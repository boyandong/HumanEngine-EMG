# emg2pose full dataset — remote server runbook

**Purpose:** acquire, extract, and verify the complete official emg2pose dataset
(431.04 GiB, 25,253 HDF5 files) on a server with enough disk, in one command.

**Status:** the pipeline below is **EXECUTED and VERIFIED end-to-end**. It was run
against the official mini archive, reconstructing all 30 files byte-for-byte
(SHA256 identical), with the resumable downloader proven bit-identical against the
official S3 ETag. See "Evidence" at the bottom.

---

## 1. Requirements

| Resource | Requirement |
|---|---|
| Free disk | **≥ 480 GiB** (431 GiB final data + 32 GiB window + 8 GiB headroom) |
| Network | HTTP access to `fb-ctrl-oss.s3.amazonaws.com`; ~10.3 h at 100 Mbit/s |
| Python | 3.9+ with **`h5py`** and **`pandas`** (only for verification) |
| OS | any (Linux/macOS/Windows all work) |
| RAM | modest — files are streamed, largest member is 72 MB |

Disk math: the naive official workflow (`curl` the tar, then `tar -xvf`) needs
**862 GiB** free simultaneously. This tooling uses a bounded sliding window and
needs **~471 GiB**. If the machine has ≥ 900 GiB free you may use the naive path,
but there is no reason to.

> **Do not run this on the local Windows machine.** It has 280 GiB free total and
> measured ~8 KB/s to this host. Both are hard blockers; see
> `docs/DATASET_ACCESS_DECISION.md`.

## 2. Copy these files to the server

```
data/tools/acquire_emg2pose_full.py      # the one-click orchestrator
data/tools/verify_emg2pose_dataset.py    # integrity verifier (called by the above)
data/tools/inspect_emg2pose_hdf5.py      # single-file schema inspector (optional)
```

Keep them in the **same directory** — the orchestrator locates the verifier
relative to itself.

## 3. Run

```bash
# 0. (optional) confirm the plan and disk math; touches nothing
python acquire_emg2pose_full.py plan --root /data/emg2pose

# 1. acquire + extract (resumable; re-run as often as you like)
python acquire_emg2pose_full.py run --root /data/emg2pose

# 2. verify every one of the 25,253 files
python acquire_emg2pose_full.py verify --root /data/emg2pose
```

`run` already performs a verification pass at the end (first 200 files) so a
mistake surfaces immediately; step 2 is the full check.

Useful flags:

| Flag | Meaning |
|---|---|
| `--root DIR` | where to build everything (default `/data/emg2pose`) |
| `--window-gib N` | sliding-window size (default 32). Lower it if disk is tight |
| `--seg-mib N` | HTTP range request size (default 64). **Raise this on high-latency links** — measured 35× throughput difference between tiny and 1 MiB requests |
| `--max-cycles N` | stop after N windows; for cron-style budgeted runs |

Long run in the background:

```bash
nohup python acquire_emg2pose_full.py run --root /data/emg2pose \
      > /data/emg2pose/run.log 2>&1 &
tail -f /data/emg2pose/run.log
```

## 4. What it produces

```
/data/emg2pose/
├── dataset/                                  <- THE DATA (this is what downstream uses)
│   ├── metadata.csv                          <- official, MD5-verified
│   └── emg2pose_dataset/<session>_<side>.hdf5   x 25,253 files
├── raw/
│   ├── emg2pose_metadata.csv                 <- same file, vendor layout
│   ├── acquire_state.json                    <- resume state; delete only to restart
│   └── emg2pose_dataset.tar                  <- shrinks to ~1 KiB as it is consumed
└── logs/
    └── verify_report.json                    <- full machine-readable verification
```

The single 431 GiB tar is **consumed and pruned** as it goes and ends at ~1 KiB;
you are never left holding both the tar and the tree.

## 5. How it works (and why it is safe to interrupt)

Per cycle, over a bounded window:

1. **Download** the next window from the official URL via HTTP Range requests
   into its correct offset in a local tar file. Completed 64 MiB segments are
   recorded in `acquire_state.json` (written atomically), so nothing is
   re-downloaded.
2. **Extract** every tar member fully contained in that window. Members already
   present with the exactly-expected size are skipped.
3. **Prune** the consumed prefix so disk usage stays bounded.
4. Members are written to `<name>.part` and atomically renamed, so an interrupted
   file never looks complete.

Correctness details that were found the hard way during testing (both were real
bugs caught by a self-test, and are fixed):

* **Coordinate spaces.** State records *remote* offsets; the tar file holds
  *local* offsets after pruning (`local = remote - base`). Mixing them made an
  earlier version read from the wrong file position.
* **Block alignment.** Tar payloads end at `offset_data + size` but are padded to
  512 bytes. Pruning at the raw member end leaves the tar starting mid-block and
  the next parse fails with `bad checksum`. The prune point is rounded up to a
  512-byte boundary.
* **Bounded reader.** The window reader must keep its own offset in sync with
  seeks, or tarfile mis-detects end-of-archive and truncates a member read.

Interrupting is safe at any point (Ctrl-C, kill, reboot). Re-run the same command
and it continues. To start completely over, delete `raw/acquire_state.json` and
`dataset/`.

## 6. Verification

```bash
python verify_emg2pose_dataset.py \
    --data     /data/emg2pose/dataset \
    --metadata /data/emg2pose/dataset/metadata.csv \
    --report   /data/emg2pose/logs/verify_report.json
```

Exit code 0 = clean, 1 = problems. It checks, per file: the `emg2pose` group and
its required attrs, the `timeseries` compound fields, `emg` = 16 ch, `joint_angles`
= 20, no NaNs, monotonic timestamps, ~2 kHz implied rate, non-degenerate payload,
plus a cross-check that every metadata `filename` exists on disk and vice versa.

**Expected result:** `25,253` files, `files_failed: 0`, `orphans on disk: 0`,
`missing (not local): 0`.

> There is **no official checksum manifest** for the dataset. The tars are
> multipart S3 uploads, so their ETags are *not* plain MD5s and cannot be used to
> validate the 431 GiB tar. Integrity therefore rests on: per-member size +
> successful HDF5 open + structural/content checks, and the MD5-verified
> `metadata.csv` as the file inventory. This limitation is inherent to the
> official distribution, not a gap in the tooling.

## 7. Storage layout for downstream research

Once acquired, put the data where the rest of the project expects it, and make
sure **not** to place it under a non-ASCII path if any Windows machine will read
it (h5py cannot open such paths — see the junction note in
`docs/EMG2POSE_DATASET_REFERENCE.md`).

If a GPU server is also where training happens (it must be, per the remote-only
rule), serve this `dataset/` directory to the training sessions and keep
`emg2pose_metadata.csv` as the single source of truth for splits.

## 8. After acquisition — the scientific caveats that matter

These are already documented in `docs/EMG2POSE_DATASET_REFERENCE.md`; repeating
because they will bite whoever trains on this data:

1. **`joint_angles` is 20-dimensional**, not 5 finger flexions. Deriving a
   5-finger state from it is a **representation decision** requiring the official
   kinematics code — not a loader detail.
2. **IK failures are all-zero joint-angle rows**, and they are not rare:
   measured median **7.0%**, max **35.0%** of frames. Mask them
   (`emg2pose.utils.get_ik_failures_mask`) or you train on garbage.
3. **`split` alone is not a generalization protocol.** All 158 train users also
   appear in test. Cross-subject claims must use
   `generalization ∈ {user, user_stage}` (2,628 test files); the cross-task axis is
   `generalization == "stage"` (3,539 test files). `emg2pose.utils.load_splits`
   keys only on `split` and silently discards this axis.
4. **The headline task is the right hand.** Both sides are distributed.

## 9. Evidence (what was actually run)

| Step | Result |
|---|---|
| `plan` against live official URL | 462,824,048,640 B = 431.04 GiB confirmed by HEAD |
| Downloader resume self-test (metadata.csv, 2 runs) | MD5 `c3f48fd51576fa7752d455f2751c8b99` = official ETag, **bit-identical** |
| Downloader idempotence (3rd run) | 0 segments remaining, downloaded nothing |
| Full orchestrator vs official mini tar (3×300 MiB windows) | 30/30 files extracted, verifier exit 0 |
| Full orchestrator vs official mini tar (7×120 MiB windows, 8 MiB segs) | exit 0, 30/30 valid |
| Byte-identity of reconstructed files | **30/30 SHA256-identical to originals, 0 mismatches** |
| Tar consumption | reduced to 1,024 B (zero padding) from 617 MiB |
| Verifier on official mini dataset | 30 files, `files_failed: 0`, `orphans: 0` |

Logs: `logs/download_full/orchestrator_regression.log`,
`logs/download_full/verify_mini_report.json`,
`logs/download_full/capacity_math.txt`.
