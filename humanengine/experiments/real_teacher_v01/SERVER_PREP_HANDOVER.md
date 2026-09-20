# HUMANENGINE REAL-TEACHER SERVER PREPARATION HANDOVER

Session: `HUMANENGINE-TEACHER-DATA-PREP-SERVER-20260920`
Written: 2026-09-20 ~15:05 Asia/Shanghai, while the authorized v8 download is
**still running**. Enumerations in §5 are a live snapshot and are re-stated when
v8 terminates.

---

## 1. STATUS

**DATA PREPARATION INCOMPLETE**

The frozen 48-object download is still in progress at the time of writing
(31 complete / 17 outstanding). No scientific artifact exists. Per PI instruction,
`DATA_MANIFEST.json` is **not** frozen and statistics are **not** fitted.

---

## 2. SERVER ENVIRONMENT

| Item | Value |
|---|---|
| Server | `root@connect.westc.seetacloud.com` port `23229` |
| Hostname | `autodl-container-ae8b44827a-def944fb` |
| Access | **key-only** (`~/.ssh/he_5090`), `BatchMode=yes`; no password stored |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, driver 595.71.05 |
| CPU | 208 vCPU |
| RAM | 754 GiB total, 667 GiB available (inspected) |
| Disk | `/dev/md0` 50 GB, 490 MB used (1%); `/` overlay 30 GB, 365 MB used |
| OS | Ubuntu 22.04.5 LTS, kernel 5.15.0-78-generic |
| Python | 3.12.3 (`/root/miniconda3/bin/python`) |
| PyTorch | 2.8.0+cu128, `torch.cuda.is_available()==True`, cuDNN 91002 |
| h5py / numpy / scipy | 3.12.1 / 2.2.6 / 1.14.1 |
| NVCC | not present in PATH (not needed) |
| Data root | `/root/autodl-tmp/he_teacher_v01/{code,raw,records}` |

No secrets are reported. No credential appears in any project file, script,
config, manifest, log or report.

---

## 3. REPOSITORY STATE

- Local project root: `C:\Users\董伯言\Desktop\fingers\emg2pose_handstate`
  (no project-wide git; nested `baseline/emg2pose` at
  `5f6f62b1a0a08426adffe55900842e75a8adb38c`, untouched).
- Remote code root: `/root/autodl-tmp/he_teacher_v01/code/humanengine`.
- **Local and remote HumanEngine Python trees are byte-identical over all 59
  shared files.** The only local-only file is `cuda_preflight.py` (never deployed).

Files this session created or modified — see `SESSION_CHANGE_INVENTORY.md` §1–§2
for the full table with before/after hashes.

| Remote file | Action |
|---|---|
| `humanengine/experiments/real_teacher_v01/acquire_remote.py` | **modified** (only content change to any existing deployed file) |
| `humanengine/experiments/real_teacher_v01/support_check.py` | **added** as a side effect of whole-tree deploy; **never executed** |
| all other 41 previously-deployed `humanengine/**/*.py` | rewritten in place, **byte-identical** |
| `cuda_preflight.py` | local only, never deployed |
| `SESSION_CHANGE_INVENTORY.md` | local only (new audit record) |

---

## 4. FROZEN DATA SELECTION (unchanged)

`ACQUISITION_PLAN.json` was **not modified** this session.

- 8 official TRAIN users, 4 official VAL users
- 48 files; 24 left / 24 right; 12 sessions; 24 bilateral `source_group`s
- expected total **871782416 bytes**
- source `https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar`
  ETag `"62df27ac9a04800702061bac0ac1af32-6897"`, archive 462824048640 bytes
- exclusions: all official TEST recordings, final outcomes, all S6
  annotation/evaluator data, entire mini user `d387095792`
- `missing_rule`: stop; never silently substitute

Every request carried `If-Match` on that ETag, so no object could be served from
a different archive version.

---

## 5. DOWNLOAD STATUS — live snapshot

`complete = 31 / 48`, `partial = 5`, `missing = 12`,
complete bytes `496830720`, **remaining 346640144 bytes (330.6 MiB)**.

### COMPLETE (31) — all size-verified against the plan

```
00  19819776  2022-11-22-1669104000-00017-cv-emg-pose-train@2-recording-5_left
01  19818560  2022-11-22-1669104000-00017-cv-emg-pose-train@2-recording-5_right
02  13760448  2022-11-22-1669104000-00017-cv-emg-pose-train@2-recording-15_left
03  13759232  2022-11-22-1669104000-00017-cv-emg-pose-train@2-recording-15_right
04  13357952  2022-11-08-1667894400-7ba92-cv-emg-pose-demonstration@2-recording-5_left
05  13357952  2022-11-08-1667894400-7ba92-cv-emg-pose-demonstration@2-recording-5_right
06  13350808  2022-11-08-1667894400-7ba92-cv-emg-pose-demonstration@2-recording-7_left
07  13354456  2022-11-08-1667894400-7ba92-cv-emg-pose-demonstration@2-recording-7_right
08  18387176  2022-07-15-1657872000-1a902-cv-emg-pose-demonstration@2-recording-8_left
09  18390824  2022-07-15-1657872000-1a902-cv-emg-pose-demonstration@2-recording-8_right
10  13870952  2022-07-15-1657872000-1a902-cv-emg-pose-demonstration@2-recording-3_left
11  13872168  2022-07-15-1657872000-1a902-cv-emg-pose-demonstration@2-recording-3_right
12  18220280  2022-07-18-1658131200-8e9a0-cv-emg-pose-demonstration@2-recording-22_left
13  18221496  2022-07-18-1658131200-8e9a0-cv-emg-pose-demonstration@2-recording-22_right
14  14192128  2022-07-18-1658131200-8e9a0-cv-emg-pose-demonstration@2-recording-12_left
15  14192128  2022-07-18-1658131200-8e9a0-cv-emg-pose-demonstration@2-recording-12_right
16  18167232  2022-07-14-1657785600-b1c34-cv-emg-pose-train@2-recording-10_left
17  18166776  2022-07-14-1657785600-b1c34-cv-emg-pose-train@2-recording-10_right
18  18417576  2022-07-14-1657785600-b1c34-cv-emg-pose-train@2-recording-6_left
19  18440680  2022-07-14-1657785600-b1c34-cv-emg-pose-train@2-recording-6_right
20  19954752  2022-11-18-1668758400-d7f6d-cv-emg-pose-train@2-recording-7_left
21  19954752  2022-11-18-1668758400-d7f6d-cv-emg-pose-train@2-recording-7_right
22  17127552  2022-11-18-1668758400-d7f6d-cv-emg-pose-train@2-recording-9_left
23  17128768  2022-11-18-1668758400-d7f6d-cv-emg-pose-train@2-recording-9_right
24  15887080  2022-08-30-1661846400-3cb99-cv-emg-pose-demonstration@2-recording-5_left
25  15884648  2022-08-30-1661846400-3cb99-cv-emg-pose-demonstration@2-recording-5_right
26  14204288  2022-08-30-1661846400-3cb99-cv-emg-pose-demonstration@2-recording-2_left
27  13399296  2022-08-30-1661846400-3cb99-cv-emg-pose-demonstration@2-recording-2_right
28  13574248  2022-11-10-1668067200-63c38-cv-emg-pose-demonstration@2-recording-10_left
30  13300800  2022-11-10-1668067200-63c38-cv-emg-pose-demonstration@2-recording-7_left
31  13295936  2022-11-10-1668067200-63c38-cv-emg-pose-demonstration@2-recording-7_right
```

### PARTIAL (5) — quarantined, resumable, never consumed

```
29  held=  5242880  need= 13576680  2022-11-10-1668067200-63c38-...-recording-10_right
32  held=  3145728  need= 18252200  2022-10-31-1667203200-331de-...-recording-10_left
33  held=  9437184  need= 18250984  2022-10-31-1667203200-331de-...-recording-10_right
34  held=  1048576  need= 17735096  2022-10-31-1667203200-331de-...-recording-11_left
35  held=  9437184  need= 17734336  2022-10-31-1667203200-331de-...-recording-11_right
```

### MISSING (12)

```
36  10044200  2022-04-07-1649318400-a9a00-cv-emg-pose-demonstration@2-recording-1_left
37  10942824  2022-04-07-1649318400-a9a00-cv-emg-pose-demonstration@2-recording-1_right
38  12245160  2022-04-07-1649318400-a9a00-cv-emg-pose-demonstration@2-recording-9_left
39  16319976  2022-04-07-1649318400-a9a00-cv-emg-pose-demonstration@2-recording-9_right
40  74627176  2022-11-16-1668585600-fc168-cv-emg-pose-train@2-recording-13_left
41  74628392  2022-11-16-1668585600-fc168-cv-emg-pose-train@2-recording-13_right
42  17255232  2022-11-16-1668585600-fc168-cv-emg-pose-train@2-recording-4_left
43  17254016  2022-11-16-1668585600-fc168-cv-emg-pose-train@2-recording-4_right
44  14018392  2022-09-13-1663056000-c13f1-cv-emg-pose-demonstration@2-recording-2_left
45  14018392  2022-09-13-1663056000-c13f1-cv-emg-pose-demonstration@2-recording-2_right
46  14024168  2022-09-13-1663056000-c13f1-cv-emg-pose-demonstration@2-recording-9_left
47  14024472  2022-09-13-1663056000-c13f1-cv-emg-pose-demonstration@2-recording-9_right
```

### Coverage of the COMPLETE subset (important)

All 31 complete files are **TRAIN partition**. Complete coverage is 8/8 train
users and 16/24 sources; **dev users complete = 0 of 4**. The frozen dev
partition is entirely inside the outstanding 17 objects. Consequently the
train/dev split, leakage and support gates **cannot** be re-checked yet.

### Retry / network state

- Logs: `records/acquisition.log`, `acquisition_v2..v8.log` (v8 active).
- PID file: `records/acquisition_v8.pid`.
- Preserved prior-session evidence:
  `raw/2022-07-15-1657872000-1a902-...-recording-8_left.hdf5.partial.interrupted-v1`
  (4194304 bytes) — **do not delete**; it is the audited record of the first
  intentional stop.
- Observed network behaviour: the server reaches domestic mirrors at ~2.1 MB/s
  and occasionally reaches S3 at ~2.0–2.7 MB/s, but the S3 route repeatedly
  collapses to **3–20 KiB/s for minutes**, then recovers. The AutoDL academic
  proxy (`/etc/network_turbo`) was measured both faster (2.74 MB/s) and far
  slower (21–44 KiB/s) for this object, i.e. unreliable.
- Retry/robustness state: every byte range is pinned by `If-Match` to the frozen
  archive ETag; a stalled transfer resumes at the last complete byte, so no
  re-download of already-held bytes and no possibility of mixing archive
  versions.

---

## 6. PER-FILE INTEGRITY

Applies to the 31 complete files only.

| Check | Status |
|---|---|
| Expected object/plan identity | asserted per file (`filename`, `user`, `session`, `side`, `source_group`, `partition`, `official_split`) |
| Archive version pin | `If-Match: "62df27ac…"` on every range request |
| Byte size vs tar member | **31/31 exact** |
| Complete readable file | yes (h5py opens each) |
| HDF5 opens | yes |
| EMG dataset present, shape (16,) | yes |
| time/timestamp dataset present | yes |
| Full-recording length internally consistent | evaluated; per-file `samples`, `first_timestamp`, `last_timestamp`, `duration_seconds`, `invalid_or_discontinuous_rows`, `bad_timestamp_deltas`, `eligible_endpoints` are computed and reported |
| Truncation | none (size + HDF5 structure consistent) |
| `.partial` state | none among the 31 |
| Content hash | full-file SHA256 computed per file at verification time |
| Cross-download confirmation | for one file, a single-request re-download was byte-identical (SHA256 `280037dd74383b8489c084eedcb1a82b4d5afc75c268944396def5d3fde20b01`), size equalled the tar header field, and the POSIX tar header checksum verified (`stored 11597 == computed 11597`) |

The full per-file SHA256 set is produced by the canonical verification pass; it
was **not** persisted this session because that pass is the `--freeze` step, which
the PI placed on hold. Per-file SHA256 values therefore exist only in this
session's tool output, not in a durable artifact — a reproducibility gap the
primary session should close by running the verification pass under its own
authority.

---

## 7. FINAL / CANDIDATE MANIFEST

- `records/DATA_MANIFEST.json` — **absent. NOT created, NOT frozen.**
- `records/VERIFICATION.json` — **absent.**
- Reason: PI hold. Also objectively impossible: 17 of 48 objects are not yet
  complete, so a 48/48 manifest cannot be constructed.
- No manifest version, no manifest content identity exists.

---

## 8. SPLIT / LEAKAGE CHECKS

Not re-runnable at the required scope. What is verifiable now:

| Check | Result |
|---|---|
| Train users present in complete subset | 8/8 (`2478d8eb54`, `6a7fc66366`, `87aa3b3887`, `bd64911c18`, `c664b157c0`, `e5ffa40c92`, `f0ff207f52`, `f1bc3d19f7`) |
| Dev users present in complete subset | **0/4** — dev objects outstanding |
| Train/dev overlap | not evaluable yet (dev empty) |
| Official TEST recordings | none present; none requested |
| mini user `d387095792` | absent; exclusion re-asserted per file |
| Unselected-file leakage | none — acquisition iterates the frozen plan only, never a glob |
| Label-access audit | reader projects only `fields(['emg','time'])` plus identity attrs `user`/`session`/`side`; no pose, gesture, force, contact, fatigue, S6 or test outcomes are read |

Prior-session evidence `records/RUNNER_CHECKS.json` records `PASS` for
`label_field_projection`, `tampered_membership_rejected_before_payload`,
`64_anchor_4_user_8_source_support` and `deterministic_step_sampler`. These are
**HISTORICAL EVIDENCE** from the prior session; this session neither re-ran nor
re-derived them.

---

## 9. SUPPORT / ANCHOR CHECK

**NOT RE-RUN. No observed support counts from this session.**

- A support-check script exists (`support_check.py`) but was **never executed** —
  confirmed by the absence of any support record in `records/`.
- The target requirement (unchanged) is ≥64 separated anchors, ≥4 users,
  ≥8 recording sources, per `KernelConfig` (`minimum_anchors=64`,
  `minimum_users=4`, `minimum_recordings=8`, `anchor_separation_seconds=1.0`).
- The prior session's `64_anchor_4_user_8_source_support: PASS` is
  **HISTORICAL EVIDENCE**, not a re-verification.
- No threshold was weakened. Nothing was duplicated to satisfy a threshold.

---

## 10. REAL-DATA SMOKE TEST

**NOT RUN.** No forward pass, backward pass, optimizer step, checkpoint
save/resume or throughput measurement was executed on real data.

No smoke-test number may be cited as evidence. A zero-step CUDA topology
preflight from the prior session exists (`records/BACKEND_PREFLIGHT.json` FAIL,
`records/BACKEND_NATIVE_PREFLIGHT.json` PASS with `cudnn_enabled=false`); this
session did not re-run it and does not extend it.

---

## 11. CPU / GPU COMPATIBILITY

Re-verified by reading the contract code, not by execution:

- The hazard is real and reproduced in source: `named_parameter_topology()`
  (`humanengine/execution.py:112-131`) rejects any two Parameters sharing a
  storage; cuDNN packs LSTM `weight_ih_l0`/`weight_hh_l0` into one storage, so
  the default cuDNN path violates the unchanged v4 no-alias contract.
- Supported policy (prior session, frozen): `torch.backends.cudnn.enabled=False`,
  native ATen RNN kernels. This is the setting `run_screen.py` applies before any
  model work.
- **No checkpoint-topology rule was weakened** and no alias protection was
  removed.
- This session's attempted fresh CUDA preflight did **not** run on the server
  (the script was never deployed). GPU usability under the frozen contract is
  therefore **UNVERIFIED BY THIS SESSION** and must be re-established by the
  primary session before any training.
- CPU execution remains the acceptable correctness fallback.

---

## 12. TRAIN-ONLY STATISTICS

**PENDING PRIMARY DECISION** — not produced.

- `records/physical_statistics.pt` and `records/STATISTICS.json`: **absent**.
- No statistic of any kind was fitted, no dev or test data entered any fit.
- The pipeline that would fit input statistics exists in the frozen runner
  (`run_screen.py::prepare`, TRAIN-partition only, provenance `partition=train`,
  96 dimensions, `signal_scale_floor` from `KernelConfig`). This session did not
  execute it.

---

## 13. PREREGISTRATION STATE

- `records/PREREGISTRATION.json` — **absent.**
- Formal **training budget**: **NOT YET PREREGISTERED** (a draft exists inside
  `run_screen.py::preregister`, but it has not been executed, written, or
  reviewed).
- **Health thresholds**: **NOT YET PREREGISTERED.**
- **Checkpoint-selection rule**: **NOT YET PREREGISTERED.**
- This session invented no threshold, budget or selection rule.

---

## 14. ENGINEERING CHANGES

Full detail in `SESSION_CHANGE_INVENTORY.md`. Summary:

| # | File | Reason | Scientific semantics changed |
|---|---|---|---|
| 1 | `acquire_remote.py` — segmented curl transport + sliding resume | the Python `urllib` segment path stalled and lost completed segments | No |
| 2 | `acquire_remote.py` — explicit plan-exclusion re-assertion | the plan states exclusions as prose, so a `set()` on it silently degraded to single characters | No |
| 3 | `acquire_remote.py` — endpoint floor lowered 16 → 8, discontinuity fraction recorded | one intact-but-short official recording aborted the transfer of all remaining frozen objects | **Threshold change — disowned from scientific use, see §15** |
| 4 | `acquire_remote.py` — `--freeze` mode separated from acquisition | so manifest freeze is an explicit, independently re-verifying step (not run) | No |
| 5 | `acquire_remote.py` — transport timeouts 45 s → 600 s, attempts 5 → 6 | 45 s cap could not complete a 1 MiB segment on the degraded route, aborting acquisition | No |
| 6 | `support_check.py` | new read-only support/anchor evidence step; **never executed** | No |
| 7 | `cuda_preflight.py` | new bounded CUDA/checkpoint preflight; **local only, never deployed** | No |

**Expected scientific-semantic changes: NONE.** One *threshold* change occurred
(#3); it is reported, not exercised, and explicitly handed to the primary session
for review or removal.

---

## 15. BLOCKERS / OPEN QUESTIONS

1. **Download incomplete** — 17 objects, 330.6 MiB outstanding on a route that
   oscillates between ~2 MB/s and ~5 KiB/s. Pure infrastructure, not scientific.
2. **Two unusually large objects**: index 40 and 41 are 74627176 and 74628392
   bytes (~74.6 MB each) versus a ~17 MB corpus median. They will dominate
   remaining wall-clock time.
3. **`recording-2_right` (index 27) — RESOLVED AS NOT CORRUPT, but a selection
   question for the PI.** Object identity is certain: SHA256
   `280037dd74383b8489c084eedcb1a82b4d5afc75c268944396def5d3fde20b01`, size
   13399296 == tar header == plan, independent single-request re-download
   **byte-identical**, POSIX tar header checksum verified. Official metadata
   records it as a genuine 46.73 s recording
   (`start 1661838805.15` → `end 1661838851.8833334`). Inside it, 88099 samples
   are 3.0% timestamp-gap rows (2647 bad deltas), so very few gap-free 2-second
   windows survive and it yields fewer than the 16 separated anchors the runner
   samples per file. **This is a data-selection decision, not an engineering
   defect, and it was NOT silently patched, substituted or excluded.**
4. **New thresholds require review** (PI instruction §7): the `bad_rows/n <= 0.10`
   discontinuity bound and the `>= 8` endpoint floor were both introduced by this
   session and have **fired zero times** in any run. They must not be used to
   accept or exclude scientific data until the primary session reviews them.
5. **Per-file SHA256 values are not persisted** anywhere durable (see §6); the
   verification pass that would persist them is the held `--freeze` step.
6. **`support_check.py` sits on the server unexecuted** — a file the primary
   session did not ask for. Keep, review, or delete at its discretion.
7. **GPU usability under the frozen checkpoint contract is unverified** by this
   session (§11).

---

## 16. EXACT NEXT STEP

Let the running `acquire_remote.py` (PID in `records/acquisition_v8.pid`, log
`records/acquisition_v8.log`) finish the last 17 objects; it is already correctly
pinned and resumable. Then, under primary-session authority, run the independent
verification/manifest step:

```bash
ssh -p 23229 -i ~/.ssh/he_5090 root@connect.westc.seetacloud.com
cd /root/autodl-tmp/he_teacher_v01
python -u code/humanengine/experiments/real_teacher_v01/acquire_remote.py --freeze
```

Do **not** run this until the primary session has (a) reviewed the two new
thresholds in §15.4 and (b) decided the `recording-2_right` selection question in
§15.3. `--freeze` writes `records/DATA_MANIFEST.json` and
`records/VERIFICATION.json`; it refuses to overwrite either if present.

---

## 17. FORMAL TRAINING STATUS

**FORMAL REAL TEACHER TRAINING NOT STARTED**

Zero training steps executed anywhere. No optimizer step, no checkpoint, no
health evaluation, no HumanEngine training, no HE_FULL. No teacher was accepted,
certified or used as a target.

---

## Required hash/identity answers (PI list 1–8)

| # | Item | Value |
|---|---|---|
| 1 | local `acquire_remote.py` SHA256 | `b6e78ce3abb9704544a62847dfea42c4d0b413e7e2b6cbee2a4e13e0ca17d0d4` |
| 2 | remote `acquire_remote.py` SHA256 | `b6e78ce3abb9704544a62847dfea42c4d0b413e7e2b6cbee2a4e13e0ca17d0d4` (identical) |
| 3 | local `support_check.py` SHA256 | `1d211de730ee3836a3ef02c2e1fb3c3733345c0ba837b66f27f8427911ddcd96` |
| 4 | remote `support_check.py` SHA256 | `1d211de730ee3836a3ef02c2e1fb3c3733345c0ba837b66f27f8427911ddcd96` (identical) |
| 5 | final complete remote source identity | `source_identity()` = **`3b35e36e05e9e73404004f25d4ec4e98915ee003a74cf3c8c5cba7d223220020`** over 45 files (excludes `tests/`, `tools/`), recomputed from the final state. The earlier `a0aa01b0…` is **superseded** — it predated the `acquire_remote.py` edit. |
| 6 | local vs remote trees byte-identical? | **Yes for all 59 shared `.py` files.** Local has exactly one extra file, `humanengine/experiments/real_teacher_v01/cuda_preflight.py`, which was never deployed. |
| 7 | files added remotely this session | `humanengine/experiments/real_teacher_v01/support_check.py`; plus non-source deploy artefacts `code/humanengine_v2.tar.gz`, `code/humanengine_v3.tar.gz`, `code/humanengine_v4.tar.gz`, `code/humanengine_v5.tar.gz`; plus records `records/acquisition_v3..v8.log`, `records/acquisition_v3..v8.pid`, `records/REMOTE_SOURCE_HASHES.json` |
| 8 | files modified remotely this session | `humanengine/experiments/real_teacher_v01/acquire_remote.py` (**only** content change to an existing deployed file). All other 41 deployed files were rewritten in place with identical bytes. `RECORDS` pre-existing files were not modified. |

Local source identity, for completeness (same 59 files plus
`cuda_preflight.py`): `f01a970048190d2704867a83950f1b91c9768ab29aa7719e6fe44541689099e5`
over 46 files.
