# emg2pose dataset — source, layout, and where every field lives

**Status of this document**

| Item | Evidence level |
|---|---|
| Official source URLs, object sizes, ETags | **VERIFIED** — live HTTP HEAD on 2026-09-17 |
| HDF5 internal schema (groups, dtypes, attrs) | **VERIFIED** — read from official `.hdf5` files with h5py |
| Archive layout (top-level dir, filename pattern) | **VERIFIED** — extracted from the official `_mini.tar` |
| File/user/session/stage counts, split tables | **VERIFIED** — computed from the official `metadata.csv`, MD5-confirmed against the S3 ETag |
| Content statistics (rates, channels, IK-failure rate) | **VERIFIED** — measured on 30 official files |
| Generalization protocol interpretation | **VERIFIED (cross-tab)** + documented |

Everything below is measured against real artifacts. Nothing here is inferred
from the README alone.

---

## 1. Official source

Canonical repository: <https://github.com/facebookresearch/emg2pose>
(paper: <https://arxiv.org/abs/2412.02725>, NeurIPS 2024 Datasets & Benchmarks)

All official data objects are served from one public S3 bucket, no credentials
required, HTTP Range requests supported:

| Object | URL | Size (bytes) | Size | ETag parts |
|---|---|---|---|---|
| **Full dataset** | `https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar` | `462,824,048,640` | **431.04 GiB** | 6897 (multipart) |
| Mini (sanity check) | `.../emg2pose_dataset_mini.tar` | `647,198,208` | 617.22 MiB | 78 (multipart) |
| Metadata CSV | `.../emg2pose_metadata.csv` | `5,674,462` | 5.41 MiB | 1 (single-part) |
| Pretrained checkpoints | `.../emg2pose_model_checkpoints.tar.gz` | `248,072,445` | 236.58 MiB | 30 (multipart) |

`Last-Modified` for all four: **2024-06-12**.

### Official recommended download method

The official README specifies plain `curl` (or `wget`), with no checksum file and
no per-file download endpoint:

```shell
curl https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar -o emg2pose_dataset.tar
tar -xvf emg2pose_dataset.tar          # -> ./emg2pose_dataset/
```

**There is no official checksum/manifest.** The S3 `ETag` is an MD5 *only* for
single-part uploads; the dataset tars are multipart, so their ETags are
`<md5-of-part-md5s>-<N>` and **cannot** be verified by a plain MD5 of the file.
Only `emg2pose_metadata.csv` (1 part) has a directly checkable MD5 ETag:
`c3f48fd51576fa7752d455f2751c8b99`.

Bucket listing (`?list-type=2`) returns **HTTP 403 AccessDenied**, so individual
HDF5 files cannot be enumerated or fetched directly from S3 — the tar is the only
official bulk access path.

### License

CC-BY-NC-SA-4.0 (non-commercial, share-alike). Relevant if results are published.

---

## 2. Dataset scale — VERIFIED against the official metadata

| Property | Value |
|---|---|
| HDF5 session files | **25,253** |
| Participants (`user`) | **193** |
| Recording sessions | **751** |
| Stages | **29** |
| Rows by split | train **17,136** / test **6,167** / val **1,950** |
| sides | left 12,654 / right 12,599 |
| `moving_hand` | both 13,121 / right 7,532 / left 4,600 |

- **2 kHz** sEMG, **16** channels per wrist, high-pass filtered.
- Each file is one *stage* of one hand — **~1 minute** (measured: median 58.6 s,
  min 45.7 s, max 236.9 s).

A file is the atomic unit: (subject, session, stage, hand side).

> Source: `data/official/emg2pose_dataset_mini/metadata.csv`
> (5,674,462 B, MD5 `c3f48fd51576fa7752d455f2751c8b99` = official S3 ETag, i.e.
> bit-identical to what the vendor serves). Full analysis:
> `logs/download_full/metadata_analysis.txt`.

---

## 3. Where each piece of information lives

This is the part that matters for writing loaders. **Everything is inside a single
HDF5 file per stage** — EMG, joint angles, timestamps, and metadata all travel
together. There is no separate "metadata store" for the time series.

### 3.1 File → disk path

```
<data_location>/
└── emg2pose_dataset/                      # top-level dir inside the tar
    ├── metadata.csv                       # global per-file table + split definition
    ├── 2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-13_left.hdf5
    ├── 2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-13_right.hdf5
    └── ...                                # 25253 files total
```

Filename pattern:

```
<YYYY-MM-DD>-<unix_epoch>-<user_hash>-cv-emg-pose-train@2-recording-<session_no>_<side>.hdf5
```

`<side>` is `left` or `right` — the two hands of one session are separate files
with different filename suffixes.

### 3.2 Inside one HDF5 file

Verified structure (from `emg2pose/tests/assets/test_data.hdf5`, official repo):

```
/emg2pose                                   (HDF5 group)
│
├── attrs  = METADATA                       <- all strings/floats, see table below
│     dataset      : 'emg2pose'
│     user         : 'd387095792'
│     session      : '2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-9'
│     stage        : 'IndividualFingerPointingSnap_both'
│     side         : 'right'
│     start        : 1670313313.1833334     (unix seconds)
│     end          : 1670313374.1           (unix seconds)
│     num_channels : 16
│     sample_rate  : 2000.0
│     filename     : '...recording-9_left'  (note: mirrors the metadata.csv row)
│
└── timeseries  (compound Dataset), shape (T,)
      dtype = [('time',          '<f8'),          # (T,)
               ('joint_angles',  '<f8', (20,)),   # (T, 20)
               ('emg',           '<f4', (16,))]   # (T, 16)
```

So, field by field:

| What you want | Where it is |
|---|---|
| **EMG** (16 ch, 2 kHz) | `f['emg2pose/timeseries']['emg']` → `(T, 16) float32` |
| **Joint angles** (ground truth) | `f['emg2pose/timeseries']['joint_angles']` → `(T, 20) float64` |
| **Timestamps** | `f['emg2pose/timeseries']['time']` → `(T,) float64`, unix seconds, monotonic |
| **Subject id** | `f['emg2pose'].attrs['user']` |
| **Session id** | `f['emg2pose'].attrs['session']` |
| **Stage name** (= the "gesture/task" label) | `f['emg2pose'].attrs['stage']` |
| **Hand side** | `f['emg2pose'].attrs['side']` |
| **Recording window** | `attrs['start']`, `attrs['end']` |
| **Sample rate** | `attrs['sample_rate']` (= 2000.0) |
| **Channel count** | `attrs['num_channels']` (= 16) |
| **Train/val/test split** | **not in the HDF5 file** — only in `metadata.csv` |

### 3.3 Two important semantic notes

**(a) `joint_angles` is 20-dimensional, and the 20 axes are the OFFICIAL ordered
set — do not invent a mapping.** It is *not* "5 fingers × 4 identical angles". The
canonical definition is in the official code (`emg2pose/constants.py`,
`NUM_JOINTS = 20` + ordered `JOINTS: list[Joint]` with explicit `index`), and it is
**load-bearing**: the official metrics index it directly
(`metrics.py`: `[j.index for j in JOINTS if finger in j.groups]`). Re-deriving your
own order silently produces wrong per-finger metrics.

| idx | axis | finger | group |
|---|---|---|---|
| 0 | `THUMB_CMC_FE` | thumb | proximal |
| 1 | `THUMB_CMC_AA` | thumb | proximal |
| 2 | `THUMB_MCP_FE` | thumb | mid |
| 3 | `THUMB_IP_FE` | thumb | distal |
| 4 | `INDEX_MCP_AA` | index | proximal |
| 5 | `INDEX_MCP_FE` | index | proximal |
| 6 | `INDEX_PIP_FE` | index | mid |
| 7 | `INDEX_DIP_FE` | index | distal |
| 8 | `MIDDLE_MCP_AA` | middle | proximal |
| 9 | `MIDDLE_MCP_FE` | middle | proximal |
| 10 | `MIDDLE_PIP_FE` | middle | mid |
| 11 | `MIDDLE_DIP_FE` | middle | distal |
| 12 | `RING_MCP_AA` | ring | proximal |
| 13 | `RING_MCP_FE` | ring | proximal |
| 14 | `RING_PIP_FE` | ring | mid |
| 15 | `RING_DIP_FE` | ring | distal |
| 16 | `PINKY_MCP_AA` | pinky | proximal |
| 17 | `PINKY_MCP_FE` | pinky | proximal |
| 18 | `PINKY_PIP_FE` | pinky | mid |
| 19 | `PINKY_DIP_FE` | pinky | distal |

**Four traps this table eliminates:**

1. **The thumb is structurally different.** Other fingers are
   `MCP_AA, MCP_FE, PIP_FE, DIP_FE`; the thumb is
   `CMC_FE, CMC_AA, MCP_FE, IP_FE`. A uniform `reshape(..., 5, 4)` "finger × DOF"
   array therefore *mixes scales within a slot* across fingers (slot 2 is `MCP_FE`
   for index but `CMC_FE` for thumb). Index by name, not by position.
2. **Real fingers occupy contiguous 4-blocks** — `thumb [0-3]`, `index [4-7]`,
   `middle [8-11]`, `ring [12-15]`, `pinky [16-19]` — in the order
   `FINGERS = ["thumb", "index", "middle", "ring", "pinky"]`.
3. **AA is a different axis, not a second flexion.** FE = flexion/extension,
   AA = abduction/adduction (official comment in `constants.py`). Measured on 30
   official files, the non-thumb proximal `MCP_AA` range is only **48.4%** of the
   `MCP_FE` range — so the axes are not interchangeable in scale either. Any
   shared normalization across the 20 dims is a modelling choice with consequences.
4. **Two of the 22 UmeTrack DOF are dropped.** UmeTrack defines
   `NUM_JOINTS_PER_HAND = 22` (`DOF_PER_FINGER = 4`, 5 fingers = 20, plus 2 wrist);
   emg2pose distributes 20 — i.e. **the 2 wrist DOF are excluded**. Forward
   kinematics in `kinematics.py` re-inserts them as zeros
   (`zeros(shape=[..., 2, ...])` concatenated before `swapaxes(-2, -1)`), so if you
   call official FK the 20-vector is padded to 22. Do not hand-pad differently.

**Units are radians**, not degrees — the official `metrics.py` computes angular
velocity as "radians / sample" and multiplies by `EMG_SAMPLE_RATE`, and
`joints_to_landmarks` converts with `torch.deg2rad` only when told the input is in
degrees (default `degrees=False`).

### 3.3.1 Machine-readable reference (generated, not transcribed)

`docs/generated/emg2pose_joint_axes.json` is produced by
`data/tools/make_joint_axis_reference.py`, which **AST-parses the official
`constants.py`** so no axis name can be mistranscribed, and re-checks five
invariants (4 DOF per finger, unique finger per axis, proximal/mid/distal coverage,
count == 20, thumb structure differs). All five pass. If upstream layout changes,
the generator exits non-zero rather than emitting a wrong table.

Use the JSON, not a hand-typed list, when hard-coding index groups into metrics,
probability models, or visualization presets.

**(b) IK failures are encoded as all-zero joint-angle rows.** The official loader
treats a row that is entirely ~0 as "no valid pose here"
(`emg2pose.utils.get_ik_failures_mask`), and training windows can skip them. A
naive loader that treats 0 as a legitimate pose will silently train on garbage
frames. **Measured on 30 official files: median 7.0% of frames are IK-failure
rows, max 35.0%, min 0.0%.** Count them, don't ignore them.

### 3.4 Measured content statistics (30 official files, all pass)

| Quantity | Measured |
|---|---|
| `sample_rate` attr | 2000.0 on 30/30 |
| Implied rate from `time` diffs | 2000.14 Hz |
| EMG channels | 16 on 30/30 |
| `joint_angles` dim | **20** on 30/30 |
| NaN in `emg` / `joint_angles` | 0 / 0 |
| Timestamps monotonic | yes, on 30/30 |
| Duration | 45.7 / 58.6 / 236.9 s (min / median / max) |
| IK-failure rows | 0.0% / 7.0% / 35.0% (min / median / max) |

Raw evidence: `logs/download_full/verify_mini_report.json`.

**(c) The right hand is the reported benchmark hand, but both sides exist.**
Files exist per side; the paper's headline task and the pretrained checkpoints are
for the right hand.

> Checked against an official file: `test_data.hdf5` has `side='right'` while its
> `filename` attr ends in `_left`, i.e. the attribute and filename disagree in
> that synthetic test asset. This is a **test fixture artifact** and is not
> evidence about the real dataset, but it is a reminder to key off the `side`
> attribute, not the filename suffix, and to verify against real files.

---

## 4. Splits and generalization protocol

Defined **only** in `metadata.csv`, consumed by `emg2pose.utils.load_splits`
(config `config/data_split/full_split.yaml`, which just points at
`${data_location}/metadata.csv`).

`metadata.csv` columns (official README):

| Column | Meaning |
|---|---|
| `filename` | base filename of the HDF5 file (no directory, no `.hdf5` extension) — used as the join key by `load_splits` |
| `user` | anonymized user id |
| `session` | recording session (multiple stages per session) |
| `stage` | stage name |
| `side` | `left` / `right` |
| `moving_hand` | whether the hand is prompted to move in that stage |
| `held_out_user` | whether this user is held out of training |
| `held_out_stage` | whether this stage is held out of training |
| `split` | `train` / `val` / `test` |
| `generalization` | `user`, `stage`, or `user_stage` |

Three generalization axes are evaluated (paper + `test_analysis.py`): **across
user**, **across stage**, and **across user and stage**.

### The `split` column alone is misleading — VERIFIED

The real structure is the `generalization` column (measured crosstab from the
official CSV):

| `generalization` | test | train | val |
|---|---|---|---|
| `none` | 0 | 17,136 | 0 |
| `stage` (held-out stage) | 3,539 | 0 | 0 |
| `user` (held-out user) | 2,172 | 0 | 1,618 |
| `user_stage` (both held out) | 456 | 0 | 332 |

Also measured:

* **All 158 train users appear in test** (train∩test = 158 users).
* **val shares 0 users with train** — val is the cleaner cross-subject signal.
* 20 users appear only in test.

> **Scientific consequence for this project.** A single "emg2pose test" number is
> not interpretable: it mixes same-subject samples with genuinely held-out ones.
> Claims about generalization must state the axis and be computed on the matching
> subset (`generalization ∈ {user, user_stage}` = 2,628 test files for
> cross-subject; `stage` = 3,539 for cross-task). Session 3 and the modelling
> sessions must not report a random-split score as evidence of cross-subject
> transfer. See `docs/DATASET_ACCESS_DECISION.md` §4.

`load_splits` returns `{split: [filename, ...]}` and applies
`df.groupby('split').sample(frac=subsample, random_state=seed)` when subsampling.
Note it keys **only on `split`**, so using it as-is silently discards the
`generalization` axis.

---

## 5. Reading the data from Python

### Official path (full training stack)

```python
from emg2pose.data import Emg2PoseSessionData, WindowedEmgDataset

with Emg2PoseSessionData(path) as d:
    d.metadata        # dict of the attrs above
    d.emg              # NOT a property; use d[:] or d.slice(t0, t1)
    window = d.slice(t0, t1)          # numpy structured array
    emg   = window['emg']             # (T, 16) float32
    ja    = window['joint_angles']    # (T, 20) float64
```

Requires `h5py, numpy, torch, hydra, omegaconf, joblib, tqdm` and the `emg2pose`
package installed (`pip install -e .`).

### Minimal path (h5py + numpy only) — verified working on this machine

`data/tools/inspect_emg2pose_hdf5.py` reads the same structure without torch:

```shell
python data/tools/inspect_emg2pose_hdf5.py <file>.hdf5
```

This was used to verify the schema above, and is the fallback when only the data
(not the training stack) is needed.

---

## 6. Local environment constraints discovered

Two hard constraints on **this machine**, both verified, that shape how the data
must be stored:

1. **h5py cannot open files under a non-ASCII path.**
   `sys.getfilesystemencoding()` is `utf-8`, but h5py encodes filenames with
   `mbcs`, so any path containing e.g. `董伯言` fails with
   `UnicodeEncodeError: 'mbcs' codec can't encode characters`.
   **Workaround (verified):** store data under a pure-ASCII path, or create an
   ASCII junction to the project:
   `cmd /c mklink /J C:\emg2pose "C:\Users\董伯言\Desktop\fingers\emg2pose_handstate"`
   then access everything through `C:\emg2pose\...`. Relative paths do **not**
   help — h5py resolves them to absolute paths internally.

2. **The dataset does not fit on local disk, and this network path is unusably
   slow.** See `docs/DATASET_ACCESS_DECISION.md` for the measured numbers.
