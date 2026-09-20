#!/usr/bin/env python3
"""Verify a local emg2pose data tree for completeness and integrity.

Checks performed
----------------
1. **File inventory** -- count ``.hdf5`` files, total bytes, per-directory roster.
2. **Metadata cross-check** -- every ``filename`` listed in
   ``emg2pose_metadata.csv`` is present on disk, and every file on disk is listed
   in the metadata (reports both orphan directions).
3. **Per-file HDF5 structure** -- opens each file, asserts the ``emg2pose`` group
   exists with its full metadata attrs, and that ``timeseries`` carries the three
   required compound fields (``time``/``joint_angles``/``emg``) with the expected
   channel counts.
4. **Content sanity** -- 2 kHz sampling implied by timestamps, monotonic time,
   no all-NaN payloads, IK-failure (all-zero joint-angle row) rate.
5. **Truncation detection** -- reports ``.part``/``.tmp`` leftovers and files that
   fail to open (classic signature of an interrupted download/extraction).

Outputs a machine-readable JSON report plus a human summary. Exit code is 0 when
no integrity problems are found, 1 otherwise, so it can gate a pipeline.

Usage
-----
    python verify_emg2pose_dataset.py --data E:\\emg2pose\\dataset \\
        --metadata emg2pose_metadata.csv --report verify_report.json

    # quick pass over the first N files only
    python verify_emg2pose_dataset.py --data ... --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

HDF5_GROUP = "emg2pose"
TIMESERIES = "timeseries"
REQUIRED_FIELDS = ("time", "joint_angles", "emg")
# Attrs the official loader's docstring declares (Emg2PoseSessionData), verified
# present in the distributed data.
REQUIRED_ATTRS = (
    "session", "side", "stage", "start", "end",
    "num_channels", "user", "sample_rate",
)
# Present in the repo's synthetic test fixture but NOT in the distributed mini
# dataset, and not required by the official loader -- warn, never fail.
OPTIONAL_ATTRS = ("dataset", "filename")
EXPECTED_EMG_CHANNELS = 16
EXPECTED_JOINT_ANGLES = 20
EXPECTED_SAMPLE_RATE = 2000.0


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TiB"


def check_one_file(path: Path, read_samples: int) -> dict:
    """Structural + content validation of a single session file."""
    rec: dict = {"file": str(path), "bytes": path.stat().st_size, "ok": False,
                 "errors": [], "warnings": []}
    try:
        with h5py.File(path, "r") as f:
            if HDF5_GROUP not in f:
                rec["errors"].append(f"missing group /{HDF5_GROUP}")
                return rec
            grp = f[HDF5_GROUP]

            missing_attrs = [a for a in REQUIRED_ATTRS if a not in grp.attrs]
            if missing_attrs:
                rec["errors"].append(f"missing attrs: {missing_attrs}")
            absent_optional = [a for a in OPTIONAL_ATTRS if a not in grp.attrs]
            if absent_optional:
                rec["warnings"].append(f"absent optional attrs: {absent_optional}")
            rec["session"] = str(grp.attrs.get("session", ""))
            rec["user"] = str(grp.attrs.get("user", ""))
            rec["stage"] = str(grp.attrs.get("stage", ""))
            rec["side"] = str(grp.attrs.get("side", ""))
            rec["sample_rate"] = float(grp.attrs.get("sample_rate", 0) or 0)

            if TIMESERIES not in grp:
                rec["errors"].append(f"missing /{HDF5_GROUP}/{TIMESERIES}")
                return rec
            ts = grp[TIMESERIES]
            fields = ts.dtype.names or ()
            missing_fields = [x for x in REQUIRED_FIELDS if x not in fields]
            if missing_fields:
                rec["errors"].append(f"missing timeseries fields: {missing_fields}")
                return rec

            n = len(ts)
            rec["samples"] = n
            if n == 0:
                rec["errors"].append("empty timeseries (0 samples)")
                return rec

            emg_shape = ts.dtype[REQUIRED_FIELDS[2]].shape
            ja_shape = ts.dtype[REQUIRED_FIELDS[1]].shape
            rec["emg_channels"] = int(emg_shape[0]) if emg_shape else 1
            rec["joint_angles"] = int(ja_shape[0]) if ja_shape else 1
            if rec["emg_channels"] != EXPECTED_EMG_CHANNELS:
                rec["warnings"].append(
                    f"emg channels {rec['emg_channels']} != {EXPECTED_EMG_CHANNELS}")
            if rec["joint_angles"] != EXPECTED_JOINT_ANGLES:
                rec["warnings"].append(
                    f"joint_angles {rec['joint_angles']} != {EXPECTED_JOINT_ANGLES}")

            k = min(n, read_samples)
            block = ts[:k]
            t = np.asarray(block["time"], dtype=np.float64)
            emg = np.asarray(block["emg"])
            ja = np.asarray(block["joint_angles"])

            rec["nan_emg"] = int(np.isnan(emg).sum())
            rec["nan_joint_angles"] = int(np.isnan(ja).sum())
            if rec["nan_emg"]:
                rec["errors"].append(f"{rec['nan_emg']} NaN values in emg block")
            if rec["nan_joint_angles"]:
                rec["errors"].append(
                    f"{rec['nan_joint_angles']} NaN values in joint_angles block")

            neg = int((np.diff(t) < 0).sum())
            if neg:
                rec["errors"].append(f"timestamps not monotonic ({neg} negative diffs)")
            if k > 1:
                dt = float(np.median(np.diff(t)))
                rec["median_dt"] = dt
                rec["implied_rate_hz"] = (1.0 / dt) if dt > 0 else None
                if dt > 0 and abs(1.0 / dt - EXPECTED_SAMPLE_RATE) > 1.0:
                    rec["warnings"].append(
                        f"implied rate {1.0/dt:.1f} Hz != {EXPECTED_SAMPLE_RATE} Hz")

            all_zero = (ja == 0).all(axis=1)
            rec["ik_failure_rows"] = int(all_zero.sum())
            rec["ik_failure_frac"] = float(all_zero.mean())

            if not np.isfinite(emg).any():
                rec["errors"].append("emg block entirely non-finite")
            if float(np.nanmax(np.abs(emg))) == 0.0:
                rec["errors"].append("emg block is all zeros")

            rec["ok"] = not rec["errors"]
    except Exception as e:
        rec["errors"].append(f"OPEN FAILED: {type(e).__name__}: {e}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="root of the extracted dataset tree")
    ap.add_argument("--metadata", type=Path, default=None,
                    help="emg2pose_metadata.csv (optional cross-check)")
    ap.add_argument("--report", type=Path, default=None, help="write JSON report here")
    ap.add_argument("--limit", type=int, default=None, help="check only first N files")
    ap.add_argument("--read-samples", type=int, default=20_000,
                    help="samples read per file for content checks")
    args = ap.parse_args()

    root: Path = args.data
    if not root.exists():
        raise SystemExit(f"data root not found: {root}")

    # NOTE: deliberately NOT calling root.resolve(). On this machine the project
    # path contains non-ASCII characters and h5py cannot open such paths, so data
    # is reached through an ASCII junction (C:\emg2pose -> ...\emg2pose_handstate).
    # resolve() would follow the junction back to the CJK target and break every
    # h5py open. We only ever build child paths by appending, and rglob() yields
    # paths that share this same prefix.
    print(f"data root : {root}")
    files = sorted(p for p in root.rglob("*.hdf5"))
    parts = sorted(p for p in root.rglob("*.part")) + sorted(p for p in root.rglob("*.tmp"))
    total_bytes = sum(p.stat().st_size for p in files)

    print(f"hdf5 files: {len(files)}")
    print(f"total size: {human(total_bytes)}")
    print(f"leftover .part/.tmp files: {len(parts)}")

    by_dir: Counter = Counter()
    for p in files:
        by_dir[str(p.parent.relative_to(root))] += 1
    print(f"directories with files: {len(by_dir)}")
    for d, c in by_dir.most_common(10):
        print(f"   {c:>6}  {d}")

    # ---- metadata cross-check ----
    meta_check: dict = {"performed": False}
    if args.metadata and args.metadata.exists():
        import pandas as pd

        df = pd.read_csv(args.metadata)
        listed = set(df["filename"].astype(str))
        on_disk = {p.stem for p in files}
        meta_check = {
            "performed": True,
            "metadata_rows": int(len(df)),
            "metadata_unique_filenames": len(listed),
            "columns": list(df.columns),
            "present": len(listed & on_disk),
            "missing_count": len(listed - on_disk),
            "missing_sample": sorted(listed - on_disk)[:10],
            "orphan_count": len(on_disk - listed),
            "orphan_sample": sorted(on_disk - listed)[:10],
            "splits": df["split"].value_counts().to_dict() if "split" in df else {},
            "users": int(df["user"].nunique()) if "user" in df else None,
            "stages": int(df["stage"].nunique()) if "stage" in df else None,
            "sessions": int(df["session"].nunique()) if "session" in df else None,
        }
        print(f"\n--- metadata cross-check ---")
        print(f"metadata rows       : {meta_check['metadata_rows']}")
        print(f"present on disk     : {meta_check['present']}")
        print(f"missing (not local) : {meta_check['missing_count']}")
        print(f"orphans on disk     : {meta_check['orphan_count']}")
        print(f"split counts        : {meta_check['splits']}")
        print(f"users / stages / sessions: {meta_check['users']} / "
              f"{meta_check['stages']} / {meta_check['sessions']}")
    elif args.metadata:
        print(f"\n!! metadata file not found: {args.metadata}")

    # ---- per-file checks ----
    subset = files[: args.limit] if args.limit else files
    print(f"\n--- per-file integrity checks ({len(subset)} files) ---")
    t0 = time.time()
    records = []
    for i, p in enumerate(subset, 1):
        rec = check_one_file(p, args.read_samples)
        records.append(rec)
        if not rec["ok"] or i == len(subset) or i % 200 == 0:
            status = "OK  " if rec["ok"] else "FAIL"
            print(f"  [{i:>6}/{len(subset)}] {status} {p.name}"
                  + ("" if rec["ok"] else f"  errors={rec['errors']}"), flush=True)
    elapsed = time.time() - t0

    bad = [r for r in records if not r["ok"]]
    warn = [r for r in records if r["warnings"] and r["ok"]]
    print(f"\nchecked {len(records)} files in {elapsed:.1f}s "
          f"({len(records)/(elapsed or 1):.1f} files/s)")
    print(f"failed : {len(bad)}")
    print(f"warned : {len(warn)}")

    summary = {
        "data_root": str(root),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_hdf5_files": len(files),
        "total_bytes": total_bytes,
        "total_human": human(total_bytes),
        "leftover_part_files": [str(p) for p in parts],
        "dirs": dict(by_dir),
        "metadata_check": meta_check,
        "files_checked": len(records),
        "files_failed": len(bad),
        "files_warned": len(warn),
        "sample_rate_attr": Counter(
            r.get("sample_rate") for r in records if r.get("sample_rate")).most_common(3),
        "sides": Counter(r.get("side") for r in records).most_common(),
        "stages": Counter(r.get("stage") for r in records).most_common(10),
        "emg_channels": Counter(r.get("emg_channels") for r in records).most_common(),
        "joint_angle_dims": Counter(r.get("joint_angles") for r in records).most_common(),
        "records": records,
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2, default=str), "utf-8")
        print(f"\nJSON report: {args.report}")

    print("\n=== VERDICT ===")
    if not files:
        print("NO DATA: no .hdf5 files found -- nothing verified.")
        return 1
    if bad:
        print(f"INTEGRITY PROBLEMS: {len(bad)} of {len(records)} checked files failed.")
        return 1
    print(f"OK: {len(records)} files structurally valid and readable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
