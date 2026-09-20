"""
verify_canonical16_on_real_data.py -- run the canonical16 adapter over the
OFFICIAL emg2pose mini dataset and record evidence.

Checks, on real recorded data:
  * shape contract (T,20) -> (T,16)
  * every retained value is bit-identical to the official source column
  * the validity mask agrees with official emg2pose.utils.get_ik_failures_mask
  * IK-failure frames really are all-zero rows and stay flagged
  * canonical16 is NOT a positional prefix of the 20D vector (name-based selection)
  * round trip restores the retained columns and NaNs the dropped ones

Writes docs/canonical16_real_data_evidence.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent
REPO = ROOT / "baseline" / "emg2pose"
DATA = ROOT / "data" / "official" / "emg2pose_dataset_mini"
DOCS = ROOT / "docs"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(ROOT / "representation"))
sys.path.insert(0, str(REPO))

import canonical_hand as ch  # noqa: E402
from emg2pose.data import Emg2PoseSessionData  # noqa: E402
from emg2pose.utils import get_ik_failures_mask  # noqa: E402


def main() -> int:
    files = sorted(DATA.glob("*.hdf5"))
    if not files:
        print(f"FATAL: no HDF5 under {DATA}", file=sys.stderr)
        return 2

    src = list(ch.resolve_source_indices(str(REPO)))
    dropped = list(ch.resolve_dropped_indices(str(REPO)))

    ev: dict = {
        "verification": "canonical16 adapter on official emg2pose mini dataset",
        "adapter": str(HERE / "canonical_hand.py"),
        "repo_commit": "5f6f62b",
        "source_indices": src,
        "dropped_indices": dropped,
        "n_files": len(files),
        "per_file": [],
    }
    lines: list[str] = []
    w = lines.append

    w("=" * 78)
    w("canonical16 ADAPTER -- REAL OFFICIAL DATA VERIFICATION")
    w("=" * 78)
    w(f"files          : {len(files)}")
    w(f"source indices : {src}")
    w(f"dropped indices: {dropped}")
    w("")

    tot = tot_invalid = 0
    all_values_exact = True
    all_mask_agree = True
    all_not_prefix = True
    all_roundtrip_ok = True
    mask_mismatches: list[str] = []

    for f in files:
        sess = Emg2PoseSessionData(hdf5_path=f)
        ja = np.asarray(sess["joint_angles"])  # (T,20), official radians
        n = ja.shape[0]
        tot += n

        canon, valid = ch.to_canonical16(ja, return_validity=True, repo_root=str(REPO))
        assert canon.shape == (n, 16), canon.shape

        # 1. exact preservation of retained columns
        exact = np.array_equal(canon, ja[:, src])
        all_values_exact &= bool(exact)

        # 2. mask agrees with the official helper
        official_mask = get_ik_failures_mask(ja)
        agree = np.array_equal(valid, official_mask)
        all_mask_agree &= bool(agree)
        if not agree:
            mask_mismatches.append(f.name)

        # 3. IK-failure frames are numerically all-zero rows.
        #
        # MEASURED CAVEAT: they are not always BIT-exact zeros. Across the official
        # mini dataset a small number of flagged frames carry float-epsilon noise
        # (max |value| = 1.11e-16, i.e. 2 ULP). The official mask uses
        # np.isclose(value, 0) (atol=1e-8, rtol=1e-5) and therefore classifies
        # these as IK failures, which is correct -- they are numerically zero.
        # We assert the same thing with an absolute tolerance.
        if (~valid).any():
            flagged = ja[~valid]
            max_abs = float(np.abs(flagged).max())
            n_exact = int((flagged == 0.0).all(axis=-1).sum())
            assert max_abs <= 1e-8, (
                f"{f.name}: flagged-invalid frame has max |value| {max_abs}, "
                "which is not numerically zero"
            )
            n_flagged = int((~valid).sum())
            ev.setdefault("ik_failure_zero_tolerance", []).append(
                {
                    "filename": f.name,
                    "n_flagged": n_flagged,
                    "n_bit_exact_zero": n_exact,
                    "n_near_zero_only": n_flagged - n_exact,
                    "max_abs_value_in_flagged": max_abs,
                }
            )

        # 4. NOT a positional prefix
        not_prefix = not np.array_equal(canon, ja[:, :16])
        all_not_prefix &= bool(not_prefix)

        # 5. round trip
        back = ch.from_canonical16(canon, repo_root=str(REPO))
        rt = np.array_equal(back[:, src], ja[:, src]) and np.isnan(back[:, dropped]).all()
        all_roundtrip_ok &= bool(rt)

        n_invalid = int((~valid).sum())
        tot_invalid += n_invalid
        ev["per_file"].append(
            {
                "filename": f.name,
                "n_samples": int(n),
                "n_ik_failure": n_invalid,
                "frac_ik_failure": n_invalid / n,
                "values_exact": bool(exact),
                "mask_agrees_with_official": bool(agree),
                "not_positional_prefix": bool(not_prefix),
                "roundtrip_ok": bool(rt),
            }
        )

    ev.update(
        {
            "n_samples_total": int(tot),
            "n_ik_failure_total": int(tot_invalid),
            "frac_ik_failure_total": tot_invalid / tot,
            "all_values_exact": bool(all_values_exact),
            "all_mask_agree_with_official": bool(all_mask_agree),
            "all_not_positional_prefix": bool(all_not_prefix),
            "all_roundtrip_ok": bool(all_roundtrip_ok),
            "mask_mismatch_files": mask_mismatches,
        }
    )

    w(f"total samples                 : {tot}")
    w(f"IK-failure (invalid) frames   : {tot_invalid} ({100*tot_invalid/tot:.3f}%)")
    w("")
    w(f"[{'PASS' if all_values_exact else 'FAIL'}] retained values bit-identical to official source columns")
    w(f"[{'PASS' if all_mask_agree else 'FAIL'}] validity mask == official get_ik_failures_mask (all {len(files)} files)")
    w(f"[{'PASS' if all_not_prefix else 'FAIL'}] canonical16 is NOT a positional prefix (name-based selection)")
    w(f"[{'PASS' if all_roundtrip_ok else 'FAIL'}] round trip restores kept cols and NaNs dropped cols")
    w("")
    w("sample rows (first file, frame 0 and first IK-failure frame):")
    sess = Emg2PoseSessionData(hdf5_path=files[0])
    ja0 = np.asarray(sess["joint_angles"])
    c0, v0 = ch.to_canonical16(ja0, return_validity=True, repo_root=str(REPO))
    w(f"  joined names : {' '.join(ch.CANONICAL16_SOURCE_JOINTS[:8])} ...")
    w(f"  canonical[0] : {' '.join(f'{x:+.4f}' for x in c0[0][:8])} ...")
    if (~v0).any():
        i_bad = int(np.argmax(~v0))
        w(f"  frame {i_bad} is IK failure -> valid={bool(v0[i_bad])}, canonical row all-zero="
          f"{bool(np.all(c0[i_bad] == 0))} (flagged, not a posture)")

    ok = all_values_exact and all_mask_agree and all_not_prefix and all_roundtrip_ok

    (DOCS / "canonical16_real_data_evidence.json").write_text(
        json.dumps(ev, indent=2), encoding="utf-8"
    )
    (DOCS / "canonical16_real_data_verification.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"\n[written] {DOCS / 'canonical16_real_data_evidence.json'}")
    print(f"RESULT: {'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
