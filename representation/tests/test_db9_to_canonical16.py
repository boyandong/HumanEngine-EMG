"""Tests for the DB9 -> canonical16 adapter.

Two independent lines of evidence, as required:

* **Synthetic values that are unique per channel.** A 22-vector whose entry *i*
  equals ``100 + i`` makes every slot identifiable. Any swap, off-by-one or
  positional-prefix assumption then produces a wrong value that the test can
  name exactly.
* **Real DB9 files**, so the mapping is exercised against what the official
  release actually contains rather than against a mock.

Run:
    python -m pytest emg2pose_handstate/representation/tests/test_db9_to_canonical16.py -v
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))          # .../emg2pose_handstate
sys.path.insert(0, os.path.join(_REPO, "representation"))
# The DB9 reader lives in Session 3's lane; the real-file tests need it.
sys.path.insert(0, os.path.join(_REPO, "posture_decoder"))

import db9_to_canonical16 as d2c  # noqa: E402

REAL_DIR = os.path.join(_REPO, "data", "ninapro_db9", "extracted")
REAL_FILES = sorted(glob.glob(os.path.join(REAL_DIR, "s_*_angles", "*.mat")))
REAL_FILES = [f for f in REAL_FILES if not os.path.basename(f).startswith("._")]

EXPECTED_DB9_COLUMNS = [0, 1, 2, 3, 4, 6, 16, 7, 8, 17, 9, 11, 18, 13, 15, 19]
EXPECTED_NODIP_COLUMNS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 13, 15]


def _frozen_signs(names=None):
    names = names or d2c.canonical16_names()
    return np.array([d2c.FROZEN_DB9_SIGN.get(n, 1.0) for n in names])


def _expected_from_db9(v, cols=EXPECTED_DB9_COLUMNS, names=None):
    """Name-based selection + degrees->radians + the frozen sign transformation.

    Written out independently of the adapter so a bug in the adapter cannot
    satisfy the test.
    """
    return np.asarray([v[c] for c in cols], dtype=float) * _frozen_signs(names)


# ---------------------------------------------------------------------------
# schema / ordering
# ---------------------------------------------------------------------------
def test_canonical16_ordering_matches_the_frozen_emg2pose_adapter():
    assert d2c.canonical16_names() == d2c.PINNED_CANONICAL16_NAMES
    assert len(d2c.canonical16_names()) == 16


def test_frozen_emg2pose_adapter_agrees_when_importable():
    try:
        import canonical_hand as ch
    except Exception:  # noqa: BLE001
        pytest.skip("frozen emg2pose adapter not importable in this environment")
    assert d2c.canonical16_names() == tuple(ch.canonical16_names())
    assert tuple(ch.resolve_source_indices()) == d2c.PINNED_CANONICAL16_SOURCE_INDICES


def test_db9_column_indices_are_pinned():
    assert list(d2c.resolve_db9_columns()) == EXPECTED_DB9_COLUMNS
    assert list(d2c.resolve_db9_columns(d2c.canonical12_noDIP_names())) == \
        EXPECTED_NODIP_COLUMNS


def test_mapping_is_not_a_positional_prefix():
    """The whole point: selection must be name-based, not columns 0..15."""
    cols = list(d2c.resolve_db9_columns())
    assert cols != list(range(16))
    # and it must genuinely interleave: the DIP block comes from columns 16..19
    assert cols[6] == 16 and cols[9] == 17 and cols[12] == 18 and cols[15] == 19


def test_schema_round_trip(tmp_path):
    p = tmp_path / "schema.json"
    d2c.write_schema(str(p))
    s = json.loads(p.read_text(encoding="utf-8"))
    assert s["ndim"] == 16
    assert s["unit"] == "radians"
    assert s["source_unit"] == "degrees"
    assert s["canonical16_names"] == list(d2c.canonical16_names())
    assert s["db9_column_indices"] == EXPECTED_DB9_COLUMNS
    assert s["not_a_positional_prefix"] is True


# ---------------------------------------------------------------------------
# unique synthetic values
# ---------------------------------------------------------------------------
def _unique_vector():
    """22 values that are all distinct, so any slot error is detectable."""
    return np.array([100.0 + i for i in range(22)])


def test_unique_values_land_in_the_right_canonical_slots():
    v = _unique_vector()
    out = d2c.to_canonical16(v, radians=False)
    expected = _expected_from_db9(v)
    assert np.allclose(out, expected)
    # and each value names its own source channel, up to the frozen sign
    signs = _frozen_signs()
    for slot, col in enumerate(EXPECTED_DB9_COLUMNS):
        assert out[slot] == (100.0 + col) * signs[slot], (
            f"canonical slot {slot} ({d2c.canonical16_names()[slot]}) took DB9 "
            f"column {col}, so it should equal {(100.0 + col) * signs[slot]}"
        )


def test_degree_to_radian_conversion_is_exact_and_named():
    v = _unique_vector()
    deg = d2c.to_canonical16(v, radians=False)
    rad = d2c.to_canonical16(v, radians=True)
    assert np.allclose(rad, np.radians(deg), rtol=1e-12, atol=1e-12)
    # a 180-degree input on a non-flipped channel must map to pi
    v2 = np.zeros(22)
    v2[0] = 180.0                     # CMC1_f -> canonical slot 0, not flipped
    assert d2c.to_canonical16(v2)[0] == pytest.approx(np.pi, rel=1e-12)


def test_no_clipping_or_scaling_is_applied():
    """Values far outside any anatomical range must pass through unchanged."""
    v = np.zeros(22)
    v[0] = 1e6                        # canonical slot 0, not sign-flipped
    v[2] = -1e6                       # canonical slot 2, not sign-flipped
    out = d2c.to_canonical16(v, radians=False)
    assert out[0] == 1e6 and out[2] == -1e6
    # the one flipped channel must be flipped and nothing else
    v3 = np.zeros(22)
    v3[1] = 1e6
    assert d2c.to_canonical16(v3, radians=False)[1] == -1e6


def test_frozen_sign_flip_is_applied_to_thumb_ab_adduction():
    """DB9 CMC1_A is negated into the emg2pose-oriented canonical16 frame.

    Frozen by the research lead from geometry (see
    tools/thumb_web_sign_test.py); this test pins it so it cannot drift.
    """
    v = np.zeros(22)
    v[1] = 42.0                       # CMC1_a, canonical slot 1
    out = d2c.to_canonical16(v, radians=False)
    assert out[1] == -42.0, "DB9 CMC1_a must be negated"
    assert d2c.FROZEN_DB9_SIGN["thumb_cmc_ab_adduction"] == -1.0
    # and it must survive into the noDIP control, which also contains slot 1
    assert d2c.to_canonical12_noDIP(v, radians=False)[1] == -42.0


def test_no_other_variable_is_sign_flipped():
    """Exactly one variable is negated; everything else keeps its DB9 polarity."""
    assert set(d2c.FROZEN_DB9_SIGN) == {"thumb_cmc_ab_adduction"}
    v = np.zeros(22)
    for i in (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21):
        v[i] = 17.0
    out16 = d2c.to_canonical16(v, radians=False)
    signs = np.array([-1.0 if n in d2c.FROZEN_DB9_SIGN else 1.0
                      for n in d2c.canonical16_names()])
    expected = np.array([v[c] for c in EXPECTED_DB9_COLUMNS]) * signs
    assert np.allclose(out16, expected)


def test_zero_is_preserved_under_the_sign_flip():
    """A pure sign flip must leave the zero alone."""
    v = np.zeros(22)
    out = d2c.to_canonical16(v)
    assert np.allclose(out, 0.0)


def test_canonical12_noDIP_drops_exactly_the_four_dip():
    v = _unique_vector()
    c16 = d2c.to_canonical16(v, radians=False)
    c12 = d2c.to_canonical12_noDIP(v, radians=False)
    keep = [i for i, n in enumerate(d2c.canonical16_names())
            if n not in d2c.CANONICAL12_NODIP_DROPPED]
    assert np.allclose(c12, c16[keep])
    for n in d2c.CANONICAL12_NODIP_DROPPED:
        assert "dip" in n


def test_arbitrary_leading_dimensions():
    v = np.stack([_unique_vector(), _unique_vector() + 1000.0])
    out = d2c.to_canonical16(v, radians=False)
    assert out.shape == (2, 16)
    assert np.allclose(out[1], _expected_from_db9(v[1]))


def test_canonical15_noThumbAA_drops_exactly_one_variable():
    """The pre-registered thumb-axis control."""
    assert len(d2c.canonical15_noThumbAA_names()) == 15
    assert d2c.CANONICAL15_NOTHUMBAA_DROPPED == ("thumb_cmc_ab_adduction",)
    assert d2c.CANONICAL15_DROPPED_COLUMN == 1
    # it is canonical16 with one column removed, nothing else
    assert d2c.canonical15_noThumbAA_names() == tuple(
        n for n in d2c.canonical16_names() if n != "thumb_cmc_ab_adduction"
    )


def test_canonical15_from_canonical16_is_a_single_column_removal():
    v = _unique_vector()
    c16 = d2c.to_canonical16(v, radians=False)
    c15 = d2c.canonical15_from_canonical16(c16)
    assert c15.shape == (15,)
    assert np.allclose(c15, np.delete(c16, 1))
    # and it agrees with converting DB9 straight into the control
    direct = d2c.to_canonical15_noThumbAA(v, radians=False)
    assert np.allclose(c15, direct)


def test_canonical15_keeps_the_frozen_sign_for_every_retained_variable():
    """Dropping the one flipped variable must not change any other sign."""
    v = _unique_vector()
    c15 = d2c.to_canonical15_noThumbAA(v, radians=False)
    names = d2c.canonical15_noThumbAA_names()
    expected = np.array([v[c] for c in
                         [EXPECTED_DB9_COLUMNS[d2c.canonical16_names().index(n)]
                          for n in names]])
    signs = np.array([d2c.FROZEN_DB9_SIGN.get(n, 1.0) for n in names])
    assert np.allclose(c15, expected * signs)
    # no remaining variable is flipped, since the only flipped one was dropped
    assert all(s == 1.0 for s in signs)


def test_control_representations_are_mutually_consistent():
    """All three are subsets of canonical16's frozen column order."""
    c16 = list(d2c.canonical16_names())
    for names in (d2c.canonical12_noDIP_names(), d2c.canonical15_noThumbAA_names()):
        idx = [c16.index(n) for n in names]
        assert idx == sorted(idx), "a control representation must preserve order"
        assert len(set(names)) == len(names)


def test_wrong_dimension_is_rejected():
    for bad in (np.zeros(16), np.zeros(21), np.zeros(23), np.zeros((3, 20))):
        with pytest.raises(d2c.Db9MappingError):
            d2c.to_canonical16(bad)


def test_validity_mask_ignores_the_officially_dead_channel():
    """MCP2_a is all-NaN but is NOT a canonical16 input, so it must not
    invalidate frames."""
    v = np.tile(_unique_vector(), (3, 1))
    v[:, 5] = np.nan                      # column 5 == MCP2_a
    mask = d2c.db9_validity_mask(v)
    assert mask.all(), "the dead MCP2_a channel must not mark frames invalid"
    # a NaN in a channel canonical16 actually uses must invalidate
    v2 = np.tile(_unique_vector(), (3, 1))
    v2[1, 0] = np.nan                     # CMC1_f
    mask2 = d2c.db9_validity_mask(v2)
    assert mask2.tolist() == [True, False, True]


def test_channel_order_verification_catches_a_swap():
    good = [f"{i+1}:{n}" for i, n in enumerate(d2c.DB9_CHANNEL_ORDER)]
    assert d2c.verify_db9_channel_order(good) == []
    swapped = list(good)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    problems = d2c.verify_db9_channel_order(swapped)
    assert len(problems) == 2


# ---------------------------------------------------------------------------
# real DB9 files
# ---------------------------------------------------------------------------
pytestmark_real = pytest.mark.skipif(
    not REAL_FILES, reason="no extracted DB9 files present"
)


@pytestmark_real
def test_real_files_declare_the_official_channel_order():
    from db9.loader import load_recording

    for path in REAL_FILES[:6]:
        rec = load_recording(path)
        assert d2c.verify_db9_channel_order(rec.order) == [], path


@pytestmark_real
def test_real_files_convert_and_are_finite_in_canonical16():
    from db9.loader import load_recording

    for path in REAL_FILES[:3]:
        rec = load_recording(path)
        c16 = d2c.to_canonical16(rec.angles)
        assert c16.shape == (rec.n_samples, 16)
        assert np.isfinite(c16).all(), f"{path}: canonical16 has non-finite values"
        # magnitudes must be plausible radians, i.e. degrees/57.3
        assert np.abs(c16).max() < 10.0


@pytestmark_real
def test_real_files_agree_with_direct_column_selection():
    """The adapter must equal an independent hand-written column selection."""
    from db9.loader import load_recording

    rec = load_recording(REAL_FILES[0])
    expected = np.radians(rec.angles[:, EXPECTED_DB9_COLUMNS]) * _frozen_signs()
    assert np.allclose(d2c.to_canonical16(rec.angles), expected, rtol=1e-12)


@pytestmark_real
def test_real_the_dead_channel_is_all_nan_and_unused():
    from db9.loader import load_recording

    rec = load_recording(REAL_FILES[0])
    assert np.isnan(rec.angles[:, 5]).all(), "MCP2_a should be entirely NaN"
    assert 5 not in EXPECTED_DB9_COLUMNS
    assert 5 not in EXPECTED_NODIP_COLUMNS
