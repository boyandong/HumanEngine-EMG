"""
Tests for the emg2pose 20D -> canonical16 adapter.

Run:
    C:\\emg2pose-venv\\Scripts\\python.exe -m pytest representation/tests -v

Design note
-----------
The central risk this suite guards against is an **index-ordering mistake that
silently passes**. Two independent devices are used:

  * `test_ordering_fingerprint` pins the EXACT resolved index tuple, so any
    re-ordering raises.
  * `test_unique_value_extraction` feeds a 20-vector where every official
    dimension carries a unique, self-describing value and checks each canonical
    slot individually. A swap anywhere fails.

The values live in *value space*, not index space, so the test cannot be passed
by "fixing" an index table -- it compares against values read out of the
official `JOINTS` name list.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).absolute().parent
REPRESENTATION_DIR = HERE.parent
sys.path.insert(0, str(REPRESENTATION_DIR))

import canonical_hand as ch  # noqa: E402

REPO_ROOT = REPRESENTATION_DIR.parent / "baseline" / "emg2pose"


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


def official_pairs() -> list[tuple[str, int]]:
    return list(ch.load_official_joints(str(REPO_ROOT)))


def official_name_to_index() -> dict[str, int]:
    return dict(official_pairs())


@pytest.fixture(scope="module")
def layout_report() -> dict:
    return ch.assert_official_layout(str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# 1. official layout resolution + validation
# --------------------------------------------------------------------------- #


def test_official_layout_is_20d(layout_report):
    assert layout_report["n_official"] == ch.EMG2POSE_NDIM == 20


def test_source_indices_match_official_joint_definition():
    """Indices are resolved from JOINTS by name -- verify against the file itself."""
    by_name = official_name_to_index()
    resolved = ch.resolve_source_indices(str(REPO_ROOT))
    for canonical_i, joint_name in enumerate(ch.CANONICAL16_SOURCE_JOINTS):
        assert resolved[canonical_i] == by_name[joint_name], (
            f"canonical slot {canonical_i} ({joint_name}) resolved to "
            f"{resolved[canonical_i]} but official index is {by_name[joint_name]}"
        )


def test_ordering_fingerprint(layout_report):
    """Pin the exact expected layout so any reordering fails loudly."""
    assert layout_report["source_indices"] == [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    assert layout_report["dropped_indices"] == [4, 8, 12, 16]


def test_official_joint_order_names(layout_report):
    assert layout_report["official_joint_order"] == [
        "THUMB_CMC_FE", "THUMB_CMC_AA", "THUMB_MCP_FE", "THUMB_IP_FE",
        "INDEX_MCP_AA", "INDEX_MCP_FE", "INDEX_PIP_FE", "INDEX_DIP_FE",
        "MIDDLE_MCP_AA", "MIDDLE_MCP_FE", "MIDDLE_PIP_FE", "MIDDLE_DIP_FE",
        "RING_MCP_AA", "RING_MCP_FE", "RING_PIP_FE", "RING_DIP_FE",
        "PINKY_MCP_AA", "PINKY_MCP_FE", "PINKY_PIP_FE", "PINKY_DIP_FE",
    ]


def test_error_if_official_repo_missing(tmp_path):
    with pytest.raises(ch.CanonicalMappingError):
        ch.load_official_joints.__wrapped__(str(tmp_path / "nope"))


def test_kept_and_dropped_partition_all_20(layout_report):
    assert set(layout_report["source_indices"]) | set(layout_report["dropped_indices"]) == set(range(20))
    assert not (set(layout_report["source_indices"]) & set(layout_report["dropped_indices"]))


def test_dropped_are_exactly_the_non_thumb_aa():
    dropped_names = list(ch.CANONICAL16_DROPPED_JOINTS)
    assert dropped_names == ["INDEX_MCP_AA", "MIDDLE_MCP_AA", "RING_MCP_AA", "PINKY_MCP_AA"]
    for n in dropped_names:
        assert n not in ch.CANONICAL16_SOURCE_JOINTS
    # the ONLY retained AA variable is the thumb's
    retained_aa = [n for n in ch.CANONICAL16_SOURCE_JOINTS if n.endswith("_AA")]
    assert retained_aa == ["THUMB_CMC_AA"]


# --------------------------------------------------------------------------- #
# 2. synthetic unique-value extraction (catches any swap)
# --------------------------------------------------------------------------- #


def _unique_20() -> np.ndarray:
    """A 20-vector where dim i holds a unique, self-describing value.

    We deliberately use values keyed off the OFFICIAL INDEX so that a wrong
    index mapping produces a wrong value, which is what we assert on.
    """
    return np.array([100.0 + i for i in range(20)], dtype=np.float64)


def test_unique_value_extraction():
    v = _unique_20()
    out = ch.to_canonical16(v, repo_root=str(REPO_ROOT))
    assert out.shape == (16,)

    by_name = official_name_to_index()
    for canonical_i, joint_name in enumerate(ch.CANONICAL16_SOURCE_JOINTS):
        expected = 100.0 + by_name[joint_name]
        assert out[canonical_i] == expected, (
            f"canonical[{canonical_i}] ({joint_name}) = {out[canonical_i]}, "
            f"expected {expected} (official index {by_name[joint_name]})"
        )


def test_unique_value_extraction_is_not_a_positional_prefix():
    """A naive im2col/reshape would give dims 0..15; assert we are NOT doing that."""
    v = _unique_20()
    out = ch.to_canonical16(v, repo_root=str(REPO_ROOT))
    assert not np.array_equal(out, v[:16]), (
        "canonical16 equals the positional prefix of the 20D vector -- this means "
        "selection is NOT being done by joint name"
    )
    # the 4 dropped ab/adduction values must be absent
    for dropped_name in ch.CANONICAL16_DROPPED_JOINTS:
        assert (100.0 + official_name_to_index()[dropped_name]) not in out


def test_batched_shapes_and_values():
    v = np.stack([_unique_20(), _unique_20() + 1000.0])  # (2,20)
    out = ch.to_canonical16(v, repo_root=str(REPO_ROOT))
    assert out.shape == (2, 16)
    assert out[1, 0] == out[0, 0] + 1000.0

    t = np.stack([np.tile(_unique_20(), (5, 1))])  # (1,5,20)
    out3 = ch.to_canonical16(t, repo_root=str(REPO_ROOT))
    assert out3.shape == (1, 5, 16)


# --------------------------------------------------------------------------- #
# 3. exact value preservation (no clipping / projection / rescaling)
# --------------------------------------------------------------------------- #


def test_values_preserved_exactly():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(7, 20)) * 3.0  # deliberately spans far outside joint limits
    out = ch.to_canonical16(v, repo_root=str(REPO_ROOT))
    src = ch.resolve_source_indices(str(REPO_ROOT))
    assert np.array_equal(out, v[:, list(src)]), "values were modified during extraction"
    # bit-exact, including out-of-anatomical-range magnitudes
    assert out.dtype == v.dtype


def test_no_clipping_applied():
    v = np.zeros((1, 20))
    v[0, 5] = 99.0   # INDEX_MCP_FE far beyond any anatomical limit
    v[0, 6] = -99.0  # INDEX_PIP_FE far below
    out = ch.to_canonical16(v, repo_root=str(REPO_ROOT))
    assert out[0, 4] == 99.0
    assert out[0, 5] == -99.0


def test_extreme_and_special_values_pass_through():
    v = np.zeros((1, 20))
    v[0, 2] = 1e12
    v[0, 9] = -1e12
    out = ch.to_canonical16(v, repo_root=str(REPO_ROOT))
    assert out[0, 2] == 1e12
    assert out[0, 7] == -1e12


# --------------------------------------------------------------------------- #
# 4. dimension rejection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [(19,), (21,), (16,), (0,), (2, 19), (2, 21)])
def test_wrong_final_dimension_rejected(bad):
    with pytest.raises(ValueError, match="final dimension"):
        ch.to_canonical16(np.zeros(bad), repo_root=str(REPO_ROOT))


def test_scalar_rejected():
    with pytest.raises(ValueError):
        ch.to_canonical16(np.float64(1.0), repo_root=str(REPO_ROOT))


def test_from_canonical16_rejects_wrong_dims():
    with pytest.raises(ValueError, match="final dimension"):
        ch.from_canonical16(np.zeros((20,)), repo_root=str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# 5. IK-failure semantics
# --------------------------------------------------------------------------- #


def test_all_zero_row_is_invalid():
    v = np.zeros((3, 20))
    v[1, 3] = 0.4  # frame 1 is a real pose
    values, valid = ch.to_canonical16(v, repo_root=str(REPO_ROOT), return_validity=True)
    assert valid.tolist() == [False, True, False]
    assert values.shape == (3, 16)


def test_ik_failure_row_is_not_laundered_into_a_posture():
    """Invalid frames must stay flagged -- never presented as a legitimate pose."""
    v = np.zeros((1, 20))
    res = ch.extract(v, repo_root=str(REPO_ROOT))
    assert res.valid.tolist() == [False]
    assert res.n_invalid == 1
    # the values are zeros (raw passthrough) but the mask says: do not trust this
    assert np.array_equal(res.values, np.zeros((1, 16)))


def test_validity_matches_official_get_ik_failures_mask():
    """Our mask must agree with the official emg2pose helper."""
    e2p = REPO_ROOT
    sys.path.insert(0, str(e2p))
    from emg2pose.utils import get_ik_failures_mask  # noqa: E402

    rng = np.random.default_rng(3)
    v = rng.normal(size=(50, 20))
    v[7] = 0.0
    v[23] = 0.0
    v[24] = 1e-12  # near-zero but not exactly zero
    ours = ch.validity_mask(v, atol=1e-8)
    theirs = get_ik_failures_mask(v)
    assert np.array_equal(ours, theirs)


def test_partially_zero_row_is_valid():
    """Only a row that is ALL zero is an IK failure."""
    v = np.zeros((1, 20))
    v[0, 0] = 1e-6
    assert ch.validity_mask(v, atol=1e-8).tolist() == [True]


def test_float_epsilon_noise_still_counts_as_ik_failure():
    """Regression test from real data.

    MEASURED: in the official mini dataset, 2 of 109 439 flagged frames in one
    session carry float-epsilon noise (max |value| = 1.11e-16, ~2 ULP) rather
    than bit-exact zeros. The official mask (`np.isclose`) classifies these as IK
    failures, which is correct -- they are numerically zero. Our mask must agree.
    """
    v = np.zeros((1, 20))
    v[0, 3] = np.finfo(np.float64).eps  # 2.22e-16, matches the real-data magnitude
    assert ch.validity_mask(v).tolist() == [False]
    assert ch.validity_mask(v, atol=1e-8).tolist() == [False]
    # but a genuinely small-but-real angle is NOT a failure
    v2 = np.zeros((1, 20))
    v2[0, 3] = 1e-6
    assert ch.validity_mask(v2).tolist() == [True]


def test_validity_shape_for_batched_input():
    v = np.zeros((2, 4, 20))
    v[0, 0, 5] = 0.3
    m = ch.validity_mask(v)
    assert m.shape == (2, 4)
    assert m[0, 0] and not m[1, 2]


def test_ik_mask_indexing_does_not_shift_values():
    """Masking is orthogonal to extraction: same values with/without mask."""
    rng = np.random.default_rng(11)
    v = rng.normal(size=(6, 20))
    v[2] = 0.0
    a = ch.to_canonical16(v, repo_root=str(REPO_ROOT))
    b, m = ch.to_canonical16(v, repo_root=str(REPO_ROOT), return_validity=True)
    assert np.array_equal(a, b)
    assert m.tolist() == [True, True, False, True, True, True]


# --------------------------------------------------------------------------- #
# 6. round trip
# --------------------------------------------------------------------------- #


def test_roundtrip_preserves_kept_dimensions():
    rng = np.random.default_rng(5)
    v = rng.normal(size=(4, 20))
    canon = ch.to_canonical16(v, repo_root=str(REPO_ROOT))
    back = ch.from_canonical16(canon, repo_root=str(REPO_ROOT))
    src = list(ch.resolve_source_indices(str(REPO_ROOT)))
    assert np.array_equal(back[:, src], v[:, src])


def test_roundtrip_marks_dropped_dims_as_nan_by_default():
    v = np.ones((2, 20))
    canon = ch.to_canonical16(v, repo_root=str(REPO_ROOT))
    back = ch.from_canonical16(canon, repo_root=str(REPO_ROOT))
    dropped = list(ch.resolve_dropped_indices(str(REPO_ROOT)))
    assert np.isnan(back[:, dropped]).all(), "dropped dims must not look like real data"
    assert not np.isnan(back[:, list(ch.resolve_source_indices(str(REPO_ROOT)))]).any()


def test_roundtrip_explicit_zero_fill_is_opt_in():
    v = np.ones((1, 20))
    canon = ch.to_canonical16(v, repo_root=str(REPO_ROOT))
    back = ch.from_canonical16(canon, fill_dropped=0.0, repo_root=str(REPO_ROOT))
    assert (back[0, list(ch.resolve_dropped_indices(str(REPO_ROOT)))] == 0.0).all()


def test_roundtrip_with_unique_values():
    v = _unique_20()
    canon = ch.to_canonical16(v, repo_root=str(REPO_ROOT))
    back = ch.from_canonical16(canon, repo_root=str(REPO_ROOT))
    by_name = official_name_to_index()
    for canonical_i, joint_name in enumerate(ch.CANONICAL16_SOURCE_JOINTS):
        assert back[by_name[joint_name]] == 100.0 + by_name[joint_name]
    for dropped_name in ch.CANONICAL16_DROPPED_JOINTS:
        assert np.isnan(back[by_name[dropped_name]])


# --------------------------------------------------------------------------- #
# 7. torch support (optional)
# --------------------------------------------------------------------------- #


def test_torch_tensor_input_returns_numpy():
    torch = pytest.importorskip("torch")
    v = torch.arange(20, dtype=torch.float32).reshape(1, 20)
    out = ch.to_canonical16(v, repo_root=str(REPO_ROOT))
    assert isinstance(out, np.ndarray)
    src = ch.resolve_source_indices(str(REPO_ROOT))
    assert np.array_equal(out, np.asarray(v)[:, list(src)])


# --------------------------------------------------------------------------- #
# 8. schema
# --------------------------------------------------------------------------- #


def test_schema_is_consistent_with_resolved_indices():
    schema = ch.build_schema(str(REPO_ROOT))
    assert schema["ndim"] == 16
    assert len(schema["variables"]) == 16
    assert len(schema["dropped_variables"]) == 4

    src = ch.resolve_source_indices(str(REPO_ROOT))
    for i, var in enumerate(schema["variables"]):
        assert var["canonical_index"] == i
        assert var["canonical_name"] == ch.CANONICAL16_NAMES[i]
        assert var["source_emg2pose_joint_name"] == ch.CANONICAL16_SOURCE_JOINTS[i]
        assert var["source_emg2pose_index"] == src[i]
        assert var["unit"] == "radians"

    by_name = official_name_to_index()
    for d in schema["dropped_variables"]:
        assert d["source_emg2pose_index"] == by_name[d["source_emg2pose_joint_name"]]


def test_schema_disclaims_full_hand_geometry():
    schema = ch.build_schema(str(REPO_ROOT))
    assert schema["does_not_fully_describe_hand_geometry"] is True
    assert "LOSSY" in schema["description"]


def test_schema_file_on_disk_matches_generated():
    on_disk = REPRESENTATION_DIR / "canonical16_schema.json"
    assert on_disk.is_file(), "canonical16_schema.json missing -- run canonical_hand.py"
    assert json.loads(on_disk.read_text(encoding="utf-8")) == ch.build_schema(str(REPO_ROOT)), (
        "canonical16_schema.json is stale; regenerate with "
        "`python representation/canonical_hand.py`"
    )


def test_canonical18_is_reserved_and_disabled():
    schema = ch.build_schema(str(REPO_ROOT))
    c18 = schema["reserved_canonical18"]
    assert c18["status"] == "DISABLED"
    assert c18["ndim"] == 18
    names = [v["variable_name"] for v in c18["reserved_variables"]]
    assert names == ["middle_ring_spread", "ring_pinky_spread"]
    for v in c18["reserved_variables"]:
        assert v["status"] == "RESERVED_DISABLED"
    # canonical16 must NOT silently be 18 wide
    assert ch.CANONICAL16_NDIM == 16
    # and no spread formula may be implemented yet
    assert not hasattr(ch, "to_canonical18"), "canonical18 must remain disabled"
