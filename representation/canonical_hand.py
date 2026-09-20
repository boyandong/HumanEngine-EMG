"""
canonical_hand.py -- the authoritative, project-wide emg2pose 20D -> canonical16
adapter.

WHAT THIS IS
------------
`canonical16` is the first cross-dataset hand representation: the 16 kinematic
variables we currently intend to match reliably between **emg2pose** and
**NinaPro DB9**. This module owns the emg2pose side only.

    emg2pose official 20D  --to_canonical16-->  canonical16 (16D)

This is a **lossy, by-design projection**: it DROPS the four non-thumb MCP
abduction/adduction variables. See docs/CANONICAL16_CONTRACT.md.

DESIGN RULES (binding)
-----------------------
1. Selection is by **official joint name resolved against
   ``emg2pose.constants.JOINTS``**, never by a hand-written positional reshape.
   `resolve_source_indices()` performs that resolution and
   `assert_official_layout()` fails loudly if the official layout ever moves.
2. Raw values are preserved **exactly** -- no clipping, anatomy projection,
   normalisation, smoothing, sign correction, or interpolation.
3. Units are **radians**, unchanged.
4. A wrong final dimension is **rejected**, not silently reshaped.
5. Official IK-failure semantics are preserved: an all-zero 20D row is an
   *invalid* frame. It is exposed through a validity mask and is **never**
   presented as a legitimate canonical16 posture.
6. canonical18 (adding middle-ring and ring-pinky spread) is **reserved but
   disabled**. No spread formula is invented here.

Author: Session 2 (emg2pose handstate)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Official emg2pose joint-angle dimensionality.
EMG2POSE_NDIM = 20
#: canonical16 dimensionality.
CANONICAL16_NDIM = 16
#: Reserved (NOT enabled) canonical18 dimensionality.
CANONICAL18_NDIM = 18

UNIT = "radians"

#: The canonical16 ordering, expressed as **official emg2pose joint names**.
#: This tuple *is* the canonical ordering: canonical index i corresponds to
#: CANONICAL16_SOURCE_JOINTS[i].
CANONICAL16_SOURCE_JOINTS: tuple[str, ...] = (
    # --- thumb (4) ---
    "THUMB_CMC_FE",   # 0  CMC flexion
    "THUMB_CMC_AA",   # 1  CMC ab/adduction
    "THUMB_MCP_FE",   # 2  MCP flexion
    "THUMB_IP_FE",    # 3  IP flexion
    # --- index (3) ---
    "INDEX_MCP_FE",   # 4
    "INDEX_PIP_FE",   # 5
    "INDEX_DIP_FE",   # 6
    # --- middle (3) ---
    "MIDDLE_MCP_FE",  # 7
    "MIDDLE_PIP_FE",  # 8
    "MIDDLE_DIP_FE",  # 9
    # --- ring (3) ---
    "RING_MCP_FE",    # 10
    "RING_PIP_FE",    # 11
    "RING_DIP_FE",    # 12
    # --- pinky (3) ---
    "PINKY_MCP_FE",   # 13
    "PINKY_PIP_FE",   # 14
    "PINKY_DIP_FE",   # 15
)

#: Human-readable canonical variable names, parallel to CANONICAL16_SOURCE_JOINTS.
CANONICAL16_NAMES: tuple[str, ...] = (
    "thumb_cmc_flexion",
    "thumb_cmc_ab_adduction",
    "thumb_mcp_flexion",
    "thumb_ip_flexion",
    "index_mcp_flexion",
    "index_pip_flexion",
    "index_dip_flexion",
    "middle_mcp_flexion",
    "middle_pip_flexion",
    "middle_dip_flexion",
    "ring_mcp_flexion",
    "ring_pip_flexion",
    "ring_dip_flexion",
    "pinky_mcp_flexion",
    "pinky_pip_flexion",
    "pinky_dip_flexion",
)

#: Anatomical description of each canonical variable.
CANONICAL16_ANATOMY: tuple[str, ...] = (
    "Thumb carpometacarpal flexion/extension",
    "Thumb carpometacarpal abduction/adduction",
    "Thumb metacarpophalangeal flexion/extension",
    "Thumb interphalangeal flexion/extension",
    "Index metacarpophalangeal flexion/extension",
    "Index proximal interphalangeal flexion/extension",
    "Index distal interphalangeal flexion/extension",
    "Middle metacarpophalangeal flexion/extension",
    "Middle proximal interphalangeal flexion/extension",
    "Middle distal interphalangeal flexion/extension",
    "Ring metacarpophalangeal flexion/extension",
    "Ring proximal interphalangeal flexion/extension",
    "Ring distal interphalangeal flexion/extension",
    "Pinky metacarpophalangeal flexion/extension",
    "Pinky proximal interphalangeal flexion/extension",
    "Pinky distal interphalangeal flexion/extension",
)

#: The four non-thumb MCP abduction/adduction variables intentionally EXCLUDED.
CANONICAL16_DROPPED_JOINTS: tuple[str, ...] = (
    "INDEX_MCP_AA",
    "MIDDLE_MCP_AA",
    "RING_MCP_AA",
    "PINKY_MCP_AA",
)

#: Reserved canonical18 variables. NOT enabled; no formula defined.
CANONICAL18_RESERVED: tuple[dict[str, str], ...] = (
    {
        "canonical_index": "16",
        "variable_name": "middle_ring_spread",
        "status": "RESERVED_DISABLED",
        "blocked_on": "DB9<->emg2pose spread mapping not yet independently validated",
    },
    {
        "canonical_index": "17",
        "variable_name": "ring_pinky_spread",
        "status": "RESERVED_DISABLED",
        "blocked_on": "DB9<->emg2pose spread mapping not yet independently validated",
    },
)


# --------------------------------------------------------------------------- #
# Official layout resolution
# --------------------------------------------------------------------------- #


class CanonicalMappingError(RuntimeError):
    """Raised when the official emg2pose joint layout cannot be reconciled."""


def _default_repo_root() -> Path:
    # NOTE: deliberately not .resolve() -- on this machine C:\emg2pose is an
    # ASCII junction and resolving it dereferences to a non-ASCII path.
    return Path(__file__).absolute().parent.parent / "baseline" / "emg2pose"


@lru_cache(maxsize=4)
def load_official_joints(repo_root: str | None = None) -> tuple[tuple[str, int], ...]:
    """Load ``(name, index)`` pairs from the OFFICIAL ``emg2pose.constants.JOINTS``.

    Loaded via importlib from the file directly so that importing this module
    does not pull in torch / the heavy emg2pose package __init__.
    """
    root = Path(repo_root) if repo_root else _default_repo_root()
    constants_py = root / "emg2pose" / "constants.py"
    if not constants_py.is_file():
        raise CanonicalMappingError(
            f"official emg2pose constants.py not found at {constants_py}. "
            "Clone https://github.com/facebookresearch/emg2pose into "
            "emg2pose_handstate/baseline/emg2pose."
        )
    spec = importlib.util.spec_from_file_location("_canonical_hand_official_constants", constants_py)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise CanonicalMappingError(f"could not load {constants_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return tuple((j.name, int(j.index)) for j in mod.JOINTS)


@lru_cache(maxsize=4)
def resolve_source_indices(repo_root: str | None = None) -> tuple[int, ...]:
    """Resolve canonical16 source indices **by official joint name**.

    Returns a 16-tuple ``source_idx`` such that
    ``canonical16[..., i] == emg2pose20[..., source_idx[i]]``.
    """
    pairs = load_official_joints(repo_root)
    by_name = dict(pairs)
    name_to_official_index = {n: i for (n, i) in pairs}

    missing = [n for n in CANONICAL16_SOURCE_JOINTS if n not in by_name]
    if missing:
        raise CanonicalMappingError(
            "canonical16 requires official emg2pose joints that are absent from "
            f"emg2pose.constants.JOINTS: {missing}. The official layout changed; "
            "canonical16 must be re-validated by the research lead."
        )

    # The official index stored on each Joint must agree with its list position.
    for pos, (name, idx) in enumerate(pairs):
        if pos != idx:
            raise CanonicalMappingError(
                f"official JOINTS list position {pos} != Joint.index {idx} for "
                f"{name!r}; emg2pose's own indexing is no longer self-consistent."
            )

    resolved = tuple(name_to_official_index[n] for n in CANONICAL16_SOURCE_JOINTS)
    if len(resolved) != CANONICAL16_NDIM:
        raise CanonicalMappingError(
            f"canonical16 resolves to {len(resolved)} source indices, expected {CANONICAL16_NDIM}"
        )
    if len(set(resolved)) != len(resolved):
        raise CanonicalMappingError(f"canonical16 resolves to duplicate source indices: {resolved}")
    return resolved


@lru_cache(maxsize=4)
def resolve_dropped_indices(repo_root: str | None = None) -> tuple[int, ...]:
    """Official indices of the 4 variables canonical16 drops."""
    pairs = load_official_joints(repo_root)
    by_name = dict(pairs)
    missing = [n for n in CANONICAL16_DROPPED_JOINTS if n not in by_name]
    if missing:
        raise CanonicalMappingError(f"dropped-joint names absent from official JOINTS: {missing}")
    return tuple(by_name[n] for n in CANONICAL16_DROPPED_JOINTS)


def assert_official_layout(repo_root: str | None = None) -> dict[str, Any]:
    """Validate that the resolved mapping still equals the expected official layout.

    Raises CanonicalMappingError on any mismatch. Returns a small report dict.
    """
    pairs = load_official_joints(repo_root)
    src = resolve_source_indices(repo_root)
    dropped = resolve_dropped_indices(repo_root)

    if len(pairs) != EMG2POSE_NDIM:
        raise CanonicalMappingError(
            f"official NUM_JOINTS changed: got {len(pairs)} joint entries, "
            f"canonical16 assumes {EMG2POSE_NDIM}."
        )

    # The four dropped variables are exactly the non-thumb MCP ab/adduction DOF.
    if set(src) | set(dropped) != set(range(EMG2POSE_NDIM)):
        raise CanonicalMappingError(
            "canonical16 + dropped variables do not partition the official 20 dims. "
            f"kept={sorted(src)} dropped={sorted(dropped)}"
        )

    interleaved_expected = (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)
    if src != interleaved_expected:
        raise CanonicalMappingError(
            "resolved canonical16 source indices differ from the expected official "
            f"layout.\n  expected: {interleaved_expected}\n  resolved: {src}\n"
            "This means the official emg2pose JOINTS ordering moved. Do NOT proceed "
            "by editing the expected tuple blindly -- re-verify the joint semantics."
        )

    if dropped != (4, 8, 12, 16):
        raise CanonicalMappingError(
            f"dropped indices differ from expected (4, 8, 12, 16): got {dropped}"
        )

    return {
        "n_official": len(pairs),
        "source_indices": list(src),
        "dropped_indices": list(dropped),
        "official_joint_order": [n for (n, _) in pairs],
    }


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def build_schema(repo_root: str | None = None) -> dict[str, Any]:
    """Build the authoritative canonical16 schema (nothing hard-coded)."""
    assert_official_layout(repo_root)
    src = resolve_source_indices(repo_root)
    dropped = resolve_dropped_indices(repo_root)
    pairs = dict(load_official_joints(repo_root))

    variables = []
    for i in range(CANONICAL16_NDIM):
        jname = CANONICAL16_SOURCE_JOINTS[i]
        variables.append(
            {
                "canonical_index": i,
                "canonical_name": CANONICAL16_NAMES[i],
                "source_emg2pose_joint_name": jname,
                "source_emg2pose_index": src[i],
                "anatomical_meaning": CANONICAL16_ANATOMY[i],
                "unit": UNIT,
                "dof_type": "FE" if jname.endswith("_FE") else "AA",
                "positive_direction": (
                    "flexion" if jname.endswith("_FE")
                    else "per-finger (no global sign); see docs/joint_angle_definition.md 3.2"
                ),
            }
        )

    return {
        "schema_name": "canonical16",
        "version": "1.0.0",
        "status": "FROZEN",
        "ndim": CANONICAL16_NDIM,
        "unit": UNIT,
        "description": (
            "The 16 kinematic variables intended to be matched reliably between "
            "emg2pose and NinaPro DB9. A LOSSY projection of emg2pose's official "
            "20D pose: the four non-thumb MCP abduction/adduction variables are "
            "dropped by design."
        ),
        "does_not_fully_describe_hand_geometry": True,
        "source": {
            "dataset": "emg2pose",
            "official_ndim": EMG2POSE_NDIM,
            "ordering_authority": "emg2pose/constants.py::JOINTS",
            "selection_method": "by official joint name, resolved at runtime",
            "repo_commit_used_for_validation": "5f6f62b",
        },
        "ik_failure_semantics": {
            "encoding": "all-zero joint_angles row",
            "helper": "emg2pose.utils.get_ik_failures_mask",
            "rule": (
                "A 20D row whose 20 values are all (close to) zero is an IK "
                "failure and MUST NOT be treated as a valid posture. It is "
                "surfaced via the validity mask returned by "
                "to_canonical16(..., return_validity=True) / validity_mask()."
            ),
            "not_a_neutral_hand": True,
        },
        "variables": variables,
        "dropped_variables": [
            {
                "source_emg2pose_joint_name": n,
                "source_emg2pose_index": pairs[n],
                "reason": "excluded from canonical16 by design (non-thumb MCP ab/adduction)",
            }
            for n in CANONICAL16_DROPPED_JOINTS
        ],
        "reserved_canonical18": {
            "status": "DISABLED",
            "ndim": CANONICAL18_NDIM,
            "note": (
                "Reserved for a future canonical18. Enabled ONLY after the "
                "DB9<->emg2pose spread mapping has been independently validated. "
                "No spread formula is defined or implemented in this module."
            ),
            "reserved_variables": [dict(v) for v in CANONICAL18_RESERVED],
        },
    }


# --------------------------------------------------------------------------- #
# core transform
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Canonical16:
    """Result of an emg2pose20 -> canonical16 extraction.

    Attributes
    ----------
    values : np.ndarray
        ``(..., 16)`` canonical16 joint angles in **radians** (raw, unmodified).
    valid : np.ndarray
        ``(...)`` bool. ``False`` where the source 20D frame was an IK failure
        (all 20 values ~0). Those rows' ``values`` are **not** a real posture.
    """

    values: np.ndarray
    valid: np.ndarray

    @property
    def n_invalid(self) -> int:
        return int((~self.valid).sum())

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape


def _is_torch(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "torch" and hasattr(x, "detach")


def validity_mask(emg2pose_angles: Any, *, atol: float = 1e-8) -> np.ndarray:
    """Official emg2pose IK-failure mask.

    ``True`` where the 20D frame is a **valid** pose, ``False`` where the frame is
    an all-zero IK failure. Mirrors ``emg2pose.utils.get_ik_failures_mask``
    (``~np.all(np.isclose(joint_angles, 0), axis=-1)``).
    """
    arr = _to_numpy(emg2pose_angles)
    _check_last_dim(arr, EMG2POSE_NDIM, "emg2pose_angles")
    return ~np.all(np.isclose(arr, 0.0, atol=atol), axis=-1)


def to_canonical16(
    emg2pose_angles: Any,
    *,
    return_validity: bool = False,
    repo_root: str | None = None,
    atol: float = 1e-8,
) -> Any:
    """Project official emg2pose 20D joint angles to canonical16.

    Parameters
    ----------
    emg2pose_angles : array-like
        ``(..., 20)`` official emg2pose joint angles in **radians**. numpy array,
        torch tensor, or anything ``np.asarray`` understands.
    return_validity : bool
        If True, return ``(canonical16, valid)`` where ``valid`` has shape
        ``(...)`` and is False for IK-failure frames.
    repo_root : str | None
        Override for locating the official emg2pose repo (tests / portability).

    Returns
    -------
    np.ndarray of shape ``(..., 16)`` (or a ``(values, valid)`` tuple)

    Notes
    -----
    * Selection is by official joint NAME (see :func:`resolve_source_indices`).
    * Values are copied **exactly**: no clipping, projection, scaling, sign
      change, smoothing or interpolation.
    * IK-failure frames are returned verbatim (as 16 zeros) but are flagged
      invalid; they are never laundered into a plausible posture.
    """
    src = resolve_source_indices(repo_root)
    arr = _to_numpy(emg2pose_angles)
    _check_last_dim(arr, EMG2POSE_NDIM, "emg2pose_angles")

    canon = np.take(arr, np.asarray(src, dtype=np.intp), axis=-1)

    if not return_validity:
        return canon
    return canon, validity_mask(arr, atol=atol)


def extract(emg2pose_angles: Any, *, repo_root: str | None = None, atol: float = 1e-8) -> Canonical16:
    """Convenience wrapper returning a :class:`Canonical16` (values + validity)."""
    values, valid = to_canonical16(
        emg2pose_angles, return_validity=True, repo_root=repo_root, atol=atol
    )
    return Canonical16(values=values, valid=valid)


def from_canonical16(
    canonical16: Any,
    *,
    fill_dropped: float = np.nan,
    repo_root: str | None = None,
) -> np.ndarray:
    """Inverse, partial map canonical16 -> emg2pose20.

    The four dropped non-thumb MCP ab/adduction slots are **unrecoverable** and
    are filled with ``fill_dropped`` (default ``NaN``) so they cannot be mistaken
    for real measurements. Pass ``fill_dropped=0.0`` only if a downstream
    consumer explicitly requires zeros AND knows they are fabricated.

    This exists for round-trip testing and for feeding pose visualisers; it is
    NOT a basis for claiming a full 20D emg2pose pose is recoverable.
    """
    src = resolve_source_indices(repo_root)
    dropped = resolve_dropped_indices(repo_root)
    arr = _to_numpy(canonical16)
    _check_last_dim(arr, CANONICAL16_NDIM, "canonical16")

    out = np.full(arr.shape[:-1] + (EMG2POSE_NDIM,), fill_dropped, dtype=arr.dtype)
    out[..., np.asarray(src, dtype=np.intp)] = arr[..., :]
    # `dropped` already holds NaN; kept explicit for readability.
    out[..., np.asarray(dropped, dtype=np.intp)] = fill_dropped
    return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _to_numpy(x: Any) -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    # Lists of python floats, etc. Use float64 to avoid precision surprises.
    return np.asarray(x)


def _check_last_dim(arr: np.ndarray, expected: int, what: str) -> None:
    if arr.ndim == 0:
        raise ValueError(f"{what} must have at least 1 dimension; got a scalar")
    if arr.shape[-1] != expected:
        raise ValueError(
            f"{what} must have final dimension {expected}, got shape {tuple(arr.shape)} "
            f"(final dim = {arr.shape[-1]}). Refusing to reshape or guess."
        )


def canonical16_names() -> list[str]:
    """Canonical variable names in canonical order."""
    return list(CANONICAL16_NAMES)


def canonical16_source_joint_names() -> list[str]:
    """Official emg2pose joint names in canonical order."""
    return list(CANONICAL16_SOURCE_JOINTS)


def write_schema(path: str | Path, repo_root: str | None = None) -> Path:
    """Write the canonical16 schema JSON (regenerated from official source)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(build_schema(repo_root), indent=2) + "\n", encoding="utf-8")
    return p


__all__ = [
    "CANONICAL16_ANATOMY",
    "CANONICAL16_DROPPED_JOINTS",
    "CANONICAL16_NAMES",
    "CANONICAL16_NDIM",
    "CANONICAL16_SOURCE_JOINTS",
    "CANONICAL18_NDIM",
    "CANONICAL18_RESERVED",
    "Canonical16",
    "CanonicalMappingError",
    "EMG2POSE_NDIM",
    "UNIT",
    "assert_official_layout",
    "build_schema",
    "canonical16_names",
    "canonical16_source_joint_names",
    "extract",
    "from_canonical16",
    "load_official_joints",
    "resolve_dropped_indices",
    "resolve_source_indices",
    "to_canonical16",
    "validity_mask",
    "write_schema",
]


if __name__ == "__main__":  # pragma: no cover
    out = write_schema(Path(__file__).absolute().parent / "canonical16_schema.json")
    rep = assert_official_layout()
    print(f"[ok] official layout validated: {rep['n_official']} dims")
    print(f"[ok] canonical16 source indices: {rep['source_indices']}")
    print(f"[ok] dropped indices           : {rep['dropped_indices']}")
    print(f"[written] {out}")
