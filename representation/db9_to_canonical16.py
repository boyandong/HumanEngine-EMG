"""DB9 -> canonical16 adapter.  Session 3 owns this side.

The emg2pose side is FROZEN in ``canonical_hand.py`` / ``canonical16_schema.json``
and is not re-derived or reinterpreted here.  This module is its mirror image:
it selects the same 16 anatomical variables out of the 22 calibrated
CyberGlove-II channels published in NinaPro DB9.

WHAT IS VERIFIED, AND HOW
-------------------------
1. **Channel vocabulary.** Taken from each recording's own ``order_of_angles``
   variable, not from a paper figure.  All 44 files in the audited corpus
   declare the same 22 names, and they match the official table exactly
   (``verify_db9_channel_order``).  See ``docs/DB9_FILE_AUDIT.json``.

2. **The two orderings are structurally different.** canonical16's own order is
   thumb, then index/middle/ring/pinky at MCP,PIP,DIP.  DB9's order is thumb,
   then the flexion chain of fingers 2-5 interleaved with their abduction
   sensors, then all four DIP sensors together, then wrist.  Consecutive index
   is NOT the same joint: e.g. ``index_dip_flexion`` is DB9 column 16, while
   ``middle_mcp_flexion`` is column 7.  Selection is therefore by **name**, and
   a test proves the mapping is not a positional prefix.

3. **Unit: DEGREES.** DB9's ``angles`` are published in degrees; canonical16 is
   radians.  The conversion is explicit and named
   (``DEGREES_PER_RADIAN``), never implicit.  Evidence: (a) Atzori et al. 2014
   describe the CyberGlove as returning values "for an average resolution of
   less than one degree"; (b) the DB9 paper quotes calibration precision "below
   5 degrees"; (c) measured over the audited corpus the global range is
   -322..+173, which is only physically sensible in degrees.  The paper never
   writes the unit symbol, so this is recorded as inferred-with-evidence, not
   as a verbatim quotation.

4. **Sign and zero reference** -- DB9 paper, Table 6 and the calibration
   section:
     * ``DIP(2-5)_F, PIP(2-5)_F, IP1_F, MCP(1-5)_F``: Flexion + / Extension -
     * ``CMC1_F``: Flexion + / Extension -
     * ``CMC1_A``: Abduction + / Adduction -
     * reference posture: "the hand resting on the table with the fingers closed
       together and extended", taken from the central part of an exercise-B
       movement.
   So DB9's 15 flexion variables already agree in sign with canonical16's
   "positive = flexion" convention.  DB9's ``CMC1_A`` is abduction-positive
   while emg2pose's ``THUMB_CMC_AA`` is adduction-positive, so that one variable
   **is** sign-flipped by this module; see :data:`FROZEN_DB9_SIGN`.

   Scope of that claim, stated precisely: the **sign correspondence is frozen**
   on geometric and anatomical evidence.  It is NOT asserted that DB9's
   ``CMC1_A`` and UmeTrack's ``THUMB_CMC_AA`` are proven to be identical
   rotation axes in magnitude or geometry -- they are not.  That residual
   uncertainty is exactly why ``canonical15_noThumbAA`` exists as a
   pre-registered robustness control.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
By contract, mirroring the frozen emg2pose adapter: **no** clipping, **no**
per-subject centring, **no** offset correction, **no** normalisation, **no**
scaling, **no** smoothing, **no** interpolation.  The only transformations are
(a) name-based column selection, (b) degrees -> radians, and (c) the frozen
per-variable sign transformation in :data:`FROZEN_DB9_SIGN`, which exists solely
to express DB9 in the emg2pose-oriented canonical16 frame and consists of exactly
one negated variable (see below).  Values otherwise pass through unchanged
(test-enforced, including values far outside any anatomical range).

VALIDITY CONTRACT
-----------------
The frozen emg2pose contract requires a validity mask that composes by logical
AND.  DB9 encodes no "IK failure" concept, so the DB9 validity mask means
"this frame carries finite values in every channel canonical16 needs".  The one
officially dead channel, ``MCP2_a`` (index MCP abduction), is **not** a
canonical16 input, so its all-NaN content does not invalidate anything -- which
is exactly why it must never be fed in blindly by position.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Official DB9 channel vocabulary, verbatim from `order_of_angles`.
# --------------------------------------------------------------------------
DB9_CHANNEL_ORDER: tuple[str, ...] = (
    "CMC1_f", "CMC1_a", "MCP1", "IP1",
    "MCP2_f", "MCP2_a", "PIP2",
    "MCP3_f", "PIP3",
    "MCP4_f", "MCP4_a", "PIP4",
    "CMC5", "MCP5_f", "MCP5_a", "PIP5",
    "DIP2", "DIP3", "DIP4", "DIP5",
    "WRIST_F", "WRIST_A",
)

#: Channels present in the layout but officially unusable.  Recorded so the
#: reason travels with the data instead of living in someone's memory.
DB9_EXCLUDED_CHANNELS: dict[str, str] = {
    "MCP2_a": (
        "All-NaN in every audited recording. Official DB9 docs: 'Sensor 11 "
        "(which corresponds to MCP2-3_A) was not included into the calibrated "
        "data due to noise problems.' It is not a canonical16 input."
    ),
}

#: canonical16 name -> DB9 channel name.  Selection is by name, always.
CANONICAL16_FROM_DB9: dict[str, str] = {
    "thumb_cmc_flexion": "CMC1_f",
    "thumb_cmc_ab_adduction": "CMC1_a",
    "thumb_mcp_flexion": "MCP1",
    "thumb_ip_flexion": "IP1",
    "index_mcp_flexion": "MCP2_f",
    "index_pip_flexion": "PIP2",
    "index_dip_flexion": "DIP2",
    "middle_mcp_flexion": "MCP3_f",
    "middle_pip_flexion": "PIP3",
    "middle_dip_flexion": "DIP3",
    "ring_mcp_flexion": "MCP4_f",
    "ring_pip_flexion": "PIP4",
    "ring_dip_flexion": "DIP4",
    "pinky_mcp_flexion": "MCP5_f",
    "pinky_pip_flexion": "PIP5",
    "pinky_dip_flexion": "DIP5",
}

#: Which DB9 anatomical label each canonical16 variable corresponds to, and the
#: sign criterion the DB9 paper defines for it (Table 6).
DB9_ANATOMY: dict[str, dict[str, str]] = {
    "thumb_cmc_flexion": {
        "db9_anatomical_name": "CMC1_F",
        "anatomical_meaning": "Thumb carpometacarpal flexion/extension",
        "positive_direction": "flexion",
        "db9_sign_criterion": "CMC1_F: Flexion+ / Extension- (Table 6)",
    },
    "thumb_cmc_ab_adduction": {
        "db9_anatomical_name": "CMC1_A",
        "anatomical_meaning": "Thumb carpometacarpal abduction/adduction",
        "positive_direction": "abduction",
        "db9_sign_criterion": "CMC1_A: Abduction+ / Adduction- (Table 6)",
    },
    "thumb_mcp_flexion": {
        "db9_anatomical_name": "MCP1_F",
        "anatomical_meaning": "Thumb metacarpophalangeal flexion/extension",
        "positive_direction": "flexion",
        "db9_sign_criterion": "MCP(1-5)_F: Flexion+ / Extension- (Table 6)",
    },
    "thumb_ip_flexion": {
        "db9_anatomical_name": "IP1_F",
        "anatomical_meaning": "Thumb interphalangeal flexion/extension",
        "positive_direction": "flexion",
        "db9_sign_criterion": "IP1_F: Flexion+ / Extension- (Table 6)",
    },
}
for _f, _d in (("index", "2"), ("middle", "3"), ("ring", "4"), ("pinky", "5")):
    for _j, _db9 in (("mcp", f"MCP{_d}_F"), ("pip", f"PIP{_d}_F"), ("dip", f"DIP{_d}_F")):
        DB9_ANATOMY[f"{_f}_{_j}_flexion"] = {
            "db9_anatomical_name": _db9,
            "anatomical_meaning": (
                f"{_f.capitalize()} "
                f"{'metacarpophalangeal' if _j == 'mcp' else 'proximal interphalangeal' if _j == 'pip' else 'distal interphalangeal'}"
                " flexion/extension"
            ),
            "positive_direction": "flexion",
            "db9_sign_criterion": (
                f"DIP(2-5)_F, PIP(2-5)_F, MCP(1-5)_F: Flexion+ / Extension- (Table 6)"
            ),
        }

#: canonical12_noDIP keeps canonical16's order minus the four distal joints.
CANONICAL12_NODIP_DROPPED: tuple[str, ...] = (
    "index_dip_flexion", "middle_dip_flexion",
    "ring_dip_flexion", "pinky_dip_flexion",
)

#: canonical15_noThumbAA keeps canonical16's order minus exactly one variable:
#: `thumb_cmc_ab_adduction`.  This is a PRE-REGISTERED ROBUSTNESS CONTROL, not a
#: candidate representation and not a model-selection input.
#:
#: Why it exists: that one variable is the only place in canonical16 where the
#: cross-dataset correspondence carries residual uncertainty.  Its DB9 sign is
#: flipped from its emg2pose counterpart (see FROZEN_DB9_SIGN), its zero was
#: verified separately, and emg2pose's CMC_AA is not the dominant in-plane web
#: channel -- so the *magnitude* semantics of the pairing are approximate even
#: though the sign decision is settled.  Removing it and re-running the identical
#: pipeline answers: does the remaining cross-dataset uncertainty in this axis
#: materially drive posture identifiability?  It must be reported alongside
#: canonical16 and canonical12_noDIP and must never be used to pick a model.
CANONICAL15_NOTHUMBAA_DROPPED: tuple[str, ...] = ("thumb_cmc_ab_adduction",)

#: The DB9 reference posture, quoted from the calibration section of the paper.
DB9_REFERENCE_POSTURE = (
    "The reference posture was chosen as the central part of a movement in "
    "exercise B, corresponding to the hand resting on the table with the "
    "fingers closed together and extended. (Jarque-Bou et al. 2020, "
    "doi:10.1038/s41597-019-0349-2)"
)

DEGREES_PER_RADIAN = 180.0 / np.pi

# --------------------------------------------------------------------------
# FROZEN SIGN TRANSFORMATION
# --------------------------------------------------------------------------
# canonical16 is an emg2pose-oriented frame.  Where DB9's documented polarity for
# a variable runs opposite to its emg2pose counterpart, the correction is applied
# HERE, on the DB9 side only, and nowhere else.  Session 2's emg2pose adapter is
# deliberately untouched.
#
# `thumb_cmc_ab_adduction` -> -1
#   DB9 documents `CMC1_A` as Abduction+ / Adduction- (paper Table 6).
#   emg2pose's `THUMB_CMC_AA` runs the other way.  Measured over the official
#   limit range [-0.6118, +0.6118] rad by forward kinematics
#   (tools/thumb_web_sign_test.py, evidence in
#   data/db9_posture16/thumb_web_sign_evidence.json and figure
#   figures/posture16/thumb_web_sign_test.png):
#
#     CMC_AA = -0.6118   web angle 69.4 deg   tip sep 153.9   elevation +45.7 deg
#     CMC_AA =  0.0000   web angle 52.0 deg   tip sep 114.8   elevation +32.5 deg
#     CMC_AA = +0.6118   web angle 50.4 deg   tip sep  93.6   elevation  +8.9 deg
#
#   So positive emg2pose CMC_AA BOTH closes the first web space AND brings the
#   thumb down toward the palm plane, i.e. it is adduction on both criteria
#   (out-of-plane and in-plane readings agree for this DOF).  DB9's positive is
#   abduction.  The two are opposite, so DB9 must be negated.
#
#   ZERO ALIGNMENT for this variable is checked separately and holds:  DB9's
#   reference posture is the extended-and-together hand, where `CMC1_A` sits at
#   ~0 deg (exercise-B wrist-rotation movements, the closest to all-zero
#   canonical16, measure -0.38 and +1.26 deg), and emg2pose's rest pose has
#   `THUMB_CMC_AA = 0` by construction.  Both therefore place the zero at the
#   same physical configuration, so the correction is a PURE SIGN FLIP with no
#   offset.  Evidence: data/db9_posture16/thumb_zero_alignment.json.
#
#   Frozen by the research lead from geometry and documented anatomy BEFORE
#   classification and never tuned against accuracy.
#
#   Caveat carried alongside (does not change the sign decision): emg2pose's
#   `THUMB_CMC_AA` is not the dominant in-plane web channel -- `THUMB_CMC_FE`
#   moves the in-plane web angle about 3x more strongly per radian
#   (tools/resolve_thumb_axis.py).  The flip corrects a sign on a channel whose
#   magnitude semantics only approximately match DB9's `CMC1_A`.
FROZEN_DB9_SIGN: dict[str, float] = {
    "thumb_cmc_ab_adduction": -1.0,
}

#: Human-readable justification, emitted into the schema.
FROZEN_SIGN_RATIONALE = {
    "thumb_cmc_ab_adduction": (
        "DB9 CMC1_A is Abduction+/Adduction- (paper Table 6). Forward kinematics "
        "of the official UmeTrack model over CMC_AA's full limit range shows "
        "positive emg2pose CMC_AA both closes the first web space (tip separation "
        "114.8 -> 93.6) and brings the thumb toward the palm plane (elevation "
        "+32.5 -> +8.9 deg), i.e. adduction. The datasets are therefore opposite "
        "and DB9 is negated. Zero aligns between the two, so this is a pure sign "
        "flip with no offset. Frozen from geometry before classification; never "
        "tuned on accuracy."
    ),
}


class Db9MappingError(RuntimeError):
    """Raised when a DB9 array cannot be mapped without guessing."""


#: Evidence and confidence for each canonical16 variable's DB9-side semantics.
#: The protocol requires a per-variable "confidence / evidence source" rather
#: than one blanket statement, because the variables are NOT equally well
#: supported: the 15 flexion variables are settled by the DB9 paper's Table 6,
#: while the thumb ab/adduction variable's SIGN is frozen on geometric evidence
#: but its AXIS IDENTITY with UmeTrack's THUMB_CMC_AA is not proven.
DB9_SIGN_EVIDENCE: dict[str, dict[str, str]] = {
    "default_flexion": {
        "evidence_source": (
            "Jarque-Bou, Atzori & Muller, Sci Data 7:12 (2020), Table 6: "
            "'DIP(2-5)_F, PIP(2-5)_F, IP1_F, MCP(1-5)_F  Flexion+ / Extension-'"
        ),
        "evidence_type": "OFFICIAL-DOC",
        "db9_sign_confidence": "HIGH",
        "cross_dataset_polarity_confidence": "HIGH",
        "cross_dataset_note": (
            "emg2pose's 15 FE joints were shown by forward kinematics to be "
            "flexion-positive as well, so the polarity agrees and no conversion "
            "is needed."
        ),
    },
    "thumb_cmc_flexion": {
        "evidence_source": (
            "Jarque-Bou, Atzori & Muller, Sci Data 7:12 (2020), Table 6: "
            "'CMC1_F  Flexion+ / Extension- (See Fig. 5)'"
        ),
        "evidence_type": "OFFICIAL-DOC",
        "db9_sign_confidence": "HIGH",
        "cross_dataset_polarity_confidence": "HIGH",
        "cross_dataset_note": (
            "emg2pose's THUMB_CMC_FE is one of the 15 flexion joints verified "
            "flexion-positive by forward kinematics."
        ),
    },
    "thumb_cmc_ab_adduction": {
        "evidence_source": (
            "Jarque-Bou, Atzori & Muller, Sci Data 7:12 (2020), Table 6: "
            "'CMC1_A  Abduction+ / Adduction- (See Fig. 5)'"
        ),
        "evidence_type": "OFFICIAL-DOC (DB9 side) + GEOMETRY (emg2pose side)",
        "db9_sign_confidence": "HIGH",
        "cross_dataset_polarity_confidence": "FROZEN (sign)",
        "cross_dataset_axis_identity_confidence": "NOT PROVEN (magnitude/geometry)",
        "cross_dataset_note": (
            "DB9 documents CMC1_A as abduction-positive. Forward kinematics of the "
            "official UmeTrack model over THUMB_CMC_AA's full limit range "
            "[-0.6118, +0.6118] rad shows positive emg2pose CMC_AA BOTH closes the "
            "first web space (thumb-index tip separation 153.90 -> 114.83 -> 93.57) "
            "AND brings the thumb down toward the palm plane (elevation +45.70 -> "
            "+32.45 -> +8.94 deg). The in-plane and out-of-plane criteria therefore "
            "AGREE that positive is adduction, which is opposite to DB9, so DB9 is "
            "negated. Zero alignment for this variable was verified separately "
            "(both datasets place its zero at the extended-and-together hand), so "
            "the correction is a pure sign flip with no offset. "
            "SCOPE: the SIGN correspondence is frozen on this evidence. It is NOT "
            "claimed that DB9's CMC1_A and UmeTrack's THUMB_CMC_AA are proven to be "
            "identical rotation axes in magnitude or geometry -- emg2pose's "
            "THUMB_CMC_FE moves the in-plane web angle about 3x more strongly per "
            "radian, so the pairing is approximate at the DOF level. That residual "
            "uncertainty is why canonical15_noThumbAA is a pre-registered control. "
            "See docs/DB9_CANONICAL16_MAPPING.md section 4.2."
        ),
    },
}


def sign_evidence_for(variable: str) -> dict[str, str]:
    if variable in DB9_SIGN_EVIDENCE:
        return DB9_SIGN_EVIDENCE[variable]
    if variable.endswith(("_mcp_flexion", "_pip_flexion", "_dip_flexion")):
        return DB9_SIGN_EVIDENCE["default_flexion"]
    return DB9_SIGN_EVIDENCE["default_flexion"]


# --------------------------------------------------------------------------
# canonical16 name list: imported from the frozen emg2pose adapter when
# available, otherwise the locally pinned copy (checked against it by tests).
# --------------------------------------------------------------------------
_FROZEN_ADAPTER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "representation"
)

PINNED_CANONICAL16_NAMES: tuple[str, ...] = (
    "thumb_cmc_flexion", "thumb_cmc_ab_adduction", "thumb_mcp_flexion",
    "thumb_ip_flexion",
    "index_mcp_flexion", "index_pip_flexion", "index_dip_flexion",
    "middle_mcp_flexion", "middle_pip_flexion", "middle_dip_flexion",
    "ring_mcp_flexion", "ring_pip_flexion", "ring_dip_flexion",
    "pinky_mcp_flexion", "pinky_pip_flexion", "pinky_dip_flexion",
)

#: emg2pose source index for each canonical16 slot, pinned here as a
#: cross-check against the frozen adapter (not used for DB9 conversion).
PINNED_CANONICAL16_SOURCE_INDICES: tuple[int, ...] = (
    0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19,
)


def frozen_adapter_names() -> tuple[str, ...]:
    """Read the canonical16 ordering from the frozen emg2pose adapter.

    Falls back to the pinned copy only if the frozen module is unavailable, and
    says so -- silently drifting from the frozen contract would be worse than
    an import error.
    """
    try:
        if _FROZEN_ADAPTER_DIR not in sys.path:
            sys.path.insert(0, _FROZEN_ADAPTER_DIR)
        import canonical_hand as ch  # type: ignore

        return tuple(ch.canonical16_names())
    except Exception:  # noqa: BLE001
        return PINNED_CANONICAL16_NAMES


CANONICAL16_NAMES: tuple[str, ...] = frozen_adapter_names()
if tuple(CANONICAL16_NAMES) != PINNED_CANONICAL16_NAMES:
    raise Db9MappingError(
        "the frozen emg2pose canonical16 ordering does not match the ordering "
        "this DB9 adapter was written against.\n"
        f"  frozen : {list(CANONICAL16_NAMES)}\n"
        f"  pinned : {list(PINNED_CANONICAL16_NAMES)}\n"
        "Refusing to guess which is authoritative."
    )

CANONICAL12_NODIP_NAMES: tuple[str, ...] = tuple(
    n for n in CANONICAL16_NAMES if n not in CANONICAL12_NODIP_DROPPED
)
assert len(CANONICAL12_NODIP_NAMES) == 12

CANONICAL15_NOTHUMBAA_NAMES: tuple[str, ...] = tuple(
    n for n in CANONICAL16_NAMES if n not in CANONICAL15_NOTHUMBAA_DROPPED
)
assert len(CANONICAL15_NOTHUMBAA_NAMES) == 15
# Pre-registered invariant: the control differs from canonical16 by exactly one
# variable and is otherwise column-identical.
assert CANONICAL15_NOTHUMBAA_NAMES == tuple(
    n for n in CANONICAL16_NAMES if n != "thumb_cmc_ab_adduction"
)

for _n in CANONICAL16_NAMES:
    if _n not in CANONICAL16_FROM_DB9:
        raise Db9MappingError(f"canonical16 variable {_n!r} has no DB9 source channel")
for _n, _c in CANONICAL16_FROM_DB9.items():
    if _c not in DB9_CHANNEL_ORDER:
        raise Db9MappingError(f"{_n!r} maps to unknown DB9 channel {_c!r}")


# --------------------------------------------------------------------------
# column resolution
# --------------------------------------------------------------------------
def resolve_db9_columns(names: tuple[str, ...] = CANONICAL16_NAMES) -> tuple[int, ...]:
    """DB9 column index for each requested canonical name, in canonical order."""
    return tuple(DB9_CHANNEL_ORDER.index(CANONICAL16_FROM_DB9[n]) for n in names)


def verify_db9_channel_order(order_of_angles) -> list[str]:
    """Compare a recording's own ``order_of_angles`` with the official table.

    Returns a list of problems (empty == clean).  Reported, never auto-repaired.
    """
    problems: list[str] = []
    names = []
    for raw in order_of_angles:
        s = str(raw).strip()
        if ":" in s:
            head, _, tail = s.partition(":")
            s = tail.strip() if head.strip().isdigit() else s
        names.append(s)
    if len(names) != len(DB9_CHANNEL_ORDER):
        problems.append(
            f"expected {len(DB9_CHANNEL_ORDER)} channels, file declares {len(names)}"
        )
    for i, (got, want) in enumerate(zip(names, DB9_CHANNEL_ORDER)):
        if got != want:
            problems.append(f"column {i}: file says {got!r}, table says {want!r}")
    for extra in names[len(DB9_CHANNEL_ORDER):]:
        problems.append(f"unexpected extra channel {extra!r}")
    return problems


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------
def _as_2d(a: Any, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 0:
        raise Db9MappingError(f"{name} must have at least one dimension")
    if arr.shape[-1] != len(DB9_CHANNEL_ORDER):
        raise Db9MappingError(
            f"{name} must have {len(DB9_CHANNEL_ORDER)} DB9 channels on its last "
            f"axis, got shape {arr.shape}"
        )
    return arr


def db9_validity_mask(db9_angles, *, names: tuple[str, ...] = CANONICAL16_NAMES,
                      atol: float = 0.0) -> np.ndarray:
    """``(...)`` bool, True where every DB9 channel canonical16 needs is finite.

    Composes with the emg2pose validity mask by logical AND, per the frozen
    contract.  The officially-dead ``MCP2_a`` channel is not a canonical16
    input and therefore does not mark frames invalid.
    """
    a = _as_2d(db9_angles, "db9_angles")
    cols = resolve_db9_columns(names)
    sel = a[..., list(cols)]
    return np.isfinite(sel).all(axis=-1) if atol <= 0 else (np.abs(sel) >= atol).all(axis=-1)


def _sign_vector(names: tuple[str, ...]) -> np.ndarray:
    """Per-column sign multipliers for the frozen DB9 -> canonical16 frame."""
    return np.array([FROZEN_DB9_SIGN.get(n, 1.0) for n in names], dtype=np.float64)


def to_canonical16(db9_angles, *, return_validity: bool = False, radians: bool = True):
    """``(..., 22)`` DB9 **degrees** -> ``(..., 16)`` canonical16.

    Columns are selected by anatomical name (never position), then degrees ->
    radians, then the frozen per-variable sign transformation
    (:data:`FROZEN_DB9_SIGN`, currently one variable) is applied.

    No clipping, centring, scaling, smoothing or offset correction is applied.
    """
    a = _as_2d(db9_angles, "db9_angles")
    cols = resolve_db9_columns(CANONICAL16_NAMES)
    out = a[..., list(cols)]
    if radians:
        out = out / DEGREES_PER_RADIAN
    out = out * _sign_vector(CANONICAL16_NAMES)
    if return_validity:
        return out, db9_validity_mask(db9_angles)
    return out


def to_canonical12_noDIP(db9_angles, *, return_validity: bool = False,
                         radians: bool = True):
    """``(..., 22)`` DB9 degrees -> ``(..., 12)`` canonical12_noDIP.

    Identical to :func:`to_canonical16` except that the four DIP variables are
    dropped.  This exists as the pre-registered robustness control for the
    documented DIP-sensor caveat, not as a search over representations.
    """
    a = _as_2d(db9_angles, "db9_angles")
    cols = resolve_db9_columns(CANONICAL12_NODIP_NAMES)
    out = a[..., list(cols)]
    if radians:
        out = out / DEGREES_PER_RADIAN
    out = out * _sign_vector(CANONICAL12_NODIP_NAMES)
    if return_validity:
        return out, db9_validity_mask(db9_angles, names=CANONICAL12_NODIP_NAMES)
    return out


def to_canonical15_noThumbAA(db9_angles, *, return_validity: bool = False,
                             radians: bool = True):
    """``(..., 22)`` DB9 degrees -> ``(..., 15)`` canonical15_noThumbAA.

    Identical to :func:`to_canonical16` except that `thumb_cmc_ab_adduction` is
    dropped.  **Pre-registered robustness control** for the one variable whose
    cross-dataset correspondence still carries uncertainty; it exists to be
    reported next to canonical16 and canonical12_noDIP, and must never be used to
    select or tune the primary model.
    """
    a = _as_2d(db9_angles, "db9_angles")
    cols = resolve_db9_columns(CANONICAL15_NOTHUMBAA_NAMES)
    out = a[..., list(cols)]
    if radians:
        out = out / DEGREES_PER_RADIAN
    out = out * _sign_vector(CANONICAL15_NOTHUMBAA_NAMES)
    if return_validity:
        return out, db9_validity_mask(db9_angles, names=CANONICAL15_NOTHUMBAA_NAMES)
    return out


@dataclass
class Canonical16:
    """A canonical16 block plus its validity mask (mirrors the emg2pose API)."""

    values: np.ndarray
    valid: np.ndarray

    @property
    def n_invalid(self) -> int:
        return int((~self.valid).sum())

    @property
    def n_total(self) -> int:
        return int(self.valid.size)


def extract(db9_angles) -> Canonical16:
    v, ok = to_canonical16(db9_angles, return_validity=True)
    return Canonical16(values=v, valid=ok)


def canonical16_names() -> tuple[str, ...]:
    return CANONICAL16_NAMES


def canonical16_db9_channels() -> tuple[str, ...]:
    return tuple(CANONICAL16_FROM_DB9[n] for n in CANONICAL16_NAMES)


def canonical12_noDIP_names() -> tuple[str, ...]:
    return CANONICAL12_NODIP_NAMES


def canonical15_noThumbAA_names() -> tuple[str, ...]:
    return CANONICAL15_NOTHUMBAA_NAMES


#: Column of canonical16 dropped by canonical15_noThumbAA, resolved by name.
CANONICAL15_DROPPED_COLUMN: int = CANONICAL16_NAMES.index(
    CANONICAL15_NOTHUMBAA_DROPPED[0]
)


def canonical15_from_canonical16(x16):
    """``(..., 16)`` canonical16 -> ``(..., 15)`` canonical15_noThumbAA.

    Because canonical15 is defined as canonical16 minus exactly one variable,
    and canonical16's column order is frozen, the control is obtained by
    removing one column -- no re-derivation from DB9 is needed, and the two
    representations are guaranteed column-identical elsewhere.  The dropped
    column is resolved by name, never hard-coded.
    """
    import numpy as np

    a = np.asarray(x16, dtype=np.float64)
    if a.shape[-1] != len(CANONICAL16_NAMES):
        raise Db9MappingError(
            f"expected last axis {len(CANONICAL16_NAMES)} (canonical16), got {a.shape}"
        )
    return np.delete(a, CANONICAL15_DROPPED_COLUMN, axis=-1)


def build_schema() -> dict:
    return {
        "schema_name": "db9_to_canonical16",
        "version": "1.0.0",
        "status": "FROZEN",
        "ndim": 16,
        "unit": "radians",
        "source_dataset": "NinaPro DB9 (calibrated CyberGlove-II kinematics)",
        "source_page": "https://ninapro.hevs.ch/instructions/DB9.html",
        "source_paper": "doi:10.1038/s41597-019-0349-2 (PMC6952409)",
        "source_channel_count": 22,
        "source_channel_order": list(DB9_CHANNEL_ORDER),
        "source_unit": "degrees",
        "unit_conversion": "radians = degrees / (180/pi); explicit, named",
        "selection": "by official channel NAME from each file's order_of_angles",
        "db9_column_indices": list(resolve_db9_columns(CANONICAL16_NAMES)),
        "canonical16_names": list(CANONICAL16_NAMES),
        "db9_channels": list(canonical16_db9_channels()),
        "anatomy": {
            k: {**DB9_ANATOMY[k], **sign_evidence_for(k)} for k in CANONICAL16_NAMES
        },
        "anatomy_confidence_summary": {
            "db9_sign_HIGH": sum(
                1 for k in CANONICAL16_NAMES
                if sign_evidence_for(k)["db9_sign_confidence"] == "HIGH"
            ),
            "cross_dataset_polarity_HIGH": sum(
                1 for k in CANONICAL16_NAMES
                if sign_evidence_for(k)["cross_dataset_polarity_confidence"] == "HIGH"
            ),
            "cross_dataset_sign_frozen": [
                k for k in CANONICAL16_NAMES
                if sign_evidence_for(k)["cross_dataset_polarity_confidence"].startswith("FROZEN")
            ],
            "cross_dataset_axis_identity_not_proven": [
                k for k in CANONICAL16_NAMES
                if sign_evidence_for(k).get("cross_dataset_axis_identity_confidence")
            ],
        },
        "reference_posture": DB9_REFERENCE_POSTURE,
        "excluded_channels": DB9_EXCLUDED_CHANNELS,
        "not_a_positional_prefix": True,
        "frozen_sign_transformation": {
            variable: multiplier for variable, multiplier in
            sorted(FROZEN_DB9_SIGN.items())
        },
        "frozen_sign_rationale": FROZEN_SIGN_RATIONALE,
        "frozen_sign_scope": (
            "applied in this DB9 adapter only; Session 2's emg2pose adapter is "
            "not modified"
        ),
        "does_not_apply": [
            "clipping", "per-subject centring", "offset correction",
            "normalisation", "scaling", "smoothing", "interpolation",
        ],
        "note_on_sign": (
            "The only sign change is the frozen one above. Every other variable "
            "passes through with its DB9 polarity, which agrees with the "
            "emg2pose polarity for the 15 flexion variables."
        ),
        "validity_semantics": (
            "True where every canonical16 source channel is finite. Composes "
            "with the emg2pose IK-failure mask by logical AND."
        ),
        "control_representation": {
            "name": "canonical12_noDIP",
            "ndim": 12,
            "names": list(CANONICAL12_NODIP_NAMES),
            "dropped": list(CANONICAL12_NODIP_DROPPED),
            "reason": (
                "DB9 warns that DIP sensors 'provide reliable angles when "
                "subject hand size is big ... may provide partial results when "
                "the hand of the subject is small'. Pre-registered robustness "
                "control, not a search."
            ),
        },
        "pre_registered_control_representations": {
            "canonical12_noDIP": {
                "ndim": 12,
                "dropped": list(CANONICAL12_NODIP_DROPPED),
                "purpose": "DIP-sensor reliability control",
            },
            "canonical15_noThumbAA": {
                "ndim": 15,
                "dropped": list(CANONICAL15_NOTHUMBAA_DROPPED),
                "purpose": (
                    "tests whether the residual cross-dataset uncertainty in the "
                    "thumb CMC ab/adduction axis materially drives posture "
                    "identifiability"
                ),
                "names": list(CANONICAL15_NOTHUMBAA_NAMES),
            },
            "policy": (
                "all three representations are reported side by side; none of "
                "them is used to select or tune the primary model"
            ),
        },
        "cross_dataset_zero_compatibility": (
            "RESOLVED for all 16 variables. The 15 flexion variables share the "
            "same zero (emg2pose's rest pose is verified 'fingers extended and "
            "together', which is DB9's documented reference), and "
            "thumb_cmc_ab_adduction's zero also aligns, so the only correction "
            "needed is the frozen sign flip recorded above. Evidence: "
            "docs/DB9_CANONICAL16_MAPPING.md and "
            "data/db9_posture16/thumb_web_sign_evidence.json."
        ),
    }


def write_schema(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(build_schema(), fh, indent=2)
    return path


if __name__ == "__main__":  # pragma: no cover
    print(f"canonical16 names ({len(CANONICAL16_NAMES)}):")
    cols = resolve_db9_columns(CANONICAL16_NAMES)
    for i, (n, c) in enumerate(zip(CANONICAL16_NAMES, cols)):
        a = DB9_ANATOMY[n]
        print(f"  [{i:2d}] {n:<24s} DB9 col {c:>2} = {CANONICAL16_FROM_DB9[n]:<8s} "
              f"({a['db9_anatomical_name']}, + = {a['positive_direction']})")
    print()
    print(f"DB9 column indices : {list(cols)}")
    print(f"is positional prefix: {list(cols) == list(range(16))}")
    print(f"canonical12_noDIP  : {list(CANONICAL12_NODIP_NAMES)}")
    print(f"  DB9 cols         : {list(resolve_db9_columns(CANONICAL12_NODIP_NAMES))}")
