from dataclasses import dataclass
import copy
import math
import torch


@dataclass(frozen=True)
class AdapterArtifact:
    version: str
    core_hash: str
    statistics_hash: str
    input_schema: tuple
    affine_state: dict
    feature_state: dict


def load_adapter(model,artifact,*,core_hash,statistics_hash,noise_scale=None,gain_limit=2.,offset_noise_fraction=.5):
    """Load a small, compatible adapter; next step resets old-version streams."""
    core=model.core
    if artifact.core_hash!=core_hash or artifact.statistics_hash!=statistics_hash or artifact.input_schema!=core.schema:
        raise ValueError("Adapter compatibility mismatch")
    if not artifact.version or artifact.version==core.adapter_id:raise ValueError("A new adapter version is required")
    if gain_limit<=1 or offset_noise_fraction<0:raise ValueError("Invalid adapter bounds")
    gain=artifact.affine_state["log_gain"];offset=artifact.affine_state["offset"]
    if not torch.isfinite(gain).all() or not torch.isfinite(offset).all() or (gain.abs()>math.log(gain_limit)).any():
        raise ValueError("Adapter gain/offset invalid")
    if noise_scale is None:
        if offset.any():raise ValueError("Offset calibration needs known noise scale")
    elif (offset.abs()>offset_noise_fraction*noise_scale).any():raise ValueError("Adapter offset outside bounds")
    if any(not torch.isfinite(v).all() for v in artifact.feature_state.values()):raise ValueError("Nonfinite feature adapter")
    old=copy.deepcopy((core.A.state_dict(),core.U.state_dict()))
    try:
        core.A.load_state_dict(artifact.affine_state,strict=True);core.U.load_state_dict(artifact.feature_state,strict=True)
    except Exception:
        core.A.load_state_dict(old[0]);core.U.load_state_dict(old[1]);raise
    core.adapter_id=artifact.version
