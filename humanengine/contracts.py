from dataclasses import dataclass, field
from typing import Optional
import torch
from torch import Tensor


@dataclass(frozen=True)
class EMGSequence:
    """Label-free teacher/data contract; no API labels or pose masks."""
    emg: Tensor                         # B,T,C
    timestamps: Tensor                  # B,T
    valid: Tensor                       # B,T boolean
    segments: Tensor                    # B,T, boundary identity

    def validate(self):
        if self.emg.ndim != 3 or any(t.shape != self.emg.shape[:2] for t in
                                   (self.timestamps,self.valid,self.segments)):
            raise ValueError("EMG sequence shape mismatch")
        if self.valid.dtype != torch.bool:
            raise ValueError("EMG valid must be boolean")
        return self


@dataclass(frozen=True)
class HEBatch:
    signal: EMGSequence
    labels: dict[str, Tensor]
    label_masks: dict[str, Tensor]
    users: tuple[str, ...]
    sessions: tuple[str, ...]
    recordings: tuple[str, ...]
    sides: tuple[str, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConvState:
    tail: Tensor
    seen: int


@dataclass(frozen=True)
class StreamState:
    conv: tuple[ConvState, ...]
    s_h: Tensor
    s_c: Tensor
    r_h: Tensor
    r_c: Tensor
    amplitude_tail: Tensor
    seen: int = 0
    last_timestamp: Optional[float] = None
    user_id: str = "unknown"
    recording_id: str = "unknown"
    adapter_id: str = "identity"
    reset_reason: str = "initial"


@dataclass(frozen=True)
class RuntimeState:
    streams: tuple[StreamState, ...]
    schema: tuple
    model_version: str


@dataclass
class HumanStateOutput:
    timestamps: Tensor
    valid: Tensor
    shared: Tensor
    residual: Tensor
    H: Tensor
    frontend: Tensor
    amplitude: Tensor
    apis: dict[str, Tensor]
    warmup: Tensor
    input_quality: dict
    model_version: str
    adapter_ids: tuple[str, ...]


@dataclass(frozen=True)
class LossTerm:
    numerator: Optional[Tensor]
    count: int
    value: Optional[Tensor]
    status: str = "OK"

    @staticmethod
    def na(reason):
        return LossTerm(None, 0, None, "NA:" + reason)


@dataclass
class LossBundle:
    terms: dict[str, LossTerm]
    total: Optional[Tensor]
    route_version: str
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AnchorMetadata:
    users: tuple[str, ...]
    recordings: tuple[str, ...]
    times: tuple[float, ...]

    def validate(self, n):
        if any(len(x) != n for x in (self.users,self.recordings,self.times)):
            raise ValueError("Anchor metadata must match flattened representations")
