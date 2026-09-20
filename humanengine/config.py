from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class KernelConfig:
    channels: int = 16
    sample_rate: float = 2000.0
    feature_dim: int = 128
    shared_dim: int = 256
    residual_dim: int = 128
    adapter_rank: int = 8
    api_projector_dim: int = 128
    predictor_hidden: int = 128
    invariance_hidden: int = 128
    invariance_dim: int = 32
    geometry_hidden: int = 64
    geometry_dim: int = 32
    kernels: tuple = (11, 5, 9)
    strides: tuple = (5, 2, 4)
    residual_kernel: int = 5
    dilations: tuple = (1, 2)
    warmup_seconds: float = 1.0
    burnin_seconds: float = 1.0
    timestamp_tolerance_samples: float = 0.1
    rms_seconds: float = 0.04
    spectrum_seconds: float = 0.1
    bands: tuple = ((40., 80.), (80., 160.), (160., 320.), (320., 500.), (500., 850.))
    signal_epsilon: float = 1e-6
    signal_scale_floor: float = 1e-3
    hann_periodic: bool = True
    future_seconds: tuple = (0.02, 0.04, 0.1)
    teacher_scales: tuple = (0.1, 0.5, 2.)
    teacher_hidden: int = 128
    teacher_ema: float = .999
    teacher_stats_momentum: float = .99
    teacher_scale_floor: float = .05
    mask_fraction: float = .3
    mask_min: float = .2
    mask_max: float = .5
    mask_block_short_fraction: float = .2
    observed_weight: float = .1
    physical_weight: float = .05
    ma_private_rule: str = "physical_weight-times-L_phys-no-family-alpha-v1"
    variance_weight: float = .01
    variance_floor: float = .1
    variance_epsilon: float = 1e-4
    minimum_anchors: int = 64
    minimum_recordings: int = 8
    minimum_users: int = 4
    anchor_separation_seconds: float = 1.
    geometry_pair_separation_seconds: float = .2
    geometry_min_pairs: int = 128
    geometry_degenerate_threshold: float = 1e-4
    geometry_distance_epsilon: float = 1e-8
    geometry_mean_epsilon: float = 1e-6
    pose_scale_floor: float = .1
    vicreg_variance_floor: float = 1.
    vicreg_covariance_weight: float = .04
    noise_multiplier: float = .02
    pose_beta: tuple = (1., .5, .05)
    reserve_alpha: float = .2
    reserve_vicreg_weight: float = .1
    cagrad_c: float = .25
    cagrad_tolerance: float = 1e-7
    cagrad_max_iterations: int = 500
    distillation_weight: float = .2
    huber_delta: float = 1.
    route_version: str = "api-detached-r-v1"
    summary_version: str = "log-rms40-band100-hann-periodic-v1"
    target_version: str = "independent-causal-3scale-2layer-v1"
    model_version: str = "he-scientific-kernel-v0.1"

    @property
    def stride(self):
        import math
        return math.prod(self.strides)

    def samples(self, seconds):
        n = round(seconds * self.sample_rate)
        if n < 1 or abs(n - seconds * self.sample_rate) > 1e-7:
            raise ValueError("Duration must have an exact positive sample support")
        return n

    def validate(self):
        if self.ma_private_rule!="physical_weight-times-L_phys-no-family-alpha-v1":
            raise ValueError("Unsupported M_A private objective rule")
        if self.channels < 1 or self.sample_rate <= 0:
            raise ValueError("Invalid input schema")
        if len(self.kernels) != len(self.strides) or any(k < s for k,s in zip(self.kernels,self.strides)):
            raise ValueError("Invalid convolution contract")
        if any(self.samples(s) % self.stride for s in self.teacher_scales):
            raise ValueError("Teacher windows must terminate on their reset-relative output tick")
        if sorted(self.teacher_scales) != list(self.teacher_scales) or len(set(self.teacher_scales)) != 3:
            raise ValueError("Three increasing teacher scales required")
        if len(self.bands) != 5 or any(not 0 <= a < b <= self.sample_rate/2 for a,b in self.bands):
            raise ValueError("Physical bands outside Nyquist")
        if not 0 <= self.cagrad_c < 1 or not 0 < self.reserve_alpha < 1:
            raise ValueError("Invalid optimizer budget")
        if not 0 <= self.teacher_ema < 1 or not 0 <= self.teacher_stats_momentum < 1:
            raise ValueError("Invalid EMA")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return KernelConfig(**raw["kernel"]).validate(), raw
