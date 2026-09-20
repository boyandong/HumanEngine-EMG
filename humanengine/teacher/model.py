import copy
import torch
from torch import nn
from ..core.frontend import CausalFrontend
from ..data.sequence import valid_support
from ..scientific_state import content_hash,statistics_content
from ..execution import executable,validate_architecture
from .certificate import TEACHER_CERTIFICATE_VERSION,evaluation_protocol


class TeacherEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.frontend=CausalFrontend(cfg)
        self.layers=nn.ModuleList((nn.LSTM(cfg.feature_dim,cfg.teacher_hidden,batch_first=True),
                                  nn.LSTM(cfg.teacher_hidden,cfg.teacher_hidden,batch_first=True)))

    def window(self,x):
        z=self.frontend.forward_sequence(x);states=[]
        if z.shape[1]==0:raise ValueError("Teacher window has no complete tick")
        for layer in self.layers:
            z,_=layer(z) # zero state at each scale's left edge
            states.append(z[:,-1])
        return torch.stack(states,1)

    def targets(self,sequence,anchors):
        sequence.validate();cfg=self.cfg
        b=sequence.emg.shape[0]
        result=sequence.emg.new_zeros(b,len(anchors),3,2,cfg.teacher_hidden)
        valid=torch.zeros(b,len(anchors),3,2,device=sequence.emg.device,dtype=torch.bool)
        for j,end in enumerate(anchors):
            end=int(end)
            for k,tau in enumerate(cfg.teacher_scales):
                n=cfg.samples(tau);good=valid_support(sequence,end,n,cfg)
                if good.any():
                    result[good,j,k]=self.window(sequence.emg[good,end-n+1:end+1])
                    valid[good,j,k]=True
        return result,valid


class RunningTargetStatistics(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        shape=(3,2,cfg.teacher_hidden)
        self.register_buffer("mean",torch.zeros(shape));self.register_buffer("second",torch.ones(shape))
        self.register_buffer("count",torch.zeros(3,2,dtype=torch.long))

    @torch.no_grad()
    def update(self,raw,valid,*,partition):
        if partition!="train":raise ValueError("Teacher statistics only update on training partition")
        for k in range(3):
            for l in range(2):
                x=raw[:,:,k,l][valid[:,:,k,l]]
                if len(x)==0:continue
                if not torch.isfinite(x).all():raise ValueError("Nonfinite teacher statistic")
                beta=self.cfg.teacher_stats_momentum if self.count[k,l] else 0.
                self.mean[k,l].mul_(beta).add_(x.mean(0),alpha=1-beta)
                self.second[k,l].mul_(beta).add_(x.square().mean(0),alpha=1-beta)
                self.count[k,l]+=len(x)

    def forward(self,raw):
        std=(self.second-self.mean.square()).clamp_min(0).sqrt().clamp_min(self.cfg.teacher_scale_floor)
        return ((raw-self.mean)/std).detach()


class FrozenTargetStatistics(nn.Module):
    def __init__(self,mean,std,provenance,cfg):
        super().__init__();self.cfg=cfg;self.provenance=dict(provenance)
        if mean.shape!=(3,2,cfg.teacher_hidden) or std.shape!=mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<0).any():
            raise ValueError("Invalid target statistics")
        if provenance.get("partition") not in ("train","synthetic") or not provenance.get("version"):
            raise ValueError("Frozen target statistics provenance missing")
        self.register_buffer("mean",mean.detach().clone());self.register_buffer("std",std.detach().clone())

    def forward(self,raw):
        return ((raw-self.mean)/self.std.clamp_min(self.cfg.teacher_scale_floor)).detach()


class FrozenTeacher(nn.Module):
    def __init__(self,encoder,statistics,artifact):
        super().__init__()
        if not artifact.get("health_passed") or not artifact.get("teacher_hash"):
            raise ValueError("Cannot export unhealthy/unversioned teacher")
        certificate=artifact.get("certificate",{})
        validate_architecture(encoder)
        from .preparation import TeacherHealth
        health=TeacherHealth(**artifact.get("health",{}))
        checks=health.checks()
        if (certificate.get("certificate_version")!=TEACHER_CERTIFICATE_VERSION
            or artifact.get("certificate_version")!=TEACHER_CERTIFICATE_VERSION
            or content_hash(certificate)!=artifact["teacher_hash"]
            or artifact.get("health",{}).get("artifact_identity")!=artifact["teacher_hash"]
            or certificate.get("ema_hash")!=content_hash(encoder.state_dict())
            or certificate.get("target_statistics_hash")!=content_hash(statistics_content(statistics))
            or certificate.get("physical_statistics_hash")!=content_hash(artifact.get("physical_statistics"))
            or content_hash(certificate.get("executable"))!=content_hash(executable(encoder,frozen=True))
            or content_hash(certificate.get("target_statistics_executable"))!=content_hash(executable(statistics,frozen=True))
            or content_hash(certificate.get("architecture"))!=content_hash(encoder.cfg.to_dict())
            or content_hash(evaluation_protocol(health))!=content_hash(certificate.get("evaluation_protocol"))
            or certificate.get("training_step")!=health.checkpoint_step
            or artifact.get("evidence_partition")!=health.partition
            or artifact.get("target_version")!=encoder.cfg.target_version
            or certificate.get("target_version")!=encoder.cfg.target_version
            or (health.partition!="synthetic" and statistics.provenance.get("partition")!="train")
            or statistics.cfg!=encoder.cfg or not all(checks.values()) or artifact.get("checks")!=checks):
            raise ValueError("Frozen teacher artifact certificate mismatch")
        self.encoder=copy.deepcopy(encoder).eval().requires_grad_(False)
        self.statistics=copy.deepcopy(statistics).eval();self.artifact=copy.deepcopy(artifact)
        self.eval();self.requires_grad_(False)

    def train(self,mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self,sequence,anchors):
        raw,valid=self.encoder.targets(sequence,anchors)
        return self.statistics(raw),valid,raw


def assert_independent(teacher,humanengine):
    a={p.data_ptr() for p in teacher.parameters()};b={p.data_ptr() for p in humanengine.parameters()}
    if a&b:raise ValueError("Teacher/HE parameter storage is shared")
    if {id(m) for m in teacher.modules()}&{id(m) for m in humanengine.modules()}:
        raise ValueError("Teacher/HE module objects shared")
    if {b.data_ptr() for b in teacher.buffers()}&{b.data_ptr() for b in humanengine.buffers()}:
        raise ValueError("Teacher/HE statistics storage shared")


@torch.no_grad()
def fit_frozen_statistics(raw,valid,provenance,cfg):
    """Separate post-freeze train-only statistics pass, never per-window scaling."""
    if provenance.get("partition") not in ("train","synthetic"):raise ValueError("Statistics partition leakage")
    mean=raw.new_empty(3,2,cfg.teacher_hidden);std=torch.empty_like(mean)
    for k in range(3):
        for l in range(2):
            values=raw[:,:,k,l][valid[:,:,k,l]]
            if len(values)<2:raise ValueError("Insufficient target-statistics support")
            mean[k,l]=values.mean(0);std[k,l]=values.std(0,unbiased=True)
    return FrozenTargetStatistics(mean,std,provenance,cfg)
