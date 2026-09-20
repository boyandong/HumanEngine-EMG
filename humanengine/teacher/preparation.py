import copy
from dataclasses import dataclass,asdict,field
import math
import torch
from torch import nn
from ..contracts import LossBundle,AnchorMetadata
from ..core.summaries import physical_summary
from ..data.sequence import valid_support
from ..data.views import validate_masked_view
from ..objectives.reductions import latent_error,masked_error,equal_mean
from ..objectives.regularizers import variance_floor
from .model import TeacherEncoder,RunningTargetStatistics,FrozenTeacher
from ..scientific_state import content_hash,statistics_content
from ..execution import executable,validate_architecture
from .certificate import TEACHER_CERTIFICATE_VERSION,evaluation_protocol



@torch.no_grad()
def physical_targets(sequence,anchors,cfg,statistics):
    out=sequence.emg.new_zeros(len(sequence.emg),len(anchors),6*cfg.channels)
    valid=torch.zeros(out.shape[:2],device=out.device,dtype=torch.bool)
    width=max(cfg.samples(cfg.rms_seconds),cfg.samples(cfg.spectrum_seconds))
    for j,end in enumerate(anchors):
        end=int(end);good=valid_support(sequence,end,width,cfg)
        if good.any():
            out[good,j]=statistics(physical_summary(sequence.emg[good,end-width+1:end+1],cfg))
            valid[good,j]=True
    return out.detach(),valid


class TeacherPreparation(nn.Module):
    def __init__(self,cfg,physical_statistics):
        super().__init__();self.cfg=cfg
        self.student=TeacherEncoder(cfg)
        self.ema=copy.deepcopy(self.student).eval().requires_grad_(False)
        self.predictors=nn.ModuleList(nn.Sequential(nn.Linear(cfg.teacher_hidden,cfg.predictor_hidden),nn.SiLU(),nn.Linear(cfg.predictor_hidden,cfg.teacher_hidden)) for _ in range(2))
        self.M_T=nn.Sequential(nn.Linear(cfg.teacher_hidden,cfg.predictor_hidden),nn.SiLU(),nn.Linear(cfg.predictor_hidden,6*cfg.channels))
        self.target_statistics=RunningTargetStatistics(cfg)
        self.physical_statistics=copy.deepcopy(physical_statistics)
        self.register_buffer("training_step",torch.tensor(0,dtype=torch.long))

    def train(self,mode=True):
        super().train(mode);self.ema.eval();return self

    @torch.no_grad()
    def update_ema(self):
        for e,s in zip(self.ema.parameters(),self.student.parameters()):
            e.mul_(self.cfg.teacher_ema).add_(s,alpha=1-self.cfg.teacher_ema)
        self.training_step.add_(1)

    def losses(self,clean,masked,anchors,time_mask,meta=None):
        cfg=self.cfg
        if clean.emg.shape!=masked.emg.shape or not torch.equal(clean.timestamps,masked.timestamps) or not torch.equal(clean.segments,masked.segments) or not torch.equal(clean.valid,masked.valid):
            raise ValueError("Views must share signal support and boundaries")
        validate_masked_view(clean,masked,time_mask,anchors,cfg)
        with torch.no_grad():raw,valid=self.ema.targets(clean,anchors);target=self.target_statistics(raw)
        observed,vo=self.student.targets(clean,anchors)
        hidden,vm=self.student.targets(masked,anchors)
        predict=lambda u:torch.stack([self.predictors[l](u[:,:,:,l]) for l in range(2)],3)
        terms={"masked":latent_error(predict(hidden),target,valid&vm,cfg.huber_delta),
               "observed":latent_error(predict(observed),target,valid&vo,cfg.huber_delta)}
        summary,psvalid=physical_targets(clean,anchors,cfg,self.physical_statistics)
        terms["physical"]=equal_mean([masked_error(self.M_T(observed[:,:,k,1]),summary,psvalid&vo[:,:,k,1],delta=cfg.huber_delta) for k in range(3)])
        variances=[]
        for k in range(3):
            for l in range(2):
                good=vo[:,:,k,l].flatten();metadata=None
                if meta is not None:
                    meta.validate(len(good));ids=good.nonzero().flatten().tolist()
                    metadata=AnchorMetadata(tuple(meta.users[i] for i in ids),tuple(meta.recordings[i] for i in ids),tuple(meta.times[i] for i in ids))
                variances.append(variance_floor(observed[:,:,k,l].flatten(0,1)[good],metadata,cfg))
        terms["variance"]=equal_mean(variances)
        total=None
        if all(terms[k].value is not None for k in ("masked","observed","physical")):
            total=terms["masked"].value+cfg.observed_weight*terms["observed"].value+cfg.physical_weight*terms["physical"].value
            if terms["variance"].value is not None:total=total+cfg.variance_weight*terms["variance"].value
        return LossBundle(terms,total,"teacher-emg-only-v1",{"variance_skipped":terms["variance"].value is None}),raw.detach(),valid

    @torch.no_grad()
    def visible_context_reference(self,masked,anchors):
        """Weak reference: frozen EMA directly encodes only the visible input."""
        raw,valid=self.ema.targets(masked,anchors)
        return self.target_statistics(raw),valid


@dataclass(frozen=True)
class TeacherHealth:
    latent_error: float
    constant_error: float
    context_error: float
    summary_error: float
    summary_constant_error: float
    raw_std_median: float
    within_recording_std: float
    amplitude_response: float
    checkpoint_step: int
    preregistered_step: int
    partition: str
    raw_std_threshold: float
    within_std_threshold: float
    amplitude_threshold: float
    artifact_identity: str = ""
    evaluation_config: dict = field(default_factory=dict)
    data_manifest_hash: str = ""
    checkpoint_identity: str = ""
    partition_identity: str = ""
    budget_identity: str = ""
    evaluation_version: str = "emg-health-evaluation-v2"

    def checks(self):
        values=(self.latent_error,self.constant_error,self.context_error,self.summary_error,self.summary_constant_error,
                self.raw_std_median,self.within_recording_std,self.amplitude_response,self.raw_std_threshold,self.within_std_threshold,self.amplitude_threshold)
        return {"finite":all(math.isfinite(v) and v>=0 for v in values),
                "constant":self.latent_error<self.constant_error,"visible_context":self.latent_error<self.context_error,
                "physical":self.summary_error<self.summary_constant_error,
                "raw_nonconstant":self.raw_std_median>self.raw_std_threshold,
                "within_recording":self.within_recording_std>self.within_std_threshold,
                "amplitude":self.amplitude_response>self.amplitude_threshold,
                "fixed_budget":self.checkpoint_step==self.preregistered_step and (self.partition=="synthetic" or self.preregistered_step>0),
                "label_free_development":self.partition in ("emg_development","synthetic")}


def teacher_artifact_content(preparation,statistics,health):
    """Call BEFORE evaluating health; retain identity with the measured evidence.

    This authenticates content consistency, not the truth of supplied measurements.
    No quality thresholds or real validation measurements are synthesized here.
    """
    validate_architecture(preparation.ema)
    if preparation.ema.cfg!=preparation.cfg or statistics.cfg!=preparation.cfg:raise ValueError("Teacher target/config mismatch")
    return {"certificate_version":TEACHER_CERTIFICATE_VERSION,
        "ema_hash":content_hash(preparation.ema.state_dict()),"architecture":preparation.cfg.to_dict(),
        "executable":executable(preparation.ema,frozen=True),
        "target_statistics_executable":executable(statistics,frozen=True),
        "target_statistics_hash":content_hash(statistics_content(statistics)),"physical_statistics_hash":content_hash(statistics_content(preparation.physical_statistics)),
        "training_step":int(preparation.training_step),"target_version":preparation.cfg.target_version,
        "preparation_version":"teacher-emg-only-v1","evaluation_protocol":evaluation_protocol(health)}


def teacher_artifact_identity(preparation,statistics,health):
    return content_hash(teacher_artifact_content(preparation,statistics,health))


def freeze_teacher(preparation,statistics,health,teacher_hash=None):
    checks=health.checks()
    if not all(checks.values()):raise ValueError("Teacher preparation FAILED: "+str(checks))
    if int(preparation.training_step)!=health.checkpoint_step:raise ValueError("Health evidence checkpoint mismatch")
    if health.partition!="synthetic" and statistics.provenance.get("partition")!="train":
        raise ValueError("Real teacher cannot use synthetic target statistics")
    identity=teacher_artifact_identity(preparation,statistics,health)
    if health.artifact_identity!=identity:raise ValueError("Teacher health artifact identity mismatch")
    if teacher_hash is not None and teacher_hash!=identity:raise ValueError("Incorrect claimed teacher hash")
    return FrozenTeacher(preparation.ema,statistics,{"health_passed":True,"checks":checks,
                          "teacher_hash":identity,"certificate_version":TEACHER_CERTIFICATE_VERSION,"health":asdict(health),
                          "certificate":teacher_artifact_content(preparation,statistics,health),
                          "physical_statistics":statistics_content(preparation.physical_statistics),
                          "target_version":preparation.cfg.target_version,
                          "evidence_partition":health.partition})
