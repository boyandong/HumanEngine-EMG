"""Composition of the specified pose + reserve kernel, without a training runner."""
import torch
from ..contracts import AnchorMetadata,LossTerm
from ..data.sequence import future_pose_targets,official_pose_valid,valid_support
from ..data.views import validate_masked_view
from ..registry import family_loss,reserve_loss
from ..teacher.preparation import physical_targets
from ..optimization.macrostep import FamilyEvaluation
from .reductions import pose_losses
from .regularizers import vicreg,pose_geometry
from .residual import residual_losses


def full_objectives(model,heads,frozen_teacher,registry,batch,masked,noisy,anchors,time_mask,
                    physical_statistics,pose_statistics,residual_reference,cfg):
    """Inputs are already sampled, independent views; no hidden RNG or state.

    Missing scientifically required coverage returns NA instead of changing budgets.
    Other real API families supply their own FamilyEvaluation through Macrostep.
    """
    clean=batch.signal
    if any(int(a)<cfg.samples(cfg.burnin_seconds) for a in anchors):
        raise ValueError("Scored anchors must be in the trainable post-burn-in region")
    for view in (masked,noisy):
        if view is None:continue
        if view.emg.shape!=clean.emg.shape or not torch.equal(view.timestamps,clean.timestamps) or not torch.equal(view.segments,clean.segments) or not torch.equal(view.valid,clean.valid):
            raise ValueError("View support mismatch")
    if not valid_support(clean,clean.emg.shape[1]-1,clean.emg.shape[1],cfg).all():
        raise ValueError("Full-objective windows must be contiguous valid EMG; split/reset before batching")
    validate_masked_view(clean,masked,time_mask,anchors,cfg)
    if any((int(a)+1)%cfg.stride for a in anchors):raise ValueError("Objective anchors must be output ticks")
    user_ids=batch.users;recordings=batch.recordings
    observed,_=model.forward_training_sequence(clean.emg,clean.timestamps,user_ids=user_ids,recording_ids=recordings)
    hidden,_=model.forward_training_sequence(masked.emg,masked.timestamps,user_ids=user_ids,recording_ids=recordings)
    indices=torch.tensor([(int(a)+1)//cfg.stride-1 for a in anchors],device=clean.emg.device)
    if indices.numel()==0 or (indices<0).any() or (indices>=observed.H.shape[1]).any():raise ValueError("Invalid objective anchors")
    meta=AnchorMetadata(tuple(u for u in batch.users for _ in anchors),
                        tuple(r for r in batch.recordings for _ in anchors),
                        tuple(float(clean.timestamps[b,int(a)]) for b in range(len(clean.emg)) for a in anchors))
    if observed.warmup[:,indices].any():raise ValueError("Scored anchors must follow declared warmup")
    pose=batch.labels.get("pose")
    if pose is None:
        # Storage placeholders are never valid observations and never a zero loss.
        pose=clean.emg.new_zeros(*clean.emg.shape[:2],20)
        pose_valid=torch.zeros(clean.emg.shape[:2],device=clean.emg.device,dtype=torch.bool)
    else:
        pose_valid=batch.label_masks["pose"] & official_pose_valid(pose) & torch.isfinite(pose).all(-1)
    future,future_valid=future_pose_targets(clean,pose,pose_valid,anchors,cfg.future_seconds,cfg)
    current=observed.apis["pose/current"][:,indices];prediction=observed.apis["pose/future"][:,indices]
    losses=pose_losses(current,prediction,pose[:,anchors],future,pose_valid[:,anchors],future_valid)
    selected=pose_valid[:,anchors].flatten();ids=selected.nonzero().flatten().tolist()
    geometry_meta=AnchorMetadata(tuple(meta.users[i] for i in ids),tuple(meta.recordings[i] for i in ids),tuple(meta.times[i] for i in ids))
    geo=pose_geometry(heads.G(observed.shared[:,indices]).flatten(0,1)[selected],pose[:,anchors].flatten(0,1)[selected],geometry_meta,pose_statistics,cfg)
    losses["geometry"]=geo
    pose_family=family_loss(registry.families["pose"],losses)
    target,target_valid,_=frozen_teacher(clean,anchors)
    physical,pvalid=physical_targets(clean,anchors,cfg,physical_statistics)
    residual=residual_losses(heads,hidden.residual[:,indices],observed.residual[:,indices],target,target_valid,physical,pvalid,meta,cfg)
    inv=LossTerm.na("noise scale/view unavailable")
    if noisy is not None:
        other,_=model.forward_training_sequence(noisy.emg,noisy.timestamps,user_ids=user_ids,recording_ids=recordings)
        inv=vicreg(heads.V(observed.shared[:,indices]),heads.V(other.shared[:,indices]),meta,cfg)
    reserve=reserve_loss(residual.total,inv.value,cfg,residual_reference)
    pose_private={"pose":pose_family.value,"G":geo.value} if pose_family.value is not None else {}
    # M_A's internal weak-anchor strength is independent of inter-family alpha.
    reserve_private={"P_R":residual.total,"M_A":cfg.physical_weight*residual.terms["physical"].value,"V":inv.value} if reserve.value is not None else {}
    return ({"pose":FamilyEvaluation(pose_family.value,pose_private),"reserve":FamilyEvaluation(reserve.value,reserve_private)},
            {"pose_components":losses,"residual":residual,"vicreg":inv,"observed":observed})
