from dataclasses import replace
import copy
import torch
import pytest
from humanengine.teacher.model import FrozenTargetStatistics,assert_independent
from humanengine.teacher.preparation import TeacherPreparation,TeacherHealth,freeze_teacher
from humanengine.data.views import masked_view,weak_noise_view
from humanengine.contracts import EMGSequence
from .conftest import sequence,signal_stats,bind_synthetic_health


def test_independent_teacher_preparation_routes_and_EMA(model,cfg):
    prep=TeacherPreparation(cfg,signal_stats(cfg,True));assert_independent(prep,model)
    clean=sequence(cfg,length=4000);masked,mask,fractions=masked_view(clean,cfg,torch.Generator().manual_seed(1))
    before={k:v.clone() for k,v in prep.ema.state_dict().items()}
    bundle,raw,valid=prep.losses(clean,masked,[3999],mask)
    assert bundle.total is not None and bundle.terms["variance"].value is None
    bundle.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in prep.student.parameters())
    assert all(p.grad is None for p in prep.ema.parameters()) and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v,prep.ema.state_dict()[k]) for k,v in before.items())
    assert not prep.target_statistics.count.any() # forward never mutates normalization
    prep.target_statistics.update(raw,valid,partition="train")
    with pytest.raises(ValueError):prep.target_statistics.update(raw,valid,partition="test")
    first=next(prep.student.parameters());ema=next(prep.ema.parameters());old=ema.clone()
    with torch.no_grad():first.add_(1.)
    prep.update_ema()
    torch.testing.assert_close(ema,cfg.teacher_ema*old+(1-cfg.teacher_ema)*first)
    reference,rv=prep.visible_context_reference(masked,[3999])
    assert reference.shape==(2,1,3,2,128) and rv.all()


def test_teacher_causal_six_targets_and_recording_support(cfg):
    prep=TeacherPreparation(cfg,signal_stats(cfg,True));seq=sequence(cfg,length=4080)
    raw,valid=prep.ema.targets(seq,[3999]);altered=seq.emg.clone();altered[:,4000:]+=999.
    other,_=prep.ema.targets(replace(seq,emg=altered),[3999])
    torch.testing.assert_close(raw,other,atol=0,rtol=0)
    assert raw.shape==(2,1,3,2,128) and valid.all()
    assert not torch.allclose(raw[:,:,0],raw[:,:,2])
    segments=seq.segments.clone();segments[:,3800:]=1
    _,valid=prep.ema.targets(replace(seq,segments=segments),[3999])
    assert valid[:,:,0].all() and not valid[:,:,1:].any()


def test_mask_protocol_and_adversarial_clean_target_leakage(model,cfg):
    clean=sequence(cfg,batch=1,length=4000);masked,mask,fractions=masked_view(clean,cfg,torch.Generator().manual_seed(7))
    assert mask[:,-40:].all() and ((fractions>=.2)&(fractions<=.5)).all()
    other=clean.emg.clone();other[mask]+=4.*torch.randn_like(other[mask])
    masked2=replace(clean,emg=torch.where(mask[:,:,None],0.,other))
    assert torch.equal(masked.emg,masked2.emg)
    a,_=model.forward_sequence(masked.emg,masked.timestamps);b,_=model.forward_sequence(masked2.emg,masked2.timestamps)
    torch.testing.assert_close(a.residual,b.residual,atol=0,rtol=0)
    prep=TeacherPreparation(cfg,signal_stats(cfg,True))
    y,_=prep.ema.targets(clean,[3999]);z,_=prep.ema.targets(replace(clean,emg=other),[3999])
    assert not torch.allclose(y,z)


def test_teacher_health_gate_and_frozen_statistics(cfg,model):
    prep=TeacherPreparation(cfg,signal_stats(cfg,True))
    stats=FrozenTargetStatistics(torch.zeros(3,2,128),torch.full((3,2,128),.01),{"partition":"synthetic","version":"fixture"},cfg)
    health=TeacherHealth(.1,1.,.5,.1,1.,.1,.1,.1,0,0,"synthetic",.01,.01,.01)
    health=bind_synthetic_health(prep,stats,health)
    teacher=freeze_teacher(prep,stats,health);assert_independent(teacher,model)
    teacher.train();assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    raw=torch.ones(1,1,3,2,128,requires_grad=True)
    torch.testing.assert_close(teacher.statistics(raw),torch.full_like(raw,20.))
    assert not teacher.statistics(raw).requires_grad
    with pytest.raises(ValueError,match="FAILED"):
        freeze_teacher(prep,stats,replace(health,raw_std_median=0.),"bad")
    with pytest.raises(ValueError,match="FAILED"):
        freeze_teacher(prep,stats,replace(health,partition="pose_validation"),"bad")


def test_no_unjustified_noise_scale(cfg):
    seq=sequence(cfg)
    with pytest.raises(ValueError,match="NA"):
        weak_noise_view(seq,cfg,None,torch.Generator())


def test_frozen_statistics_never_use_per_window_or_heldout(cfg):
    from humanengine.teacher.model import fit_frozen_statistics
    raw=torch.randn(4,2,3,2,128);valid=torch.ones(4,2,3,2,dtype=torch.bool)
    with pytest.raises(ValueError,match="leakage"):
        fit_frozen_statistics(raw,valid,{"partition":"test","version":"bad"},cfg)
    stats=fit_frozen_statistics(raw,valid,{"partition":"synthetic","version":"fixture"},cfg)
    torch.testing.assert_close(stats.mean,raw.mean((0,1)))
    before=stats.mean.clone();stats(raw+100.)
    assert torch.equal(stats.mean,before)
