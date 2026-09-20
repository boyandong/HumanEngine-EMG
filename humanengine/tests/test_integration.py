from dataclasses import replace
from pathlib import Path
import torch
import pytest
from humanengine.api import HumanEngine
from humanengine.contracts import HEBatch
from humanengine.config import load_config
from humanengine.heads.api_heads import TrainingHeads
from humanengine.registry import APIRegistry,APIFamily,Component
from humanengine.teacher.model import fit_frozen_statistics
from humanengine.teacher.preparation import TeacherPreparation,TeacherHealth,freeze_teacher
from humanengine.data.views import masked_view,weak_noise_view
from humanengine.objectives.full import full_objectives
from humanengine.optimization.factory import make_macrostep
from .conftest import sequence,signal_stats,bind_synthetic_health


def test_full_kernel_composition_and_single_synthetic_update(cfg):
    # Explicit unit-test support profile, never represented as scientific FULL evidence.
    cfg=replace(cfg,minimum_anchors=4,minimum_recordings=4,minimum_users=2,geometry_min_pairs=6)
    model=HumanEngine(cfg,signal_stats(cfg));heads=TrainingHeads(cfg)
    clean=sequence(cfg,batch=4,length=4240)
    masked,time_mask,_=masked_view(clean,cfg,torch.Generator().manual_seed(3),anchor=3999)
    noisy=weak_noise_view(clean,cfg,torch.full((2,),.01),torch.Generator().manual_seed(4))
    prep=TeacherPreparation(cfg,signal_stats(cfg,True))
    raw,valid=prep.ema.targets(clean,[3999])
    stats=fit_frozen_statistics(raw,valid,{"partition":"synthetic","version":"unit-test"},cfg)
    health=TeacherHealth(.1,1.,.5,.1,1.,.1,.1,.1,0,0,"synthetic",.01,.01,.01)
    health=bind_synthetic_health(prep,stats,health)
    teacher=freeze_teacher(prep,stats,health)
    batch=HEBatch(clean,{"pose":torch.randn(4,4240,20)}, {"pose":torch.ones(4,4240,dtype=torch.bool)},
                  ("u0","u0","u1","u1"),("s0",)*4,("r0","r1","r2","r3"),("left",)*4,("source",)*4)
    family=APIFamily("pose","v1",tuple(Component(k,b,1.,.01,"pose-valid","synthetic") for k,b in zip(("current","future","geometry"),cfg.pose_beta)),
                     ("source",),"pose-P",("current","future"),"not-yet",cfg.route_version)
    registry=APIRegistry((family,),cfg.reserve_alpha)
    from humanengine.core.summaries import SignalStatistics
    pstats=SignalStatistics(torch.zeros(20),torch.ones(20),{"partition":"synthetic","version":"test"},cfg.pose_scale_floor)
    evaluations,details=full_objectives(model,heads,teacher,registry,batch,masked,noisy,[3999],time_mask,
                                        signal_stats(cfg,True),pstats,1.,cfg)
    assert all(v.core_loss is not None for v in evaluations.values())
    macro=make_macrostep(model,heads,registry,cfg,core_optimizer_factory=lambda p:torch.optim.SGD(p,lr=1e-4),private_lr=1e-4,
                         evaluation_modules=(teacher,))
    # v3 requires each accepted graph to originate inside guarded evaluation.
    # Preserve the same full objectives and every scientific assertion below.
    def evaluate_family(key):
        return full_objectives(model,heads,teacher,registry,batch,masked,noisy,[3999],time_mask,
                               signal_stats(cfg,True),pstats,1.,cfg)[0][key]
    result=macro.step({k:(lambda k=k:evaluate_family(k)) for k in evaluations},registry.budgets())
    assert result["solution"].status=="BASE_ONLY" and macro.steps==1
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for m in (model.core.A,model.core.U) for p in m.parameters())
    assert details["residual"].terms["variance"].status=="OK"
    unlabelled=replace(batch,labels={},label_masks={})
    evaluations,_=full_objectives(model,heads,teacher,registry,unlabelled,masked,noisy,[3999],time_mask,
                                 signal_stats(cfg,True),pstats,1.,cfg)
    assert evaluations["pose"].core_loss is None and evaluations["reserve"].core_loss is not None


def test_burnin_is_detached_but_stream_values_identical(model,cfg):
    seq=sequence(cfg,batch=1,length=2400);x=seq.emg.clone().requires_grad_()
    out,_=model.forward_training_sequence(x,seq.timestamps)
    full,_=model.forward_sequence(x,seq.timestamps)
    torch.testing.assert_close(out.H,full.H,atol=1e-5,rtol=1e-5)
    grad=torch.autograd.grad(out.H[:,-1].square().sum(),x)[0]
    assert not grad[:,:2000].any() and grad[:,2000:].abs().sum()>0


def test_all_six_explicit_configs_load():
    root=Path(__file__).parents[1]/"configs"
    names=("teacher_emg_ssl_v01","he_core_v01","he_full_v01","he_routing_test","he_family_budget_test","he_cagrad_test")
    for name in names:
        cfg,raw=load_config(root/(name+".json"))
        assert cfg.route_version=="api-detached-r-v1" and not raw["execution"]["scientific_training_authorized"]
        assert raw["teacher_selection"]["budget_steps"] is None


def test_teacher_rejects_zero_mask_disguised_as_masked(cfg):
    seq=sequence(cfg,batch=1,length=4000);prep=TeacherPreparation(cfg,signal_stats(cfg,True))
    with pytest.raises(ValueError,match="endpoint"):
        prep.losses(seq,seq,[3999],torch.zeros_like(seq.valid))
