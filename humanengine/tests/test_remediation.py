"""Adversarial audit regressions. Synthetic CPU evidence only."""
import copy
import math
from dataclasses import replace
import pytest
import torch
from humanengine.api import HumanEngine
from humanengine.scientific_state import equal_state,content_hash,he_semantics
from humanengine.manifest import resume_he,teacher_checkpoint,resume_teacher,rng_state,seal,identify
from humanengine.teacher.preparation import TeacherPreparation,TeacherHealth,freeze_teacher
from humanengine.teacher.model import FrozenTargetStatistics
from humanengine.data.sampling import FamilyQuotaSampler,WindowRef,quota_mask
from humanengine.optimization.cagrad import weighted_cagrad
from humanengine.diagnostics.representation import representation_health
from .conftest import signal_stats,sequence,bind_synthetic_health
from .test_optimization import setup_macro
from .test_data_provenance import checkpoint_fixture

BUDGETS={"pose":.4,"force":.4,"reserve":.2}


def training_state(m,macro,sampler=None):
    return copy.deepcopy((m.state_dict(),[p.grad for p in m.parameters()],macro.core_optimizer.state_dict(),
        {k:o.state_dict() for k,o in macro.private_optimizers.items()},macro.state_dict(),
        sampler.state_dict() if sampler else None,rng_state()))


@pytest.mark.parametrize("target",["core","pose"])
def test_F01_data_mutation_detected(cfg,target):
    m,macro,closures,_=setup_macro(cfg);before=training_state(m,macro);original=closures["reserve"]
    def bad():
        getattr(m,target).data.add_(1)
        return original()
    closures["reserve"]=bad
    with pytest.raises(RuntimeError,match="snapshot"):macro.step(closures,BUDGETS)
    assert equal_state(before,training_state(m,macro))


def test_F01_optimizer_state_restored(cfg):
    m,macro,closures,_=setup_macro(cfg,adam=True)
    macro.step(closures,BUDGETS) # populate Adam moments and step counters
    before=training_state(m,macro);original=closures["reserve"]
    def bad():
        m.core.grad=torch.ones_like(m.core);macro.core_optimizer.step()
        macro.steps+=7
        return original()
    closures["reserve"]=bad
    with pytest.raises(RuntimeError,match="snapshot"):macro.step(closures,BUDGETS)
    assert equal_state(before,training_state(m,macro))


def test_F01_sampler_mutation_restored(cfg):
    sampler=FamilyQuotaSampler([WindowRef("s","u","ss","r","w","pose","g")],{"pose":1})
    m,macro,closures,_=setup_macro(cfg);before=training_state(m,macro,sampler);original=closures["reserve"]
    def bad():
        sampler.sample();return original()
    closures["reserve"]=bad
    with pytest.raises(RuntimeError,match="snapshot"):macro.step(closures,BUDGETS,sampler=sampler)
    assert equal_state(before,training_state(m,macro,sampler))


def test_F01_optimizer_only_mutation_rejected(cfg):
    m,macro,closures,_=setup_macro(cfg,adam=True);macro.step(closures,BUDGETS)
    before=training_state(m,macro);original=closures["reserve"]
    def bad():
        macro.core_optimizer.state[m.core]["exp_avg"].add_(1)
        return original()
    closures["reserve"]=bad
    with pytest.raises(RuntimeError,match="snapshot"):macro.step(closures,BUDGETS)
    assert equal_state(before,training_state(m,macro))


def test_F01_registered_buffer_and_metadata_restored(cfg,model):
    m,macro,closures,_=setup_macro(cfg);macro.stateful_modules=(model,)
    before=copy.deepcopy((model.state_dict(),he_semantics(model)));original=closures["reserve"]
    def bad():
        model.core.E.statistics.mean.add_(1)
        model.core.E.statistics.floor=.5
        model.core.adapter_id="contaminated"
        return original()
    closures["reserve"]=bad
    with pytest.raises(RuntimeError,match="snapshot"):macro.step(closures,BUDGETS)
    assert equal_state(before,(model.state_dict(),he_semantics(model)))


def test_F01_pure_closures_identical_parameter_contents(cfg):
    m,macro,closures,snapshots=setup_macro(cfg)
    initial=m.core.detach().clone()
    expected={k:torch.autograd.grad(fn().core_loss,m.core)[0] for k,fn in closures.items()}
    result=macro.step(closures,BUDGETS)
    assert result["snapshot_consistent"] and macro.steps==1
    assert all(torch.equal(v,initial) for v in snapshots)
    for k,g in zip(result["families"],result["gradients"]):torch.testing.assert_close(g,expected[k])


def resume_state(model,heads,opts,controller,sampler):
    return copy.deepcopy((model.state_dict(),heads.state_dict(),{k:o.state_dict() for k,o in opts.items()},
        controller.state_dict(),sampler.state_dict(),he_semantics(model),rng_state()))


@pytest.mark.parametrize("field",["mean","std"])
@pytest.mark.parametrize("reseal",[False,True])
def test_F02_statistics_content_hash(model,cfg,field,reseal):
    ck,sig,heads,opts,controller,sampler=checkpoint_fixture(model,cfg)
    ck["model"]["core.E.statistics."+field].add_(1)
    if reseal:ck=seal({k:v for k,v in ck.items() if k!="content_hash"})
    before=resume_state(model,heads,opts,controller,sampler)
    with pytest.raises(ValueError,match="hash"):resume_he(ck,sig,model,heads,opts,controller,sampler)
    assert equal_state(before,resume_state(model,heads,opts,controller,sampler))


@pytest.mark.parametrize("field",["floor","provenance","adapter","mode"])
def test_F02_floor_provenance_adapter_resume(model,cfg,field):
    ck,sig,heads,opts,controller,sampler=checkpoint_fixture(model,cfg)
    if field=="floor":model.core.E.statistics.floor=.5
    elif field=="provenance":model.core.E.statistics.provenance["version"]="different"
    elif field=="adapter":model.core.adapter_id="personal-A"
    else:model.personalization_mode()
    before=resume_state(model,heads,opts,controller,sampler)
    with pytest.raises(ValueError,match="semantics"):resume_he(ck,sig,model,heads,opts,controller,sampler)
    assert equal_state(before,resume_state(model,heads,opts,controller,sampler))


def test_F02_false_claimed_hash(model,cfg):
    from humanengine.manifest import he_checkpoint
    ck,sig,heads,opts,controller,sampler=checkpoint_fixture(model,cfg)
    sig["statistics_hash"]="false-hash"
    with pytest.raises(ValueError,match="claimed"):
        he_checkpoint(model,heads,opts,controller,sampler,sig,ck["registry"],ck["reference_scales"],ck["config"])


def test_F02_transactional_late_failure(model,cfg):
    ck,sig,heads,opts,controller,sampler=checkpoint_fixture(model,cfg)
    # A valid content seal does not make malformed RNG loadable. Exercise rollback
    # AFTER model/heads/controller/sampler loads, including metadata and gradients.
    ck["model"]["core.A.log_gain"].add_(.2)
    ck["rng"]["torch"]=torch.zeros(1,dtype=torch.uint8)
    # Deliberately create a NEW valid v3 identity to reach the late RNG loader.
    # Outer-only reseal rejection remains tested separately above.
    ck=identify({k:v for k,v in ck.items() if k not in ("content_hash","scientific_identity")})
    before=resume_state(model,heads,opts,controller,sampler)
    with pytest.raises(RuntimeError):resume_he(ck,sig,model,heads,opts,controller,sampler)
    assert equal_state(before,resume_state(model,heads,opts,controller,sampler))


def test_F02_metadata_rollback_after_load_failure(model,cfg,monkeypatch):
    ck,sig,heads,opts,controller,sampler=checkpoint_fixture(model,cfg)
    before=resume_state(model,heads,opts,controller,sampler)
    original=opts["model"].load_state_dict;calls=[]
    def fail_once(state):
        if not calls:
            calls.append(True)
            model.core.adapter_id="contaminated"
            model.core.E.statistics.floor=.5
            model.core.E.statistics.provenance["version"]="contaminated"
            model.personalization_mode()
            raise RuntimeError("injected load failure")
        return original(state)
    monkeypatch.setattr(opts["model"],"load_state_dict",fail_once)
    with pytest.raises(RuntimeError,match="injected"):resume_he(ck,sig,model,heads,opts,controller,sampler)
    assert equal_state(before,resume_state(model,heads,opts,controller,sampler))


def test_F03_direct_export_rejects_stale_certificate(cfg):
    from humanengine.teacher.model import FrozenTeacher
    prep,stats,health=teacher_fixture(cfg)
    frozen=freeze_teacher(prep,stats,health)
    next(prep.ema.parameters()).data.add_(1)
    with pytest.raises(ValueError,match="certificate"):FrozenTeacher(prep.ema,stats,frozen.artifact)


@pytest.mark.parametrize("field",["floor","mean","provenance","target"])
def test_F02_teacher_statistics_semantics(cfg,field):
    prep=TeacherPreparation(cfg,signal_stats(cfg,True));opt=torch.optim.Adam(prep.student.parameters())
    ck=teacher_checkpoint(prep,opt,cfg.to_dict(),"synthetic-data","code",{"synthetic":True})
    if field=="floor":prep.physical_statistics.floor=.5
    elif field=="provenance":prep.physical_statistics.provenance["version"]="different"
    elif field=="target":ck["preparation"]["target_statistics.mean"].add_(1)
    else:ck["preparation"]["physical_statistics.mean"].add_(1)
    before=copy.deepcopy((prep.state_dict(),opt.state_dict(),rng_state(),prep.physical_statistics.provenance,prep.physical_statistics.floor))
    with pytest.raises(ValueError):resume_teacher(ck,prep,opt,config=cfg.to_dict(),data_manifest_hash="synthetic-data",source_version="code")
    assert equal_state(before,(prep.state_dict(),opt.state_dict(),rng_state(),prep.physical_statistics.provenance,prep.physical_statistics.floor))


def teacher_fixture(cfg):
    prep=TeacherPreparation(cfg,signal_stats(cfg,True))
    stats=FrozenTargetStatistics(torch.zeros(3,2,cfg.teacher_hidden),torch.ones(3,2,cfg.teacher_hidden),{"partition":"synthetic","version":"fixture"},cfg)
    health=bind_synthetic_health(prep,stats,TeacherHealth(.1,1.,.5,.1,1.,.1,.1,.1,0,0,"synthetic",.01,.01,.01))
    return prep,stats,health


@pytest.mark.parametrize("mutate",["ema","target","physical","evaluation","manifest","step"])
def test_F03_teacher_health_bound_to_weight_hash(cfg,mutate):
    prep,stats,health=teacher_fixture(cfg)
    frozen=freeze_teacher(prep,stats,health)
    assert frozen.artifact["teacher_hash"]==health.artifact_identity
    if mutate=="ema":next(prep.ema.parameters()).data.zero_()
    elif mutate=="target":stats.mean.add_(1)
    elif mutate=="physical":prep.physical_statistics.floor=.5
    elif mutate=="evaluation":health=replace(health,evaluation_config={"changed":True})
    elif mutate=="manifest":health=replace(health,data_manifest_hash="other")
    else:prep.training_step.add_(1)
    with pytest.raises(ValueError,match="mismatch"):freeze_teacher(prep,stats,health)


def full_fixture(cfg,anchor):
    from humanengine.contracts import HEBatch
    from humanengine.heads.api_heads import TrainingHeads
    from humanengine.registry import APIRegistry,APIFamily,Component
    from humanengine.data.views import masked_view
    from humanengine.core.summaries import SignalStatistics
    from humanengine.objectives.full import full_objectives
    cfg=replace(cfg,minimum_anchors=4,minimum_recordings=4,minimum_users=2,geometry_min_pairs=6)
    m=HumanEngine(cfg,signal_stats(cfg));heads=TrainingHeads(cfg)
    clean=sequence(cfg,batch=4,length=max(anchor+241,4240))
    if anchor+1<cfg.samples(cfg.teacher_scales[-1]):
        # Audit construction: explicit 40% temporal blocks, endpoint covered.
        mask=(torch.arange(clean.emg.shape[1])%100>=60)[None].expand(4,-1)
        masked=replace(clean,emg=torch.where(mask[:,:,None],0.,clean.emg))
    else:
        masked,mask,_=masked_view(clean,cfg,torch.Generator().manual_seed(3),anchor=anchor)
    prep,stats,health=teacher_fixture(cfg);teacher=freeze_teacher(prep,stats,health)
    batch=HEBatch(clean,{"pose":torch.randn(4,clean.emg.shape[1],20)},{"pose":torch.ones_like(clean.valid)},
                  ("u0","u0","u1","u1"),("s0",)*4,("r0","r1","r2","r3"),("left",)*4,("source",)*4)
    family=APIFamily("pose","v1",tuple(Component(k,b,1.,.01,"pose-valid","synthetic") for k,b in zip(("current","future","geometry"),cfg.pose_beta)),("source",),"pose-P",("current","future"),"none",cfg.route_version)
    reg=APIRegistry((family,),cfg.reserve_alpha)
    ps=SignalStatistics(torch.zeros(20),torch.ones(20),{"partition":"synthetic","version":"test"},.1)
    def compute():return full_objectives(m,heads,teacher,reg,batch,masked,clean,[anchor],mask,signal_stats(cfg,True),ps,1.,cfg)
    return m,heads,compute


def test_F04_burnin_B_minus_1_rejected(cfg):
    # Full composition must reject before teacher scoring, even with shorter warmup.
    cfg=replace(cfg,warmup_seconds=.2)
    m,heads,compute=full_fixture(cfg,1999)
    with pytest.raises(ValueError,match="post-burn-in"):compute()


@pytest.mark.parametrize("warmup",[.2,1.])
def test_F04_burnin_B_accepted(cfg,warmup):
    # B=3999 is a valid output tick; sample B is the first trainable sample.
    cfg=replace(cfg,burnin_seconds=3999/2000,warmup_seconds=warmup)
    m,heads,compute=full_fixture(cfg,3999);ev,details=compute()
    grads=torch.autograd.grad(ev["pose"].core_loss,tuple(m.core.core_parameters()),allow_unused=True)
    assert sum(float(g.abs().sum()) for g in grads if g is not None)>0
    out=details["observed"]
    assert out.input_quality["training_eligible"][:,99].all()
    assert not out.input_quality["training_eligible"][:,:99].any()


def test_F05_unique_anchor_quota():
    w=WindowRef("s","u","ss","r","w","pose","g")
    sampler=FamilyQuotaSampler([w]*5,{"pose":1})
    p=sampler.plan_valid_quotas({"pose":{"current":7}},lambda _: {"current":list(range(5))},max_attempts=20)
    grants=[g for _,row in p["allocations"]["pose"] for g in row["current"]]
    assert p["status"]=="NA" and len(grants)==len(set(grants))==5
    assert sampler.macrosteps==0
    second=replace(w,window_id="other-window")
    sampler=FamilyQuotaSampler([w,second]*5,{"pose":1})
    p=sampler.plan_valid_quotas({"pose":{"current":7}},lambda _: {"current":[1,3,5,7,9]},max_attempts=100)
    grants=[g for _,row in p["allocations"]["pose"] for g in row["current"]]
    assert p["status"]=="COMPLETE" and len(grants)==len(set(grants))==7
    for _,row in p["allocations"]["pose"]:
        mask=quota_mask(torch.ones(10,dtype=torch.bool),row["current"])
        assert mask.nonzero().flatten().tolist()==[g.index for g in row["current"]]


def test_F07_effective_rank_registered_definition():
    x=torch.tensor([[3.,0.],[-3.,0.],[0.,1.],[0.,-1.]])
    got=representation_health(x,("u",)*4,("r",)*4,std_threshold=.01)
    assert got["effective_rank"]==pytest.approx(math.exp(-.75*math.log(.75)-.25*math.log(.25)))
    assert got["effective_rank_definition"]=="centered-singular-value-entropy-v2"
    assert representation_health(x*0,("u",)*4,("r",)*4,std_threshold=.01)["effective_rank"]==0


def test_F08_extreme_finite_cagrad_norm():
    gradients=[torch.tensor([1e20,0.]),torch.tensor([0.,1e20])]
    result=weighted_cagrad(gradients,[.8,.2],.25)
    assert result.status=="COORDINATED",result.fallback_reason
    assert torch.isfinite(result.direction).all() and torch.isfinite(result.direction_gains).all()
    assert math.isfinite(result.trust_residual) and math.isfinite(result.dual_gap)
    assert (result.direction.double()-result.base.double()).norm()<=.250001*result.base.double().norm()


@pytest.mark.parametrize("alpha",[.2,.4])
def test_F09_MA_private_weight_semantics(cfg,alpha):
    cfg=replace(cfg,reserve_alpha=alpha)
    m,heads,compute=full_fixture(cfg,3999);ev,details=compute()
    params=tuple(heads.M_A.parameters())
    raw=torch.autograd.grad(details["residual"].terms["physical"].value,params,retain_graph=True)
    assigned=torch.autograd.grad(ev["reserve"].private_losses["M_A"],params)
    assert sum(float(g.abs().sum()) for g in raw)>0
    for a,b in zip(assigned,raw):torch.testing.assert_close(a,.05*b)
