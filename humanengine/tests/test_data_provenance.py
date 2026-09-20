from dataclasses import replace
import copy
import torch
import pytest
from humanengine.data.sequence import open_hdf5_ascii,official_pose_valid,future_pose_targets,validate_split_groups
from humanengine.heads.api_heads import canonical_view,TrainingHeads,APIHead
from humanengine.manifest import content_hash,he_checkpoint,resume_he,SIGNATURE_KEYS,teacher_checkpoint,resume_teacher
from humanengine.teacher.preparation import TeacherPreparation
from humanengine.registry import APIRegistry
from humanengine.data.sampling import FamilyQuotaSampler,WindowRef
from humanengine.continual.contracts import ReplayManifest,Onboarding,distillation_loss
from humanengine.diagnostics.representation import representation_health,RepresentationExport,information_loss_diagnostic
from .conftest import sequence,signal_stats,anchor_meta
from .test_optimization import setup_macro,family
from humanengine.scientific_state import statistics_content


def test_ascii_HDF5_path_preserved_without_resolve():
    path=r"C:\emg2pose\data\fixture.hdf5";calls=[]
    result=open_hdf5_ascii(path,opener=lambda p,mode:calls.append((p,mode)) or "fixture")
    assert result=="fixture" and calls==[(path,"r")]


def test_canonical_frozen_mapping_and_validity_parity():
    from representation.canonical_hand import to_canonical16,validity_mask
    x=torch.arange(80,dtype=torch.float32).reshape(4,20).requires_grad_()
    y=canonical_view(x);torch.testing.assert_close(y,torch.from_numpy(to_canonical16(x.detach().numpy())))
    z=torch.zeros(3,20);z[1,3]=1e-6;z[2,5]=1e-9
    assert torch.equal(official_pose_valid(z),torch.from_numpy(validity_mask(z.numpy())))
    y.sum().backward();assert int((x.grad!=0).sum())==64


def test_future_target_masks_boundaries_gaps_and_tail(cfg):
    seq=sequence(cfg,batch=1,length=400);pose=torch.ones(1,400,20)
    mask=torch.ones(1,400,dtype=torch.bool);mask[:,110]=False
    segments=seq.segments.clone();segments[:,300:]=1
    future,valid=future_pose_targets(replace(seq,segments=segments),pose,mask,[79,279,359],cfg.future_seconds,cfg)
    assert not valid[0,0].any() # invalid path, even though destination labels are valid
    assert not valid[0,1].any() # recording boundary
    assert valid[0,2,0] and not valid[0,2,1:].any() # no clamp at sequence tail
    assert not future.requires_grad


def test_synchronized_source_cannot_cross_splits():
    rows=[{"source_group":"same","recording_id":"left","partition":"train"},
          {"source_group":"same","recording_id":"right","partition":"test"}]
    with pytest.raises(ValueError,match="crosses"):validate_split_groups(rows)


def checkpoint_fixture(model,cfg):
    _,controller,_,_=setup_macro(cfg)
    heads=TrainingHeads(cfg);registry=APIRegistry((family("pose"),)).to_dict()
    refs={"pose":1.,"reserve":1.};config=cfg.to_dict()
    signature={k:"synthetic-"+k for k in SIGNATURE_KEYS}
    signature.update(config_hash=content_hash(config),registry_hash=content_hash(registry),reference_scales_hash=content_hash(refs),route_version=cfg.route_version,
                     target_version=cfg.target_version,summary_version=cfg.summary_version,statistics_hash=content_hash(statistics_content(model.core.E.statistics)))
    sampler=FamilyQuotaSampler([WindowRef("s","u","ss","r","w","pose","g")],{"pose":1})
    optimizers={"model":torch.optim.Adam(model.parameters(),lr=.001)}
    checkpoint=he_checkpoint(model,heads,optimizers,controller,sampler,signature,registry,refs,config)
    return checkpoint,signature,heads,optimizers,controller,sampler


@pytest.mark.parametrize("key",["route_version","teacher_hash","target_version","registry_hash","reference_scales_hash","summary_version","data_manifest_hash","statistics_hash"])
def test_resume_provenance_mismatch_fails_before_mutation(model,cfg,key):
    checkpoint,signature,heads,optimizers,controller,sampler=checkpoint_fixture(model,cfg)
    before={k:v.clone() for k,v in model.state_dict().items()};different=dict(signature);different[key]+="-different"
    with pytest.raises(ValueError,match="incompatibility"):
        resume_he(checkpoint,different,model,heads,optimizers,controller,sampler)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_resume_restores_rng_model_and_sampler(model,cfg):
    checkpoint,signature,heads,optimizers,controller,sampler=checkpoint_fixture(model,cfg)
    expected_rng=torch.rand(3);expected_sample=sampler.sample()
    with torch.no_grad():next(model.parameters()).add_(1.)
    torch.rand(17);sampler.sample()
    resume_he(checkpoint,signature,model,heads,optimizers,controller,sampler)
    torch.testing.assert_close(torch.rand(3),expected_rng,atol=0,rtol=0)
    assert sampler.sample()==expected_sample
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in checkpoint["model"].items())


def test_teacher_checkpoint_resume_provenance(cfg):
    prep=TeacherPreparation(cfg,signal_stats(cfg,True));opt=torch.optim.SGD(prep.student.parameters(),lr=.01)
    checkpoint=teacher_checkpoint(prep,opt,cfg.to_dict(),"synthetic-data","synthetic-code",{"not_evaluated":True})
    with pytest.raises(ValueError,match="incompatibility"):
        resume_teacher(checkpoint,prep,opt,config=cfg.to_dict(),data_manifest_hash="changed-data",source_version="synthetic-code")
    resume_teacher(checkpoint,prep,opt,config=cfg.to_dict(),data_manifest_hash="synthetic-data",source_version="synthetic-code")
    assert checkpoint["training_step"]==0 and "target_statistics.mean" in checkpoint["preparation"]


def test_replay_onboarding_and_output_only_distillation(model,cfg):
    with pytest.raises(ValueError):ReplayManifest("pose","1",("s",),(("u","s","r"),),"hash",partition="test")
    model.api_heads["fixture_new"]=APIHead(cfg,1)
    gate=Onboarding("fixture_new");gate.begin(model)
    assert all(not p.requires_grad for p in model.core.parameters())
    assert all(p.requires_grad for p in model.api_heads["fixture_new"].parameters())
    with pytest.raises(ValueError):gate.enable_joint(model)
    gate.verify(finite_outputs=True,error=.1,constant_reference=1.);gate.enable_joint(model)
    assert all(p.requires_grad for p in model.core.core_parameters())
    assert all(not p.requires_grad for p in model.core.A.parameters())
    old=torch.randn(3,20,requires_grad=True);new=torch.randn(3,20,requires_grad=True)
    loss=distillation_loss(new,old,torch.ones(3,dtype=torch.bool),scale=torch.ones(20));loss.value.backward()
    assert old.grad is None and new.grad is not None


def test_representation_health_and_export_does_not_claim_physiology():
    meta=anchor_meta();x=torch.randn(64,8)
    health=representation_health(x,meta.users,meta.recordings,std_threshold=.001)
    assert 1<health["effective_rank"]<=8 and health["within_recording_variance"] is not None
    assert representation_health(torch.zeros_like(x),meta.users,meta.recordings,std_threshold=.001)["effective_rank"]==0
    with pytest.raises(ValueError,match="Teacher hash"):
        RepresentationExport("teacher",x,torch.arange(64),meta.users,meta.recordings,meta.recordings,tuple("left" for _ in range(64)),"v1","checkpoint")
    messages=information_loss_diagnostic({"raw":.9,"teacher":.4,"R":.2,"F":.3},.1)
    assert "teacher target coverage" in messages and "frontend bottleneck" in messages


def test_reference_scales_are_train_only_and_constant_predictor_based():
    from humanengine.data.references import regression_reference,classification_reference,residual_reference
    x=torch.tensor([[0.],[2.],[4.]]);valid=torch.ones(3,dtype=torch.bool)
    ref=regression_reference(x,valid,torch.ones(1),floor=.01,version="v1",partition="synthetic")
    assert ref.constant==(2.,) and ref.value==pytest.approx(4/3)
    ce=classification_reference(torch.tensor([0,0,1,1]),torch.ones(4,dtype=torch.bool),2,floor=.01,version="v1",partition="train")
    assert ce.value==pytest.approx(float(torch.tensor(2.).log()))
    target=torch.ones(2,1,3,2,128);tv=torch.ones(2,1,3,2,dtype=torch.bool)
    assert residual_reference(target,tv,floor=.01,version="v1",partition="train").value==.5
    with pytest.raises(ValueError,match="training partition"):
        regression_reference(x,valid,torch.ones(1),floor=.01,version="v1",partition="test")
