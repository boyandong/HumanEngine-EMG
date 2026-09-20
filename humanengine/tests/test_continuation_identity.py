"""Schema-v4 continuation identity and observational-purity acceptance tests."""
import copy
import random
import numpy as np
import pytest
import torch
from torch import nn

from humanengine.manifest import teacher_checkpoint,resume_teacher
from humanengine.optimization.macrostep import Macrostep,FamilyEvaluation
from humanengine.teacher.preparation import TeacherPreparation
from .conftest import signal_stats
from .test_data_provenance import checkpoint_fixture


def preparation(cfg,seed=701):
    torch.manual_seed(seed)
    return TeacherPreparation(cfg,signal_stats(cfg,True))


def checkpoint(prep,opt,cfg):
    return teacher_checkpoint(prep,opt,cfg.to_dict(),"synthetic-continuation-data",
                              "synthetic-continuation-source",{"not_evaluated":True})


def resume(value,prep,opt,cfg):
    return resume_teacher(value,prep,opt,config=cfg.to_dict(),
                          data_manifest_hash="synthetic-continuation-data",
                          source_version="synthetic-continuation-source",
                          expected_identity=value["scientific_identity"])


def synthetic_update(prep,opt,index):
    opt.zero_grad(set_to_none=True)
    chosen=list(prep.student.parameters())[:4]
    random_factor=torch.rand((),dtype=chosen[0].dtype)
    loss=sum(((p*(index+1)*random_factor).square()).mean() for p in chosen)
    loss.backward();opt.step();prep.update_ema()


def assert_optimizer_equal_by_name(left,left_opt,right,right_opt):
    ln=dict(left.named_parameters());rn=dict(right.named_parameters())
    for name in ln:
        assert name in rn
        a=left_opt.state.get(ln[name],{});b=right_opt.state.get(rn[name],{})
        assert a.keys()==b.keys()
        for key in a:
            if torch.is_tensor(a[key]):torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
            else:assert a[key]==b[key]


def test_continuous_and_checkpoint_resume_are_exactly_equivalent(cfg):
    torch.manual_seed(811);continuous=preparation(cfg,811)
    continuous_opt=torch.optim.Adam(continuous.student.parameters(),lr=3e-4,betas=(.8,.97),eps=1e-7,weight_decay=.02)
    torch.manual_seed(991)
    for index in range(5):synthetic_update(continuous,continuous_opt,index)

    torch.manual_seed(811);interrupted=preparation(cfg,811)
    interrupted_opt=torch.optim.Adam(interrupted.student.parameters(),lr=3e-4,betas=(.8,.97),eps=1e-7,weight_decay=.02)
    torch.manual_seed(991)
    for index in range(2):synthetic_update(interrupted,interrupted_opt,index)
    saved=checkpoint(interrupted,interrupted_opt,cfg)

    receiver=preparation(cfg,123)
    receiver_opt=torch.optim.Adam(receiver.student.parameters(),lr=3e-4,betas=(.8,.97),eps=1e-7,weight_decay=.02)
    resume(saved,receiver,receiver_opt,cfg)
    for index in range(2,5):synthetic_update(receiver,receiver_opt,index)

    for name,value in continuous.state_dict().items():
        torch.testing.assert_close(value,receiver.state_dict()[name],atol=0,rtol=0)
    assert int(continuous.training_step)==int(receiver.training_step)==5
    assert_optimizer_equal_by_name(continuous,continuous_opt,receiver,receiver_opt)


@pytest.mark.parametrize("change",["class","lr","betas","order","missing","extra","groups"])
def test_incompatible_optimizer_continuation_rejected(cfg,change):
    source=preparation(cfg);parameters=list(source.student.parameters())
    if change=="groups":
        source_opt=torch.optim.Adam([{"params":parameters[:2],"lr":1e-3},
                                     {"params":parameters[2:],"lr":2e-3}],betas=(.8,.9))
    else:source_opt=torch.optim.Adam(parameters,lr=1e-3,betas=(.8,.9))
    for p in parameters[:2]:p.grad=torch.randn_like(p)
    source_opt.step();saved=checkpoint(source,source_opt,cfg)
    receiver=preparation(cfg);current=list(receiver.student.parameters())
    if change=="class":candidate=torch.optim.AdamW(current,lr=1e-3,betas=(.8,.9))
    elif change=="lr":candidate=torch.optim.Adam(current,lr=2e-3,betas=(.8,.9))
    elif change=="betas":candidate=torch.optim.Adam(current,lr=1e-3,betas=(.7,.9))
    elif change=="order":
        current[4],current[6]=current[6],current[4];candidate=torch.optim.Adam(current,lr=1e-3,betas=(.8,.9))
    elif change=="missing":candidate=torch.optim.Adam(current[:-1],lr=1e-3,betas=(.8,.9))
    elif change=="extra":candidate=torch.optim.Adam([*current,next(receiver.predictors.parameters())],lr=1e-3,betas=(.8,.9))
    elif change=="groups":
        candidate=torch.optim.Adam([{"params":[current[0],*current[2:]],"lr":1e-3},
                                    {"params":[current[1]],"lr":2e-3}],betas=(.8,.9))
    with pytest.raises(ValueError,match="Optimizer continuation incompatibility"):
        resume(saved,receiver,candidate,cfg)


def test_compatible_named_optimizer_state_reaches_same_next_update(cfg):
    source=preparation(cfg);opt=torch.optim.Adam(source.student.parameters(),lr=1e-3,betas=(.8,.9))
    for i,p in enumerate(source.student.parameters()):p.grad=torch.full_like(p,(i+1)/100)
    opt.step();saved=checkpoint(source,opt,cfg)
    left=preparation(cfg,901);right=preparation(cfg,902)
    lo=torch.optim.Adam(left.student.parameters(),lr=1e-3,betas=(.8,.9))
    ro=torch.optim.Adam(right.student.parameters(),lr=1e-3,betas=(.8,.9))
    resume(saved,left,lo,cfg);resume(saved,right,ro,cfg)
    for module,optimizer in ((left,lo),(right,ro)):
        for p in module.student.parameters():p.grad=torch.zeros_like(p)
        optimizer.step()
    for name,p in left.student.named_parameters():
        torch.testing.assert_close(p,dict(right.student.named_parameters())[name],atol=0,rtol=0)


def test_parameter_storage_alias_rejected_before_load(cfg):
    source=preparation(cfg);opt=torch.optim.Adam(source.student.parameters(),lr=1e-3)
    saved=checkpoint(source,opt,cfg);receiver=preparation(cfg)
    layer=receiver.student.layers[0]
    layer.bias_hh_l0.data=layer.bias_ih_l0.data
    candidate=torch.optim.Adam(receiver.student.parameters(),lr=1e-3)
    before=copy.deepcopy(receiver.state_dict())
    with pytest.raises(ValueError,match="Shared/overlapping parameter storage"):
        resume(saved,receiver,candidate,cfg)
    assert all(torch.equal(value,receiver.state_dict()[name]) for name,value in before.items())


def test_same_parameter_name_with_incompatible_definition_rejected(cfg):
    source=preparation(cfg);opt=torch.optim.Adam(source.student.parameters(),lr=1e-3)
    saved=checkpoint(source,opt,cfg);receiver=preparation(cfg)
    layer=receiver.student.layers[0];old=layer.bias_ih_l0
    layer.bias_ih_l0=nn.Parameter(torch.zeros(old.numel()+1,dtype=old.dtype));layer._init_flat_weights()
    candidate=torch.optim.Adam(receiver.student.parameters(),lr=1e-3)
    with pytest.raises(ValueError,match="architecture|definition"):
        resume(saved,receiver,candidate,cfg)


def test_real_he_rejects_optimizer_class_and_parameter_alias(cfg,model):
    saved,signature,heads,optimizers,controller,sampler=checkpoint_fixture(model,cfg)
    incompatible={"model":torch.optim.AdamW(model.parameters(),lr=.001)}
    with pytest.raises(ValueError,match="implementation"):
        from humanengine.manifest import resume_he
        resume_he(saved,signature,model,heads,incompatible,controller,sampler,
                  expected_identity=saved["scientific_identity"])
    model.core.S.bias_hh_l0.data=model.core.S.bias_ih_l0.data
    aliased={"model":torch.optim.Adam(model.parameters(),lr=.001)}
    with pytest.raises(ValueError,match="Shared/overlapping parameter storage"):
        resume_he(saved,signature,model,heads,aliased,controller,sampler,
                  expected_identity=saved["scientific_identity"])


def diagnostic_toy(cfg):
    module=nn.Module();module.p=nn.Parameter(torch.tensor([.3,.7]));module.tag="stable"
    module.register_buffer("persistent",torch.tensor([2.]))
    module.register_buffer("ephemeral",torch.tensor([4.]),persistent=False)
    module.norm=nn.LayerNorm(2,elementwise_affine=False)
    optimizer=torch.optim.SGD(module.parameters(),lr=.01)
    macro=Macrostep(tuple(module.parameters()),{},optimizer,{},cfg,stateful_modules=(module,))
    closure=lambda:FamilyEvaluation(module.norm(module.p).square().sum(),{})
    return module,macro,closure


@pytest.mark.parametrize("change",["eps","ephemeral","persistent","parameter","mode","trainability","metadata"])
def test_mutating_diagnostic_cannot_change_future_continuation(cfg,change):
    module,macro,closure=diagnostic_toy(cfg)
    before=copy.deepcopy((module.state_dict(),module.ephemeral,module.norm.eps,module.training,
                          module.p.requires_grad,module.tag,macro.state_dict()))
    def bad():
        if change=="eps":module.norm.eps=1.
        elif change=="ephemeral":module.ephemeral.add_(1)
        elif change=="persistent":module.persistent.add_(1)
        elif change=="parameter":module.p.add_(1)
        elif change=="mode":module.training=not module.training
        elif change=="trainability":module.p.requires_grad_(False)
        else:module.tag="changed"
        return {"value":module.p.sum()}
    try:macro.step({"pose":closure,"reserve":closure},{"pose":.8,"reserve":.2},diagnostic=bad,diagnostic_module=module)
    except RuntimeError:pass
    after=(module.state_dict(),module.ephemeral,module.norm.eps,module.training,
           module.p.requires_grad,module.tag,macro.state_dict())
    assert before[0].keys()==after[0].keys()
    assert all(torch.equal(before[0][k],after[0][k]) for k in before[0])
    assert torch.equal(before[1],after[1]) and before[2:]==after[2:]


def test_pure_diagnostic_works_and_does_not_consume_rng(cfg):
    module,macro,closure=diagnostic_toy(cfg)
    random.seed(17);np.random.seed(17);torch.manual_seed(17)
    expected=(random.random(),float(np.random.rand()),torch.rand(2))
    random.seed(17);np.random.seed(17);torch.manual_seed(17)
    def pure():
        random.random();np.random.rand();torch.rand(2)
        return {"norm":module.p.norm()}
    result=macro.step({"pose":closure,"reserve":closure},{"pose":.8,"reserve":.2},diagnostic=pure,diagnostic_module=module)
    observed=(random.random(),float(np.random.rand()),torch.rand(2))
    assert result["post_losses"]["norm"]>0
    assert observed[0]==expected[0] and observed[1]==expected[1]
    torch.testing.assert_close(observed[2],expected[2],atol=0,rtol=0)


def test_successful_diagnostic_restores_controller_and_optimizer_metadata(cfg):
    module,macro,closure=diagnostic_toy(cfg);lr=macro.core_optimizer.param_groups[0]["lr"]
    def diagnostic():
        macro.steps=91;macro.core_optimizer.param_groups[0]["lr"]=.9
        return {"value":module.p.sum()}
    result=macro.step({"pose":closure,"reserve":closure},{"pose":.8,"reserve":.2},
                      diagnostic=diagnostic,diagnostic_module=module)
    assert result["post_losses"]["value"]!=0 and macro.steps==1
    assert macro.core_optimizer.param_groups[0]["lr"]==lr
