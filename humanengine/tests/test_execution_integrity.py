"""Execution v4 acceptance; manual adversaries and independent state/oracle checks.

All examples are synthetic CPU wiring tests, never scientific training evidence.
"""
import copy
import random
from dataclasses import asdict,replace
import numpy as np
import pytest
import torch
from torch import nn
from humanengine.optimization.macrostep import Macrostep,FamilyEvaluation
from humanengine.execution import SnapshotViolation,ReadOnlyEvaluation
from humanengine.transaction import UnusableTrainingState
from humanengine.manifest import (he_checkpoint,resume_he,teacher_checkpoint,resume_teacher,
                                  identify,seal)
from humanengine.teacher.preparation import TeacherPreparation,freeze_teacher,teacher_artifact_identity
from humanengine.teacher.model import FrozenTeacher
from humanengine.data.sampling import FamilyQuotaSampler,WindowRef
from .conftest import signal_stats
from .test_data_provenance import checkpoint_fixture
from .test_remediation import teacher_fixture

BUDGETS={"pose":.8,"reserve":.2}


def exact(a,b):
    """Independent equality: never calls production snapshot/hash helpers."""
    if type(a) is not type(b):return False
    if torch.is_tensor(a):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,random.Random):return a.getstate()==b.getstate()
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def state(modules,opts=(),controller=None,sampler=None):
    rows=[]
    for root in modules:
        rows.append({"tensors":[(n,p.detach(),p.requires_grad,p.grad) for n,p in root.named_parameters()],
                     "modules":{n:{"public":{k:v for k,v in vars(m).items() if not k.startswith('_')},
                                   "buffers":dict(m._buffers),"nonpersistent":set(m._non_persistent_buffers_set)}
                                for n,m in root.named_modules()}})
    return copy.deepcopy((rows,[o.state_dict() for o in opts],
                          None if controller is None else (controller.state_dict(),controller.cfg),
                          None if sampler is None else vars(sampler),
                          random.getstate(),np.random.get_state(),torch.get_rng_state()))


def toy(cfg):
    m=nn.Module();m.p=nn.Parameter(torch.tensor([.3,.7]));m.h=nn.Parameter(torch.tensor(.4))
    m.register_buffer('persistent',torch.tensor([2.]));m.register_buffer('ephemeral',torch.tensor([4.]),persistent=False)
    m.norm=nn.LayerNorm(2,elementwise_affine=False)
    opt=torch.optim.Adam([m.p],lr=.01);ho=torch.optim.Adam([m.h],lr=.01)
    for p,o in ((m.p,opt),(m.h,ho)):
        p.grad=torch.full_like(p,.7);o.step();p.grad=torch.full_like(p,.23)
    macro=Macrostep((m.p,),{'head':(m.h,)},opt,{'head':ho},cfg,stateful_modules=(m,))
    sampler=FamilyQuotaSampler([WindowRef('src','u','s','r','w','pose','g')],{'pose':1})
    def pose():
        loss=(m.h*m.p.sum()).square();return FamilyEvaluation(loss,{'head':loss})
    def reserve():return FamilyEvaluation((2*m.p).square().sum(),{})
    return m,macro,sampler,{'pose':pose,'reserve':reserve}


def snapshot(m,macro,s):return state((m,),(macro.core_optimizer,*macro.private_optimizers.values()),macro,s)


def test_independent_snapshot_oracle_rng_sensitivity(cfg):
    m,macro,s,cl=toy(cfg);before=snapshot(m,macro,s)
    assert exact(before,snapshot(m,macro,s))
    s.rng.random()
    assert not exact(before,snapshot(m,macro,s))


def test_analytic_authoritative_gradient(cfg):
    m,macro,s,cl=toy(cfg);p=m.p.detach().clone();h=m.h.detach().clone()
    oracle={'pose':2*h.square()*p.sum()*torch.ones_like(p),'reserve':8*p}
    result=macro.step(cl,BUDGETS,sampler=s)
    for name,g in zip(result['families'],result['gradients']):torch.testing.assert_close(g,oracle[name])
    assert result['snapshot_consistent'] and macro.steps==1


@pytest.mark.parametrize('name',['p','h','persistent','ephemeral'])
@pytest.mark.parametrize('transient',[False,True])
def test_write_before_compute_rejected(cfg,name,transient):
    m,macro,s,cl=toy(cfg);before=snapshot(m,macro,s);base=cl['reserve'];reached=[]
    def bad():
        target=getattr(m,name);saved=target.detach().clone()
        target.data.add_(1)
        reached.append('after write')
        loss=base()
        if transient:target.data.copy_(saved)
        return loss
    cl['reserve']=bad
    with pytest.raises(SnapshotViolation,match='protected'):macro.step(cl,BUDGETS,sampler=s)
    assert not reached and exact(before,snapshot(m,macro,s)) and macro.steps==0


@pytest.mark.parametrize('field',['eps','stride'])
@pytest.mark.parametrize('transient',[False,True])
def test_real_frontend_execution_attribute_rejected(cfg,model,field,transient):
    frontend=model.core.F;x=torch.randn(1,200,cfg.channels)
    target=frontend.norms[0] if field=='eps' else frontend.convs[0]
    old=getattr(target,field);new=1. if field=='eps' else 1
    clean=frontend.forward_sequence(x).detach()
    setattr(target,field,new);altered=frontend.forward_sequence(x).detach();setattr(target,field,old)
    assert clean.shape!=altered.shape or not torch.equal(clean,altered)
    params=tuple(frontend.parameters());opt=torch.optim.SGD(params,lr=.01)
    macro=Macrostep(params,{},opt,{},cfg,stateful_modules=(model,))
    def pure():return FamilyEvaluation(frontend.forward_sequence(x)[...,0].square().sum(),{})
    def bad():
        setattr(target,field,new)
        loss=pure()
        if transient:setattr(target,field,old)
        return loss
    before=state((model,),(opt,),macro)
    with pytest.raises(SnapshotViolation,match='attributes'):macro.step({'pose':pure,'reserve':bad},BUDGETS)
    assert exact(before,state((model,),(opt,),macro))


@pytest.mark.parametrize('kind',['loss','intermediate'])
def test_prebuilt_graph_rejected(cfg,kind):
    m,macro,s,cl=toy(cfg);outside=(2*m.p).square();before=snapshot(m,macro,s)
    outside_loss=outside.sum()
    cl['reserve']=lambda:FamilyEvaluation(outside_loss if kind=='loss' else outside.sum(),{})
    with pytest.raises(SnapshotViolation,match='outside|prebuilt'):macro.step(cl,BUDGETS,sampler=s)
    assert exact(before,snapshot(m,macro,s))


def test_caught_violation_latches(cfg):
    m,macro,s,cl=toy(cfg);before=snapshot(m,macro,s);base=cl['reserve']
    def bad():
        try:m.p.data.add_(1)
        except SnapshotViolation:pass
        return base()
    cl['reserve']=bad
    with pytest.raises(SnapshotViolation,match='previous violation'):macro.step(cl,BUDGETS,sampler=s)
    assert exact(before,snapshot(m,macro,s))


@pytest.mark.parametrize('kind',['all','commit','binding','layout','parameter_hook','module_hook','rnn_cache'])
def test_transaction_all_continuation_state(cfg,kind):
    m,macro,s,cl=toy(cfg)
    if kind=='rnn_cache':m.rnn=nn.LSTM(2,2,batch_first=True)
    before=snapshot(m,macro,s);base=cl['reserve'];old_storage=m.p.untyped_storage()._cdata
    old_grad=m.p.grad
    def corrupt():
        s.quotas['pose']=9;s.exposure['pose']=7;s.rng.random()
        random.random();np.random.rand();torch.rand(3)
        macro.steps+=3;m.h.grad=torch.zeros_like(m.h)
        macro.core_optimizer.state[m.p]['exp_avg'].add_(8)
    def bad():
        if kind=='all':corrupt()
        elif kind=='binding':m.p.data=torch.zeros_like(m.p)
        elif kind=='layout':m.p.data=m.p.data.view(torch.float16)[:2]
        elif kind=='parameter_hook':m.p.register_hook(lambda g:g*100)
        elif kind=='module_hook':m.norm.register_forward_hook(lambda *a:None)
        elif kind=='rnn_cache':m.rnn._flat_weights[0]=torch.zeros_like(m.rnn.weight_ih_l0)
        return base()
    if kind=='commit':
        def fail_step(*a,**k):
            corrupt();raise RuntimeError('injected optimizer commit failure')
        macro.private_optimizers['head'].step=fail_step
    else:cl['reserve']=bad
    with pytest.raises(RuntimeError):macro.step(cl,BUDGETS,sampler=s)
    assert exact(before,snapshot(m,macro,s))
    assert m.p.untyped_storage()._cdata==old_storage and m.p.grad is old_grad
    if kind=='parameter_hook':assert not m.p._backward_hooks
    if kind=='module_hook':assert not m.norm._forward_hooks
    if kind=='rnn_cache':assert m.rnn._flat_weights[0] is m.rnn.weight_ih_l0


def test_failed_rollback_poison_is_sticky(cfg,monkeypatch):
    import humanengine.transaction as tx
    m,macro,s,cl=toy(cfg)
    def broken_rng(state):raise RuntimeError('injected rollback failure')
    monkeypatch.setattr(tx,'restore_rng',broken_rng)
    cl['reserve']=lambda:(_ for _ in ()).throw(RuntimeError('rejected evaluation'))
    with pytest.raises(UnusableTrainingState,match='poisoned'):macro.step(cl,BUDGETS,sampler=s)
    assert all(getattr(x,'_he_unusable',False) for x in (m,macro,s,macro.core_optimizer,*macro.private_optimizers.values()))
    with pytest.raises(UnusableTrainingState,match='unusable'):macro.step(cl,BUDGETS,sampler=s)


@pytest.mark.parametrize('field',['eps','stride'])
@pytest.mark.parametrize('operation',['create','resume'])
def test_he_checkpoint_actual_executable(cfg,model,field,operation):
    ck,sig,h,opts,c,s=checkpoint_fixture(model,cfg)
    target=model.core.F.norms[0] if field=='eps' else model.core.F.convs[0]
    setattr(target,field,1. if field=='eps' else 1)
    before=state((model,h),tuple(opts.values()),c,s)
    with pytest.raises(ValueError,match='executable'):
        if operation=='create':he_checkpoint(model,h,opts,c,s,sig,ck['registry'],ck['reference_scales'],ck['config'])
        else:resume_he(ck,sig,model,h,opts,c,s)
    assert exact(before,state((model,h),tuple(opts.values()),c,s))


@pytest.mark.parametrize('field',['eps','stride'])
@pytest.mark.parametrize('operation',['create','resume'])
def test_teacher_checkpoint_actual_executable(cfg,field,operation):
    p=TeacherPreparation(cfg,signal_stats(cfg,True));opt=torch.optim.Adam(p.student.parameters())
    ck=teacher_checkpoint(p,opt,cfg.to_dict(),'data','code',{})
    target=p.ema.frontend.norms[0] if field=='eps' else p.student.frontend.convs[0]
    setattr(target,field,1. if field=='eps' else 1);before=state((p,),(opt,))
    with pytest.raises(ValueError,match='executable'):
        if operation=='create':teacher_checkpoint(p,opt,cfg.to_dict(),'data','code',{})
        else:resume_teacher(ck,p,opt,config=cfg.to_dict(),data_manifest_hash='data',source_version='code')
    assert exact(before,state((p,),(opt,)))


@pytest.mark.parametrize('field',['mean','second','count'])
@pytest.mark.parametrize('surface',['preparation','continuation'])
def test_teacher_running_state_outer_reseal_rejected(cfg,field,surface):
    p=TeacherPreparation(cfg,signal_stats(cfg,True));opt=torch.optim.Adam(p.student.parameters())
    ck=teacher_checkpoint(p,opt,cfg.to_dict(),'data','code',{})
    if surface=='preparation':ck['preparation']['target_statistics.'+field].add_(1)
    else:ck['continuation']['buffers']['target_statistics.'+field]['value'].add_(1)
    ck=seal({k:v for k,v in ck.items() if k!='content_hash'});before=state((p,),(opt,))
    with pytest.raises(ValueError,match='identity'):resume_teacher(ck,p,opt,config=cfg.to_dict(),data_manifest_hash='data',source_version='code')
    assert exact(before,state((p,),(opt,)))


def test_legitimate_running_state_and_gradients_resume(cfg):
    p=TeacherPreparation(cfg,signal_stats(cfg,True));opt=torch.optim.Adam(p.student.parameters())
    p.target_statistics.mean.fill_(.25);p.target_statistics.second.fill_(2);p.target_statistics.count.fill_(9)
    p.register_buffer('extra_ephemeral',torch.tensor([7.]),persistent=False)
    first=next(p.student.parameters());first.grad=torch.full_like(first,.37)
    ck=teacher_checkpoint(p,opt,cfg.to_dict(),'data','code',{})
    fresh=TeacherPreparation(cfg,signal_stats(cfg,True));fresh.register_buffer('extra_ephemeral',torch.tensor([0.]),persistent=False)
    fresh_opt=torch.optim.Adam(fresh.student.parameters())
    resume_teacher(ck,fresh,fresh_opt,config=cfg.to_dict(),data_manifest_hash='data',source_version='code',expected_identity=ck['scientific_identity'])
    assert torch.equal(fresh.target_statistics.mean,torch.full_like(fresh.target_statistics.mean,.25))
    assert torch.equal(fresh.target_statistics.second,torch.full_like(fresh.target_statistics.second,2))
    assert torch.equal(fresh.target_statistics.count,torch.full_like(fresh.target_statistics.count,9))
    assert torch.equal(next(fresh.student.parameters()).grad,first.grad) and float(fresh.extra_ephemeral)==7


@pytest.mark.parametrize('lane',['he','teacher'])
def test_late_resume_restores_existing_gradients(cfg,model,monkeypatch,lane):
    if lane=='he':
        ck,sig,h,opts,c,s=checkpoint_fixture(model,cfg);opt=opts['model'];modules=(model,h)
        def resume():resume_he(ck,sig,model,h,opts,c,s)
    else:
        p=TeacherPreparation(cfg,signal_stats(cfg,True));opt=torch.optim.Adam(p.student.parameters());opts={'teacher':opt}
        ck=teacher_checkpoint(p,opt,cfg.to_dict(),'data','code',{});modules=(p,);c=s=None
        def resume():resume_teacher(ck,p,opt,config=cfg.to_dict(),data_manifest_hash='data',source_version='code')
    for m in modules:
        for param in m.parameters():param.grad=torch.full_like(param,.123)
    before=state(modules,tuple(opts.values()),c,s);reached=[]
    def failure(_):
        reached.append(True)
        for m in modules:
            for param in m.parameters():param.grad=torch.full_like(param,77.)
        random.random();np.random.rand();torch.rand(2)
        if s is not None:s.quotas['pose']=91
        raise RuntimeError('late gradient failure')
    monkeypatch.setattr(opt,'load_state_dict',failure)
    with pytest.raises(RuntimeError,match='late gradient'):resume()
    assert reached and exact(before,state(modules,tuple(opts.values()),c,s))


@pytest.mark.parametrize('lane',['he','teacher'])
def test_contradictory_buffer_representations_rejected(cfg,model,lane):
    if lane=='he':
        ck,sig,h,opts,c,s=checkpoint_fixture(model,cfg)
        ck['model_continuation']['buffers']['core.E.statistics.mean']['value'].add_(1)
        ck=identify({k:v for k,v in ck.items() if k not in ('content_hash','scientific_identity')})
        call=lambda:resume_he(ck,sig,model,h,opts,c,s)
    else:
        p=TeacherPreparation(cfg,signal_stats(cfg,True));opt=torch.optim.Adam(p.student.parameters())
        ck=teacher_checkpoint(p,opt,cfg.to_dict(),'data','code',{})
        ck['continuation']['buffers']['target_statistics.mean']['value'].add_(1)
        ck=identify({k:v for k,v in ck.items() if k not in ('content_hash','scientific_identity')})
        call=lambda:resume_teacher(ck,p,opt,config=cfg.to_dict(),data_manifest_hash='data',source_version='code')
    with pytest.raises(ValueError,match='Conflicting'):call()


def test_external_checkpoint_identity_pin(cfg,model):
    ck,sig,h,opts,c,s=checkpoint_fixture(model,cfg);pin=ck['scientific_identity']
    ck['model']['core.A.log_gain'].add_(.1)
    ck=identify({k:v for k,v in ck.items() if k not in ('content_hash','scientific_identity')})
    with pytest.raises(ValueError,match='identity'):resume_he(ck,sig,model,h,opts,c,s,expected_identity=pin)


@pytest.mark.parametrize('field',['trainability','training'])
def test_checkpoint_continuation_cannot_contradict_semantics(cfg,model,field):
    ck,sig,h,opts,c,s=checkpoint_fixture(model,cfg)
    row=ck['model_continuation'][field];key=next(iter(row));row[key]=not row[key]
    ck=identify({k:v for k,v in ck.items() if k not in ('content_hash','scientific_identity')})
    with pytest.raises(ValueError,match='Conflicting'):resume_he(ck,sig,model,h,opts,c,s)


@pytest.mark.parametrize('version',[1,2,3])
def test_legacy_checkpoint_rejected(cfg,model,version):
    ck,sig,h,opts,c,s=checkpoint_fixture(model,cfg);ck['schema']='he-kernel-checkpoint-v'+str(version)
    with pytest.raises(ValueError,match='schema'):resume_he(ck,sig,model,h,opts,c,s)


@pytest.mark.parametrize('field',['eps','stride','ema','target','nonpersistent','target_nonpersistent'])
def test_frozen_teacher_direct_actual_artifact(cfg,field):
    prep,stats,health=teacher_fixture(cfg)
    if field=='nonpersistent':prep.ema.register_buffer('ephemeral',torch.tensor([1.]),persistent=False)
    if field=='target_nonpersistent':stats.register_buffer('ephemeral',torch.tensor([1.]),persistent=False)
    health=replace(health,artifact_identity=teacher_artifact_identity(prep,stats,health))
    original=freeze_teacher(prep,stats,health)
    if field=='eps':prep.ema.frontend.norms[0].eps=1.
    elif field=='stride':prep.ema.frontend.convs[0].stride=1
    elif field=='ema':next(prep.ema.parameters()).data.add_(1)
    elif field=='target':stats.mean.add_(1)
    elif field=='target_nonpersistent':stats.ephemeral.add_(1)
    else:prep.ema.ephemeral.add_(1)
    with pytest.raises(ValueError):FrozenTeacher(prep.ema,stats,original.artifact)


@pytest.mark.parametrize('field',['evaluation_version','evaluation_config','data_manifest_hash','checkpoint_identity',
    'partition','partition_identity','budget_identity','checkpoint_step','preregistered_step',
    'raw_std_threshold','within_std_threshold','amplitude_threshold','evidence_partition','target_version',
    'certificate_version','checks','health_passed'])
def test_direct_teacher_contradictory_provenance(cfg,field):
    prep,stats,health=teacher_fixture(cfg);original=freeze_teacher(prep,stats,health);artifact=copy.deepcopy(original.artifact)
    if field in ('evidence_partition','target_version','certificate_version'):artifact[field]='contradictory'
    elif field=='checks':artifact['checks']['finite']=False
    elif field=='health_passed':artifact[field]=False
    else:
        old=artifact['health'][field]
        artifact['health'][field]=({'contradiction':True} if isinstance(old,dict) else old+1 if isinstance(old,(float,int)) else 'contradictory')
    with pytest.raises(ValueError):FrozenTeacher(prep.ema,stats,artifact)


def test_valid_direct_teacher_output_parity(cfg):
    prep,stats,health=teacher_fixture(cfg);original=freeze_teacher(prep,stats,health)
    direct=FrozenTeacher(prep.ema,stats,original.artifact);x=torch.randn(1,200,cfg.channels)
    reexport=FrozenTeacher(original.encoder,original.statistics,original.artifact)
    with torch.no_grad():a=original.encoder.window(x);b=direct.encoder.window(x);c=reexport.encoder.window(x)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.testing.assert_close(a,c,atol=0,rtol=0)
