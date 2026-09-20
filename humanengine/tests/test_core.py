from dataclasses import replace
import torch
import pytest
from humanengine.core.state import detach_state
from humanengine.diagnostics.gradients import shadow_gradient
from .conftest import sequence


def test_arbitrary_chunk_equivalence_and_tick_alignment(model,cfg):
    seq=sequence(cfg);whole,_=model.forward_sequence(seq.emg,seq.timestamps)
    state=model.init_state(2);start=0;parts=[]
    for size in (1,7,31,2,67,139,3,250,300):
        end=min(start+size,seq.emg.shape[1])
        out,state=model.step(seq.emg[:,start:end],state,seq.timestamps[:,start:end]);parts.append(out)
        start=end
        if end==seq.emg.shape[1]:break
    for key in ("H","shared","residual","frontend","amplitude","timestamps"):
        actual=torch.cat([getattr(p,key) for p in parts],1)
        torch.testing.assert_close(actual,getattr(whole,key),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(whole.timestamps,seq.timestamps[:,cfg.stride-1::cfg.stride],atol=0,rtol=0)
    assert whole.H.shape[-1]==384 and whole.valid.all()


def test_future_input_causality_for_H_and_API(model,cfg):
    seq=sequence(cfg);x2=seq.emg.clone();x2[:,400:]+=100*torch.randn_like(x2[:,400:])
    a,_=model.forward_sequence(seq.emg,seq.timestamps);b,_=model.forward_sequence(x2,seq.timestamps)
    torch.testing.assert_close(a.H[:,:10],b.H[:,:10],atol=0,rtol=0)
    for name in a.apis:torch.testing.assert_close(a.apis[name][:,:10],b.apis[name][:,:10],atol=0,rtol=0)


def test_reset_gap_user_adapter_and_invalid_signal(model,cfg):
    seq=sequence(cfg,batch=1);times=seq.timestamps.clone();times[:,400:]+=1
    gap,state=model.forward_sequence(seq.emg,times,user_ids=("u1",))
    fresh,_=model.forward_sequence(seq.emg[:,400:],times[:,400:],user_ids=("u1",))
    torch.testing.assert_close(gap.H[:,10:],fresh.H)
    changed,_=model.step(seq.emg[:,:400],state,seq.timestamps[:,:400],user_ids=("u2",))
    new,_=model.forward_sequence(seq.emg[:,:400],seq.timestamps[:,:400],user_ids=("u2",))
    torch.testing.assert_close(changed.H,new.H)
    model.core.adapter_id="fixture-adapter-v2"
    adapter,_=model.step(seq.emg[:,:400],state,seq.timestamps[:,:400],user_ids=("u1",))
    new,_=model.forward_sequence(seq.emg[:,:400],seq.timestamps[:,:400],user_ids=("u1",))
    torch.testing.assert_close(adapter.H,new.H)
    bad=seq.emg.clone();bad[:,399]=float("nan")
    dropped,state=model.forward_sequence(bad,seq.timestamps)
    clean,_=model.forward_sequence(seq.emg[:,400:],seq.timestamps[:,400:])
    torch.testing.assert_close(dropped.H[:,-10:],clean.H)
    assert not dropped.input_quality["finite_input"]


def test_partial_fragment_bounded_caches_and_explicit_detach(model,cfg):
    seq=sequence(cfg,batch=1,length=4000);state=model.init_state(1)
    first,state=model.step(seq.emg[:,:3],state,seq.timestamps[:,:3]);assert first.H.shape[1]==0
    for start in range(3,4000,71):
        out,state=model.step(seq.emg[:,start:start+71],state,seq.timestamps[:,start:start+71]);state=detach_state(state)
    st=state.streams[0]
    assert st.seen==4000 and st.amplitude_tail.shape[1]==79
    assert all(c.tail.shape[1]==module.history for c,module in zip(st.conv,model.core.F.convs))
    assert st.r_h.grad_fn is None


def test_raw_amplitude_precedes_personalization(model,cfg):
    seq=sequence(cfg,batch=1)
    a,_=model.forward_sequence(seq.emg,seq.timestamps)
    with torch.no_grad():model.core.A.log_gain.fill_(.5);model.core.U.up.weight.fill_(.1)
    b,_=model.forward_sequence(seq.emg,seq.timestamps)
    torch.testing.assert_close(a.amplitude,b.amplitude,atol=0,rtol=0)
    assert not torch.allclose(a.frontend,b.frontend)
    assert model.core.U.down.weight.shape[-1]==cfg.feature_dim
    assert all(not p.requires_grad for m in (model.core.A,model.core.U) for p in m.parameters())


def test_shadow_gradient_is_observational(model,cfg):
    seq=sequence(cfg,batch=1,length=200)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    versions=[p._version for p in model.parameters()];rng=torch.get_rng_state().clone()
    states={k:v.clone() for k,v in model.state_dict().items()}
    result=shadow_gradient(model,seq.emg,seq.timestamps,lambda y:y["future"].square().mean(),optimizer)
    assert result["diagnostic_only"] and result["blocked"].norm()>0
    assert torch.equal(rng,torch.get_rng_state()) and versions==[p._version for p in model.parameters()]
    assert not optimizer.state and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in states.items())


def test_missing_labels_are_not_runtime_input_and_bad_schema_fails(model,cfg):
    seq=sequence(cfg,batch=1,length=160)
    _,state=model.forward_sequence(seq.emg[:,:80],seq.timestamps[:,:80])
    out,state=model.step(seq.emg[:,80:],state,seq.timestamps[:,80:])
    assert state.streams[0].seen==160 and not out.input_quality["reset_events"]
    with pytest.raises(ValueError,match="schema"):
        model.step(seq.emg[:,:,:1],state,seq.timestamps)


def test_explicit_reset_erases_both_memories_and_caches(model,cfg):
    seq=sequence(cfg,batch=1,length=400)
    _,state=model.forward_sequence(seq.emg,seq.timestamps)
    out,new=model.step(seq.emg,state,seq.timestamps,reset_mask=(True,))
    fresh,_=model.forward_sequence(seq.emg,seq.timestamps)
    torch.testing.assert_close(out.H,fresh.H)
    assert new.streams[0].seen==400


def test_compatible_adapter_loading_and_reset(model,cfg):
    import copy
    from humanengine.core.personalization import AdapterArtifact,load_adapter
    seq=sequence(cfg,batch=1,length=400);_,state=model.forward_sequence(seq.emg,seq.timestamps)
    affine=copy.deepcopy(model.core.A.state_dict());affine["log_gain"].fill_(.1)
    artifact=AdapterArtifact("personal-v1","core-hash","stats-hash",model.core.schema,affine,copy.deepcopy(model.core.U.state_dict()))
    with pytest.raises(ValueError,match="compatibility"):
        load_adapter(model,artifact,core_hash="wrong",statistics_hash="stats-hash")
    load_adapter(model,artifact,core_hash="core-hash",statistics_hash="stats-hash")
    out,_=model.step(seq.emg,state,seq.timestamps)
    fresh,_=model.forward_sequence(seq.emg,seq.timestamps)
    torch.testing.assert_close(out.H,fresh.H)
    assert out.input_quality["reset_events"][0][2]=="adapter_change"
    model.personalization_mode()
    assert all(not p.requires_grad for p in model.core.core_parameters())
    assert model.core.A.log_gain.requires_grad and not model.core.A.offset.requires_grad
    model.universal_mode()
    assert model.core.adapter_id=="identity" and not model.core.A.log_gain.any() and not model.core.U.up.weight.any()


def test_independent_stream_reset_does_not_change_other_stream(model,cfg):
    seq=sequence(cfg,batch=2,length=800)
    _,state=model.forward_sequence(seq.emg[:,:400],seq.timestamps[:,:400])
    normal,_=model.step(seq.emg[:,400:],state,seq.timestamps[:,400:])
    mixed,_=model.step(seq.emg[:,400:],state,seq.timestamps[:,400:],reset_mask=(True,False))
    torch.testing.assert_close(mixed.H[1],normal.H[1],atol=0,rtol=0)
