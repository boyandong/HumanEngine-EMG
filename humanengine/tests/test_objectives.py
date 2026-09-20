import torch
import pytest
from humanengine.heads.api_heads import TrainingHeads
from humanengine.objectives.reductions import masked_error,latent_error,pose_losses
from humanengine.objectives.regularizers import variance_floor,vicreg,pose_geometry,DegeneracyMonitor
from humanengine.objectives.residual import residual_losses
from humanengine.core.summaries import SignalStatistics,physical_summary
from humanengine.contracts import AnchorMetadata
from .conftest import sequence,anchor_meta


def reach(loss,groups):
    params=[p for group in groups.values() for p in group]
    grads=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
    result={};offset=0
    for key,group in groups.items():
        values=grads[offset:offset+len(group)];offset+=len(group)
        result[key]=any(g is not None and float(g.abs().sum())>0 for g in values)
    return result


@pytest.mark.parametrize("component",["current","future"])
def test_explicit_API_detached_R_reachability(model,cfg,component):
    seq=sequence(cfg,batch=1,length=400);out,_=model.forward_sequence(seq.emg,seq.timestamps)
    groups={"F":tuple(model.core.F.parameters()),"S":tuple(model.core.S.parameters()),"R":tuple(model.core.R.parameters()),
            "private":tuple(model.api_heads.parameters())}
    result=reach(out.apis["pose/"+component].square().mean(),groups)
    assert result=={"F":True,"S":True,"R":False,"private":True}
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("component",["masked","observed","physical","variance"])
def test_residual_individual_gradient_routes(model,cfg,component):
    # 64 independently reset synthetic records/windows, with explicit support metadata.
    seq=sequence(cfg,batch=64,length=80);out,_=model.forward_sequence(seq.emg,seq.timestamps)
    r=out.residual[:,-1:];heads=TrainingHeads(cfg)
    target=torch.randn(64,1,3,2,128);valid=torch.ones(64,1,3,2,dtype=torch.bool)
    phys=torch.randn(64,1,12);pv=torch.ones(64,1,dtype=torch.bool)
    bundle=residual_losses(heads,r,r,target,valid,phys,pv,anchor_meta(),cfg)
    result=reach(bundle.terms[component].value,{"F":tuple(model.core.F.parameters()),"S":tuple(model.core.S.parameters()),
                  "R":tuple(model.core.R.parameters()),"P_R":tuple(heads.P_R.parameters()),"M_A":tuple(heads.M_A.parameters())})
    assert result["F"] and result["R"] and not result["S"]
    assert result["P_R"]==(component in ("masked","observed"))
    assert result["M_A"]==(component=="physical")
    expected=bundle.terms["masked"].value+.1*bundle.terms["observed"].value+.05*bundle.terms["physical"].value+.01*bundle.terms["variance"].value
    torch.testing.assert_close(bundle.total,expected)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("component",["geometry","vicreg"])
def test_geometry_and_vicreg_gradient_routes(model,cfg,component):
    seq=sequence(cfg,batch=64,length=80);out,_=model.forward_sequence(seq.emg,seq.timestamps)
    heads=TrainingHeads(cfg);s=out.shared[:,-1];meta=anchor_meta()
    if component=="geometry":
        stats=SignalStatistics(torch.zeros(20),torch.ones(20),{"partition":"synthetic","version":"pose"},cfg.pose_scale_floor)
        term=pose_geometry(heads.G(s),torch.randn(64,20),meta,stats,cfg)
    else:
        noisy,_=model.forward_sequence(seq.emg+.001*torch.randn_like(seq.emg),seq.timestamps)
        term=vicreg(heads.V(s),heads.V(noisy.shared[:,-1]),meta,cfg)
    result=reach(term.value,{"F":tuple(model.core.F.parameters()),"S":tuple(model.core.S.parameters()),"R":tuple(model.core.R.parameters())})
    assert result=={"F":True,"S":True,"R":False}


def test_valid_count_nan_missing_and_equal_scale_reduction(cfg):
    p=torch.tensor([1.,3.,99.],requires_grad=True);t=torch.tensor([0.,1.,float("nan")])
    result=masked_error(p,t,torch.tensor([True,True,False]),kind="l1")
    assert result.count==2 and result.value==1.5
    assert masked_error(p,t,torch.zeros(3,dtype=torch.bool)).value is None
    pred=torch.zeros(1,3,3,2,128,requires_grad=True);target=torch.zeros_like(pred);valid=torch.zeros(1,3,3,2,dtype=torch.bool)
    target[:,:,0,0]=2.;valid[:,:,0,0]=True;target[:,0,1,0]=4.;valid[:,0,1,0]=True
    term=latent_error(pred,target,valid)
    assert float(term.value)==pytest.approx((1.5+3.5)/2) # not weighted 3:1 by frame counts


def test_variance_support_is_NA_and_no_neighbor_padding(cfg):
    x=torch.randn(64,128)
    short=variance_floor(x[:2],AnchorMetadata(("a","b"),("a","b"),(0.,0.)),cfg)
    assert short.value is None and short.status.startswith("NA")
    repeated=AnchorMetadata(tuple("u"+str(i%4) for i in range(64)),tuple("r"+str(i%8) for i in range(64)),tuple(i*.0001 for i in range(64)))
    assert variance_floor(x,repeated,cfg).value is None
    assert variance_floor(x,anchor_meta(),cfg).value is not None


def test_physical_summary_sinusoid_energy_and_amplitude(cfg):
    t=torch.arange(200)/cfg.sample_rate
    x=torch.sin(2*torch.pi*120*t)[None,:,None].repeat(1,1,2)
    a=physical_summary(x,cfg).reshape(1,2,6)
    b=physical_summary(2*x,cfg).reshape_as(a)
    assert a[0,0,2]>a[0,0,1]+5 and a[0,0,2]>a[0,0,3]+5
    assert float(a[0,0,2])==pytest.approx(float(torch.tensor(.5).log()),abs=1e-4)
    assert float(b[0,0,0]-a[0,0,0])==pytest.approx(float(torch.tensor(2.).log()),abs=1e-5)


def test_geometry_degeneracy_is_reported_and_stops(cfg):
    stats=SignalStatistics(torch.zeros(20),torch.ones(20),{"partition":"synthetic","version":"pose"},cfg.pose_scale_floor)
    z=torch.zeros(64,32,requires_grad=True);meta=anchor_meta()
    assert pose_geometry(z,torch.zeros(64,20),meta,stats,cfg).value is None
    term=pose_geometry(z,torch.randn(64,20),meta,stats,cfg)
    assert term.status=="DEGENERATE_STUDENT" and torch.isfinite(term.value)
    monitor=DegeneracyMonitor();monitor.observe(term);monitor.observe(term)
    with pytest.raises(RuntimeError):monitor.observe(term)
