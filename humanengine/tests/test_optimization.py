from types import SimpleNamespace
import numpy as np
import torch
import pytest
from torch import nn
from humanengine.optimization.cagrad import weighted_cagrad
from humanengine.optimization.macrostep import Macrostep,FamilyEvaluation
from humanengine.registry import APIRegistry,APIFamily,Component,family_loss,reserve_loss
from humanengine.data.sampling import FamilyQuotaSampler,WindowRef
from humanengine.contracts import LossTerm
from humanengine.diagnostics.gradients import flat_gradient


def family(name):
    return APIFamily(name,"1.0",(Component("current",1.,2.,.1,"explicit-valid","frozen-v1"),),
                     (name+"-source",),name+"-P",(name+"-head",),"none-yet","api-detached-r-v1")


def test_family_vote_invariance_E0(cfg):
    source=[WindowRef("pose-source",f"u{i%2}",f"s{i%3}",f"r{i%4}",f"w{i}","pose",f"g{i}") for i in range(8)]
    force=[WindowRef("force-source","u0","s0","r0","w0","force","force-g")]
    a=FamilyQuotaSampler(source+force,{"pose":7,"force":7},11)
    b=FamilyQuotaSampler(source*5+force,{"pose":7,"force":7},11)
    registry=APIRegistry((family("pose"),family("force")),cfg.reserve_alpha)
    assert registry.budgets()=={"force":.4,"pose":.4,"reserve":.2}
    for _ in range(5):assert a.sample()==b.sample()
    assert a.exposure==b.exposure=={"pose":35,"force":35}
    gradient=lambda sampler:torch.tensor([float(len(sampler.windows)),float(sampler.exposure["pose"])])
    torch.testing.assert_close(registry.budgets()["pose"]*gradient(a),registry.budgets()["pose"]*gradient(b),atol=0,rtol=0)
    state=a.state_dict();expected=a.sample();a.load_state_dict(state);assert a.sample()==expected


def test_family_fixed_denominator_missing_is_NA(cfg):
    f=APIFamily("pose","v1",tuple(Component(k,w,s,.01,"valid","frozen") for k,w,s in (("current",1,2),("future",.5,4),("geometry",.05,1))),
                 ("s",),"p",("h",),"none",cfg.route_version)
    terms={k:LossTerm(torch.tensor(v),1,torch.tensor(v)) for k,v in (("current",4.),("future",8.),("geometry",1.))}
    assert float(family_loss(f,terms).value)==pytest.approx((2+1+.05)/1.55)
    terms["future"]=LossTerm.na("missing")
    assert family_loss(f,terms).value is None
    assert reserve_loss(torch.tensor(2.),None,cfg,1.).value is None
    assert float(reserve_loss(torch.tensor(2.),torch.tensor(3.),cfg,2.).value)==pytest.approx(1.3/1.1)


def test_cagrad_trust_region_primal_dual_and_angular_reference():
    G=torch.tensor([[1.,0.],[0.,1.],[-.2,.6]],dtype=torch.float64);alpha=[.4,.4,.2]
    result=weighted_cagrad(list(G),alpha,.25)
    assert result.status=="COORDINATED",result.fallback_reason
    assert result.trust_residual<1e-7 and result.dual_gap<1e-6
    assert float(result.base@result.direction)>=(1-.25)*float(result.base.square().sum())-1e-9
    angles=torch.linspace(0,2*torch.pi,20001,dtype=torch.float64)
    trial=result.base[:,None]+.25*result.base.norm()*torch.stack((angles.cos(),angles.sin()))
    best=(G@trial).min(0).values.max()
    assert float((G@result.direction).min())>=float(best)-1e-6
    torch.testing.assert_close(result.coefficients@G,result.direction)


def test_cagrad_aligned_and_K1_base():
    g=[torch.tensor([1.,2.]),torch.tensor([2.,4.])]
    base=weighted_cagrad(g,[.8,.2],0)
    assert base.status=="BASE_ONLY"
    result=weighted_cagrad(g,[.8,.2],.25)
    torch.testing.assert_close(result.direction,1.25*result.base)


def test_cagrad_degenerate_fallback_missing_nonfinite_and_solver_failure():
    g=torch.tensor([1.,-1.])
    zero=weighted_cagrad([g,-g],[.5,.5],.25)
    assert zero.fallback_reason=="zero_base_gradient" and not zero.direction.any()
    onezero=weighted_cagrad([g,torch.zeros_like(g)],[.8,.2],.25)
    assert onezero.status=="FALLBACK";torch.testing.assert_close(onezero.direction,onezero.base)
    failed=weighted_cagrad([g,g*2],[.8,.2],.25,solver=lambda *a,**k:SimpleNamespace(success=False,message="injected failure"))
    assert failed.status=="FALLBACK" and "injected failure" in failed.fallback_reason
    with pytest.raises(ValueError,match="Missing"):
        weighted_cagrad([g,None],[.8,.2],.25)
    with pytest.raises(FloatingPointError):
        weighted_cagrad([g,g*float("nan")],[.8,.2],.25)
    invalid=weighted_cagrad([g,g*2],[.8,.2],.25,solver=lambda *a,**k:SimpleNamespace(success=True,x=np.array([2.,-1.])))
    assert invalid.fallback_reason=="invalid_simplex"


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.core=nn.Parameter(torch.tensor([.3,.7,.2]))
        self.pose=nn.Parameter(torch.tensor(.4));self.force=nn.Parameter(torch.tensor(.5));self.reserve=nn.Parameter(torch.tensor(.6))


def setup_macro(cfg,*,adam=False):
    m=Toy();cls=torch.optim.Adam if adam else torch.optim.SGD
    opt=cls([m.core],lr=.01);priv={k:(getattr(m,k),) for k in ("pose","force","reserve")}
    opts={k:torch.optim.SGD(v,lr=.01) for k,v in priv.items()}
    macro=Macrostep((m.core,),priv,opt,opts,cfg)
    snapshots=[]
    def closure(name):
        def evaluate():
            snapshots.append(m.core.detach().clone())
            f,s,r=m.core.unbind()
            loss=(getattr(m,name)*(r+f if name=="reserve" else f+s+r.detach())).square()
            return FamilyEvaluation(loss,{name:loss})
        return evaluate
    return m,macro,{k:closure(k) for k in priv},snapshots


def test_same_snapshot_and_private_vs_core_update_separation(cfg):
    m,macro,closures,snapshots=setup_macro(cfg)
    initial=m.core.detach().clone();private_before={k:getattr(m,k).detach().clone() for k in closures}
    expected={k:flat_gradient(closures[k]().private_losses[k],(getattr(m,k),)).clone() for k in closures}
    for p in m.parameters():p.grad=torch.full_like(p,999.) # stale grads must not contribute
    result=macro.step(closures,{"pose":.4,"force":.4,"reserve":.2},
                      diagnostic=lambda:{k:closures[k]().core_loss for k in closures},diagnostic_module=m)
    assert all(torch.equal(s,initial) for s in snapshots[:-3])
    for k in closures:
        torch.testing.assert_close(getattr(m,k),private_before[k]-.01*expected[k].reshape(()))
    assert result["gradients"][result["families"].index("pose")][-1]==0
    torch.testing.assert_close(result["actual_delta"],-.01*result["solution"].direction,atol=1e-7,rtol=1e-5)
    assert result["post_losses"] and result["snapshot_consistent"]


def test_macrostep_rejects_intra_family_parameter_update(cfg):
    m,macro,closures,_=setup_macro(cfg);before=m.core.detach().clone();base=closures["force"]
    def bad():
        with torch.no_grad():m.core.add_(1)
        return base()
    closures["force"]=bad
    with pytest.raises(RuntimeError,match="snapshot"):macro.step(closures,{"pose":.4,"force":.4,"reserve":.2})
    torch.testing.assert_close(m.core,before,atol=0,rtol=0)
    assert macro.steps==0


def test_macrostep_reports_real_Adam_delta(cfg):
    m,macro,closures,_=setup_macro(cfg,adam=True)
    result=macro.step(closures,{"pose":.4,"force":.4,"reserve":.2})
    assert not torch.allclose(result["actual_delta"],-.01*result["solution"].direction)
    for g,v in zip(result["gradients"],result["g_dot_delta"]):assert v==pytest.approx(float(g@result["actual_delta"]))


def test_macrostep_NA_has_no_partial_update(cfg):
    m,macro,closures,_=setup_macro(cfg);before={k:v.clone() for k,v in m.state_dict().items()}
    closures["reserve"]=lambda:FamilyEvaluation(None,{})
    with pytest.raises(ValueError,match="NA"):macro.step(closures,{"pose":.4,"force":.4,"reserve":.2})
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and macro.steps==0


def test_valid_anchor_quota_plans_cap_exposure_and_report_missing():
    from humanengine.data.sampling import quota_mask,AnchorIdentity
    w=WindowRef("s","u","ss","r","w","pose","g")
    sampler=FamilyQuotaSampler([w],{"pose":1})
    # See REMEDIATION_NOTES: replacement cannot manufacture unique exposure.
    plan=sampler.plan_valid_quotas({"pose":{"current":7,"future":3}},lambda w:{"current":list(range(5)),"future":[0,1]},max_attempts=3)
    assert plan["status"]=="NA"
    assert sum(len(grants["current"]) for _,grants in plan["allocations"]["pose"])==5
    assert sum(len(grants["future"]) for _,grants in plan["allocations"]["pose"])==2
    absent=sampler.plan_valid_quotas({"pose":{"current":1,"future":1}},lambda w:{"current":[0]},max_attempts=2)
    assert absent["status"]=="NA" and absent["missing"]["pose"]["future"]==1
    mask=quota_mask(torch.tensor([True,False,True,True]),[AnchorIdentity.from_window(w,i,"current") for i in [0,2]])
    assert mask.tolist()==[True,False,True,False]
