from dataclasses import dataclass
import numpy as np
import torch
from scipy.optimize import minimize


@dataclass
class DirectionResult:
    direction: torch.Tensor
    base: torch.Tensor
    status: str
    fallback_reason: str | None
    coefficients: torch.Tensor
    direction_gains: torch.Tensor
    trust_residual: float
    dual_gap: float | None


def weighted_cagrad(gradients,alpha,c,*,tolerance=1e-7,max_iterations=500,solver=None):
    """Addendum 2.3 dual; certify trust ball AND duality gap, else use g0.

    min_(w in simplex) g_w.g0 + c||g0|| ||g_w||.
    No per-family gradient normalization. One common scaling aids numerics.
    """
    if any(g is None for g in gradients):raise ValueError("Missing family gradient is NA, not zero")
    if not gradients or len(gradients)!=len(alpha):raise ValueError("Family/alpha mismatch")
    shape=gradients[0].shape
    if any(g.shape!=shape or g.ndim!=1 for g in gradients):raise ValueError("Need full matching core vectors")
    G=torch.stack([g.detach() for g in gradients]);a=torch.as_tensor(alpha,device=G.device,dtype=G.dtype)
    if not torch.isfinite(G).all() or not torch.isfinite(a).all():
        raise FloatingPointError("Nonfinite family gradient: no safe finite base update; abort macrostep")
    if (a<0).any() or abs(float(a.sum())-1)>1e-6 or not 0<=c<1:raise ValueError("Invalid simplex budget/trust coefficient")
    G64=G.double();a64=a.double()
    base=(a64@G64).to(G.dtype);bn=float(base.double().norm());radius=c*bn
    def result(d,status,reason,coeff,gap=None):
        return DirectionResult(d,base,status,reason,coeff,G64@d.double(),float((d.double()-base.double()).norm())-radius,gap)
    if c==0:return result(base,"BASE_ONLY",None,a,0.)
    if bn==0:return result(base,"FALLBACK","zero_base_gradient",a)
    if (G64.norm(dim=1)==0).any():return result(base,"FALLBACK","zero_family_gradient_degenerate_min",a)
    # Double precision KxK geometry, with a common scale rather than task whitening.
    scale=float(G64.norm(dim=1).max());g=(G64/scale).cpu().numpy()
    an=a.double().cpu().numpy();h=an@g;r=c*np.linalg.norm(h);gram=g@g.T;linear=g@h
    def objective(w):return float(w@linear+r*np.linalg.norm(w@g))
    def derivative(w):
        norm=np.linalg.norm(w@g)
        if norm<=np.finfo(float).eps:raise ArithmeticError("zero_dual_vector")
        return linear+r*(gram@w)/norm
    try:
        solved=(solver or minimize)(objective,an,jac=derivative,bounds=[(0.,1.)]*len(an),
                                    constraints={"type":"eq","fun":lambda w:w.sum()-1,"jac":lambda w:np.ones_like(w)},
                                    method="SLSQP",options={"ftol":min(tolerance**2,1e-12),"maxiter":max_iterations})
        if not solved.success:raise ArithmeticError("solver_failure: "+str(solved.message))
        w=np.asarray(solved.x);gw=w@g;norm=np.linalg.norm(gw)
        if not np.isfinite(w).all() or np.min(w)<-tolerance or abs(w.sum()-1)>tolerance:raise ArithmeticError("invalid_simplex")
        if norm<=np.finfo(float).eps:raise ArithmeticError("zero_dual_vector")
        dn=h+r*gw/norm
        gap=objective(w)-np.min(g@dn)
        residual=np.linalg.norm(dn-h)-r
        if not np.isfinite(dn).all() or residual>tolerance or gap>tolerance*(1+abs(objective(w))) or gap < -tolerance:
            raise ArithmeticError("failed_primal_dual_certificate")
        d=torch.as_tensor(dn*scale,device=G.device,dtype=G.dtype)
        if not torch.isfinite(d).all() or float((d.double()-base.double()).norm())>radius+tolerance*(1+bn):raise ArithmeticError("cast_trust_violation")
        coeff=torch.as_tensor(an+r*w/norm,device=G.device,dtype=G.dtype)
        return result(d,"COORDINATED",None,coeff,float(gap*scale*scale))
    except (ArithmeticError,ValueError,RuntimeError,FloatingPointError) as exc:
        return result(base,"FALLBACK",str(exc),a)
