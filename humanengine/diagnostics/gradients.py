from contextlib import contextmanager
import torch
from ..execution import ReadOnlyEvaluation
from ..transaction import ObservationTransaction


def flat_gradient(loss,parameters,retain_graph=True):
    params=tuple(parameters)
    gradients=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain_graph)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gradients)])


def vector(parameters):return torch.cat([p.detach().reshape(-1) for p in parameters])


@contextmanager
def observational(module,optimizer=None,*,modules=(),parameters=(),objects=()):
    """Run an observational eager callback and restore its full continuation.

    Protected tensor or executable-metadata writes reject. Registered Python
    continuation state and RNG are restored even after a successful callback.
    """
    roots=[]
    for root in (module,*modules):
        if root is not None and all(root is not old for old in roots):roots.append(root)
    params=tuple(parameters) or tuple(p for root in roots for p in root.parameters())
    registered=tuple(x for x in (optimizer,*objects) if x is not None)
    with ObservationTransaction(modules=tuple(roots),parameters=params,objects=registered):
        with ReadOnlyEvaluation(params,tuple(roots)) as guard:
            yield
            guard.check_metadata()


def shadow_gradient(model,x,timestamps,loss_fn,optimizer=None):
    """Fresh private caches; shadow path is never supplied to an optimizer."""
    with observational(model,optimizer):
        out,_=model.core.forward_sequence(x,timestamps)
        open_readout=model.api_heads["pose"](out.shared,out.residual,diagnostic_open_r=True)
        routed_readout=model.api_heads["pose"](out.shared,out.residual)
        params=model.core.core_parameters()
        full=flat_gradient(loss_fn(open_readout),params).detach().clone()
        routed=flat_gradient(loss_fn(routed_readout),params).detach().clone()
        return {"diagnostic_only":True,"full":full,"routed":routed,"blocked":full-routed}


def gradient_report(gradients,alpha,block_sizes,direction=None,delta=None):
    report={};offset=0
    for name,size in block_sizes.items():
        gs=[g[offset:offset+size] for g in gradients];offset+=size
        norms=[float(g.norm()) for g in gs]
        cos=[[None if norms[i]==0 or norms[j]==0 else float(gs[i]@gs[j])/(norms[i]*norms[j]) for j in range(len(gs))] for i in range(len(gs))]
        report[name]={"raw_norms":norms,"base_weighted_norms":[a*n for a,n in zip(alpha,norms)],"cosines":cos}
    if offset!=gradients[0].numel():raise ValueError("Block sizes do not cover the full core")
    report["direction_gains"]=None if direction is None else [float(g@direction) for g in gradients]
    report["actual_delta_inner_products"]=None if delta is None else [float(g@delta) for g in gradients]
    return report
