import torch
from .macrostep import Macrostep


def make_macrostep(model,heads,registry,cfg,*,core_optimizer_factory,private_lr,evaluation_modules=()):
    """Derive parameter ownership from model structure, never a flat model list."""
    active=set(registry.budgets())-{"reserve"}
    if not active.issubset(model.api_heads):raise ValueError("Active API lacks a private readout")
    core=model.core.core_parameters()
    groups={k:tuple(model.api_heads[k].parameters()) for k in sorted(active)}
    groups.update({k:tuple(getattr(heads,k).parameters()) for k in ("P_R","M_A","V")})
    if "pose" in active:groups["G"]=tuple(heads.G.parameters())
    return Macrostep(core,groups,core_optimizer_factory(core),
                     {k:torch.optim.SGD(v,lr=private_lr) for k,v in groups.items()},cfg,
                     stateful_modules=(model,heads,*evaluation_modules))
