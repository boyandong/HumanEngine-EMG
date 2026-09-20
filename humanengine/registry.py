from dataclasses import dataclass,asdict
import math
from .contracts import LossTerm


@dataclass(frozen=True)
class Component:
    component_id: str
    beta: float
    reference_scale: float
    reference_floor: float
    mask_definition: str
    reference_version: str

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.beta,self.reference_scale,self.reference_floor)) or self.beta<=0 or self.reference_floor<=0 or self.reference_scale<0:
            raise ValueError("Invalid frozen reference scale/beta")

    @property
    def scale(self): return max(self.reference_scale,self.reference_floor)


@dataclass(frozen=True)
class APIFamily:
    family_id: str
    semantic_version: str
    components: tuple[Component,...]
    source_ids: tuple[str,...]
    projector_id: str
    head_ids: tuple[str,...]
    replay_policy: str
    route_version: str
    active: bool = True


class APIRegistry:
    def __init__(self,families=(),reserve_alpha=.2):
        self.families={};self.reserve_alpha=reserve_alpha
        if not 0<reserve_alpha<1: raise ValueError("Invalid reserve alpha")
        for family in families:self.register(family)

    def register(self,family):
        if family.family_id in self.families:raise ValueError("Family already registered")
        if family.family_id=="reserve" or not family.components or len(set(c.component_id for c in family.components))!=len(family.components):
            raise ValueError("Invalid family components")
        self.families[family.family_id]=family

    def budgets(self):
        active=sorted(k for k,v in self.families.items() if v.active)
        if not active:raise ValueError("No explicit API family")
        return {**{k:(1-self.reserve_alpha)/len(active) for k in active},"reserve":self.reserve_alpha}

    def to_dict(self):
        return {"families":[asdict(self.families[k]) for k in sorted(self.families)],"base_budgets":self.budgets()}


def family_loss(family,components):
    missing=[c.component_id for c in family.components if c.component_id not in components or components[c.component_id].value is None]
    if missing:return LossTerm.na("incomplete family coverage: "+",".join(missing))
    value=sum(c.beta*components[c.component_id].value/c.scale for c in family.components)/sum(c.beta for c in family.components)
    return LossTerm(value,1,value)


def reserve_loss(residual,vicreg,cfg,reference_scale):
    if residual is None or vicreg is None:return LossTerm.na("incomplete reserve coverage")
    if not math.isfinite(reference_scale) or reference_scale<=0:raise ValueError("Frozen reserve reference required")
    v=(residual/reference_scale+cfg.reserve_vicreg_weight*vicreg)/(1+cfg.reserve_vicreg_weight)
    return LossTerm(v,1,v)
