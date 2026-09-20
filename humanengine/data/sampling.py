from dataclasses import dataclass,asdict
import random

QUOTA_IDENTITY_VERSION="scientific-anchor-v2"


@dataclass(frozen=True)
class AnchorIdentity:
    source_id: str
    user_id: str
    session_id: str
    recording_id: str
    window_id: str
    index: int
    component: str

    @classmethod
    def from_window(cls,window,index,component):
        return cls(window.source_id,window.user_id,window.session_id,window.recording_id,window.window_id,index,component)


@dataclass(frozen=True)
class WindowRef:
    source_id: str
    user_id: str
    session_id: str
    recording_id: str
    window_id: str
    family_id: str
    source_group: str
    partition: str = "train"


class FamilyQuotaSampler:
    """Aliases deduplicate by acquisition + window, not dataset display name."""
    def __init__(self,windows,quotas,seed=17):
        self.quotas=dict(quotas);self.rng=random.Random(seed);self.macrosteps=0
        unique={}
        for w in windows:
            if w.partition!="train":raise ValueError("Only training windows may be sampled")
            key=(w.family_id,w.source_id,w.window_id)
            if key in unique and unique[key]!=w:raise ValueError("Conflicting acquisition identity")
            unique[key]=w
        self.windows=tuple(unique[k] for k in sorted(unique));self.exposure={k:0 for k in quotas}
        if any(not any(w.family_id==k for w in self.windows) or n<1 for k,n in quotas.items()):
            raise ValueError("Family quota has no eligible source")

    def _one(self,family):
        subset=[w for w in self.windows if w.family_id==family]
        for attr in ("source_id","user_id","session_id","recording_id"):
            key=self.rng.choice(sorted(set(getattr(w,attr) for w in subset)))
            subset=[w for w in subset if getattr(w,attr)==key]
        return self.rng.choice(subset)

    def sample(self):
        result={k:tuple(self._one(k) for _ in range(n)) for k,n in sorted(self.quotas.items())}
        for k,v in result.items():self.exposure[k]+=len(v)
        self.macrosteps+=1
        return result

    def plan_valid_quotas(self,component_quotas,coverage,*,max_attempts):
        """Coverage maps components to explicit valid flattened window indices.

        Consumer must select precisely the granted identities/indices.
        Incomplete plans are NA and must not trigger a partial optimizer update.
        """
        if set(component_quotas)!=set(self.quotas):raise ValueError("Quota family mismatch")
        plans={};missing={}
        for family,quotas in sorted(component_quotas.items()):
            if not quotas or any(n<1 for n in quotas.values()):raise ValueError("Invalid component quota")
            remaining=dict(quotas);rows=[];consumed=set()
            for _ in range(max_attempts):
                window=self._one(family);available=coverage(window)
                if any(not isinstance(v,(list,tuple,range)) or any(type(i) is not int or i<0 for i in v) for v in available.values()):
                    raise ValueError("Coverage must supply explicit nonnegative anchor indices, not counts")
                granted={}
                for k,n in remaining.items():
                    candidates=[AnchorIdentity.from_window(window,i,k) for i in sorted(set(available.get(k,())))]
                    granted[k]=tuple(a for a in candidates if a not in consumed)[:n]
                    consumed.update(granted[k])
                if any(granted.values()):
                    rows.append((window,granted))
                    remaining={k:n-len(granted[k]) for k,n in remaining.items()}
                if not any(remaining.values()):break
            plans[family]=tuple(rows)
            if any(remaining.values()):missing[family]=remaining
        if not missing:
            self.macrosteps+=1
            for k,rows in plans.items():self.exposure[k]+=len(rows)
        return {"status":"NA" if missing else "COMPLETE","allocations":plans,"missing":missing,
                "quotas":component_quotas,"identity_version":QUOTA_IDENTITY_VERSION}

    def state_dict(self):
        return {"rng":self.rng.getstate(),"macrosteps":self.macrosteps,"exposure":dict(self.exposure),
                "quotas":dict(self.quotas),"windows":[asdict(w) for w in self.windows]}

    def load_state_dict(self,state):
        if state["quotas"]!=self.quotas or state["windows"]!=[asdict(w) for w in self.windows]:raise ValueError("Sampler manifest mismatch")
        self.rng.setstate(state["rng"]);self.macrosteps=state["macrosteps"];self.exposure=dict(state["exposure"])


def quota_mask(valid,grants):
    import torch
    indices=[g.index for g in grants]
    if len(set(indices))!=len(indices) or any(i<0 or i>=valid.numel() or not bool(valid.flatten()[i]) for i in indices):
        raise ValueError("Grant exceeds unique valid coverage")
    result=torch.zeros_like(valid,dtype=torch.bool)
    result.view(-1)[indices]=True
    return result
