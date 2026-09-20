import math
import torch
from ..contracts import LossTerm


def separated_indices(meta,n,cfg):
    if meta is None:return [],"anchor metadata missing"
    meta.validate(n)
    groups={}
    for i,(user,rec,t) in enumerate(zip(meta.users,meta.recordings,meta.times)):
        if not math.isfinite(t):raise ValueError("Invalid anchor timestamp")
        groups.setdefault((user,rec),[]).append((t,i))
    chosen=[]
    for values in groups.values():
        previous=-math.inf
        for t,i in sorted(values):
            if t-previous>=cfg.anchor_separation_seconds:
                chosen.append(i);previous=t
    users={meta.users[i] for i in chosen};records={(meta.users[i],meta.recordings[i]) for i in chosen}
    if len(chosen)<cfg.minimum_anchors or len(users)<cfg.minimum_users or len(records)<cfg.minimum_recordings:
        return [],"insufficient independent-anchor support"
    return chosen,"OK"


def variance_floor(x,meta,cfg,threshold=None):
    x=x.reshape(-1,x.shape[-1]);idx,reason=separated_indices(meta,len(x),cfg)
    if not idx:return LossTerm.na(reason)
    z=x[idx].float()
    value=torch.relu((cfg.variance_floor if threshold is None else threshold)-
                     torch.sqrt(z.var(0,unbiased=True)+cfg.variance_epsilon)).mean()
    return LossTerm(value,len(idx),value)


def vicreg(a,b,meta,cfg):
    if a.shape!=b.shape:raise ValueError("Paired view mismatch")
    a=a.reshape(-1,a.shape[-1]);b=b.reshape_as(a)
    idx,reason=separated_indices(meta,len(a),cfg)
    if not idx:return LossTerm.na(reason)
    a=a[idx].float();b=b[idx].float()
    inv=(a-b).square().mean()
    var=sum(torch.relu(cfg.vicreg_variance_floor-torch.sqrt(z.var(0,unbiased=True)+cfg.variance_epsilon)).mean() for z in (a,b))/2
    cov=[]
    for z in (a,b):
        z=z-z.mean(0);c=z.T@z/(len(z)-1)
        off=c-torch.diag_embed(c.diag());cov.append(off.square().sum()/z.shape[1])
    value=inv+var+cfg.vicreg_covariance_weight*sum(cov)/2
    return LossTerm(value,len(idx),value)


def pose_geometry(projected,pose,meta,pose_statistics,cfg):
    z=projected.reshape(-1,projected.shape[-1]).float();q=pose.reshape(-1,20).detach().float()
    meta.validate(len(z))
    if len(set(meta.users))<cfg.minimum_users or len(set(zip(meta.users,meta.recordings)))<cfg.minimum_recordings:
        return LossTerm.na("insufficient geometry groups")
    pairs=[(i,j) for i in range(len(z)) for j in range(i+1,len(z))
           if (meta.users[i],meta.recordings[i])!=(meta.users[j],meta.recordings[j]) or
           abs(meta.times[i]-meta.times[j])>=cfg.geometry_pair_separation_seconds]
    if len(pairs)<cfg.geometry_min_pairs:return LossTerm.na("insufficient geometry pairs")
    i,j=torch.tensor(pairs,device=z.device).T
    target=pose_statistics(q).detach()
    td=(target[i]-target[j]).square().sum(-1)
    sd=(z[i]-z[j]).square().sum(-1)
    if td.sqrt().mean()<cfg.geometry_degenerate_threshold:return LossTerm.na("degenerate pose geometry")
    tr=(td+cfg.geometry_distance_epsilon).sqrt();sr=(sd+cfg.geometry_distance_epsilon).sqrt()
    tr=tr/(tr.mean()+cfg.geometry_mean_epsilon)
    sr=sr/(sr.mean()+cfg.geometry_mean_epsilon) # retain denominator gradient
    value=torch.nn.functional.huber_loss(sr,tr,reduction="mean",delta=cfg.huber_delta)
    status="DEGENERATE_STUDENT" if sd.sqrt().mean()<cfg.geometry_degenerate_threshold else "OK"
    return LossTerm(value,len(pairs),value,status)


class DegeneracyMonitor:
    def __init__(self,limit=3):self.limit=limit;self.consecutive=0
    def observe(self,term):
        self.consecutive=self.consecutive+1 if term.status=="DEGENERATE_STUDENT" else 0
        if self.consecutive>=self.limit:raise RuntimeError("Stop development run: repeated degenerate geometry")
