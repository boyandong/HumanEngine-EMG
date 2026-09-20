from dataclasses import dataclass
import torch


def representation_health(x,users,recordings,*,std_threshold):
    x=x.detach().double().reshape(-1,x.shape[-1])
    if len(x)<2 or len(users)!=len(x) or len(recordings)!=len(x):raise ValueError("Insufficient/mismatched health metadata")
    if not torch.isfinite(x).all():raise ValueError("Nonfinite representation")
    std=x.std(0);z=x-x.mean(0);sv=torch.linalg.svdvals(z)/(len(x)-1)**.5
    spectrum=sv.square();p=sv/sv.sum() if sv.sum()>0 else sv
    rank=float(torch.exp(-(p[p>0]*p[p>0].log()).sum())) if spectrum.sum()>0 else 0.
    def within(keys):
        values=[]
        for key in sorted(set(keys)):
            indices=[i for i,k in enumerate(keys) if k==key]
            if len(indices)>1:values.append(x[indices].var(0))
        return torch.stack(values).mean(0) if values else None
    user_means=torch.stack([x[[i for i,u in enumerate(users) if u==key]].mean(0) for key in sorted(set(users))])
    return {"std":std,"dead_dimensions":int((std<std_threshold).sum()),"covariance_spectrum":spectrum,
            "effective_rank":rank,"effective_rank_definition":"centered-singular-value-entropy-v2","within_recording_variance":within(list(zip(users,recordings))),
            "within_user_variance":within(list(users)),"cross_user_mean_variance":user_means.var(0) if len(user_means)>1 else None}


@dataclass(frozen=True)
class RepresentationExport:
    kind: str
    values: torch.Tensor
    timestamps: torch.Tensor
    user: tuple[str,...]
    session: tuple[str,...]
    recording: tuple[str,...]
    side: tuple[str,...]
    representation_version: str
    checkpoint_hash: str
    teacher_hash: str | None = None

    def __post_init__(self):
        if self.kind not in ("raw","F","E","teacher","S","R","H"):raise ValueError("Unknown representation")
        n=self.values.shape[0]
        if any(len(x)!=n for x in (self.timestamps,self.user,self.session,self.recording,self.side)):raise ValueError("Export metadata mismatch")
        if not self.checkpoint_hash or not self.representation_version:raise ValueError("Export provenance required")
        if self.kind=="teacher" and not self.teacher_hash:raise ValueError("Teacher hash required")


def information_loss_diagnostic(scores,minimum_gap):
    """Interpret precomputed FAIR probe scores; never runs scientific probes."""
    messages=[]
    for a,b,label in (("raw","teacher","teacher target coverage"),("teacher","R","teacher-to-R retention"),
                       ("R","API","API readout/routing"),("raw","F","frontend bottleneck")):
        if a in scores and b in scores and scores[a]-scores[b]>minimum_gap:messages.append(label)
    if all(k in scores for k in ("S","H")) and abs(scores["S"]-scores["H"])<=minimum_gap:messages.append("residual incremental benefit not established")
    if all(k in scores for k in ("E","H")) and abs(scores["E"]-scores["H"])<=minimum_gap:messages.append("handcrafted amplitude may explain benefit")
    return messages
