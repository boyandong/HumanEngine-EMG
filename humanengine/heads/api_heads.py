import torch
from torch import nn


class APIHead(nn.Module):
    def __init__(self,cfg,output_dim,horizons=0):
        super().__init__(); self.horizons=horizons; self.output_dim=output_dim
        self.projector=nn.Sequential(nn.Linear(cfg.shared_dim+cfg.residual_dim,cfg.api_projector_dim),nn.SiLU())
        self.current=nn.Linear(cfg.api_projector_dim,output_dim)
        self.future=nn.Linear(cfg.api_projector_dim,horizons*output_dim) if horizons else None

    def forward(self,s,r,*,diagnostic_open_r=False):
        z=self.projector(torch.cat((s,r if diagnostic_open_r else r.detach()),-1))
        result={"current":self.current(z)}
        if self.future is not None:
            result["future"]=self.future(z).reshape(*z.shape[:-1],self.horizons,self.output_dim)
        return result


class TrainingHeads(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.P_R=nn.Sequential(nn.Linear(cfg.residual_dim,cfg.predictor_hidden),nn.SiLU(),
                               nn.Linear(cfg.predictor_hidden,3*2*cfg.teacher_hidden))
        self.M_A=nn.Sequential(nn.Linear(cfg.residual_dim,cfg.predictor_hidden),nn.SiLU(),nn.Linear(cfg.predictor_hidden,6*cfg.channels))
        self.V=nn.Sequential(nn.Linear(cfg.shared_dim,cfg.invariance_hidden),nn.SiLU(),nn.Linear(cfg.invariance_hidden,cfg.invariance_dim))
        self.G=nn.Sequential(nn.Linear(cfg.shared_dim,cfg.geometry_hidden),nn.SiLU(),nn.Linear(cfg.geometry_hidden,cfg.geometry_dim))


def canonical_view(pose20,repo_root=None):
    from representation.canonical_hand import resolve_source_indices
    indices=torch.tensor(resolve_source_indices(repo_root),device=pose20.device)
    if pose20.shape[-1]!=20: raise ValueError("Expected official 20D layout")
    return pose20.index_select(-1,indices)
