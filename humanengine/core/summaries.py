import torch
from torch import nn


class SignalStatistics(nn.Module):
    """Explicit, fixed train-only statistics; fixtures must say synthetic."""
    def __init__(self,mean,std,provenance,floor):
        super().__init__()
        if mean.shape!=std.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<0).any():
            raise ValueError("Invalid frozen statistics")
        if not provenance.get("partition") in ("train","synthetic") or not provenance.get("version"):
            raise ValueError("Statistics need train-only provenance")
        self.register_buffer("mean",mean.detach().clone())
        self.register_buffer("std",std.detach().clone())
        self.provenance=dict(provenance); self.floor=floor

    def forward(self,x):
        return (x-self.mean)/self.std.clamp_min(self.floor)


class CausalAmplitude(nn.Module):
    def __init__(self,cfg,statistics):
        super().__init__(); self.cfg=cfg; self.statistics=statistics
        self.width=cfg.samples(cfg.rms_seconds)
        if statistics.mean.shape!=(cfg.channels,): raise ValueError("E statistics shape")

    def step(self,x,tail,seen):
        joined=torch.cat((tail,x),dim=1)
        windows=joined.unfold(1,self.width,1) # B,T,C,W
        values=(windows.square().mean(-1).sqrt()+self.cfg.signal_epsilon).log()
        index=torch.arange(x.shape[1],device=x.device)
        keep=(index+seen+1).remainder(self.cfg.stride)==0
        return self.statistics(values[:,keep]),joined[:,-(self.width-1):] if self.width>1 else joined[:,:0]


@torch.no_grad()
def physical_summary(window,cfg):
    """B,L,C -> B,C*6; caller must ensure complete valid causal support."""
    nr,ns=cfg.samples(cfg.rms_seconds),cfg.samples(cfg.spectrum_seconds)
    if window.shape[1]<max(nr,ns): raise ValueError("Incomplete physical target support")
    rms=(window[:,-nr:].square().mean(1).sqrt()+cfg.signal_epsilon).log()
    x=window[:,-ns:]
    hann=torch.hann_window(ns,periodic=cfg.hann_periodic,device=x.device,dtype=x.dtype)
    fft=torch.fft.rfft(x*hann[None,:,None],dim=1)
    psd=fft.abs().square()/(cfg.sample_rate*hann.square().sum())
    factor=torch.full((psd.shape[1],),2.,device=x.device,dtype=x.dtype); factor[0]=1.
    if ns%2==0: factor[-1]=1.
    psd=psd*factor[None,:,None]
    freq=torch.fft.rfftfreq(ns,1/cfg.sample_rate).to(x.device)
    bands=[]
    for i,(lo,hi) in enumerate(cfg.bands):
        mask=(freq>=lo)&((freq<=hi) if i==len(cfg.bands)-1 else (freq<hi))
        bands.append((psd[:,mask].sum(1)*(cfg.sample_rate/ns)+cfg.signal_epsilon).log())
    return torch.stack((rms,*bands),dim=-1).flatten(1)
