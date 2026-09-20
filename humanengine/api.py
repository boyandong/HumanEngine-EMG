import torch
from torch import nn
from dataclasses import replace
from .core.state import DualStateCore
from .core.state import detach_state
from .heads.api_heads import APIHead


class HumanEngine(nn.Module):
    def __init__(self,cfg,amplitude_statistics):
        super().__init__(); self.cfg=cfg
        self.optimization_mode="unconfigured"
        self.core=DualStateCore(cfg,amplitude_statistics)
        self.api_heads=nn.ModuleDict({"pose":APIHead(cfg,20,len(cfg.future_seconds))})

    def init_state(self,*args,**kwargs): return self.core.init_state(*args,**kwargs)

    def step(self,emg_chunk,state,timestamps,**kwargs):
        out,state=self.core.step(emg_chunk,state,timestamps,**kwargs)
        for name,head in self.api_heads.items():
            for component,value in head(out.shared,out.residual).items():
                out.apis[name+"/"+component]=value
        return out,state

    def forward_sequence(self,x,timestamps,**kwargs):
        return self.step(x,self.init_state(len(x),device=x.device,dtype=x.dtype),timestamps,**kwargs)

    def forward_training_sequence(self,x,timestamps,**kwargs):
        """Per-view burn-in, then explicit TBPTT detach. No state between windows."""
        n=self.cfg.samples(self.cfg.burnin_seconds)
        if x.shape[1]<=n:raise ValueError("Training window has no post-burn-in samples")
        with torch.no_grad():
            first,state=self.forward_sequence(x[:,:n],timestamps[:,:n],**kwargs)
        second,state=self.step(x[:,n:],detach_state(state),timestamps[:,n:],**kwargs)
        names=("timestamps","valid","shared","residual","H","frontend","amplitude","warmup")
        joined={k:torch.cat((getattr(first,k),getattr(second,k)),1) for k in names}
        joined["apis"]={k:torch.cat((first.apis[k],second.apis[k]),1) for k in first.apis}
        joined["input_quality"]={"burnin_samples":n,"reset_events":first.input_quality["reset_events"]+second.input_quality["reset_events"]}
        joined["input_quality"]["training_eligible"]=torch.cat((torch.zeros_like(first.valid),second.valid),1)
        return replace(second,**joined),state

    def universal_mode(self):
        self.optimization_mode="universal"
        self.requires_grad_(True);self.core.A.requires_grad_(False);self.core.U.requires_grad_(False)
        with torch.no_grad():
            self.core.A.log_gain.zero_();self.core.A.offset.zero_();self.core.U.up.weight.zero_()
        self.core.adapter_id="identity"

    def personalization_mode(self,*,offset_noise_scale_known=False):
        self.optimization_mode="personalized"
        self.requires_grad_(False);self.core.A.log_gain.requires_grad_(True);self.core.U.requires_grad_(True)
        self.core.A.offset.requires_grad_(offset_noise_scale_known)
