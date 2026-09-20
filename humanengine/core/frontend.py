import torch
from torch import nn
from ..contracts import ConvState


class CausalConv(nn.Module):
    """Left-only history, output anchors s-1,2s-1,...; cache never grows."""
    def __init__(self, cin, cout, kernel, stride=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(cin,cout,kernel,dilation=dilation,groups=groups)
        self.stride = stride
        self.history = dilation*(kernel-1)
        self.cin,self.cout = cin,cout

    def init_state(self, batch, device, dtype):
        return ConvState(torch.zeros(batch,self.history,self.cin,device=device,dtype=dtype),0)

    def step(self, x, state):
        if x.shape[1] == 0:
            return x.new_empty(x.shape[0],0,self.cout),state
        joined = torch.cat((state.tail,x),dim=1)
        all_y = self.conv(joined.transpose(1,2)).transpose(1,2)
        indices = torch.arange(x.shape[1],device=x.device)
        keep = (indices+state.seen+1).remainder(self.stride)==0
        tail = joined[:,-self.history:] if self.history else joined[:,:0]
        return all_y[:,keep],ConvState(tail,state.seen+x.shape[1])


class CausalFrontend(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        layers=[]
        cin=cfg.channels
        for k,s in zip(cfg.kernels,cfg.strides):
            layers.append(CausalConv(cin,cfg.feature_dim,k,s))
            cin=cfg.feature_dim
        for d in cfg.dilations:
            layers.append(CausalConv(cin,cin,cfg.residual_kernel,dilation=d,groups=cin))
        self.convs=nn.ModuleList(layers)
        self.norms=nn.ModuleList(nn.LayerNorm(cin) for _ in layers)
        self.points=nn.ModuleList(nn.Linear(cin,cin) for _ in cfg.dilations)

    def init_state(self,batch,device,dtype):
        return tuple(c.init_state(batch,device,dtype) for c in self.convs)

    def step(self,x,state):
        states=[]
        for i,c in enumerate(self.convs):
            y,st=c.step(x,state[i]); states.append(st)
            if i<len(self.cfg.strides):
                x=self.norms[i](torch.nn.functional.silu(y))
            else:
                x=self.norms[i](x+torch.nn.functional.silu(self.points[i-len(self.cfg.strides)](y)))
        return x,tuple(states)

    def forward_sequence(self,x):
        return self.step(x,self.init_state(x.shape[0],x.device,x.dtype))[0]


class InputAffine(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.log_gain=nn.Parameter(torch.zeros(channels),requires_grad=False)
        self.offset=nn.Parameter(torch.zeros(channels),requires_grad=False)

    def forward(self,x):
        return x*self.log_gain.exp()+self.offset


class FeatureAdapter(nn.Module):
    def __init__(self,dim,rank):
        super().__init__()
        self.down=nn.Linear(dim,rank,bias=False)
        self.up=nn.Linear(rank,dim,bias=False)
        nn.init.zeros_(self.up.weight)
        self.requires_grad_(False)

    def forward(self,x):
        return x+self.up(torch.tanh(self.down(x)))
