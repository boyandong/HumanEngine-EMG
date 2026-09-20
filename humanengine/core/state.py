from dataclasses import replace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from ..contracts import RuntimeState,StreamState,HumanStateOutput
from .frontend import CausalFrontend,InputAffine,FeatureAdapter
from .summaries import CausalAmplitude


class DualStateCore(nn.Module):
    def __init__(self,cfg,amplitude_statistics):
        super().__init__(); self.cfg=cfg.validate()
        self.A=InputAffine(cfg.channels)
        self.F=CausalFrontend(cfg)
        self.U=FeatureAdapter(cfg.feature_dim,cfg.adapter_rank)
        self.E=CausalAmplitude(cfg,amplitude_statistics)
        self.S=nn.LSTM(cfg.feature_dim,cfg.shared_dim,batch_first=True)
        self.R=nn.LSTM(cfg.feature_dim+cfg.channels,cfg.residual_dim,batch_first=True)
        self.adapter_id="identity"

    @property
    def schema(self): return (self.cfg.channels,self.cfg.sample_rate,self.cfg.route_version)

    def _stream(self,device,dtype,user="unknown",recording="unknown",reason="initial"):
        z=lambda d: torch.zeros(1,1,d,device=device,dtype=dtype)
        return StreamState(self.F.init_state(1,device,dtype),z(self.cfg.shared_dim),z(self.cfg.shared_dim),
                           z(self.cfg.residual_dim),z(self.cfg.residual_dim),
                           torch.zeros(1,self.E.width-1,self.cfg.channels,device=device,dtype=dtype),
                           user_id=user,recording_id=recording,adapter_id=self.adapter_id,reset_reason=reason)

    def init_state(self,batch_size,adapter_id=None,device=None,dtype=None):
        p=next(self.parameters()); device=device or p.device; dtype=dtype or p.dtype
        if adapter_id is not None and adapter_id!=self.adapter_id:
            raise ValueError("Requested adapter has not been loaded")
        return RuntimeState(tuple(self._stream(device,dtype) for _ in range(batch_size)),self.schema,self.cfg.model_version)

    def core_parameters(self):
        return tuple(p for m in (self.F,self.S,self.R) for p in m.parameters())

    def _segment(self,x,t,st):
        f,conv=self.F.step(self.A(x),st.conv)
        e,tail=self.E.step(x,st.amplitude_tail,st.seen)
        adapted=self.U(f)
        if f.shape[1]:
            s,(sh,sc)=self.S(adapted,(st.s_h,st.s_c))
            r,(rh,rc)=self.R(torch.cat((adapted,e),-1),(st.r_h,st.r_c))
        else:
            s=x.new_empty(1,0,self.cfg.shared_dim);r=x.new_empty(1,0,self.cfg.residual_dim)
            sh,sc,rh,rc=st.s_h,st.s_c,st.r_h,st.r_c
        positions=torch.arange(x.shape[1],device=x.device)+st.seen+1
        keep=positions.remainder(self.cfg.stride)==0
        values={"shared":s[0],"residual":r[0],"frontend":f[0],"amplitude":e[0],
                "timestamps":t[keep],"warmup":positions[keep]<self.cfg.samples(self.cfg.warmup_seconds)}
        return values,replace(st,conv=conv,s_h=sh,s_c=sc,r_h=rh,r_c=rc,amplitude_tail=tail,
                              seen=st.seen+x.shape[1],last_timestamp=float(t[-1]))

    def step(self,emg_chunk,state,timestamps,*,user_ids=None,recording_ids=None,reset_mask=None):
        x=emg_chunk; cfg=self.cfg
        if x.ndim!=3 or x.shape[-1]!=cfg.channels or timestamps.shape!=x.shape[:2]:
            raise ValueError("Incompatible input schema; initialize a compatible stream")
        if state.schema!=self.schema or state.model_version!=cfg.model_version or len(state.streams)!=len(x):
            raise ValueError("Incompatible RuntimeState; explicit reinitialization required")
        if x.dtype!=next(self.parameters()).dtype or x.device!=next(self.parameters()).device:
            raise ValueError("Input dtype/device mismatch")
        if not timestamps.is_floating_point() or timestamps.dtype!=torch.float64:
            raise ValueError("Use float64 timestamps to preserve sample alignment")
        all_rows=[]; new_states=[]; events=[]
        dt=1/cfg.sample_rate; tolerance=cfg.timestamp_tolerance_samples/cfg.sample_rate
        for b,old in enumerate(state.streams):
            user=user_ids[b] if user_ids is not None else old.user_id
            recording=recording_ids[b] if recording_ids is not None else old.recording_id
            reason=None
            if reset_mask is not None and bool(reset_mask[b]): reason="explicit"
            elif user!=old.user_id: reason="user_change"
            elif recording!=old.recording_id: reason="recording_change"
            elif old.adapter_id!=self.adapter_id: reason="adapter_change"
            st=self._stream(x.device,x.dtype,user,recording,reason) if reason else old
            if reason: events.append((b,0,reason))
            parts=[]; begin=0; previous=st.last_timestamp
            # Validation splits only; each continuous segment is processed once.
            finite=(torch.isfinite(x[b]).all(-1)&torch.isfinite(timestamps[b])).tolist()
            ts=timestamps[b].tolist()
            for j in range(x.shape[1]):
                discontinuity=previous is not None and (ts[j]<=previous or abs(ts[j]-previous-dt)>tolerance)
                if not finite[j] or discontinuity:
                    if j>begin:
                        out,st=self._segment(x[b:b+1,begin:j],timestamps[b,begin:j],st);parts.append(out)
                    why="invalid_signal" if not finite[j] else "timestamp_discontinuity"
                    st=self._stream(x.device,x.dtype,user,recording,why);events.append((b,j,why))
                    begin=j+1 if not finite[j] else j
                previous=ts[j] if finite[j] else None
            if begin<x.shape[1]:
                out,st=self._segment(x[b:b+1,begin:],timestamps[b,begin:],st);parts.append(out)
            dims={"shared":cfg.shared_dim,"residual":cfg.residual_dim,"frontend":cfg.feature_dim,"amplitude":cfg.channels}
            row={k:torch.cat([p[k] for p in parts],0) if parts else x.new_empty(0,d) for k,d in dims.items()}
            for k,dtype in (("timestamps",torch.float64),("warmup",torch.bool)):
                row[k]=torch.cat([p[k] for p in parts],0) if parts else torch.empty(0,device=x.device,dtype=dtype)
            all_rows.append(row);new_states.append(st)
        packed={k:pad_sequence([r[k] for r in all_rows],batch_first=True) for k in all_rows[0]}
        width=packed["shared"].shape[1]
        valid=torch.arange(width,device=x.device)[None,:]<torch.tensor([len(r["shared"]) for r in all_rows],device=x.device)[:,None]
        out=HumanStateOutput(packed["timestamps"],valid,packed["shared"],packed["residual"],
                             torch.cat((packed["shared"],packed["residual"]),-1),packed["frontend"],packed["amplitude"],{},
                             packed["warmup"]|~valid,{"reset_events":events,"finite_input":bool(torch.isfinite(x).all())},
                             cfg.model_version,tuple(s.adapter_id for s in new_states))
        return out,RuntimeState(tuple(new_states),self.schema,cfg.model_version)

    def forward_sequence(self,x,timestamps,**kwargs):
        return self.step(x,self.init_state(x.shape[0],device=x.device,dtype=x.dtype),timestamps,**kwargs)


def detach_state(state):
    """Explicit TBPTT boundary; step never silently detaches a training graph."""
    rows=[]
    for st in state.streams:
        rows.append(replace(st,conv=tuple(replace(c,tail=c.tail.detach()) for c in st.conv),
                            s_h=st.s_h.detach(),s_c=st.s_c.detach(),r_h=st.r_h.detach(),r_c=st.r_c.detach(),
                            amplitude_tail=st.amplitude_tail.detach()))
    return replace(state,streams=tuple(rows))
