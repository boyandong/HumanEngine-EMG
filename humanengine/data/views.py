from dataclasses import replace
import torch


def masked_view(seq,cfg,generator,*,anchor=None):
    """One endpoint anchor per training window; disjoint annuli share ONE mask."""
    seq.validate()
    if anchor is not None:
        if not 0<=anchor<seq.emg.shape[1]:raise ValueError("Invalid mask anchor")
        prefix=replace(seq,emg=seq.emg[:,:anchor+1],timestamps=seq.timestamps[:,:anchor+1],valid=seq.valid[:,:anchor+1],segments=seq.segments[:,:anchor+1])
        view,mask,fractions=masked_view(prefix,cfg,generator)
        fullmask=torch.zeros_like(seq.valid);fullmask[:,:anchor+1]=mask
        return replace(seq,emg=torch.cat((view.emg,seq.emg[:,anchor+1:].clone()),1)),fullmask,fractions
    b,t,_=seq.emg.shape
    widths=[cfg.samples(v) for v in cfg.teacher_scales]
    if t<max(widths): raise ValueError("Mask construction requires full longest support")
    block=max(1,round(widths[0]*cfg.mask_block_short_fraction))
    mask=torch.zeros(b,t,dtype=torch.bool,device=seq.emg.device)
    for row in range(b):
        inner=0
        for width in widths:
            blocks=[(a,min(a+block,t-inner)) for a in range(t-width,t-inner,block)]
            target=max(1,round(len(blocks)*cfg.mask_fraction))
            if inner==0:
                selected=[len(blocks)-1]
                candidates=list(range(len(blocks)-1))
            else:
                selected=[];candidates=list(range(len(blocks)))
            order=torch.randperm(len(candidates),generator=generator).tolist()
            selected+= [candidates[i] for i in order[:target-len(selected)]]
            for i in selected:
                a,z=blocks[i];mask[row,a:z]=True
            inner=width
    fractions=torch.stack([mask[:,-w:].float().mean(1) for w in widths],1)
    if not ((fractions>=cfg.mask_min)&(fractions<=cfg.mask_max)).all():
        raise ValueError("Configured discrete mask blocks cannot meet all scale supports")
    x=torch.where(mask[:,:,None],torch.zeros_like(seq.emg),seq.emg).clone()
    return replace(seq,emg=x),mask,fractions


def weak_noise_view(seq,cfg,train_noise_scale,generator):
    if train_noise_scale is None: raise ValueError("Noise view NA: no independently justified measurement-noise scale")
    if (train_noise_scale<0).any() or not torch.isfinite(train_noise_scale).all(): raise ValueError("Invalid noise scale")
    noise=torch.randn(seq.emg.shape,generator=generator,device=seq.emg.device,dtype=seq.emg.dtype)
    return replace(seq,emg=(seq.emg+noise*cfg.noise_multiplier*train_noise_scale).clone())


def validate_masked_view(clean,masked,time_mask,anchors,cfg):
    from .sequence import valid_support
    if time_mask.shape!=clean.valid.shape or time_mask.dtype!=torch.bool:
        raise ValueError("Explicit temporal mask required")
    if not torch.equal(masked.emg,torch.where(time_mask[:,:,None],0.,clean.emg)):
        raise ValueError("Mask must explain student observations")
    block=max(1,round(cfg.samples(cfg.teacher_scales[0])*cfg.mask_block_short_fraction))
    for a in anchors:
        a=int(a)
        if a>=block-1 and not time_mask[:,a-block+1:a+1].all():
            raise ValueError("Mask must include the endpoint block")
        for tau in cfg.teacher_scales:
            width=cfg.samples(tau);start=a-width+1
            if start<0:continue
            eligible=valid_support(clean,a,width,cfg)
            ratio=time_mask[:,start:a+1].float().mean(1)[eligible]
            if ((ratio<cfg.mask_min)|(ratio>cfg.mask_max)).any():raise ValueError("Invalid scale mask coverage")
