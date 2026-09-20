import os
import torch
from ..contracts import EMGSequence


def open_hdf5_ascii(path,*,opener=None):
    """Pass caller's lexical path directly to h5py; NEVER resolve a junction."""
    if opener is None:
        import h5py
        opener=h5py.File
    return opener(os.fspath(path),"r")


def read_sequence(path,start,stop):
    if start<0 or stop<=start: raise ValueError("Invalid bounded slice")
    with open_hdf5_ascii(path) as handle:
        group=handle["emg2pose"]
        values=group["timeseries"][start:stop]
        return {"emg":values["emg"].copy(),"pose":values["joint_angles"].copy(),
                "timestamps":values["time"].copy(),"metadata":dict(group.attrs)}


def official_pose_valid(pose):
    if pose.shape[-1]!=20: raise ValueError("Official pose must have 20 angles")
    # matches np.isclose(angle,0) default atol=1e-8; finite is separate guard.
    return ~torch.isclose(pose,torch.zeros_like(pose),atol=1e-8,rtol=1e-5).all(-1)


def valid_support(seq: EMGSequence,end,width,cfg):
    seq.validate(); start=end-width+1
    if start<0 or end>=seq.emg.shape[1]:
        return torch.zeros(seq.emg.shape[0],device=seq.emg.device,dtype=torch.bool)
    x=seq.emg[:,start:end+1]; t=seq.timestamps[:,start:end+1]; seg=seq.segments[:,start:end+1]
    good=seq.valid[:,start:end+1].all(1)&torch.isfinite(x).all((1,2))&torch.isfinite(t).all(1)
    good=good&(seg==seg[:,:1]).all(1)
    if width>1:
        delta=t[:,1:]-t[:,:-1]
        good=good&(delta>0).all(1)&((delta-1/cfg.sample_rate).abs()<=cfg.timestamp_tolerance_samples/cfg.sample_rate).all(1)
    return good


def future_pose_targets(seq,pose,pose_mask,anchor_indices,horizons,cfg):
    """Exact sampled labels only; no boundary clamping or interpolation."""
    if pose.shape[:2]!=seq.emg.shape[:2] or pose_mask.shape!=pose.shape[:2]:
        raise ValueError("Pose shape mismatch")
    b=len(pose); n=len(anchor_indices)
    result=pose.new_zeros(b,n,len(horizons),20)
    valid=torch.zeros(b,n,len(horizons),dtype=torch.bool,device=pose.device)
    base=pose_mask & official_pose_valid(pose) & torch.isfinite(pose).all(-1)
    for j,a in enumerate(anchor_indices):
        for k,h in enumerate(horizons):
            end=int(a)+cfg.samples(h)
            if end>=pose.shape[1]: continue
            good=valid_support(seq,end,end-int(a)+1,cfg)&base[:,int(a):end+1].all(1)
            result[:,j,k]=torch.where(good[:,None],pose[:,end],torch.zeros_like(pose[:,end]))
            valid[:,j,k]=good
    return result.detach(),valid


def validate_split_groups(rows):
    ownership={}
    for row in rows:
        for field in ("source_group","recording_id"):
            key=(field,row[field])
            if key in ownership and ownership[key]!=row["partition"]:
                raise ValueError("Synchronized/recording group crosses split: "+str(key))
            ownership[key]=row["partition"]
