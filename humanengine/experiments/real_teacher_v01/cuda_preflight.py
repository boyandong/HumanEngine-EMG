"""Bounded CUDA execution and checkpoint-topology preflight for the teacher.

Execution-compatibility evidence only. It runs zero training steps, reads no
dataset, fits no statistic and produces no scientific result. It answers exactly
one question: can the unchanged v4 checkpoint/continuation contract be satisfied
on this server's CUDA path, and if not, is CPU the correct correctness fallback?

It also checks the exact hazard recorded by the previous session: cuDNN packs
LSTM parameters into shared storage, which the v4 no-alias contract rejects.
"""
import argparse
import copy
import json
import platform
from datetime import datetime,timezone
from pathlib import Path

import torch

from humanengine.config import KernelConfig
from humanengine.contracts import EMGSequence
from humanengine.data.views import masked_view
from humanengine.execution import named_parameter_topology
from humanengine.manifest import teacher_checkpoint,resume_teacher,file_hash
from humanengine.scientific_state import equal_state
from humanengine.teacher.model import fit_frozen_statistics
from humanengine.teacher.preparation import TeacherPreparation


def synthetic_sequence(batch,length,channels,device):
    generator=torch.Generator().manual_seed(20260920)
    emg=torch.randn(batch,length,channels,generator=generator)*0.25
    timestamps=torch.arange(1,length+1,dtype=torch.float64).div(2000.).unsqueeze(0).repeat(batch,1)
    return EMGSequence(emg.to(device),timestamps.to(device),
                       torch.ones(batch,length,dtype=torch.bool,device=device),
                       torch.zeros(batch,length,dtype=torch.long,device=device))


def topology_report(module):
    try:
        named_parameter_topology(module)
        return {'independent':True,'error':None}
    except ValueError as error:
        return {'independent':False,'error':str(error)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=3)
    arguments=parser.parse_args()
    cfg=KernelConfig().validate()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    report={'schema':'teacher-cuda-preflight-v1','recorded_utc':datetime.now(timezone.utc).isoformat(),
            'python':platform.python_version(),'torch':torch.__version__,
            'cuda_available':torch.cuda.is_available(),
            'device_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'training_steps':0,'scientific_evidence':False,
            'boundary':'execution compatibility only; not infrastructure reacceptance and not teacher quality'}
    # Physical summarisation produces 6 statistics x 16 channels = 96 features,
    # which is the dimensionality the teacher's physical statistics transform.
    physical_dim=6*cfg.channels
    statistics=fit_frozen_statistics(torch.zeros(4,2,3,2,physical_dim),
                                     torch.ones(4,2,3,2,dtype=torch.bool),
                                     {'partition':'synthetic','version':'cuda-preflight-synthetic-v1'},cfg)
    base=TeacherPreparation(cfg,statistics).train()
    report['cpu_topology']=topology_report(base)
    assert report['cpu_topology']['independent'],'CPU parameter topology already violates the no-alias contract'
    if not torch.cuda.is_available():
        report['cuda_topology']={'independent':None,'error':'cuda unavailable'}
        report['verdict']='CPU_ONLY_CUDA_UNAVAILABLE'
        arguments.out.write_text(json.dumps(report,indent=2)+'\n')
        print('PREFLIGHT',report['verdict'],flush=True)
        return
    # 1. Default cuDNN-enabled CUDA path, checked BEFORE any forward can repack.
    cudnn_model=copy.deepcopy(base).cuda()
    torch.backends.cudnn.enabled=True
    report['cudnn_enabled_topology_after_move']=topology_report(cudnn_model)
    with torch.no_grad():cudnn_model.ema.targets(synthetic_sequence(2,8000,cfg.channels,'cuda'),[7999])
    report['cudnn_enabled_topology_after_forward']=topology_report(cudnn_model)
    del cudnn_model
    # 2. Supported policy: cuDNN disabled, native ATen RNN kernels.
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    model=copy.deepcopy(base).cuda()
    report['native_topology_after_move']=topology_report(model)
    assert report['native_topology_after_move']['independent'],'Native CUDA path also violates the no-alias contract'
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=3e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    signal=synthetic_sequence(2,8000,cfg.channels,'cuda')
    anchors=[7999]
    # Same mask construction the frozen smoke test uses; keep the reference alias
    # because losses() requires both views to share support and boundaries.
    view,mask,fractions=masked_view(signal,cfg,torch.Generator().manual_seed(20260920))
    report['synthetic_mask_fractions']=fractions.mean(0).tolist()
    losses=[]
    for step in range(arguments.steps):
        optimizer.zero_grad(set_to_none=True)
        if not model.target_statistics.count.any():
            with torch.no_grad():raw,valid=model.ema.targets(signal,anchors)
            model.target_statistics.update(raw,valid,partition='train')
        bundle,raw,valid=model.losses(signal,view,anchors,mask,None)
        assert torch.isfinite(bundle.total),'Non-finite synthetic objective'
        assert all(t.value is not None for t in bundle.terms.values()),'A loss term was unavailable'
        bundle.total.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'Non-finite gradient'
        optimizer.step();model.update_ema();model.target_statistics.update(raw,valid,partition='train')
        losses.append(float(bundle.total.detach()))
    report['synthetic_steps_executed']=arguments.steps
    report['synthetic_objective_finite']=all(l==l and abs(l)!=float('inf') for l in losses)
    config={'kernel':cfg.to_dict()}
    checkpoint=teacher_checkpoint(model,optimizer,config,'preflight-manifest','preflight-source',
                                 {'partition':'preflight_not_scientific'})
    path=arguments.out.with_name('CUDA_PREFLIGHT_CHECKPOINT.pt')
    torch.save(checkpoint,path)
    receiver=TeacherPreparation(cfg,statistics).train().cuda()
    other=torch.optim.AdamW([p for p in receiver.parameters() if p.requires_grad],
                            lr=3e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    resume_teacher(torch.load(path,weights_only=False),receiver,other,config=config,
                   data_manifest_hash='preflight-manifest',source_version='preflight-source',
                   expected_identity=checkpoint['scientific_identity'])
    report['resume_state_identical']=equal_state(model.state_dict(),receiver.state_dict())
    assert report['resume_state_identical'],'Resume did not reproduce the saved state'
    report['checkpoint_bytes']=path.stat().st_size
    report['checkpoint_sha256']=file_hash(path)
    report['steps_executed_on_gpu']=arguments.steps
    report['verdict']='CUDA_NATIVE_PATH_COMPATIBLE_WITH_CHECKPOINT_CONTRACT'
    arguments.out.write_text(json.dumps(report,indent=2)+'\n')
    print('PREFLIGHT',report['verdict'],flush=True)


if __name__=='__main__':main()
