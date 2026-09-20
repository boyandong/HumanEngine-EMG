"""Bounded label-free real teacher screen. Frozen kernel is imported unchanged."""
import argparse
import copy
from dataclasses import replace,asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import h5py
import numpy as np
import torch
from humanengine.config import KernelConfig
from humanengine.contracts import EMGSequence,AnchorMetadata
from humanengine.core.summaries import SignalStatistics,physical_summary
from humanengine.data.views import masked_view
from humanengine.teacher.preparation import TeacherPreparation,TeacherHealth,teacher_artifact_identity,freeze_teacher
from humanengine.teacher.model import fit_frozen_statistics
from humanengine.manifest import teacher_checkpoint,resume_teacher,file_hash
from humanengine.scientific_state import content_hash,equal_state,statistics_content,canonical

BASE=Path('/root/autodl-tmp/he_teacher_v01')
RECORD=BASE/'records'
CFG=KernelConfig().validate()
SEED=20260920

def save_json(name,value):
    path=RECORD/name
    with path.open('x') as out:json.dump(value,out,indent=2,allow_nan=False)
    return file_hash(path)

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)

def source_identity():
    root=Path('humanengine')
    paths=[p for p in root.rglob('*.py') if not {'tests','tools','__pycache__'} & set(p.parts)]
    return content_hash({p.as_posix():file_hash(p) for p in sorted(paths)})

class Corpus:
    def __init__(self):
        self.manifest=json.loads((RECORD/'DATA_MANIFEST.json').read_text())
        assert self.manifest['status']=='frozen'
        self.identity=file_hash(RECORD/'DATA_MANIFEST.json')
        self.rows=self.manifest['records'];self.data={}
        plan_path=Path(__file__).with_name('ACQUISITION_PLAN.json')
        plan=json.loads(plan_path.read_text())
        assert file_hash(plan_path)==self.manifest['acquisition_plan_sha256']
        assert [r['filename'] for r in self.rows]==[r['filename'] for r in plan['files']]
        assert len(self.rows)==len({r['filename'] for r in self.rows})==48
        for observed,expected in zip(self.rows,plan['files']):
            assert all(observed[k]==expected[k] for k in ('user','session','source_group','side','partition','official_split','archive_member'))
        train_users={r['user'] for r in self.rows if r['partition']=='train'}
        dev_users={r['user'] for r in self.rows if r['partition']=='emg_development'}
        assert len(train_users)==8 and len(dev_users)==4 and not train_users & dev_users
        for row in self.rows:
            assert row['partition'] in ('train','emg_development') and row['official_split'] in ('train','val')
            assert row['user']!='d387095792'
            path=Path(row['path']);assert path.parent==BASE/'raw' and path.name==row['filename']+'.hdf5'
            assert path.suffix=='.hdf5' and file_hash(path)==row['sha256']
            with h5py.File(path,'r') as f:
                # No generic compound read or pose/gesture-dependent mask.
                a=f['emg2pose/timeseries'].fields(['emg','time'])[:]
            self.data[row['filename']]=(a['emg'].copy(),a['time'].copy())

    def sequence(self,items,device):
        x=np.stack([self.data[r['filename']][0][e-3999:e+1] for r,e in items])
        t=np.stack([self.data[r['filename']][1][e-3999:e+1] for r,e in items])
        signal=EMGSequence(torch.from_numpy(x).to(device),torch.from_numpy(t).to(device),
                           torch.ones(x.shape[:2],dtype=torch.bool,device=device),torch.zeros(x.shape[:2],dtype=torch.long,device=device))
        meta=AnchorMetadata(tuple(r['user'] for r,e in items),tuple(r['source_group'] for r,e in items),tuple(float(self.data[r['filename']][1][e]) for r,e in items))
        return signal,meta

    def batch(self,step):
        rng=np.random.default_rng(SEED+step)
        rows=[r for r in self.rows if r['partition']=='train']
        users=sorted({r['user'] for r in rows})
        chosen=rng.choice(users,size=4,replace=False);items=[]
        for user in chosen:
            groups=sorted({r['source_group'] for r in rows if r['user']==user})
            for group in groups:
                sides=[r for r in rows if r['source_group']==group]
                row=sides[int(rng.integers(len(sides)))]
                ends=rng.choice(row['eligible_endpoints'][::2],size=8,replace=False)
                items.extend((row,int(e)) for e in ends)
        assert len(items)==64
        return items

    def evaluation_items(self,partition):
        result=[]
        for r in self.rows:
            if r['partition']==partition:
                ends=r['eligible_endpoints']
                indices=np.linspace(0,len(ends)-1,16,dtype=int)
                assert len(set(indices))==16
                result.extend((r,int(ends[i])) for i in indices)
        return result

def optimizer(prep):
    return torch.optim.AdamW([p for p in prep.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)

def stats_load():
    record=torch.load(RECORD/'physical_statistics.pt',weights_only=False,map_location='cpu')
    return SignalStatistics(record['mean'],record['std'],record['provenance'],CFG.signal_scale_floor)

def step_once(prep,opt,corpus,step,device):
    seq,meta=corpus.sequence(corpus.batch(step),device)
    view,mask,fractions=masked_view(seq,CFG,torch.Generator().manual_seed(SEED+100000+step))
    opt.zero_grad(set_to_none=True)
    # Initialize train-derived target moments BEFORE the first scored step.
    if not prep.target_statistics.count.any():
        with torch.no_grad():raw,valid=prep.ema.targets(seq,[3999])
        prep.target_statistics.update(raw,valid,partition='train')
    bundle,raw,valid=prep.losses(seq,view,[3999],mask,meta)
    assert valid.all() and all(t.value is not None for t in bundle.terms.values())
    assert torch.isfinite(bundle.total)
    bundle.total.backward()
    grads=[p.grad for p in prep.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert all(p.grad is None for p in prep.ema.parameters())
    norm=torch.stack([g.detach().square().sum() for g in grads]).sum().sqrt().item()
    opt.step();prep.update_ema();prep.target_statistics.update(raw,valid,partition='train')
    assert all(torch.isfinite(p).all() for p in prep.parameters())
    return {**{k:float(t.value.detach()) for k,t in bundle.terms.items()},'total':float(bundle.total.detach()),
            'gradient_norm':norm,'mask_fractions':fractions.mean(0).tolist(),'valid_target_groups':valid.sum((0,1)).tolist(),
            'variance_count':bundle.terms['variance'].count,
            'ema_raw_std_median_six_groups':raw[:,0].std(0).median(-1).values.cpu().tolist()}

def prepare():
    corpus=Corpus();values=[];scope=[]
    for row in corpus.rows:
        if row['partition']!='train':continue
        x,_=corpus.data[row['filename']]
        for start in range(0,len(row['eligible_endpoints']),128):
            ends=row['eligible_endpoints'][start:start+128]
            windows=torch.from_numpy(np.stack([x[e-199:e+1] for e in ends]))
            values.append(physical_summary(windows,CFG).double())
        scope.append({'file':row['filename'],'sha256':row['sha256'],'endpoints':row['eligible_endpoints']})
    v=torch.cat(values);mean=v.mean(0).float();std=v.std(0,unbiased=True).float()
    provenance={'partition':'train','version':CFG.summary_version,'data_manifest_hash':corpus.identity,
                'count':len(v),'dimensions':96,'scope':scope,'floor':CFG.signal_scale_floor,'estimator':'sample standard deviation ddof=1'}
    torch.save({'mean':mean,'std':std,'provenance':provenance},RECORD/'physical_statistics.pt')
    save_json('STATISTICS.json',{'provenance':provenance,'mean':mean.tolist(),'std':std.tolist(),'floor_hit_fraction':float((std<CFG.signal_scale_floor).float().mean()),'artifact_sha256':file_hash(RECORD/'physical_statistics.pt')})
    print(json.dumps({'train_summary_samples':len(v),'manifest_hash':corpus.identity}),flush=True)

def smoke():
    corpus=Corpus();seed(SEED+900000);stats=stats_load();prep=TeacherPreparation(CFG,stats).train()
    seq,meta=corpus.sequence(corpus.batch(900000),'cpu')
    # Bounded numerical backend check on real inputs, not teacher-quality selection.
    with torch.no_grad():cpu_raw,cv=prep.ema.targets(seq,[3999])
    gpu=copy.deepcopy(prep).to('cuda')
    gs=replace(seq,**{k:getattr(seq,k).cuda() for k in ('emg','timestamps','valid','segments')})
    with torch.no_grad():gpu_raw,gv=gpu.ema.targets(gs,[3999])
    delta=(cpu_raw-gpu_raw.cpu()).abs().max().item()
    assert torch.equal(cv,gv.cpu())
    torch.testing.assert_close(cpu_raw,gpu_raw.cpu(),atol=2e-4,rtol=2e-3)
    opt=optimizer(gpu);started=time.perf_counter();logs=[]
    for i in range(2):logs.append(step_once(gpu,opt,corpus,900000+i,'cuda'))
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    config={'kernel':CFG.to_dict()};source=source_identity()
    checkpoint=teacher_checkpoint(gpu,opt,config,corpus.identity,source,{'partition':'smoke_not_scientific'})
    torch.save(checkpoint,RECORD/'smoke_checkpoint.pt')
    receiver=TeacherPreparation(CFG,stats_load()).train().cuda();other=optimizer(receiver)
    resume_teacher(torch.load(RECORD/'smoke_checkpoint.pt',weights_only=False),receiver,other,config=config,data_manifest_hash=corpus.identity,source_version=source,expected_identity=checkpoint['scientific_identity'])
    assert equal_state(gpu.state_dict(),receiver.state_dict())
    step_once(gpu,opt,corpus,900002,'cuda');step_once(receiver,other,corpus,900002,'cuda')
    max_resume=max((v-receiver.state_dict()[k]).abs().max().item() for k,v in gpu.state_dict().items())
    assert max_resume==0
    save_json('SMOKE.json',{'scientific_evidence':False,'device':'cuda','python':platform.python_version(),'torch':torch.__version__,
                          'cpu_gpu_max_abs':delta,'cpu_gpu_atol':2e-4,'cpu_gpu_rtol':2e-3,'resume_next_update_max_abs':max_resume,
                          'seconds_per_step':seconds/2,'batch_anchors':64,'logs':logs,'source_identity':source,
                          'backend_boundary':'real teacher runner on torch2.8 CUDA; bounded real-input checks only, not general infrastructure reacceptance'})
    print('SMOKE_PASSED',seconds/2,flush=True)

def preregister():
    smoke_record=json.loads((RECORD/'SMOKE.json').read_text())
    assert smoke_record['resume_next_update_max_abs']==0
    protocol={'version':'real-emg-health-v01','selection':'only step 1000; no earlier checkpoint selection or health-driven extension',
        'evaluation':'512 TRAIN and 256 dev anchors; 16 fixed evenly spaced eligible endpoints per file; actual same masks for predictor/context',
        'constant_reference':'TRAIN mean of EMA latent in frozen checkpoint running-stat coordinates; summary TRAIN-eval mean',
        'visible_reference':'EMA applied to the actual masked input with identical reset/context, no predictor and no hidden information',
        'physical_probe':'one fixed ridge per group: raw EMA standardized using TRAIN eval hidden moments floor .05; intercept unpenalized; alpha=1; 96 TRAIN-scaled physical summaries; no tuning; Huber delta1',
        'amplitude_probe':'raw EMA on paired .5x and 2x EMG; mean of paired RMS hidden deltas from original, median across dev windows',
        'critical_per_group_dev':{'masked_to_constant_ratio_max_exclusive':.99,'masked_to_context_ratio_max_exclusive':.99,
            'physical_to_constant_ratio_max_exclusive':.99,'raw_hidden_std_median_min_exclusive':.01,
            'within_recording_std_median_min_exclusive':.005,'amplitude_raw_delta_min_exclusive':.001,
            'dead_dimension_fraction_std_below_1e-3_max_inclusive':.25},
        'critical_consistency':{'train_raw_std_median_min_exclusive':.01,'train_within_std_min_exclusive':.005,
            'dev_to_train_masked_huber_max_inclusive':4.,'train_loss_denominator_floor':.001},
        'critical_numerical':'all valid finite six groups; all 1000 updates finite; variance support active every step; exact final step',
        'decision':'HEALTHY only if every critical check passes; else UNHEALTHY if experiment valid; no conditional override',
        'secondary':'effective rank, group correlation and user mean variance fraction descriptive only; no identity-removal intervention',
        'rationale':'1 percent error margin avoids a nominal equality pass; raw hidden floors screen near-constant bounded LSTM states; thresholds engineering screening gates, not physiological validity or statistical significance'}
    budget={'steps':1000,'batch_anchors':64,'window_seconds':2.,'effective_window_presentations':64000,'raw_sample_presentations':256000000,
        'seed':SEED,'optimizer':'AdamW','lr':3e-4,'weight_decay':1e-4,'betas':[.9,.999],'eps':1e-8,'foreach':False,'fused':False,
        'scheduler':None,'gradient_clipping':None,'checkpoint_steps':[250,500,750,1000],'scientific_health_steps':[1000],
        'train_sampling':'4 of 8 users without replacement; 2 sources per user; one random side per source; 8 endpoints without replacement from every second eligible endpoint (at least 2 seconds apart); deterministic seed+step',
        'mask_seed':'seed+100000+step; fresh CPU generator per step','evaluation_mask_seed':'seed+200000+batch_index',
        'device':'cuda','amp':False,'compile':False,'cudnn_enabled':False,'scientific_update_threads':1,'torch_cpu_threads':4,
        'estimated_training_seconds':1000*smoke_record['seconds_per_step'],
        'rationale':'bounded first screen, about one EMA time constant; balanced multiuser sampling satisfies unchanged variance support; no health-driven budget extension',
        'abort_policy':'stop on numerical/provenance/implementation failure; no sample substitution or architecture/hyperparameter rescue'}
    config={'kernel':CFG.to_dict(),'budget':budget,'health_protocol':protocol,'data_manifest_sha256':file_hash(RECORD/'DATA_MANIFEST.json'),
        'statistics_sha256':file_hash(RECORD/'physical_statistics.pt'),'source_identity':source_identity(),
        'smoke_sha256':file_hash(RECORD/'SMOKE.json'),'preregistered_unix':time.time()}
    save_json('PREREGISTRATION.json',config)
    print('PREREGISTERED',file_hash(RECORD/'PREREGISTRATION.json'),flush=True)

def train():
    config=json.loads((RECORD/'PREREGISTRATION.json').read_text());corpus=Corpus()
    assert config['source_identity']==source_identity() and config['data_manifest_sha256']==corpus.identity
    assert file_hash(RECORD/'physical_statistics.pt')==config['statistics_sha256']
    seed(SEED);prep=TeacherPreparation(CFG,stats_load()).train().cuda();opt=optimizer(prep)
    started=time.perf_counter();identities=[]
    with (RECORD/'training.jsonl').open('x') as log:
        for step in range(1000):
            values=step_once(prep,opt,corpus,step,'cuda')
            row={'step':step+1,'elapsed_seconds':time.perf_counter()-started,**values}
            log.write(json.dumps(row)+'\n');log.flush()
            if (step+1)%25==0:print(json.dumps(row),flush=True)
            if step+1 in config['budget']['checkpoint_steps']:
                checkpoint=teacher_checkpoint(prep,opt,config,corpus.identity,config['source_identity'],{'status':'not_evaluated'})
                path=RECORD/f'checkpoint_{step+1:04d}.pt';torch.save(checkpoint,path)
                identities.append({'step':step+1,'path':str(path),'sha256':file_hash(path),'scientific_identity':checkpoint['scientific_identity']})
    save_json('TRAINING_COMPLETE.json',{'steps':1000,'seconds':time.perf_counter()-started,'identities':identities,
                                      'preregistration_sha256':file_hash(RECORD/'PREREGISTRATION.json')})

def huber(a,b):
    return torch.nn.functional.huber_loss(a,b.expand_as(a),reduction='none',delta=1.)

@torch.no_grad()
def collect(prep,corpus,partition):
    items=corpus.evaluation_items(partition);collected={k:[] for k in ('raw','target','prediction','context','physical','amplitude')}
    for start in range(0,len(items),32):
        seq,_=corpus.sequence(items[start:start+32],'cuda')
        masked,mask,fractions=masked_view(seq,CFG,torch.Generator().manual_seed(SEED+200000+start//32))
        raw,valid=prep.ema.targets(seq,[3999]);target=prep.target_statistics(raw)
        hidden,mv=prep.student.targets(masked,[3999]);prediction=torch.stack([prep.predictors[l](hidden[:,:,:,l]) for l in range(2)],3)
        context,cv=prep.visible_context_reference(masked,[3999]);assert valid.all() and mv.all() and cv.all()
        phys=prep.physical_statistics(physical_summary(seq.emg,CFG))
        response=[]
        for gain in (.5,2.):
            other,ov=prep.ema.targets(replace(seq,emg=seq.emg*gain),[3999]);assert ov.all()
            response.append((other-raw).square().mean(-1).sqrt())
        amplitude=torch.stack(response).mean(0)
        vals=dict(raw=raw[:,0],target=target[:,0],prediction=prediction[:,0],context=context[:,0],physical=phys,amplitude=amplitude[:,0])
        for k,v in vals.items():assert torch.isfinite(v).all();collected[k].append(v.cpu())
    return {k:torch.cat(v) for k,v in collected.items()},items

def evaluate():
    config=json.loads((RECORD/'PREREGISTRATION.json').read_text());corpus=Corpus()
    assert config['source_identity']==source_identity()
    seed(SEED);prep=TeacherPreparation(CFG,stats_load()).train().cuda();opt=optimizer(prep)
    complete=json.loads((RECORD/'TRAINING_COMPLETE.json').read_text());record=complete['identities'][-1]
    assert record['step']==1000 and file_hash(record['path'])==record['sha256']
    checkpoint=torch.load(record['path'],weights_only=False)
    resume_teacher(checkpoint,prep,opt,config=config,data_manifest_hash=corpus.identity,source_version=config['source_identity'],expected_identity=record['scientific_identity'])
    prep.eval();train,ti=collect(prep,corpus,'train');dev,di=collect(prep,corpus,'emg_development')
    raw=train['raw'][:,None];valid=torch.ones(raw.shape[:-1],dtype=torch.bool)
    statistics=fit_frozen_statistics(raw,valid,{'partition':'train','version':'frozen-real-ema-train-eval-v01','manifest_hash':corpus.identity,'count':len(ti),'checkpoint_identity':record['scientific_identity']},CFG)
    torch.save({'state':statistics.state_dict(),'provenance':statistics.provenance},RECORD/'frozen_target_statistics.pt')
    groups=[];all_checks={}
    for k,tau in enumerate(CFG.teacher_scales):
        for l in range(2):
            tr=train['raw'][:,k,l].double();dv=dev['raw'][:,k,l].double()
            mu=tr.mean(0);std=tr.std(0).clamp_min(.05)
            a=torch.cat(((tr-mu)/std,torch.ones(len(tr),1)),1)
            b=torch.cat(((dv-mu)/std,torch.ones(len(dv),1)),1)
            penalty=torch.eye(129,dtype=torch.double);penalty[-1,-1]=0
            w=torch.linalg.solve(a.T@a+penalty,a.T@train['physical'].double())
            per={}
            for split,data,items,latent,design in [('train',train,ti,tr,a),('dev',dev,di,dv,b)]:
                mask_error=float(huber(data['prediction'][:,k,l],data['target'][:,k,l]).mean())
                constant=float(huber(data['target'][:,k,l],train['target'][:,k,l].mean(0)).mean())
                context=float(huber(data['context'][:,k,l],data['target'][:,k,l]).mean())
                summary=float(huber(design@w,data['physical'].double()).mean())
                summary_constant=float(huber(data['physical'].double(),train['physical'].double().mean(0)).mean())
                within=[]
                for source in sorted({r['filename'] for r,e in items}):
                    ids=[i for i,(r,e) in enumerate(items) if r['filename']==source]
                    within.append(float(latent[ids].std(0).median()))
                hidden_std=latent.std(0)
                singular=torch.linalg.svdvals(latent-latent.mean(0));p=singular/singular.sum().clamp_min(1e-30)
                user_means=torch.stack([latent[[i for i,(r,e) in enumerate(items) if r['user']==u]].mean(0) for u in sorted({r['user'] for r,e in items})])
                per[split]={'masked_huber':mask_error,'constant_huber':constant,'context_huber':context,
                    'masked_constant_ratio':mask_error/max(constant,1e-12),'masked_context_ratio':mask_error/max(context,1e-12),
                    'summary_huber':summary,'summary_constant_huber':summary_constant,'summary_ratio':summary/max(summary_constant,1e-12),
                    'raw_std_median':float(hidden_std.median()),'within_recording_std_median':float(np.median(within)),
                    'dead_dimension_fraction':float((hidden_std<.001).double().mean()),
                    'amplitude_delta_median':float(data['amplitude'][:,k,l].median()),
                    'effective_rank':float(torch.exp(-(p*p.clamp_min(1e-30).log()).sum())) if singular.sum()>0 else 0.,
                    'user_mean_variance_fraction':float(user_means.var(0,unbiased=False).sum()/latent.var(0,unbiased=False).sum().clamp_min(1e-30))}
            d=per['dev'];t=per['train'];checks={'constant':d['masked_constant_ratio']<.99,'visible_context':d['masked_context_ratio']<.99,
                'physical':d['summary_ratio']<.99,'raw_std':d['raw_std_median']>.01,'within':d['within_recording_std_median']>.005,
                'amplitude':d['amplitude_delta_median']>.001,'dead':d['dead_dimension_fraction']<=.25,
                'train_raw':t['raw_std_median']>.01,'train_within':t['within_recording_std_median']>.005,
                'generalization':d['masked_huber']/max(t['masked_huber'],.001)<=4.}
            name=f'{tau}s_layer{l+1}';groups.append({'group':name,**per,'checks':checks})
            all_checks.update({name+'/'+n:v for n,v in checks.items()})
    all_checks['fixed_steps']=int(prep.training_step)==1000
    trainlog=[json.loads(line) for line in (RECORD/'training.jsonl').read_text().splitlines()]
    all_checks['finite_all_updates']=len(trainlog)==1000 and all(np.isfinite([r[x] for x in ('masked','observed','physical','variance','total','gradient_norm')]).all() for r in trainlog)
    all_checks['variance_support']=all(r['variance_count']>=64*6 for r in trainlog)
    flat=dev['raw'].reshape(len(di),6,128).double();redundancy=[]
    for i in range(6):
        for j in range(i+1,6):
            a=flat[:,i]-flat[:,i].mean(0);b=flat[:,j]-flat[:,j].mean(0)
            redundancy.append({'a':i,'b':j,'centered_coordinate_cosine':float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))})
    healthy=all(all_checks.values())
    result={'status':'HEALTHY' if healthy else 'UNHEALTHY','decision':'TEACHER ACCEPTED FOR RESIDUAL PILOT' if healthy else 'DO NOT USE THIS TEACHER AS R TARGET YET',
            'groups':groups,'checks':all_checks,'group_redundancy':redundancy,'selected_fixed_budget_checkpoint':record,
            'manifest_hash':corpus.identity,'preregistration_sha256':file_hash(RECORD/'PREREGISTRATION.json'),
            'target_statistics_sha256':file_hash(RECORD/'frozen_target_statistics.pt'),
            'target_floor_fraction':float((statistics.std<.05).float().mean()),
            'claims_not_established':['HumanEngine works','R retains teacher','new API transfer','force','contact','fatigue','physiological nuisance removal'],
            'failure_class_if_unhealthy':'C/D unresolved: fixed objective/representation hypothesis not supported at this budget; B (budget) remains possible, not proved; no automatic rescue',
            'certificate_issued':False}
    if healthy:
        def avg(field):return float(np.mean([g['dev'][field] for g in groups]))
        health=TeacherHealth(avg('masked_huber'),avg('constant_huber'),avg('context_huber'),avg('summary_huber'),avg('summary_constant_huber'),
                             min(g['dev']['raw_std_median'] for g in groups),min(g['dev']['within_recording_std_median'] for g in groups),
                             min(g['dev']['amplitude_delta_median'] for g in groups),1000,1000,'emg_development',.01,.005,.001,
                             evaluation_config={**config['health_protocol'],'measured_group_results_hash':content_hash(groups),'measured_checks':all_checks,'preregistration_sha256':result['preregistration_sha256']},data_manifest_hash=corpus.identity,checkpoint_identity=record['scientific_identity'],
                             partition_identity=content_hash([{'file':r['filename'],'endpoint':e} for r,e in di]),budget_identity=content_hash(config['budget']))
        statistics=statistics.cuda();identity=teacher_artifact_identity(prep,statistics,health);health=replace(health,artifact_identity=identity)
        frozen=freeze_teacher(prep,statistics,health)
        torch.save({'ema':frozen.encoder.state_dict(),'statistics':frozen.statistics.state_dict(),'artifact':frozen.artifact},RECORD/'accepted_teacher.pt')
        save_json('TEACHER_CERTIFICATE.json',canonical(frozen.artifact));result['certificate_issued']=True;result['teacher_hash']=identity
    save_json('HEALTH_RESULTS.json',result)
    print(json.dumps({'status':result['status'],'failed':[k for k,v in all_checks.items() if not v]}),flush=True)

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Preserve v4 no-shared-Parameter-storage contract; cuDNN packs LSTM weights.
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','smoke','preregister','train','evaluate']);args=ap.parse_args()
    globals()[{'preregister':'preregister'}.get(args.phase,args.phase)]()

if __name__=='__main__':main()
