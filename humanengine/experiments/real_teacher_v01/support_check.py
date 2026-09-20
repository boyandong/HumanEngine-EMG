"""Real-data support and anchor checks against the frozen teacher data manifest.

This is an evidence step only. It reads the frozen manifest plus each file's
EMG/time fields, rebuilds the exact frozen sampling rule and reports whether the
unchanged variance-support gates (minimum_anchors / minimum_recordings /
minimum_users, anchor_separation_seconds) are satisfied by real data.

No signal is written, no model runs, no scientific statistic is fitted. The
sampling rule is read from the same frozen KernelConfig and the same seeded
construction used by the teacher runner, so a mismatch here would mean the
runner cannot satisfy its own contract.
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from humanengine.config import KernelConfig
from humanengine.scientific_state import canonical,content_hash

SEED=20260920
EXPECTED_FILES=48
TRAIN='train'
DEV='emg_development'


def file_hash(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def load_times(row):
    """Read only the allowed time field; never materialize pose/gesture labels."""
    with h5py.File(row['path'],'r') as handle:
        return row['filename'],handle['emg2pose/timeseries'].fields(['time'])[:]['time'].copy()


def batches(rows,cfg,steps):
    """Reproduce the frozen seeded TRAIN batch construction without reading EMG."""
    train=[r for r in rows if r['partition']==TRAIN]
    users=sorted({r['user'] for r in train})
    result=[]
    for step in range(steps):
        rng=np.random.default_rng(SEED+step)
        chosen=rng.choice(users,size=cfg.minimum_users,replace=False)
        items=[]
        for user in chosen:
            groups=sorted({r['source_group'] for r in train if r['user']==user})
            for group in groups:
                sides=[r for r in train if r['source_group']==group]
                row=sides[int(rng.integers(len(sides)))]
                ends=rng.choice(row['eligible_endpoints'][::2],size=8,replace=False)
                items.extend((row,int(e)) for e in ends)
        result.append(items)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--batches',type=int,default=32)
    parser.add_argument('--out',type=Path)
    arguments=parser.parse_args()
    cfg=KernelConfig().validate()
    manifest=json.loads(arguments.manifest.read_text())
    rows=manifest['records']
    assert manifest['status']=='frozen','Manifest is not frozen'
    assert len(rows)==EXPECTED_FILES==len({r['filename'] for r in rows}),'Wrong or duplicate file count'
    assert manifest['acquisition_plan_sha256']==file_hash(arguments.manifest.with_name('ACQUISITION_PLAN.json')),'Plan/manifest provenance mismatch'

    # --- split / exclusion / provenance structure -----------------------------
    train_rows=[r for r in rows if r['partition']==TRAIN]
    dev_rows=[r for r in rows if r['partition']==DEV]
    train_users=sorted({r['user'] for r in train_rows})
    dev_users=sorted({r['user'] for r in dev_rows})
    assert len(train_users)==8 and len(dev_users)==4,'Wrong train/dev user counts'
    assert not set(train_users)&set(dev_users),'Train/dev user overlap'
    # A synchronized recording group or a user must never cross the split boundary.
    ownership={}
    for row in train_rows+dev_rows:
        for key in (('source_group',row['source_group']),('session',row['session']),('user',row['user'])):
            if key in ownership and ownership[key]!=row['partition']:
                raise ValueError('Split group crosses partitions: '+repr(key))
            ownership[key]=row['partition']
    plan_path=arguments.manifest.with_name('ACQUISITION_PLAN.json')
    acquisition_plan=json.loads(plan_path.read_text())
    assert [r['filename'] for r in rows]==[r['filename'] for r in acquisition_plan['files']],'Manifest does not follow the frozen plan order'
    directories={str(Path(r['path']).parent) for r in rows}
    assert len(directories)==1,'Records are not in exactly one verified directory'
    for row in rows:
        assert row['official_split'] in ('train','val'),'Official test recording present'
        assert row['user']!='d387095792','Excluded mini source user present'
        assert 'mini' not in row['filename'].lower(),'mini source present'
        assert Path(row['path']).name==row['filename']+'.hdf5','Record path does not match its identity'
    assert {r['side'] for r in rows}=={'left','right'},'Bilateral coverage incomplete'

    # --- per-file identity re-check -------------------------------------------
    def verify(row):
        return row['filename'],file_hash(Path(row['path']))==row['sha256']
    with ThreadPoolExecutor(max_workers=8) as pool:
        digests=dict(pool.map(verify,rows))
    assert all(digests.values()),f"Content identity mismatch: {[k for k,v in digests.items() if not v]}"

    # --- real anchor support --------------------------------------------------
    sample_rows=[r for r in rows if r['eligible_endpoints']]
    with ThreadPoolExecutor(max_workers=8) as pool:
        times=dict(pool.map(load_times,sample_rows))
    plan=batches(rows,cfg,arguments.batches)
    observations=[]
    for step,items in enumerate(plan):
        assert len(items)==cfg.minimum_anchors,f'Batch {step} has {len(items)} anchors'
        users={r['user'] for r,_ in items}
        recordings={r['source_group'] for r,_ in items}
        files={r['filename'] for r,_ in items}
        assert len(users)>=cfg.minimum_users,f'Batch {step} covers {len(users)} users'
        assert len(recordings)>=cfg.minimum_recordings,f'Batch {step} covers {len(recordings)} recordings'
        separations=[]
        for row,end in items:
            t=times[row['filename']]
            separations.append(float(t[end])-float(t[end-1]))
            assert abs((float(t[end])-float(t[end-3999]))-2.0)<1e-6,f'Anchor window length wrong in {row["filename"]}'
        per_file={}
        for row,end in items:
            per_file.setdefault(row['filename'],[]).append(int(end))
        for name,ends in per_file.items():
            ordered=sorted(ends)
            gaps=[b-a for a,b in zip(ordered,ordered[1:])]
            if gaps:
                assert min(gaps)>=cfg.samples(cfg.anchor_separation_seconds),f'Anchors within one recording are closer than the required separation in {name}'
        observations.append({'step':step,'anchors':len(items),'users':len(users),'recordings':len(recordings),
                             'files':len(files),'min_same_file_endpoint_gap_samples':min((min([b-a for a,b in zip(sorted(v),sorted(v)[1:])]) for v in per_file.values() if len(v)>1),default=None)})
    minimum={'anchors':min(o['anchors'] for o in observations),
             'users':min(o['users'] for o in observations),
             'recordings':min(o['recordings'] for o in observations)}
    gates={'anchors':minimum['anchors']>=cfg.minimum_anchors,
           'users':minimum['users']>=cfg.minimum_users,
           'recordings':minimum['recordings']>=cfg.minimum_recordings}
    assert all(gates.values()),f'Support gate failed: {gates}'
    endpoints_total=sum(len(r['eligible_endpoints']) for r in rows)
    result={'schema':'real-teacher-support-check-v1','manifest':str(arguments.manifest),
            'manifest_sha256':file_hash(arguments.manifest),
            'manifest_age_check':'records re-hashed during this check',
            'batches_examined':arguments.batches,
            'required':{'anchors':cfg.minimum_anchors,'users':cfg.minimum_users,
                        'recordings':cfg.minimum_recordings,
                        'anchor_separation_samples':cfg.samples(cfg.anchor_separation_seconds)},
            'observed_minimum':minimum,'gates':gates,
            'train_users':train_users,'dev_users':dev_users,'train_dev_overlap':False,
            'files':len(rows),'train_files':len(train_rows),'dev_files':len(dev_rows),
            'sides':{'left':sum(r['side']=='left' for r in rows),'right':sum(r['side']=='right' for r in rows)},
            'sessions':len({r['session'] for r in rows}),
            'recording_sources':len({r['source_group'] for r in rows}),
            'eligible_endpoints_total':endpoints_total,
            'eligible_endpoints_min_per_file':min(len(r['eligible_endpoints']) for r in rows),
            'invalid_rows_total':sum(r['invalid_or_discontinuous_rows'] for r in rows),
            'bad_timestamp_deltas_total':sum(r['bad_timestamp_deltas'] for r in rows),
            'total_verified_bytes':sum(r['bytes'] for r in rows),
            'label_access':'EMG/time fields and identity attributes only; no pose, gesture, force, contact or S6 data',
            'observations':observations}
    result['content_identity']=content_hash(canonical(result))
    if arguments.out:
        assert not arguments.out.exists(),'Refusing to overwrite an existing support record'
        with arguments.out.open('x') as handle:json.dump(result,handle,indent=2)
        print('SUPPORT_RECORDED',file_hash(arguments.out),flush=True)
    print(json.dumps({'minimum':minimum,'gates':gates,'files':len(rows),
                      'train_users':len(train_users),'dev_users':len(dev_users),
                      'recording_sources':len({r['source_group'] for r in rows}),
                      'eligible_endpoints_total':endpoints_total}),flush=True)


if __name__=='__main__':main()
