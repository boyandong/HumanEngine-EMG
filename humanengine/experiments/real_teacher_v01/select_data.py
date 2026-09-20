"""Metadata-only deterministic acquisition plan; no signal or annotation reads."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
SEED = 'humanengine-real-teacher-v01-data-20260920'

def order(values):
    return sorted(values, key=lambda s: hashlib.sha256((SEED + '|' + s).encode()).hexdigest())

def main():
    metadata = ROOT / 'data/priority_pose_subset/raw/emg2pose_metadata.csv'
    index = ROOT / 'data/priority_pose_subset/raw/tar_index.jsonl'
    allowed = ('session','user','start','end','side','filename','held_out_user','split')
    with metadata.open(encoding='utf-8') as f:
        rows = [{k:r[k] for k in allowed} for r in csv.DictReader(f)]
    members = {Path(r['name']).stem:r for r in map(json.loads,index.read_text().splitlines()) if r['name'].endswith('.hdf5')}
    excluded_user = 'd387095792'  # entire official mini/S5/S6 source user
    selected = []
    for partition,nusers in [('train',8),('val',4)]:
        candidates = [r for r in rows if r['split']==partition and r['user']!=excluded_user]
        users = order({r['user'] for r in candidates})[:nusers]
        assert len(users)==nusers
        for user in users:
            rs = [r for r in candidates if r['user']==user]
            session = order({r['session'] for r in rs})[0]
            grouped = collections.defaultdict(list)
            for r in rs:
                if r['session']==session:
                    grouped[r['filename'].rsplit('_',1)[0]].append(r)
            sources = order({s for s,g in grouped.items() if {r['side'] for r in g}=={'left','right'} and len(g)==2})[:2]
            assert len(sources)==2
            for source in sources:
                for r in sorted(grouped[source],key=lambda r:r['side']):
                    selected.append({**r,'partition':'train' if partition=='train' else 'emg_development',
                                     'official_split':partition,'source_group':source,'archive_member':members[r['filename']]})
    assert not ({r['user'] for r in selected if r['partition']=='train'} & {r['user'] for r in selected if r['partition']=='emg_development'})
    total = sum(r['archive_member']['size'] for r in selected)
    assert total < 2 * 1024**3
    plan = {'schema':'real-teacher-acquisition-plan-v1','selection_seed':SEED,
            'status':'selected_before_EMG_access_not_yet_content_verified',
            'source_url':'https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar',
            'etag':'"62df27ac9a04800702061bac0ac1af32-6897"','archive_bytes':462824048640,
            'metadata_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest(),
            'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),
            'selection_rule':'SHA256(seed|id) order: 8 official train users, 4 official val users; 1 session/user; 2 complete bilateral recording pairs/session; no signal/label selection',
            'forbidden':'all official test recordings, final outcomes, all S6 annotation/evaluator data; entire mini user d387095792 excluded',
            'split_note':'Official train users also have separate official test recordings; those test recordings remain forbidden. Dev users are disjoint from train. No cross-user performance claim beyond this screening.',
            'missing_rule':'Stop; do not silently substitute a different recording or user',
            'files':selected,'total_bytes':total}
    (OUT/'ACQUISITION_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'files':len(selected),'bytes':total,'train_users':8,'dev_users':4,'sessions':12,'bilateral_sources':24}))

if __name__=='__main__':main()
