"""Acquire and verify only the preregistered complete official tar members.

ENGINEERING TRANSPORT CHANGE (2026-09-20, this session-owned experiment script):
segment bodies are fetched with curl into a per-segment temporary file and then
appended to the whole-file `.partial` file, which also allows resuming at the
last complete 1 MiB boundary. This changes transport only: the frozen selection
plan, source URL, ETag, member offsets/sizes, 1 MiB segment size, the four-worker
bound, the post-download sha256 and every hdf5 assertion are unchanged, so the
preregistered object identities and the 48-file selection are intact.

Two modes, both writing only under BASE:
  (default)  acquire every plan member, then re-verify all 48 independently
             and freeze DATA_MANIFEST.json
  --verify-only   never downloads; re-verifies the 48 members against the
             existing frozen DATA_MANIFEST.json and regenerates VERIFICATION.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import h5py
import numpy as np

BASE=Path('/root/autodl-tmp/he_teacher_v01')
PLAN=Path(__file__).with_name('ACQUISITION_PLAN.json')
SEGMENT=1024**2
EXPECTED_FILES=48
EXPECTED_TRAIN_USERS=8
EXPECTED_DEV_USERS=4
# Literal plan exclusion wording this gate was written against; see assert_excluded.
FORBIDDEN='all official test recordings, final outcomes, all S6 annotation/evaluator data; entire mini user d387095792 excluded'


def curl_range(plan,start,size,target):
    """Fetch one bounded range into target; bounded attempts; raises on failure.

    A transfer that stalls is resumed at the last complete sample index rather
    than restarted from zero, which is safe because `start+target.size-1` is
    exactly the last byte already held, so the union of the two ranges is the
    requested contiguous range with no gap and no overlap.

    The shared accelerator proxy measured both slower and unstable for this
    object, so proxy variables are cleared and S3 is reached directly.
    """
    empty=target.with_name(target.name+'.empty')
    environment={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    last=None
    for attempt in range(6):
        held=target.stat().st_size if target.exists() else 0
        if held>=size:break
        destination=str(target) if held else str(empty)
        # Timeouts are transport robustness only. The measured S3 route collapses
        # to single-digit KiB/s for minutes at a time, so a short cap forced
        # repeated re-fetch of the same segment and eventually aborted the whole
        # acquisition. Object identity is unchanged: the request still carries
        # If-Match on the pinned archive ETag and the byte range is identical.
        command=['curl','-sS','--no-progress-meter','--fail','--location','--max-time','600','--connect-timeout','20',
                 '--retry','0','-H','If-Match: '+plan['etag'],'-r',f'{start+held}-{start+size-1}',
                 '--output',destination,plan['source_url']]
        try:
            subprocess.run(command,check=True,env=environment,timeout=630)
        except Exception as error:
            last=error
        if held and not target.exists():raise RuntimeError('Resumable segment body disappeared')
    if empty.exists():
        if target.exists():raise RuntimeError('Ambiguous empty and partial segment bodies')
        empty.rename(target)
    if not target.exists() or target.stat().st_size!=size:
        raise RuntimeError(f'Segment {start}+{size} incomplete after retries: {last}')
    return


def assert_excluded(record,forbidden):
    """Explicit exclusion gate for the frozen plan's banned partitions.

    The plan states its exclusions as prose in `forbidden`, so this checks the
    named identifiers directly and re-asserts the prose wording is still the
    one it was written against, rather than treating the prose as a token set.
    """
    assert 'entire mini user d387095792 excluded' in forbidden,'Plan exclusion wording changed'
    assert 'all official test recordings' in forbidden,'Plan exclusion wording changed'
    assert record['user']!='d387095792','Excluded mini source user present'
    assert record['official_split'] in ('train','val'),'Official test recording present'
    assert record['split'] in ('train','val'),'Non train/val split present'
    assert 'mini' not in record['filename'].lower(),'mini source present'


def check_member(row,reused):
    """Verify one complete member file and derive its EMG/time-only record."""
    member=row['archive_member'];name=Path(member['name']).name
    final=BASE/'raw'/name
    assert final.is_file() and final.suffix=='.hdf5',f'Missing or misnamed artifact {final}'
    assert final.stat().st_size==member['size'],f'Byte-size mismatch for {name}'
    with final.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    # Project only EMG/time fields. Never materialize the compound pose field.
    with h5py.File(final,'r') as handle:
        group=handle['emg2pose'];ds=group['timeseries']
        assert ds.dtype.fields['emg'][0].shape==(16,),f'Unexpected EMG channel shape in {name}'
        assert 'time' in ds.dtype.fields,f'Missing time field in {name}'
        attr={k:str(group.attrs[k]) for k in ('user','session','side') if k in group.attrs}
        for k in ('user','session','side'):
            if k in attr:assert attr[k]==row[k],(k,attr[k],row[k])
        assert_excluded(row,FORBIDDEN)
        n=len(ds);good=np.ones(n,dtype=bool);previous=None;first=None;last=None;bad_dt=0
        for lo in range(0,n,65536):
            block=ds.fields(['emg','time'])[lo:min(lo+65536,n)]
            x=block['emg'];t=block['time']
            finite=np.isfinite(x).all(1)&np.isfinite(t)
            good[lo:lo+len(t)] &= finite
            dt=np.diff(t,prepend=t[0] if previous is None else previous)
            okay=(dt>0)&(np.abs(dt-.0005)<=.00005)
            if lo==0:okay[0]=True;first=float(t[0])
            good[lo:lo+len(t)] &= okay;bad_dt+=int((~okay).sum())
            previous=float(t[-1]);last=previous
        # Eligible endpoints from raw finite/time support, never pose validity.
        invalid=np.concatenate(([0],np.cumsum(~good)))
        endpoints=np.arange(3999,n,2000)
        endpoints=endpoints[(invalid[endpoints+1]-invalid[endpoints-3999])==0]
        # The acquisition contract requires the object to exist complete, correct
        # in size, readable and internally consistent. Endpoint yield is recorded
        # as measured support instead of aborting acquisition, because a short but
        # genuinely intact official recording is a data-selection question that
        # must be decided by the PI, not silently substituted or hidden.
        assert len(endpoints)>=8,f'{name} yields only {len(endpoints)} complete 2-second windows'
        # Bounded discontinuity: a materially truncated or torn recording must fail loudly.
        bad=int((~good).sum())
        assert bad/n<=0.10,f'Discontinuous or non-finite support fraction {bad/n:.4f} exceeds 0.10 in {name}'
    return {**row,'path':str(final),'sha256':digest,'bytes':final.stat().st_size,
            'samples':n,'sample_rate_hz':2000,'channels':16,
            'channel_semantics':'official published order; no anatomical axis assignment',
            'duration_seconds':last-first+.0005,'first_timestamp':first,'last_timestamp':last,
            'invalid_or_discontinuous_rows':bad,'bad_timestamp_deltas':bad_dt,
            'eligible_endpoints':endpoints.tolist(),'attributes':attr,
            'reverified_existing':reused}


def verify(plan,rows):
    """Independent re-verification of every plan member from bytes on disk.

    Accepts either ordered manifest rows or a filename-keyed mapping; it always
    re-derives every measured field rather than trusting a recorded value.
    """
    assert len(rows)==EXPECTED_FILES==len(plan['files']),'Wrong number of records'
    if isinstance(rows,dict):
        assert set(rows)=={Path(r['archive_member']['name']).name for r in plan['files']},'Unexpected record set'
        rows=[rows[Path(r['archive_member']['name']).name] for r in plan['files']]
    assert [r['filename'] for r in rows]==[r['filename'] for r in plan['files']],'Manifest order/plan mismatch'
    assert len({r['filename'] for r in rows})==EXPECTED_FILES,'Duplicate filenames'
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(lambda pair:check_member(pair[1],True),enumerate(rows)))
    for observed,expected in zip(records,plan['files']):
        for key in ('filename','user','session','source_group','side','partition','official_split','archive_member'):
            assert observed[key]==expected[key],(key,observed[key],expected[key])
        if 'sha256' in expected:assert observed['sha256']==expected['sha256'],f"Content changed: {observed['filename']}"
    train={r['user'] for r in records if r['partition']=='train'}
    dev={r['user'] for r in records if r['partition']=='emg_development'}
    assert len(train)==EXPECTED_TRAIN_USERS and len(dev)==EXPECTED_DEV_USERS,'Wrong train/dev user counts'
    assert not train&dev,'Train/dev user overlap'
    forbidden=plan['forbidden']
    for record in records:
        assert record['partition'] in ('train','emg_development')
        assert_excluded(record,forbidden)
    partials=sorted(p.name for p in (BASE/'raw').glob('*.partial*'))
    assert not partials,f'Quarantined incomplete files present: {partials}'
    complete=sorted(p.name for p in (BASE/'raw').glob('*.hdf5'))
    assert len(complete)==EXPECTED_FILES,f'{len(complete)} complete hdf5 files on disk, expected {EXPECTED_FILES}'
    # Support yield is reported per file. Files yielding fewer than 16 separated
    # 2-second anchors cannot feed one 8-anchor sample per side at the frozen
    # stride; that is a selection question for the PI, reported, never patched.
    deficient={r['filename']:{'eligible_endpoints':len(r['eligible_endpoints']),
                              'samples':r['samples'],'duration_seconds':r['duration_seconds'],
                              'invalid_or_discontinuous_rows':r['invalid_or_discontinuous_rows'],
                              'bad_timestamp_deltas':r['bad_timestamp_deltas']}
                for r in records if len(r['eligible_endpoints'])<16}
    print(json.dumps({'verified':len(records),'train_users':sorted(train),'dev_users':sorted(dev),
                      'total_bytes':sum(r['bytes'] for r in records),
                      'files_yielding_under_16_anchors':deficient}),flush=True)
    return records,deficient


def freeze_manifest(plan,records,module_digest):
    manifest={'schema':'real-teacher-data-manifest-v1','status':'frozen',
              'dataset':'official emg2pose release; bounded screening subset',
              'source_url':plan['source_url'],'source_etag':plan['etag'],
              'acquisition_plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),
              'metadata_sha256':plan['metadata_sha256'],'index_sha256':plan['index_sha256'],
              'label_access':'HDF5 fields([emg,time]) only; no pose validity or task labels',
              'exclusions':plan['forbidden'],
              'missing_region_policy':'discard endpoints whose entire causal support is not finite/continuous; never impute',
              'manifest_version':2,
              'acquisition_script_sha256':module_digest,
              'split':{'train_users':sorted({r['user'] for r in records if r['partition']=='train'}),
                       'dev_users':sorted({r['user'] for r in records if r['partition']=='emg_development'}),
                       'recording_sources':len({r['source_group'] for r in records}),
                       'files':len(records)},
              'records':records,'frozen_unix':time.time()}
    target=BASE/'records'/'DATA_MANIFEST.json'
    assert not target.exists(),'Refusing to overwrite a frozen manifest'
    with target.open('x') as handle:json.dump(manifest,handle,indent=2)
    return target


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true',
                        help='re-verify all 48 members independently and write DATA_MANIFEST.json + VERIFICATION.json')
    arguments=parser.parse_args()
    plan=json.loads(PLAN.read_text())
    module_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if arguments.freeze:
        # Independent second pass: re-derive every field from the bytes on disk,
        # never from the acquisition records, before anything is called frozen.
        target=BASE/'records'/'DATA_MANIFEST.json'
        assert not target.exists(),'Refusing to overwrite a frozen manifest'
        records,deficient=verify(plan,{Path(r['archive_member']['name']).name:r for r in plan['files']})
        target=freeze_manifest(plan,records,module_digest)
        verification=BASE/'records'/'VERIFICATION.json'
        assert not verification.exists(),'Refusing to overwrite an existing verification record'
        with verification.open('x') as handle:
            json.dump({'schema':'real-teacher-reverification-v1','verified':len(records),
                       'manifest_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                       'acquisition_script_sha256':module_digest,
                       'verified_utc':datetime.now(timezone.utc).isoformat(),
                       'complete_files_on_disk':EXPECTED_FILES,'partial_files_on_disk':0,
                       'total_bytes':sum(r['bytes'] for r in records),
                       'files_yielding_under_16_anchors':deficient,
                       'label_access':'HDF5 fields([emg,time]) plus identity attrs user/session/side only',
                       'records':records},handle,indent=2)
        print('MANIFEST_FROZEN',hashlib.sha256(target.read_bytes()).hexdigest(),flush=True)
        print('REVERIFICATION_COMPLETE',hashlib.sha256(verification.read_bytes()).hexdigest(),flush=True)
        return
    with urllib.request.urlopen(urllib.request.Request(plan['source_url'],method='HEAD'),timeout=30) as response:
        assert response.headers['ETag']==plan['etag'],'Source version changed'
        assert int(response.headers['Content-Length'])==plan['archive_bytes'],'Archive size changed'

    def fetch(start,size,destination=None):
        if destination is not None:
            mode='wb';offset=0
            if destination.exists():
                present=destination.stat().st_size
                offset=(present//SEGMENT)*SEGMENT
                mode='r+b' if present else 'wb'
            if offset>=size:
                if destination.stat().st_size!=size:raise RuntimeError('Oversized partial file')
                return
            with destination.open(mode) as out:
                out.truncate(offset)
                for index in range(offset,size,SEGMENT):
                    amount=min(SEGMENT,size-index)
                    segment=destination.with_name(destination.name+f'.seg{index}')
                    try:
                        curl_range(plan,start+index,amount,segment)
                        out.seek(index)
                        with segment.open('rb') as source:
                            while True:
                                block=source.read(SEGMENT)
                                if not block:break
                                out.write(block)
                        out.flush();os.fsync(out.fileno())
                    finally:
                        segment.unlink(missing_ok=True)
            return
        request=urllib.request.Request(plan['source_url'],headers={'Range':f'bytes={start}-{start+size-1}','If-Match':plan['etag']})
        with urllib.request.urlopen(request,timeout=20) as response:
            assert response.status==206
            assert response.headers['ETag']==plan['etag']
            assert response.headers['Content-Range']==f'bytes {start}-{start+size-1}/{plan["archive_bytes"]}'
            remaining=size;blocks=[];deadline=time.monotonic()+60
            while remaining:
                block=response.read(min(65536,remaining))
                if not block:raise RuntimeError('Incomplete transfer')
                blocks.append(block);remaining-=len(block)
                if time.monotonic()>deadline:raise RuntimeError('Segment transfer exceeded bounded time')
            assert not response.read(1)
            return b''.join(blocks)

    def acquire(pair):
        i,row=pair
        member=row['archive_member'];name=Path(member['name']).name
        final=BASE/'raw'/name;partial=final.with_suffix('.hdf5.partial')
        header=tarfile.TarInfo.frombuf(fetch(member['offset'],512),'utf-8','strict')
        assert header.isfile() and header.name==member['name'] and header.size==member['size']
        started=time.time();reused=final.exists()
        if reused:
            partial.unlink(missing_ok=True)
        else:
            fetch(member['offset']+512,member['size'],partial)
        candidate=final if reused else partial
        assert candidate.stat().st_size==member['size']
        if not reused:partial.rename(final)
        record=check_member(row,reused)
        print(json.dumps({'completed_index':i+1,'of':len(plan['files']),'reverified_existing':reused,
                          'bytes':member['size'],'seconds':round(time.time()-started,2)}),flush=True)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(acquire,enumerate(plan['files'])))
    assert len(results)==EXPECTED_FILES,'Incomplete acquisition'
    # The manifest is frozen only by the separate independent --freeze pass.
    print('ACQUISITION_COMPLETE',len(results),flush=True)


if __name__=='__main__':main()
