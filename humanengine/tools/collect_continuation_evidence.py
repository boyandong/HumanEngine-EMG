"""Collect continuation-closure evidence without touching protected lanes."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy
import pytest
import scipy
import torch

ROOT=Path(__file__).parents[2]
LANE=ROOT/'humanengine'
AUDIT=Path(r'D:\TEMP\humanengine_fresh_execution_reaudit')


def digest(path):
    if not path.is_file():return None
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(name,value):
    (LANE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def git(*args):
    result=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True)
    return {'exit_code':result.returncode,'stdout':result.stdout.strip(),'stderr':result.stderr.strip()}


def main():
    entry=json.loads((AUDIT/'protected_exit.json').read_text(encoding='utf-8-sig'))
    current=[]
    for row in entry:
        now=dict(row);now['entry_sha256']=row['sha256'];now['sha256']=digest(ROOT/row['path']);current.append(now)
    write('CONTINUATION_PROTECTED_BEFORE.json',entry)
    write('CONTINUATION_PROTECTED_AFTER.json',current)
    drift=[r for r in current if r['sha256']!=r['entry_sha256']]
    historical=[r for r in current if r['sha256']!=r['baseline_sha256']]

    old_rows=json.loads((AUDIT/'lane_entry.json').read_text(encoding='utf-8-sig'))
    old={str(Path(r['path'])):r['sha256'].lower() for r in old_rows
         if '__pycache__' not in Path(r['path']).parts and '.pytest_cache' not in Path(r['path']).parts}
    write('CONTINUATION_LANE_BEFORE.json',old_rows)
    excluded={'CONTINUATION_EVIDENCE.json','CONTINUATION_LANE_AFTER.json','CONTINUATION_PROTECTED_AFTER.json'}
    files=[p for base in (LANE,ROOT/'humanengine_design') for p in sorted(base.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    after={str(p):digest(p) for p in files if p.name not in excluded}
    write('CONTINUATION_LANE_AFTER.json',after)
    changed=[p for p,h in old.items() if after.get(p)!=h]
    added=[p for p in after if p not in old]
    for name in excluded:
        path=str(LANE/name)
        if path not in old and path not in added:added.append(path)

    suites=list(ET.parse(LANE/'continuation-final-suite.xml').getroot().iter('testsuite'))
    suite={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    suite['seconds']=sum(float(s.attrib.get('time',0)) for s in suites)
    preserved=json.loads((LANE/'EXECUTION_PRESERVED_CONTRACTS.json').read_text(encoding='utf-8'))
    preserved_ok=preserved['audit_sources_unchanged'] and all(r['status']=='PASS' for r in preserved['results'])
    source={str(p.relative_to(ROOT)):digest(p) for p in files if p.suffix=='.py' or 'configs' in p.parts}
    artifacts=['CONTINUATION_CLOSURE_REPORT.md','CONTINUATION_DECISIONS.md','CONTINUATION_REPRODUCTIONS_BEFORE.json',
               'continuation-final-suite.xml','EXECUTION_PRESERVED_CONTRACTS.json','README.md','IMPLEMENTATION_MAP.md','SESSION_HANDOVER.md',
               'CONTINUATION_PROTECTED_BEFORE.json','CONTINUATION_PROTECTED_AFTER.json','CONTINUATION_LANE_BEFORE.json','CONTINUATION_LANE_AFTER.json']
    evidence={
        'status':'READY FOR FINAL ACCEPTANCE AUDIT' if not any(suite[k] for k in ('failures','errors','skipped')) and preserved_ok and not drift else 'NOT READY',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'root':str(ROOT),
        'scope':'Synthetic single-process eager CPU continuation closure; not real teacher/scientific validation',
        'versions':{'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,
                    'numpy':numpy.__version__,'scipy':scipy.__version__,'pytest':pytest.__version__},
        'suite':suite,'preserved_contract_groups':len(preserved['results']),'preserved_contracts_passed':preserved_ok,
        'continuation_oracle':'5 continuous updates exactly equal 2 + checkpoint/resume + 3 updates for named model and optimizer state',
        'negative_matrix':['optimizer implementation','lr','betas','parameter order','missing parameter','extra parameter',
                           'group membership','parameter storage alias','model executable','statistics','adapter/mode'],
        'source_and_config_sha256':source,'artifact_sha256':{name:digest(LANE/name) for name in artifacts},
        'lane_files_modified':changed,'lane_files_added':added,
        'protected':{'entries':len(current),'unchanged_since_fresh_audit_handoff':len(current)-len(drift),
                     'drift_since_handoff':drift,'historical_matches':len(current)-len(historical),
                     'historical_changed':sum(r['sha256'] is not None for r in historical),
                     'historical_missing':sum(r['sha256'] is None for r in historical),
                     'authorship_process_authorization':'UNKNOWN; do not infer ownership'},
        'git':{'root':git('rev-parse','--show-toplevel'),'baseline_head':git('-C','baseline/emg2pose','rev-parse','HEAD'),
               'baseline_status':git('-C','baseline/emg2pose','status','--porcelain')},
        'findings':{key:'CLOSED IN LOCAL SYNTHETIC TESTS; FINAL INDEPENDENT ACCEPTANCE REQUIRED' for key in ('X01','X02','X03','X04')},
        'not_performed':['real teacher validation/pretraining','HE_FULL training','GPU/remote job','scientific sweep',
                         'environment installation','S6 edit/rebaseline','hidden answer inspection','performance claim']}
    write('CONTINUATION_EVIDENCE.json',evidence)
    print(json.dumps({'status':evidence['status'],'suite':suite,'preserved':preserved_ok,
                      'protected_unchanged':len(current)-len(drift),'protected_entries':len(current),
                      'historical_matches':len(current)-len(historical),'historical_differences':len(historical),
                      'modified':changed,'added':added},indent=2,ensure_ascii=True))


if __name__=='__main__':main()
