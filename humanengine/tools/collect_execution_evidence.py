"""Collect reproducible source/test/protected-file evidence, with no external writes."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy
import scipy
import pytest
import torch

ROOT=Path(__file__).parents[2]
LANE=ROOT/'humanengine'


def digest(path):
    if not path.is_file():return None
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(name,value):
    (LANE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def git(*args):
    r=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True)
    return {'exit_code':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()}


def main():
    before=json.loads((LANE/'EXECUTION_PROTECTED_BEFORE.json').read_text(encoding='utf-8-sig'))
    after=[{**r,'entry_sha256':r['sha256'],'sha256':digest(ROOT/r['path'])} for r in before]
    write('EXECUTION_PROTECTED_AFTER.json',after)
    during=[r for r in after if r['sha256']!=r['entry_sha256']]
    historical=[r for r in after if r['sha256']!=r['baseline_sha256']]
    xml=ET.parse(LANE/'execution-final-suite.xml').getroot()
    suites=list(xml.iter('testsuite'))
    summary={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    summary['seconds']=sum(float(s.attrib.get('time',0)) for s in suites)
    cases=[{'name':t.attrib['name'],'class':t.attrib.get('classname'),'seconds':float(t.attrib.get('time',0)),
            'status':'FAIL' if t.find('failure') is not None or t.find('error') is not None else 'SKIP' if t.find('skipped') is not None else 'PASS'} for t in xml.iter('testcase')]
    summary['new_tests']=sum('test_execution_integrity' in (r['class'] or '') for r in cases)
    summary['previous_tests']=summary['tests']-summary['new_tests']
    original=json.loads((LANE/'EXECUTION_LANE_BEFORE.json').read_text(encoding='utf-8-sig'))
    excluded={'EXECUTION_EVIDENCE.json','EXECUTION_LANE_AFTER.json'}
    current={str(p.relative_to(ROOT)):digest(p) for p in sorted(LANE.rglob('*'))
             if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    write('EXECUTION_LANE_AFTER.json',current)
    changed=[p for p,h in original.items() if current.get(p)!=h]
    added=[p for p in current if p not in original]
    source={p:h for p,h in current.items() if p.endswith('.py') or '\\configs\\' in p or '/configs/' in p}
    contracts=json.loads((LANE/'EXECUTION_PRESERVED_CONTRACTS.json').read_text(encoding='utf-8'))
    preserved=contracts['audit_sources_unchanged'] and all(r['status']=='PASS' for r in contracts['results'])
    artifacts=['execution-final-suite.xml','EXECUTION_PRESERVED_CONTRACTS.json','EXECUTION_REPRODUCTIONS_BEFORE.json',
               'EXECUTION_PROTECTED_BEFORE.json','EXECUTION_PROTECTED_AFTER.json','EXECUTION_LANE_BEFORE.json',
               'EXECUTION_LANE_AFTER.json','EXECUTION_DECISIONS.md','EXECUTION_REMEDIATION_REPORT.md','SESSION_HANDOVER.md']
    evidence={'status':'READY FOR FRESH INDEPENDENT RE-AUDIT' if not any(summary[k] for k in ('failures','errors','skipped')) and preserved else 'NOT READY',
              'generated_utc':datetime.now(timezone.utc).isoformat(),'root':str(ROOT),
              'scope':'Synthetic CPU; remediation-owned verification, not independent acceptance or scientific training readiness',
              'versions':{'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,
                          'numpy':numpy.__version__,'scipy':scipy.__version__,'pytest':pytest.__version__},
              'test_command':"$env:PYTHONDONTWRITEBYTECODE='1'; & 'D:\\Anaconda3\\envs\\emgforce\\python.exe' -B -m pytest humanengine/tests -q -p no:cacheprovider --junitxml=humanengine/execution-final-suite.xml",
              'suite':summary,'test_cases':cases,'preserved_contract_groups':len(contracts['results']),
              'preserved_contracts_passed':preserved,'source_and_config_sha256':source,
              'artifact_sha256':{n:digest(LANE/n) for n in artifacts},
              'lane_files_modified':changed,'lane_files_added':added,
              'protected':{'entries':len(after),'unchanged_since_entry':len(after)-len(during),
                           'drift_since_entry':during,'historical_matches':len(after)-len(historical),
                           'historical_changed':sum(r['sha256'] is not None for r in historical),
                           'historical_missing':sum(r['sha256'] is None for r in historical),'historical_differences':historical,
                           'authorship_process_authorization':'UNRESOLVED; do not infer ownership',
                           'frozen_baseline_unchanged':digest(LANE/'FROZEN_BASELINE.json')==original['humanengine\\FROZEN_BASELINE.json']},
              'git':{'root':git('rev-parse','--show-toplevel'),'baseline_head':git('-C','baseline/emg2pose','rev-parse','HEAD'),
                     'baseline_status':git('-C','baseline/emg2pose','status','--porcelain')},
              'findings':{k:'CLOSED IN LOCAL SYNTHETIC TESTS; FRESH INDEPENDENT RE-AUDIT REQUIRED' for k in ('R01','R02','R03','R04','R05','R06','N01')},
              'not_performed':['real teacher validation/pretraining','HE_FULL training','GPU/remote job','scientific sweep',
                               'environment installation','S6 edit','hidden answer inspection','scientific performance claim']}
    write('EXECUTION_EVIDENCE.json',evidence)
    print(json.dumps({'status':evidence['status'],'suite':summary,'preserved_contract_groups':len(contracts['results']),
                      'protected_unchanged_since_entry':len(after)-len(during),'protected_entries':len(after),
                      'drift_since_entry':[r['path'] for r in during],
                      'historical_matches':len(after)-len(historical),'historical_differences':len(historical),
                      'lane_files_modified':changed},indent=2,ensure_ascii=True))


if __name__=='__main__':main()
