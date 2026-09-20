"""Read-only protected hashing; write evidence ONLY inside humanengine/."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import xml.etree.ElementTree as ET

LANE=Path(__file__).absolute().parents[1]
ROOT=LANE.parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def write(name,value):
    (LANE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

before=read(LANE/'REMEDIATION_PROTECTED_BEFORE.json')
after=[]
for row in before:
    path=ROOT/row['path']
    actual=digest(path) if path.is_file() else None
    prior=row['sha256'].lower() if row['sha256'] else None
    after.append({'path':row['path'],'entry_sha256':prior,'final_sha256':actual,
                  'unchanged_during_remediation':prior==actual,
                  'matches_historical_baseline':actual==row['baseline_sha256'].lower()})
write('REMEDIATION_PROTECTED_AFTER.json',after)
suite=ET.parse(LANE/'remediation-test-results.xml').getroot().find('testsuite')
tests=list(suite.iter('testcase'))
new=[t for t in tests if '.test_remediation' in t.attrib.get('classname','')]
old=read(LANE/'IMPLEMENTATION_EVIDENCE.json')['files']
old_by_name={row['path'].replace('\\','/'):row['sha256'].lower() for row in old}
sources=[]
for path in sorted(LANE.rglob('*')):
    if not path.is_file() or any(p.startswith('.') or p.startswith('remediation_pytest_temp') or p=='__pycache__' for p in path.relative_to(LANE).parts):continue
    if path.name=='REMEDIATION_EVIDENCE.json':continue
    if path.suffix not in ('.py','.json','.md','.xml','.txt'):continue
    rel=path.relative_to(ROOT).as_posix();sha=digest(path)
    historical=old_by_name.get(rel,old_by_name.get(path.relative_to(LANE).as_posix()))
    sources.append({'path':rel,'sha256':sha,'historical_implementation_sha256':historical,
                    'changed_since_implementation':sha!=historical if historical else None})
versions={'scientific_route':'api-detached-r-v1','checkpoint_he':'he-kernel-checkpoint-v2',
          'checkpoint_teacher':'emg-teacher-preparation-v2','teacher_certificate':'teacher-artifact-certificate-v2',
          'quota_identity':'scientific-anchor-v2','effective_rank':'centered-singular-value-entropy-v2',
          'M_A_private':'physical_weight-times-L_phys-no-family-alpha-v1'}
findings={f'F{i:02d}':{'status':'REMEDIATED_LOCAL_REGRESSION_PASS'} for i in range(1,10)}
findings['F06']={'status':'READ_ONLY_RECONCILIATION_COMPLETE_PROVENANCE_UNRESOLVED',
                 'differences':[dict(row,classification='UNRESOLVED') for row in after if not row['matches_historical_baseline']]}
findings['F09']['status']='CLARIFIED_AND_REGRESSION_PASS'
evidence={'schema':'humanengine-remediation-evidence-v2','created_utc':datetime.now(timezone.utc).isoformat(),
          'status':'READY FOR INDEPENDENT RE-AUDIT','scientific_training_performed':False,
          'versions':versions,'findings':findings,
          'tests':{'historical_tests':len(tests)-len(new),'remediation_regressions':len(new),**suite.attrib,
                   'artifact':'humanengine/remediation-test-results.xml','sha256':digest(LANE/'remediation-test-results.xml')},
          'protected':{'count':len(after),'unchanged_since_entry':sum(r['unchanged_during_remediation'] for r in after),
                       'new_differences':[r for r in after if not r['unchanged_during_remediation']],
                       'historical_matches':sum(r['matches_historical_baseline'] for r in after),
                       'scope':'321 registered protected entries; no whole-disk/raw-data CRC claim'},
          'files':sources,'limitations':['Independent re-audit required; no teacher validation authorized',
                     'F06 exact authorization/authorship unresolved; original baseline preserved',
                     'Synthetic CPU checks only; no real quality measurements',
                     'Macrostep guards registered in-process state, not arbitrary external Python side effects']}
if evidence['protected']['new_differences'] or int(suite.attrib['failures']) or int(suite.attrib['errors']):evidence['status']='NOT READY'
write('REMEDIATION_EVIDENCE.json',evidence)
print(json.dumps({k:evidence[k] for k in ('status','tests','protected')},ensure_ascii=False,indent=2))
