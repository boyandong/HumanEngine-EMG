"""Read-only reuse of selected independent audit functions; write only lane evidence.

This is remediation-owned re-execution, NOT a new independent audit verdict.
Never call the external modules' run() functions (they write audit records).
"""
import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--first-audit',type=Path,required=True)
    parser.add_argument('--reaudit',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).parents[2]
    sys.path.insert(0,str(root));sys.path.insert(0,str(args.first_audit.parent))
    initial={str(p):digest(p) for p in (args.first_audit,args.reaudit)}
    spec=importlib.util.spec_from_file_location('execution_external_reaudit',args.reaudit)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    prior=audit.prior
    cases=[('causality_streaming',prior.causal_stream),('reset_isolation',prior.resets),
           ('detached_R_shadow',prior.order_route_shadow),('six_targets_causality',prior.teacher_targets_mask),
           ('residual_physics',prior.reductions_physics),('family_E0',prior.family_aliases),
           ('normal_CAGrad',prior.cagrad_numerical),('burnin_boundary_matrix',audit.f04_boundaries),
           ('unique_anchor_quotas',audit.f05_quotas),('effective_rank',audit.f07_rank),
           ('extreme_CAGrad',audit.f08_extreme),('MA_coefficient_clean_masked',audit.f09_and_views)]
    results=[]
    for name,fn in cases:
        random.seed(2319);np.random.seed(2319);torch.manual_seed(2319)
        start=time.perf_counter()
        try:
            details=fn()
            if details.get('status','PASS') not in ('PASS','COORDINATED','BASE_ONLY'):
                raise AssertionError(details)
            row={'name':name,'status':'PASS','details':details}
        except Exception as exc:row={'name':name,'status':'FAIL','error':repr(exc),'traceback':traceback.format_exc()}
        row['seconds']=time.perf_counter()-start;results.append(row)
        print(name+': '+row['status'],flush=True)
    final={str(p):digest(p) for p in (args.first_audit,args.reaudit)}
    payload={'scope':'Synthetic CPU remediation-owned re-execution of unchanged independent oracles; not fresh independent acceptance',
             'audit_source_hashes_before':initial,'audit_source_hashes_after':final,
             'audit_sources_unchanged':initial==final,'results':results}
    output=root/'humanengine'/'EXECUTION_PRESERVED_CONTRACTS.json'
    output.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    return 0 if initial==final and all(r['status']=='PASS' for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
