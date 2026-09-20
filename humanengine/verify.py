"""Explicit local verification only; never opens real data or starts training."""
import argparse
import json
from pathlib import Path
import torch
from .config import load_config
from .api import HumanEngine
from .core.summaries import SignalStatistics
from .manifest import verify_frozen


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--synthetic",action="store_true",required=True)
    parser.add_argument("--check-frozen",action="store_true")
    args=parser.parse_args();cfg,raw=load_config(args.config)
    torch.set_num_threads(1);torch.manual_seed(17)
    stats=SignalStatistics(torch.zeros(cfg.channels),torch.ones(cfg.channels),
                           {"partition":"synthetic","version":"verification-only"},cfg.signal_scale_floor)
    model=HumanEngine(cfg,stats).eval();n=cfg.samples(.2)
    x=torch.randn(1,n,cfg.channels);times=torch.arange(n,dtype=torch.float64)[None]/cfg.sample_rate
    with torch.no_grad():out,state=model.forward_sequence(x,times)
    report={"mode":"synthetic CPU forward only","scientific_training":False,"H_shape":list(out.H.shape),
            "parameter_count":sum(p.numel() for p in model.parameters()),"finite":bool(torch.isfinite(out.H).all())}
    if args.check_frozen:
        lane=Path(__file__).parent;root=lane.parent
        failures=verify_frozen(root,json.loads((lane/"FROZEN_BASELINE.json").read_text(encoding="utf-8-sig")))
        report["frozen_mismatches"]=failures
        if failures:raise RuntimeError(str(failures))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
