import random
import numpy as np
import pytest
import torch
from humanengine.config import KernelConfig
from humanengine.core.summaries import SignalStatistics
from humanengine.api import HumanEngine
from humanengine.contracts import EMGSequence,AnchorMetadata

torch.set_num_threads(1)

@pytest.fixture(autouse=True)
def seed():
    random.seed(17);np.random.seed(17);torch.manual_seed(17)

@pytest.fixture
def cfg():return KernelConfig(channels=2).validate()

def signal_stats(cfg,physical=False):
    d=6*cfg.channels if physical else cfg.channels
    return SignalStatistics(torch.zeros(d),torch.ones(d),{"partition":"synthetic","version":"test-only-identity"},cfg.signal_scale_floor)

@pytest.fixture
def model(cfg):return HumanEngine(cfg,signal_stats(cfg)).eval()

def sequence(cfg,batch=2,length=800):
    x=torch.randn(batch,length,cfg.channels)
    times=torch.arange(length,dtype=torch.float64)[None].expand(batch,-1)/cfg.sample_rate
    return EMGSequence(x,times,torch.ones(batch,length,dtype=torch.bool),torch.zeros(batch,length,dtype=torch.long))

def anchor_meta(n=64):
    return AnchorMetadata(tuple("u"+str(i%4) for i in range(n)),
                          tuple("r"+str(i%8) for i in range(n)),tuple(float(i//8)*2 for i in range(n)))


def bind_synthetic_health(prep,stats,health):
    """Identity-only fixture: no real health evaluation or thresholds."""
    from dataclasses import replace
    from humanengine.teacher.preparation import teacher_artifact_identity
    health=replace(health,evaluation_config={"synthetic_wiring_only":True},data_manifest_hash="synthetic-data",
                   checkpoint_identity="synthetic-step-0",partition_identity="synthetic",budget_identity="synthetic-no-training")
    return replace(health,artifact_identity=teacher_artifact_identity(prep,stats,health))
