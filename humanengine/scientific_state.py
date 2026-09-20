"""Deterministic content identities, independent of pickle/device/storage layout."""
import copy
import hashlib
import json
import numpy as np
import torch


def canonical(value):
    if isinstance(value, torch.Tensor):
        t=value.detach().cpu().contiguous()
        return {"tensor_dtype":str(t.dtype),"shape":list(t.shape),
                "bytes_sha256":hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value,np.ndarray):
        return {"numpy_dtype":str(value.dtype),"shape":list(value.shape),
                "bytes_sha256":hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value,dict):
        # JSON objects in scientific manifests use string keys; optimizer state
        # also uses integer IDs. Preserve their type and deterministic order.
        if all(isinstance(k,str) for k in value):return {k:canonical(v) for k,v in value.items()}
        return {"typed_mapping":sorted([[canonical(k),canonical(v)] for k,v in value.items()],key=lambda p:repr(p[0]))}
    if isinstance(value,(tuple,list)):return [canonical(v) for v in value]
    return value


def content_hash(value):
    return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")).hexdigest()


def equal_state(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return isinstance(b,type(a)) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


def statistics_content(statistics):
    return {"definition":"fixed-train-statistics-v2","state":copy.deepcopy(statistics.state_dict()),
            "floor":statistics.floor if hasattr(statistics,"floor") else statistics.cfg.teacher_scale_floor,
            "provenance":copy.deepcopy(getattr(statistics,"provenance",{"partition":"train","version":"running-ema-moments-v1"})),
            "config":statistics.cfg.to_dict() if hasattr(statistics,"cfg") else None}


def he_semantics(model):
    from .execution import executable,validate_architecture
    validate_architecture(model)
    flags={k:p.requires_grad for k,p in model.named_parameters()}
    return {"version":"he-scientific-state-v4","executable":executable(model,architecture=True),"statistics":statistics_content(model.core.E.statistics),
            "input_schema":{"version":"published-emg-btc-timestamps-v1","schema":model.core.schema},"config":model.cfg.to_dict(),
            "module_configs":{k:m.cfg.to_dict() for k,m in model.named_modules() if hasattr(m,"cfg")},
            "amplitude_window_samples":model.core.E.width,
            "summary_version":model.cfg.summary_version,"route_version":model.cfg.route_version,
            "adapter_id":model.core.adapter_id,"adapter_version":"affine-loggain-lowrank-v1",
            "trainability":flags,"module_training":{k:m.training for k,m in model.named_modules()},"mode":model.optimization_mode}


def teacher_semantics(preparation):
    from .execution import executable,validate_architecture
    validate_architecture(preparation)
    return {"version":"teacher-scientific-state-v4","executable":executable(preparation,architecture=True),"config":preparation.cfg.to_dict(),
            "trainability":{k:p.requires_grad for k,p in preparation.named_parameters()},
            "module_training":{k:m.training for k,m in preparation.named_modules()},
            "module_configs":{k:m.cfg.to_dict() for k,m in preparation.named_modules() if hasattr(m,"cfg")},
            "physical_statistics":statistics_content(preparation.physical_statistics),
            "target_definition":preparation.cfg.target_version,
            "running_target_definition":{"version":"running-ema-moments-v1","floor":preparation.cfg.teacher_scale_floor,
                                         "momentum":preparation.cfg.teacher_stats_momentum,"partition":"train"}}


def runtime_metadata(module):
    fields=("cfg","provenance","floor","adapter_id","optimization_mode","width","artifact")
    return copy.deepcopy({"attributes":{name:{key:getattr(m,key) for key in fields if hasattr(m,key)} for name,m in module.named_modules()},
                          "trainability":{name:p.requires_grad for name,p in module.named_parameters()},
                          "training":{name:m.training for name,m in module.named_modules()}})


def restore_metadata(module,snapshot):
    for name,m in module.named_modules():
        for key,value in snapshot["attributes"][name].items():setattr(m,key,copy.deepcopy(value))
        m.training=snapshot["training"][name]
    for name,p in module.named_parameters():p.requires_grad_(snapshot["trainability"][name])
