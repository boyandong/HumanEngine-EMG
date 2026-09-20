import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from .scientific_state import content_hash,statistics_content,he_semantics,teacher_semantics,equal_state
from .execution import executable,class_identity,named_parameter_topology
from .transaction import TrainingTransaction,assert_usable,rng_state,restore_rng


SIGNATURE_KEYS=("config_hash","data_manifest_hash","source_version","route_version","teacher_hash",
                "target_version","registry_hash","reference_scales_hash","summary_version","statistics_hash")


def seal(payload):
    payload=copy.deepcopy(payload)
    payload["content_hash"]=content_hash(payload)
    return payload


def verify_seal(payload):
    actual=content_hash({k:v for k,v in payload.items() if k!="content_hash"})
    if actual!=payload.get("content_hash"):raise ValueError("Checkpoint actual content hash mismatch")


def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def validate_signature(signature):
    if any(not signature.get(k) for k in SIGNATURE_KEYS):raise ValueError("Incomplete scientific compatibility signature")


def validate_model_definition(model_cfg,config,signature=None):
    definition=config.get("kernel",config)
    if content_hash(definition)!=content_hash(model_cfg.to_dict()):raise ValueError("Model/config definition mismatch")
    if signature is not None:
        for key in ("route_version","target_version","summary_version"):
            if signature[key]!=getattr(model_cfg,key):raise ValueError("Live model scientific version mismatch: "+key)


def tensor_continuation(module):
    return {"training":{n:m.training for n,m in module.named_modules()},
            "trainability":{n:p.requires_grad for n,p in module.named_parameters()},
            "gradients":{n:None if p.grad is None else p.grad.detach().clone() for n,p in module.named_parameters()},
            "buffers":{n:{"value":None if b is None else b.detach().clone(),"persistent":k not in m._non_persistent_buffers_set}
                       for prefix,m in module.named_modules() for k,b in m._buffers.items()
                       for n in [prefix+"."+k if prefix else k]}}


def validate_continuation(module,state,weights):
    current=tensor_continuation(module)
    if any(current[k].keys()!=state[k].keys() for k in current):
        raise ValueError("Continuation tensor layout mismatch")
    for n,v in state["buffers"].items():
        if v["persistent"]!=current["buffers"][n]["persistent"]:
            raise ValueError("Buffer persistence mismatch")
        if v["persistent"] and v["value"] is not None and (n not in weights or not equal_state(v["value"],weights[n])):
            raise ValueError("Conflicting persistent buffer representations")
    for n,p in module.named_parameters():
        grad=state["gradients"][n]
        if grad is not None and (grad.shape!=p.shape or grad.dtype!=p.dtype):
            raise ValueError("Continuation gradient shape/dtype mismatch")


def load_continuation(module,state):
    for n,m in module.named_modules():m.training=state["training"][n]
    for n,p in module.named_parameters():p.requires_grad_(state["trainability"][n])
    for n,p in module.named_parameters():p.grad=None if state["gradients"][n] is None else state["gradients"][n].clone().to(p)
    with torch.no_grad():
        for prefix,m in module.named_modules():
            for k,b in m._buffers.items():
                n=prefix+"."+k if prefix else k;v=state["buffers"][n]
                if v["persistent"]!=(k not in m._non_persistent_buffers_set) or (b is None)!=(v["value"] is None):
                    raise ValueError("Buffer persistence/layout mismatch")
                if b is not None:
                    if b.shape!=v["value"].shape or b.dtype!=v["value"].dtype:raise ValueError("Buffer shape/dtype mismatch")
                    b.copy_(v["value"])


def parameter_catalog(named_modules):
    """Map every supported scientific Parameter to one stable qualified name."""
    by_id={};definitions={}
    for prefix,module in named_modules:
        topology=named_parameter_topology(module)
        for row in topology["parameters"]:
            name=prefix+"."+row["name"] if prefix else row["name"]
            p=dict(module.named_parameters(remove_duplicate=False))[row["name"]]
            if id(p) in by_id:raise ValueError("Scientific parameter owned by multiple roots")
            by_id[id(p)]=name;definitions[name]={k:v for k,v in row.items() if k!="name"}
    return by_id,definitions


def optimizer_continuation(optimizer,catalog,definitions):
    """Serialize optimizer state by scientific name, never receiver slot number."""
    groups=[];owned=[]
    for group in optimizer.param_groups:
        names=[]
        for p in group["params"]:
            name=catalog.get(id(p))
            if name is None:raise ValueError("Optimizer contains an unregistered scientific parameter")
            if name in owned:raise ValueError("Optimizer parameter appears more than once: "+name)
            owned.append(name);names.append(name)
        options=copy.deepcopy({k:v for k,v in group.items() if k not in ("params","param_names")})
        groups.append({"parameters":names,"options":options})
    state={catalog[id(p)]:copy.deepcopy(value) for p,value in optimizer.state.items()
           if id(p) in catalog}
    if len(state)!=len(optimizer.state):raise ValueError("Optimizer state has an unregistered owner")
    return {"contract":"named-optimizer-continuation-v1","torch":torch.__version__,
            "implementation":class_identity(type(optimizer)),"parameter_definitions":{n:definitions[n] for n in owned},
            "param_groups":groups,"state":state}


def validate_optimizer_continuation(optimizer,saved,catalog,definitions):
    current=optimizer_continuation(optimizer,catalog,definitions)
    for key in ("contract","torch","implementation","parameter_definitions","param_groups"):
        if content_hash(current[key])!=content_hash(saved.get(key)):
            raise ValueError("Optimizer continuation incompatibility: "+key)
    owned=[n for group in saved["param_groups"] for n in group["parameters"]]
    if len(owned)!=len(set(owned)) or any(n not in owned for n in saved.get("state",{})):
        raise ValueError("Optimizer state ownership is ambiguous or unexpected")


def load_optimizer_continuation(optimizer,saved,catalog,definitions):
    validate_optimizer_continuation(optimizer,saved,catalog,definitions)
    current=optimizer.state_dict();name_to_id={}
    for live,snapshot,saved_group in zip(optimizer.param_groups,current["param_groups"],saved["param_groups"]):
        if len(live["params"])!=len(snapshot["params"]):raise ValueError("Optimizer group length changed")
        for p,pid,name in zip(live["params"],snapshot["params"],saved_group["parameters"]):
            if catalog.get(id(p))!=name:raise ValueError("Optimizer named ownership changed")
            name_to_id[name]=pid
    state={name_to_id[name]:copy.deepcopy(value) for name,value in saved["state"].items()}
    groups=[{**copy.deepcopy(group["options"]),"params":[name_to_id[n] for n in group["parameters"]]}
            for group in saved["param_groups"]]
    optimizer.load_state_dict({"state":state,"param_groups":groups})
    validate_optimizer_continuation(optimizer,saved,catalog,definitions)
    actual=optimizer_continuation(optimizer,catalog,definitions)
    if not equal_state(actual["state"],saved["state"]):raise ValueError("Optimizer state failed named post-load verification")


def verify_loaded_module(module,weights,continuation):
    actual=module.state_dict()
    if actual.keys()!=weights.keys() or any(not equal_state(actual[k],weights[k]) for k in actual):
        raise ValueError("Loaded named model state differs from selected checkpoint")
    if not equal_state(tensor_continuation(module),continuation):
        raise ValueError("Loaded continuation state differs from selected checkpoint")


def identify(payload):
    payload=copy.deepcopy(payload)
    payload["scientific_identity"]=content_hash({"definition":"scientific-checkpoint-v4","payload":payload})
    return seal(payload)


def verify_identity(payload,expected_identity=None):
    raw={k:v for k,v in payload.items() if k not in ("content_hash","scientific_identity")}
    actual=content_hash({"definition":"scientific-checkpoint-v4","payload":raw})
    if actual!=payload.get("scientific_identity") or (expected_identity is not None and actual!=expected_identity):
        raise ValueError("Scientific checkpoint identity/hash mismatch")


def validate_semantic_continuation(semantics,continuation):
    if (semantics['trainability']!=continuation['trainability']
        or semantics['module_training']!=continuation['training']):
        raise ValueError("Conflicting continuation and scientific semantics")


def he_checkpoint(model,heads,optimizers,controller,sampler,signature,registry,reference_scales,config):
    assert_usable(model,heads,controller,sampler,*optimizers.values())
    validate_signature(signature)
    validate_model_definition(model.cfg,config,signature)
    semantics=he_semantics(model)
    if signature["statistics_hash"]!=content_hash(semantics["statistics"]):raise ValueError("Incorrect claimed statistics hash")
    if signature["registry_hash"]!=content_hash(registry) or signature["reference_scales_hash"]!=content_hash(reference_scales) or signature["config_hash"]!=content_hash(config):
        raise ValueError("Checkpoint manifest contents do not match signature")
    catalog,definitions=parameter_catalog((("model",model),("heads",heads)))
    optimizer_state={k:optimizer_continuation(v,catalog,definitions) for k,v in optimizers.items()}
    all_owned=[n for record in optimizer_state.values() for group in record["param_groups"] for n in group["parameters"]]
    if len(all_owned)!=len(set(all_owned)):raise ValueError("Scientific parameter owned by multiple optimizers")
    return identify({"schema":"he-kernel-checkpoint-v4","signature":signature,"model":model.state_dict(),"scientific_state":semantics,"model_continuation":tensor_continuation(model),
                          "heads_continuation":tensor_continuation(heads),"heads_executable":executable(heads,architecture=True),
                          "training_heads":heads.state_dict(),"optimizers":optimizer_state,
                          "controller":controller.state_dict(),"sampler":sampler.state_dict(),"rng":rng_state(),
                          "registry":registry,"reference_scales":reference_scales,"config":config})


def resume_he(checkpoint,expected,model,heads,optimizers,controller,sampler,*,expected_identity=None):
    assert_usable(model,heads,controller,sampler,*optimizers.values())
    validate_signature(expected)
    if checkpoint.get("schema")!="he-kernel-checkpoint-v4":raise ValueError("Checkpoint schema mismatch")
    if checkpoint.get("signature")!=expected:raise ValueError("Scientific resume incompatibility")
    verify_seal(checkpoint)
    verify_identity(checkpoint,expected_identity)
    if content_hash(checkpoint["heads_executable"])!=content_hash(executable(heads,architecture=True)):
        raise ValueError("Training heads executable incompatibility")
    semantics=checkpoint["scientific_state"]
    if content_hash(semantics)!=content_hash(he_semantics(model)):raise ValueError("Scientific runtime semantics incompatibility")
    stats=copy.deepcopy(semantics["statistics"])
    stats["state"]={k:checkpoint["model"]["core.E.statistics."+k] for k in stats["state"]}
    if content_hash(stats)!=expected["statistics_hash"]:raise ValueError("Statistics content hash mismatch")
    validate_model_definition(model.cfg,checkpoint["config"],expected)
    for key,field in (("config_hash","config"),("registry_hash","registry"),("reference_scales_hash","reference_scales")):
        if content_hash(checkpoint[field])!=expected[key]:raise ValueError("Checkpoint content hash mismatch: "+field)
    if set(checkpoint["optimizers"])!=set(optimizers):raise ValueError("Optimizer layout mismatch")
    catalog,definitions=parameter_catalog((("model",model),("heads",heads)))
    for k,opt in optimizers.items():validate_optimizer_continuation(opt,checkpoint["optimizers"][k],catalog,definitions)
    validate_continuation(model,checkpoint["model_continuation"],checkpoint["model"])
    validate_semantic_continuation(semantics,checkpoint["model_continuation"])
    validate_continuation(heads,checkpoint["heads_continuation"],checkpoint["training_heads"])
    with TrainingTransaction(modules=(model,heads),objects=(*optimizers.values(),controller,sampler)):
        model.load_state_dict(checkpoint["model"],strict=True);heads.load_state_dict(checkpoint["training_heads"],strict=True)
        load_continuation(model,checkpoint["model_continuation"]);load_continuation(heads,checkpoint["heads_continuation"])
        verify_loaded_module(model,checkpoint["model"],checkpoint["model_continuation"])
        verify_loaded_module(heads,checkpoint["training_heads"],checkpoint["heads_continuation"])
        for k,opt in optimizers.items():load_optimizer_continuation(opt,checkpoint["optimizers"][k],catalog,definitions)
        controller.load_state_dict(checkpoint["controller"]);sampler.load_state_dict(checkpoint["sampler"]);restore_rng(checkpoint["rng"])


def teacher_checkpoint(preparation,optimizer,config,data_manifest_hash,source_version,health):
    assert_usable(preparation,optimizer)
    if not data_manifest_hash or not source_version:raise ValueError("Teacher provenance required")
    validate_model_definition(preparation.cfg,config)
    catalog,definitions=parameter_catalog((("preparation",preparation),))
    return identify({"schema":"emg-teacher-preparation-v4","preparation":preparation.state_dict(),"scientific_state":teacher_semantics(preparation),
                          "continuation":tensor_continuation(preparation),
                          "target_statistics_identity":content_hash(statistics_content(preparation.target_statistics)),
                          "optimizer":optimizer_continuation(optimizer,catalog,definitions),"training_step":int(preparation.training_step),
                          "rng":rng_state(),"config":config,"config_hash":content_hash(config),
                          "training_data_manifest_hash":data_manifest_hash,"source_version":source_version,
                          "target_version":preparation.cfg.target_version,"health_check_results":health})


def verify_frozen(root,snapshot):
    failures=[]
    for row in snapshot:
        path=Path(root)/row["path"]
        if not path.is_file() or file_hash(path).lower()!=row["sha256"].lower():failures.append(row["path"])
    return failures


def resume_teacher(checkpoint,preparation,optimizer,*,config,data_manifest_hash,source_version,expected_identity=None):
    assert_usable(preparation,optimizer)
    validate_model_definition(preparation.cfg,config)
    expected=(content_hash(config),data_manifest_hash,source_version,preparation.cfg.target_version)
    observed=tuple(checkpoint.get(k) for k in ("config_hash","training_data_manifest_hash","source_version","target_version"))
    if checkpoint.get("schema")!="emg-teacher-preparation-v4" or expected!=observed or content_hash(checkpoint["config"])!=expected[0]:
        raise ValueError("Teacher resume incompatibility")
    verify_seal(checkpoint)
    verify_identity(checkpoint,expected_identity)
    target=statistics_content(preparation.target_statistics)
    target["state"]={k:checkpoint["preparation"]["target_statistics."+k] for k in target["state"]}
    if content_hash(target)!=checkpoint["target_statistics_identity"]:raise ValueError("Running target statistics identity mismatch")
    if content_hash(checkpoint["scientific_state"])!=content_hash(teacher_semantics(preparation)):
        raise ValueError("Teacher statistics semantics incompatibility")
    physical=checkpoint["scientific_state"]["physical_statistics"]["state"]
    if any(content_hash(v)!=content_hash(checkpoint["preparation"]["physical_statistics."+k]) for k,v in physical.items()):
        raise ValueError("Teacher statistics content mismatch")
    validate_continuation(preparation,checkpoint["continuation"],checkpoint["preparation"])
    validate_semantic_continuation(checkpoint["scientific_state"],checkpoint["continuation"])
    catalog,definitions=parameter_catalog((("preparation",preparation),))
    validate_optimizer_continuation(optimizer,checkpoint["optimizer"],catalog,definitions)
    with TrainingTransaction(modules=(preparation,),objects=(optimizer,)):
        preparation.load_state_dict(checkpoint["preparation"],strict=True)
        load_continuation(preparation,checkpoint["continuation"])
        verify_loaded_module(preparation,checkpoint["preparation"],checkpoint["continuation"])
        if int(preparation.training_step)!=checkpoint["training_step"]:raise ValueError("Teacher step mismatch")
        load_optimizer_continuation(optimizer,checkpoint["optimizer"],catalog,definitions);restore_rng(checkpoint["rng"])
