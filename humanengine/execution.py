"""Executable identity and eager CPU execution protection (schema v4)."""
import copy
from dataclasses import asdict,is_dataclass
import hashlib
import inspect
from functools import lru_cache
import weakref
from pathlib import Path
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from .scientific_state import canonical,content_hash,equal_state

EXECUTION_VERSION="he-executable-v4"
# Derived RNN caches are deliberately excluded from the architectural fingerprint;
# their objects are still captured by TrainingTransaction for rollback.
SPECIAL=("_reversed_padding_repeated_twice","_flat_weights_names","_all_weights")
SEMANTIC=("cfg","provenance","floor","artifact","adapter_id","optimization_mode","training")
STRUCTURAL=set(vars(nn.Module()))-{"training"}
DERIVED={"_flat_weights","_flat_weight_refs"}


def attributes(module,architecture=False):
    result={}
    for key,value in vars(module).items():
        # Skip only explicitly handled framework bookkeeping; unknown private
        # behavior attributes are fingerprinted as well as public attributes.
        if key in STRUCTURAL or key in DERIVED:continue
        if architecture and key in SEMANTIC:continue
        if callable(value):
            raise ValueError("Instance-level executable override unsupported: "+key)
        result[key]=value
    return result


def check_supported_module(m):
    hook_fields=("_forward_hooks","_forward_pre_hooks","_backward_hooks","_backward_pre_hooks",
                 "_state_dict_hooks","_state_dict_pre_hooks","_load_state_dict_pre_hooks","_load_state_dict_post_hooks")
    if any(getattr(m,k,{}) for k in hook_fields):
        raise ValueError("Executable module hooks require an explicit versioned implementation")
    if isinstance(m,nn.RNNBase):
        weights=[getattr(m,k,None) for k in m._flat_weights_names]
        if len(weights)!=len(m._flat_weights) or any(a is not b for a,b in zip(weights,m._flat_weights)):
            raise ValueError("Executable RNN cache is inconsistent with registered weights")


@lru_cache(None)
def class_identity(cls):
    try:source=inspect.getsource(cls).encode('utf-8')
    except (OSError,TypeError):raise ValueError("Uninspectable executable module: "+str(cls))
    return {"class":cls.__module__+"."+cls.__qualname__,"source_sha256":hashlib.sha256(source).hexdigest()}


def plain(value):
    if is_dataclass(value):return plain(asdict(value))
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,set):return sorted(value)
    return canonical(value)


def executable(module,*,architecture=False,frozen=False):
    topology=named_parameter_topology(module)
    rows={}
    for name,m in module.named_modules():
        check_supported_module(m)
        attrs=attributes(m,architecture)
        if frozen and not architecture:attrs["training"]=False
        rows[name]={**class_identity(type(m)),"attributes":plain(attrs),
                    "parameters":{k:None if p is None else [list(p.shape),str(p.dtype)] for k,p in m._parameters.items()}}
        if not architecture:
            rows[name]["buffers"]={k:None if b is None else {"tensor":canonical(b),"persistent":k not in m._non_persistent_buffers_set} for k,b in m._buffers.items()}
    # Bind called helpers as well as module class bodies. Production source is
    # immutable for a run; live monkeypatch/hot-reload is not a supported model.
    root=Path(__file__).parent
    sources={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*.py')) if 'tests' not in p.relative_to(root).parts and 'tools' not in p.relative_to(root).parts}
    return {"version":EXECUTION_VERSION,"torch":torch.__version__,"production_sources":sources,
            "parameter_topology":topology,"modules":rows}


def validate_architecture(module):
    """Compare executable constants/classes to an independently cfg-built template.

    Parameter values and adaptation metadata are not architecture constants.
    Explicitly added buffers are bound in identities, not treated as architecture.
    """
    from .api import HumanEngine
    from .teacher.model import TeacherEncoder
    from .teacher.preparation import TeacherPreparation
    with torch.random.fork_rng(devices=[]):
        if isinstance(module,HumanEngine):reference=HumanEngine(module.cfg,copy.deepcopy(module.core.E.statistics))
        elif isinstance(module,TeacherPreparation):reference=TeacherPreparation(module.cfg,copy.deepcopy(module.physical_statistics))
        elif isinstance(module,TeacherEncoder):reference=TeacherEncoder(module.cfg)
        else:raise ValueError("No registered canonical architecture for "+type(module).__name__)
    # Compare dimensions and constants; dtype is allowed to follow the runtime.
    first=next(module.parameters(),None)
    if first is not None:reference.to(dtype=first.dtype)
    if content_hash(executable(module,architecture=True))!=content_hash(executable(reference,architecture=True)):
        raise ValueError("Actual executable architecture disagrees with declared config")


def storage_key(tensor):
    return (str(tensor.device),tensor.untyped_storage()._cdata)


def tensor_layout(tensor):
    return (storage_key(tensor),tuple(tensor.shape),tuple(tensor.stride()),tensor.storage_offset(),tensor.dtype)


def named_parameter_topology(module):
    """Describe the supported topology, rejecting all parameter aliasing.

    HumanEngine v0.1 has no intentional tied parameters. A Parameter registered
    under multiple paths, or distinct Parameters sharing a storage, can prevent
    ordinary in-place state loading from establishing independent named values.
    Such layouts are outside the supported continuation contract.
    """
    rows=[];objects={};storages={}
    for name,p in module.named_parameters(remove_duplicate=False):
        if p is None:continue
        if id(p) in objects:
            raise ValueError("Parameter registered under multiple scientific names: "+objects[id(p)]+", "+name)
        key=storage_key(p)
        if key in storages:
            raise ValueError("Shared/overlapping parameter storage unsupported: "+storages[key]+", "+name)
        objects[id(p)]=name;storages[key]=name
        rows.append({"name":name,"shape":list(p.shape),"dtype":str(p.dtype),
                     "stride":list(p.stride()),"storage_offset":int(p.storage_offset())})
    return {"contract":"independent-named-parameter-storage-v1","parameters":rows}


class SnapshotViolation(RuntimeError):pass


class ReadOnlyEvaluation(TorchDispatchMode):
    """Reject protected writes BEFORE execution, including .data/view aliases.

    Check actual module attributes before EVERY tensor operation, not just forward
    hooks (HE's custom .step/.window paths bypass nn.Module.__call__). Only fresh
    autograd graph inputs from this scope or authoritative leaves are accepted.
    """
    def __init__(self,parameters,modules):
        super().__init__()
        self.modules=tuple(modules)
        self.nodes=[m for root in modules for m in root.modules()]
        for m in self.nodes:check_supported_module(m)
        self.attrs=[copy.deepcopy(attributes(m)) for m in self.nodes]
        self.types=[type(m) for m in self.nodes]
        self.layouts=[(dict(m._parameters),dict(m._buffers),dict(m._modules),set(m._non_persistent_buffers_set)) for m in self.nodes]
        tensors=list(parameters)+[t for m in self.nodes for t in (*m._parameters.values(),*m._buffers.values()) if t is not None]
        self.tensors={id(t):t for t in tensors}
        self.values={key:t.detach().clone() for key,t in self.tensors.items()}
        self.flags={key:t.requires_grad for key,t in self.tensors.items()}
        self.storages={storage_key(t):key for key,t in self.tensors.items()}
        self.bindings={key:tensor_layout(t) for key,t in self.tensors.items()}
        self.produced={};self.failed=False
        self.check_metadata()

    def reject(self,reason):
        self.failed=True
        raise SnapshotViolation("Authoritative snapshot violation: "+reason)

    def check_metadata(self):
        if self.failed:self.reject("previous violation caught by evaluator")
        for m,attrs,kind,layout in zip(self.nodes,self.attrs,self.types,self.layouts):
            try:
                check_supported_module(m)
                actual_attributes=attributes(m)
            except ValueError as exc:self.reject(str(exc))
            if type(m) is not kind or not equal_state(actual_attributes,attrs):self.reject("executable attributes changed")
            for current,old in zip((m._parameters,m._buffers,m._modules),layout[:3]):
                if current.keys()!=old.keys() or any(current[k] is not old[k] for k in old):self.reject("registered object replaced")
            if m._non_persistent_buffers_set!=layout[3]:self.reject("buffer persistence changed")
        for key,t in self.tensors.items():
            if t.requires_grad!=self.flags[key] or tensor_layout(t)!=self.bindings[key]:self.reject("tensor binding/layout/trainability changed")
            if getattr(t,'_backward_hooks',None) or getattr(t,'_post_accumulate_grad_hooks',None):
                self.reject("unversioned parameter gradient hook")

    def owns(self,t):return id(t) in self.produced and self.produced[id(t)]() is t

    def require_fresh_loss(self,loss):
        if loss is not None and not self.owns(loss):self.reject("loss graph was constructed outside guarded evaluation")

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        self.check_metadata()
        for idx,a in enumerate(func._schema.arguments):
            if a.alias_info is not None and a.alias_info.is_write:
                value=args[idx] if idx<len(args) else kwargs.get(a.name)
                if any(isinstance(t,torch.Tensor) and storage_key(t) in self.storages for t in tree_leaves(value)):
                    self.reject("write to protected parameter/buffer through "+str(func))
        for t in tree_leaves((args,kwargs)):
            if not isinstance(t,torch.Tensor):continue
            key=self.storages.get(storage_key(t))
            if key is not None:
                if not torch.equal(self.tensors[key],self.values[key]):self.reject("protected tensor content changed before use")
            elif t.requires_grad and not self.owns(t):
                self.reject("foreign/prebuilt autograd graph operand")
        result=func(*args,**kwargs)
        for t in tree_leaves(result):
            if isinstance(t,torch.Tensor):self.produced[id(t)]=weakref.ref(t)
        return result
