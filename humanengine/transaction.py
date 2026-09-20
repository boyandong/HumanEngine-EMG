"""Exact registered in-process rollback without contaminated public loaders."""
import copy
import random
import numpy as np
import torch


class UnusableTrainingState(RuntimeError):pass


def assert_usable(*objects):
    if any(getattr(obj,'_he_unusable',False) for obj in objects if obj is not None):
        raise UnusableTrainingState("Training state is unusable after rollback failure; restart from verified artifacts")


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    # Attempt every independent RNG restoration even if one fails.
    errors=[]
    calls=[lambda:random.setstate(state['python']),lambda:np.random.set_state(state['numpy']),lambda:torch.set_rng_state(state['torch'])]
    if state['cuda'] is not None:calls.append(lambda:torch.cuda.set_rng_state_all(state['cuda']))
    for call in calls:
        try:call()
        except Exception as exc:errors.append(exc)
    if errors:raise RuntimeError("RNG restoration failed") from errors[0]


class TrainingTransaction:
    def __init__(self,*,modules=(),parameters=(),objects=()):
        self.roots=tuple(modules);self.objects=tuple(o for o in objects if o is not None)
        assert_usable(*self.roots,*self.objects)
        nodes={id(m):m for root in modules for m in root.modules()}
        nodes.update({id(o):o for o in self.objects})
        tensors={id(p):p for p in parameters}
        for root in modules:
            for m in root.modules():
                for t in (*m._parameters.values(),*m._buffers.values()):
                    if t is not None:tensors[id(t)]=t
        self.tensors=tensors
        self.aliases={key:t.detach() for key,t in tensors.items()}
        self.grad_objects={key:t.grad for key,t in tensors.items()}
        self.tensor_hooks={key:(copy.copy(getattr(t,'_backward_hooks',None)),
                               copy.copy(getattr(t,'_post_accumulate_grad_hooks',None))) for key,t in tensors.items()}
        self.tensor_states={key:(t.detach().clone(),t.requires_grad,
                                None if t.grad is None else t.grad.detach().clone()) for key,t in tensors.items()}
        self.memo={**nodes,**tensors}
        self.dictionaries=[(o,type(o),copy.deepcopy(vars(o),dict(self.memo))) for o in nodes.values()]
        self.rng=rng_state()

    def rollback(self):
        errors=[]
        # Restore structural metadata first (including sampler manifests), without
        # invoking load_state_dict hooks or validators on contaminated objects.
        for obj,kind,state in self.dictionaries:
            try:
                if type(obj) is not kind:obj.__class__=kind
                obj.__dict__.clear();obj.__dict__.update(copy.deepcopy(state,dict(self.memo)))
            except Exception as exc:errors.append(exc)
        for key,t in self.tensors.items():
            try:
                value,flag,grad=self.tensor_states[key]
                with torch.no_grad():
                    t.data=self.aliases[key]
                    if not torch.equal(t,value):t.copy_(value)
                    if grad is not None:self.grad_objects[key].copy_(grad)
                t.requires_grad_(flag);t.grad=self.grad_objects[key]
                t._backward_hooks,t._post_accumulate_grad_hooks=self.tensor_hooks[key]
                if not torch.equal(t,value) or t.requires_grad!=flag:raise RuntimeError("Tensor rollback verification failed")
            except Exception as exc:errors.append(exc)
        try:restore_rng(self.rng)
        except Exception as exc:errors.append(exc)
        if errors:
            for obj in (*self.roots,*self.objects):obj._he_unusable=True
            raise UnusableTrainingState("Rollback incomplete; controller/run poisoned; restart required") from errors[0]

    def __enter__(self):return self

    def __exit__(self,kind,exc,traceback):
        if kind is not None:self.rollback()
        return False


class ObservationTransaction(TrainingTransaction):
    """Always restore the captured continuation, including RNG, on success."""
    def __exit__(self,kind,exc,traceback):
        self.rollback()
        return False
