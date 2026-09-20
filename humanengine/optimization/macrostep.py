from dataclasses import dataclass
import copy
import torch
from ..scientific_state import equal_state
from ..execution import ReadOnlyEvaluation
from ..transaction import TrainingTransaction,assert_usable
from .cagrad import weighted_cagrad
from ..diagnostics.gradients import flat_gradient,vector,observational


@dataclass
class FamilyEvaluation:
    core_loss: torch.Tensor | None
    private_losses: dict[str,torch.Tensor]


class Macrostep:
    """Evaluate every family/private derivative before ANY parameter update."""
    def __init__(self,core_parameters,private_groups,core_optimizer,private_optimizers,cfg,*,stateful_modules=()):
        self.core=tuple(core_parameters);self.private={k:tuple(v) for k,v in private_groups.items()}
        self.core_optimizer=core_optimizer;self.private_optimizers=private_optimizers;self.cfg=cfg
        all_parameters=[*self.core,*[p for group in self.private.values() for p in group]]
        if len({id(p) for p in all_parameters})!=len(all_parameters):raise ValueError("Core/private groups overlap")
        if set(private_optimizers)!=set(self.private):raise ValueError("Private optimizer groups mismatch")
        for name,params,opt in [("core",self.core,core_optimizer),*[(k,v,private_optimizers[k]) for k,v in self.private.items()]]:
            actual=[p for g in opt.param_groups for p in g["params"]]
            if len(actual)!=len(params) or {id(p) for p in actual}!={id(p) for p in params}:raise ValueError("Optimizer ownership mismatch: "+name)
        self.parameters=tuple(all_parameters);self.steps=0
        self.stateful_modules=tuple(stateful_modules)

    def step(self,closures,budgets,*,diagnostic=None,diagnostic_module=None,sampler=None):
        if set(closures)!=set(budgets) or "reserve" not in closures:raise ValueError("Incomplete active-family macrostep")
        names=sorted(k for k in closures if k!="reserve")+["reserve"]
        if not names[:-1]:raise ValueError("No API family")
        assert_usable(self,*self.stateful_modules,self.core_optimizer,*self.private_optimizers.values(),sampler)
        optimizers=[self.core_optimizer,*self.private_optimizers.values()]
        with TrainingTransaction(modules=self.stateful_modules,parameters=self.parameters,
                                 objects=(*optimizers,self,sampler)):
            return self._evaluate_and_commit(closures,budgets,names,sampler,diagnostic,diagnostic_module)

    def _evaluate_and_commit(self,closures,budgets,names,sampler,diagnostic,diagnostic_module):
        before=[p.detach().clone() for p in self.parameters]
        saved_grads=[None if p.grad is None else p.grad.clone() for p in self.parameters]
        optimizers=[self.core_optimizer,*self.private_optimizers.values()]
        saved_opts=copy.deepcopy([o.state_dict() for o in optimizers])
        saved_controller=copy.deepcopy(self.state_dict());saved_cfg=self.cfg
        saved_sampler=copy.deepcopy(sampler.state_dict()) if sampler is not None else None
        grads=[];private_grads={};pre={}
        for name in names:
            # Entire forward AND backward occur under the same authoritative
            # read-only state. A transient write cannot reach the forward at all.
            with ReadOnlyEvaluation(self.parameters,self.stateful_modules) as guard:
                evaluation=closures[name]()
                guard.require_fresh_loss(evaluation.core_loss)
                for loss in evaluation.private_losses.values():guard.require_fresh_loss(loss)
                if evaluation.core_loss is None:raise ValueError("NA family: "+name)
                if not torch.isfinite(evaluation.core_loss):raise FloatingPointError("Nonfinite family loss")
                pre[name]=float(evaluation.core_loss.detach())
                grads.append(flat_gradient(evaluation.core_loss,self.core).detach())
                for key,loss in evaluation.private_losses.items():
                    if key in private_grads or key not in self.private:raise ValueError("Private loss ownership conflict")
                    private_grads[key]=flat_gradient(loss,self.private[key]).detach()
                guard.check_metadata()
            changed=any(not torch.equal(p,v) for p,v in zip(self.parameters,before))
            changed=changed or not equal_state(saved_grads,[p.grad for p in self.parameters])
            changed=changed or not equal_state(saved_opts,[o.state_dict() for o in optimizers])
            changed=changed or self.cfg!=saved_cfg or not equal_state(saved_controller,self.state_dict())
            changed=changed or (sampler is not None and not equal_state(saved_sampler,sampler.state_dict()))
            changed=changed or any(not torch.equal(t,guard.values[k]) for k,t in guard.tensors.items())
            if changed:raise RuntimeError("Family closure mutated the same-parameter snapshot or training state")
        if set(private_grads)!=set(self.private):raise ValueError("Missing private objective")
        if any(not torch.isfinite(g).all() for g in private_grads.values()):raise FloatingPointError("Nonfinite private gradient")
        c=0. if len(names)-1==1 else self.cfg.cagrad_c
        solution=weighted_cagrad(grads,[budgets[n] for n in names],c,tolerance=self.cfg.cagrad_tolerance,max_iterations=self.cfg.cagrad_max_iterations)
        intended=solution.direction
        def install(params,grad):
            start=0
            for p in params:
                p.grad=grad[start:start+p.numel()].reshape_as(p).clone();start+=p.numel()
        install(self.core,intended)
        for key,group in self.private.items():install(group,private_grads[key])
        old_core=torch.cat([v.reshape(-1) for v in before[:len(self.core)]])
        self.core_optimizer.step()
        for optimizer in self.private_optimizers.values():optimizer.step()
        self.steps+=1;delta=vector(self.core)-old_core
        post=None
        if diagnostic is not None:
            if diagnostic_module is None:raise ValueError("Observational diagnostic module required")
            with observational(diagnostic_module,modules=self.stateful_modules,parameters=self.parameters,
                               objects=(*optimizers,self,sampler)),torch.no_grad():
                post={k:float(v) for k,v in diagnostic().items()}
        return {"families":names,"solution":solution,"gradients":grads,"actual_delta":delta,
                "g_dot_delta":[float(g@delta) for g in grads],"pre_losses":pre,"post_losses":post,
                "private_gradients":private_grads,"snapshot_consistent":True}

    def state_dict(self):return {"steps":self.steps,"route_version":self.cfg.route_version}
    def load_state_dict(self,value):
        if value["route_version"]!=self.cfg.route_version:raise ValueError("Controller route mismatch")
        self.steps=value["steps"]
