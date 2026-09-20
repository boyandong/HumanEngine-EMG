from dataclasses import dataclass
import copy
import torch
from ..objectives.reductions import masked_error


@dataclass(frozen=True)
class ReplayManifest:
    family_id: str
    family_version: str
    training_source_ids: tuple[str,...]
    user_session_recording_coverage: tuple[tuple[str,str,str],...]
    data_hash: str
    partition: str = "train"

    def __post_init__(self):
        if self.partition!="train" or not self.training_source_ids or not self.data_hash:
            raise ValueError("Replay requires representative old-domain TRAINING sources")


@dataclass(frozen=True)
class OldAPIArtifact:
    model_hash: str
    family_id: str
    family_version: str
    output_definition: str
    route_version: str
    adapter_contract: str
    old_domain_manifest_hash: str


def freeze_old_api(model,artifact):
    if not all(vars(artifact).values()):raise ValueError("Incomplete old API artifact")
    return copy.deepcopy(model).eval().requires_grad_(False)


def distillation_loss(new_output,old_output,valid,*,kind="continuous",scale=None,temperature=None):
    if kind=="continuous":
        if scale is None or (scale<=0).any():raise ValueError("Fixed output scale required")
        return masked_error(new_output/scale,old_output.detach()/scale,valid)
    if kind!="categorical" or temperature is None or temperature<=0:raise ValueError("Explicit distillation likelihood required")
    from ..contracts import LossTerm
    x=new_output[valid];y=old_output.detach()[valid]
    if not len(x):return LossTerm.na("no distillation targets")
    error=torch.nn.functional.kl_div(torch.log_softmax(x/temperature,-1),torch.softmax(y/temperature,-1),reduction="sum")*temperature**2
    return LossTerm(error,len(x),error/len(x))


@dataclass
class Onboarding:
    family_id: str
    status: str = "HEAD_ONLY"

    def begin(self,model):
        if self.family_id not in model.api_heads:raise ValueError("Register and instantiate the new private head first")
        model.requires_grad_(False);model.api_heads[self.family_id].requires_grad_(True)

    def verify(self,*,finite_outputs,error,constant_reference):
        import math
        if not finite_outputs or not math.isfinite(error) or not math.isfinite(constant_reference) or not error<constant_reference:
            raise ValueError("New API not ready to enter joint training")
        self.status="READY_FOR_JOINT"

    def enable_joint(self,model):
        if self.status!="READY_FOR_JOINT":raise ValueError("Head-only health gate not passed")
        model.universal_mode();self.status="JOINT"
