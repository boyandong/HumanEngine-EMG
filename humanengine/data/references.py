"""Fit explicit reference scales once on permitted training data, then freeze."""
from dataclasses import dataclass
import torch
from ..objectives.reductions import masked_error,latent_error


@dataclass(frozen=True)
class ReferenceArtifact:
    value: float
    floor: float
    version: str
    partition: str
    constant: tuple
    definition: str


def _check(partition,floor,version):
    import math
    if partition not in ("train","synthetic") or not math.isfinite(floor) or floor<=0 or not version:
        raise ValueError("Reference must be fixed from permitted training partition with a declared floor/version")


@torch.no_grad()
def regression_reference(target,valid,coordinate_scale,*,floor,version,partition):
    _check(partition,floor,version)
    if (coordinate_scale<=0).any() or not torch.isfinite(coordinate_scale).all():raise ValueError("Coordinate scale invalid")
    rows=target[valid]
    if not len(rows) or not torch.isfinite(rows).all():raise ValueError("No finite reference support")
    center=rows.median(0).values
    error=masked_error(center.expand_as(target)/coordinate_scale,target/coordinate_scale,valid,kind="l1")
    return ReferenceArtifact(float(error.value),floor,version,partition,tuple(center.tolist()),"train constant-median scaled L1")


@torch.no_grad()
def classification_reference(labels,valid,classes,*,floor,version,partition):
    _check(partition,floor,version);y=labels[valid]
    if not len(y) or (y<0).any() or (y>=classes).any():raise ValueError("Invalid class reference support")
    frequency=torch.bincount(y,minlength=classes).double()/len(y)
    error=-frequency[y].log().mean()
    return ReferenceArtifact(float(error),floor,version,partition,tuple(frequency.tolist()),"train prevalence cross entropy")


@torch.no_grad()
def residual_reference(targets,valid,*,floor,version,partition):
    _check(partition,floor,version)
    error=latent_error(torch.zeros_like(targets),targets,valid)
    if error.value is None:raise ValueError("No teacher reference support")
    return ReferenceArtifact(float(error.value),floor,version,partition,(),"equal scale-layer Huber of normalized teacher targets versus zero")
