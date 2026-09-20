import torch
from torch.nn import functional as F
from ..contracts import LossTerm


def masked_error(pred,target,valid,kind="huber",delta=1.):
    if pred.shape!=target.shape:raise ValueError("Prediction/target shape mismatch")
    valid=valid.bool()
    while valid.ndim<pred.ndim:valid=valid.unsqueeze(-1)
    valid=valid.expand_as(pred)
    count=int(valid.sum())
    if not count:return LossTerm.na("no valid targets")
    p,t=pred[valid].float(),target.detach()[valid].float()
    if not torch.isfinite(p).all() or not torch.isfinite(t).all():raise ValueError("Non-finite valid target/prediction")
    errors=(p-t).abs() if kind=="l1" else F.huber_loss(p,t,reduction="none",delta=delta)
    numerator=errors.sum()
    return LossTerm(numerator,count,numerator/count)


def equal_mean(terms):
    active=[v for v in terms if v.value is not None]
    if not active:return LossTerm.na("no supported groups")
    value=torch.stack([v.value for v in active]).mean()
    return LossTerm(sum(v.numerator for v in active),sum(v.count for v in active),value)


def latent_error(pred,target,valid,delta=1.):
    if pred.ndim!=5 or pred.shape[-3:-1]!=(3,2):raise ValueError("Expected B,N,3,2,D")
    return equal_mean([masked_error(pred[:,:,k,l],target[:,:,k,l],valid[:,:,k,l],delta=delta)
                       for k in range(3) for l in range(2)])


def pose_losses(current,future,pose,future_pose,valid,future_valid):
    return {"current":masked_error(current,pose,valid,kind="l1"),
            "future":equal_mean([masked_error(future[:,:,i],future_pose[:,:,i],future_valid[:,:,i],kind="l1")
                                  for i in range(future.shape[2])])}
