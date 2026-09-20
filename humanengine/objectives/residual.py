from ..contracts import LossBundle
from .reductions import latent_error,masked_error
from .regularizers import variance_floor


def residual_losses(heads,r_masked,r_observed,targets,target_valid,physical,physical_valid,meta,cfg):
    shape=(*r_observed.shape[:-1],3,2,cfg.teacher_hidden)
    pred_m=heads.P_R(r_masked).reshape(shape);pred_o=heads.P_R(r_observed).reshape(shape)
    terms={"masked":latent_error(pred_m,targets,target_valid,cfg.huber_delta),
           "observed":latent_error(pred_o,targets,target_valid,cfg.huber_delta),
           "physical":masked_error(heads.M_A(r_observed),physical,physical_valid,delta=cfg.huber_delta),
           "variance":variance_floor(r_observed,meta,cfg)}
    total=None
    if all(terms[k].value is not None for k in ("masked","observed","physical")):
        total=terms["masked"].value+cfg.observed_weight*terms["observed"].value+cfg.physical_weight*terms["physical"].value
        if terms["variance"].value is not None:total=total+cfg.variance_weight*terms["variance"].value
    per_target={}
    for k in range(3):
        for l in range(2):
            sub=masked_error(pred_m[:,:,k,l],targets[:,:,k,l],target_valid[:,:,k,l],delta=cfg.huber_delta)
            per_target[f"scale{k}/layer{l}"]={"value":None if sub.value is None else float(sub.value.detach()),"valid_elements":sub.count,"status":sub.status}
    return LossBundle(terms,total,cfg.route_version,{"variance_skipped":terms["variance"].value is None,
                                                   "masked_targets":per_target,"target_valid_counts":target_valid.sum((0,1)).tolist()})
