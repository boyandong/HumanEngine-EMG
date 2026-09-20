"""Single canonical projection of the evaluation protocol, used at every gate."""
from dataclasses import asdict,is_dataclass
import copy

TEACHER_CERTIFICATE_VERSION="teacher-artifact-certificate-v3"
PROTOCOL_FIELDS=("evaluation_version","evaluation_config","data_manifest_hash","checkpoint_identity",
                 "partition","partition_identity","budget_identity","checkpoint_step","preregistered_step",
                 "raw_std_threshold","within_std_threshold","amplitude_threshold")


def evaluation_protocol(health):
    value=asdict(health) if is_dataclass(health) else health
    if any(k not in value for k in PROTOCOL_FIELDS):raise ValueError("Incomplete health evaluation protocol")
    for key in ("evaluation_version","evaluation_config","data_manifest_hash","checkpoint_identity","partition_identity","budget_identity"):
        if not value[key]:raise ValueError("Teacher artifact evaluation provenance required")
    return copy.deepcopy({key:value[key] for key in PROTOCOL_FIELDS})
