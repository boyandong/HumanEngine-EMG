# Project governance and session hygiene

## Long-term source of truth

The repository and its scientific records are the long-term source of truth. A
session is a temporary workbench. Read the applicable scientific specifications,
experiment manifests and lane-specific SESSION_HANDOVER.md before new work.
Do not rely on a long conversation as the sole record of a scientific decision.

Existing baseline, preprocessing, valid-pose conventions, canonical16, DB9 decoder,
and Session 3/4/5/6 artifacts are frozen unless the user explicitly authorizes a
specific change. Keep HumanEngine work in a new lane. Preserve S6 blinding; do not
load hidden/evaluator labels or compute model endpoints on annotation windows.

Current HumanEngine design: humanengine_design/HUMANENGINE_EMG_V0_1_ARCHITECTURE.md.
Current design handover: humanengine_design/SESSION_HANDOVER.md.
The design is a proposal, not approval to implement or train.

Focused theory addendum (2026-09-19):
humanengine_design/FOCUSED_REVIEW_MULTI_API_RESIDUAL.md.
Latest focused-review handover: humanengine_design/FOCUSED_REVIEW_HANDOVER.md.
Read both the original design and this addendum before implementation. The
addendum explicitly identifies proposed routing/target changes; the original
design and its handover remain historical comparison records, not overwritten.

## Session Hygiene Protocol — actively enforce across sessions

Actively assess whether the session should retire. Notify the user around 500K
context tokens and strongly recommend handoff at 800K+. These are user-specified
heuristics, not assertions about any model's context capacity. Also assess handoff
after two or more consecutive tool interruptions/unknown outcomes, a host/tool
crash and recovery, repeated retries or temporary wrapper construction, a new
project phase, or complex remote/background execution.

After an actual incident, preserve the old session as an archive/forensic record;
do not continue task execution in the incident session. Reconcile known and unknown
outcomes in a handover. A known read-only error with a determined outcome is not
itself evidence of an unknown remote mutation; record the distinction explicitly.

Before changing sessions, create/update the current lane's handover with:
Objective; Frozen decisions; Verified state; Scientific evidence;
Unknown/contaminated state; External runtime state (must be reverified in the new
session); Worktree state; Next actions; Do NOT; recommended new session name.
Do not overwrite another frozen lane's handover.

Use <PROJECT>-<PHASE>-<YYYYMMDD>-<PURPOSE>; incident sessions use
<PROJECT>-INCIDENT-<YYYYMMDD>-<PURPOSE>. At a design-to-implementation transition,
recommend a fresh implementation session and carry the approved specification.

## Provenance and local data access

The project root was not a Git repository on 2026-09-19; baseline/emg2pose is a
nested Git repository. Recheck before assuming commit coverage. Use content hashes
for new design provenance until the user chooses project-wide version control.
On this Windows setup, HDF5 access uses C:\emg2pose. Do not resolve that junction
back into the non-ASCII path before passing a filename to h5py.
