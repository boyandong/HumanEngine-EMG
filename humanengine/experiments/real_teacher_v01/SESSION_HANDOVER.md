# REAL EMG TEACHER v0.1 — server-first preparation

Date: 2026-09-20, Asia/Shanghai.
Execution status: REMOTE DATA ACQUISITION IN PROGRESS. Scientific outcome: NOT EVALUATED.
This is a preparation checkpoint, not the final teacher-validation report.

## Objective
Validate only the independent EMG teacher on real data with frozen provenance,
TRAIN-only statistics, fixed budget and preregistered EMG-only health criteria.
Do not train HumanEngine S/R/H, API heads or HE_FULL.

## Authority and frozen decisions
The current user task explicitly authorizes real teacher preparation/training,
superseding the historical implementation-only authorization in lane governance.
The user's server-first clarification is authoritative: full data need not be local.
Local artifacts may comprise code, metadata, manifests, small smoke samples and
experiment records. Prefer an authorized server filesystem/object store.

Preserve causal frontend, two recurrent layers of 128, independent student/EMA,
0.1/0.5/2.0-second contexts and all six targets; same physical time mask, approved
losses, train-only statistics, fixed-budget checkpoint selection. No architecture
rescue, downstream-label selection, final-test use or S6 hidden-label access.

## Verified state
- Final acceptance original located and read:
  `D:/Temp/humanengine_v01_final_acceptance/FINAL_ACCEPTANCE_REPORT.md`.
  Verdict PASS; INFRASTRUCTURE v0.1 ACCEPTED / SAFE TO BEGIN REAL TEACHER VALIDATION.
  This supersedes the older kernel handover's pending-audit status. No general
  infrastructure audit was reopened or rerun.
- Actual teacher preparation/model/certificate/config and variance support code read.
  Full recipe requires 64 separated anchors, 8 recordings and 4 users for variance.
  Do not reduce those gates merely to fit a single-user mini corpus.
- `C:/emg2pose` is the project junction. Dataset-directory file inventory found
  30 HDF5 files in official mini and 18 in the old mini copy. This was a file
  inventory, not a new EMG schema/content verification. Local absence is not a gate.
- Local runtime import check: Python 3.11.16, torch 2.4.1+cpu, h5py 3.12.1,
  scipy 1.14.1. No training or smoke update executed.
- Project data-access records still describe awaiting a server path;
  `/data/emg2pose` in the acquisition runbook is an example, not a verified mount.

## Remote discovery / external runtime — reverify
UPDATE: user authorized password login to the current westc:23229 endpoint.
Login succeeded; credential was entered through SSH's non-echoing prompt and not
saved in project files. Host autodl-container-ae8b44827a-def944fb; RTX 5090 32607 MiB;
50 GB empty data disk initially; Python 3.12.3, torch 2.8.0+cu128. No preexisting
dataset found in the checked standard mounts. Created dedicated
`/root/autodl-tmp/he_teacher_v01/{code,raw,records}`; installed h5py3.12.1/scipy1.14.1.

Official S3 HEAD verified metadata ETag c3f48fd51576fa7752d455f2751c8b99, matching
the local metadata MD5 freshly checked; full archive ETag
62df27ac9a04800702061bac0ac1af32-6897 and 462824048640 bytes match the completed index.
ACQUISITION_PLAN.json selected 8 official TRAIN users and 4 official VAL users,
one session/user, two bilateral source recordings/session: 48 files, 871782416 bytes.
All official test recordings and entire mini user d387095792 excluded. Membership
chosen using hashed IDs, never signal outcomes or stage labels. Full corpus is not needed.

Acquisition PID1736 was intentionally stopped with SIGINT after eight completed
files and a slow ninth request. PID absence and KeyboardInterrupt were confirmed.
This is a controlled transfer stop with known outcome, not a host/tool crash.
The partial file is preserved as .partial.interrupted-v1. Revised acquisition uses
1 MiB range segments, bounded retries and four concurrent complete-file workers,
reverifying completed files. Selection and source version unchanged. Current PID2278
(reverify); logs acquisition.log and acquisition_v2.log. No training started.

CUDA preflight found default cuDNN LSTM shared weight storage conflicts with the
unchanged v4 checkpoint no-alias contract. Zero-step native CUDA preflight with
cudnn.enabled=False passes. Freeze this execution setting; do not weaken checkpoint
rules. Full real-input numerical/continuation smoke remains mandatory before training.
CPU and CUDA backend evidence is not a general extension of infrastructure acceptance.

The following paragraphs preserve earlier discovery chronology; authentication is
now resolved as described above.

Current user supplied `ssh -p 23229 root@connect.westc.seetacloud.com`.
This endpoint responded. First strict-host-key attempt rejected an unknown host;
the next used OpenSSH accept-new (never replaced a mismatched existing key) and
added the new host to local known_hosts. Authentication using the existing
he_5090 key was rejected: `Permission denied (publickey,password)`.
No remote shell command executed and no remote data root is verified.
Pending user action: install he_5090.pub or provide an already-authorized local
identity path/SSH alias. Do not ask for private-key contents or store passwords.

Historical candidate found in another repository:
`C:/Users/董伯言/Desktop/EMG-project-hla-emg/docs/research_reviews/SESSION_HANDOVER.md`
and `gpu_conformance_status.md`:
- endpoint: root@connect.westb.seetacloud.com, port 39789;
- existing credential file: ~/.ssh/he_5090 (contents never read);
- historical code/data: /root/autodl-tmp/he/repo and /root/autodl-tmp/he/data.

One read-only SSH attempt used BatchMode, strict host-key checking and a 12-second
connection timeout. It exited 1 with `banner exchange: Connection to UNKNOWN port
-1: Connection refused`. The failure does not establish whether the host or any
data still exists. No remote command output was received; no write was requested.
Historical hardware, data/schema, runtime and ownership remain UNVERIFIED.
Other saved SSH aliases do not establish EMG-project authorization and were not probed.
No training/download/background job was launched by this task. Do not resume or
kill old project jobs. Obtain a current endpoint/data root, then inspect fresh.

## Scientific evidence and unknowns
No real teacher result, statistics, selected checkpoint or certificate exists from
this task. Do not label missing access UNHEALTHY or fabricate an INVALID EXPERIMENT
from an experiment that has not run. The requested final categories apply after
an interpretable execution or an actual experiment-invalidating failure.

Source acquisition/version, remote complete-file inventory, split authority,
sampling/channel semantics, invalid intervals, cohort support, prior S5/S6 use,
hardware throughput and training budget remain unresolved.

Inspection deviation: an overly broad text search for remote references matched
numeric `5090` substrings in non-hidden JSON, returning unrelated pose/annotation
payload text. Hidden/evaluator directories were excluded; no hidden answer file
was intentionally opened, but no claim of zero label-text exposure is made.
No returned values were used for modeling, thresholds, split or checkpoint decisions.
Subsequent discovery uses bounded documentation/config allowlists and precise
endpoint patterns. Keep teacher data loaders strictly EMG/time-field allowlisted.

## Worktree state
Root has no project-wide Git coverage per governance; no Git initialization done.
Nested baseline/emg2pose checked clean at
5f6f62b1a0a08426adffe55900842e75a8adb38c. Current additions are confined to this
experiment directory. Historical infrastructure evidence and source are unchanged
by this task. Source/document hashes are recorded separately.

## Next actions
1. Verify the current owned acquisition PID/log; finish the exact selected complete
   files, then freeze DATA_MANIFEST.json. Authentication and source location resolved.
2. Remotely inspect metadata/schema using only EMG, time and allowed identity fields;
   establish split/exposure boundaries before reading training payloads.
3. Freeze explicit complete recordings/shards and immutable hashes/version IDs;
   refuse partial uploads. New arrivals require another manifest/run version.
4. Freeze train/dev/test membership and exclusions. No synchronized recordings or
   adjacent windows may cross independent partitions. Preserve final-test/S6 exclusions.
5. Fit physical/input statistics only from TRAIN. Running latent statistics use
   registered TRAIN batches; final frozen latent statistics need a separate TRAIN
   pass after the fixed-budget encoder exists. Do not fabricate them beforehand.
6. Run a small real-data smoke check for schema, finite forward/backward, actual masks,
   six targets, resume and throughput. Treat it only as execution evidence.
7. Preregister practical seed/optimizer/steps/checkpoint schedule and quantitative
   per-group health gates before scientific training; freeze all input identities.
8. Execute on the server, evaluate TRAIN/dev only, export certificate only on approval,
   synchronize hashes/manifests/results back here, and issue the requested 25-part report.

## Do NOT
No local full-corpus download; no automatic full remote download just because an
acquisition script exists; no mutable growing training manifest; no partial file
consumption; no invented subjects/splits/measurements; no old-project representation
or normalization reuse without compatibility/provenance checks; no HE training.

## Session health
One intentional user interruption followed already completed read-only commands;
no unknown training outcome or crash is known. The SSH refusal is a known connection
failure, not an incident requiring retirement. Continue this session after the endpoint
is supplied unless session-health triggers arise. Reassess when remote orchestration
becomes complex. Recommended handoff name if needed:
HUMANENGINE-REAL-TEACHER-20260920-SERVER-PREFLIGHT.

After actual teacher acceptance, stop and recommend
HUMANENGINE-TEACHER-TO-RESIDUAL-PILOT-V01; otherwise diagnose without automatic rescue.
