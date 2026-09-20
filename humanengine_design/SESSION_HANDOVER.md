# HumanEngine-EMG architecture handover

Date: 2026-09-19. Lane: architecture only. Status: design delivered; no HumanEngine implementation or training performed.

## Objective

Design a theoretically justified, diagnosable HumanEngine-EMG v0.1 under partial
supervision, then hand implementation to a lower-cost coding model. The latest user
direction prioritizes theoretical architecture; channel count, sampling rate and
environment are deferred implementation concerns.

Primary deliverable: [HUMANENGINE_EMG_V0_1_ARCHITECTURE.md](HUMANENGINE_EMG_V0_1_ARCHITECTURE.md).
It contains A–O, precise routing, method decisions, diagnostics, implementation
interfaces and final nine recommendations. The new root AGENTS.md carries the
requested cross-session Session Hygiene Protocol.

## Frozen decisions

Existing emg2pose preprocessing/validity, canonical16, VEMG2Pose, DB9, S3/S4/S5/S6
and validated adapters are frozen. Runtime input is EMG only. Pose is privileged
training supervision and an explicit API, not the definition of HumanState.
Do not treat stage labels as frame-level gesture truth. No invented force,
activation, contact, fatigue, intent, or world-pose ground truth.

The NEW architecture below is a proposal awaiting scientific review, not an
empirically validated or already approved implementation:

- Shared causal local frontend, branch before recurrence into S256 and R128.
- H=[hS,hR]; complete runtime state additionally includes cell states and caches.
- Current pose→S; future pose→H; signal summary family→R.
- Signal family = masked recovery + 0.1 observed-summary anchor, using the same M
  decoder. The observed term directly pressures preservation of selected actual
  measurements; masked-only learning could retain just predictable conditional means.
- VICReg→temporary projection(S), weak noise only; fixed pose-relational geometry
  →separate projection(S), no learned teacher pretraining.
- Five objective families jointly from step 0; weights 1/.5/.2/.1/.05, with observed
  anchor inside signal at .1. These are initial engineering choices, not optimality claims.
- Personalization: frozen universal core, input affine and rank-8 feature adapter;
  identity without labeled calibration. No generic subject-adversarial removal.
- No independence/orthogonality, force-preservation, complete-belief, or intent claim.

## Verified state

- Project root: C:\Users\董伯言\Desktop\fingers\emg2pose_handstate.
- Root is not a Git repo; only baseline/emg2pose has Git coverage. Baseline HEAD
  5f6f62b1a0a08426adffe55900842e75a8adb38c; read-only status was clean.
- Existing regression checkpoint architecture was loaded read-only on CPU: TDS
  frontend, 2×512 LSTM with previous-pose feedback, 40 outputs, 5,980,328 parameters.
- Actual frontend stride is 80; native features are 25 Hz, then interpolated to
  50 Hz and eventually data rate. Batch interpolation is not a verified strict
  zero-lookahead streaming contract. Preserve B0; new HE needs its own prefix tests.
- Official mini: 30 HDF5 files, 4,218,756 rows, one user, one HDF5 session ID,
  15 stages × two hands. It cannot establish held-out-user generalization.
- Global metadata has 25,253 rows/193 users/751 sessions; this is not proof that
  all full data exist locally. full/ and priority_pose_subset/ had no HDF5.
- Old data/emg2pose_dataset_mini/ is incomplete and includes a read failure; use
  official/emg2pose_dataset_mini for the limited interface checks.
- Frozen canonical conversion detaches tensors to NumPy; new differentiable view,
  if needed, must reuse resolved indices and match the frozen adapter, not edit it.
- Current docs/CONFIRMATORY_RESULTS.md supersedes old pre-remediation audit status:
  S3 records completion of its one-shot 77-subject run; old holdout is outcome-exposed.
  This session read the report and did not reproduce those classification metrics.
- Source/config/document preservation is checked in EVIDENCE_MANIFEST.json;
  scope of hashes is explicit. No full raw-dataset CRC audit was performed here.

## Scientific evidence

Primary review includes emg2pose, CPEP, EMBridge official abstract, GenENet, VICReg,
TS2Vec/CPC, data2vec/JEPA, relational KD and adjacent representation methods.
Exact URLs and primary-vs-background verification scope are in Appendix 2.

Important findings: GenENet reconstructs RMS maps, not arbitrary raw EMG; its force
result includes downstream force supervision. CPEP behavior-group labels cannot
be imported as gesture truth. KinEMbed provides counterevidence against assuming
soft alignment universally improves continuous EMG decoding. The proposed
fixed-geometry regularizer is RKD-inspired, not a CPEP/EMBridge reproduction.

Independent architecture review found and the design incorporates: safe relational
distance denominators, VICReg anchor support, nonzero-down/zero-up adapter
initialization, stronger pose-history controls, separate runtime memory vs H,
and the masked-prediction versus observed-information preservation distinction.

## Unknown / contaminated state

- No HumanEngine accuracy, force retention, latency, optimal dimension/weights,
  cross-user benefit or calibration benefit has been measured.
- Unknown APIs may require information outside the chosen summaries and H capacity.
- Pose is filtered/interpolated; tiny-horizon forecasting cannot establish motor intent.
- EMBridge full equations were not verified due to a website verification barrier.
- Exact raw-device preprocessing and full-data remote availability are unresolved.
- Historical S3/S5 outcome exposure remains; S6 blinding remains protected. No hidden
  labels/evaluator answers were read for this design, no model endpoints computed.
- The data-audit helper ended with a known HTTP 429 rate-limit failure after delivering
  read-only findings. It was not retried. No parent host crash or unknown mutation
  outcome occurred. Distinguish this known helper failure from a host recovery incident.
- No experiment results were generated in this architecture session, so there is
  no HE experiment to treat as validated or resume.

## External runtime state — must be reverified

No training, acquisition, annotation service, remote GPU process or persistent
background job was launched by this design session. Existing external processes
belonging to other lanes were not audited; their current state is UNKNOWN.
Do not infer a server/job is running from old handovers. Before remote execution,
record host, PID/job ID, config hash, run directory, logs, checkpoint and stop/resume
procedure in the implementation lane's manifest.

## Worktree state

New design-only files are under humanengine_design/. New root AGENTS.md records
governance. No isolated Git worktree was created because root is not a Git repo;
no commit or branch is claimed. Existing frozen files were not intentionally edited.
Do not initialize/restructure project Git as a side effect of implementing the model.

## Next actions

1. Review the theoretical choices, particularly signal targets/observed anchor,
   before considering architecture approved. Further theoretical discussion can
   operate on this document rather than an ever-longer conversation.
2. For implementation, open a fresh session with the name below and read root
   AGENTS.md, this handover and the architecture specification. Do not make a new
   method survey or silently change routing.
3. Implement phases N1–N2 in a NEW humanengine/ lane. Start with contracts, causal
   state tests, masks, gradient routing and target-leakage tests.
4. Only after user authorization for training, audited data/splits and a new runtime
   preflight, prepare the first remote HE_FULL pilot in N3. It is not authorized by
   the current architecture-only task.
5. Choose 1–3 counterfactuals from diagnostics. Most likely first is FULL−geometry;
   distinguish FULL−masked, FULL−observed and FULL−signal.

## Do NOT

- Do not implement/train in the old design task merely because it remains open.
- Do not edit frozen baseline, adapters, scientific masks, S3/S4/S5/S6 or old handovers.
- Do not use mini as cross-user evidence or resample windows across held-out groups.
- Do not leak clean summaries/state into masked input, or future EMG into past output.
- Do not turn pose geometry into full-H alignment or enforce residual orthogonality.
- Do not claim fixed frequency/RMS summaries preserve all unknown physiology.
- Do not open HDF5 via Path.resolve() that dereferences C:\emg2pose to the Unicode path.
- Do not treat prior dependency inventories or external process states as current.

## Recommended next session

`HUMANENGINE-IMPLEMENTATION-20260919-HECORE-V01`

Update YYYYMMDD to the actual start date. This is a design-to-implementation phase
boundary, so a fresh session is recommended even without a crash. Preserve this
session as the architecture decision record. If a future actual incident occurs,
use HUMANENGINE-INCIDENT-YYYYMMDD-<PURPOSE>, retire that incident session, and
reconcile state before a successor resumes work.
