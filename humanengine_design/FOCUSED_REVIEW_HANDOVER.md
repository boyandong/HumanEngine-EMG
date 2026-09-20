# HumanEngine focused theory review — handover

Date: 2026-09-19. Status: focused review delivered as a proposal; no model implementation or training.

## Objective

Resolve two theoretical questions against the existing v0.1 proposal: how pose
becomes one API family among many, and whether independent EMG latent targets
should replace RMS/band summaries as residual's primary objective. Channel count,
sampling frequency and environment are deferred by the user's explicit direction.

Read `FOCUSED_REVIEW_MULTI_API_RESIDUAL.md` together with
`HUMANENGINE_EMG_V0_1_ARCHITECTURE.md`. The focused addendum covers all ten requested
outputs and answers the nine Q1 and ten Q2 subquestions explicitly. It is not
permission to implement or start teacher/HE training.

## Frozen decisions

Existing baseline, preprocessing, official validity, canonical16, DB9, validated
adapters and S3–S6 artifacts remain frozen. Protect S6 blinding; no hidden/evaluator
labels or model endpoints on annotation windows. Runtime remains EMG only.

Recommended choices awaiting adoption:

- Retain shared causal frontend, pre-recurrence split S256/R128, E amplitude path,
  H=[hS,hR], separate runtime cells/caches, and small affine/rank-8 personalization.
- API-family budgets/sampling; stable acquisition IDs prevent dataset duplication
  from increasing influence. Current/future/geometry pose share one family.
- Equal API base budget 0.8/K; representation reserve 0.2. Fixed loss reference
  scales; missing labels are NA with valid denominators and exposure accounting.
- Small-K weighted-base CAGrad on the single complete core parameter vector,
  c=0 while only pose exists, c=.25 when at least two real API families exist.
  This is a proposed extension; no per-task performance or convergence guarantee
  is imported into routed/Adam training. Fixed weights are the primary control.
- All API readouts have private projectors on [S,stopgrad(R)]. This blocks API
  gradients to R recurrence, not F-driven changes of R. It also removes the old
  direct future-pose-to-R supervision. The local routed gradient is a surrogate,
  not the full derivative of the next actual API forward after F changes.
- Prepare replay plus frozen old-output teacher interfaces now; activate when
  APIs arrive sequentially. Old-API teacher and independent EMG teacher differ.
- Prepare a separately initialized EMG-only SSL teacher with its own internal EMA,
  then freeze it for HE. No shared parameters, optimizer, stats, HE-EMA or label-
  based checkpoint selection. This deliberately changes the original no-teacher-
  pretraining lifecycle. Teacher cost belongs in comparison budgets.
- R target: three causal context scales (.1/.5/2 seconds), two teacher recurrent
  layers; masked latent + .1 observed latent + .05 observed physical summaries
  + .01 direct R variance. No default R covariance or extra future-latent loss.
- Existing weak-noise VICReg remains on S projection, under reserve budget;
  geometry remains on its separate S projection, under pose-family budget.

Numbers are initial research settings, not empirical optima or physiology facts.
Independent targets are a candidate upgrade, not a demonstrated improvement.

## Verified state

- Project: C:\Users\董伯言\Desktop\fingers\emg2pose_handstate.
- Root is still not a Git repository; no Git initialization or worktree creation.
- Nested baseline/emg2pose read-only status is clean; HEAD remains
  5f6f62b1a0a08426adffe55900842e75a8adb38c.
- All 309 protected files in the original EVIDENCE_MANIFEST.json matched their
  prior hashes; all five official checkpoint hashes matched. This checks only
  that recorded scope, not raw-dataset CRC or every file in the project.
- Original architecture, original SESSION_HANDOVER and original evidence manifest
  are preserved. Root AGENTS.md received only addendum/handover pointers; its prior
  hash and new hash are recorded in FOCUSED_REVIEW_EVIDENCE.json.
- No teacher, HumanEngine, probe, training, acquisition or evaluation code was
  implemented/run. Document content and frozen hashes were checked.

## Scientific evidence

Focused source review verified FAMO and CAGrad methods, data2vec target/EMA and
collapse details; official abstracts cover PCGrad, uncertainty weighting,
Nash-MTL, partial-label MTL, DER, LwF, I-JEPA, InfoMin, DAE and PSR. GradNorm full-
text observations came from the research helper. Prior architecture evidence for
CPC/TS2Vec/VICReg/GenENet is reused with its original scope. Exact links and reading
levels are in section 9; no literature benchmark is represented as a HE result.

An independent bounded consistency review highlighted and the draft addresses:
stop-gradient versus actual forward drift, CAGrad surrogate/optimizer limits,
fixed versus dynamic normalization, loss budget versus performance guarantee,
loss of direct future-R supervision, and teacher-only probe necessity.

## Unknown / contaminated state

- No measured multi-API gains, force/contact/fatigue retention, teacher quality,
  model convergence, runtime performance or optimal coefficients.
- Teacher may encode nuisance or lose new API information; R128 may fail to retain
  six teacher vectors. High rank/low reconstruction error is insufficient evidence.
- Disjoint task/domain datasets may be semantically non-identifiable without
  bridge labels. Actual availability of second-API and remote datasets was not audited.
- Original hardware/data issues are deferred, not solved by this review.
- S6 remains blinded. No new outcome exposure was introduced by this review.
- Two focused literature helpers (multi_api_review, residual_review) exited with
  known HTTP 429 retry-limit errors after useful partial findings. No edits or jobs
  were launched by them. They were not restarted. A previously successful helper
  performed bounded local theory/document checks; root's browser remained usable.
- These are determined read-only helper failures, not a recovered host crash or
  an unknown remote mutation. No claim that cumulative goal token accounting is
  the current context length; do not use that number as a context-size measurement.

## External runtime state — successor MUST reverify

This focused review launched no remote GPU job, teacher preparation, annotation
service, acquisition or persistent process. Existing other-lane external runtime
was not inspected and is UNKNOWN. Reverify host, PID/job ID, owner, run/config/data
hashes, log/checkpoint, and stop/resume contract before any authorized execution.
Do not resume a process based solely on an earlier session's description.

## Worktree state

New design-only documents: FOCUSED_REVIEW_MULTI_API_RESIDUAL.md,
FOCUSED_REVIEW_HANDOVER.md and FOCUSED_REVIEW_EVIDENCE.json, under humanengine_design/.
Root AGENTS.md gained navigation pointers. Historical design files remain intact.
No root Git coverage, no commit or branch claimed; use evidence hashes for this
handover. Do not initialize project-wide Git as an incidental action.

## Next actions

1. Researcher reviews the addendum's teacher lifecycle and API-to-R routing change.
   Until adopted, preserve both proposals as alternatives rather than silently
   implementing a mixture. Original coefficients/routing are not the new recipe.
2. Start a fresh implementation session for the next phase. Read root AGENTS.md,
   this handover, original design and focused addendum before edits.
3. If implementation is requested, first implement registry/masks, family/source
   budgets, route/version contracts and the synthetic gradient/causal/no-leak tests.
   Add teacher provenance and preparation interfaces separately from HE training.
4. Before training is authorized, determine suitable real second-API data and
   preregister grouping, teacher health gates, cost budget and acceptance margins.
5. Minimal research order: E0 duplication invariance; E1 fixed budgets vs CAGrad;
   E2 sequential replay; E3 summary vs independent latent targets. E4 open-R routing
   and other component comparisons are triggered by specific diagnostic failures.

## Do NOT

- Do not continue into implementation/training in this architecture-only task.
- Do not edit frozen scientific artifacts or read hidden/evaluator/S6 answers.
- Do not give five pose datasets five API-family budgets or fill missing labels
  with zeros/pseudo-force inferred from pose.
- Do not call HE's EMA an independent no-pose teacher; do not select EMG teacher
  by pose performance or allow held-out data into pretraining/statistics.
- Do not normalize away per-window amplitude or treat subject/device identity as
  uniformly irrelevant physiology.
- Do not add SSL after the solver and retain the same direction guarantee; do not
  call routed-gradient descent a proof of actual API non-regression.
- Do not mix clean state/E into masked inputs, future EMG into current targets, or
  old target versions into a resumed run.
- Do not claim residual is pose-free/disentangled, H is a complete belief state,
  or new physiological APIs are learned merely from teacher reconstruction.

## Recommended new session

`HUMANENGINE-IMPLEMENTATION-20260919-MULTIAPI-RESIDUAL`

Use the actual new start date. This is a design-to-implementation phase boundary,
so a fresh session is recommended under the user's Session Hygiene Protocol.
Keep this session as the design/evidence record. If further theoretical review
is requested instead, use `HUMANENGINE-REVIEW-YYYYMMDD-TARGET-ROUTING`.
After an actual host/tool incident with unknown outcomes, retire the incident
session as forensic record and use `HUMANENGINE-INCIDENT-YYYYMMDD-<PURPOSE>` for its
name; reconcile all external runtime state before a successor continues.
