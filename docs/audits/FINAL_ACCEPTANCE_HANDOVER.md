# Final acceptance -> real teacher validation handover

## Objective
Completed the user's bounded read-only final HumanEngine v0.1 acceptance audit. Verdict PASS. Next phase is real teacher validation; do not repeat a general kernel audit.

## Frozen decisions
A->F->U/raw E; S256/R128; detached-R explicit APIs; independent EMG teacher and .1/.5/2 s x 2 layers; residual/family/weighted-base CAGrad; burn-in; unique quotas; effective rank; M_A=.05 L_phys. v4 checkpoints, v3 teacher certificate; eager single-process single-threaded CPU, immutable Python, registered participants, no tied Parameters. No HE_FULL authorization.

## Verified state
191/191 current synthetic tests pass, zero failures/errors/skips, 92.134 s. New teacher 5 == 2+resume+3 exact across named state/moments/RNG, including all student parameters, EMA/step/running statistics/nonpersistent buffer. Actual HE factory 4 == 2+resume+2 exact across model/heads/optimizers/controller/sampler/RNG. Known negative matrix rejects, diagnostics observational, combined rollback exact, failed rollback sticky-poisons. Teacher certificate bounded regression passes. Twelve historical independent scientific oracle groups pass.

## Scientific evidence
Only synthetic implementation evidence. New scripts/oracles compare actual tensors and named optimizer state, not only hashes. Read FINAL_ACCEPTANCE_REPORT.md plus independent_results.json, he_results.json, preserved_results.json, suite.xml and integrity_final.json in this directory.

## Unknown / contaminated state
Real teacher quality, information preservation, cross-user/dataset/unseen-API transfer, physiological meaning and deployment performance remain UNKNOWN. Synthetic teacher health fixtures are not real evidence. Historical protected baseline: 308 match, 12 changed, 1 missing, attribution/authorization UNKNOWN. No demonstrated dependency into the audited synthetic path; no hidden answers read.

## External runtime state — successor MUST reverify
Audit ran only completed local CPU/hash/pytest processes. No training/GPU/remote job started. Python D:/Anaconda3/envs/emgforce/python.exe; 3.11.16 / torch 2.4.1+cpu. Other lanes' process/job ownership UNKNOWN. Reverify any later runtime, host/PID/purpose/config/data/log/checkpoint/stop method before adoption or execution.

## Worktree state
No project files modified. Entry->exit protected 321/321 stable; HumanEngine/design 113 files unchanged, zero additions. Project root outside Git. Nested baseline/emg2pose clean at 5f6f62b1a0a08426adffe55900842e75a8adb38c. All this audit's new evidence is D:/Temp/humanengine_v01_final_acceptance. Repository SESSION_HANDOVER.md remains deliberately untouched and still says ready for final acceptance; this external report supplies the newer independent decision. Carry it explicitly rather than editing frozen records during this audit.

## Next actions
Begin real teacher validation planning/preregistration in the new session: use approved real EMG manifests/splits, train-only statistics, fixed-budget selection and EMG-only health thresholds/references. Reverify actual data/runtime. Do not promote synthetic health values. Infrastructure may reopen only if empirical work reveals a concrete bug, the support boundary expands, or a new scientific API requires it.

## Do NOT
No further open-ended infrastructure audit, HE_FULL, S6 edits/rebaseline/hidden-answer inspection, architecture redesign or unsupported execution under the v0.1 acceptance claim. This audit does not execute teacher training. Preserve no-API-label teacher selection and preregistration.

## Recommended new session
HUMANENGINE-REAL-TEACHER-VALIDATION-V01

Phase transition is the handoff trigger; no incident or context-limit event occurred. Retire this completed audit as its evidence record.
