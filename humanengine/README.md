# HumanEngine-EMG v0.1 scientific kernel

This directory is a new implementation lane. It implements the approved original
architecture plus the focused addendum's API-family budgets, detached-R readout and
independent frozen EMG teacher. It is **not a trained or scientifically validated
HumanEngine**, and contains no real-data/remote training launcher.

Current status: **READY FOR FINAL ACCEPTANCE AUDIT** (2026-09-20).
191 synthetic CPU tests passed; 12 preserved independent-oracle groups passed.
This does not authorize real teacher validation or training; the only next step is
the bounded `HUMANENGINE-FINAL-ACCEPTANCE-AUDIT-V01` task.

Start with `EXECUTION_REMEDIATION_REPORT.md`, `EXECUTION_EVIDENCE.json`, and
`SESSION_HANDOVER.md`. Authoritative sources are preserved under
`humanengine_design/`; the user implementation request is preserved here as
`SPEC_IMPLEMENTATION_TASK.md`.

## Public composition

- `HumanEngine(cfg, amplitude_statistics)` exposes `init_state`, `step`,
  `forward_sequence`, and `forward_training_sequence`. State is explicit and
  per-stream. Runtime inputs are B×T×C EMG and B×T float64 timestamps.
- `forward_training_sequence` performs each view's own burn-in and detaches its
  state at the TBPTT boundary. `step` uses bounded caches, never full-history
  recomputation. No label participates in runtime state/reset decisions.
- E sees raw published EMG before A; F follows A; U follows F. F is exported before
  U for bottleneck probes. API readouts see [S,detach(R)]. `TrainingHeads` is kept
  separate from deployed runtime, as is every teacher.
- `TeacherPreparation` is fresh EMG-only student + internal EMA. `losses` requires
  explicit temporal masks. Forward does not silently mutate EMA/statistics;
  update them explicitly on registered training batches. `freeze_teacher` needs
  fixed-budget label-free health evidence and frozen post-training statistics.
  Certificate v3 requires recording `teacher_artifact_identity` BEFORE health
  evaluation, with data/checkpoint/evaluation/partition/budget identities. Export
  recomputes executable encoder/statistics and one canonical evaluation-protocol
  identity, refusing stale or contradictory evidence even on direct construction.
  Identity is not proof that
  submitted quality measurements are true.
- `full_objectives` composes the current pose/reserve kernel, with independent
  view state, burn-in, target support and precise NA handling. Additional real API
  families supply `FamilyEvaluation` using their own label definitions.
- `make_macrostep` derives the correct complete F/S/R vector and private groups.
  Closures construct fresh graphs inside an eager dispatch guard; protected writes
  are rejected before execution, with actual module semantics checked at each op.
  Forward and autograd share that guard before any optimizer step. Prebuilt graphs
  are rejected. `Macrostep` installs isolated gradients; stale `.grad` cannot contribute.
  Private G uses its own geometry loss per the latest implementation request.
- `FamilyQuotaSampler.plan_valid_quotas` supplements hierarchical window sampling
  with explicit per-component valid-target grants. Apply `quota_mask` to enforce
  grants; an incomplete plan is NA and must not update any family.
  v2 coverage supplies explicit valid indices; grants contain AnchorIdentity
  objects. Revisited windows cannot pay an anchor twice in one plan.
- Reference-scale fitters use permitted train partitions and fixed constants.
  `a_R` uses equal-scale/layer Huber of normalized teacher targets versus zero;
  its declared floor is applied once, never a moving loss/gradient norm.
- Checkpoint constructors preserve parameters, optimizers, controller, sampler,
  RNG and scientific signatures. Resume checks the live model definition before
  mutation and rolls back if a load fails.
  Checkpoint v4 additionally binds each optimizer implementation, update-rule options,
  exact ordered scientific parameter names/group membership and per-name state. It rejects
  unsupported shared/overlapping Parameter storage before load and verifies the actual
  named model/continuation/optimizer state after load. It also binds actual executable
  definitions, running statistics, all buffers, gradients and continuation modes under
  a nested scientific identity. The caller supplies
  the real `content_hash(statistics_content(model.core.E.statistics))`; a declared
  arbitrary string is rejected. Runtime frozen statistics, adapter identity and
  trainability mode must match. A/U tensors are loaded from the sealed payload.
  Use `expected_identity=` to pin the exact artifact chosen from a trusted manifest.
  Contradictory duplicate state is rejected. v1/v2/v3 checkpoints require a separately
  reviewed limited-trust migration; there is no silent load.

Macrostep factory registers model and training heads for buffer/metadata protection.
Pass the frozen teacher and other participating modules as `evaluation_modules=`.
When closures can access a sampler, pass `sampler=` to `step`. Standalone callers
must register additional accessible modules with `stateful_modules=`. Closures
must be observational. Diagnostics run inside the same eager read-only guard plus an
observation transaction that restores registered continuation state and RNG even when
the callback returns normally. This boundary is a correctness guard, not a sandbox against
arbitrary Python side effects outside registered state. Failures restore registered
parameters, persistent/nonpersistent buffers, gradients, optimizers, controller,
modules, sampler and RNG without calling contaminated public loaders. Failed
rollback poisons continuation and requires restart from verified artifacts.

Validated execution scope is eager CPU in one calling thread. Compile/JIT/GPU,
cross-thread mutation, hot monkeypatching, deliberately disabling dispatch and
arbitrary foreign-native writes are not supported guarantees. Module/parameter
hooks and instance callable overrides require a versioned implementation and
currently fail closed. See the report for exact limits and overhead.

Training eligibility is explicit in `input_quality["training_eligible"]` and is
separate from runtime warmup. Full objectives reject anchors before burn-in B.
Private M_A uses .05 L_phys by default, never multiplied by reserve_alpha.

## Constraints that remain visible

Fixed budgets are not gradient-norm percentages. Weighted-base CAGrad is a
HumanEngine extension, with one full-core solve and a primal/dual certificate.
Finite failures fall back to g0 and report why. A nonfinite g0 cannot be a safe
fallback: the macrostep raises before any update. Detached-R directions are local
surrogates; actual Adam/other optimizer deltas and pre/post losses must be audited.

Stats and reference scales require train-only provenance. There are no defaults
pretending to be fitted real data statistics. The six JSON configurations expose
constants and explicitly leave real teacher budget, fitted statistics, manifests
and health thresholds unbound. Those are later experiment inputs, not permission
to invent values. Genuine measurement-noise scale is required for the weak-noise
view; otherwise the invariant objective is NA.

Teacher health fixtures in tests are synthetic, untrained wiring fixtures and are
marked `evidence_partition=synthetic`. They are not evidence of a usable teacher.
No tests measure real pose/force/contact/fatigue performance or cross-user transfer.

## Local verification

Verified existing environment: `D:\Anaconda3\envs\emgforce\python.exe`,
Python/PyTorch CPU with torch 2.4.1+cpu, NumPy 1.26.4, SciPy 1.14.1 and pytest 8.4.1.
No dependency installation or environment modification was performed.

Run from the project root, with `PYTHONDONTWRITEBYTECODE=1` to avoid writing import
caches beside frozen modules. Tests are isolated under `humanengine/tests`; their
cache and JUnit report also live in this lane. See the final report for the exact
executed verification command and fresh independent re-audit handoff.

The HDF5 wrapper preserves the lexical `C:\emg2pose` path through `h5py.File` and
does not resolve the junction. No real HDF5 training data or S6 labels were read
for the implementation tests. Canonical16 parity uses the frozen adapter as an
oracle without changing it or adding a duplicate canonical training objective.
