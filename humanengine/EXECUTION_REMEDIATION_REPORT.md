# HUMANENGINE AUTONOMOUS EXECUTION-INTEGRITY REMEDIATION REPORT

2026-09-19. Project: `C:/Users/董伯言/Desktop/fingers/emg2pose_handstate`.
Authority: the user's autonomous execution-integrity remediation task, following
`D:/Temp/humanengine_remediation_reaudit/HUMANENGINE_REMEDIATION_REAUDIT.md`.

## 1. STATUS

**READY FOR FRESH INDEPENDENT RE-AUDIT**.

Verified local synthetic evidence: **169 passed, 0 failed, 0 errors, 0 skipped**;
96 previous tests plus 73 new cases. Twelve groups of unchanged independent
scientific-contract oracles also passed when re-executed by this remediation.
These results are not a fresh independent audit verdict, teacher-health evidence,
real teacher validation approval, or scientific validation of HumanEngine.

The scientific architecture and formulas are unchanged. No real teacher or HE
training, GPU/remote work, scientific sweep, dependency installation, S6 edit,
hidden-answer inspection or endpoint computation was performed.

## 2. ROOT CAUSES CONFIRMED

Before edits, the five selected groups from the latest audit reproduced failure:
`f01_gradient_identity`, `f01_mutations`, `f02_he`, `f02_teacher`, and
`f03_certificate`. See `EXECUTION_REPRODUCTIONS_BEFORE.json` and entry hashes.

| Finding | Confirmed cause | Local result after repair |
|---|---|---|
| R01 blocker | Equal entry/exit tensor bytes did not establish the state used by forward/backward. `.data.add_`, compute, restore accepted the wrong derivative. | Writes rejected before execution; fresh graph required; analytical derivative test passes. |
| R02 major | Declared config/state_dict omitted actual `CausalConv.stride`, `LayerNorm.eps`, and related executable constants. | Canonical executable comparison rejects live mismatches at checkpoint and teacher boundaries; per-operation checks protect macrosteps. |
| R03 major | Outer resealing replaced running target mean/second/count under insufficient artifact binding. | v3 nested scientific identity plus running-statistics identity rejects outer-only substitutions. |
| R04 major | Rollback called a sampler loader whose manifest had already been corrupted. | Structural restoration bypasses that loader; full state restores, or continuation is poisoned. |
| R05 major | Direct FrozenTeacher checked only a subset of duplicated health/certificate provenance. | One canonical evaluation-protocol projection is compared at both export and direct construction. |
| R06 minor | Resume transaction did not retain existing parameter gradients. | All registered parameter gradients are saved, loaded, and restored on late failure. |
| N01 | Nonpersistent buffers were outside state_dict-based coverage. | All registered buffers are protected and serialized in continuation state, regardless of persistence. |

Additional nearby gaps addressed: module/parameter hooks, RNN flat-weight cache
consistency, parameter storage/layout rebinding, duplicate persistent-buffer
representations, contradictory continuation/trainability metadata, and previously
constructed graphs. All findings still require fresh independent challenge.

## 3. EXTERNAL RESEARCH USED

The sources actually inspected were official **installed PyTorch 2.4.1+cpu
source/docstrings**, not live web search results. Links below identify their
official upstream counterparts. No scientific-method redesign was needed.

| Source | Why consulted / principle | Decision |
|---|---|---|
| [TorchDispatchMode source, v2.4.1](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/utils/_python_dispatch.py) | Dynamic eager operation interception, including factories; dispatch bodies run below the current mode. | Adopted for pre-operation checks, with explicit supported-execution limits. |
| Installed ATen `FunctionSchema`/`alias_info.is_write` probes | Determine write operands and `.data` storage aliasing before mutation. | Adopted. Schema arguments enumerated by index; object identity of repeated schema accesses is not assumed. |
| [functional_call](https://pytorch.org/docs/stable/generated/torch.func.functional_call.html), installed docstring | In-place writes can propagate into supplied state dictionaries. | Rejected as a sufficient immutability mechanism by itself. |
| [Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), installed `register_buffer` and `state_dict` docstrings | Nonpersistent buffers are omitted; state_dict contains shallow tensor references and excludes gradients. | Adopted explicit all-buffer, gradient, and structure snapshots. |
| [RNNBase source, v2.4.1](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/modules/rnn.py), installed `_update_flat_weights` | Flat-weight caches depend on registered parameter objects. | Adopted explicit cache-to-parameter consistency checking; preserved original parameter ownership during rollback. |

## 4. CANDIDATE SOLUTIONS CONSIDERED

| Problem | Alternatives and tradeoffs | Choice / rationale |
|---|---|---|
| Wrong-state gradients | More hashes/version checks remain vulnerable to transient writes. Stateless calls or replicas still need protection against writes inside evaluation. A restricted expression language would duplicate approved objective composition. | Guard eager operations before writes/changed execution can run, then differentiate inside the same guard. This supports existing custom `.step`/`.window` paths. |
| Executable identity | Only listing eps/stride is small but fragile. Hashing only config or state_dict misses actual objects. | Fingerprint actual class/source, behavior attributes, module tree, parameter layout, buffers where applicable, and production helper source; compare supported architectures to fresh cfg-built instances. |
| Rollback | Reusing public loaders is vulnerable to contaminated manifests/hooks. Subprocess isolation is stronger but substantially increases serialization/interface work. | Registered in-process transaction preserving module/parameter/buffer ownership; direct object-dictionary restoration and independent RNG restoration; poison on any restoration failure. |
| Running statistics | Matching fresh zero moments would reject legitimate continuation. An outer seal alone is replaceable. | Nested content identities and optional externally pinned expected identity, including legitimately nonzero running state. |
| Teacher protocol | More separately maintained field comparisons invite another omission. | One canonical protocol projection used for creation and both acceptance paths. |

The chosen approach costs CPU time and copies state. It is a correctness mechanism
for supported eager evaluators, not an isolation sandbox for arbitrary Python.
Detailed decisions and source notes are in `EXECUTION_DECISIONS.md`.

## 5. IMPLEMENTATION CHANGES

- Added `execution.py`: executable fingerprint, canonical architecture validation,
  supported-hook/cache checks, protected tensor alias/layout checks, and dispatch
  guard for forward and autograd.
- Added `transaction.py`: common snapshot/rollback, all-buffer and gradient state,
  original tensor storage restoration, RNG restoration, and poisoned-run gates.
- Reworked `optimization/macrostep.py` to evaluate fresh graphs inside guards and
  keep evaluation plus optimizer commit in one transaction. Original family/private
  gradient extraction, CAGrad objective, coefficients, and update ordering remain.
- Factory accepts `evaluation_modules=` for additional participating modules;
  integration explicitly registers its frozen teacher as well as model and heads.
- Upgraded scientific state and checkpoint schemas to v3, with continuation
  gradients/buffers/modes, executable compatibility and nested identities.
- Added `teacher/certificate.py`; strengthened artifact/export/direct-constructor
  checks. The executable identity describes the eval-mode copies actually exported.
- Added 73 acceptance cases, reproducible evidence collectors, and this handover.

## 6. SAME-SNAPSHOT SOLUTION

Every family closure and its core/private autograd evaluation run in
`ReadOnlyEvaluation`. Protected state comprises the coordinated parameters plus
all parameters/buffers in explicitly registered modules, including nonpersistent
buffers. Before each intercepted tensor operation, the guard checks actual module
attributes, module/parameter/buffer ownership, persistence, parameter trainability,
storage binding/layout, hooks, and RNN cache consistency.

ATen schema write operands sharing protected storage are rejected **before the
operation executes**. `.data` and view aliases therefore cannot perform the
demonstrated mutate→compute→restore attack. A changed eps/stride is rejected before
the changed frontend performs tensor computation. These are execution-time checks,
not an inference from equal final hashes. Violations latch even if caught by a
closure. Final checks still detect optimizer/controller/sampler/gradient changes.

Returned losses must originate in the guarded scope; foreign prebuilt autograd
intermediates are rejected. Both forward and backward are guarded. The independent
toy oracle gives `g_reserve=8*p`; at audit initialization `p=[.29,.69]`, the correct
gradient is `[2.32,5.52]`. The old attack produced `[10.32,13.52]`; the repaired API
rejects the write before the attack's forward is reached. Pure evaluation matches
the analytical core and private-head expectations.

Existing consumers must construct their objectives inside closures. Inputs and
fixed targets may be prepared beforehand, but retained autograd graphs may not be
returned. Register every participating module and pass an accessed sampler.
This does not validate whether an arbitrary caller chose the approved loss formula.

## 7. TRANSACTION / ROLLBACK SOLUTION

The common transaction covers parameters, all registered buffers, gradient values
and original gradient objects, trainability, module metadata/structure/class,
optimizer state, controller, supplied sampler, Python/NumPy/Torch RNG and initialized
CUDA RNG bookkeeping. Only CPU execution was tested. Module/parameter ownership is
preserved through memoized snapshots; original storage aliases are restored.

On failure, restoration bypasses potentially corrupted public loaders. Independent
restorations are attempted even if another restoration fails. Exact registered
state restoration permits reuse; a restoration error marks roots, optimizers,
controller and sampler `_he_unusable=True` and raises `UnusableTrainingState`.
Macrostep/resume/checkpoint gates reject poisoned continuation. Clearing this flag
is not an authorized recovery procedure: restart from independently verified state.

Tests cover simultaneous optimizer-moment, quota/manifest, sampler RNG, global RNG,
controller and gradient corruption; optimizer failure after a core update; tensor
rebinding/layout change; hook installation; cache substitution; and injected RNG
rollback failure. Successful rollback does not promise to restore external logs,
unregistered Python objects, or retained external autograd graphs.

## 8. SCIENTIFIC CHECKPOINT IDENTITY

Schemas are `he-kernel-checkpoint-v3` and `emg-teacher-preparation-v3`.
An outer content seal, nested `scientific_identity`, actual executable definition,
scientific metadata and continuation state are checked before loading. Optional
`expected_identity=` pins a checkpoint selected from a trusted manifest.

The executable definition binds module class bodies, production Python helper
source hashes, PyTorch version, actual public and non-framework-private behavior
attributes, parameter layout, and module tree. Supported HE/teacher architectures
are checked against a fresh cfg-built reference under restored RNG. It does not
treat a declared config string as proof that the live object obeys it.

Teacher running mean/second/count have their own identity. Replacing any of them
and recomputing only the outer seal fails. Legitimate nonzero moments resume into
a fresh preparation. Persistent buffers duplicated between state_dict and the
continuation must agree; training/trainability fields must agree with scientific
semantics. Nonpersistent buffers and `.grad` are explicit continuation state.

Failed HE and teacher resume restore previous gradients and other registered
state. Old v1/v2 schemas fail loudly; there is no silent migration. A party that
recomputes all nested identities creates a **different** artifact: use the external
identity pin to reject substitution of that different artifact. Hashes provide
content consistency, not signatures or proof of trusted authorship.

## 9. TEACHER ARTIFACT / HEALTH IDENTITY

Certificate schema: `teacher-artifact-certificate-v3`. The certificate binds exact
EMA contents, executable encoder and frozen-statistics behavior, target statistics,
physical-statistics content, configuration, training step and target version.
All registered encoder/statistics buffers, including nonpersistent ones, are bound.

`evaluation_protocol` canonically includes evaluation version/config, data manifest,
checkpoint identity, partition and partition identity, budget identity, checkpoint
and preregistered steps, and the three health thresholds. Export and direct
FrozenTeacher construction compare this same projection. Health checks are
recomputed, and certificate/header/version/partition/step accounts must agree.

Unchanged direct construction and re-export from a frozen object's own components
produce exactly identical synthetic encoder outputs. Stale EMA/targets/eps/stride,
changed buffers, and conflicting protocol/header fields are rejected. This certifies
what artifact/protocol supplied metrics claim to describe; it does not prove the
metric numbers were truthfully measured. Fixtures remain explicitly synthetic.

## 10. ADVERSARIAL TESTS

All rows below are new tests in `tests/test_execution_integrity.py`. Each listed
variant is an executed case. Expected outcomes and observed outcomes agreed;
all **73 passed**. Detailed case names/timing are in the JUnit and evidence JSON.

| Construction / cases | Expected | Observed / result |
|---|---|---|
| Independent snapshot oracle, unchanged then sampler RNG advanced (1) | Equality then inequality | Both detected; PASS |
| Analytical toy derivative (1) | `8*p` reserve and manual pose derivative | Equal within original numeric tolerances; PASS |
| Core/private/persistent/nonpersistent `.data` writes, persistent and transient variants (8) | Reject before write; exact state restore | Forward-after-write marker never reached; PASS |
| Real frontend eps/stride, persistent and transient (4) | Raw output demonstrably changes outside guard; guarded use rejected | Output change confirmed, state restored; PASS |
| Prebuilt loss or intermediate (2) | Reject graphs made outside scope | Rejected and restored; PASS |
| Catch a guard violation inside closure (1) | Latched rejection | No commit; PASS |
| Combined continuation corruption; late optimizer commit failure; storage rebinding; dtype/layout substitution; parameter hook; module hook; RNN cache (7) | Reject and exact restore including original parameter storage/gradient object | All restored; PASS |
| Inject rollback RNG failure (1) | Poison every continuation root and reject subsequent step | Sticky fail-closed state; PASS |
| HE checkpoint create/resume × eps/stride (4) | Reject actual config disagreement without state change | Rejected; PASS |
| Teacher checkpoint create/resume × eps/stride (4) | Same | Rejected; PASS |
| Running mean/second/count × state_dict/continuation, outer-only reseal (6) | Reject nested identity mismatch | Rejected without contamination; PASS |
| Valid nonzero running moments, nonpersistent buffer and existing gradient (1) | Resume actual saved values into fresh objects | Values exactly restored; PASS |
| Late HE/teacher loader failure after gradient/RNG mutation (2) | Loader reached; exact previous state restored | Marker reached and gradients restored; PASS |
| Contradictory buffer duplicates under new self-consistent outer/nested seal, HE/teacher (2) | Reject semantic contradiction | Rejected; PASS |
| Entire replacement artifact against externally pinned identity (1) | Reject different identity | Rejected; PASS |
| Continuation training or trainability contradicts scientific state (2) | Reject | Rejected; PASS |
| v1/v2 HE schema (2) | Loud incompatibility | Rejected; PASS |
| Direct teacher: eps, stride, EMA, target, encoder nonpersistent buffer, target nonpersistent buffer (6) | Reject stale certificate | Rejected; PASS |
| Direct teacher contradictions (17): evaluation version/config, data, checkpoint, partition, partition identity, budget, checkpoint/preregistered steps, raw/within/amplitude thresholds, evidence partition, target version, certificate version, checks, health_passed | Reject every contradictory account | Every case rejected; PASS |
| Unchanged direct teacher and frozen-component re-export (1) | Exact output parity | Exact parity; PASS |

The snapshot oracle independently compares values, metadata, gradients, optimizer
state and all RNG states; it does not use production hashing/restoration helpers.
Analytical derivatives and manually changed runtime outputs provide separate
oracles. Existing floor/provenance/adapter/mode, stale EMA/target, and scientific
regression assertions were also retained in the previous 96 tests.

## 11. FULL TEST SUITE

Final command from project root:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda3\envs\emgforce\python.exe' -B -m pytest humanengine/tests -q -p no:cacheprovider --junitxml=humanengine/execution-final-suite.xml
```

Final JUnit: **169 tests; 169 pass; 0 fail/error/skip; 86.036 seconds**.
Python/runtime versions and current source/config/artifact hashes are recorded in
`EXECUTION_EVIDENCE.json`. Test logs remain inside this lane; import bytecode and
pytest cache writes were disabled. Timings are local execution observations, not
performance benchmarks; independent CPU verification overlapped part of the run.

Development results are preserved transparently: first suite 88/96 passed (8
failures); corrected subset 48/48; first expanded suite 143/165 (22 failures caused
by the new snapshot oracle comparing Random object identities instead of RNG
state); corrected subset 74/74; final suite 169/169. The comparator was repaired
with `getstate()` equality and an explicit sensitivity test, not by omitting RNG.

No tolerance was widened or prior assertion removed. Two old fixtures were updated:
integration now computes the same approved full objectives inside guarded closures;
the deliberately malformed late-RNG checkpoint receives a new valid v3 identity
so that it still reaches the late loader. Separate outer-reseal rejection tests
remain. Previous final/historical reports and JUnit files are preserved.

## 12. PRESERVED SCIENTIFIC CONTRACTS

`tools/recheck_preserved_contracts.py` imports only guarded audit modules and calls
selected functions directly; it never invokes their writing `run()` entry points.
`EXECUTION_PRESERVED_CONTRACTS.json` records all 12 passing groups and unchanged
external audit-source hashes. Total oracle execution: approximately 12.95 seconds.

| Contract | Unchanged independent function re-executed |
|---|---|
| Future-input causality, sequence/streaming equivalence | `causal_stream` |
| Reset isolation | `resets` |
| A→F→U/raw E ordering, detached-R, shadow safety | `order_route_shadow` |
| Six causal teacher targets and masked support | `teacher_targets_mask` |
| Exact L_R and manual physical summary | `reductions_physics` |
| Family E0 alias invariance | `family_aliases` |
| Normal CAGrad geometry and failure handling | `cagrad_numerical` |
| Burn-in boundary matrix | `f04_boundaries` |
| Unique-anchor quota semantics | `f05_quotas` |
| Registered effective rank | `f07_rank` |
| Extreme finite CAGrad | `f08_extreme` |
| M_A coefficient and clean/masked isolation | `f09_and_views` |

S256/R128, detached-R route, teacher scales/objective, residual coefficients,
family budgeting, VICReg, pose geometry and M_A weak-anchor semantics were not
redesigned. Quota and burn-in production code were not edited in this remediation.

## 13. FILES MODIFIED

Existing production files: `manifest.py`, `scientific_state.py`,
`optimization/macrostep.py`, `optimization/factory.py`, `teacher/model.py`,
`teacher/preparation.py`.

New production files: `execution.py`, `transaction.py`, `teacher/certificate.py`.
Tests: existing `test_integration.py` and `test_remediation.py` fixture updates;
new `test_execution_integrity.py`.

Current documentation: README, SESSION_HANDOVER and IMPLEMENTATION_MAP.
Added decision/report/evidence records, protected/lane before/after hash inventories,
development/final JUnit records, and two reproducible tools under `humanengine/tools`.
`EXECUTION_EVIDENCE.json` contains the exact added/modified file list and hashes.
Every modification for this task is inside `humanengine/`.

## 14. HUMANENGINE INTEGRITY

Entry lane hashes were captured before source edits. Final source, tests and six
configuration hashes, JUnit digest, preserved-oracle evidence, decisions/report and
handover digests are recorded in `EXECUTION_EVIDENCE.json` / `EXECUTION_LANE_AFTER.json`.
`FROZEN_BASELINE.json` and historical implementation/remediation evidence are unchanged.

The project root remains outside Git. Nested `baseline/emg2pose` is clean at
`5f6f62b1a0a08426adffe55900842e75a8adb38c`. No commits, branches, worktrees, or Git
initialization were performed. New-lane provenance is content-hash based.

## 15. EXTERNAL / S6 DRIFT

**321/321 protected entries unchanged from this remediation's entry to exit**,
including the entry's already-missing path. This is an entry-to-exit observation,
not a claim that all files match the historical frozen baseline.

Historical comparison remains **308 matching, 12 changed, 1 missing**. All thirteen
differences existed at this task's entry; the latest audit had already observed
additional concurrent S6 drift relative to the previous remediation. Exact paths,
historical/entry/final hashes are in the protected inventories and evidence JSON.
Authorship, process ownership and authorization remain **UNRESOLVED**. No S6 file
was edited, restored, rebaselined, or interpreted as an owned change here. Hidden
answers were not parsed. No demonstrated dependency connects those records to the
synthetic HumanEngine kernel. Future real teacher manifests require their own
independent provenance validation.

## 16. KNOWN LIMITATIONS

- Validated scope: registered-state, eager CPU PyTorch in one calling thread,
  with immutable loaded implementation code. Compile/JIT/distributed/GPU execution,
  custom native kernels, cross-thread mutations, and hot monkeypatching are unverified.
- The guard is not an arbitrary-code sandbox. Deliberately disabling dispatch,
  foreign-native memory writes, and unregistered external state are outside the
  supported evaluator contract. Active module/parameter hooks and instance callable
  overrides fail closed instead of receiving an unverified guarantee.
- Checkpoint/certificate hashes establish content consistency. Recomputing every
  identity produces a different artifact; externally pin identity when selecting a
  particular checkpoint. Metric truth and authorization are not cryptographically
  attested. Untrusted pickle loading is not addressed by these in-memory helpers.
- Per-operation metadata checks, tensor comparisons, prototype validation, source
  hashing and transaction copies add overhead. No production throughput claim exists.
- Strict canonical architecture matching rejects undeclared extensions. New API
  architectures/runtime types require a versioned definition and independent tests.
- Rollback restores registered continuation state, not arbitrary external aliases,
  logging side effects or already-retained external computation graphs. Loss graphs
  must be constructed afresh in the guard.

## 17. REMAINING RISKS

The dispatch guard and broader executable/transaction abstraction need fresh
independent adversarial review, especially native operator alias-schema coverage,
derived caches, serialization semantics and supported-boundary enforcement. Local
tests are substantial evidence, not a proof covering every possible Python program.

All real teacher budgets, manifests, fitted statistics and health thresholds remain
unbound experiment inputs. Teacher usefulness, representation quality and cross-user
generalization remain **Unknown**. The original representation hypothesis was not
falsified or established by infrastructure repair. External S6 provenance remains
unresolved. None of these unknowns is recorded as zero performance or a failed
scientific experiment.

## 18. EXACT NEXT SESSION

**HUMANENGINE-REAUDIT-FRESH-EXECUTION-INTEGRITY**

Retire this remediation workbench at the phase boundary and use a fresh independent
auditor. No host crash or unknown-outcome incident occurred during this remediation.
The next task must reread root/lane AGENTS, current SESSION_HANDOVER, this report,
decisions, original scientific specifications, both independent audit reports and
current source; reverify external runtime and all source/config/protected hashes.

Audit R01–R06/N01 and nearby cases without relying on a green local suite. Preserve
the scientific architecture, historical records and S6 blinding. Only a fresh
independent acceptance decision can authorize progression toward teacher validation.
Do not start teacher/HE training, GPU/remote jobs or scientific tuning as part of
this handoff. The required Objective/Frozen decisions/Verified state/Scientific
evidence/Unknown state/External runtime/Worktree/Next actions/Do NOT fields are in
`SESSION_HANDOVER.md`.
