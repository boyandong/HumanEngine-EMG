# Scientific-kernel reconciliation

Current continuation-identity closure (2026-09-20): see
`CONTINUATION_CLOSURE_REPORT.md` and `CONTINUATION_EVIDENCE.json`. X01/X02 use
named optimizer implementation/group/state ownership in manifest; X03 uses the
independent-parameter topology contract and post-load named-state verification;
X04 uses ReadOnlyEvaluation plus ObservationTransaction. Checkpoints and scientific
state are v4; teacher artifact certificates remain v3 because their already-validated
artifact/protocol contract did not change.

Previous execution-integrity remediation (2026-09-19): see
EXECUTION_REMEDIATION_REPORT.md and EXECUTION_EVIDENCE.json. R01/R02 use
execution.py and macrostep; R03 uses manifest/scientific_state; R04/R06 use
transaction.py; R05 uses teacher/certificate plus model/preparation; N01 covers
all registered buffers. Its checkpoint/scientific-state/teacher certificates were v3.
Quota and effective-rank definitions remain v2; scientific API route remains v1.
Below is the retained mapping for the previous remediation and implementation.

Remediation authority: 2026-09-19 kernel remediation request and independent audit.
F01 maps to macrostep/factory; F02 to scientific_state/manifest; F03 to
teacher/preparation/model; F04 to api/full; F05 to data/sampling; F06 is read-only
reconciliation; F07 representation; F08 cagrad; F09 full/config. Audit-derived
regressions are grouped in tests/test_remediation.py. Checkpoint/certificate/quota
and effective-rank definitions are explicitly v2; scientific API route stays v1.

Authority: user implementation task (2026-09-19), focused addendum, original
architecture for unrelated topics, real frozen repository interfaces. The new user
task authorizes implementation/local CPU verification; historical design-only stop
conditions do not prohibit this newly authorized phase. No scientific training.

| Requirement | Source | Module / verification |
|---|---|---|
| A→F→U; raw E; S256/R128; bounded streaming | original C,J,appendix 1; task 10–14 | core/frontend,state,summaries; prefix/chunk/reset tests |
| all explicit APIs read [S,detach(R)] | addendum 5; task 15–16 | api,heads; isolated and shadow gradient tests |
| independent causal teacher, internal EMA, frozen export | addendum 4; task 24–33 | teacher/model,preparation; isolation/causality/health tests |
| six targets, observed and physical anchors, direct variance | addendum 4.3 | objectives/residual; exact reductions and reachability |
| pose geometry G(S), weak-noise VICReg V(S) | original G2; addendum 5 | objectives/regularizers; support/gradient tests |
| family/source identity, fixed scales, quotas | addendum 2; task 18–23 | registry,data/sampling; E0 duplication property |
| one complete core vector, same snapshot, private losses | addendum 2.3; task 20,43–49 | optimization; trust ball, fallback, mutation/private tests |
| replay/output retention, head-only enrollment | addendum 2.4; task 51–54 | continual/contracts; no H distillation |
| provenance, strict resume, export/probe metadata | task 58–63 | manifest,diagnostics; mismatch and observational tests |
| actual official HDF5/schema and canonical mapping | data.py,utils.py,canonical_hand.py | data/sequence,heads/canonical; frozen numeric parity |

Source-defined: six targets, routing, shapes, coefficients, budgets, physical bands,
stride anchors, support thresholds. Repository facts: `emg2pose/timeseries` compound
fields `emg,joint_angles,time`; IK valid is not-all-near-zero (finite is an additional
new-lane guard); canonical indices resolved from frozen official joint names.

Implementation choices: immutable explicit per-stream state, padded packed outputs
with valid masks when streams emit different counts; reset at discontinuities before
the next finite sample; periodic Hann window; JSON configs; SciPy constrained dual
solver with primal/dual certificate and logged base fallback; unit-test CPU threads=1.
These are not new scientific claims. Private G uses its own geometry objective per
the latest task 48, rather than letting family beta mechanically scale its optimizer.

No architecture alternatives implemented. Missing scientifically necessary metadata,
statistics, health evidence or component coverage fails explicitly; synthetic tests
provide clearly marked fixtures, never purported real training statistics. CAGrad
derivation is fully specified by addendum 2.3, so no new external survey is needed.
