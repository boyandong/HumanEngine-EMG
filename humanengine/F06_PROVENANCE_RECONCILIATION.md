# F06 read-only provenance reconciliation

2026-09-19. No S6 files restored, edited, executed, or rebaselined. No hidden
answer/evaluator label contents were read. Protected hashing does not interpret
labels. The original 321-entry frozen baseline remains intact.

Audit: eight differences (seven changed, one missing). Remediation entry: NINE
(eight changed, one missing), including the additional staged_site_manifest.json.
This difference existed before source edits. The before snapshot records old and
actual hashes. Timestamps support chronology, not authorship or authorization.

| Path relative to human_semantic_validation/ | Classification | Evidence / unresolved link |
|---|---|---|
| pilot2/logs/pilot2_state_after_revision.json | UNRESOLVED | Missing in audit and entry snapshots. pilot2/README.md:19–22,235 explicitly reference it; no authorized removal record located. |
| pilot2/logs/verifier_pilot2_A.json | UNRESOLVED | Header started_utc 2026-09-19T11:56:11.740Z; mtime 11:56:22Z. pilot2 README documents page verification; pilot3 README:143–157 documents later checks, but supplies no exact hash/authorization chain for these bytes. |
| pilot2/logs/verifier_pilot2_B.json | UNRESOLVED | mtime 11:56:31Z; documented page-verification category, no exact old/new version linkage. |
| pilot2/logs/verifier_pilot1_regression_A.json | UNRESOLVED | mtime 11:56:48Z; pilot2 README:239 documents regression verification, not these exact replacement bytes. |
| pilot2/logs/verifier_formal_regression_A.json | UNRESOLVED | Header started_utc 11:56:48.844Z, mtime 11:56:58Z; formal regression documented, exact replacement authorization unavailable. |
| pilot2/tools/stage_pilot2_site.py | UNRESOLVED | mtime 11:49:15Z; pilot2_added_annotators.json header records adding C at 11:48:15Z. Plausible adjacent evolution is not authorization for every code difference. |
| pilot2/tools/serve_pilot2.py | UNRESOLVED | mtime 11:49:02Z; same nearby C record, no authenticated code delta. |
| pilot2/tests/test_s6_pilot2.py | UNRESOLVED | mtime 11:55:26Z; pilot2 README records 58 prior tests, pilot3 records 83 later tests across lanes. Neither authenticates this exact replacement. |
| logs/staged_site_manifest.json (additional) | UNRESOLVED | generated_utc 2026-09-19T12:30:58.886280+00:00; mtime 12:30:58Z. SESSION6_STAGE_A_REMEDIATION.md:44,70 documents generated manifests. Regeneration plausible, precise authorization not established. |

Sources: governance; independent audit/handover; S6 Stage-A report/remediation;
BLINDING_SPEC; pilot2/pilot2_external/pilot3 README; added-annotator header;
page-verifier headers; staged-manifest metadata; filesystem metadata; frozen
hashes. No S6 server, verifier, scientific evaluator, or annotation workflow ran.
No entry is labeled AUTHORIZED / EXPLAINED merely from a nearby timestamp.

Root is outside Git; nested baseline is clean at
5f6f62b1a0a08426adffe55900842e75a8adb38c. This bounded reconciliation cannot establish
authorship. A lane owner with exact change records can resolve these without
exposing hidden answers. Final integrity uses the actual remediation-entry hashes,
not a rewritten baseline; machine-readable comparisons are in REMEDIATION_EVIDENCE.
