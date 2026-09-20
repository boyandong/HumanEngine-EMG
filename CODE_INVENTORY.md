# Standalone code inventory

This repository was assembled from
`C:/Users/董伯言/Desktop/fingers/emg2pose_handstate` on 2026-09-20.

## Included

- Complete `humanengine` Python package, configs, tests, runbooks and scientific
  implementation records.
- Real teacher v0.1 selection, remote acquisition, smoke, preregistration,
  training and evaluation programs.
- HumanEngine architecture and focused residual/teacher design records.
- Frozen canonical16 implementation and tests.
- The upstream emg2pose reference checkout required to resolve the official
  joint vocabulary, without its nested Git metadata or data artifacts.
- Official public emg2pose metadata and tar member index used to freeze a
  bounded real-data acquisition plan.
- Dataset acquisition/verification helpers and the final infrastructure
  acceptance report.

## Intentionally excluded

- Raw EMG/pose files (`.hdf5`, `.mat`, arrays and archives).
- Model checkpoints, optimizer state and training outputs.
- SSH keys, passwords, tokens, environment files and machine credentials.
- S6 hidden/evaluator content and annotator payloads.
- Logs, caches, generated JUnit XML and temporary files.
- Other closed project lanes (`bridge_audit`, `posture_decoder`, visualization,
  annotation UI, inference outputs) that HumanEngine does not import.

The source project remains untouched and is not replaced by this standalone
copy. Real experimental artifacts stay on the authorized server and are linked
through immutable manifests and hashes rather than committed to Git.
