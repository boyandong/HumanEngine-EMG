# HumanEngine-EMG

HumanEngine-EMG is a causal EMG representation research codebase. The current
v0.1 lane contains the scientific kernel, the independent EMG teacher, exact
checkpoint/continuation contracts, tests, frozen design records, and the
server-first real-teacher validation workflow.

Current scientific status: infrastructure v0.1 passed its bounded acceptance
audit. Real teacher validation is in progress. HumanEngine itself has not been
scientifically trained, and this repository makes no claim that force, contact,
fatigue, unseen-API transfer, or physiological state decoding has been validated.

## Repository map

- `humanengine/`: runtime package, objectives, teacher, optimization, tests,
  configs, implementation records, and experiment entrypoints.
- `humanengine/experiments/real_teacher_v01/`: frozen acquisition plan and the
  server-first teacher screening workflow.
- `humanengine_design/`: architecture and focused residual/teacher review.
- `representation/`: frozen canonical-hand mapping required by the HumanEngine
  pose interface and provenance tests.
- `baseline/emg2pose/`: pinned upstream reference needed to resolve the official
  joint vocabulary. It retains its upstream license and is not HumanEngine code.
- `data/tools/`: official emg2pose acquisition and verification helpers.
- `data/priority_pose_subset/raw/`: metadata and tar index only. No EMG payloads.
- `docs/`: dataset and server acquisition references.

## Install and verify

Python 3.11 is the accepted CPU reference environment. The real server screen
also records its exact PyTorch/CUDA runtime and performs a bounded backend smoke
check before scientific training.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest humanengine/tests -q
```

The historical accepted reference was Python 3.11.16 with PyTorch 2.4.1 CPU.
Dependency ranges in `pyproject.toml` make installation convenient; scientific
reproduction should use the versions and identities recorded by the experiment.

## Real teacher validation

The real-data workflow is server-first. Full EMG data is intentionally absent
from Git. The v0.1 plan consumes only complete, immutable official archive
members and freezes train/development membership before training.

```bash
python -B humanengine/experiments/real_teacher_v01/select_data.py
python -B humanengine/experiments/real_teacher_v01/acquire_remote.py
python -B humanengine/experiments/real_teacher_v01/run_screen.py prepare
python -B humanengine/experiments/real_teacher_v01/run_screen.py smoke
python -B humanengine/experiments/real_teacher_v01/run_screen.py preregister
python -B humanengine/experiments/real_teacher_v01/run_screen.py train
python -B humanengine/experiments/real_teacher_v01/run_screen.py evaluate
```

These commands are experiment entrypoints, not a promise that arbitrary machines
or data roots are compatible. Read the lane handover and inspect the frozen
manifest before running them.

## Data and privacy boundary

This repository excludes raw HDF5/MAT data, checkpoints, credentials, hidden S6
annotations, evaluator payloads, and training output. The included metadata and
tar index are public official emg2pose provenance required to reproduce the
predeclared subset without downloading the 431 GiB archive locally.

## License

No open-source license has been assigned to HumanEngine yet. All rights are
reserved until the project owner chooses a license. The bundled upstream
`baseline/emg2pose` reference retains its own CC BY-NC-SA 4.0 license; datasets
and other third-party material retain their own terms.
