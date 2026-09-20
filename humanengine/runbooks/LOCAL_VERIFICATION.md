# Local verification boundary

The implemented CLI `python -m humanengine.verify` is synthetic CPU forward only.
Use the exact working directory, interpreter, config and flags in implementation
report Q. That next-step command was prepared, not automatically executed.

The completed correctness suite is `humanengine/tests`; JUnit results are retained
as test-results.xml. All fixtures are synthetic. The tests do not load scientific
training data, hidden/evaluator labels or S6 endpoint windows.

Keep import bytecode disabled when verifying frozen adapters. Keep pytest caches
under the new lane. Compare existing files against FROZEN_BASELINE.json; never
rewrite old manifests to make an integrity mismatch pass.

No scientific training CLI is supplied. Real teacher/HE runs require new explicit
authorization and real train-only stats, manifests, health preregistration and an
external-runtime preflight. An available code path or config file is not permission.
