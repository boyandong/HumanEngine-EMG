# Security and research-data boundary

Do not commit credentials, private SSH keys, raw participant recordings, hidden
evaluation labels, annotator payloads, or model checkpoints. Use immutable data
manifests and content identities for scientific artifacts stored outside Git.

The v0.1 execution guarantee is limited to the support boundary documented in
`humanengine/CONTINUATION_CLOSURE_REPORT.md`. It is not an arbitrary-Python
sandbox and does not claim compile, distributed, native mutation, or hostile
runtime safety.
