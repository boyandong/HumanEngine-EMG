# Execution-integrity repair decisions

Authority: autonomous remediation request 2026-09-19; latest independent audit
D:/Temp/humanengine_remediation_reaudit/HUMANENGINE_REMEDIATION_REAUDIT.md.
Read root/lane governance, handovers, first/second audit, prior remediation/evidence,
implementation task, focused addendum and original architecture (previous turn's
readings retained). Source/adversarial behavior is decisive. No scientific redesign.

Before source edits, all five selected independent failure groups reproduced;
see EXECUTION_REPRODUCTIONS_BEFORE.json. Protected entry:321,308 historical matches,
12 changed,1 missing. No S6 attribution inferred.

## Candidates and rationale

| Root cause | Alternatives / tradeoffs | Choice |
|---|---|---|
| Transient wrong-state computation | Boundary hashes cheap but logically insufficient. functional_call/replicas isolate original objects but remain mutable inside evaluation, still need guards and disrupt custom step/window APIs. Restricted static expression language eliminates closures but duplicates approved losses and increases maintenance. | Dispatch-time read/write guard on authoritative registered tensors plus executable metadata checks at every operation; require fresh graphs produced in guarded evaluation. Reject before forbidden write or before changed semantics can compute. Higher CPU overhead accepted. |
| Incomplete executable identity | Enumerate only eps/stride (small but brittle); fingerprint actual module public/declared constants plus class code and parameter/buffer layout, validating architecture against a fresh cfg-built reference (broader, fails closed for unsupported definitions). | Actual executable fingerprint + canonical reference validation; shared by execution/checkpoints/teacher. Extensible explicit field handling rather than claiming config implies live behavior. |
| Fragile rollback | Call normal load_state_dict in reverse (hooks/manifest validation can fail); isolated subprocess per step (large serialization burden); in-process structural snapshot restoring objects directly, verify and poison on failure. | Unified transaction snapshots object dictionaries with preserved module/tensor ownership, separately restores tensors/gradients/RNG, bypasses contaminated loaders. Any restoration failure poisons continuation objects. |
| Running statistics identity | Outer seal only (self-recomputable); compare to fresh running moments (invalid for legitimate resume); nested target identity plus caller's expected checkpoint identity. | v3 content-addressed scientific identity including running state; outer reseal cannot replace it. Optional externally pinned expected identity rejects complete identity replacement. |
| Contradictory teacher provenance | More manual equality fields (duplication omissions recur); one canonical evaluation-protocol projection reused in construction/export/direct validation. | v3 certificate uses one protocol definition and derives all duplicated compatibility fields from it; exact equality enforced in all entry points. |

## Framework research (installed official source, no web claims)

PyTorch2.4.1+cpu `torch.utils._python_dispatch.TorchDispatchMode`: dynamic-scope
interception includes factories and ATen operations; nested implementation calls
run below the current mode. Used for pre-execution mutation/semantics checks.
ATen FunctionSchema alias_info.is_write inspected with add_, mul and detach probe:
.data shares protected storage and its write is interceptable. Adopted.

`torch.func.functional_call` official docstring: in-place parameter/buffer edits
are reflected in the supplied dictionary. Rejected as a sufficient immutability
mechanism alone; no claim that functional execution is intrinsically immutable.

`nn.Module.register_buffer` and `state_dict` official docstrings: nonpersistent
buffers omitted, state_dict shallow references, gradients absent. Adopted explicit
tensor/gradient/structure snapshots. `RNNBase._update_flat_weights` source: derived
flat weights refresh on changed parameter objects; preserve parameter identities
and protect public execution constants while treating flat caches as derived state.

Reference URLs for reviewers: https://pytorch.org/docs/stable/func.html ;
https://pytorch.org/docs/stable/generated/torch.func.functional_call.html ;
https://pytorch.org/docs/stable/generated/torch.nn.Module.html ;
https://github.com/pytorch/pytorch/blob/v2.4.1/torch/utils/_python_dispatch.py .
This session inspected installed versioned source/docstrings, not live web pages.

## Boundaries

Supported synthetic CPU eager PyTorch execution in the calling thread. Registered
parameters (including private), all registered buffers (persistent or not), model
execution attributes, optimizer/controller/sampler and RNG are covered. No hidden
data, external processes or arbitrary foreign-native memory sandbox. Deliberately
disabling dispatch, replacing framework code, or multi-threaded callbacks is not a
supported evaluator. Prebuilt autograd graphs are rejected; compute losses inside
the guarded closure. This strengthens the API without changing loss mathematics.

## Final implementation refinements and evidence

The executable fingerprint includes unknown private behavior attributes except
explicitly handled framework bookkeeping, actual parameter layouts, production
Python helper-source hashes, and PyTorch version. RNN flat caches must reference
registered weights. Module/state-dict/parameter hooks and callable instance overrides
are unsupported and rejected. Parameter storage, dtype/shape/stride/offset and
trainability are checked; rollback retains original storage and gradient objects.
Frozen teacher fingerprints project the eval mode actually enforced by export.

Continuation records include every registered buffer, gradient and training flag.
Duplicate persistent-buffer and trainability/mode representations must agree, even
when an internally new identity is generated. Factory evaluation_modules supports
explicit registration of teacher/other participating modules. No scientific loss,
routing, quota, burn-in or CAGrad formula was changed in this task.

Final result:169/169 CPU tests (96previous+73new), plus12 unchanged independent-oracle
groups re-executed by remediation. See EXECUTION_REMEDIATION_REPORT.md and the
machine evidence. No fresh independent acceptance or training authorization is
implied. Final scope is READY FOR FRESH INDEPENDENT RE-AUDIT.
