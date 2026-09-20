# HumanEngine v0.1 continuation-identity decisions

Date: 2026-09-20. Authority: autonomous continuation-identity closure task.

## Unified invariant

For the supported eager CPU boundary, selecting the same externally pinned
checkpoint must select the same next scientific update under identical data, RNG
and registered inputs. Continuation identity therefore binds executable model
semantics, exact named model state, independent parameter topology, optimizer
implementation and update options, optimizer state ownership by qualified name,
controller/sampler/RNG/statistics, adapter/mode and versioned scientific config.

## Chosen design

- Checkpoint/scientific-state schema v4. v1/v2/v3 fail loudly.
- Optimizers are represented by class/source identity, PyTorch version, actual
  param-group options, exact ordered qualified names and per-name state. Integer
  optimizer IDs are reconstructed only after named compatibility succeeds.
- v0.1 forbids all shared/overlapping Parameter storage and duplicate Parameter
  registration because the approved architecture has no intentional tying.
- Successful loads receive exact post-load comparison of named module state,
  continuation tensors/modes/gradients/buffers and named optimizer state.
- Diagnostics use the existing eager read-only guard plus ObservationTransaction.
  RNG and registered continuation state are restored even after normal return.

## Alternatives rejected

- Optimizer class name alone: misses implementation source and group semantics.
- Raw optimizer state_dict plus param_names: PyTorch does not use names to map
  state during load; order remains authoritative without application logic.
- Silent name-based remapping of arbitrary receiver groups: more complex and can
  hide a changed grouping contract. v0.1 rejects any mismatch instead.
- Universal storage-overlap graph support: unnecessary for the untied approved
  architecture. A fail-closed independent-storage contract is smaller and clearer.
- state_dict-only diagnostic restore: omits nonpersistent buffers and executable
  Python attributes. The registered transaction and eager guard already cover them.

## Bounded adjacent review

Current HE Macrostep has no scheduler or gradient-clipping component. Weight decay,
learning rate, betas, eps, amsgrad/maximize/foreach/fused/capturable/differentiable
and other actual optimizer group options are bound. Optimizer step counters/moments
are named state. Controller step/route, sampler state and RNG were already serialized.
Update ordering is in immutable hashed production source. Teacher preparation has no
first-class runner/scheduler; its preparation tensors, training step, optimizer and
RNG are bound, while future orchestration remains out of current implemented scope.

## Framework evidence

Installed PyTorch 2.4.1 source and official upstream documentation were inspected.
Optimizer state_dict uses integer IDs and load_state_dict zips saved IDs to current
parameters in order without ownership verification. Module load with assign=False
copies named values into existing tensors, which is why aliased receiver storages can
destroy independent checkpoint values unless topology is rejected and results checked.

