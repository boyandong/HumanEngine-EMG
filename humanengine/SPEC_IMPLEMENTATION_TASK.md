TASK — IMPLEMENT HUMANENGINE-EMG v0.1 SCIENTIFIC KERNEL

Suggested session name:

HUMANENGINE-IMPLEMENTATION-20260919-HECORE-V01


============================================================
0. ROLE
============================================================

You are the implementation owner of the HumanEngine-EMG v0.1 scientific core.

The architecture-design phase is finished.

Your task is NOT to redesign HumanEngine.

Your task is to faithfully implement the most scientifically important
algorithms, interfaces, gradient-routing rules, causal contracts and
diagnostic tests defined by the approved design documents.

This is a research implementation task.

Correct scientific semantics are more important than:

- shorter code
- prettier abstractions
- lower pose error
- convenience
- conventional deep-learning defaults
- "what is usually done"

Do not silently improve, simplify, merge or reinterpret the architecture.


============================================================
1. PRIMARY SCIENTIFIC INTENT
============================================================

HumanEngine is NOT primarily a better emg2pose model.

Its long-term purpose is:

raw human signals
→ reusable HumanState representation H
→ stable explicit APIs

The central research objective is:

train on Dataset/API set A
→ retain broadly reusable information in H
→ transfer to Dataset/API set B
with the core frozen or with minimal controlled adaptation.

Therefore:

A change that improves pose accuracy while making H less reusable for
future datasets or APIs is NOT automatically an improvement.

The implementation must preserve our ability to test:

raw EMG → new API

frontend F → new API

handcrafted amplitude path E → new API

frozen EMG teacher representation → new API

shared S → new API

residual R → new API

full H=[S,R] → new API

Cross-dataset / new-API transfer is a primary falsification criterion.


============================================================
2. PROJECT ROOT
============================================================

Project root:

C:\Users\董伯言\Desktop\fingers\emg2pose_handstate

ASCII HDF5 junction:

C:\emg2pose


============================================================
3. REQUIRED READING — DO THIS BEFORE CODING
============================================================

Read these files carefully before implementing anything:

1. root AGENTS.md

2. humanengine_design/
   FOCUSED_REVIEW_MULTI_API_RESIDUAL.md

3. humanengine_design/
   FOCUSED_REVIEW_HANDOVER.md

4. humanengine_design/
   HUMANENGINE_EMG_V0_1_ARCHITECTURE.md

5. humanengine_design/
   SESSION_HANDOVER.md

Also inspect the actual repository interfaces referenced by those documents,
especially:

- baseline/emg2pose
- representation/
- inference/
- current data loaders / masks / manifests
- canonical16 implementation
- existing frozen evaluation code

Do not rely on remembered architecture descriptions when the real repository
can answer the question.


============================================================
4. AUTHORITATIVE PRECEDENCE RULE
============================================================

FOCUSED_REVIEW_MULTI_API_RESIDUAL.md is an ADDENDUM to the original
HUMANENGINE_EMG_V0_1_ARCHITECTURE.md.

It overrides the original architecture on EXACTLY these topics:

1. multi-API supervision / family budgeting / continual API learning
2. residual learning target
3. explicit API readout and gradient routing involving R

For unrelated topics, the original architecture remains authoritative.

IMPORTANT EXAMPLES:

OLD DESIGN:
future pose reads H and directly updates R.

NEW DESIGN:
every explicit API reads:

concat(S, stopgrad(R))

and explicit API supervision does NOT directly update R.

THE NEW RULE WINS.


OLD DESIGN:
RMS / coarse spectral masked reconstruction is the main residual objective.

NEW DESIGN:
the main residual target is an independently pretrained and then frozen
multi-scale EMG teacher latent representation.

RMS / coarse spectral summaries remain only as weak observed physical anchors.

THE NEW RULE WINS.


Do NOT merge incompatible old and new rules.

Do NOT "average" them.

Do NOT preserve old future→R gradients while also claiming to implement the
new detached-R design.


============================================================
5. UNCERTAINTY / RESEARCH ESCALATION PROTOCOL
============================================================

When implementation details are uncertain, use the following hierarchy.

STEP 1 — FOCUSED REVIEW

First inspect:

FOCUSED_REVIEW_MULTI_API_RESIDUAL.md

If it explicitly defines the relevant rule, implement that rule.


STEP 2 — ORIGINAL REPORT

If the focused review does not define the issue, inspect:

HUMANENGINE_EMG_V0_1_ARCHITECTURE.md

and the relevant handover documents.


STEP 3 — REAL REPOSITORY

If documents refer to existing behavior, inspect the actual implementation,
configuration, masks, schemas and frozen interfaces.

Real code wins over stale assumptions about repository behavior.


STEP 4 — EXTERNAL RESEARCH

If the architecture documents and repository still leave an important
scientific or mathematical ambiguity:

search the literature / web.

Prefer:

1. original peer-reviewed paper or primary preprint
2. official author/project page
3. official repository / reference implementation
4. primary documentation

Examples of relevant sources include:

- CAGrad
- data2vec
- VICReg
- PCGrad
- GradNorm
- FAMO
- relational KD
- continual learning / replay
- EMG representation learning

Do NOT use blog summaries as the primary authority for mathematical behavior.

When external research is used, record:

- source
- exact principle used
- whether it matches the published method or is a HumanEngine modification
- why it resolves the ambiguity


STEP 5 — IF STILL UNCERTAIN

If uncertainty remains and the choice would materially alter:

- scientific interpretation
- gradient routing
- causal behavior
- information preservation
- validation fairness
- family budgeting
- teacher targets
- data leakage
- reproducibility

DO NOT GUESS.

Document the ambiguity precisely.

Stop that specific implementation branch.

Continue implementing unaffected parts when safe.

At final handoff, clearly report:

BLOCKED DESIGN QUESTION

rather than inventing an answer.


============================================================
6. SOURCE / INFERENCE / IMPLEMENTATION DISTINCTION
============================================================

For any nontrivial design decision, distinguish:

A. SOURCE-DEFINED RULE
directly specified by the HumanEngine design documents.

B. REPOSITORY FACT
verified from existing project code/data/config.

C. LITERATURE-DERIVED PRINCIPLE
taken from an external paper/reference.

D. IMPLEMENTATION CHOICE
engineering decision needed to realize the above.

Do not present D as if it were A or C.

Do not turn a convenient implementation choice into a scientific claim.


============================================================
7. FROZEN SCIENTIFIC ARTIFACTS
============================================================

DO NOT MODIFY:

- baseline/
- frozen VEMG2Pose code
- existing VEMG2Pose checkpoints
- existing emg2pose preprocessing
- official pose-valid / IK-invalid semantics
- representation/canonical_hand.py
- canonical16 frozen contract
- DB9 artifacts
- S3 artifacts
- S4 artifacts
- S5 artifacts
- S6 artifacts
- historical frozen scientific results/manifests

New HumanEngine implementation belongs in a NEW lane:

humanengine/

Recommended structure:

humanengine/
    contracts.py
    api.py
    registry.py
    manifest.py

    core/
    data/
    heads/
    objectives/
    teacher/
    optimization/
    continual/
    diagnostics/
    configs/
    tests/
    runbooks/

Do not initialize/restructure Git as a side effect.


============================================================
8. WINDOWS / HDF5 PATH RULE
============================================================

The non-ASCII Windows path can break h5py.

When accessing files through:

C:\emg2pose

preserve this ASCII path through the actual h5py open call.

DO NOT call Path.resolve() in a way that dereferences the junction back into:

C:\Users\董伯言\...

before h5py opens the file.

If the new data layer touches HDF5, add a regression test/helper contract.


============================================================
9. IMPLEMENTATION GOAL — SCIENTIFIC KERNEL FIRST
============================================================

This task should prioritize the SCIENTIFIC KERNEL.

Implement fully:

1. contracts / state structures
2. causal frontend F
3. correct A → F → U ordering
4. dual recurrent S/R
5. H=[S,R]
6. fixed amplitude path E
7. detached-R API routing
8. independent EMG teacher
9. teacher preparation objective
10. multi-scale frozen-teacher residual objective
11. weak physical signal anchor
12. API-family registry and budgeting
13. hierarchical family sampling
14. weighted-base CAGrad
15. private-vs-core parameter update semantics
16. replay/distillation CONTRACT
17. gradient diagnostics
18. causality / leakage / routing tests
19. provenance / checkpoint compatibility

Do NOT spend major effort yet on:

- UI
- visualization
- dashboards
- production deployment
- huge experiment runners
- full continual-learning storage infrastructure
- hyperparameter sweeps
- scientific training on the real dataset

For replay and future APIs, implement the interfaces/contracts needed for
future use, not a giant unused system.


============================================================
10. CORE DATA FLOW — EXACT ORDER
============================================================

The correct conceptual path is:

published/preprocessed EMG x
        │
        ├──────────────→ E(x)
        │                 fixed causal amplitude summary
        │                 computed BEFORE A/front-end normalization
        │
        ↓
A_u
input/channel affine personalization
        ↓
F
shared causal local frontend
        ↓
U_u
rank-8 feature adapter
        ↓
        ├──────────────→ S recurrence
        │                 hidden 256
        │
        └──────────────→ R recurrence
                          hidden 128
                          also receives E


Therefore:

f_t = U_u(F(A_u(x_<=t)))

e_t = E(x_<=t)

S receives f_t

R receives concat(f_t, e_t)

H_t = concat(S_t, R_t)


IMPORTANT:

U is AFTER F.

Do not place U directly on raw EMG.

A is the input affine adapter.

U is the feature adapter.


Universal training:

A = identity
U = identity
A/U frozen

Personalization stage:

core frozen
only explicitly authorized A/U parameters may update.


============================================================
11. H IS NOT THE COMPLETE RUNTIME STATE
============================================================

H = [S,R]

is the learned representation exposed to probes/APIs.

Complete RuntimeState also contains:

- S hidden/cell
- R hidden/cell
- convolution caches
- E summary buffers
- stride phase
- timestamps
- warmup/reset/quality state

Do not claim H alone is:

- a complete Markov state
- a Bayesian belief state
- a sufficient statistic
- full physiology


============================================================
12. CAUSAL FRONTEND
============================================================

Implement a genuinely causal frontend.

Follow the original architecture specification for:

- causal convolution structure
- stride/tick semantics
- left padding
- receptive-field alignment
- no future interpolation

Support:

forward_sequence(...)

and:

step(...)

They must agree.

No future EMG may affect outputs at time t.

No full-history recomputation inside streaming step().

No module-global recurrent state.


============================================================
13. STREAMING API
============================================================

Conceptual interface:

state = HE.init_state(
    batch_size,
    adapter_id,
    device
)

outputs, state = HE.step(
    emg_chunk,
    state,
    timestamps
)

Outputs may include:

- timestamp
- H
- shared S
- residual R
- registered API outputs
- warmup
- input quality
- model version
- adapter version

Do NOT output invented:

- force
- contact
- fatigue
- activation
- intent
- confidence probabilities

unless trained/validated APIs exist.


============================================================
14. RESET / STREAM SEMANTICS
============================================================

Reset state on:

- recording discontinuity
- invalid timestamp ordering
- user change
- adapter change
- incompatible input schema
- explicit reset event

Do NOT reset solely because pose labels are missing.

Training labels are not runtime sensor state.


============================================================
15. EXPLICIT API ROUTING — NEW AUTHORITATIVE RULE
============================================================

For every explicit API family i:

Z_i = P_i(
    concat(
        S,
        stopgrad(R)
    )
)

Default private projector candidate:

384 → 128 → SiLU

then API-specific output head(s).


Pose family includes:

- current pose
- future pose
- pose geometry

Current/future may share P_pose.

Pose geometry remains its special G(S) route as defined in the report.


CRITICAL SCIENTIFIC RULE:

Explicit API losses may READ R numerically.

Explicit API losses must NOT directly update R recurrent parameters.

Thus API loss updates:

F through the S path: YES

S: YES

R through API readout: NO

P_i/head: YES


However:

F is shared.

An API update can change F,
which changes the future numerical inputs seen by R.

Therefore:

stopgrad(R) does NOT mean
"API supervision cannot influence residual behavior."

Do not make that claim.


============================================================
16. SHADOW-GRADIENT DIAGNOSTIC
============================================================

Implement an optional DIAGNOSTIC ONLY path that removes the R stop-gradient
and measures the fully differentiable API sensitivity through R.

This shadow gradient:

- must never update training parameters
- must not alter optimizer state
- must not change RNG
- must not change caches/state
- must be clearly labeled diagnostic

Purpose:

quantify what gradient path is being intentionally blocked.


============================================================
17. POSE FAMILY
============================================================

Pose is currently the only explicit API family.

This is a DATA REALITY,
not permanent architectural privilege.

Pose family contains:

current pose
future pose
geometry regularization

Initial internal beta:

current = 1.0

future = 0.5

geometry = 0.05

Future horizons are averaged INSIDE the future component.

Geometry does NOT receive a separate API-family vote.


============================================================
18. API FAMILY REGISTRY
============================================================

Implement APIRegistry.

At minimum store:

- family_id
- semantic version
- family components
- source/acquisition IDs
- valid-mask definitions
- frozen reference scales
- beta weights
- family base budget
- projector/head IDs
- replay policy
- route version

Family boundaries must be explicit and versioned.

A dataset name is NOT an API family.


============================================================
19. FAMILY-BALANCED SAMPLING
============================================================

Training exposure hierarchy:

macrostep:

1. allocate active API-family quotas
2. choose data source within each family
3. choose user/session/recording
4. choose time window / sequence

Large datasets must not get extra optimizer steps simply because they contain
more samples.

Five pose datasets still represent ONE pose family.

Additional pose datasets may improve coverage INSIDE pose,
but do not create additional pose votes.


Stable acquisition/source identity must prevent:

- copying one dataset
- renaming one dataset
- arbitrarily splitting one dataset

from silently increasing its family influence.


============================================================
20. SAME-PARAMETER-SNAPSHOT MACROSTEP
============================================================

This is important.

For one multi-family macrostep:

all active family gradients must be evaluated from the SAME shared-core
parameter snapshot.

Do NOT:

update pose core parameters,
then compute force gradient,
then update contact,
then call that "CAGrad multi-task coordination".

Correct conceptual process:

theta_snapshot

→ compute family microbatch losses/gradients for every active family

→ compute reserve gradient

→ combine/coordinate gradients

→ perform ONE shared-core update


Private heads may have their own update semantics as specified later.


============================================================
21. FAMILY LOSS NORMALIZATION
============================================================

For family i component j:

ell_ij =
    sum(valid_ij * point_loss_ij)
    /
    sum(valid_ij)

L_i =
    (
        sum_j beta_ij * ell_ij / a_ij
    )
    /
    sum_j beta_ij


Missing label:

NA

NOT ZERO.


a_ij:

frozen BEFORE training.

Do not use:

- current batch loss
- current gradient norm
- moving task difficulty

as a_ij.


For regression:

use train-only physical/statistical scaling and declared constant-predictor
reference.

For classification:

use constant/prevalence reference when scientifically appropriate.

Near-zero reference scale:

use registered floor
or mark invalid.

Never divide by an accidental near-zero number.


============================================================
22. REPRESENTATION RESERVE FAMILY
============================================================

Besides explicit API families, retain one reserve objective family.

Initial focused-review formulation:

L_reserve =
    (
        L_R / a_R
        +
        0.1 * L_VICReg / a_V
    )
    /
    1.1


For K explicit API families:

alpha_i = 0.8 / K

alpha_reserve = 0.2


L_base =
    sum_i alpha_i * L_i
    +
    alpha_reserve * L_reserve


Initial:

a_V = 1

a_R =
frozen reference scale derived from teacher-target baseline.


All constants must be configurable and versioned.

Do not call them optimal or theoretically fundamental.


============================================================
23. WHY FAMILY BALANCING EXISTS
============================================================

The implementation must preserve this property:

pose datasets = 5
force datasets = 1

does NOT imply:

pose gets 5× the family influence.


Scientific importance is explicitly assigned at the family level.

Dataset count and sample rate must not silently define biological importance.


============================================================
24. INDEPENDENT EMG TEACHER — PURPOSE
============================================================

R must not be defined only by today's pose supervision.

The primary residual target is now a generic EMG-only learned representation.

The teacher is NOT:

- physiology truth
- force truth
- fatigue truth
- universal sufficient EMG representation

It is a broader, independently trained information-preservation proxy.


============================================================
25. TEACHER ARCHITECTURE
============================================================

Implement a SEPARATE teacher training lane.

Candidate:

causal local frontend
+
2-layer LSTM
hidden size 128 per layer


Student:

T_s

trainable.


EMA teacher:

T_bar

updates ONLY from T_s.

Initial EMA default:

T_bar =
    0.999 * T_bar
    +
    0.001 * T_s


This value is configurable.


Teacher must NOT share with HumanEngine:

- module objects
- parameters
- initialization checkpoint
- optimizer
- EMA
- normalization statistics
- predictor
- state cache


============================================================
26. TEACHER SUPERVISION CONTAMINATION RULES
============================================================

Teacher preparation must NOT use:

- pose labels
- pose pseudo-labels
- gesture labels
- force labels
- future API labels
- pose accuracy for checkpoint selection
- downstream HE accuracy for checkpoint selection

Do not use pose-valid masks to discard all otherwise valid EMG.

Teacher selection must remain API-label independent.


============================================================
27. TEACHER MULTI-SCALE TARGETS
============================================================

Focused-review candidate scales:

tau =
{
    0.1 s,
    0.5 s,
    2.0 s
}

All windows are CAUSAL and terminate at t.

No future EMG.

For each temporal scale k
and recurrent layer l:

u_(k,l,t) =
    T_EMG.layer_l(
        x[t-tau_k, t],
        reset_at_left
    )


Use BOTH teacher recurrent layers.

Do not average them into one target.

Six targets total:

3 scales
×
2 layers

each target dimension:

128


============================================================
28. TEACHER TARGET STANDARDIZATION
============================================================

After teacher preparation and freeze:

y_(k,l,t) =
    stopgrad(
        (
            u_(k,l,t) - mu_(k,l)
        )
        /
        max(
            sigma_(k,l),
            0.05
        )
    )


mu/sigma:

- train partition only
- per target dimension
- across samples
- frozen
- versioned
- never computed per-window


Do not use normalization to hide teacher collapse.

Teacher health must be checked BEFORE relying on normalized targets.


============================================================
29. TEACHER SELF-SUPERVISION PREPARATION OBJECTIVE
============================================================

Implement the focused-review candidate teacher objective.

Teacher student sees masked / observed causal windows.

EMA teacher sees clean causal windows.

Teacher student predictors:

for each recurrent layer:

128 → 128 → 128

same-layer predictor may be shared across temporal scales as specified.

Teacher preparation loss contains:

1. masked latent prediction
2. 0.1 observed latent prediction
3. 0.05 observed physical-summary prediction
4. 0.01 direct hidden variance floor


Huber-based latent targets.

Teacher summary head:

128 → 128 → summary_dim

reads second-layer clean student hidden.

Teacher physical summary head is separate from the HumanEngine physical-anchor
head.


============================================================
30. TEACHER MASKING PROTOCOL
============================================================

Do NOT reduce this to generic random masking.

Implement the focused-review initial masking contract.

Start with approximately:

30% of temporal support masked.

Scored windows should typically contain:

20%–50% masked support.


Mask must include blocks near t.

Initial block width:

approximately 20% of the shortest causal context.

Remaining masked blocks may be distributed through the causal history.


All teacher scales must be computed from the SAME actual EMG masking view where
the focused review requires it.


All student paths must be recomputed from their own masked input.

Do not reuse clean:

- features
- caches
- amplitude path
- recurrent state


============================================================
31. VISIBLE-CONTEXT TEACHER BASELINE
============================================================

Teacher preparation must support comparison to simple weak references,
including a visible-context predictor.

Low masked-latent error is not impressive if the target can be trivially
estimated from the still-visible context.

Record appropriate baseline errors.

Do not call masked recovery successful solely because the training loss is low.


============================================================
32. TEACHER HEALTH GATES
============================================================

Before a teacher can become T_EMG for HumanEngine:

support checks for:

- better than constant target predictor
- better than simple context-only reference where relevant
- observed physical-summary readout above constant reference
- non-near-constant raw hidden
- meaningful within-recording variation
- amplitude response not destroyed
- no obvious collapse


Teacher checkpoint selection:

fixed-budget / preregistered rule.

Do NOT choose whichever checkpoint later gives best pose/force/API performance.


If health gates fail:

teacher preparation = FAILED.

Do not hide failure by increasing HumanEngine capacity.


============================================================
33. HUMANENGINE RESIDUAL TARGET
============================================================

After teacher health gates pass:

freeze its EMA encoder as:

T_EMG

Discard teacher training predictors.

Teacher becomes training-only target generator.

It is NOT part of deployed HE runtime.


============================================================
34. RESIDUAL REPRESENTATIONS
============================================================

Let:

r_t^m =
residual from MASKED HumanEngine input

r_t^o =
residual from OBSERVED/CLEAN HumanEngine input


HumanEngine residual predictor:

P_R:

128 → 128 → 768


Interpret 768 as:

3 scales
×
2 teacher layers
×
128 dims


Masked and observed residuals use the SAME P_R.


P_R must not receive:

- S
- H
- API label
- teacher target as input
- clean hidden from another view
- user ID
- session ID


============================================================
35. EXACT RESIDUAL LOSS
============================================================

Implement:


L_mask_lat =

mean over valid (k,l,t):

    mean_dim
    Huber(
        P_R(r_masked)_(k,l),
        y_(k,l,t)
    )


L_obs_lat =

mean over valid (k,l,t):

    mean_dim
    Huber(
        P_R(r_observed)_(k,l),
        y_(k,l,t)
    )


L_phys =

mean over valid summaries:

    Huber(
        M_A(r_observed),
        summary(clean_EMG)
    )


L_var_R =

mean_dim

relu(
    0.1
    -
    sqrt(
        Var_batch(r_observed)
        +
        1e-4
    )
)


Then:


L_R =

L_mask_lat
+
0.1 * L_obs_lat
+
0.05 * L_phys
+
0.01 * L_var_R


Huber delta = 1.


Each scale/layer target:

average independently,
then equal-weight combination.


Masked / observed / physical terms:

independent valid counts and denominators.


Do not let a larger number of clean frames silently increase the observed
term's effective weight.


============================================================
36. RESIDUAL VARIANCE SUPPORT
============================================================

L_var_R activates only with sufficient support.

Initial review thresholds:

>=64 temporally separated anchors
>=8 recordings
>=4 users


If insufficient:

return explicit NA / skipped status.

Do NOT silently return a meaningful-looking zero.


============================================================
37. DO NOT ADD RESIDUAL DECORRELATION BY DEFAULT
============================================================

Do NOT automatically add:

- covariance whitening
- Barlow-style decorrelation on R
- orthogonality S⊥R
- mutual-information minimization
- adversarial subject removal
- generic disentanglement losses


Weak variance floor is only an anti-constant-collapse mechanism.

High effective rank does NOT prove biological usefulness.

Correlation reduction does NOT prove statistical independence.


============================================================
38. PHYSICAL SIGNAL ANCHOR
============================================================

Retain the original deterministic causal signal summaries:

- short-window log-RMS
- coarse log-band power

But their NEW ROLE is:

weak observed anchor

NOT primary residual representation target.


Use the original architecture report for the exact initial summary definition,
frequency-band formulation and causal support rules.


Do not call these:

- force
- activation
- physiology
- fatigue
- denoised ground truth


============================================================
39. NUISANCE INVARIANCE BOUNDARY
============================================================

Keep the original weak measurement-noise VICReg on S projection.

Do NOT add another default R-invariance family.

Default forbidden strong assumptions include:

- strong gain invariance
- channel permutation invariance
- sign inversion
- strong time warp
- per-window whitening
- aggressive frequency deformation
- generic subject adversary


These may delete real physiology.

Only use nuisance assumptions with specific evidence.


============================================================
40. CLEAN / MASKED / NOISY VIEW ISOLATION
============================================================

This is a critical correctness requirement.

Every view must have independent:

- input tensor
- A/F/U processing
- E path
- convolution caches
- S/R states
- summary buffers


Masked HumanEngine must never reuse:

- clean F
- clean U
- clean E
- clean R
- clean caches
- clean teacher target as model input


Teacher targets are loss targets only.


============================================================
41. ADVERSARIAL TARGET-LEAKAGE TEST
============================================================

Construct two synthetic examples such that:

masked-visible input is IDENTICAL

but:

clean hidden target / clean teacher target differs.


Required behavior:

masked HumanEngine student output must be identical.

Targets may differ.

Losses may differ.


If masked outputs differ:

clean information leaked into the student path.

The run is scientifically invalid.


============================================================
42. CAGrad — WHY IT EXISTS
============================================================

CAGrad is not a declaration of API importance.

API importance is set by explicit family budgets.

CAGrad only coordinates conflicting shared-core gradient directions.

Do not let the optimizer redefine scientific priority.


============================================================
43. CORE PARAMETER VECTOR FOR CAGrad
============================================================

Shared universal core:

theta_c =
(
    F,
    S,
    R
)


A/U frozen during universal training.

Exclude from the CAGrad vector:

- API private projectors
- API private heads
- residual predictor P_R
- physical anchor head M_A
- VICReg projector V
- pose geometry projector G
- teacher parameters


For each family gradient g_i:

represent it over the FULL theta_c vector.

If that family does not update one block:

that block's entries are ZERO.


Example:

API family:

F via S = gradient
S = gradient
R = zero

reserve:

contains gradients according to its actual R/VICReg routes.


============================================================
44. ONE CAGrad SOLVER — NOT ONE PER BLOCK
============================================================

Do NOT solve separate CAGrad problems for:

F
S
R


Use ONE shared-core gradient vector.

Blockwise gradient norms/cosines are diagnostics only.


============================================================
45. WEIGHTED-BASE CAGrad
============================================================

Base gradient:

g0 =
    sum_i alpha_i * g_i
    +
    alpha_reserve * g_reserve


Target direction:

d* =
    argmax_d
        min_j <g_j, d>

subject to:

||d - g0||_2
<=
c * ||g0||_2


Initial policy:

K = 1 explicit API family:
    c = 0

K >= 2:
    c = 0.25


Therefore with only pose today:

do NOT pay unnecessary gradient-surgery complexity.

c is configurable.


============================================================
46. CAGrad IMPLEMENTATION RESEARCH RULE
============================================================

This is a HumanEngine weighted-base extension of CAGrad,
not necessarily a verbatim original implementation.

Before coding the solver:

read the focused review.

If mathematical details remain ambiguous:

read the original CAGrad paper / official implementation.

Record the exact derivation used.


The focused review discusses the dual form.

Implement numerical checks against the primal trust-region condition.


Do NOT simply add epsilon to every degenerate denominator and call the solver
valid.


============================================================
47. CAGrad DEGENERATE CASES
============================================================

Handle explicitly:

- g0 norm = 0
- family gradient norm = 0
- multiple optimal solutions
- numerical solver failure
- missing family gradient
- non-finite gradient
- constraint violation


If no trustworthy solution is available:

fallback:

d = g0

and LOG THE FALLBACK.


Do not silently pretend CAGrad succeeded.


============================================================
48. PRIVATE PARAMETER UPDATES
============================================================

Private parameters are NOT part of CAGrad competition.

API projector/head:

updated by own family loss.

Residual predictor P_R:

updated by L_R.

Physical anchor head:

updated by its residual objective.

VICReg projector:

updated by VICReg.

Pose geometry projector:

updated by geometry objective.


Increasing number of API families must not mechanically shrink private-head
learning rates.


Do not replace the intended semantics with:

total_loss.backward()

for every parameter.


============================================================
49. OPTIMIZER REALITY
============================================================

CAGrad produces an intended gradient-space core direction.

If actual optimization later uses:

- Adam
- AdamW
- momentum
- weight decay
- gradient clipping

actual parameter delta differs from:

-eta * d


Implement diagnostics for:

- g_j dot d
- actual delta_theta
- g_j dot delta_theta
- fixed diagnostic-batch pre-update loss
- same-batch post-update loss


Do not claim:

g_j dot d > 0

guarantees real family loss improved.


============================================================
50. DO NOT STACK MULTI-TASK OPTIMIZERS BY DEFAULT
============================================================

Default path:

family budgeting
+
weighted-base CAGrad when K>=2


Do NOT simultaneously stack:

- PCGrad
- GradNorm
- FAMO
- Nash-MTL


If future diagnostics justify comparison,
that becomes a separate experiment.


============================================================
51. FUTURE API ADDITION — HEAD-ONLY WARM ENTRY
============================================================

When a genuinely new API arrives in the future:

do not immediately allow a random head to pull the shared core.

Required conceptual lifecycle:

1. register new API family
2. instantiate private projector/head
3. freeze HumanEngine core
4. perform short head-only fitting
5. verify:
   - finite outputs
   - better than appropriate constant reference
6. then enable registered joint multi-family training


This is an API onboarding stage,
NOT pose-weight decay.


============================================================
52. REPLAY / CONTINUAL API CONTRACT
============================================================

There is currently only pose.

Do not invent replay data now.

But implement the CONTRACT needed later:

- APIRegistry versions
- ReplayManifest
- OldAPIArtifact
- old output definition/version
- distillation target interface
- old-domain evaluation hooks


When new APIs are added:

save the previous old-API model as read-only teacher.


Old API retention candidate:

L_i_ret =
    L_i_label
    +
    0.2 * L_i_distill

after independent normalization.


Distillation remains inside the same old family budget.

It does NOT receive another family vote.


============================================================
53. DO NOT DISTILL THE WHOLE OLD H
============================================================

Replay/distillation should preserve:

old API behavior / compatibility

NOT freeze:

H_new ≈ H_old


HumanEngine internal representation must remain free to evolve for new APIs.


============================================================
54. REPLAY DOMAIN REQUIREMENT
============================================================

Replay should contain representative OLD-DOMAIN TRAINING inputs.

Using only new-domain inputs with old-model output distillation does not prove
old-domain retention.

Never use:

- final test
- S6 hidden/evaluator data

as replay.


============================================================
55. EXACT GRADIENT ROUTING TESTS
============================================================

For each objective independently,
use autograd.grad / isolated backward and assert reachability.


EXPLICIT API CURRENT:

F through S:
YES

S:
YES

R:
NO

private API projector/head:
YES


EXPLICIT API FUTURE:

F through S:
YES

S:
YES

R:
NO


POSE GEOMETRY:

F:
YES

S:
YES

R:
NO


VICReg:

F:
YES

S:
YES

R:
NO


MASKED LATENT:

F through R:
YES

S:
NO

R:
YES

P_R:
YES


OBSERVED LATENT:

F through R:
YES

S:
NO

R:
YES


PHYSICAL ANCHOR:

F through R:
YES

S:
NO

R:
YES


R VARIANCE:

F through R:
YES

S:
NO

R:
YES


TEACHER:

NO HumanEngine gradient.


A/U during universal training:

NO update.


Also test:

no stale gradient from the previous objective.


============================================================
56. FAMILY-VOTE INVARIANCE TEST
============================================================

Implement the focused-review E0 property test.

Create one logical pose acquisition source.

Case A:
register once.

Case B:
represent the SAME acquisition as five renamed/copied logical dataset entries
with the same stable source identity.


With identical sampling/random stream:

these must remain unchanged:

- pose family quota
- total pose family exposure
- alpha_pose
- family contribution
- base gradient contribution g0


If five aliases increase pose influence:

implementation is INVALID.


============================================================
57. CAUSALITY TESTS
============================================================

Must include:

1. perturb future EMG
   → past H unchanged

2. perturb future EMG
   → past API output unchanged

3. forward_sequence
   ==
   arbitrary step() chunking

4. different chunk partitions
   → equivalent output

5. reset removes all sequence state

6. adapter/user switch triggers correct reset

7. missing pose label does NOT reset runtime

8. EMG stream discontinuity DOES reset runtime

9. future target never crosses recording boundary

10. teacher causal targets never use >t EMG

11. masked HumanEngine never uses clean state

12. no future interpolation leakage


============================================================
58. REPRESENTATION DIAGNOSTICS
============================================================

Provide interfaces to export / probe:

- raw causal EMG
- frontend F
- amplitude path E
- teacher target representation
- S
- R
- H


Every exported representation should carry:

- timestamp
- user
- session
- recording
- side
- representation version
- checkpoint hash
- teacher hash where relevant


Do NOT run expensive scientific probes in this task.


============================================================
59. CRITICAL INFORMATION-LOSS DIAGNOSTIC LOGIC
============================================================

Future experiments must be able to distinguish:


CASE A:

raw new-API probe strong

teacher new-API probe weak

→ teacher target itself lost useful information.


CASE B:

teacher strong

R weak

→ teacher→R compression/training lost information.


CASE C:

R strong

H/API weak

→ API readout/routing failed to exploit retained information.


CASE D:

raw strong

F weak

→ shared frontend F is the information bottleneck.


CASE E:

S ≈ H on every current/future/new task

→ residual may be redundant or unused.


CASE F:

E alone explains R/H improvement

→ learned residual may mostly copy handcrafted amplitude summaries.


The code architecture must preserve our ability to measure these distinctions.


============================================================
60. REPRESENTATION HEALTH
============================================================

Support diagnostics for:

- per-dimension std
- dead dimensions
- covariance spectrum
- effective rank
- within-recording variation
- within-user variation
- cross-user variation
- device/session identity probes


But:

high rank ≠ useful physiology

low device classification ≠ nuisance successfully removed

high device classification ≠ automatically invalid


Do not optimize identity away without scientific evidence.


============================================================
61. CONFIGURATION
============================================================

Create explicit configs for at least:

teacher_emg_ssl_v01

he_core_v01

he_full_v01

he_routing_test

he_family_budget_test

he_cagrad_test


Scientific constants must be in config/manifests,
not buried in code.


Record:

- dimensions
- teacher scales
- loss weights
- beta values
- reserve alpha
- CAGrad c
- stopgrad route version
- family registry
- teacher target version
- physical summary version
- data manifest hash
- statistics hash
- code/source version where available


============================================================
62. CHECKPOINT PROVENANCE
============================================================

Teacher checkpoint must include:

- student model
- EMA teacher
- optimizer
- training step
- RNG
- target statistics
- configuration
- config hash
- training-data manifest hash
- source/code version
- health-check results


HumanEngine checkpoint must include:

- F
- S
- R
- A/U state where relevant
- E contract/version
- API registry
- private projectors/heads
- residual predictor
- physical anchor head
- optimization/controller state
- reference scales
- RNG
- sampler state
- config hash
- data manifest hash
- route version
- teacher hash/version


============================================================
63. RESUME SAFETY
============================================================

Resume must FAIL LOUDLY when scientifically incompatible.

Examples:

- different detached-R route version
- different teacher checkpoint
- different teacher target definition
- different API family registry
- different reference-scale manifest
- different physical-summary definition
- incompatible dataset manifest


Do not "best effort" resume incompatible scientific experiments.


============================================================
64. CANONICAL16 / EXISTING ADAPTERS
============================================================

Do not rewrite frozen canonical16 logic.

If a differentiable torch view is required:

create a new adapter/view using the exact resolved frozen indices.

Verify numerically against the frozen canonical adapter.

Do not modify the frozen implementation.


============================================================
65. POSE DATA REALITY
============================================================

Do not treat:

stage

as frame-level gesture truth.


Do not invent:

force
contact
fatigue
activation
intent
world pose


Current published pose is filtered/interpolated.

Therefore tiny future horizons may primarily reflect:

- persistence
- smoothness

not motor intent.


Keep future-pose interpretation conservative.


============================================================
66. DATA SPLIT / GROUPING
============================================================

Scientific evaluation units must respect:

- user
- session
- recording
- synchronized source grouping
- side
- leakage boundaries


Do not split adjacent windows from the same source across supposedly independent
sets.


Mini dataset is only for:

- schema checks
- unit tests
- interface checks
- tiny synthetic/smoke behavior

It cannot establish cross-user HumanEngine performance.


============================================================
67. LOCAL EXECUTION POLICY
============================================================

DO NOT run substantial deep-learning training locally.

Allowed:

- repository inspection
- schema inspection
- CPU forward/backward
- tiny synthetic examples
- tiny deterministic overfit sanity test if lightweight
- unit tests
- causality tests
- gradient-routing tests
- teacher/HE isolation tests
- family-budget tests
- CAGrad toy numerical tests
- configuration validation
- parameter/memory estimates


Not authorized:

- real teacher pretraining
- HE_FULL training
- GPU training
- hyperparameter sweeps
- remote jobs


Prepare the code.

Do not launch the scientific experiment.


============================================================
68. IMPLEMENTATION ORDER
============================================================

Do work in this order.


PHASE 0 — SPEC RECONCILIATION

Read all authoritative documents.

Write a short internal implementation map:

requirement
→ source document section
→ target module/test

Identify any contradictions BEFORE coding.


PHASE 1 — CONTRACTS / REGISTRY / CONFIG

Implement:

- HEBatch
- RuntimeState
- HumanStateOutput
- LossBundle
- APIRegistry
- family/source IDs
- config/provenance schema


PHASE 2 — CAUSAL CORE

Implement:

A
F
U
E
S
R
H

sequence + streaming APIs.


PHASE 3 — API ROUTING

Implement:

P_i([S, stopgrad(R)])

pose family
private projector/head
shadow-gradient diagnostic.


PHASE 4 — TEACHER

Implement:

independent teacher
EMA
multi-scale targets
masking
teacher losses
health-check interfaces.


PHASE 5 — RESIDUAL OBJECTIVE

Implement:

P_R
L_mask_lat
L_obs_lat
L_phys
L_var_R
L_R


PHASE 6 — FAMILY TRAINING SYSTEM

Implement:

- family loss normalization
- fixed reference scales
- hierarchical family quota sampler
- same-snapshot macrostep semantics
- reserve family


PHASE 7 — CAGrad

Implement:

- full-core vector construction
- weighted base gradient
- safe solver
- fallback
- diagnostics
- private update separation


PHASE 8 — CONTINUAL CONTRACT

Implement:

- ReplayManifest
- OldAPIArtifact
- distillation interface
- new API head-only onboarding contract


PHASE 9 — DIAGNOSTICS

Implement:

- gradient norm/cosine
- shadow gradient
- effective rank
- representation export
- pre/post update diagnostics


PHASE 10 — HIGH-VALUE TESTS

Run all scientific correctness tests.


============================================================
69. TEST PRIORITY
============================================================

The highest-value tests are:

1. future-input causality

2. arbitrary chunk equivalence

3. detached-R gradient routing

4. masked-clean leakage adversarial test

5. teacher/HE parameter independence

6. family-vote invariance

7. same-parameter-snapshot family gradients

8. CAGrad trust-region constraint

9. CAGrad degenerate fallback

10. private-vs-core update separation

11. target valid-count normalization

12. resume provenance mismatch failure


Do not declare the scientific kernel done if these fail.


============================================================
70. DO NOT OPTIMIZE AROUND FAILING TESTS
============================================================

If a test reveals that the implementation does not satisfy the scientific
contract:

fix the implementation.

Do not:

- weaken the assertion
- broaden tolerance without evidence
- disable the test
- reinterpret the design after seeing failure

unless the design documents themselves are shown to be inconsistent.


============================================================
71. MINIMAL EXTERNAL RESEARCH EXPECTATION
============================================================

Do not perform a giant literature survey.

External research is only needed when:

- mathematics is ambiguous
- reference implementation behavior matters
- the report intentionally delegates a detail
- implementation correctness depends on a method's exact semantics


Examples:

CAGrad solver:
check original paper/reference code if needed.

data2vec-inspired teacher:
check original method if teacher-target/EMA semantics are unclear.

VICReg:
check exact variance/covariance formulation if current code/report is ambiguous.


Research should resolve implementation ambiguity,
not reopen architecture selection.


============================================================
72. NO ARCHITECTURE DRIFT
============================================================

Do NOT spontaneously replace:

LSTM with Transformer/Mamba

dual state with one large latent

teacher with generic foundation model

CAGrad with another optimizer

detached-R with open-R

RMS anchor with raw waveform reconstruction

family budgets with dataset-size weighting


Such changes are research hypotheses,
not implementation decisions.


============================================================
73. IF YOU FIND A BETTER IDEA
============================================================

If during implementation/research you discover a potentially better method:

DO NOT silently implement it.

Record it as:

PROPOSED FUTURE BRANCH

with:

- rationale
- literature source
- expected benefit
- risk
- what current hypothesis it replaces
- minimal counterfactual experiment

Continue implementing the approved v0.1 unless the discovered issue makes the
approved design invalid or impossible.


============================================================
74. DEFINITION OF DONE
============================================================

The scientific-kernel implementation task is complete only when:

- authoritative reports were read
- precedence was respected
- A/F/U ordering is correct
- causal F works
- dual S/R works
- H=[S,R] works
- detached-R API routing is correct
- teacher is structurally independent from HE
- teacher multi-scale targets exist
- teacher masking is isolated
- residual loss matches the focused-review formula
- family registry exists
- family vote invariance passes
- same-snapshot family gradient semantics are implemented
- reserve budget is implemented
- CAGrad numerical tests pass
- private/core updates are separated
- replay interfaces exist
- causality tests pass
- target leakage tests pass
- provenance checks exist
- frozen scientific artifacts remain unchanged
- no real deep-learning training was launched


============================================================
75. FINAL REPORT FORMAT
============================================================

At completion return:


A. SPEC RECONCILIATION

- documents read
- precedence applied
- ambiguities found
- external sources consulted


B. FILES CREATED


C. FILES MODIFIED


D. CORE IMPLEMENTATION

- A
- F
- U
- E
- S
- R
- H
- streaming state


E. API ROUTING

- detached-R implementation
- private projectors
- shadow-gradient diagnostic


F. TEACHER IMPLEMENTATION

- architecture
- EMA
- scales
- masking
- objectives
- health checks


G. RESIDUAL IMPLEMENTATION

- P_R
- L_mask_lat
- L_obs_lat
- L_phys
- L_var_R
- L_R


H. MULTI-API TRAINING SYSTEM

- APIRegistry
- family budgeting
- sampler
- reference scales
- reserve


I. CAGrad

- mathematical implementation
- solver
- degeneracy handling
- fallback
- tests


J. CONTINUAL API CONTRACT

- head-only onboarding
- replay
- distillation
- versioning


K. TESTS RUN

For every high-value test:

- test name
- pass/fail
- what scientific property it verifies


L. FROZEN ARTIFACT INTEGRITY


M. DEVIATIONS FROM SPEC

If none:

NONE


N. BLOCKED DESIGN QUESTIONS

If none:

NONE


O. FUTURE BRANCH IDEAS DISCOVERED

Do not implement them.


P. REMAINING SCIENTIFIC RISKS


Q. EXACT NEXT SAFE COMMAND

Give the exact command for the next LOCAL dry-run / verification step.

Do NOT launch it automatically.


============================================================
76. STOP CONDITION
============================================================

STOP after:

implementation
+
local scientific correctness verification
+
final report.


Do NOT:

- start teacher pretraining
- start HE_FULL training
- launch remote GPU jobs
- tune hyperparameters
- claim scientific performance
- claim HumanEngine works

The next phase requires explicit user authorization.