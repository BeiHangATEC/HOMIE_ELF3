# ELF3 M4 HIM PPO Design

Date: 2026-08-17
Status: Approved by the HANDOFF/Goal blanket authorization
Scope owner: dedicated gpt-5.6-sol planning/test/acceptance agent
Implementation owner: separate execution agent

## 1. Purpose and acceptance boundary

M4 adds the algorithm layer required to train the registered 28-DOF ELF3
environment with the HOMIE History Information Model (HIM) objective on the
installed `rsl-rl-lib==3.1.2`. It covers the estimator, actor-critic, PPO,
rollout storage, symmetry transforms, runner/checkpoint integration, and
inference export.

M4 succeeds when these components satisfy their mathematical and interface
contracts in deterministic unit and integration tests. It does not add the
M5 train/play entry points, run a convergence experiment, or establish that a
policy can stand or walk. It does not add M6 MuJoCo deployment or sim2sim.

The following facts are invariant:

- ELF3 has 28 canonical DOFs and 12 policy actions. Dimensions are derived
  from `elf3_constants`, not duplicated as production literals.
- The environment observation groups are `policy` and `critic`; their current
  dimensions are respectively six 78-value actor frames and one 81-value
  critic frame.
- The estimator produces a 3-value base-velocity estimate and a 32-value
  normalized latent. The actor therefore consumes the latest actor frame plus
  those 35 values.
- M4 changes neither environment physics nor observation/reward/done semantics.
- No upstream OpenHomie, Isaac Lab, or installed `rsl_rl` file is modified.

## 2. Audited compatibility baseline

### 2.1 rsl-rl 3.1.2

The installed package is
`/home/user/miniconda3/envs/homie/lib/python3.11/site-packages/rsl_rl`. Its
current API differs materially from the old HOMIE fork:

- `OnPolicyRunner` receives `TensorDict` observations, resolves named
  `obs_groups`, constructs the algorithm through `_construct_algorithm`, and
  calls `act(obs)`, `process_env_step(obs, rewards, dones, extras)`,
  `compute_returns(obs)`, and `update()`.
- `PPO` owns the policy, transition, optimizer, KL schedule, normalization,
  timeout reward bootstrap, storage lifecycle, and optional distributed
  synchronization.
- `ActorCritic` derives input widths from `TensorDict` groups and owns
  `EmpiricalNormalization` modules. Its `load_state_dict` boolean return is
  used by runner resume logic.
- `RolloutStorage` stores a time-major `TensorDict` and its feed-forward
  generator yields the ten-field PPO minibatch tuple. Standard GAE stops at
  every done.
- Isaac Lab's `RslRlVecEnvWrapper` merges terminated/truncated into `dones`,
  keeps `time_outs` in extras for infinite-horizon tasks, and resets before the
  runner starts.
- Isaac Lab's generic exporter copies only `policy.actor`; it cannot export a
  HIM policy because estimator inference is part of the action graph.
- `EmpiricalNormalization` is an `nn.Module` with registered state. Updating
  it is separate from applying it, and evaluation mode freezes updates.

M4 must preserve these public conventions so Isaac Lab configuration and
wrapping continue to work, but it must not depend on private rsl-rl
implementation details beyond the explicitly overridden construction,
algorithm, storage, and checkpoint hooks.

### 2.2 HOMIE reference mathematics

The historical `HomieRL/rsl_rl` implementation defines the intended HIM
objective: a history encoder predicts base velocity and a normalized latent;
a target encoder embeds the next one-step actor observation; normalized
prototypes produce balanced assignments; swapped cross-prediction and velocity
MSE train the estimator. PPO adds actor and critic mirror losses.

That code is a mathematical reference, not a compatible implementation. It
uses the old tensor API, exponentiates Sinkhorn logits directly, gives the
policy optimizer ownership of estimator parameters while also using an
estimator optimizer, stores mirrored transitions by doubling the time axis,
and can supervise a terminal transition from the post-reset next observation.
Those behaviors must not be carried forward.

The newer HOMIE Isaac Lab adapter is useful corroborating evidence for the
rsl-rl 3.1 shape, branch, masking, and export boundaries, but OpenHomie M4 is
implemented and tested locally rather than importing HOMIE at runtime.

### 2.3 M3a environment contract

After each step the wrapped environment returns reset-state observations for
completed environments. Before reset, M3a records:

- `extras["terminal_critic_obs"]`: pre-reset critic observation for every env;
- `extras["terminal_critic_obs_mask"]`: rows for which that snapshot is valid;
- `extras["time_outs"]`: pure truncations only.

M4 consumes these keys without changing them. The terminal snapshot is used
only for estimator next-state supervision. PPO's existing timeout bootstrap
semantics remain separate.

## 3. Architecture alternatives

### 3.1 Recommended: thin rsl-rl 3.1 extensions

Add a focused `openhomie_isaaclab.him_rl` package. `HIMActorCritic` implements
the current policy interface, `HIMPPO` subclasses `PPO`,
`HIMRolloutStorage` subclasses `RolloutStorage`, and `HIMOnPolicyRunner`
subclasses `OnPolicyRunner`. Override only behavior that HIM genuinely
changes. Keep estimator, symmetry, and exporter as independent modules.

Advantages: the adapter follows the installed API, keeps Isaac Lab wrappers
and logging compatible, reuses tested standard behavior, and gives every HIM
addition a narrow test boundary. The cost is that rsl-rl minor-version changes
can affect overridden hooks; the exact dependency pin and explicit interface
contract tests contain that risk.

### 3.2 Full standalone learner

Copy the runner, PPO, storage, normalization, and export machinery into
OpenHomie and remove inheritance from rsl-rl.

This maximizes control and minimizes sensitivity to superclass internals, but
duplicates a large and evolving framework surface: distributed operation,
logging, observation groups, normalization, checkpoint conventions, and Isaac
Lab integration. It increases review and maintenance cost without changing
the required HIM mathematics, so it is rejected.

### 3.3 Composition around stock PPO

Keep stock `PPO` unchanged and attach estimator updates, mirror augmentation,
and terminal-supervision bookkeeping through runner callbacks or wrappers.

This has the smallest subclass surface, but the required data is inside the
transition/minibatch lifecycle. Callbacks would either reach into private
state or maintain a second rollout buffer that can drift from PPO sampling and
GAE. It cannot express branch-specific last values cleanly, so it is rejected.

The recommended thin-extension architecture is authorized because it preserves
all HANDOFF invariants and algorithm semantics.

## 4. Package and ownership boundaries

M4 adds these conceptual modules under the OpenHomie Isaac Lab package:

- `him_rl/estimator.py`: log-space Sinkhorn and estimator objective;
- `him_rl/actor_critic.py`: current rsl-rl policy contract and parameter
  ownership partition;
- `him_rl/symmetry.py`: validated spec-driven tensor transforms;
- `him_rl/storage.py`: branch-aware GAE and estimator supervision fields;
- `him_rl/ppo.py`: collection, loss computation, two optimizers, and finite
  update guards;
- `him_rl/runner.py`: construction, logging integration, atomic checkpoints,
  and resume;
- `him_rl/exporter.py`: complete estimator-plus-actor inference graph and
  parity verification;
- ELF3 agent configuration that derives dimensions and mirror data from
  canonical constants.

`him_rl` contains no ELF3 joint names. ELF3-specific dimensions and symmetry
enter through configuration. The environment imports no HIM module. Installed
`rsl_rl`, Isaac Lab, upstream G1 task files, `HomieRL/`, `HomieDeploy/`, and
`HomieHardware/` are read-only boundaries.

## 5. Estimator mathematics

Let `h_t` be the actor history and `v_(t+1)` the next critic observation's
critic-only true base velocity. Let `y_(t+1)` be the target-encoder input: the
next actor frame without its leading three velocity-command values, followed
by `v_(t+1)`. This preserves the reference HIM target width while replacing
command intent with the realized base motion.

The source encoder produces

```text
[v_hat_(t+1), z_s] = E(h_t),  z_s = normalize(z_s)
```

and the target encoder produces `z_t = normalize(T(y_(t+1)))`. Prototype rows
`P` are L2-normalized for scoring and normalized in place after every
estimator update. Scores are `S_s = z_s P^T` and `S_t = z_t P^T`.

Balanced assignments use Sinkhorn in log space. For logits shaped
`[batch, prototypes]`, divide by positive epsilon, transpose, subtract a
global `logsumexp`, then alternately normalize prototype rows to mass `1/K`
and sample columns to mass `1/B` for the configured positive iteration count.
Exponentiation happens only after normalization. The returned matrix is
`[batch, prototypes]`, finite and nonnegative, with each sample row summing
approximately to one. This retains the reference objective while preventing
overflow on large logits.

With assignments detached from autograd:

```text
L_swap = -0.5 * mean(q_s * log_softmax(S_t / tau)
                     + q_t * log_softmax(S_s / tau))
L_vel  = MSE(v_hat_(t+1), v_(t+1))
L_est  = L_vel + L_swap
```

Epsilon, temperature, iteration count, dimensions, and nonempty valid batch
are validated. All losses and gradients must be finite. A minibatch with no
valid estimator samples skips the estimator step and reports finite zero
losses; it does not synthesize supervision.

The target layout is derived from named observation segments. For the current
contract this is equivalent to velocity `critic[..., 78:81]` and target input
`critic[..., 3:81]`, but production code derives the boundaries rather than
embedding those literals. No magic slicing is allowed without a named, tested
layout helper.

## 6. Actor-critic and normalization

`HIMActorCritic` accepts the rsl-rl 3.1 constructor shape:
`(obs: TensorDict, obs_groups, num_actions, ...)`. It validates flat, distinct
policy and critic groups and derives their widths. It is feed-forward and
implements the standard `act`, `act_inference`, `evaluate`, distribution
properties, log-probability, reset, normalization update, and boolean
`load_state_dict` behavior.

The estimator consumes the complete normalized actor history. The actor
receives the latest normalized actor frame, predicted velocity, and normalized
latent. The critic receives the complete normalized critic observation. The
Gaussian action distribution follows the pinned rsl-rl scalar standard
deviation semantics and rejects nonpositive standard deviation.

Rollout storage keeps raw observations. When normalization is enabled, use
rsl-rl's `EmpiricalNormalization` modules and update each normalizer exactly
once from the observations returned by each `env.step`, before storing the
transition. Reset rows in those returned observations participate in the
ordinary update; the separate pre-reset `terminal_critic_obs` snapshot never
causes another statistics update. During minibatch evaluation, the current
collected statistics are applied consistently to all stored raw tensors.

The actor-history normalizer has the full history width. Its complete output
feeds the estimator source encoder, and its final one-step slice feeds the
actor. The estimator next-state target combines two normalized regions: its
actor-frame region uses the final one-step mean and standard-deviation slice
from the actor-history normalizer, and its true-base-velocity region uses the
critic normalizer's critic-tail mean and standard-deviation slice. The three
leading velocity commands are discarded only after the next actor frame has
been normalized. This rule applies equally to continuing next observations
and substituted terminal snapshots. It prevents actor and estimator source
features from using one coordinate system while the target silently uses
another.

The critic value path applies the critic normalizer to the complete critic
observation. Inference and export apply the actor-history normalizer before
estimator encoding and actor latest-frame extraction; they do not need the
critic normalizer because the target and value paths are training-only.
Normalizer buffers are part of the policy state dict and therefore part of
checkpoint and export parity. M4 defaults to the approved disabled setting,
but tests exercise the enabled behavior so the default cannot hide an
undefined normalization contract.

Parameter ownership is exact and disjoint:

- the policy optimizer owns actor, critic, and action-noise parameters;
- the estimator optimizer owns source encoder, target encoder, and prototypes;
- their union equals all registered actor-critic parameters.

This prevents the historical double-update bug. Actor inference treats
estimator outputs as detached features, so PPO and symmetry gradients do not
update the estimator.

## 7. PPO, symmetry, and optimizer sequencing

The PPO surrogate, clipped value loss, entropy term, KL schedule, advantage
normalization option, and timeout reward bootstrap match rsl-rl 3.1.2. M4
does not reinterpret them.

For each minibatch:

1. Re-evaluate the collected actions and values, then apply the standard
   adaptive KL update to the policy learning rate.
2. Build estimator samples from original history and terminal-safe next critic
   observations. When symmetry is active, add mirrored estimator samples with
   the same validity mask.
3. Backpropagate `L_est` only through estimator parameters, validate and clip
   estimator gradients, but defer its optimizer step.
4. Compute PPO policy/value/entropy terms plus actor and critic mirror losses.
   Mirror targets are detached.
5. Backpropagate only through policy-owned parameters, validate and clip
   gradients, and step the policy optimizer.
6. If valid estimator samples existed, step the estimator optimizer, validate
   its parameters, renormalize prototypes, and clear its gradients.

The two parameter sets are disjoint, so the deferred estimator step cannot
invalidate policy gradients. Non-finite losses, gradients, gradient norms, or
parameters raise a dedicated training error before a checkpoint can record
the update.

When the adaptive schedule changes policy learning rate, the estimator keeps
its explicitly configured learning rate. It follows policy learning rate only
when its configuration is deliberately `None`; this rule is persisted and
tested on resume.

Symmetry is not delegated to rsl-rl's generic `symmetry_cfg`, because HIM also
needs critic transforms, history transforms, estimator augmentation, and
branch-aware GAE. Supplying both mechanisms is an error.

## 8. Spec-driven symmetry

The mirror transform is built from the canonical ELF3 mirror specification:
DOF permutation/signs, policy-action permutation/signs, the ten observation
head signs, and critic-tail signs. It derives segment offsets from the
canonical layout.

Construction fails unless permutations are complete bijections, signs are
`+1/-1`, and applying each transform twice is identity. Runtime methods
validate last dimensions and preserve arbitrary leading batch dimensions,
device, and dtype.

Actor history is reshaped into frames and each frame is mirrored independently.
The critic's actor-frame prefix uses the same transform; true base velocity in
the critic tail uses its vector signs. Actions use only the policy-action spec.
Observation transforms operate on `TensorDict` using the configured policy
and critic group names and reject ambiguous shared groups.

With transform `M`, the losses are:

```text
L_actor_sym  = scale * mean(sum((pi(M(o)) - M(pi(o)).detach())^2))
L_critic_sym = scale * mean((V(M(c)) - V(c).detach())^2)
```

No ELF3 index table is handwritten in algorithm code.

## 9. Branch-aware storage and GAE

Mirrored rollouts are a second branch of each physical time step, not later
time steps. Storage therefore has logical shape
`[time, branch, env, ...]`, represented compatibly in the inherited time-major
allocation but reshaped explicitly for returns. Branch count is one without
symmetry and two with symmetry.

Each transition extends stock storage with:

- `next_critic_observations [env, critic_dim]`;
- `estimator_masks [env, 1]`.

Original and mirrored transitions are inserted adjacent within a logical time
step. `compute_returns` reshapes to `[time, branch, env, 1]`, carries one GAE
accumulator per branch/environment, and accepts final values shaped
`[branch, env, 1]`. It never lets the original branch bootstrap from the
mirrored branch or vice versa. Advantage normalization is performed over the
complete stored batch unless per-minibatch normalization is configured.

The minibatch generator preserves the rsl-rl ten-field prefix and appends next
critic observations and estimator masks. It samples all stored branches but
does not create another mirrored PPO batch during update. Storage rejects
overflow, incomplete transitions, wrong masks, unsupported recurrent use, and
return computation before the expected rollout is full.

## 10. Episode-safe estimator supervision

For continuing rows, the next critic observation returned by `env.step` is the
correct estimator target. For done rows it is post-reset and must never be
used as a next-state target.

The algorithm constructs supervision as follows:

1. Start with returned next critic observations and mask all non-done rows
   valid.
2. Require `terminal_critic_obs` and `terminal_critic_obs_mask` together.
3. Validate terminal observation shape, one mask value per environment, and
   that the terminal mask selects only done rows.
4. Replace selected done rows with pre-reset terminal observations and mark
   them valid.
5. Leave any done row without a valid terminal snapshot masked out.

This contract prevents cross-episode supervision while retaining valid final
transitions for both failure and timeout episodes. The same mask is used by
the mirrored branch. The mask affects estimator loss only; GAE still stops at
all dones, and stock timeout reward bootstrap still uses `time_outs`.

## 11. Runner and checkpoint contract

`HIMOnPolicyRunner` remains a thin `OnPolicyRunner` extension. It resolves the
standard observation groups, explicitly whitelists the HIM policy and
algorithm classes instead of using `eval`, initializes HIM storage, retains
rsl-rl logging behavior, and checks observations, actions, rewards, losses,
and reported metrics for finiteness.

Checkpoint payloads contain at least:

- full policy state, including estimator, prototypes, noise, and normalizers;
- policy optimizer state;
- estimator optimizer state;
- both effective learning rates and whether estimator LR follows policy LR;
- completed iteration number and optional infos;
- a schema/version marker sufficient to reject incompatible payloads.

Save is atomic: write a temporary file in the destination directory, then
replace the target. Resume restores model, both optimizers, learning-rate
state, and iteration. Missing estimator optimizer state is an error for
training resume, while inference-only loading may skip optimizers explicitly.
Device mapping is caller-controlled. Shape or state mismatch fails loudly;
partial policy loads are not accepted.

M4 checkpoint support is algorithm-level only. Immutable training manifests,
multi-stage orchestration, training entry points, and convergence gates remain
M5 concerns.

## 12. Inference export and numerical parity

The generic Isaac Lab exporter is insufficient because its graph contains
only `actor`. M4 defines `HIMPolicyExporter`, an inference-only deep copy of:

1. actor observation normalizer;
2. estimator source encoder;
3. latent L2 normalization;
4. latest-frame extraction and actor-input concatenation;
5. actor MLP.

Its public input is a rank-2 flat actor history and its output is deterministic
mean actions. Target encoder, prototypes, critic, action noise, and optimizers
are excluded. The exporter is in eval mode and validates input width.

TorchScript and ONNX exports must be loaded in independent runtimes and
compared with the live inference graph over multiple deterministic samples,
including zero, bounded patterned, and seeded random histories and batch sizes
greater than one. Required maximum absolute errors are:

- TorchScript: `<= 1e-7`;
- ONNX Runtime: `<= 1e-5`.

All compared outputs must be finite and have identical shape. ONNX exposes a
dynamic batch axis and uses a runtime-supported pinned opset. Missing ONNX or
ONNX Runtime is an explicit unmet acceptance prerequisite, not a skipped pass.
Export reconstruction from a checkpoint validates canonical configuration,
strict state loading, dimensions, and finite tensors before writing artifacts.

## 13. Error handling

Configuration and data errors fail at their ownership boundary:

- observation/action/layout mismatch, invalid hyperparameters, unknown class
  names, or malformed symmetry specs fail during construction;
- missing terminal key pairs, invalid masks, and transition shape errors fail
  during collection;
- invalid storage fill/branch state fails before GAE or minibatching;
- non-finite training data, losses, gradients, or parameters abort the update;
- malformed or incomplete checkpoints fail before resume;
- export shape, dependency, serialization, load, or parity failures return a
  nonzero result and do not claim a valid artifact.

No broad exception handler converts a failure into success. Scripts that later
host these functions must print internal status before Isaac Sim shutdown, as
established in M3a.

## 14. Test and acceptance design

The dedicated planning/test/acceptance agent writes M4 tests before the
separate execution agent writes production implementation. Production owners
do not weaken those tests.

Static CPU tests cover:

- log-space Sinkhorn under ordinary and extreme logits, approximate row/column
  balance, determinism, finiteness, and invalid input rejection;
- estimator target extraction that drops the three velocity commands and
  appends true base velocity, velocity MSE, swapped loss, masking, prototype
  normalization, empty-valid-batch behavior, and finite gradients;
- actor input layout, TensorDict groups, normalization state, distribution
  contract, detached estimator features, and exact optimizer partition;
- spec-derived mirror bijection, involution, frame/history/critic/action signs,
  dtype/device preservation, and actor/critic equivariance loss targets;
- one- and two-branch storage shapes, independent hand-computed GAE, no branch
  leakage, generator fields, overflow/incomplete-rollout errors, and advantage
  normalization;
- continuing, failure, timeout, missing-terminal, partial-terminal, and invalid
  terminal-mask estimator supervision truth tables;
- independent policy/estimator updates, adaptive LR rules, gradient clipping,
  non-finite rejection, and no double ownership;
- checkpoint round-trip of model, both optimizer states, normalization,
  learning rates, and iteration, plus malformed/legacy failure cases;
- TorchScript and ONNX round-trip parity at the required thresholds.

A small rsl-rl integration test constructs the HIM runner with a deterministic
fake `VecEnv`, collects at least one complete rollout containing continuing,
terminated, and truncated rows, performs an update, saves, resumes, and
performs another update. It verifies finite losses, completed iteration
accounting, terminal-safe targets, and no branch leakage without requiring
Isaac Sim.

M4 acceptance requires the complete existing static suite plus all new M4
tests to pass in the pinned `homie` environment. Any test that silently skips
ONNX parity, uses post-reset supervision for a done row, or collapses branches
onto a single GAE chain is a failed gate.

## 15. Risks and mitigations

- **Pinned API drift:** subclass hooks can change across rsl-rl versions.
  Mitigation: retain the exact 3.1.2 pin and test constructor, transition,
  generator, runner, normalization, and checkpoint interfaces explicitly.
- **Historical behavior mistaken for intent:** old HOMIE contains overflow,
  double-optimization, branch, and terminal-boundary defects. Mitigation: keep
  its loss equations but specify corrected numerical and lifecycle semantics.
- **Symmetry corruption:** ELF3 policy joints are not the first 12 runtime
  joints. Mitigation: consume the canonical name-derived mirror spec and prove
  involution; never use runtime joint positions in M4.
- **Estimator starvation:** very short episodes can reduce valid samples.
  Mitigation: use valid terminal snapshots and skip, rather than fabricate,
  an all-invalid minibatch update while reporting its count in diagnostics.
- **Checkpoint resume drift:** two optimizers and two learning rates can diverge
  after resume. Mitigation: persist and round-trip all states and LR-follow
  semantics atomically.
- **Export looks valid but omits preprocessing:** generic actor-only export can
  serialize successfully while being wrong. Mitigation: export the complete
  inference graph and make numerical parity a hard gate.

## 16. Completion language

On acceptance, the permitted M4 claim is:

> M4 HIM PPO is complete: the rsl-rl 3.1.2 adapter satisfies its estimator,
> actor-critic, dual-optimizer PPO, branch-aware storage, episode-safe
> supervision, symmetry, checkpoint, and export parity contracts.

M4 evidence does not prove learned standing or walking, training convergence,
M5 train/play workflows, or M6 sim2sim behavior.
