# ELF3 M3a Environment Core Design

Date: 2026-08-16
Status: Approved
Scope owner: dedicated gpt-5.6-sol planning/test/acceptance agent
Implementation owner: separate gpt-5.6-sol execution agent

## 1. Purpose

M3a completes the runnable Isaac Lab environment boundary for the 28-DOF
ELF3 robot in OpenHomie. It turns the existing asset, canonical constants,
reward functions, and curriculum functions into a registered `DirectRLEnv`
that can be instantiated and exercised with random actions.

M3a succeeds when the environment contract is correct, failures are explicit,
and a random-action rollout remains numerically stable across resets. It does
not establish that a policy can balance or walk. The observed
`elf3_final_2300` episode lengths of about 50 policy steps and zero timeouts
through step 318 do not demonstrate learning; they show that episodes ended
by failure before the 1000-step time limit.

## 2. Scope

### In scope

- `Elf3HomieEnvCfg`, including simulation timing, scene, robot, sensors,
  episode duration, events, command ranges, reward scales, and termination
  thresholds.
- `Elf3HomieEnv`, including runtime name resolution, reset and step lifecycle,
  observations, action application, mixed control, rewards, curriculum state,
  domain randomization, termination, truncation, and episode diagnostics.
- Gymnasium task registration as `OpenHomie-Elf3-Homie-Direct-v0`.
- A headless random-action rollout entry point for end-to-end acceptance.
- Small Isaac-independent tensor helpers where they make joint permutation,
  observation layout, action scatter, control randomization, or done semantics
  independently testable.
- Static pytest specifications and Isaac Lab integration acceptance tests,
  written by the dedicated gpt-5.6-sol planning/test/acceptance agent before
  implementation.

### Non-goals

- HIM estimator, PPO, storage, runner, symmetry, export, or any other M4 work.
- Training, checkpointing, play, convergence claims, or any other M5 work.
- MuJoCo deployment, sim2sim comparison, or any other M6 work.
- The large immutable training-plan orchestration layer from HOMIE.
- G1 changes, modifications under `HomieRL/`, `HomieDeploy/`, or
  `HomieHardware/`, and changes to upstream G1 behavior.
- Terrain support beyond the flat plane.
- Changing the canonical 28-DOF decision, robot physical parameters, default
  pose, gains, reward semantics, or curriculum stage definitions.
- Requiring a zero-action robot to remain standing for 20 seconds. Existing
  measurements show that the open-loop pose falls after about one second;
  active balance is a learned-policy responsibility.

## 3. Architecture

### 3.1 Canonical facts and assets

`openhomie_isaaclab.elf3_constants` remains the only source for joint groups,
dimensions, observation scales, gains, limits, geometry, timing, and spawn
height. Dimensions such as 78, 81, and 468 may appear in tests as expected
ELF3 results, but implementation code derives them from the canonical tables.

`elf3_articulation.py` remains responsible for the ELF3 articulation config,
including the preconverted USD, explicit actuator limits, split leg and upper
body actuator groups, and the startup massless-link correction.

### 3.2 Environment configuration

`elf3_homie_env_cfg.py` declares configuration; it does not implement step
logic. It contains:

- `DirectRLEnvCfg` simulation and policy timing from `SIM_DT` and
  `DECIMATION`;
- a flat plane, replicated scene, ELF3 articulation, and required contact
  sensing;
- episode duration from `EPISODE_LENGTH_S`;
- event terms for startup and reset randomization;
- command, curriculum, reward, and termination parameters;
- action and observation spaces derived from canonical constants.

Configuration defaults are suitable for training consumers later, while tests
and the rollout script may override environment count, device, seed, or episode
duration without changing production defaults.

### 3.3 Environment lifecycle

`elf3_homie_env.py` owns Isaac-dependent runtime state and implements the
`DirectRLEnv` hooks. At startup it resolves joints and bodies by name and
constructs explicit permutations between Isaac runtime order and canonical
HOMIE order. The mapping must support the observed nontrivial ordering, such
as canonical `waist_y_joint` mapping from runtime index 2 and canonical
`l_shoulder_y_joint` mapping from runtime index 0.

The environment owns history buffers, previous actions, commands, upper-body
targets, curriculum progress, randomized control factors, episode sums, and
termination diagnostics. Each tensor has a documented shape, device, and
ordering. State needed by future algorithms is exposed through the standard
Direct environment observation and extras contract, without importing HIM.

### 3.4 Pure tensor helpers

Narrow helpers may be placed in a focused environment support module or kept
as module-level functions when they have no Isaac dependency. They may cover:

- validating and constructing name permutations;
- canonical/runtime gather and scatter;
- observation frame assembly and history shifting;
- leg PD effort computation and clamping;
- control-randomization tensor updates;
- separation of failure termination from time-limit truncation.

These helpers are not a second source of robot constants and must accept
dimensions, indices, scales, or limits derived from `elf3_constants`.

### 3.5 Registration and rollout

`tasks/locomotion/elf3/__init__.py` registers the stable Gymnasium ID
`OpenHomie-Elf3-Homie-Direct-v0`. Its entry point references
`Elf3HomieEnv`, and its configuration entry point references
`Elf3HomieEnvCfg`. Re-importing task modules must not create a conflicting
registration.

`isaaclab_ext/scripts/random_rollout_elf3.py` starts Isaac Sim, imports the
extension to trigger registration, creates the task by Gymnasium ID, samples
finite actions in the declared 12-dimensional action space, steps through
resets, and reports a concise acceptance summary. It prints its internal
PASS/FAIL status and exit code before calling `simulation_app.close()`, because
Isaac Sim shutdown can be slow and can obscure exceptions or process status.

## 4. Data Flow

1. Gymnasium resolves the registered task and constructs `Elf3HomieEnvCfg`.
2. Isaac Lab creates the flat scene, ELF3 articulation, contact sensing, and
   startup events.
3. The environment resolves all 28 joint names and required body names. It
   validates a complete one-to-one mapping before the first reset.
4. Reset samples commands, height, upper-body targets, and configured domain
   randomization for the selected environment IDs. It clears only those
   environments' history, action, reward, and done state.
5. A policy action has exactly one value per canonical policy leg joint. The
   environment validates and clamps it, applies `ACTION_SCALE`, and scatters it
   by name-derived indices. No assumption is made that the first 12 runtime
   joints are the legs.
6. Leg torque is computed from the desired position, measured state,
   randomized kp/kd factors, and actuation offset, then clamped to the
   per-joint effort limits. The 16 upper-body joints receive position targets
   through their implicit actuator group.
7. Isaac advances `DECIMATION` physics steps per policy step.
8. Runtime joint and rigid-body state is gathered and reordered into canonical
   order. IMU angular velocity and projected gravity use Isaac Lab's wxyz
   quaternion utilities and the configured IMU body, not an Isaac Gym xyzw
   convention or an assumed root-body frame.
9. The actor frame is assembled in the fixed contract order: scaled velocity
   commands, unscaled absolute height command, scaled IMU angular velocity,
   projected gravity, all canonical joint position offsets, all canonical
   joint velocities, and previous canonical leg action. Six frames form the
   actor history; true scaled base linear velocity is appended only to the
   one-step critic observation.
10. Existing pure tensor reward terms consume the current state. The complete
    scale table is validated and rewards are scaled by policy `dt` exactly
    once. Existing curriculum functions update their owned state.
11. Failure conditions produce `terminated`; reaching the episode time limit
    produces `truncated`. `extras["time_outs"]` represents only truncation.
    Episode diagnostics and terminal observations are captured before reset.
12. Done environment IDs reset without disturbing continuing environments,
    and the next returned observations are finite and correctly shaped.

## 5. Error Handling and Invariants

The environment fails early rather than continuing with a plausible but wrong
mapping or tensor:

- A missing USD retains the existing actionable `FileNotFoundError`. USD
  staleness remains the responsibility of the existing conversion stamp check;
  M3a does not introduce a second staleness mechanism in the environment.
- Missing, extra, or duplicate runtime joint names; missing required body names;
  and non-bijective permutations raise an error during initialization.
- Action rank, batch size, or width mismatches raise `ValueError`. Non-finite
  actions or generated observations raise an error with the affected field and
  environment IDs; they are not silently replaced with zeros.
- Observation assembly verifies its final cursor against dimensions derived
  from constants. Reward scale names must exactly match the reward term set.
- Randomization validates ranges, tensor device, dtype, and selected IDs.
  Advanced-index updates use assignment semantics, for example
  `target[env_ids] = generated_values`; in-place mutation of
  `target[env_ids]` is forbidden because it mutates a copy in PyTorch.
- Torque is clamped against explicit per-joint limits before being sent to the
  articulation. A test-visible diagnostic permits checking limit compliance.
- `terminated` and `truncated` remain separate through return values and
  extras. A failure on the final allowed step is reported consistently as a
  failure termination, while `time_outs` is true only for environments whose
  sole done cause is the time limit.
- Isaac exceptions are reported with traceback and an internal nonzero exit
  code before application shutdown. No broad exception handler converts a
  failure to success.
- Implementation and scripts contain no machine-specific absolute paths.

## 6. Test Design

The dedicated gpt-5.6-sol planning/test/acceptance agent writes the tests
before the separate gpt-5.6-sol execution agent implements M3a. The execution
agent may not edit tests to make implementation pass.

### 6.1 Static and pure tensor tests

These run without Isaac Sim or a GPU:

- The environment and configuration spaces are derived from the 28-joint and
  12-policy-joint tables; no production dimension literals or absolute paths
  are introduced.
- The Gym registration points to the intended environment and configuration.
- A deliberately shuffled runtime joint list produces a bijection and exact
  canonical-to-runtime round trip; missing and duplicate names fail.
- Twelve actions scatter only to named leg joints under a nontrivial runtime
  ordering. Waist and arm targets are unchanged.
- Observation assembly checks every segment's ordering and scaling, critic-only
  linear velocity, previous-action placement, and six-frame history shift and
  reset behavior.
- PD effort calculation applies action scaling, kp/kd factors, offsets, and
  per-joint clamps.
- Control randomization changes selected environments, leaves unselected ones
  untouched, stays within configured ranges, and uses working assignment
  semantics for tensor-indexed IDs.
- Reward output is finite and the existing exact scale-name validation remains
  active.
- Done truth tables cover failure only, time limit only, neither, and failure
  on the final step; `time_outs` never aliases general done state.

### 6.2 Isaac Lab integration tests

These run in the pinned `homie` environment and require Isaac Sim:

- Import registration and create the Gymnasium task with a small number of
  environments.
- Reset and step validate actor/critic observation shapes, action width,
  device, dtype, and finiteness.
- All 28 runtime joints and required bodies resolve by name under the actual
  Isaac ordering.
- Applied leg effort remains within explicit limits and upper-body control does
  not overwrite policy leg targets.
- Resetting a subset changes only selected environment state.
- A controlled short episode duration produces at least one genuine
  `truncated`/`time_outs` result without relying on the training curve.
- A controlled failure condition produces `terminated` with `time_outs=false`.
- A 1000-policy-step random-action rollout, automatically resetting done
  environments, produces no NaN or infinity in observations, rewards, actions,
  efforts, or diagnostics.
- Runtime total mass remains approximately 43.22 kg after the massless-link
  startup event.

The timeout test may disable or neutralize failure conditions only within its
test configuration so that the time limit is reachable. Such overrides must
not change production defaults.

## 7. Acceptance Procedure

The implementation plan will name exact commands after the tests and rollout
entry point exist. Acceptance must include all of the following evidence:

1. The complete static pytest suite passes without Isaac Sim.
2. The Isaac integration suite creates the registered environment and passes
   mapping, shape, finiteness, control, reset, termination, truncation, and mass
   checks.
3. The headless random-action script completes 1000 policy steps across resets,
   reports finite values, and prints `PASS` plus internal exit code 0 before
   closing Isaac Sim.
4. A short-time-limit acceptance run observes a real timeout, and a separate
   induced-failure run observes a non-timeout termination.
5. The dedicated gpt-5.6-sol planning/test/acceptance agent independently
   executes the acceptance commands and records their real output. Code
   inspection or the execution agent self-report is insufficient.

An outer shell timeout exit status of 124 is not automatically a failed test
if Isaac Sim printed a successful internal result before a slow shutdown. If
the internal result was not printed, the run is inconclusive and must be
repeated or diagnosed; it cannot be accepted.

## 8. Completion Language

On acceptance, the permitted claim is:

> M3a Environment Core is complete: the registered 28-DOF ELF3 Isaac Lab
> environment satisfies its observation, action, control, reward, reset,
> termination, truncation, and randomization contracts, and a 1000-step random
> rollout is numerically stable across resets.

The following claims are prohibited by M3a evidence:

- the robot learned to stand or walk;
- the policy converged or improved;
- training, play, export, or sim2sim works;
- zero timeouts in `elf3_final_2300` is success;
- random rollout stability implies locomotion correctness.

Learning remains an M5 convergence question. Its evidence must include a
sustained increase in mean episode length and eventual nonzero timeouts under
the production episode limit; current approximately 50-step episodes do not
meet that standard.
