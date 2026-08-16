# ELF3 M5 Train, Play, Export, and Learning Evidence Design

Date: 2026-08-17
Status: Batch A and Batch B accepted; Batch C1/C2 contract approved and frozen
Prerequisites: accepted M3a environment core and M4 HIM PPO stack

## 1. Objective

M5 exposes the accepted ELF3 environment and HIM PPO implementation through
auditable train, play, and export workflows, then evaluates a newly trained
OpenHomie policy with quantitative learning and deterministic behavior
evidence.

M5 does not change robot dimensions, physics, rewards, terminations, training
defaults, or upstream files. Batch C1 makes the minimum production changes
needed to observe timeout activity and execute fixed evaluation scenarios; the
scenario-only branch must not change ordinary play or training behavior. A
successful process launch, a finite random rollout, a short finite training
run, video, or historical results from the reference HOMIE repository do not
establish learning convergence.

## 2. Runtime authority and environment evidence

The authoritative Isaac Lab runtime is the checkout at:

```text
/home/user/wang-sm/IsaacLab-v2.3.2
```

The imported `isaaclab` and `isaaclab.app` modules must resolve inside that
checkout. Its expected local Python package metadata is:

```text
isaaclab       0.54.2
isaaclab-rl    0.4.7
rsl-rl-lib     3.1.2
torch          2.7.0+cu128
onnx           1.21.0
onnxruntime    1.28.0
```

The lower package version numbers are the metadata shipped by the selected
Isaac Lab 2.3.2 checkout and are recorded as evidence, not treated as an
automatic mismatch. The resolved module paths and public APIs remain the
runtime authority.

The 2026-08-17 preflight observed approximately 1.3 GiB GPU memory used and
31.3 GiB free on the RTX 5090. Every GPU invocation performs a fresh
`nvidia-smi` and CUDA preflight and never terminates another process.

## 3. Architecture

### 3.1 Single user entry point

One script exposes three explicit subcommands:

```bash
python isaaclab_ext/scripts/elf3_him.py train ...
python isaaclab_ext/scripts/elf3_him.py play ...
python isaaclab_ext/scripts/elf3_him.py export ...
```

The script remains thin. It parses and validates arguments, launches
`AppLauncher`, performs delayed runtime imports, dispatches one workflow,
prints a machine-readable result, and closes the simulation application.
It does not copy the historical 5000-line orchestration layer.

### 3.2 Pure workflow layer

`openhomie_isaaclab.workflows.elf3_run` is importable without Isaac Sim,
`pxr`, or `isaaclab_rl`. It owns:

- argument and path validation;
- explicit checkpoint resolution;
- exclusive run-directory creation;
- atomic one-shot JSON evidence;
- configuration and artifact hashing;
- checkpoint lineage checks;
- TensorBoard curve parsing;
- the immutable four-scenario evaluation catalog and CLI validation;
- convergence and behavior acceptance calculations.

All structured files use structured JSON APIs, reject NaN and infinity, and
contain recursively JSON-safe values.

### 3.3 Post-launch simulation layer

`openhomie_isaaclab.workflows.elf3_sim` is imported only after
`AppLauncher` has created the simulation application. It owns:

- Gym task construction and explicit config overrides;
- `RslRlVecEnvWrapper` as the final environment wrapper;
- `HIMOnPolicyRunner` construction;
- train, resume, deterministic play, and export execution;
- scenario-only fixed-command installation and evaluation snapshots;
- finite-tensor and iteration-continuity checks;
- simulation-backed evidence collection.

### 3.4 Task registration

The existing ELF3 Gym registration keeps its accepted
`env_cfg_entry_point` and adds exactly:

```text
rsl_rl_cfg_entry_point =
openhomie_isaaclab.tasks.locomotion.elf3.agents.him_ppo_cfg:Elf3HIMRunnerCfg
```

No M3a environment implementation changes are part of this registration edit.

## 4. CLI contract

Common explicit inputs include `--run-dir`, `--device`, `--seed`, and
`--num-envs`. Train adds `--iterations`. Headless mode is explicit.

- Fresh `train` rejects a checkpoint.
- Resume is a `train` invocation with one explicit regular-file
  `--checkpoint`.
- `play` and `export` require one explicit regular-file checkpoint.
- Globs, implicit `latest`, directories, broken links, and ambiguous
  checkpoint selection are rejected.
- Resume always creates a new run directory and never mutates its parent run.
- An existing, aliased, symlinked, or non-exclusive output path is rejected.
- Ordinary `play` retains the accepted Batch B behavior: 100 steps, command
  resampling enabled, and no scenario evaluation fields.
- Scenario `play` requires exactly one of `stand`, `forward`, `turn`, or
  `crouch` plus `--steps 1000`. `--scenario` and `--steps` are rejected by
  `train` and `export`, and either field without the other is rejected.

Device, seed, environment count, and iteration count are validated before
launch. Invalid input exits nonzero without starting Isaac Sim.

## 5. AppLauncher lifecycle

The script follows the Isaac Lab 2.3.2 ordering:

1. Parse CLI and register `AppLauncher` arguments.
2. Validate all pure inputs and resource requirements.
3. Launch the simulation application.
4. Import Gym, Torch simulation consumers, `isaaclab_rl`, and
   `elf3_sim`.
5. Construct, wrap, and run the environment.
6. Close the environment.
7. Print `M5_HEADLESS_PASS` or `M5_HEADLESS_FAIL`, plus the internal exit
   code and structured result, before `simulation_app.close()`.
8. Close the simulation application in `finally`.

Tracebacks and internal failure status are printed before close so Isaac Sim
shutdown cannot disguise an exception as exit code zero.

## 6. Immutable run evidence

### 6.1 Exclusive and one-shot writes

A run directory is created exclusively. Existing paths and path aliases hard
fail. `manifest.json` and `result.json` are each written once using a
temporary file in the destination directory, flush and fsync, followed by
`os.replace`. Existing evidence is never overwritten.

A manifest without a result is classified as `INCOMPLETE`, never as a pass.

### 6.2 Manifest identity

The versioned manifest records at least:

- command, UTC timestamp, task id, seed, device, environment count, and
  requested iterations;
- normalized CLI arguments;
- Git commit and all dirty paths;
- effective environment and agent configurations;
- configuration hashes;
- ELF3 URDF and USD hashes;
- hashes of all ten M4 production files;
- Python, dependency, driver, GPU, and CUDA evidence;
- resolved Isaac Lab and Isaac Lab RL module paths;
- total and free GPU memory at launch;
- checkpoint schema and starting iteration;
- for resume, the parent checkpoint path, checkpoint hash, parent manifest
  hash, and parent iteration.

Repository paths are stored relative to the repository where possible.
Identity is established by hashes, not filenames.

### 6.3 Results and artifacts

The result records status, stable failure code, final iteration, checkpoint
path and hash, metric summaries, parity results, behavior results, artifact
hashes, and elapsed time. Run artifacts include effective config JSON,
TensorBoard events, atomic checkpoints, exports, deterministic trajectories,
and optional videos. Videos are supporting evidence only.

## 7. Checkpoint and resume semantics

M4 checkpoint schema validation is authoritative. Before simulation work,
the workflow validates:

- model state;
- both optimizer states for training resume;
- policy and estimator learning rates;
- estimator learning-rate follow mode;
- nonnegative completed iteration;
- checkpoint schema compatibility and finite scalar state.

Loading remains transactional. Fresh training starts at iteration zero.
Resuming for K iterations from iteration N must finish at exactly N + K and
must write the final checkpoint even when the save interval is not reached.
Play and export never modify checkpoint or parent-run evidence.

## 8. Deterministic inference and export

Ordinary play uses only `get_inference_policy()` under inference mode. A fixed
observation is evaluated twice on the same device; both action tensors must be
finite, shape `[num_envs, 12]`, and bitwise identical. The action hash is
recorded. Scenario play uses the same inference path but runs exactly 1000
steps with 16 environments, installs one fixed command before the first actor
observation, and records the evaluation observables and trajectory evidence
defined below.

Export creates `policy.ts` and `policy.onnx` from the same explicit
checkpoint. Fresh TorchScript and ONNX Runtime processes compare batches 1
and 4 of seeded actor histories against live `act_inference`:

- TorchScript maximum absolute error: at most `1e-7`;
- ONNX Runtime maximum absolute error: at most `1e-5`;
- all shapes equal and all values finite;
- ONNX checker passes and CPUExecutionProvider executes the graph.

The exported artifact hashes and measured maximum errors are immutable run
evidence.

## 9. Delivery batches

### Batch A: pure CLI and evidence contracts

Batch A delivers the thin CLI, pure workflow module, package marker, and
minimal task registration update. It is tested without Isaac Sim or `pxr`.
It covers command parsing, import ordering, checkpoint selection, exclusive
paths, atomic JSON, hashing, schema validation, finite curve parsing, and all
acceptance calculators.

### Batch B: headless integration

Batch B delivers the post-launch simulation module and a fixed acceptance
harness. It runs:

- 16 environments, seed 42;
- fresh training for 2 iterations;
- explicit resume for 1 iteration in a new run;
- final iteration continuity from 2 to 3;
- deterministic play;
- TorchScript and ONNX export with fresh-runtime parity.

The preflight requires CUDA execution and at least 4 GiB free GPU memory.
The current GPU evidence is expected to pass. If free memory drops below the
threshold, the harness prints `M5_HEADLESS_FAIL` and exits nonzero before
launch; it is never skipped or reported as a pass.

Batch B is archivally accepted at commit
`71c53f79c73873f6f3ee9da097dab0186ecd1b58` from the immutable evidence root
`/home/user/wang-sm/OpenHomie_m5b_71c53f7_20260817`. Its train, resume, play,
TorchScript, ONNX, child-log, and TensorBoard checks all passed. This freezes
the Batch B interfaces while permitting the explicit Batch C1 extensions.

### Batch C: convergence and behavior

Batch C is split into two test-first slices. C1 adds only the production
instrumentation and scenario-play surface absent from Batch B: one timeout
scalar, an opt-in fixed evaluation command, finite metric snapshots, scenario
CLI fields, and immutable per-scenario evidence. After independent C1
acceptance, C2 adds planner-owned convergence tests and an acceptance harness,
then consumes one explicit, from-scratch OpenHomie run; C2 changes no
production.

The canonical training budget is:

```bash
python isaaclab_ext/scripts/elf3_him.py train \
  --headless --device cuda:0 --seed 42 \
  --num-envs 4096 --iterations 2000 \
  --run-dir /absolute/new/run-directory
```

With frozen `cfg.num_steps_per_env == 50`, this is exactly
`4096 * 2000 * 50 = 409,600,000` training transitions. There is no favorable
early stop. If measured resources require fewer environments, the run records
the evidence and justification and uses:

```text
ceil(canonical_transition_budget /
     (actual_num_envs * cfg.num_steps_per_env))
```

iterations. Total transitions and every acceptance threshold remain
unchanged. `50_000` is only the agent-config ceiling, not the required M5
budget.

## 10. Learning convergence acceptance

TensorBoard data is parsed from the explicit M5 run with the official
`EventAccumulator`. The required tags use the actual rsl-rl names:

```text
Train/mean_episode_length
Train/mean_reward
Loss/value_function
Loss/surrogate
Loss/entropy
Loss/estimator_velocity
Loss/estimator_swap
Loss/actor_symmetry
Loss/critic_symmetry
Loss/learning_rate
Perf/total_fps
Episode_Termination/time_out
```

`Episode_Termination/time_out` has unit `rollout_transition_fraction`. It is
written exactly once per completed training iteration as
`sum(bool time_outs) / (num_envs * num_steps_per_env)` over that iteration's
rollout, is finite in `[0, 1]`, and resets for the next iteration. The existing
wrapped `dones` conversion to boolean and the separate boolean `time_outs`
tensor remain unchanged. Missing, truncated, foreign-run, hash-mismatched,
NaN, or infinite series fail.

The fixed acceptance is:

- final 100 recorded `mean_episode_length` points average above 300;
- that final average is at least four times the first 100-point average;
- among the final 100 timeout scalar points, at least 5 are strictly positive;
- every required scalar series is complete and finite;
- the run consumed the complete transition budget.

The timeout rule is a fraction of recorded scalar points, not a claimed
fraction of episodes. Historical HOMIE curves cannot satisfy this gate.

## 11. Deterministic behavior acceptance

The exact final checkpoint is evaluated in twelve separate 16-environment
processes: 1000 policy steps with seeds 42, 43, and 44 in each scenario. The
fixed command layout is `(vx, vy, yaw_rate, height)`; internal command column 3
remains zero and height is written to column 4. Stand uses mode
`HIGH_STAND (1)`, forward and turn use `WALK (0)`, and crouch uses
`CROUCH_LOW (2)`. The default height is the current stage's `walk_height`,
resolved to a numeric value and recorded before the first step.

Fixed commands bypass command curriculum and reset-time and periodic random
resampling only while scenario play is active. Training and ordinary play
retain the accepted resampling path. Credit is tracked per environment and
stops at that environment's first non-timeout termination; automatically reset
states after that point are never credited. Errors use only credited finite
environment steps.

| Scenario | Command | Required evidence |
|---|---|---|
| Stand | `(0, 0, 0, default_height)` | survival >= 0.95, height MAE <= 0.08 m, tilt RMS <= 0.20 rad |
| Forward | `(0.5, 0, 0, default_height)` | survival >= 0.90, vx MAE <= 0.20 m/s, height MAE <= 0.10 m |
| Turn | `(0, 0, 0.5, default_height)` | survival >= 0.90, yaw-rate MAE <= 0.25 rad/s, height MAE <= 0.10 m |
| Crouch | `(0, 0, 0, 0.80)` | survival >= 0.90, height MAE <= 0.08 m, planar-speed RMS <= 0.15 m/s |

Every seed must remain finite and meet the survival threshold, where survival
is `credited_env_steps / (16 * 1000)`. Each non-survival metric is first
computed over that seed's credited environment steps, then aggregated as the
equal-weight arithmetic mean of the three seed metrics. Height is not raw
world root Z: the measured value is exactly the reward-authoritative
torso-link-origin-to-sole quantity
`max(root_z - left_foot_z, root_z - right_foot_z) + ankle_sole_distance`.
Tilt RMS is `sqrt(mean(roll^2 + pitch^2))`; planar-speed RMS is
`sqrt(mean(vx^2 + vy^2))`. Velocity and yaw errors are mean absolute errors in
the root-link body frame.

Every scenario result records the exact resolved command and mode, seed,
`num_envs=16`, 1000 requested steps, source checkpoint path and SHA before and
after, finite flag, credited environment steps, each environment's first
non-timeout termination step and reason, timeout count, survival, scenario
metrics, action SHA, and trajectory SHA. The aggregate report references the
train and export evidence plus all twelve manifest/result hashes and records
convergence, scenario, checkpoint, export, and immutable overall PASS/FAIL.

## 12. Failure behavior

Invalid arguments, insufficient resources, registry errors, output-path
collisions, checkpoint corruption, non-finite tensors, resume discontinuity,
export parity failure, missing metrics, convergence failure, and behavior
failure all produce a stable failure code and nonzero process result.

Evidence and the PASS/FAIL sentinel are emitted before application close.
No failure may be converted into a skip or an incomplete success.

## 13. File boundaries

Batch A production may change only:

```text
isaaclab_ext/scripts/elf3_him.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/__init__.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_run.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/__init__.py
```

The only Batch B production addition is:

```text
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_sim.py
```

`isaaclab_ext/scripts/check_elf3_m5_headless.py` is a planner-owned acceptance
harness committed with the tests and frozen before production implementation.

Batch C1 production may change only:

```text
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/runner.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/elf3_homie_env.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_run.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_sim.py
```

The thin CLI script needs no change because it already delegates parsing and
request execution. C1 must not modify reward functions, reward scales,
termination rules, physics, robot constants or assets, curricula, training
defaults, the other seven generic HIM modules, or upstream paths. C2 changes
only planner-owned tests and
`isaaclab_ext/scripts/check_elf3_m5_convergence.py`, introduced after C1
independent acceptance. Any wider production change requires user approval.
