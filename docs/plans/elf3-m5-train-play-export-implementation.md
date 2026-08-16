# ELF3 M5 Implementation Plan

Date: 2026-08-17
Design: `docs/plans/elf3-m5-train-play-export-design.md`
Status: Batch A/B accepted; Batch C1 implementation contract approved and frozen

## 1. Ownership and common gates

- Planning, tests, and independent acceptance are owned by the dedicated
  gpt-5.6-sol planning agent.
- Production implementation is owned by the dedicated gpt-5.6-sol execution
  agent.
- Tests are written and committed before production for every batch.
- The execution agent never edits tests or acceptance harnesses.
- Unrelated dirty and untracked files are never staged.
- Commits use `Qingwei Ben <78202424+Elgce@users.noreply.github.com>`.

Every batch runs `git diff --check`, verifies the staged whitelist, and
records exact commands, exit codes, counts, warnings, and failures. For C1,
the four explicitly whitelisted production files are expected to change and
all other environment/HIM files, constants, assets, rewards, curriculum,
training defaults, and upstream paths are hashed before and after. After C1
acceptance, the four C1 hashes are frozen for C2.

The runtime must resolve `isaaclab` and `isaaclab.app` inside the
IsaacLab-v2.3.2 checkout. The expected package metadata mapping is
`isaaclab==0.54.2` and `isaaclab-rl==0.4.7`; this mapping is recorded
rather than treated as a release mismatch.

Current static baseline:

```text
259 passed, 8 known ONNX warnings, no skips
```

## 2. Batch A: pure CLI, registry, and run evidence

### 2.1 Test ownership

The planning agent creates and commits only:

```text
tests/test_elf3_m5_cli_contract.py
tests/test_elf3_m5_run_evidence.py
```

Suggested commit:

```text
test: define M5 CLI and run evidence contracts
```

### 2.2 CLI red contracts

`test_elf3_m5_cli_contract.py` covers:

- exactly the `train`, `play`, and `export` subcommands;
- required and rejected argument combinations;
- positive integer environment and iteration counts;
- explicit seed and device validation;
- fresh train rejecting checkpoints;
- resume, play, and export requiring one explicit regular-file checkpoint;
- rejection of globs, `latest`, directories, broken links, and ambiguous
  paths;
- existing, symlinked, and aliased run-directory rejection;
- the existing Gym `env_cfg_entry_point` remaining unchanged;
- the exact import-resolvable `rsl_rl_cfg_entry_point`;
- AST ordering that launches `AppLauncher` before imports of Gym, Torch
  simulation consumers, `isaaclab_rl`, or `elf3_sim`;
- pure modules importing in an isolated subprocess without `pxr` or Isaac
  Sim.

### 2.3 Evidence red contracts

`test_elf3_m5_run_evidence.py` covers:

- exclusive run-directory creation;
- one-shot manifest and result semantics;
- same-directory temporary files, flush, fsync, and replace;
- refusal to overwrite existing evidence;
- recursive JSON safety and `allow_nan=False`;
- stable schema versions and rejection of future or malformed schemas;
- Git, config, asset, M4 stack, runtime, GPU, and dependency identities;
- repository-relative paths and content hashes;
- parent manifest and checkpoint identity for resume;
- exact N + K iteration continuity;
- manifest-without-result classified as incomplete;
- symlink, path-alias, and overwrite attacks;
- finite TensorBoard parsing with missing, truncated, NaN, and infinity
  failures;
- first/final-window convergence calculations;
- timeout positive-point calculations;
- full transition-budget accounting;
- all deterministic behavior threshold calculations and edge cases.

### 2.4 Red command

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest -q \
  tests/test_elf3_m5_cli_contract.py \
  tests/test_elf3_m5_run_evidence.py
```

The red state must be attributable only to the absent CLI/workflow files and
the absent `rsl_rl_cfg_entry_point` in the existing task initializer.

The Batch A production whitelist is:

```text
isaaclab_ext/scripts/elf3_him.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/__init__.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_run.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/__init__.py
```

Fixture, dependency, syntax, and collection failures are test defects and
must be corrected before implementation.

### 2.5 Batch A production

The execution agent may modify only the four paths above. The workflow module
uses `argparse`, `pathlib`, structured JSON, exclusive creation, and
same-directory atomic writes. It contains no Isaac Sim, `pxr`, Gym,
`isaaclab_rl`, or simulation-side Torch import.

The CLI defines the post-launch `elf3_sim` interface without importing it
until after the application is live. Task registration changes only the new
RSL-RL config entry point while preserving the environment config entry.

### 2.6 Batch A green gates

```bash
/home/user/miniconda3/envs/homie/bin/python -m py_compile \
  isaaclab_ext/scripts/elf3_him.py \
  isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/__init__.py \
  isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_run.py

/home/user/miniconda3/envs/homie/bin/python -m pytest -q \
  tests/test_elf3_m5_cli_contract.py \
  tests/test_elf3_m5_run_evidence.py

/home/user/miniconda3/envs/homie/bin/python -m pytest -q tests
git diff --check
```

After execution-agent self-test, the planning agent independently reruns the
same gates and audits the four-file production scope.

Suggested production commit:

```text
feat: add auditable ELF3 training CLI
```

## 3. Batch B: Isaac headless workflow

### 3.1 Test and harness ownership

The planning agent creates:

```text
tests/test_elf3_m5_headless_contract.py
isaaclab_ext/scripts/check_elf3_m5_headless.py
```

The acceptance harness is test-owned. The execution agent must not modify it.

Suggested commit:

```text
test: define M5 headless workflow contract
```

### 3.2 Static and integration contracts

The tests and harness require:

- delayed post-AppLauncher runtime imports;
- `RslRlVecEnvWrapper` as the final wrapper;
- exact task and agent entry points;
- fixed 16 environments and seed 42;
- fresh two-iteration training in an exclusive run;
- explicit one-iteration resume in a second exclusive run;
- completed iteration continuity 0 to 2 to 3;
- finite observations, actions, rewards, losses, learning rates, entropy,
  estimator metrics, and checkpoint values;
- a final checkpoint for every train invocation;
- deterministic `act_inference` actions from repeated fixed input;
- play and export using an explicit checkpoint only;
- TorchScript and ONNX exports loaded in fresh runtimes;
- M4 parity thresholds and artifact hashes;
- manifest and result identity and parent lineage;
- `M5_HEADLESS_PASS`, `M5_HEADLESS_FAIL`, and internal exit code printed
  before `simulation_app.close()`.

### 3.3 Red command

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest -q \
  tests/test_elf3_m5_headless_contract.py
```

The red state must be attributable only to the missing post-launch workflow:

```text
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_sim.py
```

### 3.4 Batch B production

The execution agent creates only `workflows/elf3_sim.py`. The Batch A CLI
and pure workflow interfaces are frozen. If the frozen interface itself is
proven inconsistent by a planner-owned red test, the planning agent must
revise the contract before any additional production file is edited.

### 3.5 GPU preflight and headless command

Immediately before the harness:

```bash
nvidia-smi
```

The harness verifies CUDA Torch execution, `sm_120`, and at least 4 GiB free
GPU memory before AppLauncher. The 2026-08-17 evidence shows approximately
31.3 GiB free, so the preflight is expected to pass. If availability later
drops below the threshold, the command exits nonzero with
`M5_HEADLESS_FAIL`; it is not skipped.

The planning agent fixes the exact harness CLI in its test commit. The command
uses a new absolute run root and performs train, resume, play, and export in
one reproducible sequence.

### 3.6 Batch B green gates

Acceptance requires:

- static focused tests green;
- the real headless harness prints `M5_HEADLESS_PASS` before close;
- internal exit code zero;
- final resume iteration exactly 3;
- no skipped ONNX Runtime execution;
- all artifacts, manifests, results, and hashes present;
- the complete test suite green;
- frozen M3a and M4 hashes unchanged.

Suggested production commit:

```text
feat: integrate ELF3 headless training workflow
```

Batch B is archivally accepted at commit
`71c53f79c73873f6f3ee9da097dab0186ecd1b58` from
`/home/user/wang-sm/OpenHomie_m5b_71c53f7_20260817`; its four child processes,
checkpoint lineage, immutable play, both exports, and TensorBoard inventory
passed independent verification.

## 4. Batch C1: training instrumentation and scenario evaluation

### 4.1 C1 test ownership

The planning agent creates and commits only:

```text
tests/test_elf3_m5_training_instrumentation.py
```

It freezes only the minimum C1 production contract: the timeout scalar tag,
unit and range; unchanged boolean done boundary; opt-in fixed commands that
survive reset and periodic resampling; exact scenario-to-command/mode mapping;
finite shape-checked evaluation observables; scenario CLI validation; fixed
commands installed before the first actor observation; and no evaluation
branch in rewards, physics, terminations, or the default training path.

Suggested commit:

```text
test: define M5 training instrumentation contract
```

### 4.2 C1 red command

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest -q \
  tests/test_elf3_m5_training_instrumentation.py
```

The test must collect normally and fail only because the four whitelisted
production files do not yet expose the frozen C1 contract. Fixture,
dependency, syntax, collection, or unrelated baseline failures are test
defects and must be corrected before production.

### 4.3 C1 production

The execution agent may modify only:

```text
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/runner.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/elf3_homie_env.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_run.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_sim.py
```

`runner.py` writes `Episode_Termination/time_out` exactly once per completed
iteration. Its value is the rollout transition fraction
`sum(bool time_outs) / (num_envs * num_steps_per_env)`, finite in `[0, 1]`.
It does not replace or recast the wrapped boolean `dones` passed to HIMPPO.

`elf3_run.py` adds immutable `EVALUATION_SCENARIOS` definitions and play-only
`scenario`/`steps` request fields. Ordinary play remains 100 steps and rejects
`--steps`; scenario play requires `--scenario` and exactly `--steps 1000`.
The exact mapping is:

```text
stand:   HIGH_STAND (1), (0.0, 0.0, 0.0, default_height)
forward: WALK       (0), (0.5, 0.0, 0.0, default_height)
turn:    WALK       (0), (0.0, 0.0, 0.5, default_height)
crouch:  CROUCH_LOW (2), (0.0, 0.0, 0.0, 0.80)
```

`elf3_homie_env.py` exposes `set_evaluation_command(command, mode)` and
`get_evaluation_observables()`. The setter maps the four command values to
internal columns 0, 1, 2, and 4 while column 3 stays zero. Fixed commands
bypass random reset-time and periodic resampling only after the setter is
called. The default is disabled, so training and ordinary play are unchanged.
The observable mapping has exactly:

```text
command             [num_envs, 4]
root_lin_vel_b       [num_envs, 3]
root_ang_vel_b       [num_envs, 3]
roll_pitch           [num_envs, 2]
tracking_height      [num_envs]
```

Every value is finite. `tracking_height` is exactly
`max(root_z - left_foot_z, root_z - right_foot_z) + ankle_sole_distance`, the
same actual quantity used by `tracking_base_height`, never raw world root Z.

`elf3_sim.py` installs the fixed command immediately after raw environment
creation and before wrapping or runner construction. Only scenario play uses
that path. It runs exactly 1000 steps with 16 environments, tracks credit per
environment until its first non-timeout termination, excludes later automatic
reset data, and records the immutable command, checkpoint, finite, termination,
timeout, survival, metric, action-hash, and trajectory-hash evidence.

The thin `isaaclab_ext/scripts/elf3_him.py` remains unchanged.

### 4.4 C1 green gates

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest -q \
  tests/test_elf3_m5_training_instrumentation.py

/home/user/miniconda3/envs/homie/bin/python -m pytest -q tests
git diff --check
```

After executor self-test, the planning agent independently repeats the gates,
audits the four-file production whitelist, verifies all frozen hashes, and
runs one real 16-environment scenario play before C1 acceptance.

## 5. Batch C2: convergence and deterministic behavior evidence

### 5.1 Evidence ownership after C1 acceptance

Only after C1 independent acceptance, the planning agent creates and commits:

```text
tests/test_elf3_m5_convergence_evidence.py
isaaclab_ext/scripts/check_elf3_m5_convergence.py
```

Fixture tests do not depend on ambient evidence or an environment variable.
The frozen harness accepts one explicit `--evidence-root`; a missing or
malformed root fails nonzero without skip/xfail. The evidence root contains
one immutable training run, one exact-checkpoint export run, twelve scenario
runs, and one aggregate result referencing every manifest and result hash.

### 5.2 Resource preflight

Before the canonical run, record:

- resolved Isaac Lab 2.3.2 paths and expected package metadata mapping;
- CUDA Torch tensor execution and `sm_120`;
- `nvidia-smi` output;
- at least 12 GiB free GPU memory;
- no termination of other users' processes.

Insufficient resources fail before AppLauncher and leave explicit failure
evidence.

### 5.3 Canonical from-scratch run

```bash
python isaaclab_ext/scripts/elf3_him.py train \
  --headless \
  --device cuda:0 \
  --seed 42 \
  --num-envs 4096 \
  --iterations 2000 \
  --run-dir /absolute/new/run-directory
```

The fixed budget is exactly `4096 * 2000 * 50 = 409,600,000` transitions.
Favorable early stopping
is forbidden. A resource-backed reduction in environment count increases
iterations to at least
`ceil(canonical_budget / (actual_num_envs * cfg.num_steps_per_env))`;
it never reduces total transitions or thresholds.

### 5.4 Convergence gate

The official `EventAccumulator` parses the explicit run. Required tags are
`Train/mean_episode_length`, `Train/mean_reward`, all seven HIM loss tags plus
`Loss/learning_rate`, `Perf/total_fps`, and the new
`Episode_Termination/time_out`. Acceptance requires:

- final 100 mean episode-length points average above 300;
- final average at least four times the first 100-point average;
- at least 5 of the final 100 timeout scalar points strictly positive;
- every required reward, loss, entropy, learning-rate, throughput,
  estimator, episode-length, and timeout scalar finite;
- the complete transition budget and checkpoint chain present.

Historical HOMIE data and short M5 headless runs cannot satisfy this gate.

### 5.5 Exact checkpoint export

Read the canonical training `result.json`; require `status == "PASS"`, final
iteration 2000, a regular final checkpoint file, and a SHA-256 matching the
result. Never glob or select `latest`. Use that exact path:

```bash
python isaaclab_ext/scripts/elf3_him.py export \
  --headless --device cuda:0 --seed 42 --num-envs 1 \
  --checkpoint /absolute/final/checkpoint.pt \
  --run-dir /absolute/new/export-run
```

Both exports are loaded independently. TorchScript error must be at most
`1e-7`; ONNX Runtime error must be at most `1e-5`. The oracle is the live
`runner.get_inference_policy(device="cpu")`; batches 1 and 4 run in fresh
TorchScript and ONNX subprocesses, with ONNX checker and exactly
`CPUExecutionProvider`.

### 5.6 Deterministic play matrix

Run stand, forward, turn, and crouch scenarios for seeds 42, 43, and 44,
1000 policy steps and 16 environments each. Every invocation uses the same
exact final checkpoint and a new evidence directory. The resolved numeric
command, mode, seed, steps, and `num_envs=16` are recorded verbatim.

Acceptance uses the fixed thresholds from the design. Every seed must be
finite and meet its survival requirement, computed as
`credited_env_steps / (16 * 1000)`. Other scenario metrics are the equal-weight
mean of the three per-seed values. Per-seed and aggregate action, trajectory,
manifest, and result hashes are required. Videos do not replace numeric
evidence.

### 5.7 Evidence green command

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest -q \
  tests/test_elf3_m5_convergence_evidence.py

/home/user/miniconda3/envs/homie/bin/python \
  isaaclab_ext/scripts/check_elf3_m5_convergence.py \
  --evidence-root /absolute/completed/evidence-root

/home/user/miniconda3/envs/homie/bin/python -m pytest -q tests
```

Batch C2 changes no production code. Missing, duplicate, foreign, truncated,
non-finite, hash-mismatched, or threshold-failing evidence is a hard failure.

## 6. Final M5 audit

The planning agent independently:

1. Runs the complete suite with zero skips.
2. Repeats deterministic play and both exports from the exact final
   checkpoint.
3. Verifies manifest, result, checkpoint, export, trajectory, and parent
   hashes.
4. Verifies iteration continuity and the full transition budget.
5. Rechecks all frozen file hashes and Git scopes.
6. Reports exact commands, run directories, commit ids, checkpoint hash,
   curve windows, timeout counts, scalar finiteness, parity maxima,
   per-scenario metrics, GPU evidence, and remaining risks.

M5 is accepted only when the curve, timeout, finite-scalar, determinism,
export, and four behavior gates all pass. No partial or process-only success
is reported as M5 completion.
