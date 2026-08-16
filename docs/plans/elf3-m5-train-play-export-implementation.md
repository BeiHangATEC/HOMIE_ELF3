# ELF3 M5 Implementation Plan

Date: 2026-08-17
Design: `docs/plans/elf3-m5-train-play-export-design.md`
Status: Approved for implementation under the HANDOFF blanket authorization

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
records exact commands, exit codes, counts, warnings, and failures. M3a
environment files, all eight generic HIM modules, constants, assets, rewards,
curriculum, and upstream paths are hashed before and after each batch.

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

## 4. Batch C: convergence and deterministic behavior evidence

### 4.1 Evidence test ownership

The planning agent creates and commits only:

```text
tests/test_elf3_m5_convergence_evidence.py
```

The test reads exactly one explicit absolute run directory from
`OPENHOMIE_M5_RUN`. An absent variable or nonexistent path fails; it never
skips.

Fixtures cover passing evidence and failures for thresholds, missing tags,
foreign run identity, NaN, infinity, truncation, transition-budget mismatch,
checkpoint mismatch, artifact hash mismatch, nondeterminism, export parity,
and each behavior scenario.

Suggested commit:

```text
test: require M5 convergence and behavior evidence
```

### 4.2 Red evidence command

```bash
OPENHOMIE_M5_RUN=/absolute/nonexistent \
  /home/user/miniconda3/envs/homie/bin/python -m pytest -q \
  tests/test_elf3_m5_convergence_evidence.py
```

The failure must explicitly report missing M5 run evidence. It must not be a
skip or collection error.

### 4.3 Resource preflight

Before the canonical run, record:

- resolved Isaac Lab 2.3.2 paths and expected package metadata mapping;
- CUDA Torch tensor execution and `sm_120`;
- `nvidia-smi` output;
- at least 12 GiB free GPU memory;
- no termination of other users' processes.

Insufficient resources fail before AppLauncher and leave explicit failure
evidence.

### 4.4 Canonical from-scratch run

```bash
python isaaclab_ext/scripts/elf3_him.py train \
  --headless \
  --device cuda:0 \
  --seed 42 \
  --num-envs 4096 \
  --iterations 2000 \
  --run-dir /absolute/new/run-directory
```

The fixed budget is
`4096 * 2000 * cfg.num_steps_per_env` transitions. Favorable early stopping
is forbidden. A resource-backed reduction in environment count increases
iterations to at least
`ceil(canonical_budget / (actual_num_envs * cfg.num_steps_per_env))`;
it never reduces total transitions or thresholds.

### 4.5 Convergence gate

The official TensorBoard reader parses the explicit run. Acceptance requires:

- final 100 mean episode-length points average above 300;
- final average at least four times the first 100-point average;
- at least 5 of the final 100 timeout scalar points strictly positive;
- every required reward, loss, entropy, learning-rate, throughput,
  estimator, episode-length, and timeout scalar finite;
- the complete transition budget and checkpoint chain present.

Historical HOMIE data and short M5 headless runs cannot satisfy this gate.

### 4.6 Exact checkpoint export

Use the canonical run's exact final checkpoint:

```bash
python isaaclab_ext/scripts/elf3_him.py export \
  --headless --device cuda:0 --seed 42 --num-envs 1 \
  --checkpoint /absolute/final/checkpoint.pt \
  --run-dir /absolute/new/export-run
```

Both exports are loaded independently. TorchScript error must be at most
`1e-7`; ONNX Runtime error must be at most `1e-5`.

### 4.7 Deterministic play matrix

Run stand, forward, turn, and crouch scenarios for seeds 42, 43, and 44,
1000 policy steps each. Each invocation uses the same exact final checkpoint
and a new evidence directory. The commands and scenario overrides are
recorded verbatim in their manifests.

Acceptance uses the fixed thresholds from the design. Every seed must be
finite and meet its survival requirement. Per-seed and aggregate trajectory
hashes and metrics are required. Videos do not replace numeric evidence.

### 4.8 Evidence green command

```bash
OPENHOMIE_M5_RUN=/absolute/completed/run \
  /home/user/miniconda3/envs/homie/bin/python -m pytest -q \
  tests/test_elf3_m5_convergence_evidence.py

/home/user/miniconda3/envs/homie/bin/python -m pytest -q tests
```

Batch C changes no production code.

## 5. Final M5 audit

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
