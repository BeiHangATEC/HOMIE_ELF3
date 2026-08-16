# ELF3 M4 HIM PPO Implementation Plan

Date: 2026-08-17
Design: `docs/superpowers/specs/2026-08-17-elf3-m4-him-ppo-design.md`
Status: Approved for implementation under the HANDOFF blanket authorization

## 1. Ownership and execution rules

- The dedicated gpt-5.6-sol planning/test/acceptance agent owns this plan,
  every M4 test, and final acceptance execution.
- The implementation agent owns only the production files listed below. It
  must not edit tests to make a failure pass.
- The five M3a production files are frozen:
  `elf3_homie_env.py`, `elf3_homie_env_cfg.py`, `elf3_homie_env_core.py`,
  the ELF3 registration `__init__.py`, and `random_rollout_elf3.py`.
- Installed `rsl_rl`, Isaac Lab, upstream G1, HOMIE reference repositories,
  and all M5/M6 files are read-only.
- Dimensions and mirror data come from `elf3_constants`; production must not
  contain the reference 29-DOF dimensions 80, 83, or 480.
- Every implementation batch follows red -> minimal green -> focused suite ->
  complete `tests/` suite. A red test is changed only when the approved design
  itself is revised by the test owner.

## 2. Exact production files

Create only these M4 production files:

```text
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/__init__.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/estimator.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/actor_critic.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/symmetry.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/storage.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/ppo.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/runner.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/exporter.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/agents/__init__.py
isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/agents/him_ppo_cfg.py
```

Do not add train, play, evaluate, or export CLI scripts in M4. The exporter is
a library API; M5 owns user-facing entry points and training manifests.

## 3. Test files owned by the planning agent

```text
tests/_elf3_m4_helpers.py
tests/test_elf3_m4_estimator.py
tests/test_elf3_m4_actor_symmetry.py
tests/test_elf3_m4_storage_ppo.py
tests/test_elf3_m4_ppo_edges.py
tests/test_elf3_m4_runner_export.py
tests/test_elf3_m4_static_contract.py
```

The helper derives all widths and mirror data from canonical constants. It is
not a second source of dimensions.

## 4. Baseline gate

Before any M4 test or production change:

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest tests/ -q
```

Expected baseline: `126 passed`.

## 5. Batch A: estimator and target layout

### Red tests

`test_elf3_m4_estimator.py` specifies:

- rsl-rl 3.1.2 is installed and the module uses its public `MLP` surface;
- Sinkhorn accepts rank-2 logits, is deterministic, finite for logits near
  `+/-1000`, nonnegative, and approximately balanced;
- invalid epsilon, iterations, rank, or empty axes fail clearly;
- estimator output is `(base_velocity[3], normalized_latent[32])`;
- target velocity is the critic tail and target encoder input drops the three
  velocity commands before appending true base velocity;
- masks exclude invalid done rows and an all-false mask returns differentiable
  finite zeros without an optimizer step;
- prototype scoring/renormalization and all estimator gradients are finite;
- enabled normalization uses actor-history latest-frame statistics for the
  target actor region and critic-tail statistics for target velocity.

Run red:

```bash
python -m pytest tests/test_elf3_m4_estimator.py -q
```

Expected red: import failure for missing `openhomie_isaaclab.him_rl`, not a
syntax, fixture, dependency, or dimension error.

### Green implementation

Implement `him_rl/estimator.py` with a named target-layout helper and log-space
Sinkhorn. Export only the intended public symbols from `him_rl/__init__.py`.

Run green and baseline:

```bash
python -m pytest tests/test_elf3_m4_estimator.py -q
python -m pytest tests/ -q
```

## 6. Batch B: actor-critic and spec symmetry

### Red tests

`test_elf3_m4_actor_symmetry.py` specifies:

- the rsl-rl 3.1.2 `TensorDict` constructor and inference/value/distribution
  signatures;
- widths derived from canonical 28 DOFs and 12 actions;
- estimator features are detached from actor loss;
- actor, critic, estimator, action-noise, and normalizer state round-trip;
- normalizers update exactly once per returned step and the enabled source,
  actor, target, critic, and export paths use the approved statistics;
- policy and estimator parameter iterators are nonempty, disjoint, and cover
  all registered parameters;
- canonical actor history, critic, and action mirrors are exact involutions;
- invalid permutations/signs/groups/dimensions fail during construction.

### Green implementation

Implement `actor_critic.py` and `symmetry.py`. Reuse rsl-rl
`EmpiricalNormalization`; do not create a private running-statistics format.
No ELF3 joint indices belong in either generic module.

Commands:

```bash
python -m pytest tests/test_elf3_m4_actor_symmetry.py -q
python -m pytest tests/test_elf3_m4_estimator.py tests/test_elf3_m4_actor_symmetry.py -q
python -m pytest tests/ -q
```

## 7. Batch C: branch storage, GAE, and PPO

### Red tests

`test_elf3_m4_storage_ppo.py` specifies:

- storage subclasses the rsl-rl 3.1.2 storage and preserves its ten-field
  minibatch prefix while appending next critic observations and masks;
- one-branch GAE exactly matches stock storage;
- two interleaved branches match two independent stock GAE calculations and
  never bootstrap across branches;
- wrong final-value shape, incomplete rollout, invalid mask, overflow, and
  recurrent use fail explicitly;
- continuing/done/terminal-mask truth tables prevent cross-episode estimator
  targets while accepting valid failure and timeout terminal snapshots;
- timeout reward bootstrap remains distinct from estimator masking;
- original and mirror branches are sampled from their own distributions;
- optimizers have disjoint ownership, update both intended parameter sets,
  obey independent/following LR rules, and reject non-finite updates before
  either optimizer commits partial state.

### Green implementation

Implement `storage.py` and `ppo.py`. Store logical `[time, branch, env]`
semantics explicitly. Preserve stock PPO equations and rsl-rl public calls;
reject `rnd_cfg`, generic `symmetry_cfg`, and recurrent policies.

Commands:

```bash
python -m pytest tests/test_elf3_m4_storage_ppo.py -q
python -m pytest tests/test_elf3_m4_estimator.py \
  tests/test_elf3_m4_actor_symmetry.py tests/test_elf3_m4_storage_ppo.py -q
python -m pytest tests/ -q
```

## 8. Batch D: runner, checkpoint, and export

### Red tests

`test_elf3_m4_runner_export.py` specifies with a deterministic fake `VecEnv`:

- construction through named HIM classes and standard observation groups;
- a complete rollout with continuing, terminated, and truncated rows followed
  by a finite update;
- completed iteration accounting over repeated `learn` calls;
- atomic checkpoint save and strict resume of model, both optimizers,
  normalizers, learning rates/follow mode, iteration, and infos;
- malformed checkpoint and missing estimator optimizer failures;
- exporter contains actor-history normalizer, estimator source encoder, latent
  normalization, latest-frame extraction, and actor;
- loaded TorchScript parity is `max_abs_error <= 1e-7`;
- independently loaded ONNX Runtime parity is `max_abs_error <= 1e-5` for zero,
  patterned, and seeded random batches, including dynamic batch size > 1;
- ONNX 1.21.0 and ONNX Runtime 1.28.0 are mandatory and never skipped.

### Green implementation

Implement `runner.py` and `exporter.py`. Whitelist classes rather than using
`eval`. Use same-directory temporary files plus `os.replace` for checkpoints.
The exporter excludes critic, target encoder, prototypes, noise, and optimizer
state but includes the complete deterministic action path.

Commands:

```bash
python -m pytest tests/test_elf3_m4_runner_export.py -q
python -m pytest tests/test_elf3_m4_*.py -q
python -m pytest tests/ -q
```

Expected exporter evidence includes real ONNX Runtime execution; no skip count
is accepted.

## 9. Batch E: ELF3 agent config and static boundary

### Red tests

`test_elf3_m4_static_contract.py` specifies:

- all ten production files exist and compile;
- generic HIM modules contain no ELF3 joint-name or task imports;
- agent config derives 28/12/78/81/468 and mirror data from canonical facts;
- the config selects `HIMOnPolicyRunner`, `HIMPPO`, and `HIMActorCritic`, uses
  `policy`/`critic` observation groups, and retains approved PPO/HIM values;
- production contains neither machine-specific paths nor the reference
  29-DOF literals 80/83/480;
- installed package metadata resolves exactly `rsl-rl-lib==3.1.2`,
  `onnx==1.21.0`, and `onnxruntime==1.28.0`;
- no M3a, upstream, M5, or M6 production file changed in the M4 commit.

### Green implementation

Add the agents package and `him_ppo_cfg.py`. Use Isaac Lab config classes while
deriving every dimension and mirror field from `elf3_constants`.

Commands:

```bash
python -m pytest tests/test_elf3_m4_static_contract.py -q
python -m pytest tests/test_elf3_m4_*.py -q
python -m pytest tests/ -q
```

## 10. Final acceptance

The planning/test agent independently runs:

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest tests/ -q
/home/user/miniconda3/envs/homie/bin/python -m pytest tests/test_elf3_m4_*.py -q
git diff --check
git status --short
```

Acceptance requires zero failures and zero skips in the M4 suite, actual ONNX
Runtime parity, and inspection that the implementation commit changes only the
ten approved production files. The planning/test agent records test counts,
versions, parity maxima, and the exact commit.

## 11. M5 and later non-goals

M4 does not claim or implement training convergence, learned standing or
walking, train/play CLI workflows, curriculum orchestration, immutable run
manifests, evaluation videos, deployment commands, MuJoCo, export CLI wiring,
or sim2sim. M5 may consume the runner, checkpoint, and exporter libraries only
after M4 acceptance.
