# ELF3 M3a Environment Core Implementation Plan

Date: 2026-08-16
Design: `docs/superpowers/specs/2026-08-16-elf3-m3a-environment-core-design.md`
Scope: M3a only

## Ownership and discipline

- The dedicated gpt-5.6-sol planning/test/acceptance agent owns this plan,
  `tests/`, and final independent acceptance.
- The separate gpt-5.6-sol execution agent owns production implementation.
- The execution agent must not edit tests or broaden scope into HIM, training,
  play, export, G1, MuJoCo, or sim2sim.
- Use `/home/user/miniconda3/envs/homie/bin/python`; the shell's default Python
  skips Torch tests and does not prove the 108-test baseline.
- Do not modify `HomieRL/`, `HomieDeploy/`, or `HomieHardware/`.
- Do not copy HOMIE literals or assume runtime DOF order. Reuse formulas and
  behavior only after reconciling them with OpenHomie's 28-DOF constants.

## Required production files

Implement in this order:

1. `.../elf3/elf3_homie_env_core.py` (new, Isaac-independent tensor contract)
2. `.../elf3/elf3_homie_env_cfg.py` (new, declarative DirectRLEnv config)
3. `.../elf3/elf3_homie_env.py` (new, Isaac-dependent lifecycle)
4. `.../elf3/__init__.py` (register the exact task ID)
5. `isaaclab_ext/scripts/random_rollout_elf3.py` (new acceptance entry point)

The abbreviated prefix above is
`isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion`.

## Step 0: Preserve the baseline

Run before every milestone:

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest \
  tests/test_elf3_constants.py tests/test_elf3_usd_asset.py \
  tests/test_elf3_rewards.py tests/test_elf3_curriculum.py -q
```

Expected: `108 passed`. Skips mean the wrong interpreter was used.

## Step 1: Pure environment core

Create `elf3_homie_env_core.py` without importing Isaac Lab. Implement the
public functions specified by `tests/test_elf3_env_core_contract.py`:

- `build_name_permutation(runtime_names, canonical_names)` returns a CPU
  `torch.long` canonical-to-runtime index tensor and rejects missing, extra,
  or duplicate names.
- `gather_canonical(runtime_values, canonical_to_runtime)` gathers the final
  tensor axis into canonical order.
- `scatter_canonical(canonical_values, canonical_to_runtime, runtime_width,
  base=None)` writes only mapped slots and preserves `base` elsewhere.
- `assemble_actor_frame(...)` implements the exact 78-channel order and scales
  using `elf3_constants`; height remains unscaled.
- `shift_history(history, frame, reset_env_ids)` shifts oldest to newest and
  clears old frames for reset environments before inserting the new frame.
- `compute_leg_efforts(...)` implements randomized PD plus actuation offset and
  symmetric per-joint effort clamp.
- `apply_control_randomization(...)` accepts injected draws for deterministic
  testing and updates only selected environment IDs using assignment semantics.
- `classify_dones(failure, episode_length, max_episode_length)` returns
  `terminated`, `truncated`, and `time_outs`; failure wins on the final step.
- `virtual_sole_corners(ankle_pos_w, ankle_quat_wxyz)` returns shape
  `(num_envs, 2, 4, 3)`: four world-frame points per foot, derived only from
  `C.SOLE_LENGTH`, `C.SOLE_WIDTH`, and `C.SOLE_CENTER_OFFSET`. Quaternion input
  is wxyz. ELF3 has no G1/HOMIE contact-marker dependency.

Red command before implementation:

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest \
  tests/test_elf3_env_core_contract.py -q
```

Expected red: one failure stating that `elf3_homie_env_core.py` is missing;
remaining behavior tests skip until that interface exists. Collection errors,
fixture errors, or syntax errors are test defects and must be fixed by the
planning/test agent, not worked around in production.

Green command after implementation: the same command.

Expected green: all contract tests pass, no skip, no Isaac import or GPU use.

## Step 2: Direct environment configuration

Create `elf3_homie_env_cfg.py` as a declarative `DirectRLEnvCfg`:

- simulation dt is `C.SIM_DT`, decimation is `C.DECIMATION`, and episode length
  is `C.EPISODE_LENGTH_S`;
- action and observation widths refer directly to `C.NUM_POLICY_ACTIONS`,
  `C.num_actor_obs()`, and `C.num_critic_obs()`;
- scene contains the ELF3 articulation, plane, replicated environments, and
  contact sensor data required by rewards and termination;
- startup events include `massless_link_event()`;
- reset events and control-domain ranges match the approved HOMIE behavior,
  except the known advanced-index mutation bug must not be copied;
- command, reward, curriculum, and termination parameters use existing
  OpenHomie modules and constants without changing their semantics.

Red/green static command:

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest \
  tests/test_elf3_m3a_static_contract.py -q
```

Initially expected red: missing cfg/env/rollout files and empty registration.
After this step, cfg-related assertions may turn green while env/registration
and rollout assertions remain red. Do not weaken the remaining assertions.

## Step 3: Direct environment lifecycle

Create `elf3_homie_env.py` and implement the DirectRLEnv hooks in lifecycle
order:

1. Resolve all joint and body names and build runtime/canonical mappings.
2. Allocate commands, histories, previous actions, randomized control tensors,
   curriculum state, episode sums, and diagnostics on the environment device.
3. Reset only selected IDs; sample commands, upper-body targets, and domain
   randomization; clear per-episode state.
4. Validate 12-wide finite actions, scale them, scatter by resolved leg names,
   compute/clamp leg effort, and set upper-body position targets.
5. Gather state in canonical order and assemble actor history plus critic obs.
6. Evaluate all existing reward terms and apply scale times policy dt once.
7. Return failure as `terminated`, time limit as `truncated`, and expose
   `extras["time_outs"]` only for pure truncation. Capture terminal observation
   and diagnostics before reset.

The implementation must use Isaac Lab wxyz quaternion helpers and the IMU body
frame. It must not use Isaac Gym xyzw helpers or root-frame shortcuts.

The environment exposes these minimal read-only acceptance diagnostics. Their
names and ordering are part of M3a and must not be left for the executor to
invent:

- `canonical_to_runtime_dof_indices`: `(28,)` long tensor satisfying
  `tensor[i] == robot.joint_names.index(C.JOINT_NAMES[i])`; it is nonidentity
  for the actual ELF3 articulation.
- `applied_torques_canonical` and `effort_limits_canonical`: `(N, 28)` tensors
  in `C.JOINT_NAMES` order. Therefore `C.LOWER_DOF_INDICES` is valid for both;
  runtime-ordered private tensors must not be exposed under these names.
- `reward_term_names`: exact ordered tuple `R.REWARD_NAMES`.
- `reward_scales`: exact mapping `R.REWARD_SCALES`.
- `last_raw_reward_terms` and `last_scaled_reward_terms`: dictionaries with
  exactly the 33 reward names. For every name, scaled equals
  `raw * R.REWARD_SCALES[name] * C.policy_dt()`, proving dt is applied once.

Run after each lifecycle slice:

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest \
  tests/test_elf3_env_core_contract.py \
  tests/test_elf3_m3a_static_contract.py -q
```

Expected when Step 3 is complete: all pure/static M3a tests pass except the
still-missing registration/rollout assertions assigned to Steps 4-5.

## Step 4: Exact task registration

Update only `tasks/locomotion/elf3/__init__.py`. Register exactly:

```text
OpenHomie-Elf3-Homie-Direct-v0
```

The environment entry point must resolve `Elf3HomieEnv`; the cfg entry point
must resolve `Elf3HomieEnvCfg`. Importing/reloading must not cause a duplicate
registration failure.

The package root, `tasks/__init__.py`, and `tasks/locomotion/__init__.py` remain
lightweight and must not be changed to recursively import tasks. Every script
that needs registration, including `random_rollout_elf3.py` and the Isaac
acceptance harness, explicitly imports
`openhomie_isaaclab.tasks.locomotion.elf3` before `gym.spec` or `gym.make`.

Run:

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest \
  tests/test_elf3_m3a_static_contract.py -q
```

Expected: file and registration assertions pass; only rollout-file assertions
may remain red.

## Step 5: Random rollout and Isaac acceptance entry point

Create `isaaclab_ext/scripts/random_rollout_elf3.py` using Isaac Lab's
`AppLauncher`. Required CLI:

- `--headless`
- `--num-envs`
- `--steps` (policy steps, default 1000)
- `--seed`
- `--device`
- `--episode-length-s` test override
- `--disable-failure-termination` only for controlled timeout acceptance
- `--induce-failure` only for controlled non-timeout termination acceptance

The script must sample finite 12-wide actions, reset done environments, check
all observations/rewards/actions/efforts/diagnostics for finiteness, and print
the following before `simulation_app.close()`:

```text
random rollout steps: 1000
observed terminated: <count>
observed time_outs: <count>
M3a random rollout: PASS
random_rollout_elf3 exit code: 0
```

On exception, print traceback, `M3a random rollout: FAIL`, and a nonzero
internal exit code before closing.

Static green command:

```bash
/home/user/miniconda3/envs/homie/bin/python -m pytest \
  tests/test_elf3_m3a_static_contract.py -q
```

Expected: all static assertions pass.

## Step 6: Isaac integration verification by execution agent

Run the integration harness only after Steps 1-5 are green:

```bash
/home/user/miniconda3/envs/homie/bin/python \
  tests/isaac/check_elf3_m3a_env.py --headless --num-envs 8
```

The harness creates a deterministic nominal acceptance profile without
changing `Elf3HomieEnvCfg` defaults. Before `gym.make`, it keeps only the
`massless_link_mass` startup event and sets the following test-cfg event terms
to `None`: `physics_material`, `non_torso_link_mass`, `torso_payload`,
`hand_payload`, `torso_com`, and `push_robot`. It also disables observation
noise, action delay, control randomization, and initial-state randomization in
that cfg instance. The approximately 43.22 kg assertion therefore validates
the nominal URDF mass after the massless-link correction; it must not reject
the intentional production mass/payload randomization.

Expected lines before shutdown:

```text
task registration: PASS
runtime joint mapping: PASS (28 joints)
reset/step shapes and finiteness: PASS
effort limits and mixed control: PASS
reward wiring and single dt scaling: PASS
subset reset isolation: PASS
runtime mass: PASS (approximately 43.22 kg)
done separation for observed step: PASS
M3a Isaac integration: PASS
check_elf3_m3a_env exit code: 0
```

Then run the production rollout:

```bash
/home/user/miniconda3/envs/homie/bin/python \
  isaaclab_ext/scripts/random_rollout_elf3.py \
  --headless --num-envs 32 --steps 1000 --seed 42 --device cuda:0
```

Unlike the nominal integration harness, this command uses the unchanged
production randomization defaults. Expected: 1000 steps, no non-finite values,
PASS, internal exit code 0. Its total mass is allowed to differ from 43.22 kg
within the configured domain-randomization ranges.

Controlled timeout:

```bash
/home/user/miniconda3/envs/homie/bin/python \
  isaaclab_ext/scripts/random_rollout_elf3.py --headless --num-envs 8 \
  --steps 20 --episode-length-s 0.08 --disable-failure-termination --seed 42
```

Expected: `observed time_outs` is positive, the rollout asserts that those
entries are truncated and not terminated, PASS, internal exit code 0. This is
the only command in M3a acceptance allowed to claim timeout behavior.

Controlled failure:

```bash
/home/user/miniconda3/envs/homie/bin/python \
  isaaclab_ext/scripts/random_rollout_elf3.py --headless --num-envs 8 \
  --steps 5 --induce-failure --seed 42
```

Expected: `observed terminated` is positive, the rollout asserts `time_outs`
is false for those failures, PASS, internal exit code 0. This is the command
that proves induced-failure behavior.

Isaac shutdown can make an outer timeout return 124. Acceptance uses the
internal PASS and exit-code line printed before close. If that line is absent,
the result is inconclusive, not passing.

## Step 7: Execution-agent handoff checklist

Before handing back for independent acceptance, the execution agent reports:

- exact files changed (production files only);
- full output of the 108-test baseline;
- full output of both new static/pure test files with no skips;
- full output of the Isaac integration harness;
- full output of normal, timeout, and induced-failure rollout commands;
- confirmation that no tests, upstream G1 paths, or M4-M6 files were changed.

The planning/test/acceptance agent then reruns every command independently.
M3a completion language is limited to environment-contract correctness and
random-rollout numerical stability. It must not claim learned standing,
walking, convergence, train/play/export, or sim2sim support.
