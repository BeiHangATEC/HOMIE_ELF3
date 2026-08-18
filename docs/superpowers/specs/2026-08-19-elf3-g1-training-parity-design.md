# ELF3 G1 Training-Side Parity Design

## Status

Approved direction: align ELF3 training behavior with upstream G1 wherever the
robot geometry does not require a difference. ELF3 keeps its own absolute
height envelope of `[0.40, 1.01] m`.

## Goal

Make the active ELF3 single-stage task use the G1 command and reward semantics
while retaining ELF3-specific robot facts. In particular:

- zero-velocity commands receive the same velocity-tracking credit as G1;
- height tracking has the same formula in every command mode;
- the two sole-parallel penalties use the G1 reward scales;
- height-hold commands sample uniformly from `[0.40, 1.01] m`;
- velocity-walk and tall-stand commands keep a height of `1.01 m`.

## Parity Boundary

The following training-side behavior is shared with G1:

- one fixed command distribution with probabilities `1/2` velocity walk,
  `1/3` height hold, and `1/6` tall stand;
- velocity ranges `vx [-0.8, 1.2] m/s`, `vy [-0.5, 0.5] m/s`, and yaw rate
  `[-0.8, 0.8] rad/s`;
- four-second command resampling;
- the common HOMIE reward formulas and scales described below;
- HIM-PPO rollout, optimization, symmetry, estimator, and policy-network
  settings described below.

ELF3 retains robot-dependent values: joint count and ordering, policy and
critic observation widths, initial pose, PD gains, armature, effort and
velocity limits, contact geometry, and absolute standing height. Copying these
from G1 would describe the wrong robot and is outside this change.

## Reward Semantics

`tracking_x_vel`, `tracking_y_vel`, and `tracking_ang_vel` return their raw G1
exponential tracking rewards for every command mode. A zero command paired with
zero measured velocity therefore returns `1.0` from each raw term. The ELF3
mode-dependent subtraction is removed.

`tracking_base_height` returns only the raw exponential height-tracking reward.
The ELF3 crouch multiplier is removed, so an exact height match returns `1.0`
in walk, stand, and height-hold modes.

The scale table changes to the exact G1 values:

| Reward term | New ELF3 scale | G1 scale |
| --- | ---: | ---: |
| `feet_ground_parallel` | `-2.0` | `-2.0` |
| `feet_parallel` | `-3.0` | `-3.0` |

All other reward scales remain unchanged because they already match G1. The
shared reward change applies to both the active single-stage task and the
legacy staged ELF3 task, avoiding two incompatible ELF3 reward definitions.

## Agent Configuration

The ELF3 policy and critic MLP widths change from `[512, 256, 128]` to the G1
configuration `[512, 256, 256]`. Their input widths remain derived from the
ELF3 observation layout.

The estimator already matches G1 at encoder and target widths `[256, 256]`,
latent width `32`, `64` prototypes, temperature `3.0`, Sinkhorn epsilon `0.05`,
three Sinkhorn iterations, and gradient norm `10.0`. Its configured learning
rate changes from the fixed `1e-3` value to following the adaptive PPO learning
rate, matching the G1 update path.

The runner default maximum changes from `50,000` to `100,000` iterations to
match G1. Training entry points still require or supply an explicit iteration
count, and that runtime request remains authoritative.

TensorBoard remains the ELF3 logger because the user-facing live curve monitor
depends on its event files. The logger backend does not change optimization.
The seed remains an explicit CLI input rather than being forced to G1's default;
equal runs can use equal seeds through their launch commands.

## Height Commands

`Elf3SingleStageEnvCfg.single_stage_height_range` changes from
`(0.30, 1.01)` to `(0.40, 1.01)`. The sampler continues to draw uniformly over
the configured range only for height-hold commands. Walk and tall-stand rows
continue to use exactly `1.01 m`.

The frozen `S0..S5` and `V1..V3` stage definitions remain unchanged. They are
not used by the active single-stage launcher and are tied to prior checkpoint
and acceptance evidence. They inherit the new shared reward semantics if used
for new training.

## Compatibility

Existing checkpoints remain structurally loadable because action and
observation dimensions do not change, but changing the policy and critic MLP
widths means their model parameters do not load into the new network. The new
configuration therefore requires a fresh run. Old checkpoints remain usable
only with their recorded old configuration and source revision; they are not
valid resume inputs for the aligned configuration.

No checkpoint, training run, TensorBoard event, or generated evidence is part
of the source change.

## Validation

Focused CPU tests must prove:

1. the two sole-parallel scales equal `-2.0` and `-3.0`;
2. exact zero-velocity tracking returns `1.0` for walk, stand, and height-hold
   modes;
3. exact height tracking returns `1.0` for all modes and is never doubled;
4. the single-stage configured range is exactly `(0.40, 1.01)`;
5. deterministic height draws map to both endpoints and remain inside them;
6. walk and tall-stand commands retain the exact `1.01 m` height;
7. the legacy stage definitions are unchanged;
8. actor and critic widths are exactly `[512, 256, 256]`;
9. the estimator follows the adaptive policy learning rate and retains the G1
   architecture and loss parameters;
10. the runner default maximum is `100,000` iterations while CLI overrides
    remain effective.

After focused tests, run the full CPU test suite. Then run a small headless
Isaac Lab training smoke with a fresh output directory and verify that one PPO
iteration completes, logs all reward terms, and writes its final checkpoint.

## Acceptance Criteria

- ELF3 source contains no mode-dependent velocity reward subtraction and no
  crouch-only height multiplier.
- The active single-stage manifest reports height range `[0.40, 1.01]`.
- The effective configuration reports the G1 policy widths and estimator
  learning-rate coupling.
- Focused and full CPU tests pass.
- The headless Isaac Lab smoke completes without non-finite observations,
  rewards, or losses.
- No training artifact is committed.
