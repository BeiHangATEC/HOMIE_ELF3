# ELF3 Single-Stage G1 Command Training Design

## Status

Approved approach 1: retain ELF3's reward design while replacing staged command
training with one G1-proportioned command distribution.

## Goal

Provide an ELF3 training profile that runs from a single fixed command envelope
instead of the external `S0..S5` and `V1..V3` stage chain. The profile must
match the current upstream G1 command sampler's three outcome probabilities:

| Outcome | Probability | Command |
| --- | ---: | --- |
| Velocity walk | 1/2 | Uniform `vx`, `vy`, and yaw-rate in the full ELF3 ranges; height `1.01 m` |
| Height hold | 1/3 | Zero velocity; height uniform in `[0.30, 1.01] m` |
| Tall stand | 1/6 | Zero velocity; height `1.01 m` |

The velocity ranges are fixed for the entire run: `vx [-0.8, 1.2] m/s`,
`vy [-0.5, 0.5] m/s`, and yaw rate `[-0.8, 0.8] rad/s`. Commands still
resample every four seconds.

The command height is a torso-origin-to-sole target, matching the existing
ELF3 reward and observation convention. The lower bound is intentionally not
clamped: `0.30 m` is part of the requested training distribution. Existing
evidence only shows a measured minimum near `0.504 m` for a `0.30 m` demo
command, so this part of the distribution is out of the previously verified
tracking envelope and must be recorded as such, not silently changed.

## Design

### Profile Boundary

Add an explicit training profile selection with a `g1_single_stage` value.
Its configuration owns the full velocity envelope and `[0.30, 1.01] m` height
range. The profile is selected by the training entry point and written into
the resolved configuration and run manifest.

The existing staged profile, `elf3_stages.py`, and C3 `S0 -> V1 -> V2 -> V3`
workflow remain unchanged. They are tied to existing artifacts, signed
responses, video collection, and final-checking contracts; the new profile
does not claim compatibility with their checkpoint lineage.

### Command Sampling

Implement a pure-tensor sampler next to the existing curriculum helpers. It
will accept explicit random draws, construct the five-column command tensor,
and return the command mode required by the current reward functions.

One uniform mode draw uses the same inequalities as G1:

```text
draw < 1/3       -> height hold
draw > 1/2       -> velocity walk
otherwise        -> tall stand
```

This gives probabilities `1/3`, `1/2`, and `1/6`, respectively. Height hold
uses the existing crouch mode so the established ELF3 crouch reward shaping
continues to apply. Velocity walk and tall stand use the existing walk and
high-stand modes. No reward scale or reward formula is changed.

### Runtime Behavior

The environment dispatches between the legacy staged sampler and the new
single-stage sampler only inside command reset/resampling. Physics, action
delay, upper-body curriculum, observation construction, termination logic,
domain randomization, HIM, PPO, and symmetry remain identical.

The training CLI exposes a fresh single-stage command. It rejects a staged
checkpoint or stage-transition request under this profile, preventing an
ambiguous resume. The emitted command will request 4096 environments, seed 42,
CUDA 0, and the requested iteration count in a new run directory.

## Validation

Focused CPU tests will verify:

1. Boundary draws map to the exact G1 inequalities and each mode has the
   intended commands.
2. Height samples are inclusive of the configured `[0.30, 1.01] m` interval;
   walk and stand have height exactly `1.01 m`.
3. A deterministic large draw set has exact or tolerance-checked counts of
   `1/2`, `1/3`, and `1/6` for walk, height hold, and stand.
4. The legacy staged sampler and canonical stage definitions remain unchanged.
5. The profile configuration resolves into the manifest and incompatible
   checkpoint/stage combinations fail before Isaac Sim starts.

After focused tests, run the relevant complete Python test subset and a small
headless Isaac Lab smoke training invocation before presenting the full
4096-environment training command.

## Non-Goals

- Do not make `0.30 m` physically reachable by changing ELF3 geometry,
  joint limits, or reward semantics.
- Do not alter existing C3 artifacts, stage hashes, acceptance evidence, or
  signed-response tooling.
- Do not claim that a policy trained with this profile is comparable to the
  prior staged V3 checkpoint.
