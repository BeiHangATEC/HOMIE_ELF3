# ELF3 M5 C3 Continual Training and Evidence Design

Status: approved execution plan

## Objective

Continue the accepted S0 policy checkpoint from local/global iteration 2000
to 4000 with checkpoint-state resume. Evaluate that exact `model_4000.pt`
with a pre-registered signed-response grid. Only when both velocity axes
support the OOD diagnosis, train V1, V2, and V3 from weights only for 2000
local iterations each. The final required identity is V3 local iteration 2000
and global iteration 10000.

## Identity and Budget

Each training block uses 4096 environments and 50 policy steps per iteration,
or 204800 transitions per iteration.

| Phase | Load mode | Local start/final | Global start/final | Transitions |
| --- | --- | --- | --- | --- |
| C2 S0 | fresh | 0 / 2000 | 0 / 2000 | 409600000 |
| C3 S0 | checkpoint-state resume | 2000 / 4000 | 2000 / 4000 | 409600000 |
| V1 | weights only | 0 / 2000 | 4000 / 6000 | 409600000 |
| V2 | weights only | 0 / 2000 | 6000 / 8000 | 409600000 |
| V3 | weights only | 0 / 2000 | 8000 / 10000 | 409600000 |

The complete chain totals 2048000000 transitions. Checkpoint filenames are
stage-local; stage directory, parent hash, and manifest identity are required
to disambiguate them.

## Resume Semantics

The S0 continuation restores the model, policy optimizer, estimator optimizer,
policy learning rate, estimator learning rate, and checkpoint iteration from
`model_2000.pt`. The historical checkpoint does not contain Python, NumPy,
Torch, CUDA RNG, environment state, or upper-body runtime state. It is
therefore a checkpoint-state resume, not a bitwise stochastic continuation.

Cross-stage V1/V2/V3 loads only `model_state_dict` into a newly constructed
runner. It must prove that policy and estimator optimizer states are fresh,
that both learning rates equal the configured base `0.001`, and that local
iteration is reset to zero. Full optimizer resume is reserved for interruption
within the same stage.

## Signed-Response Grid

All grid runs use the final S0 `model_4000.pt`, S0 configuration, seed 42, 43,
or 44, 16 environments, and 1000 policy steps. Every seed has one shared
`walk_zero` baseline (WALK mode, command `[0, 0, 0, 1.01]`). Each axis then
runs commands 0.1, 0.2, 0.3, 0.4, and 0.5. No C2 pilot or aborted run is part
of this evidence root.

Every raw NPZ records pre-action snapshots, post-action outcome tensors, fixed
commands, modes, body-frame linear/angular velocities, roll/pitch, tracking
height, actions, rewards, and active/done/timeout masks. The CPU checker
reconstructs credited samples and metrics from raw data with `allow_pickle=False`.

For forward, the S0 positive support boundary is 0.3; for yaw it is 0.2. For
each seed and axis, let `d(c)` be signed response at command `c` minus that
seed's shared zero response. The grid computes in-range and out-of-range
least-squares gains for both the full and post-initial windows.

An axis is `OOD_SUPPORTED` only if all conditions hold in both windows:

- the 0.5 three-seed MAE fails its frozen behavior threshold while every
  in-range command passes it;
- in-range gain is in `[0.70, 1.30]`, out-of-range gain is at most `0.50`,
  out/in gain ratio is at most `0.60`, and aggregate 0.5 response is positive;
- every seed has positive in-range and 0.5 response and out-of-range gain
  below in-range gain;
- all fixed-command, provenance, credit, timeout, and hash checks pass.

Only both axes `OOD_SUPPORTED` authorize V1. `GATE_PASSED`,
`IN_SUPPORT_DEFICIT`, `NO_OOD_BREAK`, `MIXED`, or `INVALID` stop promotion.
Original M5 behavior thresholds remain unchanged.

## Stage Ladder

V1, V2, and V3 use the authoritative stage definitions:

- V1: x `[-0.40, 0.60]`, yaw `[-0.40, 0.40]`.
- V2: x `[-0.60, 0.90]`, yaw `[-0.60, 0.60]`.
- V3: x `[-0.80, 1.20]`, yaw `[-0.80, 0.80]`.

Each parent source and stage transition is hash-bound in an immutable manifest.
The worker rejects an out-of-order stage, a wrong parent stage, an altered
parent manifest, or a nonzero local start for weights-only training.

## Final Delivery

After V3, rerun convergence, all four three-seed behavior scenarios, and exact
TorchScript/ONNX export parity from the one V3 local-2000/global-10000
checkpoint. Record Isaac Sim videos for the accepted final checkpoint in a
separate video evidence root; videos do not replace numeric acceptance. The
parameter bundle contains the final checkpoint, training manifests/results,
resolved stage configs, hashes, global/local identity, acceptance evidence,
and video manifests with file hashes and frame metadata.
