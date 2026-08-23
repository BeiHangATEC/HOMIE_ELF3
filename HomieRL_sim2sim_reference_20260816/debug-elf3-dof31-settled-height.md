# ELF3 DOF31 settled-height debugging

Status: [BLOCKED]
Session: elf3-dof31-settled-height

## Hypotheses
1. Isaac 1.0557 m is fixed-base/FK geometry, not dynamically settled standing height.
2. MJCF foot_collision local +0.18 z creates an asset-semantic discrepancy relative to URDF contact geometry.
3. R1 still reuses geometric height in command/reset or inherits an old-stage reward override.
4. Lateral feet/knee reward scales must follow implementation sign rather than reward names.
5. Deterministic acceptance may not be loading the R1 contract.

## Evidence
- Pre-fix calibration JSON records Isaac `fix_base_link=True` FK geometry as 1.0556987101 m after one simulation step.
- URDF uses ankle collision STL; MJCF replaces it with a box at ankle-local `[0.01, 0, 0.18]`, half-size `[0.10, 0.04, 0.02]`; geometric difference is 0.2011760723 m.
- Local MuJoCo dynamic run (3000 steps, final 400) falls onto the body: torso z 0.110251 m, roll/pitch approximately 3.05/1.505 rad, `stable=false`; this rejects its use as an R1 standing-height source and confirms the asset discrepancy.
- Lateral reward functions return zero inside their windows and negative outside; therefore official positive scales 0.5/1.0 are penalty weights by implementation.
- `knee_names` intentionally contains left knee/hip and right knee/hip; implementation sums cross-side knee and hip spacing, while deterministic acceptance uses dedicated `static_knee_names` for actual knee spacing.
- R1 stage flags are all disabled and training rejects `--stage`, resume, and init checkpoints; terrain is a plane and pelvis-height scale is explicitly zero.
- Remote TCP/SSH handshake succeeds, but the provided endpoint rejects the available ED25519 key and noninteractive password authentication (`Permission denied (publickey,password)`).
- Local Isaac CPU fallback validates the new runtime path but shows zero-action default pose is not statically self-supporting: with initial root z 1.06 m, final-window torso z 0.109687 m and roll/pitch -2.18/1.452 rad (`stable=false`). Initial z 0.96 and the ATEC reference bent pose also fall. Therefore the required semantic has no valid dynamically settled standing height under the specified zero-action PD, and the script now refuses selection when `stable=false`.
- ATEC reference was consulted without modification: its default pose is hip/knee/ankle `-0.3/0.6/-0.3`, and its model uses sole capsules at ankle-local z=-0.03, unlike HomieRL MJCF +0.18-z boxes.

## Fix
- Calibration now reports separate geometric and dynamically settled sections, with free root, flat ground, zero velocities/actions, training-equivalent lower effort PD, upper default-target position PD, 3000 steps and a 400-step stability window.
- R1 contract no longer aliases geometric 1.0557 m as standing height; dynamic Isaac height is intentionally unset pending the required remote measurement. Reset root z is independently set to safe placement 1.12 m.
- Sim2Sim initial root placement now uses the reset placement value, not the height command.
