# R1 Sim2Sim Gate Debug Session

Status: OPEN
Session: r1-sim2sim-gate

## Constraints
- Steps 1-4 do not modify policy or simulator business logic.
- No HTTP instrumentation or policy behavior changes.
- Do not replace the ATEC model_25999 deployment model.
- Official DOF29 assets cannot substitute for the required DOF31 model.

## Falsifiable hypotheses
1. The current exporter produces an unmodified 504-to-12 policy with acceptable TorchScript/ONNX error.
2. No validated official DOF31 MJCF exists in the repository or ATEC tree.
3. A derived DOF31 asset can be justified against official DOF31 URDF collision geometry while preserving model contract.
4. The selected model_2500 policy remains alive for at least 30 simulated seconds at a fixed 0.974 m height command.
5. Any gate failure can be attributed using posture, relative torso-to-sole height, contact, slip, action, and timing evidence.

## Evidence log
- Checkpoint downloaded without touching ATEC model_25999: SHA256 93b28ca3ceb02369165e62f30f36b2cb81ac61eb6f12170c0b54cdc10e285492.
- Export confirmed input 504, output 12; TorchScript/ONNX max error 3.81469727e-06; ONNX SHA256 07fd35955e115815af209d5a42befb60c9a05617cafbda9657658b297afea0fb.
- ATEC audit found DOF29 MJCF and DOF31 URDF, but no validated DOF31 MJCF. DOF29 was not substituted.
- Official DOF31 ankle collision STL bounds are min [-0.09, -0.03999984, -0.041], max [0.15, 0.03999984, 0.0135] m.
- Derived DOF31 MJCF changed only mesh-relative paths and both foot collision boxes from pos z=+0.18 to a box spanning the verified collision mesh x/y and sole bottom z=-0.041 m. Inertials, joints, actuators, timestep and 43.22248028 kg total body mass were preserved.
- 30.0 simulated seconds used the selected ONNX SHA256. Result: alive=false; torso-to-supporting-sole height mean 0.12984 m, final 0.13119 m; roll max 179.90 deg; pitch max 87.64 deg; non-foot contact fraction 0.9715.
- Hypotheses 1, 2, 3 and 5 confirmed. Hypothesis 4 rejected.
- Gate decision: FAIL. This is derived-asset evidence only, not an official-asset pass. R2 must not be created.
