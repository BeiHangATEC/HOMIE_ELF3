# ELF3 Single-Stage Live Logging Design

## Goal

Start fresh ELF3 single-stage training from a normal terminal while showing
the effective request, key fixed configuration, and the existing per-iteration
HIM-PPO progress logs immediately.

## Root Cause

`conda run` captures the child process standard streams by default. Flushing
inside the Python trainer does not make those captured streams live. The
launcher, not the simulator or PPO loop, must disable that capture.

## Design

Add a POSIX shell launcher next to `elf3_single_stage_train.py`. It resolves
its own directory, executes `conda run --no-capture-output -n homie python`,
and forwards every user argument unchanged to the Python entry point. `exec`
keeps Ctrl-C and the resulting exit status attached to the actual trainer.

The Python entry point prints one JSON record before `AppLauncher` starts. The
record contains the user request and the fixed single-stage command profile.
After Isaac Lab loads the registry configuration, the workflow prints one JSON
record with the effective command envelope and the PPO settings that otherwise
only appear in `manifest.json`. The existing runner continues to print each
iteration's timing, losses, reward, episode length, and termination metrics.

## Error Handling

The shell launcher requires `conda` in `PATH`; a missing executable produces a
standard shell error. Argument validation, exclusive run-directory creation,
and training failure behavior remain owned by the existing Python workflow.
No output is redirected or forced to `/dev/tty`, so `screen`, pipes, files, and
CI retain normal stream semantics.

## Tests

Add a static test for the shell launcher: executable shebang, required
`--no-capture-output`, selected `homie` environment, target Python script, and
argument forwarding. Add pure-Python tests for both JSON summaries and retain
the existing CLI/import tests. Run the focused suite, byte-compile Python
modules, and run the launcher help command through the `homie` environment.

## Scope

The change applies only to the ELF3 single-stage entry point. It does not
modify the frozen staged environment, the C1/C3 workflows, physics, rewards,
or checkpoint semantics.
