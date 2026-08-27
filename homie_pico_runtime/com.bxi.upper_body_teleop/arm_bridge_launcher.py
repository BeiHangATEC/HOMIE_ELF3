#!/usr/bin/env python3
"""Select a Python containing the robotics Pinocchio API and run the bridge."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


CONFIG_ERROR = getattr(os, "EX_CONFIG", 78)
PROBE_TIMEOUT_S = 15.0
REQUIRED_IMPORTS = (
    "numpy",
    "zmq",
    "rclpy",
    "sensor_msgs.msg",
    "std_msgs.msg",
    "bxi_example_py_elf3",
)
PINOCCHIO_API = (
    "buildModelFromUrdf",
    "forwardKinematics",
    "computeJointJacobians",
    "integrate",
)


@dataclass(frozen=True, slots=True)
class PythonRuntime:
    executable: Path
    pinocchio_path: Path | None
    pinocchio_version: str
    pinocchio_file: str


def _resolve_python(value: str | Path) -> Path | None:
    text = str(value).strip()
    if not text:
        return None
    if os.sep not in text:
        found = shutil.which(text)
        if found is None:
            return None
        candidate = Path(os.path.abspath(found))
    else:
        candidate = Path(os.path.abspath(os.path.expanduser(text)))
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def _candidates() -> tuple[Path, ...]:
    explicit = os.environ.get("BXI_ARM_IK_PYTHON", "").strip()
    if explicit:
        candidate = _resolve_python(explicit)
        return (candidate,) if candidate is not None else ()

    values: list[str | Path] = [sys.executable]
    for variable in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        root = os.environ.get(variable, "").strip()
        if root:
            values.append(Path(root) / "bin" / "python")
    values.extend(
        (
            Path.home() / "miniconda3" / "bin" / "python",
            Path.home() / "anaconda3" / "bin" / "python",
            "python3",
        )
    )
    result: list[Path] = []
    for value in values:
        candidate = _resolve_python(value)
        if candidate is not None and candidate not in result:
            result.append(candidate)
    return tuple(result)


def _probe(executable: Path) -> tuple[PythonRuntime | None, str]:
    script = (
        "import importlib, json, pathlib, sys, sysconfig\n"
        "purelib = pathlib.Path(sysconfig.get_paths()['purelib'])\n"
        "version = f'python{sys.version_info.major}.{sys.version_info.minor}'\n"
        "cmeel = purelib / 'cmeel.prefix' / 'lib' / version / 'site-packages'\n"
        "if cmeel.is_dir(): sys.path.insert(0, str(cmeel))\n"
        "pin = importlib.import_module('pinocchio')\n"
        f"required_api = {PINOCCHIO_API!r}\n"
        "missing = [name for name in required_api if not callable(getattr(pin, name, None))]\n"
        "if missing: raise RuntimeError('wrong pinocchio package; missing robotics "
        "API: ' + ', '.join(missing))\n"
        f"imports = {REQUIRED_IMPORTS!r}\n"
        "for name in imports: importlib.import_module(name)\n"
        "print(json.dumps({'pinocchio_path': str(cmeel) if cmeel.is_dir() else '', "
        "'pinocchio_version': str(getattr(pin, '__version__', 'unknown')), "
        "'pinocchio_file': str(getattr(pin, '__file__', 'unknown'))}))\n"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            (str(executable), "-s", "-c", script),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    if completed.returncode != 0:
        output = completed.stderr.strip() or completed.stdout.strip()
        return None, output.splitlines()[-1] if output else "probe failed"
    try:
        data = json.loads(completed.stdout.strip().splitlines()[-1])
        pinocchio_path = (
            Path(data["pinocchio_path"]) if data["pinocchio_path"] else None
        )
        return (
            PythonRuntime(
                executable=executable,
                pinocchio_path=pinocchio_path,
                pinocchio_version=str(data["pinocchio_version"]),
                pinocchio_file=str(data["pinocchio_file"]),
            ),
            "",
        )
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        return None, f"invalid probe output: {exc}"


def select_runtime() -> PythonRuntime:
    candidates = _candidates()
    explicit = os.environ.get("BXI_ARM_IK_PYTHON", "").strip()
    if explicit and not candidates:
        raise RuntimeError(f"BXI_ARM_IK_PYTHON is not executable: {explicit}")
    failures = []
    for candidate in candidates:
        runtime, error = _probe(candidate)
        if runtime is not None:
            return runtime
        failures.append(f"  - {candidate}: {error}")
    detail = "\n".join(failures) if failures else "  - no Python candidates found"
    raise RuntimeError(
        "no Python runtime provides robotics Pinocchio plus ROS dependencies; "
        "the unrelated PyPI package 'pinocchio' is not compatible. Checked:\n"
        + detail
    )


def main() -> int:
    try:
        runtime = select_runtime()
        script = Path(__file__).resolve().parent / "arm_bridge.py"
        if not script.is_file():
            raise FileNotFoundError(f"arm bridge is missing: {script}")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[upper-body-arm-ik] configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR

    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if runtime.pinocchio_path:
        inherited = environment.get("PYTHONPATH", "")
        paths = [str(runtime.pinocchio_path)]
        if inherited:
            paths.extend(path for path in inherited.split(os.pathsep) if path)
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    print(
        "[upper-body-arm-ik] selected "
        f"{runtime.executable}; pinocchio={runtime.pinocchio_version} "
        f"({runtime.pinocchio_file})",
        flush=True,
    )
    os.execve(
        str(runtime.executable),
        (
            str(runtime.executable),
            "-u",
            "-B",
            "-s",
            str(script),
            *sys.argv[1:],
        ),
        environment,
    )
    return getattr(os, "EX_SOFTWARE", 70)


if __name__ == "__main__":
    raise SystemExit(main())
