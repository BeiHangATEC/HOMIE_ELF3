#!/usr/bin/env python3
"""Exec the shared SONIC PICO/RTSP runtime without modifying that Mod."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import sys


CONFIG_ERROR = getattr(os, "EX_CONFIG", 78)


def _platform_tag() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "linux-x86_64"
    if machine in {"aarch64", "arm64"}:
        return "linux-aarch64"
    return f"linux-{machine}"


def _sonic_root() -> Path:
    root = Path(__file__).resolve().parent.parent / "com.bxi.sonic"
    manifest = root / "mod.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(
            "required sibling Mod com.bxi.sonic is not installed beside "
            f"com.bxi.upper_body_teleop: {manifest}"
        )
    return root


def _resolve_command(mode: str, sonic_root: Path) -> tuple[str, ...]:
    if mode == "mediamtx":
        script = sonic_root / "mediamtx_launcher.py"
        if not script.is_file():
            raise FileNotFoundError(f"SONIC MediaMTX launcher is missing: {script}")
        return (sys.executable, str(script))
    if mode == "pico-manager":
        script = Path(__file__).resolve().parent / "auto_pose_manager.py"
        if not script.is_file():
            raise FileNotFoundError(f"upper-body PICO launcher is missing: {script}")
        return (sys.executable, str(script))
    if mode == "camera":
        binary = sonic_root / "bin" / _platform_tag() / "head_camera_rtsp_node"
        return (str(binary),)
    raise ValueError(f"unknown shared runtime mode: {mode}")


def main() -> int:
    if len(sys.argv) < 2:
        print("expected one of: mediamtx, pico-manager, camera", file=sys.stderr)
        return getattr(os, "EX_USAGE", 64)
    mode = sys.argv[1]
    try:
        sonic_root = _sonic_root()
        command = _resolve_command(mode, sonic_root)
        executable = Path(command[0]).resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"shared runtime executable is unavailable: {executable}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[upper-body-teleop] configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR

    # SONIC's selectors use this root for packaged Python, SDK and native assets.
    # This wrapper itself belongs to the separate upper-body teleop Mod.
    os.environ["BXI_MOD_ROOT"] = str(sonic_root)
    argv = (*command, *sys.argv[2:])
    os.execv(str(executable), argv)
    return getattr(os, "EX_SOFTWARE", 70)


if __name__ == "__main__":
    raise SystemExit(main())
