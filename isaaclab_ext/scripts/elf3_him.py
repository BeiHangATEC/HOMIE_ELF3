#!/usr/bin/env python3
"""Train, play, or export the ELF3 HIM policy through Isaac Lab."""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from openhomie_isaaclab.workflows.elf3_run import parse_request

COMMANDS = ("train", "play", "export")


def main(argv: list[str] | None = None) -> int:
    request = parse_request(argv)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=request.headless, device=request.device)
    simulation_app = app_launcher.app
    exit_code = 1
    result: dict[str, Any] = {
        "status": "FAIL",
        "failure_code": "UNHANDLED_EXCEPTION",
    }
    serialized_result = json.dumps(result, sort_keys=True, allow_nan=False)
    try:
        import gymnasium as gym  # noqa: F401
        import isaaclab_rl  # noqa: F401
        import torch  # noqa: F401
        from openhomie_isaaclab.workflows import elf3_sim

        result = elf3_sim.run(request)
        if not isinstance(result, dict):
            raise TypeError("elf3_sim.run() must return a mapping")
        serialized_result = json.dumps(
            result, sort_keys=True, allow_nan=False
        )
        exit_code = 0 if result.get("status") == "PASS" else 1
    except Exception:
        traceback.print_exc()
        exit_code = 1
        result = {
            "status": "FAIL",
            "failure_code": "UNHANDLED_EXCEPTION",
        }
        serialized_result = json.dumps(
            result, sort_keys=True, allow_nan=False
        )
    finally:
        try:
            sentinel = (
                "M5_HEADLESS_PASS" if exit_code == 0 else "M5_HEADLESS_FAIL"
            )
            print(sentinel, flush=True)
            print(f"M5_INTERNAL_EXIT_CODE={exit_code}", flush=True)
            print(serialized_result, flush=True)
        except Exception:
            exit_code = 1
            traceback.print_exc()
            try:
                print("M5_HEADLESS_FAIL", flush=True)
                print("M5_INTERNAL_EXIT_CODE=1", flush=True)
            except Exception:
                traceback.print_exc()
        finally:
            simulation_app.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
