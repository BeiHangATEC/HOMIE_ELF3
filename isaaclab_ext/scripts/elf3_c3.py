#!/usr/bin/env python3
"""Train one fail-closed ELF3 C3 velocity stage."""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from openhomie_isaaclab.workflows.elf3_c3 import parse_request


def main(argv: list[str] | None = None) -> int:
    request = parse_request(argv)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=request.headless, device=request.device
    )
    simulation_app = app_launcher.app
    exit_code = 1
    result: dict[str, Any] = {
        "status": "FAIL",
        "failure_code": "UNHANDLED_EXCEPTION",
    }
    try:
        import gymnasium as gym  # noqa: F401
        import isaaclab_rl  # noqa: F401
        import torch  # noqa: F401
        from openhomie_isaaclab.workflows import elf3_c3

        result = elf3_c3.run(request)
        if not isinstance(result, dict):
            raise TypeError("elf3_c3.run() must return a dictionary")
        exit_code = 0 if result.get("status") == "PASS" else 1
    except Exception:
        traceback.print_exc()
        result = {
            "status": "FAIL",
            "failure_code": "UNHANDLED_EXCEPTION",
        }
        exit_code = 1
    finally:
        try:
            sentinel = "C3_PASS" if exit_code == 0 else "C3_FAIL"
            print(sentinel, flush=True)
            print(f"C3_INTERNAL_EXIT_CODE={exit_code}", flush=True)
            print(
                json.dumps(result, sort_keys=True, allow_nan=False),
                flush=True,
            )
        except Exception:
            exit_code = 1
            traceback.print_exc()
            try:
                print("C3_FAIL", flush=True)
                print("C3_INTERNAL_EXIT_CODE=1", flush=True)
            except Exception:
                traceback.print_exc()
        finally:
            simulation_app.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
