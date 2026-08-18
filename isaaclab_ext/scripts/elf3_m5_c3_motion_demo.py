#!/usr/bin/env python3
"""Record the final ELF3 V3 scripted height-and-motion demonstration."""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Sequence

from openhomie_isaaclab.workflows import elf3_c3_motion_demo


def main(argv: Sequence[str] | None = None) -> int:
    request = elf3_c3_motion_demo.parse_request(argv)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=True,
        enable_cameras=True,
        device=request.device,
    )
    simulation_app = app_launcher.app
    exit_code = 1
    result: dict[str, Any] = {
        "status": "FAIL",
        "failure_code": "C3_MOTION_DEMO_UNHANDLED_EXCEPTION",
    }
    try:
        result = elf3_c3_motion_demo.run(request)
        exit_code = 0 if result.get("status") == "PASS" else 1
    except Exception as exc:
        traceback.print_exc()
        result = {
            "status": "FAIL",
            "failure_code": "C3_MOTION_DEMO_UNHANDLED_EXCEPTION",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        try:
            sentinel = "M5_C3_MOTION_DEMO_PASS" if exit_code == 0 else "M5_C3_MOTION_DEMO_FAIL"
            print(sentinel, flush=True)
            print(f"M5_C3_MOTION_DEMO_INTERNAL_EXIT_CODE={exit_code}", flush=True)
            print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        finally:
            simulation_app.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
