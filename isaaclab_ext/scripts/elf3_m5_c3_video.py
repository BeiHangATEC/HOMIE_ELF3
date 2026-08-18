#!/usr/bin/env python3
"""Record the fixed ELF3 M5 C3 acceptance clip through Isaac Lab."""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from openhomie_isaaclab.workflows import elf3_c3_video


def main(argv: list[str] | None = None) -> int:
    request = elf3_c3_video.parse_request(argv)
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
        "failure_code": "C3_VIDEO_UNHANDLED_EXCEPTION",
    }
    serialized_result = json.dumps(result, sort_keys=True, allow_nan=False)
    try:
        result = elf3_c3_video.run(request)
        if not isinstance(result, dict):
            raise TypeError("C3 video workflow must return a mapping")
        serialized_result = json.dumps(result, sort_keys=True, allow_nan=False)
        exit_code = 0 if result.get("status") == "PASS" else 1
    except Exception:
        traceback.print_exc()
        result = {
            "status": "FAIL",
            "failure_code": "C3_VIDEO_UNHANDLED_EXCEPTION",
        }
        serialized_result = json.dumps(result, sort_keys=True, allow_nan=False)
        exit_code = 1
    finally:
        try:
            sentinel = (
                "M5_C3_VIDEO_PASS" if exit_code == 0 else "M5_C3_VIDEO_FAIL"
            )
            print(sentinel, flush=True)
            print(f"M5_C3_VIDEO_INTERNAL_EXIT_CODE={exit_code}", flush=True)
            print(serialized_result, flush=True)
        except Exception:
            exit_code = 1
            traceback.print_exc()
        finally:
            simulation_app.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
