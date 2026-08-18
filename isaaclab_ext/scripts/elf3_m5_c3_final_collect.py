#!/usr/bin/env python3
"""Collect final ELF3 C3 V3 numeric acceptance evidence in Isaac Lab."""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Sequence

from openhomie_isaaclab.workflows import elf3_c3_final_collect


def main(argv: Sequence[str] | None = None) -> int:
    request = elf3_c3_final_collect.parse_request(argv)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, device=request.device)
    simulation_app = app_launcher.app
    exit_code = 1
    result: dict[str, Any] = {
        "status": "FAIL",
        "failure_code": "C3_FINAL_COLLECTION_UNHANDLED_EXCEPTION",
    }
    try:
        result = elf3_c3_final_collect.run(request)
        if not isinstance(result, dict):
            raise TypeError("C3 final collector must return a mapping")
        exit_code = 0 if result.get("status") == "PASS" else 1
    except Exception as exc:
        traceback.print_exc()
        result = {
            "status": "FAIL",
            "failure_code": "C3_FINAL_COLLECTION_UNHANDLED_EXCEPTION",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        exit_code = 1
    finally:
        try:
            sentinel = (
                "M5_C3_FINAL_COLLECTION_PASS"
                if exit_code == 0
                else "M5_C3_FINAL_COLLECTION_FAIL"
            )
            print(sentinel, flush=True)
            print(f"M5_C3_FINAL_COLLECTION_EXIT_CODE={exit_code}", flush=True)
            print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        except Exception:
            exit_code = 1
            traceback.print_exc()
        finally:
            simulation_app.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
