#!/usr/bin/env python3
"""Launch fresh ELF3 single-stage G1-proportioned training."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source/openhomie_isaaclab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from openhomie_isaaclab.workflows.elf3_single_stage import (
    parse_request,
    request_log_payload,
)


def main(argv: list[str] | None = None) -> int:
    request = parse_request(argv)
    print(
        json.dumps(request_log_payload(request), sort_keys=True, allow_nan=False),
        flush=True,
    )
    from isaaclab.app import AppLauncher

    application = AppLauncher(headless=request.headless, device=request.device).app
    result: dict[str, object] = {"status": "FAIL", "error": "unhandled exception"}
    exit_code = 1
    try:
        from openhomie_isaaclab.workflows.elf3_single_stage import run_training

        result = run_training(request)
        exit_code = 0 if result.get("status") == "PASS" else 1
    except Exception:
        traceback.print_exc()
    finally:
        print("ELF3_SINGLE_STAGE_PASS" if exit_code == 0 else "ELF3_SINGLE_STAGE_FAIL", flush=True)
        print(f"ELF3_SINGLE_STAGE_EXIT_CODE={exit_code}", flush=True)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        application.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
