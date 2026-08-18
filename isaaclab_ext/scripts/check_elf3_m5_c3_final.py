#!/usr/bin/env python3
"""Check ELF3 C3 final source, behavior, and export evidence on CPU."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT / "isaaclab_ext/source/openhomie_isaaclab"
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from openhomie_isaaclab.workflows import elf3_c3_final  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    try:
        request = elf3_c3_final.parse_request(argv)
        aggregate = elf3_c3_final.write_or_verify_aggregate(request)
        summary = {
            "status": aggregate["status"],
            "contract_passed": aggregate["contract"]["passed"],
            "final_acceptance_passed": aggregate["acceptance"]["overall"]["passed"],
            "checkpoint": aggregate["acceptance"]["checkpoint"],
            "scenario_runs": aggregate["acceptance"]["behavior"]["scenario_runs"],
            "aggregate": str(request.aggregate),
        }
        print("C3_FINAL_CONTRACT_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
        print("C3_FINAL_CONTRACT_PASS", flush=True)
        return 0
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "failure_code": "C3_FINAL_CONTRACT_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print("C3_FINAL_CONTRACT_JSON=" + json.dumps(failure, sort_keys=True), flush=True)
        traceback.print_exc(file=sys.stderr)
        print("C3_FINAL_CONTRACT_FAIL", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
