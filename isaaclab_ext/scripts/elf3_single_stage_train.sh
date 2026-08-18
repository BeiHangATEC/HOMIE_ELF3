#!/usr/bin/env bash
# Run fresh ELF3 single-stage training with live child-process output.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
exec conda run --no-capture-output -n homie python \
  "$SCRIPT_DIR/elf3_single_stage_train.py" "$@"
