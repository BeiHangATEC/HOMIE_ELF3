"""Test configuration: make the extension importable without installing it."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_SOURCE = REPO_ROOT / "isaaclab_ext" / "source" / "openhomie_isaaclab"

if str(EXTENSION_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXTENSION_SOURCE))
