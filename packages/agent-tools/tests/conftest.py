"""Agent-tools test configuration — adds tests dir to sys.path for imports."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow test modules to import from testing_utils.py
sys.path.insert(0, str(Path(__file__).parent))
