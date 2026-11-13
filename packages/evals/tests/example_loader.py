"""The ladder's rung directories under ``examples/``."""

from __future__ import annotations

from pathlib import Path

#: The examples directory.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

#: The ladder: every example under ``examples/`` lives in one of these directories, a rung each.
RUNGS = ("1-triage", "2-replies", "3-invoices", "4-booking", "5-writer")
