"""every exclusive suite of ``.testament.toml`` runs ``loop_timing`` tests and only those, and every shared suite none.

The loop-stall guard (``threetears.core.testing.loop_stall``) knows a test runs alone on its runner by its
``loop_timing`` marker: such a test is held to exactly 0.25 s of loop-thread CPU a turn, a test of a shared suite to
0.25 s x the host's slowdown. A suite that ran ``loop_timing`` tests beside other batches would hold them to the
reference bound on a busy host (reds the code did not earn), and an exclusive suite that ran unmarked tests would scale
their bound on a quiet host (a loosened bound hiding work).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_MANIFEST = Path(__file__).resolve().parents[2] / ".testament.toml"


def _selects_loop_timing(command: list[str]) -> bool | None:
    """whether the command's ``-m`` expression keeps ``loop_timing`` tests (True), drops them (False) or says nothing
    of them (None)."""
    expression = command[command.index("-m") + 1] if "-m" in command else ""
    terms = [term.strip() for term in expression.split(" and ")]
    if "loop_timing" in terms:
        return True
    if "not loop_timing" in terms:
        return False
    return None


def test_exclusive_suites_select_loop_timing_and_shared_suites_deselect_it() -> None:
    """Flips if an exclusive suite's ``-m`` does not require ``loop_timing``, or a shared suite's does not exclude it."""
    suites = tomllib.loads(_MANIFEST.read_text())["suites"]
    wrong = [
        f"{name}: exclusive={bool(suite.get('exclusive'))}, -m keeps loop_timing: {_selects_loop_timing(suite['command'])}"
        for name, suite in suites.items()
        if _selects_loop_timing(suite["command"]) is not bool(suite.get("exclusive"))
    ]
    assert not wrong, "suites whose isolation and loop_timing selection disagree:\n  " + "\n  ".join(wrong)
