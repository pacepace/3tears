"""The eval report kit as the tests reach it: where its documentation copy lives, how its scripts run, and whether
headless Chromium can launch for its render checks.

The kit's scripts run under the test's own interpreter in isolated mode (``-I``), so nothing in the working
directory shadows a module they import. The render check drives headless Chromium through playwright, which not
every runner has; a test that needs it skips with the launch error, so a skip there names a missing browser.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__all__ = ["KIT", "chromium_unavailable", "render_checked", "run_script"]

#: The kit's documentation copy: its stylesheet and script, the reference memos and the scripts that build and check.
KIT = Path(__file__).resolve().parents[3] / "docs" / "eval-report-kit"


def run_script(script: str, *args: Path) -> subprocess.CompletedProcess[str]:
    """One of the kit's scripts, run on ``args``."""
    return subprocess.run(
        [sys.executable, "-I", str(KIT / "scripts" / script), *(str(a) for a in args)],
        capture_output=True,
        text=True,
        check=False,
    )


def chromium_unavailable() -> str | None:
    """Why headless Chromium cannot launch here, or ``None`` when it can."""
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from playwright.sync_api import sync_playwright\n"
            "with sync_playwright() as p:\n"
            "    p.chromium.launch().close()\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        return None
    # playwright boxes its install advice after the error itself; the error line names the cause
    errors = [line.strip() for line in probe.stderr.splitlines() if "Error" in line]
    return errors[-1] if errors else f"exit code {probe.returncode}: {probe.stderr.strip()[-300:]}"


def render_checked(page: Path, shots: Path) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """``scripts/check.py`` run on ``page``, its screenshots in ``shots``: the run, and its line for each of the four
    renders (1200 and 390 px, light and dark)."""
    checked = run_script("check.py", page, Path("--out"), shots)
    runs = [line for line in checked.stdout.splitlines() if "px" in line and "scrollWidth=" in line]
    return checked, runs
