"""The eval report kit as the tests reach it: where its documentation copy lives, how its scripts run, whether
headless Chromium can launch for its render checks, what a page looks like once Chromium has drawn it, and a report
page holding the blocks a test hands it.

The kit's scripts run under the test's own interpreter in isolated mode (``-I``), so nothing in the working
directory shadows a module they import. The render check drives headless Chromium through playwright, which not
every runner has; a test that needs it skips with the launch error, so a skip there names a missing browser.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from threetears.evals.analysis import report_html
from threetears.evals.analysis.report import Report, ReportBlock
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle

__all__ = ["KIT", "chromium_unavailable", "render_checked", "rendered", "report_of", "report_page", "run_script"]

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


#: Loads a page in headless Chromium at 1200 px, waits for its script to draw and its fonts to load, runs a probe
#: (a JavaScript function's source, from stdin) and prints its result with every console error and page error.
_PROBE_RUNNER = """
import json, pathlib, sys
from playwright.sync_api import sync_playwright
page_path, probe = pathlib.Path(sys.argv[1]), sys.stdin.read()
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(page_path.resolve().as_uri())
    page.wait_for_timeout(600)
    result = page.evaluate(probe)
    browser.close()
print(json.dumps({"result": result, "errors": errors}))
"""


def rendered(page: Path, probe: str) -> tuple[Any, list[str]]:
    """``page`` as headless Chromium draws it: what the JavaScript function ``probe`` returns once the page's script
    has run, and every console error and page error it logged."""
    run = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE_RUNNER, str(page)],
        input=probe,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    return out["result"], out["errors"]


def report_of(blocks: Sequence[ReportBlock] | Sequence[dict[str, Any]]) -> Report:
    """A report holding ``blocks`` in its memo under one finding, over the toy host's bundle."""
    fields: dict[str, Any] = {
        "basis": "analysis",
        "title": "Charts",
        "finding_count": 1,
        "source": {
            "analysis_id": "a",
            "campaign_id": "c",
            "scope_id": "s",
            "behavior": "b",
            "generated_at": "2026-10-05T00:00:00+00:00",
            "generator_model": "m",
            "bundle_fingerprint": "f",
        },
        "blocks": [
            {"kind": "text", "section": "memo", "role": "finding_title", "finding": 0, "body": "Charts"},
            *blocks,
        ],
        "bundle": toyhost_bundle(),
    }
    return Report.model_validate(fields)


def report_page(blocks: Sequence[ReportBlock] | Sequence[dict[str, Any]]) -> str:
    """The page of a report holding ``blocks`` in its memo under one finding (:func:`report_of`)."""
    return report_html(report_of(blocks))
