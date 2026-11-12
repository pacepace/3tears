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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from threetears.evals.analysis import report_html
from threetears.evals.analysis.report import Report, ReportBlock
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle

__all__ = [
    "KIT",
    "MARKED_CHARTS",
    "Marked",
    "chromium_unavailable",
    "render_checked",
    "rendered",
    "report_of",
    "report_page",
    "run_script",
]

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
    """``scripts/check.py`` run on ``page``, its screenshots in ``shots``: the run, and its line for each of the eight
    renders (1200, 1104, 688 and 390 px, light and dark)."""
    checked = run_script("check.py", page, Path("--out"), shots)
    runs = [line for line in checked.stdout.splitlines() if "px" in line and "scrollWidth=" in line]
    return checked, runs


#: Loads a page in headless Chromium at a width (1200 px unless given), waits for its script to draw and its fonts to load, runs a probe
#: (a JavaScript function's source, from stdin) and prints its result with every console error and page error.
_PROBE_RUNNER = """
import json, pathlib, sys
from playwright.sync_api import sync_playwright
page_path, width, probe = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.stdin.read()
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": width, "height": 900})
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(page_path.resolve().as_uri())
    page.wait_for_timeout(600)
    result = page.evaluate(probe)
    browser.close()
print(json.dumps({"result": result, "errors": errors}))
"""


def rendered(page: Path, probe: str, *, width: int = 1200) -> tuple[Any, list[str]]:
    """``page`` as headless Chromium draws it at ``width`` px: what the JavaScript function ``probe`` returns once the
    page's script has run, and every console error and page error it logged."""
    run = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE_RUNNER, str(page), str(width)],
        input=probe,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    return out["result"], out["errors"]


def report_of(blocks: Sequence[ReportBlock] | Sequence[dict[str, Any]]) -> Report:
    """A report holding ``blocks``, over the toy host's bundle."""
    fields: dict[str, Any] = {
        "title": "Charts",
        "source": {
            "campaign_id": "c",
            "scope_id": "s",
            "behavior": "b",
            "generated_at": "2026-10-05T00:00:00+00:00",
            "bundle_fingerprint": "f",
        },
        "blocks": list(blocks),
        "bundle": toyhost_bundle(),
    }
    return Report.model_validate(fields)


def report_page(blocks: Sequence[ReportBlock] | Sequence[dict[str, Any]]) -> str:
    """The page of a report holding ``blocks`` (:func:`report_of`)."""
    return report_html(report_of(blocks))


@dataclass(frozen=True)
class Marked:
    """A chart whose every drawn value carries a marker text of its own, beside where each marker must be drawn.

    :ivar kind: The chart's kind.
    :ivar data: Its ``data-chart`` JSON.
    :ivar tips: Each mark's tooltip, as its lines, in drawn order: a value's, then its interval's ends, each in its
        place, so a low end's text drawn as the high end's reads as another tooltip.
    :ivar texts: Each marker drawn as text of its own (a note, an end label, a tile), in drawn order.
    """

    kind: str
    data: dict[str, Any]
    tips: tuple[tuple[str, ...], ...]
    texts: tuple[str, ...] = ()


#: One marked chart per kind the static page draws (and the kit draws), every value's text a marker no format spells:
#: ``test_report_kit_contract`` holds the kinds here to the static page's own registry of drawers
#: (:data:`~threetears.evals.analysis.report.chart_svg.DRAWERS`), whatever kinds the fixtures' pages draw.
MARKED_CHARTS: dict[str, Marked] = {
    "bars": Marked(
        "bars",
        {
            "rows": [
                {"label": "a", "value": 2, "lo": 1, "hi": 3, "value_text": "BV", "lo_text": "BL", "hi_text": "BH"},
                {"label": "b", "value": 1, "lo": 0.5, "value_text": "OV", "lo_text": "OL"},
            ],
            "domain": [0, 4],
            "fmt": "num",
            "level": "95%",
        },
        (("a", "BV (BL to BH)"), ("b", "OV (from OL up, no upper bound)")),
        ("BV", "OV"),
    ),
    "bars as dots": Marked(
        "bars",
        {
            "rows": [
                {"label": "a", "value": 2, "lo": 1, "hi": 3, "value_text": "DV", "lo_text": "DL", "hi_text": "DH"}
            ],
            "domain": [0, 4],
            "fmt": "num",
            "level": "95%",
            "dots": True,
        },
        (("a", "DV (DL to DH)"),),
    ),
    "diverging_bars": Marked(
        "diverging_bars",
        {
            "rows": [
                {"label": "a", "value": -1, "lo": -2, "hi": 0.5, "value_text": "NV", "lo_text": "NL", "hi_text": "NH"},
                {"label": "b", "value": 1, "lo": 0.5, "hi": 2, "value_text": "PV", "lo_text": "PL", "hi_text": "PH"},
            ],
            "domain": [-3, 3],
            "fmt": "num",
            "level": "95%",
        },
        (("a", "NV (NL to NH)"), ("b", "PV (PL to PH)")),
        ("NV", "PV"),
    ),
    "forest": Marked(
        "forest",
        {
            "rows": [
                {"label": "a", "value": -1, "lo": -2, "hi": 0.5, "value_text": "FV", "lo_text": "FL", "hi_text": "FH"}
            ],
            "domain": [-3, 3],
            "fmt": "num",
            "level": "95%",
        },
        (("a", "FV (FL to FH)"),),
        ("FV",),
    ),
    "lines": Marked(
        "lines",
        {
            "categories": ["c1", "c2"],
            "series": [
                {
                    "name": "s",
                    "values": [1, 2],
                    "lo": [0.5, 1.5],
                    "hi": [1.5, 2.5],
                    "values_text": ["LV1", "LV2"],
                    "lo_text": ["LL1", "LL2"],
                    "hi_text": ["LH1", "LH2"],
                }
            ],
            "domain": [0, 3],
            "fmt": "num",
            "level": "95%",
        },
        (("s · c1", "LV1 (LL1 to LH1)"), ("s · c2", "LV2 (LL2 to LH2)")),
    ),
    "scatter_frontier": Marked(
        "scatter_frontier",
        {
            "points": [
                {"x": 1, "y": 2, "label": "p", "x_text": "SX1", "y_text": "SY1"},
                {"x": 2, "y": 1, "label": "q", "x_text": "SX2", "y_text": "SY2"},
            ],
            "x_domain": [0, 3],
            "y_domain": [0, 3],
            "x_fmt": "num",
            "y_fmt": "num",
        },
        (("p", "SX1 · SY1"), ("q", "SX2 · SY2")),
    ),
    "gate_scatter": Marked(
        "gate_scatter",
        {
            "points": [{"x": 1, "y": 2, "label": "p", "x_text": "GX", "y_text": "GY"}],
            "x_domain": [0, 3],
            "y_domain": [0, 3],
            "x_fmt": "num",
            "y_fmt": "num",
            "reference": {"value": 1, "kind": "floor"},
            "x_reference": {"value": 2, "kind": "budget"},
        },
        (("p", "GX · GY"),),
    ),
    "dot_interval": Marked(
        "dot_interval",
        {
            "rows": [
                {"label": "a", "value": 50, "lo": 40, "hi": 60, "value_text": "IV", "lo_text": "IL", "hi_text": "IH"}
            ],
            "domain": [0, 100],
            "fmt": "pct",
            "level": "95%",
        },
        (("a", "IV (IL to IH)"),),
    ),
    "range_bars": Marked(
        "range_bars",
        {
            "groups": [
                {"label": "g", "rows": [{"label": "a", "from": 1, "to": 2, "from_text": "RF", "to_text": "RT"}]}
            ],
            "domain": [0, 3],
            "fmt": "s",
        },
        (("a · g", "RF to RT"),),
    ),
    "heatmap": Marked(
        "heatmap",
        {
            "columns": [{"label": "c"}],
            "rows": [{"label": "r", "cells": [{"value": 50, "n": 3, "value_text": "HV"}]}],
            "domain": [0, 100],
            "fmt": "pct",
        },
        (("r · c", "HV · n = 3"),),
        ("HV",),
    ),
    "diff_heatmap": Marked(
        "diff_heatmap",
        {
            "columns": [{"label": "c"}],
            "rows": [{"label": "r", "cells": [{"value": 1, "detail": 2, "value_text": "XV", "detail_text": "XD"}]}],
            "fmt": "num",
            "detail_fmt": "num",
        },
        (("r · c", "XV"),),
        ("XV", "XD"),
    ),
    "confusion": Marked(
        "confusion",
        {"labels": ["x", "y"], "matrix": [[90, 10], [5, 95]], "matrix_text": [["CA", "CB"], ["CC", "CD"]]},
        (
            ("x → x", "CA of the row"),
            ("x → y", "CB of the row"),
            ("y → x", "CC of the row"),
            ("y → y", "CD of the row"),
        ),
        ("CA", "CB", "CC", "CD"),
    ),
}
