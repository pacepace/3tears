"""Turn a finished campaign into the files people read: verdicts as data, the report as files, and its charts.

``compare_two_prompts.py`` ends by printing the campaign's report. This picks up there, from a finished
campaign, and writes what each reader needs:

- **The verdicts, as data.** A ``Report`` is a list of typed blocks, so a script reads the "Contrasts
  against the control" table directly instead of scraping Markdown.
- **The report, as files.** ``report.md`` for a pull request or an agent, ``report.html`` (no script) for a
  person, and ``bundle.json``, the evidence every number was computed from: its fingerprint is the hash the
  report cites (the last line of ``report.md``), so anyone can check which evidence a report came from.
- **The charts.** Each chart block's Vega-Lite spec as ``charts/<name>.vl.json`` (open one in
  https://vega.github.io/editor), and, when the ``[vega]`` extra is installed
  (``pip install "3tears-evals[vega]"``), an SVG of each and ``charts.html`` showing them all.

The campaign is two OFFLINE STAND-INS (keyword rules, not models) triaging 12 on-call tickets, twice each:
the point here is the files, not the eval. Run ``python packages/evals/examples/reports.py [DIR]``; every
file goes to ``DIR`` (default ``./eval-report/``), printed at the end.

Next step: an **analysis**, a model reading ``bundle.json`` and writing findings on top of this report, every
number in them still the bundle's. See ``docs/reading-reports.md`` for how the report changes, and
"Analysis generation" in ``docs/cost-and-budgets.md`` for how its cost is estimated and capped.
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from threetears.evals.analysis import ChartBlock, TableBlock, inspect_campaign_bundle, report_html, report_markdown
from threetears.evals.quick import compare
from threetears.evals.vega import VegaRenderer

# -----------------------------------------------------------------------------
# 0. A finished campaign: two keyword stand-ins over the same tickets.
# -----------------------------------------------------------------------------

CASES = [
    {"ticket": "Checkout is down for every customer since 09:00.", "label": "urgent"},
    {"ticket": "URGENT: invoices are going to the wrong customers.", "label": "urgent"},
    {"ticket": "We lost all of yesterday's orders after the migration.", "label": "urgent"},
    {"ticket": "Please fix asap, the API returns 500 on every request.", "label": "urgent"},
    {"ticket": "Possible data breach: our admin page is public.", "label": "urgent"},
    {"ticket": "Nobody on the team can sign in since the update.", "label": "urgent"},
    {"ticket": "Could you add a dark mode to the dashboard?", "label": "routine"},
    {"ticket": "How do I export last month's report to CSV?", "label": "routine"},
    {"ticket": "Please update the billing address on our account asap.", "label": "routine"},
    {"ticket": "The tooltip on the pricing page has a typo.", "label": "routine"},
    {"ticket": "Can we get a copy of the March invoice?", "label": "routine"},
    {"ticket": "Is there an API rate limit we should know about?", "label": "routine"},
]


async def baseline(case: dict) -> str:
    """Stand-in for today's prompt: urgent only when the customer says so."""
    return "urgent" if any(word in case["ticket"].lower() for word in ("urgent", "asap")) else "routine"


async def candidate(case: dict) -> str:
    """Stand-in for the new prompt: urgent when the ticket describes an outage, data loss or a breach."""
    words = ("down", "urgent", "lost", "returns 500", "breach", "nobody")
    return "urgent" if any(word in case["ticket"].lower() for word in words) else "routine"


async def main(out: Path = Path("eval-report")) -> dict[tuple[str, str], str]:
    """Run the campaign, write its files to ``out``, and return each contrast's verdict by (arm, reading)."""
    comparison = await compare(
        CASES,
        {"baseline": baseline, "candidate": candidate},
        expected=lambda case: case["label"],
        control="baseline",
        scope_id="reports-example",
        k=2,
        name="on-call triage (offline stand-ins)",
    )
    report = comparison.report  # no analysis was generated, so this is the code-only report
    out.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. The verdicts, as data: the contrasts table is a TableBlock named "comparisons",
    #    and each of its rows is a dict keyed by column.
    # -------------------------------------------------------------------------
    table = next(block for block in report.blocks if isinstance(block, TableBlock) and block.name == "comparisons")
    verdicts = {(str(row["contrast"]), str(row["reading"])): str(row["verdict"]) for row in table.rows}
    for (arm, reading), verdict in verdicts.items():
        print(f"{arm} on {reading}: {verdict}")

    # -------------------------------------------------------------------------
    # 2. The report as files, and the evidence bundle it was computed from
    #    (the same JSON the command line's ``bundle`` prints).
    # -------------------------------------------------------------------------
    (out / "report.md").write_text(report_markdown(report), encoding="utf-8")
    (out / "report.html").write_text(report_html(report), encoding="utf-8")
    bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, comparison.scope_id)
    (out / "bundle.json").write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # 3. The charts. A chart block carries an intent (what to draw), not a picture; the Vega-Lite
    #    renderer compiles each to a spec, here in the packaged light palette (a host with its own
    #    passes its style: ``VegaRenderer.for_style(host.profile.style)``).
    # -------------------------------------------------------------------------
    renderer = VegaRenderer.packaged("light")
    charts = out / "charts"
    charts.mkdir(exist_ok=True)
    svgs = []
    for index, block in enumerate(block for block in report.blocks if isinstance(block, ChartBlock) and block.intent):
        name = f"{index:02d}-" + re.sub(r"[^a-z0-9]+", "-", block.intent.title.lower()).strip("-")
        chart = renderer.draw(block.intent)
        spec = chart.spec | {"config": renderer.config()}  # the spec is colourless; the config colours it
        (charts / f"{name}.vl.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        try:
            (charts / f"{name}.svg").write_text(renderer.svg(chart), encoding="utf-8")
            svgs.append(f"{name}.svg")
        except ImportError:  # rasterising needs the [vega] extra; the spec does not
            pass
    if svgs:
        page = "".join(f'<p><img src="charts/{svg}" alt="{svg}"></p>' for svg in svgs)
        (out / "charts.html").write_text(f"<!doctype html><title>Charts</title>{page}", encoding="utf-8")
    else:
        print('No SVGs: install the [vega] extra (pip install "3tears-evals[vega]") to draw them.')

    print(f"\nWrote the report, its evidence and {len(list(charts.glob('*.vl.json')))} chart specs to {out.resolve()}")
    return verdicts


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval-report")))
