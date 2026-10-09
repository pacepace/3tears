"""How do I turn a finished campaign into the files people read: verdicts, report files and charts?

``compare_two_prompts.py`` printed the campaign's report. Here the same kind of campaign is written out
instead: the verdicts read off the ``Report`` as data, the report as Markdown and HTML beside the evidence
bundle its numbers come from, and each chart as a Vega-Lite spec (plus an SVG with the ``[vega]`` extra).
Reading a report further: ``docs/reading-reports.md``.

Run it with ``python packages/evals/examples/reports.py``; the files go to ``./eval-report/``.
It calls no model: both arms are keyword stand-ins, so it always runs offline and costs nothing.
"""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from threetears.evals.analysis import ChartBlock, TableBlock, inspect_campaign_bundle, report_html, report_markdown
from threetears.evals.quick import compare
from threetears.evals.vega import VegaRenderer

# -----------------------------------------------------------------------------
# 1. The cases: on-call tickets, and whether each one is urgent.
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

# -----------------------------------------------------------------------------
# 2. The offline stand-ins: today's rule and a better one. Nothing here is a model.
# -----------------------------------------------------------------------------

BASELINE_WORDS = ("urgent", "asap")
CANDIDATE_WORDS = ("down", "urgent", "lost", "returns 500", "breach", "nobody")


def offline_triage(words: tuple[str, ...]) -> Callable[[dict], Awaitable[str]]:
    """A keyword stand-in: urgent when the ticket mentions any of ``words``."""

    async def triage(case: dict) -> str:
        return "urgent" if any(word in case["ticket"].lower() for word in words) else "routine"

    return triage


# -----------------------------------------------------------------------------
# 3. Run the campaign, then write the files each reader needs.
# -----------------------------------------------------------------------------


async def main(out: Path = Path("eval-report")) -> dict[tuple[str, str], str]:
    """Compare the two rules, write the report, its evidence and its charts to ``out``, and return the verdicts."""
    comparison = await compare(
        CASES,
        {"baseline": offline_triage(BASELINE_WORDS), "candidate": offline_triage(CANDIDATE_WORDS)},
        expected=lambda case: case["label"],
        control="baseline",
        scope_id="reports-example",
        k=2,
    )
    report = comparison.report
    out.mkdir(parents=True, exist_ok=True)

    # The verdicts as data: the contrasts are a table block, each row a dict keyed by column.
    table = next(block for block in report.blocks if isinstance(block, TableBlock) and block.name == "comparisons")
    verdicts = {(str(row["contrast"]), str(row["reading"])): str(row["verdict"]) for row in table.rows}
    for (arm, reading), verdict in verdicts.items():
        print(f"{arm} on {reading}: {verdict}")

    (out / "report.md").write_text(report_markdown(report), encoding="utf-8")  # for a pull request or an agent
    (out / "report.html").write_text(report_html(report), encoding="utf-8")  # for a person; no script
    # The evidence every number was computed from; the report cites its fingerprint.
    bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, comparison.scope_id)
    (out / "bundle.json").write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    # A chart block carries an intent, not a picture; the Vega-Lite renderer compiles it to a spec.
    renderer = VegaRenderer.packaged("light")
    (out / "charts").mkdir(exist_ok=True)
    svgs = []
    for index, block in enumerate(block for block in report.blocks if isinstance(block, ChartBlock) and block.intent):
        name = f"{index:02d}-" + re.sub(r"[^a-z0-9]+", "-", block.intent.title.lower()).strip("-")
        chart = renderer.draw(block.intent)
        spec = chart.spec | {"config": renderer.config()}  # the config colours the spec
        (out / "charts" / f"{name}.vl.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        try:
            (out / "charts" / f"{name}.svg").write_text(renderer.svg(chart), encoding="utf-8")
            svgs.append(f"charts/{name}.svg")
        except ImportError:  # drawing an SVG needs the [vega] extra; the spec does not
            pass
    if svgs:
        page = "".join(f'<p><img src="{svg}" alt="{svg}"></p>' for svg in svgs)
        (out / "charts.html").write_text(f"<!doctype html><title>Charts</title>{page}", encoding="utf-8")
    else:
        print('No SVGs: install the [vega] extra (pip install "3tears-evals[vega]") to draw them.')

    print(f"\nWrote report.md, report.html, bundle.json and charts/ to {out.resolve()}")
    return verdicts


if __name__ == "__main__":
    asyncio.run(main())
