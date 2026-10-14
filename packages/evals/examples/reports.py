"""How do I turn a finished campaign into the files people read: verdicts, report files and charts?

The earlier examples printed a campaign's report; here it is written out. The verdicts are read as typed data
and gated as a CI step gates them, and the report goes to Markdown and HTML beside the evidence bundle its numbers
come from. New here: ``report_markdown``, ``report_html``, ``inspect_campaign_bundle`` and ``Comparison.gate``.
Reading a report further: ``docs/reading-reports.md``.

Run it with ``python reports.py`` (``README.md`` beside it says how); the files go to ``./eval-report/``. It calls
no model: both arms are keyword stand-ins, so it always runs offline and costs nothing.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from threetears.evals.analysis import inspect_campaign_bundle, report_html, report_markdown
from threetears.evals.quick import compare
from threetears.evals.run import run_blocking

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
# 2. The OFFLINE stand-ins: today's rule and a better one, keyword rules, not models.
# -----------------------------------------------------------------------------

BASELINE_WORDS = ("urgent", "asap")
CANDIDATE_WORDS = ("down", "urgent", "lost", "returns 500", "breach", "nobody")


def offline_triage(words: tuple[str, ...]) -> Callable[[dict], Awaitable[str]]:
    """A keyword stand-in: urgent when the ticket mentions any of ``words``."""

    async def triage(case: dict) -> str:
        return "urgent" if any(word in case["ticket"].lower() for word in words) else "routine"

    return triage


# -----------------------------------------------------------------------------
# 3. Run the campaign, then write the files each reader needs to ``out_dir`` and return the verdicts.
# -----------------------------------------------------------------------------


async def main(out_dir: Path = Path("eval-report")) -> dict[tuple[str, str], str]:
    print("No model is called: running OFFLINE, with keyword stand-ins for both arms.\n")
    comparison = await compare(
        CASES,
        # Name each arm for what it is: the report calls it ``candidate=<name>``.
        {"current": offline_triage(BASELINE_WORDS), "more-keywords": offline_triage(CANDIDATE_WORDS)},
        expected=lambda case: case["label"],
        control="current",
        k=2,
    )
    report = comparison.report
    out_dir.mkdir(parents=True, exist_ok=True)

    # The verdicts as data: each contrast's typed outcome, which a program branches on, beside the words printed.
    verdicts = {(row["arm"], row["reading"]): row["outcome"] for row in comparison.contrasts()}
    for row in comparison.contrasts():
        print(f"{row['arm']} on {row['reading']}: {row['outcome']} ({row['verdict']})")
    # The same verdicts as a CI gate: it fails on a regression or a guardrail not shown held, by default.
    print(comparison.gate().render())

    (out_dir / "report.md").write_text(report_markdown(report), encoding="utf-8")  # for a pull request or an agent
    (out_dir / "report.html").write_text(report_html(report), encoding="utf-8")  # for a person; no script
    # The evidence every number was computed from; the report cites its fingerprint.
    bundle = await run_blocking(  # a store read, so it runs off the event loop
        comparison.host.blocking_executor,
        inspect_campaign_bundle,
        comparison.host,
        comparison.campaign_id,
        comparison.scope_id,
    )
    (out_dir / "bundle.json").write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    print(f"\nWrote report.md, report.html and bundle.json to {out_dir.resolve()}")
    return verdicts


if __name__ == "__main__":
    asyncio.run(main())
