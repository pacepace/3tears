"""Reports to test against: the toy host's, generated through the host, and a minimal one that passes.

Shared by ``test_report.py`` and ``test_report_html_no_script.py`` — harness support, not part of the
toy host, which reaches the engine only through its public roots.
"""

from __future__ import annotations

import json
from typing import Any

from threetears.evals.analysis import (
    Report,
    TableBlock,
    assemble_campaign_bundle,
    build_report,
    report_markdown,
)
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.analysis.style import StyleViolation
from packages.evals.tests.style_support import lint_page
from threetears.evals.kernel.campaign import EvalAnalysis, EvalCampaign, EvalInsight
from threetears.evals.kernel.host import EvalHost
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, memo_payload
from packages.evals.tests.off_loop import off_loop

__all__ = [
    "page_violations",
    "stored_memo_analysis",
    "toy_campaign_host",
    "toy_report",
]


def page_violations(report: Report) -> list[StyleViolation]:
    """Every style rule a report's page breaks outside the data it quotes, its arms' names read as names.

    :param report: The report.
    :ptype report: Report
    :return: The violations; empty for a page that passes.
    :rtype: list[StyleViolation]
    """
    names = {
        str(row["arm"]).removeprefix("▸ ")
        for block in report.blocks
        if isinstance(block, TableBlock) and block.name == "scoreboard"
        for row in block.values()
    }
    return lint_page(report_markdown(report), names=names)


def toy_campaign_host(clients: Any = None) -> tuple[EvalHost, EvalCampaign]:
    """The toy host holding its corpus campaign — runs, results and the campaign — and no analysis of it."""
    host = toyhost_host(clients=clients) if clients is not None else toyhost_host()
    campaign, corpus = toyhost_campaign()
    for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
        host.storage.save_eval_run(run)
        for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
            host.storage.save_eval_result(result)
    host.storage.save_campaign(campaign)
    return host, campaign


async def stored_memo_analysis(host: EvalHost, campaign: EvalCampaign) -> tuple[EvalAnalysis, list[EvalInsight]]:
    """An analysis the memo writer wrote over the campaign, stored with its insights: the shape of every analysis
    stored before the report planner wrote new ones, which still lists, reads and renders.

    The memo writer is called directly (``generate_analysis``), since a generation through the host is the planner's.

    :param host: The host holding the campaign's runs and results.
    :param campaign: The campaign.
    :return: The stored analysis and its insights.
    """
    assembled_at = utc_now_iso()
    bundle = off_loop(assemble_campaign_bundle, host, campaign.id, campaign.scope_id)
    client = FixturedClient(json.dumps(memo_payload(bundle)))
    analysis, insights = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=client,
        prompt_id=PROMPT_ID,
        bundle_assembled_at=assembled_at,
        profile=host.profile,
    )

    def store() -> None:
        host.storage.save_analysis(analysis)
        for insight in insights:
            host.storage.save_insight(insight)

    off_loop(store)
    return analysis, insights


async def toy_report() -> tuple[EvalHost, EvalAnalysis, Report]:
    """The toy campaign's memo analysis, stored, and its one aspect's page laid out over it."""
    host, campaign = off_loop(toy_campaign_host)
    analysis, _ = await stored_memo_analysis(host, campaign)
    return host, analysis, build_report(analysis, campaign_name=campaign.name)
