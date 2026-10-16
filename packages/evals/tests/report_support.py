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
    analysis_report,
    prepare_analysis_generation,
    report_markdown,
    run_analysis_generation,
)
from threetears.evals.analysis.style import StyleViolation, lint_page
from threetears.evals.kernel.campaign import EvalAnalysis, EvalCampaign
from threetears.evals.kernel.host import EvalHost
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.toyhost_memo import PROMPT, PROMPT_ID, FixturedClient, memo_payload
from packages.evals.tests.off_loop import off_loop

__all__ = [
    "page_violations",
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


async def toy_report() -> tuple[EvalHost, EvalAnalysis, Report]:
    """The toy campaign generated through the host and read back as its report."""
    built: list[FixturedClient] = []

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> FixturedClient:
        client = FixturedClient("")
        built.append(client)
        return client

    host, campaign = off_loop(toy_campaign_host, clients)

    async def prompt() -> str:
        return PROMPT

    prepared = await prepare_analysis_generation(
        host, campaign.id, campaign.scope_id, model=None, resolve_prompt=prompt, out_of_run_cap_usd=None
    )
    built[0].completion.content = json.dumps(memo_payload(prepared.bundle))
    analysis, _ = await run_analysis_generation(host, prepared, prompt_id=PROMPT_ID, max_output_tokens=4000)
    return host, analysis, off_loop(analysis_report, host.storage, analysis.id, analysis.scope_id)
