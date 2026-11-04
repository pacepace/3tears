"""Reports to test against: the toy host's, and the analyses its retired writers stored over its corpus campaign.

Shared by ``test_report.py`` and ``test_report_html_no_script.py`` — harness support, not part of the
toy host, which reaches the engine only through its public roots.

**The retired writers' analyses are stored records, never generated.** The memo writer and the report planner wrote
the two records under ``fixtures/retired_writers/`` over the toy corpus campaign (at ``52bf24449``): each holds the
analysis as its writer stored it, with its insights, and the planner's its report plan. A host holds analyses like
these, which every read keeps serving, so a test stores the record as it was written and reads it back; nothing here
calls either writer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from threetears.evals.analysis import Report, TableBlock, build_report, report_markdown
from threetears.evals.analysis.style import StyleViolation
from packages.evals.tests.style_support import lint_page
from threetears.evals.kernel.campaign import EvalAnalysis, EvalCampaign, EvalInsight, EvalReportPlan
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.storage import EvalStorage
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.off_loop import off_loop

__all__ = [
    "page_violations",
    "stored_memo_analysis",
    "stored_plan_analysis",
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


#: The records the retired writers stored over the toy corpus campaign.
_RETIRED: Final = Path(__file__).parent / "fixtures" / "retired_writers"


def _record(name: str, campaign: EvalCampaign) -> dict[str, Any]:
    """One retired writer's stored record, refused over any campaign but the toy corpus campaign it was written over."""
    record: dict[str, Any] = json.loads((_RETIRED / f"{name}_analysis.json").read_text(encoding="utf-8"))
    written_over = (record["analysis"]["campaign_id"], record["analysis"]["scope_id"])
    if written_over != (campaign.id, campaign.scope_id):
        raise AssertionError(f"the {name} record was written over campaign {written_over}, not {campaign.id!r}")
    return record


def _stored(storage: EvalStorage, record: dict[str, Any]) -> tuple[EvalAnalysis, list[EvalInsight]]:
    """Store a record's analysis and insights as its writer stored them, and return them."""
    analysis = EvalAnalysis.model_validate(record["analysis"])
    insights = [EvalInsight.model_validate(insight) for insight in record["insights"]]
    storage.save_analysis(analysis)
    for insight in insights:
        storage.save_insight(insight)
    return analysis, insights


async def stored_memo_analysis(host: EvalHost, campaign: EvalCampaign) -> tuple[EvalAnalysis, list[EvalInsight]]:
    """The analysis the memo writer stored over the toy corpus campaign, stored again with its insights: the shape of
    every analysis stored before the report planner wrote new ones, which still lists, reads and renders.

    :param host: The host holding the campaign's runs and results.
    :param campaign: The toy corpus campaign.
    :return: The stored analysis and its insights.
    """
    record = _record("memo", campaign)
    return off_loop(_stored, host.storage, record)


async def stored_plan_analysis(host: EvalHost, campaign: EvalCampaign) -> tuple[EvalAnalysis, list[EvalInsight]]:
    """The analysis the report planner stored over the toy corpus campaign, stored again with its plan and insights:
    the shape of every analysis the planner wrote, which reads as the code-only page with the regenerate line.

    :param host: The host holding the campaign's runs and results.
    :param campaign: The toy corpus campaign.
    :return: The stored analysis and its insights; its plan reads with ``get_report_plan``.
    """
    record = _record("plan", campaign)

    def store() -> tuple[EvalAnalysis, list[EvalInsight]]:
        host.storage.save_report_plan(EvalReportPlan.model_validate(record["plan"]))
        return _stored(host.storage, record)

    return off_loop(store)


async def toy_report() -> tuple[EvalHost, EvalAnalysis, Report]:
    """The toy campaign's memo analysis, stored, and its one aspect's page laid out over it."""
    host, campaign = off_loop(toy_campaign_host)
    analysis, _ = await stored_memo_analysis(host, campaign)
    return host, analysis, build_report(analysis, campaign_name=campaign.name)
