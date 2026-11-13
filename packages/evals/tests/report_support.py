"""Reports to test against: the toy host's, and the analyses its retired writers stored over its corpus campaign.

Shared by ``test_report.py`` and ``test_report_html_no_script.py`` — harness support, not part of the
toy host, which reaches the engine only through its public roots.

**The retired writers' analyses are stored records, never generated.** The memo writer and the report planner wrote
the two records under ``fixtures/retired_writers/`` over the toy corpus campaign (at ``52bf24449``): each holds the
analysis as its writer stored it, with its insights, and the planner's its report plan. A host holds analyses like
these, which every read keeps serving, so a test stores the record as it was written and reads it back; nothing here
calls either writer. Nothing reads a stored plan or memo any more, so the package has no write for one either: a test
that stores the plan beside its analysis writes it through the host's document store (:func:`stored_plan_document`).
An aspect's page of a stored analysis is code's, over the bundle frozen with it (:func:`toy_report`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from threetears.evals.analysis import Report, TableBlock, build_code_only_report, report_markdown
from threetears.evals.analysis.report.build import analysis_campaign_bundle, analysis_writer
from threetears.evals.analysis.report.words import declared_arm_names
from threetears.evals.analysis.style import StyleViolation
from packages.evals.tests.style_support import lint_page
from threetears.evals.kernel.declaration import CampaignArm
from threetears.evals.kernel.campaign import (
    EvalAnalysis,
    EvalCampaign,
    EvalInsight,
)
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.storage import EvalStorage, save_document
from threetears.evals.schema.store_port import DocumentStore
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.off_loop import off_loop

__all__ = [
    "page_violations",
    "stored_memo_analysis",
    "stored_plan_analysis",
    "stored_plan_document",
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
    """The analysis the report planner stored over the toy corpus campaign, stored again with its insights: the shape
    of every analysis the planner wrote, which reads as the code-only page with the retired writer's line. Its plan is
    not stored (nothing reads it): :func:`stored_plan_document` stores it beside the analysis.

    :param host: The host holding the campaign's runs and results.
    :param campaign: The toy corpus campaign.
    :return: The stored analysis and its insights.
    """
    return off_loop(_stored, host.storage, _record("plan", campaign))


def stored_plan_document(
    store: DocumentStore, campaign: EvalCampaign, *, plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The report plan the planner stored beside its analysis over the toy corpus campaign, written as a host's
    document store holds it (``eval_report_plan``), never validated on the way in.

    :param store: The document store under the host's storage.
    :param campaign: The toy corpus campaign.
    :param plan: The plan body to store in place of the planner's own; None stores its own.
    :return: The stored document.
    """
    document: dict[str, Any] = dict(_record("plan", campaign)["plan"])
    if plan is not None:
        document["plan"] = plan
    save_document(store, document)
    return document


async def toy_report() -> tuple[EvalHost, EvalAnalysis, Report]:
    """The toy campaign's memo analysis, stored, and its one aspect's page: code's, over the bundle frozen with it, its
    arms named as the campaign declares them and its methods naming the analysis's writer."""
    host, campaign = off_loop(toy_campaign_host)
    analysis, _ = await stored_memo_analysis(host, campaign)
    (held,) = analysis.aspects
    listed = analysis.campaign_bundle.get("arms")
    arms = [CampaignArm.model_validate(arm) for arm in listed] if isinstance(listed, list) else []
    report = build_code_only_report(
        analysis_campaign_bundle(analysis).aspects[0],
        assembled_at=analysis.generation.bundle_assembled_at,
        campaign_name=campaign.name,
        arm_short_names=declared_arm_names(arms),
        writer=analysis_writer(analysis),
        bundle_fingerprint=held.bundle_fingerprint,
    )
    return host, analysis, report
