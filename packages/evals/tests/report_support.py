"""Reports to test against: the toy host's, and the analyses its retired writers stored over its corpus campaign.

Shared by ``test_report.py`` and ``test_report_html_no_script.py`` — harness support, not part of the
toy host, which reaches the engine only through its public roots.

**The retired writers' analyses are stored records, never generated.** The memo writer and the report planner wrote
the two records under ``fixtures/retired_writers/`` over the toy corpus campaign (at ``52bf24449``): each holds the
analysis as its writer stored it, with its insights, and the planner's its report plan. A host holds analyses like
these, which every read keeps serving, so a test stores the record as it was written and reads it back; nothing here
calls either writer. A test of the memo renderer over a memo of its own stores it with :func:`memo_analysis_over`, built
as the memo writer stored one, with no writer. Nothing reads a stored plan any more, so the package has no write for one either: a test that
stores the plan beside its analysis writes it through the host's document store (:func:`stored_plan_document`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from threetears.evals.analysis import CampaignBundle, Report, TableBlock, build_report, report_markdown
from threetears.evals.analysis.doc_analysis import aspect_analysis
from threetears.evals.analysis.references import CellAddress, resolve_reading
from threetears.evals.analysis.style import StyleViolation
from packages.evals.tests.style_support import lint_page
from threetears.evals.kernel.authored import AuthoredCampaignAnalysis
from threetears.evals.kernel.campaign import (
    EvalAnalysis,
    EvalCampaign,
    EvalInsight,
    EvidenceRow,
    FindingResolution,
    GenerationProvenance,
)
from threetears.evals.kernel.evidence_tiers import JUDGED_TIER_RULE
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.storage import EvalStorage, save_document
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.schema.store_port import DocumentStore
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.off_loop import off_loop

__all__ = [
    "memo_analysis_over",
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
    """The toy campaign's memo analysis, stored, and its one aspect's page laid out over it."""
    host, campaign = off_loop(toy_campaign_host)
    analysis, _ = await stored_memo_analysis(host, campaign)
    return host, analysis, build_report(analysis, campaign_name=campaign.name)


def memo_analysis_over(
    campaign: CampaignBundle, memo: dict[str, Any], *, generated_at: str | None = None
) -> EvalAnalysis:
    """``memo`` over ``campaign`` as the retired memo writer stored it, built with no writer: each cell a finding names
    (by the alias the writer was shown, or by its full ``cell_ref`` on the first aspect) stored by its ``cell_ref``, each
    reading resolved by code on its aspect's surface, and every aspect's bundle and the campaign's frame frozen beside
    it. Its prose is kept as written.

    :param campaign: The campaign bundle the memo was written over.
    :param memo: The authored memo, as the writer returned it.
    :param generated_at: When it was written (ISO-8601); None for now.
    :return: The analysis, not stored.
    """
    document = AuthoredCampaignAnalysis.model_validate(memo)
    surfaces = campaign.surfaces()
    aliases = campaign.cell_aliases()
    findings, resolutions = [], []
    for finding in document.findings:
        addresses = [
            aliases.get(row.cell) or CellAddress(aspect=campaign.labels[0], cell_ref=row.cell)
            for row in finding.evidence
        ]
        rows = []
        for row, where in zip(finding.evidence, addresses, strict=True):
            resolved = resolve_reading(surfaces[where.aspect], where.cell_ref, row.measure_id, row.reading)
            rows.append(
                EvidenceRow(
                    aspect=where.aspect,
                    cell_ref=where.cell_ref,
                    measure_id=row.measure_id,
                    reading=row.reading,
                    value=resolved.mean,
                    n=resolved.n,
                    n_cases=resolved.n_cases,
                    dispersion=resolved.dispersion,
                    judged_tier=resolved.judged_tier,
                )
            )
        resolutions.append(FindingResolution(evidence=rows))
        evidence = [
            row.model_copy(update={"cell": where.cell_ref})
            for row, where in zip(finding.evidence, addresses, strict=True)
        ]
        findings.append(finding.model_copy(update={"evidence": evidence}))
    recommended = document.recommended()
    arm = campaign.arm(recommended.arm) if recommended is not None else None
    first = campaign.aspects[0]
    written = generated_at if generated_at is not None else utc_now_iso()
    return EvalAnalysis(
        recommended_variant_key=arm.variant_key if arm is not None else None,
        campaign_id=campaign.campaign_id,
        subject_id=campaign.subject_id,
        subject_kind=campaign.subject_kind,
        behavior=campaign.behavior,
        observation_refs=campaign.run_ids,
        model_versions={key: value for aspect in campaign.aspects for key, value in aspect.model_versions.items()},
        scope_id=campaign.scope_id,
        generation=GenerationProvenance(
            prompt_id="eval_analysis_gen",
            prompt_version="memo-writer",
            generator_model="memo-writer-model",
            bundle_fingerprint=campaign.fingerprint(),
            generated_at=written,
            bundle_assembled_at=written,
            token_cost=0.0,
            repair_attempts=0,
            repaired_refusal=None,
            user_message_digest="sha256:memo",
            cell_model_version=first.cell_model_version,
            bundle_schema_version=first.schema_version,
            host_declarations_digest=first.host_declarations_digest,
        ),
        design_snapshot=campaign.declaration,
        document=document.model_copy(update={"findings": findings}),
        resolutions=resolutions,
        judged_tier_rule=JUDGED_TIER_RULE,
        aspects=[aspect_analysis(aspect, surfaces[aspect.aspect]) for aspect in campaign.aspects],
        campaign_bundle=campaign.frame(),
    )
