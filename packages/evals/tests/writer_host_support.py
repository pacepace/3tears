"""The toy host as the report writer's tests read it: the caller, the writer template and an operations host.

Shared by every test on the writer's path (rung 5's, the writer bank's, the writer harness's, the recorded writer's)
and by the export of rung 5's case files. The host is the toy host's own operations host
(:func:`~packages.evals.tests.fixtures.toyhost.ops.toyhost_ops_host`): the writer's kind contract and measures on its
profile, its report budget and its reading of a model's family, its template saved and its corpus campaign stored.
Its completion clients are the caller's: the writer's own models are handed in by the call that runs them (a turn model
to ``write_report``, a model factory to the writer's launchable kind), so a test passes clients only for a launch,
whose judge the host builds. Nothing here reaches the report planner, the memo generator or their fixtures.

**An analysis to read.** A test whose subject is something a stored analysis carries (its frozen aspects, its run
index, its provenance, the insights it mints) reaches one the way ``analysis_generate`` writes one today: the
report writer's document, linked by an analysis whose ``basis`` is ``doc``. :func:`stored_doc_analysis` runs the
generation's job on a host whose ``analyst`` client is scripted; :func:`doc_analysis_over` builds the same analysis
over a bundle a test shaped and holds in hand, as the job builds it once the document is stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from packages.evals.tests.doc_writer_support import ANALYST_MODEL, TOY_DOC

from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.fixtures.toyhost.ops import toyhost_ops_host
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.off_loop import off_loop
from threetears.evals.actions import Caller
from threetears.evals.actions.report_writing import campaign_report_writer
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.doc_analysis import doc_analysis
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel import EvalCampaign, EvalStorage
from threetears.evals.kernel.campaign import EvalAnalysis, EvalInsight
from threetears.evals.kernel.host import CompletionClients
from threetears.evals.kernel.report_doc import EvalReportDoc
from threetears.evals.ops import OpsHost, analysis_generate, await_job
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND
from threetears.evals.run import LaunchSettings
from threetears.evals.schema.models import EvalTemplate, utc_now_iso
from threetears.evals.storage import InMemoryDocumentStore

__all__ = ["CALLER", "WRITER_TEMPLATE", "WriterHost", "doc_analysis_over", "stored_doc_analysis", "writer_host"]

#: Who the tests call as: the toy host's scope, under a name a written campaign records.
CALLER: Final = Caller(scope_id=TOYHOST_SCOPE, identity="agent:test")

#: The writer template every seeded case is minted into.
WRITER_TEMPLATE: Final = EvalTemplate(
    id="writer-template",
    scope_id=CALLER.scope_id,
    name="Report writer",
    intent="Write the report.",
    candidate_kind=WRITER_KIND,
    # Stamped, not read from the clock: a constant built at import reads no clock.
    created_at="2026-08-01T00:00:00+00:00",
    updated_at="2026-08-01T00:00:00+00:00",
)


@dataclass(frozen=True)
class WriterHost:
    """The operations host, and the corpus campaign stored in it.

    :ivar host: The host.
    :ivar campaign: The toy host's corpus campaign, which a case can be minted from.
    """

    host: OpsHost
    campaign: EvalCampaign


def writer_host(
    *, settings: LaunchSettings = TOYHOST_LAUNCH_SETTINGS, clients: CompletionClients | None = None
) -> WriterHost:
    """The toy host's operations host, its template and the writer template saved and its corpus campaign stored.

    :param settings: The host's launch settings.
    :ptype settings: LaunchSettings
    :param clients: The host's completion clients, the judge of a writer launch among them; ``None`` for none.
    :ptype clients: CompletionClients | None
    :return: The host and its campaign.
    :rtype: WriterHost
    """
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    campaign, corpus = toyhost_campaign()
    for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
        storage.save_eval_run(run)
        for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
            storage.save_eval_result(result)
    storage.save_campaign(campaign)
    storage.save_template(WRITER_TEMPLATE)
    return WriterHost(toyhost_ops_host(storage=storage, clients=clients, settings=lambda: settings), campaign)


async def stored_doc_analysis(
    host: OpsHost, campaign_id: str, scope_id: str = CALLER.scope_id
) -> tuple[EvalAnalysis, list[EvalInsight]]:
    """The campaign's analysis as ``analysis_generate`` stores it: the report writer's job run to its end, the host's
    ``analyst`` client writing the document (a scripted one, so nothing reaches a model).

    :param host: The host, whose ``analyst`` client writes the report.
    :ptype host: OpsHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :return: The stored analysis (``basis`` ``doc``) and the insights it minted.
    :rtype: tuple[EvalAnalysis, list[EvalInsight]]
    :raises AssertionError: The job ended without storing an analysis.
    """
    started = await analysis_generate(host, campaign_id, scope_id, writer=campaign_report_writer(CALLER))
    (status,) = await await_job(host, started, scope_id)
    if status.state != "completed" or status.analysis_id is None:
        raise AssertionError(f"the generation ended {status.state}, storing no analysis: {status}")
    storage = host.eval_host.storage
    analysis = off_loop(storage.load_analysis, status.analysis_id, scope_id)
    assert analysis is not None
    minted = off_loop(storage.query_insights, scope_id, source_campaign_id=campaign_id)
    return analysis, [insight for insight in minted if insight.source_analysis_id == analysis.id]


def doc_analysis_over(bundle: CampaignBundle) -> tuple[EvalAnalysis, list[EvalInsight]]:
    """The analysis a report document over ``bundle`` is linked by, built as the generation's job builds it once the
    writer has stored the toy document (:data:`~packages.evals.tests.doc_writer_support.TOY_DOC`); nothing is stored.

    :param bundle: The campaign bundle the document's snapshot froze.
    :ptype bundle: CampaignBundle
    :return: The analysis (``basis`` ``doc``) and its insights.
    :rtype: tuple[EvalAnalysis, list[EvalInsight]]
    """
    stored = EvalReportDoc(
        scope_id=bundle.scope_id,
        snapshot_id="report-snapshot:held-in-hand",
        campaign_id=bundle.campaign_id,
        doc=TOY_DOC,
        ending="stored",
        findings=[],
        generation={},
        created_by=CALLER.identity,
    )
    built = doc_analysis(
        bundle,
        report_input(bundle),
        stored,
        input_digest="held-in-hand",
        generator_model=ANALYST_MODEL,
        skill_digest=load_skill().digest,
        bundle_assembled_at=utc_now_iso(),
        token_cost=None,
        bounces={},
    )
    return built.analysis, built.insights
