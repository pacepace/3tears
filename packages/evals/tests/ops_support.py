"""An operations host over the toy host: its template to launch, its corpus campaign to analyse, a writer to analyse it.

Shared by ``test_action_catalogue.py`` and ``test_fastmcp_transport.py`` — harness support, not part of
the toy host. The non-production parts are the models. The ``analyst`` role's client is a scripted turn model
(:class:`~packages.evals.tests.doc_writer_support.FakeAnalystClient`) whose analyst submits a report document valid on
the toy campaign's snapshot and whose editor clears it, so a generation started through ``analysis_generate`` stores a
real report document and its analysis. Every other role's is a fixtured client whose report plan is written over the
campaign's bundle as the generation assembled it (the planner's structure, then its prose), so the planner's path
(``prepare_analysis_generation`` and ``run_analysis_generation``) stores a real planned analysis
(:func:`planned_generation`). Sent the memo writer's contract, it writes the toy memo instead.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from threetears.evals.actions import Caller
from threetears.evals.actions.report_writing import campaign_report_writer
from threetears.evals.analysis import assemble_campaign_bundle, prepare_analysis_generation, run_analysis_generation
from threetears.evals.kernel.campaign import EvalAnalysis, EvalInsight
from threetears.evals.kernel import EvalCampaign, EvalStorage
from threetears.evals.kernel.host import EvalHost, HostProfile
from threetears.evals.ops import (
    AnalysisGeneration,
    JobsStarted,
    JobStatus,
    OpsHost,
    ReportBudget,
    analysis_generate,
    await_job,
    job_poll,
)
from threetears.evals.run import LaunchSettings
from threetears.evals.schema import DEFAULT_JOB_LEASE_S, DocumentStore
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.toyhost_memo import (
    FIXTURED_CALL_CEILING_USD,
    FixturedClient,
    memo_payload,
)
from packages.evals.tests.doc_writer_support import ANALYST_MODEL, FakeAnalystClient
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.toyhost_plan import PROSE_CONTRACT, STRUCTURE_CONTRACT, contract_name, replies_for
from threetears.evals.analysis.plan_prompt import EVAL_REPORT_PLAN_DEFAULT
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel.prompts.catalogue import REPORT_PLAN_PROMPT

__all__ = [
    "CALLER",
    "REPORT_BUDGET_USD",
    "WRITER",
    "generate",
    "planned_generation",
    "launched_runs",
    "RUN_MODELS",
    "TOYHOST_SCOPE",
    "TOYHOST_SUBJECT",
    "MemoWriter",
    "OpsFixture",
    "ops_fixture",
    "settled",
]

#: Who the tests call as: the toy host's scope, under a name a written campaign records.
CALLER = Caller(scope_id=TOYHOST_SCOPE, identity="agent:test")

#: The report writer a generation started as :data:`CALLER` runs, as the ``analysis_generate`` action hands it.
WRITER = campaign_report_writer(CALLER)

#: The toy host's budget for one report.
REPORT_BUDGET_USD = 10.0


class MemoWriter(FixturedClient):
    """The fixtured generator, writing the toy plan's structure and prose (or, sent the memo writer's contract, the toy
    memo) over the bundle the generation is about to send."""

    def __init__(self, host: EvalHost, campaign: EvalCampaign) -> None:
        """Bind the writer to the campaign it analyses.

        :param host: Where the campaign's runs are read.
        :ptype host: EvalHost
        :param campaign: The campaign the generation is over.
        :ptype campaign: EvalCampaign
        """
        super().__init__("")
        self._host = host
        self._campaign = campaign
        #: Set to hold the generation open until the test releases it.
        self.gate: asyncio.Event | None = None
        #: When set, what the FIRST call returns in place of the memo — an output the generator refuses,
        #: so the generation buys its repair round-trip.
        self.first_output: str | None = None
        #: When set, what every call raises once the gate opens — a provider's failure, as its SDK raises one.
        self.raises: Exception | None = None

    async def generate(self, *, system: str, user: str, response_format: Any = None, tools: Any = None) -> Any:
        """Write the memo over the campaign's bundle, then hand it back as the completion."""
        if self.gate is not None:
            await self.gate.wait()
        if self.raises is not None:
            raise self.raises
        if self.first_output is not None and not self.calls:
            self.completion.content = self.first_output
            return await super().generate(system=system, user=user, response_format=response_format, tools=tools)
        bundle = off_loop(assemble_campaign_bundle, self._host, self._campaign.id, self._campaign.scope_id)
        if contract_name(response_format) in (STRUCTURE_CONTRACT, PROSE_CONTRACT):
            self.completion.content = replies_for(lambda: report_input(bundle))(response_format)
        else:
            self.completion.content = json.dumps(memo_payload(bundle))
        return await super().generate(system=system, user=user, response_format=response_format, tools=tools)


class OpsFixture:
    """The operations host, the corpus campaign it can analyse, and the writers it built."""

    def __init__(
        self,
        host: OpsHost,
        campaign: EvalCampaign,
        writers: list[MemoWriter],
        analysts: list[FakeAnalystClient] | None = None,
    ) -> None:
        """Hold the pieces a test reads: a fixture another file builds over its own host has no analyst clients."""
        self.host = host
        self.campaign = campaign
        self.writers = writers
        self.analysts = analysts if analysts is not None else []


def ops_fixture(
    *,
    generation: bool = True,
    gate: asyncio.Event | None = None,
    ceiling_usd: float | None = FIXTURED_CALL_CEILING_USD,
    first_output: str | None = None,
    settings: LaunchSettings = TOYHOST_LAUNCH_SETTINGS,
    profile: HostProfile | None = None,
    store: DocumentStore | None = None,
    job_lease_s: float = DEFAULT_JOB_LEASE_S,
    analyst_unpriced: str | None = None,
) -> OpsFixture:
    """The toy host as an operations host: its template saved, its corpus campaign stored.

    :param generation: Whether the host generates analyses (its :class:`AnalysisGeneration`).
    :ptype generation: bool
    :param gate: Holds every generation open until set, for a test that needs one running.
    :ptype gate: asyncio.Event | None
    :param ceiling_usd: What each generator call (and each ``analyst`` turn) is priced at, at most; ``None`` for a
        writer that cannot price.
    :ptype ceiling_usd: float | None
    :param first_output: What each writer's first call returns in place of the memo, when set.
    :ptype first_output: str | None
    :param settings: The launch settings — the run and out-of-run caps, and whether they are enforced.
    :ptype settings: LaunchSettings
    :param profile: The host's vocabulary; ``None`` is the toy host's own.
    :ptype profile: HostProfile | None
    :param store: The document store under the host's storage; ``None`` is a fresh in-memory one.
    :ptype store: DocumentStore | None
    :param job_lease_s: How long the host's job leases last unless renewed.
    :ptype job_lease_s: float
    :param analyst_unpriced: Why the ``analyst`` client's turns cannot be priced, when set
        (``turn_pricing_refusal``); ``None`` for a client that prices every turn at ``ceiling_usd``.
    :ptype analyst_unpriced: str | None
    :return: The fixture.
    :rtype: OpsFixture
    """
    storage = EvalStorage(store if store is not None else InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    campaign, corpus = toyhost_campaign()
    for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
        storage.save_eval_run(run)
        for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
            storage.save_eval_result(result)
    storage.save_campaign(campaign)
    writers: list[MemoWriter] = []
    analysts: list[FakeAnalystClient] = []
    hosts: list[EvalHost] = []

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> MemoWriter | FakeAnalystClient:
        if role == "analyst":
            analyst = FakeAnalystClient(
                model_name=model or ANALYST_MODEL, gate=gate, ceiling=ceiling_usd, unpriced=analyst_unpriced
            )
            analysts.append(analyst)
            return analyst
        writer = MemoWriter(hosts[0], campaign)
        writer.gate = gate
        writer.ceiling_usd = ceiling_usd
        writer.first_output = first_output
        writers.append(writer)
        return writer

    launch, _client = toyhost_launch_host(
        profile=profile, storage=storage, clients=clients, settings=lambda: settings, job_lease_s=job_lease_s
    )
    hosts.append(launch.eval_host)

    generating = AnalysisGeneration(report_budget=lambda: ReportBudget(usd=REPORT_BUDGET_USD), budget_s=60.0)
    host = OpsHost(launch=launch, generation=generating if generation else None)
    return OpsFixture(host, campaign, writers, analysts)


async def generate(host: OpsHost, campaign_id: str, scope_id: str = TOYHOST_SCOPE, **given: Any) -> JobsStarted:
    """``analysis_generate`` as the action starts it for :data:`CALLER`: :data:`WRITER` writes the report."""
    return await analysis_generate(host, campaign_id, scope_id, writer=WRITER, **given)


async def _planner_prompt() -> str:
    # The planner's prompt, as a host registers it.
    return EVAL_REPORT_PLAN_DEFAULT


async def planned_generation(
    fixture: OpsFixture, campaign_id: str | None = None, *, model: str | None = None
) -> tuple[EvalAnalysis, list[EvalInsight]]:
    """A planned analysis of the corpus campaign, by the report planner's path (which the reporter kind runs), inline.

    :param fixture: The toy host.
    :ptype fixture: OpsFixture
    :param campaign_id: The campaign; the corpus campaign when ``None``.
    :ptype campaign_id: str | None
    :param model: The planner model, or ``None`` for the host's default.
    :ptype model: str | None
    :return: The stored analysis and its insights; its plan reads with ``get_report_plan``.
    :rtype: tuple[EvalAnalysis, list[EvalInsight]]
    """
    eval_host = fixture.host.eval_host
    target = campaign_id or fixture.campaign.id
    prepared = await prepare_analysis_generation(
        eval_host,
        target,
        fixture.campaign.scope_id,
        model=model,
        resolve_prompt=_planner_prompt,
        out_of_run_cap_usd=fixture.host.out_of_run_cap(),
    )
    return await run_analysis_generation(eval_host, prepared, prompt_id=REPORT_PLAN_PROMPT, max_output_tokens=4000)


async def settled(host: OpsHost, job_id: str, scope_id: str = TOYHOST_SCOPE) -> JobStatus:
    """Poll a job until it is done, failing after a few seconds."""
    async with asyncio.timeout(10):
        while not (status := await job_poll(host, job_id, scope_id)).done:
            await asyncio.sleep(0.01)
    return status


async def launched_runs(host: OpsHost, started: JobsStarted, scope_id: str = TOYHOST_SCOPE) -> list[str]:
    """A launch's runs, once its job has ended: awaited in the foreground, and read off its final status."""
    (status,) = await await_job(host, started, scope_id)
    return list(status.run_ids)
