"""An operations host over the toy host: its template to launch, its corpus campaign to analyse, a writer to analyse it.

Shared by ``test_action_catalogue.py`` and ``test_fastmcp_transport.py`` — harness support, not part of
the toy host. The non-production parts are the models. The ``analyst`` role's client is a scripted turn model
(:class:`~packages.evals.tests.doc_writer_support.FakeAnalystClient`) whose analyst submits a report document valid on
the toy campaign's snapshot and whose editor clears it, so a generation started through ``analysis_generate`` stores a
real report document and its analysis. Every other role's client (a judge's, a simulator's, the variation writer's) is
a :class:`~packages.evals.tests.fixtured_client.FixturedClient` that prices each call at the fixture's ceiling and
answers an empty object: no report is written in any role but the analyst's.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from threetears.evals.actions.report_writing import campaign_report_writer
from threetears.evals.kernel import EvalCampaign, EvalStorage
from threetears.evals.kernel.host import HostProfile
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
from threetears.evals.run import LaunchSettings, build_judge_service
from threetears.evals.schema import (
    DEFAULT_JOB_LEASE_S,
    CatalogRubricDim,
    DocumentStore,
    JudgeConfig,
    JudgedArtifact,
    RubricDim,
    RunDisclosure,
)
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.doc_writer_support import ANALYST_MODEL, FakeAnalystClient
from packages.evals.tests.factories import declared
from packages.evals.tests.fixtured_client import FIXTURED_CALL_CEILING_USD, FixturedClient
from packages.evals.tests.writer_host_support import CALLER

__all__ = [
    "CALLER",
    "LaunchedJudges",
    "REPORT_BUDGET_USD",
    "WRITER",
    "generate",
    "judged_as_launched",
    "launched_runs",
    "RUN_MODELS",
    "TOYHOST_SCOPE",
    "TOYHOST_SUBJECT",
    "OpsFixture",
    "ops_fixture",
    "settled",
]

#: The report writer a generation started as :data:`CALLER` runs, as the ``analysis_generate`` action hands it.
WRITER = campaign_report_writer(CALLER)

#: The toy host's budget for one report.
REPORT_BUDGET_USD = 10.0


class OpsFixture:
    """The operations host, the corpus campaign it can analyse, and the clients it built.

    :ivar writers: Every client built for a role other than the analyst's, in the order built.
    :ivar analysts: Every ``analyst`` client built.
    """

    def __init__(
        self,
        host: OpsHost,
        campaign: EvalCampaign,
        writers: list[Any],
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
    settings: LaunchSettings = TOYHOST_LAUNCH_SETTINGS,
    profile: HostProfile | None = None,
    store: DocumentStore | None = None,
    job_lease_s: float = DEFAULT_JOB_LEASE_S,
    analyst_unpriced: str | None = None,
    run_disclosure: Callable[[], RunDisclosure] | None = None,
) -> OpsFixture:
    """The toy host as an operations host: its template saved, its corpus campaign stored.

    :param generation: Whether the host generates analyses (its :class:`AnalysisGeneration`).
    :ptype generation: bool
    :param gate: Holds every generation open until set, for a test that needs one running.
    :ptype gate: asyncio.Event | None
    :param ceiling_usd: What each call of any role (each ``analyst`` turn among them) is priced at, at most; ``None``
        for a client that cannot price.
    :ptype ceiling_usd: float | None
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
    :param run_disclosure: What the host discloses about each launch (``LaunchHost.run_disclosure``); ``None``
        discloses nothing.
    :ptype run_disclosure: Callable[[], RunDisclosure] | None
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
    writers: list[Any] = []
    analysts: list[FakeAnalystClient] = []

    def clients(
        role: str, model: str | None, *, temperature: float | None = None
    ) -> FixturedClient | FakeAnalystClient:
        if role != "analyst":
            writer = FixturedClient("{}")
            writer.ceiling_usd = ceiling_usd
            writers.append(writer)
            return writer
        analyst = FakeAnalystClient(
            model_name=model or ANALYST_MODEL, gate=gate, ceiling=ceiling_usd, unpriced=analyst_unpriced
        )
        analysts.append(analyst)
        return analyst

    launch, _client = toyhost_launch_host(
        profile=profile,
        storage=storage,
        clients=clients,
        settings=lambda: settings,
        job_lease_s=job_lease_s,
        run_disclosure=run_disclosure,
    )

    generating = AnalysisGeneration(report_budget=lambda: ReportBudget(usd=REPORT_BUDGET_USD), budget_s=60.0)
    host = OpsHost(launch=launch, generation=generating if generation else None)
    return OpsFixture(host, campaign, writers, analysts)


async def generate(host: OpsHost, campaign_id: str, scope_id: str = TOYHOST_SCOPE, **given: Any) -> JobsStarted:
    """``analysis_generate`` as the action starts it for :data:`CALLER`: :data:`WRITER` writes the report."""
    return await analysis_generate(host, campaign_id, scope_id, writer=WRITER, **given)


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


@dataclass(frozen=True)
class LaunchedJudges:
    """A configured judge and a built-in one, recorded on the corpus campaign's runs and results as a launch records them.

    :ivar config: The stored config that judges ``toy.dim``.
    :ivar dim: The stored catalog dim the config names.
    :ivar run: What each run records (``EvalRun.judge_config_ids``): the config set it committed to, bare config ids.
    :ivar result: What each result records (``EvalResult.judge_config_ids``): ``<config id>@<sha12>`` for the
        configured dim and ``builtin:<sha12>`` for the built-in one.
    """

    config: JudgeConfig
    dim: CatalogRubricDim
    run: dict[str, str]
    result: dict[str, str]


def judged_as_launched(fixture: OpsFixture, store: DocumentStore) -> LaunchedJudges:
    """Record a configured judge on every run and result of the fixture's campaign, the way a launch records it.

    The corpus campaign's runs were judged by no config, so they show nothing of the two shapes a launch writes. Both
    maps come from the launch's own judge build (:func:`~threetears.evals.run.build_judge_service`) over a template
    declaring one dim configured and one built-in: the run's from :attr:`RunJudge.config_ids`, each result's from its
    service's :meth:`~threetears.evals.run.judge_service.JudgeService.judge_config_ids`.

    :param fixture: The fixture whose campaign's runs and results are written.
    :ptype fixture: OpsFixture
    :param store: The document store under the fixture's storage.
    :ptype store: DocumentStore
    :return: The config, its dim, and the two maps as recorded.
    :rtype: LaunchedJudges
    """
    eval_host = fixture.host.eval_host
    config = JudgeConfig(scope_id=TOYHOST_SCOPE, name="toy", rubric_dim_id="toy.dim", prompt_template="Judge it.")
    dim = CatalogRubricDim(
        scope_id=TOYHOST_SCOPE,
        key="toy_dim",
        dim=RubricDim(name="toy.dim", description="A toy dim.", scale="pass_fail"),
    )
    eval_host.storage.save_judge_config(config)
    eval_host.storage.save_rubric_dim(dim)
    rubric = [
        declared(dim.dim, "configured"),
        declared(RubricDim(name="toy.plain", description="A dim the built-in prompt judges.", scale="pass_fail")),
    ]
    template = toyhost_template().model_copy(update={"rubric": rubric})
    built = build_judge_service(eval_host, template, "judge/toy", judged_artifact=JudgedArtifact.DOCUMENT)
    judges = LaunchedJudges(
        config=config,
        dim=dim,
        run=built.config_ids,
        result=built.service.judge_config_ids(rubric, judged_artifact=JudgedArtifact.DOCUMENT),
    )
    for run_id in fixture.campaign.run_ids:
        assert store.merge_fields(run_id, TOYHOST_SCOPE, {"judge_config_ids": judges.run})
        for result in store.by_doc_type("eval_result", TOYHOST_SCOPE, eval_run_id=run_id):
            assert store.merge_fields(result["id"], TOYHOST_SCOPE, {"judge_config_ids": judges.result})
    return judges
