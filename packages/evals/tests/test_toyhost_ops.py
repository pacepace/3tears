"""The toy host as an operations host (``fixtures/toyhost/ops.py``) writes a campaign's report, and places models.

Held here: ``analysis_generate`` on the toy host stores a finished report document over its corpus campaign, written
on the client its ``analyst`` role returns (scripted: nothing reaches a model), with the job's ceiling above the
writer's own; the profile carries the writer's kind contract, seating the toy host's pinned role, and every writer
measure beside the toy host's own; and a model's family is the vendor its id names, or none.
"""

from __future__ import annotations

import asyncio
from typing import Any

from packages.evals.tests.doc_writer_support import FakeAnalystClient
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.ops import (
    TOYHOST_REPORT_BUDGET,
    TOYHOST_REQUEST_S,
    toyhost_model_family,
    toyhost_ops_host,
    toyhost_writer_profile,
)
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES
from packages.evals.tests.off_loop import off_loop
from threetears.evals.actions import Caller
from threetears.evals.actions.report_writing import campaign_report_writer
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.report_doc import EvalReportDoc
from threetears.evals.ops import WRITER_KIND_MEASURES, JobStatus, OpsHost, analysis_generate, job_poll
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND
from threetears.evals.run import Ceilings
from threetears.evals.storage import InMemoryDocumentStore

_CALLER = Caller(scope_id=TOYHOST_SCOPE, identity="agent:test")


def _host() -> tuple[OpsHost, str, list[FakeAnalystClient]]:
    """The toy host's operations host over its stored corpus campaign, its ``analyst`` client scripted."""
    storage = EvalStorage(InMemoryDocumentStore())
    campaign, corpus = toyhost_campaign()
    for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
        storage.save_eval_run(run)
        for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
            storage.save_eval_result(result)
    storage.save_campaign(campaign)
    analysts: list[FakeAnalystClient] = []

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> Any:
        assert role == "analyst", f"a report asks only the analyst role's client, not {role!r}'s"
        analysts.append(FakeAnalystClient())
        return analysts[-1]

    return toyhost_ops_host(storage=storage, clients=clients), campaign.id, analysts


async def _settled(host: OpsHost, job_id: str) -> JobStatus:
    async with asyncio.timeout(10):
        while not (status := await job_poll(host, job_id, TOYHOST_SCOPE)).done:
            await asyncio.sleep(0.01)
    return status


async def test_the_toy_host_writes_a_campaigns_report_on_its_analyst_client() -> None:
    host, campaign_id, analysts = off_loop(_host)

    started = await analysis_generate(host, campaign_id, TOYHOST_SCOPE, writer=campaign_report_writer(_CALLER))
    (job,) = started.jobs
    status = await _settled(host, job.job_id)

    assert (status.state, status.status) == ("completed", "stored"), status
    assert status.analysis_id is not None
    analysis = off_loop(host.eval_host.storage.load_analysis, status.analysis_id, TOYHOST_SCOPE)
    assert analysis is not None and analysis.basis == "doc"
    stored = off_loop(host.eval_host.storage.load_report_doc, analysis.report_doc_id, TOYHOST_SCOPE)
    assert isinstance(stored, EvalReportDoc) and (stored.ending, stored.campaign_id) == ("stored", campaign_id)
    assert len(analysts) == 1 and analysts[0].released


def test_the_reports_job_ceiling_sits_above_the_writers_own() -> None:
    generation = toyhost_ops_host().generation
    assert generation is not None and generation.report_budget is not None
    assert generation.report_budget() == TOYHOST_REPORT_BUDGET
    writing_s = Ceilings().writer_ceiling_s(
        generation.limits, request_s=lambda _max_tokens: TOYHOST_REQUEST_S, turns=TOYHOST_REPORT_BUDGET.turns
    )
    assert generation.budget_s > writing_s


def test_the_profile_declares_the_writers_kind_seating_the_pinned_role_and_its_measures() -> None:
    profile = toyhost_writer_profile()
    (writer,) = (kind for kind in profile.kinds if kind.kind == WRITER_KIND)
    assert writer.seats is not None and "reviewer_pool" in writer.seats
    names = set(profile.measures.names)
    assert {measure.name for measure in (*TOYHOST_MEASURES, *WRITER_KIND_MEASURES)} == names


def test_a_models_family_is_the_vendor_its_id_names_or_none() -> None:
    assert toyhost_ops_host().model_family is toyhost_model_family
    assert toyhost_model_family("anthropic/claude-sonnet") == "anthropic"
    assert toyhost_model_family("google/gemini-flash") == "google"
    assert toyhost_model_family("toy-analyst-model") is None
    assert toyhost_model_family("/unrouted") is None
