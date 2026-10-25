"""An analysis generation driven through the host: its storage, its vocabulary and its client factory.

The service operations that read the host's vocabulary take the
:class:`~threetears.evals.kernel.host.EvalHost`; the generator client is built from the host's
factory for the ``analysis`` role, not handed in by the caller. These drive the toy host's corpus
campaign from the engine's own storage through a fixtured client, and pin the refusal of a host that
supplies no client factory.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis import prepare_analysis_generation, run_analysis_generation
from threetears.evals.kernel.host import EvalHost
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.toyhost_memo import MODEL
from threetears.evals.analysis.plan_prompt import EVAL_REPORT_PLAN_DEFAULT
from threetears.evals.kernel.prompts.catalogue import REPORT_PLAN_PROMPT
from packages.evals.tests.toyhost_plan import PlanWriter
from packages.evals.tests.off_loop import off_loop


def _corpus_host(clients=None) -> tuple[EvalHost, str, str]:
    """The toy host over the engine's own storage, holding the corpus campaign and what it observed."""
    host = toyhost_host(clients=clients)
    campaign, corpus = toyhost_campaign()
    for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
        host.storage.save_eval_run(run)
        for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
            host.storage.save_eval_result(result)
    host.storage.save_campaign(campaign)
    return host, campaign.id, campaign.scope_id


async def _prompt() -> str:
    return EVAL_REPORT_PLAN_DEFAULT


async def test_a_generation_builds_its_client_from_the_hosts_factory_and_stores_through_the_host():
    asked: list[tuple[str, str | None]] = []
    built: list[PlanWriter] = []

    def clients(role, model, *, temperature=None):
        asked.append((role, model))
        # The plan is written over the report input the generation built, so it is composed lazily:
        # the factory is called after assembly, and the prepared generation carries that input.
        client = PlanWriter()
        built.append(client)
        return client

    host, campaign_id, scope_id = off_loop(_corpus_host, clients)

    prepared = await prepare_analysis_generation(
        host, campaign_id, scope_id, model=None, resolve_prompt=_prompt, out_of_run_cap_usd=None
    )
    built[0].report_input = prepared.report_input
    analysis, _insights = await run_analysis_generation(
        host, prepared, prompt_id=REPORT_PLAN_PROMPT, max_output_tokens=4000
    )

    assert asked == [("analysis", None)], "the generator is the host's analysis role, at its default model"
    assert prepared.resolved_model == MODEL, "the model recorded is the one the built client names"
    assert off_loop(host.storage.load_analysis, analysis.id, scope_id) is not None
    assert built[0].closed == 1, "the service releases the client it built"


async def test_a_host_with_no_client_factory_is_refused_before_anything_is_built():
    host, campaign_id, scope_id = off_loop(_corpus_host)

    with pytest.raises(ValueError, match="an analysis generation calls a model"):
        await prepare_analysis_generation(
            host, campaign_id, scope_id, model=None, resolve_prompt=_prompt, out_of_run_cap_usd=None
        )
