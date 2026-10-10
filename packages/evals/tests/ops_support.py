"""An operations host over the toy host: its template to launch, its corpus campaign to analyse, a writer to analyse it.

Shared by ``test_action_catalogue.py`` and ``test_fastmcp_transport.py`` — harness support, not part of
the toy host. The one non-production part is the generator model: a fixtured client whose memo is
written over the campaign's bundle as the generation assembled it, so a generation started through an
action — which assembles its own bundle — stores a real analysis.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from threetears.evals.actions import Caller
from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.kernel import EvalCampaign, EvalStorage
from threetears.evals.kernel.host import EvalHost, HostProfile
from threetears.evals.ops import AnalysisGeneration, JobStatus, OpsHost, job_poll
from threetears.evals.run import LaunchSettings
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.toyhost_memo import (
    FIXTURED_CALL_CEILING_USD,
    PROMPT,
    PROMPT_ID,
    FixturedClient,
    memo_payload,
)

__all__ = [
    "CALLER",
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


class MemoWriter(FixturedClient):
    """The fixtured generator, writing the toy memo over the bundle the generation is about to send."""

    def __init__(self, host: EvalHost, campaign: EvalCampaign) -> None:
        """Bind the writer to the campaign it analyses.

        Args:
            host: Where the campaign's runs are read.
            campaign: The campaign the generation is over.
        """
        super().__init__("")
        self._host = host
        self._campaign = campaign
        #: Set to hold the generation open until the test releases it.
        self.gate: asyncio.Event | None = None
        #: When set, what the FIRST call returns in place of the memo — an output the generator refuses,
        #: so the generation buys its repair round-trip.
        self.first_output: str | None = None

    async def generate(self, *, system: str, user: str, response_format: Any = None, tools: Any = None) -> Any:
        """Write the memo over the campaign's bundle, then hand it back as the completion."""
        if self.gate is not None:
            await self.gate.wait()
        if self.first_output is not None and not self.calls:
            self.completion.content = self.first_output
            return await super().generate(system=system, user=user, response_format=response_format, tools=tools)
        bundle = assemble_context_bundle(self._campaign, storage=self._host.storage, profile=self._host.profile)
        self.completion.content = json.dumps(memo_payload(bundle))
        return await super().generate(system=system, user=user, response_format=response_format, tools=tools)


class OpsFixture:
    """The operations host, the corpus campaign it can analyse, and the writers it built."""

    def __init__(self, host: OpsHost, campaign: EvalCampaign, writers: list[MemoWriter]) -> None:
        """Hold the pieces a test reads."""
        self.host = host
        self.campaign = campaign
        self.writers = writers


def ops_fixture(
    *,
    generation: bool = True,
    gate: asyncio.Event | None = None,
    ceiling_usd: float | None = FIXTURED_CALL_CEILING_USD,
    first_output: str | None = None,
    settings: LaunchSettings = TOYHOST_LAUNCH_SETTINGS,
    profile: HostProfile | None = None,
) -> OpsFixture:
    """The toy host as an operations host: its template saved, its corpus campaign stored.

    Args:
        generation: Whether the host generates analyses (its :class:`AnalysisGeneration`).
        gate: Holds every generation open until set, for a test that needs one running.
        ceiling_usd: What each generator call is priced at, at most; ``None`` for a writer that cannot price.
        first_output: What each writer's first call returns in place of the memo, when set.
        settings: The launch settings — the run and out-of-run caps, and whether they are enforced.
        profile: The host's vocabulary; ``None`` is the toy host's own.

    Returns:
        The fixture.
    """
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    campaign, corpus = toyhost_campaign()
    for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
        storage.save_eval_run(run)
        for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
            storage.save_eval_result(result)
    storage.save_campaign(campaign)
    writers: list[MemoWriter] = []
    hosts: list[EvalHost] = []

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> MemoWriter:
        writer = MemoWriter(hosts[0], campaign)
        writer.gate = gate
        writer.ceiling_usd = ceiling_usd
        writer.first_output = first_output
        writers.append(writer)
        return writer

    launch, _client = toyhost_launch_host(profile=profile, storage=storage, clients=clients, settings=lambda: settings)
    hosts.append(launch.eval_host)

    async def prompt() -> str:
        return PROMPT

    generating = AnalysisGeneration(prompt_id=PROMPT_ID, resolve_prompt=prompt, max_output_tokens=4000, budget_s=60.0)

    host = OpsHost(launch=launch, generation=generating if generation else None)
    return OpsFixture(host, campaign, writers)


async def settled(host: OpsHost, job_id: str, scope_id: str = TOYHOST_SCOPE) -> JobStatus:
    """Poll a job until it is done, failing after a few seconds."""
    async with asyncio.timeout(10):
        while not (status := await job_poll(host, job_id, scope_id)).done:
            await asyncio.sleep(0.01)
    return status
