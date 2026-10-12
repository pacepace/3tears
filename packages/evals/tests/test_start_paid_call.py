"""``start_paid_call``: one paid model call, admitted against its cap in the call and made in a job.

- **The happy path** admits the call in the call, makes it in the job through the budget, and ``job_poll`` returns
  what the work returned; the ledger holds the one call.
- **A call the cap cannot pay for is refused in the call**: nothing is started, nothing is spent, and the client
  built for it is released.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.kernel import AdmittedCall, OutOfRunBudget, PlannedCall, ValidationFailedError
from threetears.evals.ops import await_job, start_paid_call
from threetears.evals.run import EvalActionJob, run_blocking
from threetears.evals.schema import BoundCompletionClient
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.testing import CompletionGuard
from packages.evals.tests.jobs_conformance_support import CONFORMANCE_SCOPE, ScriptedClients, conformance_host
from packages.evals.tests.off_loop import off_loop

_CALL = PlannedCall(system="Draft a rubric.", user="the persona")


class _Draft(EvalBaseModel):
    """A drafting action's result."""

    text: str


class _ReleaseTracking:
    """A built client, passed through, that records its own release."""

    def __init__(self, inner: Any, released: list[str]) -> None:
        self._inner = inner
        self._released = released
        self.model_name: str = inner.model_name

    def price_ceiling(self, *, system: str, user: str, response_format: Any = None) -> float | None:
        return self._inner.price_ceiling(system=system, user=user, response_format=response_format)

    async def generate(self, *, system: str, user: str, response_format: Any = None, **kwargs: Any) -> Any:
        return await self._inner.generate(system=system, user=user, response_format=response_format, **kwargs)

    async def aclose(self) -> None:
        self._released.append(self.model_name)
        await self._inner.aclose()


class _ReleaseTrackingClients(ScriptedClients):
    """The scripted factory, every client it builds recording its release."""

    def __init__(self) -> None:
        super().__init__()
        self.released: list[str] = []

    def __call__(self, role: Any, model: str | None, *, temperature: float | None = None) -> Any:
        return _ReleaseTracking(super().__call__(role, model, temperature=temperature), self.released)


def _budget(host: Any, cap_usd: float) -> OutOfRunBudget:
    eval_host = host.eval_host
    return OutOfRunBudget(eval_host.storage, CONFORMANCE_SCOPE, cap_usd, blocking_executor=eval_host.blocking_executor)


async def test_the_call_is_admitted_in_the_call_and_made_in_the_job() -> None:
    clients = _ReleaseTrackingClients()
    built = off_loop(conformance_host, CompletionGuard(), clients)
    budget = _budget(built.host, cap_usd=1.0)

    async def make(client: BoundCompletionClient, admitted: AdmittedCall) -> _Draft:
        try:
            await budget.generate(client, admitted)
        finally:
            await client.aclose()
        return _Draft(text=f"drafted on {client.model_name}")

    started = await start_paid_call(
        built.host,
        CONFORMANCE_SCOPE,
        action="rubric_propose",
        role="proposer",
        model="proposer/m",
        purpose="proposer",
        budget=budget,
        call=_CALL,
        make=make,
        label="a capability rubric",
    )
    assert budget.committed_usd == pytest.approx(0.01), "admitted in the call, before the job ran"
    (done,) = await await_job(built.host, started, CONFORMANCE_SCOPE)

    assert (done.state, done.result) == ("completed", {"text": "drafted on proposer/m"})
    assert [row.purpose for row in budget.recorded] == ["proposer"]
    assert clients.released == ["proposer/m"]
    record, _ = await run_blocking(
        None,
        built.host.eval_host.storage.load_job_record,
        EvalActionJob,
        done.job_id.split(":", 1)[1],
        CONFORMANCE_SCOPE,
    )
    assert record is not None and record.label == "a capability rubric"


async def test_a_call_the_cap_cannot_pay_for_starts_nothing_and_releases_its_client() -> None:
    clients = _ReleaseTrackingClients()
    store = InMemoryDocumentStore()
    built = off_loop(conformance_host, CompletionGuard(), clients, store=store)
    budget = _budget(built.host, cap_usd=0.001)

    async def make(client: BoundCompletionClient, admitted: AdmittedCall) -> _Draft:
        raise AssertionError("a refused call is never made")

    with pytest.raises(ValidationFailedError):
        await start_paid_call(
            built.host,
            CONFORMANCE_SCOPE,
            action="rubric_propose",
            role="proposer",
            model="proposer/m",
            purpose="proposer",
            budget=budget,
            call=_CALL,
            make=make,
            label="a capability rubric",
        )

    assert budget.committed_usd == 0
    assert budget.recorded == ()
    assert clients.released == ["proposer/m"]
    assert not [
        doc for doc in store.documents.values() if doc.get("doc_type") == EvalActionJob.model_fields["doc_type"].default
    ]
