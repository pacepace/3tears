"""A campaign's report reads one analysis, however many the campaign has archived (#951).

Each stored analysis holds every aspect's frozen bundle, so a read that hydrated them all to show the newest live one
cost memory and time in proportion to the campaign's history: one read of an eight-aspect campaign cost 400–500 MB in
production. The store picks the reported analysis (:meth:`~threetears.evals.kernel.storage.EvalStorage.load_reported_analysis`)
and only that one document is shipped and read.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from threetears.evals.analysis import campaign_report, record_campaign_call, serialize_report
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.ops import analysis_generate
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.ops_support import TOYHOST_SCOPE, OpsFixture, ops_fixture, settled
from packages.evals.tests.off_loop import off_loop

#: When every archived copy was written: before the generated analysis, so the live one is the newest.
_EARLIER = "2000-01-01T00:00:{second:02d}+00:00"


class _CountingStore(InMemoryDocumentStore):
    """The in-memory reference store, counting every analysis document a read hands back."""

    def __init__(self) -> None:
        super().__init__()
        self.analyses_read = 0

    def _counted(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.analyses_read += sum(document.get("doc_type") == "eval_analysis" for document in documents)
        return documents

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        document = super().get(doc_id, scope_id)
        return None if document is None else self._counted([document])[0]

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        document, etag = super().get_with_etag(doc_id, scope_id)
        return (None if document is None else self._counted([document])[0]), etag

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        return self._counted(super().get_many(doc_type, doc_ids, scope_id, exclude=exclude, keep=keep))

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        return self._counted(
            super().by_doc_type(
                doc_type,
                scope_id,
                order_by=order_by,
                descending=descending,
                limit=limit,
                exclude=exclude,
                field_in=field_in,
                **field_eq,
            )
        )


async def _generated(store: InMemoryDocumentStore) -> tuple[OpsFixture, EvalAnalysis]:
    """The toy campaign over ``store``, and the one analysis generated of it."""
    fixture = off_loop(ops_fixture, store=store)
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    assert (await settled(fixture.host, job.job_id)).state == "completed"
    (analysis,) = off_loop(fixture.host.eval_host.storage.list_analyses_by_campaign, fixture.campaign.id, TOYHOST_SCOPE)
    return fixture, analysis


def _archive_copies(fixture: OpsFixture, analysis: EvalAnalysis, count: int) -> None:
    """Store ``count`` archived copies of ``analysis``, each written before it."""
    storage = fixture.host.eval_host.storage
    for index in range(count):
        copy = analysis.model_copy(
            update={
                "id": f"archived-{index}",
                "archived": True,
                "archived_reason": "superseded",
                "created_at": _EARLIER.format(second=index),
            }
        )
        off_loop(storage.save_analysis, copy)


async def test_the_report_reads_one_analysis_document_however_many_are_archived() -> None:
    store = _CountingStore()
    fixture, analysis = await _generated(store)
    _archive_copies(fixture, analysis, 6)
    host = fixture.host.eval_host

    store.analyses_read = 0
    page = off_loop(campaign_report, host, fixture.campaign.id, TOYHOST_SCOPE)

    assert (page.basis, page.analysis_id) == ("analysis", analysis.id)
    assert store.analyses_read == 1


async def test_the_reported_analysis_is_the_newest_live_one_whatever_is_archived_after_it() -> None:
    fixture, analysis = await _generated(InMemoryDocumentStore())
    storage = fixture.host.eval_host.storage
    older = analysis.model_copy(update={"id": "older-live", "created_at": _EARLIER.format(second=0)})
    newer = analysis.model_copy(
        update={
            "id": "newer-archived",
            "archived": True,
            "archived_reason": "wrong",
            "created_at": "2999-01-01T00:00:00+00:00",
        }
    )
    for stored in (older, newer):
        off_loop(storage.save_analysis, stored)

    reported = off_loop(storage.load_reported_analysis, fixture.campaign.id, TOYHOST_SCOPE)
    live = [
        one
        for one in off_loop(storage.list_analyses_by_campaign, fixture.campaign.id, TOYHOST_SCOPE)
        if not one.archived
    ]

    assert reported is not None and reported.id == analysis.id == live[0].id
    off_loop(storage.save_analysis, analysis.model_copy(update={"archived": True, "archived_reason": "wrong"}))
    reported = off_loop(storage.load_reported_analysis, fixture.campaign.id, TOYHOST_SCOPE)
    assert reported is not None and reported.id == "older-live"
    off_loop(storage.save_analysis, older.model_copy(update={"archived": True, "archived_reason": "wrong"}))
    assert off_loop(storage.load_reported_analysis, fixture.campaign.id, TOYHOST_SCOPE) is None
    assert off_loop(campaign_report, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).basis == "code_only"
    assert off_loop(storage.load_reported_analysis, "no-such-campaign", TOYHOST_SCOPE) is None


async def test_a_call_naming_an_analysis_that_is_not_the_report_is_refused_by_the_one_rule() -> None:
    fixture, analysis = await _generated(InMemoryDocumentStore())
    storage = fixture.host.eval_host.storage
    older = analysis.model_copy(update={"id": "older-live", "created_at": _EARLIER.format(second=0)})
    off_loop(storage.save_analysis, older)

    with pytest.raises(ValidationFailedError, match=f"its page reports {analysis.id}"):
        off_loop(
            record_campaign_call,
            fixture.host.eval_host,
            fixture.campaign.id,
            TOYHOST_SCOPE,
            arm="nope",
            called_by="tester",
            analysis_id="older-live",
        )


def _peak(read: Callable[[], object]) -> int:
    """The peak memory ``read`` allocates above what was held before it, in bytes."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        read()
        return tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()


async def test_a_reads_peak_memory_does_not_grow_with_the_analyses_archived() -> None:
    fixture, analysis = await _generated(InMemoryDocumentStore())
    host = fixture.host.eval_host

    def read() -> str:
        return serialize_report(campaign_report(host, fixture.campaign.id, TOYHOST_SCOPE), "html")

    off_loop(read)
    alone = off_loop(_peak, read)
    _archive_copies(fixture, analysis, 24)
    beside_history = off_loop(_peak, read)

    # Hydrating two dozen archived analyses held each one's bundles at once: several times the read's own peak.
    assert beside_history < alone * 1.25, (alone, beside_history)
