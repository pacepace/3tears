"""A campaign's analysis listing reads its analyses a page at a time, never every one at once (#954).

Discodon's web process restarted on 2026-11-04 when ``analyses_list`` hydrated every analysis of a campaign at once:
each carries every aspect's frozen bundle, and a campaign keeps every archived one. A line's verdict and title are
written by code from that frozen bundle, so the listing still reads each analysis whole, one page at a time
(:meth:`~threetears.evals.kernel.storage.EvalStorage.iter_analyses_by_campaign`).

Contracts held here, on the memory store and the SQLite store alike:

- **one key read, then pages**: the key read ships no bundle, and no read ships more than
  :data:`~threetears.evals.kernel.storage.ANALYSES_PAGE` whole analyses; each analysis is read whole once;
- **same listing**: every analysis is listed, newest first, as the whole read lists them;
- **bounded heap**: doubling each analysis's body raises the listing's peak heap by less than
  :data:`MAX_GROWTH_SHARE` of what it added across the campaign. A listing holding every analysis at once raises it by
  more than all of it;
- ``analyses_undescribable`` walks each campaign the same way.
"""

from __future__ import annotations

from threetears.evals.analysis.report.campaign.memo_blocks import retired_generation

import gc
import tracemalloc
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.campaign import RunIndexEntry
from threetears.evals.kernel.storage import ANALYSES_PAGE
from threetears.evals.ops import analyses_list, analyses_undescribable
from threetears.evals.schema.store_port import DocumentStore
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.storage.sqlite import SqliteDocumentStore
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import analysed_aspect, empty_bundle, make_analysis, make_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host

SCOPE = "uni-1"
CAMPAIGN = "campaign-paged"
#: The analyses the campaign holds: far more than a page.
COUNT = 200
#: Each analysis's body at the lower volume, in bytes; the higher volume is twice it.
BODY_BYTES = 16 * 1024
#: Doubling each body may raise the listing's peak heap by at most this share of what it added across the campaign
#: (``COUNT * BODY_BYTES``). Reading a page at a time raises it by a few pages' bodies; holding every analysis raises it
#: by more than the whole of it.
MAX_GROWTH_SHARE = 0.25


class _ReadsCounted:
    """A document store that records, per read, how many whole analyses (those carrying ``aspects``) it shipped."""

    def __init__(self, inner: DocumentStore) -> None:
        self.inner = inner
        self.reads: list[tuple[str, str, int]] = []

    def _count(self, method: str, doc_type: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.reads.append((method, doc_type, sum(1 for row in rows if "aspects" in row)))
        return rows

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        found = self.inner.get(doc_id, scope_id)
        self._count("get", "" if found is None else str(found.get("doc_type")), [] if found is None else [found])
        return found

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        return self.inner.get_with_etag(doc_id, scope_id)

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        return self._count(
            "get_many", doc_type, self.inner.get_many(doc_type, doc_ids, scope_id, exclude=exclude, keep=keep)
        )

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
        rows = self.inner.by_doc_type(
            doc_type,
            scope_id,
            order_by=order_by,
            descending=descending,
            limit=limit,
            exclude=exclude,
            field_in=field_in,
            **field_eq,
        )
        return self._count("by_doc_type", doc_type, rows)

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        return self.inner.iter_by_doc_type(doc_type, scope_id)

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        self.inner.upsert(document, if_match=if_match, create_only=create_only)

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        return self.inner.merge_fields(doc_id, scope_id, fields)

    def delete(self, doc_id: str, scope_id: str) -> bool:
        return self.inner.delete(doc_id, scope_id)


def _memory(_tmp_path: Path, _name: str) -> DocumentStore:
    return InMemoryDocumentStore()


def _sqlite(tmp_path: Path, name: str) -> DocumentStore:
    return SqliteDocumentStore(tmp_path / f"{name}.sqlite3")


STORES = [pytest.param(_memory, id="memory"), pytest.param(_sqlite, id="sqlite")]


def _stored(store: DocumentStore, *, body_bytes: int) -> tuple[_ReadsCounted, list[str]]:
    """``COUNT`` analyses of one campaign, each padded to ``body_bytes`` inside its aspect, stored in ``store``; and
    their ids, newest first."""
    counted = _ReadsCounted(store)
    storage = EvalStorage(counted)
    bundle = empty_bundle(campaign_id=CAMPAIGN, scope_id=SCOPE)
    frame = campaign_bundle(bundle).frame()
    ids: list[str] = []
    for index in range(COUNT):
        padding = f"body-{index}-".ljust(body_bytes, "x")
        aspect = analysed_aspect(bundle, run_index=[RunIndexEntry(run_id=f"run-{index}", key_metrics={"pad": padding})])
        analysis = make_analysis(
            id=f"analysis-{index:03d}",
            campaign_id=CAMPAIGN,
            scope_id=SCOPE,
            aspects=[aspect],
            campaign_bundle=frame,
            created_at=f"2026-11-01T{index // 60:02d}:{index % 60:02d}:00+00:00",
        )
        storage.save_analysis(analysis)
        ids.append(analysis.id)
    return counted, ids[::-1]


@pytest.mark.parametrize("make_store", STORES)
def test_the_listing_reads_one_key_read_then_a_page_of_whole_analyses_at_a_time(
    make_store: Callable[[Path, str], DocumentStore], tmp_path: Path
) -> None:
    counted, newest_first = _stored(make_store(tmp_path, "listing"), body_bytes=1024)
    host = toyhost_host(storage=EvalStorage(counted))
    counted.reads.clear()

    listing = analyses_list(host, CAMPAIGN, SCOPE)

    assert [line.id for line in listing.analyses] == newest_first
    # Each is a memo, so each line leads with the memo writer's retired line, and no line adds why it did not read.
    assert {line.needs_regeneration for line in listing.analyses} == {retired_generation("memo")}
    analyses_read = [(method, whole) for method, doc_type, whole in counted.reads if doc_type == "eval_analysis"]
    assert analyses_read[0] == ("by_doc_type", 0), "the key read ships no analysis whole"
    assert all(method == "get_many" for method, _ in analyses_read[1:])
    assert max(whole for _, whole in analyses_read) == ANALYSES_PAGE
    assert sum(whole for _, whole in analyses_read) == COUNT, "each analysis is read whole once"


@pytest.mark.parametrize("make_store", STORES)
def test_the_undescribable_listing_walks_each_campaign_a_page_at_a_time(
    make_store: Callable[[Path, str], DocumentStore], tmp_path: Path
) -> None:
    counted, _ = _stored(make_store(tmp_path, "undescribable"), body_bytes=1024)
    host = toyhost_host(storage=EvalStorage(counted))
    host.storage.save_campaign(make_campaign(id=CAMPAIGN, scope_id=SCOPE))
    counted.reads.clear()

    listing = analyses_undescribable(host, SCOPE)

    assert (listing.analyses_read, listing.analyses) == (COUNT, [])
    whole = [count for _, doc_type, count in counted.reads if doc_type == "eval_analysis"]
    assert max(whole) == ANALYSES_PAGE and sum(whole) == COUNT


def _peak_bytes(make_store: Callable[[Path, str], DocumentStore], tmp_path: Path, *, body_bytes: int) -> int:
    """The Python heap's peak while one campaign's ``COUNT`` analyses, each padded to ``body_bytes``, are listed."""
    counted, newest_first = _stored(make_store(tmp_path, f"peak-{body_bytes}"), body_bytes=body_bytes)
    host = toyhost_host(storage=EvalStorage(counted.inner))
    gc.collect()
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        listing = analyses_list(host, CAMPAIGN, SCOPE)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(listing.analyses) == len(newest_first)
    return peak - baseline


@pytest.mark.parametrize("make_store", STORES)
def test_doubling_each_analysis_s_body_raises_the_listing_s_peak_by_a_few_pages_not_the_campaign(
    make_store: Callable[[Path, str], DocumentStore], tmp_path: Path
) -> None:
    lower = _peak_bytes(make_store, tmp_path, body_bytes=BODY_BYTES)
    higher = _peak_bytes(make_store, tmp_path, body_bytes=2 * BODY_BYTES)

    added = COUNT * BODY_BYTES
    assert higher - lower < MAX_GROWTH_SHARE * added, (lower, higher, added)
