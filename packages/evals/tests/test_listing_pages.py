"""The run and campaign listings read a page at a time, and a caller reading every page reads every row once (#1145).

``runs_list`` and ``campaigns_list`` read and hydrated every row of the scope on every call, so their cost grew with
the store. Each now reads one page, newest first by ``(created_at, id)``, and answers with ``next``:

- **Every page, every row once.** Passing ``next`` back as ``cursor`` until it is ``None`` reads every run (every
  campaign) once, newest first, ties by id; the default page is 100 rows.
- **Only a labelled page is short, and it says so.** The status and archive filters are in the read, so a default
  page is full however many recent runs are archived; a document without ``archived`` reads as active. The label
  filter runs over the page after the read, so a labelled page can list fewer runs than it read: ``filtered`` counts
  them, and ``next`` is set while rows remain.
- **What an exploratory campaign feeds is not paged.** It is read among every campaign of the scope, so a declared
  campaign on another page is still named.
- **A cursor is the listing's own, under its filters.** One that does not decode, came from the other listing, or is
  read back with other filters, is refused as input.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.list_pages import LIST_PAGE_SIZE, decode_list_cursor, encode_list_cursor
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import CampaignDefinition, campaign_create, campaigns_list, runs_list
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SUBJECT
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture

#: One second apart, older than any run the fixture writes, three of them tied at one time.
_TIMES = [f"2020-01-01T00:00:{second:02d}+00:00" for second in (5, 4, 4, 4, 3, 2, 1, 0)]


def _seed(host: EvalHost, **fields: Any) -> list[str]:
    """Write one run per time in :data:`_TIMES`, ``fields`` on every one; return their ids, newest first, ties by id."""
    runs = [make_eval_run(scope_id=TOYHOST_SCOPE, created_at=at, **fields) for at in _TIMES]
    for run in runs:
        host.storage.save_eval_run(run)
    return [run.id for run in sorted(runs, key=lambda run: (run.created_at, run.id), reverse=True)]


def _every_page(host: EvalHost, **arguments: Any) -> list[Any]:
    """Every page of ``runs_list``, each read with the ``next`` of the one before, until ``next`` is None."""
    pages = [runs_list(host, TOYHOST_SCOPE, **arguments)]
    while pages[-1].next is not None:
        assert len(pages) < 50, "the walk never ends"
        pages.append(runs_list(host, TOYHOST_SCOPE, cursor=pages[-1].next, **arguments))
    return pages


def test_every_page_read_by_its_cursor_reads_every_run_once_newest_first() -> None:
    host = ops_fixture().host.eval_host
    seeded = _seed(host)
    whole = runs_list(host, TOYHOST_SCOPE)
    assert whole.next is None and whole.filtered == 0

    pages = _every_page(host, limit=3)

    listed = [line.id for page in pages for line in page.runs]
    assert listed == [line.id for line in whole.runs], "the walk is the one page, in its order"
    assert listed[-len(seeded) :] == seeded, "newest first, the three tied runs by id"
    assert [len(page.runs) for page in pages[:-1]] == [3] * (len(pages) - 1)
    assert pages[-1].next is None and 1 <= len(pages[-1].runs) <= 3


def test_the_default_page_is_a_hundred_runs_and_names_the_next() -> None:
    host = ops_fixture().host.eval_host
    stored = len(runs_list(host, TOYHOST_SCOPE).runs)
    for at in range(LIST_PAGE_SIZE + 1 - stored):
        host.storage.save_eval_run(
            make_eval_run(scope_id=TOYHOST_SCOPE, created_at=f"2019-01-01T00:{at // 60:02d}:{at % 60:02d}+00:00")
        )

    first = runs_list(host, TOYHOST_SCOPE)
    assert len(first.runs) == LIST_PAGE_SIZE and first.next is not None
    second = runs_list(host, TOYHOST_SCOPE, cursor=first.next)
    assert len(second.runs) == 1 and second.next is None
    assert {line.id for line in first.runs}.isdisjoint({line.id for line in second.runs})


def test_a_label_filtered_page_is_short_says_so_and_still_names_the_next() -> None:
    host = ops_fixture().host.eval_host
    labelled: set[str] = set()
    for turn, at in enumerate(_TIMES):
        run = make_eval_run(scope_id=TOYHOST_SCOPE, created_at=at, labels=["keep"] if turn % 3 == 0 else [])
        host.storage.save_eval_run(run)
        if turn % 3 == 0:
            labelled.add(run.id)

    pages = _every_page(host, label="keep", limit=4)

    assert {line.id for page in pages for line in page.runs} == labelled
    short = [page for page in pages if page.next is not None and len(page.runs) < 4]
    assert short, "a page the label narrowed came back short with more to read"
    assert all(page.filtered == 4 - len(page.runs) - page.unreadable for page in short)


def test_archived_runs_newer_than_every_active_one_never_shorten_a_default_page() -> None:
    host = ops_fixture().host.eval_host
    active = _seed(host)
    archived = [
        make_eval_run(scope_id=TOYHOST_SCOPE, created_at=f"2030-01-01T00:00:{second:02d}+00:00", archived=True)
        for second in range(12)
    ]
    for run in archived:
        host.storage.save_eval_run(run)

    pages = _every_page(host, limit=5)

    listed = [line.id for page in pages for line in page.runs]
    assert not set(listed) & {run.id for run in archived}
    assert set(active) <= set(listed)
    assert all(len(page.runs) == 5 for page in pages[:-1]) and all(page.filtered == 0 for page in pages)
    shown = _every_page(host, limit=5, include_archived=True)
    assert {run.id for run in archived} <= {line.id for page in shown for line in page.runs}
    assert sum(page.filtered for page in shown) == 0


def test_a_run_and_a_campaign_stored_without_the_archived_key_are_listed_by_default() -> None:
    store = InMemoryDocumentStore()
    fixture = ops_fixture(store=store)
    host = fixture.host.eval_host
    run = make_eval_run(scope_id=TOYHOST_SCOPE, created_at="2020-01-01T00:00:00+00:00")
    host.storage.save_eval_run(run)
    for doc_id in (run.id, fixture.campaign.id):
        stored = store.get(doc_id, TOYHOST_SCOPE)
        assert stored is not None and stored.pop("archived") is False
        store.upsert(stored)

    assert run.id in {line.id for page in _every_page(host) for line in page.runs}
    assert fixture.campaign.id in {line.id for line in campaigns_list(host, TOYHOST_SCOPE, archived=False).campaigns}
    assert fixture.campaign.id not in {line.id for line in campaigns_list(host, TOYHOST_SCOPE, archived=True).campaigns}


def test_a_cursor_that_does_not_decode_or_is_another_listings_is_refused() -> None:
    host = ops_fixture().host.eval_host
    with pytest.raises(ValidationFailedError) as garbled:
        runs_list(host, TOYHOST_SCOPE, cursor="not-a-cursor")
    assert tuple(garbled.value.parameters) == ("cursor",)
    with pytest.raises(ValidationFailedError, match="campaigns_list"):
        runs_list(
            host,
            TOYHOST_SCOPE,
            cursor=encode_list_cursor("campaigns_list", {"archived": False}, "2020-01-01T00:00:00+00:00", "c"),
        )
    with pytest.raises(ValidationFailedError):
        runs_list(host, TOYHOST_SCOPE, limit=0)


def test_a_cursor_read_back_with_other_filters_is_refused_naming_them() -> None:
    host = ops_fixture().host.eval_host
    _seed(host)
    first = runs_list(host, TOYHOST_SCOPE, limit=2)
    assert first.next is not None
    assert runs_list(host, TOYHOST_SCOPE, limit=3, cursor=first.next, status="all").runs, "page size and 'all' ride"
    for changed, filters in (
        ("status", {"status": "completed"}),
        ("label", {"label": "keep"}),
        ("include_archived", {"include_archived": True}),
    ):
        with pytest.raises(ValidationFailedError, match="other filters") as refused:
            runs_list(host, TOYHOST_SCOPE, limit=2, cursor=first.next, **filters)
        assert changed in refused.value.parameters
    campaigns = campaigns_list(host, TOYHOST_SCOPE, limit=1, archived=None)
    if campaigns.next is not None:
        with pytest.raises(ValidationFailedError, match="other filters"):
            campaigns_list(host, TOYHOST_SCOPE, limit=1, cursor=campaigns.next, archived=False)


def test_a_cursor_round_trips_the_place_it_names() -> None:
    for value in ("2020-01-01T00:00:00+00:00", None, 3, "ünïcode"):
        cursor = encode_list_cursor("runs_list", {"label": "é", "status": None}, value, "run-é")
        assert decode_list_cursor("runs_list", {"status": None, "label": "é"}, cursor) == (value, "run-é")


def _sweep(host: EvalHost, name: str, run_ids: list[str]) -> str:
    """An exploratory campaign holding ``run_ids``."""
    return campaign_create(
        host,
        CampaignDefinition(
            name=name,
            subject_id=TOYHOST_SUBJECT.subject_id,
            behavior="speed",
            aspects=[{"label": "speed", "run_ids": run_ids}],
            exploratory=True,
        ),
        TOYHOST_SCOPE,
        created_by=CALLER.identity,
    ).id


def test_campaign_pages_read_every_campaign_once_and_feeds_reach_across_pages() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    sweeps = [_sweep(host, f"sweep {at}", fixture.campaign.run_ids) for at in range(3)]

    first = campaigns_list(host, TOYHOST_SCOPE, limit=1)
    pages = [first]
    while pages[-1].next is not None:
        assert len(pages) < 10, "the walk never ends"
        pages.append(campaigns_list(host, TOYHOST_SCOPE, limit=1, cursor=pages[-1].next))

    listed = [line for page in pages for line in page.campaigns]
    assert [line.id for line in listed] == [line.id for line in campaigns_list(host, TOYHOST_SCOPE).campaigns]
    assert {line.id for line in listed} == {*sweeps, fixture.campaign.id}
    assert all(len(page.campaigns) == 1 for page in pages)
    for line in listed:
        if line.id in sweeps:
            assert line.feeds == [fixture.campaign.id], "the declared campaign it feeds is on another page"
    with pytest.raises(ValidationFailedError, match="runs_list"):
        campaigns_list(host, TOYHOST_SCOPE, cursor=encode_list_cursor("runs_list", {}, None, "r"))
