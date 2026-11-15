"""the operator's listing and its count, moved onto keyed helpers, give the rows the hand-written statements did.

:meth:`~threetears.scheduled_jobs.collections.ScheduledJobCollection.list_jobs` was ``LIMIT``/``OFFSET`` over every
schedule and :meth:`~threetears.scheduled_jobs.collections.ScheduledJobCollection.count_jobs` a ``count(*)`` of them;
each is now a keyed page (``keyset_page`` from a cursor) and a bounded count (one keyset page of the sort columns, its cap plus one) over
``idx_scheduled_jobs_kind_name_key`` (v007). Each case freezes the statement the collection sent before (``old_sql``)
and runs the converted method on the same rows through
:func:`~threetears.core.testing.same_results.assert_same_results`: empty, one row, many, NULL names, ties in the sort,
two kinds, a page boundary and one past it.

The paging case reads every page by cursor and compares the pages, joined, with the old statement read whole: no row
repeated, none skipped, in the old order, NULL names last, ties broken by ``job_id``.

Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.same_results import Seeded, assert_same_results
from threetears.scheduled_jobs.collections import ScheduledJobCollection
from threetears.scheduled_jobs.migrations import register as register_scheduled_jobs

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration

_NS = uuid.UUID("0f4a7c2e-6b1d-4c55-9e3a-2b7d8f1a5c90")
_T0 = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
_STATUSES = ("active", "paused", "active", "expired")
#: the page a listing case sits its boundary at; a multiple of the ties' run of three
_PAGE = 6

_COLUMNS = (
    "partition_key",
    "job_id",
    "kind",
    "payload",
    "schedule_type",
    "schedule_config",
    "status",
    "next_fire_at",
    "last_fired_at",
    "missed_fire_policy",
    "name",
    "date_created",
    "date_updated",
)
_SELECT = ", ".join(_COLUMNS)
_FILTERS = "WHERE ($1::text[] IS NULL OR kind = ANY($1)) AND ($2::text[] IS NULL OR status = ANY($2)) "
_ORDER = "ORDER BY kind ASC, name ASC NULLS LAST, job_id ASC "
#: the listing as list_jobs sent it before (one page, by offset), frozen
_OLD_PAGE = f"SELECT {_SELECT} FROM scheduled_jobs {_FILTERS}{_ORDER}LIMIT $3 OFFSET $4"  # noqa: S608 - test text
#: the same listing read whole: what every page, joined, must be
_OLD_WHOLE = f"SELECT {_SELECT} FROM scheduled_jobs {_FILTERS}{_ORDER}"  # noqa: S608 - test text
#: the count as count_jobs sent it before, frozen, and capped as the bounded count answers it
_OLD_COUNT = f"SELECT count(*) FROM scheduled_jobs {_FILTERS}"  # noqa: S608 - test text


def _job(seed: Seeded) -> Mapping[str, Any]:
    """one schedule: the kind by tenant (so a kind filter crosses the boundary), its name the sort's tie (NULL when
    ``seed.null``), each in its own partition as real seeders write them."""
    return {
        "partition_key": uuid.uuid5(_NS, f"partition-{seed.index}"),
        "job_id": uuid.uuid5(_NS, f"job-{seed.case}-{seed.index}"),
        "kind": f"kind_{seed.tenant}",
        "payload": json.dumps({"n": seed.index}),
        "schedule_type": "interval",
        "schedule_config": json.dumps({"seconds": 60 + seed.index}),
        "status": _STATUSES[seed.index % 4],
        "next_fire_at": None if seed.null else _T0 + timedelta(minutes=seed.index),
        "last_fired_at": None,
        "missed_fire_policy": "coalesce",
        "name": None if seed.null else f"job {seed.tie:03d}",
        "date_created": _T0,
        "date_updated": _T0,
    }


def _jobs(conn: Any) -> ScheduledJobCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=conn)
    return ScheduledJobCollection(
        registry=registry,
        config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
        nats_client=None,
    )


def _rows(entities: Sequence[Any]) -> list[dict[str, Any]]:
    return [{column: entity.to_dict()[column] for column in _COLUMNS} for entity in entities]


def _first_page(
    kinds: Sequence[str] | None, statuses: Sequence[str] | None
) -> Callable[[Any], Awaitable[list[dict[str, Any]]]]:
    async def helper(conn: Any) -> list[dict[str, Any]]:
        page = await _jobs(conn).list_jobs(kinds=kinds, statuses=statuses, limit=_PAGE)
        return _rows(page.jobs)

    return helper


def _every_page(
    kinds: Sequence[str] | None, statuses: Sequence[str] | None
) -> Callable[[Any], Awaitable[list[dict[str, Any]]]]:
    """every page, read by the cursor each page answered, joined: a page of 2 so every case crosses pages."""

    async def helper(conn: Any) -> list[dict[str, Any]]:
        jobs = _jobs(conn)
        read: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            page = await jobs.list_jobs(kinds=kinds, statuses=statuses, limit=2, cursor=cursor)
            read.extend(_rows(page.jobs))
            cursor = page.next_cursor
            if cursor is None:
                break
        return read

    return helper


def _counted(
    kinds: Sequence[str] | None, statuses: Sequence[str] | None, at_most: int
) -> Callable[[Any], Awaitable[list[dict[str, Any]]]]:
    async def helper(conn: Any) -> list[dict[str, Any]]:
        counted = await _jobs(conn).count_jobs(kinds=kinds, statuses=statuses, at_most=at_most)
        return [{"count": counted.count, "more": counted.more}]

    return helper


@dataclass(frozen=True)
class _Site:
    old_sql: str
    params: Sequence[Any]
    helper: Callable[[Any], Awaitable[Sequence[Mapping[str, Any]]]]
    order_by: Sequence[str] | None = ("kind", "name", "job_id")
    ordered: bool | None = None


def _capped(at_most: int) -> str:
    """the old count, answered as the bounded count answers it: the count up to the cap, and whether more match."""
    return f"SELECT least(n, {at_most}) AS count, n > {at_most} AS more FROM ({_OLD_COUNT}) AS c(n)"  # noqa: S608


_SITES: dict[str, _Site] = {
    "first page, unfiltered": _Site(_OLD_PAGE, (None, None, _PAGE, 0), _first_page(None, None)),
    "first page, by kind": _Site(_OLD_PAGE, (["kind_0"], None, _PAGE, 0), _first_page(["kind_0"], None)),
    "first page, by status": _Site(
        _OLD_PAGE, (None, ["active", "expired"], _PAGE, 0), _first_page(None, ["active", "expired"])
    ),
    "first page, by kind and status": _Site(
        _OLD_PAGE, (["kind_1", "kind_0"], ["paused"], _PAGE, 0), _first_page(["kind_1", "kind_0"], ["paused"])
    ),
    "every page by cursor, unfiltered": _Site(_OLD_WHOLE, (None, None), _every_page(None, None)),
    "every page by cursor, by kind and status": _Site(
        _OLD_WHOLE,
        (
            ["kind_1"],
            ["active"],
        ),
        _every_page(["kind_1"], ["active"]),
    ),
    "count, unfiltered, under the cap": _Site(_capped(1000), (None, None), _counted(None, None, 1000), None, False),
    "count, by kind and status, under the cap": _Site(
        _capped(1000), (["kind_0"], ["active", "paused"]), _counted(["kind_0"], ["active", "paused"], 1000), None, False
    ),
    "count, unfiltered, past a cap of 5": _Site(_capped(5), (None, None), _counted(None, None, 5), None, False),
    "count, by status, past a cap of 3": _Site(
        _capped(3), (None, ["active"]), _counted(None, ["active"], 3), None, False
    ),
}


@pytest.fixture
async def conn(pg_schema: tuple[str, str]) -> AsyncIterator[asyncpg.Connection]:
    url, schema = pg_schema
    connection = await asyncpg.connect(url)
    try:
        await connection.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_scheduled_jobs(runner, held_contract=True)
        await runner.apply_for_platform_schema(AsyncpgStore(connection))
        yield connection
    finally:
        await connection.close()


@pytest.mark.parametrize("site", list(_SITES), ids=list(_SITES))
async def test_the_keyed_listing_gives_the_old_statements_rows(conn: asyncpg.Connection, site: str) -> None:
    """Flips if a page of list_jobs (the first, or every page joined by cursor) answers other rows or another order
    than the LIMIT/OFFSET statement it replaced, or count_jobs another count (or "more") than count(*) capped, on
    PostgreSQL or YugabyteDB."""
    case = _SITES[site]
    await assert_same_results(
        conn,
        table="scheduled_jobs",
        row=_job,
        old_sql=case.old_sql,
        params=case.params,
        helper=case.helper,
        order_by=case.order_by,
        ordered=case.ordered,
        page_size=_PAGE,
    )
