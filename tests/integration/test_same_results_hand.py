"""integration: the hand conversions onto the keyed helpers read and write what the statements they replaced did.

The rewrite tool's manifest holds the point reads it rewrote; a conversion made by hand (a page, a drain, a keyed or
prefix delete) is held here, by the same equivalence harness
(:func:`threetears.core.testing.same_results.assert_same_results`): every seed case (empty, one, many, NULLs, ties, a
tenant boundary, a page boundary and one past it), the old statement against the helper call as the module now makes
it, on the session's migrated schema (``conn`` on PostgreSQL, ``yugabyte_conn`` on YugabyteDB; each skips on the
other engine).

- langgraph ``ThreeTierCheckpointSaver``: ``_l3_get_tuple``'s latest checkpoint (``ORDER BY checkpoint_id DESC
  LIMIT 1`` -> a ``keyset_page`` of one) and its pending writes (``ORDER BY idx`` -> ``read_by`` by the
  checkpoint's key, sorted by ``idx``); ``alist`` (newest first: ``keyset_rows``, a ``limit`` page through
  ``keyset_page``, ``before`` as a ``where`` fragment); ``adelete_thread`` (``delete_keyed`` by the thread);
  ``adelete_customer_threads`` (``LIKE '<prefix>%'`` -> ``delete_keyed(prefix=)``); anonymization's pages of a
  thread (past a tuple cursor -> ``keyset_page`` by its cursor) and its rewrites by key (``update_keyed``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from operator import itemgetter
from typing import Any

import pytest

from threetears.core.keyed import Fragment, Sort, delete_keyed, keyset_page, keyset_rows, read_by, update_keyed
from threetears.core.testing.same_results import Seeded, assert_same_results

#: tenant 0's thread and customer prefix; tenant 1's rows are another customer's
_THREAD = "c0/t"
_PREFIX = "c0/"
_CHECKPOINT_COLUMNS = ("checkpoint_id", "parent_checkpoint_id", "type", "checkpoint", "metadata_")


@pytest.fixture(params=["conn", "yugabyte_conn"])
def db(request: pytest.FixtureRequest) -> Any:
    """the session's migrated schema on whichever engine it is (the other skips)."""
    return request.getfixturevalue(request.param)


def _checkpoint(seed: Seeded) -> dict[str, Any]:
    return {
        "thread_id": f"c{seed.tenant}/t",
        "checkpoint_ns": "",
        "checkpoint_id": f"{seed.tie:04d}-{seed.index:04d}",
        "parent_checkpoint_id": None if seed.null else f"p{seed.index}",
        "type": "msgpack",
        "checkpoint": f"cp{seed.index}".encode(),
        "metadata_": None if seed.null else b"m",
    }


def _write(seed: Seeded) -> dict[str, Any]:
    return {
        "thread_id": f"c{seed.tenant}/t",
        "checkpoint_ns": "",
        "checkpoint_id": f"cp{seed.tie % 2}",
        "task_id": f"task{seed.index % 3}",
        "task_path": "",
        "idx": seed.index,
        "channel": "ch",
        "type": None if seed.null else "msgpack",
        "blob": f"w{seed.index}".encode(),
    }


def _deleting(write: Callable[[Any], Awaitable[Any]]) -> Callable[[Any], Awaitable[Sequence[Mapping[str, Any]]]]:
    async def run(conn: Any) -> Sequence[Mapping[str, Any]]:
        await write(conn)
        return []

    return run


async def test_a_checkpoint_s_pending_writes_are_the_rows_its_key_named_in_idx_order(db: Any) -> None:
    """Flips if ``_l3_get_tuple`` reads other writes than its ``ORDER BY idx`` statement, or in another order."""

    async def helper(conn: Any) -> list[Any]:
        found = await read_by(
            conn,
            "checkpoint_writes",
            by={"thread_id": _THREAD, "checkpoint_ns": "", "checkpoint_id": "cp0"},
            columns=("task_id", "idx", "channel", "type", "blob"),
            records=True,
        )
        return sorted(found, key=itemgetter("idx"))

    await assert_same_results(
        db,
        table="checkpoint_writes",
        row=_write,
        old_sql=(
            "SELECT task_id, idx, channel, type, blob FROM checkpoint_writes "
            "WHERE thread_id = $1 AND checkpoint_ns = $2 AND checkpoint_id = $3 ORDER BY idx"
        ),
        params=[_THREAD, "", "cp0"],
        helper=helper,
        order_by=["idx"],
    )


async def test_a_thread_s_latest_checkpoint_is_the_newest_row(db: Any) -> None:
    """Flips if ``_l3_get_tuple``'s latest read (a page of one, newest first by the key) answers another checkpoint
    than its ``ORDER BY checkpoint_id DESC LIMIT 1`` statement did."""

    async def helper(conn: Any) -> list[dict[str, Any]]:
        page = await keyset_page(
            conn,
            "checkpoints",
            columns=_CHECKPOINT_COLUMNS,
            by={"thread_id": _THREAD, "checkpoint_ns": ""},
            order=[Sort("checkpoint_id", cast=None, descending=True)],
            limit=1,
        )
        return page.rows

    sql = (
        f"SELECT {', '.join(_CHECKPOINT_COLUMNS)} FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2"
        " ORDER BY checkpoint_id DESC LIMIT 1"
    )
    await assert_same_results(
        db, table="checkpoints", row=_checkpoint, old_sql=sql, params=[_THREAD, ""], helper=helper, page_size=1
    )


@pytest.mark.parametrize("limit", [None, 3])
@pytest.mark.parametrize("before", [None, "0003-0000"])
async def test_a_thread_s_listing_is_the_rows_newest_first(db: Any, limit: int | None, before: str | None) -> None:
    """Flips if ``alist`` lists other checkpoints than its statement (``before``, ``limit``), or in another order."""
    by = {"thread_id": _THREAD, "checkpoint_ns": ""}
    where = [Fragment('t."checkpoint_id" < $1', (before,))] if before else []
    order = [Sort("checkpoint_id", cast=None, descending=True)]

    async def helper(conn: Any) -> list[dict[str, Any]]:
        if limit is None:
            return await keyset_rows(
                conn, "checkpoints", columns=_CHECKPOINT_COLUMNS, by=by, order=order, where=where, page=5
            )
        page = await keyset_page(
            conn, "checkpoints", columns=_CHECKPOINT_COLUMNS, by=by, order=order, limit=limit, where=where
        )
        return page.rows

    sql = f"SELECT {', '.join(_CHECKPOINT_COLUMNS)} FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2"
    params: list[Any] = [_THREAD, ""]
    if before:
        sql += " AND checkpoint_id < $3"
        params.append(before)
    sql += " ORDER BY checkpoint_id DESC"
    if limit is not None:
        sql += f" LIMIT {limit}"
    await assert_same_results(
        db, table="checkpoints", row=_checkpoint, old_sql=sql, params=params, helper=helper, page_size=5
    )


@pytest.mark.parametrize("table", ["checkpoints", "checkpoint_writes"])
async def test_a_thread_s_purge_deletes_the_rows_its_key_named(db: Any, table: str) -> None:
    """Flips if ``adelete_thread`` deletes other rows than its ``DELETE ... WHERE thread_id = $1``."""
    await assert_same_results(
        db,
        table=table,
        row=_checkpoint if table == "checkpoints" else _write,
        old_sql=f"DELETE FROM {table} WHERE thread_id = $1",
        params=[_THREAD],
        helper=_deleting(lambda c: delete_keyed(c, table, by={"thread_id": _THREAD})),
        state_sql=f"SELECT * FROM {table}",
    )


@pytest.mark.parametrize("table", ["checkpoints", "checkpoint_writes"])
async def test_a_customer_s_purge_deletes_the_threads_its_prefix_named(db: Any, table: str) -> None:
    """Flips if ``adelete_customer_threads`` deletes other rows than its ``LIKE '<prefix>%'``."""
    await assert_same_results(
        db,
        table=table,
        row=_checkpoint if table == "checkpoints" else _write,
        old_sql=f"DELETE FROM {table} WHERE thread_id LIKE $1",
        params=[f"{_PREFIX}%"],
        helper=_deleting(lambda c: delete_keyed(c, table, prefix={"thread_id": _PREFIX})),
        state_sql=f"SELECT * FROM {table}",
    )


#: what anonymization pages, per table: the columns it reads and its key's order past the thread
_ANONYMIZED: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "checkpoints": (
        ("checkpoint_ns", "checkpoint_id", "type", "checkpoint", "metadata_"),
        ("checkpoint_ns", "checkpoint_id"),
    ),
    "checkpoint_writes": (
        ("checkpoint_ns", "checkpoint_id", "task_id", "idx", "channel", "type", "blob"),
        ("checkpoint_ns", "checkpoint_id", "task_id", "idx"),
    ),
}


@pytest.mark.parametrize("table", ["checkpoints", "checkpoint_writes"])
async def test_an_anonymized_thread_s_pages_are_its_rows_in_key_order(db: Any, table: str) -> None:
    """Flips if ``_anonymize_checkpoint_rows`` / ``_anonymize_write_rows``, walking a thread's rows a ``keyset_page``
    at a time by the cursor each answers, read other rows, or in another order, than the thread's rows by its key's
    order (the hand-written pages past a tuple cursor read exactly those)."""
    columns, key = _ANONYMIZED[table]

    async def helper(conn: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            page = await keyset_page(
                conn,
                table,
                columns=columns,
                by={"thread_id": _THREAD},
                order=[Sort(column, cast=None) for column in key],
                limit=5,
                cursor=cursor,
            )
            rows += page.rows
            if page.next_cursor is None:
                return rows
            cursor = page.next_cursor

    await assert_same_results(
        db,
        table=table,
        row=_checkpoint if table == "checkpoints" else _write,
        old_sql=f"SELECT {', '.join(columns)} FROM {table} WHERE thread_id = $1 ORDER BY {', '.join(key)}",
        params=[_THREAD],
        helper=helper,
        order_by=list(key),
        page_size=5,
    )


async def test_an_anonymized_checkpoint_is_rewritten_by_its_key(db: Any) -> None:
    """Flips if the anonymized checkpoint's ``update_keyed`` writes other rows than its ``UPDATE ... WHERE thread_id
    AND checkpoint_ns AND checkpoint_id``."""
    key = {"thread_id": _THREAD, "checkpoint_ns": "", "checkpoint_id": "0001-0001"}
    await assert_same_results(
        db,
        table="checkpoints",
        row=_checkpoint,
        old_sql=(
            "UPDATE checkpoints SET checkpoint = $1, metadata_ = $2 "
            "WHERE thread_id = $3 AND checkpoint_ns = $4 AND checkpoint_id = $5"
        ),
        params=[b"anon", b"m2", *key.values()],
        helper=_deleting(
            lambda c: update_keyed(c, "checkpoints", values={"checkpoint": b"anon", "metadata_": b"m2"}, by=key)
        ),
        state_sql="SELECT * FROM checkpoints",
    )


async def test_an_anonymized_write_is_rewritten_by_its_key(db: Any) -> None:
    """Flips if the anonymized pending write's ``update_keyed`` writes other rows than its ``UPDATE ... WHERE`` its
    five key columns."""
    key = {"thread_id": _THREAD, "checkpoint_ns": "", "checkpoint_id": "cp1", "task_id": "task1", "idx": 1}
    await assert_same_results(
        db,
        table="checkpoint_writes",
        row=_write,
        old_sql=(
            "UPDATE checkpoint_writes SET blob = $1 WHERE thread_id = $2 AND checkpoint_ns = $3 "
            "AND checkpoint_id = $4 AND task_id = $5 AND idx = $6"
        ),
        params=[b"anon", *key.values()],
        helper=_deleting(lambda c: update_keyed(c, "checkpoint_writes", values={"blob": b"anon"}, by=key)),
        state_sql="SELECT * FROM checkpoint_writes",
    )
