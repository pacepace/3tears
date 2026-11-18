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
- skills ``AgentSkillCollection.increment_outcome_counts`` (two statements chosen by the outcome -> one
  ``update_keyed`` whose ``values`` the outcome picks: ``Increment(1)``, ``now()``) and workspace's version stamp (four
  modules' ``UPDATE workspaces SET current_version = GREATEST(current_version, $1)`` ->
  ``tools.helpers.raise_workspace_version``; the reset's plain set -> ``update_keyed``): held as write entries of the
  rewrite harness (:func:`threetears.core.testing.rewrites.assert_rewrite_same`), the table after each compared.
- pages whose key now runs the way their order does (``date_created DESC`` completed by the key ``DESC``, so one
  index walks the page; completed ascending, YugabyteDB read and sorted them): memory's
  ``MemoryRefsCollection.find_by_conversation`` (v037) and workspace's
  ``WorkspaceFileVersionCollection.find_by_workspace`` / ``find_by_workspace_and_path`` (v006), each against the
  statement as it read before its key was completed (``ORDER BY date_created DESC LIMIT n``): rows tied on the date
  may come in either order, and of those tied with a page's last row only how many (``limited``).
- workspace ``tools.helpers``: ``next_journal_version`` (``COALESCE(MAX(version), 0)`` -> ``aggregate_keyed`` ``Max``)
  and ``enrich_workspace_identity`` (the namespace row by its whole primary key -> ``read_row``); wake
  ``rate_limit`` active-schedule cap (its two ``COUNT(*)`` statements -> ``aggregate_keyed``), through
  ``check_active_schedule_cap`` (create) and ``resume_schedule_serialized`` (resume, the wake left out).
- wake ``rate_limit`` agent and user fire counts (one statement summing two joined counts -> two ``aggregate_keyed``
  counts, each naming its wakes by a keyed sub-select of its source table), through ``check_fire_limits`` and
  ``check_rate_limit``; wake ``manual.claim_manual_fire`` (one ``UPDATE ... RETURNING`` feeding the ``INSERT`` -> an
  ``update_keyed`` guarded by the gap and "no fire in flight", then the insert, in one transaction), its table after
  compared too.
- scheduled-jobs ``JobFireCollection.latest_for_jobs`` (``DISTINCT ON`` over the listed jobs -> one ``keyset_page`` of
  one row a job) and ``reap_stale_dispatching`` (a hand ``UPDATE`` with a kind ``EXISTS`` -> ``update_keyed`` with
  the ``EXISTS`` as a guard), its table after compared too.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from operator import itemgetter
from types import SimpleNamespace
from typing import Any

import pytest

from threetears.agent.memory.collections import MemoryRefsCollection
from threetears.agent.wake import manual, rate_limit
from threetears.agent.wake.entities import WakeScheduleEntity
from threetears.agent.workspace.entities import Workspace
from threetears.core import keyed
from threetears.agent.workspace.collections import WorkspaceFileVersionCollection
from threetears.agent.workspace.tools import helpers as workspace_helpers
from threetears.agent.workspace.tools.helpers import next_journal_version
from threetears.core.collections.registry import CollectionRegistry
from threetears.scheduled_jobs.collections import REAPED_DISPATCH_ERROR, JobFireCollection
from threetears.core.config import DefaultCoreConfig
from threetears.core.keyed import Fragment, Sort, delete_keyed, keyset_page, keyset_rows, read_by, update_keyed
from threetears.core.testing.rewrites import assert_rewrite_same
from threetears.core.testing.same_results import Seeded, assert_same_results

pytestmark = pytest.mark.integration

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
                order=[Sort(column, cast="integer" if column == "idx" else "text") for column in key],
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


def _entry(table: str, old_sql: str, sets: list[dict[str, Any]], pins: list[dict[str, Any]]) -> dict[str, Any]:
    """a hand conversion as the rewrite harness's write entry: the old statement, the SETs and pins the helper sends."""
    return {
        "shape": "write",
        "helper": "update_keyed",
        "table": table,
        "old_sql": old_sql,
        "sets": sets,
        "pins": pins,
        "returning": [],
    }


_SKILL_KEY = [
    {"column": "agent_id", "kind": "param", "param": 1},
    {"column": "skill_id", "kind": "param", "param": 2},
]
_WORKSPACE_KEY = [
    {"column": "workspace_id", "kind": "param", "param": 3},
    {"column": "agent_id", "kind": "param", "param": 4},
]


@pytest.mark.parametrize(
    "entry",
    [
        _entry(
            "agent_skills",
            "UPDATE agent_skills SET success_count = success_count + 1, date_updated = now() "
            "WHERE agent_id = $1 AND skill_id = $2",
            [
                {"column": "success_count", "kind": "increment", "literal": 1},
                {"column": "date_updated", "kind": "now"},
            ],
            _SKILL_KEY,
        ),
        _entry(
            "agent_skills",
            "UPDATE agent_skills SET failure_count = failure_count + 1, last_failure_at = now(), date_updated = now() "
            "WHERE agent_id = $1 AND skill_id = $2",
            [
                {"column": "failure_count", "kind": "increment", "literal": 1},
                {"column": "last_failure_at", "kind": "now"},
                {"column": "date_updated", "kind": "now"},
            ],
            _SKILL_KEY,
        ),
        _entry(
            "workspaces",
            "UPDATE workspaces SET current_version = GREATEST(current_version, $1), date_updated = $2 "
            "WHERE workspace_id = $3 AND agent_id = $4",
            [
                {
                    "column": "current_version",
                    "kind": "fragment",
                    "literal": "GREATEST(current_version, $1)",
                    "params": [1],
                },
                {"column": "date_updated", "kind": "param", "param": 2},
            ],
            _WORKSPACE_KEY,
        ),
        _entry(
            "workspaces",
            "UPDATE workspaces SET current_version = $1, date_updated = $2 WHERE workspace_id = $3 AND agent_id = $4",
            [
                {"column": "current_version", "kind": "param", "param": 1},
                {"column": "date_updated", "kind": "param", "param": 2},
            ],
            _WORKSPACE_KEY,
        ),
    ],
    ids=["skill-success", "skill-failure", "workspace-raise", "workspace-reset"],
)
async def test_a_hand_keyed_write_leaves_the_table_as_its_statement_did(db: Any, entry: dict[str, Any]) -> None:
    """Flips if a hand conversion's update_keyed (an increment, a GREATEST fragment, a plain set) leaves its table
    other than the statement it replaced did."""
    await assert_rewrite_same(db, entry)


_NS = uuid.UUID("2f0c51f4-5b0e-4f5e-8a77-61c0de0f4a99")
_T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
#: tenant 0's and tenant 1's agent (one conversation: the agent is the boundary) and workspace
_AGENTS = (uuid.uuid5(_NS, "agent-0"), uuid.uuid5(_NS, "agent-1"))
_CONVERSATION = uuid.uuid5(_NS, "conversation")
_WORKSPACES = (uuid.uuid5(_NS, "workspace-0"), uuid.uuid5(_NS, "workspace-1"))
#: the page each read asks (the harness's page boundary sits at it)
_KEYED_PAGE = 5
_REF_COLUMNS = ("conversation_id", "item_id", "item_type", "short_desc", "date_created", "agent_id")
_VERSION_COLUMNS = ("version_id", "workspace_id", "relative_path", "version", "sha256", "action", "date_created")


def _collection(db: Any, kind: type[Any]) -> Any:
    registry = CollectionRegistry()
    registry.configure(l3_pool=db)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    if kind is WorkspaceFileVersionCollection:
        return kind(registry, config, db)
    return kind(registry, config)


def _ref(seed: Seeded) -> dict[str, Any]:
    # newer with each row, so past the tenant boundary the other agent's refs are the conversation's newest: a read
    # that dropped the agent pin would page them
    return {
        "conversation_id": _CONVERSATION,
        "item_id": uuid.uuid5(_NS, f"ref-{seed.index}"),
        "item_type": "memory",
        "short_desc": f"ref {seed.index}",
        "date_created": _T0 + timedelta(hours=seed.tie),
        "date_updated": _T0,
        "agent_id": _AGENTS[seed.tenant],
    }


def _version(seed: Seeded) -> dict[str, Any]:
    return {
        "version_id": uuid.uuid5(_NS, f"version-{seed.index}"),
        "workspace_id": _WORKSPACES[seed.tenant],
        # two paths interleaved, so the path read keeps some of a workspace's rows and drops others
        "relative_path": f"p{seed.index % 2}.md",
        "version": seed.index,
        "content": b"c",
        "sha256": "0" * 64,
        "action": "update",
        "label": None if seed.null else f"l{seed.index}",
        "actor_id": _AGENTS[0],
        "correlation_id": _NS,
        "date_created": _T0 - timedelta(hours=seed.tie),
    }


async def _workspaces(db: Any) -> None:
    """both tenants' workspaces, the parents every journal row references (seeded before the harness's cases)."""
    await db.executemany(
        "INSERT INTO workspaces (workspace_id, agent_id, name, created_by, current_version, date_created,"
        " date_updated) VALUES ($1, $2, $3, $2, 0, $4, $4) ON CONFLICT DO NOTHING",
        [(workspace, _AGENTS[0], f"w{n}", _T0) for n, workspace in enumerate(_WORKSPACES)],
    )


def _dicts(entities: Sequence[Any]) -> list[dict[str, Any]]:
    return [entity.to_dict() for entity in entities]


async def test_an_agent_s_refs_page_is_its_newest_refs(db: Any) -> None:
    """Flips if find_by_conversation's page (completed by item_id DESC, v037) is not the agent's newest refs in the
    conversation, newest first, as ORDER BY date_created DESC LIMIT read them (the collection hands them back oldest
    first; reversed here to the read's order)."""

    async def helper(conn: Any) -> list[dict[str, Any]]:
        found = await _collection(conn, MemoryRefsCollection).find_by_conversation(
            _CONVERSATION, _AGENTS[0], limit=_KEYED_PAGE
        )
        return list(reversed(_dicts(found)))

    await assert_same_results(
        db,
        table="conversation_memory_refs",
        row=_ref,
        old_sql=(
            f"SELECT {', '.join(_REF_COLUMNS)} FROM conversation_memory_refs"
            " WHERE conversation_id = $1 AND agent_id = $2 ORDER BY date_created DESC LIMIT $3"
        ),
        params=[_CONVERSATION, _AGENTS[0], _KEYED_PAGE],
        helper=helper,
        columns=_REF_COLUMNS,
        order_by=["date_created"],
        page_size=_KEYED_PAGE,
        limited=True,
    )


@pytest.mark.parametrize("path", [None, "p0.md"], ids=["workspace", "path"])
async def test_a_workspace_s_history_page_is_its_newest_versions(db: Any, path: str | None) -> None:
    """Flips if find_by_workspace's or find_by_workspace_and_path's page (completed by version_id DESC, v006) is not
    the workspace's (or the path's) newest journal rows, newest first, as ORDER BY date_created DESC LIMIT read them."""
    await _workspaces(db)

    async def helper(conn: Any) -> list[dict[str, Any]]:
        versions = _collection(conn, WorkspaceFileVersionCollection)
        if path is None:
            return _dicts(await versions.find_by_workspace(_WORKSPACES[0], _KEYED_PAGE))
        return _dicts(await versions.find_by_workspace_and_path(_WORKSPACES[0], path, _KEYED_PAGE))

    narrowed = "" if path is None else " AND relative_path = $3"
    await assert_same_results(
        db,
        table="workspace_file_versions",
        row=_version,
        old_sql=(
            f"SELECT {', '.join(_VERSION_COLUMNS)} FROM workspace_file_versions"
            f" WHERE workspace_id = $1{narrowed} ORDER BY date_created DESC LIMIT $2"
        ),
        params=[_WORKSPACES[0], _KEYED_PAGE] + ([] if path is None else [path]),
        helper=helper,
        columns=_VERSION_COLUMNS,
        order_by=["date_created"],
        page_size=_KEYED_PAGE,
        limited=True,
    )


async def test_a_path_s_next_journal_version_is_one_past_its_largest(db: Any) -> None:
    """Flips if ``next_journal_version``'s keyed largest version (``aggregate_keyed`` ``Max`` over the UNIQUE
    (workspace_id, relative_path, version) prefix) is not the ``COALESCE(MAX(version), 0)`` its statement read: another
    path's or workspace's version, or no 0 for a path with no journal row."""
    await _workspaces(db)

    async def helper(conn: Any) -> list[dict[str, Any]]:
        return [{"max_version": await next_journal_version(conn, _WORKSPACES[0], "p0.md") - 1}]

    await assert_same_results(
        db,
        table="workspace_file_versions",
        row=_version,
        old_sql=(
            "SELECT COALESCE(MAX(version), 0) AS max_version FROM workspace_file_versions"
            " WHERE workspace_id = $1 AND relative_path = $2"
        ),
        params=[_WORKSPACES[0], "p0.md"],
        helper=helper,
    )


def _namespace(seed: Seeded) -> dict[str, Any]:
    """a namespace row: every other one the platform's (no customer), the rest a tenant's customer's."""
    platform = seed.index % 2 == 1
    return {
        "row_scope": "platform" if platform else "customer",
        "namespace_id": uuid.uuid5(_NS, f"namespace-{seed.index}"),
        "name": f"ns-{seed.index}",
        "namespace_type": "shared" if platform else "workspace",
        "customer_id": None if platform else _AGENTS[seed.tenant],
        "date_created": _T0,
        "date_updated": _T0 + timedelta(hours=seed.tie),
    }


class _Unrouted:
    """the session's connection for a read bound to a namespace (``with_options(namespace=)``): the schema under test
    is already on the path, so the routing keyword is dropped."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def fetch(self, sql: str, *params: Any, **routing: Any) -> list[Any]:
        del routing
        return list(await self._conn.fetch(sql, *params))


@pytest.mark.parametrize("index", [0, 1], ids=["customer", "platform"])
async def test_a_workspace_s_customer_is_its_namespace_row_s(
    db: Any, monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    """Flips if ``enrich_workspace_identity``'s keyed read of the namespace row (both row scopes and the id: the whole
    primary key) answers another row than its ``row_scope IN ('platform', 'customer')`` statement did."""
    kept: list[Any] = []
    real = keyed.read_row  # the function the helpers module imported, and the name the spy replaces there

    async def spy(*args: Any, **kwargs: Any) -> Any:
        row = await real(*args, **kwargs)
        kept.append(row)
        return row

    monkeypatch.setattr(workspace_helpers, "read_row", spy)
    target = uuid.uuid5(_NS, f"namespace-{index}")

    async def helper(conn: Any) -> list[dict[str, Any]]:
        kept.clear()
        await workspace_helpers.enrich_workspace_identity(
            Workspace({"id": target, "customer_id": None}), _Unrouted(conn)
        )
        (row,) = kept
        return [] if row is None else [dict(row)]

    await assert_same_results(
        db,
        table="namespaces",
        row=_namespace,
        old_sql="SELECT customer_id FROM namespaces WHERE row_scope IN ('platform', 'customer') AND namespace_id = $1",
        params=[target],
        helper=helper,
    )


def _schedule(seed: Seeded) -> dict[str, Any]:
    """a wake of its tenant's agent: active, paused, active, expired in turn; every third active one protected."""
    status = ("active", "paused", "active", "expired")[seed.index % 4]
    return {
        "conversation_id": _CONVERSATION,
        "schedule_id": uuid.uuid5(_NS, f"schedule-{seed.index}"),
        "user_id": _AGENTS[0],
        "agent_id": _AGENTS[seed.tenant],
        "schedule_type": "cron",
        "status": status,
        "protected": status == "active" and seed.index % 3 == 2,
        "name": None if seed.null else f"wake {seed.index}",
        "date_created": _T0 + timedelta(hours=seed.tie),
        "date_updated": _T0,
    }


class _OnePool:
    """a pool of the session's one connection (the harness's transaction is open on it)."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        yield self._conn

    async def fetch(self, query: str, *args: Any) -> Any:
        return await self._conn.fetch(query, *args)


class _Resumed:
    """the schedule collection's resume, as resume_schedule_serialized calls it after the cap: nothing written."""

    async def resume(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


@pytest.mark.parametrize("excluding", [None, 0], ids=["create", "resume"])
async def test_an_agent_s_active_count_is_its_unprotected_active_wakes(
    db: Any, monkeypatch: pytest.MonkeyPatch, excluding: int | None
) -> None:
    """Flips if the cap's keyed count (``aggregate_keyed`` over idx_wake_schedules_agent_status) counts other wakes
    than its ``COUNT(*)`` did: another agent's, a paused or expired one, a protected one, or (on resume) the wake
    being resumed. Read through the front doors: ``check_active_schedule_cap`` (create) and
    ``resume_schedule_serialized`` (resume), the count each sends kept by a spy."""
    left_out = None if excluding is None else uuid.uuid5(_NS, f"schedule-{excluding}")
    kept: list[Any] = []
    real = keyed.aggregate_keyed

    async def spy(*args: Any, **kwargs: Any) -> Any:
        rows = await real(*args, **kwargs)
        kept.append(rows)
        return rows

    monkeypatch.setattr(rate_limit, "aggregate_keyed", spy)

    async def helper(conn: Any) -> list[dict[str, Any]]:
        kept.clear()
        if left_out is None:
            await rate_limit.check_active_schedule_cap(agent_id=_AGENTS[0], cap=10**6, pool=conn)
        else:
            await rate_limit.resume_schedule_serialized(
                collection=_Resumed(),  # type: ignore[arg-type]
                agent_id=_AGENTS[0],
                conversation_id=_CONVERSATION,
                schedule_id=left_out,
                next_fire_at=_T0,
                cap=10**6,
                pool=_OnePool(conn),
            )
        ((counted,),) = kept
        return [{"n": counted["n"]}]

    old_sql = (
        "SELECT COUNT(*) AS n FROM agent_wake_schedules WHERE agent_id = $1 AND status = 'active' AND NOT protected"
    )
    await assert_same_results(
        db,
        table="agent_wake_schedules",
        row=_schedule,
        old_sql=old_sql if left_out is None else old_sql + " AND schedule_id != $2",
        params=[_AGENTS[0]] if left_out is None else [_AGENTS[0], left_out],
        helper=helper,
    )


#: each tenant's own conversation and user (the agent and the user are the boundary of the fire counts)
_CONVERSATIONS = (uuid.uuid5(_NS, "conversation-0"), uuid.uuid5(_NS, "conversation-1"))
_USERS = (uuid.uuid5(_NS, "user-0"), uuid.uuid5(_NS, "user-1"))


def _wake(seed: Seeded) -> dict[str, Any]:
    """a wake of its tenant's agent and user, in its tenant's conversation; every third one protected."""
    return {
        "conversation_id": _CONVERSATIONS[seed.tenant],
        "schedule_id": uuid.uuid5(_NS, f"wake-{seed.index}"),
        "user_id": _USERS[seed.tenant],
        "agent_id": _AGENTS[seed.tenant],
        "schedule_type": "cron",
        "status": "active",
        "protected": seed.index % 3 == 2,
        "name": None if seed.null else f"wake {seed.index}",
        "date_created": _T0 + timedelta(hours=seed.tie),
        "date_updated": _T0,
    }


#: a fire's (status, hours ago, source) for each wake: inside the day fired, silent, run by hand, not run; and a
#: fire of the day before
_WAKE_FIRES = (
    ("fired", 1, "scheduled_tick"),
    ("fired_silent", 2, "scheduled_tick"),
    ("fired", 3, "manual"),
    ("skipped_busy", 1, "scheduled_tick"),
    ("fired", 30, "scheduled_tick"),
)
#: and for each subscription
_HOOK_FIRES = (("fired", 1, "webhook"), ("fired_silent", 2, "webhook"), ("fired", 30, "webhook"))


async def _fires_beside(conn: Any, wakes: list[Mapping[str, Any]]) -> None:
    """two subscriptions of each tenant's agent and user, and the fires of every wake and subscription; one fire of
    tenant 0's first subscription written in tenant 1's conversation (no key ties a webhook fire's conversation to its
    subscription's, so a count by the subscription alone would take it)."""
    now = datetime.now(UTC)
    fire = 0

    async def fired(
        conversation: Any, status: str, hours: int, source: str, *, schedule: Any = None, hook: Any = None
    ) -> None:
        nonlocal fire
        fire += 1
        await conn.execute(
            "INSERT INTO wake_fires (conversation_id, fire_id, schedule_id, webhook_subscription_id, actual_fired_at,"
            " status, fire_source, requested_by) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            conversation,
            uuid.uuid5(_NS, f"fire-{fire}"),
            schedule,
            hook,
            now - timedelta(hours=hours),
            status,
            source,
            "user:x" if source == "manual" else None,
        )

    for tenant in (0, 1):
        for k in (0, 1):
            hook = uuid.uuid5(_NS, f"hook-{tenant}-{k}")
            await conn.execute(
                "INSERT INTO webhook_subscriptions (conversation_id, subscription_id, user_id, agent_id, name,"
                " verification_scheme, secret_ciphertext) VALUES ($1, $2, $3, $4, $5, 'generic_hmac_sha256', 'x'::bytea)",
                _CONVERSATIONS[tenant],
                hook,
                _USERS[tenant],
                _AGENTS[tenant],
                f"hook {tenant} {k}",
            )
            for status, hours, source in _HOOK_FIRES:
                await fired(_CONVERSATIONS[tenant], status, hours, source, hook=hook)
    await fired(_CONVERSATIONS[1], "fired", 1, "webhook", hook=uuid.uuid5(_NS, "hook-0-0"))
    for wake in wakes:
        for status, hours, source in _WAKE_FIRES:
            await fired(wake["conversation_id"], status, hours, source, schedule=wake["schedule_id"])


class _Kept:
    """the keyed counts rate_limit sends, each one's answer kept (a spy on its ``aggregate_keyed``)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.counted: list[int] = []
        real = keyed.aggregate_keyed

        async def spy(*args: Any, **kwargs: Any) -> Any:
            rows = await real(*args, **kwargs)
            self.counted.append(int(rows[0]["n"]))
            return rows

        monkeypatch.setattr(rate_limit, "aggregate_keyed", spy)


@pytest.mark.parametrize("count_manual", [True, False], ids=["manual-counted", "manual-left-out"])
async def test_an_agent_s_fire_count_is_its_wakes_and_subscriptions_fires(
    db: Any, monkeypatch: pytest.MonkeyPatch, count_manual: bool
) -> None:
    """Flips if the agent's fire count (two keyed counts, its wakes' fires and its subscriptions' fires, each by a
    sub-select of its source table) counts other fires than the one statement summing two joined counts did: another
    agent's, a protected wake's, a fire not run, one of the day before, one run by hand when those are left out, or a
    subscription's fire written in another conversation. Read through check_fire_limits, its counts kept by a spy."""
    kept = _Kept(monkeypatch)

    async def helper(conn: Any) -> list[dict[str, Any]]:
        kept.counted.clear()
        trigger = SimpleNamespace(
            conversation_id=_CONVERSATIONS[0],
            schedule_id=None,
            webhook_subscription_id=uuid.uuid5(_NS, "no-such-hook"),
            fire_source="webhook",
            agent_id=_AGENTS[0],
            user_id=_USERS[0],
            protected=False,
        )
        limits = SimpleNamespace(manual_fires_count=count_manual, per_wake=10**6, per_agent=10**6)
        assert await rate_limit.check_fire_limits(trigger, conn, limits) is None  # type: ignore[arg-type]
        _wake, wakes, hooks = kept.counted
        return [{"n": wakes + hooks}]

    await assert_same_results(
        db,
        table="agent_wake_schedules",
        row=_wake,
        old_sql=(
            "SELECT "
            "(SELECT COUNT(*) FROM wake_fires wf "
            " JOIN agent_wake_schedules ws ON wf.conversation_id = ws.conversation_id AND wf.schedule_id = ws.schedule_id "
            " WHERE ws.agent_id = $1 AND NOT ws.protected AND wf.actual_fired_at > $2 "
            " AND wf.status IN ('fired', 'fired_silent') AND ($3 OR wf.fire_source <> 'manual')) "
            "+ "
            "(SELECT COUNT(*) FROM wake_fires wf "
            " JOIN webhook_subscriptions ws"
            " ON wf.conversation_id = ws.conversation_id AND wf.webhook_subscription_id = ws.subscription_id "
            " WHERE ws.agent_id = $1 AND wf.actual_fired_at > $2 "
            " AND wf.status IN ('fired', 'fired_silent')) "
            "AS n"
        ),
        params=[_AGENTS[0], datetime.now(UTC) - timedelta(hours=rate_limit.RATE_LIMIT_WINDOW_HOURS), count_manual],
        helper=helper,
        related=_fires_beside,
    )


async def test_a_user_s_fire_count_is_its_wakes_and_subscriptions_fires(
    db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the user's fire count (two keyed counts by a sub-select of each source table) counts other fires than
    the one statement summing two joined counts did: another user's, a silent one or one not run, one of the day
    before, or a subscription's fire written in another conversation. Read through check_rate_limit."""
    kept = _Kept(monkeypatch)

    async def helper(conn: Any) -> list[dict[str, Any]]:
        kept.counted.clear()
        trigger = SimpleNamespace(
            conversation_id=uuid.uuid5(_NS, "no-such-conversation"),
            schedule_id=None,
            fire_source="scheduled_tick",
            user_id=_USERS[0],
        )
        config = SimpleNamespace(max_fires_per_conv_per_day=10**6, max_fires_per_user_per_day=10**6)
        assert await rate_limit.check_rate_limit(trigger, conn, config) is None  # type: ignore[arg-type]
        _conversation, wakes, hooks = kept.counted
        return [{"n": wakes + hooks}]

    await assert_same_results(
        db,
        table="agent_wake_schedules",
        row=_wake,
        old_sql=(
            "SELECT "
            "(SELECT COUNT(*) FROM wake_fires wf "
            " JOIN agent_wake_schedules ws ON wf.conversation_id = ws.conversation_id AND wf.schedule_id = ws.schedule_id "
            " WHERE ws.user_id = $1 AND wf.actual_fired_at > $2 AND wf.status = 'fired') "
            "+ "
            "(SELECT COUNT(*) FROM wake_fires wf "
            " JOIN webhook_subscriptions ws"
            " ON wf.conversation_id = ws.conversation_id AND wf.webhook_subscription_id = ws.subscription_id "
            " WHERE ws.user_id = $1 AND wf.actual_fired_at > $2 AND wf.status = 'fired') "
            "AS n"
        ),
        params=[_USERS[0], datetime.now(UTC) - timedelta(hours=rate_limit.RATE_LIMIT_WINDOW_HOURS)],
        helper=helper,
        related=_fires_beside,
    )


#: the manual fire's claim instant, its gap, and its fire's id (the claim's own is pinned to it: uuid7 is replaced)
_CLAIMED_AT = _T0 + timedelta(days=1)
_GAP = timedelta(minutes=10)
_MANUAL_FIRE = uuid.uuid5(_NS, "manual-fire")


def _manual_wake(seed: Seeded) -> dict[str, Any]:
    """a wake last run by hand 0, 6, 10 (the gap's edge) or 12 minutes before the claim, or never (NULLs)."""
    return {
        **_wake(seed),
        "protected": False,
        "last_manual_fire_at": None if seed.null else _CLAIMED_AT - timedelta(minutes=(0, 6, 10, 12)[seed.index % 4]),
    }


async def _in_flight_beside(conn: Any, wakes: list[Mapping[str, Any]]) -> None:
    """a fire in flight of every fifth wake from the second; a fire that ended of every other."""
    for index, wake in enumerate(wakes):
        await conn.execute(
            "INSERT INTO wake_fires (conversation_id, fire_id, schedule_id, actual_fired_at, status, fire_source)"
            " VALUES ($1, $2, $3, $4, $5, 'scheduled_tick')",
            wake["conversation_id"],
            uuid.uuid5(_NS, f"in-flight-{index}"),
            wake["schedule_id"],
            _CLAIMED_AT - timedelta(hours=1),
            "dispatching" if index % 5 == 1 else "fired",
        )


@pytest.mark.parametrize(
    "target", [0, 1, 2, 3, 6], ids=["too-soon", "in-flight", "gap-edge", "past-gap", "edge-in-flight"]
)
async def test_a_manual_claim_takes_the_limit_its_statement_took(
    db: Any, monkeypatch: pytest.MonkeyPatch, target: int
) -> None:
    """Flips if the manual claim (a keyed UPDATE of the stamp, guarded by the gap and "no fire in flight", then the
    insert, in one transaction) claims where the one UPDATE ... RETURNING feeding the INSERT did not, or the other way:
    inside the gap, at its edge, past it, never run by hand, a fire in flight, the wake gone; or leaves the wake's
    stamp or the fire row otherwise. The single-claimer path; the race is test_manual_fire_and_grid's."""
    monkeypatch.setattr(manual, "uuid7", lambda: _MANUAL_FIRE)
    wake_id = uuid.uuid5(_NS, f"wake-{target}")
    tenant_conversation = _CONVERSATIONS[0]

    async def helper(conn: Any) -> list[dict[str, Any]]:
        # named by tenant 0's conversation, as the statement names it: a wake of tenant 1's is not found
        wake = WakeScheduleEntity(
            {
                "conversation_id": tenant_conversation,
                "schedule_id": wake_id,
                "user_id": _USERS[0],
                "agent_id": _AGENTS[0],
                "execution_mode": "inline",
                "schedule_type": "cron",
                "name": "wake",
                "task_prompt": "do it",
                "context_from_schedule_id": None,
                "skill_id": None,
                "include_conversation_history": False,
                "protected": False,
            }
        )
        try:
            claim = await manual.claim_manual_fire(
                _OnePool(conn),
                wake,
                requested_by="user:x",
                min_gap=_GAP,
                now=_CLAIMED_AT,
            )
        except manual.ManualFireRefused:
            return []
        return [{"fire_id": claim.fire_id}]

    claimed_old = (
        "WITH claim AS ("
        " UPDATE agent_wake_schedules s SET last_manual_fire_at = $3::timestamptz"
        " WHERE s.conversation_id = $1 AND s.schedule_id = $2"
        " AND (s.last_manual_fire_at IS NULL"
        " OR s.last_manual_fire_at <= $3::timestamptz - make_interval(secs => $4::double precision))"
        " AND NOT EXISTS (SELECT 1 FROM wake_fires f WHERE f.conversation_id = s.conversation_id"
        " AND f.schedule_id = s.schedule_id AND f.status = 'dispatching')"
        " RETURNING s.conversation_id, s.schedule_id)"
        " INSERT INTO wake_fires (conversation_id, fire_id, schedule_id, webhook_subscription_id, scheduled_fire_at,"
        " actual_fired_at, status, display_suppressed, fire_source, requested_by)"
        " SELECT c.conversation_id, $5::uuid, c.schedule_id, NULL, NULL, $3::timestamptz, 'dispatching', false,"
        " 'manual', $6::text FROM claim c RETURNING fire_id"
    )
    await assert_same_results(
        db,
        table="agent_wake_schedules",
        row=_manual_wake,
        old_sql=claimed_old,
        # the target's conversation: tenant 0's (each seed case places it there or seeds no such wake)
        params=[tenant_conversation, wake_id, _CLAIMED_AT, _GAP.total_seconds(), _MANUAL_FIRE, "user:x"],
        helper=helper,
        related=_in_flight_beside,
        state_sql=(
            "SELECT s.schedule_id, s.last_manual_fire_at, f.fire_id, f.conversation_id, f.status, f.fire_source,"
            " f.requested_by, f.actual_fired_at, f.scheduled_fire_at, f.display_suppressed"
            " FROM agent_wake_schedules s LEFT JOIN wake_fires f ON f.schedule_id = s.schedule_id"
        ),
    )


#: each tenant's partition of scheduled jobs; the sweep's instant and age
_JOB_PARTITIONS = (uuid.uuid5(_NS, "jobs-0"), uuid.uuid5(_NS, "jobs-1"))
_SWEPT_AT = _T0
_STALE = timedelta(minutes=10)
_FIRE_COLUMNS = ("partition_key", "fire_id", "job_id", "scheduled_fire_at", "actual_fired_at", "status", "error")


def _job(seed: Seeded) -> dict[str, Any]:
    """a scheduled job of its tenant's partition, of kind k0 or k1 in turn."""
    return {
        "partition_key": _JOB_PARTITIONS[seed.tenant],
        "job_id": uuid.uuid5(_NS, f"job-{seed.index}"),
        "kind": ("k0", "k1")[seed.index % 2],
        "schedule_type": "interval",
        "status": "active",
        "name": None if seed.null else f"job {seed.index}",
        "date_created": _T0 + timedelta(hours=seed.tie),
        "date_updated": _T0,
    }


async def _job_fires_beside(conn: Any, jobs: list[Mapping[str, Any]]) -> None:
    """0 to 3 fires of each job, in flight, done or failed in turn, at distinct instants: some in flight past the
    sweep's age, some inside it."""
    for index, job in enumerate(jobs):
        for turn in range(index % 4):
            await conn.execute(
                "INSERT INTO job_fires (partition_key, fire_id, job_id, scheduled_fire_at, actual_fired_at, status)"
                " VALUES ($1, $2, $3, $4, $4, $5)",
                job["partition_key"],
                uuid.uuid5(_NS, f"job-fire-{index}-{turn}"),
                job["job_id"],
                _SWEPT_AT - timedelta(minutes=7 * turn + index),
                ("dispatching", "succeeded", "failed")[turn % 3],
            )


class _OneAtATime:
    """the harness's one connection as a pool the collection may ask several reads of at once: they take it in turn
    (a pool lends each its own connection; one asyncpg connection runs one statement at a time)."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self._turn = asyncio.Lock()

    async def fetch(self, query: str, *args: Any) -> Any:
        async with self._turn:
            return await self._conn.fetch(query, *args)

    async def execute(self, query: str, *args: Any) -> Any:
        async with self._turn:
            return await self._conn.execute(query, *args)


def _job_fires(pool: Any) -> JobFireCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return JobFireCollection(registry=registry, config=config, nats_client=None)


async def test_the_listed_jobs_latest_fires_are_each_one_s_newest(db: Any) -> None:
    """Flips if latest_for_jobs (one keyed page of one row a job) answers other fires than the DISTINCT ON over the
    listed jobs did: not each job's newest, a job never fired present, an unlisted job's fire, another partition's
    job left out."""
    listed = [uuid.uuid5(_NS, f"job-{index}") for index in range(0, 26, 1)] + [uuid.uuid5(_NS, "no-such-job")]

    async def helper(conn: Any) -> list[dict[str, Any]]:
        found = await _job_fires(_OneAtATime(conn)).latest_for_jobs(listed[::2] + listed[-1:])
        return [{column: getattr(fire, column) for column in _FIRE_COLUMNS} for fire in found.values()]

    await assert_same_results(
        db,
        table="scheduled_jobs",
        row=_job,
        old_sql=(
            f"SELECT DISTINCT ON (job_id) {', '.join(_FIRE_COLUMNS)} FROM job_fires WHERE job_id = ANY($1)"
            " ORDER BY job_id, actual_fired_at DESC"
        ),
        params=[listed[::2] + listed[-1:]],
        helper=helper,
        ordered=False,
        related=_job_fires_beside,
    )


async def test_the_stale_sweep_fails_the_kinds_old_in_flight_fires(db: Any) -> None:
    """Flips if reap_stale_dispatching (a keyed write over the fires in flight, the job's kind as a keyed EXISTS)
    fails other fires than its UPDATE did: one of another kind, one in flight inside the age, one that ended."""

    async def helper(conn: Any) -> list[dict[str, Any]]:
        reaped = await _job_fires(conn).reap_stale_dispatching(_SWEPT_AT, older_than=_STALE, kinds=["k0"])
        return [{"n": reaped}]

    await assert_same_results(
        db,
        table="scheduled_jobs",
        row=_job,
        old_sql=(
            "WITH reaped AS (UPDATE job_fires SET status = 'failed', error = $1"
            " WHERE status = 'dispatching' AND actual_fired_at < $2"
            " AND EXISTS (SELECT 1 FROM scheduled_jobs j WHERE j.job_id = job_fires.job_id AND j.kind = ANY($3))"
            " RETURNING partition_key, fire_id) SELECT count(*) AS n FROM reaped"
        ),
        params=[REAPED_DISPATCH_ERROR, _SWEPT_AT - _STALE, ["k0"]],
        helper=helper,
        related=_job_fires_beside,
        state_sql="SELECT fire_id, status, error FROM job_fires",
    )
