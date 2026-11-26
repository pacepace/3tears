"""a rescope keeps every permission cache right, live and after a restore from the tables.

A namespace is keyed by ``namespace_id`` alone and ``row_scope`` is a moveable column (agent_acl v007, Pace
2026-11-25): a rescope is an in-place update that records the scope it left on the row, and the change is heard under
both scopes. What is held here, from the caches a pod keeps:

- ``AclCache``'s per-namespace entries (keyed ``(group_id, namespace_id)``, evicted by the namespace's id) for the
  rescoped namespace go, and no other namespace's; the next evaluation reads the row's new customer and answers from
  it. A customer-authored role reaches only its own customer's namespaces, so a namespace moved off that customer
  must stop granting it (a cache that kept the entry would go on granting);
- ``CallerAccessCache`` (one answer per person, dropped by any namespace change) is dropped;
- the namespace collection's own cached row, held under the key it had before the move, goes (after a restore: live,
  the old composite key's eviction was named too).

Live: the rescope's write records the move in its own statement, and its broadcast names the key with the scope it
joined and the scope it left. After a restore: the records a restore of a lost change log reads back from the table
(the row under its own scope, and once more under the scope its move left) evict the same entries on a follower that
never heard the live broadcast.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from collections.abc import AsyncIterator
from unittest.mock import MagicMock
from uuid import UUID, uuid4, uuid7

from sqlalchemy import MetaData

from threetears.agent.acl import (
    CallerAccessCache,
    CallerKey,
    EvaluationContext,
    Group,
    GroupMembership,
    GroupNamespaceKey,
    MemberType,
    Namespace,
    NamespaceCollection,
    Role,
    RoleAssignment,
    ScopeType,
    bind_acl_cache_to_access_tables,
    bind_caller_cache_to_access_tables,
    evaluate_decision,
)
from threetears.agent.acl.cache import AclCache
from threetears.agent.acl.tables import register_rbac_l1_tables
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import CacheInvalidationMessage, CollectionRegistry
from threetears.core.collections.base import NO_L2
from threetears.core.collections.epoch_change import ChangedRow, EpochChange
from threetears.core.collections.table_restore import StampedTable, TableChanges, restored_tables
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import next_epoch

from .fake_loaders import FakeStore, make_cache

#: the customer whose namespace it is before the move
_CUSTOMER = UUID("0193a2b4-0000-7000-8000-0000000000c1")


class _Source:
    """the namespaces table's write generation: every advance succeeds."""

    def __init__(self) -> None:
        self.count = 0

    async def current(self, table_name: str) -> str:
        return f"inc:{self.count}"

    async def advance(self, table_name: str) -> str:
        self.count += 1
        return f"inc:{self.count}"


class _Pool:
    """L3 as the rescope reads and writes it: the row's scope and customer, then the keyed UPDATE that moves it.

    :ivar sent: every statement, whitespace folded, with its parameters
    """

    def __init__(self, namespace_id: UUID) -> None:
        self.namespace_id = namespace_id
        self.sent: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        folded = " ".join(sql.split())
        self.sent.append((folded, params))
        if folded.startswith("SELECT"):
            assert self.namespace_id in params, params
            return [{"row_scope": "customer", "customer_id": _CUSTOMER}]
        assert folded.startswith('UPDATE "namespaces"'), folded
        return [{"namespace_id": self.namespace_id}]


def _granted(namespace_id: UUID, other_id: UUID) -> tuple[FakeStore, UUID, UUID]:
    """a platform group a user is in, granted a customer-authored role on two namespaces of that customer.

    :return: the store, the user and the group
    """
    user, group_id, role_id = uuid4(), uuid4(), uuid4()
    store = FakeStore()
    store.add_group(Group(id=group_id, name="readers", customer_id=None))
    store.add_role(
        Role(
            id=role_id,
            name="Reader",
            permissions={"workspace": frozenset({"read"})},
            is_built_in=False,
            customer_id=_CUSTOMER,
        )
    )
    store.add_membership(GroupMembership(group_id, MemberType.USER, user, None))
    for scoped in (namespace_id, other_id):
        store.add_assignment(
            RoleAssignment(
                id=uuid4(),
                role_id=role_id,
                group_id=group_id,
                scope_type=ScopeType.NAMESPACE,
                scope_namespace_id=scoped,
                scope_namespace_type=None,
                scope_customer_id=None,
            )
        )
    return store, user, group_id


def _namespace(namespace_id: UUID, customer_id: UUID | None) -> Namespace:
    return Namespace(
        id=namespace_id, customer_id=customer_id, namespace_type="workspace", owner_agent_id=None, owner_namespace=None
    )


def _row(namespace_id: UUID, customer_id: UUID | None) -> dict[str, Any]:
    """the namespace's row as L1 holds it."""
    return {
        "row_scope": "platform" if customer_id is None else "customer",
        "namespace_id": namespace_id,
        "name": f"ws.{namespace_id.hex}",
        "namespace_type": "workspace",
        "customer_id": customer_id,
    }


def _key(collection: NamespaceCollection, row: dict[str, Any]) -> Any:
    """the key the collection holds ``row`` under, as its declaration says (one column: the bare value)."""
    columns = collection.primary_key_columns
    values = tuple(row[column] for column in columns)
    return values[0] if len(values) == 1 else values


async def _held(
    registry: CollectionRegistry, store: FakeStore, user: UUID, namespace_id: UUID, other_id: UUID
) -> tuple[AclCache, CallerAccessCache[str], CallerKey]:
    """the caches a pod keeps, bound to the registry, each holding what it answered before the move.

    :return: the access cache, the per-caller cache and the caller it holds
    """
    acl = make_cache(store)
    bind_acl_cache_to_access_tables(registry, acl)
    callers: CallerAccessCache[str] = CallerAccessCache()
    bind_caller_cache_to_access_tables(registry, callers)
    callers.followed_by(lambda: True)
    for held in (namespace_id, other_id):
        context = EvaluationContext(namespace=_namespace(held, _CUSTOMER), action="read", user_id=user)
        assert await evaluate_decision(context, cache=acl), "the customer's role grants on the customer's namespace"
    caller = CallerKey(agent_id=uuid4(), user_id=user)
    assert callers.put(caller, "the namespaces this person sees", fence=callers.read_fence())
    return acl, callers, caller


async def _assert_the_move_was_heard(
    acl: AclCache,
    callers: CallerAccessCache[str],
    caller: CallerKey,
    store: FakeStore,
    *,
    user: UUID,
    group_id: UUID,
    namespace_id: UUID,
    other_id: UUID,
) -> None:
    """the rescoped namespace's entries are gone and no other's, the person's answer is gone, and the next evaluation
    answers from the row's new customer, as a cache that never held anything would."""
    assert acl.get_group_namespace(GroupNamespaceKey(group_id, namespace_id)) is None, "the stale entry is dropped"
    assert acl.get_group_namespace(GroupNamespaceKey(group_id, other_id)) is not None, "another namespace's is kept"
    assert callers.get(caller) is None, "the person's answer is dropped"
    moved = EvaluationContext(namespace=_namespace(namespace_id, None), action="read", user_id=user)
    fresh = await evaluate_decision(moved, cache=make_cache(store))
    assert fresh is False, "a customer's role grants nothing on a platform namespace"
    assert await evaluate_decision(moved, cache=acl) is fresh


async def _stale_answer(store: FakeStore, user: UUID, namespace_id: UUID) -> bool:
    """what a cache that held the pre-move entry and heard nothing answers after the move: why eviction matters."""
    stale = make_cache(store)
    before = EvaluationContext(namespace=_namespace(namespace_id, _CUSTOMER), action="read", user_id=user)
    await evaluate_decision(before, cache=stale)
    after = EvaluationContext(namespace=_namespace(namespace_id, None), action="read", user_id=user)
    return await evaluate_decision(after, cache=stale)


def _l1() -> SQLiteBackend:
    backend = SQLiteBackend(db_name=f"rescope_caches_{uuid7().hex}")
    metadata = MetaData()
    register_rbac_l1_tables(metadata)
    backend.initialize(metadata)
    return backend


def _collection(registry: CollectionRegistry) -> NamespaceCollection:
    return NamespaceCollection(
        registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), nats_client=NO_L2
    )


async def test_a_live_rescope_drops_the_stale_permissions_and_answers_from_the_new_scope() -> None:
    """Flips if a rescope stops evicting the moved namespace's AclCache entries (or evicts another's), stops dropping
    the per-caller answers, writes the move without recording the scope it left, or broadcasts it without naming the
    key with both scopes."""
    namespace_id, other_id = uuid4(), uuid4()
    store, user, group_id = _granted(namespace_id, other_id)
    assert await _stale_answer(store, user, namespace_id) is True, "a cache that heard nothing would keep granting"
    pool = _Pool(namespace_id)
    registry = CollectionRegistry()
    registry.configure(l1_backend=_l1(), l3_pool=pool)  # type: ignore[arg-type]
    registry.set_generation_source(_Source())
    heard: list[tuple[list[str], dict[str, Any] | None]] = []

    def hear(message: CacheInvalidationMessage) -> None:
        heard.append((list(message.ids), dict(message.columns) if message.columns is not None else None))

    registry.register_derived_cache(
        "namespaces",
        on_row=hear,
        on_table_dropped=lambda: heard.append((["dropped"], None)),
        on_partition=registry.slice_as_row("namespaces", hear),
    )
    namespaces = _collection(registry)
    acl, callers, caller = await _held(registry, store, user, namespace_id, other_id)

    outcome = await namespaces.rescope(namespace_id, customer_id=None)

    assert (outcome.moved, outcome.previous_row_scope, outcome.row_scope) == (True, "customer", "platform")
    await _assert_the_move_was_heard(
        acl, callers, caller, store, user=user, group_id=group_id, namespace_id=namespace_id, other_id=other_id
    )
    # the move is recorded in the write itself, from the row as it was: the scope it left and the move's epoch
    [update] = [sql for sql, _ in pool.sent if sql.startswith("UPDATE")]
    assert '"row_scope_moved_from" = CASE WHEN "row_scope" IS DISTINCT FROM' in update, update
    assert '"row_scope_moved_epoch" = CASE WHEN "row_scope" IS DISTINCT FROM' in update, update
    # the key once with the scope it joined and once with the scope it left
    assert heard == [
        ([f"{namespace_id}"], {"row_scope": "platform"}),
        ([f"{namespace_id}"], {"row_scope": "customer"}),
    ], heard


class _Catalog:
    """the keeper's L3: the catalog's restore declaration for ``namespaces``, as its DDL comments its restore index,
    and a ``COPY`` a statement that answers no rows (the reads' shapes are what is asked of here)."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.sent.append((query, args))
        [comment] = [
            each for each in NamespaceCollection.schema.create_statements(yugabyte=False) if each.startswith("COMMENT")
        ]
        declared = comment.split(" IS '", 1)[1][:-1].replace("''", "'")
        return [{"table_schema": "platform", "table_name": "namespaces", "declared": declared, "run_markers": False}]

    async def copy_from_query(self, query: str, *args: Any, output: Any, **_: Any) -> str:
        self.sent.append((query, args))
        return "COPY 0"

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_Catalog]:
        yield self

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


async def _restored_rows(before: dict[str, Any], after: dict[str, Any]) -> list[ChangedRow]:
    """the rows a restore of a lost log puts back for a namespace that moved since X, as its reads name them: the
    bucket's read names the row's key and invalidation columns as the row holds them now; a moveable table's moves read
    names it again, each moved column as the move left it. Each read's shape is checked against the SQL the restore
    sends, so the rows are the ones the database would answer.

    :param before: the row before the move
    :ptype before: dict[str, Any]
    :param after: the row after it
    :ptype after: dict[str, Any]
    :return: the rows, in the order the reads come
    :rtype: list[ChangedRow]
    """
    pool = _Catalog()
    [table] = await restored_tables(pool)
    assert isinstance(table, StampedTable), table
    floor = next_epoch()
    assert [change async for change in TableChanges(pool, [table], floor, MagicMock(), None)] == []
    reads = [query for query, _ in pool.sent if " AS t " in query]
    keys = "ARRAY[" + ", ".join(f't."{column}"::text' for column in table.keys) + "]"
    assert all(keys in query for query in reads), reads[:2]
    rows = [
        ChangedRow(
            ids=[str(after[column]) for column in table.keys],
            columns={column: str(after[column]) for column in table.columns} or None,
        )
    ]
    if table.moved_columns:
        moves = [query for query in reads if "_moved_from" in query]
        assert moves, "a moveable table is read again for its moved rows"
        for column in table.moved_columns:
            assert all(f't."{column}_moved_from"::text' in query for query in moves), moves[0]
        rows.append(
            ChangedRow(
                ids=[str(after[column]) for column in table.keys],
                columns={
                    column: str(before[column] if column in table.moved_columns else after[column])
                    for column in table.columns
                },
            )
        )
    return rows


async def test_a_restore_from_the_tables_drops_the_stale_permissions_a_rescope_left() -> None:
    """Flips if a restore of a lost change log reads a rescoped namespace back under a key its followers never held
    (the old ``(row_scope, namespace_id)``: the row comes back under its new scope alone, and the copy a follower held
    under the old one stays), or stops reading it back under the scope it left; or if the records it puts back stop
    evicting the namespace's AclCache entries and the per-caller answers on a follower that missed the live
    broadcast."""
    namespace_id, other_id = uuid4(), uuid4()
    store, user, group_id = _granted(namespace_id, other_id)
    before, after = _row(namespace_id, _CUSTOMER), _row(namespace_id, None)
    registry = CollectionRegistry()
    registry.configure(l1_backend=_l1())
    namespaces = _collection(registry)
    namespaces.write_to_cache_sync(before)
    assert await namespaces.get(_key(namespaces, before)) is not None
    acl, callers, caller = await _held(registry, store, user, namespace_id, other_id)

    rows = await _restored_rows(before, after)
    await registry.apply_epoch_change(EpochChange(table="namespaces", token="restored:1", rows=rows))

    await _assert_the_move_was_heard(
        acl, callers, caller, store, user=user, group_id=group_id, namespace_id=namespace_id, other_id=other_id
    )
    assert await namespaces.get(_key(namespaces, before)) is None, "the row the follower held before the move is gone"
