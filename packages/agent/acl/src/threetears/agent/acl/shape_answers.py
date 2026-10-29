"""answers derived once per access shape, shared by every replica, and dropped by the rows they reach.

An answer to an access question -- which reads a person may make, which scopes they read -- is a
function of their membership shape (:mod:`threetears.agent.acl.access_shapes`) and of rows everyone
shares. :class:`AccessShapeAnswers` keeps one answer per ``(kind, customer, shape)``:

- **L1, then L2, then derived once.** A replica answers from its L1 with no L2 or database read when
  it holds a valid answer; otherwise from L2, which every replica reads; otherwise it derives, under
  an in-process gate and the cross-pod build lock
  (:class:`~threetears.core.collections.derived.DerivedCollection`), so one replica in the cluster
  derives a shape and the others wait for its answer. Every answer derived or read from L2 is kept
  in L1. There is no L3: an answer is derived, and derived again on any miss.
- **No lifetimes; valid by the epoch system.** An answer carries a *stamp* -- every table's write
  generation as this replica had applied it before the derivation read anything -- and its
  *footprint* deps (the groups, roles and agents it can reach, and a whole-table dep on each table).
  :class:`ShapeDependencies` records, per dep, the generation of the last row it applied that reaches
  the dep, and a floor per table (where it began trusting the table). An answer is served only when
  every table's stamp is at or past the floor and at or past the last change of every dep it has.
  That holds whichever replica derived the answer: a generation is applied here only once every row
  of it has been, and a row is broadcast after it commits, so an answer stamped at or past a change
  read the change, and one stamped before it is never served by a replica that has applied it.
- **Narrow.** A role or group change makes stale only the answers whose footprint reaches it; a
  person's own membership row makes nothing stale (it moves that person to another shape, which is
  the caller's per-person cache, :func:`~threetears.agent.acl.caller_cache.bind_caller_cache_to_memberships`).
  A heard change also evicts the L1 entries it reaches, everywhere it is heard, and deletes their L2
  entries when stale; an L2 entry nobody indexed is caught by the stamp when it is read.
- **The read fence.** An answer derived while a change reaching any answer landed in this process is
  returned to its caller and kept in neither tier.
- **Trusted only while followed.** Nothing is served or kept unless every table is followed with its
  watch running (:func:`threetears.agent.acl.generation_follow.follow_access_shape_answers`); a
  question asked then is derived afresh.
"""

from __future__ import annotations

import asyncio
import functools
import json
import threading
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final
from uuid import UUID

from sqlalchemy import MetaData
from threetears.core.collections.base import NATS_CLIENT_FROM_REGISTRY
from threetears.core.collections.derived import BuildLock, DerivedCollection
from threetears.core.collections.generation import (
    GenerationMarks,
    NoWriteGeneration,
    WriteGenerationDeclaration,
    split_generation_token,
)
from threetears.core.collections.schema_backed import DATETIMETZ_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.collections.tier_reads import ReadTier, record_tier_read
from threetears.core.entities.base import BaseEntity
from threetears.observe import contain, get_logger

from threetears.agent.acl.access_shapes import (
    ANY_ROLE,
    AccessFootprint,
    AccessShape,
    ShapeActors,
    access_footprint,
    agent_dep,
    group_dep,
    role_dep,
    table_dep,
)
from threetears.agent.acl.access_tables import (
    _GROUP_ID_AT,
    _ROLE_ID_AT,
    DegradedEvictions,
    _column,
    _key_part,
    _member_kind,
    _uuid,
)
from threetears.agent.acl.collections import (
    GroupCollection,
    GroupMemberCollection,
    RoleAssignmentCollection,
    RoleCollection,
)
from threetears.agent.acl.types import MemberType

if TYPE_CHECKING:
    from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
    from threetears.core.config import CoreConfig

    from threetears.agent.acl.cache import AclCache

__all__ = [
    "AccessShapeAnswers",
    "ShapeAnswer",
    "ShapeAnswerOutcome",
    "ShapeDependencies",
]

log = get_logger(__name__)

_GROUPS: Final = GroupCollection.schema.name
_GROUP_MEMBERS: Final = GroupMemberCollection.schema.name
_ROLES: Final = RoleCollection.schema.name
_ROLE_ASSIGNMENTS: Final = RoleAssignmentCollection.schema.name

#: the characters a kind may carry: it is a key segment
_KIND_CHARS: Final = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


def _row_reach(message: CacheInvalidationMessage) -> frozenset[str] | None:
    """the deps one row reaches: none for a person's own membership, ``None`` for the whole table.

    - ``group_members``: a user's row reaches no shared dep (the person moves to another shape); an
      agent's reaches that agent; a nested group's reaches the child group;
    - ``role_assignments``: the group holding it;
    - ``roles``: the role (and every answer that could not list its roles);
    - ``groups``: the group;
    - any other table, or a row that does not say what it names: the whole table.

    :param message: the row broadcast
    :ptype message: CacheInvalidationMessage
    :return: the deps, or ``None`` for the whole table
    :rtype: frozenset[str] | None
    """
    reach: frozenset[str] | None = None
    if message.table == _GROUP_MEMBERS:
        kind = _member_kind(_column(message, "member_type"))
        member = _uuid(_column(message, "member_id"))
        if member is not None and kind == MemberType.USER.value:
            reach = frozenset()
        elif member is not None and kind == MemberType.AGENT.value:
            reach = frozenset((agent_dep(member),))
        elif member is not None and kind == MemberType.GROUP.value:
            reach = frozenset((group_dep(member),))
    elif message.table == _ROLE_ASSIGNMENTS:
        group = _uuid(_column(message, "group_id"))
        reach = None if group is None else frozenset((group_dep(group),))
    elif message.table == _ROLES:
        role = _uuid(_key_part(message, _ROLE_ID_AT))
        reach = None if role is None else frozenset((role_dep(role), ANY_ROLE))
    elif message.table == _GROUPS:
        group = _uuid(_key_part(message, _GROUP_ID_AT))
        reach = None if group is None else frozenset((group_dep(group),))
    return reach


#: the tables whose rows say what they reach; a row of one that does not is counted as degraded
_PLACED_TABLES: Final = frozenset((_GROUP_MEMBERS, _ROLE_ASSIGNMENTS, _ROLES, _GROUPS))


#: a generation as ``(incarnation, count)``
type _Generation = tuple[str, int]


def _generation(token: str | None) -> _Generation | None:
    """a token as ``(incarnation, count)``, or ``None`` when it is not one.

    :param token: the token
    :ptype token: str | None
    :return: the parts
    :rtype: tuple[str, int] | None
    """
    return None if token is None else split_generation_token(token)


class ShapeDependencies:
    """what this replica has applied of the tables answers are derived from, and which deps changed when.

    Bound to a registry's row broadcasts for ``tables`` (:meth:`bind`), it keeps its own
    :class:`GenerationMarks`, fed by exactly the rows it is told, so a generation is recorded here only
    once every row of it has reached this cache and everything bound before it -- the access cache
    among them -- which makes the mark a sound stamp for an answer about to be derived
    (:meth:`stamp`). It records, per dep, the generation of the last row reaching it, and per table the
    floor it began trusting from; :meth:`valid` judges a stamp against both. Thread-safe.

    :param registry: the registry whose listener hears the rows, and whose marks it starts from
    :ptype registry: CollectionRegistry
    :param tables: every table the answers are derived from
    :ptype tables: Sequence[str]
    :param degraded: where rows that did not say what they reach are counted; a new record when ``None``
    :ptype degraded: DegradedEvictions | None
    """

    def __init__(
        self, registry: CollectionRegistry, tables: Sequence[str], *, degraded: DegradedEvictions | None = None
    ) -> None:
        self._registry = registry
        self._degraded = degraded if degraded is not None else DegradedEvictions()
        self._tables: tuple[str, ...] = tuple(dict.fromkeys(tables))
        self._marks = GenerationMarks()
        self._floor: dict[str, _Generation | None] = dict.fromkeys(self._tables)
        # dep -> table -> the newest generation of a row of that table reaching it
        self._changed: dict[str, dict[str, _Generation]] = {}
        self._fence = 0
        self._listeners: list[Callable[[frozenset[str] | None], None]] = []
        self._watching: Callable[[], bool] | None = None
        self._lock = threading.Lock()

    @property
    def degraded(self) -> DegradedEvictions:
        """the rows heard that did not say what they reach, by table: each made its whole table stale.

        :return: the record
        :rtype: DegradedEvictions
        """
        return self._degraded

    @property
    def tables(self) -> tuple[str, ...]:
        """every table the answers are derived from.

        :return: the tables
        :rtype: tuple[str, ...]
        """
        return self._tables

    def followed_by(self, watching: Callable[[], bool] | None) -> None:
        """record how to ask whether every table's watch is running; ``None`` when nobody follows.

        :param watching: answers whether every watch is running now
        :ptype watching: Callable[[], bool] | None
        :return: nothing
        :rtype: None
        """
        self._watching = watching
        if watching is None:
            self._notify(None)

    def bind(self) -> Callable[[], None]:
        """start from the registry's marks and hear every row of :attr:`tables` from here.

        Call with the tables already followed on the registry and with no await between that and this,
        so no row is heard by the registry and not by this.

        :return: a call that removes the registrations
        :rtype: Callable[[], None]
        """
        removers: list[Callable[[], None]] = []
        with self._lock:
            for table in self._tables:
                self._adopt(table)
        for table in self._tables:
            removers.append(
                self._registry.register_derived_cache(
                    table,
                    on_row=functools.partial(self._on_row, table),
                    on_table_dropped=functools.partial(self._on_dropped, table),
                )
            )

        def remove() -> None:
            for remover in removers:
                remover()

        return remove

    def add_listener(self, listener: Callable[[frozenset[str] | None], None]) -> Callable[[], None]:
        """call ``listener`` with the deps each heard change reaches, or ``None`` when everything is stale.

        :param listener: called synchronously inside the row's handling; must not block
        :ptype listener: Callable[[frozenset[str] | None], None]
        :return: a call that removes it
        :rtype: Callable[[], None]
        """
        self._listeners.append(listener)

        def remove() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return remove

    def _adopt(self, table: str, *, past_now: bool = False) -> None:
        """take the registry's mark for ``table`` and trust the table from the newest generation it heard.

        :param table: the table
        :ptype table: str
        :param past_now: trust only generations after the newest heard: for a drop that moved no mark,
            whose reach is unknown at the generation this replica is already at
        :ptype past_now: bool
        :return: nothing
        :rtype: None
        """
        self._marks.adopt(table, self._registry.generation_marks)
        floor = _generation(self._marks.highest_heard(table))
        self._floor[table] = floor if floor is None or not past_now else (floor[0], floor[1] + 1)
        for tables in self._changed.values():
            tables.pop(table, None)

    def _on_row(self, table: str, message: CacheInvalidationMessage) -> None:
        """record what one row reaches, at its generation, and count it toward its advance.

        :param table: the table the row belongs to
        :ptype table: str
        :param message: the row broadcast
        :ptype message: CacheInvalidationMessage
        :return: nothing
        :rtype: None
        """
        reach = _row_reach(message)
        if reach is None and table in _PLACED_TABLES:
            self._degraded.record(table, "a row that does not say what it reaches makes its whole table stale")
        deps = frozenset((table_dep(table),)) if reach is None else reach
        with self._lock:
            if deps:
                self._fence += 1
                changed_at = _generation(message.generation)
                if changed_at is None:
                    # a row naming no generation: stale until this replica applies a later one
                    newest = _generation(self._marks.highest_heard(table))
                    changed_at = None if newest is None else (newest[0], newest[1] + 1)
                if changed_at is None:
                    self._floor[table] = None
                else:
                    for dep in deps:
                        tables = self._changed.setdefault(dep, {})
                        held = tables.get(table)
                        if held is None or held[0] != changed_at[0] or held[1] < changed_at[1]:
                            tables[table] = changed_at
            if message.generation is not None and message.bump_rows is not None:
                self._marks.hear(table, message.generation, message.bump_rows)
        if deps:
            self._notify(deps)

    def _on_dropped(self, table: str) -> None:
        """start ``table`` again from the registry's judged mark: nothing derived before is trusted.

        :param table: the table dropped in this process
        :ptype table: str
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._fence += 1
            # a drop after a judged generation moved the registry's mark past everything this replica
            # stamped; a drop that moved nothing leaves the reach unknown at the generation it is at
            unmoved = self._marks.highest_heard(table) == self._registry.generation_marks.highest_heard(table)
            self._adopt(table, past_now=unmoved and self._marks.recorded(table) is not None)
        self._notify(None)

    def _notify(self, deps: frozenset[str] | None) -> None:
        """tell every listener what a change reached.

        :param deps: the deps, or ``None`` for everything
        :ptype deps: frozenset[str] | None
        :return: nothing
        :rtype: None
        """
        for listener in tuple(self._listeners):
            listener(deps)

    @property
    def trusted(self) -> bool:
        """whether answers may be served and kept: followed, every watch running, every table stamped.

        :return: ``True`` when trusted
        :rtype: bool
        """
        watching = self._watching
        return watching is not None and watching() and self.stamp() is not None

    def fence(self) -> int:
        """the fence to take before deriving: it moves with every row that reaches any answer.

        :return: the count now
        :rtype: int
        """
        with self._lock:
            return self._fence

    def stamp(self) -> dict[str, str] | None:
        """every table's generation as applied here, for an answer about to be derived.

        :return: table to token, or ``None`` when a table is not yet trusted
        :rtype: dict[str, str] | None
        """
        stamp: dict[str, str] = {}
        with self._lock:
            for table in self._tables:
                token = self._marks.recorded(table)
                if token is None or self._floor.get(table) is None:
                    return None
                stamp[table] = token
        return stamp

    def valid(self, stamp: Mapping[str, str], deps: Iterable[str]) -> bool:
        """whether an answer with this stamp and these deps is current by everything applied here.

        :param stamp: the answer's stamp
        :ptype stamp: Mapping[str, str]
        :param deps: the answer's deps
        :ptype deps: Iterable[str]
        :return: ``True`` when it may be served
        :rtype: bool
        """
        stamped: dict[str, _Generation] = {}
        with self._lock:
            for table in self._tables:
                at = _generation(stamp.get(table))
                floor = self._floor.get(table)
                if at is None or floor is None or at[0] != floor[0] or at[1] < floor[1]:
                    return False
                stamped[table] = at
            for dep in deps:
                for table, changed_at in self._changed.get(dep, {}).items():
                    at = stamped.get(table)
                    if at is None or at[0] != changed_at[0] or at[1] < changed_at[1]:
                        return False
        return True


class ShapeAnswerOutcome(StrEnum):
    """how an answer was given."""

    #: this replica's L1 held it
    L1 = "l1"
    #: read from L2, where a replica (this one, or a peer) put it
    L2 = "l2"
    #: derived here, and kept
    DERIVED = "derived"
    #: derived here and kept nowhere: not trusted, or a change landed while it was derived
    UNCACHED = "uncached"


@dataclass(frozen=True, slots=True)
class ShapeAnswer:
    """an answer, and how it was given.

    :ivar body: the answer's text, as the compute returned it
    :ivar outcome: how it was given
    """

    body: str
    outcome: ShapeAnswerOutcome


class ShapeAnswerRow(BaseEntity):
    """one kept answer: ``scope``, ``shape``, ``context``, ``stamp``, ``deps``, ``body``."""


@dataclass
class _Pending:
    """the question a read of one key is asking, for the collection's hooks.

    :ivar shape: the shape asked about
    :ivar context: what else the answer depends on, matched exactly
    :ivar agent_ids: the agents whose side the answer evaluates
    :ivar compute: derives the answer as the shape's actor; ``None`` when nothing can be answered
    :ivar ran: whether this read derived
    :ivar storable: whether what it derived may be kept
    :ivar saved: whether what it derived was put in L2
    :ivar fence: the fence taken before it derived
    """

    shape: AccessShape
    context: str
    agent_ids: frozenset[UUID]
    compute: Callable[[UUID], Awaitable[str | None]]
    ran: bool = False
    storable: bool = False
    saved: bool = False
    fence: int = 0


_PENDING: ContextVar[_Pending | None] = ContextVar("access_shape_pending", default=None)


def _schema(table_name: str) -> TableSchema:
    """the L1 table of the answers.

    :param table_name: the table
    :ptype table_name: str
    :return: the schema
    :rtype: TableSchema
    """
    return TableSchema(
        name=table_name,
        primary_key=("scope", "shape"),
        columns=[
            Column("scope", STRING_TYPE),
            Column("shape", STRING_TYPE),
            Column("context", STRING_TYPE),
            Column("stamp", STRING_TYPE),
            Column("deps", STRING_TYPE),
            Column("body", STRING_TYPE),
            Column("date_created", DATETIMETZ_TYPE, nullable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
    )


class AccessShapeAnswers(DerivedCollection[ShapeAnswerRow]):
    """answers kept per ``(kind, customer, shape)`` in L1 and L2; see the module docstring.

    Construct after the registry's L1 backend and L2 client are set, bind and follow with
    :func:`threetears.agent.acl.generation_follow.follow_access_shape_answers`, and ask with
    :meth:`answer`.

    :param registry: the registry carrying the L1 backend, the L2 client and key scope
    :ptype registry: CollectionRegistry
    :param config: core configuration
    :ptype config: CoreConfig
    :param nats_client: the L2 client; the registry's when omitted
    :ptype nats_client: Any
    :param table_name: the table the keys are named by
    :ptype table_name: str
    :param acl_cache: the access cache answers are evaluated through; its membership loader must be
        ``actors``
    :ptype acl_cache: AclCache
    :param actors: the loader that answers for each shape's actor
    :ptype actors: ShapeActors
    :param dependencies: the generations applied here, bound to the tables the answers derive from
    :ptype dependencies: ShapeDependencies
    :param build_lock: the cross-pod build lock; a ``NatsBuildLock`` over the registry's client when omitted
    :ptype build_lock: BuildLock | None
    :raises ValueError: when ``acl_cache`` does not read memberships through ``actors``
    """

    primary_key_column: str | tuple[str, ...] = ("scope", "shape")
    #: an L1 copy is checked against the stamp on every read, so it is never served stale
    caches_in_l1: ClassVar[bool] = True
    write_generation: ClassVar[WriteGenerationDeclaration] = NoWriteGeneration(
        reason="an answer carries the access tables' generations it was derived at and is checked against them on "
        "every read; its own table's writes need no generation"
    )
    derived_values_tier: ClassVar[ReadTier] = ReadTier.L2
    l2_is_the_store: bool = False

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        nats_client: Any = NATS_CLIENT_FROM_REGISTRY,
        *,
        table_name: str,
        acl_cache: AclCache,
        actors: ShapeActors,
        dependencies: ShapeDependencies,
        build_lock: BuildLock | None = None,
    ) -> None:
        if acl_cache.membership_loader is not actors:
            raise ValueError("the access cache must read memberships through the shape actors it answers for")
        self._untrusted_logged = False
        self._table_name = table_name
        self._acl_cache = acl_cache
        self._actors = actors
        self._dependencies = dependencies
        # key -> its deps, and dep -> keys, for the entries this replica holds in L1
        self._held: dict[tuple[str, str], frozenset[str]] = {}
        self._by_dep: dict[str, set[tuple[str, str]]] = {}
        self._background: set[asyncio.Task[None]] = set()
        super().__init__(registry, config, nats_client, None, build_lock=build_lock)
        self.l3_pool = None
        if self._l1 is not None:
            self._l1.initialize(_schema(table_name).to_sqlalchemy_table(MetaData()).metadata)
        dependencies.add_listener(self._on_change)

    @property
    def table_name(self) -> str:
        """the table the answers' keys are named by.

        :return: the table name
        :rtype: str
        """
        return self._table_name

    @property
    def entity_class(self) -> type[ShapeAnswerRow]:
        """the entity class.

        :return: :class:`ShapeAnswerRow`
        :rtype: type[ShapeAnswerRow]
        """
        return ShapeAnswerRow

    @property
    def held(self) -> int:
        """how many answers this replica holds in L1.

        :return: the count
        :rtype: int
        """
        return len(self._held)

    async def settled(self) -> None:
        """wait for the L2 deletions changes have started here to finish.

        :return: nothing
        :rtype: None
        """
        while self._background:
            await asyncio.gather(*tuple(self._background), return_exceptions=True)

    @staticmethod
    def key_of(kind: str, customer_id: UUID, shape: AccessShape) -> tuple[str, str]:
        """the key of a kind of answer for one customer and shape.

        :param kind: the kind of answer: lowercase letters, digits and ``-``
        :ptype kind: str
        :param customer_id: the customer it is for
        :ptype customer_id: UUID
        :param shape: the shape
        :ptype shape: AccessShape
        :return: ``(scope, shape digest)``
        :rtype: tuple[str, str]
        :raises ValueError: when ``kind`` cannot be a key segment
        """
        if not kind or not set(kind) <= _KIND_CHARS:
            raise ValueError(f"{kind!r} cannot be a key segment: lowercase letters, digits and '-'")
        return f"{kind}-{customer_id.hex}", shape.digest

    async def answer(
        self,
        *,
        kind: str,
        customer_id: UUID,
        shape: AccessShape,
        context: str,
        agent_ids: Collection[UUID],
        compute: Callable[[UUID], Awaitable[str | None]],
    ) -> ShapeAnswer | None:
        """the answer for ``shape``: from L1, L2, or derived once and kept in both.

        :param kind: the kind of answer (a key segment)
        :ptype kind: str
        :param customer_id: the customer it is for
        :ptype customer_id: UUID
        :param shape: the shape of the person asking
        :ptype shape: AccessShape
        :param context: everything else the answer depends on that the tables do not (a route table's
            digest, say); an answer kept under another context is derived again
        :ptype context: str
        :param agent_ids: the agents whose side the answer evaluates
        :ptype agent_ids: Collection[UUID]
        :param compute: derives the answer's text evaluating as the given actor, or returns ``None``
            when it cannot be answered (nothing is kept); whatever it raises reaches the caller
        :ptype compute: Callable[[UUID], Awaitable[str | None]]
        :return: the answer, or ``None`` when ``compute`` returned ``None``
        :rtype: ShapeAnswer | None
        """
        key = self.key_of(kind, customer_id, shape)
        given: ShapeAnswer | None
        if not self.trusted:
            given = await self._uncached(shape, compute)
        else:
            held = self._select_from_l1(key)
            if held is not None and self._current(held, context):
                record_tier_read(self.table_name, ReadTier.L1)
                given = ShapeAnswer(str(held["body"]), ShapeAnswerOutcome.L1)
            else:
                given = await self._read_or_derive(key, shape, context, agent_ids, compute)
        return given

    @property
    def trusted(self) -> bool:
        """whether answers are served and kept now: the tables followed, and the access cache followed too.

        An access cache nobody follows holds rows nothing drops, and an answer derived through it would
        be stamped current and shared with every replica.

        :return: ``True`` when they are
        :rtype: bool
        """
        return self._dependencies.trusted and self._acl_cache.followed and self._acl_cache.trusted

    async def _uncached(
        self, shape: AccessShape, compute: Callable[[UUID], Awaitable[str | None]]
    ) -> ShapeAnswer | None:
        """derive afresh and keep nothing: what a question gets while answers are not trusted.

        Nothing held while a watch could miss a row is served once trust returns, so what this replica
        holds is dropped.

        :param shape: the shape
        :ptype shape: AccessShape
        :param compute: derives the answer as the given actor
        :ptype compute: Callable[[UUID], Awaitable[str | None]]
        :return: the answer, or ``None``
        :rtype: ShapeAnswer | None
        """
        if self._held:
            self._on_change(None)
        if not self._untrusted_logged:
            self._untrusted_logged = True
            log.warning(
                "access-shape answers are not trusted (a table or the access cache is not followed); each is "
                "derived afresh and kept nowhere until they are",
                extra={"extra_data": {"table": self.table_name}},
            )
        record_tier_read(self.table_name, ReadTier.L3)
        body = await compute(self._actors.actor_for(shape))
        return None if body is None else ShapeAnswer(body, ShapeAnswerOutcome.UNCACHED)

    async def _read_or_derive(
        self,
        key: tuple[str, str],
        shape: AccessShape,
        context: str,
        agent_ids: Collection[UUID],
        compute: Callable[[UUID], Awaitable[str | None]],
    ) -> ShapeAnswer | None:
        """the answer from L2, or derived once in the cluster; kept in L1 when current.

        Its outcome says where it is kept: ``derived`` when this read put it in L2, ``l2`` when it read
        it there, ``uncached`` when it derived and kept it nowhere.

        :param key: the key
        :ptype key: tuple[str, str]
        :param shape: the shape
        :ptype shape: AccessShape
        :param context: the context asked under
        :ptype context: str
        :param agent_ids: the agents whose side the answer evaluates
        :ptype agent_ids: Collection[UUID]
        :param compute: derives the answer as the given actor
        :ptype compute: Callable[[UUID], Awaitable[str | None]]
        :return: the answer, or ``None`` when ``compute`` answered nothing
        :rtype: ShapeAnswer | None
        """
        self._untrusted_logged = False
        ticket = self.scan_ticket()
        pending = _Pending(shape=shape, context=context, agent_ids=frozenset(agent_ids), compute=compute)
        token = _PENDING.set(pending)
        try:
            row, tier = await self.fetch_from_store_with_tier(key)
        finally:
            _PENDING.reset(token)
        record_tier_read(self.table_name, tier)
        given: ShapeAnswer | None = None
        if row is not None:
            if pending.ran:
                outcome = ShapeAnswerOutcome.DERIVED if pending.saved else ShapeAnswerOutcome.UNCACHED
            else:
                outcome = ShapeAnswerOutcome.L2
            # kept here too only while nothing reaching an answer has landed since and it is current
            keep = outcome is not ShapeAnswerOutcome.UNCACHED and self._current(row, context)
            if keep and (not pending.ran or self._dependencies.fence() == pending.fence):
                if self.write_to_cache_sync(row, read_since=ticket):
                    self._index(key, frozenset(json.loads(row["deps"])))
            given = ShapeAnswer(str(row["body"]), outcome)
        return given

    def _current(self, row: Mapping[str, Any], context: str) -> bool:
        """whether a kept answer is for ``context`` and valid by everything applied here.

        :param row: the kept answer
        :ptype row: Mapping[str, Any]
        :param context: the context asked under
        :ptype context: str
        :return: ``True`` when it may be served
        :rtype: bool
        """
        if row.get("context") != context:
            return False
        try:
            stamp = json.loads(row["stamp"])
            deps = json.loads(row["deps"])
        except KeyError, TypeError, ValueError:
            # NOSILENT: a row this release cannot read is no answer; the caller derives one
            return False
        return isinstance(stamp, dict) and isinstance(deps, list) and self._dependencies.valid(stamp, deps)

    def _index(self, key: tuple[str, str], deps: frozenset[str]) -> None:
        """record an L1 entry's deps, so a change reaching one evicts it.

        :param key: the entry's key
        :ptype key: tuple[str, str]
        :param deps: its deps
        :ptype deps: frozenset[str]
        :return: nothing
        :rtype: None
        """
        self._unindex(key)
        self._held[key] = deps
        for dep in deps:
            self._by_dep.setdefault(dep, set()).add(key)

    def _unindex(self, key: tuple[str, str]) -> None:
        """forget an L1 entry's deps.

        :param key: the entry's key
        :ptype key: tuple[str, str]
        :return: nothing
        :rtype: None
        """
        for dep in self._held.pop(key, frozenset()):
            keys = self._by_dep.get(dep)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    del self._by_dep[dep]

    def _on_change(self, deps: frozenset[str] | None) -> None:
        """evict the L1 entries a change reaches, and delete their L2 entries once stale.

        :param deps: what the change reached, or ``None`` for everything
        :ptype deps: frozenset[str] | None
        :return: nothing
        :rtype: None
        """
        if deps is None:
            reached = set(self._held)
        else:
            reached = {key for dep in deps for key in self._by_dep.get(dep, ())}
        for key in reached:
            self._unindex(key)
            self.evict_from_cache_sync(key)
        if reached:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # NOSILENT: no loop to delete L2 entries on (a synchronous caller); the stamp keeps them unserved
                return
            task = loop.create_task(self._delete_stale(tuple(reached)))
            self._background.add(task)
            task.add_done_callback(self._background.discard)

    async def _delete_stale(self, keys: tuple[tuple[str, str], ...]) -> None:
        """delete each key's L2 entry when it is stale here, at the revision read: a newer one stands.

        :param keys: the keys
        :ptype keys: tuple[tuple[str, str], ...]
        :return: nothing
        :rtype: None
        """
        try:
            kv = await self._ensure_kv()
            if kv is None:
                return
            for key in keys:
                entry = await kv.get_entry(key=self.l2_key(key))
                if entry is None:
                    continue
                raw, revision = entry
                row = self.deserialize(raw)
                if not self._current(row, str(row.get("context"))):
                    await kv.delete(key=self.l2_key(key), revision=revision)
        # prawduct:allow prawduct/broad-except -- background work no caller awaits: whatever it raises is recorded
        # here or nowhere, and an entry left in L2 is still checked against its stamp on every read
        except Exception as exc:  # noqa: BLE001
            contain(exc, log=log, what="stale access-shape answers were not deleted from L2", level="warning")

    # ------------------------------------------------------------------
    # DerivedCollection's contract
    # ------------------------------------------------------------------

    def derive_key(self, request: Any) -> Any:
        """``(kind, customer_id, shape)`` onto its key.

        :param request: the triple
        :ptype request: Any
        :return: the key
        :rtype: Any
        """
        kind, customer_id, shape = request
        return self.key_of(kind, customer_id, shape)

    async def load_derived(self, entity_id: Any) -> dict[str, Any] | None:
        """an answer in L2 valid for the question this read asks, or ``None``.

        :param entity_id: the key
        :ptype entity_id: Any
        :return: the row, or ``None``
        :rtype: dict[str, Any] | None
        """
        pending = _PENDING.get()
        row = await self._get_from_l2(entity_id)
        if row is None or pending is None or not self._current(row, pending.context):
            return None
        return row

    async def compute(self, entity_id: Any) -> dict[str, Any] | None:
        """derive the answer as the shape's actor, stamped with what was applied before any read.

        :param entity_id: the key
        :ptype entity_id: Any
        :return: the row, or ``None`` when the compute answered nothing
        :rtype: dict[str, Any] | None
        :raises LookupError: when reached by a read other than :meth:`answer`, which brings the question
        """
        pending = _PENDING.get()
        if pending is None:
            raise LookupError(
                f"{self.table_name}: an answer is derived only through answer(), which brings its question"
            )
        pending.ran = True
        pending.fence = self._dependencies.fence()
        stamp = self._dependencies.stamp()
        actor = self._actors.actor_for(pending.shape)
        footprint: AccessFootprint = await access_footprint(
            self._acl_cache, shape=pending.shape, agent_ids=pending.agent_ids
        )
        body = await pending.compute(actor)
        if body is None:
            return None
        scope, digest = self.normalize_pk(entity_id)
        row = {
            "scope": scope,
            "shape": digest,
            "context": pending.context,
            "stamp": json.dumps(stamp or {}, sort_keys=True),
            "deps": json.dumps(sorted(footprint.deps(self._dependencies.tables))),
            "body": body,
        }
        # kept only when nothing reaching any answer landed meanwhile, and current by what is applied here
        pending.storable = (
            stamp is not None and self._dependencies.fence() == pending.fence and self._current(row, pending.context)
        )
        return row

    async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None, *, conn: Any = None) -> int:
        """put the answer in L2 while the deriving replica holds the key's build lock, unless it may not be kept.

        :param data: the row
        :ptype data: dict[str, Any]
        :param original_timestamp: unused
        :ptype original_timestamp: Any
        :param conn: unused
        :ptype conn: Any
        :return: ``1`` when it was put, else ``0``
        :rtype: int
        """
        del original_timestamp, conn
        pending = _PENDING.get()
        if pending is not None and pending.storable and self._dependencies.fence() == pending.fence:
            pending.saved = await self._save_to_l2((data["scope"], data["shape"]), data)
        return 1 if pending is not None and pending.saved else 0

    async def delete_from_store(self, entity_id: Any) -> None:
        """nothing: there is no durable tier.

        :param entity_id: unused
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        del entity_id

    def serialize(self, data: dict[str, Any]) -> bytes:
        """a row as L2 holds it: JSON.

        :param data: the row
        :ptype data: dict[str, Any]
        :return: JSON bytes
        :rtype: bytes
        """
        return json.dumps({key: data[key] for key in ("scope", "shape", "context", "stamp", "deps", "body")}).encode(
            "utf-8"
        )

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """a row back from L2.

        :param data: JSON bytes
        :ptype data: bytes
        :return: the row
        :rtype: dict[str, Any]
        """
        row: dict[str, Any] = json.loads(data)
        return row
