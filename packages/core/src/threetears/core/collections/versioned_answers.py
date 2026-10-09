"""answers computed once per version of their data, shared by every replica, and retired when the version moves.

A read whose answer is a pure function of (the data's version, the request) -- a report's rows at a
version of its copy, say -- need be computed once per version, by one replica, and read by the rest.
:class:`VersionedAnswers` is that cache: a :class:`~threetears.core.collections.derived.DerivedCollection`
keyed by ``(version, request digest)`` whose value is the answer gzip-compressed, held in L2 alone.

- **L2 only.** The shared ``{ns}-collections`` bucket, under the owner's key scope, is what every
  replica reads; there is no L3 (the answer is derived, and rebuilt on any miss) and no L1
  (:attr:`~BaseCollection.caches_in_l1` is off; a caller keeps its own in-process copy if it wants
  one).
- **Computed once.** A miss computes under the in-process gate and the cross-pod build lock, so one
  replica derives a key and the others wait for its value. A tool pod passes a
  :class:`~threetears.core.collections.derived.LeaseBuildLock`: it may not declare a bucket.
- **Never expires; retired by version, oldest first.** An entry is the answer at one version, so it
  is never stale and carries no lifetime (owner ruling, 2026-10-08: caches are invalidated by the
  epoch system, never by a TTL). Each version comes with an ``order`` that grows with the data (the
  caller's: for a copy of epochs, their sum), because versions are digests and say nothing of which
  is newer. :meth:`current_version` retires every version of a LOWER order, never a higher one, so a
  replica still on an older copy cannot delete the answers the rest have moved on to.
- **Retirement reads an index, never a listing.** A pod's grant on the shared bucket is key-addressed
  (get, put, compare-and-set, delete under its own scope; no consumer), so the entries cannot be
  listed. The owner keeps an index as rows of a small L2-only collection beside the answers, changed
  only by :meth:`~BaseCollection.l2_cas_mutate`: ``versions`` (each version's order, and a ``floor``
  below which no version is recorded any more) and, per version, sixteen shards of request digests
  (``{version}.{first hex digit}``), so no one value is rewritten by every request.
- **Race-free by one ordering.** A computing replica records its version (refused at or below the
  floor) and its digest before it computes, and after its entry lands it reads the floor again: a
  version retired meanwhile has its entry and digest removed by the replica that wrote them.
  Retirement raises the floor first, then empties each shard by compare-and-set (re-reading whatever
  arrived since it read), then forgets the version. So every entry is either reachable from the index
  or removed by its own writer; a retirement interrupted part way leaves the version listed below the
  floor, and the next one finishes it.
- **A failure is never cached, and bookkeeping never fails a read.** ``compute`` raising (a conflict,
  a refusal, data not ready) reaches the caller and leaves nothing behind. The index failing (L2
  unreachable, contention past the retry budget) answers the read uncached and writes no entry, so
  nothing unindexed is ever stored.

**Not bounded here: delete markers.** A KV delete writes a marker, and every answer's key is new, so
the bucket accumulates one marker per retired answer until the bucket's owner purges them; the
retirement keeps the answers themselves from accumulating, not their markers.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Literal

from threetears.core.collections.base import NATS_CLIENT_FROM_REGISTRY, BaseCollection
from threetears.core.collections.derived import BuildLock, DerivedCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError
from threetears.nats.errors import KvError
from threetears.observe import get_logger

__all__ = ["AnswerNotComputable", "VersionedAnswer", "VersionedAnswers"]

log = get_logger(__name__)

#: a version as a key segment: KV-grammar characters without ``_`` (l2_key joins a composite key's
#: parts with it) or ``.`` (a shard's name is ``{version}.{digit}``)
_VERSION: Final = re.compile(r"^[-=a-zA-Z0-9]+$")

#: rounds of re-reading a shard that changed while it was being emptied
_SHARD_ROUNDS: Final = 8

#: the name of the index row holding each version's order and the floor
_VERSIONS: Final = "versions"

#: what a write to the index can raise: transport, a grant refusal, or the retry budget spent
_INDEX_ERRORS: Final = (KvError, ConcurrentModificationError)

_Action = tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]


class AnswerNotComputable(LookupError):
    """a read reached ``compute`` without :meth:`VersionedAnswers.answer`: only it knows how to compute."""


@dataclass
class _Pending:
    """the compute one :meth:`VersionedAnswers.answer` call brought, and whether it ran."""

    compute: Callable[[], Awaitable[str]]
    ran: bool = False


#: the compute of the :meth:`VersionedAnswers.answer` call this task is inside, for ``compute`` to find:
#: a context variable, so concurrent callers never see each other's and a cancelled one leaves nothing
_PENDING: ContextVar[_Pending | None] = ContextVar("versioned_answer_pending", default=None)


class VersionedAnswer(BaseEntity):
    """one answer: ``version``, ``request`` (a digest), ``body`` (base64 gzip)."""

    primary_key_field: str = "request"


class _IndexRow(BaseEntity):
    """one row of the index: ``name``, ``members``, and on ``versions`` the ``floor``."""

    primary_key_field: str = "name"


class _AnswerIndex(BaseCollection[_IndexRow]):
    """the index of a :class:`VersionedAnswers`: rows in L2 alone, changed only by compare-and-set."""

    primary_key_column: str | tuple[str, ...] = ("name",)
    caches_in_l1: ClassVar[bool] = False

    def __init__(self, registry: CollectionRegistry, config: CoreConfig, nats_client: Any, *, table_name: str) -> None:
        self._table_name = table_name
        super().__init__(registry, config, nats_client, None)
        # L2 alone: the index is rebuilt by the answers it indexes, never read through from a store
        self.l3_pool = None

    @property
    def table_name(self) -> str:
        """the table the index's keys are named by.

        :return: the table name
        :rtype: str
        """
        return self._table_name

    @property
    def entity_class(self) -> type[_IndexRow]:
        """the entity class.

        :return: :class:`_IndexRow`
        :rtype: type[_IndexRow]
        """
        return _IndexRow

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """nothing: there is no store behind the index.

        :param entity_id: unused
        :ptype entity_id: Any
        :return: ``None``
        :rtype: dict[str, Any] | None
        """
        del entity_id
        return None

    async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None, *, conn: Any = None) -> int:
        """nothing: there is no store behind the index.

        :param data: unused
        :ptype data: dict[str, Any]
        :param original_timestamp: unused
        :ptype original_timestamp: Any
        :param conn: unused
        :ptype conn: Any
        :return: ``0``
        :rtype: int
        """
        del data, original_timestamp, conn
        return 0

    async def delete_from_store(self, entity_id: Any) -> None:
        """nothing: there is no store behind the index.

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
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """a row back from L2.

        :param data: JSON bytes
        :ptype data: bytes
        :return: the row
        :rtype: dict[str, Any]
        """
        row: dict[str, Any] = json.loads(data)
        return row


class VersionedAnswers(DerivedCollection[VersionedAnswer]):
    """gzip answers keyed by ``(version, request digest)``, in L2 alone, retired when the version moves.

    :param registry: the owner's registry, carrying its L2 client and key scope
    :ptype registry: CollectionRegistry
    :param config: core configuration
    :ptype config: CoreConfig
    :param nats_client: the L2 client; the registry's when omitted
    :ptype nats_client: Any
    :param table_name: the table the keys are named by: one per kind of answer (the index is
        ``{table_name}_index``)
    :ptype table_name: str
    :param build_lock: the cross-pod build lock; a tool pod passes a ``LeaseBuildLock``
    :ptype build_lock: BuildLock | None
    """

    primary_key_column: str | tuple[str, ...] = ("version", "request")
    caches_in_l1: ClassVar[bool] = False

    #: how long a replica waits for a peer computing the same answer before computing it too: an
    #: answer takes up to a few seconds on a large read
    peer_wait_seconds: ClassVar[float] = 15.0

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        nats_client: Any = NATS_CLIENT_FROM_REGISTRY,
        *,
        table_name: str,
        build_lock: BuildLock | None = None,
    ) -> None:
        self._table_name = table_name
        super().__init__(registry, config, nats_client, None, build_lock=build_lock)
        # L2 alone: no durable tier to pull through (a miss computes)
        self.l3_pool = None
        self._index = _AnswerIndex(registry, config, nats_client, table_name=f"{table_name}_index")
        #: the highest order this replica has retired below, and the highest it has been asked to
        self._retired_below: int | None = None
        self._wanted: int | None = None
        self._retiring: asyncio.Task[None] | None = None

    @property
    def table_name(self) -> str:
        """the table the answers' keys are named by.

        :return: the table name
        :rtype: str
        """
        return self._table_name

    @property
    def entity_class(self) -> type[VersionedAnswer]:
        """the entity class.

        :return: :class:`VersionedAnswer`
        :rtype: type[VersionedAnswer]
        """
        return VersionedAnswer

    # ------------------------------------------------------------------
    # the public surface
    # ------------------------------------------------------------------

    async def answer(self, version: str, request: str, compute: Callable[[], Awaitable[str]], *, order: int) -> bytes:
        """the answer to ``request`` at ``version``, gzip-compressed: cached, or computed once and cached.

        :param version: the data's version the answer is at
        :ptype version: str
        :param request: the request, whole (digested for the key): two requests asking the same thing
            must spell it the same
        :ptype request: str
        :param compute: computes the answer's text at exactly ``version``; it raises rather than answer
            at another, and whatever it raises reaches the caller uncached
        :ptype compute: Callable[[], Awaitable[str]]
        :param order: where ``version`` stands among the data's versions: higher is newer
        :ptype order: int
        :return: the answer, gzip bytes
        :rtype: bytes
        :raises ValueError: when ``version`` cannot be a key segment
        """
        key = self.key_of(version, request)
        if not await self._recorded(version, key[1], order):
            # retired, older than what was retired, or the index unreachable: answered, not stored
            return gzip.compress((await compute()).encode("utf-8"), mtime=0)
        pending = _Pending(compute)
        token = _PENDING.set(pending)
        try:
            row = await self.ensure(key)
        finally:
            _PENDING.reset(token)
        if row is None:
            raise RuntimeError(f"{self.table_name}: no answer for {key}; compute returned nothing")
        if pending.ran:
            await self._confirm(version, key[1])
        return base64.b64decode(row["body"])

    def key_of(self, version: str, request: str) -> tuple[str, str]:
        """the key of ``request``'s answer at ``version``.

        :param version: the data's version
        :ptype version: str
        :param request: the request, whole
        :ptype request: str
        :return: ``(version, sha256 hex of request)``
        :rtype: tuple[str, str]
        :raises ValueError: when ``version`` is not KV-grammar characters without ``_`` or ``.``
        """
        if not _VERSION.match(version):
            raise ValueError(f"{version!r} cannot be a version key segment: KV-grammar characters, no '_' or '.'")
        return version, hashlib.sha256(request.encode("utf-8")).hexdigest()

    def current_version(self, version: str, order: int) -> None:
        """note the data's current version; when it is newer than any seen, retire the older ones.

        Cheap to call on every read that learns the version: it does nothing for an order this
        replica has already retired below or asked to, and otherwise starts (or extends) the one
        background retirement. A retirement that fails is tried again on the next call.

        :param version: the version the caller's read is at
        :ptype version: str
        :param order: its order: higher is newer
        :ptype order: int
        :return: nothing
        :rtype: None
        """
        del version
        if self._wanted is not None and order <= self._wanted:
            return
        self._wanted = order
        if self._retiring is None or self._retiring.done():
            self._retiring = asyncio.get_running_loop().create_task(self._retire_until_current())

    async def retire_older_than(self, order: int) -> int:
        """delete every answer this owner holds at a version of a lower order.

        :param order: the order of the version now current
        :ptype order: int
        :return: how many answers were deleted
        :rtype: int
        :raises KvError: when the index or an entry could not be read or changed
        :raises ConcurrentModificationError: when the index kept changing past the retry budget
        """
        taken: dict[str, int] = {}

        def raise_floor(row: dict[str, Any] | None) -> _Action:
            nonlocal taken
            members: dict[str, int] = dict(row["members"]) if row else {}
            floor = row.get("floor") if row else None
            taken = {version: older for version, older in members.items() if older < order}
            new_floor = order - 1 if floor is None else max(floor, order - 1)
            if row is not None and new_floor == floor:
                return ("noop", None)
            return ("upsert", {"name": _VERSIONS, "members": members, "floor": new_floor})

        # first the floor: from here no replica records a version below it, so what is listed is final
        await self._index.l2_cas_mutate(_VERSIONS, raise_floor)
        deleted = 0
        for version, older in sorted(taken.items()):
            for shard in "0123456789abcdef":
                deleted += await self._empty_shard(f"{version}.{shard}")
            await self._index.l2_cas_mutate(_VERSIONS, _forget(version, older))
        if deleted:
            log.info("retired old answers: table=%s below=%d deleted=%d", self.table_name, order, deleted)
        return deleted

    # ------------------------------------------------------------------
    # the index
    # ------------------------------------------------------------------

    async def _recorded(self, version: str, digest: str, order: int) -> bool:
        """record ``version`` and the digest in the index before computing, unless it is retired.

        :param version: the version
        :ptype version: str
        :param digest: the request's digest
        :ptype digest: str
        :param order: the version's order
        :ptype order: int
        :return: ``True`` when the answer may be stored; ``False`` when its version is at or below the
            floor, or the index could not be written (logged)
        :rtype: bool
        """
        refused = False

        def add_version(row: dict[str, Any] | None) -> _Action:
            nonlocal refused
            members: dict[str, int] = dict(row["members"]) if row else {}
            floor = row.get("floor") if row else None
            refused = floor is not None and order <= floor
            if refused or members.get(version) == order:
                return ("noop", None)
            members[version] = order
            return ("upsert", {"name": _VERSIONS, "members": members, "floor": floor})

        try:
            await self._index.l2_cas_mutate(_VERSIONS, add_version)
            if not refused:
                shard = f"{version}.{digest[0]}"
                await self._index.l2_cas_mutate(shard, _add_member(shard, digest))
        except _INDEX_ERRORS as exc:
            # NOSILENT: the read is answered uncached; an answer the index does not name is never stored
            log.warning("answer index unwritable; answering uncached: table=%s error=%s", self.table_name, exc)
            return False
        return not refused

    async def _confirm(self, version: str, digest: str) -> None:
        """after this replica's answer landed: if its version was retired meanwhile, take the answer back.

        :param version: the version
        :ptype version: str
        :param digest: the request's digest
        :ptype digest: str
        :return: nothing
        :rtype: None
        """
        try:
            versions = await self._index.ensure(_VERSIONS)
            floor = versions.get("floor") if versions else None
            if floor is not None and versions is not None and versions["members"].get(version, floor) <= floor:
                await self.l2_cas_mutate((version, digest), _delete_present)
                shard = f"{version}.{digest[0]}"
                await self._index.l2_cas_mutate(shard, _remove_member(shard, digest))
        except _INDEX_ERRORS as exc:
            # NOSILENT: the one window left open is this replica's own answer outliving its version
            log.warning(
                "answer landed on a retired version and could not be taken back: table=%s version=%s error=%s",
                self.table_name,
                version,
                exc,
            )

    async def _empty_shard(self, name: str) -> int:
        """delete every answer a shard names, then the shard, re-reading whatever arrived meanwhile.

        :param name: the shard's name, ``{version}.{digit}``
        :ptype name: str
        :return: how many answers were deleted
        :rtype: int
        """
        version = name.split(".", 1)[0]
        deleted = 0
        for _ in range(_SHARD_ROUNDS):
            row = await self._index.ensure(name)
            seen = frozenset(row["members"]) if row else frozenset()
            for digest in sorted(seen):
                outcome = await self.l2_cas_mutate((version, digest), _delete_present)
                deleted += outcome.action == "deleted"
            emptied = False

            def drop_if_unchanged(current: dict[str, Any] | None, seen: frozenset[str] = seen) -> _Action:
                nonlocal emptied
                emptied = current is None or frozenset(current["members"]) <= seen
                return ("delete", None) if emptied else ("noop", None)

            await self._index.l2_cas_mutate(name, drop_if_unchanged)
            if emptied:
                break
        return deleted

    async def _retire_until_current(self) -> None:
        """retire below the highest order asked for, until it is reached or a retirement fails.

        :return: nothing
        :rtype: None
        """
        while self._wanted is not None and (self._retired_below is None or self._wanted > self._retired_below):
            target = self._wanted
            try:
                await self.retire_older_than(target)
            except _INDEX_ERRORS as exc:
                # NOSILENT: the old answers wait for the next call, which tries again
                log.warning("retiring old answers failed: table=%s below=%d error=%s", self.table_name, target, exc)
                self._wanted = self._retired_below
                return
            self._retired_below = target

    # ------------------------------------------------------------------
    # DerivedCollection's contract
    # ------------------------------------------------------------------

    def derive_key(self, request: Any) -> Any:
        """a ``(version, request)`` pair onto its key.

        :param request: ``(version, request text)``
        :ptype request: Any
        :return: the key
        :rtype: Any
        """
        version, text = request
        return self.key_of(version, text)

    async def load_derived(self, entity_id: Any) -> dict[str, Any] | None:
        """the answer a peer has put in L2, or ``None``.

        :param entity_id: the key
        :ptype entity_id: Any
        :return: the row, or ``None``
        :rtype: dict[str, Any] | None
        """
        return await self._get_from_l2(entity_id)

    async def compute(self, entity_id: Any) -> dict[str, Any] | None:
        """compute the answer for ``entity_id`` with the calling :meth:`answer`'s ``compute``, and compress it.

        :param entity_id: the key
        :ptype entity_id: Any
        :return: the row
        :rtype: dict[str, Any] | None
        :raises AnswerNotComputable: when reached by a read other than :meth:`answer` (``get``,
            ``get_for``), which brings no way to compute
        """
        pending = _PENDING.get()
        if pending is None:
            raise AnswerNotComputable(
                f"{self.table_name}: an answer is computed only through answer(), which brings its compute"
            )
        pending.ran = True
        version, digest = self.normalize_pk(entity_id)
        text = await pending.compute()
        body = gzip.compress(text.encode("utf-8"), mtime=0)
        return {"version": version, "request": digest, "body": base64.b64encode(body).decode("ascii")}

    async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None, *, conn: Any = None) -> int:
        """nothing: L2 is the only tier, and the read that computed the row seeds it there.

        :param data: the row
        :ptype data: dict[str, Any]
        :param original_timestamp: unused
        :ptype original_timestamp: Any
        :param conn: unused
        :ptype conn: Any
        :return: ``0``
        :rtype: int
        """
        del data, original_timestamp, conn
        return 0

    async def delete_from_store(self, entity_id: Any) -> None:
        """nothing: there is no durable tier; retirement deletes from L2.

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
        return json.dumps({key: data[key] for key in ("version", "request", "body")}).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """a row back from L2.

        :param data: JSON bytes
        :ptype data: bytes
        :return: the row
        :rtype: dict[str, Any]
        """
        row: dict[str, Any] = json.loads(data)
        return row


def _delete_present(row: dict[str, Any] | None) -> _Action:
    """delete the row when there is one.

    :param row: the row, or ``None``
    :ptype row: dict[str, Any] | None
    :return: the action
    :rtype: _Action
    """
    return ("delete", None) if row is not None else ("noop", None)


def _add_member(name: str, member: str) -> Callable[[dict[str, Any] | None], _Action]:
    """a shard change adding ``member``.

    :param name: the shard's name
    :ptype name: str
    :param member: the digest
    :ptype member: str
    :return: the change
    :rtype: Callable[[dict[str, Any] | None], _Action]
    """

    def change(row: dict[str, Any] | None) -> _Action:
        members: list[str] = list(row["members"]) if row else []
        if member in members:
            return ("noop", None)
        return ("upsert", {"name": name, "members": [*members, member]})

    return change


def _remove_member(name: str, member: str) -> Callable[[dict[str, Any] | None], _Action]:
    """a shard change removing ``member``, deleting the shard when it empties.

    :param name: the shard's name
    :ptype name: str
    :param member: the digest
    :ptype member: str
    :return: the change
    :rtype: Callable[[dict[str, Any] | None], _Action]
    """

    def change(row: dict[str, Any] | None) -> _Action:
        members: list[str] = list(row["members"]) if row else []
        if member not in members:
            return ("noop", None)
        rest = [kept for kept in members if kept != member]
        return ("delete", None) if not rest else ("upsert", {"name": name, "members": rest})

    return change


def _forget(version: str, order: int) -> Callable[[dict[str, Any] | None], _Action]:
    """a ``versions`` change taking ``version`` off, once its shards are empty.

    :param version: the version retired
    :ptype version: str
    :param order: its order, as retirement read it
    :ptype order: int
    :return: the change
    :rtype: Callable[[dict[str, Any] | None], _Action]
    """

    def change(row: dict[str, Any] | None) -> _Action:
        members: dict[str, int] = dict(row["members"]) if row else {}
        if members.get(version) != order or row is None:
            return ("noop", None)
        del members[version]
        return ("upsert", {"name": _VERSIONS, "members": members, "floor": row.get("floor")})

    return change
