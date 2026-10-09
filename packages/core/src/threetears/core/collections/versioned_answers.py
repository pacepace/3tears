"""answers computed once per version of their data, shared by every replica, and retired when the version moves.

A read whose answer is a pure function of (the data's version, the request) -- a report's rows at a
version of its copy, say -- need be computed once per version, by one replica, and read by the rest.
:class:`VersionedAnswers` is that cache: a :class:`~threetears.core.collections.derived.DerivedCollection`
keyed by ``(version, request digest)`` whose value is the answer gzip-compressed, held in L2 alone.

- **L2 only.** The shared ``{ns}-collections`` bucket, under the owner's key scope, is what every
  replica reads; there is no L3 (the answer is derived, and rebuilt on any miss) and no L1 (a caller
  keeps its own in-process copy if it wants one). A refused or failed L2 write costs a recompute,
  never a wrong answer.
- **Computed once.** A miss computes under the in-process gate and the cross-pod build lock, so one
  replica derives a key and the others wait for its value (:class:`DerivedCollection`). A tool pod
  passes a :class:`~threetears.core.collections.derived.LeaseBuildLock`: it may not declare a bucket.
- **Never expires; retired by version.** An entry is the answer at one version, so it is never stale
  and carries no lifetime. When the data's version moves, :meth:`retire_all_but` deletes every entry
  of another version (:meth:`current_version` schedules it once per change seen). A replica still on
  the old version may compute an old entry again after that; the next retirement takes it.
- **Retirement reads an index, never a listing.** A pod's grant on the shared bucket is key-addressed
  (get, put, compare-and-set, delete under its own scope; no consumer), so the entries cannot be
  listed. Each owner keeps, under its scope, the set of versions it holds answers for
  (``{table}.versions``) and per version the set of request digests (``{table}.index.{version}``),
  each a JSON list changed by compare-and-set; a computing replica records its key before it
  computes, so an entry is never left out of the index.
- **A failure is never cached.** ``compute`` raising (a conflict, a refusal, data not ready) reaches
  the caller and leaves nothing behind.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
import random
import re
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Final

from threetears.core.collections.base import NATS_CLIENT_FROM_REGISTRY
from threetears.core.collections.derived import BuildLock, DerivedCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.nats.errors import KvError
from threetears.observe import get_logger

__all__ = ["VersionedAnswer", "VersionedAnswers"]

log = get_logger(__name__)

#: a version as a key segment: KV-grammar characters, and no ``_``, which joins the key's two parts, and
#: no ``.``, which would make ``index.{version}`` more than one token
_VERSION: Final = re.compile(r"^[-=a-zA-Z0-9]+$")

#: compare-and-set rounds on an index set before giving up, and the jitter between them
_INDEX_ATTEMPTS: Final = 20
_INDEX_BACKOFF_SECONDS: Final = 0.02


class VersionedAnswer(BaseEntity):
    """one answer: ``version``, ``request`` (a digest), ``body`` (base64 gzip)."""

    primary_key_field: str = "request"


class VersionedAnswers(DerivedCollection[VersionedAnswer]):
    """gzip answers keyed by ``(version, request digest)``, in L2 alone, retired when the version moves.

    :param registry: the owner's registry, carrying its L2 client and key scope
    :ptype registry: CollectionRegistry
    :param config: core configuration
    :ptype config: CoreConfig
    :param nats_client: the L2 client; the registry's when omitted
    :ptype nats_client: Any
    :param table_name: the table the keys are named by: one per kind of answer
    :ptype table_name: str
    :param build_lock: the cross-pod build lock; a tool pod passes a ``LeaseBuildLock``
    :ptype build_lock: BuildLock | None
    """

    primary_key_column: str | tuple[str, ...] = ("version", "request")

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
        # L2 alone: no durable tier to pull through (a miss computes), and no L1 copy to evict
        self.l3_pool = None
        self._l1 = None
        self._computing: dict[tuple[Any, ...], Callable[[], Awaitable[str]]] = {}
        self._current: str | None = None
        self._retiring: set[asyncio.Task[int]] = set()

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

    async def answer(self, version: str, request: str, compute: Callable[[], Awaitable[str]]) -> bytes:
        """the answer to ``request`` at ``version``, gzip-compressed: cached, or computed once and cached.

        :param version: the data's version the answer is at
        :ptype version: str
        :param request: the request, whole (digested for the key): two requests asking the same thing
            must spell it the same
        :ptype request: str
        :param compute: computes the answer's text at exactly ``version``; it raises rather than answer
            at another, and whatever it raises reaches the caller uncached
        :ptype compute: Callable[[], Awaitable[str]]
        :return: the answer, gzip bytes
        :rtype: bytes
        :raises ValueError: when ``version`` cannot be a key segment
        """
        key = self.key_of(version, request)

        async def recorded() -> str:
            # in the index before the entry exists, so retirement can always find it
            await self._add_to_set(self._index_key("versions"), version)
            await self._add_to_set(self._index_key(f"index.{version}"), key[1])
            return await compute()

        # any caller's compute serves the key: the answer is a function of the key alone
        self._computing[key] = recorded
        try:
            row = await self.ensure(key)
        finally:
            if self._computing.get(key) is recorded:
                del self._computing[key]
        if row is None:
            raise RuntimeError(f"{self.table_name}: no answer for {key}; compute returned nothing")
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
            raise ValueError(f"{version!r} cannot be a version key segment: KV-grammar characters, no '_' and no '.'")
        return version, hashlib.sha256(request.encode("utf-8")).hexdigest()

    def current_version(self, version: str) -> None:
        """note the data's current version; when it moved, retire every other version's answers.

        Cheap to call on every read that learns the version: it does nothing until the version
        changes, and then schedules one retirement in the background.

        :param version: the version the caller's read is at
        :ptype version: str
        :return: nothing
        :rtype: None
        """
        if version == self._current:
            return
        self._current = version
        task = asyncio.get_running_loop().create_task(self.retire_all_but(version))
        self._retiring.add(task)
        task.add_done_callback(self._retiring.discard)

    async def retire_all_but(self, version: str) -> int:
        """delete every answer this owner holds at a version other than ``version``.

        Reads the owner's index (see the module docstring): each other version's digests, each of
        their entries deleted, then the version's index, then the version taken off the set.

        :param version: the version to keep
        :ptype version: str
        :return: how many answers were deleted
        :rtype: int
        """
        deleted = 0
        try:
            kv = await self._ensure_kv()
            if kv is not None:
                versions_key = self._index_key("versions")
                held, _ = await self._read_set(versions_key)
                for old in sorted(held - {version}):
                    index_key = self._index_key(f"index.{old}")
                    digests, _ = await self._read_set(index_key)
                    for digest in sorted(digests):
                        if await kv.delete(key=self.l2_key((old, digest))):
                            deleted += 1
                    await kv.delete(key=index_key)
                    await self._remove_from_set(versions_key, old)
        except KvError as exc:
            # NOSILENT: an answer left behind is only space until the next retirement; never wrong
            log.warning("retiring old answers failed: table=%s keep=%s error=%s", self.table_name, version, exc)
        if deleted:
            log.info("retired old answers: table=%s keep=%s deleted=%d", self.table_name, version, deleted)
        return deleted

    def _index_key(self, name: str) -> str:
        """an index key under this owner's scope: ``{scope}.{table}.{name}``.

        :param name: ``versions`` or ``index.{version}``
        :ptype name: str
        :return: the key
        :rtype: str
        """
        return f"{self._registry.kv_key_scope}.{self.table_name}.{name}"

    async def _read_set(self, key: str) -> tuple[set[str], int]:
        """an index set and the revision it was read at (``0`` when the key never held one).

        :param key: the index key
        :ptype key: str
        :return: the members and the revision
        :rtype: tuple[set[str], int]
        """
        kv = await self._ensure_kv()
        if kv is None:
            return set(), 0
        value, revision = await kv.get_latest(key=key)
        members: set[str] = set(json.loads(value)) if value else set()
        return members, revision

    async def _change_set(self, key: str, member: str, *, add: bool) -> None:
        """add ``member`` to the index set at ``key``, or remove it, by compare-and-set.

        :param key: the index key
        :ptype key: str
        :param member: the member
        :ptype member: str
        :param add: whether to add (else remove)
        :ptype add: bool
        :return: nothing
        :rtype: None
        :raises KvError: when the set kept changing under every attempt
        """
        kv = await self._ensure_kv()
        if kv is None:
            return
        for _ in range(_INDEX_ATTEMPTS):
            members, revision = await self._read_set(key)
            if (member in members) == add:
                return
            changed = members | {member} if add else members - {member}
            value = json.dumps(sorted(changed)).encode("utf-8")
            if await kv.update(key=key, value=value, revision=revision) is not None:
                return
            await asyncio.sleep(random.uniform(0, _INDEX_BACKOFF_SECONDS))  # noqa: S311 - jitter, not secrecy
        raise KvError(f"{key}: the index kept changing; {member!r} not {'added' if add else 'removed'}")

    async def _add_to_set(self, key: str, member: str) -> None:
        """add ``member`` to the index set at ``key``.

        :param key: the index key
        :ptype key: str
        :param member: the member
        :ptype member: str
        :return: nothing
        :rtype: None
        """
        await self._change_set(key, member, add=True)

    async def _remove_from_set(self, key: str, member: str) -> None:
        """remove ``member`` from the index set at ``key``.

        :param key: the index key
        :ptype key: str
        :param member: the member
        :ptype member: str
        :return: nothing
        :rtype: None
        """
        await self._change_set(key, member, add=False)

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
        """compute the answer for ``entity_id`` with the waiting caller's ``compute``, and compress it.

        :param entity_id: the key
        :ptype entity_id: Any
        :return: the row
        :rtype: dict[str, Any] | None
        """
        version, digest = self.normalize_pk(entity_id)
        compute = self._computing[(version, digest)]
        text = await compute()
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
        """nothing: there is no durable tier; :meth:`retire_all_but` deletes from L2.

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
