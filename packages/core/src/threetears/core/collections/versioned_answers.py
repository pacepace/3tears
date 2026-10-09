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
- **A failure is never cached.** ``compute`` raising (a conflict, a refusal, data not ready) reaches
  the caller and leaves nothing behind.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
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

#: a version as a key segment: KV-grammar characters, and no ``_``, which joins the key's two parts
_VERSION: Final = re.compile(r"^[-=.a-zA-Z0-9]+$")


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
        # any caller's compute serves the key: the answer is a function of the key alone
        self._computing[key] = compute
        try:
            row = await self.ensure(key)
        finally:
            if self._computing.get(key) is compute:
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
        :raises ValueError: when ``version`` is not KV-grammar characters without ``_``
        """
        if not _VERSION.match(version):
            raise ValueError(f"{version!r} cannot be a version key segment: KV-grammar characters, no '_'")
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

        :param version: the version to keep
        :ptype version: str
        :return: how many answers were deleted
        :rtype: int
        """
        scope = self._registry.kv_key_scope
        deleted = 0
        try:
            kv = await self._ensure_kv()
            if kv is None or scope is None:
                return 0
            # a prefix ending on a token boundary: the server filters to this owner's own keys, the
            # one listing a pod's grant admits
            prefix = f"{scope}.{self.table_name}."
            for key in await kv.list_keys(prefix=prefix):
                kept_version = key[len(prefix) :].split("_", 1)[0]
                if kept_version != version and await kv.delete(key=key):
                    deleted += 1
        except KvError as exc:
            # NOSILENT: an answer left behind is only space until the next retirement; never wrong
            log.warning("retiring old answers failed: table=%s keep=%s error=%s", self.table_name, version, exc)
        if deleted:
            log.info("retired old answers: table=%s keep=%s deleted=%d", self.table_name, version, deleted)
        return deleted

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
