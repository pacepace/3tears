"""collection whose key is *derived* from a request and whose value is *computed*.

:class:`BaseCollection` caches rows by primary key. that serves reads whose
identity is already discrete -- a user id, a conversation id -- and does not
serve reads whose identity is continuous. a bounding box, an arbitrary time
window, or an offset/limit page names a region of a space rather than a row,
so no two callers produce the same cache key and the hit rate across pods is
zero. every such read in this codebase is currently annotated
``# cache-bypass: ... not by-pk`` and goes straight to L3.

the fix is not a different cache. it is **quantization**: collapse the
continuous request onto a discrete grid, and the grid cell becomes a primary
key that the existing three tiers already handle unmodified. a geographic
tile (``z/x/y``) is one instance; an hour bucket and a fixed-size page are
others.

this class supplies the two things that quantization needs and
:class:`BaseCollection` does not have:

- :meth:`derive_key` -- the request-to-key contract, so the quantization is
  declared in one place instead of being open-coded by each caller (and
  therefore disagreed on by each caller).
- a **compute-on-miss** :meth:`fetch_from_store`, because a derived value has
  no row waiting in L3 the first time it is asked for. the miss path is
  single-flighted twice over: an in-process :class:`asyncio.Lock` so
  concurrent tasks on one pod compute once, and
  :func:`~threetears.nats.nats_distributed_lock` so concurrent *pods* compute
  once. derivation is typically the expensive step -- if it were cheap there
  would be no reason to cache it -- so an unguarded miss on a popular key is a
  stampede.

subclasses implement :meth:`derive_key`, :meth:`compute`, and
:meth:`load_derived`, plus :class:`BaseCollection`'s own
:meth:`~BaseCollection.save_to_store` /
:meth:`~BaseCollection.delete_from_store` / serialization hooks. note that
the durable tier here is pluggable exactly as it is on the base class: a
derived value may live in a table, in an object store, or anywhere else that
can answer "do you already have this key".

the value is a pure function of the key. a subclass whose ``compute`` depends
on mutable source data must therefore fold the source generation into the key
(see the ``source_version`` discussion in the geo tile design), otherwise a
cached value outlives the inputs that justified it.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, ClassVar, Generic

from threetears.core.collections.base import BaseCollection, EntityT
from threetears.nats import LockHeld, nats_distributed_lock
from threetears.observe import get_logger, traced

__all__ = ["DerivedCollection"]

log = get_logger(__name__)


class DerivedCollection(BaseCollection[EntityT], Generic[EntityT]):
    """three-tier collection over values computed from a quantized key.

    see the module docstring for why this exists. the constructor signature is
    :class:`BaseCollection`'s, unchanged.
    """

    #: JetStream KV bucket holding the cross-pod build locks. a bucket pins
    #: one TTL for every key in it (see :func:`nats_distributed_lock`), so a
    #: subclass needing a different build timeout declares its own bucket
    #: rather than passing a different ttl into the shared one.
    build_lock_bucket: ClassVar[str] = "derived-build-locks"

    #: grace given to a peer pod that holds the build lock before deriving
    #: locally anyway. bounded deliberately -- see :meth:`_await_peer_derivation`.
    peer_grace_seconds: ClassVar[float] = 0.25

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # per-key in-process gate, dropped once nobody holds or awaits it, so
        # this does not grow with the number of keys ever seen. the lock's own
        # ``locked()`` is the liveness signal -- a separate reference count
        # would be a second source of truth for the same fact.
        self._inflight: dict[tuple[Any, ...], asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def derive_key(self, request: Any) -> Any:
        """quantize a continuous request onto this collection's key grid.

        the whole point of the class: a caller hands over the thing it
        actually has (a viewport, an instant, an offset) and receives the
        discrete key that stands for it. two requests that fall in the same
        cell MUST produce equal keys, or the cache cannot share anything
        between them.

        implementations must be pure and total -- no I/O, no clock reads --
        because the key is computed on every request including cache hits.

        :param request: caller-domain request (subclass-defined shape)
        :ptype request: Any
        :return: pk value (single-pk) or tuple of pk values (composite-pk),
            in the shape :meth:`~BaseCollection.normalize_pk` accepts
        :rtype: Any
        """
        ...

    @abstractmethod
    async def load_derived(self, entity_id: Any) -> dict[str, Any] | None:
        """read an already-derived value from the durable tier, or ``None``.

        the cheap existence check that decides whether :meth:`compute` has to
        run. called on the L1+L2 miss path, and again after each single-flight
        gate is acquired, since a peer task or peer pod may have derived the
        value while this caller waited.

        must not derive anything itself -- returning ``None`` is how this
        method says "not built yet".

        :param entity_id: pk value or tuple of pk values
        :ptype entity_id: Any
        :return: row data on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        ...

    @abstractmethod
    async def compute(self, entity_id: Any) -> dict[str, Any] | None:
        """derive the value for ``entity_id``.

        called at most once per key per pod at a time, under both single-flight
        gates. returning ``None`` means the key names nothing derivable (an
        out-of-range tile, an empty bucket) and is cached as a miss rather than
        retried on every request.

        the returned dict must carry the pk columns named in
        :attr:`~BaseCollection.primary_key_columns`, since the framework and
        :meth:`~BaseCollection.save_to_store` both key off them.

        :param entity_id: pk value or tuple of pk values
        :ptype entity_id: Any
        :return: derived row data, or ``None`` if nothing is derivable
        :rtype: dict[str, Any] | None
        """
        ...

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    @traced
    async def get_for(self, request: Any) -> EntityT | None:
        """resolve a caller-domain request through the full three-tier path.

        equivalent to ``await collection.get(collection.derive_key(request))``;
        provided so callers never handle the derived key themselves and
        therefore cannot quantize it inconsistently.

        :param request: caller-domain request (subclass-defined shape)
        :ptype request: Any
        :return: entity on hit, ``None`` when nothing is derivable
        :rtype: EntityT | None
        """
        return await self.get(self.derive_key(request))

    # ------------------------------------------------------------------
    # compute-on-miss durable tier
    # ------------------------------------------------------------------

    @traced
    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """return the derived value, computing it if the durable tier lacks it.

        this is :class:`BaseCollection`'s L3 read hook, so the promotion into
        L2 and L1 that follows a hit is the base class's and is unchanged --
        the only difference is that a durable miss here builds rather than
        reporting absence.

        :param entity_id: pk value or tuple of pk values
        :ptype entity_id: Any
        :return: row data, or ``None`` when nothing is derivable
        :rtype: dict[str, Any] | None
        """
        key = self.normalize_pk(entity_id)
        existing = await self.load_derived(key)
        if existing is not None:
            return existing
        return await self._derive_single_flight(key)

    async def _derive_single_flight(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        """run :meth:`compute` for ``key`` at most once per pod at a time."""
        gate = self._inflight.get(key)
        if gate is None:
            gate = asyncio.Lock()
            self._inflight[key] = gate
        try:
            async with gate:
                # a peer task on this pod may have derived it while we queued.
                existing = await self.load_derived(key)
                if existing is not None:
                    return existing
                return await self._derive_cross_pod(key)
        finally:
            # only the last leaver sees an unlocked gate; anyone still queued
            # holds a reference to this same object and is unaffected either way.
            if not gate.locked():
                self._inflight.pop(key, None)

    async def _derive_cross_pod(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        """hold the cross-pod build lock, then derive and persist."""
        lock_key = self.build_lock_key(key)
        try:
            async with nats_distributed_lock(
                self._nats_client,
                lock_key,
                bucket_name=self.build_lock_bucket,
            ):
                # a peer POD may have derived it while we queued.
                existing = await self.load_derived(key)
                if existing is not None:
                    return existing
                derived = await self.compute(key)
                if derived is not None:
                    await self.save_to_store(derived)
                return derived
        except LockHeld:
            return await self._await_peer_derivation(key)

    async def _await_peer_derivation(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        """give a peer pod's in-progress derivation one grace period, then derive.

        a single bounded wait rather than a poll loop. the grace exists because
        the common case is a peer that is merely a moment ahead of us, and
        re-reading once is far cheaper than duplicating an expensive
        derivation. when the value still is not there, deriving locally is the
        correct trade: a peer that died mid-build holds its lock until the KV
        TTL expires, and no caller should inherit that latency. the cost of
        being wrong is one duplicated compute; the cost of waiting on a dead
        peer is a stalled request.

        subclasses whose derivations are unusually cheap or unusually expensive
        can override :data:`peer_grace_seconds` to shift that trade.
        """
        await asyncio.sleep(self.peer_grace_seconds)
        existing = await self.load_derived(key)
        if existing is not None:
            return existing
        log.warning(
            "derived value did not land from peer pod within %.2fs, deriving locally: table=%s key=%s",
            self.peer_grace_seconds,
            self.table_name,
            key,
        )
        derived = await self.compute(key)
        if derived is not None:
            await self.save_to_store(derived)
        return derived

    def build_lock_key(self, entity_id: Any) -> str:
        """cross-pod lock key for one derived key.

        namespaced by table so two collections quantizing onto similar grids
        cannot collide in the shared bucket. overridable for subclasses whose
        pk values do not render usefully with ``str``.

        :param entity_id: pk value or tuple of pk values
        :ptype entity_id: Any
        :return: lock key
        :rtype: str
        """
        parts = "/".join(str(part) for part in self.normalize_pk(entity_id))
        return f"{self.table_name}/{parts}"

    @property
    def inflight_derivations(self) -> int:
        """number of keys currently being derived or queued on this pod.

        operational visibility: a number that stays high under steady load
        means derivations are outpacing requests, and a number that grows
        without bound means the gate is not being released.

        :return: count of live in-process derivation gates
        :rtype: int
        """
        return len(self._inflight)
