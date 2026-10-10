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
  concurrent tasks on one pod compute once, and a cross-pod :class:`BuildLock`
  so concurrent *pods* compute once. derivation is typically the expensive step -- if it were cheap there
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
import hashlib
import re
from abc import abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Final, Generic, Protocol

from threetears.core.collections.base import BaseCollection, EntityT
from threetears.observe import get_logger, traced

if TYPE_CHECKING:
    from threetears.core.coordination.lease import KVLease
    from threetears.nats.kv import KvCapable

__all__ = ["BuildLock", "BuildLockHeld", "DerivedCollection", "LeaseBuildLock", "NatsBuildLock"]

log = get_logger(__name__)

#: the JetStream KV key grammar; a lease key outside it is hashed
_KV_KEY_GRAMMAR: Final = re.compile(r"^[-/_=.a-zA-Z0-9]+$")


class BuildLockHeld(Exception):
    """another pod holds the build lock on this key: wait for its value rather than derive it too."""


class BuildLock(Protocol):
    """the cross-pod build lock: held while one pod derives a key, so the others wait for its value."""

    def holding(self, key: str) -> AbstractAsyncContextManager[None]:
        """hold the lock on ``key`` for the body.

        :param key: the lock key (:meth:`DerivedCollection.build_lock_key`)
        :ptype key: str
        :return: a context manager holding the lock while it is entered
        :rtype: AbstractAsyncContextManager[None]
        :raises BuildLockHeld: on entry, when another holder has it
        """
        ...


class NatsBuildLock:
    """the build lock on a bucket of its own (:func:`~threetears.core.coordination.nats_distributed_lock`), declared on first use.

    The default, for an infrastructure identity that may declare a bucket. A tool pod may not -- its
    grant holds no stream-management verb -- so a pod's collection takes a :class:`LeaseBuildLock` over
    a bucket the hub declared.

    :param client: the connected NATS client
    :ptype client: KvCapable
    :param bucket_name: the lock bucket's suffix; it pins one TTL for every key in it
    :ptype bucket_name: str
    """

    def __init__(self, client: KvCapable, bucket_name: str) -> None:
        self._client = client
        self._bucket_name = bucket_name

    @asynccontextmanager
    async def holding(self, key: str) -> AsyncIterator[None]:
        """hold ``key`` for the body.

        :param key: the lock key
        :ptype key: str
        :return: an iterator yielding once, while the lock is held
        :rtype: AsyncIterator[None]
        :raises BuildLockHeld: when another holder has it
        """
        from threetears.core.coordination import LockHeld, nats_distributed_lock

        try:
            # cancel_on_loss=False: the build lock only stops a stampede of identical
            # derivations; losing it mid-build costs one duplicate compute, and interrupting
            # the build would fail the read that is waiting on it. LockHeld is raised only on
            # entry: the body is a derivation, which takes no lock of this kind.
            async with nats_distributed_lock(self._client, key, bucket_name=self._bucket_name, cancel_on_loss=False):
                yield
        except LockHeld as exc:
            raise BuildLockHeld(key) from exc


class LeaseBuildLock:
    """the build lock as a lease (:class:`~threetears.core.coordination.lease.KVLease`) in a bucket another identity declared.

    What a tool pod's derived collection takes: the pod binds the hub-declared ``leases`` bucket
    (``create_if_missing=False``) under its own key scope, so replicas of one pod contend for one key
    and no other pod can see or release it. The lease is never renewed: a build outliving
    ``ttl_seconds`` lets one more pod derive the same value, which is what the lock exists to make
    rare, not impossible. A key outside the KV grammar is hashed.

    :param lease: the lease factory, bound to its bucket and key scope
    :ptype lease: KVLease
    :param ttl_seconds: how long a holder that died keeps the others waiting
    :ptype ttl_seconds: int
    """

    def __init__(self, lease: KVLease, *, ttl_seconds: int = 60) -> None:
        self._lease = lease
        self._ttl_seconds = ttl_seconds

    @asynccontextmanager
    async def holding(self, key: str) -> AsyncIterator[None]:
        """hold ``key`` for the body, refusing at once when it is held.

        :param key: the lock key
        :ptype key: str
        :return: an iterator yielding once, while the lease is held
        :rtype: AsyncIterator[None]
        :raises BuildLockHeld: when another holder has it
        """
        from threetears.core.coordination.lease import LeaseUnavailable

        name = key if _KV_KEY_GRAMMAR.match(key) else hashlib.sha256(key.encode("utf-8")).hexdigest()
        try:
            handle = await self._lease.acquire(name, ttl_seconds=self._ttl_seconds, max_wait_seconds=0)
        except LeaseUnavailable as exc:
            raise BuildLockHeld(key) from exc
        async with handle:
            yield


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

    #: total time a pod that lost the build lock waits for the winner's result
    #: before deriving locally. must be on the scale of a derivation, since a
    #: budget shorter than one guarantees every loser duplicates the winner's
    #: work -- see :meth:`_await_peer_derivation`.
    peer_wait_seconds: ClassVar[float] = 5.0

    #: how often the loser re-reads the durable tier while waiting. short, so
    #: a fast peer releases the waiter promptly.
    peer_poll_interval: ClassVar[float] = 0.1

    def __init__(self, *args: Any, build_lock: BuildLock | None = None, **kwargs: Any) -> None:
        """build the collection as :class:`BaseCollection` does, with the cross-pod build lock.

        :param args: :class:`BaseCollection`'s positional arguments
        :ptype args: Any
        :param build_lock: the cross-pod build lock; ``None`` takes a :class:`NatsBuildLock` on
            :attr:`build_lock_bucket` when there is a NATS client, and none without one
        :ptype build_lock: BuildLock | None
        :param kwargs: :class:`BaseCollection`'s keyword arguments
        :ptype kwargs: Any
        """
        super().__init__(*args, **kwargs)
        if build_lock is None and self._nats_client is not None:
            build_lock = NatsBuildLock(self._nats_client, self.build_lock_bucket)
        self._build_lock = build_lock
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
        """hold the cross-pod build lock, then derive and persist.

        with no build lock there is no peer pod to coordinate with: the
        in-process gate the caller holds is the whole single-flight, so the
        value is derived directly.
        """
        if self._build_lock is None:
            return await self._derive_and_save(key)
        try:
            async with self._build_lock.holding(self.build_lock_key(key)):
                # a peer POD may have derived it while we queued.
                return await self._derive_and_save(key)
        except BuildLockHeld:
            return await self._await_peer_derivation(key)

    async def _derive_and_save(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        """derive the value unless it has landed meanwhile, and persist it."""
        existing = await self.load_derived(key)
        if existing is not None:
            return existing
        derived = await self.compute(key)
        if derived is not None:
            await self.save_to_store(derived)
        return derived

    async def _await_peer_derivation(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        """wait for a peer pod's in-progress derivation, then derive if it never lands.

        a **bounded retry**, not a background pump: it re-reads the durable
        tier at :data:`peer_poll_interval` until either the value appears or
        :data:`peer_wait_seconds` is spent. the budget has to be on the scale
        of a derivation, because a derivation is expensive by definition -- if
        it were cheap there would be nothing to cache and no lock to contend
        for. an earlier revision waited a single sub-second grace, which meant
        every loser duplicated the winner's work: the exact stampede the
        cross-pod lock exists to prevent, reintroduced underneath it. that bug
        was invisible to unit tests, which pass ``nats_client=None`` and never
        take the lock at all.

        deriving locally after the budget expires is the deliberate trade. a
        peer that died mid-build holds its lock until the KV TTL expires
        (60s by default), and no caller should inherit that latency. one
        duplicated compute costs CPU; waiting on a dead peer stalls a request.
        """
        attempts = max(1, int(self.peer_wait_seconds / self.peer_poll_interval))
        for _ in range(attempts):
            await asyncio.sleep(self.peer_poll_interval)
            existing = await self.load_derived(key)
            if existing is not None:
                return existing
        log.warning(
            "derived value did not land from peer pod within %.2fs, deriving locally: table=%s key=%s",
            self.peer_wait_seconds,
            self.table_name,
            key,
        )
        derived = await self.compute(key)
        if derived is not None:
            await self.save_to_store(derived)
        return derived

    @asynccontextmanager
    async def derivation_paused(self, entity_id: Any) -> AsyncIterator[None]:
        """hold off any derivation of ``entity_id`` for the body, or refuse when one is running.

        For a caller that must act on a key's derived value knowing no derivation of it is in flight
        -- one deciding a value that is absent will stay absent. Holds this pod's in-process gate and
        the cross-pod build lock, so a derivation on any pod either finished before or starts after.

        :param entity_id: pk value or tuple of pk values
        :ptype entity_id: Any
        :return: an iterator yielding once, while no derivation of the key can run
        :rtype: AsyncIterator[None]
        :raises BuildLockHeld: when a derivation of the key is running, here or on another pod
        """
        key = self.normalize_pk(entity_id)
        gate = self._inflight.get(key)
        if gate is not None and gate.locked():
            raise BuildLockHeld(self.build_lock_key(key))
        if self._build_lock is None:
            yield
        else:
            async with self._build_lock.holding(self.build_lock_key(key)):
                yield

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
