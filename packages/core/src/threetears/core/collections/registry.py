"""Collection registry — DI container + table_name lookup + cache coherence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ValidationError
from threetears.nats import KV_KEY_SCOPE_GRAMMAR, Subjects
from threetears.nats.errors import PublishError
from threetears.core.collections.scan_cache import ScanCache
from threetears.core.exceptions import InvalidL2ScopeError, L2ScopeNotConfiguredError
from threetears.observe import get_logger
from uuid_utils import uuid7

if TYPE_CHECKING:
    from threetears.core.backends import L3Backend

    # annotation-only. `Subjects` above is a genuine runtime use, but it lives
    # in a nats-py-free submodule, so importing it eagerly costs nothing.
    from threetears.nats import NatsClient, Subscription

__all__ = [
    "DEFAULT_L1_MAX_AGE_SECONDS",
    "CacheInvalidationMessage",
    "CollectionRegistry",
]

log = get_logger(__name__)

#: Default bound applied when a collection opts into L1 expiry without naming a
#: number. Not a fleet-wide default: nothing is bounded until a collection asks,
#: because the collections here differ by orders of magnitude in read volume and
#: in the staleness they tolerate, and one number would be tuned for the worst.
#:
#: Chosen against the L2 bucket TTL rather than measured. L2 self-expires at
#: 7200s, so a bound below it means a refetch usually resolves at L2 rather than
#: L3, and a bound above it is largely inert. This sits under that with room.
#: Revisit against real read volume; it moves without structural change, which
#: is why the per-collection knob was the design decision and this value was not.
DEFAULT_L1_MAX_AGE_SECONDS: float = 3600.0


def _as_l3_backend(l3_pool: Any) -> Any:
    """normalize a raw L3 transport to a :class:`DurableStore`-capable backend.

    The collection CRUD lifecycle routes through the structured
    :class:`~threetears.core.backends.protocol.DurableStore` ops, so the resolved
    ``l3_pool`` must expose them. A backend that already satisfies ``DurableStore``
    (the named :class:`~threetears.core.backends.sql.SqlL3Backend`, or a non-SQL
    backend such as scriob's ``GitL3Backend``) is returned **unchanged**. A raw
    transport that only speaks raw SQL (a bare asyncpg ``Pool`` or the
    ``NatsProxyL3Backend``) is wrapped in a ``SqlL3Backend`` so it gains the
    structured ops (which generate SQL). ``None`` passes through.

    Imported lazily to keep this module free of an import-time dependency on the
    backends package.

    :param l3_pool: a raw transport, a ``DurableStore`` backend, or ``None``.
    :ptype l3_pool: Any
    :return: a ``DurableStore``-capable backend, or ``None``.
    :rtype: Any
    """
    if l3_pool is None:
        return None
    from threetears.core.backends.protocol import DurableStore
    from threetears.core.backends.sql import SqlL3Backend

    if isinstance(l3_pool, DurableStore):
        return l3_pool
    return SqlL3Backend(l3_pool)


class CacheInvalidationMessage(BaseModel):
    """typed wire envelope for cross-pod cache invalidation broadcasts.

    publishers emit one message per write op; every other pod
    subscribed to :func:`Subjects.cache_invalidate` evicts the named
    entity from its local L1 cache. ``ids`` is always an array whose
    length matches the target collection's
    :attr:`BaseCollection.primary_key_columns` declaration -- single-pk
    collections emit length-1 arrays; composite-pk collections emit
    length-N arrays in declared column order.

    typed Pydantic envelope replaces the previous raw-JSON-bytes
    publish; the typed wrapper enforces wire-format consistency and
    surfaces drift at parse time rather than via opaque key errors
    deep in the listener.

    :ivar table: target table name (matches :attr:`BaseCollection.table_name`)
    :ivar ids: stringified pk values in declared column order
    :ivar origin: opaque per-registry id of the publisher. a receiving
        registry skips any message carrying its OWN origin -- a single
        pod both publishes and subscribes on this subject, and acting on
        its own invalidation would evict the L1 row it just wrote (the
        fresh value is already local; the broadcast is for OTHER pods).
        ``None`` for messages from a pre-origin publisher -> the receiver
        cannot prove self-origin and evicts (the historical behaviour),
        which is safe: a redundant local eviction only forces a
        pull-through, never stale data.
    """

    table: str
    ids: list[str]
    origin: str | None = None


class CollectionRegistry:
    """Registry for collection instances with dependency injection.

    Holds default L1/L2/L3 dependencies and per-collection overrides.
    Collections register themselves and resolve their dependencies through
    the registry.
    """

    def __init__(self) -> None:
        self._l1_backend: Any = None
        self._l2_client: Any = None
        self._l3_pool: L3Backend | None = None
        # The leading segment of every L2 key this registry's collections write
        # (``{scope}.{table}.{body}``). One per registry, because a registry is
        # the unit a principal wires: the scope is the sharing boundary, so every
        # collection on one registry must land in one place or replicas of the
        # same principal stop seeing each other's writes.
        self._kv_key_scope: str | None = None
        # Whether a collection on this registry may CREATE the shared collections
        # bucket, or must bind to one somebody else declared.
        #
        # ``True`` is the default and stays the default. ``NatsKvBucket._reopen``
        # re-runs the opener with this flag, and it exists because a single-node
        # NATS restart on ephemeral JetStream storage wiped every bucket and
        # silenced the wake scheduler in production -- a fleet that cannot
        # recreate the bucket runs with L2 off, at WARNING, until whoever
        # declares it restarts. Readers-not-writers is enforced by the NATS
        # GRANT, where a denied create is fail-closed by construction.
        #
        # It lives here rather than as a literal in ``base.py`` so one
        # deployment's bucket-ownership policy is not baked into the library.
        self._l2_create_if_missing: bool = True
        self._collections: dict[str, Any] = {}  # table_name -> collection instance
        self._overrides: dict[str, dict[str, Any]] = {}  # table_name -> {l1_backend, l2_client, l3_pool}
        # Deliberately NOT in ``_overrides``: ``register()`` hard-resets that dict
        # when given any tier override, and wiring commonly configures before
        # registering, so a bound stored there would be silently dropped by a
        # later ``register()`` call. Separate dict, separate lifetime.
        self._l1_max_ages: dict[str, float | None] = {}
        # Per-registry (effectively per-pod) identity stamped on every
        # invalidation this registry publishes, so its own listener can
        # skip self-published messages and avoid evicting rows it just
        # wrote. An opaque token, never used as a UUID.
        self._origin_id: str = str(uuid7())  # convert at border: invalidation wire-envelope origin token
        self._scan_cache: ScanCache | None = None
        # The invalidation listener's lifecycle state, set together by
        # :meth:`start_invalidation_listener` and cleared together by
        # :meth:`stop_invalidation_listener`. The subscription being non-``None``
        # IS the "a listener is live" answer -- deliberately not a separate
        # boolean, which would be one more thing to drift out of step with the
        # handle it describes. The client is retained solely so ``stop`` can
        # route the unsubscribe back through it, keeping the client's own
        # subscription bookkeeping correct.
        self._nats_client: NatsClient | None = None
        self._invalidation_subscription: Subscription | None = None

    def configure(
        self,
        l1_backend: Any = None,
        l2_client: Any = None,
        l3_pool: L3Backend | None = None,
        kv_key_scope: str | None = None,
        l2_create_if_missing: bool | None = None,
    ) -> None:
        """Set default dependencies for all collections.

        **Every argument merges into existing state rather than replacing it** -- a ``None``
        leaves whatever was configured before untouched. That is what makes two-pass wiring
        (scope first and client later, or the reverse) the normal shape it already is at
        several call sites.

        The L2-scope refusal is therefore evaluated over the MERGED state at the end of the
        call, never over this call's arguments. A per-call check would refuse the first pass
        of every two-pass site.

        :param l1_backend: default pod-local L1 backend, or ``None`` to leave it unchanged
        :ptype l1_backend: Any
        :param l2_client: default NATS client backing L2, or ``None`` to leave it unchanged
        :ptype l2_client: Any
        :param l3_pool: default durable L3 backend, or ``None`` to leave it unchanged
        :ptype l3_pool: L3Backend | None
        :param kv_key_scope: the leading segment of every L2 key this registry's collections
            write, from :func:`threetears.nats.kv_key_scope_for`; ``None`` leaves it unchanged
        :ptype kv_key_scope: str | None
        :param l2_create_if_missing: whether this registry's collections may CREATE the
            shared collections bucket. defaults to ``True`` and should be left there until
            the declaring identity re-declares the bucket on every NATS reconnect --
            otherwise a NATS restart that wipes JetStream leaves L2 off fleet-wide, at
            WARNING, until that identity restarts. ``None`` leaves it unchanged
        :ptype l2_create_if_missing: bool | None
        :return: nothing
        :rtype: None
        :raises InvalidL2ScopeError: if ``kv_key_scope`` falls outside the scope grammar
        :raises L2ScopeNotConfiguredError: if the merged state holds an L2 client and no scope
        """
        if l2_create_if_missing is not None:
            self._l2_create_if_missing = l2_create_if_missing
        if kv_key_scope is not None:
            if not KV_KEY_SCOPE_GRAMMAR.match(kv_key_scope):
                raise InvalidL2ScopeError(
                    f"kv_key_scope {kv_key_scope!r} does not match "
                    f"{KV_KEY_SCOPE_GRAMMAR.pattern}; the scope is one NATS subject token, so a "
                    f"'.' or '/' in it renders two and the per-principal $KV grant stops matching"
                )
            self._kv_key_scope = kv_key_scope
        if l1_backend is not None:
            self._l1_backend = l1_backend
            # a cache built over the previous backend would keep answering from
            # it; drop it so the next access rebuilds over the new one.
            self._scan_cache = None
        if l2_client is not None:
            self._l2_client = l2_client
        if l3_pool is not None:
            self._l3_pool = _as_l3_backend(l3_pool)
        if self._l2_client is not None and self._kv_key_scope is None:
            raise L2ScopeNotConfiguredError(
                "an L2 client is wired with no kv_key_scope: every collection on this registry "
                "would write into the shared collections bucket under a key no per-principal "
                "grant can name. pass kv_key_scope=threetears.nats.kv_key_scope_for(...)"
            )

    @property
    def kv_key_scope(self) -> str | None:
        """the leading segment of every L2 key this registry's collections write.

        Read by :meth:`BaseCollection.l2_key` on every L2 access. ``None`` means unwired, which
        is legitimate for an L1-only or L3-only registry and is a hard error for one holding an
        L2 client -- :meth:`configure` refuses that combination outright, and ``l2_key`` backstops
        the ``nats_client=``-direct construction path that never reaches ``configure``.

        :return: the scope segment, or ``None`` when no scope is wired
        :rtype: str | None
        """
        return self._kv_key_scope

    @property
    def l2_create_if_missing(self) -> bool:
        """whether this registry's collections may create the shared collections bucket.

        Read by :meth:`BaseCollection._ensure_kv` when it resolves the bucket. ``True`` is
        the default and the only value anything sets today: flipping it to ``False`` turns
        off :meth:`threetears.nats.NatsKvBucket._reopen`'s restart self-heal for the
        collections bucket, which is only safe once the declaring identity re-declares the
        bucket on every NATS reconnect.

        :return: ``True`` when a collection may create the bucket
        :rtype: bool
        """
        return self._l2_create_if_missing

    def register(
        self,
        collection: Any,
        *,
        l1_backend: Any = None,
        l2_client: Any = None,
        l3_pool: L3Backend | None = None,
    ) -> None:
        """Register a collection instance with optional per-collection overrides."""
        table = collection.table_name
        self._collections[table] = collection
        if l1_backend or l2_client or l3_pool:
            self._overrides[table] = {}
            if l1_backend:
                self._overrides[table]["l1_backend"] = l1_backend
            if l2_client:
                self._overrides[table]["l2_client"] = l2_client
            if l3_pool:
                self._overrides[table]["l3_pool"] = _as_l3_backend(l3_pool)

    def bind_table(
        self,
        table_name: str,
        *,
        l1_backend: Any = None,
        l2_client: Any = None,
        l3_pool: L3Backend | None = None,
    ) -> None:
        """pin per-table backend overrides BEFORE the Collection is constructed.

        :class:`BaseCollection.__init__` reads ``l3_pool`` (and friends)
        from the registry via :meth:`get_l3_pool` immediately and then
        auto-registers. :meth:`register`'s ``l3_pool=`` kwarg records
        an override but fires too late -- the Collection has already
        snapped its pool from the registry default. ``bind_table``
        records the override under the table name so the subsequent
        Collection construction reads the intended backend on its
        first :meth:`get_l3_pool` call.

        used by multi-pool agent-side bootstraps (three-tier-task-01
        Phase C2: the rbac metadata Collections need a separate
        :class:`NatsProxyL3Backend` pool bound to ``system.platform.rbac``
        because the broker route different namespaces to different
        schemas). every key left as ``None`` is ignored so callers can
        layer l1 / l2 / l3 bindings independently.

        :param table_name: target table name (matches
            :attr:`BaseCollection.table_name` on the Collection that
            will later be constructed)
        :ptype table_name: str
        :param l1_backend: L1 backend override for this table, or
            ``None`` to leave any existing binding untouched
        :ptype l1_backend: Any
        :param l2_client: L2 client override for this table
        :ptype l2_client: Any
        :param l3_pool: L3 pool override for this table
        :ptype l3_pool: Any
        :return: nothing
        :rtype: None
        """
        if l1_backend is None and l2_client is None and l3_pool is None:
            return
        existing = self._overrides.setdefault(table_name, {})
        if l1_backend is not None:
            existing["l1_backend"] = l1_backend
        if l2_client is not None:
            existing["l2_client"] = l2_client
        if l3_pool is not None:
            existing["l3_pool"] = _as_l3_backend(l3_pool)

    def get_collection(self, table_name: str) -> Any | None:
        """Look up a registered collection by table name."""
        return self._collections.get(table_name)

    def get_l1_backend(self, table_name: str) -> Any:
        """Get L1 backend for a collection (override or default)."""
        overrides = self._overrides.get(table_name, {})
        return overrides.get("l1_backend", self._l1_backend)

    def set_l1_max_age(self, table_name: str, max_age_seconds: float | None = DEFAULT_L1_MAX_AGE_SECONDS) -> None:
        """Bound how long ``table_name``'s L1 rows may be served after a pull-through.

        **Per collection, and off until one asks.** Collections in this repo
        differ by orders of magnitude in read volume and in how much staleness
        they can tolerate, so a single fleet-wide number would be tuned for the
        worst of them and wrong for the rest. Nothing is bounded until a
        collection sets a value.

        Not every backend can honour one. ``DuckDBBackend`` injects no stamp and
        raises ``NotImplementedError`` on the first repairing read rather than
        silently never expiring -- so a bound accepted here surfaces at read
        time, not at configuration time. It has no production construction site
        today, which is why this is a documented sharp edge rather than a
        validation: the check would have to reach into the backend a collection
        happens to hold, and the refusal it would duplicate is already loud.

        A collection with no L3 pool is refused a bound at the point of use
        (:attr:`BaseCollection.l1_max_age_seconds`), regardless of what is set
        here: with nothing to pull through from, an expired row is not a miss,
        it is a deletion.

        :param table_name: the collection's table
        :ptype table_name: str
        :param max_age_seconds: the bound, or ``None`` to turn expiry off.
            Omitted, it is :data:`DEFAULT_L1_MAX_AGE_SECONDS` -- opting in
            without a number is the supported way to take the default.
        :ptype max_age_seconds: float | None
        :return: nothing
        :rtype: None
        :raises ValueError: if ``max_age_seconds`` is not positive
        """
        if max_age_seconds is not None and max_age_seconds <= 0:
            raise ValueError(
                f"l1 max age for {table_name!r} must be positive or None, got {max_age_seconds!r}; "
                f"a zero or negative bound expires every row on the read that follows its own write, "
                f"which is not a cache",
            )
        self._l1_max_ages[table_name] = max_age_seconds

    def get_l1_max_age(self, table_name: str) -> float | None:
        """Return the configured L1 max age for a collection, or ``None``.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the bound in seconds, or ``None`` when expiry is off
        :rtype: float | None
        """
        return self._l1_max_ages.get(table_name)

    @property
    def scan_cache(self) -> ScanCache:
        """the pod's visibility-scan cache, created on first use.

        Built over the DEFAULT L1 backend rather than a per-table override: a
        scan spans tables (its visibility predicate reads the RBAC tables), so
        it has no single owning table's backend to live in.

        :return: the scan cache
        :rtype: ScanCache
        """
        if self._scan_cache is None:
            self._scan_cache = ScanCache(self._l1_backend)
        return self._scan_cache

    def get_l2_client(self, table_name: str) -> Any:
        """Get L2 client for a collection (override or default)."""
        overrides = self._overrides.get(table_name, {})
        return overrides.get("l2_client", self._l2_client)

    def get_l3_pool(self, table_name: str) -> L3Backend | None:
        """Get L3 pool for a collection (override or default)."""
        overrides = self._overrides.get(table_name, {})
        return cast("L3Backend | None", overrides.get("l3_pool", self._l3_pool))

    # ------------------------------------------------------------------
    # Cache coherence -- cross-pod L1 invalidation via typed NATS pub/sub
    # ------------------------------------------------------------------

    async def start_invalidation_listener(self, nats_client: NatsClient) -> None:
        """subscribe to cache invalidation signals from other pods.

        wire envelope is :class:`CacheInvalidationMessage` -- typed
        Pydantic, serialized via ``model_dump_json()``. on receipt the
        listener evicts the named entity from local L1 cache for the
        matching collection. callers must invoke after every
        Collection has registered.

        narrow exception scope: malformed payloads (Pydantic
        :class:`ValidationError`) and unknown-table receipts log and
        skip; programming errors (``AttributeError`` / ``TypeError``)
        propagate so wrapper drift / collection-misregistration
        surfaces immediately rather than as silent
        invalidation-skips.

        **idempotent while a listener is live.** a second call is a
        no-op rather than a second consumer: two subscriptions on one
        subject in one process means every broadcast is handled twice,
        and a teardown that releases one of them leaves the other
        running. call :meth:`stop_invalidation_listener` first to
        rebind to a different client.

        :param nats_client: connected typed NATS wrapper client
        :ptype nats_client: NatsClient
        :return: nothing
        :rtype: None
        :raises SubscribeError: if the underlying subscribe fails to
            register (transport / config error)
        """
        if self._invalidation_subscription is not None:
            log.debug(
                "Invalidation listener already running",
                extra={"extra_data": {"subject": Subjects.cache_invalidate().path}},
            )
            return

        async def _on_invalidation(message: CacheInvalidationMessage) -> None:
            # Skip invalidations this registry published itself. A single
            # pod both publishes and subscribes on this subject; acting on
            # our own broadcast would evict the L1 row save_entity just
            # wrote, so a subsequent L1-only read of that entity (e.g. an
            # entity field accessor) would miss. The broadcast exists to
            # evict OTHER pods, which carry a different origin.
            if message.origin is not None and message.origin == self._origin_id:
                return

            # Evict dependent SCANS first, and unconditionally. Every check
            # below returns early for a table this pod does not hold as a
            # collection -- and `role_assignments` / `group_members` are exactly
            # that on an agent pod, while being precisely the tables whose write
            # must drop a cached visibility scan. Ordering this after those
            # guards would leave a revoked grant readable until the TTL lapsed.
            self.scan_cache.drop_for_table(message.table)

            collection = self._collections.get(message.table)
            if collection is None:
                # unknown-table receipts are expected during partial
                # rollouts (sender has a Collection the receiver does
                # not). log + skip without warning.
                return

            pk_cols = collection.primary_key_columns
            if len(message.ids) != len(pk_cols):
                log.warning(
                    "Invalidation pk arity mismatch",
                    extra={
                        "extra_data": {
                            "table": message.table,
                            "expected_columns": list(pk_cols),
                            "received_values": len(message.ids),
                        },
                    },
                )
                return
            entity_id = tuple(message.ids)

            # EVICT L2 HERE, ahead of the L1 guards below, and the placement is the
            # requirement rather than a preference.
            #
            # Keys are per-principal (``{scope}.{table}.{body}``), so the writer updates
            # only ITS OWN key. A peer that drops L1 and pulls through reads L2 first --
            # its own, still-stale key -- and re-caches the value the broadcast was sent
            # to retract. ``max_age`` on the bucket is unlimited and no collection sets an
            # L1 bound, so nothing heals it: a revoked grant would be enforced forever,
            # which is worse than the exposure per-principal keys exist to close. Under
            # the old single shared key this could not happen (the peer read the writer's
            # fresh value), so the eviction is a scoping-induced obligation and lands with
            # it.
            #
            # Behind ``l1 is None`` / ``not l1.has_table(...)`` it would skip exactly the
            # collections whose L1 schema was never initialised -- and those are precisely
            # the ones that would then keep the stale L2 value forever. L2 presence is
            # independent of L1 presence.
            #
            # NOT ``invalidate_cache``, which looks right and RE-PUBLISHES: the origin
            # filter above only skips SELF, so every receiver would rebroadcast under its
            # own origin, unbounded.
            #
            # The arity check is hoisted above this rather than left where it was, because
            # ``l2_key`` normalises the pk and raises on a mismatch; it touches no L1.
            await collection.delete_l2_entry(entity_id)

            l1 = self.get_l1_backend(message.table)
            if l1 is None:
                return
            if hasattr(l1, "has_table") and not l1.has_table(message.table):
                # this pod's L1 backend was never initialize()'d with this
                # table's schema -- its OWN collections never touch it
                # locally, so there is nothing to evict. Same "unknown
                # receipts are expected during partial rollouts" treatment
                # as the `collection is None` case above: without this
                # check, `l1.delete_by_id` below raises straight through
                # (sqlite3.OperationalError: no such table / DuckDB's
                # equivalent CatalogException) on every single broadcast
                # for a table this pod doesn't cache, which is the common
                # case for any agent that doesn't use every framework
                # feature (e.g. a security-scanning agent hearing knowledge-
                # subsystem invalidations for `concepts`/`playbook_entries`
                # it never reads). `hasattr` guards backends (or test
                # doubles) that predate this method.
                return

            l1.delete_by_id(message.table, entity_id, pk_cols)

        # SubscribeError propagates deliberately: cache coherence is not
        # optional, so a process that cannot subscribe must fail its startup
        # rather than run on silently as an island. Both fields stay unset on
        # that path, so a failed start leaves nothing behind for ``stop`` to
        # find and a retry is a clean first start.
        subscription = await nats_client.subscribe_typed(
            subject=Subjects.cache_invalidate(),
            message_type=CacheInvalidationMessage,
            cb=_on_invalidation,
        )
        self._nats_client = nats_client
        self._invalidation_subscription = subscription

    async def stop_invalidation_listener(self) -> None:
        """drop this registry's cache-invalidation subscription.

        the teardown half of :meth:`start_invalidation_listener`, for a
        process's shutdown path. after it returns the registry holds no
        subscription and no client, and a later ``start`` subscribes
        again -- the registry is reusable, not one-shot.

        **a no-op when no listener is live**, whether one was never
        started or was already stopped. callers run this from a
        ``finally`` / ``close()`` alongside other teardown, where a
        raise would abandon the teardown steps after it.

        **no exception handling here, deliberately.**
        :meth:`NatsClient.unsubscribe` already absorbs the transport
        failures a shutdown produces: it returns early on an
        already-closed subscription, and it wraps both the raw
        unsubscribe and the dispatch-task join in its own handlers that
        log at WARNING. so unsubscribing on a draining connection does
        not raise, and a catch here would be code that never runs while
        looking like it guards something. what WOULD reach a caller is a
        genuine programming error, which must surface.

        the handle is released before the unsubscribe is awaited, so the
        registry is restartable even on the paths that unwind slowly.

        :return: nothing
        :rtype: None
        """
        subscription = self._invalidation_subscription
        nats_client = self._nats_client
        if subscription is None or nats_client is None:
            return
        self._invalidation_subscription = None
        self._nats_client = None
        await nats_client.unsubscribe(subscription)
        log.info(
            "Invalidation listener stopped",
            extra={"extra_data": {"subject": Subjects.cache_invalidate().path}},
        )

    async def publish_invalidation(
        self,
        nats_client: NatsClient | None,
        table_name: str,
        entity_id: Any,
    ) -> None:
        """publish cache invalidation signal for an entity.

        called by :class:`BaseCollection` after any write operation.
        emits :class:`CacheInvalidationMessage` via the typed wrapper.
        ``ids`` is the stringified pk-value tuple in declared column
        order.

        narrow exception scope: only :class:`PublishError` is logged
        and swallowed. The write has already succeeded, and nats-py
        buffers a publish issued while disconnected and flushes it on
        reconnect, so an ordinary outage loses nothing.

        The older justification here -- that a peer would "pull a
        stale-but-still-correct row from L3" -- does not hold and is
        recorded as wrong rather than deleted: a peer that missed the
        eviction keeps serving its own L1 and never re-reads. What
        actually bounds that staleness is the L1 max-age, and only for
        a collection that has opted into one. programming errors
        propagate.

        :param nats_client: connected typed NATS wrapper client;
            ``None`` short-circuits (no-op)
        :ptype nats_client: NatsClient | None
        :param table_name: target table name
        :ptype table_name: str
        :param entity_id: pk value (single-pk) or tuple of pk values
            in declared order (composite-pk)
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        # A LOCAL write evicts LOCAL scans, before anything touches the bus.
        #
        # The listener below deliberately ignores this registry's own
        # broadcasts: for a by-pk row that is right, because `save_entity` just
        # wrote the freshest copy into L1 and evicting it would force a needless
        # re-read. A SCAN is the opposite. The write changed WHICH ROWS MATCH,
        # so the result this pod has cached is stale the instant it commits --
        # and it is the one pod guaranteed never to hear about it.
        #
        # Shipped without this in 0.23.6: a hub that imported knowledge kept
        # serving the pre-import concept set until the TTL lapsed. Deploy
        # content, ask a question, get the old answer, with nothing to indicate
        # why. Caught by the concept-visibility tests, which write and then read
        # back through the same registry.
        #
        # Deliberately ahead of the `nats_client is None` return: local eviction
        # is not a broadcast and must not be skipped when there is no bus (devx,
        # tests, a pod whose NATS is down).
        self.scan_cache.drop_for_table(table_name)
        if nats_client is None:
            return
        if isinstance(entity_id, tuple):
            values = entity_id
        else:
            values = (entity_id,)
        ids = [str(v) for v in values]  # convert at border: invalidation wire-envelope pk values
        message = CacheInvalidationMessage(table=table_name, ids=ids, origin=self._origin_id)
        try:
            await nats_client.publish(
                subject=Subjects.cache_invalidate(),
                message=message,
            )
        except PublishError as exc:
            log.warning(
                "Failed to publish invalidation signal",
                extra={
                    "extra_data": {
                        "table": table_name,
                        "entity_id": str(
                            entity_id
                        ),  # convert at border: invalidation-publish-failed log extra_data field
                        "error": str(exc),
                    },
                },
            )
        except ValidationError as exc:
            # wire envelope failed validation -- programming error,
            # but propagating would mask the calling write op. log
            # loud and continue; surfaces as a real failure in CI
            # (there is no such counter, and nothing in
            # test_cache_coherence.py asserts on one -- the claim this
            # comment used to make was simply untrue).
            log.error(
                "Invalidation envelope validation failed",
                extra={
                    "extra_data": {
                        "table": table_name,
                        "entity_id": str(
                            entity_id
                        ),  # convert at border: invalidation-envelope-invalid log extra_data field
                        "error": str(exc),
                    },
                },
            )

    def clear(self) -> None:
        """Remove all registered collections, overrides and L1 bounds (for tests).

        **Does NOT stop the invalidation listener**, and cannot: unsubscribing
        is awaitable and this method is synchronous. A registry cleared while a
        listener is live keeps that subscription and the callback goes on
        firing. The by-pk eviction below it is inert -- the collection lookup
        misses and returns -- but the scan-cache drop sits ABOVE that guard and
        runs on every broadcast regardless, so a cleared registry is not an
        idle one. Call :meth:`stop_invalidation_listener` alongside this in any
        teardown that started a listener.
        """
        self._collections.clear()
        self._overrides.clear()
        # The bound keeps its own dict so ``register()`` cannot wipe it, but a
        # separate lifetime is not an unbounded one: a table re-registered after
        # a clear would otherwise inherit a bound nobody in the new setup asked
        # for, which is the same silent-config class of bug in the other
        # direction.
        self._l1_max_ages.clear()
