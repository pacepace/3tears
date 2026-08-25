"""LangGraph checkpoint saver with three-tier caching.

implements :class:`~langgraph.checkpoint.base.BaseCheckpointSaver`
over an :class:`~threetears.langgraph.protocols.AsyncQueryExecutor`
with optional L1 (pod-local, fast) and L2 (shared, network-backed)
read caches in front of the L3 database tier:

- L3 (database): source of truth. reached via the
  :class:`AsyncQueryExecutor` protocol. trusted services pass a
  :class:`~threetears.langgraph.protocols.AsyncpgPoolAdapter` over
  their :class:`asyncpg.Pool`; sandboxed agents pass
  :class:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend`
  directly because it implements the protocol natively.
- L2 (distributed cache): optional hot cache (e.g. NATS KV, Redis).
- L1 (local cache): optional in-memory/local cache (e.g. SQLite).

all L1 and L2 operations degrade gracefully on failure -- cache
misses fall through to the next tier, and cache write failures are
logged and swallowed so the graph never crashes due to cache
infrastructure issues.

L3 does NOT degrade in general: it is the source of truth, and a
failed read or checkpoint write must reach the caller. The single
exception is the crash-recovery half of
:meth:`ThreeTierCheckpointSaver.aput_writes`, which LangGraph calls
from executor teardown where a raise kills a turn that has already
answered. That carve-out is scoped by channel, not by tier -- the
control-channel writes arriving at the same method still raise,
because losing one changes what the run does. See that method for
the full rule.

every saver states a tenancy decision at construction:
``scope: CheckpointScope`` has no default, so a caller names one
customer, says in writing that it deliberately names none, or says
that it serves many and each call will name its own. when a customer
is resolved it is folded into the stored ``thread_id`` and so into
every key the call addresses at all three tiers -- see
:meth:`ThreeTierCheckpointSaver.storage_thread_id` for why the
customer lives in the key rather than in a column, and
:meth:`ThreeTierCheckpointSaver.adelete_customer_threads` for the
purge that tenancy exists to make possible. an unscoped saver is
byte-for-byte what it was before tenancy existed, which is what makes
the required parameter adoptable without moving a row -- see the class
docstring for the per-consumer upgrade path.

namespace-task-01 phase 8.5l-4 merged the former
``ProxyCheckpointSaver`` into this class after Pace's pushback
on the "genuinely distinct deployment targets" claim. the split
was 95% duplicate code separated only by the database-parameter
type and a couple of private helper names. the unified class
takes the protocol, so both deployment contexts (direct pool via
the adapter; sandboxed agent via the NATS L3 proxy) flow through
one implementation with no parallel path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any, cast
from uuid import UUID

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from threetears.langgraph.checkpoint_scope import CheckpointScope
from threetears.langgraph.protocols import (
    AsyncQueryExecutor,
    CheckpointL1Cache,
    CheckpointL2Cache,
    CheckpointL2PrefixCache,
    FlushCallback,
)
from threetears.langgraph.serde import UUIDSafeSerializer
from threetears.observe import get_logger

__all__ = [
    "ThreeTierCheckpointSaver",
]

log = get_logger(__name__)

_DEFAULT_L2_BUCKET = "checkpoints"

#: separates the customer from the caller's own thread id inside a stored
#: ``thread_id``. a UUID's text form contains no ``/``, so the split point is
#: unambiguous and a customer prefix can never be forged by a thread id.
_CUSTOMER_SEPARATOR = "/"

#: width of the ``thread_id`` column in
#: :mod:`threetears.langgraph.migrations.v001_create_checkpoint_tables`.
#: the customer prefix is checked against it here so an overflow surfaces as an
#: arithmetic error naming both halves, rather than as a driver error or (on a
#: database that truncates rather than rejects) as two customers sharing a row.
_MAX_THREAD_ID_LENGTH = 255


class ThreeTierCheckpointSaver(BaseCheckpointSaver[int]):
    """LangGraph checkpoint saver using three-tier caching.

    L1 and L2 are optional cache layers in front of the L3
    (database) source of truth. reads check L1 -> L2 -> L3,
    promoting hits into warmer tiers. writes always go to L3 first,
    then warm L2 and L1 opportunistically.

    the database tier is reached through the
    :class:`AsyncQueryExecutor` protocol so one implementation
    serves two deployment contexts:

    - trusted services (e.g. the hub) hold a direct
      :class:`asyncpg.Pool` and wrap it in
      :class:`~threetears.langgraph.protocols.AsyncpgPoolAdapter`
      before passing to this saver.
    - sandboxed agents hold a
      :class:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend`,
      which already implements the protocol natively, so they pass
      it straight through.

    **Tenancy.** ``scope`` is REQUIRED and has no default, so building a saver
    forces one of exactly three answers:

    - ``CheckpointScope.for_customer(customer_id)`` -- the customer is folded
      into the stored thread id (:meth:`storage_thread_id`) and therefore into
      every key the saver addresses: the ``thread_id`` bound into L3 statements,
      the L2 bucket key, and the L1 thread key. A saver scoped to one customer
      cannot NAME another customer's row, which is a structural property rather
      than a predicate a later statement can forget. See
      :meth:`storage_thread_id` for why the customer lives in the key rather
      than in a column.
    - ``CheckpointScope.from_config(key="customer_id")`` -- the saver serves
      MANY customers and resolves each call's customer out of
      ``config["configurable"][key]``, folding it into the key exactly as above.
      Fails CLOSED: a missing key, a ``None``, or a non-``UUID`` raises rather
      than degrading to the un-tenanted keyspace.
    - ``CheckpointScope.unscoped(reason="...")`` -- the saver addresses the
      un-tenanted keyspace, deliberately, with the reason recorded and logged.

    The parameter used to be ``customer_id: UUID | None = None``, and the
    DEFAULT was the defect: saying nothing meant "address every customer's
    keyspace", which is what every caller said, so tenancy was a convention
    rather than a gate. Omitting ``scope`` is now a :class:`TypeError` at
    construction.

    **The multi-tenant case, and why ``from_config`` exists.** The first two
    answers both assume ONE customer per saver INSTANCE. A host that serves
    every customer from one process, with one compiled graph and therefore one
    process-lifetime saver built in lifespan startup before any request exists,
    can say neither honestly: it has no single customer to name, and it is not
    un-tenanted -- it has many customers who must not share a keyspace. That
    host is not a corner case; it is the shape metallm already has and the shape
    the survey engine's admin pod is designed to have.

    ``from_config`` keeps the one saver and moves the customer onto the work.
    LangGraph already threads a ``RunnableConfig`` into every checkpoint call
    that reads or writes -- ``aget_tuple``, ``alist``, ``aput``, ``aput_writes``
    -- so the customer travels with the request that knows it, and one saver
    keeps every customer in a distinct keyspace with no per-request construction
    and no re-compilation.

    The two purge methods, :meth:`adelete_thread` and
    :meth:`adelete_customer_threads`, receive no config. They take the customer
    as a keyword-only argument instead, which leaves every existing positional
    call working under the other two scopes, and they REFUSE rather than guess
    when a config-resolved saver is asked to delete without one.

    **Upgrading an existing consumer.** The minimum viable change is one line::

        saver = ThreeTierCheckpointSaver(
            executor=executor,
            scope=CheckpointScope.unscoped(reason="single-tenant deployment"),
        )

    That is a legitimate destination for a genuinely SINGLE-TENANT deployment,
    not a placeholder. An unscoped saver produces byte-identical keys and
    statements to one built before tenancy existed, so it reads the rows that
    already exist and MIGRATES NOTHING.

    It is the wrong destination for a host that serves many customers from one
    process. That host wants ``from_config``, whose per-call resolution costs it
    one dict key rather than a construction-lifetime change -- see the metallm
    worked example below.

    Adopting a real customer later is a data change, not a code change:
    existing rows live under a bare thread id and a scoped saver will not find
    them, so they have to be re-keyed --
    ``UPDATE checkpoints SET thread_id = $customer || '/' || thread_id ...``
    plus the same over ``checkpoint_writes``, and the cached L2 bundles
    invalidated. No such script ships here, and cannot: which customer owns
    which thread lives in the HOST's own tables (a sessions table, a
    conversations table), which this library has never seen. Each consumer
    writes that mapping query itself.

    **Worked example: metallm, the multi-tenant case.** Its migration is
    ``from_config`` and it is two edits, neither of which changes a lifetime or
    a construction site's shape.

    1. At the build site, ``api/src/graph/checkpoint.py:292`` (the
       ``return ThreeTierCheckpointSaver(...)`` inside
       ``build_checkpoint_saver``), pass the scope::

           return ThreeTierCheckpointSaver(
               AsyncpgPoolAdapter(postgres_pool),
               scope=CheckpointScope.from_config(),
               l1_cache=l1_cache,
               l2_cache=l2_cache,
               l2_bucket=BUCKET_CHECKPOINTS,
               flush_callback=flush_callback,
           )

       Nothing about the singleton changes: it is still built once in lifespan
       startup at ``api/src/main.py:906`` and still baked into the compiled graph
       by ``get_graph(checkpointer=checkpointer)`` at ``api/src/main.py:915``.

    2. At the call site, ``api/src/graph/metallm_graph.py:1707`` (the
       ``config: RunnableConfig = {...}`` literal in ``run_conversation``), add
       one key beside the ``thread_id`` already set on line 1709::

           config: RunnableConfig = {
               "configurable": {
                   "thread_id": str(conversation_id),
                   "customer_id": user_id,
                   ...
               },
           }

       ``user_id: UUID`` is a parameter of ``run_conversation`` itself
       (``api/src/graph/metallm_graph.py:925``) and is in scope at that literal,
       so the customer is one dict key away and no plumbing is needed.

    Why ``user_id`` IS the customer here: metallm's tenant model maps
    ``customer_id -> user_id``, one customer per user, stated in
    ``api/src/services/metallm_memory_authorizer.py:449-467``
    (``metallm_memory_namespace_for_user``, whose docstring says "each user is
    their own customer" and whose body returns ``METALLM_AGENT_ID, user_id``).
    So metallm is not a single-tenant deployment that happens to lack a
    customer -- it is a MULTI-TENANT one whose customer is per request, which is
    exactly the case ``from_config`` was added for. Its checkpoints are
    currently isolated only by ``thread_id`` unguessability; after this they are
    isolated by the key itself, at all three tiers.

    The same data caveat as any other adoption applies and is not waived by
    ``from_config``: metallm's EXISTING checkpoint rows live under a bare
    ``thread_id``, and a config-resolved saver will not find them, so a cutover
    either re-keys them (``UPDATE checkpoints SET thread_id = <customer> || '/'
    || thread_id``, the same over ``checkpoint_writes``, L2 bundles invalidated)
    or accepts that in-flight conversations start a fresh checkpoint history.
    The mapping from conversation to user lives in metallm's own tables, so that
    statement is metallm's to write; no script ships here.

    An earlier revision of this docstring recorded metallm as "could not
    determine a customer" and pointed it at ``unscoped``. That is superseded and
    was a limitation of the two-answer API, not of metallm: the customer was
    always in scope at the call site, it just had nowhere to go. ``unscoped`` is
    NOT metallm's path.

    Where the customer is available today, from a reading of each consumer:

    - ``14-eng-ai-bot-agents`` bootstrap (``phases/backend.py``) -- available.
      ``state.customer_id`` is a sibling attribute on the same
      ``BootstrapState`` already in scope, checked non-None earlier in the same
      function. The pod serves one customer for its whole life, so a
      process-lifetime saver can be scoped to it directly.
    - ``14-eng-ai-survey`` (``core/checkpointer_factory.py``) -- available, one
      call away. ``get_platform_identity().customer_id`` is a process-wide
      accessor over the same one-customer-per-pod environment; it is simply not
      wired into the factory yet. Also a process-lifetime saver over a
      single-tenant deployment, so scoping it is safe.
    - ``scriob`` delete-session route (``chat/routes.py``) -- available.
      ``identity.tenant_id`` is a local in the same handler, used two lines
      above the construction. Per-request construction on a multi-tenant
      server, which is the shape a scoped saver fits best.
    - ``scriob`` history read (``chat/turn.py``, ``read_message_history``) --
      NOT in scope, one hop away. The caller holds ``identity.tenant_id``; the
      function takes only a pool and a conversation id, so adopting a customer
      means threading one parameter through one call.
    - ``scriob`` turn build (``chat/turn.py``, ``_build_compiled``) -- NOT in
      scope, two to three hops away. The same ``identity.tenant_id`` is already
      threaded down this chain for summarization wiring, so the path exists.
      ``from_config`` is also open to it, and is the better fit if the compiled
      graph is ever shared across tenants.
    - ``metallm`` (``api/src/graph/checkpoint.py``) -- available, one dict key
      away, via ``from_config``. Multi-tenant per process: see the worked
      example above for the two edits and the line numbers.

    :param executor: async query executor for database operations
    :ptype executor: AsyncQueryExecutor
    :param scope: required tenancy decision -- one customer, a customer per
        call, or an explicit reasoned opt-out
    :ptype scope: CheckpointScope
    :param l1_cache: optional L1 local cache (e.g. SQLite)
    :ptype l1_cache: CheckpointL1Cache | None
    :param l2_cache: optional L2 distributed cache (e.g. NATS KV)
    :ptype l2_cache: CheckpointL2Cache | None
    :param l2_bucket: bucket/namespace for L2 cache keys
    :ptype l2_bucket: str
    :param flush_callback: optional async callback invoked after
        each checkpoint write to drain pending writes; returns the
        count of items flushed
    :ptype flush_callback: FlushCallback | None
    """

    def __init__(
        self,
        executor: AsyncQueryExecutor,
        *,
        scope: CheckpointScope,
        l1_cache: CheckpointL1Cache | None = None,
        l2_cache: CheckpointL2Cache | None = None,
        l2_bucket: str = _DEFAULT_L2_BUCKET,
        flush_callback: FlushCallback | None = None,
    ) -> None:
        """initialize checkpoint saver.

        ``scope`` is keyword-only and has no default. Keyword-only because a
        second positional argument next to ``executor`` is a transposition
        nothing would catch; no default because the default was the defect this
        parameter replaces.

        :param executor: async query executor for database operations
        :ptype executor: AsyncQueryExecutor
        :param scope: required tenancy decision for every key this saver
            addresses
        :ptype scope: CheckpointScope
        :param l1_cache: optional L1 local cache
        :ptype l1_cache: CheckpointL1Cache | None
        :param l2_cache: optional L2 distributed cache
        :ptype l2_cache: CheckpointL2Cache | None
        :param l2_bucket: bucket name for L2 cache keys
        :ptype l2_bucket: str
        :param flush_callback: optional post-write flush callback
        :ptype flush_callback: FlushCallback | None
        :return: nothing
        :rtype: None
        """
        super().__init__()
        self.serde = UUIDSafeSerializer()
        self._exec = executor
        self._l1 = l1_cache
        self._l2 = l2_cache
        self._l2_bucket = l2_bucket
        self._scope = scope
        self._flush_callback = flush_callback

    # ------------------------------------------------------------------
    # Tenancy helpers
    # ------------------------------------------------------------------

    def customer_for_config(self, config: Mapping[str, Any] | None) -> UUID | None:
        """the customer a config-bearing call addresses, per this saver's scope.

        Every read and write path calls this once and threads the result down to
        the key builders, so the three scope answers differ in one place rather
        than at every call site. A ``for_customer`` saver returns its own
        customer and never consults the config; an ``unscoped`` one returns
        ``None``; a ``from_config`` one reads the config and RAISES rather than
        degrading when the customer is missing, ``None``, or not a
        :class:`~uuid.UUID`.

        :param config: the call's runnable config, or None
        :ptype config: Mapping[str, Any] | None
        :return: the customer this call addresses, or None when the scope names
            none
        :rtype: UUID | None
        :raises TypeError: when a config-resolved scope finds a non-UUID value
        :raises ValueError: when a config-resolved scope finds no usable value
        """
        return self._scope.customer_for_config(config)

    def _customer_prefix(self, customer: UUID | None) -> str | None:
        """render a resolved customer as the key prefix, or None for no customer.

        The single place a customer UUID becomes text, so the border is one line
        rather than one per statement.

        :param customer: the customer this call addresses, or None
        :ptype customer: UUID | None
        :return: the prefix including its separator, or None
        :rtype: str | None
        """
        if customer is None:
            return None
        return f"{customer}{_CUSTOMER_SEPARATOR}"  # convert at border: storage key text

    def storage_thread_id(self, thread_id: str, *, customer: UUID | None = None) -> str:
        """map a caller's thread id to the one this saver stores it under.

        Returns ``thread_id`` unchanged when the call names no customer, and
        ``"<customer_id>/<thread_id>"`` when it names one.

        ``customer`` reconciles against the saver's scope through
        :meth:`~threetears.langgraph.checkpoint_scope.CheckpointScope.customer_for_operation`,
        so omitting it means "use the answer the scope already holds". Under
        :meth:`~threetears.langgraph.checkpoint_scope.CheckpointScope.for_customer`
        and
        :meth:`~threetears.langgraph.checkpoint_scope.CheckpointScope.unscoped`
        there IS such an answer and every existing call keeps working unchanged.
        Under
        :meth:`~threetears.langgraph.checkpoint_scope.CheckpointScope.from_config`
        there is not, so omitting it RAISES.

        That last refusal is the backstop that makes the multi-tenant scope
        structural rather than a checklist: this is the only key builder in the
        class, so a path added later that forgets to resolve its customer cannot
        produce an un-tenanted key -- it fails instead.

        **Why the customer lives in the key rather than in a column.** The
        checkpoint tables are keyed ``(thread_id, checkpoint_ns,
        checkpoint_id)``. A ``customer_id`` column only makes a row unique
        THROUGH its customer if it joins that primary key, and altering the key
        of a table that holds live rows in several deployments is a different
        and much larger change than this one. Folding the customer into the
        leading key column buys the same uniqueness with no DDL at all: two
        customers using the same caller-chosen thread id occupy two rows, not
        one. It also reaches tiers a column cannot -- the L2 bucket and the L1
        cache are key-value stores with no columns to filter on, and tenanting
        L3 while leaving the KV bucket shared would be the leak this exists to
        close.

        It also works over both deployment transports unchanged. The proxied
        route cannot carry a customer: ``customer_scope`` exists only on the L3
        backend's READ methods, not on ``execute``, and the hub broker consults
        it only for its ``system.platform.rbac`` SELECT carve-out. A key scheme
        needs nothing from the transport.

        **What this is and is not.** It is defence in depth and a purge handle,
        not an authorization system. A host still decides which customer a
        request belongs to and builds the saver accordingly; what this adds is
        that a saver built for the wrong customer reads nothing rather than
        reading someone else's conversation. What the required ``scope``
        parameter adds on top is that the decision cannot be skipped -- a
        deployment addressing the un-tenanted keyspace did so on purpose and
        said why.

        The scheme fails CLOSED, which is the reason it beats a column: a bare
        thread id under a scoped saver matches nothing, whereas a statement that
        forgot a ``customer_id`` predicate would return every customer's rows.

        :param thread_id: conversation/thread identifier as the caller knows it
        :ptype thread_id: str
        :param customer: the customer this call addresses; omit to use the
            scope's own answer, which a config-resolved scope does not have
        :ptype customer: UUID | None
        :return: the identifier this saver reads and writes under
        :rtype: str
        :raises TypeError: when customer is neither None nor a UUID
        :raises ValueError: when the customer cannot be reconciled with the
            scope, or when the customer prefix would push the composite past the
            width of the ``thread_id`` column
        """
        prefix = self._customer_prefix(
            self._scope.customer_for_operation(customer, operation="storage_thread_id"),
        )
        if prefix is None:
            return thread_id
        composite = f"{prefix}{thread_id}"
        if len(composite) > _MAX_THREAD_ID_LENGTH:
            raise ValueError(
                f"customer-scoped thread id is {len(composite)} characters, over the "
                f"{_MAX_THREAD_ID_LENGTH}-character thread_id column: the customer prefix costs "
                f"{len(prefix)}, leaving {_MAX_THREAD_ID_LENGTH - len(prefix)} "
                f"for a thread id of {len(thread_id)}",
            )
        return composite

    # ------------------------------------------------------------------
    # L1 helpers
    # ------------------------------------------------------------------

    async def l1_get(self, thread_id: str, checkpoint_ns: str, *, customer: UUID | None = None) -> bytes | None:
        """read from L1 cache, returning None on miss or error.

        :param thread_id: conversation/thread identifier
        :ptype thread_id: str
        :param checkpoint_ns: checkpoint namespace
        :ptype checkpoint_ns: str
        :param customer: the customer this call addresses; omit to use the
            scope's own answer
        :ptype customer: UUID | None
        :return: cached blob or None
        :rtype: bytes | None
        """
        result: bytes | None = None
        if self._l1 is not None:
            # scoped OUTSIDE the guard: an over-long composite, or a missing
            # customer under a config-resolved scope, is a caller error that
            # fails identically every time, and reporting it as an L1 fault would
            # degrade it into a warning the caller never acts on -- which for the
            # missing customer would leave the un-tenanted key being read.
            storage_thread_id = self.storage_thread_id(thread_id, customer=customer)
            try:
                result = await self._l1.get(storage_thread_id, checkpoint_ns)
            except Exception:
                log.warning("L1 checkpoint read failed", exc_info=True)
                result = None
        return result

    async def l1_put(self, thread_id: str, checkpoint_ns: str, data: bytes, *, customer: UUID | None = None) -> None:
        """write to L1 cache, swallowing errors.

        :param thread_id: conversation/thread identifier
        :ptype thread_id: str
        :param checkpoint_ns: checkpoint namespace
        :ptype checkpoint_ns: str
        :param data: serialized cache blob
        :ptype data: bytes
        :param customer: the customer this call addresses; omit to use the
            scope's own answer
        :ptype customer: UUID | None
        :return: nothing
        :rtype: None
        """
        if self._l1 is None:
            return
        storage_thread_id = self.storage_thread_id(thread_id, customer=customer)
        try:
            await self._l1.put(storage_thread_id, checkpoint_ns, data)
        except Exception:
            log.warning("L1 checkpoint write failed", exc_info=True)

    async def l1_delete(self, thread_id: str, *, customer: UUID | None = None) -> None:
        """delete a thread's L1 entry, swallowing errors.

        :param thread_id: conversation/thread identifier
        :ptype thread_id: str
        :param customer: the customer this call addresses; omit to use the
            scope's own answer
        :ptype customer: UUID | None
        :return: nothing
        :rtype: None
        """
        if self._l1 is None:
            return
        storage_thread_id = self.storage_thread_id(thread_id, customer=customer)
        try:
            await self._l1.delete(storage_thread_id)
        except Exception:
            log.warning("L1 checkpoint delete failed", exc_info=True)

    # ------------------------------------------------------------------
    # L2 helpers
    # ------------------------------------------------------------------

    def l2_key(self, thread_id: str, checkpoint_ns: str, *, customer: UUID | None = None) -> str:
        """build L2 cache key from thread and namespace.

        Built on :meth:`storage_thread_id`, so a key carries the call's customer
        and two customers sharing one bucket cannot read each other's bundles
        even when they chose the same thread id. Under a config-resolved scope
        that bucket is shared by construction -- one saver, one bucket, every
        customer -- which is exactly why the key rather than the bucket carries
        the customer.

        :param thread_id: conversation/thread identifier as the caller knows it
        :ptype thread_id: str
        :param checkpoint_ns: checkpoint namespace
        :ptype checkpoint_ns: str
        :param customer: the customer this call addresses; omit to use the
            scope's own answer
        :ptype customer: UUID | None
        :return: composite cache key
        :rtype: str
        """
        storage_thread_id = self.storage_thread_id(thread_id, customer=customer)
        if checkpoint_ns == "":
            result = storage_thread_id
        else:
            result = f"{storage_thread_id}.{checkpoint_ns}"
        return result

    async def l2_get(self, thread_id: str, checkpoint_ns: str, *, customer: UUID | None = None) -> bytes | None:
        """read from L2 cache, returning None on miss or error.

        :param thread_id: conversation/thread identifier
        :ptype thread_id: str
        :param checkpoint_ns: checkpoint namespace
        :ptype checkpoint_ns: str
        :param customer: the customer this call addresses; omit to use the
            scope's own answer
        :ptype customer: UUID | None
        :return: cached blob or None
        :rtype: bytes | None
        """
        result: bytes | None = None
        if self._l2 is not None:
            # scoped OUTSIDE the guard, for the reason given in :meth:`l1_get`
            key = self.l2_key(thread_id, checkpoint_ns, customer=customer)
            try:
                result = await self._l2.get(self._l2_bucket, key)
            except Exception:
                log.warning("L2 checkpoint read failed", exc_info=True)
                result = None
        return result

    async def l2_put(self, thread_id: str, checkpoint_ns: str, data: bytes, *, customer: UUID | None = None) -> None:
        """write to L2 cache, swallowing errors.

        :param thread_id: conversation/thread identifier
        :ptype thread_id: str
        :param checkpoint_ns: checkpoint namespace
        :ptype checkpoint_ns: str
        :param data: serialized cache blob
        :ptype data: bytes
        :param customer: the customer this call addresses; omit to use the
            scope's own answer
        :ptype customer: UUID | None
        :return: nothing
        :rtype: None
        """
        if self._l2 is None:
            return
        key = self.l2_key(thread_id, checkpoint_ns, customer=customer)
        try:
            await self._l2.put(self._l2_bucket, key, data)
        except Exception:
            log.warning("L2 checkpoint write failed", exc_info=True)

    async def l2_delete(self, thread_id: str, checkpoint_ns: str, *, customer: UUID | None = None) -> None:
        """delete one thread+namespace L2 entry, swallowing errors.

        Takes ``checkpoint_ns`` because L2 is keyed on it and the protocol offers
        only exact-key deletes -- no prefix sweep, no listing. This used to
        hardcode ``""`` and so cleared only the root-namespace key, which meant a
        caller invalidating a namespaced thread silently left the stale bundle in
        place. The parameter makes the caller name the key it wrote, and turns
        "every namespace" from a thing the signature implied into a thing a caller
        has to loop for.

        :param thread_id: conversation/thread identifier
        :ptype thread_id: str
        :param checkpoint_ns: checkpoint namespace whose entry to drop
        :ptype checkpoint_ns: str
        :param customer: the customer this call addresses; omit to use the
            scope's own answer
        :ptype customer: UUID | None
        :return: nothing
        :rtype: None
        """
        if self._l2 is None:
            return
        key = self.l2_key(thread_id, checkpoint_ns, customer=customer)
        try:
            await self._l2.delete(self._l2_bucket, key)
        except Exception:
            log.warning("L2 checkpoint delete failed", exc_info=True)

    async def l2_delete_prefix(self, prefix: str) -> bool:
        """sweep every L2 key under ``prefix``, reporting whether it happened.

        :meth:`l2_delete` is exact-key because :class:`CheckpointL2Cache` offers
        nothing else, which is why a purge could clear a thread's root-namespace
        bundle and leave every ``thread.checkpoint_ns`` bundle cached. A cache
        that also satisfies :class:`CheckpointL2PrefixCache` closes that; one
        that does not is not an error, so the caller learns which happened from
        the return value and can say so rather than purging silently and
        incompletely.

        Failures degrade like every other L2 operation: a purge whose L3 half
        already committed must not abort because a cache timed out. What the
        caller loses is the sweep, which the ``False`` return reports.

        :param prefix: key prefix to sweep, including any separator
        :ptype prefix: str
        :return: True when a prefix-capable cache completed the sweep
        :rtype: bool
        """
        if not isinstance(self._l2, CheckpointL2PrefixCache):
            return False
        try:
            await self._l2.delete_prefix(self._l2_bucket, prefix)
        except Exception:
            log.warning("L2 checkpoint prefix sweep failed", extra={"prefix": prefix}, exc_info=True)
            return False
        return True

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def serialize_checkpoint_tuple(
        self,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        parent_checkpoint_id: str | None,
        pending_writes: list[tuple[str, str, Any]] | None,
    ) -> bytes:
        """serialize full checkpoint tuple for cache storage (L1/L2).

        public extension point for subclasses that customize
        checkpoint envelope serialization. override in tandem with
        :meth:`deserialize_checkpoint_tuple` to change the on-wire
        cache format; the base implementation prepends the serde
        type tag so the matching deserializer can recover the
        encoded payload without ambiguity.

        :param checkpoint: checkpoint state
        :ptype checkpoint: Checkpoint
        :param metadata: checkpoint metadata
        :ptype metadata: CheckpointMetadata
        :param parent_checkpoint_id: parent checkpoint id or None
        :ptype parent_checkpoint_id: str | None
        :param pending_writes: optional pending-writes list
        :ptype pending_writes: list[tuple[str, str, Any]] | None
        :return: encoded cache blob
        :rtype: bytes
        """
        bundle = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "parent_checkpoint_id": parent_checkpoint_id,
            "pending_writes": pending_writes,
        }
        _type, blob = self.serde.dumps_typed(bundle)
        type_bytes = _type.encode("utf-8")
        return len(type_bytes).to_bytes(4, "big") + type_bytes + blob

    def deserialize_checkpoint_tuple(self, data: bytes) -> dict[str, Any]:
        """deserialize checkpoint tuple from cache blob.

        public extension point for subclasses that customize
        checkpoint envelope serialization. must round-trip with
        :meth:`serialize_checkpoint_tuple`; the base implementation
        reads the serde type tag prepended by the matching serializer
        and hands the remainder to the serde's typed loader.

        :param data: cache blob produced by
            :meth:`serialize_checkpoint_tuple`
        :ptype data: bytes
        :return: decoded bundle dict
        :rtype: dict[str, Any]
        """
        type_len = int.from_bytes(data[:4], "big")
        type_str = data[4 : 4 + type_len].decode("utf-8")
        blob = data[4 + type_len :]
        result: dict[str, Any] = self.serde.loads_typed((type_str, blob))
        return result

    def _bundle_to_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        bundle: dict[str, Any],
    ) -> CheckpointTuple:
        """convert a deserialized cache bundle back to CheckpointTuple.

        :param thread_id: conversation/thread identifier
        :ptype thread_id: str
        :param checkpoint_ns: checkpoint namespace
        :ptype checkpoint_ns: str
        :param bundle: bundle from
            :meth:`deserialize_checkpoint_tuple`
        :ptype bundle: dict[str, Any]
        :return: reconstituted checkpoint tuple
        :rtype: CheckpointTuple
        """
        checkpoint = bundle["checkpoint"]
        metadata = bundle["metadata"]
        parent_checkpoint_id = bundle.get("parent_checkpoint_id")
        pending_writes = bundle.get("pending_writes")

        cp_id = checkpoint.get("id", "")

        result_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": cp_id,
            },
        }
        parent_config: RunnableConfig | None = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                },
            }
            if parent_checkpoint_id
            else None
        )

        return CheckpointTuple(
            config=result_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    # ------------------------------------------------------------------
    # Async interface -- required by LangGraph
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """fetch latest or specific checkpoint: L1 -> L2 -> L3.

        :param config: runnable config with ``thread_id`` in
            ``configurable``
        :ptype config: RunnableConfig
        :return: checkpoint tuple or None when nothing is stored
        :rtype: CheckpointTuple | None
        :raises TypeError: when a config-resolved scope finds a non-UUID customer
        :raises ValueError: when a config-resolved scope finds no usable customer
        """
        # resolved FIRST, before any tier is touched: under a config-resolved
        # scope a missing customer must refuse rather than read, and a refusal
        # that lands after a read has already returned another customer's row is
        # no refusal at all.
        customer = self.customer_for_config(config)
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        if checkpoint_id is None:
            # --- L1 attempt ---
            cached = await self.l1_get(thread_id, checkpoint_ns, customer=customer)
            if cached is not None:
                try:
                    bundle = self.deserialize_checkpoint_tuple(cached)
                    return self._bundle_to_tuple(thread_id, checkpoint_ns, bundle)
                except Exception:
                    log.warning(
                        "L1 checkpoint deserialization failed, falling through",
                        exc_info=True,
                    )

            # --- L2 attempt ---
            cached = await self.l2_get(thread_id, checkpoint_ns, customer=customer)
            if cached is not None:
                try:
                    bundle = self.deserialize_checkpoint_tuple(cached)
                    tup = self._bundle_to_tuple(thread_id, checkpoint_ns, bundle)
                    await self.l1_put(thread_id, checkpoint_ns, cached, customer=customer)
                    return tup
                except Exception:
                    log.warning(
                        "L2 checkpoint deserialization failed, falling through",
                        exc_info=True,
                    )

        # --- L3 (executor-backed) ---
        return await self._l3_get_tuple(thread_id, checkpoint_ns, checkpoint_id, customer=customer)

    async def _l3_get_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str | None,
        *,
        customer: UUID | None,
    ) -> CheckpointTuple | None:
        """load checkpoint from the executor (L3 tier).

        :param thread_id: conversation/thread identifier
        :ptype thread_id: str
        :param checkpoint_ns: checkpoint namespace
        :ptype checkpoint_ns: str
        :param checkpoint_id: specific checkpoint id or None for
            latest
        :ptype checkpoint_id: str | None
        :param customer: the customer this call addresses, already resolved by
            the caller
        :ptype customer: UUID | None
        :return: checkpoint tuple or None
        :rtype: CheckpointTuple | None
        """
        # the statements address the customer-scoped keyspace; every config
        # built below reports the caller's own thread id, because LangGraph
        # feeds a returned config straight back into the next call and a leaked
        # prefix would be scoped a second time.
        storage_thread_id = self.storage_thread_id(thread_id, customer=customer)

        if checkpoint_id:
            row = await self._exec.fetchrow(
                "SELECT checkpoint_id, parent_checkpoint_id, type, "
                "checkpoint, metadata_ "
                "FROM checkpoints "
                "WHERE thread_id = $1 AND checkpoint_ns = $2 "
                "AND checkpoint_id = $3",
                storage_thread_id,
                checkpoint_ns,
                checkpoint_id,
            )
        else:
            row = await self._exec.fetchrow(
                "SELECT checkpoint_id, parent_checkpoint_id, type, "
                "checkpoint, metadata_ "
                "FROM checkpoints "
                "WHERE thread_id = $1 AND checkpoint_ns = $2 "
                "ORDER BY checkpoint_id DESC LIMIT 1",
                storage_thread_id,
                checkpoint_ns,
            )

        if row is None:
            return None

        cp_id = row["checkpoint_id"]
        parent_id = row["parent_checkpoint_id"]
        cp_type = row["type"]
        cp_blob = bytes(row["checkpoint"])
        md_blob = bytes(row["metadata_"])

        checkpoint: Checkpoint = self.serde.loads_typed((cp_type or "msgpack", cp_blob))
        metadata: CheckpointMetadata = cast(
            CheckpointMetadata,
            (self.serde.loads_typed((cp_type or "msgpack", md_blob)) if md_blob and md_blob != b"\x00" else {}),
        )

        write_rows = await self._exec.fetch(
            "SELECT task_id, channel, type, blob "
            "FROM checkpoint_writes "
            "WHERE thread_id = $1 AND checkpoint_ns = $2 "
            "AND checkpoint_id = $3 "
            "ORDER BY idx",
            storage_thread_id,
            checkpoint_ns,
            cp_id,
        )
        pending_writes: list[tuple[str, str, Any]] = []
        for wr in write_rows:
            pending_writes.append(
                (
                    wr["task_id"],
                    wr["channel"],
                    self.serde.loads_typed(
                        (wr["type"] or "msgpack", bytes(wr["blob"])),
                    ),
                ),
            )

        result_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": cp_id,
            },
        }
        parent_config: RunnableConfig | None = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                },
            }
            if parent_id
            else None
        )

        tup = CheckpointTuple(
            config=result_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

        # Warm L1 and L2
        try:
            cache_blob = self.serialize_checkpoint_tuple(
                checkpoint,
                metadata,
                parent_id,
                pending_writes,
            )
            await self.l2_put(thread_id, checkpoint_ns, cache_blob, customer=customer)
            await self.l1_put(thread_id, checkpoint_ns, cache_blob, customer=customer)
        except Exception:
            log.warning("Failed to warm caches after L3 read", exc_info=True)

        return tup

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """list checkpoints via executor.

        :param config: runnable config with ``thread_id``
        :ptype config: RunnableConfig | None
        :param filter: optional metadata filter
        :ptype filter: dict[str, Any] | None
        :param before: only return checkpoints before this config
        :ptype before: RunnableConfig | None
        :param limit: max number of checkpoints to return
        :ptype limit: int | None
        :return: async iterator of checkpoint tuples
        :rtype: AsyncIterator[CheckpointTuple]
        :raises TypeError: when a config-resolved scope finds a non-UUID customer
        :raises ValueError: when a config-resolved scope finds no usable customer
        """
        if config is None:
            return

        customer = self.customer_for_config(config)
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")

        query = (
            "SELECT checkpoint_id, parent_checkpoint_id, type, "
            "checkpoint, metadata_ "
            "FROM checkpoints "
            "WHERE thread_id = $1 AND checkpoint_ns = $2"
        )
        params: list[Any] = [self.storage_thread_id(thread_id, customer=customer), checkpoint_ns]

        if before and (before_id := get_checkpoint_id(before)):
            query += f" AND checkpoint_id < ${len(params) + 1}"
            params.append(before_id)

        query += " ORDER BY checkpoint_id DESC"

        if limit is not None:
            query += f" LIMIT ${len(params) + 1}"
            params.append(limit)

        rows = await self._exec.fetch(query, *params)

        for row in rows:
            cp_id = row["checkpoint_id"]
            parent_id = row["parent_checkpoint_id"]
            cp_type = row["type"]
            cp_blob = bytes(row["checkpoint"])
            md_blob = bytes(row["metadata_"])

            checkpoint: Checkpoint = self.serde.loads_typed(
                (cp_type or "msgpack", cp_blob),
            )
            metadata: CheckpointMetadata = cast(
                CheckpointMetadata,
                (self.serde.loads_typed((cp_type or "msgpack", md_blob)) if md_blob and md_blob != b"\x00" else {}),
            )

            if filter and not all(metadata.get(k) == v for k, v in filter.items()):
                continue

            result_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": cp_id,
                },
            }
            parent_config: RunnableConfig | None = (
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    },
                }
                if parent_id
                else None
            )

            yield CheckpointTuple(
                config=result_config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=None,
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """store a checkpoint: write to L3, warm L2 and L1, flush.

        :param config: runnable config with ``thread_id``
        :ptype config: RunnableConfig
        :param checkpoint: checkpoint state to store
        :ptype checkpoint: Checkpoint
        :param metadata: checkpoint metadata
        :ptype metadata: CheckpointMetadata
        :param new_versions: channel version updates
        :ptype new_versions: ChannelVersions
        :return: config with ``checkpoint_id`` set
        :rtype: RunnableConfig
        :raises TypeError: when a config-resolved scope finds a non-UUID customer
        :raises ValueError: when a config-resolved scope finds no usable customer
        """
        # resolved before anything is serialized or written, so a call that
        # names no customer under a config-resolved scope writes nothing at all
        # rather than landing a row in the un-tenanted keyspace.
        customer = self.customer_for_config(config)
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        serializable_metadata = get_checkpoint_metadata(config, metadata)

        cp_type, cp_blob = self.serde.dumps_typed(checkpoint)
        _md_type, md_blob = self.serde.dumps_typed(serializable_metadata)

        # --- L3: executor (source of truth) ---
        await self._exec.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
            "type, checkpoint, metadata_) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7) "
            "ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) "
            "DO UPDATE SET parent_checkpoint_id = EXCLUDED.parent_checkpoint_id, "
            "type = EXCLUDED.type, checkpoint = EXCLUDED.checkpoint, "
            "metadata_ = EXCLUDED.metadata_",
            self.storage_thread_id(thread_id, customer=customer),
            checkpoint_ns,
            checkpoint["id"],
            parent_checkpoint_id,
            cp_type,
            cp_blob,
            md_blob,
        )

        result_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            },
        }

        # --- Warm L2 and L1 caches ---
        try:
            cache_blob = self.serialize_checkpoint_tuple(
                checkpoint,
                serializable_metadata,
                parent_checkpoint_id,
                [],
            )
            await self.l2_put(thread_id, checkpoint_ns, cache_blob, customer=customer)
            await self.l1_put(thread_id, checkpoint_ns, cache_blob, customer=customer)
        except Exception:
            log.warning("Failed to warm caches after L3 write", exc_info=True)

        # --- Flush callback ---
        if self._flush_callback is not None:
            try:
                flushed = await self._flush_callback()
                if flushed > 0:
                    log.debug(
                        "Flushed pending writes on checkpoint",
                        extra={"flushed_count": flushed},
                    )
            except Exception:
                log.warning(
                    "Failed to flush pending writes on checkpoint",
                    exc_info=True,
                )

        return result_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """store intermediate writes, degrading only where degrading is safe.

        Two kinds of row arrive here and they get OPPOSITE treatment, because the
        cost of losing them differs in kind:

        - **crash-recovery rows** (ordinary channel writes) let a CRASHED run
          resume. Losing one costs resumability of a turn that has already
          answered, so a failure DEGRADES with a warning.
        - **control-channel writes** (the members of
          :data:`~langgraph.checkpoint.base.WRITES_IDX_MAP` --
          ``__error__``, ``__scheduled__``, ``__interrupt__``, ``__resume__``)
          carry the run's control flow, not merely its resumability. A failure
          RAISES.

        Why the split is not cosmetic: pregel builds a state snapshot's
        interrupts out of the very rows this method writes
        (``saved.pending_writes``), and
        :func:`~threetears.langgraph.streaming.detect_interrupt` derives "the
        graph PAUSED" solely from that snapshot. So swallowing a failed
        ``__interrupt__`` write leaves a snapshot with no interrupt in it, the
        turn ends as an ordinary ``StreamEndEvent``, and a human-approval gate is
        silently skipped with its payload lost. ``detect_interrupt`` already
        refuses to swallow a failing ``aget_state`` for exactly this reason; a
        blanket guard here would reintroduce that bug one layer down.

        LangGraph calls this from its executor teardown, so anything raised here
        propagates out of ``run_graph`` and terminates the whole turn. For a
        crash-recovery row that trade is inverted -- it discards an answer that
        already exists to protect the ability to recover an answer nobody needs
        any more. Observed on a cluster: two identical L3 timeouts on the same
        NATS subject inside one log window. The first hit memory retrieval, which
        guards its call, and soft-failed -- that turn answered normally. The
        second hit this method, which guarded nothing, and the turn died as
        ``AGENT_FAILED``. For a control-channel write the trade runs the other
        way: a turn that dies loudly is recoverable, an approval gate that
        vanishes is not.

        The write set is persisted in ONE statement (see :meth:`_write_pending`),
        so a failure leaves NO rows behind. What that costs is a REPLAY, not a
        lost turn: pregel treats a task with any writes as already run, so a task
        with none looks not-run and its node RE-EXECUTES when the run resumes,
        repeating whatever side effects it already performed. The reason to
        prefer that over a truncated set is that pregel also applies a task's
        pending writes to channels, so half a set resumes by SKIPPING the node
        with partially updated channels -- silent divergence, which is worse than
        doing the work twice.

        :param config: runnable config with ``thread_id`` and
            ``checkpoint_id``
        :ptype config: RunnableConfig
        :param writes: list of (channel, value) tuples
        :ptype writes: Sequence[tuple[str, Any]]
        :param task_id: task identifier for crash recovery
        :ptype task_id: str
        :param task_path: optional task path
        :ptype task_path: str
        :return: nothing
        :rtype: None
        :raises TypeError: when a config-resolved scope finds a non-UUID customer
        :raises ValueError: when a config-resolved scope finds no usable customer
        :raises Exception: whatever the write raises, when the set contains a
            control-channel write
        """
        control_channels = sorted({channel for channel, _ in writes if channel in WRITES_IDX_MAP})

        # resolved OUTSIDE the degrade guard below, alongside serialization and
        # for the same reason: the guard exists for transport faults on the
        # crash-recovery path, and a host that never supplied a customer is a
        # programming error that fails identically every time. Degrading it would
        # hide a permanent tenancy bug behind one warning per turn.
        customer = self.customer_for_config(config)

        # ``or {}`` rather than a .get default: a config carrying an explicit
        # ``configurable: None`` would still raise AttributeError on the next
        # lookup, which is the shape a malformed config most often takes.
        configurable = config.get("configurable") or {}
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable.get("checkpoint_id")

        # Serialization happens OUTSIDE the guard below. An unserializable
        # channel value is a programming error, not a transport fault: it fails
        # identically on every retry, so degrading it would hide a permanent bug
        # behind one warning per turn. Only the write itself is guarded.
        #
        # The isinstance guard below, rather than an unconditional call:
        # ``thread_id`` arrives out of the config unvalidated, and a non-str one
        # has no scoped form. Passing it through unchanged keeps the existing
        # behaviour -- the statement rejects it, so no row lands -- whereas
        # coercing it would invent a thread id nobody asked for.
        rows = self._build_write_rows(
            writes,
            task_id,
            storage_thread_id=(
                self.storage_thread_id(thread_id, customer=customer) if isinstance(thread_id, str) else thread_id
            ),
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            task_path=task_path,
        )

        try:
            await self._write_pending(rows)
        except Exception:  # prawduct:allow prawduct/broad-except -- teardown path: a crash-recovery row must degrade, never terminate the turn; control-channel writes re-raise below
            if control_channels:
                # Losing these changes what the run DOES. Fail the turn loudly
                # rather than let it resolve to a wrong control flow. Logged
                # before re-raising because the channel names are the whole
                # diagnosis and the traceback does not carry them.
                log.warning(
                    "Failed to store control-channel writes; re-raising rather than skipping the control flow",
                    extra={
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "control_channels": control_channels,
                    },
                    exc_info=True,
                )
                raise
            # WARNING, not debug: a silent replay is worth one line, and the
            # cluster incident that motivated this guard was invisible until
            # someone read a traceback.
            log.warning(
                "Failed to store crash-recovery writes; this task will re-execute if the run resumes",
                extra={
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "write_count": len(writes),
                },
                exc_info=True,
            )
        else:
            # isinstance rather than a cast: thread_id comes from the config
            # unvalidated, and a write that succeeded without one has nothing
            # cacheable to invalidate.
            if control_channels and isinstance(thread_id, str):
                # The cached bundle is written by :meth:`aput` with
                # ``pending_writes=[]``, and :meth:`aget_tuple` serves it
                # verbatim when no ``checkpoint_id`` is pinned -- which is
                # exactly how ``detect_interrupt`` reads state. Without this
                # invalidation a wired L1/L2 would hide the interrupt row we
                # just persisted, and the approval gate would vanish anyway.
                #
                # Deliberately NOT via l1_delete / l2_delete: those swallow their
                # own failures by design, and a silently-failed invalidation here
                # leaves the cache serving the pre-interrupt bundle -- the same end
                # state as never writing the row, which is the failure the re-raise
                # above exists to prevent. This invalidation is load-bearing, so it
                # fails the way the write does.
                await self._invalidate_for_control_write(
                    thread_id,
                    checkpoint_ns if isinstance(checkpoint_ns, str) else "",
                    customer=customer,
                )

    async def _invalidate_for_control_write(
        self,
        thread_id: str,
        checkpoint_ns: str,
        *,
        customer: UUID | None,
    ) -> None:
        """drop the cached bundle for a control-channel write, RAISING on failure.

        The read caches hold a bundle :meth:`aput` wrote with
        ``pending_writes=[]``, and :meth:`aget_tuple` serves it verbatim when no
        ``checkpoint_id`` is pinned -- which is exactly how interrupt detection
        reads state. So a control-channel row that reached L3 is still invisible
        until the cache is dropped, and an invalidation that fails quietly leaves
        the run in the same place as one that never happened: no interrupt in the
        snapshot, an ordinary end of turn, a human-approval gate skipped.

        That is why this bypasses :meth:`l1_delete` / :meth:`l2_delete`. Those
        degrade on purpose, which is right for opportunistic cache warming and
        wrong here -- losing this invalidation changes what the run does, so it
        gets the same treatment as losing the write itself.

        L1 is dropped across every namespace by its protocol; L2 is exact-key, so
        only the namespace just written is cleared.

        :param thread_id: conversation/thread identifier
        :ptype thread_id: str
        :param checkpoint_ns: the namespace whose cached bundle is now stale
        :ptype checkpoint_ns: str
        :param customer: the customer this call addresses, already resolved by
            the caller
        :ptype customer: UUID | None
        :return: nothing
        :rtype: None
        :raises Exception: whatever the cache raises; the caller does not degrade
        """
        if self._l1 is not None:
            await self._l1.delete(self.storage_thread_id(thread_id, customer=customer))
        if self._l2 is not None:
            await self._l2.delete(self._l2_bucket, self.l2_key(thread_id, checkpoint_ns, customer=customer))

    def _build_write_rows(
        self,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        *,
        storage_thread_id: Any,
        checkpoint_ns: Any,
        checkpoint_id: Any,
        task_path: str,
    ) -> list[tuple[Any, ...]]:
        """serialize a write set into per-row parameter tuples, deduplicated.

        Deduplication mirrors the reference saver's ``put_writes`` exactly, and
        the two halves differ: an ordinary channel (``idx >= 0``) is FIRST-wins,
        a control channel (negative index, from
        :data:`~langgraph.checkpoint.base.WRITES_IDX_MAP`) is LAST-wins. Getting
        that backwards drops a second ``Command(resume=...)`` against the same
        checkpoint and task, so the graph resumes on the STALE earlier approve or
        deny -- a wrong human decision applied silently, which is the failure the
        control-channel guard exists to prevent.

        Deduplicating here rather than in SQL is forced: the statement upserts
        control rows with ``DO UPDATE``, and Postgres refuses a command whose own
        ``VALUES`` list hits one conflict key twice ("cannot affect row a second
        time"). Two writes on one control channel collapse to that channel's
        single index, so the collision is reachable, not theoretical.

        Splitting this out of the write also puts serialization OUTSIDE the
        caller's degrade guard, so an unserializable value surfaces as the
        programming error it is instead of degrading identically forever.

        The ids are typed :class:`~typing.Any` because they arrive out of
        ``config["configurable"]`` unvalidated; they ride as bound parameters, so
        a missing one fails the statement rather than corrupting a row. They are
        keyword-only for the same reason they are all ``Any``: three
        interchangeable ids in a row is a transposition that no type checker
        would catch.

        :param storage_thread_id: the customer-scoped identifier the rows are
            stored under, from :meth:`storage_thread_id`
        :ptype storage_thread_id: Any
        :param checkpoint_ns: checkpoint namespace
        :ptype checkpoint_ns: Any
        :param checkpoint_id: checkpoint identifier
        :ptype checkpoint_id: Any
        :param writes: list of (channel, value) tuples
        :ptype writes: Sequence[tuple[str, Any]]
        :param task_id: task identifier for crash recovery
        :ptype task_id: str
        :param task_path: task path
        :ptype task_path: str
        :return: one parameter tuple per surviving write, in insertion order
        :rtype: list[tuple[Any, ...]]
        """
        by_idx: dict[int, tuple[Any, ...]] = {}
        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            if write_idx >= 0 and write_idx in by_idx:
                # ordinary channel: the first write for an index wins
                continue
            w_type, w_blob = self.serde.dumps_typed(value)
            by_idx[write_idx] = (
                storage_thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                write_idx,
                channel,
                w_type,
                w_blob,
            )
        return list(by_idx.values())

    async def _write_pending(self, rows: Sequence[tuple[Any, ...]]) -> None:
        """persist prepared write rows as ONE statement, raising on failure.

        The whole set goes in a single multi-row ``INSERT`` so it is
        all-or-nothing: a failure leaves no rows at all, never a truncated set. A
        statement-per-write cannot promise that, because the
        :class:`~threetears.langgraph.protocols.AsyncQueryExecutor` protocol
        declares no transaction. A batch API was the alternative and was
        rejected: ``NatsProxyL3Backend`` does expose
        ``execute_batch(..., transaction=True)``, but the protocol does not
        declare it and the direct-pool adapter does not implement it, so taking
        that route means widening the protocol and teaching the adapter
        transactions to buy an atomicity that one statement already has.

        The upsert is asymmetric on purpose, matching the reference saver: an
        ordinary channel keeps the row already stored, a control channel
        overwrites it. ``DO UPDATE ... WHERE checkpoint_writes.idx < 0`` says
        exactly that in one clause -- for a non-negative index the predicate is
        false and the statement leaves the existing row alone.

        Verified against the engine rather than assumed: a mid-statement failure
        commits zero rows, a control row is overwritten while an ordinary row is
        not, and a conflict key repeated inside one statement is rejected outright
        (hence the caller's deduplication).

        One statement means one parameter list, so a set is ultimately bounded by
        what the transport accepts -- the bind-parameter ceiling on a direct pool,
        ``max_payload`` on the NATS proxy. Real write sets are a handful of
        channels; a set large enough to hit either limit fails the statement and
        takes the caller's degrade path.

        :param rows: parameter tuples from :meth:`_build_write_rows`
        :ptype rows: Sequence[tuple[Any, ...]]
        :return: nothing
        :rtype: None
        :raises Exception: whatever the executor raises; the caller decides
            whether to degrade
        """
        if not rows:
            return

        row_placeholders: list[str] = []
        params: list[Any] = []
        for row in rows:
            # placeholder count derives from the row itself, so adding a column
            # cannot desynchronise the SQL from the parameter list
            row_placeholders.append("(" + ", ".join(f"${len(params) + n}" for n in range(1, len(row) + 1)) + ")")
            params.extend(row)

        await self._exec.execute(
            "INSERT INTO checkpoint_writes "
            "(thread_id, checkpoint_ns, checkpoint_id, task_id, "
            "task_path, idx, channel, type, blob) "
            f"VALUES {', '.join(row_placeholders)} "
            "ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) "
            "DO UPDATE SET "
            "task_path = EXCLUDED.task_path, "
            "channel = EXCLUDED.channel, "
            "type = EXCLUDED.type, "
            "blob = EXCLUDED.blob "
            "WHERE checkpoint_writes.idx < 0",
            *params,
        )

    async def adelete_thread(self, thread_id: str, *, customer: UUID | None = None) -> None:
        """delete all checkpoints and writes for a thread from all tiers.

        This is one of the two methods LangGraph hands NO ``RunnableConfig``, so
        a config-resolved scope has nothing to read a customer out of. The
        customer therefore arrives as a KEYWORD-ONLY argument, which keeps the
        existing call shape intact: this method has a live production consumer
        that passes a rendered session id positionally and nothing else, and
        under ``for_customer`` or ``unscoped`` the saver still answers from its
        own scope, so that call keeps working byte for byte.

        Under
        :meth:`~threetears.langgraph.checkpoint_scope.CheckpointScope.from_config`
        the argument is REQUIRED and its absence raises. A delete that cannot
        know its customer must refuse: purging under the bare thread id would
        address the un-tenanted keyspace (the wrong rows, and none of the
        caller's), and picking a customer would be a guess about whose data to
        destroy. Naming a customer a ``for_customer`` saver was not built for, or
        naming one at all on an ``unscoped`` saver, is refused for the same
        reason and names the scope in the error.

        :param thread_id: conversation/thread identifier as the caller knows it
        :ptype thread_id: str
        :param customer: whose thread this is; required under a config-resolved
            scope, refused when it contradicts any other scope
        :ptype customer: UUID | None
        :return: nothing
        :rtype: None
        :raises TypeError: when customer is neither None nor a UUID
        :raises ValueError: when the customer cannot be reconciled with the scope
        """
        # reconciled here rather than left to :meth:`storage_thread_id` so a
        # refusal names THIS method -- the one the caller invoked -- instead of
        # the key builder it happens to delegate to.
        resolved = self._scope.customer_for_operation(customer, operation="adelete_thread")
        storage_thread_id = self.storage_thread_id(thread_id, customer=resolved)

        await self._exec.execute(
            "DELETE FROM checkpoint_writes WHERE thread_id = $1",
            storage_thread_id,
        )
        await self._exec.execute(
            "DELETE FROM checkpoints WHERE thread_id = $1",
            storage_thread_id,
        )

        # L2 is exact-key, so the root-namespace entry needs its own delete and
        # every ``thread.checkpoint_ns`` entry needs a sweep. A cache that
        # cannot sweep leaves those namespaced bundles cached -- the gap this
        # method carried as a comment until an L2 was actually wired in front of
        # it. It is reported rather than assumed away: a purge that answers an
        # erasure request has to say what it could not reach.
        swept = await self.l2_delete_prefix(f"{storage_thread_id}.")
        await self.l2_delete(thread_id, "", customer=resolved)
        if not swept and self._l2 is not None:
            log.warning(
                "L2 cache cannot sweep by prefix; bundles cached under a non-empty checkpoint_ns "
                "survive this purge. Give the cache a delete_prefix (CheckpointL2PrefixCache) to close it.",
                extra={"l2_bucket": self._l2_bucket},
            )

        # L1 drops the thread across every namespace by its own protocol.
        await self.l1_delete(thread_id, customer=resolved)

    async def adelete_customer_threads(self, *, customer: UUID | None = None) -> None:
        """delete every checkpoint and write belonging to this saver's customer.

        The per-thread purge answers an erasure request aimed at one
        conversation; this answers one aimed at a whole customer -- tenant
        offboarding, or an erasure whose subject is the tenant rather than a
        conversation within it. Tenancy without a purge path is why the column
        would have been added in the first place, so the two ship together.

        The second of the two methods that receive no ``RunnableConfig``, so the
        customer arrives the same way it does on :meth:`adelete_thread` -- as a
        keyword-only argument reconciled against the scope. What each scope makes
        of it:

        - ``for_customer`` -- omit it and the saver purges the customer it was
          built for, exactly as before. Restating that same customer is accepted;
          naming a DIFFERENT one is refused, so there is still no way to ask an
          instance to erase a customer it was not built for.
        - ``from_config`` -- the argument is required, because a saver serving
          many customers with no config in hand cannot know which tenant is being
          offboarded, and omitting it would leave a pattern matching every row.
        - ``unscoped`` -- refused either way, for the reason it always was: the
          pattern would match every row in the table.

        The L3 half matches on the customer's key prefix. A UUID's text form
        contains neither ``%`` nor ``_``, so the pattern needs no ``ESCAPE``
        clause and cannot widen beyond the one customer.

        Caches are best-effort and say so. L2 is swept when it can be; L1's
        protocol deletes one thread at a time with no way to enumerate a
        customer's threads, so a wired L1 keeps its entries until they are
        evicted. Both are reported at WARNING rather than passed over, because
        a blob still served from cache after a purge is the failure the purge
        exists to prevent.

        :param customer: the customer to purge; required under a config-resolved
            scope, refused when it contradicts any other scope
        :ptype customer: UUID | None
        :return: nothing
        :rtype: None
        :raises TypeError: when customer is neither None nor a UUID
        :raises ValueError: when the customer cannot be reconciled with the
            scope, or when no customer is named at all, since the pattern would
            then match every row in the table
        """
        prefix = self._customer_prefix(
            self._scope.customer_for_operation(customer, operation="adelete_customer_threads"),
        )
        if prefix is None:
            raise ValueError(
                "adelete_customer_threads() needs a saver built with CheckpointScope.for_customer(...), "
                "or a customer= argument on a CheckpointScope.from_config(...) saver. This saver is "
                f"unscoped (reason: {self._scope.reason}), so it has no customer to scope the purge to "
                "and the pattern would match every thread in the table.",
            )

        pattern = f"{prefix}%"

        await self._exec.execute(
            "DELETE FROM checkpoint_writes WHERE thread_id LIKE $1",
            pattern,
        )
        await self._exec.execute(
            "DELETE FROM checkpoints WHERE thread_id LIKE $1",
            pattern,
        )

        if self._l2 is not None and not await self.l2_delete_prefix(prefix):
            log.warning(
                "L2 cache cannot sweep by prefix; this customer's cached checkpoint bundles survive "
                "the purge. Give the cache a delete_prefix (CheckpointL2PrefixCache) to close it.",
                extra={"l2_bucket": self._l2_bucket},
            )
        if self._l1 is not None:
            log.warning(
                "L1 cache deletes one thread at a time and cannot enumerate a customer's threads; "
                "this customer's cached checkpoint bundles survive the purge until they are evicted.",
            )

    # ------------------------------------------------------------------
    # Sync methods -- not supported (async-only application)
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """not supported. use :meth:`aget_tuple`.

        :param config: runnable config (ignored)
        :ptype config: RunnableConfig
        :return: never returns
        :rtype: CheckpointTuple | None
        :raises NotImplementedError: always
        """
        raise NotImplementedError(
            "ThreeTierCheckpointSaver is async-only. Use aget_tuple().",
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """not supported. use :meth:`alist`.

        :param config: runnable config (ignored)
        :ptype config: RunnableConfig | None
        :param filter: ignored
        :ptype filter: dict[str, Any] | None
        :param before: ignored
        :ptype before: RunnableConfig | None
        :param limit: ignored
        :ptype limit: int | None
        :return: never returns
        :rtype: Iterator[CheckpointTuple]
        :raises NotImplementedError: always
        """
        raise NotImplementedError(
            "ThreeTierCheckpointSaver is async-only. Use alist().",
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """not supported. use :meth:`aput`.

        :param config: runnable config (ignored)
        :ptype config: RunnableConfig
        :param checkpoint: ignored
        :ptype checkpoint: Checkpoint
        :param metadata: ignored
        :ptype metadata: CheckpointMetadata
        :param new_versions: ignored
        :ptype new_versions: ChannelVersions
        :return: never returns
        :rtype: RunnableConfig
        :raises NotImplementedError: always
        """
        raise NotImplementedError(
            "ThreeTierCheckpointSaver is async-only. Use aput().",
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """not supported. use :meth:`aput_writes`.

        :param config: runnable config (ignored)
        :ptype config: RunnableConfig
        :param writes: ignored
        :ptype writes: Sequence[tuple[str, Any]]
        :param task_id: ignored
        :ptype task_id: str
        :param task_path: ignored
        :ptype task_path: str
        :return: never returns
        :rtype: None
        :raises NotImplementedError: always
        """
        raise NotImplementedError(
            "ThreeTierCheckpointSaver is async-only. Use aput_writes().",
        )

    def delete_thread(self, thread_id: str) -> None:
        """not supported. use :meth:`adelete_thread`.

        :param thread_id: conversation/thread identifier (ignored)
        :ptype thread_id: str
        :return: never returns
        :rtype: None
        :raises NotImplementedError: always
        """
        raise NotImplementedError(
            "ThreeTierCheckpointSaver is async-only. Use adelete_thread().",
        )
