"""spec test for L2 key scoping (coll-task-03).

Every key in the shared ``{ns}-collections`` NATS KV bucket is
``{scope}.{table}.{body}``. **One tier: every key is scoped, always.** A two-tier
design with an opt-in shared scope was worked through and dropped -- ``$KV.`` grants are
pub-and-sub with no split, ``_pull_through`` writes L2 on every miss so no principal is
ever a pure reader, and the tables nominated for sharing carry ``customer_id``. There is
therefore no ``L2Scope`` enum anywhere in the implementation; if one appears, the tier
decision has been reopened by accident.

This file owns the SCOPE delta only. The SHA-256 body-hashing invariant, the composite-pk
join and the grammar-safe passthrough belong to ``test_base_collection.py``'s
``TestL2KeyGrammarSafe``; asserting them again here would give two homes to one contract.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import (
    CacheInvalidationMessage,
    CollectionRegistry,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import (
    InvalidL2ScopeError,
    L2ScopeError,
    L2ScopeNotConfiguredError,
)
from threetears.nats import Principal, Subjects, kv_key_scope_for
from threetears.nats.errors import KvError

_HUB_SCOPE = kv_key_scope_for(Principal.HUB)
_AGENT_SCOPE = kv_key_scope_for(Principal.AGENT_POD, agent_id=uuid.uuid4())


# ---------------------------------------------------------------------------
# test infrastructure
# ---------------------------------------------------------------------------


def _make_metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "test_entities",
        metadata,
        Column("id", String(255), primary_key=True),
        Column("name", String(255)),
        Column("score", Integer),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    return metadata


class StubEntity(BaseEntity):
    primary_key_field = "id"


class StubCollection(BaseCollection[StubEntity]):
    """single-pk collection with a dict L3, for the scope assertions."""

    def __init__(
        self,
        registry: CollectionRegistry,
        config: DefaultCoreConfig,
        nats_client: Any = None,
        l3_rows: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._l3_rows = l3_rows if l3_rows is not None else {}
        super().__init__(registry, config, nats_client)

    @property
    def table_name(self) -> str:
        return "test_entities"

    @property
    def entity_class(self) -> type[StubEntity]:
        return StubEntity

    async def fetch_from_store(self, entity_id: object) -> dict[str, Any] | None:
        return self._l3_rows.get(str(entity_id))

    async def save_to_store(self, data: dict[str, Any], original_timestamp: datetime | None = None) -> int:
        self._l3_rows[str(data.get("id"))] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: object) -> None:
        self._l3_rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        raw: dict[str, Any] = json.loads(data)
        for key in ("date_created", "date_updated"):
            if key in raw and isinstance(raw[key], str):
                raw[key] = datetime.fromisoformat(raw[key])
        return raw


# parity-with: threetears.nats.kv.NatsKvBucket
class _InMemoryKvBucket:
    """in-memory stand-in for the wrapper's KV bucket, kw-only like the real one."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def get(self, *, key: str) -> bytes | None:
        return self.store.get(key)

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
        raw = self.store.get(key)
        return None if raw is None else (raw, 1)

    async def put(self, *, key: str, value: bytes) -> int:
        self.store[key] = value
        return len(self.store)

    async def create(self, *, key: str, value: bytes) -> int | None:
        if key in self.store:
            return None
        self.store[key] = value
        return 1

    async def update(self, *, key: str, value: bytes, revision: int) -> int | None:  # noqa: ARG002
        self.store[key] = value
        return 1

    async def delete(self, *, key: str, revision: int | None = None) -> bool:
        existed = key in self.store
        self.store.pop(key, None)
        return existed or revision is None


class _SharedNatsBus:
    """one KV bucket plus typed pub/sub fan-out, shared by every pod in a test.

    Deliberately ONE bucket across pods: after scoping the isolation is intra-bucket, so a
    test that gave each pod its own store could not observe a collision even if the key
    shape reintroduced one.
    """

    def __init__(self) -> None:
        self.bucket = _InMemoryKvBucket()
        self._subscribers: dict[str, list[tuple[Any, Any]]] = {}

    async def kv_bucket(self, *, name: str, **_: Any) -> _InMemoryKvBucket:  # noqa: ARG002
        return self.bucket

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:  # noqa: ARG002
        for cb, message_type in self._subscribers.get(str(subject), []):
            await cb(message_type.model_validate_json(message.model_dump_json()))

    async def subscribe_typed(self, *, subject: Any, cb: Any, message_type: Any, **_: Any) -> None:
        self._subscribers.setdefault(str(subject), []).append((cb, message_type))


def _make_l1() -> SQLiteBackend:
    backend = SQLiteBackend(db_name=f"scope_{uuid.uuid4().hex[:8]}")
    backend.initialize(_make_metadata())
    return backend


#: stands in for a durable backend handle. :class:`StubCollection` overrides
#: ``fetch_from_store`` / ``save_to_store`` / ``delete_from_store`` against a dict, so the
#: handle itself is never used -- what it DECLARES is that this collection has an
#: authoritative L3 tier, which is what makes its L2 a cache rather than a source of truth.
_DECLARES_L3 = object()


def _make_pod(
    bus: _SharedNatsBus,
    scope: str,
    l3_rows: dict[str, dict[str, Any]],
    config: DefaultCoreConfig,
    *,
    with_l3: bool = True,
) -> tuple[StubCollection, CollectionRegistry]:
    """one principal: its own L1 and scope, the shared bucket and L3."""
    registry = CollectionRegistry()
    registry.configure(l1_backend=_make_l1(), l2_client=bus, kv_key_scope=scope)
    collection = StubCollection(registry, config, nats_client=bus, l3_rows=l3_rows)
    if with_l3:
        collection.l3_pool = _DECLARES_L3  # type: ignore[assignment]
    return collection, registry


@pytest.fixture()
def config_always() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


@pytest.fixture()
def bus() -> _SharedNatsBus:
    return _SharedNatsBus()


@pytest.fixture()
def scoped_registry() -> CollectionRegistry:
    registry = CollectionRegistry()
    registry.configure(l1_backend=_make_l1(), kv_key_scope=_HUB_SCOPE)
    return registry


# ---------------------------------------------------------------------------
# L2S-01 / L2S-03 -- the key shape and what it isolates
# ---------------------------------------------------------------------------


class TestScopedKeyShape:
    """``l2_key`` emits ``{scope}.{table}.{body}`` and nothing else declares a tier."""

    def test_undeclared_collection_keys_scope_table_body(
        self, scoped_registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """a collection that declares nothing still gets the scope segment."""
        collection = StubCollection(scoped_registry, config_always)

        assert collection.l2_key("e1") == f"{_HUB_SCOPE}.test_entities.e1"

    def test_no_l2_scope_enum_exists(self) -> None:
        """a single-tier design has no scope ENUM; one appearing means the tier split is back."""
        import threetears.core.collections.base as base_module
        import threetears.core.collections.registry as registry_module

        for module in (base_module, registry_module):
            assert not hasattr(module, "L2Scope"), f"{module.__name__} declares an L2Scope enum"

    def test_two_scopes_never_collide(self, config_always: DefaultCoreConfig) -> None:
        """the same table and the same pk under two principals are two distinct keys."""
        hub = CollectionRegistry()
        hub.configure(l1_backend=_make_l1(), kv_key_scope=_HUB_SCOPE)
        agent = CollectionRegistry()
        agent.configure(l1_backend=_make_l1(), kv_key_scope=_AGENT_SCOPE)

        hub_key = StubCollection(hub, config_always).l2_key("e1")
        agent_key = StubCollection(agent, config_always).l2_key("e1")

        assert hub_key != agent_key

    def test_two_registries_on_one_scope_agree(self, config_always: DefaultCoreConfig) -> None:
        """replicas of one principal MUST land on one key or L2 stops being a shared cache."""
        first = CollectionRegistry()
        first.configure(l1_backend=_make_l1(), kv_key_scope=_HUB_SCOPE)
        second = CollectionRegistry()
        second.configure(l1_backend=_make_l1(), kv_key_scope=_HUB_SCOPE)

        assert StubCollection(first, config_always).l2_key("e1") == StubCollection(second, config_always).l2_key("e1")

    def test_scope_survives_body_hashing(
        self, scoped_registry: CollectionRegistry, config_always: DefaultCoreConfig
    ) -> None:
        """an out-of-grammar pk hashes the BODY only; the scope stays readable in the grant."""
        collection = StubCollection(scoped_registry, config_always)

        key = collection.l2_key("cust:story:main:scene.md")

        scope, _, remainder = key.partition(".")
        table, _, body = remainder.partition(".")
        assert scope == _HUB_SCOPE
        assert table == "test_entities"
        assert len(body) == 64


# ---------------------------------------------------------------------------
# L2S-02 / L2S-04 -- fail at wiring time, over merged state
# ---------------------------------------------------------------------------


class TestConfigureRefusesAnUnscopedL2Client:
    """``configure()`` is the primary gate, and it reads registry state, not arguments."""

    def test_l2_client_without_a_scope_raises_naming_the_argument(self) -> None:
        """the message has to say which argument is missing or the fix is a guess."""
        registry = CollectionRegistry()

        with pytest.raises(L2ScopeNotConfiguredError, match="kv_key_scope"):
            registry.configure(l2_client=object())

    def test_scope_then_client_succeeds(self) -> None:
        """two-pass wiring, scope first: normal at several call sites."""
        registry = CollectionRegistry()

        registry.configure(kv_key_scope=_HUB_SCOPE)
        registry.configure(l2_client=object())

        assert registry.kv_key_scope == _HUB_SCOPE

    def test_client_then_scope_is_refused_on_the_first_pass(self) -> None:
        """the reverse order fails LOUD on the pass that leaves the registry unscoped.

        The refusal is over merged state, so it fires the moment a client is present with
        no scope -- it does not wait to see whether a later call supplies one. Passing the
        client first is therefore a wiring order to fix, not a shape to tolerate: an
        unscoped L2 client is exactly the state this refusal exists to prevent a process
        from reaching.
        """
        registry = CollectionRegistry()

        with pytest.raises(L2ScopeNotConfiguredError):
            registry.configure(l2_client=object())

    def test_a_later_call_that_supplies_neither_still_passes(self) -> None:
        """once wired, an unrelated ``configure`` pass must not re-raise on merged state."""
        registry = CollectionRegistry()
        registry.configure(kv_key_scope=_HUB_SCOPE, l2_client=object())

        registry.configure(l1_backend=_make_l1())

        assert registry.kv_key_scope == _HUB_SCOPE

    def test_l1_only_registry_needs_no_scope(self) -> None:
        """the refusal is about L2; an L1-only or L3-only registry is untouched."""
        registry = CollectionRegistry()

        registry.configure(l1_backend=_make_l1())

        assert registry.kv_key_scope is None

    @pytest.mark.parametrize("bad", ["hub.replica", "hub/replica", "", "hub replica", "hub>", "hub*"])
    def test_scope_outside_the_grammar_is_refused(self, bad: str) -> None:
        """the scope is ONE subject token: a dot or a slash in it silently defeats the grant."""
        registry = CollectionRegistry()

        with pytest.raises(InvalidL2ScopeError):
            registry.configure(kv_key_scope=bad)

    def test_the_scope_grammar_is_stricter_than_the_key_grammar(self) -> None:
        """``_KV_KEY_GRAMMAR`` admits ``.`` and ``/``; reusing it here would validate nothing."""
        from threetears.core.collections.base import _KV_KEY_GRAMMAR  # noqa: PLC0415

        assert _KV_KEY_GRAMMAR.match("hub.replica")
        registry = CollectionRegistry()
        with pytest.raises(InvalidL2ScopeError):
            registry.configure(kv_key_scope="hub.replica")


# ---------------------------------------------------------------------------
# L2S-05 -- the backstop, and the type it must not be
# ---------------------------------------------------------------------------


class TestBackstopRaiseInL2Key:
    """the ``nats_client=``-direct path never calls ``configure``, so ``l2_key`` backstops it."""

    def test_unscoped_registry_raises_from_l2_key(self, config_always: DefaultCoreConfig) -> None:
        """a collection handed a client directly on an unscoped registry cannot key anything."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=_make_l1())
        collection = StubCollection(registry, config_always, nats_client=_SharedNatsBus())

        with pytest.raises(L2ScopeNotConfiguredError):
            collection.l2_key("e1")

    def test_the_backstop_is_not_a_kv_error(self) -> None:
        """a ``KvError`` would be SWALLOWED at three of ``l2_key``'s four call sites.

        ``_get_from_l2`` / ``_save_to_l2`` / ``_delete_from_l2`` each degrade a ``KvError``
        to a warning, so the fleet would run with L2 silently off -- the exact degradation
        the fail-loud decision exists to prevent. ``l2_cas_mutate`` deliberately does not
        degrade, so a ``KvError`` would additionally be inconsistent between the four.
        """
        assert not issubclass(L2ScopeNotConfiguredError, KvError)
        assert not issubclass(InvalidL2ScopeError, KvError)
        assert not issubclass(L2ScopeError, KvError)

    @pytest.mark.asyncio
    async def test_the_degrading_l2_accessors_do_not_catch_it(self, config_always: DefaultCoreConfig) -> None:
        """the read path must surface the missing scope, not warn and return ``None``."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=_make_l1())
        collection = StubCollection(
            registry,
            config_always,
            nats_client=_SharedNatsBus(),
            l3_rows={"e1": {"id": "e1", "name": "Alice", "score": 1}},
        )

        with pytest.raises(L2ScopeNotConfiguredError):
            await collection.get("e1")


# ---------------------------------------------------------------------------
# L2S-06 / L2S-08 -- the scope helper
# ---------------------------------------------------------------------------


class TestKvKeyScopeFor:
    """the ONE producer of a scope value."""

    def test_an_infra_scope_is_the_bare_principal_value(self) -> None:
        """an infra principal has one identity per service and no per-connection id."""
        assert kv_key_scope_for(Principal.REGISTRY) == Principal.REGISTRY.value

    def test_an_infra_scope_is_accepted_by_configure(self) -> None:
        """the collision ban lives in the helper, NOT in ``configure``.

        ``configure`` sees a bare string and cannot tell an infra scope from a pod one, so a
        blanket "may not equal a Principal value" check there would refuse every infra
        principal at startup.
        """
        registry = CollectionRegistry()

        registry.configure(kv_key_scope=kv_key_scope_for(Principal.REGISTRY))

        assert registry.kv_key_scope == "registry"

    @pytest.mark.parametrize("principal", [Principal.REGISTRY, Principal.HUB, Principal.GATEWAY])
    def test_an_infra_principal_takes_no_id(self, principal: Principal) -> None:
        """passing one means the caller believes it is used; say so rather than ignore it."""
        with pytest.raises(ValueError, match="takes no id"):
            kv_key_scope_for(principal, agent_id=uuid.uuid4())

    def test_agent_pod_refuses_pod_id(self) -> None:
        """for AGENT_POD the callout derives pod id from the ATTACKER-INFLUENCED connect name."""
        with pytest.raises(ValueError, match="never pod_id"):
            kv_key_scope_for(Principal.AGENT_POD, pod_id=uuid.uuid4())

    def test_agent_pod_requires_agent_id(self) -> None:
        """no fallback: a fleet of agents on one shared scope is the failure being prevented."""
        with pytest.raises(ValueError, match="agent_id"):
            kv_key_scope_for(Principal.AGENT_POD)

    def test_tool_pod_requires_pod_id(self) -> None:
        """``tool_pods.id`` is the pod's authenticated ``claims.sub``; there is no substitute."""
        with pytest.raises(ValueError, match="pod_id"):
            kv_key_scope_for(Principal.TOOL_POD)

    def test_tool_pod_scope_is_per_deployment_not_per_replica(self) -> None:
        """replicas share ``tool_pods.id``, so they share a scope -- L2 stays a shared cache."""
        pod_id = uuid.uuid4()

        assert kv_key_scope_for(Principal.TOOL_POD, pod_id=pod_id) == kv_key_scope_for(
            Principal.TOOL_POD, pod_id=str(pod_id)
        )

    def test_a_non_uuid_id_is_refused(self) -> None:
        """a scope derived from a free-form string is not provably collision-free."""
        with pytest.raises(ValueError, match="uuid"):
            kv_key_scope_for(Principal.TOOL_POD, pod_id="scrape-zone-alpha")

    def test_distinct_pods_get_distinct_scopes(self) -> None:
        """L2S-03's isolation half, at the helper."""
        first = kv_key_scope_for(Principal.TOOL_POD, pod_id=uuid.uuid4())
        second = kv_key_scope_for(Principal.TOOL_POD, pod_id=uuid.uuid4())

        assert first != second

    def test_a_pod_derived_scope_never_collides_with_a_bare_principal(self) -> None:
        """L2S-08: the ban constrains POD-derived scopes only, and it holds by construction."""
        bare = {principal.value for principal in Principal}

        produced = {kv_key_scope_for(Principal.AGENT_POD, agent_id=uuid.uuid4()) for _ in range(32)} | {
            kv_key_scope_for(Principal.TOOL_POD, pod_id=uuid.uuid4()) for _ in range(32)
        }

        assert not (produced & bare)

    def test_a_scope_is_never_derived_from_a_sanitized_name(self) -> None:
        """the sanitizer is ``.``->``-`` and non-injective; two names collapse, two uuids never do.

        Source already records the collapse for mcp names. A scope built that way would hand
        two tool pods one scope, which is exactly what this work exists to prevent.
        """
        from threetears.nats import sanitize_subject_segment  # noqa: PLC0415

        assert sanitize_subject_segment("zone.alpha") == sanitize_subject_segment("zone-alpha")

        first, second = uuid.uuid4(), uuid.uuid4()
        assert kv_key_scope_for(Principal.TOOL_POD, pod_id=first) != kv_key_scope_for(Principal.TOOL_POD, pod_id=second)

    def test_every_produced_scope_satisfies_the_scope_grammar(self) -> None:
        """what the helper mints must be what ``configure`` accepts, for every principal."""
        registry = CollectionRegistry()
        scopes = [kv_key_scope_for(p) for p in Principal if p not in (Principal.AGENT_POD, Principal.TOOL_POD)]
        scopes.append(kv_key_scope_for(Principal.AGENT_POD, agent_id=uuid.uuid4()))
        scopes.append(kv_key_scope_for(Principal.TOOL_POD, pod_id=uuid.uuid4()))

        for scope in scopes:
            registry.configure(kv_key_scope=scope)


# ---------------------------------------------------------------------------
# L2S-09 -- invalidation evicts the receiver's OWN scoped L2 entry
# ---------------------------------------------------------------------------


class TestInvalidationEvictsL2:
    """the defect scoping introduces, and the reason it lands in the same commit."""

    @pytest.mark.asyncio
    async def test_peer_refuses_the_revoked_value_the_hub_retracted(
        self, bus: _SharedNatsBus, config_always: DefaultCoreConfig
    ) -> None:
        """hub revokes, peer refuses.

        Without the L2 eviction the peer drops L1, pulls through, hits ITS OWN stale scoped
        key -- which the hub's write never touched -- and re-caches the revoked value. The
        bucket's ``max_age`` is unlimited and no collection sets an L1 bound, so nothing
        heals it: the revoked grant would be enforced forever, which is worse than the
        exposure per-principal keys exist to close.
        """
        l3: dict[str, dict[str, Any]] = {"g1": {"id": "g1", "name": "granted", "score": 1}}
        hub, hub_registry = _make_pod(bus, _HUB_SCOPE, l3, config_always)
        peer, peer_registry = _make_pod(bus, _AGENT_SCOPE, l3, config_always)
        await hub_registry.start_invalidation_listener(bus)
        await peer_registry.start_invalidation_listener(bus)

        # both principals cache the grant; each writes its OWN key into the shared bucket.
        await hub.ensure("g1")
        await peer.ensure("g1")
        assert f"{_HUB_SCOPE}.test_entities.g1" in bus.bucket.store
        assert f"{_AGENT_SCOPE}.test_entities.g1" in bus.bucket.store

        # the hub revokes: L3 changes, the hub's own key is refreshed, invalidation fires.
        l3["g1"] = {"id": "g1", "name": "revoked", "score": 1}
        await bus.publish(
            subject=Subjects.cache_invalidate(),
            message=CacheInvalidationMessage(table="test_entities", ids=["g1"], origin="hub-origin"),
        )

        # the peer's own scoped key is gone, so its pull-through reaches L3.
        assert f"{_AGENT_SCOPE}.test_entities.g1" not in bus.bucket.store
        entity = await peer.get("g1")
        assert entity is not None
        assert entity.name == "revoked"

    @pytest.mark.asyncio
    async def test_eviction_happens_before_the_l1_guards(
        self, bus: _SharedNatsBus, config_always: DefaultCoreConfig
    ) -> None:
        """a collection whose L1 schema was never initialised must still drop its L2 entry.

        Behind ``l1 is None`` / ``not l1.has_table(...)`` the eviction would skip exactly the
        collections that keep the stale value forever. L2 presence is independent of L1
        presence.
        """
        registry = CollectionRegistry()
        registry.configure(l1_backend=SQLiteBackend(db_name=f"bare_{uuid.uuid4().hex[:8]}"), kv_key_scope=_HUB_SCOPE)
        registry.configure(l2_client=bus)
        collection = StubCollection(registry, config_always, nats_client=bus, l3_rows={})
        collection.l3_pool = _DECLARES_L3  # type: ignore[assignment]
        await registry.start_invalidation_listener(bus)
        key = collection.l2_key("g1")
        bus.bucket.store[key] = collection.serialize({"id": "g1", "name": "stale", "score": 1})

        await bus.publish(
            subject=Subjects.cache_invalidate(),
            message=CacheInvalidationMessage(table="test_entities", ids=["g1"], origin="elsewhere"),
        )

        assert key not in bus.bucket.store

    @pytest.mark.asyncio
    async def test_an_uncached_entity_writes_no_delete_marker(
        self, bus: _SharedNatsBus, config_always: DefaultCoreConfig
    ) -> None:
        """a KV delete publishes a marker UNCONDITIONALLY, so the eviction gates on presence.

        Ungated, every receiver would write one marker per broadcast for every entity it
        never cached, into a memory-storage bucket with ``history=1``, unlimited ``max_age``
        and no ``max_bytes``.
        """
        deleted: list[str] = []

        async def _record_delete(*, key: str, revision: int | None = None) -> bool:  # noqa: ARG001
            deleted.append(key)
            return True

        bus.bucket.delete = _record_delete  # type: ignore[method-assign]
        _, registry = _make_pod(bus, _AGENT_SCOPE, {}, config_always)
        await registry.start_invalidation_listener(bus)

        await bus.publish(
            subject=Subjects.cache_invalidate(),
            message=CacheInvalidationMessage(table="test_entities", ids=["never-cached"], origin="elsewhere"),
        )

        assert deleted == []

    @pytest.mark.asyncio
    async def test_a_collection_with_no_l3_is_never_evicted(
        self, bus: _SharedNatsBus, config_always: DefaultCoreConfig
    ) -> None:
        """where L2 IS the source of truth, evicting is deleting.

        ``HeartbeatCollection``, the presence room index, ``PodAffinityCollection`` and
        ``IdentityGenerationCollection`` all run L1+L2 with no L3. Removing one of their keys
        does not force a refetch -- there is nothing to refetch from -- it destroys the row.
        The identity fence is the sharp case: it fails OPEN on a missing generation, so a lost
        key admits a superseded connection rather than refusing it.

        The staleness the eviction exists to prevent also cannot arise here: with no L3 there
        is no pull-through to re-cache anything, and a peer principal's copy under its own
        scope is its own truth rather than a stale view of somebody else's.

        This is the same reasoning ``l1_max_age_seconds`` already applies to L1 expiry.
        """
        collection, registry = _make_pod(bus, _AGENT_SCOPE, {}, config_always, with_l3=False)
        await registry.start_invalidation_listener(bus)
        key = collection.l2_key("g1")
        bus.bucket.store[key] = collection.serialize({"id": "g1", "name": "the-truth", "score": 1})

        await bus.publish(
            subject=Subjects.cache_invalidate(),
            message=CacheInvalidationMessage(table="test_entities", ids=["g1"], origin="elsewhere"),
        )

        assert key in bus.bucket.store

    @pytest.mark.asyncio
    async def test_the_eviction_does_not_rebroadcast(
        self, bus: _SharedNatsBus, config_always: DefaultCoreConfig
    ) -> None:
        """implementing L2S-09 via ``invalidate_cache`` would re-publish under a new origin.

        The origin filter only skips SELF, so every receiver would rebroadcast and the fan-out
        would be unbounded. The receiver must delete its own key and publish nothing.
        """
        published: list[Any] = []
        original_publish = bus.publish

        async def _counting_publish(*, subject: Any, message: Any, reply_to: Any = None) -> None:
            published.append(message)
            await original_publish(subject=subject, message=message, reply_to=reply_to)

        collection, registry = _make_pod(bus, _AGENT_SCOPE, {}, config_always)
        await registry.start_invalidation_listener(bus)
        bus.bucket.store[collection.l2_key("g1")] = collection.serialize({"id": "g1", "name": "x", "score": 1})
        bus.publish = _counting_publish  # type: ignore[method-assign]

        await bus.publish(
            subject=Subjects.cache_invalidate(),
            message=CacheInvalidationMessage(table="test_entities", ids=["g1"], origin="elsewhere"),
        )

        assert len(published) == 1, "the receiver rebroadcast its own eviction"
