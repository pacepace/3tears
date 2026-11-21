"""a tool pod's own L1+L2 collection stack (``coll-task-07c``).

Before this, a tool pod got L1 and nothing else: ``_tool_pod`` granted two KV buckets and neither
was ``{ns}-collections``, and the pod appeared in neither direction of the cross-platform
invalidation subject. So it could hold a ``BaseCollection`` that silently ran with L2 disabled --
four of the five ``l2_key`` call sites catch ``KvError`` and degrade to a WARNING, with
``_ensure_kv`` deliberately INSIDE the catch, so an ungranted pod presents as a dead cache rather
than a crash.

These tests pin the four properties that make the stack safe rather than merely present:

- the key scope is ``tool_pods.id``, the pod's authenticated ``claims.sub``, so replicas share and
  two pods never collide;
- the shared bucket is BOUND, never declared -- the pod's grant carries ``STREAM.INFO`` and no
  ``CREATE``;
- the bucket is opened EAGERLY, before the registry is configured, so a configuration this process
  refuses raises at wiring rather than in a request path under load;
- the invalidation listener is started through ``coll-task-01``'s API and can be stopped again.
"""

from __future__ import annotations

import uuid
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, MetaData, String, Table

from threetears.agent.acl.collections import GroupMemberCollection
from threetears.agent.tools.bootstrap import build_tool_pod_collection_stack
from threetears.agent.tools.l1_cache import TOOL_POD_L1_DB_NAME, create_tool_pod_l1_backend
from threetears.core.collections import BaseCollection
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.namespaces import build_namespace_schema_name
from threetears.core.tenancy import TENANT_BY_SCHEMA, PodTenants, epoch_change_key, schema_tenant
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationSource
from threetears.nats import Principal, kv_key_scope_for
from threetears.nats.errors import KvConfigMismatch, KvError
from threetears.nats.subject_permissions import CROSS_PLATFORM_CACHE_INVALIDATE
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

_POD_A: Final[str] = "01947100-0000-7000-8000-0000000000aa"
_POD_B: Final[str] = "01947100-0000-7000-8000-0000000000bb"

#: a tool pod that owns no data space: it reads the platform tables' ``_all`` records and no schema's
_NO_SPACE: Final[PodTenants] = PodTenants()

#: the provider space a tool pod owns: namespace ``tools.<provider>``, schema ``ns_<namespace id hex>``
_SPACE_SCHEMA: Final[str] = build_namespace_schema_name(uuid.UUID("0192f0c4-7a10-7000-8000-00000000c0de"))

#: an access-machinery table: declared no-tenant (its class is imported so the reader finds that declaration)
_ACCESS_TABLE: Final[str] = "group_members"
assert GroupMemberCollection.tenancy.kind == "none"

#: a provider-space table: declared tenant-by-schema, as every table a pod's tables.yaml declares is
_OWN_TABLE: Final[str] = "tool_pod_tenants_test_rows"


class _OwnRows(BaseCollection[Any]):
    """a table in the pod's own provider space, declared so the reader picks its keys by its tenancy."""

    tenancy = TENANT_BY_SCHEMA

    @property
    def table_name(self) -> str:
        return _OWN_TABLE


def _tables() -> MetaData:
    """one trivial table so the L1 factory has something to mirror.

    :return: metadata carrying a single-column table
    :rtype: MetaData
    """
    metadata = MetaData()
    Table("widgets", metadata, Column("id", String(64), primary_key=True), info={L1_EVICTION_INFO: TEST_L1_TABLE})
    return metadata


def _nats_client() -> MagicMock:
    """a stand-in for the canonical wrapper client the stack builder takes.

    :return: mock client whose bucket bind succeeds and whose subscribe hands back a handle
    :rtype: MagicMock
    """
    client = MagicMock()
    client.raw = MagicMock()
    client.ensure_kv_bucket = AsyncMock(return_value=MagicMock())
    client.subscribe_typed = AsyncMock(return_value=MagicMock())
    client.unsubscribe = AsyncMock()
    return client


class TestTheL1Factory:
    """TP-05: the pod constructs no ``SQLiteBackend`` of its own."""

    def test_it_mirrors_the_declared_tables(self) -> None:
        backend = create_tool_pod_l1_backend(_tables())
        assert backend.has_table("widgets") is True

    def test_the_database_is_named_so_one_process_shares_one_tier(self) -> None:
        """an anonymous in-memory database would give every collection its own, unshared L1."""
        assert TOOL_POD_L1_DB_NAME == "tool_pod_l1_cache"


class TestTheStackIsScopedToTheToolPodId:
    """TP-01 / TP-03: ``tool_pods.id`` is the sharing boundary."""

    async def test_the_scope_is_derived_from_the_pod_id(self) -> None:
        registry = await build_tool_pod_collection_stack(
            nats_client=_nats_client(),
            pod_id=_POD_A,
            l1_metadata=_tables(),
            tenants=_NO_SPACE,
        )
        assert registry.kv_key_scope == kv_key_scope_for(Principal.TOOL_POD, pod_id=_POD_A)

    async def test_two_tool_pods_never_share_a_scope(self) -> None:
        a = await build_tool_pod_collection_stack(
            nats_client=_nats_client(), pod_id=_POD_A, l1_metadata=_tables(), tenants=_NO_SPACE
        )
        b = await build_tool_pod_collection_stack(
            nats_client=_nats_client(), pod_id=_POD_B, l1_metadata=_tables(), tenants=_NO_SPACE
        )
        assert a.kv_key_scope != b.kv_key_scope

    async def test_replicas_of_one_pod_share_one_scope(self) -> None:
        """``tool_pods.id`` is configured once per DEPLOYMENT, so two processes carrying it are two
        replicas of one pod and must land on one scope -- otherwise L2 is not a cross-pod cache."""
        one = await build_tool_pod_collection_stack(
            nats_client=_nats_client(), pod_id=_POD_A, l1_metadata=_tables(), tenants=_NO_SPACE
        )
        two = await build_tool_pod_collection_stack(
            nats_client=_nats_client(), pod_id=_POD_A, l1_metadata=_tables(), tenants=_NO_SPACE
        )
        assert one.kv_key_scope == two.kv_key_scope

    async def test_a_pod_id_that_is_not_a_uuid_fails_closed(self) -> None:
        """the grant side refuses the same input, so accepting it here would key the cache under a
        scope no grant can ever name -- a dead cache that logs nothing."""
        with pytest.raises(ValueError, match="uuid"):
            await build_tool_pod_collection_stack(
                nats_client=_nats_client(), pod_id="pod-alpha", l1_metadata=_tables(), tenants=_NO_SPACE
            )

    async def test_the_composed_key_leads_with_the_scope(self) -> None:
        """the property the whole grant rests on, asserted on a registry the builder produced."""
        registry = await build_tool_pod_collection_stack(
            nats_client=_nats_client(), pod_id=_POD_A, l1_metadata=_tables(), tenants=_NO_SPACE
        )
        scope = kv_key_scope_for(Principal.TOOL_POD, pod_id=_POD_A)
        assert registry.kv_key_scope is not None
        assert registry.kv_key_scope == scope


class TestTheBucketIsBoundEagerlyAndNeverDeclared:
    """TP-06 (``coll-task-04a`` KVC-05) plus the bind-not-declare rule."""

    async def test_it_binds_the_shared_bucket_before_anything_reads_it(self) -> None:
        client = _nats_client()
        await build_tool_pod_collection_stack(
            nats_client=client, pod_id=_POD_A, l1_metadata=_tables(), tenants=_NO_SPACE
        )
        client.ensure_kv_bucket.assert_awaited_once_with(name="collections", create_if_missing=False)

    async def test_the_registry_refuses_to_create_the_bucket(self) -> None:
        """the pod's minted grant carries ``STREAM.INFO`` and no ``CREATE``: a declare can only ever
        cost a JetStream deadline the broker never answers, then bind anyway."""
        registry = await build_tool_pod_collection_stack(
            nats_client=_nats_client(), pod_id=_POD_A, l1_metadata=_tables(), tenants=_NO_SPACE
        )
        assert registry.l2_create_if_missing is False

    async def test_a_config_mismatch_aborts_before_any_collection_is_wired(self) -> None:
        """the raise is the point, and so is WHERE it lands.

        ``BaseCollection._ensure_kv`` resolves the bucket on first read, so a lazily-discovered
        mismatch surfaces under load. Raising here means nothing is configured and no listener is
        bound -- the process dies at wiring, which is the only place an operator can act on it.
        """
        client = _nats_client()
        client.ensure_kv_bucket = AsyncMock(side_effect=KvConfigMismatch("allow_direct differs"))

        with pytest.raises(KvConfigMismatch):
            await build_tool_pod_collection_stack(
                nats_client=client, pod_id=_POD_A, l1_metadata=_tables(), tenants=_NO_SPACE
            )

        client.subscribe_typed.assert_not_awaited()
        assert client.ensure_kv_bucket.await_count == 1

    async def test_an_unbindable_bucket_is_retried_then_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the cold-cluster race: the hub declares this bucket and nothing sequences a pod behind it."""
        import asyncio

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        client = _nats_client()
        client.ensure_kv_bucket = AsyncMock(side_effect=KvError("bucket not found"))

        with pytest.raises(KvError, match="could not be bound"):
            await build_tool_pod_collection_stack(
                nats_client=client, pod_id=_POD_A, l1_metadata=_tables(), tenants=_NO_SPACE
            )

        assert client.ensure_kv_bucket.await_count > 1


class TestTheInvalidationListenerIsStartedAndStoppable:
    """TP-04: through ``coll-task-01``'s API, both halves."""

    async def test_the_listener_is_bound_to_the_global_subject(self) -> None:
        client = _nats_client()
        await build_tool_pod_collection_stack(
            nats_client=client, pod_id=_POD_A, l1_metadata=_tables(), tenants=_NO_SPACE
        )
        subjects = [call.kwargs.get("subject") for call in client.subscribe_typed.await_args_list]
        assert CROSS_PLATFORM_CACHE_INVALIDATE in [str(s) for s in subjects if s is not None]

    async def test_the_listener_can_be_stopped(self) -> None:
        """``coll-task-01`` gave the registry a handle and a stop method; without them a pod that
        shut down left a subscription behind on a client it no longer owns."""
        client = _nats_client()
        registry = await build_tool_pod_collection_stack(
            nats_client=client, pod_id=_POD_A, l1_metadata=_tables(), tenants=_NO_SPACE
        )
        await registry.stop_invalidation_listener()
        client.unsubscribe.assert_awaited_once()


class TestTheBrokerAdvancesThePodsWriteGenerations:
    """epoch-task-06 stage 2: a tool pod cannot write the epoch bucket, so the L3 broker advances."""

    async def test_the_registry_carries_the_broker_generation_source(self) -> None:
        from threetears.core.backends import BrokerGenerationSource

        registry = await build_tool_pod_collection_stack(
            nats_client=_nats_client(), pod_id=_POD_A, l1_metadata=_tables(), tenants=_NO_SPACE
        )
        assert isinstance(registry.generation_source, BrokerGenerationSource)

    async def test_the_source_reads_the_epoch_bucket(self) -> None:
        """stage 3: the pod reads generations (absence caching, following a table) through its grant."""
        from threetears.core.collections.generation import source_reads

        registry = await build_tool_pod_collection_stack(
            nats_client=_nats_client(), pod_id=_POD_A, l1_metadata=_tables(), tenants=_NO_SPACE
        )
        assert source_reads(registry.generation_source)
        assert registry.readable_generation_source is registry.generation_source


class TestTheScopeIsNotAccidentallyUnique:
    """a scope that differed per CALL would pass the isolation half and brick the sharing half."""

    def test_the_helper_is_a_pure_function_of_the_pod_id(self) -> None:
        pod = uuid.UUID(_POD_A)
        assert kv_key_scope_for(Principal.TOOL_POD, pod_id=pod) == kv_key_scope_for(Principal.TOOL_POD, pod_id=str(pod))


class TestThePodReadsTheChangeLogUnderItsOwnTenants:
    """the hub grants a tool pod its OWN change-log keys, and NATS matches the literal subject.

    A ``*.{table}.e`` read is never admitted by a per-tenant grant, so a tool pod's reader must ask for
    concrete keys: ``s-{its schema}.{table}.e`` for a table in its provider space, ``_all.{table}.e`` for
    the access machinery. Each case below runs against a change log that refuses every key the pod's
    grant does not name, so a wildcard read fails exactly as the broker would fail it.
    """

    async def _stack_over(self, bus: FakeNatsClient, tenants: PodTenants) -> object:
        client = _nats_client()
        client.change_log = bus.change_log
        registry = await build_tool_pod_collection_stack(
            nats_client=client, pod_id=_POD_A, l1_metadata=_tables(), tenants=tenants
        )
        return registry.generation_source

    async def _log_with(self, bus: FakeNatsClient) -> tuple[str, str]:
        source = EpochGenerationSource(bus)
        await source.start_log()
        own = await source.advance_with_change(
            _OWN_TABLE, [ChangedRow(ids=["r1"], tenant=schema_tenant(_SPACE_SCHEMA))]
        )
        access = await source.advance_with_change(_ACCESS_TABLE, None)
        return own, access

    async def test_a_pod_with_a_space_reads_its_schema_key_and_the_access_tables_all_key(self) -> None:
        bus = FakeNatsClient()
        own, access = await self._log_with(bus)
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        own_key = epoch_change_key(_OWN_TABLE, schema_tenant(_SPACE_SCHEMA))
        access_key = epoch_change_key(_ACCESS_TABLE, None)
        log.granted_reads = (own_key, access_key)
        source = await self._stack_over(bus, PodTenants(schema=_SPACE_SCHEMA))

        assert await source.current(_OWN_TABLE) == own  # type: ignore[attr-defined]
        assert await source.current(_ACCESS_TABLE) == access  # type: ignore[attr-defined]
        assert own_key == f"s-{_SPACE_SCHEMA}.{_OWN_TABLE}.e"
        assert access_key == "_all.group_members.e"
        assert set(log.read_keys) == {own_key, access_key}
        assert not [key for key in log.read_keys if key.startswith("*")], "a wildcard read no grant admits"

    async def test_a_pod_with_no_space_reads_the_access_tables_all_key(self) -> None:
        bus = FakeNatsClient()
        _, access = await self._log_with(bus)
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        log.granted_reads = ("_all.group_members.e",)
        source = await self._stack_over(bus, _NO_SPACE)

        assert await source.current(_ACCESS_TABLE) == access  # type: ignore[attr-defined]
        assert log.read_keys == ["_all.group_members.e"]

    async def test_a_pod_with_no_space_is_refused_a_schema_table_rather_than_reading_every_tenant(self) -> None:
        """no schema of its own means no key for a schema table; the wildcard is a platform reader's only."""
        bus = FakeNatsClient()
        await self._log_with(bus)
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        source = await self._stack_over(bus, _NO_SPACE)

        with pytest.raises(GenerationUnavailableError, match="reads no schema"):
            await source.current(_OWN_TABLE)  # type: ignore[attr-defined]
        assert log.read_keys == []

    async def test_a_pod_s_reader_is_never_a_platform_reader(self) -> None:
        """``None`` is how a platform service asks for every tenant; a tool pod never may."""
        with pytest.raises(TypeError, match="PodTenants"):
            await build_tool_pod_collection_stack(
                nats_client=_nats_client(),
                pod_id=_POD_A,
                l1_metadata=_tables(),
                tenants=None,  # type: ignore[arg-type]
            )
