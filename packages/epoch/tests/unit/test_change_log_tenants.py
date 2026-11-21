"""the epoch change log keyed by tenant: a pod reads each table under the keys its declared tenancy names.

What these hold, on the in-memory log (the live NATS behaviour is pinned in
``tests/integration/test_change_log_live.py``):

- a write's rows go under each row's tenant key; a platform table's whole-table or keys-unknown change under
  ``_all``, and an agent-schema table's under the schema the writer names;
- a pod of customer A walks a platform table's A records and ``_all``'s, never B's, and B's writes never make
  it behind; every customer's pod applies an ``_all`` record;
- a pod reading its own schema's table asks for its own schema's key and nothing else (a fake that refuses
  every subject its grant does not name proves it), and sees a truncate of its own schema, never another's;
- a tool pod's owner stack reads an agent's table under that agent's schema key;
- a reader is given a pod's PodTenants, never a list of tenants;
- a record retention took from a follower's own key leaves it behind (``aged_out``), as per table before;
- a platform reader (no tenants) reads every tenant through the wildcard;
- the restore puts each tenant's notes, whole-table ones too, back under its own key;
- every table 3tears hands an owner to declare among its own tables (the epochs, the fence, the coordination
  tables) is read by a tool pod of a provider space under ``s-ns_<hex>.<table>.e``, the key the hub's broker
  writes it under, never ``_all`` or the wildcard.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from threetears.agent.acl.collections import GroupMemberCollection
from threetears.core.collections import BaseCollection
from threetears.core.collections.change_notes import NotedChange
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.generation import advance_for_write
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import SchemaBackedCollection, collection_for_schema
from threetears.core.collections.scope_epochs import scope_epochs_collection
from threetears.core.coordination.source_token import source_tokens_schema
from threetears.core.coordination.tables import COORDINATION_TABLE_SCHEMAS
from threetears.core.exceptions import ChangeLogKeyRefusedError, GenerationUnavailableError
from threetears.core.namespaces import build_namespace_schema_name
from threetears.core.tenancy import (
    ALL_TENANTS,
    TENANT_BY_SCHEMA,
    PodTenants,
    epoch_change_key,
    rows_with_tenants,
    schema_tenant,
    tenant_column,
    value_tenant,
)
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationReader, EpochGenerationSource
from threetears.nats.errors import KvError

_TABLE = "tenancy_test_widgets"
_OWN_TABLE = "tenancy_test_notebook"
_A = value_tenant("customer-a")
_B = value_tenant("customer-b")
_POD_A = PodTenants(schema="agent_a", customer="customer-a")
_POD_B = PodTenants(schema="agent_b", customer="customer-b")


class _Widgets(BaseCollection[Any]):
    """a platform table with a customer column: declared here so a pod's reader reads it by its declaration."""

    tenancy = tenant_column("customer_id")

    @property
    def table_name(self) -> str:
        return _TABLE


class _Notebook(BaseCollection[Any]):
    """an agent-schema table: every agent has its own."""

    tenancy = TENANT_BY_SCHEMA

    @property
    def table_name(self) -> str:
        return _OWN_TABLE


class _RefusalRecordingClient(FakeNatsClient):
    """a client that records refused subjects as the real one does: a test sets what it still holds."""

    def __init__(self) -> None:
        super().__init__()
        self.refused: set[str] = set()

    def publish_refused_since(self, fragment: str, since: float) -> str | None:
        return next((subject for subject in self.refused if fragment.lower() in subject.lower()), None)


def _row(key: str, tenant: str | None) -> ChangedRow:
    return ChangedRow(ids=[key], tenant=tenant)


async def _ids_after(reader: EpochGenerationReader, after: str) -> list[str]:
    head = await reader.read(_TABLE)
    assert head is not None
    walk = await reader.changes_after(_TABLE, after, head)
    assert walk.complete, walk
    return [row.ids[0] for change in walk.changes for row in change.rows] + [
        "*" for change in walk.changes if change.whole_table
    ]


class TestARecordGoesUnderItsTenantsKey:
    async def test_rows_split_by_tenant_and_the_whole_table_goes_on_all(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        await source.advance_with_change(_TABLE, [_row("a1", _A), _row("b1", _B), _row("a2", _A)])
        await source.advance_with_change(_TABLE, None)
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        keys = [entry.key for entry in await log.records_after(f"*.{_TABLE}.e", 0, 100)]
        assert keys == [epoch_change_key(_TABLE, _A), epoch_change_key(_TABLE, _B), epoch_change_key(_TABLE, None)]
        assert epoch_change_key(_TABLE, None).startswith(f"{ALL_TENANTS}.")


class TestAFollowerReadsOnlyItsTenants:
    async def test_it_walks_its_own_and_all_never_anothers(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        start = await EpochGenerationReader(bus).read(_TABLE)
        assert start is not None
        await source.advance_with_change(_TABLE, [_row("a1", _A)])
        await source.advance_with_change(_TABLE, [_row("b1", _B)])
        await source.advance_with_change(_TABLE, None)
        await source.advance_with_change(_TABLE, [_row("a2", _A)])

        assert await _ids_after(EpochGenerationReader(bus, tenants=_POD_A), start) == ["a1", "a2", "*"]
        assert await _ids_after(EpochGenerationReader(bus, tenants=_POD_B), start) == ["b1", "*"]
        assert sorted(await _ids_after(EpochGenerationReader(bus), start)) == ["*", "a1", "a2", "b1"], (
            "a platform reader reads every tenant through the wildcard"
        )

    async def test_another_tenants_writes_never_make_it_behind(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        reader = EpochGenerationReader(bus, tenants=_POD_A)
        follower = CollectionRegistry()
        follower.follow_generation(_TABLE)
        await follower.catch_up_generation(_TABLE, await reader.read(_TABLE), changes=reader)
        for n in range(5):
            await source.advance_with_change(_TABLE, [_row(f"b{n}", _B)])
        await follower.catch_up_generation(_TABLE, await reader.read(_TABLE), changes=reader)
        assert not follower.behind_on(_TABLE)

        await source.advance_with_change(_TABLE, [_row("a1", _A)])
        token = await reader.read(_TABLE)
        await follower.catch_up_generation(_TABLE, token, changes=reader)
        assert not follower.behind_on(_TABLE)
        assert follower.generation_marks.recorded(_TABLE) == token

    async def test_every_tenants_follower_applies_an_all_record(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        pods = {"a": _POD_A, "b": _POD_B, "none": PodTenants(schema="agent_x")}
        followers = {name: CollectionRegistry() for name in pods}
        readers = {name: EpochGenerationReader(bus, tenants=pod) for name, pod in pods.items()}
        for tenant, registry in followers.items():
            registry.follow_generation(_TABLE)
            await registry.catch_up_generation(_TABLE, await readers[tenant].read(_TABLE), changes=readers[tenant])
        whole = await source.advance_with_change(_TABLE, None)
        for tenant, registry in followers.items():
            token = await readers[tenant].read(_TABLE)
            assert token == whole, f"{tenant} sees the whole-table record as its newest"
            await registry.catch_up_generation(_TABLE, token, changes=readers[tenant])
            assert registry.generation_marks.recorded(_TABLE) == whole

    async def test_a_record_retention_took_from_its_own_key_leaves_it_behind(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        reader = EpochGenerationReader(bus, tenants=_POD_A)
        mark = await reader.read(_TABLE)
        assert mark is not None
        lost = await source.advance_with_change(_TABLE, [_row("a1", _A)])
        await source.advance_with_change(_TABLE, [_row("a2", _A)])
        bus.change_log(EPOCH_CHANGES_BUCKET).age_out(int(lost.rpartition(":")[2]))
        head = await reader.read(_TABLE)
        assert head is not None
        walk = await reader.changes_after(_TABLE, mark, head)
        assert not walk.complete and walk.reason == "aged_out"


class TestARestorePutsEachTenantBack:
    async def test_notes_of_two_tenants_go_back_under_their_own_keys(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        lost = (await source.log_state()).incarnation  # type: ignore[union-attr]
        bus.change_log(EPOCH_CHANGES_BUCKET).vanish()
        restored = await source.begin_restore(lost)
        assert restored is not None
        await source.finish_restore(
            [
                NotedChange(table=_TABLE, tenant=_A, rows=(_row("a1", _A),)),
                NotedChange(table=_TABLE, tenant=_B, rows=(_row("b1", _B),)),
                NotedChange(table=_TABLE, whole_table=True),
            ],
            incarnation=restored,
            lost=lost,
            covered_from=0,
        )
        start = f"{restored}:0"
        assert await _ids_after(EpochGenerationReader(bus, tenants=_POD_A), start) == ["a1", "*"]
        assert await _ids_after(EpochGenerationReader(bus, tenants=_POD_B), start) == ["b1", "*"]

    async def test_a_schema_tables_whole_table_note_goes_back_under_its_schema(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        lost = (await source.log_state()).incarnation  # type: ignore[union-attr]
        bus.change_log(EPOCH_CHANGES_BUCKET).vanish()
        restored = await source.begin_restore(lost)
        assert restored is not None
        own = schema_tenant("agent_a")
        await source.finish_restore(
            [NotedChange(table=_OWN_TABLE, tenant=own, whole_table=True)],
            incarnation=restored,
            lost=lost,
            covered_from=0,
        )
        start = f"{restored}:0"
        assert await _ids_after_table(EpochGenerationReader(bus, tenants=_POD_A), _OWN_TABLE, start) == ["*"]
        assert await EpochGenerationReader(bus, tenants=_POD_B).read(_OWN_TABLE) == start


@pytest.mark.parametrize("value", ["customer-a", "3f7c1c1e-0000-4000-8000-000000000001", "a b/c", "x" * 200])
def test_a_tenant_token_is_one_subject_token(value: str) -> None:
    token = value_tenant(value)
    assert "." not in token and " " not in token and len(token) <= 64
    assert epoch_change_key(_TABLE, token).split(".")[0] == token


class TestAPodReadsItsOwnSchemaTableUnderItsOwnKeyOnly:
    async def test_it_asks_for_no_subject_but_its_own_schemas(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        await source.advance_with_change(_OWN_TABLE, [_row("n1", schema_tenant("agent_a"))])
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        # the pod's grant, as its host builds it: its own schema whole, a platform table per customer and _all
        log.granted_reads = (
            f"{schema_tenant('agent_a')}.>",
            epoch_change_key(_TABLE, _A),
            epoch_change_key(_TABLE, None),
        )
        reader = EpochGenerationReader(bus, tenants=_POD_A)
        start = f"{(await source.log_state()).incarnation}:0"  # type: ignore[union-attr]
        assert await _ids_after_table(reader, _OWN_TABLE, start) == ["n1"]
        assert set(log.read_keys) == {epoch_change_key(_OWN_TABLE, schema_tenant("agent_a"))}
        log.read_keys.clear()
        await reader.read(_TABLE)
        assert set(log.read_keys) == {epoch_change_key(_TABLE, _A), epoch_change_key(_TABLE, None)}

    async def test_a_truncate_in_one_schema_is_seen_by_its_pod_and_not_by_anothers(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        mine, theirs = EpochGenerationReader(bus, tenants=_POD_A), EpochGenerationReader(bus, tenants=_POD_B)
        before_mine, before_theirs = await mine.read(_OWN_TABLE), await theirs.read(_OWN_TABLE)
        truncate = await source.advance_with_change(_OWN_TABLE, None, tenant=schema_tenant("agent_a"))
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        assert [entry.key for entry in await log.records_after(f"*.{_OWN_TABLE}.e", 0, 100)] == [
            epoch_change_key(_OWN_TABLE, schema_tenant("agent_a"))
        ], "a schema table's truncate goes under its schema, never _all"
        assert await mine.read(_OWN_TABLE) == truncate
        assert before_mine is not None and await _ids_after_table(mine, _OWN_TABLE, before_mine) == ["*"]
        assert await theirs.read(_OWN_TABLE) == before_theirs, "another agent's pod never sees it"

    async def test_a_tool_pods_owner_stack_reads_the_agents_schema_key(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        owner = schema_tenant("agent_owner")
        token = await source.advance_with_change(_OWN_TABLE, [_row("n1", owner)])
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        # a tool pod granted that agent's one table (agent_table_grants), and no customer of its own
        log.granted_reads = (epoch_change_key(_OWN_TABLE, owner),)
        reader = EpochGenerationReader(bus, tenants=PodTenants(schema="agent_owner"))
        assert await reader.read(_OWN_TABLE) == token
        assert set(log.read_keys) == {epoch_change_key(_OWN_TABLE, owner)}

    async def test_a_pod_with_no_schema_cannot_read_a_schema_table(self) -> None:
        bus = FakeNatsClient()
        await EpochGenerationSource(bus).start_log()
        with pytest.raises(GenerationUnavailableError, match="reads no schema"):
            await EpochGenerationReader(bus, tenants=PodTenants(customer="customer-a")).read(_OWN_TABLE)

    async def test_a_pod_of_no_schema_is_refused_as_a_wiring_error(self) -> None:
        bus = FakeNatsClient()
        await EpochGenerationSource(bus).start_log()
        reader = EpochGenerationReader(bus, tenants=PodTenants(customer="customer-a"))
        with pytest.raises(ChangeLogKeyRefusedError) as refused:
            await reader.read(_OWN_TABLE)
        assert refused.value.table == _OWN_TABLE and refused.value.keys == ()
        assert refused.value.tenants == PodTenants(customer="customer-a")

    async def test_a_key_the_broker_refuses_is_refused_once_and_never_asked_again(self) -> None:
        """a pod wired with a schema its grant does not name: the first read is refused, and every later read
        of the table is answered at once, without asking the broker (and waiting on it) again."""
        bus = FakeNatsClient()
        await EpochGenerationSource(bus).start_log()
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        log.granted_reads = (f"{schema_tenant('agent_granted')}.>",)
        reader = EpochGenerationReader(bus, tenants=PodTenants(schema="agent_wrong"))
        wrong_key = epoch_change_key(_OWN_TABLE, schema_tenant("agent_wrong"))

        with pytest.raises(ChangeLogKeyRefusedError) as first:
            await reader.read(_OWN_TABLE)
        assert first.value.keys == (wrong_key,)
        assert first.value.tenants == PodTenants(schema="agent_wrong")
        assert "PodTenants.schema" in str(first.value), "the error names the fix"
        assert log.read_keys == [wrong_key]
        with pytest.raises(ChangeLogKeyRefusedError):
            await reader.read(_OWN_TABLE)
        state = await reader.log_state()
        assert state is not None
        with pytest.raises(ChangeLogKeyRefusedError):
            await reader.changes_after(_OWN_TABLE, f"{state.incarnation}:0", f"{state.incarnation}:1")
        assert log.read_keys == [wrong_key], "a refused table is never asked for again"

    async def test_a_refusal_a_new_credential_cleared_is_asked_again(self) -> None:
        """the client clears its refusals when a new credential takes over (which may name the key): the reader
        asks the broker again then, and reads the table once the grant names it."""
        bus = _RefusalRecordingClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        token = await source.advance_with_change(_OWN_TABLE, [_row("n1", schema_tenant("agent_a"))])
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        log.granted_reads = ()
        reader = EpochGenerationReader(bus, tenants=_POD_A)
        own_key = epoch_change_key(_OWN_TABLE, schema_tenant("agent_a"))

        with pytest.raises(ChangeLogKeyRefusedError) as refused:
            await reader.read(_OWN_TABLE)
        bus.refused = {f"$KV.{EPOCH_CHANGES_BUCKET}.{own_key}"}
        with pytest.raises(ChangeLogKeyRefusedError):
            await reader.read(_OWN_TABLE)
        assert log.read_keys == [own_key], "held while the client still records the refusal"
        assert refused.value.keys == (own_key,)

        bus.refused = set()  # a renewal's successor: the client cleared what the old credential heard
        log.granted_reads = None
        assert await reader.read(_OWN_TABLE) == token

    async def test_an_outage_stays_a_passing_failure(self) -> None:
        """a log that cannot be reached is not a refusal: it is not remembered, and the next read asks again."""
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        token = await source.advance_with_change(_OWN_TABLE, [_row("n1", schema_tenant("agent_a"))])
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        reader = EpochGenerationReader(bus, tenants=_POD_A)
        log.unreachable = KvError("broker down")

        with pytest.raises(GenerationUnavailableError) as down:
            await reader.read(_OWN_TABLE)
        assert not isinstance(down.value, ChangeLogKeyRefusedError)
        log.unreachable = None
        assert await reader.read(_OWN_TABLE) == token

    def test_a_list_of_tenants_is_refused(self) -> None:
        with pytest.raises(TypeError, match="PodTenants"):
            EpochGenerationReader(FakeNatsClient(), tenants=[_A])  # type: ignore[arg-type]


async def _ids_after_table(reader: EpochGenerationReader, table: str, after: str) -> list[str]:
    head = await reader.read(table)
    assert head is not None
    walk = await reader.changes_after(table, after, head)
    assert walk.complete, walk
    return [row.ids[0] for change in walk.changes for row in change.rows] + [
        "*" for change in walk.changes if change.whole_table
    ]


class TestTheAccessMachineryIsFollowedAcrossCustomers:
    """every pod's access evaluator reads groups and roles cross-customer by id, so every pod sees every change."""

    async def test_a_pod_applies_a_group_row_changed_under_another_customer(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        table = "group_members"
        assert GroupMemberCollection.tenancy.kind == "none"
        reader = EpochGenerationReader(bus, tenants=_POD_A)
        follower = CollectionRegistry()
        follower.follow_generation(table)
        await follower.catch_up_generation(table, await reader.read(table), changes=reader)
        mark = await reader.read(table)
        assert mark is not None
        # customer B's membership changes: the writer keys it by the table's declaration
        changed = rows_with_tenants(
            GroupMemberCollection.tenancy, [ChangedRow(ids=["gm-b"], columns={"customer_id": "customer-b"})]
        )
        token = await source.advance_with_change(table, changed)
        assert await reader.read(table) == token, "customer A's pod sees customer B's change"
        assert await _ids_after_table(reader, table, mark) == ["gm-b"]
        await follower.catch_up_generation(table, token, changes=reader)
        assert not follower.behind_on(table)
        assert follower.generation_marks.recorded(table) == token


class TestAWriterThatKnowsTheSchemaNamesEachRecordsTenant:
    """``advance_for_write``: the broker names the schema its statement ran in; 3tears applies the one rule."""

    async def test_each_table_by_its_declaration(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        own = schema_tenant("agent_a")
        await advance_for_write(source, _OWN_TABLE, [ChangedRow(ids=["n1"])], schema="agent_a")
        await advance_for_write(source, _OWN_TABLE, None, schema="agent_a", keys_unknown=True)
        await advance_for_write(
            source, _TABLE, [ChangedRow(ids=["w1"], columns={"customer_id": "customer-a"})], schema="platform"
        )
        await advance_for_write(source, _TABLE, None, schema="platform")
        await advance_for_write(source, "group_members", [ChangedRow(ids=["g1"])], schema="platform")
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        keys = [entry.key for entry in await log.records_after(">", 0, 100)]
        assert keys == [
            epoch_change_key(_OWN_TABLE, own),
            epoch_change_key(_OWN_TABLE, own),
            epoch_change_key(_TABLE, _A),
            epoch_change_key(_TABLE, None),
            epoch_change_key("group_members", None),
        ]


#: a tool pod's own provider space, in the hub's spelling (``ns_<namespace id hex>``)
_SPACE = build_namespace_schema_name(uuid.UUID("0192f0c4-7a10-7000-8000-00000000c0de"))

#: the collection classes over the tables 3tears hands an owner to declare among its own tables, built as the owner
#: builds them and held here: the reader finds a table's declaration through the classes alive in the process (the
#: coordination tables' classes are defined by their module)
_OWNER_TABLES: tuple[type[SchemaBackedCollection[Any]], ...] = (
    scope_epochs_collection(),
    collection_for_schema(source_tokens_schema()),
)
_OWNER_TABLE_NAMES: tuple[str, ...] = (
    *(cls.schema.name for cls in _OWNER_TABLES),
    *(schema.name for schema in COORDINATION_TABLE_SCHEMAS),
)


class TestAnOwnersOwnTablesAreReadUnderItsOwnSchemaKey:
    """a tool pod of a provider space reads the tables it declares among its own under ``s-ns_<hex>``.

    The hub grants such a pod ``s-ns_<hex>.>`` (its own space) and the access tables' ``_all`` keys, and its broker
    writes a switched table's changes under the schema the statement ran in. A table with no declared tenancy would
    be read under ``_all``: refused by that grant, and missing the records even if granted.
    """

    @pytest.mark.parametrize("table", _OWNER_TABLE_NAMES)
    async def test_the_pod_reads_its_schema_key_and_nothing_else(self, table: str) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        # the hub's broker, which knows the schema its statement ran in
        token = await advance_for_write(source, table, [ChangedRow(ids=["r1"])], schema=_SPACE)
        own_key = f"s-{_SPACE}.{table}.e"
        log = bus.change_log(EPOCH_CHANGES_BUCKET)
        assert [entry.key for entry in await log.records_after(">", 0, 100)] == [own_key]
        log.read_keys.clear()
        # the pod's grant, as the hub builds it for a tool pod of a provider space
        log.granted_reads = (f"{schema_tenant(_SPACE)}.>", epoch_change_key("group_members", None))
        reader = EpochGenerationReader(bus, tenants=PodTenants(schema=_SPACE))
        assert await reader.read(table) == token
        assert set(log.read_keys) == {own_key}
        assert not [key for key in log.read_keys if key.startswith(("*", f"{ALL_TENANTS}."))]
