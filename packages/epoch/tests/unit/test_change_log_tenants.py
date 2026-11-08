"""the epoch change log keyed by tenant: a follower reads only its own tenants' keys and ``_all``.

What these hold, on the in-memory log (the live NATS behaviour is pinned in
``tests/integration/test_change_log_live.py``):

- a write's rows go under each row's tenant key, and a whole-table or keys-unknown change under ``_all``;
- a follower of tenant A walks A's records and ``_all``'s, never B's, and B's writes never make it behind;
- every tenant's follower applies an ``_all`` record;
- a record retention took from a follower's own key leaves it behind (``aged_out``), as per table before;
- a platform reader (no tenants) reads every tenant through the wildcard;
- the restore puts each tenant's notes back under its own key.
"""

from __future__ import annotations

import pytest
from threetears.core.collections.change_notes import NotedChange
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.tenancy import ALL_TENANTS, epoch_change_key, schema_tenant, value_tenant
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EPOCH_CHANGES_BUCKET, EpochGenerationReader, EpochGenerationSource

_TABLE = "widgets"
_A = value_tenant("customer-a")
_B = value_tenant("customer-b")


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

        assert await _ids_after(EpochGenerationReader(bus, tenants=[_A]), start) == ["a1", "a2", "*"]
        assert await _ids_after(EpochGenerationReader(bus, tenants=[_B]), start) == ["b1", "*"]
        assert sorted(await _ids_after(EpochGenerationReader(bus), start)) == ["*", "a1", "a2", "b1"], (
            "a platform reader reads every tenant through the wildcard"
        )

    async def test_another_tenants_writes_never_make_it_behind(self) -> None:
        bus = FakeNatsClient()
        source = EpochGenerationSource(bus)
        await source.start_log()
        reader = EpochGenerationReader(bus, tenants=[_A])
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
        followers = {tenant: CollectionRegistry() for tenant in (_A, _B, schema_tenant("agent_x"))}
        readers = {tenant: EpochGenerationReader(bus, tenants=[tenant]) for tenant in followers}
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
        reader = EpochGenerationReader(bus, tenants=[_A])
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
        assert await _ids_after(EpochGenerationReader(bus, tenants=[_A]), start) == ["a1", "*"]
        assert await _ids_after(EpochGenerationReader(bus, tenants=[_B]), start) == ["b1", "*"]


@pytest.mark.parametrize("value", ["customer-a", "3f7c1c1e-0000-4000-8000-000000000001", "a b/c", "x" * 200])
def test_a_tenant_token_is_one_subject_token(value: str) -> None:
    token = value_tenant(value)
    assert "." not in token and " " not in token and len(token) <= 64
    assert epoch_change_key(_TABLE, token).split(".")[0] == token
