"""RowFollower: each changed row goes to its table's keyed read, a load's reach to its prefix read; never a whole read."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from threetears.core.collections.registry import CacheInvalidationMessage
from threetears.epoch import RowFollower


def _message(table: str, key: str) -> CacheInvalidationMessage:
    return CacheInvalidationMessage(table=table, ids=[key])


async def test_a_changed_row_is_read_by_its_key() -> None:
    """Flips if a row message reaches no read, the wrong table's, or a read of the table whole."""
    read: list[tuple[str, str]] = []

    async def models(key: str) -> None:
        read.append(("models", key))

    async def providers(key: str) -> None:
        read.append(("providers", key))

    dropped: list[str] = []
    follower = RowFollower({"models": models, "providers": providers}, lambda: dropped.append("dropped"), name="test")
    follower.on_row("models", _message("models", "m1"))
    follower.on_row("providers", _message("providers", "p1"))
    follower.on_row("models", CacheInvalidationMessage(table="models", ids=[]))
    await follower.settled()
    assert sorted(read) == [("models", "m1"), ("providers", "p1")]
    assert dropped == []
    await follower.stop()


async def test_a_load_s_reach_is_read_by_its_prefix() -> None:
    """Flips if a load from files drops the cache instead of reading the prefix it wrote."""
    reached: list[dict[str, str]] = []

    async def row(key: str) -> None:
        raise AssertionError(key)

    async def prefix(reach: Mapping[str, str]) -> None:
        reached.append(dict(reach))

    follower = RowFollower({"results": row}, lambda: None, name="test", partition_reads={"results": prefix})
    follower.on_partition("results", {"state": "TX"})
    await follower.settled()
    assert reached == [{"state": "TX"}]


def test_a_partition_read_of_a_table_it_does_not_follow_is_refused() -> None:
    async def row(key: str) -> None:
        return None

    with pytest.raises(ValueError, match="does not follow"):
        RowFollower({"a": row}, lambda: None, name="test", partition_reads={"b": lambda reach: row("")})


async def test_a_composite_key_table_s_row_is_read_by_its_whole_key() -> None:
    """Flips if a whole-key read gets only the key's lead (its tenant: every row of the tenant), a key out of its
    declared order, or if a lead-keyed table beside it stops getting its single column."""
    whole: list[tuple[str, ...]] = []
    lead: list[str] = []

    async def agents(key: tuple[str, ...]) -> None:
        whole.append(key)

    async def providers(key: str) -> None:
        lead.append(key)

    follower = RowFollower({"providers": providers}, lambda: None, name="test", key_reads={"agents": agents})
    follower.on_row("agents", CacheInvalidationMessage(table="agents", ids=["customer-1", "agent-9"]))
    follower.on_row("providers", _message("providers", "p1"))
    follower.on_row("agents", CacheInvalidationMessage(table="agents", ids=[]))
    await follower.settled()
    assert whole == [("customer-1", "agent-9")]
    assert lead == ["p1"]
    await follower.stop()


async def test_a_follower_of_whole_keys_alone_follows_and_drops_as_any() -> None:
    """Flips if a follower given only whole-key reads is refused, or reads a single-column key as a tuple of one."""
    whole: list[tuple[str, ...]] = []
    dropped: list[str] = []

    async def models(key: tuple[str, ...]) -> None:
        whole.append(key)

    follower = RowFollower({}, lambda: dropped.append("dropped"), name="test", key_reads={"models": models})
    follower.on_row("models", _message("models", "m1"))
    await follower.settled()
    assert whole == [("m1",)]
    assert dropped == []


def test_a_table_given_both_a_lead_read_and_a_whole_key_read_is_refused() -> None:
    async def lead(key: str) -> None:
        return None

    async def whole(key: tuple[str, ...]) -> None:
        return None

    with pytest.raises(ValueError, match="each take one row read"):
        RowFollower({"agents": lead}, lambda: None, name="test", key_reads={"agents": whole})
    with pytest.raises(ValueError, match="at least one table"):
        RowFollower({}, lambda: None, name="test", key_reads={})


async def test_a_partition_read_may_name_a_whole_key_table() -> None:
    """Flips if a load's reach of a composite-key table is refused because the table has no lead read."""
    reached: list[dict[str, str]] = []

    async def whole(key: tuple[str, ...]) -> None:
        raise AssertionError(key)

    async def prefix(reach: Mapping[str, str]) -> None:
        reached.append(dict(reach))

    follower = RowFollower(
        {}, lambda: None, name="test", key_reads={"results": whole}, partition_reads={"results": prefix}
    )
    follower.on_partition("results", {"state": "TX"})
    await follower.settled()
    assert reached == [{"state": "TX"}]
