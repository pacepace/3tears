"""RowFollower: each changed row goes to its table's keyed read, a load's reach to its prefix read; never a whole read."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from threetears.core.collections.registry import CacheInvalidationMessage
from threetears.epoch import RowFollower, lead_slice_reads


def _message(table: str, key: str) -> CacheInvalidationMessage:
    return CacheInvalidationMessage(table=table, ids=[key])


async def _no_slice(reach: Mapping[str, str]) -> None:
    raise AssertionError(f"no slice is read here: {dict(reach)}")


async def test_a_changed_row_is_read_by_its_key() -> None:
    """Flips if a row message reaches no read, the wrong table's, or a read of the table whole."""
    read: list[tuple[str, str]] = []

    async def models(key: str) -> None:
        read.append(("models", key))

    async def providers(key: str) -> None:
        read.append(("providers", key))

    dropped: list[str] = []
    follower = RowFollower(
        {"models": models, "providers": providers},
        lambda: dropped.append("dropped"),
        name="test",
        partition_reads={"models": _no_slice, "providers": _no_slice},
    )
    follower.on_row("models", _message("models", "m1"))
    follower.on_row("providers", _message("providers", "p1"))
    follower.on_row("models", CacheInvalidationMessage(table="models", ids=[]))
    await follower.settled()
    assert sorted(read) == [("models", "m1"), ("providers", "p1")]
    assert dropped == []
    await follower.stop()


@pytest.mark.timeout(10)
async def test_settled_returns_when_a_read_finished_but_its_done_callback_has_not_run() -> None:
    """Flips (hangs: the loop spins without a turn, so only the timeout ends it) if settled() re-gathers a finished
    read still listed: gather over finished tasks completes without yielding, so the done-callback that unlists the
    read never runs."""
    read: list[str] = []

    async def models(key: str) -> None:
        read.append(key)

    follower = RowFollower({"models": models}, lambda: None, name="test", partition_reads={"models": _no_slice})
    follower.on_row("models", _message("models", "m1"))
    # one turn: the read runs to its end, and its done-callback is queued behind this task's resumption
    await asyncio.sleep(0)
    assert read == ["m1"]
    await follower.settled()
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

    follower = RowFollower(
        {"providers": providers},
        lambda: None,
        name="test",
        key_reads={"agents": agents},
        partition_reads={"agents": _no_slice, "providers": _no_slice},
    )
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

    follower = RowFollower(
        {},
        lambda: dropped.append("dropped"),
        name="test",
        key_reads={"models": models},
        partition_reads={"models": _no_slice},
    )
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
        RowFollower({"agents": lead}, lambda: None, name="test", key_reads={"agents": whole}, partition_reads={})
    with pytest.raises(ValueError, match="at least one table"):
        RowFollower({}, lambda: None, name="test", key_reads={}, partition_reads={})


def test_a_table_with_no_partition_read_is_refused() -> None:
    """Flips if a follower is built with a table whose slices it could not read (a load or a clear would then drop the
    cache whole, which decision 33 removed)."""

    async def row(key: str) -> None:
        return None

    with pytest.raises(ValueError, match=r"\['b'\] have no partition read"):
        RowFollower({"a": row, "b": row}, lambda: None, name="test", partition_reads={"a": _no_slice})


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


async def test_a_slice_is_read_by_its_key_s_lead_through_the_row_read() -> None:
    """Flips if a lead slice read reads anything but the row read of the prefix's lead value."""
    read: list[str] = []

    async def by_lead(value: str) -> None:
        read.append(value)

    reads = {"audience": by_lead}
    follower = RowFollower(
        reads, lambda: None, name="test", partition_reads=lead_slice_reads(reads, {"audience": "dataset_id"})
    )
    follower.on_partition("audience", {"dataset_id": "d1"})
    follower.on_partition("audience", {"dataset_id": "d2", "customer_id": "c1"})
    await follower.settled()
    assert sorted(read) == ["d1", "d2"]


async def test_a_lead_slice_of_a_prefix_not_leading_the_key_is_refused() -> None:
    """Flips if a prefix without the key's lead is read as something else (decision 34)."""

    async def by_lead(value: str) -> None:
        raise AssertionError(f"read {value}")

    slices = lead_slice_reads({"audience": by_lead}, {"audience": "dataset_id"})
    with pytest.raises(ValueError, match="leading the key"):
        await slices["audience"]({"customer_id": "c1"})


def test_a_lead_slice_read_names_every_table_s_lead() -> None:
    """Flips if a table read with no lead column named gets a partition read anyway."""

    async def by_lead(value: str) -> None:
        return None

    with pytest.raises(ValueError, match="no lead key column"):
        lead_slice_reads({"audience": by_lead}, {})
