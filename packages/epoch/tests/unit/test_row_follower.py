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
