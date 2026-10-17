"""GitDurableStore over a real repository: reads, writes, compare-and-swap, conflict modes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.core.backends.protocol import BulkDeletingStore, BulkDurableStore, DurableStore
from threetears.git_store import ChangeConflictError, GitDurableStore

T0 = datetime(2026, 10, 17, 9, 0, tzinfo=UTC)


def keeper(tile: str, *, batch: str = "V3", when: datetime = T0, **extra: Any) -> dict[str, Any]:
    return {"character": "aria", "batch": batch, "tile": tile, "date_updated": when, **extra}


def key(tile: str, batch: str = "V3") -> dict[str, str]:
    return {"character": "aria", "batch": batch, "tile": tile}


def commits(repo: Any) -> int:
    return len(repo.git.log(limit=1000))


async def test_it_is_every_store_the_collections_reach_for(make_repo: Any) -> None:
    store = GitDurableStore(await make_repo())
    assert isinstance(store, DurableStore)
    assert isinstance(store, BulkDurableStore)
    assert isinstance(store, BulkDeletingStore)


async def test_a_write_is_a_file_and_a_commit_and_reads_back(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    assert await store.upsert("keepers", keeper("12", score=9), pk=("character", "batch", "tile")) == 1
    assert (repo.config.work_tree / "aria/keepers/V3/12.json").is_file()
    assert commits(repo) == 1
    assert await store.fetch_one("keepers", key("12")) == keeper("12", score=9)
    assert await store.fetch_one("keepers", key("13")) is None


async def test_scan_filters_and_narrows_by_path(make_repo: Any) -> None:
    store = GitDurableStore(await make_repo())
    await store.upsert_many(
        "keepers",
        [keeper("1", score=1), keeper("2", score=2), keeper("1", batch="V4", score=1)],
        max_rows=100,
        max_bytes=1_000_000,
    )
    assert len(await store.scan("keepers")) == 3
    assert [row["tile"] for row in await store.scan("keepers", {"batch": "V3"})] == ["1", "2"]
    assert [row["batch"] for row in await store.scan("keepers", {"score": 1})] == ["V3", "V4"]


async def test_many_rows_are_one_commit(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert_many("keepers", [keeper(str(n)) for n in range(5)], max_rows=1, max_bytes=1)
    assert commits(repo) == 1
    await store.delete_many("keepers", [("aria", "V3", "1"), ("aria", "V3", "2")], max_rows=1)
    assert commits(repo) == 2
    assert len(await store.scan("keepers")) == 3


async def test_a_compare_and_swap_hit_writes(make_repo: Any) -> None:
    store = GitDurableStore(await make_repo())
    await store.upsert("keepers", keeper("12"), pk=None)
    later = T0 + timedelta(minutes=1)
    assert await store.upsert("keepers", keeper("12", when=later, score=3), cas=T0) == 1
    stored = await store.fetch_one("keepers", key("12"))
    assert stored is not None and stored["score"] == 3


async def test_a_compare_and_swap_miss_writes_nothing(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("keepers", keeper("12", score=1))
    stale = T0 - timedelta(minutes=1)
    assert await store.upsert("keepers", keeper("12", when=T0 + timedelta(minutes=1), score=2), cas=stale) == 0
    stored = await store.fetch_one("keepers", key("12"))
    assert stored is not None and stored["score"] == 1
    assert commits(repo) == 1


async def test_a_row_deleted_since_it_was_read_is_a_miss_not_a_reinsert(make_repo: Any) -> None:
    store = GitDurableStore(await make_repo())
    await store.upsert("keepers", keeper("12"))
    await store.delete("keepers", key("12"))
    assert await store.upsert("keepers", keeper("12", when=T0 + timedelta(minutes=1)), cas=T0) == 0
    assert await store.fetch_one("keepers", key("12")) is None


async def test_a_compare_and_swap_on_a_table_without_a_version_is_refused(make_repo: Any) -> None:
    store = GitDurableStore(await make_repo())
    with pytest.raises(ValueError, match="no version column"):
        await store.upsert("notes", {"character": "aria", "text": "x"}, cas=T0)


async def test_on_conflict_ignore_and_raise(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("keepers", keeper("12", score=1))
    assert await store.upsert("keepers", keeper("12", score=2), on_conflict="ignore") == 0
    with pytest.raises(ChangeConflictError) as raised:
        await store.upsert("keepers", keeper("12", score=3), on_conflict="raise")
    assert raised.value.rows[0].reason == "exists"
    stored = await store.fetch_one("keepers", key("12"))
    assert stored is not None and stored["score"] == 1
    assert await store.upsert("keepers", keeper("13"), on_conflict="raise") == 1


async def test_writing_the_same_bytes_again_makes_no_commit(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("keepers", keeper("12"))
    await store.upsert("keepers", keeper("12"))
    assert commits(repo) == 1


async def test_deleting_a_missing_row_is_no_error_and_no_commit(make_repo: Any) -> None:
    repo = await make_repo()
    await GitDurableStore(repo).delete("keepers", key("nope"))
    assert commits(repo) == 0


async def test_history_and_read_at_follow_one_row(make_repo: Any) -> None:
    store = GitDurableStore(await make_repo())
    await store.upsert("recipes", {"character": "aria", "look": "one", "date_updated": T0})
    later = T0 + timedelta(hours=1)
    await store.upsert("recipes", {"character": "aria", "look": "two", "date_updated": later}, cas=T0)
    await store.upsert("keepers", keeper("12"))
    history = await store.history("recipes", {"character": "aria"})
    assert [entry.subject for entry in history] == ["recipes aria: save", "recipes aria: save"]
    first = await store.read_at(history[1].sha, "recipes", {"character": "aria"})
    assert first is not None and first["look"] == "one"


async def test_a_conn_that_is_not_this_repositorys_change_set_is_refused(make_repo: Any) -> None:
    store = GitDurableStore(await make_repo())
    other = await make_repo("other", key="other")
    async with other.change("elsewhere") as change_set:
        with pytest.raises(ValueError, match="ChangeSet of repository"):
            await store.upsert("keepers", keeper("1"), conn=change_set)
    with pytest.raises(ValueError):
        await store.delete("keepers", key("1"), conn=object())
