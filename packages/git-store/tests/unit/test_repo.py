"""GitRepo without NATS: hand commits noticed, crash recovery, hand edits adopted, origin pushed and never forced."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from threetears.git_store import (
    Author,
    Change,
    FileConflictError,
    GitDurableStore,
    GitRepo,
    GitRepoConfig,
    GitStoreError,
    RowChange,
)

T0 = datetime(2026, 10, 17, 9, 0, tzinfo=UTC)


def keeper(tile: str, **extra: Any) -> dict[str, Any]:
    return {"character": "aria", "batch": "V3", "tile": tile, "date_updated": T0, **extra}


def key(tile: str) -> dict[str, str]:
    return {"character": "aria", "batch": "V3", "tile": tile}


async def test_a_hand_commit_is_noticed_and_its_rows_reported(make_repo: Any, git_cli: Any) -> None:
    repo = await make_repo()
    seen: list[RowChange] = []

    async def record(rows: list[RowChange]) -> None:
        seen.extend(rows)

    repo.on_rows_changed(record)
    store = GitDurableStore(repo)
    await store.upsert_many("keepers", [keeper("1"), keeper("2")], max_rows=9, max_bytes=9)
    assert {row.pk["tile"] for row in seen} == {"1", "2"} and all(row.local for row in seen)
    seen.clear()

    root = repo.config.work_tree
    (root / "aria/keepers/V3/1.json").write_text('{"date_updated": "2026-10-17T10:00:00.000000+00:00"}\n')
    (root / "aria/keepers/V3/2.json").unlink()
    (root / "aria/card.md").write_text("not a row\n")
    git_cli(root, "add", "-A")
    git_cli(root, "commit", "-qm", "by hand")

    rows = await repo.sync()
    assert {(row.table, row.pk["tile"], row.deleted, row.local) for row in rows} == {
        ("keepers", "1", False, False),
        ("keepers", "2", True, False),
    }
    assert rows == seen
    assert await repo.sync() == []  # seen once


async def test_a_hand_commit_made_while_no_store_was_running_is_reported_on_open(
    make_repo: Any, git_cli: Any, tmp_path: Path, layout: Any
) -> None:
    first = await make_repo()
    await GitDurableStore(first).upsert("keepers", keeper("1"))
    await first.aclose()
    root = first.config.work_tree
    (root / "aria/keepers/V3/1.json").unlink()
    git_cli(root, "commit", "-qam", "removed while the store was down")

    again = GitRepo(GitRepoConfig(key="characters", work_tree=root), layout, nats=None)
    seen: list[RowChange] = []

    async def record(rows: list[RowChange]) -> None:
        seen.extend(rows)

    again.on_rows_changed(record)  # bound before open, so takeover's report reaches it
    await again.open()
    try:
        assert [(row.pk["tile"], row.deleted) for row in seen] == [("1", True)]
    finally:
        await again.aclose()


async def test_an_interrupted_change_is_put_back_by_the_next_writer(make_repo: Any, layout: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("keepers", keeper("1", score=1))
    await repo.aclose()
    root = repo.config.work_tree
    # what a writer killed between "files written" and "committed" leaves behind
    journal = repo.git.git_path("git-store/inflight.json")
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({"paths": ["aria/keepers/V3/1.json", "aria/keepers/V3/2.json"]}))
    (root / "aria/keepers/V3/1.json").write_text('{"score": 99}\n')
    (root / "aria/keepers/V3/2.json").write_text('{"score": 2}\n')
    repo.git.run("add", "aria/keepers/V3/2.json")
    (root / "aria/keepers/V3/3.json.tmp.0123").write_text("half a write")

    again = GitRepo(GitRepoConfig(key="characters", work_tree=root), layout, nats=None)
    await again.open()
    try:
        assert '"score": 1' in (root / "aria/keepers/V3/1.json").read_text()
        assert not (root / "aria/keepers/V3/2.json").exists()
        assert not journal.exists()
        assert again.git.dirty() == [("??", "aria/keepers/V3/3.json.tmp.0123")]  # not journaled: left for a person
    finally:
        await again.aclose()


async def test_hand_edits_are_adopted_by_commit_paths(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("keepers", keeper("1", score=1))
    seen: list[RowChange] = []

    async def record(rows: list[RowChange]) -> None:
        seen.extend(rows)

    repo.on_rows_changed(record)
    root = repo.config.work_tree
    target = root / "aria/keepers/V3/1.json"
    target.write_text(target.read_text().replace('"score": 1', '"score": 7'))
    (root / "aria/training-log.md").write_text("tried a new look\n")
    (root / "aria/elsewhere.txt").write_text("not this one\n")

    result = await repo.commit_paths(
        ["aria/keepers/V3/1.json", "aria/training-log.md"],
        "fix a score by hand",
        author=Author("Pace", "pace@example.org"),
    )
    assert result.sha is not None
    assert set(result.paths) == {"aria/keepers/V3/1.json", "aria/training-log.md"}
    assert [(row.pk["tile"], row.local) for row in seen] == [("1", True)]
    assert repo.git.log(limit=1)[0].author == Author("Pace", "pace@example.org")
    assert ("??", "aria/elsewhere.txt") in repo.git.dirty()
    stored = await store.fetch_one("keepers", key("1"))
    assert stored is not None and stored["score"] == 7

    everything = await repo.commit_paths(None, "and the rest")
    assert everything.paths == ("aria/elsewhere.txt",)
    assert (await repo.commit_paths(None, "nothing left")).sha is None


async def test_commit_files_writes_the_bytes_as_one_commit(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("keepers", keeper("1", score=1))
    seen: list[RowChange] = []

    async def record(rows: list[RowChange]) -> None:
        seen.extend(rows)

    repo.on_rows_changed(record)
    root = repo.config.work_tree
    row = (root / "aria/keepers/V3/1.json").read_bytes().replace(b'"score": 1', b'"score": 7')
    base = repo.git.head()
    result = await repo.commit_files(
        {"aria/training-log.md": b"tried a new look\n", "aria/keepers/V3/1.json": row},
        "log and a fixed score",
        author=Author("Pace", "pace@example.org"),
        base=base,
    )
    assert result.sha is not None and result.paths == ("aria/keepers/V3/1.json", "aria/training-log.md")
    assert (root / "aria/training-log.md").read_text() == "tried a new look\n"
    assert [(r.pk["tile"], r.local) for r in seen] == [("1", True)]  # a row among them is reported
    stored = await store.fetch_one("keepers", key("1"))
    assert stored is not None and stored["score"] == 7
    assert repo.git.dirty() == []

    same = await repo.commit_files({"aria/training-log.md": b"tried a new look\n"}, "no change")
    assert same.sha is None
    gone = await repo.commit_files({"aria/training-log.md": None}, "drop the log")
    assert gone.paths == ("aria/training-log.md",) and not (root / "aria/training-log.md").exists()


async def test_commit_files_refuses_a_file_changed_since_its_base(make_repo: Any) -> None:
    repo = await make_repo()
    base = (await repo.commit_files({"aria/notes.md": b"one\n", "aria/other.md": b"a\n"}, "first")).sha
    await repo.commit_files({"aria/notes.md": b"two\n"}, "second", base=base)
    with pytest.raises(FileConflictError) as refused:
        await repo.commit_files({"aria/notes.md": b"three\n", "aria/other.md": b"b\n"}, "stale", base=base)
    assert refused.value.paths == ("aria/notes.md",)
    assert (repo.config.work_tree / "aria/notes.md").read_text() == "two\n"
    assert (repo.config.work_tree / "aria/other.md").read_text() == "a\n"  # refused whole
    ok = await repo.commit_files({"aria/other.md": b"b\n"}, "untouched since base", base=base)
    assert ok.sha is not None
    with pytest.raises(GitStoreError, match="has no commit"):
        await repo.commit_files({"aria/other.md": b"c\n"}, "unknown base", base="0" * 40)


async def test_commit_files_refuses_over_a_hand_edit(make_repo: Any) -> None:
    repo = await make_repo()
    await repo.commit_files({"aria/notes.md": b"one\n"}, "first")
    (repo.config.work_tree / "aria/notes.md").write_text("edited by hand\n")
    with pytest.raises(GitStoreError, match="hand edits"):
        await repo.commit_files({"aria/notes.md": b"two\n"}, "over it")
    assert (repo.config.work_tree / "aria/notes.md").read_text() == "edited by hand\n"


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", ".git/config", ""])
async def test_commit_files_refuses_a_path_outside_the_tree(make_repo: Any, path: str) -> None:
    with pytest.raises(ValueError):
        await (await make_repo()).commit_files({path: b"x"}, "no")


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", ".git/config", ""])
async def test_commit_paths_refuses_a_path_outside_the_tree(make_repo: Any, path: str) -> None:
    with pytest.raises(ValueError):
        await (await make_repo()).commit_paths([path], "no")


async def test_a_commit_can_carry_its_original_date(make_repo: Any) -> None:
    repo = await make_repo()
    when = datetime(2025, 3, 1, 12, 0, tzinfo=UTC)
    await repo.commit(
        [Change("recipes", {"character": "aria"}, {"character": "aria", "date_updated": when})], "recipe v1", date=when
    )
    assert repo.git.log(limit=1)[0].date == when


async def test_each_commit_is_pushed_to_origin(make_repo: Any, origin: Path) -> None:
    repo = await make_repo(remote_url=str(origin))
    result = await repo.commit([Change("keepers", key("1"), keeper("1"))], "first")
    assert result.pushed
    remote = subprocess.run(["git", "rev-parse", "main"], cwd=origin, capture_output=True, text=True, check=True)
    assert remote.stdout.strip() == result.sha
    status = await repo.status()
    assert (status.ahead, status.behind, status.push_blocked) == (0, 0, None)
    assert status.last_push_at is not None


async def test_a_clone_of_origin_is_made_when_the_tree_is_missing(make_repo: Any, origin: Path) -> None:
    first = await make_repo("first", remote_url=str(origin))
    await first.commit([Change("keepers", key("1"), keeper("1"))], "first")
    second = await make_repo("second", remote_url=str(origin), key="characters-2")
    assert (second.config.work_tree / "aria/keepers/V3/1.json").is_file()


async def test_origin_moved_elsewhere_stops_pushing_and_is_never_forced(
    make_repo: Any, origin: Path, git_cli: Any, tmp_path: Path
) -> None:
    repo = await make_repo(remote_url=str(origin))
    await repo.commit([Change("keepers", key("1"), keeper("1"))], "first")
    # someone else pushes to origin behind the writer's back
    other = tmp_path / "other"
    git_cli(tmp_path, "clone", "-q", str(origin), str(other))
    (other / "stray.txt").write_text("x\n")
    git_cli(other, "add", "stray.txt")
    git_cli(other, "commit", "-qm", "elsewhere")
    git_cli(other, "push", "-q", "origin", "main")
    origin_head = git_cli(origin, "rev-parse", "main").strip()

    result = await repo.commit([Change("keepers", key("2"), keeper("2"))], "second")
    assert result.sha is not None and not result.pushed  # committed locally, not pushed
    status = await repo.status()
    assert status.push_blocked is not None
    assert git_cli(origin, "rev-parse", "main").strip() == origin_head  # origin untouched
    third = await repo.commit([Change("keepers", key("3"), keeper("3"))], "third")
    assert not third.pushed  # stays stopped until a person decides
    assert not await repo.push()  # an explicit push is still refused by git, never forced
    assert git_cli(origin, "rev-parse", "main").strip() == origin_head


async def test_status_lists_hand_edits_and_the_sweep_warns_about_old_ones(
    make_repo: Any, caplog: pytest.LogCaptureFixture
) -> None:
    repo = await make_repo(safety_idle=timedelta(0), sweep_interval=timedelta(milliseconds=50))
    await GitDurableStore(repo).upsert("keepers", keeper("1"))
    (repo.config.work_tree / "aria/keepers/V3/1.json").write_text("{}\n")
    status = await repo.status()
    assert [(item.path, item.status) for item in status.dirty] == [("aria/keepers/V3/1.json", " M")]
    assert status.dirty[0].since is not None
    assert status.is_writer and status.writer is None
    with caplog.at_level(logging.WARNING, logger="threetears.git_store.repo"):
        for _ in range(40):
            if any("git_store.dirty" in record.getMessage() for record in caplog.records):
                break
            await asyncio.sleep(0.05)
    assert any("git_store.dirty" in record.getMessage() for record in caplog.records)


async def test_trailers_must_be_single_lines(make_repo: Any) -> None:
    repo = await make_repo()
    with pytest.raises(ValueError):
        await repo.commit([Change("keepers", key("1"), keeper("1"))], "x", trailers={"Op-Seq": "1\n2"})
    with pytest.raises(ValueError):
        await repo.commit([Change("keepers", key("1"), keeper("1"))], "   ")


@pytest.mark.parametrize("made", [True, False], ids=["existing folder", "folder not made yet"])
async def test_a_work_tree_inside_another_repo_is_refused(
    make_repo: Any, git_cli: Any, tmp_path: Path, made: bool
) -> None:
    # found adopting it in samhla: its characters folder sits inside its code checkout, and a store opened there
    # committed into the code repository (--is-inside-work-tree is "true" for any folder of a checkout)
    outer = tmp_path / "outer"
    outer.mkdir()
    git_cli(outer, "init", "-q", "-b", "main")
    (outer / "README").write_text("code\n")
    git_cli(outer, "add", "README")
    git_cli(outer, "commit", "-q", "-m", "code")
    if made:
        (outer / "chars").mkdir()
    with pytest.raises(GitStoreError, match="is inside the git repo"):
        await make_repo(work_tree=outer / "chars")
    log = subprocess.run(["git", "-C", str(outer), "log", "--oneline"], capture_output=True, text=True, check=True)
    assert len(log.stdout.splitlines()) == 1  # nothing landed in the outer repository


async def test_a_repo_of_its_own_inside_another_is_used(make_repo: Any, git_cli: Any, tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    outer.mkdir()
    git_cli(outer, "init", "-q", "-b", "main")
    inner = outer / "data"
    inner.mkdir()
    git_cli(inner, "init", "-q", "-b", "main")  # nested, but its own root: fine
    repo = await make_repo(work_tree=inner)
    assert repo.git.exists()


async def test_a_failed_commit_leaves_no_folder_it_made(make_repo: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = await make_repo()
    root = repo.config.work_tree
    await repo.commit_files({"kept.md": b"x\n"}, "first")
    (root / "empty-before").mkdir()

    def refuse(*_args: Any, **_kwargs: Any) -> str:
        raise GitStoreError("the commit failed")

    monkeypatch.setattr(repo.git, "commit", refuse)
    with pytest.raises(GitStoreError, match="the commit failed"):
        await repo.commit_files({"new/a/x.md": b"x\n", "new/b/y.md": b"y\n"}, "a scaffold")
    assert not (root / "new").exists()
    assert (root / "empty-before").is_dir()  # an empty folder that was there already stays
    assert not repo.git.git_path("git-store/inflight.json").exists()


async def test_after_a_crash_the_next_writer_removes_the_folders_the_change_made(make_repo: Any, layout: Any) -> None:
    repo = await make_repo()
    root = repo.config.work_tree
    await repo.commit_files({"kept.md": b"x\n"}, "first")
    journal = repo.git.git_path("git-store/inflight.json")
    journal.write_text(json.dumps({"paths": ["new/a/x.md"], "dirs": ["new/a", "new"]}))
    (root / "new/a").mkdir(parents=True)
    (root / "new/a/x.md").write_text("half\n")  # the writer died between the write and the commit
    await repo.aclose()

    again = GitRepo(repo.config, layout, nats=None)
    await again.open()
    try:
        assert not (root / "new").exists() and not journal.exists()
    finally:
        await again.aclose()
