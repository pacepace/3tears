"""saved, not yet committed: OpLogTail on an in-memory log with a toy op, against real git, without NATS."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from threetears.git_store import (
    CommitResult,
    DivergedError,
    GitRepo,
    GitStoreError,
    OpLogTail,
    committed_seq_from,
)
from threetears.git_store.git import GitError
from threetears.git_store.wire import ForwardedRequest, decode_reply, decode_request, encode_failure, encode_request
from threetears.nats.oplog import AppendResult, OpRecord


class SequenceConflict(Exception):
    """the sender's expected sequence is not the log's last."""


class MemoryLog:
    """an op-log in memory: append with an expected-sequence check and id dedup, replay from a sequence."""

    def __init__(self) -> None:
        self.records: list[OpRecord] = []

    async def last_seq(self) -> int:
        return self.records[-1].seq if self.records else 0

    async def append(self, *, payload: bytes, op_id: str, expected_last_seq: int) -> AppendResult:
        for record in self.records:
            if record.op_id == op_id:
                return AppendResult(record.seq, True)
        if expected_last_seq != await self.last_seq():
            raise SequenceConflict(f"expected {expected_last_seq}, log is at {await self.last_seq()}")
        seq = await self.last_seq() + 1
        self.records.append(OpRecord(seq, payload, op_id))
        return AppendResult(seq, False)

    async def replay(self, from_seq: int) -> AsyncIterator[OpRecord]:
        for record in list(self.records):
            if record.seq >= from_seq:
                yield record


# the toy op: {"path": ..., "add": "a line"} appends a line; {"path": ..., "set": "text"} replaces the file


def op(path: str, **change: str) -> bytes:
    return json.dumps({"path": path, **change}).encode()


async def apply(repo: GitRepo, ops: Sequence[dict[str, str]]) -> list[str]:
    """rebuild each touched file from HEAD, then apply its ops in order."""
    texts: dict[str, str] = {}
    for item in ops:
        path = item["path"]
        if path not in texts:
            base = repo.git.show("HEAD", path) if repo.git.head() else None
            texts[path] = (base or b"").decode()
        texts[path] = item["set"] if "set" in item else texts[path] + item["add"] + "\n"
    for path, text in texts.items():
        target = repo.config.work_tree / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return list(texts)


def tail(log: MemoryLog, scope: str | None = "stories/one") -> OpLogTail[dict[str, str]]:
    return OpLogTail(log, scope=scope, decode=json.loads, apply=apply)


def last_message(repo: GitRepo) -> str:
    entry = repo.git.last_commit()
    assert entry is not None
    return entry[2]


async def seeded(make_repo: Any, git_cli: Any, **settings: Any) -> GitRepo:
    """a repository with two stories committed by hand."""
    repo = await make_repo(**settings)
    root = repo.config.work_tree
    for story in ("one", "two"):
        (root / f"stories/{story}").mkdir(parents=True)
        (root / f"stories/{story}/scene.md").write_text("start\n")
    git_cli(root, "add", "-A")
    git_cli(root, "commit", "-qm", "two stories")
    await repo.sync()
    return repo


async def test_a_save_is_written_but_not_committed_until_the_intent(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    log = MemoryLog()
    one = tail(log)
    await repo.add_tail(one)
    head = repo.git.head()

    await one.save(repo, payload=op("stories/one/scene.md", add="first"), op_id="a")
    await one.save(repo, payload=op("stories/one/scene.md", add="second"), op_id="b")

    scene = repo.config.work_tree / "stories/one/scene.md"
    assert scene.read_text() == "start\nfirst\nsecond\n"
    assert repo.git.head() == head  # saved, not committed
    assert repo.pending_paths() == {"stories/one/scene.md"}
    assert (await repo.status()).dirty == ()  # the store's own work is not a hand edit

    result = await repo.commit_intent("stories/one", "edit")
    assert result.sha is not None and result.paths == ("stories/one/scene.md",)
    assert committed_seq_from(last_message(repo)) == 2
    assert repo.pending_paths() == set()
    assert (await repo.commit_intent("stories/one", "edit again")).sha is None  # nothing new


async def test_a_retried_save_is_not_applied_twice(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    one = tail(MemoryLog())
    await repo.add_tail(one)
    first = await one.save(repo, payload=op("stories/one/scene.md", add="x"), op_id="same")
    again = await one.save(repo, payload=op("stories/one/scene.md", add="x"), op_id="same")
    assert again.deduplicated and again.seq == first.seq
    assert (repo.config.work_tree / "stories/one/scene.md").read_text() == "start\nx\n"


async def test_edits_that_net_to_nothing_make_no_commit(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    one = tail(MemoryLog())
    await repo.add_tail(one)
    head = repo.git.head()
    await one.save(repo, payload=op("stories/one/scene.md", set="changed\n"), op_id="a")
    await one.save(repo, payload=op("stories/one/scene.md", set="start\n"), op_id="b")
    assert (await repo.commit_intent("stories/one", "edit")).sha is None
    assert repo.git.head() == head


async def test_an_op_the_check_refuses_is_never_logged(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    log = MemoryLog()
    one = tail(log)
    await repo.add_tail(one)

    def refuse(_pending: Sequence[dict[str, str]], new: dict[str, str]) -> None:
        raise ValueError(f"{new['path']} does not exist")

    with pytest.raises(ValueError, match="does not exist"):
        await one.save(repo, payload=op("stories/one/nope.md", add="x"), op_id="a", check=refuse)
    assert log.records == []


async def test_a_new_writer_rebuilds_saved_edits_over_a_dirty_folder(
    make_repo: Any, git_cli: Any, layout: Any, tmp_path: Path
) -> None:
    repo = await seeded(make_repo, git_cli)
    log = MemoryLog()
    one = tail(log)
    await repo.add_tail(one)
    await one.save(repo, payload=op("stories/one/scene.md", add="saved"), op_id="a")
    await repo.aclose()

    root = repo.config.work_tree
    (root / "stories/one/scene.md").write_text("half-written garbage\n")  # the old writer died mid-write
    (root / "stories/two/scene.md").write_text("a hand edit elsewhere\n")

    again = GitRepo(repo.config, layout, nats=None)
    await again.add_tail(tail(log))  # before open: open is the takeover
    await again.open()
    try:
        assert (root / "stories/one/scene.md").read_text() == "start\nsaved\n"
        assert (root / "stories/two/scene.md").read_text() == "a hand edit elsewhere\n"  # outside the folder: kept
        assert [item.path for item in (await again.status()).dirty] == ["stories/two/scene.md"]
    finally:
        await again.aclose()


async def test_adding_a_tail_on_the_writer_replays_it(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    log = MemoryLog()
    log.records.append(OpRecord(1, op("stories/one/scene.md", add="from the log"), "a"))
    await repo.add_tail(tail(log))
    assert (repo.config.work_tree / "stories/one/scene.md").read_text() == "start\nfrom the log\n"
    assert repo.pending_paths("stories/one") == {"stories/one/scene.md"}


async def test_a_trailer_ahead_of_the_log_counts_nothing_committed_until_head_moves(
    make_repo: Any, git_cli: Any
) -> None:
    repo = await seeded(make_repo, git_cli)
    old = MemoryLog()
    one = tail(old)
    await repo.add_tail(one)
    for n in range(3):
        await one.save(repo, payload=op("stories/one/scene.md", add=f"old {n}"), op_id=f"old{n}")
    await repo.commit_intent("stories/one", "edit")
    assert committed_seq_from(last_message(repo)) == 3

    fresh = MemoryLog()  # the log was reset: git's trailer (3) is ahead of it
    repo.remove_tail("stories/one")
    again = tail(fresh)
    await repo.add_tail(again)
    for n in range(3):  # the log grows to the stale trailer's own number with no commit
        await again.save(repo, payload=op("stories/one/scene.md", add=f"new {n}"), op_id=f"new{n}")
    assert await again.committed_through(repo) == 0  # sticky: still nothing of this log is committed
    assert (repo.config.work_tree / "stories/one/scene.md").read_text().endswith("new 0\nnew 1\nnew 2\n")

    await repo.commit_intent("stories/one", "edit")  # HEAD moves: the marker heals
    assert await again.committed_through(repo) == 3
    assert repo.pending_paths() == set()


async def test_each_folder_keeps_its_own_trailer_and_commits(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    one, two = tail(MemoryLog(), "stories/one"), tail(MemoryLog(), "stories/two")
    await repo.add_tail(one)
    await repo.add_tail(two)
    for n in range(3):
        await one.save(repo, payload=op("stories/one/scene.md", add=f"{n}"), op_id=f"{n}")
    await two.save(repo, payload=op("stories/two/scene.md", add="two"), op_id="t")

    result = await repo.commit_intent("stories/one", "edit one")
    assert result.paths == ("stories/one/scene.md",)
    assert repo.pending_paths() == {"stories/two/scene.md"}  # the sibling's saved edit is left out
    await repo.commit_intent("stories/two", "edit two")
    assert await one.committed_through(repo) == 3  # the sibling's later commit (Op-Seq 1) doesn't count here
    assert await two.committed_through(repo) == 1


async def test_tails_may_not_overlap(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    await repo.add_tail(tail(MemoryLog(), "stories/one"))
    with pytest.raises(ValueError, match="overlaps"):
        await repo.add_tail(tail(MemoryLog(), "stories"))
    with pytest.raises(ValueError, match="overlaps"):
        await repo.add_tail(tail(MemoryLog(), None))


async def test_a_structural_commit_in_the_folder_takes_the_saved_edits_along(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    one = tail(MemoryLog())
    await repo.add_tail(one)
    await one.save(repo, payload=op("stories/one/scene.md", add="saved"), op_id="a")

    result = await repo.commit_files({"stories/one/new.md": b"a new entry\n"}, "create an entry")
    assert set(result.paths) == {"stories/one/new.md", "stories/one/scene.md"}
    assert committed_seq_from(last_message(repo)) == 1  # the trailer says the saved edit is in git
    assert repo.pending_paths() == set()
    assert repo.git.dirty() == []

    await repo.commit_files({"stories/two/new.md": b"elsewhere\n"}, "outside the folder")
    assert "Op-Seq" not in last_message(repo)


async def test_a_write_over_a_saved_file_is_refused(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    one = tail(MemoryLog())
    await repo.add_tail(one)
    await one.save(repo, payload=op("stories/one/scene.md", add="saved"), op_id="a")
    with pytest.raises(GitStoreError, match="saved edits not yet committed"):
        await repo.commit_files({"stories/one/scene.md": b"whole new text\n"}, "overwrite")
    assert (repo.config.work_tree / "stories/one/scene.md").read_text() == "start\nsaved\n"


async def test_a_hand_commit_in_the_folder_carries_the_trailer(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    one = tail(MemoryLog())
    await repo.add_tail(one)
    await one.save(repo, payload=op("stories/one/scene.md", add="saved"), op_id="a")
    (repo.config.work_tree / "stories/one/notes.md").write_text("by hand\n")

    result = await repo.commit_paths(None, "adopt a hand edit")
    assert set(result.paths) == {"stories/one/notes.md", "stories/one/scene.md"}
    assert committed_seq_from(last_message(repo)) == 1


async def test_a_safety_commit_waits_for_the_idle_window(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    one = tail(MemoryLog())
    await repo.add_tail(one)
    await one.save(repo, payload=op("stories/one/scene.md", add="saved"), op_id="a")
    head = repo.git.head()
    assert (await repo.commit_intent("stories/one", "safety", idle=timedelta(hours=6))).sha is None
    assert repo.git.head() == head  # the folder's last commit is seconds old
    assert (await repo.commit_intent("stories/one", "safety", idle=timedelta(0))).sha is not None


async def test_the_sweep_commits_saved_edits_left_idle(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli, safety_idle=timedelta(0), sweep_interval=timedelta(milliseconds=50))
    one = tail(MemoryLog())
    await repo.add_tail(one)
    await one.save(repo, payload=op("stories/one/scene.md", add="saved"), op_id="a")
    for _ in range(100):
        if not repo.pending_paths():
            break
        await asyncio.sleep(0.05)
    assert repo.pending_paths() == set()
    assert committed_seq_from(last_message(repo)) == 1


async def test_commit_listeners_hear_every_commit(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    one = tail(MemoryLog())
    await repo.add_tail(one)
    heard: list[CommitResult] = []

    async def listen(result: CommitResult) -> None:
        heard.append(result)

    stop = repo.on_committed(listen)
    await one.save(repo, payload=op("stories/one/scene.md", add="saved"), op_id="a")
    assert heard == []  # a save is not a commit
    await repo.commit_intent("stories/one", "edit")
    await repo.commit_files({"stories/two/new.md": b"x\n"}, "create")
    stop()
    await repo.commit_files({"stories/two/other.md": b"y\n"}, "create")
    assert [result.paths for result in heard] == [("stories/one/scene.md",), ("stories/two/new.md",)]


async def test_a_save_needs_the_folders_tail(make_repo: Any, git_cli: Any) -> None:
    repo = await seeded(make_repo, git_cli)
    with pytest.raises(GitStoreError, match="add_tail first"):
        async with repo.pending("stories/one"):
            pass
    with pytest.raises(GitStoreError, match="no pending tail"):
        await repo.commit_intent("stories/one", "edit")


def test_commit_intent_crosses_the_wire() -> None:
    request = ForwardedRequest(op="commit_intent", scope="stories/one", idle=21600.0, message="edit\n")
    back = decode_request(encode_request(request))
    assert (back.op, back.scope, back.idle, back.message) == ("commit_intent", "stories/one", 21600.0, "edit\n")


def test_the_last_trailer_counts() -> None:
    assert committed_seq_from("edit\n\nOp-Seq: 3\nOp-Seq: 7\n") == 7
    assert committed_seq_from("edit\n") == 0


async def test_without_nats_a_products_operation_runs_here(make_repo: Any) -> None:
    repo = await make_repo()

    async def run(payload: bytes) -> bytes:
        return payload.upper()

    assert await repo.forward("apply_op", b"x", local=run) == b"X"


def test_git_and_value_failures_keep_their_type_across_the_wire() -> None:
    with pytest.raises(GitError) as failed:
        decode_reply(encode_failure(GitError(["commit", "-q"], 128, "fatal: index.lock exists")))
    assert failed.value.returncode == 128 and failed.value.git_args == ("commit", "-q") and failed.value.index_locked
    with pytest.raises(ValueError, match="not a path inside"):
        decode_reply(encode_failure(ValueError("'../x' is not a path inside the working tree")))


def outside_push(origin: Path, tmp_path: Path, git_cli: Any, files: dict[str, str], name: str = "outside") -> str:
    """someone pushes to origin from a clone of their own (a desktop editor, a merged PR)."""
    clone = tmp_path / name
    if not clone.exists():
        git_cli(tmp_path, "clone", "-q", str(origin), str(clone))
    git_cli(clone, "pull", "-q", "--ff-only")
    for path, text in files.items():
        (clone / path).parent.mkdir(parents=True, exist_ok=True)
        (clone / path).write_text(text)
    git_cli(clone, "add", "-A")
    git_cli(clone, "commit", "-qm", "from outside")
    git_cli(clone, "push", "-q", "origin", "HEAD:main")
    return git_cli(clone, "rev-parse", "HEAD").strip()


async def test_a_pull_takes_an_outside_push_and_reports_its_rows(
    make_repo: Any, git_cli: Any, origin: Path, tmp_path: Path
) -> None:
    repo = await make_repo(remote_url=str(origin))
    await repo.commit_files({"stories/one/scene.md": b"start\n"}, "a story")
    one = tail(MemoryLog())
    await repo.add_tail(one)
    await one.save(repo, payload=op("stories/one/scene.md", add="saved"), op_id="a")
    seen: list[Any] = []

    async def record(rows: list[Any]) -> None:
        seen.extend(rows)

    repo.on_rows_changed(record)
    pushed = outside_push(
        origin,
        tmp_path,
        git_cli,
        {"aria/keepers/V3/1.json": '{"date_updated": null}\n', "stories/one/scene.md": "start\nby hand\n"},
    )

    result = await repo.pull()
    assert result.sha == pushed == repo.git.head()
    assert set(result.paths) == {"aria/keepers/V3/1.json", "stories/one/scene.md"}
    assert any(row.table == "keepers" and row.pk["tile"] == "1" and not row.local for row in seen)
    # the saved edit is rebuilt on the pulled text
    assert (repo.config.work_tree / "stories/one/scene.md").read_text() == "start\nby hand\nsaved\n"
    assert (await repo.pull()).paths == ()  # nothing more to take


async def test_a_pull_refuses_a_branch_that_diverged_and_changes_nothing(
    make_repo: Any, git_cli: Any, origin: Path, tmp_path: Path
) -> None:
    repo = await make_repo(remote_url=str(origin), push="never")
    await repo.commit_files({"a.md": b"a\n"}, "first")
    await repo.push()
    await repo.commit_files({"b.md": b"b\n"}, "local, not pushed")
    outside_push(origin, tmp_path, git_cli, {"c.md": "c\n"})
    head = repo.git.head()
    with pytest.raises(DivergedError):
        await repo.pull()
    assert repo.git.head() == head and not (repo.config.work_tree / "c.md").exists()


async def test_a_pull_without_an_origin_is_refused(make_repo: Any) -> None:
    repo = await make_repo()
    with pytest.raises(GitStoreError, match="no origin"):
        await repo.pull()


def test_a_diverged_refusal_crosses_the_wire() -> None:
    with pytest.raises(DivergedError) as refused:
        decode_reply(encode_failure(DivergedError("main", "a" * 40, "b" * 40)))
    assert refused.value.head == "a" * 40 and "both moved on" in str(refused.value)
