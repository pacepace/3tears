"""a hand edit the store adopts gets a new version, so a copy loaded before the edit cannot save over it.

A row's version is its ``date_updated``; a save carries the version it read and lands only while the file still
holds it. A person editing a row's file by hand rarely touches ``date_updated``, so before the writer adopted the
edit unchanged and a copy loaded before it passed the compare-and-swap and wrote over the person's work. The writer
now stamps every adopted row edit with a version newer than any it held -- edits in its own working tree
(``commit_paths``) and edits sent from another tree as bytes (``commit_files``) alike.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from threetears.core.backends.git import GitDurableStore

T0 = datetime(2026, 11, 25, 9, 0, 0, tzinfo=UTC)


def _keeper(tile: str, **extra: Any) -> dict[str, Any]:
    return {"character": "aria", "batch": "V3", "tile": tile, "verdict": "keep", "date_updated": T0, **extra}


def _key(tile: str) -> dict[str, str]:
    return {"character": "aria", "batch": "V3", "tile": tile}


def _edit(path: Any, **changes: Any) -> None:
    """change a row's file as a person does in an editor: only the fields they mean to change."""
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data, indent=2) + "\n")


async def test_an_edit_committed_in_the_writers_tree_gets_a_newer_version(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("keepers", _keeper("t1"))
    loaded = await store.fetch_one("keepers", _key("t1"))
    assert loaded is not None
    _edit(repo.config.work_tree / "aria/keepers/V3/t1.json", verdict="reject")

    await repo.commit_paths(None, "a person rejects t1")

    adopted = await store.fetch_one("keepers", _key("t1"))
    assert adopted is not None
    assert adopted["verdict"] == "reject"
    assert adopted["date_updated"] > T0
    # the copy loaded before the edit is refused, and the edit stands
    stale = {**loaded, "verdict": "keep again", "date_updated": T0 + timedelta(seconds=1)}
    assert await store.upsert("keepers", stale, cas=loaded["date_updated"]) == 0
    stored = await store.fetch_one("keepers", _key("t1"))
    assert stored is not None
    assert stored["verdict"] == "reject"


async def test_an_edit_sent_as_bytes_from_another_tree_gets_a_newer_version(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("keepers", _keeper("t2"))
    loaded = await store.fetch_one("keepers", _key("t2"))
    assert loaded is not None
    edited = json.loads((repo.config.work_tree / "aria/keepers/V3/t2.json").read_text())
    edited["verdict"] = "reject"

    await repo.commit_files({"aria/keepers/V3/t2.json": (json.dumps(edited, indent=2) + "\n").encode()}, "edited")

    adopted = await store.fetch_one("keepers", _key("t2"))
    assert adopted is not None
    assert adopted["verdict"] == "reject"
    assert adopted["date_updated"] > loaded["date_updated"]
    assert await store.upsert("keepers", {**loaded, "verdict": "x"}, cas=loaded["date_updated"]) == 0


async def test_an_edit_that_moved_the_version_back_still_gets_a_newer_one(make_repo: Any) -> None:
    """the new version is newer than the one the writer held, whatever the edit wrote in its place."""
    repo = await make_repo()
    store = GitDurableStore(repo)
    later = T0 + timedelta(days=30)
    await store.upsert("keepers", _keeper("t3", date_updated=later))
    _edit(repo.config.work_tree / "aria/keepers/V3/t3.json", date_updated=T0.isoformat(), verdict="reject")

    await repo.commit_paths(None, "an edit pasted from an old copy")

    adopted = await store.fetch_one("keepers", _key("t3"))
    assert adopted is not None
    assert adopted["date_updated"] > later


async def test_a_table_with_no_version_is_adopted_as_edited(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("notes", {"character": "aria", "text": "first"})
    path = repo.config.work_tree / "aria/notes.json"
    _edit(path, text="by hand")
    before = path.read_bytes()

    await repo.commit_paths(None, "a note by hand")

    assert path.read_bytes() == before
    assert await store.fetch_one("notes", {"character": "aria"}) == {"character": "aria", "text": "by hand"}


async def test_a_file_that_is_not_a_row_and_a_removed_row_are_committed_as_they_are(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("keepers", _keeper("t4"))
    (repo.config.work_tree / "aria/keepers/V3/t4.json").unlink()
    (repo.config.work_tree / "README.txt").write_text("not a row\n")

    result = await repo.commit_paths(None, "a removal and a readme")

    assert result.sha is not None
    assert await store.fetch_one("keepers", _key("t4")) is None
    assert (repo.config.work_tree / "README.txt").read_text() == "not a row\n"
