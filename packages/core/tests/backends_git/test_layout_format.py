"""a git store records the format its layout writes, migrates an older repository once, and refuses a newer one.

A layout's codecs and paths are a stored format: a repository written under an older one is misread by a newer
layout. ``TemplateLayout(format_version=, migrate_format=)`` names the format and how to reach it; the writer, when
it takes the repository over, runs the migration and records the new version in ONE commit.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from pathlib import Path

from threetears.core.backends.git import (
    FormatVersioned,
    GitDurableStore,
    GitRepo,
    GitRepoConfig,
    GitStoreError,
    JsonCodec,
    NewerLayoutFormatError,
    TableLayout,
    TemplateLayout,
)

from .conftest import MakeRepo

T0 = datetime(2026, 11, 25, 9, 0, 0, tzinfo=UTC)
#: where a repository records its layout format
FORMAT_FILE = ".threetears/layout-format"
_TABLES = [TableLayout("keepers", "{character}/keepers/{tile}.json", ("character", "tile"), JsonCodec())]


async def _verdict_to_decision(repo: Any, from_version: int) -> Mapping[str, bytes | None]:
    """format 0 -> 1: a keeper's ``verdict`` is now its ``decision``."""
    assert from_version == 0
    files: dict[str, bytes | None] = {}
    for path in sorted((repo.config.work_tree).glob("*/keepers/*.json")):
        data = json.loads(path.read_text())
        data["decision"] = data.pop("verdict")
        files[str(path.relative_to(repo.config.work_tree))] = (json.dumps(data, indent=2) + "\n").encode()
    return files


def _commits(repo: Any) -> int:
    return len(repo.git.log(limit=1000))


async def test_an_older_repository_is_migrated_once_in_one_commit(make_repo: MakeRepo) -> None:
    old = await make_repo("data", layout=TemplateLayout(_TABLES))
    await GitDurableStore(old).upsert(
        "keepers", {"character": "aria", "tile": "t1", "verdict": "keep", "date_updated": T0}
    )
    await old.aclose()

    before = _commits(old)
    new_layout = TemplateLayout(_TABLES, format_version=1, migrate_format=_verdict_to_decision)
    new = await make_repo("data", layout=new_layout)

    row = await GitDurableStore(new).fetch_one("keepers", {"character": "aria", "tile": "t1"})
    assert row is not None
    assert (row.get("decision"), "verdict" in row) == ("keep", False)
    assert (new.config.work_tree / FORMAT_FILE).read_text().strip() == "1"
    assert _commits(new) == before + 1
    await new.aclose()

    again = await make_repo("data", layout=new_layout)
    assert _commits(again) == before + 1


async def test_a_newer_repository_is_refused(make_repo: MakeRepo) -> None:
    newer = await make_repo("data", layout=TemplateLayout(_TABLES, format_version=2))
    assert (newer.config.work_tree / FORMAT_FILE).read_text().strip() == "2"
    await newer.aclose()
    with pytest.raises(NewerLayoutFormatError, match="format 2"):
        await make_repo("data", layout=TemplateLayout(_TABLES, format_version=1))


async def test_an_older_repository_with_no_migration_is_refused(make_repo: MakeRepo) -> None:
    old = await make_repo("data", layout=TemplateLayout(_TABLES))
    await GitDurableStore(old).upsert(
        "keepers", {"character": "aria", "tile": "t1", "verdict": "keep", "date_updated": T0}
    )
    await old.aclose()
    with pytest.raises(GitStoreError, match="no migrate_format"):
        await make_repo("data", layout=TemplateLayout(_TABLES, format_version=1))


@pytest.mark.parametrize("migrates", [False, True], ids=["no-migration", "a-migration-touching-files"])
async def test_a_new_repository_is_stamped_at_its_layouts_format_with_no_migration(
    make_repo: MakeRepo, migrates: bool
) -> None:
    """a repository holding no row and recording no format is new: nothing in it was written under format 0."""
    calls: list[int] = []

    async def rewrite_every_keeper(repo: Any, from_version: int) -> Mapping[str, bytes | None]:
        calls.append(from_version)
        return await _verdict_to_decision(repo, from_version)

    layout = TemplateLayout(_TABLES, format_version=3, migrate_format=rewrite_every_keeper if migrates else None)
    repo = await make_repo("data", layout=layout)
    assert (repo.config.work_tree / FORMAT_FILE).read_text().strip() == "3"
    assert calls == []
    await GitDurableStore(repo).upsert(
        "keepers", {"character": "aria", "tile": "t1", "decision": "keep", "date_updated": T0}
    )
    await repo.aclose()
    again = await make_repo("data", layout=layout)
    assert calls == []
    assert await GitDurableStore(again).fetch_one("keepers", {"character": "aria", "tile": "t1"}) is not None


async def test_a_writer_that_meets_a_format_migrated_elsewhere_refuses_it(make_repo: MakeRepo, origin: Path) -> None:
    """an older release's writer and its clone follow origin onto a newer format: each refuses rather than misread."""
    old_layout = TemplateLayout(_TABLES)
    first = await make_repo("first", remote_url=str(origin), layout=old_layout)
    await GitDurableStore(first).upsert(
        "keepers", {"character": "aria", "tile": "t1", "verdict": "keep", "date_updated": T0}
    )
    lagging = await make_repo("lagging", remote_url=str(origin), layout=old_layout, key="characters-lagging")
    await lagging.aclose()

    # a newer release takes the repository over on its own clone, migrates it and pushes
    newer = await make_repo(
        "newer",
        remote_url=str(origin),
        layout=TemplateLayout(_TABLES, format_version=1, migrate_format=_verdict_to_decision),
        key="characters-newer",
    )
    assert (newer.config.work_tree / FORMAT_FILE).read_text().strip() == "1"

    # the older process follows origin onto it: the pull is refused, and so is every read after it
    with pytest.raises(NewerLayoutFormatError, match="format 1"):
        await first.pull()
    with pytest.raises(NewerLayoutFormatError):
        await first.read_row("keepers", {"character": "aria", "tile": "t1"})
    with pytest.raises(NewerLayoutFormatError):
        await GitDurableStore(first).fetch_one("keepers", {"character": "aria", "tile": "t1"})
    # and every change built under its older layout: none is committed here or sent on
    with pytest.raises(NewerLayoutFormatError):
        await GitDurableStore(first).upsert(
            "keepers", {"character": "aria", "tile": "t2", "verdict": "keep", "date_updated": T0}
        )

    # an older writer taking over a clone still at the old format catches up onto the new one and refuses it
    again = GitRepo(
        GitRepoConfig(key="characters-lagging", work_tree=lagging.config.work_tree, remote_url=str(origin)),
        old_layout,
        nats=None,
    )
    try:
        with pytest.raises(NewerLayoutFormatError, match="format 1"):
            await again.open()
    finally:
        await again.aclose()


def test_a_template_layout_names_its_format_and_a_bare_layout_does_not() -> None:
    assert isinstance(TemplateLayout(_TABLES, format_version=1), FormatVersioned)


async def test_a_format_file_holding_no_version_is_refused(make_repo: MakeRepo) -> None:
    repo = await make_repo("data", layout=TemplateLayout(_TABLES))
    await repo.aclose()
    path = repo.config.work_tree / FORMAT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\u00b2\n")
    with pytest.raises(GitStoreError, match="not a format version"):
        await make_repo("data", layout=TemplateLayout(_TABLES, format_version=1))


async def test_a_layout_that_never_migrated_records_nothing(make_repo: MakeRepo) -> None:
    repo = await make_repo("data", layout=TemplateLayout(_TABLES))
    assert not (repo.config.work_tree / FORMAT_FILE).exists()


def test_a_negative_version_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 0"):
        TemplateLayout(_TABLES, format_version=-1)
