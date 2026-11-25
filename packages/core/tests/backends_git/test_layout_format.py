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

from threetears.core.backends.git import GitDurableStore, GitStoreError, JsonCodec, TableLayout, TemplateLayout
from threetears.core.backends.git.repo import FORMAT_FILE

from .conftest import MakeRepo

T0 = datetime(2026, 11, 25, 9, 0, 0, tzinfo=UTC)
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
    newer = await make_repo("data", layout=TemplateLayout(_TABLES, format_version=2, migrate_format=_none))
    assert (newer.config.work_tree / FORMAT_FILE).read_text().strip() == "2"
    await newer.aclose()
    with pytest.raises(GitStoreError, match="format 2"):
        await make_repo("data", layout=TemplateLayout(_TABLES, format_version=1))


async def test_an_older_repository_with_no_migration_is_refused(make_repo: MakeRepo) -> None:
    old = await make_repo("data", layout=TemplateLayout(_TABLES))
    await old.aclose()
    with pytest.raises(GitStoreError, match="no migrate_format"):
        await make_repo("data", layout=TemplateLayout(_TABLES, format_version=1))


async def test_a_layout_that_never_migrated_records_nothing(make_repo: MakeRepo) -> None:
    repo = await make_repo("data", layout=TemplateLayout(_TABLES))
    assert not (repo.config.work_tree / FORMAT_FILE).exists()


def test_a_negative_version_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 0"):
        TemplateLayout(_TABLES, format_version=-1)


async def _none(repo: Any, from_version: int) -> Mapping[str, bytes | None]:
    return {}
