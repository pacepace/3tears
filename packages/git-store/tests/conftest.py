"""fixtures for the git-store suites: a layout, opened repositories, and a bare origin, all on real git."""

from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from threetears.git_store import GitRepo, GitRepoConfig, JsonCodec, TableLayout, TemplateLayout

MakeRepo = Callable[..., Awaitable[GitRepo]]


def run_git(cwd: Path, *args: str) -> str:
    """run git as a person would, outside the store, with an identity of its own.

    :param cwd: where
    :ptype cwd: Path
    :param args: the arguments after ``git``
    :ptype args: str
    :return: stdout
    :rtype: str
    """
    result = subprocess.run(
        ["git", "-c", "user.name=Person", "-c", "user.email=person@example.org", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


GitCli = Callable[..., str]


@pytest.fixture()
def git_cli() -> GitCli:
    """git as a person runs it, beside the store."""
    return run_git


@pytest.fixture()
def layout() -> TemplateLayout:
    """a character-shaped layout: keepers per tile, one recipe, notes without a version, numbered rejects."""
    codec = JsonCodec()
    return TemplateLayout(
        [
            TableLayout("keepers", "{character}/keepers/{batch}/{tile}.json", ("character", "batch", "tile"), codec),
            TableLayout("recipes", "{character}/recipe.json", ("character",), codec),
            TableLayout("notes", "{character}/notes.json", ("character",), codec, version_column=None),
            TableLayout(
                "rejects",
                "{character}/rejects/{batch}/{tile}-{n}.json",
                ("character", "batch", "tile", "n"),
                codec,
                pk_types={"n": int},
            ),
        ]
    )


@pytest.fixture()
def origin(tmp_path: Path) -> Path:
    """an empty bare repository to push to."""
    path = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(path)], check=True)
    return path


@pytest.fixture()
async def make_repo(tmp_path: Path, layout: TemplateLayout) -> AsyncIterator[MakeRepo]:
    """open repositories without NATS (each is its own writer); every one is closed after the test."""
    opened: list[GitRepo] = []

    async def make(name: str = "data", **settings: Any) -> GitRepo:
        chosen = settings.pop("layout", layout)
        settings.setdefault("work_tree", tmp_path / name)
        settings.setdefault("key", "characters")
        repo = GitRepo(GitRepoConfig(**settings), chosen, nats=None)
        await repo.open()
        opened.append(repo)
        return repo

    yield make
    for repo in opened:
        await repo.aclose()
