"""fixtures for the git backend's unit suite: a layout, opened repositories, and a bare origin, all on real git."""

from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from packages.core.tests.backends_git.support import bare_origin, character_layout
from threetears.core.backends.git import GitRepo, GitRepoConfig, TemplateLayout

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
    return character_layout()


@pytest.fixture()
def origin(tmp_path: Path) -> Path:
    """an empty bare repository to push to."""
    return bare_origin(tmp_path)


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
