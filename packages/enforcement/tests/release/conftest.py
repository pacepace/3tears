"""scratch git repositories for the release-tooling tests.

Every repository here is built under ``tmp_path`` and nothing else: the tool is
only ever pointed at a directory this file created, and :func:`git` refuses any
other. ``uvx`` is replaced by a stub on ``PATH``, so a "relock" is a line the stub
appends to ``uv.lock`` naming the directory it relocked -- which is how a test
tells a released-mode relock (in the scratch worktree) from a local one. With
``UVX_STUB_PROBE`` set, it also records each relocked directory that has a
sibling of that name beside it in ``UVX_STUB_PROBE_OUT``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from packages.enforcement.tests.release.scratch_repos import SCRATCH_ROOTS, commit_all, git

_UVX_STUB = """#!/bin/sh
# the release tool calls: uvx uv@latest lock --directory DIR --quiet
dir="$4"
printf '%s\\n' "$dir" >> "$UVX_STUB_LOG"
case "$dir" in
  *"${UVX_STUB_FAIL_MATCH:-<never>}"*) echo "stub relock failed in $dir" >&2; exit 1 ;;
esac
if [ -n "${UVX_STUB_PROBE:-}" ] && [ -d "$(dirname "$dir")/$UVX_STUB_PROBE" ]; then
  printf '%s\\n' "$dir" >> "$UVX_STUB_PROBE_OUT"
fi
printf '# relocked %s\\n' "$dir" >> "$dir/uv.lock"
"""


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """a parent directory for scratch repositories, with git identity and a ``uvx`` stub.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :param monkeypatch: pytest's environment patcher
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the parent; repositories are created inside it
    :rtype: Path
    """
    parent = (tmp_path / "work").resolve()
    parent.mkdir()
    SCRATCH_ROOTS[:] = [parent]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "uvx"
    stub.write_text(_UVX_STUB)
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("UVX_STUB_LOG", str(tmp_path / "uvx.log"))
    monkeypatch.delenv("UVX_STUB_FAIL_MATCH", raising=False)
    monkeypatch.delenv("UVX_STUB_PROBE", raising=False)
    for key, value in {
        "GIT_AUTHOR_NAME": "release-test",
        "GIT_AUTHOR_EMAIL": "release-test@example.invalid",
        "GIT_COMMITTER_NAME": "release-test",
        "GIT_COMMITTER_EMAIL": "release-test@example.invalid",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
    }.items():
        monkeypatch.setenv(key, value)
    return parent


def _changelog(unreleased: str, sections: str) -> str:
    """a Keep a Changelog file.

    :param unreleased: body under ``## [Unreleased]``
    :ptype unreleased: str
    :param sections: everything after it
    :ptype sections: str
    :return: the file
    :rtype: str
    """
    return f"# Changelog\n\n## [Unreleased]\n\n{unreleased}\n{sections}"


@pytest.fixture
def make_repo(scratch: Path) -> Callable[..., Path]:
    """builds a one-project repository: pyproject, uv.lock, release notes, committed.

    :param scratch: the scratch parent
    :ptype scratch: Path
    :return: factory ``(name="proj", version="0.1.0", unreleased=..., sections="", tags=(), notes=None)``
    :rtype: Callable[..., Path]
    """

    def build(
        name: str = "proj",
        version: str = "0.1.0",
        unreleased: str = "### Added\n\n- a new thing\n",
        sections: str = "",
        tags: tuple[str, ...] = (),
        notes: dict[str, str] | None = None,
    ) -> Path:
        root = scratch / name
        root.mkdir()
        git(root, "init", "-q", "-b", "main")
        (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "{version}"\n')
        (root / "uv.lock").write_text(
            f'version = 1\n\n[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {{ editable = "." }}\n'
        )
        if notes is None:
            (root / "CHANGELOG.md").write_text(_changelog(unreleased, sections))
        else:
            for path, text in notes.items():
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text(text)
        commit_all(root, "initial")
        for tag in tags:
            git(root, "tag", "-a", tag, "-m", tag)
        return root

    return build


@pytest.fixture
def uvx_log(tmp_path: Path) -> Callable[[], list[str]]:
    """the directories the ``uvx`` stub was asked to relock, in order.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: reader
    :rtype: Callable[[], list[str]]
    """

    def read() -> list[str]:
        log = tmp_path / "uvx.log"
        return log.read_text().splitlines() if log.exists() else []

    return read
