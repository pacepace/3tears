"""helpers for driving git in the release-tooling tests' scratch repositories.

:func:`git` refuses any directory that is not under a root registered by the
``scratch`` fixture, so no test can point git -- or the release tool -- at a real
checkout.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

from threetears.enforcement.release import load_release_config, run_release

__all__ = ["SCRATCH_ROOTS", "TODAY", "commit_all", "git", "release", "snapshot"]

#: the date every test run writes into the notes.
TODAY = "2026-09-30"

#: the root every scratch repository must live under, set per test.
SCRATCH_ROOTS: list[Path] = []


def _require_scratch(root: Path) -> None:
    """refuses a directory that is not under a registered scratch root.

    :param root: directory a test is about to act on
    :ptype root: Path
    :return: nothing
    :rtype: None
    """
    assert any(root.resolve().is_relative_to(scratch) for scratch in SCRATCH_ROOTS), f"{root} is not a scratch repo"


def release(root: Path, action: str, version: str | None = None) -> tuple[int, str, str]:
    """runs the release tool against a scratch repository.

    :param root: scratch repository
    :ptype root: Path
    :param action: release action
    :ptype action: str
    :param version: the version ``verify`` checks
    :ptype version: str | None
    :return: exit status, stdout, stderr
    :rtype: tuple[int, str, str]
    """
    _require_scratch(root)
    out, err = io.StringIO(), io.StringIO()
    status = run_release(load_release_config(root), action, TODAY, out, err, version)
    return status, out.getvalue(), err.getvalue()


def git(root: Path, *args: str) -> str:
    """runs git in a scratch repository, refusing any directory a test did not create.

    :param root: scratch repository
    :ptype root: Path
    :param args: git arguments
    :ptype args: str
    :return: stdout, stripped
    :rtype: str
    """
    _require_scratch(root)
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def commit_all(root: Path, message: str = "commit") -> None:
    """commits everything in *root*.

    :param root: scratch repository
    :ptype root: Path
    :param message: commit message
    :ptype message: str
    :return: nothing
    :rtype: None
    """
    git(root, "add", "--all")
    git(root, "commit", "-qm", message)


def snapshot(root: Path) -> tuple[dict[str, bytes], str, str]:
    """every tracked-or-not file's bytes, the index, and the status, for a byte-identity check.

    :param root: scratch repository
    :ptype root: Path
    :return: files, ``git ls-files -s``, ``git status --porcelain``
    :rtype: tuple[dict[str, bytes], str, str]
    """
    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }
    return files, git(root, "ls-files", "-s"), git(root, "status", "--porcelain")
