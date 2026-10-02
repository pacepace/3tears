"""release versions and the git reads every release tool shares.

A release is a plain ``X.Y.Z`` declared once, as ``[project] version`` in a
``pyproject.toml``, and recorded by an annotated ``vX.Y.Z`` tag on the commit
that shipped it. Everything here reads; nothing writes.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "RELEASE_TAG",
    "GitResult",
    "Version",
    "format_version",
    "parse_version",
    "project_version",
    "release_tags",
    "run_git",
]

#: a release tag, ``vX.Y.Z`` and nothing else.
RELEASE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

type Version = tuple[int, int, int]


@dataclass(frozen=True)
class GitResult:
    """one finished git command.

    :param returncode: process exit status
    :ptype returncode: int
    :param stdout: raw standard output
    :ptype stdout: bytes
    :param stderr: standard error, decoded
    :ptype stderr: str
    """

    returncode: int
    stdout: bytes
    stderr: str

    @property
    def ok(self) -> bool:
        """whether git exited zero.

        :return: ``True`` on exit status 0
        :rtype: bool
        """
        return self.returncode == 0

    @property
    def text(self) -> str:
        """standard output decoded as utf-8, surrounding whitespace removed.

        :return: decoded stdout
        :rtype: str
        """
        return self.stdout.decode("utf-8", errors="replace").strip()


def run_git(root: Path, *args: str, stdin: bytes | None = None) -> GitResult:
    """runs one git command against the repository at *root*.

    Never raises on a git failure: the caller reads :attr:`GitResult.ok`, because
    for most reads here a failure is information (a missing tag, a missing object)
    rather than a crash.

    :param root: repository root
    :ptype root: Path
    :param args: git arguments
    :ptype args: str
    :param stdin: bytes for git's standard input, if any
    :ptype stdin: bytes | None
    :return: exit status and both streams
    :rtype: GitResult
    """
    done = subprocess.run(["git", "-C", str(root), *args], input=stdin, capture_output=True, check=False)
    return GitResult(done.returncode, done.stdout, done.stderr.decode("utf-8", errors="replace").strip())


def parse_version(text: str) -> Version:
    """parses a plain ``X.Y.Z``.

    :param text: version string
    :ptype text: str
    :return: ``(major, minor, patch)``
    :rtype: Version
    :raises ValueError: if *text* is not three dot-separated integers
    """
    parts = text.strip().split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"version {text!r} is not X.Y.Z")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def format_version(version: Version) -> str:
    """renders a version as ``X.Y.Z``.

    :param version: parsed version
    :ptype version: Version
    :return: dotted form
    :rtype: str
    """
    return ".".join(str(part) for part in version)


def project_version(pyproject_text: str) -> Version:
    """reads ``[project] version`` out of pyproject source.

    :param pyproject_text: ``pyproject.toml`` contents
    :ptype pyproject_text: str
    :return: declared version
    :rtype: Version
    :raises ValueError: if there is no ``[project] version`` or it is not ``X.Y.Z``
    """
    project = tomllib.loads(pyproject_text).get("project", {})
    if "version" not in project:
        raise ValueError("no [project] version is declared")
    return parse_version(str(project["version"]))


def release_tags(root: Path) -> dict[str, Version]:
    """every ``vX.Y.Z`` tag visible in the repository at *root*.

    :param root: repository root
    :ptype root: Path
    :return: tag name onto its version
    :rtype: dict[str, Version]
    """
    tags: dict[str, Version] = {}
    for line in run_git(root, "tag", "--list", "v*").text.splitlines():
        matched = RELEASE_TAG.match(line.strip())
        if matched is not None:
            tags[line.strip()] = (int(matched[1]), int(matched[2]), int(matched[3]))
    return tags
