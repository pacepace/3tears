"""where a repository writes its version, read from ``[tool.threetears-release]``.

A repository with one ``pyproject.toml`` and a ``CHANGELOG.md`` (or a prawduct
change-log) needs no table at all: the defaults describe it. A workspace that
versions many members in lockstep -- 3tears -- names them::

    [tool.threetears-release]
    version-file = "packages/core/pyproject.toml"   # the canonical declaration
    lockstep-files = ["packages/*/pyproject.toml", "packages/agent/*/pyproject.toml"]
    smoke-tests = ["packages/*/tests/test_smoke.py", "packages/agent/*/tests/test_smoke.py"]
    bake-file = "docker-bake.hcl"
    family-prefix = "3tears"                         # intra-family bounds move with the minor
    notes = "CHANGELOG.md"
    notes-style = "3tears"                           # keepachangelog | 3tears | prawduct

Every glob is matched with :meth:`pathlib.Path.glob`, so ``packages/*`` is one
level deep and a nested deployable with its own version is not swept in.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

__all__ = ["NOTES_STYLES", "ConfigError", "ReleaseConfig", "load_release_config"]

#: every notes style a repository may name.
NOTES_STYLES = frozenset({"keepachangelog", "3tears", "prawduct"})

_KEYS = frozenset(
    {
        "version-file",
        "lockstep-files",
        "smoke-tests",
        "bake-file",
        "family-prefix",
        "notes",
        "notes-style",
        "command",
    }
)


class ConfigError(Exception):
    """``[tool.threetears-release]`` cannot be used as written."""


@dataclass(frozen=True)
class ReleaseConfig:
    """everything the release tool writes in one repository.

    :param root: repository root
    :ptype root: Path
    :param version_file: repo-relative ``pyproject.toml`` whose version is canonical
    :ptype version_file: str
    :param lockstep_files: further ``pyproject.toml`` files that carry the same version
    :ptype lockstep_files: tuple[str, ...]
    :param smoke_tests: files asserting ``__version__ == "X.Y.Z"``
    :ptype smoke_tests: tuple[str, ...]
    :param bake_file: a ``docker-bake.hcl`` whose ``VERSION`` default and ``:vX.Y.Z``
        refs follow the version, if any
    :ptype bake_file: str | None
    :param family_prefix: package prefix whose ``>=X.Y.0,<X.Y+1.0`` bounds follow the minor
    :ptype family_prefix: str | None
    :param notes_file: repo-relative release notes
    :ptype notes_file: str
    :param notes_style: ``keepachangelog``, ``3tears`` or ``prawduct``
    :ptype notes_style: str
    :param command: how the repo invokes the tool, for the messages it prints
    :ptype command: str
    """

    root: Path
    version_file: str
    lockstep_files: tuple[str, ...]
    smoke_tests: tuple[str, ...]
    bake_file: str | None
    family_prefix: str | None
    notes_file: str
    notes_style: str
    command: str

    @property
    def version_files(self) -> tuple[str, ...]:
        """the canonical version file, then every other lockstep ``pyproject.toml``.

        :return: repo-relative paths, canonical first
        :rtype: tuple[str, ...]
        """
        return (self.version_file, *[path for path in self.lockstep_files if path != self.version_file])

    @property
    def edited_files(self) -> tuple[str, ...]:
        """every file a bump rewrites, except the lock and the notes.

        :return: repo-relative paths
        :rtype: tuple[str, ...]
        """
        return (*self.version_files, *self.smoke_tests, *((self.bake_file,) if self.bake_file else ()))


def _globbed(root: Path, patterns: list[str]) -> tuple[str, ...]:
    """every file under *root* matching any of *patterns*, sorted.

    :param root: repository root
    :ptype root: Path
    :param patterns: repo-relative globs
    :ptype patterns: list[str]
    :return: repo-relative paths
    :rtype: tuple[str, ...]
    """
    found = {path.relative_to(root).as_posix() for pattern in patterns for path in root.glob(pattern) if path.is_file()}
    return tuple(sorted(found))


def _string_list(table: dict[str, object], key: str) -> list[str]:
    """a list-of-strings value from the table, empty when absent.

    :param table: the parsed ``[tool.threetears-release]`` table
    :ptype table: dict[str, object]
    :param key: key to read
    :ptype key: str
    :return: the strings
    :rtype: list[str]
    :raises ConfigError: if the value is not a list of strings
    """
    value = table.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"[tool.threetears-release] {key} must be a list of strings")
    return list(value)


def _optional_string(table: dict[str, object], key: str) -> str | None:
    """a string value from the table, ``None`` when absent.

    :param table: the parsed ``[tool.threetears-release]`` table
    :ptype table: dict[str, object]
    :param key: key to read
    :ptype key: str
    :return: the string, or ``None``
    :rtype: str | None
    :raises ConfigError: if the value is present and not a string
    """
    value = table.get(key)
    if value is not None and not isinstance(value, str):
        raise ConfigError(f"[tool.threetears-release] {key} must be a string")
    return value


def load_release_config(root: Path) -> ReleaseConfig:
    """reads the release configuration of the repository at *root*.

    :param root: repository root holding ``pyproject.toml``
    :ptype root: Path
    :return: the configuration, defaults filled in
    :rtype: ReleaseConfig
    :raises ConfigError: if the table names an unknown key or a bad value, or the
        notes cannot be found
    """
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    table = data.get("tool", {}).get("threetears-release", {})
    unknown = sorted(set(table) - _KEYS)
    if unknown:
        raise ConfigError(f"[tool.threetears-release] has unknown keys: {', '.join(unknown)}")
    notes_file = _optional_string(table, "notes")
    notes_style = _optional_string(table, "notes-style")
    if notes_file is None:
        if (root / "CHANGELOG.md").is_file():
            notes_file = "CHANGELOG.md"
        elif (root / ".prawduct" / "change-log.md").is_file():
            notes_file = ".prawduct/change-log.md"
        else:
            raise ConfigError(
                "no CHANGELOG.md and no .prawduct/change-log.md -- there is nowhere to record the release"
            )
    if notes_style is None:
        notes_style = "prawduct" if notes_file.endswith("change-log.md") else "keepachangelog"
    if notes_style not in NOTES_STYLES:
        raise ConfigError(f"[tool.threetears-release] notes-style {notes_style!r} is not one of {sorted(NOTES_STYLES)}")
    if not (root / notes_file).is_file():
        raise ConfigError(f"the release notes {notes_file} do not exist")
    version_file = _optional_string(table, "version-file") or "pyproject.toml"
    if not (root / version_file).is_file():
        raise ConfigError(f"the version file {version_file} does not exist")
    return ReleaseConfig(
        root=root,
        version_file=version_file,
        lockstep_files=_globbed(root, _string_list(table, "lockstep-files")),
        smoke_tests=_globbed(root, _string_list(table, "smoke-tests")),
        bake_file=_optional_string(table, "bake-file"),
        family_prefix=_optional_string(table, "family-prefix"),
        notes_file=notes_file,
        notes_style=notes_style,
        command=_optional_string(table, "command") or "./scripts/bump-version.sh",
    )
