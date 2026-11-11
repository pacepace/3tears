"""Enforcement: ``.testament.toml``'s ``[areas]`` put every test file in exactly one area.

Areas divide the unit suite for reporting and selection (``testament run --not-area evals``,
``testament time --by area``). The table names each package rather than ending in a catch-all
``packages/**``, because a catch-all overlaps every area above it and ``testament areas check
--strict`` reports each overlap. The cost of naming packages is drift: a new package's tests
land in no area, and ``--area platform`` quietly leaves them out. This test is the local,
pool-free form of ``areas check --strict`` over the files in this tree:

- every test file is matched by some area's glob;
- no test file is matched by two areas (the first would take it, hiding the second's glob);
- every glob matches some test file.

Globs match as testament matches them: ``fnmatch`` against the file's repo-relative path, where
``*`` crosses ``/``. ``packages/scrape/sidecar`` is skipped, as the root pytest config skips it.
"""

from __future__ import annotations

import fnmatch
import os
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / ".testament.toml"

#: trees pytest never collects (the root pyproject's ``--ignore``), relative to the repo root
_IGNORED = ("packages/scrape/sidecar",)


def _areas() -> dict[str, list[str]]:
    """Return the manifest's areas, in declared order.

    :return: each area's name and globs
    :rtype: dict[str, list[str]]
    """
    areas = tomllib.loads(_MANIFEST.read_text())["areas"]
    return {str(name): [str(glob) for glob in globs] for name, globs in areas.items()}


def _test_files() -> list[str]:
    """Return every pytest test file under ``packages/`` and ``tests/``, repo-relative.

    :return: posix paths, sorted
    :rtype: list[str]
    """
    found: list[str] = []
    for top in ("packages", "tests"):
        for directory, subdirs, files in os.walk(_REPO_ROOT / top):
            rel_dir = Path(directory).relative_to(_REPO_ROOT).as_posix()
            subdirs[:] = [
                d
                for d in subdirs
                if not d.startswith(".")
                and d not in {"__pycache__", "node_modules"}
                and f"{rel_dir}/{d}" not in _IGNORED
            ]
            found += [
                f"{rel_dir}/{name}"
                for name in files
                if name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))
            ]
    return sorted(found)


def _matching(path: str, areas: dict[str, list[str]]) -> list[str]:
    """Return the areas whose globs match ``path``, in declared order.

    :param path: a repo-relative test file
    :ptype path: str
    :param areas: the manifest's areas
    :ptype areas: dict[str, list[str]]
    :return: area names
    :rtype: list[str]
    """
    return [name for name, globs in areas.items() if any(fnmatch.fnmatchcase(path, glob) for glob in globs)]


def test_the_tree_has_test_files() -> None:
    """a walk that finds nothing would pass the checks below vacuously."""
    assert len(_test_files()) > 100


def test_every_test_file_is_in_an_area() -> None:
    """a new package's tests need its glob, in the area it belongs to (``platform`` by default)."""
    areas = _areas()
    unassigned = [path for path in _test_files() if not _matching(path, areas)]
    assert not unassigned, (
        f"{len(unassigned)} test files are in no area of .testament.toml's [areas], e.g. {unassigned[:5]}: "
        "add a glob for their package"
    )


def test_no_test_file_is_in_two_areas() -> None:
    """an overlap hides the later glob; ``areas check --strict`` reports it."""
    areas = _areas()
    overlaps = {path: names for path in _test_files() if len(names := _matching(path, areas)) > 1}
    assert not overlaps, f"test files matched by two areas: {dict(list(overlaps.items())[:5])}"


def test_every_glob_matches_a_test_file() -> None:
    """a glob for a package that was removed or renamed is dead; drop or fix it."""
    files = _test_files()
    dead = [
        (name, glob)
        for name, globs in _areas().items()
        for glob in globs
        if not any(fnmatch.fnmatchcase(path, glob) for path in files)
    ]
    assert not dead, f"globs in .testament.toml's [areas] that match no test file: {dead}"
