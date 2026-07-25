"""
enforcement: every intra-family 3tears dependency must be version-bounded.

The 3tears packages version in LOCKSTEP -- one release moves all of them to the
same number. Until this guard existed, they declared each other with no bound at
all::

    dependencies = [
        "3tears",
        "3tears-observe",
    ]

which tells pip that ANY published version of a sibling is acceptable. Two
distinct failures come out of that, and both have cost real time:

**A mixed family installs silently.** pip is free to pair 0.19.0 of one package
with 0.14.0 of another, and it will when a constraint elsewhere makes that the
easier solve. The hub hit exactly this: an unpinned transitive resolved
``3tears-object-store`` to 0.18.0 while everything else was 0.19.0, and 0.18.0
predates ``build_object_key``'s ``path=`` parameter -- so the image built clean
and failed at runtime. Nothing in the build says "you have a mixed family".

**Resolution explodes, and blames the wrong package.** With ~17 published
versions per package and ~25 packages declaring each other unbounded, pip
backtracks across the cross-product. It gives up with ``ResolutionImpossible``
or ``resolution-too-deep``, naming whichever node it happened to be holding --
which is essentially never the package at fault. One such failure reported
``no matching distributions available for your environment: 3tears-agent-tools``
when the real cause was a stale ``protobuf`` pin three levels away in a
consumer's constraints file. That message sends you hunting registries and
extras for hours.

Bounding each sibling to its own minor line makes a mixed family UNRESOLVABLE
rather than merely unlikely, and collapses the search space so a genuine
conflict is reported against the package that actually conflicts.

Static parsing only (no network, no install), consistent with the rest of
``tests/enforcement``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*", "packages/agent/*")

#: a dependency entry naming a 3tears family package, with optional extras and
#: an optional version specifier. `3tears-agent-tools[all]>=0.19.0,<0.20.0`
#: splits into name / extras / specifier.
_FAMILY_DEP = re.compile(
    r"^(?P<name>3tears(?:-[a-z0-9-]+)?)"
    r"(?P<extras>\[[a-z0-9,_-]+\])?"
    r"(?P<spec>.*)$"
)


def _package_dirs() -> list[Path]:
    """return every workspace package directory holding a pyproject.

    :return: package directories, sorted for stable test ids
    :rtype: list[Path]
    """
    found: list[Path] = []
    for glob in _PACKAGE_GLOBS:
        for path in sorted(_REPO_ROOT.glob(glob)):
            if (path / "pyproject.toml").is_file():
                found.append(path)
    return found


def _all_declared_deps(data: dict) -> list[tuple[str, str]]:
    """return ``(section, entry)`` for every declared dependency of a package.

    covers required dependencies AND every optional-dependency extra, because an
    unbounded sibling inside an extra resolves exactly as badly as one in the
    main list -- and extras are where they hide.

    :param data: parsed pyproject mapping
    :ptype data: dict
    :return: (section label, raw dependency string) pairs
    :rtype: list[tuple[str, str]]
    """
    project = data.get("project") or {}
    entries: list[tuple[str, str]] = [("dependencies", dep) for dep in (project.get("dependencies") or [])]
    for extra, deps in (project.get("optional-dependencies") or {}).items():
        entries.extend((f"optional-dependencies.{extra}", dep) for dep in deps or [])
    return entries


_PACKAGES = _package_dirs()


def test_packages_were_discovered() -> None:
    """the globs must actually match -- a silent zero would pass every test below.

    :return: none
    :rtype: None
    :raises AssertionError: if no packages were found
    """
    assert len(_PACKAGES) > 10, (
        f"only {len(_PACKAGES)} workspace packages matched {_PACKAGE_GLOBS}. The "
        "layout changed and this guard is now inspecting almost nothing."
    )


@pytest.mark.parametrize("package_dir", _PACKAGES, ids=lambda p: p.name)
def test_intra_family_deps_are_bounded(package_dir: Path) -> None:
    """every 3tears sibling dependency carries a version specifier.

    :param package_dir: workspace package directory under test
    :ptype package_dir: Path
    :return: none
    :rtype: None
    :raises AssertionError: if a sibling is declared without a bound
    """
    data = tomllib.loads((package_dir / "pyproject.toml").read_text(encoding="utf-8"))
    unbounded: list[str] = []
    for section, dep in _all_declared_deps(data):
        match = _FAMILY_DEP.match(dep.strip())
        if match is None:
            continue
        if not match["spec"].strip():
            unbounded.append(f"{section}: {dep}")
    assert not unbounded, (
        f"{package_dir.name} declares 3tears siblings with no version bound:\n  "
        + "\n  ".join(unbounded)
        + "\n\nThe family versions in lockstep, so declare each sibling against its "
        "own minor line, e.g. `3tears-observe>=0.19.0,<0.20.0`. Unbounded lets pip "
        "install a MIXED family (0.14 core with 0.19 tools) and makes a genuine "
        "conflict surface against an unrelated package."
    )


@pytest.mark.parametrize("package_dir", _PACKAGES, ids=lambda p: p.name)
def test_intra_family_bounds_match_own_version(package_dir: Path) -> None:
    """a sibling's bound must be the declaring package's own minor line.

    a bound that has drifted (``>=0.18.0,<0.19.0`` inside a 0.19.0 package) is
    worse than none: it looks deliberate, and it pins the consumer to a family
    that cannot include this very package.

    :param package_dir: workspace package directory under test
    :ptype package_dir: Path
    :return: none
    :rtype: None
    :raises AssertionError: if a sibling bound does not match this package's line
    """
    data = tomllib.loads((package_dir / "pyproject.toml").read_text(encoding="utf-8"))
    version = (data.get("project") or {}).get("version")
    assert version, f"{package_dir.name} declares no version"
    major, minor, *_ = version.split(".")
    expected = f">={major}.{minor}.0,<{major}.{int(minor) + 1}.0"

    drifted: list[str] = []
    for section, dep in _all_declared_deps(data):
        match = _FAMILY_DEP.match(dep.strip())
        if match is None:
            continue
        spec = match["spec"].strip()
        if spec and spec != expected:
            drifted.append(f"{section}: {dep}  (expected {match['name']}{expected})")
    assert not drifted, (
        f"{package_dir.name} is version {version} but bounds siblings to a different "
        "line:\n  " + "\n  ".join(drifted) + "\n\nThe family releases in lockstep; a "
        "sibling bound must name this package's own minor line."
    )
