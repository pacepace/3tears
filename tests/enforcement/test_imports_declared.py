"""enforcement: every third-party module a package imports comes from a distribution that package declares.

3tears ships this gate (:mod:`threetears.enforcement.imports_declared`) to the repos that consume
it, and runs it here on each of its own packages: a package's imports are checked against its own
``pyproject.toml``, hard dependencies and extras alike, with ``threetears`` first-party (the family's
own cross-package declarations are ``test_dependency_alignment.py``'s).

Without it a package could import sqlalchemy, pydantic or cryptography because ``3tears`` happened
to install them, and break the day core stopped doing so -- which is the change that moving core's
peripheral weight behind extras makes. 3tears-object-store was one: it imported cryptography and
pydantic and declared neither.

An optional import -- inside a function, or under ``try/except ImportError`` -- is held to the same
rule, and is declared as an extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.imports_declared import ImportsDeclaredConfig, run_imports_declared_enforcement

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: contracts packages import no third-party module at all; ``test_dependency_alignment.py`` holds
#: them to that, and this gate's non-vacuity guard would refuse them for having nothing to check.
_CONTRACT_PACKAGES = frozenset({"packages/access-contracts", "packages/media-contracts", "packages/tool-schema"})

_PACKAGES = sorted(
    pyproject.parent.relative_to(_REPO_ROOT).as_posix()
    for pattern in ("packages/*/pyproject.toml", "packages/agent/*/pyproject.toml")
    for pyproject in _REPO_ROOT.glob(pattern)
)


def test_the_packages_were_found() -> None:
    """a glob that stopped matching would pass every package by checking none."""
    assert len(_PACKAGES) > 20, _PACKAGES
    assert _CONTRACT_PACKAGES <= set(_PACKAGES), sorted(_CONTRACT_PACKAGES - set(_PACKAGES))


@pytest.mark.parametrize("package", [package for package in _PACKAGES if package not in _CONTRACT_PACKAGES])
def test_every_import_comes_from_a_declared_distribution(package: str) -> None:
    run_imports_declared_enforcement(
        ImportsDeclaredConfig(repo_root=_REPO_ROOT / package, first_party=frozenset({"threetears"}))
    )
