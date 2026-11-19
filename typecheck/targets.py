"""What the typecheck suite checks: every workspace package's import roots, and the evals example hosts.

The roots are read off each package's ``src/threetears/`` tree rather than listed, so a new package is
checked from its first commit with no entry to add: a directory holding an ``__init__.py`` is a root, and
one without (``threetears/agent/``) is a namespace that is descended into. CLAUDE.md, "Namespace
packages", is the rule that makes the walk sound.

``tests/enforcement/test_typecheck_targets.py`` holds the root pyproject's ``[tool.mypy] files``, which a
bare ``mypy`` reads, to the same set of source trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: the scrape sidecar is its own deployable with its own venv (pyproject's ``addopts`` ignores it too)
_NOT_WORKSPACE = ("packages/scrape/sidecar",)


@dataclass(frozen=True)
class Package:
    """one workspace package and the import roots its ``src/threetears/`` tree holds.

    :param path: the package's directory relative to the repo root (``packages/agent/memory``)
    :ptype path: str
    :param roots: its import roots (``threetears.agent.memory``), sorted
    :ptype roots: tuple[str, ...]
    """

    path: str
    roots: tuple[str, ...]

    @property
    def name(self) -> str:
        """the name ``scripts/test.sh`` and ``scripts/typecheck.sh`` take: the path under ``packages/``.

        :return: ``core``, ``agent/memory``
        :rtype: str
        """
        return self.path.removeprefix("packages/")

    @property
    def test_id(self) -> str:
        """the package's parameter id in the typecheck suite: its name with dots for slashes.

        :return: ``core``, ``agent.memory``
        :rtype: str
        """
        return self.name.replace("/", ".")


def _roots(directory: Path, dotted: str) -> list[str]:
    """the import roots under one directory of a ``src/threetears`` tree.

    :param directory: a directory of the tree
    :ptype directory: Path
    :param dotted: its dotted module name
    :ptype dotted: str
    :return: every root at or under it
    :rtype: list[str]
    """
    found: list[str] = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir() or child.name == "__pycache__":
            continue
        if (child / "__init__.py").is_file():
            found.append(f"{dotted}.{child.name}")
        else:
            found.extend(_roots(child, f"{dotted}.{child.name}"))
    return found


def packages(root: Path = REPO_ROOT) -> tuple[Package, ...]:
    """every workspace package with a ``src/threetears`` tree, by path.

    :param root: the repository root
    :ptype root: Path
    :return: the packages and their import roots
    :rtype: tuple[Package, ...]
    """
    found: list[Package] = []
    for src in sorted((root / "packages").glob("**/src/threetears")):
        package = src.parent.parent.relative_to(root).as_posix()
        if ".venv" in src.parts or package.startswith(_NOT_WORKSPACE):
            continue
        roots = tuple(sorted(_roots(src, "threetears")))
        if roots:
            found.append(Package(package, roots))
    return tuple(found)


#: the evals example hosts: reference code an adopter copies, held to strict mypy against the engine's
#: public roots. They live under a ``tests/`` tree, which no package root covers, so they are named by
#: directory, with the repository root on the path because they import their fixture package by its
#: repository path.
EXAMPLE_HOSTS = (
    "packages/evals/tests/fixtures/toyhost",
    "packages/evals/tests/fixtures/courierhost",
)
