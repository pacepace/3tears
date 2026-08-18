"""
enforcement: a package whose public API grew may not ship as a patch release.

**The rule this holds is a recorded product decision, not an inference.** Backlog
item BLD-7QM3, decided 2026-07-26: *an intra-family API addition requires a minor
bump.* Two other directions were considered there and rejected -- raising each
bound's floor to the release that carries the new API (more precise, but the
bound cannot be written until the release version is known, which puts a moving
part inside the very mechanism whose failure mode is an unresolvable family), and
guarding the imports to degrade (wrong wherever the seam is not optional).

**Why the rule exists.** Every intra-family dependency is bounded to a MINOR line
-- ``3tears>=0.24.0,<0.25.0`` -- which is what makes a mixed family unresolvable
rather than merely unlikely (``test_intra_family_version_bounds.py``, and
CLAUDE.md's own warning). That bound's floor is the minor. So when API is added
inside a minor, every earlier patch in the range satisfies the bound and lacks
the symbol: pip resolves a family that installs clean, builds clean, and raises
``ImportError`` on first use. Bounding a package that grew mid-minor is a
guardrail with a hole exactly the width of the change.

**It has happened, which is why this is a test and not a note.** v0.24.7 shipped
as a patch carrying a new ``threetears.langgraph.tool_structure`` module and
eleven new names in ``threetears.langgraph``'s ``__all__``. Every gate was
green: the bounds were correctly shaped, the suite passed, the release workflow
succeeded. ``test_intra_family_version_bounds.py`` checks bound SHAPE and
structurally cannot see this class of defect -- BLD-7QM3 said so in advance, and
said the guard that would catch it compares the public surface at the last
release tag against HEAD. This is that guard.

**What it can and cannot see.** It compares names a consumer can import or call
by name: ``__all__`` entries, and the public methods of classes those entries
name. It does NOT see a method added to a private class that a public method
returns -- v0.24.7's other addition, ``fetchval`` on ``core``'s
``_ProxyConnection``, is exactly that shape and passes this check. Such an object
is public in every way that matters to a caller and private to every check that
reads names, and closing it needs return-type analysis rather than a name sweep.
The limit is recorded here rather than papered over: this guard makes the common
case impossible, not every case.

Static parsing against git blobs -- no import, no install, no network --
consistent with the rest of ``tests/enforcement``.
"""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*", "packages/agent/*")
_RELEASE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _git(*args: str) -> str:
    """run a read-only git command in the repo, returning stdout.

    :param args: git arguments
    :ptype args: str
    :return: stdout, stripped
    :rtype: str
    """
    done = subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else ""


def _latest_release_tag() -> tuple[str, tuple[int, int, int]] | None:
    """the highest ``vX.Y.Z`` tag this clone can see.

    :return: the tag name and its parsed version, or ``None`` when the clone has
        no release tags (a shallow checkout, or a fork before its first release)
    :rtype: tuple[str, tuple[int, int, int]] | None
    """
    best: tuple[str, tuple[int, int, int]] | None = None
    for line in _git("tag", "--list", "v*").splitlines():
        matched = _RELEASE_TAG.match(line.strip())
        if matched is None:
            continue
        version = (int(matched[1]), int(matched[2]), int(matched[3]))
        if best is None or version > best[1]:
            best = (line.strip(), version)
    return best


def _workspace_version() -> tuple[int, int, int]:
    """the version every workspace package currently declares, read from core.

    The family moves in lockstep and ``bump-version.sh --verify`` polices that,
    so one package answers for all of them.

    :return: the current ``(major, minor, patch)``
    :rtype: tuple[int, int, int]
    """
    data = tomllib.loads((_REPO_ROOT / "packages/core/pyproject.toml").read_text())
    parts = data["project"]["version"].split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _public_surface(tree: ast.Module) -> set[str]:
    """the names one module offers a consumer, read from its AST.

    ``__all__`` is the declared surface, and the public methods of a class it
    names are reachable through it. Everything else is either private or not
    importable by name.

    :param tree: a parsed module
    :ptype tree: ast.Module
    :return: exported names, plus ``Class.method`` for exported classes
    :rtype: set[str]
    """
    exported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        exported |= set(ast.literal_eval(node.value))
                    except ValueError, TypeError:
                        return set()

    surface = set(exported)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in exported:
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and not item.name.startswith("_"):
                    surface.add(f"{node.name}.{item.name}")
    return surface


def _surface_at(ref: str, package_dir: str) -> dict[str, set[str]]:
    """every module's public surface under *package_dir* at git ref *ref*.

    :param ref: a git ref (tag, sha)
    :ptype ref: str
    :param package_dir: repo-relative package directory
    :ptype package_dir: str
    :return: module path -> its public surface; missing paths are simply absent
    :rtype: dict[str, set[str]]
    """
    listing = _git("ls-tree", "-r", "--name-only", ref, f"{package_dir}/src")
    surfaces: dict[str, set[str]] = {}
    for path in listing.splitlines():
        if not path.endswith(".py"):
            continue
        # A module under a `_`-prefixed name is internal by convention, so growth there is not
        # API a consumer is entitled to import and cannot create the mixed-resolve hazard. What
        # a private module exports to a PUBLIC one still shows up, in that public module's own
        # `__all__` -- so this narrows the noise without opening a way through.
        parts = Path(path).parts
        if any(part.startswith("_") and part != "__init__.py" for part in parts):
            continue
        source = _git("show", f"{ref}:{path}")
        if not source:
            continue
        try:
            surfaces[path] = _public_surface(ast.parse(source))
        except SyntaxError:
            continue
    return surfaces


def _package_dirs() -> list[str]:
    """every workspace package directory holding a pyproject.

    :return: repo-relative package directories, sorted for stable test ids
    :rtype: list[str]
    """
    found: list[str] = []
    for glob in _PACKAGE_GLOBS:
        for path in sorted(_REPO_ROOT.glob(glob)):
            if (path / "pyproject.toml").is_file():
                found.append(path.relative_to(_REPO_ROOT).as_posix())
    return sorted(set(found))


class TestApiGrowthRequiresAMinorBump:
    """A patch release may fix things. It may not add things a consumer can import."""

    def test_no_package_grew_its_public_api_inside_a_patch_bump(self) -> None:
        """Grown surface on a patch line fails here, naming what grew.

        :return: nothing
        :rtype: None
        """
        tag = _latest_release_tag()
        if tag is None:
            pytest.skip("no vX.Y.Z tags visible; a shallow clone cannot compare against a release")

        tag_name, tag_version = tag
        current = _workspace_version()

        if current[:2] != tag_version[:2]:
            # A minor or major bump moves every intra-family bound onto a line whose every
            # published version carries the new API. Growth is exactly what this is for.
            return

        grown: list[str] = []
        for package_dir in _package_dirs():
            before = _surface_at(tag_name, package_dir)
            after = _surface_at("HEAD", package_dir)
            for module_path, names in sorted(after.items()):
                added = names - before.get(module_path, set())
                if added:
                    grown.append(f"{module_path}: {', '.join(sorted(added))}")

        assert not grown, (
            f"these modules gained public API since {tag_name}, but the workspace version "
            f"{'.'.join(map(str, current))} is a PATCH bump on the same minor line.\n\n"
            f"Every intra-family bound reads >={current[0]}.{current[1]}.0,<{current[0]}.{current[1] + 1}.0, "
            f"so pip may resolve a sibling published earlier on this line that lacks these names -- "
            f"a family that installs clean and ImportErrors at runtime.\n\n"
            f"The decision on record (backlog BLD-7QM3, 2026-07-26) is that an intra-family API "
            f"addition ships in a MINOR bump. Run ./scripts/bump-version.sh minor.\n\n"
            + "\n".join(f"  {line}" for line in grown)
        )
