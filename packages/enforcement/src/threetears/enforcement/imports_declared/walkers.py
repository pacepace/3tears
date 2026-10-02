"""the two inputs of the declared-imports gate, and the comparison between them.

Pure functions over a repo tree and the installed environment; :mod:`.runner` turns their
answers into a pytest verdict.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from collections.abc import Iterator
from importlib.metadata import Distribution, distributions
from pathlib import Path, PurePosixPath
from typing import Final
from urllib.parse import urlparse
from urllib.request import url2pathname

from threetears.enforcement.imports_declared.config import ImportsDeclaredConfig

__all__ = [
    "canonical_distribution_name",
    "declared_distributions",
    "editable_module_files",
    "imported_modules",
    "is_governed",
    "module_name",
    "module_owners",
    "requirement_name",
    "undeclared_imports",
    "unresolved_imports",
]

#: file suffixes that make a module importable. ``.pyi`` is absent on purpose: a stub-only
#: distribution (``types-PyYAML``) provides no runtime module and must not own one.
_MODULE_SUFFIXES: Final[tuple[str, ...]] = (".py", ".so", ".pyd")

#: PEP 503 normalisation: runs of ``-``, ``_`` and ``.`` collapse to one ``-``.
_NAME_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"[-_.]+")

#: separators that end a requirement's name, in the order they can appear.
_REQUIREMENT_NAME_ENDS: Final[re.Pattern[str]] = re.compile(r"[\s\[;<>=!~@(]")


def canonical_distribution_name(name: str) -> str:
    """normalise a distribution name the way the index does.

    :param name: distribution name as written in metadata or a requirement
    :ptype name: str
    :return: lowercased name with separator runs folded to one hyphen
    :rtype: str
    """
    return _NAME_SEPARATORS.sub("-", name.strip()).lower()


def requirement_name(requirement: str) -> str:
    """extract the distribution name a requirement string names.

    ``"3tears-iam[saml]>=0.22,<1"`` names ``3tears-iam``; ``"uvicorn[standard]>=0.30"``
    names ``uvicorn``.

    :param requirement: PEP 508 requirement string
    :ptype requirement: str
    :return: canonical distribution name
    :rtype: str
    """
    return canonical_distribution_name(_REQUIREMENT_NAME_ENDS.split(requirement.strip(), maxsplit=1)[0])


def declared_distributions(manifest: Path) -> set[str]:
    """every distribution a manifest declares, normalised for comparison.

    reads runtime dependencies, optional-dependency extras and dependency groups alike: an
    import under ``src/`` must be satisfied by a RUNTIME declaration, but a dev-only
    declaration is still a declaration, and reporting one as undeclared would send whoever
    reads the failure looking for the wrong thing.

    :param manifest: a ``pyproject.toml``
    :ptype manifest: Path
    :return: canonical names of every declared distribution
    :rtype: set[str]
    """
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    project = data.get("project", {})
    declared: list[str] = list(project.get("dependencies", []))
    for extra in (project.get("optional-dependencies") or {}).values():
        declared.extend(extra)
    for group in (data.get("dependency-groups") or {}).values():
        declared.extend(dep for dep in group if isinstance(dep, str))
    return {requirement_name(dep) for dep in declared}


def editable_module_files(dist: Distribution) -> Iterator[PurePosixPath]:
    """yield the module paths an EDITABLE distribution provides, relative to its import root.

    an editable install records no module files at all: ``dist.files`` lists only the
    ``.dist-info`` metadata and a ``.pth`` shim, so the file-list walk sees nothing and every
    import it provides resolves to no owner. That failure is ASYMMETRIC in the worst
    direction: the resolution check goes red, which is recoverable, while the rule itself
    goes green and VACUOUS, because an empty owner map has no offenders to report.

    PEP 610 records the source directory in ``direct_url.json``, surfaced as ``dist.origin``,
    so the module list is recoverable by walking the checkout: its ``src/`` tree when it has
    one, otherwise its top-level packages that carry an ``__init__.py``.

    :param dist: an installed distribution, editable or not
    :ptype dist: Distribution
    :return: iterator of import-root-relative module paths, empty for a non-editable one
    :rtype: Iterator[PurePosixPath]
    """
    origin = getattr(dist, "origin", None)
    dir_info = getattr(origin, "dir_info", None)
    url = getattr(origin, "url", "") or ""
    if not getattr(dir_info, "editable", False) or not url.startswith("file:"):
        return
    root = Path(url2pathname(urlparse(url).path))
    src = root / "src"
    if src.is_dir():
        packages = [src]
    else:
        packages = [child for child in root.iterdir() if (child / "__init__.py").is_file()]
    for package in packages:
        base = src if package == src else root
        for module_file in package.rglob("*.py"):
            yield PurePosixPath(*module_file.relative_to(base).parts)


def module_name(file: PurePosixPath) -> str | None:
    """turn a distribution file path into the dotted module it makes importable.

    ``PIL/Image.py`` is ``PIL.Image``; ``PIL/_imaging.cpython-314-darwin.so`` is
    ``PIL._imaging``; ``threetears/iam/__init__.py`` is ``threetears.iam``. Metadata, data
    files and scripts (``../../bin/x``, ``pillow-12.0.dist-info/RECORD``) are not modules.

    :param file: path as a distribution's file list records it
    :ptype file: PurePosixPath
    :return: dotted module path, or ``None`` when the file is not an importable module
    :rtype: str | None
    """
    parts = file.parts
    result: str | None = None
    if parts and file.name.endswith(_MODULE_SUFFIXES):
        stem = file.name.split(".", 1)[0]
        dotted = [*parts[:-1], stem]
        if all(part.isidentifier() for part in dotted):
            result = ".".join(dotted).removesuffix(".__init__")
    return result


def module_owners() -> dict[str, set[str]]:
    """map each installed module, and every package above it, to the distributions providing it.

    built from the installed distributions' own file lists (and editable checkouts), so it
    reflects what is on disk rather than what anyone believes the layout to be. A namespace
    package (``threetears``, ``google.cloud``) is owned by every distribution contributing to
    it, while ``threetears.agent.tools.base_tool`` is owned by ``3tears-agent-tools`` alone.

    :return: dotted module path to canonical owning distribution names
    :rtype: dict[str, set[str]]
    """
    owners: dict[str, set[str]] = {}
    for dist in distributions():
        name = canonical_distribution_name(dist.metadata["Name"] or "")
        if not name:
            continue
        files = (*(PurePosixPath(str(f)) for f in dist.files or ()), *editable_module_files(dist))
        for file in files:
            module = module_name(file)
            if module is None:
                continue
            segments = module.split(".")
            for depth in range(1, len(segments) + 1):
                owners.setdefault(".".join(segments[:depth]), set()).add(name)
    return owners


def is_governed(module: str, first_party: frozenset[str]) -> bool:
    """report whether an absolute import is a third-party one the gate governs.

    :param module: dotted module path as written in the import
    :ptype module: str
    :param first_party: the repo's own import roots
    :ptype first_party: frozenset[str]
    :return: ``False`` for the standard library, ``__future__`` and the repo's own packages
    :rtype: bool
    """
    top = module.split(".", 1)[0]
    return top not in sys.stdlib_module_names and top not in first_party and top != "__future__"


def imported_modules(config: ImportsDeclaredConfig) -> dict[str, set[Path]]:
    """every governed module imported under the configured source roots, with its importers.

    an import inside ``try: ... except ImportError`` is held to the same rule as any other:
    an optional import of ours is still ours to declare, as an extra.

    :param config: the repo's gate configuration
    :ptype config: ImportsDeclaredConfig
    :return: dotted module path to the repo-relative files importing it
    :rtype: dict[str, set[Path]]
    :raises SyntaxError: when a source file does not parse -- a file the gate cannot read
        is not a file it checked
    """
    found: dict[str, set[Path]] = {}
    for source_root in config.source_roots:
        for path in sorted((config.repo_root / source_root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                for name in names:
                    if is_governed(name, config.first_party):
                        found.setdefault(name, set()).add(path.relative_to(config.repo_root))
    return found


def unresolved_imports(imports: dict[str, set[Path]], owners: dict[str, set[str]]) -> list[str]:
    """imports nothing installed provides; they would fail at runtime.

    their fix is a sync, not a declaration, so they are reported apart from the rule.

    :param imports: dotted module path to the files importing it
    :ptype imports: dict[str, set[Path]]
    :param owners: dotted module path to owning distribution names
    :ptype owners: dict[str, set[str]]
    :return: sorted unresolved module paths
    :rtype: list[str]
    """
    return sorted(module for module in imports if module not in owners)


def undeclared_imports(
    imports: dict[str, set[Path]],
    owners: dict[str, set[str]],
    declared: set[str],
) -> dict[str, tuple[list[str], list[str]]]:
    """find imports whose every owning distribution is undeclared.

    one declared owner is enough: where a module path is shipped by several distributions
    (``google.cloud``), declaring any one is what makes the import work, and the walk cannot
    tell which one an import means.

    :param imports: dotted module path to the files importing it
    :ptype imports: dict[str, set[Path]]
    :param owners: dotted module path to owning distribution names
    :ptype owners: dict[str, set[str]]
    :param declared: canonical names of declared distributions
    :ptype declared: set[str]
    :return: offending module to (its owners, the files importing it)
    :rtype: dict[str, tuple[list[str], list[str]]]
    """
    offenders: dict[str, tuple[list[str], list[str]]] = {}
    for module, importers in imports.items():
        owned_by = owners.get(module, set())
        if owned_by and not owned_by & declared:
            offenders[module] = (sorted(owned_by), sorted(str(p) for p in importers))
    return offenders
