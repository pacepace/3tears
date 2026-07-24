"""AST walkers extracting declared vs actual workspace dependencies.

import-context classification drives the rules:

- **top** -- unguarded module-level import. requires a declared hard
  dependency or coverage by an optional extra.
- **guarded** -- inside a ``try`` whose handlers catch ``ImportError``
  / ``ModuleNotFoundError``. sanctioned optional-dependency shape;
  never flagged missing.
- **deferred** -- inside a function or method body. sanctioned
  lazy-import shape; never flagged missing.
- **type_checking** -- inside ``if TYPE_CHECKING:``. zero runtime
  cost; never flagged missing.

every bucket counts as *usage* for staleness: a declared dependency is
stale only when no module imports it through any path.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from threetears.enforcement.common import Violation

from threetears.enforcement.dependency_alignment.config import DependencyAlignmentConfig

__all__ = [
    "PackageInfo",
    "contract_purity_violations",
    "dependency_alignment_violations",
    "dist_symbol",
]


def dist_symbol(dist: str) -> str:
    """sanitize a distribution name into an exemption-grammar symbol.

    the shared exemption parser requires ``[A-Za-z_][A-Za-z_0-9]*``;
    distribution names (``3tears``, ``3tears-agent-wake``) violate
    both the leading-digit and the dash rule. mapping: the bare core
    distribution ``3tears`` becomes ``core``; any ``3tears-`` prefix is
    stripped and dashes become underscores.

    :param dist: PyPI distribution name
    :ptype dist: str
    :return: exemption-safe symbol (``core``, ``agent_wake``, ...)
    :rtype: str
    """
    if dist == "3tears":
        return "core"
    result = dist.removeprefix("3tears-").replace("-", "_")
    return result


@dataclass
class PackageInfo:
    """one workspace package's declared and actual dependency facts.

    :ivar name: distribution name from ``[project] name``
    :ivar root: package directory (holds ``pyproject.toml``)
    :ivar namespace: import namespace this package owns
        (``threetears.agent.tools``), or ``None`` when no
        ``src/threetears`` tree exists
    :ivar hard_deps: distribution names from ``[project] dependencies``
    :ivar extra_deps: distribution names from every optional extra
    :ivar top_imports: (module, file, line) per unguarded module-top
        ``threetears.*`` import
    :ivar other_imports: modules imported via guarded / deferred /
        ``TYPE_CHECKING`` paths (usage for staleness only)
    """

    name: str
    root: Path
    namespace: str | None
    hard_deps: frozenset[str]
    extra_deps: frozenset[str]
    top_imports: list[tuple[str, Path, int]] = field(default_factory=list)
    other_imports: set[str] = field(default_factory=set)


def _base_dist_name(requirement: str) -> str:
    """strip a PEP 508 requirement down to its distribution name.

    :param requirement: requirement string (``"3tears-nats>=0.9"``)
    :ptype requirement: str
    :return: bare distribution name
    :rtype: str
    """
    result = requirement
    for sep in (">=", "==", "<=", "~=", ">", "<", "[", ";", " "):
        result = result.split(sep)[0]
    return result.strip()


def _import_namespace(pkg_root: Path) -> str | None:
    """find the leaf import namespace a package owns under ``src/threetears``.

    walks namespace directories (no ``__init__.py``) until the first
    directory carrying an ``__init__.py`` -- the leaf package per the
    repo's implicit-namespace convention.

    :param pkg_root: package directory
    :ptype pkg_root: Path
    :return: dotted namespace, or ``None`` when the package has no
        ``src/threetears`` tree
    :rtype: str | None
    """
    base = pkg_root / "src" / "threetears"
    if not base.is_dir():
        return None

    def descend(directory: Path, prefix: str) -> str | None:
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            if (child / "__init__.py").exists():
                return f"{prefix}.{child.name}"
            nested = descend(child, f"{prefix}.{child.name}")
            if nested is not None:
                return nested
        return None

    return descend(base, "threetears")


def _is_type_checking_test(test: ast.expr) -> bool:
    """true for ``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:`` tests."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _handler_catches_import_error(handler: ast.ExceptHandler) -> bool:
    """true when an except handler catches ImportError/ModuleNotFoundError."""
    exc = handler.type
    names: list[str] = []
    if exc is None:
        return False
    candidates = exc.elts if isinstance(exc, ast.Tuple) else [exc]
    for cand in candidates:
        if isinstance(cand, ast.Name):
            names.append(cand.id)
        elif isinstance(cand, ast.Attribute):
            names.append(cand.attr)
    return any(n in ("ImportError", "ModuleNotFoundError") for n in names)


def _is_threetears(module: str) -> bool:
    """true for ``threetears`` and any ``threetears.*`` dotted module."""
    return module == "threetears" or module.startswith("threetears.")


def _collect_imports(
    tree: ast.Module,
    file: Path,
    top: list[tuple[str, Path, int]],
    other: set[str],
) -> None:
    """classify every ``threetears.*`` import in one module.

    thin filter over :func:`_collect_all_imports`.

    :param tree: parsed module
    :ptype tree: ast.Module
    :param file: source path (for violation reporting)
    :ptype file: Path
    :param top: accumulator for unguarded module-top imports
    :ptype top: list[tuple[str, Path, int]]
    :param other: accumulator for guarded/deferred/TYPE_CHECKING imports
    :ptype other: set[str]
    """
    all_top: list[tuple[str, Path, int]] = []
    all_shielded: set[str] = set()
    _collect_all_imports(tree, file, all_top, all_shielded)
    top.extend(entry for entry in all_top if _is_threetears(entry[0]))
    other.update(module for module in all_shielded if _is_threetears(module))


def _load_package(pkg_root: Path) -> PackageInfo | None:
    """parse one package directory into a :class:`PackageInfo`.

    :param pkg_root: directory holding a ``pyproject.toml``
    :ptype pkg_root: Path
    :return: package facts, or ``None`` for non-package directories
    :rtype: PackageInfo | None
    """
    pyproject = pkg_root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    data = tomllib.loads(pyproject.read_text())
    project = data.get("project")
    if project is None:
        return None
    hard = frozenset(_base_dist_name(d) for d in project.get("dependencies", []))
    extras = frozenset(_base_dist_name(d) for group in project.get("optional-dependencies", {}).values() for d in group)
    info = PackageInfo(
        name=project["name"],
        root=pkg_root,
        namespace=_import_namespace(pkg_root),
        hard_deps=hard,
        extra_deps=extras,
    )
    src = pkg_root / "src"
    if src.is_dir():
        for py in sorted(src.rglob("*.py")):
            try:
                tree = ast.parse(py.read_text())
            except SyntaxError:
                continue
            _collect_imports(tree, py, info.top_imports, info.other_imports)
    return info


def contract_purity_violations(config: DependencyAlignmentConfig) -> list[Violation]:
    """flag non-stdlib imports inside designated contracts packages.

    a contracts package's entire value is an empty dependency closure;
    any import outside the allowlist (stdlib, the package's own
    namespace, :attr:`DependencyAlignmentConfig.contract_extra_allowed`)
    silently re-acquires the weight the extraction removed.
    ``TYPE_CHECKING``-shielded imports are permitted -- they cost
    nothing at runtime.

    :param config: per-repo enforcement config
    :ptype config: DependencyAlignmentConfig
    :return: ``contract.impure`` violations
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for rel in config.contract_packages:
        pkg_root = config.repo_root / rel
        namespace = _import_namespace(pkg_root)
        allowed_prefixes = tuple(config.contract_extra_allowed) + ((namespace,) if namespace else ())
        src = pkg_root / "src"
        if not src.is_dir():
            continue
        for py in sorted(src.rglob("*.py")):
            tree = ast.parse(py.read_text())
            top: list[tuple[str, Path, int]] = []
            shielded: set[str] = set()
            _collect_all_imports(tree, py, top, shielded)
            for module, file, line in top:
                root = module.split(".")[0]
                if root in sys.stdlib_module_names:
                    continue
                if any(module == p or module.startswith(p + ".") for p in allowed_prefixes):
                    continue
                violations.append(
                    Violation(
                        category="contract.impure",
                        file=file,
                        line=line,
                        symbol=root,
                        reason=(
                            f"contracts package {rel} imports {module} at module top; contracts "
                            f"may import only the stdlib, their own namespace, and configured "
                            f"extras. move the implementation out or shield under TYPE_CHECKING"
                        ),
                    )
                )
    return violations


def _collect_all_imports(
    tree: ast.Module,
    file: Path,
    top: list[tuple[str, Path, int]],
    shielded: set[str],
) -> None:
    """like :func:`_collect_imports` but for every module, not just threetears.

    reuses the same context classification; guarded / deferred /
    ``TYPE_CHECKING`` imports land in ``shielded``.

    :param tree: parsed module
    :ptype tree: ast.Module
    :param file: source path (for violation reporting)
    :ptype file: Path
    :param top: accumulator for unguarded module-top imports
    :ptype top: list[tuple[str, Path, int]]
    :param shielded: accumulator for shielded imports
    :ptype shielded: set[str]
    """

    def record(node: ast.stmt, is_shielded: bool) -> None:
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        for module in modules:
            if is_shielded:
                shielded.add(module)
            else:
                top.append((module, file, node.lineno))

    def visit(node: ast.stmt, is_shielded: bool) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            record(node, is_shielded)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in node.body:
                visit(child, True)
            return
        if isinstance(node, ast.If):
            branch_shielded = is_shielded or _is_type_checking_test(node.test)
            for child in node.body:
                visit(child, branch_shielded)
            for child in node.orelse:
                visit(child, is_shielded)
            return
        if isinstance(node, ast.Try):
            try_shielded = is_shielded or any(_handler_catches_import_error(h) for h in node.handlers)
            for child in node.body:
                visit(child, try_shielded)
            for handler in node.handlers:
                for stmt in handler.body:
                    visit(stmt, is_shielded)
            for group in (node.orelse, node.finalbody):
                for child in group:
                    visit(child, is_shielded)
            return
        # distinct name from the ``child`` bound by the branches above:
        # those iterate ``list[ast.stmt]`` bodies, whereas
        # ``iter_child_nodes`` yields the wider ``ast.AST``. reusing the
        # name would rebind an already-narrowed local to a supertype,
        # which mypy rejects even though the isinstance guard below makes
        # it correct at runtime.
        for descendant in ast.iter_child_nodes(node):
            if isinstance(descendant, ast.stmt):
                visit(descendant, is_shielded)

    for stmt in tree.body:
        visit(stmt, False)


def dependency_alignment_violations(config: DependencyAlignmentConfig) -> list[Violation]:
    """compare declared dependencies with actual imports per package.

    :param config: per-repo enforcement config
    :ptype config: DependencyAlignmentConfig
    :return: ``dependency.missing`` and ``dependency.stale`` violations
    :rtype: list[Violation]
    """
    packages: list[PackageInfo] = []
    for pattern in config.package_globs:
        for pkg_root in sorted(config.repo_root.glob(pattern)):
            info = _load_package(pkg_root)
            if info is not None:
                packages.append(info)

    namespace_to_dist = {p.namespace: p.name for p in packages if p.namespace is not None}

    def dist_for(module: str) -> str | None:
        best: tuple[str, str] | None = None
        for ns, dist in namespace_to_dist.items():
            if module == ns or module.startswith(ns + "."):
                if best is None or len(ns) > len(best[0]):
                    best = (ns, dist)
        return best[1] if best else None

    violations: list[Violation] = []
    for pkg in packages:
        declared = pkg.hard_deps | pkg.extra_deps
        used_dists: set[str] = set()

        for module, file, line in pkg.top_imports:
            dist = dist_for(module)
            if dist is None or dist == pkg.name:
                continue
            used_dists.add(dist)
            if dist not in declared:
                violations.append(
                    Violation(
                        category="dependency.missing",
                        file=file,
                        line=line,
                        symbol=dist_symbol(dist),
                        reason=(
                            f"{pkg.name} imports {module} at module top but declares no "
                            f"dependency on {dist} (hard or extra). declare it, guard it, "
                            f"or defer it"
                        ),
                    )
                )
        for module in pkg.other_imports:
            dist = dist_for(module)
            if dist is not None and dist != pkg.name:
                used_dists.add(dist)

        workspace_dist_names = {p.name for p in packages}
        for dep in sorted(pkg.hard_deps & workspace_dist_names):
            if dep not in used_dists:
                violations.append(
                    Violation(
                        category="dependency.stale",
                        file=pkg.root / "pyproject.toml",
                        line=1,
                        symbol=dist_symbol(dep),
                        reason=(
                            f"{pkg.name} declares {dep} but no module under its src/ imports "
                            f"it through any path. remove the declaration"
                        ),
                    )
                )

    return violations
