"""enforcement test for raw dict / OrderedDict persistent state.

detects improper use of raw dicts and OrderedDicts as persistent state
in __init__ methods. persistent state should use 3tears L1 backends
(SQLiteBackend) for pod-local cache, or NATS KV for cross-instance
shared state.

scans ALL 3tears packages: core, registry, agent-memory, agent-tools,
langgraph. relative paths are computed from the packages/ directory.

three-tier classification:
- ALLOWLIST: genuinely ephemeral, non-serializable objects (always ok)
- KNOWN_VIOLATIONS: tracked for future migration (acknowledged, not failed)
- unknown violations: test FAILS (must be added to one of the above lists)
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# packages/ directory is the common ancestor for all src roots
_PACKAGES_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# each package's source directory to scan
_SRC_ROOTS: list[tuple[str, Path]] = [
    ("core", _PACKAGES_ROOT / "core" / "src" / "threetears" / "core"),
    ("registry", _PACKAGES_ROOT / "registry" / "src" / "threetears" / "registry"),
    # post-namespace-move (2026-05-06) the agent-* family lives under
    # packages/agent/{name}/. The package_prefix labels keep the
    # historical "agent-memory" / "agent-tools" form because it's
    # what the allowlist below uses; only the disk path changed.
    ("agent-memory", _PACKAGES_ROOT / "agent" / "memory" / "src" / "threetears" / "agent" / "memory"),
    ("agent-tools", _PACKAGES_ROOT / "agent" / "tools" / "src" / "threetears" / "agent" / "tools"),
    ("langgraph", _PACKAGES_ROOT / "langgraph" / "src" / "threetears" / "langgraph"),
]


# ---------------------------------------------------------------------------
# genuinely ephemeral / non-serializable -- never needs migration
# (package_prefix/relative_path, class_name, attr_name, justification)
# ---------------------------------------------------------------------------
ALLOWLIST: list[tuple[str, str, str, str]] = [
    # -- core: collection infrastructure (IS the infrastructure) --
    (
        "core/collections/registry.py",
        "CollectionRegistry",
        "_collections",
        "registry managing L1/L2/L3 backends, IS the infrastructure",
    ),
    (
        "core/collections/registry.py",
        "CollectionRegistry",
        "_overrides",
        "per-collection backend overrides, IS the infrastructure",
    ),
    (
        "core/data/store.py",
        "DataStore",
        "_collections",
        "registry of collection instances, IS the infrastructure",
    ),
    (
        "core/data/migrations/runner.py",
        "MigrationRunner",
        "_packages",
        "static config, PackageMigrations registered once at startup",
    ),
    (
        "core/data/migrations/registry.py",
        "PackageMigrations",
        "_versions",
        "static config, migration callables registered once at startup",
    ),
    (
        "core/data/migrations/registry.py",
        "PackageMigrations",
        "_downgrades",
        "static config, downgrade callables registered once at startup",
    ),
    # -- core: L1/L2 cache backends (non-serializable connection state) --
    (
        "core/cache/kv.py",
        "NatsKvClient",
        "_buckets",
        "live NATS KV connection references, non-serializable",
    ),
    (
        "core/cache/sqlite.py",
        "SQLiteBackend",
        "_schema_info",
        "schema metadata for type-aware serialization, infrastructure",
    ),
    (
        "core/cache/duckdb.py",
        "DuckDBBackend",
        "_schema_info",
        "schema metadata for type-aware serialization, infrastructure",
    ),
    # -- agent-tools: tool serving infrastructure --
    (
        "agent-tools/server.py",
        "ToolServer",
        "_tools",
        "live TearsTool instances, non-serializable",
    ),
    (
        "agent-tools/registry.py",
        "ToolRegistry",
        "_factories",
        "static config, tool factories registered once at startup",
    ),
    (
        "agent-tools/builtin/analyze_media.py",
        "AnalyzeMediaTool",
        "_analyzers",
        "constructor-injected config mapping, not cached state",
    ),
    # -- core: SqlL3Backend table→TableSchema registration map --
    (
        "core/backends/sql.py",
        "SqlL3Backend",
        "_schemas",
        "static config, table→TableSchema registered once at collection "
        "construction (startup); non-serializable (stores TableSchema dataclass "
        "refs, not data), read-only on the CRUD hot path",
    ),
    # -- registry: tool catalog (synced from NATS KV) --
    (
        "registry/catalog.py",
        "ToolCatalog",
        "_entries",
        "synced from NATS KV on load, not a local-only cache",
    ),
    # -- langgraph: typed framework-event class registry --
    (
        "langgraph/events.py",
        "FrameworkEventRegistry",
        "_by_name",
        "static config, FrameworkEvent class objects registered once at "
        "import time; non-serializable (stores Python class refs, not data)",
    ),
]

# ---------------------------------------------------------------------------
# known violations tracked for future migration to L1 / NATS KV
# (package_prefix/relative_path, class_name, attr_name, migration_target)
# ---------------------------------------------------------------------------
KNOWN_VIOLATIONS: list[tuple[str, str, str, str]] = [
    (
        "core/collections/flush.py",
        "WriteBuffer",
        "_buf",
        "pending write buffer in raw dict, migrate to SQLiteBackend L1",
    ),
]

_ALLOWLIST_SET: set[tuple[str, str, str]] = {(path, cls, attr) for path, cls, attr, _ in ALLOWLIST}

_KNOWN_VIOLATIONS_SET: set[tuple[str, str, str]] = {(path, cls, attr) for path, cls, attr, _ in KNOWN_VIOLATIONS}


def _is_dict_literal(node: ast.expr) -> bool:
    """check if AST node is empty dict literal ``{}``.

    :param node: AST expression node to check
    :ptype node: ast.expr
    :return: True if node is empty dict literal
    :rtype: bool
    """
    result = isinstance(node, ast.Dict) and len(node.keys) == 0
    return result


def _is_dict_call(node: ast.expr) -> bool:
    """check if AST node is ``dict()`` call with no arguments.

    :param node: AST expression node to check
    :ptype node: ast.expr
    :return: True if node is dict() call
    :rtype: bool
    """
    result = (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and len(node.args) == 0
        and len(node.keywords) == 0
    )
    return result


def _is_ordered_dict_call(node: ast.expr) -> bool:
    """check if AST node is ``OrderedDict()`` call.

    :param node: AST expression node to check
    :ptype node: ast.expr
    :return: True if node is OrderedDict() call
    :rtype: bool
    """
    result = (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OrderedDict"
        and len(node.args) == 0
    )
    return result


def _is_dict_or_fallback(node: ast.expr) -> bool:
    """check if AST node is ``x or {}`` pattern.

    :param node: AST expression node to check
    :ptype node: ast.expr
    :return: True if node is BoolOp with empty dict fallback
    :rtype: bool
    """
    if not isinstance(node, ast.BoolOp):
        return False
    if not isinstance(node.op, ast.Or):
        return False
    result = any(_is_dict_literal(v) for v in node.values)
    return result


def _is_dict_type_annotation(annotation: ast.expr | None) -> bool:
    """check if annotation references dict or OrderedDict type.

    :param annotation: AST annotation node to check
    :ptype annotation: ast.expr | None
    :return: True if annotation is dict or OrderedDict type
    :rtype: bool
    """
    if annotation is None:
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id in ("dict", "OrderedDict")
    if isinstance(annotation, ast.Subscript):
        if isinstance(annotation.value, ast.Name):
            return annotation.value.id in ("dict", "OrderedDict")
    result = False
    return result


def _is_bad_value(node: ast.expr) -> bool:
    """check if AST expression is dict literal, dict(), OrderedDict(), or x or {}.

    :param node: AST expression node to check
    :ptype node: ast.expr
    :return: True if value represents raw dict state initialization
    :rtype: bool
    """
    result = _is_dict_literal(node) or _is_dict_call(node) or _is_ordered_dict_call(node) or _is_dict_or_fallback(node)
    return result


def _get_self_attr_name(target: ast.expr) -> str | None:
    """extract attribute name from ``self.xxx`` or ``self._xxx`` target, or None.

    the walker accepts both public and private attribute names because
    the Collection-primitive rule applies regardless of python name
    privacy: a bespoke in-memory dict of domain state on a class that
    also has cache-style public api is a cache wrapper and belongs in
    a BaseCollection, whether the dict is named ``_cache`` or ``cache``.

    :param target: AST expression node representing assignment target
    :ptype target: ast.expr
    :return: attribute name if target is self.xxx, else None
    :rtype: str | None
    """
    if not isinstance(target, ast.Attribute):
        return None
    if not isinstance(target.value, ast.Name):
        return None
    if target.value.id != "self":
        return None
    result = target.attr
    return result


def _find_dict_state_violations(
    tree: ast.Module,
) -> list[tuple[str, str, int]]:
    """find all self._xxx dict state assignments in __init__ methods.

    walks AST looking for ClassDef nodes containing __init__ FunctionDef,
    then checks Assign and AnnAssign nodes for dict/OrderedDict state.

    :param tree: parsed AST module
    :ptype tree: ast.Module
    :return: list of (class_name, attr_name, line_number) tuples
    :rtype: list[tuple[str, str, int]]
    """
    violations: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if child.name != "__init__":
                continue
            for stmt in ast.walk(child):
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    attr_name = _get_self_attr_name(stmt.target)
                    if attr_name is None:
                        continue
                    if _is_bad_value(stmt.value) or (
                        _is_dict_type_annotation(stmt.annotation) and _is_bad_value(stmt.value)
                    ):
                        violations.append((node.name, attr_name, stmt.lineno))
                elif isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        attr_name = _get_self_attr_name(target)
                        if attr_name is None:
                            continue
                        if _is_bad_value(stmt.value):
                            violations.append((node.name, attr_name, stmt.lineno))
    result = violations
    return result


def _collect_src_files() -> list[tuple[str, Path]]:
    """collect all Python source files across all 3tears packages.

    returns tuples of (composite_rel_path, absolute_path) where
    composite_rel_path is package_prefix/relative_file_path.

    :return: sorted list of (composite path, absolute path) tuples
    :rtype: list[tuple[str, Path]]
    """
    all_files: list[tuple[str, Path]] = []
    for pkg_prefix, src_root in _SRC_ROOTS:
        if not src_root.exists():
            continue
        for py_file in sorted(src_root.rglob("*.py")):
            rel = str(py_file.relative_to(src_root))
            composite = f"{pkg_prefix}/{rel}"
            all_files.append((composite, py_file))
    result = sorted(all_files, key=lambda t: t[0])
    return result


_SRC_FILES = _collect_src_files()
_SRC_IDS = [composite for composite, _ in _SRC_FILES]
_SRC_PATHS = [path for _, path in _SRC_FILES]


def _get_composite_path(src_file: Path) -> str:
    """resolve composite path (package_prefix/rel) for given source file.

    :param src_file: absolute path to Python source file
    :ptype src_file: Path
    :return: composite path string
    :rtype: str
    """
    for pkg_prefix, src_root in _SRC_ROOTS:
        try:
            rel = str(src_file.relative_to(src_root))
            return f"{pkg_prefix}/{rel}"
        except ValueError:
            continue
    result = str(src_file)
    return result


class TestDictStateDetection:
    """detect raw dict / OrderedDict used as persistent state in __init__."""

    @pytest.mark.parametrize("src_file", _SRC_PATHS, ids=_SRC_IDS)
    def test_no_unknown_dict_state(self, src_file: Path) -> None:
        """scan source file for unknown dict state violations in __init__ methods.

        :param src_file: path to Python source file to scan
        :ptype src_file: Path
        """
        content = src_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(content, filename=str(src_file))
        except SyntaxError:
            return

        rel_path = _get_composite_path(src_file)
        violations = _find_dict_state_violations(tree)
        unknown: list[str] = []
        for class_name, attr_name, lineno in violations:
            key = (rel_path, class_name, attr_name)
            if key in _ALLOWLIST_SET:
                continue
            if key in _KNOWN_VIOLATIONS_SET:
                continue
            unknown.append(f"  {rel_path}:{lineno} {class_name}.{attr_name}")
        if unknown:
            detail = "\n".join(unknown)
            pytest.fail(
                f"{rel_path} has unknown dict state violation(s):\n{detail}\n"
                f"Use a 3tears L1 backend (SQLiteBackend) for pod-local cache,\n"
                f"or NATS KV for cross-instance shared state.\n"
                f"If this is genuinely ephemeral, add to ALLOWLIST with justification."
            )

    @pytest.mark.parametrize("src_file", _SRC_PATHS, ids=_SRC_IDS)
    def test_no_ordered_dict_import(self, src_file: Path) -> None:
        """warn on OrderedDict imports -- almost no valid use in this codebase.

        :param src_file: path to Python source file to scan
        :ptype src_file: Path
        """
        content = src_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(content, filename=str(src_file))
        except SyntaxError:
            return

        rel_path = _get_composite_path(src_file)
        imports: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "collections":
                continue
            if node.names is None:
                continue
            for alias in node.names:
                if alias.name == "OrderedDict":
                    imports.append(f"  line {node.lineno}")

        # allow files that have known violations using OrderedDict
        has_known_od = any(rel_path == path for path, _, _, _ in KNOWN_VIOLATIONS if path == rel_path) or any(
            rel_path == path for path, _, _, _ in ALLOWLIST if path == rel_path
        )

        if imports and not has_known_od:
            detail = "\n".join(imports)
            pytest.fail(
                f"{rel_path} imports OrderedDict from collections:\n{detail}\n"
                f"OrderedDict has almost no valid use in this codebase.\n"
                f"Use a 3tears L1 backend or NATS KV instead."
            )


class TestDictStateAllowlistIntegrity:
    """verify allowlist and known violations reference real code."""

    def test_allowlist_entries_exist(self) -> None:
        """verify every allowlist entry matches real class and attribute in source.

        ensures allowlist does not contain stale entries referencing code
        that has been refactored or removed.
        """
        missing: list[str] = []
        for composite_path, class_name, attr_name, _ in ALLOWLIST:
            # find absolute path from composite
            found_file: Path | None = None
            for pkg_prefix, src_root in _SRC_ROOTS:
                if composite_path.startswith(f"{pkg_prefix}/"):
                    rel = composite_path[len(pkg_prefix) + 1 :]
                    candidate = src_root / rel
                    if candidate.exists():
                        found_file = candidate
                    break
            if found_file is None:
                missing.append(f"  {composite_path}: file not found")
                continue
            content = found_file.read_text(encoding="utf-8")
            try:
                tree = ast.parse(content, filename=str(found_file))
            except SyntaxError:
                missing.append(f"  {composite_path}: syntax error")
                continue
            found = False
            for hit_cls, hit_attr, _ in _find_dict_state_violations(tree):
                if hit_cls == class_name and hit_attr == attr_name:
                    found = True
                    break
            if not found:
                missing.append(
                    f"  {composite_path}:{class_name}.{attr_name}: not found in AST (stale allowlist entry?)"
                )
        if missing:
            detail = "\n".join(missing)
            pytest.fail(f"ALLOWLIST contains entries not found in source:\n{detail}")

    def test_known_violations_exist(self) -> None:
        """verify every known violation entry matches real class and attribute in source.

        ensures known violations list does not contain stale entries
        referencing code that has already been migrated.
        """
        missing: list[str] = []
        for composite_path, class_name, attr_name, _ in KNOWN_VIOLATIONS:
            # find absolute path from composite
            found_file: Path | None = None
            for pkg_prefix, src_root in _SRC_ROOTS:
                if composite_path.startswith(f"{pkg_prefix}/"):
                    rel = composite_path[len(pkg_prefix) + 1 :]
                    candidate = src_root / rel
                    if candidate.exists():
                        found_file = candidate
                    break
            if found_file is None:
                missing.append(f"  {composite_path}: file not found")
                continue
            content = found_file.read_text(encoding="utf-8")
            try:
                tree = ast.parse(content, filename=str(found_file))
            except SyntaxError:
                missing.append(f"  {composite_path}: syntax error")
                continue
            found = False
            for hit_cls, hit_attr, _ in _find_dict_state_violations(tree):
                if hit_cls == class_name and hit_attr == attr_name:
                    found = True
                    break
            if not found:
                missing.append(f"  {composite_path}:{class_name}.{attr_name}: not found in AST (already migrated?)")
        if missing:
            detail = "\n".join(missing)
            pytest.fail(f"KNOWN_VIOLATIONS contains entries not found in source:\n{detail}")
