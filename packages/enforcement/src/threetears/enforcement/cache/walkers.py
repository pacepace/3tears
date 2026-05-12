"""walkers for the cache-primitive-usage enforcement domain.

four walkers implement the four contract surfaces:

- :func:`find_sqlite_constructions` — flag every ``SQLiteBackend(...)``
  call outside the sanctioned factories. ``tests/`` trees are always
  permitted regardless of the configured allowlist; the canonical
  hardcodes that.
- :func:`find_wrapper_classes` — flag classes that store a
  :class:`SQLiteBackend` field AND expose cache api (``get`` / ``put``
  / ``set`` / ``delete`` / ``upsert``) AND do not transitively subclass
  any of the configured base-collection names. uses the common
  inheritance graph so chains crossing path-dep package boundaries
  resolve correctly.
- :func:`find_direct_pool_access` — flag ``pool.fetch`` / ``fetchrow``
  / ``fetchval`` / ``execute`` / ``executemany`` calls whose first SQL
  argument names a table mapped in ``collection_table_allowlist``.
  allowed sites: inside the Collection class itself, or preceded by a
  ``# cache-bypass: <reason>`` comment within ten lines above the call.
- :func:`find_missing_collections` — walk every migration file
  (basename matches ``v\\d+_*.py`` under a ``migrations/`` directory),
  extract ``CREATE TABLE`` statements, and verify every table in the
  config's allowlist points to a class that transitively subclasses a
  base-collection name. **uses** :func:`transitively_subclasses_any
  <threetears.enforcement.common.inheritance.transitively_subclasses_any>`
  — this is the bug fix that replaces the canonical's direct-only
  ``_class_subclasses_base_collection`` helper.

translation note: the canonical hardcoded a ``_COLLECTION_BASE_NAMES``
frozenset with intermediate bases (``SchemaBackedCollection``, etc.) as
a workaround for the missing transitive walk. this lift drops that
workaround entirely — the only configurable terminal name is
``BaseCollection`` (by default) and the transitive walk handles every
intermediate base automatically.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from threetears.enforcement.common import (
    ClassBaseGraph,
    Violation,
    collect_class_base_graph,
    iter_python_files,
    parse_python_file,
    relative_posix_path,
    transitively_subclasses_any,
)

__all__ = [
    "find_direct_pool_access",
    "find_missing_collections",
    "find_sqlite_constructions",
    "find_wrapper_classes",
]


_SQLITE_CONSTRUCTION_CATEGORY = "cache.sqlite_construction"
_WRAPPER_CLASS_CATEGORY = "cache.wrapper_class"
_POOL_ACCESS_CATEGORY = "cache.pool_access"
_MISSING_COLLECTION_CATEGORY = "cache.missing_collection"


# ----------------------------------------------------------------------
# walker 1 — SQLiteBackend construction outside sanctioned factories
# ----------------------------------------------------------------------


def find_sqlite_constructions(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
    allowed_sites: frozenset[str],
) -> list[Violation]:
    """flag every ``SQLiteBackend(...)`` call outside sanctioned sites.

    accepted call shapes:

    - bare-name ``SQLiteBackend(...)`` (the import-and-call form)
    - dotted ``module.SQLiteBackend(...)`` (the qualified form)

    permitted sites:

    - any file under a ``tests/`` directory (the canonical hardcodes
      this — test fixtures construct backends directly to exercise the
      cache tier)
    - files whose forward-slash repo-relative path is in
      ``allowed_sites`` (named factories like
      ``hub/common/l1_cache.py``)

    every other site emits one violation per call. ``symbol`` is
    ``"SQLiteBackend"`` and ``line`` is the call's lineno.

    :param scan_roots: where to look for violations; only iterates
        these roots and does not consult the inheritance graph
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root for relative-path comparison against
        ``allowed_sites``
    :ptype repo_root: Path
    :param allowed_sites: forward-slash relative paths permitted to
        construct :class:`SQLiteBackend`
    :ptype allowed_sites: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in scan_roots:
        for source in iter_python_files(root):
            rel = relative_posix_path(source, repo_root)
            if _is_under_tests(source) or _is_under_tests_rel(rel):
                continue
            if rel in allowed_sites:
                continue
            tree = parse_python_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_target_name(node.func)
                if name != "SQLiteBackend":
                    continue
                violations.append(
                    Violation(
                        category=_SQLITE_CONSTRUCTION_CATEGORY,
                        file=source,
                        line=node.lineno,
                        symbol="SQLiteBackend",
                        reason=(
                            "SQLiteBackend constructed outside the "
                            f"sanctioned factories; add {rel} to the "
                            "allowlist (with rationale) or route through "
                            "a Collection instead"
                        ),
                    )
                )
    return violations


def _is_under_tests(path: Path) -> bool:
    """true iff any path component equals ``tests``.

    :param path: absolute file path
    :ptype path: Path
    :return: whether ``path`` lives under a ``tests`` directory
    :rtype: bool
    """
    return "tests" in path.parts


def _is_under_tests_rel(rel: str) -> bool:
    """true iff any forward-slash segment of ``rel`` equals ``tests``.

    redundant with :func:`_is_under_tests` for typical absolute paths,
    but the relative form catches cases where the absolute path uses
    a leading symlink that elides the ``tests`` segment.

    :param rel: forward-slash relative path
    :ptype rel: str
    :return: whether any segment is ``tests``
    :rtype: bool
    """
    return "tests" in rel.split("/")


def _call_target_name(func: ast.expr) -> str | None:
    """return the last-segment name of a call target, or ``None``.

    :param func: ast expression in a :class:`ast.Call.func` slot
    :ptype func: ast.expr
    :return: bare name (``Name.id`` or ``Attribute.attr``) or ``None``
    :rtype: str | None
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


# ----------------------------------------------------------------------
# walker 2 — bespoke cache wrapper classes
# ----------------------------------------------------------------------


def find_wrapper_classes(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
    inheritance_roots: tuple[Path, ...],
    base_collection_names: frozenset[str],
    cache_method_names: frozenset[str],
) -> list[Violation]:
    """flag classes that wrap :class:`SQLiteBackend` without subclassing a Collection.

    fingerprint:

    - the class stores a :class:`SQLiteBackend` field via any of:
      annotated class-body attribute (``_l1: SQLiteBackend``),
      annotated init-body assign (``self._l1: SQLiteBackend = ...``),
      inline construction in init (``self._l1 = SQLiteBackend(...)``),
      or constructor injection (``def __init__(self, l1: SQLiteBackend
      | None): self._l1 = l1``).
    - the class exposes a public method whose name is in
      ``cache_method_names`` (``get`` / ``put`` / ``set`` / ``delete``
      / ``upsert``), OR the class calls SQLiteBackend's row-level data
      api (``upsert`` / ``select_by_id`` / ``delete_by_id`` /
      ``execute_query``) on a stored backend from a method body.
    - the class does NOT transitively subclass any name in
      ``base_collection_names`` — the chain through intermediate bases
      like ``SchemaBackedCollection`` is walked automatically.

    one violation per offending class. ``symbol`` is the class name;
    ``line`` is the :class:`ast.ClassDef.lineno`.

    :param scan_roots: where to look for violations; iterates only
        these roots when scanning class definitions
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for parity with sibling
        walkers; relative-path rendering happens at report time)
    :ptype repo_root: Path
    :param inheritance_roots: where to build the cross-package
        class-base graph used by the transitive subclass walk; MUST
        include any path-dep targets that declare intermediate
        Collection bases (``SchemaBackedCollection``, etc.) so chains
        crossing repo boundaries resolve.
    :ptype inheritance_roots: tuple[Path, ...]
    :param base_collection_names: terminal class names that satisfy
        "this is a Collection." default is ``{"BaseCollection"}`` —
        transitive walking handles intermediate bases automatically.
    :ptype base_collection_names: frozenset[str]
    :param cache_method_names: public method names whose presence,
        combined with a :class:`SQLiteBackend` field, fingerprints a
        wrapper class.
    :ptype cache_method_names: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    _ = repo_root
    base_graph = collect_class_base_graph(inheritance_roots)
    violations: list[Violation] = []
    for root in scan_roots:
        for source in iter_python_files(root):
            tree = parse_python_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if transitively_subclasses_any(
                    node.name,
                    base_collection_names,
                    base_graph,
                ):
                    continue
                sqlite_attrs = _class_sqlite_attr_names(node)
                if not sqlite_attrs:
                    continue
                methods = _class_cache_methods(node, cache_method_names)
                data_api_calls = _class_calls_sqlite_data_api(
                    node,
                    sqlite_attrs,
                )
                if not methods and not data_api_calls:
                    continue
                if methods:
                    reason = (
                        f"class '{node.name}' wraps SQLiteBackend and "
                        f"exposes cache api {sorted(methods)!r} without "
                        "extending BaseCollection; convert to a Collection "
                        "subclass"
                    )
                else:
                    reason = (
                        f"class '{node.name}' wraps SQLiteBackend and "
                        f"calls its data api {data_api_calls!r} from "
                        "method bodies without extending BaseCollection "
                        "(domain-verb wrapper); convert to a Collection "
                        "subclass"
                    )
                violations.append(
                    Violation(
                        category=_WRAPPER_CLASS_CATEGORY,
                        file=source,
                        line=node.lineno,
                        symbol=node.name,
                        reason=reason,
                    )
                )
    return violations


_SQLITE_DATA_API_METHODS: frozenset[str] = frozenset(
    {
        "upsert",
        "select_by_id",
        "delete_by_id",
        "execute_query",
    }
)


def _annotation_names_sqlite_backend(ann: ast.AST | None) -> bool:
    """true iff ``ann`` textually names :class:`SQLiteBackend`.

    matches:

    - bare ``SQLiteBackend``
    - dotted ``some.module.SQLiteBackend``
    - union ``SQLiteBackend | None``
    - subscripted ``Optional[SQLiteBackend]``
    - tuples (variadic Generic args)

    :param ann: annotation ast node, or ``None``
    :ptype ann: ast.AST | None
    :return: whether the annotation names :class:`SQLiteBackend`
    :rtype: bool
    """
    if ann is None:
        return False
    if isinstance(ann, ast.Name):
        return ann.id == "SQLiteBackend"
    if isinstance(ann, ast.Attribute):
        return ann.attr == "SQLiteBackend"
    if isinstance(ann, ast.BinOp):
        return _annotation_names_sqlite_backend(ann.left) or _annotation_names_sqlite_backend(ann.right)
    if isinstance(ann, ast.Subscript):
        return _annotation_names_sqlite_backend(ann.value) or _annotation_names_sqlite_backend(ann.slice)
    if isinstance(ann, ast.Tuple):
        return any(_annotation_names_sqlite_backend(elt) for elt in ann.elts)
    return False


def _class_sqlite_attr_names(cls: ast.ClassDef) -> frozenset[str]:
    """return ``self.<attr>`` names on ``cls`` that hold a :class:`SQLiteBackend`.

    three sources contribute:

    - annotated class-body attribute (``_l1: SQLiteBackend``)
    - init-body annotated assignments and inline constructions
      (``self._l1: SQLiteBackend = ...`` / ``self._l1 =
      SQLiteBackend(...)``)
    - constructor injection: an ``__init__`` parameter typed
      :class:`SQLiteBackend` that is stored on a ``self.<attr>``
      target (``def __init__(self, l1: SQLiteBackend): self._l1 = l1``)

    :param cls: class definition
    :ptype cls: ast.ClassDef
    :return: frozen set of attr names; empty when the class stores no
        backend
    :rtype: frozenset[str]
    """
    names: set[str] = set()

    for item in cls.body:
        if isinstance(item, ast.AnnAssign):
            if _annotation_names_sqlite_backend(item.annotation):
                target = item.target
                if isinstance(target, ast.Name):
                    names.add(target.id)

    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name != "__init__":
            continue
        sqlite_params: set[str] = set()
        args = item.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if _annotation_names_sqlite_backend(arg.annotation):
                sqlite_params.add(arg.arg)
        for sub in ast.walk(item):
            if isinstance(sub, ast.AnnAssign):
                if _annotation_names_sqlite_backend(sub.annotation):
                    target = sub.target
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        names.add(target.attr)
            if isinstance(sub, ast.Assign):
                rhs = sub.value
                rhs_is_sqlite_call = False
                if isinstance(rhs, ast.Call):
                    func = rhs.func
                    if isinstance(func, ast.Name) and func.id == "SQLiteBackend":
                        rhs_is_sqlite_call = True
                    if isinstance(func, ast.Attribute) and func.attr == "SQLiteBackend":
                        rhs_is_sqlite_call = True
                rhs_is_sqlite_param = isinstance(rhs, ast.Name) and rhs.id in sqlite_params
                if not (rhs_is_sqlite_call or rhs_is_sqlite_param):
                    continue
                for tgt in sub.targets:
                    if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                        names.add(tgt.attr)
    return frozenset(names)


def _class_cache_methods(
    cls: ast.ClassDef,
    cache_method_names: frozenset[str],
) -> list[str]:
    """return public methods on ``cls`` whose name is in ``cache_method_names``.

    underscore-prefixed methods are intentionally skipped; they are
    private and do not constitute a public cache api.

    :param cls: class definition
    :ptype cls: ast.ClassDef
    :param cache_method_names: cache-api vocabulary
    :ptype cache_method_names: frozenset[str]
    :return: list of matching public method names in source order
    :rtype: list[str]
    """
    result: list[str] = []
    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = item.name
        if name.startswith("_"):
            continue
        if name in cache_method_names:
            result.append(name)
    return result


def _class_calls_sqlite_data_api(
    cls: ast.ClassDef,
    sqlite_attrs: frozenset[str],
) -> list[str]:
    """return data-api method names called on a stored :class:`SQLiteBackend`.

    walks every method body of ``cls`` and returns
    :data:`_SQLITE_DATA_API_METHODS` entries that the class invokes on
    a ``self.<attr>`` receiver where ``<attr>`` is a known backend
    field. also accepts a local rebind (``l1 = self._l1;
    l1.upsert(...)``) by tracking simple ``Name = self.<attr>``
    assignments.

    :param cls: class definition
    :ptype cls: ast.ClassDef
    :param sqlite_attrs: ``self`` attr names that hold a
        :class:`SQLiteBackend` (from :func:`_class_sqlite_attr_names`)
    :ptype sqlite_attrs: frozenset[str]
    :return: sorted list of distinct data-api method names called
    :rtype: list[str]
    """
    if not sqlite_attrs:
        return []
    hits: set[str] = set()
    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local_binds: set[str] = set(sqlite_attrs)
        for sub in ast.walk(item):
            if isinstance(sub, ast.Assign):
                rhs = sub.value
                if (
                    isinstance(rhs, ast.Attribute)
                    and isinstance(rhs.value, ast.Name)
                    and rhs.value.id == "self"
                    and rhs.attr in sqlite_attrs
                ):
                    for tgt in sub.targets:
                        if isinstance(tgt, ast.Name):
                            local_binds.add(tgt.id)
        for sub in ast.walk(item):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in _SQLITE_DATA_API_METHODS:
                continue
            receiver = func.value
            receiver_is_self_attr = (
                isinstance(receiver, ast.Attribute)
                and isinstance(receiver.value, ast.Name)
                and receiver.value.id == "self"
                and receiver.attr in sqlite_attrs
            )
            receiver_is_local_bind = isinstance(receiver, ast.Name) and receiver.id in local_binds
            if receiver_is_self_attr or receiver_is_local_bind:
                hits.add(func.attr)
    return sorted(hits)


# ----------------------------------------------------------------------
# walker 3 — direct pool access to Collection-backed tables
# ----------------------------------------------------------------------


_POOL_METHOD_NAMES: frozenset[str] = frozenset(
    {
        "fetch",
        "fetchrow",
        "fetchval",
        "execute",
        "executemany",
    }
)


_SQL_TARGET_RE = re.compile(
    r"\b(?:from|into|update|join|table)\s+([A-Za-z_][A-Za-z_0-9]*)",
    re.IGNORECASE,
)


def find_direct_pool_access(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
    collection_table_allowlist: dict[str, str],
) -> list[Violation]:
    """flag ``pool.<method>(<sql>)`` whose SQL touches a Collection-backed table.

    matched call shapes:

    - any :class:`ast.Call` whose ``.func`` is a
      :class:`ast.Attribute` with ``.attr`` in
      :data:`_POOL_METHOD_NAMES` (``fetch`` / ``fetchrow`` /
      ``fetchval`` / ``execute`` / ``executemany``).
    - the receiver's name (last :class:`ast.Name` / :class:`ast.
      Attribute` segment) must contain ``"pool"`` or ``"store"`` —
      this filters out unrelated ``str.fetch`` / ``dict.get`` shapes
      that share method names with pool wrappers.
    - the first positional argument is a string literal (or joined
      f-string of literals) whose text matches the
      ``FROM/INTO/UPDATE/JOIN/TABLE <name>`` regex with at least one
      table in ``collection_table_allowlist``.

    permitted sites:

    - the call lives inside a class whose name is the value of an
      entry in ``collection_table_allowlist`` (a Collection's own
      method body is allowed to use the pool).
    - the call is preceded by a ``# cache-bypass: <reason>`` comment
      in the ten lines above the call (case-insensitive substring
      match).

    one violation per ``(call, table)`` pair when multiple allowlisted
    tables appear in the same SQL. ``symbol`` is the table name;
    ``line`` is the call's lineno.

    :param scan_roots: where to look for violations; only iterates
        these roots
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for parity with sibling
        walkers)
    :ptype repo_root: Path
    :param collection_table_allowlist: ``{table_name: collection_class_name}``
        mapping
    :ptype collection_table_allowlist: dict[str, str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    _ = repo_root
    violations: list[Violation] = []
    allowed_tables = set(collection_table_allowlist.keys())
    collection_names = set(collection_table_allowlist.values())
    for root in scan_roots:
        for source in iter_python_files(root):
            tree = parse_python_file(source)
            if tree is None:
                continue
            try:
                raw_lines = source.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            parents = _build_parent_map(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                if func.attr not in _POOL_METHOD_NAMES:
                    continue
                if not _receiver_is_pool_like(func.value):
                    continue
                tables = _extract_sql_tables(node)
                hits = tables & allowed_tables
                if not hits:
                    continue
                if _has_bypass_comment(raw_lines, node.lineno):
                    continue
                enclosing = _enclosing_class(node, parents)
                if enclosing is not None and enclosing.name in collection_names:
                    continue
                for table in sorted(hits):
                    violations.append(
                        Violation(
                            category=_POOL_ACCESS_CATEGORY,
                            file=source,
                            line=node.lineno,
                            symbol=table,
                            reason=(
                                f"direct pool.{func.attr}() on table "
                                f"'{table}' which has a Collection "
                                f"({collection_table_allowlist[table]}); "
                                "go through the Collection or add a "
                                "'# cache-bypass: <reason>' comment"
                            ),
                        )
                    )
    return violations


def _receiver_is_pool_like(receiver: ast.expr) -> bool:
    """true iff the call receiver's name contains ``pool`` or ``store``.

    :param receiver: the call's receiver expression
        (``func.value``)
    :ptype receiver: ast.expr
    :return: whether the receiver looks like a pool / store
    :rtype: bool
    """
    name = ""
    if isinstance(receiver, ast.Name):
        name = receiver.id
    elif isinstance(receiver, ast.Attribute):
        name = receiver.attr
    lowered = name.lower()
    return "pool" in lowered or "store" in lowered


def _extract_sql_tables(call_node: ast.Call) -> set[str]:
    """extract every table-name reference from the call's first SQL arg.

    accepts either a plain string-literal argument or a joined
    f-string composed of string-literal segments (interpolation slots
    are skipped). returns the empty set for dynamic SQL.

    :param call_node: ast call node
    :ptype call_node: ast.Call
    :return: set of lowercase table names mentioned in the SQL
    :rtype: set[str]
    """
    if not call_node.args:
        return set()
    first = call_node.args[0]
    text: str | None = None
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        text = first.value
    elif isinstance(first, ast.JoinedStr):
        parts: list[str] = []
        for v in first.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
        text = "".join(parts)
    if text is None:
        return set()
    return {match.group(1).lower() for match in _SQL_TARGET_RE.finditer(text)}


def _has_bypass_comment(raw_lines: list[str], lineno: int) -> bool:
    """true iff a ``cache-bypass:`` marker appears within ten lines above ``lineno``.

    :param raw_lines: full source split on newlines
    :ptype raw_lines: list[str]
    :param lineno: 1-based line of the call
    :ptype lineno: int
    :return: whether a bypass marker is present
    :rtype: bool
    """
    for prev_line in range(max(0, lineno - 10), lineno):
        if prev_line >= len(raw_lines):
            continue
        if "cache-bypass:" in raw_lines[prev_line]:
            return True
    return False


def _build_parent_map(module: ast.Module) -> dict[int, ast.AST]:
    """build a child-id -> parent-node map for ``module``.

    :param module: parsed module
    :ptype module: ast.Module
    :return: mapping from ``id(child)`` to its immediate parent
    :rtype: dict[int, ast.AST]
    """
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(module):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _enclosing_class(
    node: ast.AST,
    parents: dict[int, ast.AST],
) -> ast.ClassDef | None:
    """return the innermost enclosing :class:`ast.ClassDef`, or ``None``.

    :param node: target node
    :ptype node: ast.AST
    :param parents: child-id -> parent map from
        :func:`_build_parent_map`
    :ptype parents: dict[int, ast.AST]
    :return: the enclosing class def, or ``None``
    :rtype: ast.ClassDef | None
    """
    current: ast.AST | None = node
    while current is not None:
        current = parents.get(id(current))
        if isinstance(current, ast.ClassDef):
            return current
    return None


# ----------------------------------------------------------------------
# walker 4 — migration-defined tables without Collections (the bug fix)
# ----------------------------------------------------------------------


_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?!IF\b|NOT\b|EXISTS\b)"
    r"([A-Za-z_][A-Za-z_0-9\.]*)",
    re.IGNORECASE,
)


_MIGRATION_FILENAME_RE = re.compile(r"^v\d+_.+\.py$")


def find_missing_collections(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
    inheritance_roots: tuple[Path, ...],
    collection_table_allowlist: dict[str, str],
    migration_table_allowlist: frozenset[str],
    base_collection_names: frozenset[str],
) -> list[Violation]:
    """flag migration-defined tables whose Collection class is missing or non-Collection.

    strategy:

    1. build the cross-package class-base graph via
       :func:`~threetears.enforcement.common.inheritance.collect_class_base_graph`
       over ``inheritance_roots`` so the transitive walk sees
       intermediate Collection bases (``SchemaBackedCollection``,
       etc.) regardless of which package they live in.
    2. walk every file under a ``migrations/`` directory below
       ``scan_roots`` whose basename matches the canonical
       ``v\\d+_<suffix>.py`` template (helper modules ``template.py`` /
       ``drift.py`` / ``__init__.py`` are skipped).
    3. extract ``CREATE TABLE`` statements from SQL-shaped string
       literals in the module ast (docstrings excluded — module /
       class / function docstrings cannot create tables).
    4. for each table:

       - strip schema prefix (``platform.customers`` → ``customers``)
       - if in ``migration_table_allowlist``: skip
       - if not in ``collection_table_allowlist``: emit "no mapping"
         violation
       - otherwise verify the named class transitively subclasses any
         name in ``base_collection_names`` via
         :func:`~threetears.enforcement.common.inheritance.transitively_subclasses_any`.
         **this is the bug fix**: the canonical's
         ``_class_subclasses_base_collection`` only walked direct
         AST bases, so chains through ``SchemaBackedCollection``
         (in a sibling path-dep package) were missed and reported
         as spurious violations.

    one violation per offending table. ``symbol`` is the table name;
    ``line`` is the line of the SQL string in the migration file.

    :param scan_roots: where to look for migration files; the walker
        only iterates these roots when discovering ``migrations/``
        directories
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for parity with sibling
        walkers; relative-path rendering happens at report time)
    :ptype repo_root: Path
    :param inheritance_roots: where to build the cross-package class-
        base graph; MUST include any path-dep targets that declare
        intermediate Collection bases declared in sibling packages so
        chains crossing repo boundaries resolve
    :ptype inheritance_roots: tuple[Path, ...]
    :param collection_table_allowlist: ``{table_name:
        expected_collection_class_name}`` mapping
    :ptype collection_table_allowlist: dict[str, str]
    :param migration_table_allowlist: tables that legitimately lack
        Collections (bookkeeping, LangGraph state, sibling-repo-owned)
    :ptype migration_table_allowlist: frozenset[str]
    :param base_collection_names: terminal class names that satisfy
        "this is a Collection." default callers pass
        ``{"BaseCollection"}``; the transitive walk handles
        intermediate bases automatically.
    :ptype base_collection_names: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    _ = repo_root
    base_graph = collect_class_base_graph(inheritance_roots)
    violations: list[Violation] = []
    seen: set[str] = set()
    for root in scan_roots:
        for source in iter_python_files(root):
            if "migrations" not in source.parts:
                continue
            if not _MIGRATION_FILENAME_RE.match(source.name):
                continue
            tree = parse_python_file(source)
            if tree is None:
                continue
            for lineno, text in _extract_sql_strings(tree):
                for match in _CREATE_TABLE_RE.finditer(text):
                    table_raw = match.group(1).lower()
                    if "." in table_raw:
                        table_raw = table_raw.rsplit(".", 1)[1]
                    if table_raw in seen:
                        continue
                    seen.add(table_raw)
                    if table_raw in migration_table_allowlist:
                        continue
                    expected = collection_table_allowlist.get(table_raw)
                    if expected is None:
                        violations.append(
                            Violation(
                                category=_MISSING_COLLECTION_CATEGORY,
                                file=source,
                                line=lineno,
                                symbol=table_raw,
                                reason=(
                                    f"migration creates table '{table_raw}' "
                                    "with no mapping in "
                                    "collection_table_allowlist; either "
                                    "add a Collection + allowlist entry "
                                    "or add to migration_table_allowlist "
                                    "with rationale"
                                ),
                            )
                        )
                        continue
                    if not _class_is_collection(
                        expected,
                        base_collection_names,
                        base_graph,
                    ):
                        violations.append(
                            Violation(
                                category=_MISSING_COLLECTION_CATEGORY,
                                file=source,
                                line=lineno,
                                symbol=table_raw,
                                reason=(
                                    f"table '{table_raw}' expects "
                                    f"{expected} but no class by that "
                                    "name transitively subclasses any of "
                                    f"{sorted(base_collection_names)!r}; "
                                    "either declare the Collection or "
                                    "extend the inheritance chain"
                                ),
                            )
                        )
    return violations


def _class_is_collection(
    class_name: str,
    base_collection_names: frozenset[str],
    base_graph: ClassBaseGraph,
) -> bool:
    """true iff ``class_name`` exists in ``base_graph`` and reaches a target.

    a class that is not present in the graph at all (i.e., not declared
    in any scanned src root) is treated as a non-Collection — the
    migration's mapping points at a class that simply does not exist.
    a class that exists but does not transitively reach any
    ``base_collection_names`` is also a non-Collection.

    :param class_name: name to query
    :ptype class_name: str
    :param base_collection_names: terminal class names that satisfy
        "this is a Collection"
    :ptype base_collection_names: frozenset[str]
    :param base_graph: cross-package class-base graph
    :ptype base_graph: ClassBaseGraph
    :return: whether the class is a Collection
    :rtype: bool
    """
    if class_name not in base_graph:
        return False
    return transitively_subclasses_any(
        class_name,
        base_collection_names,
        base_graph,
    )


def _extract_sql_strings(tree: ast.Module) -> list[tuple[int, str]]:
    """extract every non-docstring SQL-shaped string literal from ``tree``.

    walks :class:`ast.Constant` and :class:`ast.JoinedStr` nodes,
    skipping module / class / function docstrings (the first
    :class:`ast.Expr` wrapping a string constant in each body). returns
    ``(lineno, text)`` pairs in source order so the report's line
    numbers point at the SQL literal, not the migration's docstring.

    :param tree: parsed ast module
    :ptype tree: ast.Module
    :return: ``(lineno, string_value)`` pairs
    :rtype: list[tuple[int, str]]
    """
    result: list[tuple[int, str]] = []
    docstring_nodes: set[int] = set()
    for container in ast.walk(tree):
        if isinstance(
            container,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            body = getattr(container, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstring_nodes:
                continue
            result.append((node.lineno, node.value))
        elif isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for piece in node.values:
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                    parts.append(piece.value)
            if parts:
                result.append((node.lineno, "".join(parts)))
    return result
