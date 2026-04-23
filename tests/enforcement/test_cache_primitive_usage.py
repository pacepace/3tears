"""enforcement test for the 3tears-Collection-as-the-primitive contract.

the platform's state-management primitive is
:class:`~threetears.core.collections.base.BaseCollection`: every
stateful data surface — persistent, cached, ephemeral, secret — is
expected to be a ``BaseCollection`` with pluggable L1/L2/L3 tiers.
rolling a bespoke SQLite cache directly on top of
:class:`~threetears.core.cache.sqlite.SQLiteBackend` bypasses the
registry, the cross-pod invalidation stream, and the L2 KV tier. this
test surfaces four violation classes that indicate the pattern has
been re-introduced:

- **no bespoke ``SQLiteBackend(...)`` construction**. the only
  sanctioned sites are small named factories (hub + agent pod). any
  other site is a bespoke wrapper in disguise.
- **no bespoke cache wrapper classes**. a class that stores a
  ``SQLiteBackend`` as an instance attribute AND exposes a public
  key/value api (``get`` / ``put`` / ``set`` / ``delete`` / ``upsert``)
  AND does not subclass ``BaseCollection`` is a wrapper; the platform
  already has the wrapper (Collections).
- **no direct pool access to tables with Collections**. ``pool.fetch``
  / ``fetchrow`` / ``fetchval`` / ``execute`` with SQL referencing a
  table that has a Collection bypasses every cache tier. the Collection
  IS the access path.
- **every migration-defined table has a Collection**. if a migration
  creates a table, there must be a ``class XxxCollection(BaseCollection[
  ...])`` somewhere in ``src/``. bookkeeping tables
  (``_schema_migrations``) are allowlisted with rationale.

run mode is controlled by the ``CACHE_ENFORCEMENT_MODE`` env var:

- ``strict`` (default, flipped in 8.5e) — raises ``pytest.fail`` on
  any violation. any new bespoke ``SQLiteBackend`` wrapper or
  direct-pool bypass fails CI.
- ``report`` — prints every violation with ``file:line:reason``;
  asserts nothing. retained as an opt-in debugging aid for bulk
  refactors; production pipelines never set this.

exemptions live in ``tests/enforcement/_cache_exemptions.txt``. each
entry requires a preceding ``# rationale: <specific reason>`` line; the
parser rejects entries without one. blanket exemptions without a
rationale are banned outright.

:see: docs/namespace-task-01-everything-is-a-namespace.md -- Phase
    8.5a / 8.5b / 8.5c / 8.5d / 8.5e lifecycle.

CANONICAL SOURCE: this file is maintained in the ``14-eng-ai-bot`` repo
under ``tests/enforcement/test_cache_primitive_usage.py`` and vendored
verbatim into the sibling repos (``3tears``, ``14-eng-ai-bot-agents``,
``14-eng-ai-bot-agent-admin``). updates MUST land in every copy in the
same commit. the allowlist constants
(:data:`_ALLOWED_SQLITE_CONSTRUCTION_SITES`,
:data:`_COLLECTION_TABLE_ALLOWLIST`,
:data:`_MIGRATION_TABLE_ALLOWLIST`) are scoped to this repo's valid
sites — each vendored copy substitutes its own.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

import pytest

# ------------------------------------------------------------------------
# config
# ------------------------------------------------------------------------

MODE_REPORT = "report"
MODE_STRICT = "strict"
# strict is the current production default (flipped in namespace-task-01
# phase 8.5e). report mode is retained as an opt-in debugging aid for
# bulk refactors; production and CI pipelines run strict.
_DEFAULT_MODE = MODE_STRICT

_SKIP_DIRS = frozenset({
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".git",
    "dist",
    "build",
    ".eggs",
    "_build",
})


# allowed construction sites for SQLiteBackend (paths relative to repo
# root). every other site is a violation. each entry carries a short
# inline rationale documenting why the site is sanctioned.
_ALLOWED_SQLITE_CONSTRUCTION_SITES: frozenset[str] = frozenset({
    # 3tears library has no sanctioned src/ construction sites; the
    # SQLiteBackend class itself lives here but ``class`` definitions
    # are not ``Call`` nodes, so this allowlist applies only to
    # ``SQLiteBackend(...)`` invocations. tests and test fixtures are
    # permitted to construct backends directly; they cover the
    # Collection-behavior tests that need a clean SQLite tier.
})


# tables defined in migrations that legitimately lack Collections.
# bookkeeping tables, extension-created tables, etc. keep this list
# tiny and document every entry.
_MIGRATION_TABLE_ALLOWLIST: frozenset[str] = frozenset({
    # internal migration runner state — not a business entity
    "_schema_migrations",
    # checkpoints / checkpoint_writes are accessed via
    # ``threetears.langgraph.ThreeTierCheckpointSaver`` and
    # ``threetears.langgraph.ProxyCheckpointSaver`` (LangGraph
    # ``BaseCheckpointSaver`` contract). the LangGraph interface is
    # sync-and-async and mandates shape details that cannot be
    # expressed through ``BaseCollection``. 3tears wraps the three-tier
    # backend under the LangGraph contract so the platform gets L1/L2/L3
    # while satisfying LangGraph.
    "checkpoints",
    "checkpoint_writes",
    # conversation_memory_refs: composite primary key (conversation_id,
    # item_id) which BaseCollection's single-PK contract cannot model
    # without a schema migration adding a synthetic surrogate id or
    # extending BaseCollection to support composite keys. Tracked as
    # follow-on work under namespace-task-01 phase 8.5l; the wrapper
    # class (MemoryLedger) that owns reads/writes is exempted in the
    # matching _cache_exemptions.txt entry with the same rationale.
    "conversation_memory_refs",
})


# logical mapping from migration table name to expected Collection
# class name (simple by-name match). used by test 3 to know which
# tables to enforce and by test 4 to verify Collections exist. kept as
# an inline dict so additions stay visible on code review.
_COLLECTION_TABLE_ALLOWLIST: dict[str, str] = {
    # 3tears package-owned tables (every table in this dict is
    # created by a migration under ``packages/*/migrations/`` and has
    # a Collection somewhere under ``packages/*/src/``)
    "conversations": "ConversationsCollection",
    "context_items": "ContextItemCollection",
    "memories": "MemoriesCollection",
    "media": "MediaCollection",
    "media_content": "MediaContentCollection",
    "memory_chunks": "MemoryChunkCollection",
    "workspaces": "WorkspaceCollection",
    "workspace_files": "WorkspaceFileCollection",
    "workspace_file_versions": "WorkspaceFileVersionCollection",
}


# cache-like method names that flag a class as a "wrapper" when
# combined with a SQLiteBackend field. kept liberal so any new
# bespoke wrapper hits the walker.
_CACHE_METHOD_NAMES: frozenset[str] = frozenset({
    "get", "put", "set", "delete", "upsert", "remove",
})


# public method names on :class:`~threetears.core.cache.sqlite.SQLiteBackend`
# that read/write row state. a class that stores a SQLiteBackend AND
# calls one of these methods on it from any method body is using
# SQLiteBackend's data api directly — regardless of what public verbs
# it exposes. this is the fingerprint of a bespoke wrapper dressed in
# domain-specific method names (``revoke`` / ``record_failure`` /
# ``dispatch`` / ``consume_token``) that the generic-verb walker in
# :data:`_CACHE_METHOD_NAMES` misses. namespace-task-01 phase 8.5f
# added this domain-verb detection path.
_SQLITE_DATA_API_METHODS: frozenset[str] = frozenset({
    "upsert",
    "select_by_id",
    "delete_by_id",
    "execute_query",
})


# ------------------------------------------------------------------------
# public dataclasses
# ------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheViolation:
    """one detected cache-enforcement violation.

    :ivar test_name: short id (``sqlite_construction``, ``wrapper_class``,
        ``pool_access``, ``missing_collection``)
    :ivar file: absolute source path
    :ivar line: 1-based line number
    :ivar symbol: offending symbol (class / function / table name)
    :ivar reason: human-readable explanation
    """

    test_name: str
    file: Path
    line: int
    symbol: str
    reason: str

    def format(self, repo_root: Path) -> str:
        """format a violation as ``[test] rel:line:symbol  -- reason``.

        :param repo_root: repo root for relative-path rendering
        :ptype repo_root: Path
        :return: single-line rendering
        :rtype: str
        """
        try:
            rel = self.file.relative_to(repo_root)
        except ValueError:
            rel = self.file
        result = f"[{self.test_name}] {rel}:{self.line}:{self.symbol}  -- {self.reason}"
        return result


@dataclass(frozen=True)
class CacheExemption:
    """one exemption entry parsed from ``_cache_exemptions.txt``.

    :ivar file: source-relative path (as written)
    :ivar line: line number within that file
    :ivar symbol: offending symbol as reported by the walker
    :ivar rationale: justification from the preceding ``# rationale:`` line
    """

    file: str
    line: int
    symbol: str
    rationale: str


# ------------------------------------------------------------------------
# exemptions parser
# ------------------------------------------------------------------------

_ENTRY_RE = re.compile(
    r"^(?P<file>[^\s:][^:]*):(?P<line>[^:]+):(?P<symbol>[A-Za-z_][A-Za-z_0-9]*)\s*$"
)


def parse_cache_exemptions(path: Path) -> list[CacheExemption]:
    """parse a cache-exemptions file.

    every entry line must be preceded by a line matching
    ``# rationale: <non-empty reason>``. blank lines are skipped.
    non-rationale comment lines are ignored.

    :param path: path to exemptions file
    :ptype path: Path
    :return: parsed entries in file order
    :rtype: list[CacheExemption]
    :raises ValueError: malformed file (missing rationale, bad entry
        shape, non-integer line number)
    """
    entries: list[CacheExemption] = []
    if not path.exists():
        return entries
    text = path.read_text(encoding="utf-8")
    pending: str | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            stripped = line.lstrip("#").strip()
            if stripped.lower().startswith("rationale:"):
                rationale = stripped[len("rationale:"):].strip()
                if not rationale:
                    raise ValueError(
                        f"{path}:{lineno}: '# rationale:' must be followed "
                        f"by a non-empty reason"
                    )
                pending = rationale
            continue
        match = _ENTRY_RE.match(line)
        if match is None:
            raise ValueError(
                f"{path}:{lineno}: malformed entry; expected "
                f"'file:line:symbol' triple, got {line!r}"
            )
        if pending is None:
            raise ValueError(
                f"{path}:{lineno}: entry has no preceding '# rationale: ...' line"
            )
        line_field = match.group("line")
        try:
            line_int = int(line_field)
        except ValueError as exc:
            raise ValueError(
                f"{path}:{lineno}: line number must be an integer, got "
                f"{line_field!r}"
            ) from exc
        entries.append(
            CacheExemption(
                file=match.group("file"),
                line=line_int,
                symbol=match.group("symbol"),
                rationale=pending,
            )
        )
        pending = None
    return entries


# ------------------------------------------------------------------------
# shared helpers
# ------------------------------------------------------------------------


def find_src_roots(repo_root: Path) -> tuple[Path, ...]:
    """locate all ``src/`` directories under ``repo_root``.

    :param repo_root: repo root path to search
    :ptype repo_root: Path
    :return: tuple of discovered src roots, sorted for stable order
    :rtype: tuple[Path, ...]
    """
    candidates: set[Path] = set()
    for candidate in repo_root.rglob("src"):
        if not candidate.is_dir():
            continue
        if any(part in _SKIP_DIRS for part in candidate.parts):
            continue
        candidates.add(candidate)
    return tuple(sorted(candidates))


def _iter_python_files(root: Path) -> Iterable[Path]:
    """yield ``.py`` files under root, skipping cache / venv directories.

    :param root: directory to walk
    :ptype root: Path
    :return: iterator of python files
    :rtype: Iterable[Path]
    """
    for path in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        yield path


def _parse_file(path: Path) -> ast.Module | None:
    """parse a python file, returning None on syntax error.

    :param path: file to parse
    :ptype path: Path
    :return: parsed module AST or None
    :rtype: ast.Module | None
    """
    result: ast.Module | None
    try:
        source = path.read_text(encoding="utf-8")
        result = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        result = None
    return result


def _relpath_str(path: Path, repo_root: Path) -> str:
    """produce a forward-slash relative path for allowlist matching.

    :param path: absolute path
    :ptype path: Path
    :param repo_root: repo root
    :ptype repo_root: Path
    :return: ``src/pkg/mod.py``-style relative path
    :rtype: str
    """
    try:
        result = str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        result = str(path)
    return result.replace("\\", "/")


# ------------------------------------------------------------------------
# walker 1 — SQLiteBackend construction outside sanctioned factories
# ------------------------------------------------------------------------


def find_sqlite_constructions(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    allowed_sites: frozenset[str],
) -> list[CacheViolation]:
    """flag every ``SQLiteBackend(...)`` call outside the allowed sites.

    :param src_roots: all src roots in the repo
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root for allowlist matching
    :ptype repo_root: Path
    :param allowed_sites: relative paths permitted to construct
        ``SQLiteBackend``
    :ptype allowed_sites: frozenset[str]
    :return: list of violations
    :rtype: list[CacheViolation]
    """
    violations: list[CacheViolation] = []
    for root in src_roots:
        for source in _iter_python_files(root):
            rel = _relpath_str(source, repo_root)
            if rel in allowed_sites:
                continue
            tree = _parse_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name != "SQLiteBackend":
                    continue
                violations.append(
                    CacheViolation(
                        test_name="sqlite_construction",
                        file=source,
                        line=node.lineno,
                        symbol="SQLiteBackend",
                        reason=(
                            "SQLiteBackend constructed outside the sanctioned "
                            f"factories; add {rel} to the allowlist (with "
                            "rationale) or route through a Collection instead"
                        ),
                    )
                )
    return violations


# ------------------------------------------------------------------------
# walker 2 — bespoke cache wrapper classes
# ------------------------------------------------------------------------


def _annotation_names_sqlite_backend(ann: ast.AST | None) -> bool:
    """true iff the annotation textually names ``SQLiteBackend``.

    matches:

    - ``SQLiteBackend`` — bare name
    - ``some.module.SQLiteBackend`` — attribute ending in the name
    - ``SQLiteBackend | None`` / ``Optional[SQLiteBackend]`` —
      union / subscript shapes; the walker recurses into
      :class:`ast.BinOp`, :class:`ast.Subscript`, and
      :class:`ast.Tuple` so every optional / parameterised variant
      surfaces.

    :param ann: annotation AST node, or ``None``
    :ptype ann: ast.AST | None
    :return: whether the annotation textually names ``SQLiteBackend``
    :rtype: bool
    """
    result = False
    if ann is None:
        pass
    elif isinstance(ann, ast.Name) and ann.id == "SQLiteBackend":
        result = True
    elif isinstance(ann, ast.Attribute) and ann.attr == "SQLiteBackend":
        result = True
    elif isinstance(ann, ast.BinOp):
        result = (
            _annotation_names_sqlite_backend(ann.left)
            or _annotation_names_sqlite_backend(ann.right)
        )
    elif isinstance(ann, ast.Subscript):
        inner = ann.slice
        result = (
            _annotation_names_sqlite_backend(ann.value)
            or _annotation_names_sqlite_backend(inner)
        )
    elif isinstance(ann, ast.Tuple):
        result = any(
            _annotation_names_sqlite_backend(elt) for elt in ann.elts
        )
    return result


def _init_param_names_of_sqlite_type(cls: ast.ClassDef) -> frozenset[str]:
    """return every ``__init__`` parameter whose annotation is SQLiteBackend.

    captures the constructor-injected shape bespoke wrappers use
    (``def __init__(self, l1_backend: SQLiteBackend | None = None)``)
    which the earlier walker missed — it only matched a field that
    was either annotated on the class body with ``SQLiteBackend`` or
    that was the RHS of a ``= SQLiteBackend(...)`` call. the
    constructor-injection shape stores the already-constructed
    backend on ``self.<attr>`` without ever constructing a fresh one
    inside the class, so the old rules never fire.

    :param cls: class definition
    :ptype cls: ast.ClassDef
    :return: set of parameter names typed as ``SQLiteBackend``
    :rtype: frozenset[str]
    """
    names: set[str] = set()
    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name != "__init__":
            continue
        args = item.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if _annotation_names_sqlite_backend(arg.annotation):
                names.add(arg.arg)
    return frozenset(names)


def _init_stores_param_on_self(
    cls: ast.ClassDef, param_names: frozenset[str],
) -> bool:
    """true iff ``__init__`` assigns one of ``param_names`` to ``self.<x>``.

    matches ``self._l1 = l1_backend`` / ``self.store = backend`` /
    ``self.l1: SQLiteBackend = l1_backend`` patterns where the RHS
    is a plain :class:`ast.Name` whose id is in ``param_names``.

    :param cls: class definition
    :ptype cls: ast.ClassDef
    :param param_names: constructor parameter names whose type is
        ``SQLiteBackend``
    :ptype param_names: frozenset[str]
    :return: whether the constructor stores one of those parameters
        onto a ``self`` attribute
    :rtype: bool
    """
    result = False
    if not param_names:
        return result
    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name != "__init__":
            continue
        for sub in ast.walk(item):
            targets: list[ast.AST] = []
            rhs: ast.AST | None = None
            if isinstance(sub, ast.Assign):
                targets = list(sub.targets)
                rhs = sub.value
            elif isinstance(sub, ast.AnnAssign):
                targets = [sub.target]
                rhs = sub.value
            if rhs is None:
                continue
            if not (isinstance(rhs, ast.Name) and rhs.id in param_names):
                continue
            for tgt in targets:
                if (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                ):
                    result = True
                    break
            if result:
                break
        if result:
            break
    return result


def _class_sqlite_attr_names(cls: ast.ClassDef) -> frozenset[str]:
    """return every ``self.<attr>`` name on ``cls`` that holds a SQLiteBackend.

    three sources contribute to the set:

    - annotated class-body attributes (``self._l1: SQLiteBackend``)
    - init-body annotated assignments + inline constructions
      (``self._l1: SQLiteBackend = ...`` / ``self._l1 = SQLiteBackend(...)``)
    - constructor-injection: ``__init__`` parameter typed
      ``SQLiteBackend`` that is stored on a ``self.<attr>`` target

    the resulting set is consumed by
    :func:`_class_calls_sqlite_data_api` to check whether the class
    reaches into the backend's row-level api from inside any method
    body. the combination (store a SQLiteBackend + call its data api
    + not a ``BaseCollection``) is the fingerprint of a domain-verb
    wrapper.

    :param cls: class definition node
    :ptype cls: ast.ClassDef
    :return: frozen set of attribute names on ``self`` that hold a
        SQLiteBackend; empty when the class stores no backend
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
                rhs: ast.AST | None = sub.value
                rhs_is_sqlite_call = False
                if isinstance(rhs, ast.Call):
                    func = rhs.func
                    if isinstance(func, ast.Name) and func.id == "SQLiteBackend":
                        rhs_is_sqlite_call = True
                    if isinstance(func, ast.Attribute) and func.attr == "SQLiteBackend":
                        rhs_is_sqlite_call = True
                rhs_is_sqlite_param = (
                    isinstance(rhs, ast.Name) and rhs.id in sqlite_params
                )
                if not (rhs_is_sqlite_call or rhs_is_sqlite_param):
                    continue
                for tgt in sub.targets:
                    if (
                        isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                    ):
                        names.add(tgt.attr)
    return frozenset(names)


def _class_calls_sqlite_data_api(
    cls: ast.ClassDef, sqlite_attrs: frozenset[str],
) -> list[str]:
    """return every SQLiteBackend data-api method called on a stored backend.

    walks every method body and returns the
    :data:`_SQLITE_DATA_API_METHODS` entries that the class calls on a
    ``self.<attr>`` receiver where ``<attr>`` is a known SQLiteBackend
    field. the walker also accepts assignments that rebind the
    backend onto a local (``l1 = self._l1; l1.upsert(...)``) so
    wrappers cannot escape detection by introducing a local alias.

    :param cls: class definition node
    :ptype cls: ast.ClassDef
    :param sqlite_attrs: names of ``self`` attributes holding a
        SQLiteBackend (from :func:`_class_sqlite_attr_names`)
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
            receiver_is_local_bind = (
                isinstance(receiver, ast.Name)
                and receiver.id in local_binds
            )
            if receiver_is_self_attr or receiver_is_local_bind:
                hits.add(func.attr)
    return sorted(hits)


def _class_has_sqlite_field(cls: ast.ClassDef) -> bool:
    """true iff the class body annotates or assigns an SQLiteBackend field.

    matches three shapes:

    - annotated attribute (``self._l1: SQLiteBackend``)
    - init assignment with inline construction
      (``self._l1 = SQLiteBackend(...)``)
    - constructor-injection pattern: ``__init__`` takes a parameter
      annotated ``SQLiteBackend`` (possibly ``SQLiteBackend | None``)
      and stores it on ``self`` (``def __init__(self, l1_backend:
      SQLiteBackend | None = None): self._l1 = l1_backend``). this
      final shape is what bespoke hub wrappers
      (:class:`UnifiedAccessCache`, :class:`LoginLockout`,
      :class:`AdminRateLimiter` et al.) use; the earlier walker
      missed it because no ``SQLiteBackend(...)`` call appeared
      inside the class body. namespace-task-01 phase 8.5c sharpened
      the walker to catch it.

    :param cls: class definition node
    :ptype cls: ast.ClassDef
    :return: whether the class stores a SQLiteBackend
    :rtype: bool
    """
    # annotated attributes on class body
    for item in cls.body:
        if isinstance(item, ast.AnnAssign):
            if _annotation_names_sqlite_backend(item.annotation):
                return True
    # init-body annotations + inline constructions
    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name != "__init__":
            continue
        for sub in ast.walk(item):
            if isinstance(sub, ast.AnnAssign):
                if _annotation_names_sqlite_backend(sub.annotation):
                    return True
            if isinstance(sub, ast.Assign):
                value = sub.value
                if isinstance(value, ast.Call):
                    func = value.func
                    if isinstance(func, ast.Name) and func.id == "SQLiteBackend":
                        return True
                    if isinstance(func, ast.Attribute) and func.attr == "SQLiteBackend":
                        return True
    # constructor-injection shape: parameter typed SQLiteBackend and
    # stored on self in the body
    param_names = _init_param_names_of_sqlite_type(cls)
    if _init_stores_param_on_self(cls, param_names):
        return True
    return False


def _class_has_cache_methods(cls: ast.ClassDef) -> list[str]:
    """return every public method whose name matches the cache vocabulary.

    :param cls: class definition
    :ptype cls: ast.ClassDef
    :return: list of matching method names
    :rtype: list[str]
    """
    result: list[str] = []
    for item in cls.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = item.name
            if name.startswith("_"):
                continue
            if name in _CACHE_METHOD_NAMES:
                result.append(name)
    return result


def _class_subclasses_base_collection(cls: ast.ClassDef) -> bool:
    """true iff the class explicitly names ``BaseCollection`` as a base.

    textual match on the last segment of each base expression, so
    ``class X(BaseCollection[T])`` / ``class X(foo.BaseCollection)`` /
    ``class X(BaseCollection)`` all qualify.

    :param cls: class definition
    :ptype cls: ast.ClassDef
    :return: whether the class extends BaseCollection
    :rtype: bool
    """
    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id == "BaseCollection":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseCollection":
            return True
        if isinstance(base, ast.Subscript):
            inner = base.value
            if isinstance(inner, ast.Name) and inner.id == "BaseCollection":
                return True
            if isinstance(inner, ast.Attribute) and inner.attr == "BaseCollection":
                return True
    return False


def find_wrapper_classes(
    src_roots: tuple[Path, ...],
    repo_root: Path,
) -> list[CacheViolation]:
    """flag every bespoke cache wrapper class (SQLiteBackend + cache api).

    :param src_roots: all src roots in the repo
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root (unused; retained for signature symmetry)
    :ptype repo_root: Path
    :return: list of violations
    :rtype: list[CacheViolation]
    """
    violations: list[CacheViolation] = []
    _ = repo_root
    for root in src_roots:
        for source in _iter_python_files(root):
            tree = _parse_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if _class_subclasses_base_collection(node):
                    continue
                if not _class_has_sqlite_field(node):
                    continue
                methods = _class_has_cache_methods(node)
                # domain-verb detection (namespace-task-01 phase 8.5f).
                sqlite_attrs = _class_sqlite_attr_names(node)
                data_api_calls = _class_calls_sqlite_data_api(
                    node, sqlite_attrs,
                )
                if not methods and not data_api_calls:
                    continue
                if methods:
                    reason = (
                        f"class '{node.name}' wraps SQLiteBackend and exposes "
                        f"cache api {sorted(methods)!r} without extending "
                        "BaseCollection; convert to a Collection subclass"
                    )
                else:
                    reason = (
                        f"class '{node.name}' wraps SQLiteBackend and calls "
                        f"its data api {data_api_calls!r} from method bodies "
                        "without extending BaseCollection (domain-verb "
                        "wrapper); convert to a Collection subclass"
                    )
                violations.append(
                    CacheViolation(
                        test_name="wrapper_class",
                        file=source,
                        line=node.lineno,
                        symbol=node.name,
                        reason=reason,
                    )
                )
    return violations


# ------------------------------------------------------------------------
# walker 3 — direct pool access to Collection-backed tables
# ------------------------------------------------------------------------

# pool method names whose SQL text is inspected for violations.
_POOL_METHOD_NAMES: frozenset[str] = frozenset({
    "fetch", "fetchrow", "fetchval", "execute", "executemany",
})

# first-argument SQL-text regex: matches ``FROM table`` / ``INTO table`` /
# ``UPDATE table`` / ``JOIN table`` / ``DELETE FROM table``. case-insensitive.
_SQL_TARGET_RE = re.compile(
    r"\b(?:from|into|update|join|table)\s+([A-Za-z_][A-Za-z_0-9]*)",
    re.IGNORECASE,
)


def _extract_sql_tables(call_node: ast.Call) -> set[str]:
    """extract every table name referenced by the call's SQL-text arg.

    inspects the first positional argument when it is a string literal
    (including joined constants). returns the empty set when the
    argument is dynamic.

    :param call_node: AST call node (``pool.fetch(...)`` etc.)
    :ptype call_node: ast.Call
    :return: set of table names mentioned in the SQL text
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


def _class_containing(
    node: ast.AST, module: ast.Module,
) -> ast.ClassDef | None:
    """return the innermost enclosing class for a node, or None.

    walks the module's parents map; since ast does not record parents
    by default we build a cheap parent map per module call.

    :param node: target node
    :ptype node: ast.AST
    :param module: enclosing module
    :ptype module: ast.Module
    :return: enclosing ClassDef or None
    :rtype: ast.ClassDef | None
    """
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(module):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    current: ast.AST | None = node
    while current is not None:
        current = parents.get(id(current))
        if isinstance(current, ast.ClassDef):
            return current
    return None


def find_direct_pool_access(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    table_allowlist: dict[str, str],
) -> list[CacheViolation]:
    """flag ``pool.<method>(<sql>)`` that touches a table with a Collection.

    allowed sites:

    - inside the Collection class itself (matches on class name in
      ``table_allowlist``)
    - any call preceded by a ``# cache-bypass: <reason>`` comment on the
      same or prior line (inspected via the raw source).

    :param src_roots: all src roots
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root (unused; retained for signature symmetry)
    :ptype repo_root: Path
    :param table_allowlist: ``{table_name: collection_class_name}``
    :ptype table_allowlist: dict[str, str]
    :return: list of violations
    :rtype: list[CacheViolation]
    """
    violations: list[CacheViolation] = []
    _ = repo_root
    allowed_tables = set(table_allowlist.keys())
    collection_names = set(table_allowlist.values())
    for root in src_roots:
        for source in _iter_python_files(root):
            tree = _parse_file(source)
            if tree is None:
                continue
            raw_lines = source.read_text(encoding="utf-8").splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                if func.attr not in _POOL_METHOD_NAMES:
                    continue
                # infer the called object's name (best-effort)
                obj = func.value
                obj_name = ""
                if isinstance(obj, ast.Name):
                    obj_name = obj.id
                elif isinstance(obj, ast.Attribute):
                    obj_name = obj.attr
                # filter: only care about pool-like receivers. avoids
                # flagging ``str.fetch`` / ``dict.get`` / etc. shapes.
                if "pool" not in obj_name.lower() and "store" not in obj_name.lower():
                    continue
                tables = _extract_sql_tables(node)
                hits = tables & allowed_tables
                if not hits:
                    continue
                # allow calls preceded by ``# cache-bypass: ...`` comment
                bypass = False
                for prev_line in range(
                    max(0, node.lineno - 3), node.lineno
                ):
                    if prev_line >= len(raw_lines):
                        continue
                    if "cache-bypass:" in raw_lines[prev_line]:
                        bypass = True
                        break
                if bypass:
                    continue
                # allow: call originates inside the collection class itself
                enclosing = _class_containing(node, tree)
                if enclosing is not None and enclosing.name in collection_names:
                    continue
                for table in sorted(hits):
                    violations.append(
                        CacheViolation(
                            test_name="pool_access",
                            file=source,
                            line=node.lineno,
                            symbol=table,
                            reason=(
                                f"direct pool.{func.attr}() on table '{table}' "
                                f"which has a Collection ({table_allowlist[table]}); "
                                "go through the Collection or add a "
                                "'# cache-bypass: <reason>' comment"
                            ),
                        )
                    )
    return violations


# ------------------------------------------------------------------------
# walker 4 — migration-defined tables without Collections
# ------------------------------------------------------------------------

# CREATE TABLE regex — requires at least one space after EXISTS when
# that clause is present, and the captured name must be an identifier
# (not the keyword ``IF``, ``NOT``, ``EXISTS``). the previous regex
# collapsed ``IF\s+NOT\s+EXISTS\s+`` into an optional group that could
# match zero characters when the text lacked the trailing whitespace
# (docstring/backtick breaks), which let ``IF`` itself be captured as
# the table name. the rewritten pattern uses two alternatives so the
# captured identifier is always the table token that follows.
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?!IF\b|NOT\b|EXISTS\b)"
    r"([A-Za-z_][A-Za-z_0-9\.]*)",
    re.IGNORECASE,
)

# only scan files under a ``migrations/`` directory whose filename
# matches the canonical ``v<digits>_<suffix>.py`` migration template
# (``template.py`` and ``drift.py`` live under ``migrations/`` but are
# helper modules, not migrations). the walker previously scanned every
# ``.py`` file that happened to sit under a ``migrations/`` directory,
# which surfaced every docstring mentioning ``CREATE TABLE IF NOT
# EXISTS`` as a phantom migration.
_MIGRATION_FILENAME_RE = re.compile(r"^v\d+_.+\.py$")


def _extract_sql_strings(tree: ast.Module) -> list[tuple[int, str]]:
    """extract every SQL-shaped string literal from the module AST.

    walks :class:`ast.Constant` and :class:`ast.JoinedStr` nodes,
    ignoring module/class/function docstrings (the first :class:`ast
    .Expr` wrapping a string constant in a body). returns ``(lineno,
    text)`` pairs for downstream regex matching.

    using the AST rather than raw text ensures ``CREATE TABLE IF NOT
    EXISTS`` fragments embedded inside module/class docstrings
    (``drift.py``, ``template.py``) are never considered; only real
    SQL-string assignments contribute.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: ``(lineno, string_value)`` tuples in source order
    :rtype: list[tuple[int, str]]
    """
    result: list[tuple[int, str]] = []
    docstring_nodes: set[int] = set()
    for container in ast.walk(tree):
        if isinstance(container, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(container, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
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


def find_missing_collections(
    src_roots: tuple[Path, ...],
    repo_root: Path,
    table_allowlist: dict[str, str],
    migration_allowlist: frozenset[str],
) -> list[CacheViolation]:
    """flag tables from CREATE TABLE migrations that have no Collection class.

    strategy:

    1. walk every ``migrations/`` directory under each src root and
       match files whose basename follows the
       ``v<digits>_<suffix>.py`` migration naming convention —
       helper modules (``template.py``, ``drift.py``, ``__init__.py``)
       are skipped.
    2. extract CREATE TABLE statements by scanning only the SQL-shaped
       string literals in the module AST (docstrings excluded) — no
       migration execution required and no false positives from
       backtick-quoted examples inside module documentation.
    3. for every extracted table, check ``table_allowlist`` for the
       expected Collection class name, then confirm a ``class X(
       BaseCollection...)`` with that name exists somewhere under
       ``src/``.

    :param src_roots: all src roots
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root for relative-path rendering
    :ptype repo_root: Path
    :param table_allowlist: ``{table_name: collection_class_name}``
    :ptype table_allowlist: dict[str, str]
    :param migration_allowlist: set of migration-defined tables that
        legitimately lack Collections (``_schema_migrations``, etc.)
    :ptype migration_allowlist: frozenset[str]
    :return: list of violations
    :rtype: list[CacheViolation]
    """
    _ = repo_root
    # gather every declared Collection class name across src
    declared_collections: set[str] = set()
    for root in src_roots:
        for source in _iter_python_files(root):
            tree = _parse_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if _class_subclasses_base_collection(node):
                    declared_collections.add(node.name)

    # walk every migration file and extract CREATE TABLE targets
    violations: list[CacheViolation] = []
    seen: set[str] = set()
    for root in src_roots:
        for source in _iter_python_files(root):
            if "migrations" not in source.parts:
                continue
            if not _MIGRATION_FILENAME_RE.match(source.name):
                continue
            tree = _parse_file(source)
            if tree is None:
                continue
            for lineno, text in _extract_sql_strings(tree):
                for match in _CREATE_TABLE_RE.finditer(text):
                    table_raw = match.group(1).lower()
                    # strip schema prefix (``platform.customers`` ->
                    # ``customers``)
                    if "." in table_raw:
                        table_raw = table_raw.rsplit(".", 1)[1]
                    if table_raw in seen:
                        continue
                    seen.add(table_raw)
                    if table_raw in migration_allowlist:
                        continue
                    expected = table_allowlist.get(table_raw)
                    if expected is None:
                        violations.append(
                            CacheViolation(
                                test_name="missing_collection",
                                file=source,
                                line=lineno,
                                symbol=table_raw,
                                reason=(
                                    f"migration creates table '{table_raw}' with no "
                                    "mapping in _COLLECTION_TABLE_ALLOWLIST; either "
                                    "add a Collection + allowlist entry or add to "
                                    "_MIGRATION_TABLE_ALLOWLIST with rationale"
                                ),
                            )
                        )
                        continue
                    if expected not in declared_collections:
                        violations.append(
                            CacheViolation(
                                test_name="missing_collection",
                                file=source,
                                line=lineno,
                                symbol=table_raw,
                                reason=(
                                    f"table '{table_raw}' expects "
                                    f"{expected} but no class by that name "
                                    "subclasses BaseCollection anywhere in src/"
                                ),
                            )
                        )
    return violations


# ------------------------------------------------------------------------
# exemption application + mode resolution
# ------------------------------------------------------------------------


def apply_cache_exemptions(
    violations: list[CacheViolation],
    exemptions: list[CacheExemption],
    repo_root: Path,
) -> list[CacheViolation]:
    """filter violations against exemption entries (file, line, symbol).

    :param violations: raw walker output
    :ptype violations: list[CacheViolation]
    :param exemptions: parsed exemptions
    :ptype exemptions: list[CacheExemption]
    :param repo_root: repo root for relative-path comparison
    :ptype repo_root: Path
    :return: violations that did not match any exemption
    :rtype: list[CacheViolation]
    """
    result: list[CacheViolation] = []
    keys = {(e.file, e.line, e.symbol) for e in exemptions}
    for violation in violations:
        try:
            rel = str(violation.file.relative_to(repo_root))
        except ValueError:
            rel = str(violation.file)
        rel = rel.replace("\\", "/")
        if (rel, violation.line, violation.symbol) in keys:
            continue
        result.append(violation)
    return result


def resolve_mode() -> str:
    """read and validate the ``CACHE_ENFORCEMENT_MODE`` env var.

    :return: normalized mode string
    :rtype: str
    :raises ValueError: if the env var holds an unknown value
    """
    raw = os.environ.get("CACHE_ENFORCEMENT_MODE", _DEFAULT_MODE).strip().lower()
    if raw not in (MODE_REPORT, MODE_STRICT):
        raise ValueError(
            f"CACHE_ENFORCEMENT_MODE must be '{MODE_REPORT}' or "
            f"'{MODE_STRICT}', got {raw!r}"
        )
    return raw


# ------------------------------------------------------------------------
# repo anchoring
# ------------------------------------------------------------------------


def _find_repo_root(start: Path) -> Path:
    """walk upward from ``start`` to the nearest ``pyproject.toml``.

    :param start: path to start from
    :ptype start: Path
    :return: repo root
    :rtype: Path
    :raises RuntimeError: no pyproject.toml ancestor found
    """
    current = start.resolve()
    result: Path | None = None
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            result = candidate
            break
    if result is None:
        raise RuntimeError(f"no pyproject.toml ancestor found above {start}")
    return result


_REPO_ROOT = _find_repo_root(Path(__file__))
_EXEMPTIONS_PATH = Path(__file__).parent / "_cache_exemptions.txt"


# ------------------------------------------------------------------------
# pytest entry points — four tests, shared walker output
# ------------------------------------------------------------------------


def _collect_all() -> tuple[list[CacheViolation], tuple[Path, ...]]:
    """run every walker and return combined violations.

    :return: (violations, src_roots_discovered)
    :rtype: tuple[list[CacheViolation], tuple[Path, ...]]
    """
    src_roots = find_src_roots(_REPO_ROOT)
    v1 = find_sqlite_constructions(
        src_roots, _REPO_ROOT, _ALLOWED_SQLITE_CONSTRUCTION_SITES,
    )
    v2 = find_wrapper_classes(src_roots, _REPO_ROOT)
    v3 = find_direct_pool_access(
        src_roots, _REPO_ROOT, _COLLECTION_TABLE_ALLOWLIST,
    )
    v4 = find_missing_collections(
        src_roots, _REPO_ROOT,
        _COLLECTION_TABLE_ALLOWLIST, _MIGRATION_TABLE_ALLOWLIST,
    )
    return v1 + v2 + v3 + v4, src_roots


def _emit_report(
    violations: list[CacheViolation],
    src_roots: tuple[Path, ...],
    exemptions: list[CacheExemption],
    mode: str,
) -> str:
    """build the human-readable report string.

    :param violations: filtered violations
    :ptype violations: list[CacheViolation]
    :param src_roots: discovered src roots
    :ptype src_roots: tuple[Path, ...]
    :param exemptions: loaded exemptions
    :ptype exemptions: list[CacheExemption]
    :param mode: active mode
    :ptype mode: str
    :return: rendered report
    :rtype: str
    """
    by_test: dict[str, list[CacheViolation]] = {
        "sqlite_construction": [],
        "wrapper_class": [],
        "pool_access": [],
        "missing_collection": [],
    }
    for violation in violations:
        by_test.setdefault(violation.test_name, []).append(violation)
    lines: list[str] = [
        f"repo_root: {_REPO_ROOT}",
        f"src_roots: {[str(r) for r in src_roots]}",
        f"mode: {mode}",
        f"exemptions_loaded: {len(exemptions)}",
        f"violations_total: {len(violations)}",
        f"  sqlite_construction: {len(by_test['sqlite_construction'])}",
        f"  wrapper_class:       {len(by_test['wrapper_class'])}",
        f"  pool_access:         {len(by_test['pool_access'])}",
        f"  missing_collection:  {len(by_test['missing_collection'])}",
    ]
    for violation in sorted(
        violations,
        key=lambda v: (v.test_name, str(v.file), v.line, v.symbol),
    ):
        lines.append(violation.format(_REPO_ROOT))
    return "\n".join(lines)


class TestCachePrimitiveUsage:
    """four enforcement tests across the 3tears Collection primitive."""

    def test_no_bespoke_sqlite_backend_construction(self) -> None:
        """SQLiteBackend is constructed only inside sanctioned factories."""
        src_roots = find_src_roots(_REPO_ROOT)
        exemptions = parse_cache_exemptions(_EXEMPTIONS_PATH)
        raw = find_sqlite_constructions(
            src_roots, _REPO_ROOT, _ALLOWED_SQLITE_CONSTRUCTION_SITES,
        )
        filtered = apply_cache_exemptions(raw, exemptions, _REPO_ROOT)
        mode = resolve_mode()
        report = _emit_report(filtered, src_roots, exemptions, mode)
        print(report, file=sys.stderr)
        if mode == MODE_REPORT:
            return
        if filtered:
            pytest.fail(
                f"cache-enforcement: {len(filtered)} bespoke "
                f"SQLiteBackend construction(s):\n{report}"
            )

    def test_no_bespoke_cache_wrapper_classes(self) -> None:
        """no class stores SQLiteBackend + exposes cache api without BaseCollection."""
        src_roots = find_src_roots(_REPO_ROOT)
        exemptions = parse_cache_exemptions(_EXEMPTIONS_PATH)
        raw = find_wrapper_classes(src_roots, _REPO_ROOT)
        filtered = apply_cache_exemptions(raw, exemptions, _REPO_ROOT)
        mode = resolve_mode()
        report = _emit_report(filtered, src_roots, exemptions, mode)
        print(report, file=sys.stderr)
        if mode == MODE_REPORT:
            return
        if filtered:
            pytest.fail(
                f"cache-enforcement: {len(filtered)} bespoke cache "
                f"wrapper class(es):\n{report}"
            )

    def test_no_direct_pool_access_to_collection_tables(self) -> None:
        """pool.fetch/execute never targets a Collection-backed table directly."""
        src_roots = find_src_roots(_REPO_ROOT)
        exemptions = parse_cache_exemptions(_EXEMPTIONS_PATH)
        raw = find_direct_pool_access(
            src_roots, _REPO_ROOT, _COLLECTION_TABLE_ALLOWLIST,
        )
        filtered = apply_cache_exemptions(raw, exemptions, _REPO_ROOT)
        mode = resolve_mode()
        report = _emit_report(filtered, src_roots, exemptions, mode)
        print(report, file=sys.stderr)
        if mode == MODE_REPORT:
            return
        if filtered:
            pytest.fail(
                f"cache-enforcement: {len(filtered)} direct-pool-access "
                f"violation(s):\n{report}"
            )

    def test_all_tables_have_collections(self) -> None:
        """every migration-defined table has a matching Collection class."""
        src_roots = find_src_roots(_REPO_ROOT)
        exemptions = parse_cache_exemptions(_EXEMPTIONS_PATH)
        raw = find_missing_collections(
            src_roots, _REPO_ROOT,
            _COLLECTION_TABLE_ALLOWLIST, _MIGRATION_TABLE_ALLOWLIST,
        )
        filtered = apply_cache_exemptions(raw, exemptions, _REPO_ROOT)
        mode = resolve_mode()
        report = _emit_report(filtered, src_roots, exemptions, mode)
        print(report, file=sys.stderr)
        if mode == MODE_REPORT:
            return
        if filtered:
            pytest.fail(
                f"cache-enforcement: {len(filtered)} table(s) missing "
                f"Collections:\n{report}"
            )
