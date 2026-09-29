"""Enforcement -- Column.column_type matches migration-defined SQL type.

review-task-01 finding D-10 + partition-hardening-task-01 sub-task 5.

the platform's datetime contract distinguishes ``DATETIME_TYPE``
(``TIMESTAMP``, naive UTC) from ``DATETIMETZ_TYPE`` (``TIMESTAMPTZ``,
aware UTC) at the Python codec layer. asyncpg's ``TIMESTAMPTZ`` codec
calls ``astimezone(UTC)`` on every parameter -- a naive datetime is
interpreted as the **client's local timezone**, silently shifting the
wire value by the local-tz offset on non-UTC hosts. invisible on a
UTC CI/prod host; breaks CAS predicates on PDT/EST/etc. dev machines.

this enforcement walker locks the contract: every Column declared as
``DATETIMETZ_TYPE`` must have a migration-defined SQL type of
``TIMESTAMPTZ`` (and vice versa). a future contributor cannot widen
``Column("timestamp", DATETIMETZ_TYPE)`` against a ``TIMESTAMP`` column
without flipping a migration in the same change.

design:

- AST-only, <15s execution. walks every TableSchema declaration in the
  package source tree to extract ``(table_name, column_name,
  column_type)`` tuples for every datetime column. then walks every
  migration file under ``**/migrations/**/v*.py`` to extract the latest
  SQL type for each column via ``CREATE TABLE`` and ``ALTER TABLE``
  literals. a mismatch surfaces as a clear violation.
- migration precedence: ``ALTER TABLE ... ALTER COLUMN ... TYPE T``
  supersedes the original ``CREATE TABLE`` declaration. when multiple
  migrations touch the same column, the highest-versioned one wins.
- rendered migrations: a migration may build its DDL from the same
  ``TableSchema`` the collection reads (``table_def_for``) instead of
  repeating the columns as a SQL literal. Such a table has one
  declaration and cannot drift, but it leaves no literal for the
  regexes to read, so the check moves onto the renderer's type map,
  which must send ``DATETIMETZ_TYPE`` to ``TIMESTAMPTZ``. A renderer
  with a wrong or missing map fails, and so does every table it
  renders.
- exemptions: tables tracked only via inline raw SQL (no Column
  declaration anywhere) live in ``_column_type_alignment_exemptions.txt``
  with a ``# rationale: ...`` line per entry, matching the underscore-
  audit convention.

runtime is well under the 15s CLAUDE.md budget; the walker is
deterministic and side-effect-free.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

__all__: list[str] = []


# resolves to the 3tears repo root: this file lives at
# packages/core/tests/enforcement/, four levels under the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# package source roots to walk for TableSchema declarations, DERIVED rather than listed.
#
# A hand-written list was wrong twice over. Three of its five entries were spelled
# ``packages/agent-tools`` against a tree that has ``packages/agent/tools``, so they matched no
# directory and were skipped in silence while the gate read green. And five was never the number:
# eleven packages declare a ``TableSchema``, so six were unwalked by construction -- among them
# ``agent/acl``, ``agent/knowledge`` and ``datasources``, which all declare ``DATETIMETZ_TYPE``
# columns. A gate whose coverage is a list drifts from the tree the moment the tree grows.
#
# Both globs are required: flat packages live at ``packages/<name>/src`` and the agent family at
# ``packages/agent/<name>/src``. ``test_every_derived_root_exists`` fails when either matches
# nothing, which is the false-green this derivation exists to prevent -- the same shape
# ``threetears.enforcement.memory_only_kv`` and ``test_negative_cache_write_paths.py`` use.
_PACKAGE_SRC_ROOTS: list[Path] = sorted(path for path in _REPO_ROOT.glob("packages/*/src") if path.is_dir()) + sorted(
    path for path in _REPO_ROOT.glob("packages/*/*/src") if path.is_dir()
)

# package migration roots to walk for SQL declarations: every ``migrations`` package beneath a
# source root, so a package that grows one is covered the day it does.
_PACKAGE_MIGRATION_ROOTS: list[Path] = sorted(
    {path for root in _PACKAGE_SRC_ROOTS for path in root.glob("**/migrations") if path.is_dir()}
)

_DATETIME_TYPE_NAMES = frozenset({"DATETIMETZ_TYPE"})

# expected SQL type tag per Column type name.
# collections-task-05 eliminated DATETIME_TYPE / TIMESTAMP from the
# platform; DATETIMETZ_TYPE / TIMESTAMPTZ is now the only datetime
# column type 3tears recognises.
_EXPECTED_SQL_TYPE: dict[str, str] = {
    "DATETIMETZ_TYPE": "TIMESTAMPTZ",
}


@dataclass(frozen=True)
class ColumnDecl:
    """one Column declaration extracted from a TableSchema body.

    :cvar table: table name from the enclosing ``TableSchema(name=...)``
    :cvar column: column name (Column's first positional arg)
    :cvar column_type_name: ``DATETIME_TYPE`` or ``DATETIMETZ_TYPE``
    :cvar source: source file path
    :cvar lineno: line number of the Column call
    """

    table: str
    column: str
    column_type_name: str
    source: Path
    lineno: int


@dataclass(frozen=True)
class SqlColumnType:
    """one column-type assignment extracted from a migration file.

    :cvar table: target table name (lowercased)
    :cvar column: target column name (lowercased)
    :cvar sql_type: ``TIMESTAMP`` or ``TIMESTAMPTZ``
    :cvar source: source file path
    :cvar lineno: line number of the migration string literal
    """

    table: str
    column: str
    sql_type: str
    source: Path
    lineno: int


def _extract_column_call_table_and_col(
    call: ast.Call,
) -> tuple[str | None, str | None, str | None]:
    """if ``call`` is ``Column("name", TYPE_TAG, ...)``, return ``(name, type_name)``.

    returns ``(None, None, None)`` for non-matching calls.

    :param call: ast.Call node
    :ptype call: ast.Call
    :return: ``(column_name, type_tag_name, type_tag_name)`` triple
        where type_tag_name is the textual identifier (e.g.
        ``"DATETIME_TYPE"``); ``None`` triple for non-Column calls
    :rtype: tuple[str | None, str | None, str | None]
    """
    if not (isinstance(call.func, ast.Name) and call.func.id == "Column"):
        return (None, None, None)
    if len(call.args) < 2:
        return (None, None, None)
    name_node = call.args[0]
    type_node = call.args[1]
    if not (isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)):
        return (None, None, None)
    if not isinstance(type_node, ast.Name):
        return (None, None, None)
    return (name_node.value, type_node.id, type_node.id)


def _extract_table_name_from_schema(schema_call: ast.Call) -> str | None:
    """return the ``name=`` keyword arg value from a ``TableSchema(...)`` call.

    :param schema_call: ast.Call for ``TableSchema(...)``
    :ptype schema_call: ast.Call
    :return: table name string, or ``None`` if no ``name=`` keyword
    :rtype: str | None
    """
    result: str | None = None
    for kw in schema_call.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
            value = kw.value.value
            if isinstance(value, str):
                result = value
                break
    return result


def _collect_column_decls(src_root: Path) -> list[ColumnDecl]:
    """walk ``src_root`` and extract Column declarations of datetime types.

    finds every ``TableSchema(name=..., columns=[Column(...), ...])``
    pattern. table-name resolution is via the ``name=`` keyword arg on
    the enclosing TableSchema call; if the schema is built dynamically
    (no inline ``name=`` literal), the columns inside that schema are
    skipped (the alignment check has nothing to anchor against).

    :param src_root: package source root
    :ptype src_root: Path
    :return: list of :class:`ColumnDecl`
    :rtype: list[ColumnDecl]
    """
    out: list[ColumnDecl] = []
    if not src_root.exists():
        return out
    for path in src_root.rglob("*.py"):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Name) and node.func.id == "TableSchema"):
                continue
            table = _extract_table_name_from_schema(node)
            if table is None:
                continue
            # find the columns= keyword and walk its inline list
            for kw in node.keywords:
                if kw.arg != "columns":
                    continue
                if not isinstance(kw.value, ast.List):
                    continue
                for elt in kw.value.elts:
                    if not isinstance(elt, ast.Call):
                        continue
                    col_name, type_name, _ = _extract_column_call_table_and_col(elt)
                    if col_name is None or type_name is None:
                        continue
                    if type_name not in _DATETIME_TYPE_NAMES:
                        continue
                    out.append(
                        ColumnDecl(
                            table=table.lower(),
                            column=col_name.lower(),
                            column_type_name=type_name,
                            source=path,
                            lineno=elt.lineno,
                        ),
                    )
    return out


# regex for ``CREATE TABLE [IF NOT EXISTS] [schema.]<table> (...)`` ;
# captures the table name + the parenthesised column block.
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:[a-zA-Z_]\w*\.)?([a-zA-Z_]\w*)\s*\(\s*(.*?)\s*\)\s*$",
    re.IGNORECASE | re.DOTALL,
)

# inside a CREATE TABLE column block, a column declaration has shape
# ``<name> <type> [other modifiers...]``; type is captured up to the
# first comma / newline / NULL/NOT NULL keyword. we pre-split the
# block on commas (handling nested parens for compound types like
# ``vector(1024)``).
_COLUMN_LINE_RE = re.compile(
    r"^\s*([a-zA-Z_]\w*)\s+(TIMESTAMPTZ|TIMESTAMP)(?:\s|,|$|\()",
    re.IGNORECASE,
)

# regex for ``ALTER TABLE [schema.]<table> ALTER COLUMN <col> TYPE <type>``.
_ALTER_TYPE_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:[a-zA-Z_]\w*\.)?([a-zA-Z_]\w*)\s+"
    r"ALTER\s+COLUMN\s+([a-zA-Z_]\w*)\s+TYPE\s+(TIMESTAMPTZ|TIMESTAMP)\b",
    re.IGNORECASE,
)

# regex for ``ALTER TABLE [schema.]<table> RENAME COLUMN <old> TO <new>``.
#
# A rename carries the column's type to its new name, and without this the renamed column reads
# as declared-but-never-migrated: `conversation_memory_refs.date_created` arrives this way (v013
# converted `date_added` to TIMESTAMPTZ, v014 renamed it), and the gate reported it missing the
# moment its package root was spelled correctly.
_ALTER_RENAME_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:[a-zA-Z_]\w*\.)?([a-zA-Z_]\w*)\s+"
    r"RENAME\s+COLUMN\s+([a-zA-Z_]\w*)\s+TO\s+([a-zA-Z_]\w*)\b",
    re.IGNORECASE,
)

# regex for ``ALTER TABLE [schema.]<table> ADD COLUMN [IF NOT EXISTS] <col> <type>``.
_ALTER_ADD_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:[a-zA-Z_]\w*\.)?([a-zA-Z_]\w*)\s+"
    r"ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"([a-zA-Z_]\w*)\s+(TIMESTAMPTZ|TIMESTAMP)\b",
    re.IGNORECASE,
)


def _split_top_level_commas(block: str) -> list[str]:
    """split a CREATE TABLE column block on top-level commas (paren-aware).

    handles nested parens like ``vector(1024)`` so the inner comma list
    is not split.

    :param block: column declaration block, no leading/trailing parens
    :ptype block: str
    :return: list of raw column-line strings
    :rtype: list[str]
    """
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in block:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _collect_sql_types_from_literal(
    literal: str,
    source: Path,
    lineno: int,
) -> list[SqlColumnType]:
    """extract column-type assignments from one SQL string literal.

    handles three shapes:

    - ``CREATE TABLE [schema.]<table> (... <col> TIMESTAMP[TZ] ...)``
    - ``ALTER TABLE [schema.]<table> ALTER COLUMN <col> TYPE
      TIMESTAMP[TZ]``
    - ``ALTER TABLE [schema.]<table> ADD COLUMN [IF NOT EXISTS] <col>
      TIMESTAMP[TZ]``

    :param literal: SQL string literal
    :ptype literal: str
    :param source: source file path the literal lives in
    :ptype source: Path
    :param lineno: line number where the literal starts
    :ptype lineno: int
    :return: list of :class:`SqlColumnType`
    :rtype: list[SqlColumnType]
    """
    out: list[SqlColumnType] = []
    upper = literal.upper()
    # CREATE TABLE: parse the column block
    if "CREATE TABLE" in upper:
        m = _CREATE_TABLE_RE.search(literal)
        if m is not None:
            table = m.group(1).lower()
            block = m.group(2)
            for col_line in _split_top_level_commas(block):
                cm = _COLUMN_LINE_RE.match(col_line)
                if cm is None:
                    continue
                col_name = cm.group(1).lower()
                sql_type = cm.group(2).upper()
                out.append(
                    SqlColumnType(
                        table=table,
                        column=col_name,
                        sql_type=sql_type,
                        source=source,
                        lineno=lineno,
                    ),
                )
    # ALTER TABLE ALTER COLUMN ... TYPE
    for m in _ALTER_TYPE_RE.finditer(literal):
        out.append(
            SqlColumnType(
                table=m.group(1).lower(),
                column=m.group(2).lower(),
                sql_type=m.group(3).upper(),
                source=source,
                lineno=lineno,
            ),
        )
    # ALTER TABLE ADD COLUMN
    for m in _ALTER_ADD_RE.finditer(literal):
        out.append(
            SqlColumnType(
                table=m.group(1).lower(),
                column=m.group(2).lower(),
                sql_type=m.group(3).upper(),
                source=source,
                lineno=lineno,
            ),
        )
    return out


def _apply_renames(
    literal: str,
    latest: dict[tuple[str, str], SqlColumnType],
    source: Path,
    lineno: int,
) -> None:
    """carry a renamed column's type to its new name.

    ``ALTER TABLE t RENAME COLUMN old TO new`` changes no type, so the type the chain established
    for ``old`` is the type of ``new``. Without this the renamed column looks declared but never
    migrated, which is how `conversation_memory_refs.date_created` read as a violation.

    :param literal: SQL string literal
    :ptype literal: str
    :param latest: the resolved-so-far map, mutated in place
    :ptype latest: dict[tuple[str, str], SqlColumnType]
    :param source: the migration file, for the report
    :ptype source: Path
    :param lineno: line number of the literal
    :ptype lineno: int
    :return: nothing
    :rtype: None
    """
    for m in _ALTER_RENAME_RE.finditer(literal):
        table = m.group(1).lower()
        old_column = m.group(2).lower()
        new_column = m.group(3).lower()
        carried = latest.pop((table, old_column), None)
        if carried is None:
            continue
        latest[(table, new_column)] = SqlColumnType(
            table=table,
            column=new_column,
            sql_type=carried.sql_type,
            source=source,
            lineno=lineno,
        )


def _extract_version(path: Path) -> int:
    """parse the migration version from ``vNNN_*.py`` filename.

    files that do not match the convention are sorted last (version 0).

    :param path: migration file path
    :ptype path: Path
    :return: parsed integer version, or 0 on no match
    :rtype: int
    """
    m = re.match(r"v(\d+)_", path.name)
    return int(m.group(1)) if m is not None else 0


def _collect_sql_column_types(
    migration_root: Path,
) -> dict[tuple[str, str], SqlColumnType]:
    """walk ``migration_root`` and resolve the latest SQL type per column.

    files are processed in version order (``v001`` -> ``v002`` -> ...);
    later entries overwrite earlier ones for the same
    ``(table, column)`` key. the final dict reflects the post-chain
    state.

    :param migration_root: directory containing ``vNNN_*.py`` files
    :ptype migration_root: Path
    :return: ``(table, column) -> SqlColumnType`` mapping (latest wins)
    :rtype: dict[tuple[str, str], SqlColumnType]
    """
    latest: dict[tuple[str, str], SqlColumnType] = {}
    if not migration_root.exists():
        return latest
    files = sorted(migration_root.glob("v*.py"), key=_extract_version)
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literal = node.value
                # a RENAME carries a type without naming one, so it passes this filter too.
                upper = literal.upper()
                if "TIMESTAMP" not in upper and "RENAME COLUMN" not in upper:
                    continue
                for hit in _collect_sql_types_from_literal(
                    literal,
                    path,
                    node.lineno,
                ):
                    latest[(hit.table, hit.column)] = hit
                _apply_renames(literal, latest, path, node.lineno)
            elif isinstance(node, ast.JoinedStr):
                # join f-string fragments into a best-effort literal so
                # f"ALTER TABLE x ADD COLUMN {colname} TIMESTAMP" is not
                # missed (the column name part may be a variable).
                parts: list[str] = []
                for v in node.values:
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        parts.append(v.value)
                    else:
                        parts.append("__PLACEHOLDER__")
                literal = "".join(parts)
                # a RENAME carries a type without naming one, so it passes this filter too.
                upper = literal.upper()
                if "TIMESTAMP" not in upper and "RENAME COLUMN" not in upper:
                    continue
                for hit in _collect_sql_types_from_literal(
                    literal,
                    path,
                    node.lineno,
                ):
                    latest[(hit.table, hit.column)] = hit
                _apply_renames(literal, latest, path, node.lineno)
    return latest


@dataclass(frozen=True)
class ColumnExemption:
    """one entry from the column-type-alignment exemption file."""

    table: str
    column: str
    rationale: str


def parse_column_exemptions(path: Path) -> list[ColumnExemption]:
    """parse the column-type-alignment exemption file.

    file shape (matches the underscore-audit / cache-primitive
    convention):

    .. code-block:: text

        # rationale: <specific reason for this exemption>
        <table>:<column>

    blank lines and ``#``-prefixed comment lines (other than rationale)
    are skipped. every entry MUST be preceded by a non-empty rationale
    line; entries without one are a parser error.

    :param path: exemption file path
    :ptype path: Path
    :return: list of :class:`ColumnExemption`
    :rtype: list[ColumnExemption]
    :raises ValueError: on a missing / blank rationale or malformed
        entry line
    """
    out: list[ColumnExemption] = []
    if not path.exists():
        return out
    pending_rationale: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# rationale:"):
            text = line[len("# rationale:") :].strip()
            if not text:
                raise ValueError(
                    f"{path}: blank rationale (write '# rationale: <reason>')",
                )
            pending_rationale = text
            continue
        if line.startswith("#"):
            continue
        # entry line
        if pending_rationale is None:
            raise ValueError(
                f"{path}: entry {line!r} has no preceding rationale line",
            )
        parts = line.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"{path}: malformed entry {line!r}; expected table:column",
            )
        table, column = parts
        out.append(
            ColumnExemption(
                table=table.strip().lower(),
                column=column.strip().lower(),
                rationale=pending_rationale,
            ),
        )
        pending_rationale = None
    return out


_EXEMPTION_FILE = Path(__file__).resolve().parent / "_column_type_alignment_exemptions.txt"

# the renderer seam: a migration may build its DDL from the same TableSchema the collection reads,
# instead of repeating the columns as a SQL literal. Such a table cannot drift -- there is one
# declaration -- but it also has no literal for the regexes above to read. The check does not go
# away: it moves onto the renderer's own type map, which must send DATETIMETZ_TYPE to TIMESTAMPTZ
# for every table it renders. Anything else (a renderer that maps it to TIMESTAMP, or a rendered
# table whose renderer declares no map) is a violation, same as a mismatched literal.
_RENDERER_FUNCTION = "table_def_for"
_RENDERER_TYPE_MAP = "_DDL_TYPES"


def _rendered_schema_sets(migration_roots: list[Path]) -> set[str]:
    """the names of the schema collections a migration actually renders.

    Keyed on the render CALL SITE, not on module co-location: a migration that renders
    ``for schema in COORDINATION_TABLE_SCHEMAS: ... table_def_for(schema)`` names that
    collection, and only the tables inside it are covered by the renderer's type map. A fifth
    table declared in the same module but left out of the rendered set is then still checked
    against a SQL literal, which is the drift this gate exists for.

    :param migration_roots: package migration roots to walk
    :ptype migration_roots: list[Path]
    :return: the module-level names iterated at a render call site
    :rtype: set[str]
    """
    rendered: set[str] = set()
    for root in migration_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except OSError, SyntaxError:
                continue
            for loop in ast.walk(tree):
                if not isinstance(loop, ast.For):
                    continue
                if not isinstance(loop.target, ast.Name) or not isinstance(loop.iter, ast.Name):
                    continue
                renders_the_loop_variable = any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == _RENDERER_FUNCTION
                    and any(isinstance(arg, ast.Name) and arg.id == loop.target.id for arg in call.args)
                    for call in ast.walk(loop)
                )
                if renders_the_loop_variable:
                    rendered.add(loop.iter.id)
    return rendered


def _renderer_maps_datetimetz_correctly(module: ast.Module) -> bool:
    """whether a renderer module's type map sends ``DATETIMETZ_TYPE`` to a timestamptz column.

    :param module: parsed renderer module
    :ptype module: ast.Module
    :return: ``True`` when the map is present and correct
    :rtype: bool
    """
    for node in ast.walk(module):
        if not isinstance(node, ast.AnnAssign | ast.Assign):
            continue
        targets = [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)
        if not any(isinstance(t, ast.Name) and t.id == _RENDERER_TYPE_MAP for t in targets):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            continue
        for key, mapped in zip(value.keys, value.values, strict=False):
            if isinstance(key, ast.Name) and key.id == "DATETIMETZ_TYPE":
                return (
                    isinstance(mapped, ast.Constant)
                    and isinstance(mapped.value, str)
                    and mapped.value.upper() == _EXPECTED_SQL_TYPE["DATETIMETZ_TYPE"]
                )
    return False


def _class_table_names(tree: ast.Module) -> dict[str, str]:
    """map each class that declares ``schema = TableSchema(name=...)`` to that table name.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: class name -> table name
    :rtype: dict[str, str]
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign | ast.AnnAssign):
                continue
            targets = [statement.target] if isinstance(statement, ast.AnnAssign) else list(statement.targets)
            if not any(isinstance(t, ast.Name) and t.id == "schema" for t in targets):
                continue
            value = statement.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "TableSchema":
                table = _extract_table_name_from_schema(value)
                if table is not None:
                    out[node.name] = table.lower()
    return out


def _tables_in_rendered_sets(tree: ast.Module, rendered_sets: set[str]) -> set[str]:
    """the tables named by a module's rendered schema collections.

    Resolves ``NAME = (SomeCollection.schema, OtherCollection.schema)`` against the classes
    declared in the same module, so the exempt set is exactly what the migration renders.

    :param tree: parsed module
    :ptype tree: ast.Module
    :param rendered_sets: the collection names a migration renders
    :ptype rendered_sets: set[str]
    :return: the table names those collections cover
    :rtype: set[str]
    """
    by_class = _class_table_names(tree)
    tables: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign | ast.Assign):
            continue
        targets = [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)
        if not any(isinstance(t, ast.Name) and t.id in rendered_sets for t in targets):
            continue
        for element in ast.walk(node):
            if (
                isinstance(element, ast.Attribute)
                and element.attr == "schema"
                and isinstance(element.value, ast.Name)
                and element.value.id in by_class
            ):
                tables.add(by_class[element.value.id])
    return tables


def _rendered_tables(src_roots: list[Path], migration_roots: list[Path]) -> tuple[set[str], list[str]]:
    """tables whose DDL a migration renders from their own schema, and any renderer violations.

    A rendered table's columns are checked through the renderer's type map rather than through a
    SQL literal, so a renderer with a wrong or missing map fails here instead of silently
    exempting every table it renders.

    :param src_roots: package source roots to walk
    :ptype src_roots: list[Path]
    :param migration_roots: package migration roots to walk
    :ptype migration_roots: list[Path]
    :return: rendered table names, and violation messages for bad renderers
    :rtype: tuple[set[str], list[str]]
    """
    rendered: set[str] = set()
    violations: list[str] = []
    rendered_sets = _rendered_schema_sets(migration_roots)
    if not rendered_sets:
        return (rendered, violations)
    for src_root in src_roots:
        if not src_root.exists():
            continue
        for path in src_root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except OSError, SyntaxError:
                continue
            tables = _tables_in_rendered_sets(tree, rendered_sets)
            if not tables:
                continue
            if not _renderer_maps_datetimetz_correctly(tree):
                violations.append(
                    f"{path}: declares {len(tables)} table(s) a migration renders "
                    f"({', '.join(sorted(tables))}), but its {_RENDERER_TYPE_MAP} does not map "
                    f"DATETIMETZ_TYPE to {_EXPECTED_SQL_TYPE['DATETIMETZ_TYPE']}"
                )
                continue
            rendered |= tables
    return (rendered, violations)


def test_every_derived_root_exists() -> None:
    """the roots are derived, so the failure to guard against is a glob that matched nothing.

    A list of roots drifted from the tree twice: three entries were spelled
    ``packages/agent-tools`` against a tree that has ``packages/agent/tools`` and matched nothing,
    and six packages declaring a ``TableSchema`` were never listed at all. Deriving the roots
    fixes both, and moves the failure mode: a glob that stops matching (a layout change, a
    relocated repo root) would silently walk nothing and read green. So assert the globs found
    something, and that everything they found is real.

    :return: nothing
    :rtype: None
    :raises AssertionError: when a glob matched nothing, or a derived root does not exist
    """
    assert _PACKAGE_SRC_ROOTS, (
        f"no package source roots under {_REPO_ROOT}/packages -- the walker would check nothing "
        f"and this gate would read green"
    )
    assert _PACKAGE_MIGRATION_ROOTS, (
        "no migration roots beneath any package source root -- no SQL column type would be "
        "discovered and every declaration would read as unmigrated"
    )
    missing = [str(root) for root in (*_PACKAGE_SRC_ROOTS, *_PACKAGE_MIGRATION_ROOTS) if not root.exists()]
    assert not missing, "derived roots that do not exist (so nothing in them is checked):\n  " + "\n  ".join(missing)


def test_column_type_alignment() -> None:
    """every datetime Column matches its migration-defined SQL type.

    walks the package source tree for Column declarations and the
    migration tree for SQL types; reports any mismatch with file
    + line context. exemption file entries skip the check for the
    specific ``(table, column)`` pair with a documented rationale.

    :return: nothing
    :rtype: None
    :raises AssertionError: on any unexempted mismatch
    """
    violations: list[str] = []
    column_decls: list[ColumnDecl] = []
    for src_root in _PACKAGE_SRC_ROOTS:
        column_decls.extend(_collect_column_decls(src_root))

    sql_types: dict[tuple[str, str], SqlColumnType] = {}
    for migration_root in _PACKAGE_MIGRATION_ROOTS:
        for k, v in _collect_sql_column_types(migration_root).items():
            sql_types[k] = v

    exemptions = {(e.table, e.column) for e in parse_column_exemptions(_EXEMPTION_FILE)}
    rendered, violations = _rendered_tables(_PACKAGE_SRC_ROOTS, _PACKAGE_MIGRATION_ROOTS)

    for decl in column_decls:
        key = (decl.table, decl.column)
        if key in exemptions:
            continue
        if decl.table in rendered:
            # its migration renders this column from this same declaration, and the renderer's
            # type map was checked above; there is no second place for it to drift from.
            continue
        sql = sql_types.get(key)
        if sql is None:
            # genuine drift: a Column declared but no migration declares
            # the SQL column type. either the column is added by a
            # path the walker cannot see (rawer SQL helper) or the
            # migration was never written. surface the gap.
            violations.append(
                f"{decl.source}:{decl.lineno}: {decl.table}.{decl.column}: "
                f"Column declared as {decl.column_type_name} but no "
                f"matching CREATE TABLE / ALTER TABLE entry found in any "
                f"migration tree (add an entry to "
                f"_column_type_alignment_exemptions.txt with a rationale "
                f"if the column is genuinely tracked outside the "
                f"migration system)",
            )
            continue
        expected_sql = _EXPECTED_SQL_TYPE[decl.column_type_name]
        if sql.sql_type != expected_sql:
            violations.append(
                f"{decl.source}:{decl.lineno}: {decl.table}.{decl.column}: "
                f"Column declared as {decl.column_type_name} but migration "
                f"{sql.source.name} (line {sql.lineno}) defines as "
                f"{sql.sql_type} (expected {expected_sql})",
            )

    if violations:
        formatted = "\n".join(violations)
        raise AssertionError(
            f"column-type alignment: {len(violations)} mismatch(es)\n{formatted}",
        )


def test_column_type_alignment_finds_at_least_one_decl() -> None:
    """sanity check: walker discovers at least one datetime Column.

    if zero Column declarations of either datetime type are surfaced,
    the alignment check is silently a no-op and the contract is not
    actually enforced. surface the regression at CI time.

    :return: nothing
    :rtype: None
    """
    decls: list[ColumnDecl] = []
    for src_root in _PACKAGE_SRC_ROOTS:
        decls.extend(_collect_column_decls(src_root))
    assert len(decls) > 0, (
        "no datetime Column declarations found across any package "
        "source root -- the alignment walker is silently a no-op"
    )


def test_column_type_alignment_finds_at_least_one_sql_type() -> None:
    """sanity check: walker discovers at least one migration SQL type.

    :return: nothing
    :rtype: None
    """
    sql_types: dict[tuple[str, str], SqlColumnType] = {}
    for migration_root in _PACKAGE_MIGRATION_ROOTS:
        for k, v in _collect_sql_column_types(migration_root).items():
            sql_types[k] = v
    assert len(sql_types) > 0, (
        "no datetime SQL columns discovered across any migration tree -- the alignment walker is silently a no-op"
    )


# ---------------------------------------------------------------------------
# walker self-tests (positive + negative + exemption)
# ---------------------------------------------------------------------------


class TestColumnTypeAlignmentWalker:
    """fixture-driven self-tests for the alignment walker."""

    def test_positive_clean_alignment(self, tmp_path: Path) -> None:
        """a TableSchema + matching CREATE TABLE produces zero violations."""
        src = tmp_path / "src" / "pkg"
        src.mkdir(parents=True)
        (src / "coll.py").write_text(
            "from x import Column, TableSchema, DATETIMETZ_TYPE\n"
            "schema = TableSchema(\n"
            "    name='audit_events',\n"
            "    columns=[Column('timestamp', DATETIMETZ_TYPE)],\n"
            ")\n",
            encoding="utf-8",
        )
        mig = tmp_path / "mig"
        mig.mkdir()
        (mig / "v001_create.py").write_text(
            "SQL = '''CREATE TABLE audit_events (\n"
            "    id UUID PRIMARY KEY,\n"
            "    timestamp TIMESTAMPTZ NOT NULL\n"
            ")'''\n",
            encoding="utf-8",
        )
        decls = _collect_column_decls(tmp_path / "src")
        sql_types = _collect_sql_column_types(mig)
        assert decls != []
        assert ("audit_events", "timestamp") in sql_types
        assert sql_types[("audit_events", "timestamp")].sql_type == "TIMESTAMPTZ"
        assert decls[0].column_type_name == "DATETIMETZ_TYPE"

    def test_negative_mismatched_alignment(self, tmp_path: Path) -> None:
        """DATETIMETZ_TYPE on a TIMESTAMP column surfaces a mismatch."""
        src = tmp_path / "src" / "pkg"
        src.mkdir(parents=True)
        (src / "coll.py").write_text(
            "from x import Column, TableSchema, DATETIMETZ_TYPE\n"
            "schema = TableSchema(\n"
            "    name='items',\n"
            "    columns=[Column('date_created', DATETIMETZ_TYPE)],\n"
            ")\n",
            encoding="utf-8",
        )
        mig = tmp_path / "mig"
        mig.mkdir()
        (mig / "v001_create.py").write_text(
            "SQL = '''CREATE TABLE items (\n    id UUID PRIMARY KEY,\n    date_created TIMESTAMP NOT NULL\n)'''\n",
            encoding="utf-8",
        )
        decls = _collect_column_decls(tmp_path / "src")
        sql_types = _collect_sql_column_types(mig)
        # the walker would emit a mismatch; exercise the pieces.
        assert decls[0].column_type_name == "DATETIMETZ_TYPE"
        assert sql_types[("items", "date_created")].sql_type == "TIMESTAMP"
        assert _EXPECTED_SQL_TYPE[decls[0].column_type_name] != sql_types[("items", "date_created")].sql_type

    def test_alter_type_supersedes_create(self, tmp_path: Path) -> None:
        """``ALTER TABLE ALTER COLUMN TYPE`` overrides the CREATE TABLE form."""
        mig = tmp_path / "mig"
        mig.mkdir()
        (mig / "v001_create.py").write_text(
            "SQL = '''CREATE TABLE x (id UUID PRIMARY KEY, t TIMESTAMP)'''\n",
            encoding="utf-8",
        )
        (mig / "v002_alter.py").write_text(
            "SQL = '''ALTER TABLE x ALTER COLUMN t TYPE TIMESTAMPTZ USING t AT TIME ZONE 'UTC' '''\n",
            encoding="utf-8",
        )
        sql_types = _collect_sql_column_types(mig)
        assert sql_types[("x", "t")].sql_type == "TIMESTAMPTZ"

    def test_exemption_file_entry_is_respected(self, tmp_path: Path) -> None:
        """an entry in the exemption file skips the alignment check."""
        path = tmp_path / "ex.txt"
        path.write_text(
            "# rationale: tracked via a custom helper that inlines the SQL "
            "outside any migration\n"
            "side_table:date_legacy\n",
            encoding="utf-8",
        )
        entries = parse_column_exemptions(path)
        assert entries == [
            ColumnExemption(
                table="side_table",
                column="date_legacy",
                rationale="tracked via a custom helper that inlines the SQL outside any migration",
            ),
        ]

    def test_exemption_file_rejects_blank_rationale(self, tmp_path: Path) -> None:
        """``# rationale:`` with no text after the colon is a parser error."""
        path = tmp_path / "ex.txt"
        path.write_text(
            "# rationale:\nside_table:date_legacy\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="rationale"):
            parse_column_exemptions(path)

    def test_exemption_file_rejects_entry_without_rationale(
        self,
        tmp_path: Path,
    ) -> None:
        """entry line not preceded by a rationale is a parser error."""
        path = tmp_path / "ex.txt"
        path.write_text("side_table:date_legacy\n", encoding="utf-8")
        with pytest.raises(ValueError, match="rationale"):
            parse_column_exemptions(path)
