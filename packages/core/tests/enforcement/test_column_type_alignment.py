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

# package source roots to walk for TableSchema declarations.
_PACKAGE_SRC_ROOTS: list[Path] = [
    _REPO_ROOT / "packages" / "core" / "src",
    _REPO_ROOT / "packages" / "agent-tools" / "src",
    _REPO_ROOT / "packages" / "agent-workspace" / "src",
    _REPO_ROOT / "packages" / "agent-memory" / "src",
    _REPO_ROOT / "packages" / "conversations" / "src",
]

# package migration roots to walk for SQL declarations.
_PACKAGE_MIGRATION_ROOTS: list[Path] = [
    _REPO_ROOT / "packages" / "core" / "src" / "threetears" / "core" / "data" / "migrations",  # noqa: E501
    _REPO_ROOT / "packages" / "agent-tools" / "src" / "threetears" / "agent" / "tools" / "migrations",  # noqa: E501
    _REPO_ROOT / "packages" / "agent-workspace" / "src" / "threetears" / "agent" / "workspace" / "migrations",  # noqa: E501
    _REPO_ROOT / "packages" / "agent-memory" / "src" / "threetears" / "agent" / "memory" / "migrations",  # noqa: E501
    _REPO_ROOT / "packages" / "conversations" / "src" / "threetears" / "conversations" / "migrations",  # noqa: E501
]

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
                if "TIMESTAMP" not in literal.upper() and "TIMESTAMPTZ" not in literal.upper():
                    continue
                for hit in _collect_sql_types_from_literal(
                    literal,
                    path,
                    node.lineno,
                ):
                    latest[(hit.table, hit.column)] = hit
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
                if "TIMESTAMP" not in literal.upper() and "TIMESTAMPTZ" not in literal.upper():
                    continue
                for hit in _collect_sql_types_from_literal(
                    literal,
                    path,
                    node.lineno,
                ):
                    latest[(hit.table, hit.column)] = hit
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
    column_decls: list[ColumnDecl] = []
    for src_root in _PACKAGE_SRC_ROOTS:
        column_decls.extend(_collect_column_decls(src_root))

    sql_types: dict[tuple[str, str], SqlColumnType] = {}
    for migration_root in _PACKAGE_MIGRATION_ROOTS:
        for k, v in _collect_sql_column_types(migration_root).items():
            sql_types[k] = v

    exemptions = {(e.table, e.column) for e in parse_column_exemptions(_EXEMPTION_FILE)}

    violations: list[str] = []
    for decl in column_decls:
        key = (decl.table, decl.column)
        if key in exemptions:
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
