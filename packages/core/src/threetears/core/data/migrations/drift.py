"""
drift detection: compare live schema to expected schema.

the drift detector answers a single operator question: "does the DB
match what my migrations claim is applied?" it does so by:

1. running every applied migration in preview mode against the live
   DB to collect the CREATE TABLE / ALTER TABLE statements that would
   have been issued to build the current state.
2. parsing those statements to derive an "expected" set of tables and
   columns.
3. querying the live DB's ``information_schema`` for actual tables and
   columns.
4. computing three difference sets and returning a structured report:
   - expected tables missing from the DB (should not happen; the
     migration ran but its DDL is not reflected in pg metadata)
   - extra tables in the DB not claimed by any migration (hand-DDL,
     likely unintentional)
   - column-level mismatches (missing, extra, or type-differing)

parsing is intentionally simple: it handles the ``CREATE TABLE IF NOT
EXISTS <name> ( <cols> )`` shape and the ``ALTER TABLE <name> ADD
COLUMN [IF NOT EXISTS] <col> <type>`` shape, which together cover
every migration in the current tree. unrecognized DDL does NOT crash
the detector — it is logged and ignored so drift reports do not
spuriously fail on a future migration that uses a different idiom.
callers that write novel DDL are responsible for teaching the parser
about it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from threetears.observe import get_logger

__all__ = [
    "DriftReport",
    "ExpectedTable",
    "diff_expected_live",
    "parse_ddl_to_expected",
    "snapshot_live_schema",
]

log = get_logger(__name__)


_INFO_SCHEMA_TABLES_SQL = (
    "SELECT table_name FROM information_schema.tables "
    "WHERE table_schema = current_schema() "
    "  AND table_type = 'BASE TABLE' "
    "ORDER BY table_name"
)

_INFO_SCHEMA_COLUMNS_SQL = (
    "SELECT table_name, column_name, data_type "
    "FROM information_schema.columns "
    "WHERE table_schema = current_schema() "
    "ORDER BY table_name, ordinal_position"
)


# a table reference is either an unquoted ident, a quoted ident, or a
# schema-qualified ident (``schema.name`` / ``"schema"."name"``). every
# migration in the platform tree today mixes these three shapes, so the
# parser must handle all three. the capture groups below strip the
# schema prefix via :func:`_extract_table_name` so drift comparison uses
# unqualified names (matches ``information_schema.table_name``).
_TABLE_REF = r'(?P<table_ref>(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\.(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))?)'

# captures "CREATE TABLE IF NOT EXISTS <name> ( <body> )"
_RE_CREATE_TABLE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    + _TABLE_REF
    + r"\s*\((?P<body>.*)\)\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)

# captures "ALTER TABLE <name> ADD COLUMN [IF NOT EXISTS] <col> <type>"
# the type character class permits digits, commas, whitespace, and
# parentheses so ``NUMERIC(10, 6)`` parses as one token.
_RE_ADD_COLUMN = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?"
    + _TABLE_REF
    + r"\s+ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?P<col>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<type>[A-Za-z0-9_ ,()]+?)(?:\s+(?:NOT|DEFAULT|REFERENCES|UNIQUE|CHECK|PRIMARY)|$)",
    flags=re.IGNORECASE,
)

# captures "ALTER TABLE <name> DROP COLUMN [IF EXISTS] <col>"
_RE_DROP_COLUMN = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?"
    + _TABLE_REF
    + r"\s+DROP\s+COLUMN\s+(?:IF\s+EXISTS\s+)?"
    r"(?P<col>[A-Za-z_][A-Za-z0-9_]*)",
    flags=re.IGNORECASE,
)


def _extract_table_name(table_ref: str) -> str:
    """
    strip quotes and schema prefix from a captured table reference.

    ``platform.agents`` -> ``agents``; ``"platform"."agents"`` ->
    ``agents``. the drift check compares to ``information_schema.
    table_name`` which is always unqualified, so the schema prefix is
    noise at diff time.

    :param table_ref: raw captured text
    :ptype table_ref: str
    :return: unqualified, unquoted table name
    :rtype: str
    """
    result = table_ref.strip()
    if "." in result:
        # split on the last dot — simple heuristic good for the
        # shapes migrations use today.
        result = result.rsplit(".", 1)[1]
    result = result.strip('"')
    return result


def _extract_schema_prefix(table_ref: str) -> str | None:
    """
    extract the schema prefix from a table reference, if any.

    ``platform.agents`` -> ``"platform"``; ``agents`` -> ``None``.

    :param table_ref: raw captured text
    :ptype table_ref: str
    :return: schema portion, or None for unqualified references
    :rtype: str | None
    """
    stripped = table_ref.strip()
    if "." not in stripped:
        return None
    head = stripped.rsplit(".", 1)[0]
    result = head.strip().strip('"')
    return result


@dataclass
class ExpectedTable:
    """
    expected shape of one table derived from parsed DDL.

    :ivar name: unquoted table name
    :ivar columns: mapping of column name to normalized data type
    :ivar schema: optional schema the DDL qualified the table with;
        ``None`` means the DDL was unqualified and the table is
        assumed to live in the runner's current search_path schema
    """

    name: str
    columns: dict[str, str] = field(default_factory=dict)
    schema: str | None = None


@dataclass
class DriftReport:
    """
    structured drift-detection result.

    every field is a list so the JSON output has stable shape. an empty
    report (every list empty) means the live schema matches expectations.

    :ivar missing_tables: tables migrations claim exist that are not in
        the DB
    :ivar extra_tables: tables in the DB that no migration created
    :ivar missing_columns: (table, column) expected but absent from the
        live table
    :ivar extra_columns: (table, column, type) present in the live
        table but not declared by any migration
    :ivar type_mismatches: (table, column, expected_type, actual_type)
        where the column exists but with a different type
    """

    missing_tables: list[str] = field(default_factory=list)
    extra_tables: list[str] = field(default_factory=list)
    missing_columns: list[tuple[str, str]] = field(default_factory=list)
    extra_columns: list[tuple[str, str, str]] = field(default_factory=list)
    type_mismatches: list[tuple[str, str, str, str]] = field(default_factory=list)

    def is_clean(self) -> bool:
        """
        return True if every drift bucket is empty.

        :return: True when the live schema matches expectations
        :rtype: bool
        """
        result = (
            not self.missing_tables
            and not self.extra_tables
            and not self.missing_columns
            and not self.extra_columns
            and not self.type_mismatches
        )
        return result

    def as_dict(self) -> dict[str, Any]:
        """
        render the report as a JSON-serializable dict.

        :return: dict with every drift bucket under a stable key
        :rtype: dict[str, Any]
        """
        result: dict[str, Any] = {
            "clean": self.is_clean(),
            "missing_tables": list(self.missing_tables),
            "extra_tables": list(self.extra_tables),
            "missing_columns": [
                {"table": t, "column": c} for t, c in self.missing_columns
            ],
            "extra_columns": [
                {"table": t, "column": c, "data_type": dt}
                for t, c, dt in self.extra_columns
            ],
            "type_mismatches": [
                {"table": t, "column": c, "expected": e, "actual": a}
                for t, c, e, a in self.type_mismatches
            ],
        }
        return result

    def as_human_lines(self) -> list[str]:
        """
        render the report as plaintext lines for stdout.

        :return: list of human-readable report lines
        :rtype: list[str]
        """
        lines: list[str] = []
        if self.is_clean():
            lines.append("schema matches expectations; no drift.")
            return lines
        if self.missing_tables:
            lines.append("missing tables (migration claimed but DB does not have):")
            for t in self.missing_tables:
                lines.append(f"  - {t}")
        if self.extra_tables:
            lines.append("extra tables (in DB but no migration created them):")
            for t in self.extra_tables:
                lines.append(f"  - {t}")
        if self.missing_columns:
            lines.append("missing columns (expected but absent from live table):")
            for tbl, col in self.missing_columns:
                lines.append(f"  - {tbl}.{col}")
        if self.extra_columns:
            lines.append("extra columns (in DB but no migration declared them):")
            for tbl, col, dt in self.extra_columns:
                lines.append(f"  - {tbl}.{col} ({dt})")
        if self.type_mismatches:
            lines.append("column type mismatches (expected vs actual):")
            for tbl, col, exp, act in self.type_mismatches:
                lines.append(f"  - {tbl}.{col}: expected {exp!r}, actual {act!r}")
        return lines


# ---------------------------------------------------------------------
# DDL parsing
# ---------------------------------------------------------------------


def _normalize_type(raw: str) -> str:
    """
    normalize a column type so migration-declared types can be compared
    to information_schema.data_type values.

    :param raw: raw type text from DDL source
    :ptype raw: str
    :return: lower-cased canonical type
    :rtype: str
    """
    clean = raw.strip().lower()
    # stripping anything past the first open paren folds VARCHAR(255) to
    # VARCHAR; information_schema.data_type reports "character varying"
    # for varchar columns regardless of length, so the length is noise
    # for drift comparison.
    paren = clean.find("(")
    if paren >= 0:
        clean = clean[:paren].strip()
    # map the handful of declared-vs-info_schema aliases the tree uses.
    alias_map = {
        "varchar": "character varying",
        "int": "integer",
        "int4": "integer",
        "int8": "bigint",
        "bool": "boolean",
        "timestamptz": "timestamp with time zone",
        "timestamp": "timestamp without time zone",
        "jsonb": "jsonb",
        "uuid": "uuid",
        "text": "text",
        "bytea": "bytea",
        "decimal": "numeric",
        "float8": "double precision",
        "float4": "real",
    }
    result = alias_map.get(clean, clean)
    return result


def _split_table_body(body: str) -> list[str]:
    """
    split a CREATE TABLE body into one entry per top-level comma.

    parentheses are balanced so ``DECIMAL(10,4)`` doesn't split into
    two entries. the split is simple enough to inline here rather than
    pulling in a full SQL parser.

    :param body: text inside the table's parentheses
    :ptype body: str
    :return: list of trimmed column/constraint clauses
    :rtype: list[str]
    """
    depth = 0
    current: list[str] = []
    result: list[str] = []
    for char in body:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        result.append(tail)
    return result


def _parse_column_clause(clause: str) -> tuple[str, str] | None:
    """
    parse one CREATE TABLE body clause into (column_name, data_type).

    returns None if the clause is a table-level constraint rather than a
    column definition (``PRIMARY KEY (...)``, ``FOREIGN KEY ...``,
    ``UNIQUE ...``, ``CHECK (...)``, ``CONSTRAINT ...``). those do not
    contribute columns.

    :param clause: single comma-separated clause from the table body
    :ptype clause: str
    :return: (column_name, normalized_type) or None for constraints
    :rtype: tuple[str, str] | None
    """
    first = clause.split(None, 1)
    if not first:
        return None
    head = first[0].upper()
    if head in {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"}:
        return None
    name = first[0].strip('"')
    if len(first) < 2:
        return None
    # second token is the type; take up to the next space that precedes
    # NOT / DEFAULT / REFERENCES / UNIQUE / CHECK / PRIMARY / COLLATE
    rest = first[1].strip()
    match = re.match(
        r"^(?P<type>[A-Za-z0-9_ ,()]+?)"
        r"(?:\s+(?:NOT|DEFAULT|REFERENCES|UNIQUE|CHECK|PRIMARY|COLLATE)|$)",
        rest,
        flags=re.IGNORECASE,
    )
    if match is None:
        raw_type = rest
    else:
        raw_type = match.group("type")
    result = (name, _normalize_type(raw_type))
    return result


def parse_ddl_to_expected(statements: list[str]) -> dict[str, ExpectedTable]:
    """
    parse a sequence of DDL strings into expected tables + columns.

    each statement is examined in turn; CREATE TABLE creates a new
    entry, ALTER TABLE ADD COLUMN extends an existing entry, ALTER
    TABLE DROP COLUMN removes an entry. unrecognized statements are
    logged and skipped so an unparseable DDL does not abort the whole
    drift check.

    :param statements: ordered list of DDL strings from preview output
    :ptype statements: list[str]
    :return: mapping of table_name -> ExpectedTable
    :rtype: dict[str, ExpectedTable]
    """
    expected: dict[str, ExpectedTable] = {}
    for raw in statements:
        stmt = raw.strip().rstrip(";").strip()
        if not stmt:
            continue
        upper = stmt.upper()
        if upper.startswith("CREATE TABLE"):
            match = _RE_CREATE_TABLE.match(stmt)
            if match is None:
                log.debug("drift: could not parse CREATE TABLE: %s", stmt[:120])
                continue
            ref = match.group("table_ref")
            name = _extract_table_name(ref)
            schema_prefix = _extract_schema_prefix(ref)
            body = match.group("body")
            columns: dict[str, str] = {}
            for clause in _split_table_body(body):
                parsed = _parse_column_clause(clause)
                if parsed is not None:
                    col_name, col_type = parsed
                    columns[col_name] = col_type
            expected[name] = ExpectedTable(
                name=name, columns=columns, schema=schema_prefix
            )
        elif upper.startswith("ALTER TABLE"):
            add_match = _RE_ADD_COLUMN.match(stmt)
            if add_match is not None:
                name = _extract_table_name(add_match.group("table_ref"))
                col = add_match.group("col")
                col_type = _normalize_type(add_match.group("type"))
                table = expected.setdefault(name, ExpectedTable(name=name))
                table.columns[col] = col_type
                continue
            drop_match = _RE_DROP_COLUMN.match(stmt)
            if drop_match is not None:
                name = _extract_table_name(drop_match.group("table_ref"))
                col = drop_match.group("col")
                existing_table = expected.get(name)
                if existing_table is not None:
                    existing_table.columns.pop(col, None)
                continue
            log.debug("drift: ALTER TABLE shape not recognized: %s", stmt[:120])
        # CREATE INDEX / DROP INDEX do not contribute to column-level
        # drift; skip silently.
    return expected


# ---------------------------------------------------------------------
# live snapshot
# ---------------------------------------------------------------------


async def snapshot_live_schema(store: Any) -> dict[str, dict[str, str]]:
    """
    collect the live schema's tables + columns into an expected-shaped dict.

    queries information_schema for tables and columns in the current
    search_path schema. excludes the ``_schema_migrations`` bookkeeping
    table since drift checking against it would always flag it as
    "extra."

    :param store: DataStore-shape with execute/query surface
    :ptype store: Any
    :return: mapping of table_name -> (column_name -> data_type)
    :rtype: dict[str, dict[str, str]]
    """
    table_rows = await store.query(_INFO_SCHEMA_TABLES_SQL)
    col_rows = await store.query(_INFO_SCHEMA_COLUMNS_SQL)
    live: dict[str, dict[str, str]] = {}
    for row in table_rows:
        name = row["table_name"]
        if name == "_schema_migrations":
            continue
        live[name] = {}
    for row in col_rows:
        table = row["table_name"]
        if table not in live:
            continue
        live[table][row["column_name"]] = str(row["data_type"]).lower()
    return live


def diff_expected_live(
    expected: dict[str, ExpectedTable],
    live: dict[str, dict[str, str]],
    current_schema: str | None = None,
) -> DriftReport:
    """
    compute a structured drift report from expected vs live mappings.

    when ``current_schema`` is provided, expected tables that were
    declared with an explicit schema prefix that differs from
    ``current_schema`` are excluded from the comparison. those tables
    live in another schema and the drift check for the current schema
    cannot see them. without that filter every cross-schema table the
    migrations create would surface as ``missing_tables``, drowning
    real drift in noise.

    :param expected: parsed expectations from migration DDL
    :ptype expected: dict[str, ExpectedTable]
    :param live: live schema as snapshotted from information_schema
    :ptype live: dict[str, dict[str, str]]
    :param current_schema: the schema the live snapshot was taken from;
        used to filter out cross-schema expected tables
    :ptype current_schema: str | None
    :return: structured drift report
    :rtype: DriftReport
    """
    report = DriftReport()
    if current_schema is not None:
        scoped_expected = {
            name: tbl
            for name, tbl in expected.items()
            if tbl.schema is None or tbl.schema == current_schema
        }
    else:
        scoped_expected = expected
    expected_names = set(scoped_expected.keys())
    live_names = set(live.keys())
    for missing in sorted(expected_names - live_names):
        report.missing_tables.append(missing)
    for extra in sorted(live_names - expected_names):
        report.extra_tables.append(extra)
    for table_name in sorted(expected_names & live_names):
        exp_cols = scoped_expected[table_name].columns
        live_cols = live[table_name]
        for col in sorted(set(exp_cols) - set(live_cols)):
            report.missing_columns.append((table_name, col))
        for col in sorted(set(live_cols) - set(exp_cols)):
            report.extra_columns.append((table_name, col, live_cols[col]))
        for col in sorted(set(exp_cols) & set(live_cols)):
            expected_type = exp_cols[col]
            actual_type = live_cols[col]
            if expected_type != actual_type:
                report.type_mismatches.append(
                    (table_name, col, expected_type, actual_type)
                )
    return report
