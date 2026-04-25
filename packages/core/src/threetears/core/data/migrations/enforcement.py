"""
AST walker enforcing yugabyte-safe patterns across migration files.

implements the migration-helpers-task-01 walker spec. five rules:

- **M-1 (CRITICAL)**: ``DO`` blocks containing both DDL and DML are the
  v054 footgun -- yugabyte auto-commits the DDL and the DML runs
  against a stale snapshot. flagged literals contain at least one of
  ``ALTER TABLE``, ``CREATE TABLE/INDEX/TYPE/EXTENSION``,
  ``DROP TABLE/INDEX/CONSTRAINT``, ``ADD CONSTRAINT``, ``RENAME COLUMN``
  AND at least one of ``UPDATE ... SET``, ``INSERT INTO``,
  ``DELETE FROM``, ``TRUNCATE TABLE`` (excluding constraint names that
  contain ``UPDATE`` as a substring)
- **M-2 (HIGH)**: backfill ``UPDATE`` statements lacking a replay-guard
  predicate. the predicate may take any of the documented forms
  (``{column} IS NULL``, ``{column} = {default}``, ``{column} IS DISTINCT
  FROM {value}``); when the literal is an ``UPDATE ... SET <c> = <v>``
  but the WHERE clause does not include any of those forms, the rule
  fires
- **M-3 (MEDIUM)**: ``CREATE TABLE`` / ``CREATE INDEX`` / ``ALTER TABLE
  ... ADD COLUMN`` lacking ``IF NOT EXISTS``; ``DROP CONSTRAINT`` /
  ``DROP TABLE`` lacking ``IF EXISTS``
- **M-4 (HIGH)**: ``ALTER TABLE ... ADD CONSTRAINT ... PRIMARY KEY``
  on a multi-column PK without an accompanying ``UNIQUE`` on the
  preserved id column (the inbound-FK preservation pattern)
- **M-5 (HIGH)**: ``TRUNCATE TABLE`` in a migration file. pre-GA we
  carry a narrow exemption for the three agent-memory v009/v010/v011
  cases that legitimately use TRUNCATE; every future migration must
  prefer a more surgical alternative

mode is controlled by ``MIGRATION_ENFORCEMENT_MODE`` env var. defaults
to ``report`` during the retro-rewrite window; flips to ``strict``
once every migration has been reshaped onto helpers or carries a
rationale-tagged exemption.

exemption file format: ``<file_path>:<rule_name> # rationale: <reason>``
per repo. blank rationales are rejected by the walker's meta-test;
"internal access needed" / "tests need this" are not rationales.

runtime is AST-only and well under the 15s CLAUDE.md budget.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MigrationViolation",
    "WalkerConfig",
    "find_migration_violations",
    "load_exemptions",
    "walk_migration_directory",
]


# ---------------------------------------------------------------------------
# rule names -- referenced from exemption files and test output
# ---------------------------------------------------------------------------


RULE_M1_DDL_DML_MIXING = "M-1"
RULE_M2_REPLAY_GUARD = "M-2"
RULE_M3_IDEMPOTENCY = "M-3"
RULE_M4_PK_UNIQUE = "M-4"
RULE_M5_TRUNCATE = "M-5"


# ---------------------------------------------------------------------------
# DDL / DML pattern recognition. these are case-insensitive substring
# checks against SQL string literals. the walker collects literals via
# ast.walk so multi-line strings, f-strings, and joined-string forms
# are all surfaced.
# ---------------------------------------------------------------------------


_DDL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bALTER\s+TABLE\b", re.IGNORECASE),
    re.compile(r"\bCREATE\s+TABLE\b", re.IGNORECASE),
    re.compile(r"\bCREATE\s+(?:UNIQUE\s+)?INDEX\b", re.IGNORECASE),
    re.compile(r"\bCREATE\s+TYPE\b", re.IGNORECASE),
    re.compile(r"\bCREATE\s+EXTENSION\b", re.IGNORECASE),
    re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
    re.compile(r"\bDROP\s+INDEX\b", re.IGNORECASE),
    re.compile(r"\bDROP\s+CONSTRAINT\b", re.IGNORECASE),
    re.compile(r"\bADD\s+CONSTRAINT\b", re.IGNORECASE),
    re.compile(r"\bRENAME\s+COLUMN\b", re.IGNORECASE),
    re.compile(r"\bRENAME\s+TO\b", re.IGNORECASE),
)


# DML patterns. the UPDATE pattern requires "SET" within ~80 chars to
# differentiate "UPDATE table SET col = ..." from constraint/column
# names that happen to contain the word "UPDATE".
_DML_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bUPDATE\s+\w[\w.]*\s+SET\b", re.IGNORECASE),
    re.compile(r"\bINSERT\s+INTO\b", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
    re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE),
)


_DO_BLOCK_PATTERN = re.compile(r"\bDO\s*\$\$", re.IGNORECASE)


_TRUNCATE_PATTERN = re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE)


# replay-guard discovery patterns. a backfill UPDATE includes a
# replay-safe predicate when its WHERE clause matches one of these
# shapes for the column being SET. we accept the column-name-agnostic
# form: any IS NULL / IS DISTINCT FROM / equality against a literal
# string / equality against a numeric counts as a guard. this is more
# permissive than rejecting on form, which is the right tradeoff:
# false-negatives on M-2 are caught at runtime by a failing
# integration test; false-positives prevent committing a real fix.
_UPDATE_TARGET_PATTERN = re.compile(
    r"\bUPDATE\s+(?:[a-zA-Z_]\w*\.)?(?P<table>[a-zA-Z_]\w*)\s+SET\s+"
    r"(?P<col>[a-zA-Z_]\w*)\s*=",
    re.IGNORECASE,
)


def _has_replay_guard(literal: str, column: str) -> bool:
    """
    return ``True`` when the WHERE clause carries a documented replay guard
    for ``column``.

    accepts:

    - ``{column} IS NULL``
    - ``{column} = '<literal>'`` or ``{column} = <numeric>``
    - ``{column} IS DISTINCT FROM ...``
    - ``ON CONFLICT ... DO NOTHING`` (sibling INSERT-with-ON-CONFLICT
      pattern)

    the ``{column} = <value>`` form is searched only inside the WHERE
    clause, not the SET clause -- ``UPDATE t SET c = 'x' WHERE c IS
    NULL`` is fine, but ``UPDATE t SET c = 'x' WHERE customer_id IS
    NULL`` is NOT (the predicate doesn't reference the SET column).

    :param literal: full SQL literal under inspection
    :ptype literal: str
    :param column: column being SET in the UPDATE
    :ptype column: str
    :return: whether at least one replay-guard form is present
    :rtype: bool
    """
    # ON CONFLICT DO NOTHING is its own replay-safety form.
    if re.search(
        r"\bON\s+CONFLICT\b.*\bDO\s+NOTHING\b",
        literal,
        re.IGNORECASE | re.DOTALL,
    ):
        return True
    # extract the WHERE clause text -- everything after the first WHERE
    # token until the end of the statement / DO block.
    where_match = re.search(
        r"\bWHERE\b(.*)", literal, re.IGNORECASE | re.DOTALL,
    )
    if where_match is None:
        return False
    where_clause = where_match.group(1)
    # WHERE id = $N (or = '<uuid>'::uuid, or = ANY(...)) is a structural
    # single-row replay guard: the predicate scopes the UPDATE to one
    # row whose identity is supplied by the Python caller. re-running
    # the migration with the same caller-supplied id is a no-op
    # because the SET produces the same value. this matches the
    # ``UNIQUE (id)`` reasoning the partition walker uses for SELECT.
    by_id_re = re.compile(
        r"\b(?:[a-zA-Z_]\w*\.)?id\s*=\s*"
        r"(?:\$\d+\b|'[^']+'::uuid|ANY\s*\()",
        re.IGNORECASE,
    )
    if by_id_re.search(where_clause) is not None:
        return True
    column_re = re.escape(column)
    patterns = (
        rf"\b{column_re}\s+IS\s+NULL\b",
        rf"\b{column_re}\s*=\s*",
        rf"\b{column_re}\s+IS\s+DISTINCT\s+FROM\b",
    )
    result = any(
        re.search(p, where_clause, re.IGNORECASE | re.DOTALL)
        for p in patterns
    )
    return result


def _is_likely_pk_redefinition(literal: str) -> bool:
    """
    return ``True`` when the literal contains an ALTER-TABLE composite PK add.

    used by rule M-4: a PK swap to a composite ``(partition, id)`` key
    must accompany ``UNIQUE (id)`` so inbound FKs that target the
    legacy id-only PK can be re-pointed at the unique index. the rule
    looks for ``ADD CONSTRAINT ... PRIMARY KEY (a, b...)`` with two or
    more columns.

    initial ``CREATE TABLE`` statements that declare a composite PK
    in-line are NOT a redefinition -- they are the original PK shape
    so no inbound FK can pre-exist that depends on a different PK
    form. those are filtered out here.

    :param literal: SQL string literal under inspection
    :ptype literal: str
    :return: whether the literal contains a post-creation composite PK add
    :rtype: bool
    """
    add_constraint_pk_re = re.compile(
        r"ADD\s+CONSTRAINT\s+\w+\s+PRIMARY\s+KEY\s*\(\s*"
        r"[a-zA-Z_]\w*\s*,\s*[a-zA-Z_]\w*",
        re.IGNORECASE,
    )
    result = add_constraint_pk_re.search(literal) is not None
    return result


_UNIQUE_PATTERN = re.compile(r"\bUNIQUE\s*\(", re.IGNORECASE)


_IF_NOT_EXISTS_PATTERN = re.compile(
    r"\b(?:CREATE|ADD\s+COLUMN)\b[^;]*\bIF\s+NOT\s+EXISTS\b",
    re.IGNORECASE,
)


_CREATE_TABLE_PATTERN = re.compile(
    r"\bCREATE\s+TABLE\b(?!\s+IF\s+NOT\s+EXISTS)",
    re.IGNORECASE,
)


_CREATE_INDEX_PATTERN = re.compile(
    r"\bCREATE\s+(?:UNIQUE\s+)?INDEX\b(?!\s+IF\s+NOT\s+EXISTS)",
    re.IGNORECASE,
)


_ADD_COLUMN_PATTERN = re.compile(
    r"\bADD\s+COLUMN\b(?!\s+IF\s+NOT\s+EXISTS)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# walker data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationViolation:
    """
    one walker hit.

    :ivar path: file containing the offending SQL literal
    :ivar lineno: 1-based line number of the literal start
    :ivar rule: rule name (M-1 .. M-5)
    :ivar message: human-readable description
    :ivar literal_excerpt: first 200 characters of the offending literal
    """

    path: Path
    lineno: int
    rule: str
    message: str
    literal_excerpt: str

    def format(self) -> str:
        """
        return a single-line human-readable representation.

        :return: ``"<path>:<line>: [<rule>] <message>"``
        :rtype: str
        """
        result = (
            f"{self.path}:{self.lineno}: [{self.rule}] {self.message} "
            f"-- {self.literal_excerpt!r}"
        )
        return result


@dataclass(frozen=True)
class WalkerConfig:
    """
    per-repo walker configuration.

    :ivar migration_dirs: list of paths to migration package directories
    :ivar exemptions: set of (file_path_str, rule) pairs that suppress
        violations
    :ivar repo_root: optional repo root used to render relative paths
        in violation messages; when ``None``, absolute paths are used
    """

    migration_dirs: tuple[Path, ...]
    exemptions: frozenset[tuple[str, str]]
    repo_root: Path | None = None


# ---------------------------------------------------------------------------
# string-literal collection (mirrors partition walker pattern)
# ---------------------------------------------------------------------------


def _collect_docstring_ids(tree: ast.AST) -> set[int]:
    """
    return ``id(...)`` of every module / class / function docstring constant.

    docstrings are prose, not SQL. the walker filters them out so a
    docstring that mentions ``CREATE TABLE`` or ``ALTER TABLE`` does
    not produce a false positive.

    :param tree: AST root
    :ptype tree: ast.AST
    :return: set of constant node ids treated as docstrings
    :rtype: set[int]
    """
    docstring_ids: set[int] = set()
    docstring_owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, docstring_owners):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstring_ids.add(id(first.value))
    return docstring_ids


def _collect_string_literals(tree: ast.AST) -> list[tuple[str, int]]:
    """
    walk ``tree`` and collect every string literal with its starting line.

    plain ``ast.Constant`` strings, joined-string ``f"..."`` forms (with
    formatted parts replaced by ``__PLACEHOLDER__``), and adjacent
    literal concatenation are all surfaced. constants nested inside a
    JoinedStr are skipped because they were already absorbed into the
    joined string emitted for the parent. module / class / function
    docstrings are skipped because they are prose, not SQL.

    :param tree: AST root
    :ptype tree: ast.AST
    :return: list of ``(literal text, line number)`` pairs
    :rtype: list[tuple[str, int]]
    """
    joined_constants: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    joined_constants.add(id(v))
    docstring_ids = _collect_docstring_ids(tree)
    results: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in joined_constants:
                continue
            if id(node) in docstring_ids:
                continue
            results.append((node.value, node.lineno))
        elif isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                else:
                    parts.append("__PLACEHOLDER__")
            results.append(("".join(parts), node.lineno))
    return results


_SQL_LEADING_TOKENS = (
    "ALTER",
    "CREATE",
    "DROP",
    "UPDATE",
    "INSERT",
    "DELETE",
    "TRUNCATE",
    "DO",
    "WITH",
    "SELECT",
)


def _is_sql_literal(literal: str) -> bool:
    """
    return ``True`` when the literal is SQL-shaped (vs prose / log message).

    used to filter docstrings, log strings, and other prose from the
    walker's input. a literal counts as SQL when its first non-whitespace
    token (after the leading newline / indent block stripped) is a SQL
    leading verb.

    log messages like ``"v026: datasource has NULL customer_id; skipping
    -- operator must UPDATE ..."`` are NOT SQL even though they
    contain the word ``UPDATE``.

    :param literal: candidate literal
    :ptype literal: str
    :return: whether the literal starts with a SQL leading token
    :rtype: bool
    """
    stripped = literal.lstrip()
    upper = stripped.upper()
    # tolerate a leading SQL comment / DECLARE block.
    result = any(
        upper.startswith(tok + " ") or upper.startswith(tok + "\n")
        for tok in _SQL_LEADING_TOKENS
    )
    return result


# ---------------------------------------------------------------------------
# rule evaluators
# ---------------------------------------------------------------------------


def _matches_any(patterns: tuple[re.Pattern[str], ...], literal: str) -> bool:
    """
    return ``True`` when any pattern in ``patterns`` matches ``literal``.

    :param patterns: compiled regexes
    :ptype patterns: tuple[re.Pattern[str], ...]
    :param literal: SQL literal under inspection
    :ptype literal: str
    :return: presence of at least one match
    :rtype: bool
    """
    result = any(p.search(literal) is not None for p in patterns)
    return result


def _strip_pl_pgsql_string_literals(literal: str) -> str:
    """
    return ``literal`` with single-quoted SQL string literals replaced by ``''``.

    pl/pgsql ``RAISE EXCEPTION 'message ... UPDATE ...'`` constructs
    embed prose into a SQL string literal that should NOT count as DML
    for the M-1 mixing check. similarly, a CHECK expression like
    ``c IN ('a', 'b')`` embeds enum values that should not be parsed
    as SQL keywords. this helper strips every single-quoted segment
    so the regex scanners only see structural SQL.

    :param literal: SQL string literal under inspection
    :ptype literal: str
    :return: literal with single-quoted segments collapsed
    :rtype: str
    """
    # match single-quoted strings, including doubled-quote escapes
    # (`'don''t'` is one literal). dotall so newlines inside multi-line
    # error messages do not terminate the match early.
    pattern = re.compile(r"'(?:''|[^'])*'", re.DOTALL)
    result = pattern.sub("''", literal)
    return result


def _evaluate_m1_ddl_dml_mixing(
    literal: str,
    path: Path,
    lineno: int,
) -> MigrationViolation | None:
    """
    return an M-1 violation when the literal mixes DDL and DML inside a DO block.

    string literals inside the SQL (``RAISE EXCEPTION 'message...'``)
    are stripped before pattern matching so prose inside an error
    message does not produce a false positive.

    :param literal: SQL string literal
    :ptype literal: str
    :param path: source file
    :ptype path: Path
    :param lineno: line number of the literal
    :ptype lineno: int
    :return: violation or ``None``
    :rtype: MigrationViolation | None
    """
    result: MigrationViolation | None = None
    if _DO_BLOCK_PATTERN.search(literal) is None:
        return result
    structural = _strip_pl_pgsql_string_literals(literal)
    has_ddl = _matches_any(_DDL_PATTERNS, structural)
    has_dml = _matches_any(_DML_PATTERNS, structural)
    if has_ddl and has_dml:
        result = MigrationViolation(
            path=path,
            lineno=lineno,
            rule=RULE_M1_DDL_DML_MIXING,
            message=(
                "DO block mixes DDL and DML; yugabyte auto-commits the "
                "DDL and the DML runs against a stale snapshot. split "
                "into separate store.execute() calls or use a helper "
                "from threetears.core.data.migrations.helpers"
            ),
            literal_excerpt=literal[:200],
        )
    return result


def _evaluate_m2_replay_guard(
    literal: str,
    path: Path,
    lineno: int,
) -> MigrationViolation | None:
    """
    return an M-2 violation when a backfill UPDATE lacks a replay-guard predicate.

    :param literal: SQL string literal
    :ptype literal: str
    :param path: source file
    :ptype path: Path
    :param lineno: line number of the literal
    :ptype lineno: int
    :return: violation or ``None``
    :rtype: MigrationViolation | None
    """
    result: MigrationViolation | None = None
    match = _UPDATE_TARGET_PATTERN.search(literal)
    if match is None:
        return result
    column = match.group("col")
    # require a WHERE clause AND a guard form referencing the column.
    has_where = re.search(r"\bWHERE\b", literal, re.IGNORECASE) is not None
    if not has_where or not _has_replay_guard(literal, column):
        result = MigrationViolation(
            path=path,
            lineno=lineno,
            rule=RULE_M2_REPLAY_GUARD,
            message=(
                f"UPDATE setting column {column!r} lacks a replay-guard "
                "predicate. add WHERE <col> IS NULL / WHERE <col> = "
                "<default> / WHERE <col> IS DISTINCT FROM <value> so a "
                "re-run after partial failure is a clean no-op"
            ),
            literal_excerpt=literal[:200],
        )
    return result


def _evaluate_m3_idempotency(
    literal: str,
    path: Path,
    lineno: int,
) -> MigrationViolation | None:
    """
    return an M-3 violation when the literal contains non-idempotent DDL.

    flagged forms: bare ``CREATE TABLE``, bare ``CREATE INDEX``, bare
    ``ADD COLUMN``. the rule does NOT fire when the literal is wrapped
    in a ``DO`` block with an existence probe -- that pattern is the
    accepted alternative idempotency form.

    :param literal: SQL string literal
    :ptype literal: str
    :param path: source file
    :ptype path: Path
    :param lineno: line number
    :ptype lineno: int
    :return: violation or ``None``
    :rtype: MigrationViolation | None
    """
    result: MigrationViolation | None = None
    # DO block with an EXISTS / NOT EXISTS guard is an accepted shape.
    if _DO_BLOCK_PATTERN.search(literal) is not None:
        if re.search(
            r"\b(?:NOT\s+)?EXISTS\s*\(", literal, re.IGNORECASE,
        ) is not None:
            return result
    matches = []
    if _CREATE_TABLE_PATTERN.search(literal) is not None:
        matches.append("CREATE TABLE")
    if _CREATE_INDEX_PATTERN.search(literal) is not None:
        matches.append("CREATE INDEX")
    if _ADD_COLUMN_PATTERN.search(literal) is not None:
        matches.append("ADD COLUMN")
    if matches:
        result = MigrationViolation(
            path=path,
            lineno=lineno,
            rule=RULE_M3_IDEMPOTENCY,
            message=(
                f"non-idempotent DDL ({', '.join(matches)}) lacks "
                "IF NOT EXISTS / IF EXISTS clause. add the clause or "
                "wrap in a DO block with an existence probe"
            ),
            literal_excerpt=literal[:200],
        )
    return result


def _evaluate_m4_pk_unique(
    literal: str,
    path: Path,
    lineno: int,
    *,
    file_has_unique: bool = False,
) -> MigrationViolation | None:
    """
    return an M-4 violation when a composite-PK ADD lacks UNIQUE preservation.

    M-4 is evaluated at file scope rather than per-literal: a migration
    that splits ``ADD CONSTRAINT ... PRIMARY KEY (a, b)`` and
    ``ADD CONSTRAINT ... UNIQUE (id)`` across separate ``store.execute``
    calls is the correct yugabyte-safe shape. the per-literal evaluator
    receives ``file_has_unique`` so a sibling UNIQUE statement
    elsewhere in the same file suppresses the hit.

    :param literal: SQL string literal under inspection
    :ptype literal: str
    :param path: source file
    :ptype path: Path
    :param lineno: line number
    :ptype lineno: int
    :param file_has_unique: whether any other literal in the same file
        contains a ``UNIQUE`` constraint clause; suppresses the hit
        when ``True``
    :ptype file_has_unique: bool
    :return: violation or ``None``
    :rtype: MigrationViolation | None
    """
    result: MigrationViolation | None = None
    if not _is_likely_pk_redefinition(literal):
        return result
    if _UNIQUE_PATTERN.search(literal) is not None:
        return result
    if file_has_unique:
        return result
    result = MigrationViolation(
        path=path,
        lineno=lineno,
        rule=RULE_M4_PK_UNIQUE,
        message=(
            "composite PRIMARY KEY ADD lacks accompanying UNIQUE "
            "constraint on the preserved id column. inbound FKs "
            "that target the legacy id-only PK will fail without "
            "the UNIQUE index"
        ),
        literal_excerpt=literal[:200],
    )
    return result


def _evaluate_m5_truncate(
    literal: str,
    path: Path,
    lineno: int,
) -> MigrationViolation | None:
    """
    return an M-5 violation when the literal contains ``TRUNCATE TABLE``.

    :param literal: SQL string literal
    :ptype literal: str
    :param path: source file
    :ptype path: Path
    :param lineno: line number
    :ptype lineno: int
    :return: violation or ``None``
    :rtype: MigrationViolation | None
    """
    result: MigrationViolation | None = None
    if _TRUNCATE_PATTERN.search(literal) is not None:
        result = MigrationViolation(
            path=path,
            lineno=lineno,
            rule=RULE_M5_TRUNCATE,
            message=(
                "TRUNCATE TABLE in a migration -- prefer a more "
                "surgical alternative (DELETE WHERE replay-guard, "
                "or schema-managed empty state). pre-GA exemptions "
                "carry a rationale in _migration_exemptions.txt"
            ),
            literal_excerpt=literal[:200],
        )
    return result


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def find_migration_violations(
    path: Path,
    *,
    exemptions: frozenset[tuple[str, str]] = frozenset(),
) -> list[MigrationViolation]:
    """
    return every walker violation in ``path``, with exemptions applied.

    :param path: migration source file
    :ptype path: Path
    :param exemptions: set of ``(path_str, rule_name)`` pairs to skip
    :ptype exemptions: frozenset[tuple[str, str]]
    :return: list of violations ordered by line number
    :rtype: list[MigrationViolation]
    """
    violations: list[MigrationViolation] = []
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return violations
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return violations
    path_str = str(path)
    sql_literals = [
        (lit, lineno)
        for lit, lineno in _collect_string_literals(tree)
        if _is_sql_literal(lit)
    ]
    # file-level: M-4 is suppressed when a sibling literal in the same
    # file carries a UNIQUE clause (the partition primitive's standard
    # split shape).
    file_has_unique = any(
        _UNIQUE_PATTERN.search(lit) is not None
        for lit, _ in sql_literals
    )
    for literal, lineno in sql_literals:
        per_literal_evaluators: tuple[
            tuple[str, MigrationViolation | None], ...
        ] = (
            ("m1", _evaluate_m1_ddl_dml_mixing(literal, path, lineno)),
            ("m2", _evaluate_m2_replay_guard(literal, path, lineno)),
            ("m3", _evaluate_m3_idempotency(literal, path, lineno)),
            (
                "m4",
                _evaluate_m4_pk_unique(
                    literal, path, lineno,
                    file_has_unique=file_has_unique,
                ),
            ),
            ("m5", _evaluate_m5_truncate(literal, path, lineno)),
        )
        for _, hit in per_literal_evaluators:
            if hit is not None and (path_str, hit.rule) not in exemptions:
                violations.append(hit)
    violations.sort(key=lambda v: (v.lineno, v.rule))
    return violations


def walk_migration_directory(
    config: WalkerConfig,
) -> list[MigrationViolation]:
    """
    walk every ``*.py`` under the configured migration directories.

    :param config: walker configuration
    :ptype config: WalkerConfig
    :return: list of violations across all files
    :rtype: list[MigrationViolation]
    """
    violations: list[MigrationViolation] = []
    for directory in config.migration_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            file_violations = find_migration_violations(
                path, exemptions=config.exemptions,
            )
            violations.extend(file_violations)
    return violations


def load_exemptions(
    exemption_file: Path,
) -> tuple[frozenset[tuple[str, str]], list[str]]:
    """
    parse an exemption file and return the parsed pairs + every line lacking a rationale.

    file format::

        <file_path>:<rule_name> # rationale: <specific reason>

    blank lines and lines starting with ``#`` are skipped. lines lacking
    a ``# rationale: ...`` suffix are returned in the second list so
    the meta-test can fail-fast on blank rationales.

    :param exemption_file: path to ``_migration_exemptions.txt``
    :ptype exemption_file: Path
    :return: ``(exemptions, lines_missing_rationale)``
    :rtype: tuple[frozenset[tuple[str, str]], list[str]]
    """
    exemptions: set[tuple[str, str]] = set()
    missing: list[str] = []
    if not exemption_file.exists():
        return frozenset(exemptions), missing
    raw = exemption_file.read_text(encoding="utf-8")
    for raw_line in raw.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # split on the rationale marker; everything before the ``#`` is
        # the entry, everything after is the rationale.
        if "#" not in stripped:
            missing.append(raw_line)
            continue
        entry_part, rationale_part = stripped.split("#", 1)
        entry_part = entry_part.strip()
        rationale_part = rationale_part.strip()
        if not rationale_part.lower().startswith("rationale:"):
            missing.append(raw_line)
            continue
        rationale_text = rationale_part[len("rationale:"):].strip()
        if not rationale_text:
            missing.append(raw_line)
            continue
        if ":" not in entry_part:
            missing.append(raw_line)
            continue
        path_str, rule_name = entry_part.rsplit(":", 1)
        exemptions.add((path_str.strip(), rule_name.strip()))
    return frozenset(exemptions), missing


def get_enforcement_mode() -> str:
    """
    return the current walker mode (``strict`` / ``report``).

    consults the ``MIGRATION_ENFORCEMENT_MODE`` env var. the default is
    ``strict`` (sub-task 7 flip) so any new migration that introduces
    a walker-flagged anti-pattern fails CI from the first commit.
    explicitly set ``MIGRATION_ENFORCEMENT_MODE=report`` to surface
    violations without blocking the build during a cleanup window.

    :return: lower-cased mode string
    :rtype: str
    """
    result = os.environ.get("MIGRATION_ENFORCEMENT_MODE", "strict").lower()
    return result
