"""what one SQL statement reads or writes whole: the text half of the no-whole-reads rule.

A statement is judged from its text alone, each placeholder of an f-string read as ``{}``. Every ``SELECT`` in
it (a subquery, a part of a ``UNION``, the read of an ``INSERT ... SELECT``) is judged on its own level: the
relations its ``FROM`` and ``JOIN`` name, whether a ``WHERE`` (or a join onto bound values) carries a key, and
whether it aggregates. A relation is a data table unless it is a catalog, a CTE of the statement, a function
(``unnest($1)``, ``read_parquet(...)``) or a subquery.

A predicate carries a key when it holds a bound value: a placeholder (``$1``, ``%s``, ``%(name)s``, ``?``,
``:name``), an interpolation the walk cannot see (``{}``, inside the ``WHERE`` only), or an equality with a
literal (``scope = '*'``). ``WHERE pending IS NOT NULL`` carries none: it is a scan of the table with a filter.

Kinds, each the start of a finding's symbol:

- ``whole-select``: a read of a data table with no ``WHERE`` and no ``LIMIT``;
- ``whole-aggregate``: ``count``, ``GROUP BY``, ``DISTINCT``, a sum or a fingerprint over a data table with no
  bound predicate, ``LIMIT`` or not;
- ``unbound-where``: a ``WHERE`` that holds no bound value;
- ``truncate``, ``whole-delete``, ``whole-update``: a write to every row;
- ``whole-maintenance``: ``ANALYZE`` / ``VACUUM`` of a table, which reads it whole;
- ``whole-copy-out``: ``COPY <table> TO``.

A ``SELECT ... FROM <table>`` with nothing after the table is a *head* (:data:`HEAD`): the caller adds the
predicate (a keyset pager does), so a head is judged where it is run, not where it is written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = [
    "HEAD",
    "KEYED_COUNT",
    "is_sql",
    "judge_sql",
]

#: the kind a bare ``SELECT ... FROM <table>`` is given: judged where it runs
HEAD: Final[str] = "head"

#: the kind a count or fingerprint narrowed by a key is given: a verification scan of the key's rows, judged only in
#: the code a repo names as load code (``verify-aggregate``)
KEYED_COUNT: Final[str] = "keyed-count"

#: a count or a fingerprint: the shapes of re-verifying data by scanning it
_VERIFY: Final[re.Pattern[str]] = re.compile(
    r"\b(?:count|md5|sha256|hashtext|bit_xor|string_agg|checksum|fingerprint\w*)\s*\(", re.IGNORECASE
)

_BLOCK_COMMENT: Final[re.Pattern[str]] = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT: Final[re.Pattern[str]] = re.compile(r"--[^\n]*")
_QUOTED: Final[re.Pattern[str]] = re.compile(r"'(?:[^']|'')*'")

#: the start of a statement: a verb, then the shape that verb has in SQL (so English prose is not SQL)
_STATEMENT: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:"
    r"(?:WITH\b.*\bSELECT\b.*\bFROM\b)|(?:SELECT\b.+\bFROM\s)|(?:INSERT\s+(?:OR\s+\w+\s+)?INTO\s)|"
    r"(?:UPDATE\s+[\w.\"{}]+(?:\s+(?:AS\s+)?\w+)?\s+SET\s)|(?:DELETE\s+FROM\s)|(?:TRUNCATE\b)|(?:MERGE\s+INTO\s)|"
    r"(?:COPY\s)|(?:CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP\w*\s+)?(?:TABLE|VIEW)\b.*\bAS\b)|"
    r"(?:ANALYZE\b)|(?:VACUUM\b)"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_NAME: Final[str] = r"((?:\"[^\"]+\"|[\w{}]+)(?:\.(?:\"[^\"]+\"|[\w{}]+))*)"
_SELECT: Final[re.Pattern[str]] = re.compile(r"\bSELECT\b", re.IGNORECASE)
_SET_OPERATOR: Final[re.Pattern[str]] = re.compile(r"\b(UNION|INTERSECT|EXCEPT)\b", re.IGNORECASE)
_FROM_OR_JOIN: Final[re.Pattern[str]] = re.compile(rf"\b(FROM|JOIN)\s+(?:ONLY\s+|LATERAL\s+)?(\(|{_NAME})", re.I)
_WHERE: Final[re.Pattern[str]] = re.compile(r"\bWHERE\b", re.IGNORECASE)
_AFTER_WHERE: Final[re.Pattern[str]] = re.compile(
    r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT|OFFSET|HAVING|WINDOW|RETURNING|FOR\s+(?:UPDATE|SHARE|NO\s+KEY)|ON\s+CONFLICT|"
    r"QUALIFY)\b",
    re.IGNORECASE,
)
_LIMITED: Final[re.Pattern[str]] = re.compile(r"\b(LIMIT|FETCH\s+FIRST|FETCH\s+NEXT|TOP)\b", re.IGNORECASE)
_AGGREGATE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:GROUP\s+BY|DISTINCT)\b|\b(?:count|sum|avg|min|max|string_agg|array_agg|bit_xor|bit_or|md5|sha256|"
    r"hashtext|listagg|median|bool_and|bool_or|checksum|fingerprint\w*|approx_count_distinct)\s*\(",
    re.IGNORECASE,
)
_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\$\d+|%s|%\(\w+\)s|\?|(?<![:\w]):[A-Za-z_]\w*")
_LITERAL_EQUALITY: Final[re.Pattern[str]] = re.compile(
    r"(?<![<>!])=\s*(?:'_'|(?:-?\d+(?:\.\d+)?|TRUE|FALSE)\b)|\bIN\s*\(\s*(?:'_'|-?\d)|\bFALSE\b|\b1\s*=\s*0\b", re.I
)
_INTERPOLATION: Final[str] = "{}"
#: a predicate no row passes: a part that only types a ``UNION``'s columns
_NO_ROWS: Final[re.Pattern[str]] = re.compile(r"WHERE\s+(?:1\s*=\s*0|FALSE)", re.IGNORECASE)

#: an equality between two columns, one qualified: in a subquery, a key from the outer row
_COLUMN_EQUALITY: Final[re.Pattern[str]] = re.compile(
    r"[\w\"]+(?:\.[\w\"]+)?\s*=\s*[\w\"]+\.[\w\"]+|[\w\"]+\.[\w\"]+\s*=\s*[\w\"]+"
)
_CATALOG: Final[re.Pattern[str]] = re.compile(
    r"(?i)^\"?(pg_|information_schema|sqlite_|duckdb_|sys\.|svv_|stl_|stv_|svl_|pg_catalog|dual$|"
    r"generate_series|unnest|values$)"
)
_VIEW: Final[re.Pattern[str]] = re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP\w*\s+)?VIEW\b", re.I)
_CTE: Final[re.Pattern[str]] = re.compile(
    r"(?:\bWITH(?:\s+RECURSIVE)?|,)\s*(\"?[\w{}]+\"?)\s*(?:\([^()]*\)\s*)?AS\s*(?:NOT\s+)?(?:MATERIALIZED\s+)?\(",
    re.IGNORECASE,
)
_TRUNCATE: Final[re.Pattern[str]] = re.compile(rf"^\s*TRUNCATE\s+(?:TABLE\s+)?(?:ONLY\s+)?{_NAME}", re.I)
_DELETE: Final[re.Pattern[str]] = re.compile(rf"\bDELETE\s+FROM\s+(?:ONLY\s+)?{_NAME}", re.I)
_UPDATE: Final[re.Pattern[str]] = re.compile(
    rf"(?<!FOR )(?<!DO )\bUPDATE\s+(?:ONLY\s+)?{_NAME}(?:\s+(?:AS\s+)?\w+)?\s+SET\b", re.I
)
_MAINTENANCE: Final[re.Pattern[str]] = re.compile(rf"^\s*(ANALYZE|VACUUM)(?:\s+\(?[A-Z, ]*\)?)?\s+{_NAME}", re.I)
_COPY_OUT: Final[re.Pattern[str]] = re.compile(rf"^\s*COPY\s+{_NAME}(?:\s*\([^)]*\))?\s+TO\b", re.I)


@dataclass(frozen=True)
class _SqlFinding:
    """one way a statement reads or writes whole.

    :ivar kind: the kind (module docstring), or :data:`HEAD`
    :ivar table: the relation, as written (``{}`` for an interpolated name)
    """

    kind: str
    table: str


def _clean(sql: str) -> str:
    """the statement without comments, each quoted literal as ``'_'``, on one line.

    :param sql: the text
    :ptype sql: str
    :return: the cleaned text
    :rtype: str
    """
    text = _LINE_COMMENT.sub(" ", _BLOCK_COMMENT.sub(" ", sql))
    return " ".join(_QUOTED.sub("'_'", text).split())


def is_sql(text: str) -> bool:
    """whether ``text`` starts as a SQL statement this rule judges: its verb upper case (``select`` and ``with`` may be
    lower case; ``Select ...`` and ``delete from ...`` are prose) and the shape that verb has in SQL.

    :param text: the text
    :ptype text: str
    :return: whether it does
    :rtype: bool
    """
    cleaned = _clean(text)
    verb = cleaned.split(" ", 1)[0] if cleaned else ""
    cased = verb.isupper() or (verb.islower() and verb in {"select", "with"})
    return cased and _STATEMENT.match(cleaned) is not None


def _statement_start(text: str) -> int:
    """where a statement that is a bare read starts its ``SELECT``: 0 when it starts ``SELECT``, else -1 (a write's
    read, or a ``WITH``'s, is never a head).

    :param text: cleaned SQL
    :ptype text: str
    :return: the index
    :rtype: int
    """
    return 0 if re.match(r"SELECT\b", text, re.IGNORECASE) is not None else -1


def _depths(text: str) -> list[int]:
    """each character's parenthesis depth (an opening parenthesis is at its outer depth).

    :param text: cleaned SQL
    :ptype text: str
    :return: one depth per character
    :rtype: list[int]
    """
    depths: list[int] = []
    depth = 0
    for char in text:
        if char == ")":
            depth = max(0, depth - 1)
        depths.append(depth)
        if char == "(":
            depth += 1
    return depths


def _own_level(text: str, depths: list[int], start: int, depth: int) -> tuple[int, str]:
    """the end of the clause starting at ``start``, and its text with every deeper level blanked.

    :param text: cleaned SQL
    :ptype text: str
    :param depths: :func:`_depths` of it
    :ptype depths: list[int]
    :param start: where the clause starts
    :ptype start: int
    :param depth: its depth
    :ptype depth: int
    :return: ``(end, masked)``: the clause is ``text[start:end]``; ``masked`` is it with deeper text as spaces
    :rtype: tuple[int, str]
    """
    end = len(text)
    for index in range(start + 1, len(text)):
        if depths[index] < depth:
            end = index
            break
    masked = "".join(char if depths[start + offset] <= depth else " " for offset, char in enumerate(text[start:end]))
    operator = _SET_OPERATOR.search(masked)
    if operator is not None:
        end = start + operator.start()
        masked = masked[: operator.start()]
    return end, masked


def _is_data(name: str, after: str, ctes: set[str]) -> bool:
    """whether a relation a ``FROM`` or ``JOIN`` names is a data table.

    :param name: the relation as written
    :ptype name: str
    :param after: the text right after it
    :ptype after: str
    :param ctes: the statement's CTE names
    :ptype ctes: set[str]
    :return: False for a catalog, a CTE or a function call
    :rtype: bool
    """
    bare = name.strip('"')
    return not (
        _CATALOG.match(name) is not None or bare in ctes or bare.lower() in ctes or after.lstrip().startswith("(")
    )


def _bound(span: str, where: str | None, *, correlated: bool = False) -> bool:
    """whether a clause's predicate carries a key.

    :param span: the clause from its first relation on (nested levels included)
    :ptype span: str
    :param where: its ``WHERE`` text, or None when it has none
    :ptype where: str | None
    :param correlated: whether the clause is a subquery, where an equality with an outer row's column is a key
    :ptype correlated: bool
    :return: whether a bound value narrows it
    :rtype: bool
    """
    return (
        _PLACEHOLDER.search(span) is not None
        or (where is not None and (_INTERPOLATION in where or _LITERAL_EQUALITY.search(where) is not None))
        or _LITERAL_EQUALITY.search(span) is not None
        or (correlated and where is not None and _COLUMN_EQUALITY.search(where) is not None)
    )


def _outer_bound(text: str, depths: list[int], start: int, depth: int) -> bool:
    """whether a derived table (``FROM (SELECT ...) alias``) is narrowed by its outer ``WHERE``, which the planner
    pushes into it.

    :param text: cleaned SQL
    :ptype text: str
    :param depths: :func:`_depths` of it
    :ptype depths: list[int]
    :param start: where the inner ``SELECT`` starts
    :ptype start: int
    :param depth: its depth
    :ptype depth: int
    :return: whether it is a derived table whose outer clause holds a bound ``WHERE``
    :rtype: bool
    """
    derived = depth > 0 and re.search(r"\b(FROM|JOIN)\s*\(\s*$", text[:start], re.IGNORECASE) is not None
    bound = False
    if derived:
        close = next((i for i in range(start, len(text)) if depths[i] < depth), len(text))
        _, outer = _own_level(text, depths, close, depth - 1)
        where_match = _WHERE.search(outer)
        if where_match is not None:
            closing = _AFTER_WHERE.search(outer, where_match.end())
            where = text[close + where_match.start() : close + (closing.start() if closing else len(outer))]
            bound = _bound(where, where)
    return bound


def _judge_select(text: str, depths: list[int], start: int, ctes: set[str]) -> _SqlFinding | None:
    """judge one ``SELECT`` at its own level.

    :param text: cleaned SQL
    :ptype text: str
    :param depths: :func:`_depths` of it
    :ptype depths: list[int]
    :param start: where the ``SELECT`` keyword starts
    :ptype start: int
    :param ctes: the statement's CTE names
    :ptype ctes: set[str]
    :return: the finding, or None when it reads by a key, reads no data table, or is bounded by a ``LIMIT``
    :rtype: _SqlFinding | None
    """
    depth = depths[start]
    end, masked = _own_level(text, depths, start, depth)
    relations = [
        (match.group(3), match.start(), match.end())
        for match in _FROM_OR_JOIN.finditer(masked)
        if match.group(2) != "(" and _is_data(match.group(3), text[start + match.end() : start + match.end() + 2], ctes)
    ]
    finding: _SqlFinding | None = None
    if relations:
        table, first, table_end = relations[0]
        opening = next(match.start() for match in _FROM_OR_JOIN.finditer(masked))
        where_match = _WHERE.search(masked)
        where: str | None = None
        if where_match is not None:
            closing = _AFTER_WHERE.search(masked, where_match.end())
            where = text[start + where_match.start() : start + (closing.start() if closing else len(masked))]
        clauses = _AFTER_WHERE.search(masked, table_end)
        # text interpolated between the table and its clauses (``FROM t {where}``, ``FROM t{filters}``) may be the
        # predicate: the walk cannot see it, so it does not judge the read
        unseen = _INTERPOLATION in masked[table_end : clauses.start() if clauses else len(masked)] or (
            table.endswith(_INTERPOLATION) and len(table) > len(_INTERPOLATION)
        )
        table = table.removesuffix(_INTERPOLATION) or table
        bound = (
            unseen
            or _bound(text[start + opening : end], where, correlated=depth > 0)
            or _outer_bound(text, depths, start, depth)
        )
        aggregate = _AGGREGATE.search(masked[:first]) is not None or re.search(r"\bGROUP\s+BY\b", masked, re.IGNORECASE)
        limited = _LIMITED.search(masked) is not None
        if where is not None and _NO_ROWS.fullmatch(where.strip()) is not None:
            finding = None
        elif not masked[table_end:].strip() and start == _statement_start(text) and not aggregate and not unseen:
            finding = _SqlFinding(HEAD, table)
        elif not bound and aggregate:
            finding = _SqlFinding("whole-aggregate", table)
        elif where is None and not limited and not bound:
            finding = _SqlFinding("whole-select", table)
        elif where is not None and not bound:
            finding = _SqlFinding("unbound-where", table)
        elif _VERIFY.search(masked[:first]) is not None:
            finding = _SqlFinding(KEYED_COUNT, table)
    return finding


def _judge_write(text: str, depths: list[int]) -> list[_SqlFinding]:
    """the whole-table writes and maintenance in one statement.

    :param text: cleaned SQL
    :ptype text: str
    :param depths: :func:`_depths` of it
    :ptype depths: list[int]
    :return: the findings
    :rtype: list[_SqlFinding]
    """
    findings: list[_SqlFinding] = []
    truncate = _TRUNCATE.match(text)
    if truncate is not None:
        findings.append(_SqlFinding("truncate", truncate.group(1)))
    maintenance = _MAINTENANCE.match(text)
    if maintenance is not None and _CATALOG.match(maintenance.group(2)) is None:
        findings.append(_SqlFinding("whole-maintenance", maintenance.group(2)))
    copy_out = _COPY_OUT.match(text)
    if copy_out is not None:
        findings.append(_SqlFinding("whole-copy-out", copy_out.group(1)))
    for pattern, kind in ((_DELETE, "whole-delete"), (_UPDATE, "whole-update")):
        for match in pattern.finditer(text):
            depth = depths[match.start()]
            _, masked = _own_level(text, depths, match.start(), depth)
            relative = match.end() - match.start()
            where_match = _WHERE.search(masked, relative)
            where: str | None = None
            if where_match is not None:
                closing = _AFTER_WHERE.search(masked, where_match.end())
                where = text[
                    match.start() + where_match.start() : match.start() + (closing.start() if closing else len(masked))
                ]
            clauses = _AFTER_WHERE.search(masked, relative)
            unseen = match.group(1).endswith(_INTERPOLATION) or (
                _INTERPOLATION in masked[relative : clauses.start() if clauses else len(masked)]
                and kind == "whole-delete"
            )
            if where is None and unseen:
                continue
            if where is None:
                findings.append(_SqlFinding(kind, match.group(1)))
            elif not _bound(where, where):
                findings.append(_SqlFinding("unbound-where", match.group(1)))
    return findings


def judge_sql(sql: str) -> list[_SqlFinding]:
    """every way one statement reads or writes whole.

    :param sql: the statement, each interpolation as ``{}``
    :ptype sql: str
    :return: the findings in text order; a lone :data:`HEAD` for a bare ``SELECT ... FROM <table>``; empty when
        every read and write carries a key
    :rtype: list[_SqlFinding]
    """
    text = _clean(sql)
    depths = _depths(text)
    ctes = {match.group(1).strip('"') for match in _CTE.finditer(text) if depths[match.start()] == 0}
    ctes |= {name.lower() for name in ctes}
    findings = _judge_write(text, depths)
    if _VIEW.match(text) is None:
        for match in _SELECT.finditer(text):
            judged = _judge_select(text, depths, match.start(), ctes)
            if judged is not None:
                findings.append(judged)
    return findings
