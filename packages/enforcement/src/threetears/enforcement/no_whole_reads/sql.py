"""what one SQL statement reads or writes whole: the text half of the no-whole-reads rule.

A statement is judged from its text alone, each placeholder of an f-string read as ``{}`` (``{pk}`` when the code
names it the key: ``{pk}``, ``{key_column}``, ``{primary_key}``). Every ``SELECT`` in it (a subquery, a part of a
``UNION``, the read of an ``INSERT ... SELECT``) is judged on its own level, and so is every ``UPDATE`` and
``DELETE``: the relations its ``FROM`` and ``JOIN`` name, the predicates its ``WHERE`` and ``ON`` hold, and whether
it aggregates. A relation is a data table unless it is a catalog, a CTE of the statement, a function
(``unnest($1)``, ``read_parquet(...)``) or a subquery.

**Keyed means keyed.** A read of a data table is keyed only when one of these holds, judged against the table's
declared key and indexes (:mod:`threetears.enforcement.no_whole_reads.schema`):

- **(a) its key:** its bound equality, ``IN`` or ``= ANY`` predicates cover a leading prefix of the table's primary
  key. A bound value is a placeholder (``$1``, ``%s``, ``%(name)s``, ``?``, ``:name``), an interpolation, a literal,
  a subquery (judged on its own level), a column of an outer row (a correlated subquery), or a column of a relation
  that is itself keyed or bound (``JOIN unnest($1) k ON t.id = k.id``, a join to a CTE seeded by a bound key);
- **(b) an index declared for it:** those predicates cover the leading column of an index whose ``purpose`` names
  the function (or SQL constant) the statement is written in;
- **(c) a keyset page:** ordered by the next key column after a bound key prefix, its only other predicate a range
  on that column, and a ``LIMIT``.

A filter on anything else (a status, a kind, a flag, a type, a timestamp without the key) is a whole read with a
filter: ``non-key-where``. A table whose schema is declared nowhere has no known key, so a predicate on it proves
nothing: ``unknown-schema``. A ``LIMIT 1`` with no predicate and no order reads one row and passes.

Kinds, each the start of a finding's symbol:

- ``whole-select``: a read of a data table with no predicate, or ordered by no key under its ``LIMIT``;
- ``whole-aggregate``: ``count``, ``GROUP BY``, ``DISTINCT``, a sum or a fingerprint over a data table it does not
  read by key;
- ``unbound-where``: a ``WHERE`` that holds no bound value;
- ``non-key-where``: a ``WHERE`` with bound values, none of them a key (a), purposed index (b) or keyset page (c);
- ``unknown-schema``: a predicate on a table whose key the walk cannot find;
- ``truncate``, ``whole-delete``, ``whole-update``: a write to every row;
- ``whole-maintenance``: ``ANALYZE`` / ``VACUUM`` of a table, which reads it whole;
- ``whole-copy-out``: ``COPY <table> TO``.

A ``SELECT ... FROM <table>`` with nothing after the table is a *head* (:data:`HEAD`): the caller adds the
predicate (a keyset pager does), so a head is judged where it is run, not where it is written. Text interpolated
between a table and its clauses (``FROM t {where}``), or a whole predicate that is one interpolation, may hold the
key: the walk cannot see it, so it does not judge that read.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from threetears.enforcement.no_whole_reads.schema import SchemaCatalog, bare_table

__all__ = [
    "HEAD",
    "KEYED_COUNT",
    "Reader",
    "is_sql",
    "judge_sql",
]

#: the kind a bare ``SELECT ... FROM <table>`` is given: judged where it runs
HEAD: Final[str] = "head"

#: the kind a count or fingerprint narrowed by a key is given: a verification scan of the key's rows, judged only in
#: the code a repo names as load code (``verify-aggregate``)
KEYED_COUNT: Final[str] = "keyed-count"

#: the column an interpolation the code names as the key stands for (``{pk}``, ``{key_column}``)
PK_TOKEN: Final[str] = "{pk}"

#: an interpolation's name that says it is the key column(s)
_KEYISH: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:^|_)(?:pk\w*|primary_keys?|keys?|key_?col\w*|key_?columns?|key_?names?|id_?col\w*)$"
)

#: a count or a fingerprint: the shapes of re-verifying data by scanning it
_VERIFY: Final[re.Pattern[str]] = re.compile(
    r"\b(?:count|md5|sha256|hashtext|bit_xor|string_agg|checksum|fingerprint\w*)\s*\(", re.IGNORECASE
)

_BLOCK_COMMENT: Final[re.Pattern[str]] = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT: Final[re.Pattern[str]] = re.compile(r"--[^\n]*")
_QUOTED: Final[re.Pattern[str]] = re.compile(r"'(?:[^']|'')*'")
_BRACED: Final[re.Pattern[str]] = re.compile(r"\{([^{}]*)\}")

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
_FROM: Final[re.Pattern[str]] = re.compile(r"\bFROM\b", re.IGNORECASE)
_WHERE: Final[re.Pattern[str]] = re.compile(r"\bWHERE\b", re.IGNORECASE)
_AFTER_WHERE: Final[re.Pattern[str]] = re.compile(
    r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT|OFFSET|HAVING|WINDOW|RETURNING|FOR\s+(?:UPDATE|SHARE|NO\s+KEY)|ON\s+CONFLICT|"
    r"QUALIFY|FETCH\s+(?:FIRST|NEXT))\b",
    re.IGNORECASE,
)
#: where a ``FROM`` list ends
_FROM_END: Final[re.Pattern[str]] = re.compile(rf"\bWHERE\b|{_AFTER_WHERE.pattern}", re.IGNORECASE)
_LIMITED: Final[re.Pattern[str]] = re.compile(r"\b(LIMIT|FETCH\s+FIRST|FETCH\s+NEXT|TOP)\b", re.IGNORECASE)
_LIMIT_ONE: Final[re.Pattern[str]] = re.compile(r"\b(?:LIMIT\s+1|FETCH\s+(?:FIRST|NEXT)\s+1?\s*ROWS?\s+ONLY)\b", re.I)
_ORDER_BY: Final[re.Pattern[str]] = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_AGGREGATE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:GROUP\s+BY|DISTINCT)\b|\b(?:count|sum|avg|min|max|string_agg|array_agg|bit_xor|bit_or|md5|sha256|"
    r"hashtext|listagg|median|bool_and|bool_or|checksum|fingerprint\w*|approx_count_distinct)\s*\(",
    re.IGNORECASE,
)
_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\$\d+|%s|%\(\w+\)s|\?|(?<![:\w]):[A-Za-z_]\w*")
_INTERPOLATION: Final[re.Pattern[str]] = re.compile(r"\{(?:pk)?\}")
#: a predicate no row passes: a part that only types a ``UNION``'s columns
_NO_ROWS: Final[re.Pattern[str]] = re.compile(r"WHERE\s+(?:1\s*=\s*0|FALSE)", re.IGNORECASE)
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
    rf"(?<!FOR )(?<!DO )\bUPDATE\s+(?:ONLY\s+)?{_NAME}(?:\s+(?:AS\s+)?(\w+))?\s+SET\b", re.I
)
_MAINTENANCE: Final[re.Pattern[str]] = re.compile(rf"^\s*(ANALYZE|VACUUM)(?:\s+\(?[A-Z, ]*\)?)?\s+{_NAME}", re.I)
_COPY_OUT: Final[re.Pattern[str]] = re.compile(rf"^\s*COPY\s+{_NAME}(?:\s*\([^)]*\))?\s+TO\b", re.I)

#: a join keyword or a comma: where one item of a ``FROM`` list ends
_JOIN_SPLIT: Final[re.Pattern[str]] = re.compile(
    r"\b(?:NATURAL\s+)?(?:(?:LEFT|RIGHT|FULL)(?:\s+OUTER)?\s+|INNER\s+|CROSS\s+)?JOIN\b|,", re.IGNORECASE
)
#: words that end a relation, never its alias
_NOT_ALIAS: Final[frozenset[str]] = frozenset(
    {
        "ON", "USING", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "CROSS", "FULL", "NATURAL", "OUTER", "GROUP",
        "ORDER", "LIMIT", "OFFSET", "HAVING", "WINDOW", "UNION", "INTERSECT", "EXCEPT", "RETURNING", "FOR", "SET",
        "TABLESAMPLE", "FETCH", "QUALIFY", "LATERAL", "WITH",
    }
)  # fmt: skip
#: the first comparison of a predicate: what it compares its column with
_COMPARISON: Final[re.Pattern[str]] = re.compile(
    r"->>?|#>>?|@>|<@|&&|<>|!=|>=|<=|=|>|<|\bNOT\s+IN\b|\bIN\b|\bNOT\s+BETWEEN\b|\bBETWEEN\b|\bIS\b|\bNOT\s+I?LIKE\b|"
    r"\bI?LIKE\b|\bSIMILAR\b|~",
    re.IGNORECASE,
)
_COLUMN: Final[re.Pattern[str]] = re.compile(r"^(?:(\"[^\"]+\"|\w+)\s*\.\s*)?(\"[^\"]+\"|[A-Za-z_]\w*|\{pk\}|\{\})$")
_CAST: Final[re.Pattern[str]] = re.compile(r"::\s*[\w ]+(?:\(\d+(?:,\s*\d+)?\))?(?:\[\])*\s*$")
_VALUE_WORDS: Final[frozenset[str]] = frozenset({"TRUE", "FALSE", "NULL", "CURRENT_DATE", "CURRENT_TIMESTAMP"})


@dataclass(frozen=True)
class _SqlFinding:
    """one way a statement reads or writes whole.

    :ivar kind: the kind (module docstring), or :data:`HEAD`
    :ivar table: the relation, as written (``{}`` for an interpolated name)
    """

    kind: str
    table: str


def _never(_: str) -> bool:
    return False


@dataclass(frozen=True)
class Reader:
    """what a statement is judged against: the declared schema, and the names of the code it is written in.

    :ivar schema: every table's key and indexes
    :ivar names: ``<path>::<qualified name>`` of the function (or SQL constant) the statement is written in: an
        index whose ``purpose`` names one of them keys the read (b)
    :ivar transient: whether a relation is a change set being applied (a stage, rows registered from memory): it
        holds only the rows a change named, so a join on it is keyed
    """

    schema: SchemaCatalog = field(default_factory=SchemaCatalog)
    names: frozenset[str] = frozenset()
    transient: Callable[[str], bool] = _never


def _interpolation(match: re.Match[str]) -> str:
    return PK_TOKEN if _KEYISH.search(match.group(1).strip()) else "{}"


def _clean(sql: str) -> str:
    """the statement without comments, each quoted literal as ``'_'``, each interpolation as ``{}`` (or ``{pk}``
    when it names the key), on one line.

    :param sql: the text
    :ptype sql: str
    :return: the cleaned text
    :rtype: str
    """
    text = _LINE_COMMENT.sub(" ", _BLOCK_COMMENT.sub(" ", sql))
    return " ".join(_BRACED.sub(_interpolation, _QUOTED.sub("'_'", text)).split())


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


def _mask(text: str) -> str:
    """``text`` with everything inside parentheses blanked (the parentheses kept), so a split sees one level.

    :param text: a piece of cleaned SQL
    :ptype text: str
    :return: the masked text, the same length
    :rtype: str
    """
    out: list[str] = []
    depth = 0
    for char in text:
        if char == ")":
            depth = max(0, depth - 1)
        out.append(char if depth == 0 else " ")
        if char == "(":
            depth += 1
    return "".join(out)


def _unwrap(text: str) -> str:
    """``text`` without parentheses that enclose all of it.

    :param text: a predicate
    :ptype text: str
    :return: it unwrapped
    :rtype: str
    """
    stripped = text.strip()
    while stripped.startswith("(") and stripped.endswith(")") and _mask(stripped).count("(") == 1:
        stripped = stripped[1:-1].strip()
    return stripped


def _split(text: str, pattern: re.Pattern[str]) -> list[str]:
    """split ``text`` where ``pattern`` matches outside parentheses.

    :param text: a piece of cleaned SQL
    :ptype text: str
    :param pattern: the separator
    :ptype pattern: re.Pattern[str]
    :return: the parts
    :rtype: list[str]
    """
    masked = _mask(text)
    parts: list[str] = []
    last = 0
    for match in pattern.finditer(masked):
        parts.append(text[last : match.start()])
        last = match.end()
    parts.append(text[last:])
    return [part.strip() for part in parts if part.strip()]


_AND: Final[re.Pattern[str]] = re.compile(r"\bAND\b", re.IGNORECASE)
_OR: Final[re.Pattern[str]] = re.compile(r"\bOR\b", re.IGNORECASE)
_BETWEEN_AND: Final[re.Pattern[str]] = re.compile(r"(\bBETWEEN\s+\S+\s+)AND\b", re.IGNORECASE)


def _conjuncts(text: str) -> list[str]:
    """the predicates ``text`` ANDs together, nested parentheses of ANDs opened.

    :param text: a ``WHERE`` or ``ON`` condition
    :ptype text: str
    :return: the predicates
    :rtype: list[str]
    """
    found: list[str] = []
    for part in _split(_BETWEEN_AND.sub(r"\1_BETWEEN_AND_", text), _AND):
        part = part.replace("_BETWEEN_AND_", "AND")
        inner = _unwrap(part)
        if inner != part.strip() and len(_split(inner, _AND)) > 1 and len(_split(inner, _OR)) == 1:
            found.extend(_conjuncts(inner))
        else:
            found.append(inner)
    return found


def _strip_cast(text: str) -> str:
    stripped = _unwrap(text)
    previous = None
    while previous != stripped:
        previous = stripped
        stripped = _unwrap(_CAST.sub("", stripped))
        cast = re.fullmatch(r"CAST\s*\((.*)\s+AS\s+[\w ]+(?:\[\])*\)", stripped, re.IGNORECASE)
        if cast is not None:
            stripped = cast.group(1).strip()
    return stripped


def _column(text: str) -> tuple[str, str] | None:
    """the column an operand is: ``(qualifier, column)``, the qualifier empty when there is none.

    :param text: an operand
    :ptype text: str
    :return: it, or None when it is a value or an expression
    :rtype: tuple[str, str] | None
    """
    match = _COLUMN.match(_strip_cast(text))
    if match is None:
        return None
    column = match.group(2).strip('"')
    if column.upper() in _VALUE_WORDS or (column == "{}" and match.group(1) is None):
        return None
    return (match.group(1) or "").strip('"').lower(), column.lower()


def _row(text: str) -> list[tuple[str, str]] | None:
    """the columns a row operand ``(a, b)`` is.

    :param text: an operand
    :ptype text: str
    :return: them, or None when it is no row of columns
    :rtype: list[tuple[str, str]] | None
    """
    stripped = text.strip()
    if not (stripped.startswith("(") and stripped.endswith(")")):
        return None
    parts = _split(stripped[1:-1], re.compile(","))
    columns = [_column(part) for part in parts]
    return [column for column in columns if column is not None] if len(parts) > 1 and all(columns) else None


def _bound_value(text: str) -> bool:
    """whether an operand is a bound value: placeholders, interpolations, literals and functions of them, or a
    subquery (judged on its own level).

    :param text: an operand
    :ptype text: str
    :return: whether it is
    :rtype: bool
    """
    stripped = text.strip()
    if _SELECT.search(stripped) is not None:
        return True
    if not (
        _PLACEHOLDER.search(stripped)
        or _INTERPOLATION.search(stripped)
        or "'_'" in stripped
        or re.search(r"(?<![\w.])-?\d", stripped)
        or re.search(r"\b(?:TRUE|FALSE)\b", stripped, re.IGNORECASE)
    ):
        return False
    rest = _PLACEHOLDER.sub(" ", _INTERPOLATION.sub(" ", stripped.replace("'_'", " ")))
    rest = re.sub(r"::\s*[\w ]+(?:\[\])*", " ", rest)
    rest = re.sub(r"\bAS\s+\w+(?:\[\])*", " ", rest, flags=re.IGNORECASE)
    rest = re.sub(r"\b\w+\s*\(", "(", rest)
    rest = re.sub(r"\b(?:ARRAY|ANY|TRUE|FALSE|NULL|INTERVAL)\b", " ", rest, flags=re.IGNORECASE)
    rest = re.sub(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e-?\d+)?", " ", rest)
    return re.search(r"[A-Za-z_]", rest) is None


@dataclass(frozen=True)
class _Atom:
    """one predicate, as the key judgement reads it.

    :ivar kind: ``eq`` (a column equals a bound value), ``range`` (a column against a bound bound), ``join`` (a
        column equals another column), ``filter`` (anything else), ``unseen`` (an interpolation the walk cannot see)
    :ivar column: the column, ``(qualifier, name)``
    :ivar other: the other column of a join
    :ivar bound: whether the predicate holds a bound value at all
    """

    kind: str
    column: tuple[str, str] = ("", "")
    other: tuple[str, str] = ("", "")
    bound: bool = False


def _atoms(predicate: str) -> list[_Atom]:
    """what one predicate binds.

    :param predicate: one conjunct
    :ptype predicate: str
    :return: its atoms
    :rtype: list[_Atom]
    """
    text = _unwrap(predicate)
    bound = _PLACEHOLDER.search(text) is not None or _INTERPOLATION.search(text) is not None or "'_'" in text
    bound = bound or re.search(r"=\s*-?\d|\b(?:TRUE|FALSE)\b", text, re.IGNORECASE) is not None
    if _INTERPOLATION.fullmatch(text) or re.fullmatch(r"(?:NOT\s+)?\{\}", text, re.IGNORECASE):
        return [_Atom("unseen")]
    masked = _mask(text)
    branches = _split(text, _OR)
    if len(branches) > 1:
        # ``k = $1 OR k = $2``: every branch binds the same column, so the read is by those keys
        each = [_atoms(branch) for branch in branches]
        columns = {atoms[0].column for atoms in each if len(atoms) == 1 and atoms[0].kind == "eq"}
        if len(columns) == 1 and all(len(atoms) == 1 and atoms[0].kind == "eq" for atoms in each):
            return [_Atom("eq", columns.pop(), bound=True)]
        return [_Atom("filter", bound=bound)]
    if re.match(r"\s*(?:NOT\b|EXISTS\b)", masked, re.IGNORECASE):
        return [_Atom("filter", bound=bound)]
    operator = _COMPARISON.search(masked)
    if operator is None:
        return [_Atom("filter", bound=bound)]
    left, right = text[: operator.start()].strip(), text[operator.end() :].strip()
    op = " ".join(operator.group(0).upper().split())
    atoms: list[_Atom] = []
    if op == "=":
        lcol, rcol = _column(left), _column(right)
        lrow = _row(left)
        if lcol and rcol:
            atoms.append(_Atom("join", lcol, rcol, bound))
        elif lcol and _bound_value(right):
            atoms.append(_Atom("eq", lcol, bound=True))
        elif rcol and _bound_value(left):
            atoms.append(_Atom("eq", rcol, bound=True))
        elif lrow and _bound_value(right):
            atoms.extend(_Atom("eq", column, bound=True) for column in lrow)
        elif lrow and _row(right):
            atoms.extend(_Atom("join", a, b, bound) for a, b in zip(lrow, _row(right) or [], strict=False))
    elif op == "IN":
        lcol, lrow = _column(left), _row(left)
        inner = _unwrap(right)
        if right.strip().startswith("(") and (
            _SELECT.match(inner) or all(_bound_value(part) for part in _split(inner, re.compile(",")))
        ):
            atoms.extend(_Atom("eq", column, bound=True) for column in ([lcol] if lcol else lrow or []))
    elif op in {">", ">=", "<", "<="}:
        lcol, rcol, lrow = _column(left), _column(right), _row(left)
        if lcol and _bound_value(right):
            atoms.append(_Atom("range", lcol, bound=True))
        elif rcol and _bound_value(left):
            atoms.append(_Atom("range", rcol, bound=True))
        elif lrow and _bound_value(right):
            atoms.append(_Atom("range", lrow[0], bound=True))
    elif op == "BETWEEN":
        lcol = _column(left)
        if lcol and all(_bound_value(part) for part in re.split(r"\bAND\b", right, flags=re.IGNORECASE)):
            atoms.append(_Atom("range", lcol, bound=True))
    return atoms or [_Atom("filter", bound=bound)]


@dataclass
class _Relation:
    """one item of a ``FROM`` list.

    :ivar name: the relation as written
    :ivar alias: what the statement calls it (its name when it has no alias), lower-cased
    :ivar kind: ``data`` (a table), ``source`` (bound rows: a function of bound values, a bound CTE, a stage, a
        derived table), ``other`` (a catalog, an unbound CTE, a function of nothing bound)
    :ivar on: its ``ON`` condition
    :ivar using: its ``USING`` columns
    """

    name: str
    alias: str
    kind: str
    on: str = ""
    using: tuple[str, ...] = ()


def _relations(text: str, masked: str, ctes: set[str], bound_ctes: frozenset[str], reader: Reader) -> list[_Relation]:
    """the relations one ``FROM`` list (or ``USING`` list) names.

    :param text: the list, cleaned SQL
    :ptype text: str
    :param masked: the same span with deeper levels blanked
    :ptype masked: str
    :param ctes: the statement's CTE names
    :ptype ctes: set[str]
    :param bound_ctes: those seeded by a bound key
    :ptype bound_ctes: frozenset[str]
    :param reader: the judgement's context
    :ptype reader: Reader
    :return: the relations, in order
    :rtype: list[_Relation]
    """
    relations: list[_Relation] = []
    last = 0
    pieces: list[tuple[int, int]] = []
    for match in _JOIN_SPLIT.finditer(masked):
        pieces.append((last, match.start()))
        last = match.end()
    pieces.append((last, len(masked)))
    for begin, end in pieces:
        item, shown = text[begin:end], masked[begin:end]
        head = re.match(rf"\s*(?:LATERAL\s+|ONLY\s+)?(\(|{_NAME})", shown, re.IGNORECASE)
        if head is None:
            continue
        after = head.end()
        if head.group(1) == "(":
            close = shown.find(")", after)
            close = len(shown) if close < 0 else close + 1
            body = item[head.start(1) : close]
            name, kind, after = body, "source", close
        else:
            name = head.group(2)
            bare = name.strip('"')
            if shown[after:].lstrip().startswith("("):
                opening = after + (len(shown[after:]) - len(shown[after:].lstrip()))
                close = shown.find(")", opening)
                close = len(shown) if close < 0 else close + 1
                args = item[opening:close]
                bound = _PLACEHOLDER.search(args) is not None or _INTERPOLATION.search(args) is not None
                kind, after = ("source" if bound else "other"), close
            elif bare in bound_ctes or bare.lower() in bound_ctes:
                kind = "source"
            elif bare in ctes or bare.lower() in ctes or _CATALOG.match(name) is not None:
                kind = "other"
            elif reader.transient(bare):
                kind = "source"
            else:
                kind = "data"
        alias_match = re.match(r"\s*(?:AS\s+)?(\w+)", shown[after:], re.IGNORECASE)
        alias = name
        if alias_match is not None and alias_match.group(1).upper() not in _NOT_ALIAS:
            alias = alias_match.group(1)
        on_match = re.search(r"\bON\b", shown[after:], re.IGNORECASE)
        using_match = re.search(r"\bUSING\s*\(", shown[after:], re.IGNORECASE)
        on = item[after + on_match.end() :] if on_match is not None else ""
        using: tuple[str, ...] = ()
        if using_match is not None:
            opening = after + using_match.end() - 1
            close = item.find(")", opening)
            using = tuple(part.strip().strip('"').lower() for part in item[opening + 1 : close].split(","))
        relations.append(_Relation(name, alias.strip('"').lower(), kind, on, using))
    return relations


@dataclass
class _Level:
    """what one statement level reads, as the key judgement needs it.

    :ivar relations: its ``FROM`` / ``JOIN`` relations (or a write's target first)
    :ivar atoms: the atoms of its ``WHERE`` and every ``ON``
    :ivar where: its ``WHERE`` text, None when it has none
    :ivar order: the column it orders by first, ``(qualifier, name)``
    :ivar limited: whether a ``LIMIT`` bounds it
    :ivar outer: columns an outer ``WHERE`` binds, pushed into a derived table
    :ivar correlated: whether it is a subquery, where a column of a relation it does not name is an outer row's
    """

    relations: list[_Relation]
    atoms: list[_Atom]
    where: str | None
    order: tuple[str, str] | None
    limited: bool
    outer: set[str] = field(default_factory=set)
    correlated: bool = False


def _owner(level: _Level, qualifier: str) -> list[int]:
    """the relations of a level a column's qualifier names (every data relation, for no qualifier).

    :param level: the level
    :ptype level: _Level
    :param qualifier: the qualifier, lower-cased, empty for none
    :ptype qualifier: str
    :return: the indexes of the relations it may belong to
    :rtype: list[int]
    """
    if qualifier:
        return [i for i, rel in enumerate(level.relations) if qualifier in {rel.alias, rel.name.strip('"').lower()}]
    data = [i for i, rel in enumerate(level.relations) if rel.kind == "data"]
    return data if data else list(range(len(level.relations)))


def _known(level: _Level, qualifier: str) -> bool:
    return not qualifier or bool(_owner(level, qualifier))


def _keys(table: str, reader: Reader) -> tuple[list[tuple[str, ...]], list[tuple[str, tuple[str, ...]]]]:
    """the keys a read of ``table`` may use: its primary keys, and its indexes whose purpose names the reader.

    :param table: the relation as written
    :ptype table: str
    :param reader: the judgement's context
    :ptype reader: Reader
    :return: ``(primary keys, [(index name, columns)])``; a table interpolated in generic code has the key the code
        names (``{pk}``)
    :rtype: tuple[list[tuple[str, ...]], list[tuple[str, tuple[str, ...]]]]
    """
    if bare_table(table) is None:
        return [(PK_TOKEN,)], []
    declared = reader.schema.table(table)
    if declared is None:
        return [], []
    purposed = [
        (index.name, index.columns)
        for index in declared.indexes.values()
        if index.purpose and index.purpose in reader.names and index.columns
    ]
    return [*declared.primary_keys, *([(PK_TOKEN,)] if declared.primary_keys else [])], purposed


def _keyed(level: _Level, reader: Reader) -> tuple[list[bool], list[bool]]:
    """which data relations of a level are read by key, and which have a known schema.

    :param level: the level
    :ptype level: _Level
    :param reader: the judgement's context
    :ptype reader: Reader
    :return: ``(keyed, known)``, one flag each per relation
    :rtype: tuple[list[bool], list[bool]]
    """
    count = len(level.relations)
    eq: list[set[str]] = [set() for _ in range(count)]
    ranged: list[set[str]] = [set() for _ in range(count)]
    other: list[int] = [0] * count
    joins: list[tuple[int, str, int, str]] = []
    for column in level.outer:
        for i in _owner(level, ""):
            eq[i].add(column)
    for atom in level.atoms:
        qualifier, name = atom.column
        if atom.kind == "join":
            oq, oname = atom.other
            mine, theirs = _owner(level, qualifier), _owner(level, oq)
            if qualifier and not mine and theirs:
                # ``outer.x = t.y``: the outer row's column binds this level's
                for i in theirs:
                    eq[i].add(oname)
            elif oq and not theirs and mine:
                for i in mine:
                    eq[i].add(name)
            elif not qualifier and not oq and level.correlated:
                for i in mine:
                    eq[i].add(name)
                    eq[i].add(oname)
            else:
                joins.extend((a, name, b, oname) for a in mine for b in theirs if a != b)
        elif atom.kind in {"eq", "range"} and _known(level, qualifier):
            for i in _owner(level, qualifier):
                (eq if atom.kind == "eq" else ranged)[i].add(name)
        elif atom.kind == "filter":
            for i in _owner(level, ""):
                other[i] += 1
    for index, relation in enumerate(level.relations):
        for column in relation.using:
            joins.extend((index, column, earlier, column) for earlier in range(index))
    # bound rows key what joins them; a catalog, an unbound CTE or a function of nothing bound keys nothing
    keyed = [relation.kind == "source" for relation in level.relations]
    known = [True] * count
    grew = True
    while grew:
        grew = False
        for a, column, b, other_column in joins:
            for x, cx, y, cy in ((a, column, b, other_column), (b, other_column, a, column)):
                if keyed[y] and cx not in eq[x]:
                    eq[x].add(cx)
                    grew = True
        for index, relation in enumerate(level.relations):
            if keyed[index]:
                continue
            primary, purposed = _keys(relation.name, reader)
            # generic code over an interpolated table knows the key it names ({pk}), and nothing else about it
            known[index] = bool(primary) and bare_table(relation.name) is not None
            hit = _key_hit(primary, purposed, eq[index], ranged[index], other[index], level, index, reader, relation)
            if hit:
                keyed[index] = True
                grew = True
    return keyed, known


def _key_hit(
    primary: list[tuple[str, ...]],
    purposed: list[tuple[str, tuple[str, ...]]],
    eq: set[str],
    ranged: set[str],
    filters: int,
    level: _Level,
    index: int,
    reader: Reader,
    relation: _Relation,
) -> bool:
    """whether a relation's bound columns make its read keyed: (a) a key prefix, (b) a purposed index's leading
    column, (c) a keyset page.

    :param primary: its primary keys
    :ptype primary: list[tuple[str, ...]]
    :param purposed: its indexes whose purpose names the reader, ``(name, columns)``
    :ptype purposed: list[tuple[str, tuple[str, ...]]]
    :param eq: its columns bound by equality
    :ptype eq: set[str]
    :param ranged: its columns bound by a range
    :ptype ranged: set[str]
    :param filters: how many other predicates may filter it
    :ptype filters: int
    :param level: the level
    :ptype level: _Level
    :param index: its position in the level
    :ptype index: int
    :param reader: the judgement's context
    :ptype reader: Reader
    :param relation: the relation
    :ptype relation: _Relation
    :return: whether it is keyed
    :rtype: bool
    """
    if any(key and key[0] in eq for key in primary):
        return True
    for name, columns in purposed:
        if columns[0] in eq or (columns[0] in ranged and level.limited):
            table = bare_table(relation.name)
            if table is not None:
                reader.schema.used.add((table, name))
            return True
    order = level.order
    if level.limited and order is not None and index in _owner(level, order[0]) and not filters:
        for key in [*primary, *(columns for _, columns in purposed)]:
            prefix = 0
            while prefix < len(key) and key[prefix] in eq:
                prefix += 1
            if prefix < len(key) and key[prefix] == order[1] and not (eq - set(key[:prefix])) and ranged <= {order[1]}:
                return True
    return False


def _order(masked: str, text: str) -> tuple[str, str] | None:
    """the column a level orders by first.

    :param masked: the level, deeper levels blanked
    :ptype masked: str
    :param text: the same span, cleaned SQL
    :ptype text: str
    :return: ``(qualifier, column)``, or None
    :rtype: tuple[str, str] | None
    """
    match = _ORDER_BY.search(masked)
    if match is None:
        return None
    closing = re.compile(r"\b(LIMIT|OFFSET|FOR|FETCH)\b", re.IGNORECASE).search(masked, match.end())
    items = _split(text[match.end() : closing.start() if closing else len(text)], re.compile(","))
    if not items:
        return None
    first = re.sub(r"\s+(ASC|DESC|NULLS\s+(FIRST|LAST)).*$", "", items[0], flags=re.IGNORECASE).strip()
    row = _row(first)
    return row[0] if row else _column(first)


def _bound_ctes(text: str, depths: list[int], ctes: set[str], reader: Reader) -> set[str]:
    """the statement's CTEs whose seed carries a key: the first ``SELECT`` of the body (before any ``UNION``) holds a
    bound value and reads no data table whole.

    :param text: cleaned SQL
    :ptype text: str
    :param depths: :func:`_depths` of it
    :ptype depths: list[int]
    :param ctes: the statement's CTE names
    :ptype ctes: set[str]
    :param reader: the judgement's context
    :ptype reader: Reader
    :return: the bound CTEs' names, as written and lower-cased
    :rtype: set[str]
    """
    bound: set[str] = set()
    for match in _CTE.finditer(text):
        if depths[match.start()] != 0:
            continue
        seed = _SELECT.search(text, match.end())
        if seed is None or depths[seed.start()] != 1:
            continue
        end, masked = _own_level(text, depths, seed.start(), 1)
        level = _level(text, depths, seed.start(), end, masked, ctes, frozenset(bound), reader)
        reads = any(relation.kind == "data" for relation in level.relations) if level is not None else False
        where = _WHERE.search(masked)
        # a seed reading no table binds the key in what it selects; one reading a table binds it in its WHERE, so a
        # placeholder in a whole read's select list binds nothing
        placeholder = _PLACEHOLDER.search(
            text[seed.start() + where.end() : end] if reads and where else text[seed.start() : end]
        )
        keyed = placeholder is not None and (not reads or where is not None)
        if keyed and _judge_select(text, depths, seed.start(), ctes, frozenset(bound), reader) is None:
            name = match.group(1).strip('"')
            bound |= {name, name.lower()}
    return bound


def _level(
    text: str,
    depths: list[int],
    start: int,
    end: int,
    masked: str,
    ctes: set[str],
    bound_ctes: frozenset[str],
    reader: Reader,
) -> _Level | None:
    """read one ``SELECT`` level: its relations, predicates, order and limit.

    :param text: cleaned SQL
    :ptype text: str
    :param depths: :func:`_depths` of it
    :ptype depths: list[int]
    :param start: where the ``SELECT`` starts
    :ptype start: int
    :param end: where its level ends
    :ptype end: int
    :param masked: the level with deeper text blanked
    :ptype masked: str
    :param ctes: the statement's CTE names
    :ptype ctes: set[str]
    :param bound_ctes: those seeded by a bound key
    :ptype bound_ctes: frozenset[str]
    :param reader: the judgement's context
    :ptype reader: Reader
    :return: the level, or None when it reads from nothing
    :rtype: _Level | None
    """
    opening = _FROM.search(masked)
    if opening is None:
        return None
    closing = _FROM_END.search(masked, opening.end())
    list_end = closing.start() if closing else len(masked)
    relations = _relations(
        text[start + opening.end() : start + list_end], masked[opening.end() : list_end], ctes, bound_ctes, reader
    )
    where_match = _WHERE.search(masked, opening.end())
    where: str | None = None
    if where_match is not None:
        after = _AFTER_WHERE.search(masked, where_match.end())
        where = text[start + where_match.end() : start + (after.start() if after else len(masked))]
    atoms = [atom for predicate in _conjuncts(where or "") for atom in _atoms(predicate)]
    for relation in relations:
        atoms.extend(atom for predicate in _conjuncts(relation.on) for atom in _atoms(predicate))
    depth = depths[start]
    level = _Level(
        relations,
        atoms,
        where,
        _order(masked, text[start : start + len(masked)]),
        _LIMITED.search(masked) is not None,
        correlated=depth > 0,
    )
    level.outer = _outer_columns(text, depths, start, depth)
    return level


def _outer_columns(text: str, depths: list[int], start: int, depth: int) -> set[str]:
    """the columns a derived table's outer ``WHERE`` binds (``FROM (SELECT ...) d WHERE d.k = $1``), which the
    planner pushes into it.

    :param text: cleaned SQL
    :ptype text: str
    :param depths: :func:`_depths` of it
    :ptype depths: list[int]
    :param start: where the inner ``SELECT`` starts
    :ptype start: int
    :param depth: its depth
    :ptype depth: int
    :return: the bound column names
    :rtype: set[str]
    """
    if not (depth > 0 and re.search(r"\b(FROM|JOIN)\s*\(\s*$", text[:start], re.IGNORECASE)):
        return set()
    close = next((i for i in range(start, len(text)) if depths[i] < depth), len(text))
    _, outer = _own_level(text, depths, close, depth - 1)
    where_match = _WHERE.search(outer)
    if where_match is None:
        return set()
    closing = _AFTER_WHERE.search(outer, where_match.end())
    where = text[close + where_match.end() : close + (closing.start() if closing else len(outer))]
    return {atom.column[1] for p in _conjuncts(where) for atom in _atoms(p) if atom.kind == "eq"}


def _unkeyed_kind(level: _Level, known: bool, aggregate: bool) -> str:
    """the kind of an unkeyed read.

    :param level: its level
    :ptype level: _Level
    :param known: whether its table's schema is declared
    :ptype known: bool
    :param aggregate: whether the level aggregates
    :ptype aggregate: bool
    :return: the kind
    :rtype: str
    """
    bound = any(atom.bound for atom in level.atoms)
    if bound and not known:
        kind = "unknown-schema"
    elif aggregate:
        kind = "whole-aggregate"
    elif level.where is None and not any(atom.kind != "join" for atom in level.atoms):
        kind = "whole-select"
    elif not bound:
        kind = "unbound-where"
    else:
        kind = "non-key-where"
    return kind


def _judge_select(
    text: str,
    depths: list[int],
    start: int,
    ctes: set[str],
    bound_ctes: frozenset[str],
    reader: Reader,
) -> _SqlFinding | None:
    """judge one ``SELECT`` at its own level.

    :param text: cleaned SQL
    :ptype text: str
    :param depths: :func:`_depths` of it
    :ptype depths: list[int]
    :param start: where the ``SELECT`` keyword starts
    :ptype start: int
    :param ctes: the statement's CTE names
    :ptype ctes: set[str]
    :param bound_ctes: the statement's CTEs whose seed carries a key (:func:`_bound_ctes`)
    :ptype bound_ctes: frozenset[str]
    :param reader: the judgement's context
    :ptype reader: Reader
    :return: the finding, or None when every table it reads is read by key, or it reads none
    :rtype: _SqlFinding | None
    """
    depth = depths[start]
    end, masked = _own_level(text, depths, start, depth)
    level = _level(text, depths, start, end, masked, ctes, bound_ctes, reader)
    if level is None or not any(relation.kind == "data" for relation in level.relations):
        return None
    first = next(relation for relation in level.relations if relation.kind == "data")
    opening = _FROM.search(masked)
    table_end = masked.find(first.name, opening.end() if opening else 0) + len(first.name)
    clauses = _FROM_END.search(masked, table_end)
    between = masked[table_end : clauses.start() if clauses else len(masked)]
    # text interpolated between the table and its clauses (``FROM t {where}``, ``FROM t{filters}``), or a predicate
    # that is one interpolation, may be the key: the walk cannot see it, so it does not judge the read
    unseen = (
        _INTERPOLATION.search(between) is not None
        or (first.name.endswith("{}") and len(first.name) > 2)
        or any(atom.kind == "unseen" for atom in level.atoms)
    )
    aggregate = _AGGREGATE.search(masked[: opening.start() if opening else 0]) is not None or bool(
        re.search(r"\bGROUP\s+BY\b", masked, re.IGNORECASE)
    )
    table = first.name.removesuffix("{}") if first.name.endswith("{}") and len(first.name) > 2 else first.name
    finding: _SqlFinding | None = None
    if level.where is not None and _NO_ROWS.fullmatch(f"WHERE {level.where.strip()}"):
        return None
    if unseen:
        # a count or fingerprint is a verification scan whatever narrows it, so load code still names it
        verify = _VERIFY.search(masked[: opening.start() if opening else 0]) is not None
        return _SqlFinding(KEYED_COUNT, table) if verify else None
    if (
        not masked[table_end:].strip()
        and len(level.relations) == 1
        and start == _statement_start(text)
        and not aggregate
    ):
        return _SqlFinding(HEAD, table)
    keyed, known = _keyed(level, reader)
    for index, relation in enumerate(level.relations):
        if relation.kind != "data" or keyed[index]:
            continue
        probe = (
            level.where is None
            and not aggregate
            and level.order is None
            and _LIMIT_ONE.search(masked) is not None
            and not any(atom.kind != "join" for atom in level.atoms)
        )
        if probe and len(level.relations) == 1:
            return None
        return _SqlFinding(_unkeyed_kind(level, known[index], aggregate), relation.name)
    if _VERIFY.search(masked[: opening.start() if opening else 0]) is not None:
        finding = _SqlFinding(KEYED_COUNT, table)
    return finding


def _judge_write(
    text: str, depths: list[int], ctes: set[str], bound_ctes: frozenset[str], reader: Reader
) -> list[_SqlFinding]:
    """the whole-table writes and maintenance in one statement, and each ``UPDATE`` / ``DELETE`` not by key.

    :param text: cleaned SQL
    :ptype text: str
    :param depths: :func:`_depths` of it
    :ptype depths: list[int]
    :param ctes: the statement's CTE names
    :ptype ctes: set[str]
    :param bound_ctes: those seeded by a bound key
    :ptype bound_ctes: frozenset[str]
    :param reader: the judgement's context
    :ptype reader: Reader
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
                    match.start() + where_match.end() : match.start() + (closing.start() if closing else len(masked))
                ]
            clauses = _AFTER_WHERE.search(masked, relative)
            table = match.group(1)
            unseen = (
                table.endswith("{}")
                and len(table) > 2
                or (
                    _INTERPOLATION.search(masked[relative : clauses.start() if clauses else len(masked)]) is not None
                    and kind == "whole-delete"
                    and where is None
                )
            )
            if unseen:
                continue
            if where is None:
                findings.append(_SqlFinding(kind, table))
                continue
            alias_text = masked[relative:] if kind == "whole-delete" else ""
            alias_match = re.match(r"\s*(?:AS\s+)?(\w+)", alias_text, re.IGNORECASE)
            alias = match.group(2) if kind == "whole-update" and match.group(2) else table
            if alias_match is not None and alias_match.group(1).upper() not in _NOT_ALIAS:
                alias = alias_match.group(1)
            target = _Relation(table, alias.strip('"').lower(), "data")
            sources = re.search(r"\b(?:USING|FROM)\b(.*?)(?=\bWHERE\b)", masked[relative:], re.IGNORECASE)
            others: list[_Relation] = []
            if sources is not None:
                offset = match.start() + relative + sources.start(1)
                others = _relations(
                    text[offset : offset + len(sources.group(1))], sources.group(1), ctes, bound_ctes, reader
                )
            atoms = [atom for predicate in _conjuncts(where) for atom in _atoms(predicate)]
            if any(atom.kind == "unseen" for atom in atoms):
                continue
            # a write narrowed by a column of a relation it does not name reads by an outer row's key
            level = _Level([target, *others], atoms, where, None, False, correlated=True)
            keyed, known = _keyed(level, reader)
            if not keyed[0]:
                findings.append(_SqlFinding(_unkeyed_kind(level, known[0], False), table))
    return findings


def judge_sql(sql: str, reader: Reader | None = None) -> list[_SqlFinding]:
    """every way one statement reads or writes whole.

    :param sql: the statement, each interpolation as ``{}``
    :ptype sql: str
    :param reader: the schema it is judged against and the names of the code it is written in; None judges it
        against no schema at all, so every predicate on a table is ``unknown-schema``
    :ptype reader: Reader | None
    :return: the findings in text order; a lone :data:`HEAD` for a bare ``SELECT ... FROM <table>``; empty when
        every read and write carries a key
    :rtype: list[_SqlFinding]
    """
    reader = reader if reader is not None else Reader()
    text = _clean(sql)
    depths = _depths(text)
    ctes = {match.group(1).strip('"') for match in _CTE.finditer(text) if depths[match.start()] == 0}
    ctes |= {name.lower() for name in ctes}
    bound_ctes = frozenset(_bound_ctes(text, depths, ctes, reader))
    findings = _judge_write(text, depths, ctes, bound_ctes, reader)
    if _VIEW.match(text) is None:
        for match in _SELECT.finditer(text):
            judged = _judge_select(text, depths, match.start(), ctes, bound_ctes, reader)
            if judged is not None:
                findings.append(judged)
    return findings
