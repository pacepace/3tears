"""
enforcement: every ``extraction_status`` spelling in SQL is a named constant.

``MediaInfo.extraction_status`` is a plain ``str`` field whose values live in
two places at once. ``media-contracts`` names them
(``EXTRACTION_STATUS_*``); ``agent-memory`` writes them into DDL --
``agent-memory``'s v021 migration declares the column ``TEXT NOT NULL
DEFAULT 'none'``, and v022 builds a partial index on
``WHERE extraction_status = 'pending'``.

The partial index is what makes drift expensive. It is the extraction work
queue: rows enter it by carrying the exact string ``'pending'``. Change the
constant without changing the index predicate and the two stop matching --
nothing raises, nothing fails a type check, and the queue simply reports that
there is no work to do. A silent empty queue is indistinguishable from a
quiet day.

So the DDL is the canonical statement of this vocabulary and the constants
name what is stored (docs/search-spec.md §3.5, ruled before the constants
were written). This guard holds that direction: every status spelling
appearing in SQL must be one ``media-contracts`` names. The reverse is not
required -- ``refused`` is contract vocabulary that no migration has needed,
which is the normal state for a value the database stores but never
predicates on.

Two mechanisms, because the two sides are written differently. The constants
are imported (``media-contracts`` is dependency-free and importing it costs
nothing); the DDL is scanned as text, because the spellings live *inside* SQL
strings where an AST walk sees one opaque constant.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from threetears.media.contracts import (
    EXTRACTION_STATUS_COMPLETE,
    EXTRACTION_STATUS_FAILED,
    EXTRACTION_STATUS_NONE,
    EXTRACTION_STATUS_PENDING,
    EXTRACTION_STATUS_REFUSED,
    MediaInfo,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where a status spelling can reach SQL. Scanned as a tree rather than as
#: named migration files so a v023 that adds a spelling is covered without
#: this list being edited -- an enforcement test nobody remembers to update
#: is one that stops enforcing.
_SQL_BEARING_ROOTS = ("packages/agent/memory/src",)

#: The vocabulary, by the constant that names it.
_NAMED_STATUSES = frozenset(
    {
        EXTRACTION_STATUS_NONE,
        EXTRACTION_STATUS_PENDING,
        EXTRACTION_STATUS_COMPLETE,
        EXTRACTION_STATUS_FAILED,
        EXTRACTION_STATUS_REFUSED,
    }
)

#: A quoted lowercase value within reach of an ``extraction_status`` mention.
#: The window spans both shapes this vocabulary is written in: inline SQL
#: (``... extraction_status = 'pending'``, ``extraction_status TEXT NOT NULL
#: DEFAULT 'none'``) and the column declaration, where the name and its
#: server default are adjacent arguments rather than one string
#: (``Column("extraction_status", STRING_TYPE, server_default="'none'::text")``).
_STATUS_WINDOW = re.compile(r"extraction_status.{0,160}?'([a-z_]+)'", re.DOTALL)


def _statuses_in(text: str) -> set[str]:
    """Status spellings quoted near an ``extraction_status`` mention.

    :param text: the file contents to scan
    :ptype text: str
    :return: every quoted lowercase value found in range of a mention
    :rtype: set[str]
    """
    return set(_STATUS_WINDOW.findall(text))


def _sql_bearing_files() -> list[Path]:
    """Every Python file that could write a status spelling into SQL.

    :return: the ``.py`` paths under the SQL-bearing source roots
    :rtype: list[Path]
    """
    found: list[Path] = []
    for root in _SQL_BEARING_ROOTS:
        found.extend(sorted((_REPO_ROOT / root).rglob("*.py")))
    return found


def test_every_status_spelling_in_sql_is_a_named_constant() -> None:
    """No SQL may predicate on or default to a status the contract omits.

    The failure this refuses is the unnamed spelling: DDL that stores
    ``'extracting'`` while the contract knows ``pending``, so a consumer
    comparing against the constants never matches the rows and never learns
    why.
    """
    unnamed: dict[str, set[str]] = {}
    for path in _sql_bearing_files():
        found = _statuses_in(path.read_text(encoding="utf-8"))
        stray = found - _NAMED_STATUSES
        if stray:
            unnamed[str(path.relative_to(_REPO_ROOT))] = stray

    assert not unnamed, (
        "status spellings in SQL that media-contracts does not name: "
        f"{unnamed}. Add the constant to "
        "packages/media-contracts/src/threetears/media/contracts/protocols.py, "
        "or fix the SQL -- but read docs/search-spec.md §3.5 first: a spelling "
        "already in a column default or an index predicate is changed by a "
        "migration, not by editing the constant."
    )


def test_the_work_queue_predicate_still_matches_the_constant() -> None:
    """v022's partial index and ``EXTRACTION_STATUS_PENDING`` agree.

    Named separately from the scan above because this pair is the one whose
    disagreement is silent: the index would still build, the writes would
    still succeed, and the extraction queue would just come back empty.
    """
    migrations = _REPO_ROOT / "packages/agent/memory/src/threetears/agent/memory/migrations"
    predicates = {
        path.name: _statuses_in(text)
        for path in sorted(migrations.rglob("*.py"))
        if "extraction_status =" in (text := path.read_text(encoding="utf-8"))
    }

    assert predicates, "no migration predicates on extraction_status -- did the index move?"
    for name, found in predicates.items():
        assert EXTRACTION_STATUS_PENDING in found, (
            f"{name} predicates on extraction_status but not on {EXTRACTION_STATUS_PENDING!r}: {found}"
        )


def test_the_field_is_not_narrowed_to_a_closed_type() -> None:
    """``extraction_status`` stays ``str | None``.

    The ruling in §3.5 is that this vocabulary is open at the type level: the
    column carries no CHECK constraint, consumers assign and compare bare
    ``str``, and a producer may store a value the constants have not caught up
    with. Narrowing to a ``Literal`` or ``StrEnum`` would make every such
    value a type error at the reader rather than an unrecognised status it can
    ignore -- and would force a ruling on ``None`` versus ``'none'``, which is
    a migration, not a contract edit.
    """
    annotation = MediaInfo.__annotations__["extraction_status"]

    assert annotation == "str | None", (
        f"MediaInfo.extraction_status is annotated {annotation!r}, not 'str | None'. "
        "See docs/search-spec.md §3.5 -- this vocabulary is deliberately open."
    )


#: Shapes the window regex must classify, kept beside it because the scan
#: above passes trivially against an extractor that finds nothing: every
#: spelling in the tree already complies, so a narrowed regex breaks no test
#: unless one exists to break.
_EXTRACTOR_CASES = [
    pytest.param("\"WHERE extraction_status = 'pending'\"", {"pending"}, id="index-predicate"),
    pytest.param(
        "\"ALTER TABLE media ADD COLUMN extraction_status TEXT NOT NULL DEFAULT 'none'\"",
        {"none"},
        id="column-default-in-sql",
    ),
    pytest.param(
        'Column("extraction_status", STRING_TYPE, server_default="\'none\'::text")',
        {"none"},
        id="column-declaration-across-arguments",
    ),
    pytest.param(
        'Column("extraction_status",\n    STRING_TYPE,\n    server_default="\'none\'::text",\n)',
        {"none"},
        id="column-declaration-across-lines",
    ),
    pytest.param('"ON media (extraction_status) "', set(), id="mention-with-no-value"),
    pytest.param("\"metadata_json JSONB DEFAULT '{}'\"", set(), id="unrelated-column"),
]


@pytest.mark.parametrize(("source", "expected"), _EXTRACTOR_CASES)
def test_the_extractor_finds_a_status_in_the_shapes_that_carry_one(source: str, expected: set[str]) -> None:
    """The detector itself, so a narrowed window fails here rather than silently.

    :param source: the fragment under inspection
    :ptype source: str
    :param expected: the spellings the window must recover from it
    :ptype expected: set[str]
    """
    assert _statuses_in(source) == expected
