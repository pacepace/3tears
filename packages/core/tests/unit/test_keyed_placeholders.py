"""a ``$n`` is a placeholder only where SQL reads one: never text inside a quoted literal, a quoted name, a dollar-quoted
string or a comment (:mod:`threetears.interfaces.keyed.filters`).

Reports' Contest Details for California's 'Proposition: 40 - Impose one-time wealth tax over $1 billion' failed with
``ValueError: a fragment names $1 but binds 0 parameters``: the race's name, a literal in the query
``keyed.ranked_order`` orders (a :class:`Derived` binding nothing), was read as naming ``$1`` when the relation was
renumbered into the statement (2026-11-18, plan-2 probe). Every builder that moves or fills placeholders reads them
the one way.
"""

from __future__ import annotations

import duckdb
import pytest

from threetears.core.keyed import (
    Derived,
    Fragment,
    Ranked,
    Sort,
    ThreadedDuckDB,
    keyset_rows,
    literal_select,
    ranked_order,
)
from threetears.interfaces.keyed.filters import one_use_each, renumbered

#: the race whose name broke the page
_TITLE = "Proposition: 40 - Impose one-time wealth tax over $1 billion"

#: texts holding a ``$`` that is no placeholder, each as SQL writes it
_NOT_PLACEHOLDERS = (
    f"'{_TITLE}'",
    "'$'",
    "'$$'",
    "'costs $10 or $1'",
    "'it''s $2'",
    "E'it\\'s $3'",
    '"col$1"',
    '"a ""$2"" b"',
    "$$ a dollar-quoted $1 $$",
    "$body$ $2 and $3 $body$",
    "-- a comment naming $1\n",
    "/* a block naming $2 */",
)


@pytest.mark.parametrize("text", _NOT_PLACEHOLDERS)
def test_text_holding_a_dollar_is_not_renumbered(text: str) -> None:
    """Flips if renumbered reads a ``$n`` inside a literal, a quoted name, a dollar-quoted string or a comment as a
    placeholder: refused as naming a parameter it binds none of, or moved as one."""
    sql = f"t.name = {text}"
    assert renumbered(Fragment(sql, ()), 3) == sql


@pytest.mark.parametrize("text", _NOT_PLACEHOLDERS)
def test_real_placeholders_beside_such_text_are_renumbered_and_the_text_is_not(text: str) -> None:
    sql = f"t.a = $1 AND t.name = {text} AND t.b = $2"
    expected = f"t.a = $4 AND t.name = {text} AND t.b = $5"
    assert renumbered(Fragment(sql, (1, 2)), 3) == expected


def test_a_placeholder_is_still_refused_when_it_binds_nothing() -> None:
    with pytest.raises(ValueError, match=r"names \$2 but binds 1"):
        renumbered(Fragment("t.a = $1 AND t.b = $2 AND t.c = '$9'", (1,)), 0)


def test_two_digit_placeholders_and_a_bare_dollar_sign() -> None:
    params = tuple(range(10))
    assert renumbered(Fragment("t.a = $10 AND t.b = $1 AND t.c = 'x' || $2", params), 1) == (
        "t.a = $11 AND t.b = $2 AND t.c = 'x' || $3"
    )
    assert renumbered(Fragment("t.a = $1 AND t.price LIKE '$%'", (1,)), 2) == "t.a = $3 AND t.price LIKE '$%'"


@pytest.mark.parametrize("text", _NOT_PLACEHOLDERS)
def test_one_use_each_leaves_such_text_alone(text: str) -> None:
    sql, params = one_use_each(f"SELECT $1, $1 WHERE x = {text}", ["v"])
    assert (sql, params) == (f"SELECT $1, $2 WHERE x = {text}", ["v", "v"])


def test_the_race_whose_name_broke_the_page_is_ordered() -> None:
    """Flips if ranked_order refuses (or renumbers) a relation whose text holds the race's name as a literal."""
    db = duckdb.connect()
    db.execute("CREATE TABLE races (name VARCHAR, votes BIGINT)")
    db.execute(
        "INSERT INTO races VALUES (?, 1), ('Proposition: 3 - Bonds', 2), ('Proposition: 10 - Rent', 3)", [_TITLE]
    )
    relation = Derived(f"SELECT name, votes FROM races WHERE name = '{_TITLE}' OR name LIKE 'Proposition: %'")
    step = ranked_order(relation, [Ranked("name", "regexp_extract(v, '[0-9]+')::int, v")])
    assert step.params == ()
    rows = db.execute(step.sql).fetchall()
    assert [row[0] for row in rows] == ["Proposition: 3 - Bonds", "Proposition: 10 - Rent", _TITLE]


def test_a_literal_select_fills_placeholders_and_never_text_like_one() -> None:
    """Flips if a literal select fills a ``$n`` inside its own filter's literal with a value."""
    written = literal_select("races", columns=["name"], by={"votes": 1}, where=[Fragment(f"t.name <> '{_TITLE}'", ())])
    assert f"'{_TITLE}'" in written
    assert "$" not in written.replace(f"'{_TITLE}'", "")


async def test_a_keyset_read_through_a_filter_naming_the_race() -> None:
    """a listing's own filter holding the name as a literal reads its rows, its placeholders renumbered past the
    listing's."""
    db = duckdb.connect()
    db.execute("CREATE TABLE races (race_id BIGINT PRIMARY KEY, name VARCHAR)")
    db.execute("INSERT INTO races VALUES (1, ?), (2, 'other')", [_TITLE])
    rows = await keyset_rows(
        ThreadedDuckDB(db),
        "races",
        columns=["race_id", "name"],
        by={},
        order=[Sort("race_id")],
        where=[Fragment(f"t.name = '{_TITLE}' AND t.race_id >= $1", (1,))],
    )
    assert [row["name"] for row in rows] == [_TITLE]


def _natural_order(column: str) -> str:
    """reports' ``evaluate.natural_order`` key, as its Contest Details, County Types and Result Comparison pages
    order their labels: the text with its first run of digits compared as a number."""
    text = f"typeof({column}) = 'VARCHAR'"
    as_text = f"CAST({column} AS VARCHAR)"
    return (
        f"CASE WHEN {text} THEN regexp_replace({as_text}, '[0-9]+', '0', 'g') END, "
        f"CASE WHEN {text} THEN TRY_CAST(regexp_extract({as_text}, '[0-9]+') AS BIGINT) END, {column}"
    )


def test_a_reports_page_naming_the_race_orders_its_rows() -> None:
    """Flips on the shape that failed reports' pages for CA's Proposition 40: a page's query naming the race as a
    literal (its filter), its rows ordered by ``ranked_order`` over reports' natural-order key, a plain sort first."""
    db = duckdb.connect()
    db.execute("CREATE TABLE results (state VARCHAR, race VARCHAR, county VARCHAR, votes BIGINT)")
    db.executemany(
        "INSERT INTO results VALUES ('CA', ?, ?, ?)",
        [(_TITLE, "County 10", 5), (_TITLE, "County 2", 7), ("Proposition: 3 - Bonds", "County 1", 1)],
    )
    page = f"SELECT race, county, votes FROM results WHERE state = 'CA' AND race = '{_TITLE}'"
    step = ranked_order(
        Derived(page),
        [Ranked("race", _natural_order("v")), Ranked("county", _natural_order("v"))],
        first=[Sort("votes", descending=True, nulls="last")],
    )
    assert step.params == ()
    assert f"'{_TITLE}'" in step.sql
    assert [row[1] for row in db.execute(step.sql).fetchall()] == ["County 2", "County 10"]
