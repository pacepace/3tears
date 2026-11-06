"""A statement's operation, read from a plain statement in a few searches, is exactly the one its tokens give, and a
page's multi-row ``INSERT`` is labelled in a few milliseconds at most: the event loop labels every statement it sends.

A trailing line comment changes nothing a statement does, and puts it off the plain path onto the tokens, so a
statement labelled the same with and without one is labelled the same both ways."""

from __future__ import annotations

import random
import time

import pytest

from threetears.core.backends.statement_operation import statement_operation

#: a line comment closing a statement: read only as tokens, and changing nothing the statement does
_READ_AS_TOKENS = "\n-- read as tokens"

#: statements whose names, quoting and words put a keyword or a mark where it is not code
_TRICKY = [
    "SELECT 1",
    "select a, b from t where x = $1",
    "INSERT INTO t (a) VALUES ($1), ($2) ON CONFLICT (a) DO UPDATE SET a = excluded.a",
    "insert into t values ('on conflict')",
    'INSERT INTO "t" ("on conflict") VALUES ($1)',
    "INSERT INTO t VALUES ($1) xON CONFLICT",
    "INSERT INTO t VALUES ($1) x9ON CONFLICT",
    "INSERT INTO t VALUES ($1) 9ON CONFLICT",
    "INSERT INTO t VALUES ($1) ON\n\tCONFLICT DO NOTHING",
    "INSERT INTO t VALUES ($1) ON CONFLICTS",
    "WITH update AS (SELECT 1) DELETE FROM t",
    'UPDATE "t" SET "a" = $1 WHERE "k" = $2',
    '"x" SELECT',
    "(SELECT 1)",
    "  delete from t",
    "SELECT $$ (DELETE) $$",
    "SELECT 1 -- ON CONFLICT",
    "SELECT /* nested */ 1",
    "",
]

#: the pieces the fuzz joins: every kind of token and its edges
_PIECES = [
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "ON",
    "CONFLICT",
    "WITH",
    "AS",
    "(",
    ")",
    ",",
    " ",
    "\n",
    "\t",
    "'",
    "E'",
    "\\",
    '"',
    '""',
    "$",
    "$1",
    "$$",
    "$t$",
    "--",
    "/",
    "*",
    "x",
    "E",
    "a$b",
    "_1",
    "9",
    ";",
    "=",
    "é",
    "\x1c",
]


@pytest.mark.parametrize("statement", _TRICKY)
def test_a_plain_statement_is_labelled_as_its_tokens_label_it(statement: str) -> None:
    assert statement_operation(statement) == statement_operation(statement + _READ_AS_TOKENS)


def test_any_mix_of_quotes_comments_and_words_is_labelled_the_same_both_ways() -> None:
    draw = random.Random(1029)
    plain = 0
    for _ in range(30_000):
        statement = "".join(draw.choice(_PIECES) for _ in range(draw.randrange(1, 14)))
        plain += not any(mark in statement for mark in ("'", "--", "/*", "$$", "$t$"))
        assert statement_operation(statement) == statement_operation(statement + _READ_AS_TOKENS), repr(statement)
    assert plain > 5_000, "the fuzz hardly reached the plain path"


def test_a_pages_upsert_is_labelled_in_a_few_milliseconds() -> None:
    rows = ", ".join(f"(${n * 4 + 1}, ${n * 4 + 2}, ${n * 4 + 3}, ${n * 4 + 4})" for n in range(1000))
    # a table of its own per page, as each page's text differs: no label is remembered from the page before
    pages = [
        f'INSERT INTO "t{page}" ("a", "b", "c", "d") VALUES {rows} ON CONFLICT ("a") DO UPDATE SET "b" = excluded."b"'
        for page in range(10)
    ]
    began = time.perf_counter()
    labels = {statement_operation(page) for page in pages}
    took = (time.perf_counter() - began) / len(pages)
    assert labels == {"upsert"} == {statement_operation(pages[0] + _READ_AS_TOKENS)}
    # about 1 ms on an M-series laptop, against about 20 ms read as tokens
    assert took < 0.005, f"{took * 1000:.2f} ms"
