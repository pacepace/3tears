"""every keyed helper sends the statements it sent before its builders moved to :mod:`threetears.interfaces.keyed`.

The pure half of the keyed helpers (their vocabulary, and the builders writing a call's SQL and binds) lives in the
dependency-free interfaces package; core keeps the runners. The move changed no statement: each case below runs a
helper on a recording reader, and the text and binds it sent (each bind by its ``repr``, so a type change shows)
are compared with ``keyed_statements.golden.json``, recorded from the helpers as they were before the move
(``RECORD_KEYED_GOLDEN=1`` writes it again; never to make a failing case pass).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from threetears.core import keyed
from threetears.core.keyed import (
    AnyOf,
    Derived,
    Dialect,
    Fragment,
    Increment,
    KeyBatch,
    Lock,
    Selected,
    Sort,
)

_GOLDEN = Path(__file__).with_name("keyed_statements.golden.json")

_WHEN = datetime(2026, 11, 11, 12, 30, tzinfo=UTC)
_ID = uuid.UUID("0190d2f4-6c2e-7e3a-8b1c-1234567890ab")


class _Recorder:
    """a reader keeping every statement it is sent and answering from a queue of row lists (none when empty)."""

    def __init__(
        self,
        *answers: list[dict[str, Any]],
        dialect: Dialect = Dialect.POSTGRES,
        native: bool = False,
    ) -> None:
        self.sql_dialect = dialect
        self.binds_native_arrays = native
        self._answers = list(answers)
        self.sent: list[list[str]] = []

    def _keep(self, query: str, params: Sequence[Any]) -> None:
        self.sent.append([query, repr(tuple(params))])

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self._keep(query, params)
        return self._answers.pop(0) if self._answers else []

    async def execute(self, query: str, *params: Any) -> str:
        self._keep(query, params)
        verb = query.split()[0].upper() if not query.startswith("/*") else query.split("*/")[1].split()[0].upper()
        return "INSERT 0 2" if verb == "INSERT" else f"{verb} 2"

    async def copy_from_query(self, query: str, *params: Any, output: Any, **options: Any) -> None:
        self._keep(query, [*params, options])
        await output(b"")


Case = Callable[[], Awaitable[list[list[str]]]]


def _run(make: Callable[[_Recorder], Awaitable[Any]], recorder: _Recorder | None = None) -> Case:
    async def case() -> list[list[str]]:
        reader = recorder if recorder is not None else _Recorder()
        await make(reader)
        return reader.sent

    return case


_ORDER = (Sort("name"), Sort("id", cast="uuid"))
_MIXED = (Sort("due_at", cast="timestamptz", descending=True, nullable=True), Sort("id", cast="bigint"))
_EXPR = (Sort("lname", expression='lower(t."name")'), Sort("id", cast="bigint"))


def _cases() -> dict[str, Case]:
    cases: dict[str, Case] = {
        # point reads
        "read_by.eq": _run(lambda c: keyed.read_by(c, "s.t", by={"a": "k", "b": 2}, columns=("a", "b"))),
        "read_by.all_columns": _run(lambda c: keyed.read_by(c, "t", by={"a": None})),
        "read_by.anyof_selected_lock": _run(
            lambda c: keyed.read_by(
                c,
                "t",
                by={"a": AnyOf(["x", "y"], cast="uuid"), "z": AnyOf([1, 2], as_list=True), "w": AnyOf([3], cast=None)},
                columns=("a", Selected("CAST(b AS VARCHAR)", "b")),
                lock=Lock.UPDATE_SKIP_LOCKED,
            )
        ),
        "read_row.lock_nowait": _run(lambda c: keyed.read_row(c, "t", by={"id": _ID}, lock=Lock.UPDATE_NOWAIT)),
        "read_row.records": _run(lambda c: keyed.read_row(c, "t", by={"id": 1}, columns=("v",), records=True)),
        "read_value": _run(lambda c: keyed.read_value(c, "t", by={"id": 1, "flag": True}, column="v")),
        "read_by.derived": _run(lambda c: keyed.read_by(c, Derived("SELECT 1 AS a"), by={"a": 1}, columns=("a",))),
        # keyed batches
        "read_keyed.one_column": _run(
            lambda c: keyed.read_keyed(
                c, "t", keys=KeyBatch(("id",), [(1,), (2,), (1,)], ("bigint",)), columns=("id", "v")
            )
        ),
        "read_keyed.one_column_native": _run(
            lambda c: keyed.read_keyed(
                c, "t", keys=KeyBatch(("id",), [(1,), (2,)], ("bigint",)), columns=("id",), by={"tenant": "x"}
            ),
            _Recorder(native=True),
        ),
        "read_keyed.native_text_values": _run(
            lambda c: keyed.read_keyed(c, "t", keys=KeyBatch(("id",), [("1",), ("2",)], ("bigint",)), columns=("id",)),
            _Recorder(native=True),
        ),
        "read_keyed.two_columns_bounded": _run(
            lambda c: keyed.read_keyed(
                c,
                "t",
                keys=KeyBatch(("a", "b"), [("s", 1), ("s", 2), ("s", 3)], ("text", "bigint")),
                columns=("a", "b"),
                by={"tenant": AnyOf(["x"])},
                where=[Fragment('t."v" > $1', (5,))],
            )
        ),
        "read_keyed.two_columns_scattered": _run(
            lambda c: keyed.read_keyed(
                c,
                "t",
                keys=KeyBatch(("a", "b"), [("s", 1), ("u", 2), ("w", 3)], ("text", "date")),
                columns=("a",),
            )
        ),
        "read_keyed.null_safe_lock": _run(
            lambda c: keyed.read_keyed(
                c,
                "t",
                keys=KeyBatch(("a", "b"), [(None, 1), ("s", 2)], ("text", "bigint"), null_safe=True),
                columns=("a", "b"),
                lock=Lock.UPDATE,
                where=[Fragment('t."v" = $1', ("q",))],
            )
        ),
        "read_keyed.as_list_batches": _run(
            lambda c: keyed.read_keyed(
                c,
                "documents",
                keys=KeyBatch(("id",), [("a",), ("b",), ("c",)], as_list=True),
                columns=("body",),
                by={"scope_id": "s", "doc_type": "run"},
                batch=2,
            )
        ),
        "read_keyed.as_list_two_columns": _run(
            lambda c: keyed.read_keyed(
                c, "t", keys=KeyBatch(("a", "b"), [("x", 1), ("y", 2)], as_list=True), columns=("a",), by={"c": 1}
            )
        ),
        "read_joined": _run(
            lambda c: keyed.read_joined(
                c, [{"p": 1}, {"p": 2}, {"p": None}], "t", on={"id": "p"}, columns=("id", "v"), casts={"id": "bigint"}
            )
        ),
        "any_rows": _run(
            lambda c: keyed.any_rows(c, "t", ("a",), limit=5, by={"a": 1}, where=[Fragment('t."b" < $1', (2,))])
        ),
        # pages
        "keyset_page.first": _run(
            lambda c: keyed.keyset_page(c, "t", columns=("name", "id"), by={"tenant": "x"}, order=_ORDER, limit=10)
        ),
        "keyset_page.cursor_by_row": _run(
            lambda c: keyed.keyset_page(
                c,
                "t",
                columns=("name", "id"),
                by={"tenant": "x", "kind": AnyOf(["a", "b"])},
                order=_ORDER,
                limit=10,
                cursor=keyed.keyset_cursor(_ORDER, {"name": "n", "id": _ID}),
                where=[Fragment('t."due" <= $1', (_WHEN,))],
                visible=[keyed.exists_keyed("grants", on={"doc": "id"}, by={"who": "me", "role": AnyOf(["r"])})],
                lock=Lock.SHARE,
            )
        ),
        "keyset_page.cursor_yugabyte": _run(
            lambda c: keyed.keyset_page(
                c,
                "t",
                columns=("name", "id"),
                by={"tenant": "x"},
                order=_ORDER,
                limit=10,
                cursor=keyed.keyset_cursor(_ORDER, {"name": "n", "id": _ID}),
            ),
            _Recorder(dialect=Dialect.YUGABYTE),
        ),
        "keyset_page.nullable_desc_null_cursor": _run(
            lambda c: keyed.keyset_page(
                c,
                "t",
                columns=("due_at", "id"),
                by={},
                order=_MIXED,
                limit=3,
                cursor=keyed.keyset_cursor(_MIXED, {"due_at": None, "id": 7}),
            )
        ),
        "keyset_page.nullable_desc_value_cursor": _run(
            lambda c: keyed.keyset_page(
                c,
                "t",
                columns=("due_at", "id"),
                by={"a": None},
                order=_MIXED,
                limit=3,
                cursor=keyed.keyset_cursor(_MIXED, {"due_at": _WHEN, "id": 7}),
            )
        ),
        "keyset_page.expression_more": _run(
            lambda c: keyed.keyset_page(c, "t", columns=("id",), by={"k": 1}, order=_EXPR, limit=1),
            _Recorder([{"id": 1, "lname": "a"}, {"id": 2, "lname": "b"}]),
        ),
        "keyset_page.uncast_after_more": _run(
            lambda c: keyed.keyset_page(
                c,
                "documents",
                columns=("seq", "body"),
                by={"scope_id": "s"},
                order=(Sort("seq", cast=None),),
                limit=1,
                cursor=keyed.keyset_cursor((Sort("seq", cast=None),), {"seq": 4}),
            ),
            _Recorder(dialect=Dialect.DUCKDB),
        ),
        "keyset_rows.pages": _run(
            lambda c: keyed.keyset_rows(
                c, "t", columns=("name", "id"), by={"tenant": "x"}, order=_ORDER, page=2, where=[Fragment("TRUE")]
            ),
            _Recorder(
                [{"name": "a", "id": _ID}, {"name": "b", "id": _ID}, {"name": "c", "id": _ID}],
                [{"name": "d", "id": _ID}],
            ),
        ),
        "keyset_rows.uncast_pages": _run(
            lambda c: keyed.keyset_rows(
                c,
                "documents",
                columns=("id", "seq"),
                by={"scope_id": "s", "doc_type": "r"},
                order=(Sort("seq", cast=None),),
                page=1,
            ),
            _Recorder([{"id": "a", "seq": 1}, {"id": "b", "seq": 2}], [{"id": "c", "seq": 3}]),
        ),
        "keyset_rows.mixed_pages_yugabyte": _run(
            lambda c: keyed.keyset_rows(c, "t", columns=("due_at", "id"), by={"k": 1}, order=_MIXED, page=1),
            _Recorder(
                [{"due_at": _WHEN, "id": 1}, {"due_at": None, "id": 2}],
                [{"due_at": None, "id": 3}],
                dialect=Dialect.YUGABYTE,
            ),
        ),
        "copy_keyed": _run(
            lambda c: keyed.copy_keyed(
                c,
                "t",
                columns=("a", Selected('t."b"::text', "b")),
                by={"k": 1},
                order=_ORDER,
                where=[Fragment('t."v" > $1', (1,))],
                force_quote=("a",),
            )
        ),
        # writes
        "update_keyed.by_where_returning": _run(
            lambda c: keyed.update_keyed(
                c,
                "s.t",
                values={"status": "done", "at": Fragment("now() + $1", ("1 day",)), "n": Increment(1)},
                by={"id": _ID, "gone": None},
                where=[Fragment('"status" = $1', ("open",))],
                returning=("id", "status"),
            ),
            _Recorder([{"id": _ID, "status": "done"}]),
        ),
        "update_keyed.keys_batches": _run(
            lambda c: keyed.update_keyed(
                c,
                "t",
                values={"v": 1},
                keys=KeyBatch(("a", "b"), [("x", 1), ("y", 2), ("z", 3)], ("text", "bigint")),
                by={"tenant": "q"},
                batch=2,
            )
        ),
        "update_keyed.keys_native": _run(
            lambda c: keyed.update_keyed(c, "t", values={"v": 1}, keys=KeyBatch(("id",), [(1,), (2,)], ("bigint",))),
            _Recorder(native=True),
        ),
        "update_keyed.keys_as_list": _run(
            lambda c: keyed.update_keyed(
                c, "t", values={"v": 1}, keys=KeyBatch(("a", "b"), [("x", 1), ("y", 2)], as_list=True)
            )
        ),
        "update_keyed.prefix": _run(lambda c: keyed.update_keyed(c, "t", values={"v": 1}, prefix={"signal": "name/"})),
        "delete_keyed.by": _run(lambda c: keyed.delete_keyed(c, "t", by={"scope_id": "s", "id": "d"})),
        "delete_keyed.where_returning": _run(
            lambda c: keyed.delete_keyed(
                c, "t", by={"key": "k"}, where=[Fragment('"expires_at" > $1', (_WHEN,))], returning=("payload",)
            ),
            _Recorder([{"payload": {}}]),
        ),
        "delete_keyed.null_safe_keys": _run(
            lambda c: keyed.delete_keyed(
                c, "t", keys=KeyBatch(("a", "b"), [(None, "1"), ("s", "2")], ("text", "uuid"), null_safe=True)
            )
        ),
        "delete_keyed.single_key_text": _run(lambda c: keyed.delete_keyed(c, "t", keys=KeyBatch(("a",), [("x",)]))),
        "delete_keyed.empty_keys": _run(lambda c: keyed.delete_keyed(c, "t", keys=KeyBatch(("a",), []))),
        # sweeps
        "delete_before.pages": _run(
            lambda c: keyed.delete_before(
                "t", collection=None, key=("hashed",), column="expires_at", before=_WHEN, page=2, conn=c
            ),
            _Recorder([{"hashed": "a"}, {"hashed": "b"}], [{"hashed": "c"}]),
        ),
        "delete_before.by_most_pages": _run(
            lambda c: keyed.delete_before(
                "s.t",
                collection=None,
                key=("a", "b"),
                column="at",
                before=_WHEN,
                page=1,
                by={"purpose": "p"},
                most_pages=1,
                conn=c,
            ),
            _Recorder([{"a": 1, "b": 2}]),
        ),
        "update_before.pages": _run(
            lambda c: keyed.update_before(
                "t",
                collection=None,
                key=("id",),
                column="at",
                before=_WHEN,
                set_to={"status": "stale"},
                by={"status": "dispatching"},
                page=5,
                conn=c,
            )
        ),
    }
    return cases


_CASES = _cases()


def _recorded() -> dict[str, list[list[str]]]:
    return json.loads(_GOLDEN.read_text(encoding="utf-8")) if _GOLDEN.is_file() else {}


@pytest.mark.parametrize("name", sorted(_CASES))
async def test_a_helper_sends_the_statements_it_sent_before_the_move(name: str) -> None:
    """Flips if moving a builder changed one character of a statement, or the value or type of one bind."""
    sent = await _CASES[name]()
    if os.environ.get("RECORD_KEYED_GOLDEN") == "1":
        golden = _recorded()
        golden[name] = sent
        _GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    assert sent == _recorded()[name]


def test_every_recorded_case_is_still_run() -> None:
    """Flips if a case is dropped from the corpus: the golden names exactly the cases run."""
    assert sorted(_recorded()) == sorted(_CASES)
