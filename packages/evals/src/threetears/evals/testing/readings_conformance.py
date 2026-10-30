"""The readings conformance kit: every rule of the optional ``ResultReadings`` capability, as a case a store runs.

A store that offers :class:`~threetears.evals.schema.ResultReadings` beside the
:class:`~threetears.evals.schema.DocumentStore` port promises the engine the same answers it would get from
:meth:`~threetears.evals.schema.DocumentStore.by_doc_type`, only narrowed, grouped and in one backend call. This
module states each of those promises as a check, in the store conformance kit's pattern
(:mod:`threetears.evals.testing.store_conformance`): a :class:`ResultReadingsConformanceCase` takes a fresh, empty
store that offers the capability, and returns or raises
:class:`~threetears.evals.testing.StoreConformanceFailure`, whose message names the case and the rule it broke. A
store that does not offer the capability needs none of this: the engine reads it through the port alone.

**A case also takes a hook installer.** Two promises are about the backend rather than the answer: a read is ONE
backend call, and a backend failure raises. No check can see a backend from outside, so the adopter passes
``install_backend_hook``: a function that takes a callback and arranges for the store to call it once per query it
issues (a statement executed, a cursor opened, a request sent), replacing any callback installed before. Fetching
the next page of a query already issued is not another query: a store that streams through a server-side cursor
calls the hook when it opens the cursor and not on each ``FETCH``. A case counts the calls
with it, or makes the backend fail by installing a callback that raises; the store must let that exception reach the
caller, as itself or as the cause of its own. Under pytest, for a store whose driver can count its statements::

    import pytest
    from threetears.evals.testing import RESULT_READINGS_CONFORMANCE_CASES, ResultReadingsConformanceCase

    @pytest.mark.parametrize("case", RESULT_READINGS_CONFORMANCE_CASES, ids=lambda case: case.name)
    def test_my_store_offers_readings(case: ResultReadingsConformanceCase, tmp_path) -> None:
        store = MyDocumentStore(tmp_path / "evals.sqlite")
        case.run(store, install_backend_hook=store.connection.on_statement)

``packages/evals/tests/test_readings_conformance_sqlite.py`` installs it on a SQLite connection, and
``test_readings_conformance_memory.py`` on the in-memory store's dict.

**Every case is mandatory** for a store that offers the capability. Each case writes only under the kit's own scopes
and ``doc_type`` values, through the port's ``upsert``, and installs the hook only after its writes.

**What it cannot see.** Speed and memory: a store that answers correctly by reading every whole document passes. And
a backend whose order is not stable between reads can pass by luck, since each case compares reads made moments apart.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from threetears.evals.schema.store_port import DocumentStore, GroupMeasure, ResultReadings, narrow_items, omit_paths
from threetears.evals.testing.store_conformance import StoreConformanceFailure

__all__ = [
    "RESULT_READINGS_CONFORMANCE_CASES",
    "ResultReadingsConformanceCase",
]

_SCOPE = "readings-conformance-scope-a"
_OTHER_SCOPE = "readings-conformance-scope-b"
_TYPE = "readings_conformance_doc"
_OTHER_TYPE = "readings_conformance_other"


@dataclass(frozen=True)
class ResultReadingsConformanceCase:
    """One rule of the readings capability, as a check over a store and its backend hook.

    :ivar name: A stable identifier, for a test id (``"grouped.sum_int"``).
    :ivar rule: The rule, in one sentence, as the capability states it.
    :ivar check: The check. Takes a fresh, empty store that offers the capability, and the hook installer; returns,
        or raises :class:`~threetears.evals.testing.StoreConformanceFailure`.
    """

    name: str
    rule: str
    check: Callable[[DocumentStore, Callable[[Callable[[], None]], None]], None]

    def run(self, store: DocumentStore, install_backend_hook: Callable[[Callable[[], None]], None]) -> None:
        """Run the case against ``store``, which must be fresh, empty, and offer the capability.

        :param store: The adapter under test.
        :ptype store: DocumentStore
        :param install_backend_hook: Takes a callback and makes the store call it once per query it issues (a
            statement executed or a cursor opened, never a page fetched from one), replacing any callback installed
            before. An exception the callback raises is the backend failing.
        :ptype install_backend_hook: Callable[[Callable[[], None]], None]
        :raises StoreConformanceFailure: The store broke the rule, or does not offer the capability; the message
            names the case, the rule and what was observed. Anything else the store raises propagates unchanged.
        """
        try:
            if not isinstance(store, ResultReadings):
                raise StoreConformanceFailure(
                    f"{type(store).__name__} does not offer ResultReadings (iter_narrowed and grouped)"
                )
            self.check(store, install_backend_hook)
        except StoreConformanceFailure as failure:
            raise StoreConformanceFailure(f"{self.name}: {self.rule} — {failure}") from None


# --- helpers ----------------------------------------------------------------------------------------


class _BackendDown(Exception):
    """The failure a case's hook raises as the backend's own."""


def _doc(doc_id: str, *, scope: str = _SCOPE, doc_type: str = _TYPE, **fields: Any) -> dict[str, Any]:
    return {"id": doc_id, "scope_id": scope, "doc_type": doc_type, **fields}


def _write(store: DocumentStore, *documents: dict[str, Any]) -> None:
    for document in documents:
        store.upsert(document)


def _readings(store: DocumentStore) -> ResultReadings:
    if not isinstance(store, ResultReadings):
        raise StoreConformanceFailure(f"{type(store).__name__} does not offer ResultReadings")
    return store


def _expect(condition: bool, observed: str) -> None:
    if not condition:
        raise StoreConformanceFailure(observed)


def _expect_equal(actual: object, expected: object, what: str) -> None:
    _expect(actual == expected, f"{what}: expected {expected!r}, got {actual!r}")


def _expect_refused(error: type[Exception], call: Callable[[], object], what: str) -> None:
    try:
        call()
    except error:
        # NOSILENT: the refusal IS the conformance check passing; the failure is the call being accepted
        return
    raise StoreConformanceFailure(f"{what} was accepted; expected {error.__name__}")


def _ids(documents: Sequence[dict[str, Any]]) -> list[Any]:
    return [document.get("id") for document in documents]


def _narrowed(
    store: DocumentStore,
    *,
    exclude: Sequence[str] = (),
    exclude_in_items: Mapping[str, Sequence[str]] | None = None,
    field_in: Mapping[str, Sequence[Any]] | None = None,
    scope: str = _SCOPE,
    doc_type: str = _TYPE,
    **field_eq: Any,
) -> list[dict[str, Any]]:
    rows = _readings(store).iter_narrowed(
        doc_type, scope, exclude=exclude, exclude_in_items=exclude_in_items or {}, field_in=field_in, **field_eq
    )
    _expect(isinstance(rows, Iterator), f"iter_narrowed returned {type(rows).__name__}, not an iterator")
    return list(rows)


def _grouped(
    store: DocumentStore,
    by: Sequence[str],
    measures: Mapping[str, GroupMeasure],
    *,
    field_in: Mapping[str, Sequence[Any]] | None = None,
    scope: str = _SCOPE,
    doc_type: str = _TYPE,
    **field_eq: Any,
) -> list[dict[str, Any]]:
    return _readings(store).grouped(doc_type, scope, by=by, measures=measures, field_in=field_in, **field_eq)


def _one_group(store: DocumentStore, measures: Mapping[str, GroupMeasure], **field_eq: Any) -> dict[str, Any]:
    groups = _grouped(store, [], measures, **field_eq)
    _expect(len(groups) == 1, f"grouped with no by paths answered {len(groups)} groups, expected one: {groups!r}")
    return groups[0]


def _key(value: Any) -> tuple[str, Any]:
    """A group's ``by`` value as a dict key that keeps ``True`` and ``1`` apart."""
    return ("boolean" if isinstance(value, bool) else "value", value)


def _by_key(groups: list[dict[str, Any]], path: str) -> dict[tuple[str, Any], dict[str, Any]]:
    """The groups keyed by their ``path`` value; two groups with one value fail the case."""
    keyed: dict[tuple[str, Any], dict[str, Any]] = {}
    for group in groups:
        key = _key(group.get(path))
        _expect(key not in keyed, f"two groups share the by value {key[1]!r}: {groups!r}")
        keyed[key] = group
    return keyed


#: Rows written in an order that is neither id order nor any field's order, so a store that sorts fails.
_ROWS = (
    _doc(
        "r3",
        run="a",
        rank=2,
        scores=[{"name": "s", "value": 1, "reasoning": "long"}, "plain", {"reasoning": "only"}],
        meta={"note": "x", "big": "drop"},
    ),
    _doc("r1", run="b", rank=None, scores="not a list", meta={"big": "drop"}),
    _doc("r9", run="a", rank=True, scores=[{"value": 2, "reasoning": "long", "sub": {"reasoning": "deep"}}]),
    _doc("r5", run="c", flag=1, scores=[]),
    _doc("r2", run="a", rank=1, flag=True, scores=[{"value": 3}], meta="text"),
)


# --- narrowed reads ---------------------------------------------------------------------------------


def _narrowed_equals_by_doc_type(store: DocumentStore, _install: Callable[[Callable[[], None]], None]) -> None:
    _write(store, *_ROWS, _doc("o1", doc_type=_OTHER_TYPE, run="a"))
    whole = store.by_doc_type(_TYPE, _SCOPE)
    _expect_equal(_ids(_narrowed(store)), _ids(whole), "iter_narrowed's rows and order beside by_doc_type's")
    _expect_equal(_narrowed(store), whole, "iter_narrowed with nothing to narrow")
    exclude = ["meta.big", "rank"]
    items = {"scores": ["reasoning"]}
    expected = [narrow_items(omit_paths(row, exclude), items) for row in store.by_doc_type(_TYPE, _SCOPE, run="a")]
    _expect_equal(
        _narrowed(store, exclude=exclude, exclude_in_items=items, run="a"),
        expected,
        "iter_narrowed beside by_doc_type with omit_paths and narrow_items applied",
    )


def _list_items_are_narrowed(store: DocumentStore, _install: Callable[[Callable[[], None]], None]) -> None:
    _write(
        store,
        _doc(
            "n1",
            scores=[{"value": 1, "reasoning": "r", "sub": {"reasoning": "s", "keep": 1}}, "plain", 7, None, ["x"]],
            nested={"list": [{"drop": 1, "keep": 2}]},
            obj={"reasoning": "an object, not a list"},
            text="reasoning",
        ),
    )
    rows = _narrowed(
        store,
        exclude_in_items={
            "scores": ["reasoning", "sub.reasoning"],
            "nested.list": ["drop"],
            "obj": ["reasoning"],
            "text": ["reasoning"],
            "absent": ["reasoning"],
            "text.deeper": ["x"],
        },
    )
    _expect_equal(
        rows,
        [
            _doc(
                "n1",
                scores=[{"value": 1, "sub": {"keep": 1}}, "plain", 7, None, ["x"]],
                nested={"list": [{"keep": 2}]},
                obj={"reasoning": "an object, not a list"},
                text="reasoning",
            )
        ],
        "exclude_in_items: each object item narrowed, other items, non-list fields and dead paths left alone",
    )
    _expect_equal(
        store.get("n1", _SCOPE),
        _doc(
            "n1",
            scores=[{"value": 1, "reasoning": "r", "sub": {"reasoning": "s", "keep": 1}}, "plain", 7, None, ["x"]],
            nested={"list": [{"drop": 1, "keep": 2}]},
            obj={"reasoning": "an object, not a list"},
            text="reasoning",
        ),
        "the stored document after a narrowed read",
    )


def _each_read_is_one_backend_call(store: DocumentStore, install: Callable[[Callable[[], None]], None]) -> None:
    _write(store, *(_doc(f"c{index:04d}", run="a" if index % 2 else "b", n=index) for index in range(600)))
    calls: list[int] = []
    install(lambda: calls.append(1))
    rows = _readings(store).iter_narrowed(_TYPE, _SCOPE, exclude=["n"], run="a")
    _expect_equal(len(calls), 1, "backend calls made by the iter_narrowed call itself")
    _expect_equal(len(list(rows)), 300, "rows iter_narrowed yielded")
    _expect_equal(len(calls), 1, "backend calls after every row of iter_narrowed was yielded")
    calls.clear()
    _grouped(store, ["run"], {"rows": ("count", None), "n": ("values", "n"), "top": ("max", "n")})
    _expect_equal(len(calls), 1, "backend calls made by one grouped read")
    calls.clear()
    _expect_equal(_narrowed(store, field_in={"run": []}), [], "iter_narrowed with an empty field_in")
    _expect_equal(_grouped(store, ["run"], {"rows": ("count", None)}, field_in={"run": []}), [], "an empty field_in")
    _expect_equal(len(calls), 0, "backend calls made for an empty field_in, which matches nothing")


def _reads_stay_in_their_scope(store: DocumentStore, _install: Callable[[Callable[[], None]], None]) -> None:
    _write(
        store,
        _doc("s1", run="a", n=1),
        _doc("s1", scope=_OTHER_SCOPE, run="a", n=10),
        _doc("s2", scope=_OTHER_SCOPE, run="a", n=20),
        _doc("t1", doc_type=_OTHER_TYPE, run="a", n=100),
    )
    _expect_equal(_ids(_narrowed(store, run="a")), ["s1"], "iter_narrowed in scope a")
    _expect_equal(_narrowed(store)[0]["n"], 1, "the scope-a document of an id stored in two scopes")
    _expect_equal(_ids(_narrowed(store, scope=_OTHER_SCOPE)), ["s1", "s2"], "iter_narrowed in scope b")
    _expect_equal(
        _grouped(store, ["run"], {"rows": ("count", None), "n": ("sum_int", "n")}),
        [{"run": "a", "rows": 1, "n": 1}],
        "grouped in scope a, beside another scope and another doc_type",
    )
    _expect_equal(_grouped(store, [], {"n": ("sum_int", "n")}, scope=_OTHER_SCOPE), [{"n": 30}], "grouped in scope b")


#: Predicate sets whose matches a store reading through ``==`` or text would get wrong.
_PREDICATES: tuple[tuple[Mapping[str, Sequence[Any]] | None, dict[str, Any]], ...] = (
    (None, {"flag": True}),
    (None, {"flag": 1}),
    (None, {"rank": 1}),
    (None, {"rank": None}),
    (None, {"run": "a", "rank": 2}),
    ({"rank": [1, None]}, {}),
    ({"flag": [True]}, {}),
    ({"run": ["a", "c"]}, {"flag": None}),
    ({"code": ["1"]}, {}),
    ({"code": [1]}, {}),
)


def _predicates_match_by_doc_type(store: DocumentStore, _install: Callable[[Callable[[], None]], None]) -> None:
    _write(store, *_ROWS, _doc("t1", run="a", code="1"), _doc("t2", run="a", code=1))
    for field_in, field_eq in _PREDICATES:
        expected = _ids(store.by_doc_type(_TYPE, _SCOPE, field_in=field_in, **field_eq))
        what = f"field_in={field_in!r} field_eq={field_eq!r}"
        _expect_equal(_ids(_narrowed(store, field_in=field_in, **field_eq)), expected, f"iter_narrowed's ids, {what}")
        counted = _grouped(store, [], {"rows": ("count", None)}, field_in=field_in, **field_eq)
        _expect_equal(sum(group["rows"] for group in counted), len(expected), f"grouped's row count, {what}")
    _expect_refused(TypeError, lambda: _narrowed(store, field_in={"run": "ab"}), "iter_narrowed with a string field_in")
    _expect_refused(
        ValueError, lambda: _narrowed(store, field_in={"run": ["a"]}, run="a"), "iter_narrowed naming a field twice"
    )
    _expect_refused(
        TypeError,
        lambda: _grouped(store, [], {"rows": ("count", None)}, field_in={"run": "ab"}),
        "grouped with a string field_in",
    )


# --- grouped reads ----------------------------------------------------------------------------------


def _groups_follow_the_json_type(store: DocumentStore, _install: Callable[[Callable[[], None]], None]) -> None:
    _write(
        store,
        _doc("g1", key=True, deep={"k": "x"}),
        _doc("g2", key=1, deep={"k": "x"}),
        _doc("g3", key=None, deep={"k": "y"}),
        _doc("g4", deep={"k": "x"}),
        _doc("g5", key="1", deep="not an object"),
        _doc("g6", key=1),
    )
    keyed = _by_key(_grouped(store, ["key"], {"rows": ("count", None)}), "key")
    _expect_equal(
        {key: group["rows"] for key, group in keyed.items()},
        {_key(True): 1, _key(1): 2, _key(None): 2, _key("1"): 1},
        "group sizes by key (True and 1 apart, absent and null together, a number apart from its text)",
    )
    deep = _by_key(_grouped(store, ["deep.k"], {"rows": ("count", None)}), "deep.k")
    _expect_equal(
        {key: group["rows"] for key, group in deep.items()},
        {_key("x"): 3, _key("y"): 1, _key(None): 2},
        "group sizes by a dotted path (through a non-object, or absent, reads as null)",
    )
    pairs = _grouped(store, ["key", "deep.k"], {"rows": ("count", None)})
    _expect_equal(len(pairs), 6, f"groups by two paths: {pairs!r}")
    _expect_equal(_grouped(store, [], {"rows": ("count", None)}), [{"rows": 6}], "grouped with no by paths")
    _write(store, _doc("g7", key={"an": "object"}))
    _expect_refused(TypeError, lambda: _grouped(store, ["key"], {"rows": ("count", None)}), "grouping by an object")


def _count_and_count_null(store: DocumentStore, _install: Callable[[Callable[[], None]], None]) -> None:
    _write(
        store,
        _doc("k1", cost=0.5, usage={"in": 3}),
        _doc("k2", cost=None, usage={"in": None}),
        _doc("k3", usage="text"),
        _doc("k4", cost=0, usage={"in": 0}),
        _doc("k5", cost=False),
    )
    _expect_equal(
        _one_group(
            store, {"rows": ("count", None), "unpriced": ("count_null", "cost"), "no_in": ("count_null", "usage.in")}
        ),
        {"rows": 5, "unpriced": 2, "no_in": 3},
        "count, and count_null over absent and null (never 0 or false)",
    )


def _sum_int_sums_integers_only(store: DocumentStore, _install: Callable[[Callable[[], None]], None]) -> None:
    _write(
        store,
        _doc("i1", run="a", tokens=9007199254740991, usage={"in": 2}),
        _doc("i2", run="a", tokens=-3, usage={"in": None}),
        _doc("i3", run="a", tokens=None),
        _doc("i4", run="b"),
        _doc("w1", run="bool", tokens=True),
        _doc("w2", run="float", tokens=1.5),
        _doc("w3", run="whole_float", tokens=2.0),
        _doc("w4", run="text", tokens="3"),
    )
    _expect_equal(
        _grouped(store, ["run"], {"tokens": ("sum_int", "tokens"), "in": ("sum_int", "usage.in")}, run="a"),
        [{"run": "a", "tokens": 9007199254740988, "in": 2}],
        "sum_int over integers, skipping absent and null",
    )
    _expect_equal(
        _grouped(store, ["run"], {"tokens": ("sum_int", "tokens")}, run="b"),
        [{"run": "b", "tokens": 0}],
        "sum_int over a group with no integers",
    )
    for run in ("bool", "float", "whole_float", "text"):
        _expect_refused(
            TypeError,
            partial(_grouped, store, ["run"], {"tokens": ("sum_int", "tokens")}, run=run),
            f"sum_int over a {run} value",
        )


def _min_and_max(store: DocumentStore, _install: Callable[[Callable[[], None]], None]) -> None:
    _write(
        store,
        _doc("m1", run="n", v=9),
        _doc("m2", run="n", v=10),
        _doc("m3", run="n", v=-0.5),
        _doc("m4", run="n", v=None),
        _doc("m5", run="t", v="Zebra"),
        _doc("m6", run="t", v="apple"),
        _doc("m7", run="t", v="éclair"),
        _doc("m8", run="t"),
        _doc("m9", run="none", v=None),
        _doc("x1", run="mixed", v=1),
        _doc("x2", run="mixed", v="1"),
        _doc("x3", run="bool", v=True),
        _doc("x4", run="object", v={"a": 1}),
    )
    numbers = _grouped(store, ["run"], {"low": ("min", "v"), "high": ("max", "v")}, run="n")
    _expect_equal(numbers, [{"run": "n", "low": -0.5, "high": 10}], "min and max over numbers, compared as numbers")
    texts = _grouped(store, ["run"], {"low": ("min", "v"), "high": ("max", "v")}, run="t")
    _expect_equal(texts, [{"run": "t", "low": "Zebra", "high": "éclair"}], "min and max over strings, by code point")
    empty = _grouped(store, ["run"], {"low": ("min", "v"), "high": ("max", "v")}, run="none")
    _expect_equal(empty, [{"run": "none", "low": None, "high": None}], "min and max over a group of nulls")
    extremes: tuple[GroupMeasure, ...] = (("min", "v"), ("max", "v"))
    for run in ("mixed", "bool", "object"):
        for measure in extremes:
            _expect_refused(
                TypeError,
                partial(_grouped, store, ["run"], {"v": measure}, run=run),
                f"{measure[0]} over a {run} group",
            )


def _values_come_in_stored_order(store: DocumentStore, _install: Callable[[Callable[[], None]], None]) -> None:
    _write(
        store,
        _doc("v3", run="a", cost=0.1, extra={"k": [1, 2]}),
        _doc("v1", run="b", cost=0.2),
        _doc("v9", run="a", cost=None),
        _doc("v2", run="a"),
        _doc("v5", run="a", cost=True, extra="x"),
    )
    keyed = _by_key(_grouped(store, ["run"], {"cost": ("values", "cost"), "extra": ("values", "extra")}), "run")
    # The port lets a store's by_doc_type order be insertion or id order, so the expected pairs are the store's own
    # read of the group's rows, in its order, each id paired with its own row's value.
    rows = store.by_doc_type(_TYPE, _SCOPE, run="a")
    _expect_equal(
        keyed.get(_key("a")),
        {
            "run": "a",
            "cost": [[row.get("id"), row.get("cost")] for row in rows],
            "extra": [[row.get("id"), row.get("extra")] for row in rows],
        },
        "values: [row_id, value] pairs in by_doc_type's order, each value its own row's",
    )
    _expect_equal(keyed.get(_key("b")), {"run": "b", "cost": [["v1", 0.2]], "extra": [["v1", None]]}, "group b")


#: Values a store that carries numbers through text, or compares with Python's ``==``, would hand back changed: a
#: double needing 17 significant digits, the largest finite double, negative zero, an integral real beside its
#: integer, and ``false`` beside ``0``.
_EXACT_VALUES: tuple[Any, ...] = (
    0.1 + 0.2,
    123456789.12345678,
    1.7976931348623157e308,
    5e-324,
    -0.0,
    1,
    1.0,
    0,
    False,
    "0.1",
    [1, 2.5, {"k": 0.30000000000000004}],
    None,
)


def _values_are_exact(store: DocumentStore, _install: Callable[[Callable[[], None]], None]) -> None:
    _write(store, *(_doc(f"x{index:02d}", run="a", v=value) for index, value in enumerate(_EXACT_VALUES)))
    _write(store, _doc("x99", run="a"))
    groups = _grouped(store, ["run"], {"v": ("values", "v")})
    # The contract is agreement with the store's own reads, not with what was written: a store may normalise a value
    # as it writes it (a jsonb column reads -0.0 back as 0.0, and an integral double back as an integer), and the
    # engine reads both paths through that store.
    expected = [[row.get("id"), row.get("v")] for row in store.by_doc_type(_TYPE, _SCOPE, run="a")]
    _expect(len(groups) == 1, f"grouped by one run answered {len(groups)} groups: {groups!r}")
    # repr tells -0.0 from 0.0, 1.0 from 1 and False from 0, which == does not.
    _expect_equal(
        repr(groups[0]["v"]),
        repr(expected),
        "values over doubles, zeros, integers and false, as by_doc_type reads them",
    )


def _a_grouping_the_port_does_not_define_is_refused(
    store: DocumentStore, _install: Callable[[Callable[[], None]], None]
) -> None:
    _write(store, _doc("q1", run="a", n=1))
    refusals: tuple[tuple[str, Sequence[str], Mapping[str, Any]], ...] = (
        ("an unknown kind", ["run"], {"m": ("mean", "n")}),
        ("count naming a path", ["run"], {"m": ("count", "n")}),
        ("values naming no path", ["run"], {"m": ("values", None)}),
        ("a measure named as a by path", ["run"], {"run": ("count", None)}),
    )
    for what, by, measures in refusals:
        _expect_refused(ValueError, partial(_grouped, store, by, measures), f"grouped with {what}")


# --- empty answers and failures ---------------------------------------------------------------------


def _an_empty_match_is_empty(store: DocumentStore, _install: Callable[[Callable[[], None]], None]) -> None:
    _expect_equal(_narrowed(store), [], "iter_narrowed over an empty store")
    _expect_equal(_grouped(store, ["run"], {"rows": ("count", None)}), [], "grouped over an empty store")
    _expect_equal(_grouped(store, [], {"rows": ("count", None)}), [], "grouped with no by paths over an empty store")
    _write(store, _doc("e1", run="a", n=1))
    _expect_equal(_narrowed(store, run="z"), [], "iter_narrowed matching nothing")
    measures: dict[str, GroupMeasure] = {"rows": ("count", None), "n": ("sum_int", "n"), "low": ("min", "n")}
    _expect_equal(_grouped(store, [], measures, run="z"), [], "grouped with no by paths matching nothing")
    _expect_equal(_grouped(store, ["run"], measures, run="z"), [], "grouped matching nothing")


def _caused_by(raised: BaseException, cause: type[BaseException]) -> bool:
    seen: BaseException | None = raised
    while seen is not None:
        if isinstance(seen, cause):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def _expect_backend_failure(call: Callable[[], object], what: str) -> None:
    try:
        answer = call()
    except Exception as raised:
        if not _caused_by(raised, _BackendDown):
            raise
        # NOSILENT: the backend's failure reaching the caller IS the check passing
        return
    raise StoreConformanceFailure(f"{what} answered {answer!r} while the backend failed")


def _a_backend_failure_raises(store: DocumentStore, install: Callable[[Callable[[], None]], None]) -> None:
    _write(store, *_ROWS)

    def down() -> None:
        raise _BackendDown("the backend is down")

    install(down)
    _expect_backend_failure(lambda: _narrowed(store), "iter_narrowed")
    _expect_backend_failure(lambda: _grouped(store, ["run"], {"rows": ("count", None)}), "grouped")
    _expect_backend_failure(lambda: _grouped(store, [], {"rows": ("count", None)}), "grouped with no by paths")


RESULT_READINGS_CONFORMANCE_CASES: tuple[ResultReadingsConformanceCase, ...] = (
    ResultReadingsConformanceCase(
        "narrowed.equals_by_doc_type",
        "iter_narrowed yields by_doc_type's rows for the same predicates, in its order, with exclude and "
        "exclude_in_items applied",
        _narrowed_equals_by_doc_type,
    ),
    ResultReadingsConformanceCase(
        "narrowed.list_items",
        "exclude_in_items narrows each object item of a list field, and leaves other items, non-list fields and "
        "dead paths alone",
        _list_items_are_narrowed,
    ),
    ResultReadingsConformanceCase(
        "calls.one_per_read",
        "each iter_narrowed and grouped issues one query, at the call, and an empty field_in issues none",
        _each_read_is_one_backend_call,
    ),
    ResultReadingsConformanceCase(
        "scope.isolated",
        "a narrowed or grouped read sees only its own scope and doc_type",
        _reads_stay_in_their_scope,
    ),
    ResultReadingsConformanceCase(
        "predicates.match_by_doc_type",
        "predicates match what they match on by_doc_type: True is not 1, None is absent or null, and the same refusals",
        _predicates_match_by_doc_type,
    ),
    ResultReadingsConformanceCase(
        "grouped.by_json_type",
        "groups are keyed on the stored JSON type: True and 1 apart, absent and null together, objects refused",
        _groups_follow_the_json_type,
    ),
    ResultReadingsConformanceCase(
        "grouped.count_and_count_null",
        "count counts rows and count_null counts rows whose value is absent or null",
        _count_and_count_null,
    ),
    ResultReadingsConformanceCase(
        "grouped.sum_int",
        "sum_int sums JSON integers, skips nulls, gives 0 for none, and refuses booleans and non-integers",
        _sum_int_sums_integers_only,
    ),
    ResultReadingsConformanceCase(
        "grouped.min_max",
        "min and max skip nulls, compare numbers as numbers and strings by code point, and refuse a mix",
        _min_and_max,
    ),
    ResultReadingsConformanceCase(
        "grouped.values_in_stored_order",
        "values gives [row_id, value] pairs in by_doc_type's order within each group",
        _values_come_in_stored_order,
    ),
    ResultReadingsConformanceCase(
        "grouped.values_exact",
        "values gives each value exactly as this store's own by_doc_type returns it: every double to its last bit, "
        "negative zero, 1 apart from 1.0, false apart from 0, wherever its reads keep them apart",
        _values_are_exact,
    ),
    ResultReadingsConformanceCase(
        "grouped.undefined_grouping_refused",
        "an unknown kind, a count naming a path, a measure naming none, or a measure named as a by path is refused",
        _a_grouping_the_port_does_not_define_is_refused,
    ),
    ResultReadingsConformanceCase(
        "empty.answers",
        "an empty match gives no rows and no groups, with or without by paths",
        _an_empty_match_is_empty,
    ),
    ResultReadingsConformanceCase(
        "failure.raises",
        "a backend failure reaches the caller and is never an empty answer",
        _a_backend_failure_raises,
    ),
)
