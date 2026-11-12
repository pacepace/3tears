"""A checked bounded-relation producer (reports #39, coordinator 2026-11-12): a function that returns a copy's
relations already cut to a key (``limited_snapshot``: each to the reader's states) lets a read over a relation the copy
accessor names (``snapshot.relation(table)``) be judged keyed by that key with no condition of its own -- only while the
gate sees the producer bind the key in every relation it returns, every other source of the relations is checked or
declared, and a reader of every state is the one place the producer is skipped."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from threetears.enforcement.no_whole_reads import BoundBy, NoWholeReadsConfig, no_whole_reads_findings

#: a finding naming one read: ``<path>:<line> in <function>: <kind> <what> -- ...``
_HIT: re.Pattern[str] = re.compile(r"^\S+:\d+ in (\S+): (\S+ \S+) -- ")


def _symbols(config: NoWholeReadsConfig) -> set[tuple[str, str]]:
    """``(function, symbol)`` of every read the gate names."""
    return {
        (match.group(1), match.group(2))
        for match in (_HIT.match(finding) for finding in no_whole_reads_findings(config))
        if match is not None
    }


_TABLES = """
    STATE_COLUMN = "join_state_code"
    BENCHMARKS = "benchmarks"
"""

#: a copy's evaluator as reports writes it: the snapshot, its producer limiting it to a reader's states (a national
#: table left whole), the every-state base, a derived snapshot, a reader of either, and reads with no state condition
_EVALUATE = """
    from dataclasses import dataclass, field, replace

    from tables import BENCHMARKS, STATE_COLUMN

    SHARED_VIEWS = frozenset({BENCHMARKS, "settings"})
    FEED = "feed_county"


    def quote_ident(name):
        return '"' + name.replace('"', '""') + '"'


    @dataclass(frozen=True)
    class Snapshot:
        backend: object
        columns: dict
        relations: dict = field(default_factory=dict)
        states: frozenset | None = None

        def relation(self, table):
            return self.relations.get(table, quote_ident(table))


    def _feed_sql(relation, states, kept):
        own = ", ".join(sorted(states))
        return f"(SELECT * REPLACE ({own}) FROM {relation} WHERE {kept})"


    def limited_snapshot(views, states, *, column=STATE_COLUMN):
        listed = ", ".join(sorted(states)) or "NULL"
        kept = f"{quote_ident(column)} IN ({listed})"
        relations = {
            table: views.relation(table)
            if table in SHARED_VIEWS
            else _feed_sql(views.relation(table), states, kept)
            if table == FEED
            else f"(SELECT * FROM {views.relation(table)} WHERE {kept})"
            for table in views.columns
        }
        return replace(views, relations=relations, states=frozenset(states))


    def base_views(backend, columns):
        return Snapshot(backend=backend, columns=columns, relations={t: f"views.{t}" for t in columns})


    def over_read(backend, views):
        return Snapshot(backend=backend, columns=views.columns, relations=views.relations, states=views.states)


    def with_namesakes(views):
        relations = dict(views.relations)
        relations["candidates"] = f"(SELECT v.* FROM {views.relation('candidates')} AS v)"
        return replace(views, relations=relations)


    def with_house(views, race):
        relations = {**views.relations, **{t: _house(views, t, race) for t in ("geos",)}}
        return replace(views, relations=relations)


    def _house(views, table, race):
        return f"(SELECT * FROM {views.relation(table)} WHERE office_key = {race})"


    class Reading:
        def read_for(self, views, states):
            if states is None:
                return views
            return limited_snapshot(views, states)


    def distinct_rows(snapshot, table):
        return snapshot.backend.execute_query(
            f"SELECT DISTINCT v FROM (SELECT * FROM {snapshot.relation(table)}) AS t WHERE v IS NOT NULL ORDER BY 1"
        )


    def panel(snapshot, table):
        return snapshot.backend.execute_query(f"SELECT * FROM {snapshot.relation(table)} WHERE office = 'x'")


    def benchmarks(snapshot):
        return snapshot.backend.execute_query(f"SELECT * FROM {snapshot.relation(BENCHMARKS)}")
"""

_REASON = "the copy's every-state views: a reader of some states reads them only through limited_snapshot"

_BOUND = BoundBy(
    producer="evaluate.limited_snapshot",
    key=("join_state_code",),
    reason="every relation a reader of some states reads is cut to their states by limited_snapshot",
    unlimited="SHARED_VIEWS",
    unlimited_reason="the national tables hold every row under the national code: no state of their own to cut by",
    sources={"evaluate.base_views": _REASON},
    every_state=("Reading.read_for",),
)


def _config(tmp_path: Path, evaluate: str = _EVALUATE, bound: BoundBy | None = _BOUND) -> NoWholeReadsConfig:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "tables.py").write_text(textwrap.dedent(_TABLES), encoding="utf-8")
    (src / "evaluate.py").write_text(textwrap.dedent(evaluate), encoding="utf-8")
    (tmp_path / "_no_whole_reads_exemptions.txt").write_text("", encoding="utf-8")
    return NoWholeReadsConfig(
        repo_root=tmp_path,
        src_roots=(src,),
        ledger_dir=tmp_path,
        dependency_schema=False,
        table_lists=(),
        local_relations={"*": ("join_state_code",)},
        bounded_producers={} if bound is None else {"Snapshot.relation": bound},
    )


def _bounded(tmp_path: Path, evaluate: str = _EVALUATE, bound: BoundBy | None = _BOUND) -> list[str]:
    return [finding for finding in no_whole_reads_findings(_config(tmp_path, evaluate, bound)) if "bounded" in finding]


def _edit(old: str, new: str) -> str:
    source = textwrap.dedent(_EVALUATE)
    assert source.count(old) == 1, old
    return source.replace(old, new)


def test_a_read_over_a_bounded_relation_is_keyed_by_the_producers_key(tmp_path: Path) -> None:
    """Flips if a read through the copy accessor with no state condition is not judged keyed by the bound, or a read
    of a table the producer leaves whole (a national one, by its constant) is no longer judged as it is."""
    found = _symbols(_config(tmp_path))
    functions = {function for function, _ in found}
    assert "distinct_rows" not in functions, found
    assert "panel" not in functions, found
    assert ("benchmarks", "whole-select copy.benchmarks") in found, found
    assert not _bounded(tmp_path), _bounded(tmp_path)


def test_an_undeclared_producer_leaves_the_reads_whole(tmp_path: Path) -> None:
    """Flips if a read through the copy accessor passes with no bounded producer declared."""
    found = _symbols(_config(tmp_path, bound=None))
    assert ("distinct_rows", "whole-select copy.{}") in found, found
    assert ("panel", "non-key-where copy.{}") in found, found


@pytest.mark.parametrize(
    ("old", "new", "named"),
    [
        # a relation returned with no condition on the key
        (
            'else f"(SELECT * FROM {views.relation(table)} WHERE {kept})"',
            'else f"(SELECT * FROM {views.relation(table)})"',
            "limited_snapshot",
        ),
        # the condition's column defaults to another column
        (
            "def limited_snapshot(views, states, *, column=STATE_COLUMN):",
            'def limited_snapshot(views, states, *, column="office"):',
            "limited_snapshot",
        ),
        # the condition reads nothing the walk can tell
        (
            'kept = f"{quote_ident(column)} IN ({listed})"',
            'kept = f"{quote_ident(column)} IS NOT NULL OR TRUE"',
            "limited_snapshot",
        ),
        # the helper handed the condition drops it
        (
            'return f"(SELECT * REPLACE ({own}) FROM {relation} WHERE {kept})"',
            'return f"(SELECT * REPLACE ({own}) FROM {relation})"',
            "limited_snapshot",
        ),
        # a table left whole other than by the declared constant
        ("if table in SHARED_VIEWS", "if table in {'geos'}", "limited_snapshot"),
        # a caller overriding the column
        (
            "return limited_snapshot(views, states)",
            'return limited_snapshot(views, states, column="office")',
            "Reading.read_for",
        ),
        # a second source of the relations the gate does not check
        (
            "def over_read(backend, views):",
            "def widened(views):\n    return replace(views, relations={t: f'other.{t}' for t in views.columns})\n\n\ndef over_read(backend, views):",
            "widened",
        ),
        # a derived snapshot adding a relation read from elsewhere
        (
            """relations["candidates"] = f"(SELECT v.* FROM {views.relation('candidates')} AS v)\"""",
            """relations["candidates"] = "other.candidates\"""",
            "with_namesakes",
        ),
        # the every-state reader's branch gone: the producer reached for every reader, or skipped for none
        ("        if states is None:\n            return views\n", "", "Reading.read_for"),
        # the producer reached for a reader of every state
        ("            return views\n", "            return limited_snapshot(views, frozenset())\n", "Reading.read_for"),
    ],
)
def test_the_gate_names_the_site_that_breaks_the_bound(tmp_path: Path, old: str, new: str, named: str) -> None:
    """Flips if a producer that stops binding the key, a source of the relations from elsewhere, or a reader of every
    state handed the producer passes, or is not named where it is."""
    source = _edit(old, new)
    assert source != textwrap.dedent(_EVALUATE)
    found = _bounded(tmp_path, source)
    assert any(f" in {named}: bounded-producer Snapshot.relation -- " in finding for finding in found), found


def test_the_declaration_itself_is_checked(tmp_path: Path) -> None:
    """Flips if a bound naming no column (``()`` is a whole read, never a bound), a producer the walk cannot find, a
    stale source, a source with no reason, an unlimited set the walk cannot resolve, or an every-state reader the walk
    cannot find passes."""
    cases = {
        "no column": BoundBy(producer=_BOUND.producer, key=(), reason=_BOUND.reason),
        "no producer": BoundBy(producer="evaluate.limited", key=_BOUND.key, reason=_BOUND.reason),
        "stale source": BoundBy(
            producer=_BOUND.producer,
            key=_BOUND.key,
            reason=_BOUND.reason,
            unlimited="SHARED_VIEWS",
            unlimited_reason=_BOUND.unlimited_reason,
            sources={**_BOUND.sources, "evaluate.panel": _REASON},
            every_state=_BOUND.every_state,
        ),
        "short reason": BoundBy(
            producer=_BOUND.producer,
            key=_BOUND.key,
            reason=_BOUND.reason,
            unlimited="SHARED_VIEWS",
            unlimited_reason=_BOUND.unlimited_reason,
            sources={"evaluate.base_views": "because"},
            every_state=_BOUND.every_state,
        ),
        "unresolved unlimited": BoundBy(
            producer=_BOUND.producer,
            key=_BOUND.key,
            reason=_BOUND.reason,
            unlimited="NATIONAL",
            unlimited_reason=_BOUND.unlimited_reason,
            sources=_BOUND.sources,
            every_state=_BOUND.every_state,
        ),
        "no every-state reader": BoundBy(
            producer=_BOUND.producer,
            key=_BOUND.key,
            reason=_BOUND.reason,
            unlimited="SHARED_VIEWS",
            unlimited_reason=_BOUND.unlimited_reason,
            sources=_BOUND.sources,
            every_state=("Reading.read_every",),
        ),
    }
    for case, bound in cases.items():
        assert _bounded(tmp_path, bound=bound), case


def test_an_accessor_other_than_the_copys_is_refused(tmp_path: Path) -> None:
    """Flips if a bound is declared on an accessor the walk does not read as a copy's relation."""
    config = _config(tmp_path)
    config = NoWholeReadsConfig(**{**config.__dict__, "bounded_producers": {"Snapshot.view": _BOUND}})
    assert any("bounded" in finding for finding in no_whole_reads_findings(config))
