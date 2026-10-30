"""Which path served a read of readings or groups, recorded, and the two paths' answers alike (W4b P4).

``EvalStorage.iter_result_readings`` and ``result_groups`` read on the store's path when the store offers
``ResultReadings`` (one ``iter_narrowed`` or one ``grouped`` call), and on the engine's otherwise (keys, then pages
by id). Held here:

- **the path** is the store's on the reference store as it is, on the sqlite file store, and on a store wrapped to
  hide the capability the engine's; the read reports it, and the ``eval.read_tier`` line carries it
  (``path=`` and ``store_calls=``);
- **no silent fallback:** a store that offers the capability and fails raises what it raised, at the call or part
  way through its rows, and the engine's reads are never made after it;
- **store calls exact** on both paths, counted against the backend calls the store was asked for;
- **the same answers:** every campaign the reading's fields were recorded over reads the same readings, JSON for
  JSON and in the same order, on both paths, for the scope, for several runs named out of the store's order, and for
  one run; and the same groups, by run and across runs, ``values`` in the same order;
- **what the store ships:** the verbatim fields and the judge's prose no reading carries are asked out on both paths.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``EvalStorage.iter_result_readings``: never taking the store's path; taking it for a store without the capability;
  catching the store's failure and reading again on the engine's path; reporting no store call for a read that
  matched nothing; handing several runs out in the store's order rather than run by run.
- ``EvalStorage.result_groups``: the same, for ``grouped``; reading the engine path's keys run by run, so a
  grouping across runs hands its ``values`` in another order than the store's.
- the engine path's read by id: asking ``get_many`` for the verbatim fields.
- ``phase_log.served_by``: not reaching the open call's line; replacing its store calls rather than adding to them.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.kernel.phase_log import phase_logged
from threetears.evals.kernel.result_reading import ResultReading
from threetears.evals.kernel.storage import READINGS_PAGE, EvalStorage
from threetears.evals.kernel.run_pages import VERBATIM_RESULT_FIELDS
from threetears.evals.schema.models import RubricScore
from threetears.evals.schema.store_port import DocumentStore, GroupMeasure, ResultReadings
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.storage.sqlite import SqliteDocumentStore
from packages.evals.tests.factories import make_eval_result
from packages.evals.tests.hot_path_support import HOT_PATH_SCOPE, hot_path_corpus
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture
from packages.evals.tests.read_path_support import CountingStore, EngineOnlyStore
from packages.evals.tests.fixtures.courierhost import COURIER_SCOPE
from packages.evals.tests.test_readings_match_results import as_json, reading_json
from packages.evals.tests.test_result_reading_fields import (
    COURIER_CAPTURE,
    ENRICHED_CASES,
    ENRICHED_SCOPE,
    REGRESSED_SCOPE,
    add_a_regressed_step,
    enrich,
)

SCOPE = "uni-1"
LOGGER = "threetears.evals.kernel.phase_log"
LINE = re.compile(r"eval\.read_tier lens=(?P<lens>\S+) (?P<fields>.*)")

#: A grouping's measures, one of each kind, over fields every campaign's results carry.
MEASURES: dict[str, GroupMeasure] = {
    "n": ("count", None),
    "unpriced": ("count_null", "cost_usd"),
    "cost": ("values", "cost_usd"),
    "rescores": ("values", "judge_rescores"),
    "first": ("min", "scored_at"),
    "last": ("max", "scored_at"),
    "turns": ("sum_int", "turns_delivered"),
    "total_ms": ("values", "latency.total_ms"),
}


def _scaled(inner: InMemoryDocumentStore) -> str:
    hot_path_corpus(scale=1, store=inner)
    return HOT_PATH_SCOPE


def _scaled_two_aspects(inner: InMemoryDocumentStore) -> str:
    hot_path_corpus(scale=1, two_aspects=True, store=inner)
    return HOT_PATH_SCOPE


def _regressed(inner: InMemoryDocumentStore) -> str:
    corpus = hot_path_corpus(scale=1, scope_id=REGRESSED_SCOPE, store=inner)
    add_a_regressed_step(corpus.host, REGRESSED_SCOPE)
    return REGRESSED_SCOPE


def _enriched(inner: InMemoryDocumentStore) -> str:
    corpus = hot_path_corpus(scale=1, scope_id=ENRICHED_SCOPE, cases_per_run=ENRICHED_CASES, store=inner)
    enrich(corpus.host, ENRICHED_SCOPE)
    return ENRICHED_SCOPE


def _toyhost(inner: InMemoryDocumentStore) -> str:
    off_loop(ops_fixture, store=inner)
    return TOYHOST_SCOPE


def _courierhost(inner: InMemoryDocumentStore) -> str:
    for document in json.loads(COURIER_CAPTURE.read_text()):
        inner.upsert(document)
    return COURIER_SCOPE


#: Every campaign the reading's fields were recorded over (``test_result_reading_fields.py``), written into a store
#: the test holds: its builder, which returns the scope it wrote.
CAMPAIGN_BUILDERS: dict[str, Callable[[InMemoryDocumentStore], str]] = {
    "scaled": _scaled,
    "scaled_two_aspects": _scaled_two_aspects,
    "regressed": _regressed,
    "enriched": _enriched,
    "toyhost": _toyhost,
    "courierhost": _courierhost,
}


def _both(inner: DocumentStore) -> tuple[EvalStorage, EvalStorage]:
    """Storage over ``inner`` as it is, and over it with its narrowed and grouped reads hidden."""
    return EvalStorage(inner), EvalStorage(EngineOnlyStore(inner))


def _stored(count: int, runs: Sequence[str] = ("run-1",)) -> CountingStore:
    """A counting store holding ``count`` results, dealt over ``runs`` in turn, its calls so far forgotten."""
    store = CountingStore()
    storage = EvalStorage(store)
    for index in range(count):
        storage.save_eval_result(make_eval_result(id=f"r-{index:05d}", eval_run_id=runs[index % len(runs)]))
    store.calls.clear()
    return store


def _lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, str]]:
    """Every ``eval.read_tier`` line logged, as its ``key=value`` fields."""
    found = []
    for record in caplog.records:
        if record.name == LOGGER and (match := LINE.match(record.getMessage())):
            found.append(dict(pair.split("=", 1) for pair in match["fields"].split()))
    return found


def _sorted_groups(groups: list[dict[str, Any]]) -> str:
    """Groups as JSON, in the order of their ``by`` values (a grouped read promises none)."""
    return as_json(sorted(groups, key=lambda group: json.dumps([group.get("eval_run_id"), group.get("model")])))


class TestThePathIsTheStores:
    def test_the_reference_store_as_it_is_serves_readings_in_one_narrowed_read(self) -> None:
        store = _stored(5)
        assert isinstance(store, ResultReadings)

        walked = EvalStorage(store).iter_result_readings(["run-1"], SCOPE)

        assert [reading.id for reading in walked] == [f"r-{index:05d}" for index in range(5)]
        assert (walked.path, walked.store_calls, store.calls) == ("store", 1, ["iter_narrowed"])

    def test_a_store_hiding_the_capability_serves_readings_on_the_engine_s_path(self) -> None:
        store = _stored(5)
        hidden = EngineOnlyStore(store)
        assert not isinstance(hidden, ResultReadings)

        walked = EvalStorage(hidden).iter_result_readings(["run-1"], SCOPE)

        assert len(list(walked)) == 5
        assert (walked.path, walked.store_calls, store.calls) == ("engine", 2, ["by_doc_type", "get_many"])

    def test_the_reference_store_serves_groups_in_one_grouped_read(self) -> None:
        store = _stored(6, runs=("run-1", "run-2"))

        grouped = EvalStorage(store).result_groups(["run-1", "run-2"], SCOPE, by=["eval_run_id"], measures=MEASURES)

        assert sorted(group["n"] for group in grouped.groups) == [3, 3]
        assert (grouped.path, grouped.store_calls, store.calls) == ("store", 1, ["grouped"])

    def test_a_store_hiding_the_capability_serves_groups_on_the_engine_s_path(self) -> None:
        store = _stored(6, runs=("run-1", "run-2"))

        grouped = EvalStorage(EngineOnlyStore(store)).result_groups(
            ["run-1", "run-2"], SCOPE, by=["eval_run_id"], measures=MEASURES
        )

        assert sorted(group["n"] for group in grouped.groups) == [3, 3]
        assert (grouped.path, grouped.store_calls, store.calls) == ("engine", 2, ["by_doc_type", "get_many"])

    def test_the_sqlite_file_store_serves_both_on_the_store_s_path_with_the_engine_s_answers(
        self, tmp_path: Path
    ) -> None:
        """The readings path end to end over a real file store: the toy campaign's documents in a sqlite file, read
        through the seams on its own narrowed and grouped reads, answer as its documents read on the engine's path."""
        source = InMemoryDocumentStore()
        scope_id = _toyhost(source)
        sqlite = SqliteDocumentStore(str(tmp_path / "evals.db"))
        for document in source.by_doc_type("eval_result", scope_id):
            sqlite.upsert(document)
        on_store, on_engine = _both(sqlite)
        runs = list(dict.fromkeys(reading.eval_run_id for reading in on_engine.iter_result_readings(None, scope_id)))
        assert len(runs) > 1

        for named in (None, runs[::-1]):
            store_read = on_store.iter_result_readings(named, scope_id)
            engine_read = on_engine.iter_result_readings(named, scope_id)
            store_readings = [reading_json(reading) for reading in store_read]
            assert store_readings == [reading_json(reading) for reading in engine_read]
            assert store_readings
            assert (store_read.path, store_read.store_calls, engine_read.path) == ("store", 1, "engine")

        store_groups = on_store.result_groups(runs[::-1], scope_id, by=["eval_run_id"], measures=MEASURES)
        engine_groups = on_engine.result_groups(runs[::-1], scope_id, by=["eval_run_id"], measures=MEASURES)
        assert _sorted_groups(store_groups.groups) == _sorted_groups(engine_groups.groups)
        assert (store_groups.path, store_groups.store_calls) == ("store", 1)


class _FailingReadings(CountingStore):
    """A store offering the capability whose narrowed read fails: at the call, or after ``rows_first`` rows."""

    def __init__(self, *, rows_first: int | None = None) -> None:
        super().__init__()
        self.rows_first = rows_first

    def iter_narrowed(
        self,
        doc_type: str,
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        exclude_in_items: Mapping[str, Sequence[str]] = {},
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> Iterator[dict[str, Any]]:
        rows = super().iter_narrowed(
            doc_type, scope_id, exclude=exclude, exclude_in_items=exclude_in_items, field_in=field_in, **field_eq
        )
        if self.rows_first is None:
            raise ConnectionError("the backend went away")
        return self._then_fail(rows, self.rows_first)

    @staticmethod
    def _then_fail(rows: Iterator[dict[str, Any]], count: int) -> Iterator[dict[str, Any]]:
        for _ in range(count):
            yield next(rows)
        raise ConnectionError("the backend went away part way")

    def grouped(
        self,
        doc_type: str,
        scope_id: str,
        *,
        by: Sequence[str],
        measures: Mapping[str, GroupMeasure],
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append("grouped")
        raise ConnectionError("the backend went away")


def _failing(**kwargs: Any) -> _FailingReadings:
    store = _FailingReadings(**kwargs)
    storage = EvalStorage(store)
    for index in range(4):
        storage.save_eval_result(make_eval_result(id=f"r-{index:05d}", eval_run_id="run-1"))
    store.calls.clear()
    return store


class TestNoSilentFallback:
    def test_a_narrowed_read_that_fails_at_the_call_raises_and_is_never_read_again(self) -> None:
        store = _failing()

        walked = EvalStorage(store).iter_result_readings(["run-1"], SCOPE)
        with pytest.raises(ConnectionError, match="went away"):
            list(walked)

        assert store.calls == ["iter_narrowed"], "no key read, no read by id: the engine's path is never taken"
        assert walked.path == "store"

    def test_a_narrowed_read_that_fails_part_way_raises_and_is_never_read_again(self) -> None:
        store = _failing(rows_first=2)

        walked = EvalStorage(store).iter_result_readings(None, SCOPE)
        with pytest.raises(ConnectionError, match="part way"):
            list(walked)

        assert store.calls == ["iter_narrowed"]
        assert (walked.path, walked.store_calls) == ("store", 1)

    def test_a_grouped_read_that_fails_raises_and_is_never_read_again(self) -> None:
        store = _failing()

        with pytest.raises(ConnectionError, match="went away"):
            EvalStorage(store).result_groups(["run-1"], SCOPE, by=["eval_run_id"], measures=MEASURES)

        assert store.calls == ["grouped"]


class TestStoreCallsExact:
    @pytest.mark.parametrize("named", [None, ["run-1"], ["run-2", "run-1", "run-3", "run-2"]], ids=str)
    def test_the_store_s_path_makes_one_call_whatever_it_reads(self, named: list[str] | None) -> None:
        store = _stored(READINGS_PAGE + 5, runs=("run-1", "run-2", "run-3"))

        walked = EvalStorage(store).iter_result_readings(named, SCOPE)
        readings = list(walked)

        assert readings
        assert (walked.store_calls, store.calls) == (1, ["iter_narrowed"])

    def test_the_store_s_path_counts_its_call_for_a_read_that_matches_nothing(self) -> None:
        store = _stored(3)

        walked = EvalStorage(store).iter_result_readings(["run-none"], SCOPE)

        assert (list(walked), walked.store_calls, store.calls) == ([], 1, ["iter_narrowed"])

    @pytest.mark.parametrize("hide", [False, True], ids=["store", "engine"])
    def test_no_runs_named_asks_nothing_on_either_path(self, hide: bool) -> None:
        store = _stored(3)
        storage = EvalStorage(EngineOnlyStore(store) if hide else store)

        walked = storage.iter_result_readings([], SCOPE)
        grouped = storage.result_groups([], SCOPE, by=["eval_run_id"], measures=MEASURES)

        assert (list(walked), walked.store_calls, grouped.groups, grouped.store_calls) == ([], 0, [], 0)
        assert store.calls == []

    def test_the_engine_s_path_makes_a_key_read_a_run_and_a_read_by_id_a_page(self) -> None:
        store = _stored(2 * READINGS_PAGE + 5, runs=("run-1", "run-2", "run-3"))

        walked = EvalStorage(EngineOnlyStore(store)).iter_result_readings(["run-3", "run-1", "run-3", "run-2"], SCOPE)
        readings = list(walked)

        assert len(readings) == 2 * READINGS_PAGE + 5
        assert walked.store_calls == 3 + 3 == len(store.calls)
        assert store.calls == ["by_doc_type"] * 3 + ["get_many"] * 3

    def test_the_engine_s_path_groups_with_one_key_read_and_a_read_by_id_a_page(self) -> None:
        store = _stored(2 * READINGS_PAGE + 5, runs=("run-1", "run-2", "run-3"))

        grouped = EvalStorage(EngineOnlyStore(store)).result_groups(
            ["run-3", "run-1", "run-2"], SCOPE, by=["eval_run_id"], measures=MEASURES
        )

        assert sum(group["n"] for group in grouped.groups) == 2 * READINGS_PAGE + 5
        assert grouped.store_calls == 1 + 3 == len(store.calls)
        assert store.calls == ["by_doc_type"] + ["get_many"] * 3


class TestTheLineCarriesThePath:
    @pytest.mark.parametrize(("hide", "path", "calls"), [(False, "store", "1"), (True, "engine", "3")])
    def test_a_walk_inside_a_logged_call_reports_its_path_and_calls(
        self, hide: bool, path: str, calls: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = _stored(READINGS_PAGE + 5)
        storage = EvalStorage(EngineOnlyStore(store) if hide else store)

        with caplog.at_level(logging.INFO, logger=LOGGER), phase_logged("probe", scope_id=SCOPE, campaign_id="c"):
            list(storage.iter_result_readings(["run-1"], SCOPE))

        ((fields,),) = [_lines(caplog)]
        assert (fields["path"], fields["store_calls"]) == (path, calls)
        assert fields["store_ms"] and fields["hydrate_ms"], "the store's time and the hydration's are charged"

    def test_reads_on_both_paths_in_one_call_report_both_and_add_their_calls(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = _stored(3, runs=("run-1", "run-2"))

        with caplog.at_level(logging.INFO, logger=LOGGER), phase_logged("probe", scope_id=SCOPE, campaign_id="c"):
            list(EvalStorage(store).iter_result_readings(None, SCOPE))
            EvalStorage(EngineOnlyStore(store)).result_groups(None, SCOPE, by=["eval_run_id"], measures=MEASURES)

        ((fields,),) = [_lines(caplog)]
        assert (fields["path"], fields["store_calls"]) == ("engine+store", "3")

    def test_a_walk_left_part_way_has_reported_the_calls_it_made(self, caplog: pytest.LogCaptureFixture) -> None:
        store = _stored(2 * READINGS_PAGE + 5)

        with caplog.at_level(logging.INFO, logger=LOGGER), phase_logged("probe", scope_id=SCOPE, campaign_id="c"):
            walked = iter(EvalStorage(EngineOnlyStore(store)).iter_result_readings(["run-1"], SCOPE))
            next(walked)

        ((fields,),) = [_lines(caplog)]
        assert (fields["path"], fields["store_calls"]) == ("engine", "2")


class _ShippingStore(CountingStore):
    """The counting store, keeping each row its reads of results hand out (``shipped``), the key read's aside."""

    def __init__(self) -> None:
        super().__init__()
        self.shipped: list[dict[str, Any]] = []

    def get_many(self, doc_type: str, doc_ids: Sequence[str], scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        rows = super().get_many(doc_type, doc_ids, scope_id, **kwargs)
        self.shipped.extend(rows)
        return rows

    def iter_narrowed(
        self,
        doc_type: str,
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        exclude_in_items: Mapping[str, Sequence[str]] = {},
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> Iterator[dict[str, Any]]:
        for row in super().iter_narrowed(
            doc_type, scope_id, exclude=exclude, exclude_in_items=exclude_in_items, field_in=field_in, **field_eq
        ):
            self.shipped.append(row)
            yield row


class TestWhatTheStoreShips:
    @pytest.fixture(scope="class")
    def store(self) -> _ShippingStore:
        """A store holding one result with a payload, the judge's narration and a score's reasoning."""
        store = _ShippingStore()
        EvalStorage(store).save_eval_result(
            make_eval_result(
                id="r-1",
                kind_payload={"transcript": "y" * 100},
                judge_reasoning="the judge's narration",
                rubric_scores=[
                    RubricScore(
                        dim="conversation.tone",
                        score=4,
                        scale="ordinal",
                        axis="capability",
                        judge_temperature=0.0,
                        reasoning="the score's reasoning",
                    )
                ],
            )
        )
        return store

    @staticmethod
    def _shipped(store: _ShippingStore, *, hide: bool) -> tuple[list[ResultReading], list[dict[str, Any]]]:
        """The readings a read on one path hands out, and the rows the store shipped it."""
        store.shipped = []
        storage = EvalStorage(EngineOnlyStore(store) if hide else store)
        return list(storage.iter_result_readings(None, SCOPE)), store.shipped

    def test_neither_path_asks_for_the_verbatim_fields_or_the_judge_s_narration(self, store: _ShippingStore) -> None:
        narrowed, on_store = self._shipped(store, hide=False)
        by_id, on_engine = self._shipped(store, hide=True)

        assert [reading_json(reading) for reading in narrowed] == [reading_json(reading) for reading in by_id]
        assert on_store and on_engine
        for row in (*on_store, *on_engine):
            assert not {*VERBATIM_RESULT_FIELDS, "judge_reasoning"} & set(row), sorted(row)

    def test_the_store_s_path_leaves_each_score_s_reasoning_out_too(self, store: _ShippingStore) -> None:
        _, on_store = self._shipped(store, hide=False)

        ((row,),) = [on_store]
        assert row["rubric_scores"] and all("reasoning" not in score for score in row["rubric_scores"])


@pytest.fixture(scope="module", params=list(CAMPAIGN_BUILDERS))
def campaign_store(request: pytest.FixtureRequest) -> tuple[InMemoryDocumentStore, str]:
    """One campaign written into a store this module holds, and its scope."""
    inner = InMemoryDocumentStore()
    return inner, CAMPAIGN_BUILDERS[request.param](inner)


class TestBothPathsReadAlike:
    def test_every_campaign_reads_the_same_readings_on_both_paths(
        self, campaign_store: tuple[InMemoryDocumentStore, str]
    ) -> None:
        inner, scope_id = campaign_store
        on_store, on_engine = _both(inner)
        whole = list(on_engine.iter_result_readings(None, scope_id))
        runs = list(dict.fromkeys(reading.eval_run_id for reading in whole))
        assert len(runs) > 1

        for named in (None, [*runs[::-1], runs[0]], runs[:1]):
            store_read = on_store.iter_result_readings(named, scope_id)
            engine_read = on_engine.iter_result_readings(named, scope_id)
            readings: list[ResultReading] = list(store_read)
            assert [reading_json(reading) for reading in readings] == [
                reading_json(reading) for reading in engine_read
            ], named
            assert (store_read.path, engine_read.path) == ("store", "engine")
        assert len(readings) < len(whole)

    @pytest.mark.parametrize("by", [["eval_run_id"], ["model"], []], ids=str)
    def test_every_campaign_reads_the_same_groups_on_both_paths(
        self, by: list[str], campaign_store: tuple[InMemoryDocumentStore, str]
    ) -> None:
        inner, scope_id = campaign_store
        on_store, on_engine = _both(inner)
        runs = list(dict.fromkeys(reading.eval_run_id for reading in on_engine.iter_result_readings(None, scope_id)))

        for named in (None, runs[::-1]):
            store_groups = on_store.result_groups(named, scope_id, by=by, measures=MEASURES)
            engine_groups = on_engine.result_groups(named, scope_id, by=by, measures=MEASURES)
            assert _sorted_groups(store_groups.groups) == _sorted_groups(engine_groups.groups), named
            assert (store_groups.path, engine_groups.path) == ("store", "engine")
