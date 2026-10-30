"""A compact per-result reading reads as the result it was built from, to the byte.

``ResultReading`` carries exactly what the reader families read of a stored result
(``test_result_reading_fields.py`` pins that set), so every reader answers a list of readings as it answers the
results they were built from. Checked three ways, each output compared as JSON byte for byte:

- **the families end to end:** the aggregates and the planning block read over the oracle's campaigns, once as
  stored and once with every result the store hydrates handed out as its reading (``EvalResult.from_dict`` patched,
  as the access recorder patches it). The bundle family reads readings itself (``EvalStorage.iter_result_readings``)
  and re-reads the trials it keeps whole, by id, for their prose; it is read once as it is and once with each reading
  the storage builds handed out as the result it was built of (``ResultReading.of`` patched), so its readers read
  whole results;
- **each rule** of the retyped modules (``kernel/result_condition.py``, ``kernel/scoring.py``, the read-side folds of
  ``kernel/usage_capture.py`` and ``analysis/reporting.py``) over the enriched campaign, whose results carry every
  record list;
- **each whole-result copy** a reader makes (``model_copy`` with a change): a second judge's passes narrowed to some
  of them (``analysis/bundle/judges.py``), read with more than one pass on a result, and contended latency withheld
  (``analysis/contention.py``).

The engine-path seams (``EvalStorage.iter_result_readings`` and ``result_groups``) are checked here too: the readings
they hand out are the readings of what a whole read returns, in its order, with the store calls they report.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any
from unittest import mock

import pytest
from pydantic import BaseModel

from threetears.evals.analysis import judge_drift
from threetears.evals.analysis import service as analysis_service
from threetears.evals.analysis.contention import withhold_contended_latency
from threetears.evals.analysis.reporting import (
    pooled_composite_basis,
    pooled_served_models,
    project_score_records,
    request_settings_readings,
    self_judging_models,
    served_reading,
    unconfirmed_model_identities,
)
from threetears.evals.kernel.result_condition import (
    candidate_failure_cause,
    classify_result,
    counted_goal_verdicts,
    counted_rubric_scores,
    counted_score,
    delivered_a_turn,
    harness_faulted,
    judge_truncated_dims,
    judged_on_every_dim,
    trial_exclusion,
)
from threetears.evals.kernel.result_reading import ResultReading, ScoreView, SharedValues
from threetears.evals.kernel.scoring import (
    capability_scores,
    composite_basis,
    compute_async_delivery_summary,
    compute_composite_summary,
    compute_cost_summary,
    compute_dimension_summary,
    compute_latency_summary,
    compute_pass_hat_k,
    compute_per_case_composites,
    has_pass_criterion,
    pool_pass_hat_k,
    reconstruct_completeness,
    result_composite,
)
from threetears.evals.kernel.storage import READINGS_PAGE, EvalStorage
from threetears.evals.kernel.usage_capture import (
    count_substituted_deliveries,
    production_replicating_cost,
    production_rows,
    program_cost,
    spend_observed,
)
from threetears.evals.run.reads import list_runs
from threetears.evals.schema.models import (
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    EvalResult,
    EvalRun,
    RubricScore,
    SecondJudge,
    SecondJudging,
)
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_result, memory_storage
from packages.evals.tests.read_path_support import CountingStore, EngineOnlyStore
from packages.evals.tests.test_result_reading_fields import (
    CAMPAIGNS,
    PINNED,
    REPORT_NOW,
    Campaign,
    campaign,
    read_aggregates,
    read_bundle,
    read_planning,
    regressed_steps,
)

#: The campaign whose results carry every record list (re-judges, repeats, second judgings, deliveries, preconditions,
#: latency, simulator usage), so each rule's walk over them runs.
ENRICHED = "enriched"

#: The thresholds a score's bar is read at: below, at and above every level of the 1–5 scale.
THRESHOLDS = range(0, 7)


def canonical(value: Any) -> Any:
    """``value`` as JSON reads it: a model dumped, a dataclass by its fields, a mapping's keys as text, an enum by its
    value, a set sorted."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: canonical(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(canonical(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def as_json(value: Any) -> str:
    """``value`` as JSON text (:func:`canonical`), keys sorted."""
    return json.dumps(canonical(value), sort_keys=True, ensure_ascii=False)


def score_json(score: ScoreView | None) -> Any:
    """A score as the readers read it: each view attribute, and whether it clears each threshold."""
    if score is None:
        return None
    return {
        "dim": score.dim,
        "scale": score.scale,
        "axis": score.axis,
        "score": score.score,
        "served_model": score.served_model,
        "judge_temperature": score.judge_temperature,
        "judge_sent_settings": canonical(score.judge_sent_settings),
        "output_fingerprint": score.output_fingerprint,
        "criterion_fingerprint": score.criterion_fingerprint,
        "label_key": canonical(score.label_key),
        "normalized": score.normalized,
        "clears": [score.clears(threshold) for threshold in THRESHOLDS],
    }


@functools.cache
def stored(name: str) -> tuple[Campaign, tuple[EvalRun, ...], tuple[EvalResult, ...]]:
    """The campaign ``name``, its runs and every result of its scope, as stored."""
    read = campaign(name)
    runs = tuple(list_runs(read.host, read.scope_id))
    return read, runs, tuple(read.host.storage.query_eval_results(read.scope_id))


def readings_of(results: Sequence[EvalResult]) -> list[ResultReading]:
    """Each result's reading, built through one :class:`SharedValues`, as a read builds them."""
    shared = SharedValues()
    return [ResultReading.of(result, shared) for result in results]


@pytest.fixture(scope="module")
def enriched() -> tuple[Campaign, tuple[EvalRun, ...], tuple[EvalResult, ...], list[ResultReading]]:
    """The enriched campaign, its runs, its results and their readings."""
    read, runs, results = stored(ENRICHED)
    return read, runs, results, readings_of(results)


class TestEveryFamilyReadsReadingsAsResults:
    """The aggregates, the planning block and the bundle, over every campaign the recorder reads them over, answer the
    same whether their readers read results or their readings."""

    @staticmethod
    def _as_readings(family: Callable[..., list[str]], read: Campaign, *args: Any) -> tuple[list[str], int]:
        """``family`` read with every result the store hydrates handed out as its reading, and how many it handed
        out."""
        hydrate = EvalResult.from_dict
        shared = SharedValues()
        handed: list[str] = []

        def reading_of(data: dict[str, Any]) -> Any:
            reading = ResultReading.of(hydrate(data), shared)
            handed.append(reading.id)
            return reading

        with mock.patch.object(EvalResult, "from_dict", reading_of):
            return family(read, *args), len(handed)

    @staticmethod
    def _as_results(family: Callable[..., list[str]], read: Campaign) -> tuple[list[str], int]:
        """``family`` read with every reading the storage builds handed out as the result it was built of, and how
        many it handed out."""
        handed: list[str] = []

        def whole(result: EvalResult, shared: SharedValues | None = None) -> EvalResult:
            handed.append(result.id)
            return result

        with mock.patch.object(ResultReading, "of", staticmethod(whole)):
            return family(read), len(handed)

    # The bundle reads readings as it is, so the read it is compared with is the one whose readers read results.
    @pytest.mark.parametrize("name", CAMPAIGNS)
    def test_the_bundle(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(analysis_service, "utc_now_iso", lambda: REPORT_NOW)
        read = campaign(name)

        outputs, handed = self._as_results(read_bundle, read)

        assert handed > 0, "the bundle read no reading, so it read no result in a reading's place"
        assert read_bundle(read) == outputs

    @pytest.mark.parametrize("name", (*CAMPAIGNS, "regressed"))
    def test_the_aggregates(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(analysis_service, "utc_now_iso", lambda: REPORT_NOW)
        read = campaign(name)
        steps = regressed_steps(read)
        family = functools.partial(read_aggregates, lenses=name != "regressed")

        outputs, handed = self._as_readings(family, read, steps)

        assert handed > 0, "the family read no result, so it read no reading"
        assert outputs == family(read, steps)

    # Not over the toy host's campaign: its planning block reads no result there (``test_result_reading_fields.py``
    # pins nothing it reads over it), so a reading would check nothing.
    @pytest.mark.parametrize("name", ("scaled", "courierhost", ENRICHED))
    def test_the_planning_block(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(analysis_service, "utc_now_iso", lambda: REPORT_NOW)
        read = campaign(name)

        outputs, handed = self._as_readings(read_planning, read)

        assert handed > 0, "the planning block read no result, so it read no reading"
        assert outputs == read_planning(read)


class TestEachRuleReadsReadingsAsResults:
    """Every rule of the retyped modules answers the enriched campaign's readings as it answers its results."""

    PER_RESULT: dict[str, Callable[[Any], Any]] = {
        "candidate_failure_cause": candidate_failure_cause,
        "classify_result": classify_result,
        "delivered_a_turn": delivered_a_turn,
        "harness_faulted": harness_faulted,
        "counted_goal_verdicts": lambda r: [
            (check.expression, check.passed, check.detail, passed) for check, passed in counted_goal_verdicts(r) or []
        ],
        "counted_rubric_scores": lambda r: [
            (score_json(score), counted) for score, counted in counted_rubric_scores(r) or []
        ],
        "counted_score": lambda r: [counted_score(r, score) for score in r.judge_scores()],
        "trial_exclusion": trial_exclusion,
        "judge_truncated_dims": judge_truncated_dims,
        "judged_on_every_dim": judged_on_every_dim,
        "capability_scores": lambda r: [score_json(score) for score in capability_scores(r)],
        "judge_scores": lambda r: [score_json(score) for score in r.judge_scores()],
        "judge_score": lambda r: [score_json(r.judge_score(score.dim)) for score in r.judge_scores()],
        "result_composite": result_composite,
        "composite_basis": composite_basis,
        "has_pass_criterion": has_pass_criterion,
        "spend_observed": lambda r: spend_observed(r.usage, r.cost_roles),
        "production_replicating_cost": lambda r: production_replicating_cost(
            r.usage, substituted_deliveries=count_substituted_deliveries(r), candidate_kind=r.candidate_kind
        ),
        "production_rows": lambda r: [
            (row.role, row.model, row.cost_usd) for row in production_rows(r.usage, candidate_kind=r.candidate_kind)
        ],
        "program_cost": lambda r: program_cost(r.usage),
        "count_substituted_deliveries": count_substituted_deliveries,
        "served_reading": served_reading,
    }

    @pytest.mark.parametrize("rule", sorted(PER_RESULT))
    def test_each_per_result_rule(self, rule: str, enriched: Any) -> None:
        _, _, results, readings = enriched
        read = self.PER_RESULT[rule]

        assert as_json([read(reading) for reading in readings]) == as_json([read(result) for result in results])

    def test_each_fold_over_the_campaign(self, enriched: Any) -> None:
        read, runs, results, readings = enriched
        cells = {run.id: run.candidate_model for run in runs}

        def folds(over: Sequence[Any]) -> dict[str, str]:
            return {
                "compute_pass_hat_k": as_json(compute_pass_hat_k(over)),
                "pool_pass_hat_k": as_json(pool_pass_hat_k(over, cell_of_run=cells, k=2)),
                "compute_latency_summary": as_json(compute_latency_summary(over)),
                "compute_async_delivery_summary": as_json(compute_async_delivery_summary(over)),
                "compute_cost_summary": as_json(compute_cost_summary(over)),
                "compute_dimension_summary": as_json(compute_dimension_summary(over)),
                "compute_composite_summary": as_json(compute_composite_summary(over)),
                "compute_per_case_composites": as_json(compute_per_case_composites(over)),
                "reconstruct_completeness": as_json([reconstruct_completeness(run, over) for run in runs]),
                "pooled_served_models": as_json(pooled_served_models(over)),
                "pooled_composite_basis": as_json(pooled_composite_basis(over)),
                "request_settings_readings": as_json(request_settings_readings(runs, over)),
                "unconfirmed_model_identities": as_json(unconfirmed_model_identities(runs, over)),
                "self_judging_models": as_json([self_judging_models(run, over) for run in runs]),
                "project_score_records": as_json(
                    project_score_records(list(runs), over, archived_run_ids=set(), profile=read.host.profile)
                ),
            }

        assert folds(readings) == folds(results)


class TestEachWholeResultCopyReadsAlike:
    """A copy of a reading with a change reads as the same copy of its result."""

    def test_a_second_judges_passes_narrowed_to_some_of_them(self, enriched: Any) -> None:
        """Two passes on every judged result, each narrowed to one and to both, as the judge-change link narrows them
        (``analysis/bundle/judges.py``), then read for drift."""
        _, _, stored_results, _ = enriched
        results = [_with_a_second_pass(result) for result in stored_results if result.judge_seconds]
        readings = readings_of(results)
        assert results and all(len(result.judge_seconds) == 2 for result in results)

        for pass_ids in ({"second-pass"}, {"second-pass-2"}, {"second-pass", "second-pass-2"}):

            def narrowed(over: Sequence[Any], keep: set[str] = pass_ids) -> list[Any]:
                return [
                    r.model_copy(update={"judge_seconds": [j for j in r.judge_seconds if j.pass_id in keep]})
                    for r in over
                ]

            for pass_id in (*sorted(pass_ids), None):
                assert as_json(judge_drift(narrowed(readings), pass_id=pass_id)) == as_json(
                    judge_drift(narrowed(results), pass_id=pass_id)
                )

    def test_contended_latency_withheld(self, enriched: Any) -> None:
        """Every fourth enriched result was read under concurrency: its latency, phase timings, deliveries' elapsed time
        and elapsed host measures are withheld, each through a copy (``analysis/contention.py``)."""
        read, _, results, readings = enriched
        measures = read.host.profile.measures

        from_results = withhold_contended_latency(list(results), measures)
        from_readings = withhold_contended_latency(readings, measures)

        assert [reading_json(r) for r in from_readings] == [reading_json(ResultReading.of(r)) for r in from_results]
        assert sum(1 for r in from_readings if r.latency is None) > 0, "no result's latency was withheld"

    def test_a_copy_names_only_what_the_reading_carries(self) -> None:
        reading = ResultReading.of(make_eval_result(id="r-1"))

        assert reading.model_copy(update={"cost_usd": 1.5}).cost_usd == 1.5
        assert reading.model_copy().cost_usd == reading.cost_usd
        with pytest.raises(TypeError):
            reading.model_copy(update={"judge_reasoning": "prose the reading does not carry"})


def _with_a_second_pass(result: EvalResult) -> EvalResult:
    """``result`` with a second pass of its second judge, under another judge, its scores one higher."""
    (first,) = result.judge_seconds
    again = SecondJudging(
        pass_id="second-pass-2",
        judge=SecondJudge(model="third-judge-model", temperature=0.0),
        sample_fraction=1.0,
        sample_seed=1,
        scores=[
            entry.model_copy(
                update={
                    "second": entry.second
                    and entry.second.model_copy(
                        update={"score": min(5, entry.second.score + 1), "served_model": "third-judge-model"}
                    )
                }
            )
            for entry in first.scores
        ],
        judge_config_ids=dict(first.judge_config_ids),
    )
    return result.model_copy(update={"judge_seconds": [first, again]})


def reading_json(reading: ResultReading) -> str:
    """A reading as JSON, its record lists read alike whether a copy holds them as lists or tuples."""
    return as_json(reading)


class TestTheSeamsHandOutTheReadingsOfAWholeRead:
    """``EvalStorage.iter_result_readings`` and ``result_groups`` on the engine's path: over a store whose narrowed and
    grouped reads are hidden (``test_read_path_recorded.py`` holds the store's path to the same answers)."""

    def test_the_readings_of_named_runs_in_the_store_order_run_by_run(self, enriched: Any) -> None:
        read, runs, results, _ = enriched
        named = [run.id for run in runs[:3]]
        whole = [
            result for run_id in named for result in read.host.storage.query_eval_results_by_run(run_id, read.scope_id)
        ]
        storage = _engine_storage(results)

        walked = storage.iter_result_readings([*named, named[0]], read.scope_id)
        readings = list(walked)

        assert [reading_json(r) for r in readings] == [reading_json(ResultReading.of(r)) for r in whole]
        assert walked.path == "engine"
        pages = -(-len(whole) // READINGS_PAGE)
        assert walked.store_calls == len(named) + pages, "one key read per run named once, one read by id a page"

    def test_the_readings_of_a_scope_in_the_store_order(self, enriched: Any) -> None:
        read, _, results, _ = enriched

        walked = _engine_storage(results).iter_result_readings(None, read.scope_id)

        assert [reading_json(r) for r in walked] == [reading_json(ResultReading.of(r)) for r in results]
        assert walked.store_calls == 1 + -(-len(results) // READINGS_PAGE)

    def test_a_read_past_a_page_is_read_a_page_at_a_time(self) -> None:
        storage, store = _counted_storage()
        for index in range(READINGS_PAGE + 5):
            storage.save_eval_result(make_eval_result(id=f"r-{index:05d}", eval_run_id="run-1"))
        store.calls.clear()

        walked = storage.iter_result_readings(["run-1"], "uni-1")
        readings = iter(walked)
        first = next(readings)

        assert first.id == "r-00000"
        assert store.calls == ["by_doc_type", "get_many"], "the second page is read when the walk reaches it"
        assert walked.store_calls == 2
        rest = list(readings)
        assert [first.id, *(r.id for r in rest)] == [f"r-{index:05d}" for index in range(READINGS_PAGE + 5)]
        assert store.calls == ["by_doc_type", "get_many", "get_many"]
        assert walked.store_calls == 3

    def test_a_read_is_walked_once(self) -> None:
        storage, _ = memory_storage()
        storage.save_eval_result(make_eval_result(id="r-1"))
        walked = storage.iter_result_readings(None, "uni-1")
        assert [r.id for r in walked] == ["r-1"]

        with pytest.raises(RuntimeError, match="walked once"):
            list(walked)

    def test_no_runs_named_asks_nothing(self) -> None:
        storage, store = _counted_storage()

        walked = storage.iter_result_readings([], "uni-1")

        assert (list(walked), walked.store_calls, store.calls) == ([], 0, [])

    def test_the_groups_are_the_stores_own_over_the_same_rows(self, enriched: Any) -> None:
        read, runs, _, _ = enriched
        named = [run.id for run in runs[:4]]
        by = ["eval_run_id"]
        measures: dict[str, Any] = {
            "n": ("count", None),
            "unpriced": ("count_null", "cost_usd"),
            "cost": ("values", "cost_usd"),
            "rescores": ("values", "judge_rescores"),
            "first": ("min", "scored_at"),
            "last": ("max", "scored_at"),
            "turns": ("sum_int", "turns_delivered"),
            "total_ms": ("values", "latency.total_ms"),
        }
        store = InMemoryDocumentStore()
        storage = EvalStorage(EngineOnlyStore(store))
        for run_id in named:
            for result in read.host.storage.query_eval_results_by_run(run_id, read.scope_id):
                storage.save_eval_result(result)

        grouped = storage.result_groups(named, read.scope_id, by=by, measures=measures)

        expected = store.grouped(
            "eval_result", read.scope_id, by=by, measures=measures, field_in={"eval_run_id": named}
        )
        assert as_json(sorted(grouped.groups, key=lambda g: g["eval_run_id"])) == as_json(
            sorted(expected, key=lambda g: g["eval_run_id"])
        )
        assert (grouped.path, grouped.store_calls) == ("engine", 2), "one key read of every run named, one page by id"

    def test_shared_values_hold_each_equal_value_once(self, enriched: Any) -> None:
        _, _, results, readings = enriched

        alone = [ResultReading.of(result) for result in results]

        assert [reading_json(r) for r in readings] == [reading_json(r) for r in alone]
        settings = {id(s.judge_sent_settings) for r in readings for s in r.judge_scores() if s.judge_sent_settings}
        distinct = {as_json(s.judge_sent_settings) for r in readings for s in r.judge_scores() if s.judge_sent_settings}
        assert len(settings) == len(distinct), "each distinct settings model is held once across the read"


class TestAScoreReadsAsItsScore:
    """A score's derived reads (its label key, its 0–1 value, whether it clears a bar) are the stored score's."""

    def test_on_every_score_of_every_campaign(self) -> None:
        """Over every campaign together (the courier's is graded by code alone and carries none)."""
        compared = 0
        for name in (*CAMPAIGNS, "regressed"):
            _, _, results = stored(name)
            shared = SharedValues()
            pairs = [
                (score_json(score), score_json(reading))
                for result in results
                for score, reading in zip(
                    result.judge_scores(), ResultReading.of(result, shared).judge_scores(), strict=True
                )
            ]
            compared += len(pairs)
            assert all(read == stored_score for stored_score, read in pairs), name
        assert compared > 1000, "too few scores to compare"


def _counted_storage() -> tuple[EvalStorage, CountingStore]:
    """Storage reading on the engine's path over a store that names each call it is asked for."""
    store = CountingStore()
    return EvalStorage(EngineOnlyStore(store)), store


def _engine_storage(results: Sequence[EvalResult]) -> EvalStorage:
    """Storage reading on the engine's path over a store holding ``results``, written in the order given."""
    storage = EvalStorage(EngineOnlyStore(InMemoryDocumentStore()))
    for result in results:
        storage.save_eval_result(result)
    return storage


#: Pinned paths a reading leaves out by design: a score's reasoning, read only by the bundle's trials section, which
#: re-reads the rows it lists whole.
LEFT_OUT = frozenset({"outcome_score.reasoning", "rubric_scores[].reasoning", "transcript_score.reasoning"})


def _reached(holders: list[Any], segments: list[str]) -> bool:
    """Whether ``segments`` (a pinned path's steps) reach their end from some holder: its last step is read off a
    holder that exists (whatever the value read, ``None`` included). A step a holder lacks raises."""
    live = [holder for holder in holders if holder is not None]
    if not live:
        return False
    step, rest = segments[0], segments[1:]
    if step == "==":
        return True
    if step.endswith("()"):
        return all(callable(getattr(holder, step[:-2])) for holder in live)
    if step.endswith("[]"):
        held = [getattr(holder, step[:-2]) for holder in live]
        return True if not rest else _reached([item for value in held for item in (value or ())], rest)
    values = [getattr(holder, step) for holder in live]
    return True if not rest else _reached(values, rest)


class TestEveryPinnedPathResolvesOnAReading:
    """Each attribute path any family reads of a result (``test_result_reading_fields.py``'s pins) resolves on the
    readings of results that carry it: none fails, and each reaches its end on some reading."""

    def test_over_every_campaign(self) -> None:
        readings = [ResultReading.of(result) for name in (*CAMPAIGNS, "regressed") for result in stored(name)[2]]
        paths = sorted(frozenset().union(*PINNED.values()) - LEFT_OUT)

        unreached = [path for path in paths if not _reached(readings, path.split("."))]

        assert not unreached, f"pinned paths no reading reaches: {unreached}"

    def test_a_left_out_path_is_one_the_reading_does_not_carry(self) -> None:
        def axis(dim: str) -> RubricScore:
            return RubricScore(dim=dim, score=4, scale="ordinal", axis="capability", reasoning="the judge's words")

        reading = ResultReading.of(
            make_eval_result(
                id="r-1",
                rubric_scores=[axis("conversation.tone")],
                transcript_score=axis(TRANSCRIPT_DIM_ID),
                outcome_score=axis(OUTCOME_DIM_ID),
            )
        )

        for path in sorted(LEFT_OUT):
            with pytest.raises(AttributeError):
                _reached([reading], path.split("."))
