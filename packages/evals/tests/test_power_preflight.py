"""The power pre-flight beside a launch's price (#594), checked against data with a known answer.

A launch estimate states, per reading, the smallest difference its paired comparison would find with 80%
power: "with N cases and k repeats this campaign can detect Δ ≥ x". That is a number, so it is checked the
way every number here is, by drawing:

- **the power arithmetic** agrees with an independent reference (``simulation_support.paired_t_power``, a
  4,000-panel Simpson integral that shares no code with the engine's quadrature);
- **the stated Δ is found about four times in five** when the variance components are known, through the
  family rule a bundle applies (``family_verdicts``: per-case means, the paired test, Holm over the family —
  pinned to the bundle by ``test_simulated_multiple_comparisons``), and at least that often where it was
  planned without credit for a pairing the arms in fact share;
- **the variance components are recovered** from earlier runs with known components;
- **the estimate reads the store**: earlier runs of the template give the block its runs and its basis, and a
  template with none, or with no repeated cases, says it cannot estimate rather than borrowing a variance.
"""

from __future__ import annotations

import asyncio
import math
import random
import tracemalloc
from collections.abc import Callable
from dataclasses import replace

from typing import Any

import pytest

from threetears.evals.analysis.stats import (
    DETECTABLE_POWER,
    SIGNIFICANCE_ALPHA,
    paired_case_variance,
    paired_detectable_difference,
    paired_t_power,
    variance_components,
)
from threetears.evals.analysis import reads
from threetears.evals.analysis.bundle import comparisons
from threetears.evals.analysis.bundle.comparisons import planning_readings
from threetears.evals.kernel import EvalStorage, ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.schema.models import EvalResult, RubricScore
from threetears.evals.ops import DetectableEffects, LaunchArguments, launch_estimate
from threetears.evals.ops.lenses import detectable_effects, estimate_text
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.ops_support import TOYHOST_SCOPE, TOYHOST_SUBJECT, ops_fixture
from packages.evals.tests.run_pages_support import BatchRecordingStore
from packages.evals.tests.simulation_support import (
    ClusteredDesign,
    at_least,
    case_means,
    draw_paired_arms,
    family_verdicts,
    within,
)
from packages.evals.tests.simulation_support import paired_t_power as reference_power

#: A host measure with a better end (lower), on the quality axis and with no declared range: what a family tests.
READING = "fields_stripped"
SCOPE = "power-scope"
TEMPLATE = "tpl-power"


class TestThePowerArithmetic:
    @pytest.mark.parametrize("n_pairs", [2, 3, 5, 8, 15, 40])
    @pytest.mark.parametrize("alpha", [SIGNIFICANCE_ALPHA, SIGNIFICANCE_ALPHA / 4])
    def test_it_agrees_with_an_independent_integral(self, n_pairs: int, alpha: float) -> None:
        for effect_size in (0.3, 0.8, 1.5, 3.0):
            engine = paired_t_power(n_pairs, effect_size, 1.0, alpha=alpha)
            assert engine == pytest.approx(reference_power(n_pairs, effect_size, alpha), abs=0.002)

    def test_the_detectable_difference_is_the_power_inverted_and_scales_with_the_spread(self) -> None:
        delta = paired_detectable_difference(10, 1.0, alpha=SIGNIFICANCE_ALPHA)
        assert reference_power(10, delta, SIGNIFICANCE_ALPHA) == pytest.approx(DETECTABLE_POWER, abs=0.002)
        assert paired_detectable_difference(10, 2.5, alpha=SIGNIFICANCE_ALPHA) == pytest.approx(2.5 * delta)

    def test_too_few_pairs_or_no_spread_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two pairs"):
            paired_t_power(1, 1.0, 1.0, alpha=SIGNIFICANCE_ALPHA)
        with pytest.raises(ValueError, match="must be positive"):
            paired_detectable_difference(5, 0.0, alpha=SIGNIFICANCE_ALPHA)


def _detection_rate(
    rng: random.Random, design: ClusteredDesign, *, correlation: float, readings: int, effect: float, replicates: int
) -> float:
    """How often the family rule calls the one moved reading improved, the other ``readings − 1`` unmoved."""
    cases = [f"case-{index}" for index in range(design.n_cases)]
    found = 0
    for _ in range(replicates):
        family = []
        for reading in range(readings):
            control, contrast = draw_paired_arms(
                rng, design, effect=effect if reading == 0 else 0.0, correlation=correlation
            )
            family.append((dict(zip(cases, case_means(control))), dict(zip(cases, case_means(contrast))), True))
        found += family_verdicts(family)[0].verdict == "improved"
    return found / replicates


class TestTheStatedDifferenceIsFoundFourTimesInFive:
    """With the variance components known, the stated Δ is detected at 80%, within 4 Monte-Carlo SEs."""

    #: 2,000 replicates: SE at 80% is 0.0089, a 4-SE band of ±0.036.
    REPLICATES = 2000

    @pytest.mark.parametrize(
        ("n_cases", "repeats", "between_sd", "repeat_sd", "correlation", "readings"),
        [(8, 3, 1.0, 0.5, 0.5, 1), (5, 3, 1.0, 0.5, 0.0, 1), (15, 2, 0.5, 1.0, 0.3, 1), (10, 3, 1.0, 0.5, 0.5, 3)],
    )
    def test_through_the_family_rule(
        self, n_cases: int, repeats: int, between_sd: float, repeat_sd: float, correlation: float, readings: int
    ) -> None:
        """The truth of ``draw_paired_arms``: a case's difference of means has variance
        ``2 between² (1 − ρ) + 2 repeat² / k``; the family of ``readings`` asks the moved one to clear α/m."""
        rng = random.Random(f"power-{n_cases}-{repeats}-{correlation}-{readings}")
        design = ClusteredDesign(n_cases=n_cases, repeats=repeats, between_case_sd=between_sd, repeat_sd=repeat_sd)
        difference_sd = math.sqrt(2 * between_sd**2 * (1 - correlation) + 2 * repeat_sd**2 / repeats)
        delta = paired_detectable_difference(n_cases, difference_sd, alpha=SIGNIFICANCE_ALPHA / readings)

        rate = _detection_rate(
            rng, design, correlation=correlation, readings=readings, effect=delta, replicates=self.REPLICATES
        )

        assert within(rate, DETECTABLE_POWER, self.REPLICATES), f"Δ={delta:.3f} found {rate:.4f} of the time"

    def test_planned_with_no_credit_for_pairing_it_is_found_at_least_as_often(self) -> None:
        """The ``unrelated_case_effects`` basis takes two arms' case levels as unrelated; where they agree (ρ = 0.6)
        the pairing cancels more, so the stated Δ is found more often than 80%, never less."""
        rng = random.Random("power-unrelated-basis")
        design = ClusteredDesign(n_cases=10, repeats=3, between_case_sd=1.0, repeat_sd=0.5)
        delta = paired_detectable_difference(10, math.sqrt(2 * 1.0 + 2 * 0.25 / 3), alpha=SIGNIFICANCE_ALPHA)

        rate = _detection_rate(rng, design, correlation=0.6, readings=1, effect=delta, replicates=self.REPLICATES)

        assert rate >= at_least(DETECTABLE_POWER, self.REPLICATES)


class TestTheComponentsAreRecovered:
    def test_from_runs_with_known_components(self) -> None:
        """A large seeded history recovers each component within four of its own standard errors."""
        rng = random.Random("components")
        design = ClusteredDesign(n_cases=200, repeats=3, between_case_sd=1.0, repeat_sd=0.5)
        control, contrast = draw_paired_arms(rng, design, correlation=0.5)

        components = variance_components([control, contrast])
        paired = paired_case_variance([(control, contrast)], 0.25 if components is None else components.within_case)

        assert components is not None and paired is not None
        within_se = 0.25 * math.sqrt(2 / components.within_df)
        assert components.within_case == pytest.approx(0.25, abs=4 * within_se)
        # Between: the case means' variance (1 + 0.25/3) over 199 df, less a near-exact noise share.
        assert components.between_case == pytest.approx(1.0, abs=4 * (1.0 + 0.25 / 3) * math.sqrt(2 / 398))
        # Two arms at ρ = 0.5 disagree about a case by 2 · 1² · (1 − 0.5) = 1.
        assert paired[0] == pytest.approx(1.0, abs=4 * (1.0 + 2 * 0.25 / 3) * math.sqrt(2 / 199))

    def test_fewer_than_two_repeated_cases_estimate_nothing(self) -> None:
        assert variance_components([[[1.0], [2.0], [3.0, 3.5]]]) is None
        assert variance_components([[[1.0, 1.2], [2.0, 2.4]]]) is not None


def _store_history(
    storage: EvalStorage,
    runs: dict[str, dict[str, list[float]]],
    *,
    template_id: str = TEMPLATE,
    scope: str = SCOPE,
    payload: int = 0,
) -> None:
    """Save one run per model, each case's repeats carrying ``READING`` at the given values, and ``payload`` characters
    of a kind's verbatim payload each when it is more than none."""
    for model, by_case in runs.items():
        run = make_eval_run(
            id=f"run-{model}",
            scope_id=scope,
            template_id=template_id,
            candidate_model=model,
            status="completed",
            k_runs=max(len(values) for values in by_case.values()),
            test_case_ids=sorted(by_case),
        )
        storage.save_eval_run(run)
        for case, values in by_case.items():
            for repeat, value in enumerate(values, start=1):
                storage.save_eval_result(
                    make_eval_result(
                        id=f"{model}-{case}-{repeat}",
                        eval_run_id=run.id,
                        scope_id=scope,
                        model=model,
                        test_case_id=case,
                        k_iteration=repeat,
                        host_measures={READING: value},
                        kind_payload={"events": f"{model}-{case}-{repeat}:" + "x" * payload} if payload else None,
                    )
                )


def _host() -> EvalHost:
    return replace(ops_fixture().host.eval_host, storage=EvalStorage(InMemoryDocumentStore()))


def _reading(block: DetectableEffects) -> object:
    (effect,) = [effect for effect in block.effects if effect.name == READING]
    return effect


class TestTheEstimateReadsEarlierRuns:
    #: The truth the history is drawn around: case levels SD 1, repeats SD 0.5, two arms agreeing at ρ = 0.5.
    DESIGN = ClusteredDesign(n_cases=60, repeats=3, between_case_sd=1.0, repeat_sd=0.5)

    def _history(self, host: EvalHost, *, paired: bool) -> None:
        control, contrast = draw_paired_arms(random.Random("history"), self.DESIGN, correlation=0.5)
        cases = [f"case-{index:02d}" for index in range(self.DESIGN.n_cases)]
        runs = {"model-a": dict(zip(cases, control))}
        if paired:
            runs["model-b"] = dict(zip(cases, contrast))
        _store_history(host.storage, runs)

    @pytest.mark.parametrize(
        ("paired", "basis", "true_variance"),
        [(True, "paired_runs", 2 * 1.0 * 0.5), (False, "unrelated_case_effects", 2 * 1.0)],
    )
    def test_the_stated_difference_holds_its_power_under_the_truth(
        self, paired: bool, basis: str, true_variance: float
    ) -> None:
        """The figure is planned from estimates, so it is checked against the truth the history was drawn from:
        at 60 earlier cases the variance it planned from is within about a fifth of the truth, so the power the
        stated Δ really has lies within 0.70-0.88 (the band the estimate's own error allows; this seed's is
        printed in any failure)."""
        host = _host()
        self._history(host, paired=paired)

        block = detectable_effects(host, SCOPE, TEMPLATE, n_cases=12, k_runs=2, n_arms=2)

        effect = _reading(block)
        assert effect.basis == basis and effect.delta is not None
        assert effect.run_ids == (["run-model-a", "run-model-b"] if paired else ["run-model-a"])
        true_sd = math.sqrt(true_variance + 2 * 0.25 / 2)
        power = reference_power(12, effect.delta / true_sd, block.per_comparison_alpha)
        assert 0.70 <= power <= 0.88, f"Δ={effect.delta:.3f} has power {power:.3f} under the truth"

    def test_a_template_with_no_earlier_run_cannot_estimate(self) -> None:
        host = _host()
        _store_history(host.storage, {"model-a": {"c1": [1.0, 2.0], "c2": [3.0, 3.0]}}, template_id="another")

        block = detectable_effects(host, SCOPE, TEMPLATE, n_cases=10, k_runs=3, n_arms=2)

        assert block.effects == [] and block.run_ids == []
        assert block.cannot_estimate is not None and "no earlier run of template 'tpl-power'" in block.cannot_estimate

    def test_earlier_runs_with_no_repeated_case_cannot_estimate_the_reading(self) -> None:
        host = _host()
        _store_history(host.storage, {"model-a": {f"c{index}": [float(index)] for index in range(6)}})

        effect = _reading(detectable_effects(host, SCOPE, TEMPLATE, n_cases=10, k_runs=3, n_arms=2))

        assert effect.delta is None and effect.basis is None
        assert (
            effect.cannot_estimate is not None
            and "fewer than two earlier cases were repeated" in effect.cannot_estimate
        )

    def test_one_planned_case_cannot_be_tested(self) -> None:
        host = _host()
        self._history(host, paired=False)

        block = detectable_effects(host, SCOPE, TEMPLATE, n_cases=1, k_runs=3, n_arms=2)

        assert block.cannot_estimate == "a paired test needs two cases per arm; the launch plans 1"


def test_the_launch_estimate_prints_one_line_per_reading() -> None:
    """Through the operation: earlier runs of the toy template give each reading its line."""
    fixture = ops_fixture()
    _store_history(
        fixture.host.eval_host.storage,
        {"m-0": {f"case-{index}": [1.0 + index, 1.5 + index, 0.8 + index] for index in range(4)}},
        template_id=toyhost_template().id,
        scope=TOYHOST_SCOPE,
    )
    arguments = LaunchArguments(
        template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=["m-1", "m-2"], k_runs=2
    )

    estimate = asyncio.run(launch_estimate(fixture.host, arguments, TOYHOST_SCOPE))

    block = estimate.detectable_effect
    assert block is not None and block.run_ids == ["run-m-0"] and block.k_runs == 2
    assert [effect.name for effect in block.effects if effect.delta is not None] == [READING]
    text = estimate_text(estimate)
    assert f"Holm over {block.family_size} comparison(s)" in text
    for effect in block.effects:
        if effect.delta is not None:
            assert f"- {effect.name}: with {block.n_cases} cases and 2 repeats this campaign can detect Δ ≥ " in text
        else:
            assert f"- {effect.name}: cannot estimate: {effect.cannot_estimate}" in text
    assert "assumes: the planned arms run the same cases" in text


class TestThePlanningReadIsHeldToTheReadBudget:
    """Every earlier run's results are read, so past the read tier's budget the block is refused before any result is
    read (#996 is the store computing it). Two runs of 6 cases × 2: 24 results by their matrices."""

    def _host(self) -> tuple[BatchRecordingStore, EvalHost]:
        store = BatchRecordingStore()
        host = replace(ops_fixture().host.eval_host, storage=EvalStorage(store))
        by_case = {f"c{index}": [float(index), float(index) + 0.5] for index in range(6)}
        _store_history(host.storage, {"model-a": by_case, "model-b": by_case})
        store.clear()
        return store, host

    def test_one_result_past_the_budget_is_refused_reading_no_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store, host = self._host()
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 23)

        with pytest.raises(ValidationFailedError) as refused:
            detectable_effects(host, SCOPE, TEMPLATE, n_cases=6, k_runs=2, n_arms=2)

        assert refused.value.details == {"runs": 2, "results_estimated": 24, "result_budget": 23}
        assert refused.value.remedy.startswith("archive the template's earlier runs the plan need not read")
        assert store.typed_reads == [] and store.id_reads == [], "refused before any result was read"

    def test_at_the_budget_it_plans_as_before(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store, host = self._host()
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 24)

        block = detectable_effects(host, SCOPE, TEMPLATE, n_cases=6, k_runs=2, n_arms=2)

        assert block.run_ids == ["run-model-a", "run-model-b"] and _reading(block)
        assert all(keys_only for _, keys_only, _ in store.typed_reads), "no run was read whole"
        assert sum(store.id_reads) == 24

    def test_a_read_returning_more_than_its_runs_promised_past_the_budget_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, host = self._host()
        # Two cells past what run-model-a's matrix promises: 24 by the matrices, 26 stored.
        for extra in (3, 4):
            host.storage.save_eval_result(
                make_eval_result(
                    id=f"model-a-c0-{extra}",
                    eval_run_id="run-model-a",
                    scope_id=SCOPE,
                    model="model-a",
                    test_case_id="c0",
                    k_iteration=extra,
                    host_measures={READING: 1.0},
                )
            )
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 25)

        with pytest.raises(ValidationFailedError) as refused:
            detectable_effects(host, SCOPE, TEMPLATE, n_cases=6, k_runs=2, n_arms=2)

        assert refused.value.details == {
            "runs": 2,
            "results_estimated": 24,
            "result_budget": 25,
            "results_returned": 26,
        }
        assert store.id_reads == [], "counted off the runs' keys, refused before any result was read"


class TestALaunchEstimatePastTheBudgetIsPricedAndSaysWhyItCannotPlan:
    """The price does not depend on the planning block, so a planning read past the budget is the block's disclosure,
    never a refusal of the estimate: the toy template's earlier runs hold 24 results by their matrices."""

    def _fixture(self) -> tuple[BatchRecordingStore, Any, LaunchArguments]:
        store = BatchRecordingStore()
        fixture = ops_fixture(store=store)
        by_case = {f"case-{index}": [1.0 + index, 1.5 + index] for index in range(6)}
        _store_history(
            fixture.host.eval_host.storage,
            {"m-0": by_case, "m-9": by_case},
            template_id=toyhost_template().id,
            scope=TOYHOST_SCOPE,
        )
        arguments = LaunchArguments(
            template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=["m-1", "m-2"], k_runs=2
        )
        return store, fixture, arguments

    def test_over_the_budget_it_is_priced_and_its_planning_block_says_why_reading_no_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, fixture, arguments = self._fixture()
        corpus = sum(
            len(run.test_case_ids) * run.k_runs
            for run in fixture.host.eval_host.storage.query_eval_runs(TOYHOST_SCOPE)
            if run.template_id == toyhost_template().id and not run.archived
        )
        within = asyncio.run(launch_estimate(fixture.host, arguments, TOYHOST_SCOPE))
        assert within.detectable_effect is not None and within.detectable_effect.run_ids, "it plans within the budget"
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", corpus - 1)
        store.clear()

        estimate = asyncio.run(launch_estimate(fixture.host, arguments, TOYHOST_SCOPE))

        assert estimate.arms and estimate.arms == within.arms, "priced as within the budget"
        assert estimate.total_predicted_usd == within.total_predicted_usd
        assert estimate.would_launch == within.would_launch
        block = estimate.detectable_effect
        assert block is not None and block.effects == [] and block.run_ids == []
        assert block.cannot_estimate is not None
        assert f"hold about {corpus} results by their matrices, past the read tier's budget of {corpus - 1}" in (
            block.cannot_estimate
        )
        assert "archive the template's earlier runs the plan need not read" in block.cannot_estimate
        assert f"- cannot estimate: {block.cannot_estimate}" in estimate_text(estimate)
        assert store.row_reads() == [], "no result of the template's earlier runs was read"

    def test_a_direct_call_still_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, fixture, _ = self._fixture()
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 1)

        with pytest.raises(ValidationFailedError, match="detectable_effects would aggregate about"):
            detectable_effects(
                fixture.host.eval_host, TOYHOST_SCOPE, toyhost_template().id, n_cases=6, k_runs=2, n_arms=2
            )


def _peak(read: Callable[[], object]) -> int:
    """The peak memory ``read`` allocates above what was held before it, in bytes."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        read()
        return tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()


class TestThePlanningReadHoldsAPageNotTheHistory:
    """The planning block reads every result of every earlier run of the template: in a live history, dozens of runs
    of hundreds of results, each carrying tens of kilobytes verbatim. Read whole, they were all held at once (a 2 GB
    pod was killed twice on 2026-10-30 by launch estimates). Walked a page at a time, each result read down to its
    values before the next page, a history four times as long adds only those values: far less per result than one
    result weighs."""

    #: The characters of verbatim payload each stored result carries.
    PAYLOAD = 16_000

    def _runs(self, first: int, last: int) -> dict[str, dict[str, list[float]]]:
        return {
            f"model-{index:02d}": {f"c{case}": [case + index / 10, case + 0.5, case + 0.25] for case in range(10)}
            for index in range(first, last)
        }

    def test_four_times_the_history_adds_its_values_not_its_results(self) -> None:
        host = _host()
        _store_history(host.storage, self._runs(0, 3), payload=self.PAYLOAD)

        def plan() -> DetectableEffects:
            return detectable_effects(host, SCOPE, TEMPLATE, n_cases=10, k_runs=3, n_arms=2)

        plan()
        short = _peak(plan)
        _store_history(host.storage, self._runs(3, 12), payload=self.PAYLOAD)
        block = plan()
        long = _peak(plan)

        assert len(block.run_ids) == 12 and _reading(block)
        added = 9 * 10 * 3
        # Held whole, each added result costs at least its payload; read down to its values, a small fraction of it.
        assert (long - short) / added < self.PAYLOAD / 4, (short, long)


class TestTheCrossRunRulesHoldAcrossPages:
    """The planning read keeps each result's values and applies what reads across runs once every run is read, so a
    rule decided by a later run reaches a result read pages earlier. Each result is its own page here."""

    @pytest.fixture(autouse=True)
    def _a_result_a_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(comparisons, "RUN_RESULTS_PAGE", 1)

    def _result(self, run: str, case: str, k: int, **fields: Any) -> EvalResult:
        return make_eval_result(
            id=f"{run}-{case}-{k}",
            eval_run_id=f"run-{run}",
            scope_id=SCOPE,
            model=run,
            test_case_id=case,
            k_iteration=k,
            **fields,
        )

    def test_a_miss_a_run_grading_otherwise_and_a_boundary_are_read_as_the_whole_history_reads_them(self) -> None:
        host = _host()
        refused = {"candidate_error": "refused", "host_measures": {}}
        boundary_tone = RubricScore(
            dim="conversation.tone", score=3, scale="ordinal", axis="boundary", judge_temperature=0.0
        )
        history = {
            # Read first: its failure on c1 is a miss only because run b, read last, classifies c1.
            # Its one page also scores conversation.tone as a boundary: a guardrail, however many capability scores of
            # it are read after.
            "c": [self._result("c", "c1", 1, **refused, rubric_scores=[boundary_tone])],
            # Classifies c0; its failure there is a miss.
            "a": [
                self._result("a", "c0", 1, host_measures={"match": True}),
                self._result("a", "c0", 2, host_measures={"match": True}),
                self._result("a", "c0", 3, **refused),
            ],
            # Classifies c1, but answers c0 without a verdict on its last page: it grades otherwise, so its failure on
            # c1 is no miss.
            "b": [
                self._result("b", "c1", 1, host_measures={"match": True}),
                self._result("b", "c1", 2, **refused),
                self._result("b", "c0", 1, host_measures={}),
            ],
        }
        runs = []
        for model, results in history.items():
            run = make_eval_run(
                id=f"run-{model}",
                scope_id=SCOPE,
                template_id=TEMPLATE,
                candidate_model=model,
                status="completed",
                k_runs=3,
                test_case_ids=["c0", "c1"],
            )
            host.storage.save_eval_run(run)
            runs.append(run)
            for result in results:
                host.storage.save_eval_result(result)
        walked = {run.id: iter(history[run.candidate_model]) for run in runs}

        readings = {reading.name: reading for reading in planning_readings(runs, walked, profile=host.profile)}

        assert readings["accuracy"].repeats == {
            "run-c": {"c1": [0.0]},
            "run-a": {"c0": [1.0, 1.0, 0.0]},
            "run-b": {"c1": [1.0]},
        }
        assert "conversation.tone" not in readings, "scored once as a boundary, on the first page read: a guardrail"
