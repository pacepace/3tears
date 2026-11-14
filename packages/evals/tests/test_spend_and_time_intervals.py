"""Which measures the bundle reads on the interval for a skewed mean, and what each surface says where it has none.

``stats.observed_mean_interval(..., skewed=True)`` holds its level on a spend or a time
(``tests/test_simulated_floored_mean.py``); this file holds the wiring. Through the assembled bundle: a measure in
dollars off the cost axis, one in seconds, one on each of the cost and latency axes and the engine's own
``total_ms`` take that interval, exactly,
and a count and a score keep theirs. With under five cases above zero a spend or a time states none, and a bar
proposal, a reading's cell and an evidence row each say so, with how many cases would give one, rather than a false
reason or the mean ± its standard error.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

import pytest

from threetears.evals.analysis import (
    AnalysisContextBundle,
    assemble_context_bundle,
    build_code_only_report,
    report_markdown,
)
from threetears.evals.analysis.stats import SKEWED_INTERVAL_WITHHELD, SMALL_N_BAND_FLOOR, observed_mean_interval
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.host import SHARED_CORE, HostProfile, MeasureRegistry
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.schema import EvalResult, GoalStateOutcome, LatencyMetrics
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import fixture_variant_key, make_campaign, make_eval_result, make_eval_run

_SCOPE = "uni-1"
_CHECK = "state.answer.delivered == true"
_MODEL = "alpha"


def _descriptor(name: str, unit: str | None, **more: object) -> MetricDescriptor:
    return MetricDescriptor(
        name=name,
        reader_name=name.replace("_", " ").capitalize(),
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description=f"The {name.replace('_', ' ')} of one turn.",
        higher_is_better=False,
        unit=unit,
        **more,  # type: ignore[arg-type]
    )


#: name -> (descriptor, whether the bundle reads it on the skewed interval).
_MEASURES: dict[str, tuple[MetricDescriptor, bool]] = {
    "judge_spend": (_descriptor("judge_spend", "usd", nonnegative=True), True),
    "wait_seconds": (_descriptor("wait_seconds", "s", nonnegative=True), True),
    "queue_depth": (_descriptor("queue_depth", None, nonnegative=True, merit_axis="latency"), True),
    "billed_units": (_descriptor("billed_units", None, nonnegative=True, merit_axis="cost"), True),
    "parcels_sent": (_descriptor("parcels_sent", "count", nonnegative=True), False),
    "tokens_read": (_descriptor("tokens_read", "tokens", nonnegative=True), False),
    "drift": (_descriptor("drift", "usd"), False),
    "share_kept": (_descriptor("share_kept", None, value_range=(0.0, 1.0)), False),
}

_PROFILE = HostProfile(
    host_id="spend-and-time",
    host_sweepables=SHARED_CORE,
    measures=MeasureRegistry([descriptor for descriptor, _ in _MEASURES.values()]),
)


def _bundle(
    values: Mapping[str, list[float]], *, total_ms: list[float], cost_usd: list[float]
) -> AnalysisContextBundle:
    """One arm, one observation per case: case ``i`` carries each measure's ``i``-th value."""
    storage = EvalStorage(InMemoryDocumentStore())
    cases = [f"case-{index:02d}" for index in range(len(total_ms))]
    run = make_eval_run(
        id="run-alpha", scope_id=_SCOPE, candidate_model=_MODEL, k_runs=1, test_case_ids=cases, status="completed"
    )
    storage.save_eval_run(run)
    for index, case in enumerate(cases):
        result: EvalResult = make_eval_result(
            scope_id=_SCOPE,
            eval_run_id=run.id,
            test_case_id=case,
            model=_MODEL,
            variant_key=fixture_variant_key(_MODEL),
            latency=LatencyMetrics(total_ms=total_ms[index]),
            cost_usd=cost_usd[index],
            host_measures={name: series[index] for name, series in values.items()},
            goal_state_outcomes=[GoalStateOutcome(expression=_CHECK, passed=True)],
            rubric_scores=[],
        )
        storage.save_eval_result(result)
    campaign = make_campaign(scope_id=_SCOPE, run_ids=[run.id])
    storage.save_campaign(campaign)
    return assemble_context_bundle(campaign, storage=storage, profile=_PROFILE)


def _skewed_draws(seed: str, n: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.lognormvariate(0.0, 1.0) for _ in range(n)]


def _summaries(bundle: AnalysisContextBundle) -> dict[str, object]:
    (cell,) = bundle.cell_measures
    return {summary.name: summary for summary in cell.measures.measures}


N = 12


@pytest.fixture(scope="module")
def twelve_cases() -> tuple[dict[str, list[float]], dict[str, object]]:
    values = {
        name: [min(1.0, value / 10.0) for value in _skewed_draws(name, N)]
        if name == "share_kept"
        else _skewed_draws(name, N)
        for name in _MEASURES
    }
    values["total_ms"] = [1000.0 * value for value in _skewed_draws("total_ms", N)]
    values["cost_usd"] = [0.01 * value for value in _skewed_draws("cost_usd", N)]
    bundle = _bundle(
        {name: values[name] for name in _MEASURES}, total_ms=values["total_ms"], cost_usd=values["cost_usd"]
    )
    return values, _summaries(bundle)


@pytest.mark.parametrize(
    ("name", "skewed"), [*((name, skewed) for name, (_, skewed) in _MEASURES.items()), ("total_ms", True)]
)
def test_a_spend_or_a_time_takes_the_skewed_interval_and_nothing_else_does(
    name: str, skewed: bool, twelve_cases: tuple[dict[str, list[float]], dict[str, object]]
) -> None:
    """Exactly: the bundle's ends are the engine's for the same observations, so dropping the wiring fails here."""
    values, summaries = twelve_cases
    summary = summaries[name]
    descriptor = _MEASURES[name][0] if name in _MEASURES else None
    value_range = descriptor.value_range if descriptor is not None else None
    floor = descriptor.interval_floor if descriptor is not None else 0.0
    cases = list(range(N))
    wide = observed_mean_interval(values[name], cases=cases, value_range=value_range, floor=0.0, skewed=True)
    plain = observed_mean_interval(values[name], cases=cases, value_range=value_range, floor=floor)
    assert wide is not None and plain is not None
    stated = (summary.ci_low, summary.ci_high)  # type: ignore[attr-defined]
    if skewed:
        assert stated == pytest.approx(wide, rel=1e-12)
        assert wide[1] > plain[1] * 1.05, "on these draws the skewed interval is visibly the wider one"
    else:
        assert stated == pytest.approx(plain, rel=1e-12)
        if value_range is None:
            assert stated != pytest.approx(wide, rel=1e-6)


FEW = SMALL_N_BAND_FLOOR - 1


@pytest.fixture(scope="module")
def four_cases() -> AnalysisContextBundle:
    values = {name: _skewed_draws(name, FEW) for name in _MEASURES}
    values["share_kept"] = [0.2, 0.4, 0.5, 0.9]
    return _bundle(values, total_ms=[900.0, 1100.0, 1500.0, 4000.0], cost_usd=[0.01, 0.012, 0.02, 0.09])


def test_under_five_cases_a_spend_or_a_time_states_no_interval_and_a_count_still_does(
    four_cases: AnalysisContextBundle,
) -> None:
    summaries = _summaries(four_cases)
    for name in ("total_ms", "judge_spend", "wait_seconds", "queue_depth", "billed_units"):
        summary = summaries[name]
        assert (summary.ci_low, summary.ci_high) == (None, None), name  # type: ignore[attr-defined]
        assert summary.sem is not None, "the standard error is still estimated"  # type: ignore[attr-defined]
    assert summaries["parcels_sent"].ci_high is not None  # type: ignore[attr-defined]


def test_the_reason_says_how_many_cases_would_give_one() -> None:
    assert SKEWED_INTERVAL_WITHHELD == (
        f"no interval: a spend or a time states one from {SMALL_N_BAND_FLOOR} cases above zero, and this has fewer"
    )


def test_a_readings_cell_prints_the_mean_and_the_reason_never_the_mean_and_its_standard_error(
    four_cases: AnalysisContextBundle,
) -> None:
    report = build_code_only_report(four_cases, assembled_at="2026-11-14T00:00:00+00:00")
    page = report_markdown(report)
    withheld = [line for line in page.splitlines() if SKEWED_INTERVAL_WITHHELD in line]
    assert withheld, "no line of the page gives the reason"
    assert all("±" not in line for line in withheld)
