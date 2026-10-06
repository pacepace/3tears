"""Every measure shape is summarised in its own terms, over its own population.

* **Boolean** — a condition that held or did not: a rate with a Wilson interval, never a percentile.
* **Text** — words a host recorded (a ruling, a reason): every observation listed whole, nothing
  aggregated, and no reference may read one as a number.
* **Classifier** — ``match`` is a boolean, ``confusion_cell`` is categorical, and each label's
  precision, recall and F1 are derived from the confusion counts.
* **Population** — a ``scored`` measure leaves out every result the harness faulted, an
  ``all_observed`` one keeps them, and every summary states which; a measure that declares none takes
  the population of the surface asking, so a cell and a run summary report one name over the
  population each always used.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.references import resolve_reading
from threetears.evals.analysis.stats import wilson_interval
from threetears.evals.contracts import (
    EvalResult,
    MetricDescriptor,
    classifier_label_measure,
    classifier_label_of,
    confusion_cell,
)
from threetears.evals.contracts.analysis_measures import MeasureCollection, MeasureSummary
from threetears.evals.contracts.host import SHARED_CORE, HostProfile, MeasureRegistry
from threetears.evals.contracts.metrics import confusion_of, describe_measure
from threetears.evals.contracts.surface import CellFacts, DecisionSurface, MeasureFacts
from packages.evals.tests.bundle_support import one_batch_bundle
from packages.evals.tests.factories import make_eval_result


def _measure(name: str, data_type: str, **extra: object) -> MetricDescriptor:
    return MetricDescriptor(
        name=name,
        data_type=data_type,  # type: ignore[arg-type]
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description=f"{name}, for the test.",
        **extra,  # type: ignore[arg-type]
    )


_PROFILE = HostProfile(
    host_id="shapes",
    host_sweepables=SHARED_CORE,
    measures=MeasureRegistry(
        (
            _measure("on_target", "boolean", higher_is_better=True),
            _measure("ruling_text", "text"),
            _measure("spend_usd", "numeric", higher_is_better=False, population="all_observed"),
            _measure("grade", "numeric", higher_is_better=True, population="scored"),
            _measure("bare", "numeric", higher_is_better=True),
        )
    ),
)


def _result(case: str, faulted: bool = False, **measures: bool | float | str) -> EvalResult:
    return make_eval_result(
        test_case_id=case,
        host_measures=measures,
        infra_error="apparatus: the rig broke" if faulted else None,
        rubric_scores=[],
        goal_state_outcomes=[],
    )


def _collection(results: list[EvalResult], surface: str = "cell") -> MeasureCollection:
    """The measures one surface of the assembled bundle summarises over ``results``.

    ``cell`` is the one cell's facts, whose undeclared population is ``scored``; ``run`` is the batch's
    run summary, whose undeclared population is ``all_observed``.
    """
    bundle = one_batch_bundle(results, profile=_PROFILE)
    if surface == "run":
        (summary,) = bundle.run_summaries
        return summary.measures
    (cell,) = bundle.cell_measures
    return cell.measures


def _summaries(results: list[EvalResult], surface: str = "cell") -> dict[str, MeasureSummary]:
    return {summary.name: summary for summary in _collection(results, surface).measures}


# --- boolean -------------------------------------------------------------------------------------------


def test_a_boolean_measure_is_a_rate_with_a_wilson_interval() -> None:
    results = [_result(f"c{i}", on_target=held) for i, held in enumerate([True, True, True, False])]

    summary = _summaries(results)["on_target"]

    assert (summary.rate, summary.n_true, summary.n) == (0.75, 3, 4)
    assert (summary.ci_low, summary.ci_high) == pytest.approx(wilson_interval(3, 4))
    assert summary.mean is None and summary.p50 is None, "a condition is counted, never averaged into a distribution"


def test_a_rate_at_an_end_keeps_its_width() -> None:
    low, high = wilson_interval(3, 3)  # type: ignore[misc]
    assert high == 1.0 and low < 0.5, "three for three is not certainty"


def test_a_number_under_a_boolean_name_is_dropped_and_reported() -> None:
    collection = _collection([_result("c1", on_target=1.0)])

    assert "on_target" not in {summary.name for summary in collection.measures}
    assert "on_target" in collection.unreported_observations


def test_a_boolean_reads_its_rate_as_the_point_estimate() -> None:
    surface = _surface({"on_target": _summaries([_result("c1", on_target=True), _result("c2", on_target=False)])})
    assert resolve_reading(surface, cell_ref("cell", "rig"), "on_target", "measure").mean == 0.5


# --- text ----------------------------------------------------------------------------------------------


def test_a_text_measure_lists_every_observation_whole_and_aggregates_nothing() -> None:
    rulings = ["Grapple holds: the ogre is Large.", "Opportunity attack allowed.", "Opportunity attack allowed."]
    summary = _summaries([_result(f"c{i}", ruling_text=text) for i, text in enumerate(rulings)])["ruling_text"]

    assert summary.texts == rulings, "listed in order, duplicates kept — nothing is counted or deduplicated"
    assert (summary.mean, summary.rate, summary.categories) == (None, None, {})


def test_a_text_measure_is_never_a_number_in_a_reference() -> None:
    surface = _surface({"ruling_text": _summaries([_result("c1", ruling_text="Allowed.")])})
    with pytest.raises(UnresolvableReference, match="it is text"):
        resolve_reading(surface, cell_ref("cell", "rig"), "ruling_text", "measure")


def test_a_summary_takes_exactly_one_shape() -> None:
    with pytest.raises(ValueError, match="exactly one shape"):
        MeasureSummary(
            name="x", attribution_scope="end_to_end", population="scored", n=1, rate=1.0, n_true=1, texts=["y"]
        )
    with pytest.raises(ValueError, match="needs both rate and n_true"):
        MeasureSummary(name="x", attribution_scope="end_to_end", population="scored", n=1, rate=1.0)


# --- classifier ----------------------------------------------------------------------------------------


def test_a_confusion_cell_round_trips_labels_carrying_the_arrow() -> None:
    assert confusion_of(confusion_cell("go → left", "attack")) == ("go → left", "attack")
    assert confusion_of("not a cell") is None
    with pytest.raises(ValueError):
        confusion_cell("", "attack")


def test_per_label_precision_recall_and_f1_come_from_the_confusion_counts() -> None:
    """Four cases: attack→attack twice, attack→move once, move→move once."""
    cells = [("attack", "attack"), ("attack", "attack"), ("attack", "move"), ("move", "move")]
    results = [
        _result(f"c{i}", match=expected == predicted, confusion_cell=confusion_cell(expected, predicted))
        for i, (expected, predicted) in enumerate(cells)
    ]

    summaries = _summaries(results)

    assert summaries["match"].rate == 0.75
    assert summaries["confusion_cell"].categories == {"attack → attack": 2, "attack → move": 1, "move → move": 1}
    precision_attack = summaries[classifier_label_measure("precision", "attack")]
    assert (precision_attack.rate, precision_attack.n) == (1.0, 2)
    recall_attack = summaries[classifier_label_measure("recall", "attack")]
    assert (recall_attack.rate, recall_attack.n) == (pytest.approx(2 / 3), 3)
    precision_move = summaries[classifier_label_measure("precision", "move")]
    assert (precision_move.rate, precision_move.n) == (0.5, 2)
    f1_attack = summaries[classifier_label_measure("f1", "attack")]
    assert (f1_attack.mean, f1_attack.n) == (pytest.approx(0.8), 3), "n is the label's support: predicted or expected"
    f1_move = summaries[classifier_label_measure("f1", "move")]
    assert (f1_move.mean, f1_move.n) == (pytest.approx(2 / 3), 2)


def test_a_label_predicted_but_never_expected_has_no_recall_and_no_f1() -> None:
    """F1 is the harmonic mean of precision and recall, so a label missing recall has none — never a mean of 0.0
    stated over n=0. Its precision exists (0 of 1), and is reported."""
    cells = [("attack", "attack"), ("attack", "move")]
    results = [
        _result(f"c{i}", match=expected == predicted, confusion_cell=confusion_cell(expected, predicted))
        for i, (expected, predicted) in enumerate(cells)
    ]

    summaries = _summaries(results)

    assert summaries[classifier_label_measure("precision", "move")].rate == 0.0
    assert classifier_label_measure("recall", "move") not in summaries
    assert classifier_label_measure("f1", "move") not in summaries
    assert summaries[classifier_label_measure("f1", "attack")].mean == pytest.approx(2 / 3)


def test_an_f1_reading_states_that_it_has_no_spread_by_construction() -> None:
    """ "Unestimable at n=…" implies more data would supply a spread; F1 is one value from one matrix, at any n."""
    cells = [("attack", "attack"), ("attack", "attack"), ("attack", "move"), ("move", "move")]
    results = [
        _result(f"c{i}", match=expected == predicted, confusion_cell=confusion_cell(expected, predicted))
        for i, (expected, predicted) in enumerate(cells)
    ]
    summaries = _summaries(results)
    f1 = classifier_label_measure("f1", "attack")
    recall = classifier_label_measure("recall", "attack")
    surface = _surface({"cell": {f1: summaries[f1], recall: summaries[recall]}})

    assert resolve_reading(surface, cell_ref("cell", "rig"), f1, "measure").dispersion.startswith(
        "none by construction"
    )
    assert "CI" in resolve_reading(surface, cell_ref("cell", "rig"), recall, "measure").dispersion


def test_accuracy_is_derived_from_match_as_the_classifier_s_quality_reading() -> None:
    """``match`` is the one verdict a kind lands; ``accuracy`` is its per-observation 0/1, with a mean to rank on."""
    results = [_result(f"c{i}", match=held) for i, held in enumerate([True, True, True, False])]

    summaries = _summaries(results)

    assert summaries["match"].rate == 0.75
    accuracy = summaries["accuracy"]
    assert (accuracy.mean, accuracy.n, accuracy.rate) == (0.75, 4, None), "a numeric reading, so a family can test it"
    assert describe_measure("accuracy", _PROFILE.measures).merit_axis == "quality"
    assert describe_measure("match", _PROFILE.measures).merit_axis is None, "one verdict, one reading on the axis"


def test_no_match_no_accuracy_and_a_non_bool_match_is_reported_under_its_own_name() -> None:
    assert "accuracy" not in _summaries([_result("c1", on_target=True)])

    collection = _collection([_result("c1", match=1.0)])

    assert "match" in collection.unreported_observations
    assert not {"match", "accuracy"} & {summary.name for summary in collection.measures}


def test_a_confusion_cell_that_does_not_split_is_dropped_and_reported() -> None:
    collection = _collection([_result("c1", confusion_cell="attack->move")])
    assert "confusion_cell" in collection.unreported_observations
    assert not any(summary.name.startswith("classifier:") for summary in collection.measures)


# --- population ----------------------------------------------------------------------------------------


def test_each_measure_is_computed_over_its_declared_population() -> None:
    """One faulted result: it spent money (kept by `all_observed`), and its grade counts for nothing."""
    results = [_result("c1", spend_usd=0.01, grade=0.9), _result("c2", faulted=True, spend_usd=0.03, grade=0.1)]

    summaries = _summaries(results)

    assert (summaries["spend_usd"].n, summaries["spend_usd"].population) == (2, "all_observed")
    assert (summaries["grade"].n, summaries["grade"].population) == (1, "scored")
    assert summaries["grade"].mean == 0.9


def test_a_measure_declaring_no_population_takes_the_surfaces() -> None:
    results = [_result("c1", bare=1.0), _result("c2", faulted=True, bare=0.0)]

    on_a_cell = _summaries(results, "cell")["bare"]
    on_a_run = _summaries(results, "run")["bare"]

    assert (on_a_cell.n, on_a_cell.population) == (1, "scored")
    assert (on_a_run.n, on_a_run.population) == (2, "all_observed")


def _surface(summaries: dict[str, dict[str, MeasureSummary]]) -> DecisionSurface:
    measures = [summary for by_name in summaries.values() for summary in by_name.values()]
    return DecisionSurface(
        cells=[
            CellFacts(
                variant_key="cell",
                apparatus_class_id="rig",
                run_ids=["run"],
                n_observations=2,
                measures=MeasureCollection(measures=sorted(measures, key=lambda m: m.name)),
            )
        ],
        measures={summary.name: MeasureFacts() for summary in measures},
    )


def test_a_proportion_with_more_hits_than_trials_is_refused() -> None:
    with pytest.raises(ValueError, match="0 <= n_true <= n"):
        wilson_interval(5, 4)
    with pytest.raises(ValueError, match="0 <= n_true <= n"):
        wilson_interval(-1, 4)
    assert wilson_interval(0, 0) is None, "no observations, no rate to bound"


def test_a_per_label_name_round_trips_and_refuses_a_blank_label() -> None:
    name = classifier_label_measure("recall", "go | left")
    assert classifier_label_of(name) == ("recall", "go | left")
    assert classifier_label_of("classifier:accuracy:attack") is None, "accuracy is not a per-label statistic"
    described = describe_measure(name, _PROFILE.measures)
    assert (described.family, described.value_range, described.higher_is_better) == ("classifier", (0.0, 1.0), True)
    with pytest.raises(ValueError):
        classifier_label_measure("precision", "  ")
