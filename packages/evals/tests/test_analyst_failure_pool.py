"""The failures an analyst's examples are a sample of, and the cases two aspects share.

Contracts held here (``threetears.evals.analysis.report_input_build.report_input`` with ``counted``, as
``analyst.snapshot.analyst_input`` builds a snapshot's input; ``analyst.facts.examples_list`` and
``campaign_overview``; the ``count.examples``, ``count.failures`` and ``overlap`` key families; the example cards):

- **each slice's failed trials are counted off the bundle**, a failure as the examples count one (a nought on the
  primary judged dimension, else a failed trial), so every kept example falls in a slice that failed at least as
  often as it was kept from; an aspect whose bundle left failures out is not counted, and reads as not known;
- **examples_list says what the examples are a sample of**: per aspect and arm the filters keep, how many it froze
  against how many failed, and why not where the snapshot does not count them (a bare input);
- **an example carries the start of what its case asked**, where the bundle holds it;
- **a planner's input counts nothing**, so its view is what it was;
- **which cases two aspects ran** is counted off the bundle's case ids, keyed and listed by the overview, and reads
  as not known where the snapshot does not hold them;
- **an example card names its aspect** where the input holds more than one.

Every expectation is computed from the bundle or the input, never typed.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from packages.evals.tests.analyst_support import (
    KAIRO,
    SHAPE_NAMES,
    budget_inputs,
    fixture_inputs,
    fixture_snapshots,
    shape_bundle,
)
from packages.evals.tests.fixtures.report_plans.dataset import kairo_bundle
from threetears.evals.analysis.analyst.blocks import block_render
from threetears.evals.analysis.analyst.facts import campaign_overview, examples_list
from threetears.evals.analysis.analyst.snapshot import analyst_input, frozen_input
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.errors import UnstatedFigure
from threetears.evals.analysis.report.figure_keys import OVERLAP_WORDS, reader_name, resolve
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import EXAMPLE_EXCERPT_CHARS, planner_view, report_input
from threetears.evals.kernel.report_plan import ChartQuery


def _bundles() -> dict[str, CampaignBundle]:
    return {KAIRO: kairo_bundle(), **{name: shape_bundle(name) for name in SHAPE_NAMES}}


#: Every campaign frozen from its bundle, by name.
BUNDLES = _bundles()


def _counted(name: str) -> ReportInput:
    if name == KAIRO:
        return frozen_input(fixture_snapshots("s")[KAIRO])
    return budget_inputs()[name]


def _failures(bundle: CampaignBundle, primary: str | None) -> dict[tuple[str, str, str | None], int]:
    """Each slice's failed trials, read off the bundle's trials as the examples count a failure."""
    counted: dict[tuple[str, str, str | None], int] = {}
    for aspect in bundle.aspects:
        for trial in aspect.trials:
            key = (aspect.aspect, bundle.label_of(trial.arm), trial.stratum)
            failed = trial.judged.get(primary) == 0 if primary is not None else trial.state == "failed"
            counted[key] = counted.get(key, 0) + int(failed)
    return counted


def _every_failure_held(bundle: CampaignBundle) -> bool:
    return all(
        aspect.trials_omitted == 0 or any(trial.state == "passed" for trial in aspect.trials)
        for aspect in bundle.aspects
    )


@pytest.mark.parametrize("name", sorted(BUNDLES))
def test_each_slices_failed_trials_are_counted_off_the_bundle(name: str) -> None:
    bundle, held = BUNDLES[name], _counted(name)
    assert _every_failure_held(bundle), "each fixture bundle holds every failure"
    assert held.failed_trials is not None
    read = {(row.aspect, row.arm, row.stratum): row.failed for row in held.failed_trials}
    assert read == _failures(bundle, held.primary_judged)
    for example in held.examples:
        kept = sum(
            (e.aspect, e.arm, e.stratum) == (example.aspect, example.arm, example.stratum) for e in held.examples
        )
        assert read[example.aspect or "", example.arm or "", example.stratum] >= kept


def test_an_aspect_whose_bundle_left_failures_out_is_not_counted_and_reads_as_not_known() -> None:
    bundle = shape_bundle("s3")
    capped, whole = bundle.aspects
    failing = [trial for trial in capped.trials if trial.state != "passed"]
    left_out = capped.model_copy(update={"trials": failing[:1], "trials_omitted": len(capped.trials) - 1})
    held = analyst_input(bundle.model_copy(update={"aspects": [left_out, whole]}))
    assert held.failed_trials is not None
    assert {row.aspect for row in held.failed_trials} == {whole.aspect}
    arm = held.arms[0].id
    with pytest.raises(UnstatedFigure):
        resolve(f"count.failures.{capped.aspect}.{arm}", held)
    with pytest.raises(UnstatedFigure):
        resolve("count.failures", held)
    pair = f"overlap.{capped.aspect}.{whole.aspect}"
    assert resolve(f"{pair}.same", held).value == OVERLAP_WORDS["unknown"]
    with pytest.raises(UnstatedFigure):
        resolve(f"{pair}.shared", held)


def _inputs() -> dict[str, ReportInput]:
    held = dict(fixture_inputs())
    return held | {name: _counted(name) for name in BUNDLES}


#: Every input examples_list is held to: the bare fixtures, and every campaign frozen from its bundle.
INPUTS = _inputs()


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_examples_list_says_how_many_it_kept_of_how_many_failed(name: str) -> None:
    held = INPUTS[name]
    listing = examples_list(held, {}, limit=200)
    named = {(e.aspect, e.arm) for e in held.examples if e.aspect is not None and e.arm is not None}
    named |= {(row.aspect, row.arm) for row in held.failed_trials or ()}
    assert {(row.aspect, str(row.arm.value)) for row in listing.kept_from} == named
    for row in listing.kept_from:
        arm = str(row.arm.value)
        kept = [e for e in held.examples if (e.aspect, e.arm) == (row.aspect, arm)]
        assert row.kept.value == len(kept)
        if held.failed_trials is None:
            assert row.failed is None and row.unknown, "a bare input does not count the run's failures"
            continue
        failed = sum(r.failed for r in held.failed_trials if (r.aspect, r.arm) == (row.aspect, arm))
        assert row.failed is not None and row.failed.value == failed >= len(kept)
        assert not row.unknown


@pytest.mark.parametrize("name", sorted(BUNDLES))
def test_a_slice_kept_from_more_failures_than_it_shows_says_so(name: str) -> None:
    held = _counted(name)
    rows = examples_list(held, {}, limit=200).kept_from
    short = [row for row in rows if row.failed is not None and row.failed.value > row.kept.value]
    expected = {
        (row.aspect, row.arm)
        for row in held.failed_trials or ()
        if sum(r.failed for r in held.failed_trials or () if (r.aspect, r.arm) == (row.aspect, row.arm))
        > sum((e.aspect, e.arm) == (row.aspect, row.arm) for e in held.examples)
    }
    assert {(row.aspect, str(row.arm.value)) for row in short} == expected
    assert expected, "every bundle fixture keeps fewer examples than some slice failed"


def test_a_stratum_filter_counts_within_the_stratum() -> None:
    held = _counted(KAIRO)
    example = next(e for e in held.examples if e.stratum is not None)
    rows = examples_list(held, {}, stratum_filter=example.stratum, limit=200).kept_from
    assert rows and all(row.stratum == example.stratum for row in rows)
    for row in rows:
        arm = str(row.arm.value)
        kept = sum((e.aspect, e.arm, e.stratum) == (row.aspect, arm, example.stratum) for e in held.examples)
        failed = sum(
            r.failed
            for r in held.failed_trials or ()
            if (r.aspect, r.arm, r.stratum) == (row.aspect, arm, example.stratum)
        )
        assert row.kept.value == kept
        assert row.failed is not None and row.failed.value == failed


@pytest.mark.parametrize("name", sorted(BUNDLES))
def test_the_campaign_counts_read_every_kept_example_and_every_failure(name: str) -> None:
    held = _counted(name)
    assert resolve("count.examples", held).value == len(held.examples)
    assert resolve("count.failures", held).value == sum(row.failed for row in held.failed_trials or ())


@pytest.mark.parametrize("name", sorted(fixture_inputs()))
def test_a_bare_input_counts_its_examples_and_refuses_its_failures_as_not_known(name: str) -> None:
    held = fixture_inputs()[name]
    assert resolve("count.examples", held).value == len(held.examples)
    with pytest.raises(UnstatedFigure):
        resolve("count.failures", held)


def test_each_example_carries_the_start_of_what_its_case_asked() -> None:
    bundle = kairo_bundle()
    held = _counted(KAIRO)
    trials = {(aspect.aspect, trial.trial_id): trial for aspect in bundle.aspects for trial in aspect.trials}
    named = bundle.trial_ids()
    assert any(example.input for example in held.examples), "Kairo's trials record what each case asked"
    for example in held.examples:
        trial = trials[named[example.trial or ""]]
        if not trial.first_input:
            assert example.input is None
            continue
        assert example.input is not None and len(example.input) <= EXAMPLE_EXCERPT_CHARS
        assert trial.first_input.startswith(example.input[:20])


@pytest.mark.parametrize("name", sorted(BUNDLES))
def test_a_planners_input_counts_nothing_and_its_view_is_unchanged(name: str) -> None:
    built = report_input(BUNDLES[name])
    assert built.failed_trials is None and built.case_overlap is None
    assert all(example.input is None for example in built.examples)
    assert "failed_trials" not in planner_view(built) and "case_overlap" not in planner_view(built)


@pytest.mark.parametrize("name", sorted(BUNDLES))
def test_which_cases_each_two_aspects_ran_is_counted_off_the_bundles_case_ids(name: str) -> None:
    bundle, held = BUNDLES[name], _counted(name)
    cases = {aspect.aspect: {trial.case_id for trial in aspect.trials} for aspect in bundle.aspects}
    pairs = list(combinations([aspect.aspect for aspect in bundle.aspects], 2))
    assert [row.aspects for row in held.case_overlap or ()] == pairs
    for first, second in pairs:
        shared, only_first, only_second = (
            len(cases[first] & cases[second]),
            len(cases[first] - cases[second]),
            len(cases[second] - cases[first]),
        )
        assert resolve(f"overlap.{first}.{second}.shared", held).value == shared
        assert resolve(f"overlap.{first}.{second}.only_first", held).value == only_first
        assert resolve(f"overlap.{second}.{first}.only_first", held).value == only_second
        word = "same" if not only_first and not only_second else "apart" if not shared else "partly"
        assert resolve(f"overlap.{first}.{second}.same", held).value == OVERLAP_WORDS[word]


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_the_overview_says_whether_each_aspect_ran_the_same_cases_as_each_later_one(name: str) -> None:
    held = INPUTS[name]
    overview = campaign_overview(held, "snapshot", limit=10_000)
    ids = [aspect.id for aspect in held.aspects]
    for row in overview.aspects:
        later = ids[ids.index(row.aspect) + 1 :]
        assert [figure.key for figure in row.same_cases] == [f"overlap.{row.aspect}.{other}.same" for other in later]
        if held.case_overlap is None:
            assert all(figure.value == OVERLAP_WORDS["unknown"] for figure in row.same_cases)


#: A query of every example the input holds.
_ALL_EXAMPLES = ChartQuery.model_validate(
    {
        "source": "examples",
        "aspect": "",
        "measures": [],
        "arms": [],
        "strata": [],
        "limit": 0,
        "transform": "none",
        "small_multiples_by": "none",
    }
)


@pytest.mark.parametrize("name", sorted(BUNDLES))
def test_an_example_card_names_its_aspect_where_the_input_holds_more_than_one(name: str) -> None:
    held = _counted(name)
    rendered = block_render(held, block_kind="examples", query=_ALL_EXAMPLES, limit=200)
    for row, example in zip(rendered.rows, held.examples, strict=False):
        meta = row.cells[0].split(" · ")
        aspect = reader_name("aspect", example.aspect or "", held)
        assert (aspect in meta) == (len(held.aspects) > 1), (meta, aspect)
