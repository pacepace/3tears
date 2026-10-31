"""The report input built from a campaign's bundle, and the planner's budgeted view of it.

``report_input(bundle)`` ports the report kit's adapter onto the bundle's typed objects, and ``planner_view(input)``
is what the planner reads. Contracts held here, on the four shape campaigns (``report_shapes_support``) and on the
kit's real Kairo input (``fixtures/report_plans/r5/input.json``):

- the input validates, as the model and against the docs schema, for a change, a choice and a measurement;
- nothing the bundle holds is silently dropped: every arm, reading, pair reading, contrast (one with no interval, or
  untested, with its reason), contrast family, floor, bar, measured figure and caveat reaches the input, counted against the bundle;
- figures are copied, never recomputed: a reading's change and interval, a contrast's delta, a pair's change;
- a shown trial's excerpt reads as the reply did, its JSON escapes decoded;
- a wider example pool shows up to ``per_slice`` failed trials per aspect, arm and stratum, every slice's first before
  any slice's second and a slice's other cases before a repeat, its first round the planner's examples; and a
  failure excerpt is centred on the passage the judge's reason quotes, else is the text's end, each cut end marked;
- the planner's view stays within its budget, trims in the design's order and says what it cut, never cuts the arms,
  readings, floors, caveats or the asked aspects' and judged dimensions' contrasts, and refuses when what it may not
  cut is over the budget alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from packages.evals.tests.fixtures.report_plans.dataset import kairo_bundle
from packages.evals.tests.report_kit_support import KIT
from packages.evals.tests.report_shapes_support import SHAPES, Shape
from threetears.evals.analysis.bundle.schema import LateCriterion as BundleLateCriterion
from threetears.evals.analysis.campaign_bundle import (
    CAMPAIGN_MESSAGE_CHARS_PER_ASPECT,
    CampaignBundle,
    build_campaign_bundle,
)
from threetears.evals.analysis.report_input import Example, ReportInput, Stat, Stratum
from threetears.evals.analysis.report_input_build import EXAMPLE_EXCERPT_CHARS, planner_view, report_input
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.kernel.errors import RefusedInput

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "report_plans"
SCHEMA = KIT / "schema" / "report-input.schema.json"


def _bundle(shape: Shape) -> CampaignBundle:
    return build_campaign_bundle(shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings)


@pytest.fixture(scope="module")
def bundles() -> dict[str, CampaignBundle]:
    return {name: _bundle(make()) for name, make in SHAPES.items()}


def _view(text: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(text)
    return loaded


def _kairo() -> ReportInput:
    return ReportInput.model_validate(json.loads((FIXTURES / "r5" / "input.json").read_text(encoding="utf-8")))


# --- the input carries what the bundle holds ----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SHAPES))
def test_the_input_is_a_report_input(bundles: dict[str, CampaignBundle], name: str) -> None:
    built = report_input(bundles[name])
    document = json.loads(built.model_dump_json(by_alias=True, exclude_none=True))
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(document)
    assert ReportInput.model_validate(document) == built
    assert built.meta.kind == bundles[name].kind
    assert built.meta.synthetic is False


@pytest.mark.parametrize("name", list(SHAPES))
def test_every_arm_and_reading_reaches_the_input(bundles: dict[str, CampaignBundle], name: str) -> None:
    bundle = bundles[name]
    built = report_input(bundle)
    assert [arm.id for arm in built.arms][: len(bundle.arms)] == [arm.label for arm in bundle.arms]
    assert {arm.id for arm in built.arms if arm.role == "control"} == {
        arm.label for arm in bundle.arms if arm.role == "control"
    }
    assert [aspect.id for aspect in built.aspects] == bundle.labels
    readings = {(aspect.id, reading.arm): reading for aspect in built.aspects for reading in aspect.readings}
    assert len(readings) == len(bundle.readings.readings)
    for row in bundle.readings.readings:
        copied = readings[row.aspect, row.arm_label]
        assert (copied.reading, copied.change, copied.within_interval, copied.gain_interval) == (
            row.reading,
            row.change,
            row.within_interval,
            row.gain_interval,
        )
        assert copied.cases == [row.control_cases, row.arm_cases]
    pairs = [(aspect.id, pair) for aspect in built.aspects for pair in aspect.pairs]
    assert len(pairs) == len(bundle.readings.pairs)
    for (aspect, pair), row in zip(pairs, bundle.readings.pairs, strict=True):
        assert (aspect, pair.arms, pair.change, pair.interval) == (
            row.aspect,
            (row.first_label, row.second_label),
            row.change,
            row.interval,
        )
    assert len(built.floors) == len(bundle.readings.floors)
    assert [(floor.arm, floor.status) for floor in built.floors] == [
        (status.arm_label, status.status) for status in bundle.readings.floors
    ]
    assert len(built.bars) == len(bundle.readings.bars)


@pytest.mark.parametrize("name", list(SHAPES))
def test_every_contrast_reaches_the_input_an_untested_one_with_its_reason(
    bundles: dict[str, CampaignBundle], name: str
) -> None:
    bundle = bundles[name]
    built = report_input(bundle)
    held = [
        (aspect.aspect, comparison)
        for aspect in bundle.aspects
        for family in aspect.multiple_comparisons.families
        for comparison in family.comparisons
    ]
    assert len(built.contrasts) == len(held)
    assert len(built.contrast_families) == sum(len(aspect.multiple_comparisons.families) for aspect in bundle.aspects)
    copied = {(c.aspect, c.arm, c.measure, c.kind): c for c in built.contrasts}
    for aspect, comparison in held:
        contrast = copied[aspect, bundle.label_of(comparison.contrast.variant_key), comparison.name, comparison.reading]
        assert (contrast.verdict, contrast.delta, contrast.interval, contrast.p_adjusted) == (
            comparison.verdict,
            comparison.delta,
            comparison.interval,
            comparison.p_adjusted,
        )
        if comparison.delta is None:
            assert contrast.reason == comparison.untested_reason
    for contrast in built.contrasts:
        family = next(f for f in built.contrast_families if f.aspect == contrast.aspect)
        assert contrast.level == family.level


def test_a_contrast_with_no_interval_is_carried(bundles: dict[str, CampaignBundle]) -> None:
    unbounded = [c for c in report_input(bundles["s4"]).contrasts if c.interval is None]
    assert [(c.measure, c.verdict, c.delta) for c in unbounded] == [
        ("production_replicating_cost", "not_separated", 0.0)
    ]


def test_a_measurements_readings_reach_its_measures(bundles: dict[str, CampaignBundle]) -> None:
    bundle = bundles["s3"]
    built = report_input(bundle)
    lead = built.aspects[0].id
    assert bundle.readings.measures
    for measured in bundle.readings.measures:
        measure = built.measures[measured.measure]
        figures = measure.per_arm if measured.aspect == lead else measure.by_aspect[measured.aspect]
        assert figures[measured.arm_label].mean == measured.value
    assert set(built.judged) == {"answer.clear", "answer.correct"}
    assert built.primary_judged == "answer.correct"


@pytest.mark.parametrize("name", list(SHAPES))
def test_a_measure_whose_figures_carry_intervals_states_the_level_they_are_read_at(
    bundles: dict[str, CampaignBundle], name: str
) -> None:
    built = report_input(bundles[name])
    stated = 0
    for key, measure in built.measures.items():
        figures = [*measure.per_arm.values(), *(stat for arms in measure.by_aspect.values() for stat in arms.values())]
        if any(stat.ci is not None for stat in figures):
            assert measure.level == f"{100 * INTERVAL_LEVEL:g}%", key
            stated += 1
        else:
            assert measure.level is None, key
    assert stated, "the shape carries a measure with intervals"


def test_every_other_aspects_figures_sit_beside_the_lead_aspects(bundles: dict[str, CampaignBundle]) -> None:
    bundle = bundles["s1"]
    built = report_input(bundle)
    lead, *others = built.aspects
    assert lead.scoreboard is None and all(aspect.scoreboard for aspect in others)
    assert set(built.scoreboard) == {bundle.label_of(row.variant_key) for row in bundle.aspects[0].scoreboard}
    for aspect in others:
        assert aspect.scoreboard is not None
        held = bundle.aspect(aspect.id).scoreboard
        assert {label: (row.ran, row.passed) for label, row in aspect.scoreboard.items()} == {
            bundle.label_of(row.variant_key): (row.ran, row.passed) for row in held
        }
        assert all(aspect.id in measure.by_aspect for measure in built.measures.values() if measure.per_arm)


def test_a_choice_declares_its_bars_as_criteria(bundles: dict[str, CampaignBundle]) -> None:
    built = report_input(bundles["s2"])
    assert [(c.kind, c.measure, c.direction, c.threshold) for c in built.criteria] == [
        ("bar", "cost_usd", "lower_is_better", 0.001),
        ("bar", "total_ms", "lower_is_better", 800.0),
    ]
    assert all(arm.role == "challenger" for arm in built.arms)


def test_every_caveat_the_bundle_holds_reaches_the_input(bundles: dict[str, CampaignBundle]) -> None:
    bundle = bundles["s1"]
    first = bundle.aspects[0]
    run = first.design.contrasts[0].run_ids[0]
    late = BundleLateCriterion(criterion="margin", name=None, set_at="2026-10-02T09:00:00+00:00")
    changed = first.model_copy(
        update={
            "short_runs": {run: "ran short: 19 of 20 cases produced a usable measurement"},
            "reading_scope": first.reading_scope.model_copy(update={"late_criteria": [late]}),
        }
    )
    built = report_input(bundle.model_copy(update={"aspects": [changed, *bundle.aspects[1:]]}))
    caveats = built.caveats
    assert [(s.run, s.arm, s.note) for s in caveats.short_runs] == [
        (run, bundle.label_of(first.design.contrasts[0].variant_key), changed.short_runs[run])
    ]
    assert [(c.criterion, c.set_at) for c in caveats.late_criteria] == [("margin", late.page_date())]
    apparatus = {confound.dimension for aspect in bundle.aspects for confound in aspect.apparatus_confounds}
    assert {confound.dimension for confound in caveats.apparatus_confounds} == apparatus
    assert apparatus <= set(caveats.confounds)
    tiers = {tier.rubric_dim for aspect in bundle.aspects for tier in aspect.judge_evidence_tiers}
    assert set(caveats.judge_tiers) == tiers
    unreported = {o for aspect in bundle.aspects for o in aspect.telemetry.measures.unreported_observations}
    assert unreported <= set(caveats.unreported)


def test_examples_show_the_candidates_failures_first(bundles: dict[str, CampaignBundle]) -> None:
    built = report_input(bundles["s1"])
    control = next(arm.id for arm in built.arms if arm.role == "control")
    arms = [example.arm for example in built.examples]
    first_control = arms.index(control) if control in arms else len(arms)
    assert arms[0] != control and all(arm == control for arm in arms[first_control:])
    assert all(example.reason for example in built.examples)
    assert len({(e.aspect, e.arm, e.stratum, e.outcome) for e in built.examples}) == len(built.examples)


def test_an_examples_excerpt_reads_as_the_reply_did(bundles: dict[str, CampaignBundle]) -> None:
    bundle = bundles["s3"]
    first = bundle.aspects[0]
    failed = next(trial for trial in first.trials if trial.judged.get("answer.correct") == 0)
    escaped = 'queued: {"content": "dead boiler \\u2014 landlord\'s job\\nkeep \\"replies\\""}'
    trials = [
        trial.model_copy(update={"excerpt": escaped}) if trial.trial_id == failed.trial_id else trial
        for trial in first.trials
    ]
    changed = bundle.model_copy(update={"aspects": [first.model_copy(update={"trials": trials}), *bundle.aspects[1:]]})
    shown = next(e for e in report_input(changed).examples if e.case == failed.case_label and e.aspect == first.aspect)
    assert shown.excerpt == 'queued: {"content": "dead boiler — landlord\'s job\nkeep "replies""}'


def _failed_by_slice(bundle: CampaignBundle, primary: str | None) -> dict[tuple[str, str, str | None], list[str]]:
    """Each slice's failed trials' cases, by (aspect, arm, stratum): failed on the primary judged dimension."""
    held: dict[tuple[str, str, str | None], list[str]] = {}
    arms = {key: arm.label for arm in bundle.arms for key in bundle.arm_keys(arm)}
    for aspect in bundle.aspects:
        for trial in aspect.trials:
            if primary is not None and trial.judged.get(primary) == 0:
                held.setdefault((aspect.aspect, arms.get(trial.arm, trial.arm), trial.stratum), []).append(
                    trial.case_label
                )
    return held


def _turns(examples: list[Example]) -> list[int]:
    """Each example's turn in its slice: how many examples before it share its aspect, arm and stratum."""
    seen: dict[tuple[str | None, str | None, str | None], int] = {}
    turns: list[int] = []
    for example in examples:
        key = (example.aspect, example.arm, example.stratum)
        turns.append(seen.get(key, 0))
        seen[key] = turns[-1] + 1
    return turns


def test_a_wider_pool_shows_up_to_per_slice_failures_per_slice_each_slices_first_before_any_second() -> None:
    bundle = kairo_bundle()
    planner = report_input(bundle, examples=1000)
    wide = report_input(bundle, examples=1000, per_slice=3)
    failed = _failed_by_slice(bundle, wide.primary_judged)
    assert any(len(cases) > 1 for cases in failed.values()), "the control: a slice holds more than one failure"
    shown: dict[tuple[str | None, str | None, str | None], list[str | None]] = {}
    for example in wide.examples:
        shown.setdefault((example.aspect, example.arm, example.stratum), []).append(example.case)
    assert {key: len(cases) for key, cases in shown.items()} == {
        key: min(len(cases), 3) for key, cases in failed.items()
    }
    for key, cases in shown.items():
        distinct = len(set(failed[(str(key[0]), str(key[1]), key[2])]))
        assert len(set(cases)) == min(len(cases), distinct), "a slice's other cases come before a repeat"
    turns = _turns(wide.examples)
    assert turns == sorted(turns), "every slice's first before any slice's second"
    firsts = [example for example, turn in zip(wide.examples, turns, strict=True) if turn == 0]
    assert [(e.trial, e.case, e.arm, e.stratum) for e in firsts] == [
        (e.trial, e.case, e.arm, e.stratum) for e in planner.examples
    ], "the first round is the planner's pool"
    capped = report_input(bundle, examples=len(firsts) + 2, per_slice=3)
    assert len(capped.examples) == len(firsts) + 2 and _turns(capped.examples)[-1] == 1


def _with_excerpt(bundle: CampaignBundle, text: str, reason: str) -> tuple[CampaignBundle, str]:
    """The bundle with one failed trial's excerpt and judge's reason replaced, and that trial's case."""
    first = bundle.aspects[0]
    failed = next(trial for trial in first.trials if trial.judged.get("answer.correct") == 0)
    judge = [judged.model_copy(update={"reason": reason}) for judged in failed.judge]
    trials = [
        trial.model_copy(update={"excerpt": text, "judge": judge}) if trial.trial_id == failed.trial_id else trial
        for trial in first.trials
    ]
    aspects = [first.model_copy(update={"trials": trials}), *bundle.aspects[1:]]
    return bundle.model_copy(update={"aspects": aspects}), failed.case_label


def _shown(bundle: CampaignBundle, case: str) -> str:
    built = report_input(bundle, examples=1000, per_slice=3, excerpt="failure")
    excerpt = next(e.excerpt for e in built.examples if e.case == case and e.aspect == bundle.aspects[0].aspect)
    assert excerpt is not None
    return excerpt


def test_a_failure_excerpt_is_centred_on_the_passage_the_reason_quotes(bundles: dict[str, CampaignBundle]) -> None:
    passage = "I'd rather tell you that than make up a synth name"
    before, after = "tool output line\n" * 80, "queued the next record\n" * 80
    reason = f"It fails: \u201c{passage.replace(chr(39), chr(0x2019)).upper()},\u201d which breaks the rule."
    changed, case = _with_excerpt(bundles["s3"], before + passage + after, json.loads(f'"{reason}"'))
    excerpt = _shown(changed, case)
    assert len(excerpt) <= EXAMPLE_EXCERPT_CHARS and passage in excerpt
    assert excerpt.startswith("…") and excerpt.endswith("…")
    left, right = excerpt[1:-1].split(passage)
    assert abs(len(left) - len(right)) <= 1
    head = report_input(changed, examples=1000).examples
    assert all(passage not in (e.excerpt or "") for e in head), "the control: the start of the text misses it"


def test_a_failure_excerpt_with_no_quote_found_is_the_texts_end_and_a_short_one_is_whole(
    bundles: dict[str, CampaignBundle],
) -> None:
    text = "".join(f"line {index} of the transcript\n" for index in range(200))
    changed, case = _with_excerpt(bundles["s3"], text, "It fails: \u201cthis passage appears nowhere at all\u201d.")
    excerpt = _shown(changed, case)
    assert len(excerpt) == EXAMPLE_EXCERPT_CHARS and excerpt.startswith("…") and text.endswith(excerpt[1:])
    short, case = _with_excerpt(bundles["s3"], "a short reply", "It fails: \u201ca short reply\u201d.")
    assert _shown(short, case) == "a short reply"


# --- the planner's view -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SHAPES))
def test_the_planners_view_of_a_shape_fits_its_budget_whole(bundles: dict[str, CampaignBundle], name: str) -> None:
    built = report_input(bundles[name])
    text = planner_view(built)
    assert len(text) <= CAMPAIGN_MESSAGE_CHARS_PER_ASPECT * len(built.aspects)
    view = _view(text)
    assert "trimmed" not in view
    assert len(view["contrasts"]) == len(built.contrasts)
    assert len(view["examples"]) == len(built.examples)


def _protected(report: ReportInput, view: dict[str, Any]) -> None:
    """The view keeps every arm, reading, floor and caveat, and every contrast on an asked aspect or judged dimension."""
    assert [arm["id"] for arm in view["arms"]] == [arm.id for arm in report.arms]
    assert [[r["arm"] for r in a["readings"]] for a in view["aspects"]] == [
        [r.arm for r in a.readings] for a in report.aspects
    ]
    assert len(view.get("floors", [])) == len(report.floors)
    assert len(view["caveats"].get("late_criteria", [])) == len(report.caveats.late_criteria)
    assert len(view["caveats"].get("short_runs", [])) == len(report.caveats.short_runs)
    lead = report.aspects[0].id
    asked = {aspect.id for aspect in report.aspects if aspect.asked} or {lead}
    kept = {(c.get("aspect", lead), c["arm"], c["measure"], c.get("kind")) for c in view["contrasts"]}
    for contrast in report.contrasts:
        where = contrast.aspect or lead
        if contrast.kind == "judged" or where in asked:
            assert (where, contrast.arm, contrast.measure, contrast.kind) in kept


def test_the_planners_view_of_kairo_fits_its_budget_and_says_what_it_cut() -> None:
    report = _kairo()
    text = planner_view(report)
    assert len(text) <= CAMPAIGN_MESSAGE_CHARS_PER_ASPECT
    view = _view(text)
    note = view["trimmed"]
    assert "examples past the first 4" in note and "strata past the 12 largest" in note
    assert len(view["examples"]) == 4 and all(len(e.get("excerpt", "")) <= 200 for e in view["examples"])
    assert [s["id"] for s in view["strata"]] == [s.id for s in report.strata[:12]]
    assert view["strata"][0]["id"] == max(report.strata, key=lambda s: s.n_cases or 0).id
    _protected(report, view)


def _inflated(report: ReportInput) -> ReportInput:
    """``report`` with many long examples and strata, and every aspect but the first no longer asked."""
    arms = [arm.id for arm in report.arms]
    examples = [
        Example(trial=f"T{i}", case=f"case-{i}", arm=arms[-1], reason="r" * 150, excerpt="x" * 400) for i in range(30)
    ]
    strata = [
        Stratum(
            id=f"stratum-{i}",
            n_cases=100 - i,
            per_arm={arm: {"a.one": Stat(mean=0.5, n=9), "a.two": Stat(mean=3.5, n=9)} for arm in arms},
        )
        for i in range(40)
    ]
    aspects = [report.aspects[0], *(aspect.model_copy(update={"asked": False}) for aspect in report.aspects[1:])]
    return report.model_copy(
        update={"examples": examples, "strata": strata, "aspects": aspects, "primary_judged": "a.one"}
    )


def test_the_planners_view_trims_in_order_within_budget(bundles: dict[str, CampaignBundle]) -> None:
    report = _inflated(report_input(bundles["s1"]))
    whole = len(planner_view(report, budget=10**9))
    steps = []
    for budget in (whole - 2_000, whole - 14_000, whole - 23_000, whole - 30_000, whole - 32_000):
        text = planner_view(report, budget=budget)
        assert len(text) <= budget
        view = _view(text)
        _protected(report, view)
        steps.append(view)
    notes = [view["trimmed"] for view in steps]
    assert "excerpts" in notes[0] and "examples past" not in notes[0]
    assert "examples past the first 4" in notes[1] and "strata past" not in notes[1]
    assert "strata past the 12 largest" in notes[2] and "contrasts outside" not in notes[2]
    assert len(steps[2]["strata"]) >= 12
    assert "contrasts outside the asked aspects" in notes[3]
    assert {c["aspect"] for c in steps[3]["contrasts"]} >= {report.aspects[0].id}
    assert "strata's figures on anything but a.one" in notes[4]
    assert all(set(figures) == {"a.one"} for figures in steps[4]["strata"][-1]["per_arm"].values())


def test_the_planners_view_refuses_what_it_may_not_cut(bundles: dict[str, CampaignBundle]) -> None:
    report = report_input(bundles["s1"])
    with pytest.raises(RefusedInput, match="never trimmed"):
        planner_view(report, budget=5_000)
