"""An analyst's examples show the failure, every dimension's reason and the case's input (engine item 5).

Discodon reported three faults in the report input's examples: the excerpt was the head of a 2,000-character tail,
only the primary dimension's reason was kept, and ``Example.input`` / ``expected`` were never filled. Contracts held
here, on the analyst's input (``analyst_input``, what every snapshot is frozen from):

- **the excerpt is cut around the failure**, centred on the passage the reason quotes, never the head of the tail;
- **every judged dimension's reason** rides on the example (``Example.reasons``, by dimension, the primary's among
  them), and the analyst's example row shows them; ``reason`` stays the primary's;
- **the case's input is filled** wherever the trial has one; ``expected`` stays unset, since no trial the bundle holds
  carries an expected answer;
- **a planner's input is unchanged**: it carries no ``reasons`` key, so its view and its digest are as they were.
"""

from __future__ import annotations

from packages.evals.tests.fixtures.report_plans.dataset import kairo_bundle
from threetears.evals.analysis.analyst.facts import examples_list
from threetears.evals.analysis.analyst.snapshot import analyst_input
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, TrialRow
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.report_input import Example, ReportInput
from threetears.evals.analysis.report_input_build import EXAMPLE_EXCERPT_CHARS, report_input

SNAPSHOT = "report-snapshot:" + "0" * 32


def _trial_of(bundle: CampaignBundle, example: Example) -> TrialRow:
    """The bundle's trial an example shows, by the trial id the input names it with."""
    assert example.trial is not None
    where, trial_id = bundle.trial_ids()[example.trial]
    aspect: AnalysisContextBundle = next(held for held in bundle.aspects if held.aspect == where)
    return next(trial for trial in aspect.trials if trial.trial_id == trial_id)


def _planted(bundle: CampaignBundle, text: str, reason: str) -> tuple[CampaignBundle, TrialRow]:
    """The bundle with the first shown example's excerpt replaced, and every dimension's reason set to ``reason``."""
    shown = _trial_of(bundle, analyst_input(bundle).examples[0])
    planted = shown.model_copy(
        update={"excerpt": text, "judge": [judged.model_copy(update={"reason": reason}) for judged in shown.judge]}
    )
    aspects = [
        aspect.model_copy(update={"trials": [planted if trial is shown else trial for trial in aspect.trials]})
        for aspect in bundle.aspects
    ]
    return bundle.model_copy(update={"aspects": aspects}), planted


def _shown(bundle: CampaignBundle, held: ReportInput, trial: TrialRow) -> Example:
    return next(example for example in held.examples if _trial_of(bundle, example).trial_id == trial.trial_id)


def test_the_excerpt_is_cut_around_the_failure_never_the_head_of_the_tail() -> None:
    passage = "do I look like a man to you, Jules"
    text = "queued a track and searched again\n" * 40 + passage + "\nplayed the next record" * 40
    bundle, trial = _planted(kairo_bundle(), text, f"It fails: “{passage}” is out of character.")

    example = _shown(bundle, analyst_input(bundle), trial)

    assert example.excerpt is not None and passage in example.excerpt
    assert len(example.excerpt) <= EXAMPLE_EXCERPT_CHARS
    assert not example.excerpt.startswith(text[:40]), "not the head of the tail"


def test_every_judged_dimensions_reason_rides_on_the_example_the_primarys_among_them() -> None:
    bundle = kairo_bundle()
    held = analyst_input(bundle)
    assert held.examples
    for example in held.examples:
        trial = _trial_of(bundle, example)
        given = {judged.dim: judged.reason.strip() for judged in trial.judge if judged.reason.strip()}
        assert example.reasons == given, example.trial
        assert len(given) > 1, "Kairo's trials are judged on several dimensions"
        assert example.reason in given.values(), "reason stays the primary dimension's"


def test_the_analysts_example_row_shows_every_dimensions_reason() -> None:
    held = analyst_input(kairo_bundle())
    rows = examples_list(held, {}, limit=len(held.examples)).examples
    assert rows and all(
        row.reasons == (example.reasons or {}) for row, example in zip(rows, held.examples, strict=True)
    )


def test_the_cases_input_is_filled_where_the_trial_has_one_and_expected_stays_unset() -> None:
    bundle = kairo_bundle()
    held = analyst_input(bundle)
    for example in held.examples:
        trial = _trial_of(bundle, example)
        assert bool(example.input) == bool(trial.first_input.strip()), example.trial
        assert example.expected is None, "no trial the bundle holds carries an expected answer"
    assert any(example.input for example in held.examples)


def test_a_planners_input_carries_no_reasons_so_its_digest_is_as_it_was() -> None:
    held = report_input(kairo_bundle())
    assert held.examples and all(example.reasons is None for example in held.examples)
    assert '"reasons"' not in held.model_dump_json(by_alias=True)
