"""Multiple comparisons: each declared question's family is corrected as one, and verdicts read the adjusted p.

Ten comparisons at α=0.05 find a chance "difference" more often than not, so the bundle tests each
contrast against the control on every reading a live question asks about, Holm-corrects inside the
question's family, and reads each verdict off the adjusted p. The writer is shown the adjusted p only.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``holm_adjust``: dropping the running maximum (adjusted p no longer monotone); multiplying by ``m``
  for every rank (plain Bonferroni); removing the out-of-range refusal.
- ``_multiple_comparisons``: reading the verdict off ``p_raw`` instead of ``p_adjusted``; correcting
  over each comparison alone (``holm_adjust([p])``); dropping the control-variant filter on pairs.
- ``build_user_message``: removing the ``p_raw`` deletion.
- ``_compare`` (through the bundle): disabling the paired no-spread branch, so a deterministic gap is named as a
  shortage of cases; reading the gap by the t-test's refusal instead of ``separation_p``, so a constant
  shift over twelve cases reads untested.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import pytest

from threetears.evals.analysis import (
    EVAL_ANALYSIS_GEN_DEFAULT,
    AnalysisContextBundle,
    ComparisonFamily,
    assemble_context_bundle,
)
from threetears.evals.analysis.generator import build_user_message
from threetears.evals.analysis.stats import composite_significance, holm_adjust, separation_p
from threetears.evals.contracts import EvalCampaign, EvalResult, Question, RubricScore
from threetears.evals.contracts.models import LatencyMetrics
from threetears.evals.contracts.host import HostProfile, MeasureRegistry
from threetears.evals.contracts.metrics import MeritAxis
from packages.evals.tests.factories import fixture_variant_key, make_eval_result, make_eval_run, minimal_declaration
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile

#: Per-case score differences, contrast minus control, over twelve cases. Raw paired p ≈ 0.026: a
#: separation at α=0.05 when tested alone, and not once nine other comparisons share its family.
BORDERLINE = (1, 1, 1, 1, 1, 1, 0, 0, -1, 1, 0, 0)

#: A difference of zero mean with spread: raw p = 1.
NOISE = (1, -1) * 6

#: A clear improvement with spread: survives any correction a family of ten applies.
CLEAR = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1)

CONTROL = "control-model"
CONTRAST = "contrast-model"


def _dims(n: int) -> list[str]:
    return [f"extraction.d{index}" for index in range(n)]


def _bundle(
    differences: Sequence[Sequence[int]],
    *,
    merit_axes: list[MeritAxis] | None = None,
    control: str | None = CONTROL,
    questions: bool = True,
    cases: int = 12,
    matched: tuple[Sequence[bool], Sequence[bool]] | None = None,
    accuracy: tuple[Sequence[float], Sequence[float]] | None = None,
    profile: HostProfile | None = None,
    rigs: Sequence[str] = ("",),
    contrast_cases: Sequence[int] | None = None,
    unpaired: int | None = None,
    apparatus: bool = False,
) -> AnalysisContextBundle:
    """A control and one contrast, scored on one judged dimension per entry of ``differences``, on each of ``rigs``.

    The control scores 3 on every dimension of every case; the contrast scores ``3 + difference``.
    Results carry no goal check and no cost, so the judged dimensions are the whole family — unless
    ``matched`` gives each side's per-case classifier verdicts, landed as ``match``, or ``accuracy`` each side's
    per-case ``field_accuracy``. ``contrast_cases`` names the cases the contrast ran (default: all of them);
    ``unpaired`` instead gives the contrast that many cases of its own, none shared with the control, so the
    comparison is unpaired.
    ``apparatus`` gives every result a judge phase and a blended spend that differ sharply between the arms —
    the rig's own readings, which no family may test.
    """
    dims = _dims(len(differences))
    runs = []
    results: dict[str, list[EvalResult]] = {}
    for model, rig in ((model, rig) for model in (CONTROL, CONTRAST) for rig in rigs):
        # A rig is the reviewer pool the toy host records as apparatus; "" leaves the run on the default one.
        payload = {"toyhost": {"reviewer_pool": rig}} if rig else {}
        run = make_eval_run(status="completed", candidate_model=model, host_payload=payload)
        runs.append(run)
        results[run.id] = [
            make_eval_result(
                id=f"{model}-{rig}-{case}",
                eval_run_id=run.id,
                scope_id=run.scope_id,
                model=model,
                test_case_id=f"tc-u{case:02d}" if unpaired is not None and model == CONTRAST else f"tc-{case:02d}",
                goal_state_outcomes=[],
                cost_usd=(0.5 if model == CONTRAST else 0.1) + 0.01 * case if apparatus else None,
                latency=(
                    LatencyMetrics(judge_ms=(9000.0 if model == CONTRAST else 1000.0) + 10.0 * case)
                    if apparatus
                    else None
                ),
                host_measures=(
                    {"match": matched[model == CONTRAST][case]}
                    if matched is not None
                    else {"field_accuracy": accuracy[model == CONTRAST][case]}
                    if accuracy is not None
                    else {}
                ),
                rubric_scores=[
                    RubricScore(dim=dim, score=3 + (diffs[case] if model == CONTRAST else 0), scale="ordinal")
                    for dim, diffs in zip(dims, differences, strict=True)
                ],
            )
            for case in range(cases if unpaired is None or model == CONTROL else unpaired)
            if model == CONTROL or contrast_cases is None or case in contrast_cases
        ]
    declaration = minimal_declaration(control=fixture_variant_key(control) if control else None)
    if questions:
        declaration = declaration.model_copy(
            update={
                "questions": [
                    Question(id="q-better", text="is the contrast better?", merit_axes=merit_axes or ["quality"])
                ]
            }
        )
    campaign = EvalCampaign(
        scope_id=runs[0].scope_id,
        name="family of comparisons",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[run.id for run in runs],
        declared_design=declaration,
        created_by="test:fixture",
    )
    return assemble_context_bundle(
        campaign, storage=ToyhostStorage(runs, results), profile=profile if profile is not None else toyhost_profile()
    )


def _family(bundle: AnalysisContextBundle) -> ComparisonFamily:
    (family,) = bundle.multiple_comparisons.families
    return family


# --- the arithmetic ----------------------------------------------------------------------------------


class TestHolm:
    def test_the_worked_family(self) -> None:
        # Sorted: 0.005×4=0.02, 0.01×3=0.03, 0.03×2=0.06, 0.04×1=0.04 → raised to 0.06 by monotonicity.
        assert holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])

    def test_an_adjusted_p_is_never_below_one_ranked_beneath_it(self) -> None:
        adjusted = holm_adjust([0.04, 0.045])
        assert adjusted == pytest.approx([0.08, 0.08]), "the larger p's own product (0.045) is raised to 0.08"

    def test_it_is_capped_at_one_and_a_family_of_one_is_uncorrected(self) -> None:
        assert holm_adjust([0.6, 0.7]) == pytest.approx([1.0, 1.0])
        assert holm_adjust([0.026]) == pytest.approx([0.026])
        assert holm_adjust([]) == []

    @pytest.mark.parametrize("stray", [-0.1, 1.5, float("nan")])
    def test_a_value_that_is_not_a_probability_is_refused(self, stray: float) -> None:
        with pytest.raises(ValueError, match="must lie in"):
            holm_adjust([0.01, stray])

    def test_a_cap_on_the_true_hypotheses_caps_the_multiplier(self) -> None:
        """Shaffer's refinement: with at most two of four hypotheses true, no p is multiplied by more than two."""
        # Sorted: 0.01×min(4,2)=0.02, 0.02×min(3,2)=0.04, 0.5×2=1.0, 0.6×1=0.6 → raised to 1.0 by monotonicity.
        assert holm_adjust([0.5, 0.01, 0.6, 0.02], max_true=2) == pytest.approx([1.0, 0.02, 1.0, 0.04])
        assert holm_adjust([0.01, 0.04, 0.03, 0.005], max_true=4) == holm_adjust([0.01, 0.04, 0.03, 0.005])
        with pytest.raises(ValueError, match="max_true"):
            holm_adjust([0.01], max_true=0)


# --- the bundle --------------------------------------------------------------------------------------


class TestAFamilyOfTenCorrectsAVerdictAwayAndTheWriterSeesOnlyTheAdjustedP:
    """Done when: a family of 10 has a verdict that survives raw p and fails adjusted p, and the memo prompt sees only the adjusted figure."""

    def test_the_borderline_reading_separates_on_raw_p_and_not_on_adjusted(self) -> None:
        family = _family(_bundle([BORDERLINE, *[NOISE] * 9]))

        assert family.family_size == 10
        assert len(family.comparisons) == 10 and family.n_untested == 0
        borderline = next(c for c in family.comparisons if c.name == "extraction.d0")
        assert borderline.test == "paired" and borderline.p_raw is not None and borderline.p_adjusted is not None
        assert borderline.p_raw < 0.05, "the raw p alone would have called this a separation"
        assert borderline.p_adjusted >= 0.05
        assert borderline.p_adjusted == pytest.approx(min(1.0, 10 * borderline.p_raw))
        assert borderline.verdict == "not_separated"
        assert "10 comparisons" in family.disclosure and "Holm" in family.disclosure

    def test_the_same_reading_alone_is_a_separation_so_the_family_made_the_difference(self) -> None:
        family = _family(_bundle([BORDERLINE]))

        (alone,) = family.comparisons
        assert family.family_size == 1
        assert alone.p_adjusted == pytest.approx(alone.p_raw)
        assert alone.verdict == "improved"

    def test_the_writer_is_shown_the_adjusted_p_and_never_the_raw_one(self) -> None:
        bundle = _bundle([BORDERLINE, *[NOISE] * 9])
        borderline = next(c for c in _family(bundle).comparisons if c.name == "extraction.d0")
        assert borderline.p_raw is not None and borderline.p_adjusted is not None

        message = build_user_message(bundle)

        assert "p_raw" not in message
        assert repr(borderline.p_raw) not in message, "the raw figure must not reach the writer under any key"
        assert repr(borderline.p_adjusted) in message
        assert bundle.multiple_comparisons.families[0].comparisons[0].p_raw is not None, (
            "the raw p stays on the bundle for audit; only the writer's view drops it"
        )

    def test_the_prompt_tells_the_writer_to_quote_only_the_adjusted_p(self) -> None:
        assert "`multiple_comparisons`" in EVAL_ANALYSIS_GEN_DEFAULT
        assert "quote only its `p_adjusted`" in EVAL_ANALYSIS_GEN_DEFAULT
        assert "`family_size`" in EVAL_ANALYSIS_GEN_DEFAULT


class TestVerdicts:
    def test_a_clear_improvement_survives_a_family_of_ten(self) -> None:
        family = _family(_bundle([CLEAR, *[NOISE] * 9]))
        clear = next(c for c in family.comparisons if c.name == "extraction.d0")
        assert clear.p_adjusted is not None and clear.p_adjusted < 0.05
        assert clear.verdict == "improved"
        assert all(c.verdict == "not_separated" for c in family.comparisons if c is not clear)

    def test_a_clear_decline_is_a_regression(self) -> None:
        family = _family(_bundle([tuple(-d for d in CLEAR)]))
        (decline,) = family.comparisons
        assert decline.delta is not None and decline.delta < 0
        assert decline.verdict == "regressed"

    def test_an_untested_comparison_says_which_refusal_it_hit(self) -> None:
        # Every one of five cases moved by exactly +1: the exact sign-flip p is 2^-4 at best, above α.
        gap = _family(_bundle([(1,) * 5], cases=5))
        (moved,) = gap.comparisons
        assert (moved.verdict, moved.p_raw, moved.test, moved.hedges_g) == ("untested", None, None, None)
        assert moved.untested_reason is not None
        assert "same amount" in moved.untested_reason and "0.0625" in moved.untested_reason
        assert "needs 6 shared cases" in moved.untested_reason
        assert gap.family_size == 0 and gap.n_untested == 1

        # One case: too few on a side for any test.
        thin = _family(_bundle([(1,)], cases=1))
        (single,) = thin.comparisons
        assert single.verdict == "untested"
        assert single.untested_reason is not None and "fewer than two" in single.untested_reason

    @pytest.mark.parametrize("cases", [6, 12])
    def test_a_constant_shift_is_separated_by_the_exact_sign_flip_p(self, cases: int) -> None:
        """Every shared case moved by +1: the frontier, the mechanism reads and history call that separated, so here too."""
        family = _family(_bundle([(1,) * cases], cases=cases))
        (moved,) = family.comparisons
        assert (moved.test, moved.p_raw) == ("paired", 2.0 ** (1 - cases))
        assert moved.p_raw == separation_p([3.0] * cases, [4.0] * cases, paired=True)
        assert moved.verdict == "improved" and family.family_size == 1
        assert moved.hedges_g is None, "no spread, so no finite effect size"
        assert moved.interval is None, "the exact test has no interval to invert"

    def test_identical_values_are_tested_and_not_separated(self) -> None:
        (same,) = _family(_bundle([(0,) * 12])).comparisons
        assert (same.verdict, same.test, same.p_raw, same.hedges_g) == ("not_separated", "paired", 1.0, None)

    @pytest.mark.parametrize(("n_control", "n_contrast"), [(4, 4), (3, 5)])
    def test_two_constant_sides_unpaired_are_tested_by_the_exact_permutation_p(
        self, n_control: int, n_contrast: int
    ) -> None:
        (comparison,) = _family(
            _bundle([(1,) * max(n_control, n_contrast)], cases=n_control, unpaired=n_contrast)
        ).comparisons
        assert comparison.test == "unpaired"
        assert comparison.p_raw == pytest.approx(2 / math.comb(n_control + n_contrast, n_control))
        assert comparison.p_raw is not None and comparison.p_raw < 0.05
        assert comparison.hedges_g is None and comparison.untested_reason is None

    def test_two_constant_sides_too_few_for_the_exact_p_read_untested_and_say_why(self) -> None:
        family = _family(_bundle([(1,) * 4], cases=3, unpaired=4))
        (comparison,) = family.comparisons
        assert (comparison.verdict, comparison.test, comparison.p_raw, comparison.p_adjusted) == (
            "untested",
            None,
            None,
            None,
        )
        assert family.n_untested == 1, "an untested comparison is counted as untested, not corrected over"
        assert comparison.untested_reason is not None
        assert "each side's values are constant" in comparison.untested_reason
        assert "3 and 4 cases" in comparison.untested_reason and "0.05714" in comparison.untested_reason

    def test_comparisons_are_against_the_control_only(self) -> None:
        family = _family(_bundle([CLEAR]))
        (comparison,) = family.comparisons
        assert comparison.control.variant_key == fixture_variant_key(CONTROL)
        assert comparison.contrast.variant_key == fixture_variant_key(CONTRAST)
        assert comparison.control.n_cases == comparison.contrast.n_cases == 12


class TestWhatAFamilyCovers:
    def test_a_question_about_cost_draws_no_verdict_from_a_judged_dimension(self) -> None:
        quality = _family(_bundle([CLEAR], merit_axes=["quality"]))
        assert [c.name for c in quality.comparisons if c.reading == "judged"] == ["extraction.d0"], (
            "the positive control: a quality question does compare the judged dimension"
        )

        family = _family(_bundle([CLEAR], merit_axes=["cost"]))
        assert family.merit_axes == ["cost"]
        assert not [c for c in family.comparisons if c.reading == "judged"]

    def test_no_question_means_one_campaign_wide_family_over_every_reading(self) -> None:
        """A campaign that asked nothing is not licensed to report chance differences: every comparison is one family."""
        bundle = _bundle([CLEAR, *[NOISE] * 9], questions=False)

        family = _family(bundle)
        assert bundle.multiple_comparisons.withheld is None
        assert (family.question_id, family.merit_axes, family.family_size) == (None, [], 10)
        assert family.disclosure.startswith("This campaign declares no live question, so every comparison")

    def test_a_chance_difference_in_a_campaign_with_no_question_does_not_separate(self) -> None:
        """The Done-when case, with no question declared: raw p ≈ 0.026 separates alone and not among ten."""
        borderline = next(
            c
            for c in _family(_bundle([BORDERLINE, *[NOISE] * 9], questions=False)).comparisons
            if c.name.endswith("d0")
        )
        assert borderline.p_raw is not None and borderline.p_raw < 0.05
        assert borderline.verdict == "not_separated"
        assert borderline.p_adjusted is not None and borderline.p_adjusted >= 0.05

    def test_no_control_means_no_family(self) -> None:
        bundle = _bundle([CLEAR], control=None)
        assert bundle.multiple_comparisons.families == []
        assert bundle.multiple_comparisons.withheld is not None
        assert "No control resolved" in bundle.multiple_comparisons.withheld

    def test_the_writer_is_not_licensed_to_separate_arms_without_a_family(self) -> None:
        """The prompt once let a no-family campaign separate arms on a dimension's own sem, with no correction."""
        assert "in a campaign with no family, where it clears that dimension's own noise floor" not in (
            EVAL_ANALYSIS_GEN_DEFAULT
        )
        assert (
            "Where the bundle carries no family (`multiple_comparisons.withheld` says why), no comparison between arms is separated"
            in (EVAL_ANALYSIS_GEN_DEFAULT)
        )
        assert "with no question declared, on every merit-axis reading as one campaign-wide family" in (
            EVAL_ANALYSIS_GEN_DEFAULT
        )

    def test_a_contrast_is_paired_with_the_control_on_its_own_rig_only(self) -> None:
        """Two rigs: each contrast cell meets the control cell of its rig, never the other's."""
        family = _family(_bundle([CLEAR], rigs=("pool-a", "pool-b")))

        rigs = {c.control.apparatus_class_id for c in family.comparisons}
        assert len(rigs) == 2, "the fixture puts the cells on two rigs"
        assert all(c.control.apparatus_class_id == c.contrast.apparatus_class_id for c in family.comparisons)
        assert family.family_size == 1 * 1 * 2, "one contrast arm x one reading x two rigs"

    def test_the_bundle_test_and_the_stats_test_are_one_computation(self) -> None:
        family = _family(_bundle([BORDERLINE]))
        (comparison,) = family.comparisons
        expected = composite_significance([3.0] * 12, [3.0 + d for d in BORDERLINE], paired=True)
        assert comparison.p_raw == pytest.approx(expected.p_value)

    def test_a_classifier_s_verdict_is_one_comparison_on_the_quality_axis(self) -> None:
        """``match`` lands once and ``accuracy`` is derived from it: the family tests the verdict once, under ``accuracy``."""
        control = [True] * 6 + [False] * 6
        contrast = [True] * 11 + [False]
        family = _family(_bundle([], matched=(control, contrast), merit_axes=["quality"]))

        measured = [c for c in family.comparisons if c.reading == "measure"]
        assert [c.name for c in measured] == ["accuracy"]
        (accuracy,) = measured
        assert (accuracy.control.mean, accuracy.contrast.mean) == (0.5, pytest.approx(11 / 12))
        assert family.family_size == 1

        cost = _family(_bundle([], matched=(control, contrast), merit_axes=["cost"]))
        assert not [c for c in cost.comparisons if c.name == "accuracy"], "it sits on quality, not on every axis"


# --- materiality: a separated difference too small to act on is labelled so -----------------------------


def _accuracy_threshold(threshold: float | None) -> HostProfile:
    profile = toyhost_profile()
    measures = tuple(
        descriptor.model_copy(update={"materiality_threshold": threshold})
        if descriptor.name == "field_accuracy"
        else descriptor
        for descriptor in TOYHOST_MEASURES
    )
    return replace(profile, measures=MeasureRegistry(measures, families=profile.measures.families))


#: The contrast extracts about two points better on every case, with spread: separated after any correction.
_ACCURACY = ([0.80, 0.81, 0.79, 0.80] * 3, [0.82, 0.83, 0.82, 0.82] * 3)


@pytest.mark.parametrize(("threshold", "expected"), [(0.05, "immaterial"), (0.01, "material"), (None, "material")])
def test_a_separated_comparison_carries_the_materiality_of_its_delta(threshold: float | None, expected: str) -> None:
    """The verdict says the arms separated; materiality says whether the separation is worth acting on — the
    host's threshold read by the one predicate every surface uses, so a memo cannot crown a winner on a
    difference the host declared too small to matter."""
    family = _family(_bundle([], accuracy=_ACCURACY, profile=_accuracy_threshold(threshold)))
    (comparison,) = [c for c in family.comparisons if c.name == "field_accuracy"]

    assert comparison.verdict == "improved"
    assert comparison.delta == pytest.approx(0.02, abs=0.005)
    assert comparison.materiality == expected


def test_the_report_and_the_writer_see_an_immaterial_verdict_labelled() -> None:
    """Both surfaces that list a comparison's verdict: the report's contrast table and the analysis writer's bundle."""
    from threetears.evals.analysis.report.build import build_code_only_report

    profile = _accuracy_threshold(0.05)
    bundle = _bundle([], accuracy=_ACCURACY, profile=profile)
    report = build_code_only_report(bundle, measures=profile.measures, assembled_at="2026-10-06T00:00:00Z")
    (table,) = [block for block in report.blocks if getattr(block, "name", None) == "comparisons"]
    (row,) = [row for row in table.rows if row["reading"] == "field_accuracy"]  # type: ignore[attr-defined]

    assert (
        row["verdict"].startswith("improved")
        and "immaterial: the observed delta is below the host's materiality threshold" in row["verdict"]
    )
    assert '"materiality":"immaterial"' in build_user_message(bundle).replace(" ", "")


# --- what a contrast states beside its verdict: interval, effect size, equivalence, the cases it read ---------


def test_a_contrast_carries_its_interval_and_effect_size_at_the_family_level() -> None:
    """Two readings, so each interval is at 1 − α/2 = 97.5%, from the same paired test as the p."""
    from threetears.evals.analysis.stats import difference_interval

    family = _family(_bundle([CLEAR, NOISE]))
    assert family.interval_level == pytest.approx(0.975)
    clear = next(c for c in family.comparisons if c.name == "extraction.d0")
    assert clear.interval == pytest.approx(
        difference_interval([3.0] * 12, [3.0 + d for d in CLEAR], paired=True, confidence=0.975)
    )
    assert clear.interval is not None and clear.interval[0] > 0, "a separation this clear excludes zero"
    assert clear.hedges_g == composite_significance([3.0] * 12, [3.0 + d for d in CLEAR], paired=True).hedges_g
    assert "97.5%" in family.disclosure


#: Two arms within a point of each other on every case: shown inside a five-point margin.
_ALIKE = ([0.80, 0.81, 0.79, 0.80] * 3, [0.80, 0.80, 0.80, 0.81] * 3)


@pytest.mark.parametrize(("threshold", "expected"), [(0.05, "equivalent"), (None, "not_separated")])
def test_equivalent_is_claimed_only_by_the_equivalence_test_against_a_declared_margin(
    threshold: float | None, expected: str
) -> None:
    """With no margin a move that does not separate says nothing either way; with one, TOST can show it inside."""
    family = _family(_bundle([], accuracy=_ALIKE, profile=_accuracy_threshold(threshold)))
    (comparison,) = [c for c in family.comparisons if c.name == "field_accuracy"]

    assert comparison.verdict == expected
    assert comparison.equivalence_margin == threshold
    assert family.n_equivalence_tests == (1 if threshold else 0)
    if threshold:
        assert comparison.equivalence_p_raw is not None and comparison.equivalence_p_adjusted is not None
        assert comparison.equivalence_p_adjusted < 0.05
        assert (
            comparison.interval is not None and -threshold < comparison.interval[0] < comparison.interval[1] < threshold
        )
        assert "equivalence_p_raw" not in build_user_message(
            _bundle([], accuracy=_ALIKE, profile=_accuracy_threshold(threshold))
        )
    else:
        assert comparison.equivalence_p_raw is None and comparison.equivalence_p_adjusted is None


def test_the_means_and_counts_are_over_the_cases_the_test_read_and_the_dropped_ones_are_counted() -> None:
    """The contrast ran nine of the control's twelve cases: the test pairs over nine, and so do its means."""
    accuracy = ([0.5] * 9 + [0.9] * 3, [0.6, 0.62, 0.61, 0.6, 0.63, 0.6, 0.61, 0.62, 0.6] + [0.0] * 3)
    family = _family(_bundle([], accuracy=accuracy, contrast_cases=range(9)))
    (comparison,) = [c for c in family.comparisons if c.name == "field_accuracy"]

    assert comparison.test == "paired"
    assert (comparison.control.n_cases, comparison.contrast.n_cases) == (9, 9)
    assert (comparison.control.n_left_out, comparison.contrast.n_left_out) == (3, 0)
    assert comparison.control.mean == pytest.approx(0.5), "the control's own mean over twelve cases is 0.6"
    assert comparison.delta == pytest.approx(sum(accuracy[1][:9]) / 9 - 0.5)


def test_the_campaign_wide_family_leaves_out_the_rig_s_own_readings() -> None:
    """With no question, every reading on a merit axis is tested — and judge_ms and cost_usd sit on none.

    A judged A/B once reported "judge_ms … improved on the control": the judge phase's time and the blended
    spend that includes the judge's are what it cost to MEASURE an arm, not anything the arm did.
    """
    family = _family(_bundle([CLEAR], questions=False, apparatus=True))
    names = {c.name for c in family.comparisons}
    assert "extraction.d0" in names
    assert not names & {"judge_ms", "cost_usd", "program_cost", "async_wait_ms"}


def test_the_report_states_each_contrast_s_means_cases_interval_and_effect_size() -> None:
    from threetears.evals.analysis.report.build import build_code_only_report

    profile = _accuracy_threshold(None)
    accuracy = ([0.5] * 9 + [0.9] * 3, [0.6, 0.62, 0.61, 0.6, 0.63, 0.6, 0.61, 0.62, 0.6] + [0.0] * 3)
    bundle = _bundle([], accuracy=accuracy, contrast_cases=range(9), profile=profile)
    report = build_code_only_report(bundle, measures=profile.measures, assembled_at="2026-10-06T00:00:00Z")
    (table,) = [block for block in report.blocks if getattr(block, "name", None) == "comparisons"]
    (row,) = [row for row in table.rows if row["reading"] == "field_accuracy"]  # type: ignore[attr-defined]

    assert row["control_mean"] == pytest.approx(0.5)
    assert row["cases"] == "9 paired; 3 of the control's left out, not run by the other side"
    assert isinstance(row["interval"], str) and row["interval"].endswith("at 95%")
    assert row["hedges_g"] is not None and row["hedges_g"] > 0
