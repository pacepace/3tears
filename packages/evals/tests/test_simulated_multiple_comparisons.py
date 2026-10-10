"""The campaign family's multiple-comparison correction, checked against data with a known truth (#601).

A campaign question's family is every contrast against the control on every reading it asks about,
Holm-corrected as one, and a verdict is read off the adjusted p. ``test_multiple_comparisons.py`` pins
the arithmetic on fixed inputs. This file checks what the correction is for:

- **family-wise error at α**: with no true difference anywhere, the chance that ANY comparison in the
  family reads ``improved`` or ``regressed`` is at most α, under independence and under the dependence a
  real family has (readings off the same cells; contrasts sharing one control);
- **under a partial null**, the family error over the comparisons with no true difference stays at α while
  the one with a difference is found at least as often as a Bonferroni test would find it, and almost never
  in the wrong direction.

The simulation runs the family rule composed from the engine's public statistics
(:func:`~packages.evals.tests.simulation_support.family_verdicts`), because thousands of bundles would take
minutes. ``TestTheBundleAppliesTheRule`` closes the gap: on seeded campaigns covering every branch of the
rule, every comparison the assembled bundle publishes carries exactly the rule's p, adjusted p and verdict.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest

from threetears.evals.analysis import AnalysisContextBundle, ComparisonFamily, assemble_context_bundle
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA, holm_adjust
from threetears.evals.contracts import EvalCampaign, EvalResult, Question
from packages.evals.tests.factories import fixture_variant_key, make_eval_result, make_eval_run, minimal_declaration
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.simulation_support import (
    ClusteredDesign,
    at_least,
    at_most,
    case_means,
    family_verdicts,
    paired_t_power,
    within,
)


def _correlated_normals(rng: random.Random, count: int, correlation: float) -> list[float]:
    """``count`` standard normals with pairwise correlation ``correlation``."""
    shared = rng.gauss(0.0, 1.0)
    return [math.sqrt(correlation) * shared + math.sqrt(1.0 - correlation) * rng.gauss(0.0, 1.0) for _ in range(count)]


def _p_of_z(z: float) -> float:
    """The two-sided normal p of ``z``."""
    return math.erfc(abs(z) / math.sqrt(2.0))


class TestHolmHoldsTheFamilyWiseError:
    """``holm_adjust`` alone: valid p's in, family-wise error at most α out."""

    #: 20,000 replicates: SE at α is sqrt(0.05 * 0.95 / 20000) = 0.0015, a 4-SE band of ±0.006.
    REPLICATES = 20000

    @pytest.mark.parametrize("family_size", [2, 5, 10, 20])
    def test_under_independence_it_is_exactly_bonferronis_first_step(self, family_size: int) -> None:
        """Under the global null Holm rejects anything only if its smallest p clears α/m, so with
        independent uniform p's its family-wise error is exactly ``1 - (1 - α/m)^m`` (0.0494 at m=2, 0.0488
        at m=20) — a known answer, not just a bound."""
        rng = random.Random(f"holm-independent-{family_size}")
        errors = sum(
            1
            for _ in range(self.REPLICATES)
            if min(holm_adjust([rng.random() for _ in range(family_size)])) < SIGNIFICANCE_ALPHA
        )
        rate = errors / self.REPLICATES
        expected = 1.0 - (1.0 - SIGNIFICANCE_ALPHA / family_size) ** family_size
        assert within(rate, expected, self.REPLICATES), (
            f"m={family_size}: family-wise error {rate:.4f}, exact {expected:.4f}"
        )

    @pytest.mark.parametrize(("family_size", "correlation"), [(5, 0.5), (10, 0.8), (20, 0.3)])
    def test_under_positive_dependence_it_stays_at_most_alpha(self, family_size: int, correlation: float) -> None:
        """Several readings off the same cells are positively correlated test statistics; Holm assumes nothing
        about dependence, so its error stays at most α (and falls below it as the tests move together)."""
        rng = random.Random(f"holm-dependent-{family_size}-{correlation}")
        errors = 0
        for _ in range(self.REPLICATES):
            p_values = [_p_of_z(z) for z in _correlated_normals(rng, family_size, correlation)]
            errors += min(holm_adjust(p_values)) < SIGNIFICANCE_ALPHA
        rate = errors / self.REPLICATES
        assert rate <= at_most(SIGNIFICANCE_ALPHA, self.REPLICATES), (
            f"m={family_size}, ρ={correlation}: family-wise error {rate:.4f} against α={SIGNIFICANCE_ALPHA}"
        )


def _campaign_family(
    rng: random.Random,
    design: ClusteredDesign,
    *,
    contrasts: int,
    readings: int,
    reading_correlation: float,
    effects: Sequence[Sequence[float]],
) -> list[tuple[dict[str, float], dict[str, float], bool]]:
    """One campaign's family: ``contrasts`` arms against one control on ``readings`` readings, case-paired.

    Every arm runs the same cases. An observation of one reading is a case level shared by every arm
    (between-case SD), a per-arm case deviation (SD 0.5), and repeat noise; the readings of one observation
    are correlated ``reading_correlation`` through a shared component, as several measures off one response
    are. ``effects[c][r]`` is contrast ``c``'s true shift on reading ``r``, in units of its per-case
    difference SD (d_z).

    Returns:
        The family, contrast-major, as :func:`family_verdicts` takes it (higher is better on every reading).
    """
    cases = [f"case-{index}" for index in range(design.n_cases)]
    case_levels = [[rng.gauss(0.0, design.between_case_sd) for _ in range(readings)] for _ in cases]
    arm_sd = 0.5
    difference_sd = math.sqrt(2 * arm_sd**2 + 2 * design.repeat_sd**2 / design.repeats)

    def arm(shift: Sequence[float]) -> list[dict[str, float]]:
        per_reading: list[dict[str, float]] = [{} for _ in range(readings)]
        for case, levels in zip(cases, case_levels, strict=True):
            deviation = [arm_sd * z for z in _correlated_normals(rng, readings, reading_correlation)]
            for reading in range(readings):
                observations = [
                    levels[reading] + deviation[reading] + shift[reading] * difference_sd + design.repeat_sd * z
                    for z in (rng.gauss(0.0, 1.0) for _ in range(design.repeats))
                ]
                per_reading[reading][case] = case_means([observations])[0]
        return per_reading

    control = arm([0.0] * readings)
    family = []
    for contrast_effects in effects[:contrasts]:
        contrast = arm(contrast_effects)
        family.extend((control[reading], contrast[reading], True) for reading in range(readings))
    return family


class TestTheCampaignFamilyHoldsTheFamilyWiseError:
    """The whole family rule — per-case means, paired tests, Holm, verdict — on campaigns with a known truth."""

    #: 3,000 replicates: SE at α is 0.0040, a 4-SE band of ±0.016.
    REPLICATES = 3000

    @pytest.mark.parametrize(
        ("n_cases", "repeats", "contrasts", "readings"),
        [(3, 3, 1, 3), (5, 3, 3, 4), (10, 1, 2, 6), (15, 5, 1, 2)],
    )
    def test_with_no_difference_anywhere_no_verdict_is_reached_beyond_alpha(
        self, n_cases: int, repeats: int, contrasts: int, readings: int
    ) -> None:
        rng = random.Random(f"family-null-{n_cases}-{repeats}-{contrasts}-{readings}")
        design = ClusteredDesign(n_cases=n_cases, repeats=repeats, between_case_sd=1.0, repeat_sd=0.5)
        null = [[0.0] * readings] * contrasts
        errors = 0
        for _ in range(self.REPLICATES):
            family = _campaign_family(
                rng, design, contrasts=contrasts, readings=readings, reading_correlation=0.5, effects=null
            )
            errors += any(verdict.verdict in ("improved", "regressed") for verdict in family_verdicts(family))
        rate = errors / self.REPLICATES
        assert rate <= at_most(SIGNIFICANCE_ALPHA, self.REPLICATES), (
            f"{n_cases} cases x {repeats}, {contrasts} contrasts x {readings} readings: "
            f"family-wise error {rate:.4f} against α={SIGNIFICANCE_ALPHA}"
        )

    def test_under_a_partial_null_the_true_difference_is_found_and_the_rest_held_at_alpha(self) -> None:
        """One reading of four moves by d_z = 1.2 at 10 cases; the other three do not move.

        The family error over the three unmoved readings stays at most α; the moved one is found at
        least as often as a Bonferroni test at α/4 finds it, computed from the noncentral t (a separation needs
        its Bonferroni interval to exclude 0, which is that test, #597); and it is called a regression — the
        wrong direction — at most α/2 of the time. No separation is ever read beside an interval reaching 0.
        """
        rng = random.Random("family-partial-null")
        design = ClusteredDesign(n_cases=10, repeats=3, between_case_sd=1.0, repeat_sd=0.5)
        effects = [[1.2, 0.0, 0.0, 0.0]]
        found = wrong_way = null_errors = 0
        for _ in range(self.REPLICATES):
            family = _campaign_family(rng, design, contrasts=1, readings=4, reading_correlation=0.5, effects=effects)
            verdicts = family_verdicts(family)
            assert not any(
                v.verdict in ("improved", "regressed")
                and v.interval is not None
                and v.interval[0] <= 0 <= v.interval[1]
                for v in verdicts
            ), "a separation beside an interval that includes 0 (#597)"
            found += verdicts[0].verdict == "improved"
            wrong_way += verdicts[0].verdict == "regressed"
            null_errors += any(
                verdict.verdict != "not_separated" and verdict.verdict != "untested" for verdict in verdicts[1:]
            )
        bonferroni_power = paired_t_power(10, 1.2, SIGNIFICANCE_ALPHA / 4)
        assert null_errors / self.REPLICATES <= at_most(SIGNIFICANCE_ALPHA, self.REPLICATES)
        assert found / self.REPLICATES >= at_least(bonferroni_power, self.REPLICATES), (
            f"power {found / self.REPLICATES:.4f} against Bonferroni's {bonferroni_power:.4f}"
        )
        assert wrong_way / self.REPLICATES <= at_most(SIGNIFICANCE_ALPHA / 2, self.REPLICATES)


class TestEveryVerdictTheFamilyReaches:
    """Equivalence verdicts and intervals join the family, and its error stays at α over all of them.

    Four readings: two that do not move, with a wide margin, so the arms are often (rightly) shown
    equivalent while any separation called on them is false; and two that move by exactly their margin, the
    boundary where an equivalence claim is false as often as it can be while the separation is real. Every
    verdict that could be wrong is counted. Equivalence is tested only on a declared range (#695), so the
    normal readings, which declare none, reach no equivalent verdict, and the pass-rate readings do.
    """

    #: 3,000 replicates: SE at α is 0.0040, a 4-SE band of ±0.016.
    REPLICATES = 3000

    def test_no_verdict_of_either_kind_is_wrong_beyond_alpha_and_the_intervals_cover_together(self) -> None:
        rng = random.Random("family-equivalence")
        design = ClusteredDesign(n_cases=10, repeats=3, between_case_sd=1.0, repeat_sd=0.5)
        difference_sd = math.sqrt(2 * 0.5**2 + 2 * design.repeat_sd**2 / design.repeats)
        effects = [[0.0, 0.0, 0.6, -0.6]]
        margins = [3.0 * difference_sd, 3.0 * difference_sd, 0.6 * difference_sd, 0.6 * difference_sd]
        truths = [effect * difference_sd for effect in effects[0]]
        wrong = equivalences = uncovered = 0
        for _ in range(self.REPLICATES):
            family = _campaign_family(rng, design, contrasts=1, readings=4, reading_correlation=0.5, effects=effects)
            verdicts = family_verdicts(family, margins=margins)
            wrong += any(v.verdict in ("improved", "regressed") for v in verdicts[:2]) or any(
                v.verdict == "equivalent" for v in verdicts[2:]
            )
            equivalences += sum(v.verdict == "equivalent" for v in verdicts[:2])
            uncovered += any(
                v.interval is None or not v.interval[0] <= truth <= v.interval[1]
                for v, truth in zip(verdicts, truths, strict=True)
            )
        assert wrong / self.REPLICATES <= at_most(SIGNIFICANCE_ALPHA, self.REPLICATES), (
            f"family-wise error over every verdict {wrong / self.REPLICATES:.4f} against α={SIGNIFICANCE_ALPHA}"
        )
        assert uncovered / self.REPLICATES <= at_most(SIGNIFICANCE_ALPHA, self.REPLICATES), (
            f"the intervals failed to cover together {uncovered / self.REPLICATES:.4f} of the time"
        )
        assert equivalences == 0, "a margin with no declared range is never tested for equivalence (#695)"

    #: 1,500 replicates: SE at α is 0.0056, a 4-SE band of ±0.023.
    BOUNDED_REPLICATES = 1500

    def test_on_a_declared_range_equivalence_joins_the_family_and_its_error_stays_at_alpha(self) -> None:
        """Pass rates over 30 cases at five repeats, on their declared range [0, 1] with a margin of 0.3."""
        rng = random.Random("family-equivalence-bounded")
        margin, shifts = 0.3, [0.0, 0.0, 0.3, -0.3]
        cases = [f"case-{index}" for index in range(30)]
        wrong = equivalences = 0
        for _ in range(self.BOUNDED_REPLICATES):
            family = []
            for shift in shifts:
                # Each case's own rate, in [0.3, 0.7] so a shift of ±0.3 stays a probability and the true
                # difference is exactly the shift.
                rates = {case: rng.uniform(0.3, 0.7) for case in cases}
                control = {case: sum(rng.random() < rates[case] for _ in range(5)) / 5 for case in cases}
                contrast = {case: sum(rng.random() < rates[case] + shift for _ in range(5)) / 5 for case in cases}
                family.append((control, contrast, True))
            verdicts = family_verdicts(family, margins=[margin] * 4, value_ranges=[(0.0, 1.0)] * 4)
            wrong += any(v.verdict in ("improved", "regressed") for v in verdicts[:2]) or any(
                v.verdict == "equivalent" for v in verdicts[2:]
            )
            equivalences += sum(v.verdict == "equivalent" for v in verdicts[:2])
        assert wrong / self.BOUNDED_REPLICATES <= at_most(SIGNIFICANCE_ALPHA, self.BOUNDED_REPLICATES), (
            f"family-wise error over every verdict {wrong / self.BOUNDED_REPLICATES:.4f} against α={SIGNIFICANCE_ALPHA}"
        )
        assert equivalences / (2 * self.BOUNDED_REPLICATES) > 0.5, "the fixture must reach the equivalent verdict"


class TestIdenticalArmsOnPassFailAnswers:
    """Two identical arms answering a label bank, graded pass/fail twice over: a scorer and accuracy.

    The shape a quick comparison of two prompts that change nothing has: each case has the answer it usually
    gets (right on about three cases in four), every answer departs from it one time in twelve, to another
    label, each case is run twice (k=2), and the family holds the scorer's pass rate and the classifier's
    accuracy, which read the same answers. The values are coarse (a case's mean is 0, 1/2 or 1) and mostly
    tied, the regime a t-test is least sure of. No difference exists, so ANY ``improved`` or ``regressed``
    is a false separation, and the family's chance of one must stay at most α at every bank size.

    Through ``compare()`` itself, 1,500 seeded comparisons per bank size gave 0.004 (12 cases), 0.023 (24)
    and 0.021 (48); this runs the same rule fast.
    """

    #: 3,000 replicates: SE at α is 0.0040, a 4-SE band of ±0.016.
    REPLICATES = 3000
    LABELS = 4
    DEPARTS = 1 / 12

    def _arm(self, rng: random.Random, usual_right: Sequence[bool], repeats: int) -> dict[str, float]:
        """Each case's pass rate over its repeats; a departure from a right usual answer is wrong, and from a
        wrong one is right one time in three (it lands on one of the other three labels)."""
        rates = {}
        for case, right in enumerate(usual_right):
            passes = 0
            for _ in range(repeats):
                if rng.random() < self.DEPARTS:
                    passes += (not right) and rng.randrange(self.LABELS - 1) == 0
                else:
                    passes += right
            rates[f"case-{case}"] = passes / repeats
        return rates

    @pytest.mark.parametrize("n_cases", [12, 24, 48])
    def test_the_family_separates_them_at_most_alpha_of_the_time(self, n_cases: int) -> None:
        rng = random.Random(f"identical-pass-fail-{n_cases}")
        errors = 0
        for _ in range(self.REPLICATES):
            usual_right = [rng.random() < 0.75 for _ in range(n_cases)]
            control, contrast = self._arm(rng, usual_right, 2), self._arm(rng, usual_right, 2)
            # The scorer and accuracy read the same answers, so the family holds one comparison twice.
            family = [(control, contrast, True), (control, contrast, True)]
            errors += any(v.verdict in ("improved", "regressed") for v in family_verdicts(family))
        rate = errors / self.REPLICATES
        assert rate <= at_most(SIGNIFICANCE_ALPHA, self.REPLICATES), (
            f"{n_cases} cases x 2: family-wise false separation {rate:.4f} against α={SIGNIFICANCE_ALPHA}"
        )


# --- the bundle applies the rule ----------------------------------------------------------------------

_CONTROL = "control-model"
_ARMS = ("control-model", "contrast-one", "contrast-two")
#: The two readings the seeded campaigns carry: the toy host's own quality measures, one better high and
#: one better low, so the verdict's direction is exercised both ways.
_READINGS = {"field_accuracy": True, "fields_stripped": False}
#: The readings' declared ranges in the toy host: a rate is bounded, a count is not.
_RANGES: dict[str, tuple[float, float]] = {"field_accuracy": (0.0, 1.0)}


def _seeded_campaign(
    rng: random.Random,
    *,
    n_cases: int,
    repeats: int,
    shift: float,
    cases_of: dict[str, Sequence[int]] | None = None,
    noise: float = 1.0,
) -> tuple[AnalysisContextBundle, dict[tuple[str, str], dict[str, float]]]:
    """Assemble a three-arm campaign with seeded readings, and return each arm's per-case means too.

    ``contrast-one`` is shifted by ``shift`` on ``field_accuracy``; nothing else moves. ``noise`` scales each
    observation's own noise around its case's level: at 0 every arm reads its case's level exactly, so a shift
    moves every case by one amount and the comparison has no spread. ``cases_of`` names
    which case indices an arm ran (default: the first ``n_cases``; up to ``n_cases + 4``), so a contrast can
    share some or only one of the control's cases and take the paired or the unpaired branch.

    Returns:
        The bundle, and ``{(arm, reading): {case id: per-case mean}}`` as the observations were written.
    """
    runs, results = [], {}
    written: dict[tuple[str, str], dict[str, float]] = {}
    case_levels = {index: (rng.gauss(0.0, 0.08), rng.gauss(0.0, 1.0)) for index in range(n_cases + 5)}
    for model in _ARMS:
        run = make_eval_run(status="completed", candidate_model=model)
        runs.append(run)
        members: list[EvalResult] = []
        for index in (cases_of or {}).get(model, range(n_cases)):
            case = f"tc-{index:02d}"
            for repeat in range(1, repeats + 1):
                accuracy = 0.6 + case_levels[index][0] + noise * rng.gauss(0.0, 0.05)
                accuracy = min(1.0, max(0.0, accuracy + (shift if model == "contrast-one" else 0.0)))
                stripped = 5.0 + case_levels[index][1] + noise * rng.gauss(0.0, 0.5)
                values = {"field_accuracy": round(accuracy, 6), "fields_stripped": round(stripped, 6)}
                for reading, value in values.items():
                    written.setdefault((model, reading), {}).setdefault(case, 0.0)
                    written[(model, reading)][case] += value / repeats
                members.append(
                    make_eval_result(
                        id=f"{model}-{case}-{repeat}",
                        eval_run_id=run.id,
                        scope_id=run.scope_id,
                        model=model,
                        test_case_id=case,
                        k_iteration=repeat,
                        goal_state_outcomes=[],
                        cost_usd=None,
                        rubric_scores=[],
                        host_measures=values,
                    )
                )
        results[run.id] = members
    declaration = minimal_declaration(control=fixture_variant_key(_CONTROL)).model_copy(
        update={"questions": [Question(id="q-quality", text="is a contrast better?", merit_axes=["quality"])]}
    )
    campaign = EvalCampaign(
        scope_id=runs[0].scope_id,
        name="seeded family",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[run.id for run in runs],
        declared_design=declaration,
        created_by="test:fixture",
    )
    bundle = assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=toyhost_profile())
    return bundle, written


def _family(bundle: AnalysisContextBundle) -> ComparisonFamily:
    (family,) = bundle.multiple_comparisons.families
    return family


class TestTheBundleAppliesTheRule:
    """Every comparison the bundle publishes carries the rule's p, adjusted p and verdict — the link that lets
    the simulations above stand for the bundle."""

    @pytest.mark.parametrize(
        ("label", "n_cases", "repeats", "shift", "cases_of", "branch"),
        [
            ("no difference, paired", 5, 3, 0.0, None, ("contrast-one", "paired", "not_separated")),
            ("a clear difference, paired", 6, 2, 0.25, None, ("contrast-one", "paired", "improved")),
            (
                "a small difference, partly shared cases",
                8,
                2,
                0.04,
                {"contrast-two": range(3, 8)},
                ("contrast-two", "paired", "not_separated"),
            ),
            (
                "one shared case, so unpaired",
                8,
                3,
                0.35,
                {"contrast-one": range(7, 13)},
                ("contrast-one", "unpaired", "improved"),
            ),
            # No spread: every case moved by one amount, so no t exists. On field_accuracy's declared range the
            # bounded test decides, and eight cases moved 0.25 on a 0-1 rate cannot show a mean moved (#597);
            # fields_stripped declares no range and is not separated. The mirror reads both as the bundle does.
            ("every case moved by one amount", 8, 1, 0.25, None, ("contrast-one", "paired", "not_separated")),
            ("every one of forty cases moved by one amount", 40, 1, 0.25, None, ("contrast-one", "paired", "improved")),
        ],
        ids=["null-paired", "clear-paired", "partly-shared", "one-shared-unpaired", "no-spread-paired", "no-spread-40"],
    )
    def test_each_comparison_is_the_rule(
        self,
        label: str,
        n_cases: int,
        repeats: int,
        shift: float,
        cases_of: dict[str, Sequence[int]] | None,
        branch: tuple[str, str, str],
    ) -> None:
        """``branch`` is the rule branch the campaign exists to reach — ``(contrast, test, verdict)`` on
        ``field_accuracy`` — asserted so a seeded campaign that drifted off its branch fails rather than
        passing over the easy case."""
        rng = random.Random(f"bundle-rule-{label}")
        noise = 0.0 if label.startswith("every") else 1.0
        bundle, written = _seeded_campaign(
            rng, n_cases=n_cases, repeats=repeats, shift=shift, cases_of=cases_of, noise=noise
        )
        family = _family(bundle)
        names = {fixture_variant_key(model): model for model in _ARMS}
        assert {comparison.name for comparison in family.comparisons} == set(_READINGS), (
            "the family holds exactly the seeded readings, so the rule below sees the whole family"
        )
        expected = family_verdicts(
            [
                (
                    written[(names[comparison.control.variant_key], comparison.name)],
                    written[(names[comparison.contrast.variant_key], comparison.name)],
                    _READINGS[comparison.name],
                )
                for comparison in family.comparisons
            ],
            value_ranges=[_RANGES.get(comparison.name) for comparison in family.comparisons],
        )
        for comparison, rule in zip(family.comparisons, expected, strict=True):
            where = f"{label}: {names[comparison.contrast.variant_key]} on {comparison.name}"
            assert comparison.p_raw == pytest.approx(rule.p_raw, rel=1e-9, abs=1e-12), where
            assert comparison.p_adjusted == pytest.approx(rule.p_adjusted, rel=1e-9, abs=1e-12), where
            assert comparison.verdict == rule.verdict, where
            assert (comparison.interval is None) == (rule.interval is None), where
            if comparison.interval is not None and rule.interval is not None:
                assert comparison.interval == pytest.approx(rule.interval, rel=1e-9, abs=1e-12), where
        contrast, test, verdict = branch
        (reached,) = [
            comparison
            for comparison in family.comparisons
            if names[comparison.contrast.variant_key] == contrast and comparison.name == "field_accuracy"
        ]
        assert (reached.test, reached.verdict) == (test, verdict), f"{label}: the campaign missed its branch"
