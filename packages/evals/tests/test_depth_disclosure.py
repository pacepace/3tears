"""Each arm's depth is stated, a comparison of pass^k names its common depth, and what limits an interval is said.

Arms pooled from runs of different ``k``, or a run stopped part-way, differ in depth across arms and across the cases
of one arm. Nothing read changes for it: every reading and frontier point states each arm's cases, trials and repeats
per case; a frontier subject names the deepest depth every arm reached, decided per case, and reads each point there
over all its runs; a depth some arm never reached is unmeasured for it; and a pass-rate reading says which side limits
its interval and how much of the difference's variance lies between cases, which extra repeats cannot lower.

Every expected value below is worked by hand from the fixture's attempts.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis import assemble_context_bundle, campaign_readings
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.depth import (
    Depth,
    common_depth,
    common_depth_sentence,
    depth_of,
    depth_sentence,
    depths_differ,
    pass_hat_k_over,
    times,
    unmeasured_sentence,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL, bounded_mean_interval, contrast_cases, contrast_samples
from threetears.evals.kernel.declaration import CampaignArm
from threetears.evals.kernel.identity import IDENTITY_VERSION, derive_context_identity, derive_variant_identity
from threetears.evals.kernel.scoring import compute_pass_hat_k
from threetears.evals.schema.models import EvalResult, EvalRun, RoleUsage
from packages.evals.tests.factories import (
    make_aspect_campaign,
    make_eval_run,
    make_scored_result,
    minimal_aspect_design,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

PROFILE = toyhost_profile()

#: The four cases every fixture arm runs.
CASES = ["tc1", "tc2", "tc3", "tc4"]


def _launched(run_id: str, model: str, k: int) -> EvalRun:
    """A run of one arm (its candidate model) at planned depth ``k``, stamped as the launch stamps it."""
    run = make_eval_run(
        id=run_id, status="completed", template_digest="digest-1", candidate_model=model, k_runs=k, test_case_ids=CASES
    )
    context = derive_context_identity(run, PROFILE)
    variant = derive_variant_identity(run=run, profile=PROFILE)
    return run.model_copy(
        update={
            "context_key": context.context_key,
            "context_components": context.context_components,
            "identity_version": IDENTITY_VERSION,
            "variant_levers": variant.levers,
        }
    )


def _key(run: EvalRun) -> str:
    return derive_variant_identity(run=run, profile=PROFILE).variant_key


def _results(run: EvalRun, outcomes: dict[str, tuple[bool, ...]]) -> list[EvalResult]:
    """One result per attempt, each case's attempts in order, every candidate call naming the model that answered."""
    usage = [
        RoleUsage(
            role="candidate", model=run.candidate_model, cost_usd=0.01, call_count=1, served_model=run.candidate_model
        )
    ]
    return [
        make_scored_result(
            test_case_id=case, run_id=run.id, k=k, goal_passes=(passed,), model=run.candidate_model
        ).model_copy(
            update={
                "variant_key": _key(run),
                "identity_version": IDENTITY_VERSION,
                "scope_id": run.scope_id,
                "usage": usage,
            }
        )
        for case, attempts in outcomes.items()
        for k, passed in enumerate(attempts, start=1)
    ]


class _Arms:
    """Two arms on one aspect: ``A`` the control, ``B`` the candidate, each one run with its own attempts."""

    def __init__(
        self,
        a: dict[str, tuple[bool, ...]],
        b: dict[str, tuple[bool, ...]],
        *,
        k_a: int,
        k_b: int,
    ) -> None:
        self.run_a, self.run_b = _launched("run-a", "sonnet", k_a), _launched("run-b", "haiku", k_b)
        self.key_a, self.key_b = _key(self.run_a), _key(self.run_b)
        self.results = {self.run_a.id: _results(self.run_a, a), self.run_b.id: _results(self.run_b, b)}
        self.arms = [
            CampaignArm(label="A", role="control", variant_key=self.key_a),
            CampaignArm(label="B", role="candidate", variant_key=self.key_b),
        ]
        self.campaign = make_aspect_campaign(
            label="character",
            run_ids=[self.run_a.id, self.run_b.id],
            question="Is B better on character?",
            design=minimal_aspect_design(merit_axes=[]),
            margin=10.0,
            control=self.arms[0],
            scope_id=self.run_a.scope_id,
            subject_id=self.run_a.subject_snapshot.subject_id,
            behavior="b",
        )
        self.bundle = assemble_context_bundle(
            self.campaign, storage=ToyhostStorage([self.run_a, self.run_b], self.results), profile=PROFILE
        )

    def readings(self, bundle: AnalysisContextBundle | None = None) -> Any:
        return campaign_readings(self.campaign, {"character": bundle or self.bundle}, arms=self.arms)

    def point(self, key: str) -> Any:
        return next(point for point in self.bundle.frontier.subjects[0].points if point.variant_key == key)


#: Arm A run twice on every case, arm B once: the shares A 1, 1/2, 0, 1 and B 1, 0, 1, 1.
_TWICE = {"tc1": (True, True), "tc2": (True, False), "tc3": (False, False), "tc4": (True, True)}
_ONCE = {"tc1": (True,), "tc2": (False,), "tc3": (True,), "tc4": (True,)}


# =============================================================================
# The depth facts
# =============================================================================


def test_an_arm_s_depth_is_its_cases_trials_and_repeats_per_case() -> None:
    depth = depth_of([2, 2, 1, 2])

    assert depth == Depth(cases=4, trials=7, repeats_min=1, repeats_median=2.0, repeats_max=2)
    assert times(depth) == "1 to 2 times"
    assert times(Depth(cases=60, trials=60, repeats_min=1, repeats_median=1.0, repeats_max=1)) == "once"
    assert times(Depth(cases=3, trials=6, repeats_min=2, repeats_median=2.0, repeats_max=2)) == "twice"
    # An even count of cases reads the median between the middle two.
    assert depth_of([1, 3]) == Depth(cases=2, trials=4, repeats_min=1, repeats_median=2.0, repeats_max=3)
    assert depth_of([]) is None and depth_of([0]) is None


def test_the_common_depth_is_the_deepest_every_arm_reached_counted_per_case() -> None:
    # A reached 3 on tc1 and 1 on tc2; B 2 on both. Every arm reached 2 (B's deepest), and only tc1 is 2 deep on both.
    common = common_depth(
        {"A": {"tc1": [True, True, False], "tc2": [True]}, "B": {"tc1": [True, True], "tc2": [False, True]}}
    )

    assert common.k == 2
    assert common.cases == frozenset({"tc1"})
    assert common.left_out == {"A": 1, "B": 1}
    # A deeper arm is read at the common depth over all its runs: C(2, 2) / C(3, 2) = 1/3 on tc1.
    assert pass_hat_k_over({"tc1": [True, True, False], "tc2": [True]}, common.cases, 2) == pytest.approx(1 / 3)
    # A case one arm never ran is left out of a comparison that needs both.
    assert common_depth({"A": {"tc1": [True], "tc2": [True]}, "B": {"tc1": [False]}}).left_out == {"A": 1, "B": 0}
    assert common_depth({}).k is None


def test_the_sentences_say_the_depths_in_plain_words() -> None:
    once = Depth(cases=60, trials=60, repeats_min=1, repeats_median=1.0, repeats_max=1)
    twice = Depth(cases=60, trials=120, repeats_min=2, repeats_median=2.0, repeats_max=2)

    assert depth_sentence("character", {"1017": twice, "1016": once, "1015": once}) == (
        "1017 was run twice on character, the others once; that tightens 1017's own estimate, not the comparison much."
    )
    assert depth_sentence("character", {"1017": twice, "1016": once}) == (
        "1017 was run twice on character, 1016 once; that tightens 1017's own estimate, not the comparison much."
    )
    assert depth_sentence("character", {"1017": once, "1016": once}) is None
    assert not depths_differ([once, once, None]) and depths_differ([once, twice])
    assert common_depth_sentence(1, left_out=0) == (
        "pass^k is compared at depth 1, the deepest every arm reached, so it compares pass rates, not reliability."
    )
    assert common_depth_sentence(2, left_out=1) == (
        "pass^k is compared at depth 2, the deepest every arm reached, each arm read over all its runs; 1 case some "
        "arm measured fewer times are left out."
    )
    assert unmeasured_sentence(2, ["B"]) == "pass^2 is unmeasured for B, which ran no case twice."
    assert unmeasured_sentence(2, []) is None


def test_a_pass_hat_k_entry_counts_the_cases_short_of_k() -> None:
    # tc1 scored twice, tc2 once (a run stopped part-way): pass^2 reads tc1 alone and counts tc2 as short of it.
    results = [
        make_scored_result(test_case_id="tc1", k=1, goal_passes=(True,)),
        make_scored_result(test_case_id="tc1", k=2, goal_passes=(True,)),
        make_scored_result(test_case_id="tc2", k=1, goal_passes=(True,)),
    ]

    entry = compute_pass_hat_k(results, k=2)[("m1", "r1")]

    assert (entry["n_cases_at_k"], entry["n_cases_short_of_k"], entry["n_test_cases"]) == (1, 1, 2)
    assert entry["pass_hat_k"] == 1.0


# =============================================================================
# Two arms at different depths, end to end
# =============================================================================


def test_each_arm_s_depth_is_on_the_scoreboard_and_its_trials_beside_its_shares() -> None:
    arms = _Arms(_TWICE, _ONCE, k_a=2, k_b=1)
    boards = {board.variant_key: board for board in arms.bundle.scoreboard}

    assert boards[arms.key_a].depth == Depth(cases=4, trials=8, repeats_min=2, repeats_median=2.0, repeats_max=2)
    assert boards[arms.key_b].depth == Depth(cases=4, trials=4, repeats_min=1, repeats_median=1.0, repeats_max=1)
    [passed] = [contrast for contrast in arms.bundle.floor_contrasts if contrast.reading == "trial"]
    assert passed.case_values == ([1.0, 0.5, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0])
    assert passed.case_trials == ([2, 2, 2, 2], [1, 1, 1, 1])
    # The cost contrast carries no trials: its depth is the pass rate's.
    assert all(contrast.case_trials is None for contrast in arms.bundle.floor_contrasts if contrast.reading != "trial")


def test_a_reading_states_both_arms_depths_and_what_limits_its_interval() -> None:
    arms = _Arms(_TWICE, _ONCE, k_a=2, k_b=1)

    [row] = arms.readings().readings

    assert row.control_depth == Depth(cases=4, trials=8, repeats_min=2, repeats_median=2.0, repeats_max=2)
    assert row.arm_depth == Depth(cases=4, trials=4, repeats_min=1, repeats_median=1.0, repeats_max=1)
    # Every one of the four paired cases ran twice on A and once on B.
    assert row.depth_differs == 4
    precision = row.precision
    assert precision is not None
    # Each side's own half-width is its bounded mean interval's on its shares, in points.
    a_low, a_high = bounded_mean_interval([1.0, 0.5, 0.0, 1.0], (0.0, 1.0), confidence=INTERVAL_LEVEL) or (0, 0)
    b_low, b_high = bounded_mean_interval([1.0, 0.0, 1.0, 1.0], (0.0, 1.0), confidence=INTERVAL_LEVEL) or (0, 0)
    assert precision.half_widths == {
        arms.key_a: pytest.approx(50 * (a_high - a_low)),
        arms.key_b: pytest.approx(50 * (b_high - b_low)),
    }
    assert precision.limited_by == (arms.key_a if a_high - a_low > b_high - b_low else arms.key_b)
    # A's repeats: tc2 passed one of two (squares about its mean 1/4 + 1/4 over 1 df), the rest agree; four df in
    # all, so the within-case variance is 0.5 / 4.
    assert precision.within_case == pytest.approx(0.125)
    # The paired differences B - A are 0, -1/2, 1, 0: mean 1/8, sample variance 19/48. Repeat noise takes out
    # 0.125 * (1/2 + 1/1), leaving 19/48 - 3/16 = 5/24 between cases, 5/24 / (5/24 + 3/16) = 10/19 of the whole.
    assert precision.between_case == pytest.approx(5 / 24)
    assert precision.between_share == pytest.approx(10 / 19)


def test_a_frontier_point_states_the_depth_it_delivered_and_its_subject_the_common_depth() -> None:
    arms = _Arms(_TWICE, _ONCE, k_a=2, k_b=1)
    subject = arms.bundle.frontier.subjects[0]
    a, b = arms.point(arms.key_a), arms.point(arms.key_b)

    assert a.depth == Depth(cases=4, trials=8, repeats_min=2, repeats_median=2.0, repeats_max=2)
    assert b.depth == Depth(cases=4, trials=4, repeats_min=1, repeats_median=1.0, repeats_max=1)
    # The planned depth is the shallowest k_runs (1), and B's deepest case (1) is the deepest every arm reached.
    assert (subject.k, subject.common_depth, subject.unmeasured_for) == (1, 1, [])
    # A at depth 1 over all its runs: the mean of its shares, (1 + 1/2 + 0 + 1) / 4.
    assert (a.common_pass_hat_k, a.n_common_cases, a.n_cases_left_out) == (pytest.approx(0.625), 4, 0)
    assert (b.common_pass_hat_k, b.n_common_cases, b.n_cases_left_out) == (pytest.approx(0.75), 4, 0)


def test_a_depth_above_an_arm_s_reach_is_unmeasured_for_it_and_the_common_depth_falls_to_its_reach() -> None:
    # Both planned k = 2, but B's run stopped after one attempt at each case.
    arms = _Arms(_TWICE, _ONCE, k_a=2, k_b=2)
    subject = arms.bundle.frontier.subjects[0]
    a, b = arms.point(arms.key_a), arms.point(arms.key_b)

    assert subject.k == 2
    assert subject.unmeasured_for == [arms.key_b]
    assert b.pass_hat_k is None and b.n_cases_short_of_k == 4
    # A's pass^2 at the planned depth: C(2,2)/C(2,2) on tc1 and tc4, 0 on tc2 and tc3.
    assert (a.pass_hat_k, a.n_pass_cases, a.n_cases_short_of_k) == (pytest.approx(0.5), 4, 0)
    assert subject.common_depth == 1
    assert (a.common_pass_hat_k, b.common_pass_hat_k) == (pytest.approx(0.625), pytest.approx(0.75))


def test_depths_mixed_within_one_arm_are_stated_and_its_short_case_is_counted() -> None:
    # A's tc3 was cut to one attempt: three cases twice, one once.
    mixed = {**_TWICE, "tc3": (False,)}
    arms = _Arms(mixed, _TWICE, k_a=2, k_b=2)
    a, b = arms.point(arms.key_a), arms.point(arms.key_b)
    subject = arms.bundle.frontier.subjects[0]

    assert a.depth == Depth(cases=4, trials=7, repeats_min=1, repeats_median=2.0, repeats_max=2)
    assert (a.n_pass_cases, a.n_cases_short_of_k) == (3, 1)
    # Both reached 2, so 2 is the common depth, and tc3 counts there for neither arm: A ran it once.
    assert subject.common_depth == 2
    assert (a.n_common_cases, a.n_cases_left_out, b.n_cases_left_out) == (3, 1, 1)
    # Over tc1, tc2, tc4 at depth 2 each arm reads 1, 0, 1.
    assert a.common_pass_hat_k == pytest.approx(2 / 3) and b.common_pass_hat_k == pytest.approx(2 / 3)
    [row] = arms.readings().readings
    assert row.depth_differs == 1  # tc3: once on A, twice on B.


def test_a_campaign_run_once_on_every_case_states_one_depth_and_compares_pass_rates() -> None:
    once_a = {"tc1": (True,), "tc2": (True,), "tc3": (False,), "tc4": (True,)}
    arms = _Arms(once_a, _ONCE, k_a=1, k_b=1)
    subject = arms.bundle.frontier.subjects[0]
    [row] = arms.readings().readings

    assert (
        row.control_depth == row.arm_depth == Depth(cases=4, trials=4, repeats_min=1, repeats_median=1.0, repeats_max=1)
    )
    assert row.depth_differs == 0
    assert (subject.k, subject.common_depth, subject.unmeasured_for) == (1, 1, [])
    # No case repeated, so no within-case spread can be told from the between-case spread.
    assert row.precision is not None and row.precision.within_case is None and row.precision.between_share is None


# =============================================================================
# Nothing read changes
# =============================================================================

#: The fields the depth disclosure adds to a reading; every other field is as it was.
_ADDED = {"control_depth", "arm_depth", "depth_differs", "precision"}


def test_no_existing_reading_value_moves_and_none_reads_the_depth_facts() -> None:
    arms = _Arms(_TWICE, _ONCE, k_a=2, k_b=1)
    stripped = arms.bundle.model_copy(
        update={
            "scoreboard": [board.model_copy(update={"depth": None}) for board in arms.bundle.scoreboard],
            "floor_contrasts": [
                contrast.model_copy(update={"case_trials": None}) for contrast in arms.bundle.floor_contrasts
            ],
        }
    )

    [with_depth] = arms.readings().readings
    [without] = arms.readings(stripped).readings

    assert with_depth.model_dump(exclude=_ADDED) == without.model_dump(exclude=_ADDED)
    # And the values themselves, worked by hand: B's shares average 3/4 and A's 5/8, 12.5 points apart, over every
    # trial A passed 5 of 8 and B 3 of 4, each on 4 cases.
    assert with_depth.change == pytest.approx(12.5)
    assert (with_depth.control_passed, with_depth.control_trials, with_depth.arm_passed, with_depth.arm_trials) == (
        5,
        8,
        3,
        4,
    )
    assert (with_depth.control_cases, with_depth.arm_cases) == (4, 4)


def test_no_existing_frontier_value_moves() -> None:
    arms = _Arms(_TWICE, _ONCE, k_a=2, k_b=2)
    a, b = arms.point(arms.key_a), arms.point(arms.key_b)

    # pass^k still reads the planned depth on each point's own cases, and its curve every depth a case reached.
    assert (a.k, a.pass_hat_k, a.n_pass_cases) == (2, pytest.approx(0.5), 4)
    assert a.pass_hat_k_curve == [
        {"k": 1, "pass_hat_k": pytest.approx(0.625), "n_cases": 4},
        {"k": 2, "pass_hat_k": pytest.approx(0.5), "n_cases": 4},
    ]
    assert (b.k, b.pass_hat_k, b.n_pass_cases) == (2, None, 0)
    assert b.pass_hat_k_curve == [{"k": 1, "pass_hat_k": pytest.approx(0.75), "n_cases": 4}]


def test_contrast_cases_are_the_cases_contrast_samples_reads() -> None:
    control = {"tc2": 0.5, "tc1": 1.0, "tc3": 0.0}
    contrast = {"tc3": 1.0, "tc1": 0.0, "tc9": 1.0}

    cases_a, cases_b, paired = contrast_cases(control, contrast)

    assert (cases_a, cases_b, paired) == (["tc1", "tc3"], ["tc1", "tc3"], True)
    assert contrast_samples(control, contrast) == ([1.0, 0.0], [0.0, 1.0], True)
    # Sharing one case, each side reads its own, sorted.
    assert contrast_cases({"b": 1.0, "a": 1.0}, {"a": 1.0, "z": 0.0}) == (["a", "b"], ["a", "z"], False)


def test_a_floor_contrast_refuses_trials_that_do_not_line_up_with_its_shares() -> None:
    [passed] = [
        contrast
        for contrast in _Arms(_TWICE, _ONCE, k_a=2, k_b=1).bundle.floor_contrasts
        if contrast.reading == "trial"
    ]
    data = passed.model_dump(mode="json")

    with pytest.raises(ValueError, match="line up"):
        type(passed).model_validate({**data, "case_trials": [[2, 2, 2], [1, 1, 1, 1]]})
    with pytest.raises(ValueError, match="at least one usable trial"):
        type(passed).model_validate({**data, "case_trials": [[2, 2, 2, 0], [1, 1, 1, 1]]})
