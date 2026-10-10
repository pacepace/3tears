"""The bundle's frontier takes the campaign's bar on the measure it ranks on, and a judged bar names its tier (#679).

The frontier ranks on pass^k and reads a bar only on it, so the bundle passes it the campaign's effective bar
(declared, else registered) on pass^k and no other. Every other case leaves the bar withheld, and the reason
says which bars exist and on what measure, or why the one on pass^k could not be passed. Each point's
``bar_decision`` is the three-valued interval rule every bar is read by, so a variant below the bar is named
``missed`` only when its whole pass^k interval is under it.

A bar verdict on a judged dimension carries the evidence tier of the judges that scored the cell; a verdict on a
measured quantity carries None.

Mutations that turn this file red: assembling the frontier without a bar again; passing a bar on another
measure onto pass^k; dropping the tier from a judged verdict, or setting one
on a measured verdict; the gate refusing a declared pass^k bar again, or admitting one outside [0, 1].
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.kernel.declaration import BarOverride, CampaignDesign, refuse_an_undeclarable_design
from threetears.evals.kernel.host.bars import Bar, BarRegistry
from threetears.evals.kernel.metrics import FRONTIER_RANKING_MEASURE
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_BARS, toyhost_profile

#: The toy campaign's two arms read pass^k 0 (interval to ~0.30) and 1/3 (interval ~0.08 to ~0.68), so a bar
#: at 0.5 has one arm wholly under it and one straddling it.
_BAR = 0.5


def _bundle(*bars: BarOverride, registered: tuple[Bar, ...] = TOYHOST_BARS) -> AnalysisContextBundle:
    """The toy campaign's bundle with ``bars`` as its declared bars, under the host's ``registered`` ones."""
    campaign, storage = toyhost_campaign()
    assert campaign.declared_design is not None
    # model_construct: the bundle validates the design itself, which is what refuses two bars on one measure.
    design = CampaignDesign.model_construct(**{**dict(campaign.declared_design), "bars": list(bars)})
    profile = replace(toyhost_profile(), bars=BarRegistry(registered))
    return assemble_context_bundle(
        campaign.model_copy(update={"declared_design": design}), storage=storage, profile=profile
    )


def _on_pass_hat_k(threshold: float = _BAR) -> BarOverride:
    return BarOverride(measure_id=FRONTIER_RANKING_MEASURE, threshold=threshold, direction="higher_is_better")


class TestTheFrontierTakesTheBarOnPassHatK:
    def test_a_declared_bar_on_pass_hat_k_sets_the_frontier_bar_and_names_the_variant_below_it(self) -> None:
        bundle = _bundle(_on_pass_hat_k())

        assert bundle.frontier.bar == _BAR
        assert bundle.frontier_bar_withheld is None
        (subject,) = bundle.frontier.subjects
        decisions = sorted(point.bar_decision or "" for point in subject.points)
        assert decisions == ["missed", "undecided"], "one arm wholly under the bar, one straddling it"
        (below,) = [point for point in subject.points if point.bar_decision == "missed"]
        assert below.pass_hat_k_ci_high is not None and below.pass_hat_k_ci_high < _BAR
        assert (subject.n_cleared_bar, subject.n_undecided_bar) == (0, 1)

    def test_the_pass_hat_k_adjudication_points_at_the_frontier_rather_than_saying_nothing_read_it(self) -> None:
        (bar,) = [bar for bar in _bundle(_on_pass_hat_k()).bar_adjudications if bar.measure_id == "pass_hat_k"]

        assert bar.state == "names_no_stored_measure" and bar.verdicts == []
        assert bar.reason is not None and "frontier read this bar" in bar.reason


class TestTheFrontierBarIsWithheld:
    def test_a_bar_on_another_measure_leaves_the_frontier_withheld_and_says_which_bars_exist(self) -> None:
        bundle = _bundle(BarOverride(measure_id="cost_usd", threshold=0.025, direction="lower_is_better"))

        assert bundle.frontier.bar is None
        reason = bundle.frontier_bar_withheld
        assert reason is not None
        assert "n_cleared_bar is a default of 0" in reason
        assert "ranks on pass_hat_k" in reason
        assert "cost_usd ≤ 0.025 (declared)" in reason and "field_accuracy ≥ 0.92 (registered)" in reason
        assert all(subject.n_cleared_bar == 0 for subject in bundle.frontier.subjects)

    def test_two_bars_on_pass_hat_k_never_reach_the_frontier(self) -> None:
        """The frontier takes one bar; two on its measure are refused by the declaration before any pick."""
        with pytest.raises(ValueError, match=r"more than one bar on: pass_hat_k"):
            _bundle(_on_pass_hat_k(0.4), _on_pass_hat_k(0.6))

    def test_a_bar_on_pass_hat_k_outside_its_range_is_withheld_not_raised(self) -> None:
        """The gate refuses one at authoring; a campaign that reached the bundle another way is not a crash."""
        bundle = _bundle(_on_pass_hat_k(1.5))

        assert bundle.frontier.bar is None
        assert bundle.frontier_bar_withheld is not None and "[0, 1]" in bundle.frontier_bar_withheld


class TestAJudgedBarNamesItsTier:
    def test_a_judged_bars_verdicts_carry_the_tier_of_the_judges_that_scored_each_cell(self) -> None:
        bundle = _bundle(BarOverride(measure_id=TOYHOST_JUDGED_DIMENSION, threshold=3.0, direction="higher_is_better"))

        (bar,) = [bar for bar in bundle.bar_adjudications if bar.measure_id == TOYHOST_JUDGED_DIMENSION]
        (judged,) = [measure for measure in bundle.judged_measures if measure.name == TOYHOST_JUDGED_DIMENSION]
        arm_tiers = {(arm.variant_key, arm.apparatus_class_id): arm.evidence_tier for arm in judged.arms}
        assert bar.state == "adjudicated" and bar.verdicts
        for verdict in bar.verdicts:
            assert verdict.judge_evidence_tier is not None
            assert verdict.judge_evidence_tier == arm_tiers[(verdict.variant_key, verdict.apparatus_class_id)]

    def test_a_measured_bars_verdicts_carry_no_tier(self) -> None:
        (bar,) = [bar for bar in _bundle().bar_adjudications if bar.measure_id == "field_accuracy"]

        assert bar.state == "adjudicated" and bar.verdicts
        assert all(verdict.judge_evidence_tier is None for verdict in bar.verdicts)


def _gate(*bars: BarOverride) -> None:
    campaign, _storage = toyhost_campaign()
    assert campaign.declared_design is not None
    design = campaign.declared_design.model_copy(update={"bars": list(bars)})
    refuse_an_undeclarable_design(design, behavior=campaign.behavior, template=None, profile=toyhost_profile())


class TestTheGateAdmitsTheBarTheFrontierReads:
    """A declared bar on pass^k is read now, by the frontier, so authoring no longer refuses it as never read."""

    def test_a_declared_bar_on_pass_hat_k_is_admitted(self) -> None:
        _gate(_on_pass_hat_k())

    def test_one_outside_pass_hat_ks_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            _gate(_on_pass_hat_k(1.5))

    def test_one_read_the_wrong_way_round_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"pass_hat_k is declared higher-is-better"):
            _gate(BarOverride(measure_id=FRONTIER_RANKING_MEASURE, threshold=0.5, direction="lower_is_better"))

    def test_another_composite_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match=r"mean_composite"):
            _gate(BarOverride(measure_id="mean_composite", threshold=0.5, direction="higher_is_better"))
