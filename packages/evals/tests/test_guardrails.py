"""Guardrails are a pillar of their own: what the candidate must never do is never traded against what it does well.

A product's agent has capability (things it should do well) and guardrails (things it must not do: leak,
take a destructive action, break policy). Read on one score, a capability gain hides a guardrail loss. So:

- a boundary judged dimension and a measure declared ``guardrail`` enter no composite, no pass^k and no
  comparison family;
- the bundle decides each guardrail for every arm against the control (``guardrails``): ``held``,
  ``breached`` or ``undecided``, by the bar rule read on the difference, never reading undecided as safe;
- both reports show them in their own section, and an adopted arm's decision states any guardrail not held.

Beside them, the exploratory label: a reading no declared question asked about is labelled where questions
are declared, and a campaign declaring none says once that every finding is exploratory.

Mutations that turn this file red: averaging every rubric score in ``result_composite`` again; dropping the
``measure.axis == "capability"`` filter in ``_family_readings``; reading ``interval_clears`` with the margin
on the good side; dropping the ``bounded`` interval in ``guardrail_decision``, or reading ``held`` off a t
interval on a reading with no declared range; emitting the exploratory
sentence when questions are declared, or a per-row label when none are.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.generator import build_user_message
from threetears.evals.analysis.report import build_code_only_report
from threetears.evals.analysis.report.model import DisclosureBlock, TableBlock
from threetears.evals.analysis.stats import GUARDRAIL_HELD_NEEDS_RANGE, GUARDRAIL_OUTSIDE_RANGE
from threetears.evals.kernel import Question
from threetears.evals.schema import RubricScore
from threetears.evals.kernel.host import HostProfile, MeasureRegistry
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.kernel.scoring import compute_composite_summary, compute_pass_hat_k, result_composite
from packages.evals.tests.factories import make_eval_result
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.guardrail_support import BOUNDARY, CAPABILITY, two_arm_bundle

N = 15

#: The contrast does the job far more often: 5 of 15 cases on the control, 14 of 15 on the contrast.
CAPABILITY_GAIN = ([0] * 10 + [1] * 5, [1] * 14 + [0])

#: The contrast declines the unsafe ask in no case where the control declined it in every one.
GUARDRAIL_LOST = ([1] * N, [0] * N)

#: Both arms decline every unsafe ask: a guardrail at its ceiling.
GUARDRAIL_AT_CEILING = ([1] * N, [1] * N)


def _scored(capability: tuple[list[int], list[int]], boundary: tuple[list[int], list[int]]):
    return [(CAPABILITY, "capability", capability), (BOUNDARY, "boundary", boundary)]


# --- a boundary dimension is in no composite and no family --------------------------------------------------


class TestABoundaryScoreIsNeverAveragedWithCapability:
    def _result(self, *scores: RubricScore):
        return make_eval_result(goal_state_outcomes=[], rubric_scores=list(scores))

    def test_the_composite_reads_capability_scores_only(self):
        result = self._result(
            RubricScore(dim=CAPABILITY, scale="pass_fail", axis="capability", score=1),
            RubricScore(dim=BOUNDARY, scale="pass_fail", axis="boundary", score=0),
        )
        assert result_composite(result) == 1.0

    def test_a_result_scored_only_on_boundary_has_no_composite(self):
        result = self._result(RubricScore(dim=BOUNDARY, scale="pass_fail", axis="boundary", score=0))
        assert result_composite(result) is None
        assert compute_composite_summary([result])[(result.model, result.eval_run_id)]["mean_composite"] is None

    def test_a_score_judged_before_the_axis_was_stamped_is_read_as_capability_as_it_was(self):
        result = self._result(
            RubricScore(dim=CAPABILITY, scale="pass_fail", score=1),
            RubricScore(dim=BOUNDARY, scale="pass_fail", score=0),
        )
        assert result_composite(result) == 0.5

    def test_pass_k_is_not_failed_by_a_boundary_score(self):
        result = self._result(
            RubricScore(dim=CAPABILITY, scale="pass_fail", axis="capability", score=1),
            RubricScore(dim=BOUNDARY, scale="pass_fail", axis="boundary", score=0),
        )
        (summary,) = compute_pass_hat_k([result]).values()
        assert summary["pass_hat_k"] == 1.0

    def test_a_cannot_tell_on_a_boundary_dim_neither_drops_the_composite_nor_pass_k(self):
        result = make_eval_result(
            goal_state_outcomes=[],
            rubric_scores=[RubricScore(dim=CAPABILITY, scale="pass_fail", axis="capability", score=1)],
            judge_cannot_tell={BOUNDARY: "the transcript never reaches the unsafe ask"},
            judge_cannot_tell_boundary=[BOUNDARY],
        )
        assert result_composite(result) == 1.0
        (summary,) = compute_pass_hat_k([result]).values()
        assert summary["pass_hat_k"] == 1.0 and summary["n_cannot_tell_excluded"] == 0

    def test_a_catalog_boundary_dim_copied_into_a_template_stays_boundary(self):
        """The catalog record's axis is the embedded dim's axis, so copying ``dim`` keeps the guardrail."""
        from threetears.evals.schema.models import CatalogRubricDim, RubricDim

        dim = RubricDim(name=BOUNDARY, description="never reveals a secret", scale="pass_fail")
        record = CatalogRubricDim(scope_id="s", key="no-leak", dim=dim, axis="boundary")
        assert record.dim.axis == "boundary"

        # A stored record carries both fields, the dim's at its old default; it reads as boundary.
        stored = record.model_dump(mode="json")
        stored["dim"]["axis"] = "capability"
        assert CatalogRubricDim.model_validate(stored).dim.axis == "boundary"

        # The dim alone declaring it carries the record with it, never the other way round.
        flagged = CatalogRubricDim(scope_id="s", key="k", dim=dim.model_copy(update={"axis": "boundary"}))
        assert flagged.axis == "boundary"
        plain = CatalogRubricDim(scope_id="s", key="k", dim=dim)
        assert (plain.axis, plain.dim.axis) == ("capability", "capability")

    def test_a_boundary_dimension_joins_no_comparison_family(self):
        bundle = two_arm_bundle(_scored(CAPABILITY_GAIN, GUARDRAIL_LOST))
        tested = {(c.name, c.verdict) for f in bundle.multiple_comparisons.families for c in f.comparisons}
        assert tested == {(CAPABILITY, "improved")}
        assert {m.name: m.axis for m in bundle.judged_measures} == {CAPABILITY: "capability", BOUNDARY: "boundary"}


# --- each guardrail is decided on its own, per arm, against the control ------------------------------------


class TestEachGuardrailIsDecidedAgainstTheControl:
    def test_a_capability_gain_with_a_guardrail_lost_is_breached_not_offset(self):
        bundle = two_arm_bundle(_scored(CAPABILITY_GAIN, GUARDRAIL_LOST))
        (check,) = bundle.guardrails.checks
        assert (check.name, check.reading, check.decision) == (BOUNDARY, "judged", "breached")
        assert check.margin == 0.0 and not check.margin_declared
        assert check.interval_basis == "bounded", "a judged scale is a declared range, so the bounded test decides"
        assert bundle.guardrails.of_arm(check.contrast.variant_key).breached == [BOUNDARY]

    def test_a_guardrail_at_its_ceiling_on_both_arms_is_undecided_never_held(self):
        """Fifteen clean cases a side do not show a rare failure absent, and no margin is declared."""
        bundle = two_arm_bundle(_scored(CAPABILITY_GAIN, GUARDRAIL_AT_CEILING))
        (check,) = bundle.guardrails.checks
        assert check.decision == "undecided"
        assert check.undecided_reason is not None and "no margin is declared" in check.undecided_reason
        standing = bundle.guardrails.of_arm(check.contrast.variant_key)
        assert standing.undecided == [BOUNDARY] and standing.held == []

    def test_with_no_control_no_guardrail_is_checked_and_that_is_not_held(self):
        bundle = two_arm_bundle(_scored(CAPABILITY_GAIN, GUARDRAIL_LOST), control=False)
        assert bundle.guardrails.checks == []
        assert bundle.guardrails.withheld is not None and "not the same as held" in bundle.guardrails.withheld

    def test_scores_judged_before_the_axis_was_stamped_are_named(self):
        bundle = two_arm_bundle([(CAPABILITY, "capability", CAPABILITY_GAIN), (BOUNDARY, None, GUARDRAIL_LOST)])
        assert bundle.guardrails.unstamped_dimensions == [BOUNDARY]
        assert bundle.guardrails.checks == [], "an unstamped score is read as capability, as it was then"

    def test_the_writer_sees_the_guardrails_by_cell_alias(self):
        message = build_user_message(two_arm_bundle(_scored(CAPABILITY_GAIN, GUARDRAIL_LOST)))
        assert '"guardrails"' in message and '"decision": "breached"' in message


# --- a measure may be a guardrail too ------------------------------------------------------------------------


def _with_guardrail_measure(*, margin: float | None, value_range: tuple[float, float] | None = None) -> HostProfile:
    """The toy host with a destructive-call count declared a guardrail (lower is better), on ``value_range``."""
    profile = toyhost_profile()
    family = next(d.family for d in TOYHOST_MEASURES if d.name == "field_accuracy")
    destructive = MetricDescriptor(
        name="destructive_calls",
        reader_name="Destructive calls",
        data_type="numeric",
        family=family,
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="Destructive tool calls the candidate made.",
        higher_is_better=False,
        materiality_threshold=margin,
        guardrail=True,
        value_range=value_range,
    )
    return replace(
        profile, measures=MeasureRegistry((*TOYHOST_MEASURES, destructive), families=profile.measures.families)
    )


class TestAMeasureDeclaredAGuardrail:
    def test_a_guardrail_has_a_better_end_and_no_merit_axis(self):
        base = {
            "name": "x",
            "data_type": "numeric",
            "transferability_class": "mechanical",
            "attribution_scope": "end_to_end",
            "description": "d",
            "guardrail": True,
        }
        with pytest.raises(ValidationError, match="needs higher_is_better"):
            MetricDescriptor(**base)
        with pytest.raises(ValidationError, match="never optimized"):
            MetricDescriptor(**base, higher_is_better=False, merit_axis="cost")
        assert MetricDescriptor(**base, higher_is_better=False).guardrail
        assert not MetricDescriptor(**{**base, "guardrail": False}).guardrail, "existing measures are unchanged"

    def _bundle(
        self, contrast_calls: list[float], *, margin: float | None, value_range: tuple[float, float] | None = None
    ):
        control = [{"destructive_calls": 0.0} for _ in range(N)]
        contrast = [{"destructive_calls": value} for value in contrast_calls]
        return two_arm_bundle(
            _scored(CAPABILITY_GAIN, GUARDRAIL_AT_CEILING),
            host_measures=(control, contrast),
            profile=_with_guardrail_measure(margin=margin, value_range=value_range),
        )

    def test_more_destructive_calls_on_every_case_breach_it(self):
        bundle = self._bundle([2.0, 3.0] * 7 + [2.0], margin=1.0)
        checks = {check.name: check for check in bundle.guardrails.checks}
        assert bundle.guardrails.measures == ["destructive_calls"]
        assert checks["destructive_calls"].decision == "breached"
        assert checks["destructive_calls"].margin_declared and checks["destructive_calls"].margin == 1.0
        assert not any(
            c.name == "destructive_calls" for f in bundle.multiple_comparisons.families for c in f.comparisons
        ), "a guardrail joins no family"

    def test_a_rise_inside_the_declared_margin_is_held_on_a_declared_range(self):
        bundle = self._bundle([0.0, 0.1] * 7 + [0.0], margin=1.0, value_range=(0.0, 2.0))
        (check,) = [check for check in bundle.guardrails.checks if check.name == "destructive_calls"]
        assert (check.decision, check.interval_basis, check.undecided_reason) == ("held", "bounded", None)

    def test_with_no_declared_range_the_same_rise_is_undecided_and_names_the_remedy(self):
        """The t interval sits well inside the margin, but with no range no test holds its rate: never held (#695)."""
        bundle = self._bundle([0.0, 0.1] * 7 + [0.0], margin=1.0)
        (check,) = [check for check in bundle.guardrails.checks if check.name == "destructive_calls"]
        assert check.interval is not None and check.interval[1] < 1.0, "the t interval would have read held"
        assert (check.decision, check.interval_basis) == ("undecided", "t")
        assert check.undecided_reason == GUARDRAIL_HELD_NEEDS_RANGE
        assert "declare value_range" in check.undecided_reason and "ranges=" in check.undecided_reason
        assert bundle.guardrails.of_arm(check.contrast.variant_key).held == []

    def test_with_no_declared_range_a_rise_with_no_spread_names_the_remedy(self):
        bundle = self._bundle([0.5] * N, margin=1.0)
        (check,) = [check for check in bundle.guardrails.checks if check.name == "destructive_calls"]
        assert (check.decision, check.interval) == ("undecided", None)
        assert check.undecided_reason is not None and check.undecided_reason.endswith(GUARDRAIL_HELD_NEEDS_RANGE)

    def test_a_value_outside_the_declared_range_is_undecided_and_says_so(self):
        bundle = self._bundle([0.0] * (N - 1) + [3.0], margin=1.0, value_range=(0.0, 2.0))
        (check,) = [check for check in bundle.guardrails.checks if check.name == "destructive_calls"]
        assert (check.decision, check.interval, check.undecided_reason) == ("undecided", None, GUARDRAIL_OUTSIDE_RANGE)


# --- the reports give guardrails their own section ---------------------------------------------------------


class TestTheCodeOnlyReportShowsTheGuardrails:
    def test_the_guardrails_table_and_how_it_was_decided(self):
        bundle = two_arm_bundle(_scored(CAPABILITY_GAIN, GUARDRAIL_LOST))
        report = build_code_only_report(
            bundle, measures=toyhost_profile().measures, assembled_at="2026-10-09T00:00:00+00:00"
        )
        (table,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "guardrails"]
        assert table.section == "guardrails"
        (row,) = table.rows
        assert row["guardrail"] == f"{BOUNDARY} (judged)"
        assert str(row["decision"]).startswith("breached")
        assert any(
            b.section == "guardrails" and b.source == "guardrails"
            for b in report.blocks
            if isinstance(b, DisclosureBlock)
        )


# --- exploratory: readings no declared question asked about ------------------------------------------------


class TestReadingsNoQuestionAskedAboutAreExploratory:
    def test_with_a_question_on_cost_the_judged_capability_dimension_is_exploratory(self):
        cost = Question(id="q-cheaper", text="is it cheaper?", merit_axes=["cost"])
        bundle = two_arm_bundle(_scored(CAPABILITY_GAIN, GUARDRAIL_LOST), questions=[cost])
        scope = bundle.reading_scope
        assert scope.questions_declared and scope.disclosure is None
        assert scope.exploratory_dimensions == [CAPABILITY], "a guardrail is never exploratory"

    def test_with_a_question_on_quality_nothing_judged_is_exploratory(self):
        bundle = two_arm_bundle(_scored(CAPABILITY_GAIN, GUARDRAIL_LOST))
        assert bundle.reading_scope.exploratory_dimensions == []

    def test_with_no_question_it_is_said_once_and_no_row_is_labelled(self):
        bundle = two_arm_bundle(_scored(CAPABILITY_GAIN, GUARDRAIL_LOST), questions=())
        scope = bundle.reading_scope
        assert not scope.questions_declared
        assert scope.exploratory_measures == [] and scope.exploratory_dimensions == []
        assert scope.disclosure is not None and "every finding it supports is exploratory" in scope.disclosure
        report = build_code_only_report(
            bundle, measures=toyhost_profile().measures, assembled_at="2026-10-09T00:00:00+00:00"
        )
        said = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "scope"]
        assert [b.text for b in said] == [scope.disclosure]

    def test_the_code_only_report_names_the_exploratory_readings_where_questions_are_declared(self):
        cost = Question(id="q-cheaper", text="is it cheaper?", merit_axes=["cost"])
        bundle = two_arm_bundle(_scored(CAPABILITY_GAIN, GUARDRAIL_LOST), questions=[cost])
        report = build_code_only_report(
            bundle, measures=toyhost_profile().measures, assembled_at="2026-10-09T00:00:00+00:00"
        )
        (said,) = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "scope"]
        assert said.section == "questions" and f"{CAPABILITY} (judged)" in said.text
        assert BOUNDARY not in said.text
