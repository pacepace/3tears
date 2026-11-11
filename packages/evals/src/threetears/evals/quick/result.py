"""``Result``: what an eval ran, for one arm or a comparison of many, and what its campaign's report says.

Every eval's run ends in one :class:`Result`, whatever its runner: one arm, or a mapping of arms whose first key is
the control. Everything here is the engine's own path, composed:

- **Each arm is one run**, labelled by its arm name, all into one host and one scope, over one content-addressed case
  set. Two arms with different names are two variants. Named plainly (``{"baseline": a, "terse": b}``), an arm's name
  is the run's level of the host's arm lever (:data:`~threetears.evals.quick.one_call.ARM_LEVER`), every arm at one
  shared candidate model, so the report calls the arm ``candidate=<name>``. Named as ``k=v`` pairs
  (``"prompt=v2, effort=low"``), each pair is one factor of a factorial design: ``model`` is the run's candidate model
  and every other factor is a lever the engine names ``callable.<factor>`` and keys into the variant.
- **Every arm is started in one launch**, so they share one launch's schedule and capacity, and a
  refusal on the last arm leaves none run.
- **The campaign declares its design** — one axis per factor, a controlled stimulus, a commissioned apparatus and the
  repeats each arm ran — through :func:`~threetears.evals.analysis.create_campaign`, which gates the declaration
  against the host's vocabulary as it gates any other. One arm is a campaign of one run, its own control.
- **The control is named from the control arm's run**, by the arm's own name, as the campaign is created: its variant
  key is the one that run's observations carry.
- **The report is the campaign's report**: with no analysis generated, the code-only report, whose contrasts table
  tests every other arm against the control on every reading, Holm-corrected, and says which separated.

**Printed, a result is quiet** (:meth:`Result.render`): the verdict, each arm's scoreboard and readings, the top
misses, and where the runs are stored. What is never quieted is printed whatever is asked: errors, exclusions and
their causes, a run stopped short, unpriced spend, the self-judging warning, and how the judge stands against
people's labels. Every stable disclosure is in the report's Methods and in ``render(full=True)``.

A factor's effect at another level of another factor is read against a second control over the same runs:
:meth:`Result.against` files it as a second campaign, with no run repeated.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from threetears.evals.kernel.offload import run_blocking
from threetears.evals.analysis import (
    DEFAULT_FAIL_ON,
    ConfusionCount,
    GateResult,
    LabelStatistics,
    Verdict,
    VerdictKind,
    gate_verdicts,
    CampaignReport,
    Report,
    TableBlock,
    campaign_report,
    campaign_report_markdown,
    create_campaign,
    get_campaign,
    judge_agreement,
    judge_self_agreement,
    variant_key_of_run,
)
from threetears.evals.kernel import (
    ACCURACY_MEASURE,
    CALIBRATION_MIN_RESULTS,
    CONFUSION_CELL_MEASURE,
    MATCH_MEASURE,
    ArmGuardrails,
    GuardrailMargin,
    GuardrailReadings,
    TierCriterion,
    agreement_statistic,
    calibration_criterion,
    confusion_cell,
    confusion_of,
    separation_criterion,
    tier_of,
)
from threetears.evals.schema import CalibrationRating, EvalResult, RubricScale
from threetears.evals.schema.models import DECLARED_NAME_CHARS
from threetears.evals.schema.aspect_label import aspect_label_for
from threetears.evals.kernel.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.kernel.text_block import one_line
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.bundle.trials import trial_order
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.confusion import confusion_matrix, label_statistics
from threetears.evals.analysis.summary import CaseResult, DimensionSummary, EvalSummary
from threetears.evals.analysis.quantities import PERCENT
from threetears.evals.kernel.metrics import METRIC_DESCRIPTORS, MetricDescriptor, reads_as_rate, reads_as_share
from threetears.evals.analysis.report.cost_unit import MoneyRuler, money_ruler
from threetears.evals.analysis.report.figures import report_rulers
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.blocks import DisclosureBlock
from threetears.evals.analysis.report.build import (
    aspect_verdict,
    build_code_only_report_in_forms,
    contrast_rows,
    report_cell_labels,
)
from threetears.evals.analysis.report.campaign.names import measure_name_of
from threetears.evals.analysis.report.verdict import VerdictLine
from threetears.evals.analysis.report.words import campaign_arm_tags, judged_heading, stratum_names
from threetears.evals.quick.graders import UNUSABLE_ANSWER, expected_label, predicted_label
from threetears.evals.quick.judged import Judge, JudgeExamples
from threetears.evals.quick.one_call import ARM_LEVER, CALLABLE_KIND
from threetears.evals.run import get_result_trace, list_results

#: Who an eval's campaign and its control are recorded as created by, unless the caller says.
CREATED_BY = "eval"


#: The factors that are levers of the engine or the host, never of the callable kind: no kind prefix names them.
_UNPREFIXED = frozenset({CANDIDATE_MODEL_LEVER, ARM_LEVER})

#: How many misses the quiet print shows; the rest are in the report and :meth:`Result.misses`.
QUIET_MISSES = 5

#: The file a judged result writes beside its report for a person to label: each judged answer, the verdict withheld.
TO_LABEL = "to_label.csv"

#: The columns of :data:`TO_LABEL`, which ``Judge(labels=...)`` reads back once ``label`` is filled in.
LABEL_COLUMNS: tuple[str, ...] = (
    "case",
    "repeat",
    "dimension",
    "case_material",
    "answer",
    "label",
    "reason",
    "rater",
    "run_id",
    "result_id",
    "output_fingerprint",
    "criterion_fingerprint",
)

#: How many characters of an answer a miss line shows where the graded field cannot be read off it.
_MISS_ANSWER_CHARS = 120

#: What a label is read as, on a pass/fail dimension.
_PASS_FAIL = {"pass": 1, "fail": 0, "1": 1, "0": 0, "true": 1, "false": 0}


def coordinates(arm: str) -> dict[str, str] | None:
    """An arm named as ``k=v`` pairs, by factor (``"prompt=v2, effort=low"`` is two factors), or ``None`` for a name.

    :raises ValueError: A name that mixes pairs and plain words, or names one factor twice.
    """
    parts = [part.strip() for part in arm.split(",")]
    if not any("=" in part for part in parts):
        return None
    pairs: dict[str, str] = {}
    for part in parts:
        factor, equals, level = part.partition("=")
        if not equals or not factor.strip() or not level.strip():
            raise ValueError(
                f"the arm {arm!r} names its factors as k=v pairs, and {part!r} is not one: name every factor at a "
                'level ("prompt=v2, effort=low"), or name the arm plainly'
            )
        if factor.strip() in pairs:
            raise ValueError(f"the arm {arm!r} names {factor.strip()!r} twice; each factor is at one level")
        pairs[factor.strip()] = level.strip()
    return pairs


@dataclass(frozen=True)
class JudgeCalibration:
    """How a judge's scores on one dimension agree with a person's labels, and the evidence tier that bears.

    :ivar dimension: The judged dimension.
    :ivar labels: The labels paired with a judge score.
    :ivar kappa: Cohen's kappa on pass/fail, the quadratic-weighted one on 1-5; ``None`` when undefined.
    :ivar lower: The kappa's one-sided 95% lower bound; ``None`` when it has none.
    :ivar upper: Its one-sided 97.5% upper bound; ``None`` when it has none.
    :ivar calibration: How the calibration criterion reads: ``met``, ``not_met``, ``undecided`` or ``insufficient``.
    :ivar tier: The evidence tier the judge's readings bear: ``calibrated``, ``separation``, ``incidental`` or
        ``undetermined``.
    """

    dimension: str
    labels: int
    kappa: float | None
    lower: float | None
    upper: float | None
    calibration: str
    tier: str

    def line(self) -> str:
        """The calibration as one line."""
        if self.kappa is None:
            figure = "kappa undefined"
        else:
            forms = PageForms.off_page()
            # How far the judge agrees with people is no measure of the eval's, so it is keyed to none.
            figure = f"kappa {forms.reading(self.kappa).text}"
            if self.lower is not None:
                figure += f" (lower bound {forms.reading(self.lower).text})"
        return f"{self.dimension}: {self.labels} label(s), {figure}, calibration {self.calibration}; tier {self.tier}"


@dataclass(frozen=True)
class Result:
    """What an eval ran — one arm, or every arm of a comparison — and what its campaign's report says.

    ``print(result)`` is the quiet summary (:meth:`render`); :meth:`write` writes the report.

    :ivar name: The eval's name, which titles its campaign and report.
    :ivar scope_id: The scope the campaign and its runs are stored in.
    :ivar control: The arm every other arm is tested against: the runner mapping's first key, or the one arm.
    :ivar arms: Each arm's run summary, by arm name, in the order the arms were given.
    :ivar report: The eval's one aspect's page, laid out over the bundle of the page's one fold (:attr:`page`) with
        each arm named by its variant: every reading here is read off its bundle, verdicts and blocks.
    :ivar host: The host the runs and the campaign are stored in, for reading them further.
    :ivar campaign_id: The campaign holding every arm's run.
    :ivar page: The campaign's page, as :func:`~threetears.evals.analysis.campaign_report` reads it: what
        :meth:`write` writes and every other door serves.
    :ivar factors: The factors each arm name names a level of, in order: ``("candidate",)`` when the arms are named
        plainly on the arm lever, ``("model",)`` when the one arm or the arms are named as models.
    :ivar store: Where the runs are stored, in words: the file, or that they are held in memory.
    :ivar fields: The eval's field graders, in the order listed; the classifier is the first not declared a guardrail.
    :ivar grader_readings: The measure each grader reads, in the order the eval lists its graders: the classifier's
        ``accuracy``, any other field's or function's own name, a judge's every dimension. A comparison's first line
        leads on the first of these the contrasts tested; empty when the result was built without them.
    :ivar question: The one sentence the comparison answers, as the eval declared it and its campaign asks it; ``None``
        when none was declared, and every reading is a lead.
    :ivar margin: How far the pass rate may fall below the control before it is shown worse, in the measure's units (a
        share of 0 to 1), as the eval declared it: a non-inferiority question's. ``None`` declares none: a question
        with no margin asks whether a candidate does better (a superiority question), read against zero.
    :ivar judge: The eval's judge, or ``None``.
    :ivar disclosures: What the run must say about how it was run, beside its readings: a tool recording replayed or
        made, a logged runner's file.
    :ivar calibrations: How the judge agrees with the labels ``Judge(labels=...)`` names, per dimension
        (:func:`calibrate`); empty with no judge, no labels, or no label that pairs.
    :ivar judge_examples: How the judge graded its examples (``Judge(examples=...)``) in the run, each on every
        repeat, beside its agreement with the labels and never pooled into it; ``None`` with no judge, or a judge
        with no examples.
    :ivar to_label: Every judged answer on every dimension, as a row of :data:`TO_LABEL`, the verdict withheld.
    :ivar contrast_arms: The arm each row of the report's contrasts table tests, by name, row for row; ``None`` for a
        row no arm's variant matches.
    :ivar contrast_measures: The key of the measure each row of the contrasts table reads, row for row.
    :ivar kind: The kind every arm ran — the callable kind, or the judged one — whose levers the campaign's axes name.
    :ivar guardrail_readings: Every guardrail, decided for each arm against the control; empty when none is declared.
    :ivar arm_variants: Each arm's variant key, by name: what the bundle names the arm by.
    :ivar labelled_arms: Each arm's name, by the words the report names it by (``baseline``).
    """

    name: str
    scope_id: str
    control: str
    arms: dict[str, EvalSummary]
    report: Report
    host: EvalHost
    campaign_id: str
    page: CampaignReport
    factors: tuple[str, ...] = (CANDIDATE_MODEL_LEVER,)
    store: str = ""
    fields: tuple[str, ...] = ()
    judge: Judge | None = field(default=None, repr=False, compare=False)
    disclosures: tuple[str, ...] = ()
    calibrations: tuple[JudgeCalibration, ...] = ()
    judge_examples: JudgeExamples | None = None
    to_label: tuple[dict[str, str], ...] = field(default=(), repr=False, compare=False)
    contrast_arms: tuple[str | None, ...] = field(default=(), repr=False, compare=False)
    contrast_measures: tuple[str | None, ...] = field(default=(), repr=False, compare=False)
    kind: str = field(default=CALLABLE_KIND, repr=False, compare=False)
    guardrail_readings: GuardrailReadings = field(default_factory=GuardrailReadings, repr=False, compare=False)
    arm_variants: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    labelled_arms: dict[str, str | None] = field(default_factory=dict, repr=False, compare=False)
    grader_readings: tuple[str, ...] = field(default=(), repr=False, compare=False)
    question: str | None = None
    margin: float | None = None

    # --- one arm, or the arm named ----------------------------------------------------------------------------------

    @property
    def summary(self) -> EvalSummary:
        """The one arm's run summary.

        :raises ValueError: The result is a comparison: name the arm (``result.arms["baseline"]``).
        """
        return self._arm(None)

    @property
    def run_id(self) -> str:
        """The one arm's run id.

        :raises ValueError: The result is a comparison: read :attr:`run_ids`.
        """
        return self._arm(None).run_id

    @property
    def run_ids(self) -> dict[str, str]:
        """Each arm's run id, by arm name."""
        return {arm: summary.run_id for arm, summary in self.arms.items()}

    @property
    def template_id(self) -> str | None:
        """The template every arm ran: one, since every arm ran the same cases."""
        return self.arms[self.control].template_id

    def results(self, arm: str | None = None) -> list[CaseResult]:
        """Every result of one arm: each case's answer on each repeat, its grades, and why it failed or was excluded.

        :param arm: The arm, by name; ``None`` for the one arm of a result that has one.
        :ptype arm: str | None
        :return: The arm's results, as :meth:`~threetears.evals.analysis.summary.EvalSummary.results` reads them.
        :rtype: list[CaseResult]
        :raises ValueError: ``arm`` names no arm, or is ``None`` on a comparison.
        """
        return self._arm(arm).results()

    def misses(self, arm: str | None = None) -> list[CaseResult]:
        """The results one arm missed, each saying why.

        :param arm: The arm, by name; ``None`` for the one arm of a result that has one.
        :ptype arm: str | None
        :return: The arm's misses, as :meth:`~threetears.evals.analysis.summary.EvalSummary.misses` reads them.
        :rtype: list[CaseResult]
        :raises ValueError: ``arm`` names no arm, or is ``None`` on a comparison.
        """
        return self._arm(arm).misses()

    def _arm(self, arm: str | None) -> EvalSummary:
        if arm is None:
            if len(self.arms) != 1:
                raise ValueError(
                    f"this result compares {len(self.arms)} arms ({', '.join(map(repr, self.arms))}); name the arm"
                )
            return next(iter(self.arms.values()))
        _refuse_an_unknown_arm(self.arms, arm)
        return self.arms[arm]

    @property
    def _classifier(self) -> str | None:
        """The field the eval classifies on: the first field graded as no measure of its own name, since every other
        field (one listed after it, or a guardrail listed before it) is graded as a pass/fail of its name."""
        graded = {measure.name for summary in self.arms.values() for measure in summary.measures}
        return next((name for name in self.fields if name not in graded), None)

    # --- labels and calibration --------------------------------------------------------------------------------------

    def labels(
        self, grader: str | None = None, arm: str | None = None
    ) -> tuple[list[ConfusionCount], list[LabelStatistics]]:
        """One field grader's confusion matrix and each label's precision, recall and F1.

        The classifier's (the first field not declared a guardrail) are the run's own, as its summary carries them; any other field's are
        counted the same way, over the same results, from each answer's label in that field.

        :param grader: The field, by name; ``None`` for the classifier.
        :ptype grader: str | None
        :param arm: The arm, by name; ``None`` for the one arm of a result that has one.
        :ptype arm: str | None
        :return: The confusion matrix, by expected then predicted label, and each label's statistics, by label.
        :rtype: tuple[list[ConfusionCount], list[LabelStatistics]]
        :raises ValueError: No field grader of that name (or none at all), or an arm the result does not have.
        """
        summary = self._arm(arm)
        if not self.fields:
            raise ValueError("this eval grades no field, so it has no labels; a field grader is one named by a string")
        classifier = self._classifier
        name = (classifier or self.fields[0]) if grader is None else grader
        if name not in self.fields:
            raise ValueError(f"{name!r} is no field grader of this eval; its fields are {', '.join(self.fields)}")
        if name == classifier and summary.confusion:
            return list(summary.confusion), list(summary.labels)
        cells: dict[str, int] = {}
        observations: list[tuple[str, str]] = []
        for result in summary.results():
            if result.outcome == "excluded" or not isinstance(result.input, dict):
                continue
            predicted = UNUSABLE_ANSWER if result.outcome == "failed" else predicted_label(result.answer, name)
            cell = confusion_cell(expected_label(result.input, name), predicted)
            cells[cell] = cells.get(cell, 0) + 1
            observations.append((cell, result.case))
        return confusion_matrix(cells), label_statistics(observations)

    def calibration_lines(self) -> list[str]:
        """How the judge stands against people, as the quiet print says it; empty for an unjudged eval."""
        judge = self.judge
        if judge is None:
            return []
        if judge.labels is None:
            return [
                f"judge: 0 labels: judge uncalibrated (Judge(labels=...) names a person's labels){self._next_step()}"
            ]
        if not judge.labels.is_file():
            return [f"judge: 0 labels: judge uncalibrated ({judge.labels.name} is not there yet){self._next_step()}"]
        read = self.calibrations
        # The sheet by its name, never the path it was read from: a path is the machine's, not the reading's.
        if not read:
            return [f"judge: 0 labels paired from {judge.labels.name}: judge uncalibrated{self._next_step()}"]
        return [f"judge against {judge.labels.name}: {entry.line()}" for entry in read]

    def _next_step(self) -> str:
        """What an uncalibrated judge's line ends with: the step that calibrates it, read off the sheet it is taken on.

        The calibrated tier's floor is :data:`~threetears.evals.kernel.CALIBRATION_MIN_RESULTS` labelled results per
        dimension, and :data:`TO_LABEL` holds one row per judged answer per dimension, so the step says how many rows
        the sheet holds for each, and, where the fewest fall short of the floor, how many more judged answers it needs.
        """
        rows = {dim.name: 0 for dim in self.judge.dims} if self.judge is not None else {}
        for row in self.to_label:
            rows[row["dimension"]] = rows.get(row["dimension"], 0) + 1
        fewest, most = min(rows.values(), default=0), max(rows.values(), default=0)
        held = f"{fewest}" if fewest == most else f"{fewest} to {most}"
        sheet = f"the {TO_LABEL} the report writes, which holds {held} per dimension"
        unseen = "without looking at the judge's verdict"
        if fewest >= CALIBRATION_MIN_RESULTS:
            return f"; next step: rate about {CALIBRATION_MIN_RESULTS} outputs of each dimension in {sheet}, {unseen}"
        return (
            f"; next step: rate every output in {sheet}, {unseen}; the calibrated tier's floor is "
            f"{CALIBRATION_MIN_RESULTS} per dimension, so it needs {CALIBRATION_MIN_RESULTS - fewest} more judged "
            "answers per dimension"
        )

    # --- the report --------------------------------------------------------------------------------------------------

    def __str__(self) -> str:
        """The quiet summary (:meth:`render`)."""
        return self.render()

    def render(self, *, full: bool = False) -> str:
        """The result as text: quiet by default, everything with ``full=True``.

        Quiet is the verdict, each arm's scoreboard and readings, the top misses and where the runs are stored. Never
        quieted: errors, exclusions and their causes, a run stopped short, unpriced spend, the self-judging warning,
        and how the judge stands against people's labels.

        :param full: Every arm's whole summary and the whole report, Methods included, as Markdown.
        :ptype full: bool
        :return: The text.
        :rtype: str
        """
        # Every arm's total in one form, the page's for its test costs (plain dollars to two significant figures), and
        # every role's share of one in another.
        totals = money_ruler((summary.total_cost_usd for summary in self.arms.values()), per=0)
        roles = money_ruler(
            (role.cost_usd for summary in self.arms.values() for role in summary.spend if role.counted), per=0
        )
        forms = PageForms.off_page()
        if full:
            return "\n\n".join(
                [
                    *(self._full_summary(forms, summary, totals, roles) for summary in self.arms.values()),
                    campaign_report_markdown(self.page),
                ]
            )
        one_arm = len(self.arms) == 1
        lines = [self._verdict_line()]
        # How far its judge can be trusted is read with the verdict: against people's labels, then on its examples.
        judged = self.calibration_lines() + (self.judge_examples.lines() if self.judge_examples is not None else [])
        if one_arm:
            # One arm's verdict is how its cases came out, and how far its judge can be trusted is read with it.
            lines += judged
        for arm, summary in self.arms.items():
            label = "" if len(self.arms) == 1 else f"{arm}{' (control)' if arm == self.control else ''}: "
            lines.append(f"{label}{summary.results_text}")
            # an operator's reason for a cancel, a predicted label, a refused check's reason: each can run over lines
            lines += [f"  {one_line(line)}" for line in (summary.stopped_text,) if line is not None]
            lines += [f"  {one_line(line)}" for line in self._measure_texts(forms, summary, arm)]
            if summary.confusion:
                lines.append("  per label:")
                lines += [f"    {one_line(_label_text(forms, statistics))}" for statistics in summary.labels]
            lines += [f"  {one_line(_dimension_text(forms, dimension))}" for dimension in summary.judged]
            lines += [f"  {one_line(line)}" for line in summary.goal_check_lines]
            # What it cost, priced or not, and the warnings no quiet print drops.
            for line in (
                _total_spend(forms, summary, totals, roles),
                summary.self_judging_warning,
                summary.model_identity_warning,
            ):
                if line is not None:
                    lines.append(f"  {one_line(line)}")
            lines += [f"  error: {one_line(error)}" for error in summary.errors]
            lines += [f"  excluded: {cause}" for cause in _exclusion_causes(summary)]
        if not one_arm:
            lines += judged
        order = self._page_order()
        misses = sorted(
            ((arm, miss) for arm, summary in self.arms.items() for miss in summary.misses()),
            key=lambda pair: order(pair[0], pair[1]),
        )
        if misses:
            lines.append(f"top misses ({min(len(misses), QUIET_MISSES)} of {len(misses)} failed trials):")
            for arm, miss in misses[:QUIET_MISSES]:
                who = "" if len(self.arms) == 1 else f"{arm}, "
                lines.append(f"  case {miss.case} ({who}repeat {miss.repeat}): {one_line('; '.join(self._why(miss)))}")
        lines += self.disclosures
        lines.append(f"stored in {self.store}, scope {self.scope_id!r}")
        return "\n".join(lines)

    def _why(self, miss: CaseResult) -> list[str]:
        """Why a miss missed, as the quiet print says it: a classifier's wrong label as the graded field's value, never
        the whole answer it was read from (a runner's answer object can run to hundreds of characters).

        A scored result whose classifier label did not match carries that reason first
        (:meth:`~threetears.evals.analysis.summary.CaseResult.of`); every other reason is printed as the summary has it.
        The label is the one the run graded, read off the result's confusion cell; an answer it read no label off is
        shown cut to :data:`_MISS_ANSWER_CHARS`, as is a label that runs past it.
        """
        reasons = list(miss.missed_because)
        field_name, answer = self._classifier, miss.answer
        if field_name is None or miss.outcome != "scored" or miss.scores.get(MATCH_MEASURE) is not False or not reasons:
            return reasons
        cell = miss.scores.get(CONFUSION_CELL_MEASURE)
        read = confusion_of(cell) if isinstance(cell, str) else None
        label = read[1] if read is not None else predicted_label(answer, field_name)
        if label == UNUSABLE_ANSWER:
            graded = f"answered no {field_name} it can be graded on: {_compact(answer)}"
        elif isinstance(answer, Mapping):
            graded = f"answered {field_name} {_cut(repr(label))}"
        else:
            # The answer is its own label, so the label is what is shown.
            graded = f"answered {_cut(repr(label))}"
        expected = "" if miss.expected is None else f", expected {miss.expected!r}"
        return [f"{graded}{expected}", *reasons[1:]]

    def _page_order(self) -> Callable[[str, CaseResult], tuple[str, int, str, int, int]]:
        """Where an arm's result stands in the page's Trials order (:func:`trial_order`), so the console lists misses
        as the page does: case by case, then arm as the bundle lists them, then repeat."""
        arm_of_variant = {variant: key for key, variant in self.arm_variants.items()}
        position = {
            arm_of_variant.get(row.variant_key): index for index, row in enumerate(self.report.bundle.scoreboard)
        }
        return lambda arm, result: trial_order(result.case_id, position.get(arm, len(position)), result.repeat)

    def _guardrail_said(self, arm: str) -> dict[str, str]:
        """What this comparison came to on each guardrail for ``arm``, by measure name, as its measure line says it:
        ``guardrail undecided against v1``; the control's, that the other arms are decided against it."""
        if arm == self.control:
            return {
                check.name: "the guardrail the other arms are decided against"
                for check in self.guardrail_readings.checks
            }
        return {row["measure_id"]: f"guardrail {row['outcome']} against {self.control}" for row in self.guardrails(arm)}

    def _measure_texts(self, forms: PageForms, summary: EvalSummary, arm: str) -> list[str]:
        """Each measure's line, the classifier's ``match`` under its field's name and its matrix's count left out.

        In a comparison a guardrail's line says what the comparison decided on it for ``arm``; alone, that a guardrail
        is decided only in one.
        """
        texts: list[str] = []
        lines = (
            summary.measure_lines if len(self.arms) == 1 else summary.compared_measure_lines(self._guardrail_said(arm))
        )
        held = {measure.name: measure for measure in summary.measures}
        catalog = self.report.bundle.measure_catalog
        for measure in lines:
            if measure.name == CONFUSION_CELL_MEASURE or measure.state == "not_carried":
                continue
            # prose-canary: allow — MeasureLine.text is the summary's own line, not anything a model wrote
            said = measure.text.removeprefix(measure.name)
            figures = held.get(measure.name)
            descriptor = catalog.get(measure.name) or METRIC_DESCRIPTORS.get(measure.name)
            if measure.state == "measured" and figures is not None:
                counted = _counted_rate(measure.name, descriptor)
                rate = counted or (descriptor is not None and reads_as_share(descriptor))
                level = self._page_rate(measure.name, figures.mean) if rate else None
                said = _measured_text(
                    forms, measure.name, said, figures.mean, figures.n, rate=rate, counted=counted, level=level
                )
            # Each measure under the name the page gives it, never its key.
            texts.append(f"{self._printed_name(measure.name)}{said}")
        if (omitted := summary.uncarried_text) is not None:
            texts.append(omitted)
        return texts

    def _full_summary(self, forms: PageForms, summary: EvalSummary, totals: MoneyRuler, roles: MoneyRuler) -> str:
        """An arm's whole summary as the full print writes it: the run summary's lines
        (:meth:`~threetears.evals.analysis.summary.EvalSummary.render`), each figure in the form the quiet print and
        the page state it (the review's P43, A4): a measure, its strata's included, under the page's name, a rate in
        percent with its count of trials; each label's precision and recall in percent with their intervals; a judged
        dimension as the quiet print says it; every amount of spend in the form of its set, a total as the page's test
        cost. Every other line is the summary's own."""
        said: dict[str, str] = {}
        said.update(
            zip(summary.label_lines, (_label_text(forms, statistics) for statistics in summary.labels), strict=True)
        )
        said.update(
            zip(
                summary.dimension_lines,
                (_dimension_text(forms, dimension) for dimension in summary.judged),
                strict=True,
            )
        )
        if summary.total_spend_text is not None and summary.total_cost_usd is not None:
            total = forms.money_on(totals, summary.total_cost_usd).text
            said[summary.total_spend_text] = f"total spend: {total}, the results' cost"
        spent = {role.role: role.cost_usd for role in summary.spend if role.cost_usd is not None}
        lines = []
        for line in summary.render().splitlines():
            body = line.lstrip()
            indent = line[: len(line) - len(body)]
            if body in said:
                body = said[body]
            elif (measured := _FULL_MEASURE.fullmatch(body)) is not None:
                body = self._full_measure(forms, measured)
            elif (role := _ROLE_SPEND.fullmatch(body)) is not None and role["role"] in spent:
                body = f"{role['role']} spend: {forms.money_on(roles, spent[role['role']]).text}{role['rest']}"
            lines.append(indent + body)
        return "\n".join(lines)

    def _full_measure(self, forms: PageForms, measured: re.Match[str]) -> str:
        """A run summary's measure line as the full print writes it: under the page's name, a rate in percent with its
        count of trials, any other measure with its trials named (:func:`_measured_text`)."""
        name = measured["name"]
        descriptor = self.report.bundle.measure_catalog.get(name) or METRIC_DESCRIPTORS.get(name)
        counted = not measured["unit"] and _counted_rate(name, descriptor)
        rate = counted or (not measured["unit"] and descriptor is not None and reads_as_share(descriptor))
        mean = float(measured["mean"]) if rate else 0.0
        level = self._page_rate(name, mean) if rate else None
        said = _measured_text(
            forms, name, measured["said"], mean, int(measured["n"]), rate=rate, counted=counted, level=level
        )
        return f"{self._printed_name(name)}{said}"

    def _verdict_line(self) -> str:
        """The first line: the eval's name, then its aspect's verdict line as its page prints it
        (:func:`~threetears.evals.analysis.report.build.aspect_verdict`): one arm's pass rate as its count, or what the
        comparison came to on the measure the eval's first grader reads, what else the evidence tells apart, and each
        guardrail an arm is not shown safe on. The page, its lead, this line and the exit line say one sentence.
        """
        return f"{self.name}: {self._verdict().text}"

    def _verdict(self) -> VerdictLine:
        """The eval's verdict line, as its aspect's page opens with it, led on its graders' order."""
        return aspect_verdict(self.report.bundle, measure_order=self.grader_readings, forms=self._aspect_forms())

    def _page_rate(self, measure: str, rate: float | None) -> str | None:
        """A measure's rate as :attr:`page` prints an arm's level of it (one form per measure, number rule 1): at the
        places the code page's scorecard reads it at where that states it (one arm's campaign), else the places the
        aspect's page reads its contrasts' levels at (a comparison's); None where neither states it, or for no rate.
        """
        if rate is None:
            return None
        code = PageForms.of_campaign(self.page.readings, {fold.aspect: fold.report for fold in self.page.aspects})
        if any((form.key, form.kind) == (measure, "measure") for form in code.measures):
            return code.level(measure, rate).text
        aspect = self._aspect_forms()
        places = aspect.contrast_places(measure, "measure")
        if places is None:
            return None
        bundle = self.report.bundle
        ruler = report_rulers(bundle_decision_surface(bundle), bundle.measure_catalog).of(measure, "measure")
        return aspect.ruled_level(ruler, rate, places, measure=measure, kind="measure").text

    def _aspect_forms(self) -> PageForms:
        """The aspect page's forms, over the campaign's set of each measure, as the campaign's page reads them: so the
        verdict line, the contrasts' rows and :attr:`report` print every figure as :attr:`page` does."""
        return _aspect_forms(self.page, self.report.bundle)

    def _printed_name(self, measure: str) -> str:
        """A measure as the page names it (:func:`~threetears.evals.analysis.report.campaign.names.measure_name_of`):
        ``Accuracy``, ``Number score``. The classifier's ``match`` is its accuracy, which the page leads on, so it is
        named as the page names that."""
        key = ACCURACY_MEASURE if measure == MATCH_MEASURE and self._classifier is not None else measure
        return measure_name_of(self.report.bundle, key)

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        """Write the campaign's page as one self-contained file, its Markdown, its JSON and its aspect's bundle, and the
        judged answers to label.

        :param out_dir: The directory; created when missing.
        :ptype out_dir: Path | str
        :return: Each file written, by role, as :meth:`~threetears.evals.analysis.CampaignReport.write` names them
            (``html``, ``markdown``, ``json``, ``bundle:<aspect>``), and, for a judged eval, ``to_label``
            (:data:`TO_LABEL`): each judged answer on each dimension, the judge's verdict withheld, for a person to
            label and ``Judge(labels=...)`` to read back.
        :rtype: dict[str, Path]
        """
        written = self.page.write(out_dir)
        if self.judge is not None:
            path = Path(out_dir) / TO_LABEL
            _write_to_label(path, self.to_label)
            written["to_label"] = path
        return written

    def exit_code(self) -> int:
        """What a command line exits with for this result: 0, 1 when a run did not complete, 4 when a gate failed.

        A comparison is gated on its verdicts (:meth:`gate`, the default outcomes); one arm has nothing to gate.
        """
        if any(summary.status != "completed" for summary in self.arms.values()):
            return 1
        if len(self.arms) > 1 and self.report.verdicts and self.gate().failed:
            return 4
        return 0

    def why_failed(self) -> str | None:
        """Why :meth:`exit_code` is 4, in one line and the report's words: the result's sentence, as the page leads
        with it and the first line prints it, then ``the gate failed on ...``, naming each verdict it failed on
        (:meth:`gate`); ``None`` for any other exit code, whose cause the summary already says.
        """
        if self.exit_code() != 4:
            return None
        lead, gate = self._verdict().lead, self.gate()
        # A gate failing only on what the sentence already says names it once, in the sentence.
        if all(f"{failed.heading}: {failed.words}" in lead for failed in gate.failures):
            return f"{lead} The gate failed on it."
        return f"{lead} The gate {gate.why()}."

    def contrasts(self, reading: str | None = None) -> list[dict[str, Any]]:
        """Every contrast against the control the report tested, the ones its page leaves out under the case floor too.

        :param reading: Only the rows on this reading, by its key (``"accuracy"``) or as the report heads it
            (``"Accuracy"``); ``None`` keeps every row.
        :ptype reading: str | None
        :return: One row per arm and reading, keyed ``arm`` (the arm tested, by its key in :attr:`arms`: the name
            you gave it), ``measure_id`` (the key of the measure read, to cite or filter
            on), ``question`` (in the words it was asked), ``reading`` (the measure as the report heads it),
            ``contrast`` (the arm, as the report names it), ``control`` (the control, as the report names it),
            ``control_mean`` and ``arm_mean`` (over the cases the test read), ``cases`` (how many, paired or not,
            and any one side ran that the test left out), ``delta`` (arm minus control), ``interval`` (on the
            delta, simultaneous over the family), ``hedges_g`` (the standardized effect), ``p_adjusted`` (Holm,
            over the campaign's family), ``verdict`` (in words, printed from ``outcome``) and ``outcome`` (the
            typed verdict: ``improved``, ``regressed``, ``equivalent``, ``not_separated`` or ``untested``, which
            code branches on); empty when the report tested nothing.
        :rtype: list[dict[str, Any]]
        """
        rows = contrast_rows(self.report.bundle, forms=self._aspect_forms())
        typed = [verdict for verdict in self.report.verdicts if verdict.kind == "contrast"]
        arms = self.contrast_arms if len(self.contrast_arms) == len(rows) else (None,) * len(rows)
        keys = self.contrast_measures if len(self.contrast_measures) == len(rows) else (None,) * len(rows)
        outcomes = [verdict.outcome for verdict in typed] if len(typed) == len(rows) else [None] * len(rows)
        return [
            {"arm": arm, "measure_id": key, **row, "outcome": outcome}
            for row, arm, key, outcome in zip(rows, arms, keys, outcomes, strict=True)
            if reading is None or reading in (key, row["reading"])
        ]

    def verdicts(self, reading: str | None = None, *, kind: VerdictKind | None = None) -> list[Verdict]:
        """The report's typed verdicts: every contrast against the control, guardrail and bar, as code reads them.

        The source of truth for every verdict the report prints, which is rendered from these: branch on
        :attr:`~threetears.evals.analysis.Verdict.outcome`, never on the words.

        :param reading: Only the verdicts on this reading, by its key (``"accuracy"``) or as the report heads it
            (``"Accuracy"``); ``None`` keeps every one.
        :ptype reading: str | None
        :param kind: Only ``"contrast"``, ``"guardrail"`` or ``"bar"`` verdicts; ``None`` keeps every kind.
        :ptype kind: VerdictKind | None
        :return: The verdicts, guardrails first, then bars, then contrasts, each in its table's row order.
        :rtype: list[Verdict]
        """
        return [
            verdict
            for verdict in self.report.verdicts
            if (reading is None or reading in (verdict.name, verdict.heading))
            and (kind is None or verdict.kind == kind)
        ]

    def gate(self, fail_on: Iterable[str] = DEFAULT_FAIL_ON, *, readings: Iterable[str] | None = None) -> GateResult:
        """Gate a build on this comparison's verdicts, as ``python -m threetears.evals gate`` does on a campaign.

        :param fail_on: The outcomes that fail it (:data:`~threetears.evals.analysis.GATE_TOKENS`). The default,
            :data:`~threetears.evals.analysis.DEFAULT_FAIL_ON`, fails on a regressed contrast, a breached
            guardrail and an undecided guardrail.
        :ptype fail_on: Iterable[str]
        :param readings: Only the verdicts on these readings, by key or heading; ``None`` gates every verdict.
        :ptype readings: Iterable[str] | None
        :return: The gate's result: ``failed`` when a named outcome occurred, ``passed`` only when every verdict was
            decided and none failed, ``undecided`` otherwise — never a pass.
        :rtype: GateResult
        :raises ValueError: A ``fail_on`` name that is no outcome, or a reading no verdict is on.
        """
        return gate_verdicts(self.report.verdicts, fail_on=fail_on, readings=readings)

    def strata(self, arm: str | None = None, *, reading: str | None = None) -> list[dict[str, Any]]:
        """The rows of the report's "By stratum" table: each arm's figures over all cases, then per kind of case.

        Present when the cases declare strata (each case's reserved ``stratum`` key). Each arm opens with a
        ``cases`` row saying how many cases and observations each column rests on, a stratum of fewer than
        ``STRATUM_MIN_CASES`` (10) cases marked too few to read alone, then one row per reading.

        :param arm: Only this arm's rows, by its key in :attr:`arms`; ``None`` keeps every arm's.
        :ptype arm: str | None
        :param reading: Only the rows on this reading, as the report heads it (``"Accuracy"``), or ``"cases"``;
            ``None`` keeps every row.
        :ptype reading: str | None
        :return: One row per arm and reading, keyed ``arm`` (the arm, by its key in :attr:`arms`), ``label`` (the arm as
            the report names it), ``reading``, ``all`` (the figure over every case) and ``strata`` (the figure
            over each stratum's cases alone, by stratum: its declared name, else its id (``edge_case``), as the
            case declared it, never the name the page prints (``edge case``); the cases declaring none under
            ``"(no stratum)"``; a stratum the arm has no case in is absent); empty when no case declares a stratum.
        :rtype: list[dict[str, Any]]
        :raises ValueError: ``arm`` names no arm.
        """
        if arm is not None:
            _refuse_an_unknown_arm(self.arms, arm)
        tables = [block for block in self.report.blocks if isinstance(block, TableBlock) and block.name == "strata"]
        # Each stratum by its key here (its declared name, else its id), from the name its column prints.
        bundle = self.report.bundle
        ids = sorted(
            {held.stratum for cell in bundle.cell_measures for held in cell.strata if held.stratum is not None}
        )
        keyed = {
            printed: bundle.stratum_names.get(stratum, stratum)
            for stratum, printed in stratum_names(ids, bundle.stratum_names).items()
        }
        rows: list[dict[str, Any]] = []
        for table in tables:
            headers = {column.key: keyed.get(column.header, column.header) for column in table.columns}
            for row in table.values():
                label = str(row["arm"]).removeprefix("▸ ")
                key = self.labelled_arms.get(label)
                if (arm is not None and key != arm) or (reading is not None and row["reading"] != reading):
                    continue
                rows.append(
                    {
                        "arm": key,
                        "label": label,
                        "reading": row["reading"],
                        "all": row["all"],
                        "strata": {
                            headers[column]: value
                            for column, value in row.items()
                            if column not in ("arm", "reading", "all") and value is not None
                        },
                    }
                )
        return rows

    def guardrails(self, arm: str | None = None) -> list[dict[str, Any]]:
        """The rows of the report's "Guardrails" table: each guardrail, for each arm.

        A guardrail (``grader(guard=...)``, ``Judge(guard=...)``) is decided on its own for each arm against the
        control, never in the contrasts: ``held`` when the arm is shown no worse than the control by more than the
        margin, ``breached`` when it is shown worse by more, and ``undecided`` otherwise, which is never safe.

        :param arm: Only this arm's rows, by its key in :attr:`arms`; ``None`` keeps every arm's.
        :ptype arm: str | None
        :return: One row per guardrail and arm, keyed ``arm`` (the arm checked, by its key in :attr:`arms`),
            ``measure_id`` (the guardrail's key: a scorer's name or a judged dimension's), ``outcome`` (``held``,
            ``breached`` or ``undecided``), ``contrast`` (the arm, as the report names it), and the table's other
            columns: ``guardrail`` (as the report heads it), ``control_mean`` and ``arm_mean``, ``cases``,
            ``delta`` (arm minus control), ``interval`` (on the delta), ``margin`` and ``decision`` (the outcome in
            words, with why when undecided); empty when no reading is a guardrail or no control resolved.
        :rtype: list[dict[str, Any]]
        :raises ValueError: ``arm`` names no arm.
        """
        if arm is not None:
            _refuse_an_unknown_arm(self.arms, arm)
        rows = [
            row
            for block in self.report.blocks
            if isinstance(block, TableBlock) and block.name == "guardrails"
            for row in block.values()
        ]
        checks = self.guardrail_readings.checks
        if len(checks) != len(rows):
            return []
        arm_of_variant = {variant: key for key, variant in self.arm_variants.items()}
        return [
            {
                "arm": arm_of_variant.get(check.contrast.variant_key),
                "measure_id": check.name,
                "outcome": check.decision,
                "contrast": row["arm"],
                **{key: value for key, value in row.items() if key != "arm"},
            }
            for check, row in zip(checks, rows, strict=True)
            if arm is None or arm_of_variant.get(check.contrast.variant_key) == arm
        ]

    def guardrail_standing(self, arm: str) -> ArmGuardrails:
        """Where one arm stands on every guardrail: breached anywhere, else undecided anywhere, else held.

        An arm with any guardrail ``breached`` is not to be adopted, whatever its contrasts show it gained; one
        ``undecided`` is not known to be safe, which says nothing either way about whether it is.

        :param arm: The arm, by its key in :attr:`arms`.
        :ptype arm: str
        :return: The guardrails the arm breached, those it is undecided on, and those it held, each by key; all empty
            for the control, which no guardrail is checked against itself, and when no reading is a guardrail.
        :rtype: ArmGuardrails
        :raises ValueError: ``arm`` names no arm.
        """
        _refuse_an_unknown_arm(self.arms, arm)
        variant = self.arm_variants.get(arm)
        if variant is None:
            return ArmGuardrails(breached=[], undecided=[], held=[])
        return self.guardrail_readings.of_arm(variant)

    async def against(
        self,
        control: str,
        *,
        name: str | None = None,
        created_by: str = CREATED_BY,
        question: str | None = None,
        margin: float | None = None,
    ) -> Result:
        """The same runs read against another control, or a question sharpened once they ran: a second campaign over
        them, with no run repeated.

        The engine tests every arm against one control, so in a factorial the effect of one factor at the control's
        level of the other is in :attr:`report`, and its effect at another level is read here — the prompt's effect on
        the second model is the contrast against the first prompt on that model. Each campaign Holm-corrects its own
        contrasts, so the two reports are two families, not one.

        A question (or its margin) declared here is asked of runs whose readings already exist, as an exploratory run
        read again once its question sharpens is: never refused, and read as asked, its page saying once that its
        criteria were set after the results there existed, so they were not fixed in advance.

        :param control: The arm to test every other against, by name; this result's own control reads the same arms
            again.
        :ptype control: str
        :param name: The new campaign's name; ``None`` names it after this one and the new control.
        :ptype name: str | None
        :param created_by: Who the campaign and its control are recorded as created by.
        :ptype created_by: str
        :param question: The question the new campaign asks; ``None`` asks this result's own, with its margin.
        :ptype question: str | None
        :param margin: The question's margin, in the measure's units, as :class:`~threetears.evals.Eval`'s ``margin``;
            ``None`` keeps this result's own where no new question is given, and declares none with a new one.
        :ptype margin: float | None
        :return: A result over the same arms and runs, against ``control``.
        :rtype: Result
        :raises ValueError: ``control`` names no arm; a blank question; a margin outside 0 to 1, or with no question
            to read it under.
        :raises ValidationFailedError: The host refuses the campaign's declaration.
        """
        _refuse_an_unknown_arm(self.arms, control)
        if question is not None and not question.strip():
            raise ValueError("an eval's question is filed on its campaign, and this one is blank")
        asked = question if question is not None else self.question
        held = margin if margin is not None or question is not None else self.margin
        refuse_an_unusable_margin(held, asked)

        def declared() -> Result:
            campaign = get_campaign(self.host.storage, self.campaign_id, self.scope_id)
            design = campaign.aspects[0].design
            levels = {
                arm: dict.fromkeys(self.factors, arm) if (pairs := coordinates(arm)) is None else pairs
                for arm in self.arms
            }
            again = declare(
                self.host,
                self.arms,
                self.factors,
                levels=_levels_of(self.arms, self.factors, levels),
                kind=self.kind,
                control=control,
                scope_id=self.scope_id,
                name=name or f"{self.name}, against {control}",
                behavior=campaign.behavior,
                repetitions=design.intended_repetitions if design is not None else None,
                created_by=created_by,
                guardrail_margins=design.guardrail_margins if design is not None else (),
                measure_latency=design.measure_latency if design is not None else False,
                question=asked,
                margin=held,
                measure_order=self.grader_readings,
            )
            return with_judge_examples(
                replace(
                    again,
                    store=self.store,
                    fields=self.fields,
                    judge=self.judge,
                    disclosures=self.disclosures,
                    calibrations=self.calibrations,
                    to_label=self.to_label,
                ),
                self.judge_examples,
            )

        # It reads this result's campaign and files a new one, so it runs off the loop.
        return await run_blocking(self.host.blocking_executor, declared)


def _levels_of(
    arms: Mapping[str, EvalSummary], factors: tuple[str, ...], named: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, str]]:
    """Each arm's level of each factor: its ``k=v`` pairs, its name on the arm lever, or its run's model."""
    levels: dict[str, dict[str, str]] = {}
    for arm, summary in arms.items():
        pairs = dict(named.get(arm, {}))
        if CANDIDATE_MODEL_LEVER in factors and CANDIDATE_MODEL_LEVER not in (coordinates(arm) or {}):
            pairs[CANDIDATE_MODEL_LEVER] = summary.candidate_model
        levels[arm] = {factor: pairs[factor] for factor in factors}
    return levels


def refuse_an_unusable_margin(margin: float | None, question: str | None) -> None:
    """Refuse a margin no reading could carry: one that is no share strictly between 0 and 1, or one with no question
    to read it under.

    :param margin: The margin, in the measure's units (a share of 0 to 1); ``None`` declares none.
    :ptype margin: float | None
    :param question: The question it would be read under; ``None`` asks none.
    :ptype question: str | None
    :raises ValueError: Naming what is wrong with it.
    """
    if margin is None:
        return
    if isinstance(margin, bool) or not isinstance(margin, int | float) or not math.isfinite(margin):
        raise ValueError(f"margin= is a share of the pass rate strictly between 0 and 1, not {margin!r}")
    if not 0 < margin < 1:
        raise ValueError(
            f"margin= is a share of the pass rate strictly between 0 and 1 (0.05 is 5 points), not {margin!r}"
        )
    if question is None:
        raise ValueError(
            "margin= is how far a candidate may fall below the control before the question's answer reads it worse, "
            "and this eval asks no question for it to apply to: pass question= with it, or leave margin out"
        )


def _refuse_an_unknown_arm(arms: Mapping[str, Any], arm: str) -> None:
    if arm not in arms:
        raise ValueError(f"{arm!r} names no arm; the arms are {', '.join(map(repr, arms))}")


#: A measure's line as the summary writes it: its mean, its count, its range, anything left out, and what follows.
_MEASURE_LINE = re.compile(
    r"^: mean (?P<mean>\S+) \(n=(?P<n>\d+), min (?P<low>[^,]+), max (?P<high>[^,)]+)(?P<rest>[^)]*)\)(?P<tail>.*)$"
)


#: A run summary's measure line, its name and what follows it: a measure carried by some result, with its mean.
_FULL_MEASURE = re.compile(
    r"^(?P<name>[^\s:]+(?: [^\s:]+)*?)(?P<said>: mean (?P<mean>-?[\d.]+)(?P<unit>[^\s(]*(?: [^\s(]+)?) \(n=(?P<n>\d+), .*)$"
)

#: A run summary's spend line for one role: ``candidate spend: $0.00175 over 40 call(s)``.
_ROLE_SPEND = re.compile(r"^(?P<role>[\w-]+) spend: [<>]? ?\$[\d.,]+(?P<rest> over .*)$")


def _aspect_forms(page: CampaignReport, bundle: AnalysisContextBundle) -> PageForms:
    """An aspect's page's forms over its campaign's set of each measure, as the campaign's page lays the aspect out
    (:meth:`~threetears.evals.analysis.report.forms.PageForms.of_aspects`, its aspects and the code page's readings)."""
    return PageForms.of_aspects([fold.report.bundle for fold in page.aspects], page.readings)[bundle.aspect]


def _counted_rate(name: str, descriptor: MetricDescriptor | None) -> bool:
    """Whether a measure is a rate counted over trials, which the quiet print states with its count: a label match,
    a measure that reads as a rate (:func:`~threetears.evals.kernel.metrics.reads_as_rate`), or a boolean's share."""
    return (
        name == MATCH_MEASURE
        or reads_as_rate("measure", name, descriptor=descriptor)
        or (descriptor is not None and descriptor.data_type == "boolean")
    )


def _measured_text(
    forms: PageForms,
    measure: str,
    said: str,
    mean: float | None,
    n: int,
    *,
    rate: bool,
    counted: bool = True,
    level: str | None = None,
) -> str:
    """A measure's figures as the quiet print says them, after its name: a rate in percent (number rule 4, as every
    page reads it), with its count of trials where it is counted over them (``: 83%, 75 of 90 trials``), else with
    its trials named (a host's fraction: ``: 38% (12 trials)``), the level as the campaign's page prints it (``level``,
    one form per measure, number rule 1) where the page states it, else as a lone rate; any other measure's mean with
    its trials named (``: mean 950 ms (90 trials, min 700 ms, max 1.2 s)``). Anything the line leaves out, and what
    follows it (a guardrail's standing), is kept."""
    read = _MEASURE_LINE.fullmatch(said)
    if read is None or mean is None:
        return said
    rest, tail = read["rest"].removeprefix(", "), read["tail"]
    if rate:
        if level is None:
            level = forms.in_unit(100 * mean, PERCENT, role="level", measure=measure).text
        if not counted:
            return f": {level} ({n} trials{f', {rest}' if rest else ''}){tail}"
        hits = f"{forms.hits(mean, n, measure=measure).text} of {n} trials"
        return f": {level}, {hits}{f' ({rest})' if rest else ''}{tail}"
    left = f", {rest}" if rest else ""
    return f": mean {read['mean']} ({n} trials, min {read['low']}, max {read['high']}{left}){tail}"


def _proportion_text(
    forms: PageForms,
    name: str,
    rate: float | None,
    hits: int,
    n: int,
    cases: int,
    interval: tuple[float, float] | None,
) -> str:
    """A rate with its count of trials, its cases where they repeat, and its interval, in percent."""
    if rate is None:
        return f"{name} none (no trial)"
    over = f" over {cases} case{'' if cases == 1 else 's'}" if cases < n else ""
    # The rate and its interval at one precision, its interval's own (number rules 1 to 3): ``90% (… 58 to 98%)``.
    level, ends = forms.lone_rate(100 * rate, None if interval is None else (100 * interval[0], 100 * interval[1]))
    return f"{name} {level.text} ({hits} of {n} trials{over}{'' if ends is None else f', {ends.text}'})"


def _label_text(forms: PageForms, statistics: LabelStatistics) -> str:
    """One label's line as the quiet print says it: precision and recall in percent with their counts and intervals
    (``precision 100% (15 of 15 trials over 5 cases, [43.8%, 100%])``), then F1 in percent."""
    label = statistics.label
    shown = repr(label) if label != label.strip() or len(label.splitlines()) > 1 else label
    precision = _proportion_text(
        forms,
        "precision",
        statistics.precision,
        statistics.correct,
        statistics.predicted,
        statistics.predicted_cases,
        statistics.precision_interval,
    )
    recall = _proportion_text(
        forms,
        "recall",
        statistics.recall,
        statistics.correct,
        statistics.expected,
        statistics.expected_cases,
        statistics.recall_interval,
    )
    f1 = "F1 none" if statistics.f1 is None else f"F1 {forms.in_unit(100 * statistics.f1, PERCENT, role='level').text}"
    return f"{shown}: {precision}, {recall}, {f1}"


def _dimension_text(forms: PageForms, dimension: DimensionSummary) -> str:
    """One judged dimension's line as the quiet print says it: under the name the page gives it
    (:func:`~threetears.evals.analysis.report.words.judged_heading`) and its scale, a pass/fail one in percent with its
    count of trials, a scored one's mean with its trials named, and how often the judge could not tell."""
    scale = {"ordinal": ", scored 1-5", "pass_fail": ", scored pass/fail", None: ""}[dimension.scale]
    line = f"{judged_heading(dimension.name)}{scale}: "
    if dimension.mean is None or dimension.minimum is None or dimension.maximum is None:
        line += "no result carries a score"
    elif dimension.scale == "pass_fail":
        line += (
            f"{forms.in_unit(100 * dimension.mean, PERCENT, role='level', measure=dimension.name, kind='judged').text}, "
            f"{forms.hits(dimension.mean, dimension.n, measure=dimension.name, kind='judged').text} of "
            f"{dimension.n} trials"
        )
    else:
        line += (
            f"mean {forms.lone(dimension.mean, measure=dimension.name, kind='judged').text} ({dimension.n} trials, "
            f"min {forms.lone(dimension.minimum, measure=dimension.name, kind='judged').text}, "
            f"max {forms.lone(dimension.maximum, measure=dimension.name, kind='judged').text})"
        )
    if dimension.cannot_tell:
        line += f", the judge could not tell on {dimension.cannot_tell}"
    return line


def _compact(answer: Any) -> str:
    """An answer the graded field cannot be read off, as a miss line shows it, cut by :func:`_cut`: a mapping as
    compact JSON without its ``id``, the text an answer JSON could not hold was stored as (its ``repr``) as it is, and
    any other value as compact JSON."""
    if isinstance(answer, str):
        return _cut(answer)
    if isinstance(answer, Mapping):
        answer = {key: value for key, value in answer.items() if key != "id"}
    return _cut(json.dumps(answer, separators=(",", ":"), default=str))


def _cut(text: str) -> str:
    """``text`` cut at :data:`_MISS_ANSWER_CHARS`, an ellipsis marking the cut."""
    return text if len(text) <= _MISS_ANSWER_CHARS else text[: _MISS_ANSWER_CHARS - 1] + "…"


def _total_spend(forms: PageForms, summary: EvalSummary, totals: MoneyRuler, roles: MoneyRuler) -> str | None:
    """The total-spend line, split by role where more than one role the total counts spent, each role's dollars read
    off its usage rows (:attr:`~threetears.evals.analysis.summary.EvalSummary.spend`): the total in ``totals``' one
    form, the form the page writes the arms' test costs in, each role's in ``roles``': ``total spend: $0.017, the
    results' cost (candidate $0.0018, judge $0.015)``. A total unknown in full as the summary says it."""
    said = summary.total_spend_text
    if said is None or summary.total_cost_usd is None:
        return said
    said = f"total spend: {forms.money_on(totals, summary.total_cost_usd).text}, the results' cost"
    spent = [(role.role, role.cost_usd) for role in summary.spend if role.counted and role.cost_usd]
    if len(spent) < 2:
        return said
    return f"{said} ({', '.join(f'{role} {forms.money_on(roles, cost).text}' for role, cost in spent)})"


def _exclusion_causes(summary: EvalSummary) -> list[str]:
    """Each distinct reason an arm's results were excluded, with how many: never dropped from the quiet print."""
    counted: dict[str, int] = {}
    for result in summary.results():
        if result.outcome == "excluded":
            for error in result.errors or ["no reason recorded"]:
                counted[error] = counted.get(error, 0) + 1
    return [f"{count} x {error}" for error, count in counted.items()]


def _write_to_label(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    """Write the judged answers for a person to label, in :data:`LABEL_COLUMNS`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as written:
        writer = csv.DictWriter(written, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read_labels(path: Path, scope_id: str, scales: Mapping[str, RubricScale]) -> list[CalibrationRating]:
    """The labels a person filled in on a :data:`TO_LABEL` file, as ratings to pair with the judge's scores.

    :raises ValueError: A missing column, a label not on its dimension's scale, or a dimension the judge does not score.
    """
    with path.open(encoding="utf-8", newline="") as read:
        rows = list(csv.DictReader(read))
    if rows and (missing := [column for column in LABEL_COLUMNS if column not in rows[0]]):
        raise ValueError(f"{path} lacks the column(s) {', '.join(missing)}; label the {TO_LABEL} a result wrote")
    ratings: list[CalibrationRating] = []
    for number, row in enumerate(rows, start=2):
        label = (row.get("label") or "").strip().lower()
        if not label:
            continue
        dimension = row["dimension"]
        scale = scales.get(dimension)
        if scale is None:
            raise ValueError(f"{path} line {number} labels {dimension!r}, which the judge does not score")
        if scale == "pass_fail":
            score = _PASS_FAIL.get(label)
        else:
            score = int(label) if label.isdigit() and 1 <= int(label) <= 5 else None
        if score is None:
            scale_words = "pass or fail" if scale == "pass_fail" else "1 to 5"
            raise ValueError(f"{path} line {number} labels {dimension!r} {label!r}; its scale is {scale_words}")
        ratings.append(
            CalibrationRating(
                scope_id=scope_id,
                run_id=row["run_id"],
                result_id=row["result_id"],
                rubric_dim=dimension,
                rater=(row.get("rater") or "").strip() or path.name,
                rater_kind="person",
                scale=scale,
                score=score,
                reason=(row.get("reason") or "").strip() or f"labelled in {path.name}",
                output_fingerprint=row["output_fingerprint"] or None,
                criterion_fingerprint=row["criterion_fingerprint"] or None,
                rated_withheld=f"read from {path.name}, which carries the answer and its case as the person read them",
            )
        )
    return ratings


def _criterion(entry: Any, *, of: Any) -> TierCriterion:
    """A tier criterion over one agreement reading, or an insufficient one where there is none."""
    if entry is None:
        insufficient: TierCriterion = of(0, 0, None, None)
        return insufficient
    statistic = agreement_statistic(entry.scale, entry.kappa, entry.weighted_kappa)
    criterion: TierCriterion = of(entry.n, entry.results, statistic, entry.agreement_interval)
    return criterion


def calibrate(
    judge: Judge | None, host: EvalHost, arms: Mapping[str, EvalSummary], scope_id: str
) -> list[JudgeCalibration]:
    """How ``judge`` agrees with the labels it names, per dimension, over every arm's results. Reads the store.

    Labels pair with the judge's scores of byte-identical answers on the same dimension, in this run or the one they
    were written on, so a replayed run reads the labels written on its recording's.

    :return: One per dimension a label pairs with; empty with no judge, no labels file, or no label in it.
    :rtype: list[JudgeCalibration]
    :raises ValueError: A labels file with a missing column, or a label not on its dimension's scale.
    """
    if judge is None or judge.labels is None or not judge.labels.is_file():
        return []
    ratings = _read_labels(judge.labels, scope_id, {dim.name: dim.scale for dim in judge.dims})
    results: list[EvalResult] = [
        result for summary in arms.values() for result in list_results(host.storage, summary.run_id, scope_id)
    ]
    separation = {entry.rubric_dim: entry for entry in judge_self_agreement(results).dimensions}
    calibrations: list[JudgeCalibration] = []
    for entry in judge_agreement(ratings, results).dimensions:
        calibrated = _criterion(entry, of=calibration_criterion)
        separated = _criterion(separation.get(entry.rubric_dim), of=separation_criterion)
        calibrations.append(
            JudgeCalibration(
                dimension=entry.rubric_dim,
                labels=entry.n,
                kappa=agreement_statistic(entry.scale, entry.kappa, entry.weighted_kappa),
                lower=None if entry.agreement_interval is None else entry.agreement_interval[0],
                upper=None if entry.agreement_interval is None else entry.agreement_interval[1],
                calibration=calibrated.state,
                tier=tier_of(calibrated, separated),
            )
        )
    return calibrations


#: The id the page's line on how the judge graded its examples carries.
JUDGE_EXAMPLES_ANCHOR = "judge-examples"


def with_judge_examples(result: Result, examples: JudgeExamples | None) -> Result:
    """``result`` carrying how its judge graded its examples, its page saying so; ``result`` itself for ``None``.

    The page says it in Design & provenance, where it states how far the judge's scores can be read (the judge's
    calibration against people's labels), after that table: one sentence, then the misses. It is the eval's reading,
    taken in its call and not stored with the runs, so the line is set on this result's page; a page read again from
    the store has none, as it has no ``to_label.csv``.

    :param result: The result.
    :ptype result: Result
    :param examples: How the judge graded its examples, or ``None`` where it has none.
    :ptype examples: JudgeExamples | None
    :return: The result, its page with the line.
    :rtype: Result
    """
    if examples is None:
        return result
    line = DisclosureBlock(
        section="provenance", source="calibration", text=examples.sentence(), anchor=JUDGE_EXAMPLES_ANCHOR
    )
    return replace(
        result, judge_examples=examples, page=result.page.model_copy(update={"blocks": [*result.page.blocks, line]})
    )


def judged_rows(
    host: EvalHost, arms: Mapping[str, EvalSummary], scope_id: str, case_names: Mapping[str, str]
) -> list[dict[str, str]]:
    """Every judged answer on every dimension, as a :data:`TO_LABEL` row, the judge's score withheld. Reads the store.

    :param host: Where the runs are stored.
    :ptype host: EvalHost
    :param arms: Each arm's summary.
    :ptype arms: Mapping[str, EvalSummary]
    :param scope_id: Their scope.
    :ptype scope_id: str
    :param case_names: What each test case is called, by its id.
    :ptype case_names: Mapping[str, str]
    :return: The rows, arm by arm, in the order the store lists each run's results.
    :rtype: list[dict[str, str]]
    """
    rows: list[dict[str, str]] = []
    for summary in arms.values():
        for result in list_results(host.storage, summary.run_id, scope_id):
            trace = get_result_trace(host.storage, result) if result.has_trace else None
            evidence = trace.judge_evidence if trace is not None else None
            if evidence is None:
                continue
            for score in result.rubric_scores:
                rows.append(
                    {
                        "case": case_names.get(result.test_case_id, result.test_case_id),
                        "repeat": str(result.k_iteration),
                        "dimension": score.dim,
                        "case_material": evidence.case_material,
                        "answer": evidence.artifact,
                        "label": "",
                        "reason": "",
                        "rater": "",
                        "run_id": result.eval_run_id,
                        "result_id": result.id,
                        "output_fingerprint": score.output_fingerprint or "",
                        "criterion_fingerprint": score.criterion_fingerprint or "",
                    }
                )
    return rows


def _campaign_name(name: str) -> str:
    """An eval's name as its campaign holds it: whole within :data:`DECLARED_NAME_CHARS`, else cut at a word to fit,
    an ellipsis marking the cut (its first characters where its first word alone runs past)."""
    if len(name) <= DECLARED_NAME_CHARS:
        return name
    text = " ".join(name.split())
    room = text[: DECLARED_NAME_CHARS - 1]
    cut = room.rsplit(" ", 1)[0] if " " in room else room
    return cut.rstrip(" ,;:.-") + "…"


def declare(
    host: EvalHost,
    arms: Mapping[str, EvalSummary],
    factors: tuple[str, ...],
    *,
    levels: Mapping[str, Mapping[str, str]],
    kind: str,
    control: str,
    scope_id: str,
    name: str,
    behavior: str,
    repetitions: int | None,
    created_by: str,
    guardrail_margins: Sequence[GuardrailMargin] = (),
    measure_latency: bool = False,
    declared_at: str | None = None,
    question: str | None = None,
    margin: float | None = None,
    measure_order: Sequence[str] = (),
) -> Result:
    """File the arms' runs as one campaign, one axis per factor, designate ``control``, and read its report.

    ``question`` is the one sentence the comparison answers, filed as the campaign's question: every reading on the
    aspect is then read as asked rather than as a lead. ``margin`` (a share of 0 to 1, the measure's units) is the
    author's own, filed on the aspect in points: a non-inferiority question, the pass rate's drop read against it.
    ``None`` files none, and nothing is assumed: a question with no margin is a superiority question, its drop read
    against zero. ``question`` ``None`` asks none, and a margin without one is refused. ``measure_order`` is each
    grader's measure in the order the eval lists them (:attr:`Result.grader_readings`), filed on the aspect's design so
    the campaign's page leads on the first of them, as the printed line does; empty files none.

    ``declared_at`` is when the caller fixed the design and the control, where that was before its arms ran: an eval
    fixes both in the call that launches them, so the campaign filed once they have run is written once, dated then
    (:func:`~threetears.evals.analysis.create_campaign`'s ``declared_at``), never as chosen once its readings were
    visible. None dates them now, as reading the same runs against another control (:meth:`Result.against`) does.

    The page is the campaign's code-only page, as :func:`~threetears.evals.analysis.campaign_report` reads a
    campaign no analysis has been generated for — which a campaign filed a moment ago is — and the report its one
    aspect's page over the same bundle, read off that bundle, so each contrast the report tests is matched to the arm
    whose run carries its variant key rather than to the words the report names it by. ``levels`` is each arm's level of
    each factor, by arm name. A campaign of one arm is one run, its own control.
    """
    ordered = [control, *(arm for arm in arms if arm != control)]
    prefix = host.profile.kind_contract(kind).lever_prefix
    axes = []
    for factor in factors:
        values = list(dict.fromkeys(levels[arm][factor] for arm in ordered))
        axes.append(
            {
                "axis_id": factor if factor in _UNPREFIXED else f"{prefix}.{factor}",
                "values": [{"content": value, "display": value} for value in values],
                "rationale": (
                    f"does any arm do better than {control}"
                    if len(factors) == 1
                    else f"does moving {factor} change what the arms score, against {control}"
                ),
            }
        )
    design: dict[str, Any] = {"axes": axes, "held_fixed": {"stimulus": "controlled", "apparatus": "commissioned"}}
    if repetitions is not None:
        design["intended_repetitions"] = repetitions
    if measure_latency:
        design["measure_latency"] = True
    if guardrail_margins:
        design["guardrail_margins"] = [entry.model_dump() for entry in guardrail_margins]
    if measure_order:
        design["measure_order"] = list(dict.fromkeys(measure_order))
    # One aspect, the eval's: every arm ran the one template the eval authored, so the aspect names it (a judged
    # guardrail's margin is declared on a dimension of its rubric, which the declaration gate reads from it). A margin
    # on the pass rate is filed only where the eval's author wrote one; none is ever assumed.
    aspect: dict[str, Any] = {
        "label": aspect_label_for(behavior),
        "run_ids": [summary.run_id for summary in arms.values()],
        "design": design,
    }
    refuse_an_unusable_margin(margin, question)
    if margin is not None:
        aspect["margin"] = round(100 * margin, 10)
    if arms[control].template_id is not None:
        aspect["template_id"] = arms[control].template_id
    # The control arm is named as the eval names it, so every page reads `hasty (control) vs careful`.
    control_arm = {"label": control, "role": "control", "from_run_ids": [arms[control].run_id]}
    # The campaign's name is held to the cap every write of one is (a qualifier beside its title): an eval's name that
    # runs past it is cut at a word there, and kept whole in the campaign's description. The result keeps it whole.
    stored = _campaign_name(name)
    defined: dict[str, Any] = {
        "name": stored,
        "subject_id": name,
        "behavior": behavior,
        "aspects": [aspect],
        "arms": [control_arm],
    }
    if stored != name:
        defined["description"] = name
    if question is not None:
        defined["question"] = question
    campaign = create_campaign(
        host.storage,
        defined,
        scope_id=scope_id,
        created_by=created_by,
        profile=host.profile,
        declared_at=declared_at,
    )
    page = campaign_report(host, campaign.id, scope_id)
    (fold,) = page.aspects
    bundle = fold.report.bundle
    # The same bundle laid out alone, its arms named by their variants as the eval names them, not by the campaign's
    # labels the page's fold reads.
    arm_variants = {
        arm: variant
        for arm, summary in arms.items()
        if (variant := variant_key_of_run(list_results(host.storage, summary.run_id, scope_id))) is not None
    }
    # Its arms tagged as the campaign's page tags them, in the order the eval gives them (the control first, as the
    # campaign and its report document list them), so an arm is one letter and one colour on all three pages.
    control_key = next((arm.variant_key for arm in campaign.arms if arm.role == "control"), None)
    report = build_code_only_report_in_forms(
        bundle,
        assembled_at=fold.report.source.generated_at,
        campaign_name=campaign.name,
        arm_tags=campaign_arm_tags([bundle], control_key, arms=list(arm_variants.values())),
        measure_order=list(dict.fromkeys(measure_order)),
        # In the campaign page's forms, so the report reads every figure as the page's fold of it does.
        forms=_aspect_forms(page, bundle),
    )
    arm_of_variant = {variant: arm for arm, variant in arm_variants.items()}
    surface = bundle_decision_surface(bundle)
    labels = report_cell_labels(bundle)
    labelled_arms = {
        labels[cell_ref(cell.variant_key, cell.apparatus_class_id)]: arm_of_variant.get(cell.variant_key)
        for cell in surface.cells
    }
    return Result(
        campaign_id=campaign.id,
        scope_id=scope_id,
        name=name,
        control=control,
        arms=dict(arms),
        report=report,
        page=page,
        host=host,
        factors=factors,
        contrast_arms=tuple(
            arm_of_variant.get(tested.contrast.variant_key)
            for family in bundle.multiple_comparisons.families
            for tested in family.comparisons
        ),
        contrast_measures=tuple(
            tested.name for family in bundle.multiple_comparisons.families for tested in family.comparisons
        ),
        kind=kind,
        guardrail_readings=bundle.guardrails,
        arm_variants=arm_variants,
        labelled_arms=labelled_arms,
        grader_readings=tuple(measure_order),
        question=question,
        margin=margin,
    )


__all__ = [
    "CREATED_BY",
    "JUDGE_EXAMPLES_ANCHOR",
    "LABEL_COLUMNS",
    "QUIET_MISSES",
    "TO_LABEL",
    "JudgeCalibration",
    "Result",
    "calibrate",
    "coordinates",
    "declare",
    "judged_rows",
    "refuse_an_unusable_margin",
    "with_judge_examples",
]
