"""``Result``: what an eval ran, for one arm or a comparison of many, and what its campaign's report says.

Every eval's run ends in one :class:`Result`, whatever its runner: one arm, or a mapping of arms whose first key is
the control. Everything here is the engine's own path, composed:

- **Each arm is one run**, labelled by its arm name, all into one host and one scope, over one content-addressed case
  set. Two arms with different names are two variants. Named plainly (``{"baseline": a, "terse": b}``), an arm's name
  is the run's level of the host's arm lever (:data:`~threetears.evals.quick.one_call.ARM_LEVER`), every arm at one
  shared candidate model, so the report calls the arm ``candidate=<name>``. Named as ``k=v`` pairs
  (``"prompt=v2, effort=low"``), each pair is one factor of a factorial design: ``model`` is the run's candidate model
  and every other factor is a lever the engine names ``callable.<factor>`` and keys into the variant. Such an arm is
  named by its factors, in the order the eval declares them, on every surface (:func:`_shown_arm_name`:
  ``Prompt v2, effort low``), never by the lever the engine keys it under.
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

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from threetears.evals.kernel.offload import run_blocking
from threetears.evals.analysis import (
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
    create_campaign,
    variant_key_of_run,
)
from threetears.evals.analysis.gate import DEFAULT_FAIL_ON
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown
from threetears.evals.analysis.campaigns import get_campaign
from threetears.evals.kernel import (
    ACCURACY_MEASURE,
    CONFUSION_CELL_MEASURE,
    MATCH_MEASURE,
    ArmGuardrails,
    GuardrailMargin,
    GuardrailReadings,
    confusion_cell,
)
from threetears.evals.kernel.metrics import confusion_of
from threetears.evals.analysis.blind_read import BlindClose, blind_sheet, close_blind_read, draw_blind_read
from threetears.evals.schema import BlindReadReport, CalibrationRating, CompletionClient, EvalResult
from threetears.evals.schema.blind_read import BlindAnswer, BlindArm, BlindRead
from threetears.evals.schema.models import DECLARED_NAME_CHARS
from threetears.evals.schema.aspect_label import aspect_label_for
from threetears.evals.kernel.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.kernel.text_block import one_line
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, ArmScoreboard
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.bundle.comparisons import (
    LATE_QUESTION_SENTENCE,
    REREAD_SENTENCE,
    read_by_bet,
    undo_lowering_ignored,
)
from threetears.evals.quick.continuation import MEMORY_WHERE, UNKEPT_SENTENCE
from threetears.evals.analysis.bundle.trials import pass_count, trial_order
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.confusion import confusion_matrix, label_statistics
from threetears.evals.analysis.run_ending import stopped_short
from threetears.evals.analysis.summary import CaseResult, DimensionSummary, EvalSummary
from threetears.evals.analysis.quantities import PERCENT
from threetears.evals.kernel.metrics import (
    METRIC_DESCRIPTORS,
    MetricDescriptor,
    describe_measure,
    goal_check_of,
    reads_as_rate,
    reads_as_share,
)
from threetears.evals.analysis.report.cost_unit import MoneyRuler, money_ruler
from threetears.evals.analysis.report.figures import report_rulers
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.blocks import DisclosureBlock
from threetears.evals.analysis.report.aspect.contrasts import aspect_verdict, contrast_rows, report_cell_labels
from threetears.evals.analysis.report.build import build_code_only_report_in_forms
from threetears.evals.analysis.report.verdict import VerdictLine
from threetears.evals.analysis.style import UNDO_LOWERING_IGNORED_SENTENCE, arm_display_name, arm_name
from threetears.evals.analysis.report.words import campaign_arm_tags, judged_heading, stratum_names
from threetears.evals.quick.graders import UNUSABLE_ANSWER, expected_label, predicted_label
from threetears.evals.kernel.out_of_run import OutOfRunBudget
from threetears.evals.quick.blind import (
    BLIND_READ_CSV,
    BLIND_READ_HTML,
    answer_sheet,
    blind_read_csv,
    blind_read_html,
    judge_lines,
    keep_reviewed,
    read_labels,
)
from threetears.evals.run.blind_review import REVIEWER_PURPOSE, plan_review
from threetears.evals.quick.judged import BorrowedJudgeClient, Judge, JudgeExamples, judge_client
from threetears.evals.quick.one_call import ARM_LEVER, CALLABLE_KIND
from threetears.evals.kernel.candidate_kind import QUICK_PATH_FILER
from threetears.evals.run import get_result_trace, list_results
from threetears.evals.analysis.style import measure_name, measure_name_of

#: Who an eval's campaign and its control are recorded as created by, unless the caller says: the quick path's own
#: name, which the analysis reads a quick-path comparison's origin by
#: (:data:`~threetears.evals.kernel.candidate_kind.QUICK_PATH_FILER`).
CREATED_BY = QUICK_PATH_FILER


#: The factors that are levers of the engine or the host, never of the callable kind: no kind prefix names them.
_UNPREFIXED = frozenset({CANDIDATE_MODEL_LEVER, ARM_LEVER})

#: How many misses the quiet print shows; the rest are in the report and :meth:`Result.misses`.
QUIET_MISSES = 5

#: How many characters of an answer a miss line shows where the graded field cannot be read off it.
_MISS_ANSWER_CHARS = 120


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


def _shown_arm_name(arm: str) -> str:
    """The one name every surface prints for an arm: the campaign's page, the verdict line, the quiet print's
    scoreboard and misses, and a program's rows.

    An arm named as ``k=v`` pairs is named by its factors, in the order the eval declares them, each as its reader
    name and its level, by the one rule a name code computes from levers is printed by
    (:func:`~threetears.evals.analysis.style.arm_display_name`): ``"prompt=v2, effort=low"`` is ``Prompt v2, effort
    low``. Never by the lever the engine keys a factor under (``callable.prompt``), whose kind is no reader's word,
    nor in the order the variant sorts its levers in. An arm named plainly is printed as named.

    :param arm: The arm, by its key in the eval's runners.
    :return: Its name as a reader is shown it.
    """
    pairs = coordinates(arm)
    if pairs is None:
        return arm
    # Each factor's reader name is the factor as the eval's author wrote it, as the campaign stores it on its axis.
    named = {factor: factor for factor in pairs}
    shown = arm_name(", ".join(f"{factor}={level}" for factor, level in pairs.items()), declared=named)
    return arm_display_name(shown.label, shown.levers, named)


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
    :ivar calibrations: How the judge agrees with the person's labels ``Judge(labels=...)`` names, per dimension
        (:func:`calibrations_of`); empty with no judge, no labels, or no person's label that pairs.
    :ivar judge_examples: How the judge graded its examples (``Judge(examples=...)``) in the run, each on every
        repeat, beside its agreement with the labels and never pooled into it; ``None`` with no judge, or a judge
        with no examples.
    :ivar blind_read: What the run's blind read found, as typed data: how many cases were read and whether as a random
        sample, the judge's agreement per criterion with people and with a model, the preference with its test, and
        what a model's reading cost. ``None`` with no judge, or no judged answer to show.
    :ivar blind_read_files: What a judged result writes for a blind read, by file name: ``blind_read.html`` (the
        run's judged answers in one seeded random order, with no arm, run or judge score on the page) and
        ``blind_read.csv`` (the same order as rows). Empty with no judge, or no judged answer to show.
    :ivar judge_read: What the print says of the judge against whoever read it blind (:meth:`calibration_lines`),
        line by line; empty for an unjudged eval.
    :ivar reviews_kept: The file a model reviewer's answers are kept in (:meth:`reviewed_by`), or ``None``.
    :ivar contrast_arms: The arm each row of the report's contrasts table tests, by name, row for row; ``None`` for a
        row no arm's variant matches.
    :ivar contrast_measures: The key of the measure each row of the contrasts table reads, row for row.
    :ivar kind: The kind every arm ran — the callable kind, or the judged one — whose levers the campaign's axes name.
    :ivar guardrail_readings: Every guardrail, decided for each arm against the control; empty when none is declared.
    :ivar arm_variants: Each arm's variant key, by name: what the bundle names the arm by.
    :ivar labelled_arms: Each arm's name, by the words the report names it by (``baseline``).
    :ivar arm_labels: The name every surface prints each arm named as factors by (:func:`_shown_arm_name`), by its
        variant key; empty for arms named plainly, which every surface names by their variants.
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
    blind_read: BlindReadReport | None = field(default=None, repr=False, compare=False)
    blind_read_files: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    judge_read: tuple[str, ...] = field(default=(), repr=False, compare=False)
    reviews_kept: Path | None = field(default=None, repr=False, compare=False)
    contrast_arms: tuple[str | None, ...] = field(default=(), repr=False, compare=False)
    contrast_measures: tuple[str | None, ...] = field(default=(), repr=False, compare=False)
    kind: str = field(default=CALLABLE_KIND, repr=False, compare=False)
    guardrail_readings: GuardrailReadings = field(default_factory=GuardrailReadings, repr=False, compare=False)
    arm_variants: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    labelled_arms: dict[str, str | None] = field(default_factory=dict, repr=False, compare=False)
    arm_labels: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
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
        :return: The arm's results, as :meth:`~threetears.evals.analysis.summary.EvalSummary.results` reads them.
        :raises ValueError: ``arm`` names no arm, or is ``None`` on a comparison.
        """
        return self._arm(arm).results()

    def misses(self, arm: str | None = None) -> list[CaseResult]:
        """The results one arm missed, each saying why.

        :param arm: The arm, by name; ``None`` for the one arm of a result that has one.
        :return: The arm's misses, as :meth:`~threetears.evals.analysis.summary.EvalSummary.misses` reads them.
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
        :param arm: The arm, by name; ``None`` for the one arm of a result that has one.
        :return: The confusion matrix, by expected then predicted label, and each label's statistics, by label.
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
        """How the judge stands against whoever read its answers blind, as the quiet print says it; empty for an
        unjudged eval."""
        return list(self.judge_read)

    async def reviewed_by(
        self,
        model: str | CompletionClient,
        *,
        max_cost_usd: float | None = None,
        kept: str | os.PathLike[str] | None = None,
    ) -> Result:
        """This result with its blind read answered by a model, read apart from any person and never as one.

        The model is shown the sheet a person is (the outputs in their drawn order and slots, the case each answers,
        the criteria as worded; no arm, run or judge score), one call per case. The judge's agreement with it is
        agreement with that model. It is not read as a check of the judge where it is the judge's own model, or where
        which model served either side was not recorded.

        **Its calls are spend like any other.** Each is admitted before it is made, held to ``max_cost_usd`` over
        everything the eval has landed (its arms, its judge's examples), and written to the store's out-of-run ledger
        under the purpose ``reviewer``: the cost of checking the judge, in no candidate's cost. The print says what
        they cost on a line of their own.

        **A case it answered is not asked again.** With ``kept`` its answers are written there as a labels file, and
        a later call over the same draw reads them back and asks only the cases the file does not hold. A reply that
        cannot be used leaves its case not answered, counted and said, and the rest are asked.

        The read is drawn again from the stored answers, which draw the same order, and the labels file is read
        again. A result carries one model's reading: a second call with another model replaces the first, unless
        both are kept.

        :param model: The reviewer: a model id, a :class:`~threetears.evals.providers.Model`, or any completion
            client that names its model.
        :param max_cost_usd: The most the eval may spend, these calls included; ``None`` for no cap.
        :param kept: The file its answers are kept in and read back from; ``None`` keeps none.
        :return: The result, its blind read carrying the model's answers beside any already there.
        :raises ValueError: A client naming no model, or an unjudged result or one with no judged answer to read.
        """
        judge = self.judge
        if judge is None or not self.blind_read_files:
            raise ValueError("a model answers a judged result's blind read, and this result has none")
        client, name = judge_client(model)
        lent = BorrowedJudgeClient(client, model_name=name)
        kept_at = None if kept is None else Path(kept)

        def recorded() -> tuple[BlindRead, tuple[CalibrationRating, ...], list[EvalResult], list[BlindAnswer]]:
            runs = self._arm_runs()
            closed, outside = read_blind(judge, self.host, self.arms, self.control, self.scope_id, self.name, runs=runs)
            if closed is None:
                raise ValueError("the stored answers no longer hold a judged answer to read blind")
            drawn = closed.read
            results = _results_of(self.host, self.arms, self.scope_id, runs)
            if kept_at is None or not kept_at.is_file():
                return drawn, outside, results, []
            # Only answers on this very draw are read back: an answer to another draw is asked for again.
            earlier = read_labels(kept_at, drawn, results).answers
            mine = [answer for answer in earlier if (answer.rater, answer.made_by) == (name, "model")]
            return drawn, outside, results, mine

        read, loose, results, earlier = await run_blocking(self.host.blocking_executor, recorded)
        report = read.report
        have = {answer.item for answer in earlier}
        landed = [summary.total_cost_usd for summary in self.arms.values()]
        if self.judge_examples is not None and self.judge_examples.spend.calls:
            landed.append(None if self.judge_examples.spend.unpriced_calls else self.judge_examples.spend.cost_usd)
        budget = OutOfRunBudget(
            store=self.host.storage,
            scope_id=self.scope_id,
            cap_usd=None,
            template_id=self.template_id,
            campaign_id=self.campaign_id,
            blocking_executor=self.host.blocking_executor,
        )
        sheet = blind_sheet(read)
        # Planned, then admitted here, then made: the reading makes only what this call admitted.
        planned = plan_review(sheet, {item.item for item in read.items} - have)
        admitted = (
            dict(zip(planned, budget.admit(lent, REVIEWER_PURPOSE, list(planned.values())), strict=True))
            if planned
            else {}
        )
        try:
            answers, review = await answer_sheet(
                sheet,
                lent,
                budget,
                admitted,
                reused=len(have),
                max_cost_usd=max_cost_usd,
                landed_usd=None
                if any(cost is None for cost in landed)
                else math.fsum(c for c in landed if c is not None),
            )
        finally:
            if isinstance(model, str):
                # A client built here from a model id is this call's to close; a caller's own is the caller's.
                await client.aclose()

        def closed() -> Result:
            others = [answer for answer in read.answers if (answer.rater, answer.made_by) != (name, "model")]
            reviews = [review_ for review_ in read.reviews if review_.reviewer != name]
            again = close_blind_read(
                read.model_copy(
                    update={"answers": [*others, *earlier, *answers], "reviews": [*reviews, review], "report": None}
                ),
                results,
                other_ratings=loose,
                unread=report.unread if report is not None else (),
                judge_models=[judge.model],
            )
            if kept_at is not None:
                keep_reviewed(kept_at, again.read, name)
            return replace(self, reviews_kept=kept_at, **blind_reading(judge, again, self.arms, kept_at))

        return await run_blocking(self.host.blocking_executor, closed)

    def _arm_runs(self) -> dict[str, list[str]]:
        """Each arm's runs, as its campaign holds them, oldest first: one, or a continued comparison's every run
        (:func:`~threetears.evals.quick.one_call.launch_arms`). Reads the store."""
        campaign = get_campaign(self.host.storage, self.campaign_id, self.scope_id)
        arm_of = {variant: arm for arm, variant in self.arm_variants.items()}
        runs: dict[str, list[str]] = {arm: [] for arm in self.arms}
        for run_id in campaign.aspects[0].run_ids:
            arm = arm_of.get(variant_key_of_run(list_results(self.host.storage, run_id, self.scope_id)) or "")
            if arm is not None:
                runs[arm].append(run_id)
        return {arm: held or [self.arms[arm].run_id] for arm, held in runs.items()}

    # --- the report --------------------------------------------------------------------------------------------------

    def __str__(self) -> str:
        """The quiet summary (:meth:`render`)."""
        return self.render()

    def render(self, *, full: bool = False) -> str:
        """The result as text: quiet by default, everything with ``full=True``.

        Quiet is the verdict, each arm's scoreboard and readings, the top misses and where the runs are stored. Never
        quieted: errors, exclusions and their causes, a run stopped short, unpriced spend, the self-judging warning,
        and how the judge stands against people's labels. A run that did not finish (one that failed, as a replay miss
        fails one, was cancelled, was stopped by its budget or its provider, or is still running) leads with what
        stopped it and prints no verdict: its readings follow, said to cover only the results recorded before it
        stopped.

        :param full: Every arm's whole summary and the whole report, Methods included, as Markdown.
        :return: The text.
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
                    *(self._full_summary(forms, arm, summary, totals, roles) for arm, summary in self.arms.items()),
                    campaign_report_markdown(self.page),
                ]
            )
        one_arm = len(self.arms) == 1
        lines = self._stopped_lines() or [*self._advice_line(), self._verdict_line(), *self._reread_line()]
        # How far its judge can be trusted is read with the verdict: against people's labels, then on its examples.
        judged = self.calibration_lines() + (self.judge_examples.lines() if self.judge_examples is not None else [])
        if one_arm:
            # One arm's verdict is how its cases came out, and how far its judge can be trusted is read with it.
            lines += judged
        for arm, summary in self.arms.items():
            label = (
                "" if len(self.arms) == 1 else f"{_shown_arm_name(arm)}{' (control)' if arm == self.control else ''}: "
            )
            lines.append(f"{label}{summary.results_text}")
            # an operator's reason for a cancel, a predicted label, a refused check's reason: each can run over lines
            lines += [f"  {one_line(line)}" for line in (summary.stopped_text,) if line is not None]
            lines += [f"  {one_line(line)}" for line in self._measure_texts(forms, summary, arm)]
            # A label's precision and recall over no answer graded are no rates either (:meth:`_answered_nothing`).
            if summary.confusion and not self._answered_nothing(arm):
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
                who = "" if len(self.arms) == 1 else f"{_shown_arm_name(arm)}, "
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
        control = _shown_arm_name(self.control)
        return {row["measure_id"]: f"guardrail {row['outcome']} against {control}" for row in self.guardrails(arm)}

    def _measure_texts(self, forms: PageForms, summary: EvalSummary, arm: str) -> list[str]:
        """Each measure's line, the classifier's ``match`` under its field's name and its matrix's count left out.

        In a comparison a guardrail's line says what the comparison decided on it for ``arm``; alone, that a guardrail
        is decided only in one.
        """
        if self._answered_nothing(arm):
            # No answer was graded, so no measure has a figure: each reads none, for the one reason, never 0%; those
            # some result carries first, the rest where the line counting them would stand, as the full print has them.
            return [
                f"{self._unscored_name(measure.name)}: {_NOTHING_GRADED}"
                for measure in sorted(summary.measure_lines, key=lambda measure: measure.state == "not_carried")
                if measure.name != CONFUSION_CELL_MEASURE
            ]
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

    def _full_summary(
        self, forms: PageForms, arm: str, summary: EvalSummary, totals: MoneyRuler, roles: MoneyRuler
    ) -> str:
        """An arm's whole summary as the full print writes it: the run summary's lines
        (:meth:`~threetears.evals.analysis.summary.EvalSummary.render`), each figure in the form the quiet print and
        the page state it (the review's P43, A4): a measure, its strata's included, under the page's name, a rate in
        percent with its count of trials; each label's precision and recall in percent with their intervals; a judged
        dimension as the quiet print says it; every amount of spend in the form of its set, a total as the page's test
        cost. Every other line is the summary's own. Where the arm graded no answer (:meth:`_answered_nothing`), every
        measure reads :data:`_NOTHING_GRADED`, as the quiet print says it, and no label's precision or recall is
        printed."""
        nothing = self._answered_nothing(arm)
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
        dropped = {"per label:", *summary.label_lines} if nothing else set()
        uncarried = [measure.name for measure in summary.measure_lines if measure.state == "not_carried"]
        for line in summary.render().splitlines():
            body = line.lstrip()
            indent = line[: len(line) - len(body)]
            if body in dropped:
                continue
            if nothing and body == summary.uncarried_text:
                lines += [f"{indent}{self._unscored_name(name)}: {_NOTHING_GRADED}" for name in uncarried]
                continue
            if nothing and (measured := _FULL_MEASURE.fullmatch(body)) is not None:
                body = f"{self._unscored_name(measured['name'])}: {_NOTHING_GRADED}"
            elif body in said:
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

    def _answered_nothing(self, arm: str) -> bool:
        """Whether ``arm``'s runner failed every graded trial without answering, so no grader read an answer of its:
        the one decision the first line says it by (``... so no answer was graded``,
        :attr:`~threetears.evals.analysis.bundle.trials.PassCount.none_answered`), which every measure line of the arm
        then reads none by, never 0%."""
        board = self._board(arm)
        return board is not None and pass_count(self.report.bundle, board).none_answered

    def _graded_nothing(self, arm: str) -> bool:
        """Whether no answer of ``arm``'s was graded: its runner raised on every graded trial
        (:meth:`_answered_nothing`), or no trial of its was graded at all (each lost to the test setup, or none ran)."""
        board = self._board(arm)
        if board is None:
            return True
        count = pass_count(self.report.bundle, board)
        return count.graded == count.candidate_failed

    def _board(self, arm: str) -> ArmScoreboard | None:
        """``arm``'s row in the bundle's scoreboard, its census; None where it ran no trial the bundle holds."""
        variant = self.arm_variants.get(arm)
        return next((board for board in self.report.bundle.scoreboard if board.variant_key == variant), None)

    def _verdict_line(self) -> str:
        """The first line: the eval's name, then its aspect's verdict line as its page prints it
        (:func:`~threetears.evals.analysis.report.aspect.contrasts.aspect_verdict`): one arm's pass rate as its count, or what the
        comparison came to on the measure the eval's first grader reads, what else the evidence tells apart, and each
        guardrail an arm is not shown safe on. The page, its lead, this line and the exit line say one sentence.
        """
        return f"{self.name}: {self._verdict().text}"

    def _advice_line(self) -> list[str]:
        """The print's opening line, above the verdict line: the eval's name, then what to do (:attr:`VerdictLine.advice`,
        the page's verdict block's first sentences); empty where there is no advice to say."""
        advice = self._verdict().advice
        return [f"{self.name}: {advice}"] if advice else []

    def _reread_line(self) -> list[str]:
        """What a reader may do with the comparison the first line reads, one line under it: read it again as cases are
        added (:data:`~threetears.evals.analysis.bundle.comparisons.REREAD_SENTENCE`, its family's disclosure on the
        page), where the mixture bet read its family
        (:func:`~threetears.evals.analysis.bundle.comparisons.read_by_bet`) and its runs are kept in a store a later run
        continues it from; where they were kept in memory, that running it again is a new comparison
        (:data:`~threetears.evals.quick.continuation.UNKEPT_SENTENCE`), which is what the page's sentence, said of a
        store that keeps the runs, comes to there. Empty for one arm, or a family no bet read. A question set after the
        results (:data:`~threetears.evals.analysis.bundle.comparisons.LATE_QUESTION_SENTENCE`), and a lowering of how hard the
        switch is to undo declared after them and ignored
        (:data:`~threetears.evals.analysis.style.UNDO_LOWERING_IGNORED_SENTENCE`), say so first, as the
        page's disclosure does: its bet leans no way."""
        families = self.report.bundle.multiple_comparisons.families
        if not any(read_by_bet(family) for family in families):
            return []
        late = [LATE_QUESTION_SENTENCE] if any(f.primary is not None and f.lean is None for f in families) else []
        declared = self.report.bundle.declared
        late += (
            [UNDO_LOWERING_IGNORED_SENTENCE]
            if any(undo_lowering_ignored(family, declared) for family in families)
            else []
        )
        return [*late, UNKEPT_SENTENCE if self.store == MEMORY_WHERE else REREAD_SENTENCE]

    def _stopped_lines(self) -> list[str]:
        """The first lines of the print of a run that did not finish, in place of its verdict; empty when every arm's
        run completed.

        Whether there is a verdict to open with is the one decision every surface asks
        (:func:`~threetears.evals.analysis.run_ending.stopped_short`), here over each arm's run
        (:attr:`EvalSummary.ending`). Each unfinished arm's ending first, in the run's words (a replay miss names the
        call its recording does not hold; a cancel or a budget stop gives its reason), then how many results the
        readings below it cover. Such a run prints no verdict: it ended where it stopped (a replay miss fails it, owner
        ruling, #1162), so its results are whichever cells happened to finish first, and a verdict over them is the
        reading the stop exists to prevent.
        """
        short = stopped_short(summary.ending for summary in self.arms.values())
        if short is None:
            return []
        arm_of = {summary.run_id: arm for arm, summary in self.arms.items()}
        one_arm = len(self.arms) == 1
        said = short.sentences(lambda ending: "the run" if one_arm else _shown_arm_name(arm_of[ending.run_id]))
        return [
            *(f"{self.name}: {sentence}" for sentence in said),
            short.covers(sum(summary.n_results for summary in self.arms.values())),
        ]

    def _verdict(self) -> VerdictLine:
        """The eval's verdict line, as its aspect's page opens with it, led on its graders' order."""
        return aspect_verdict(
            self.report.bundle,
            measure_order=self.grader_readings,
            arm_labels=self.arm_labels or None,
            forms=self._aspect_forms(),
        )

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
        """A measure as the page names it (:func:`~threetears.evals.analysis.style.measure_name_of`):
        ``Accuracy``, ``Number score``. The classifier's ``match`` is its accuracy, which the page leads on, so it is
        named as the page names that."""
        key = ACCURACY_MEASURE if measure == MATCH_MEASURE and self._classifier is not None else measure
        return measure_name_of(self.report.bundle, key)

    def _unscored_name(self, measure: str) -> str:
        """A measure of an arm that graded no answer, as :meth:`_printed_name` names it where it scored: one no trial
        scored is in no catalog the bundle holds, so it is named as the host declares it (``Right score``), never by
        its key."""
        if measure == MATCH_MEASURE or measure in self.report.bundle.measure_catalog or goal_check_of(measure):
            return self._printed_name(measure)
        return measure_name(measure, describe_measure(measure, self.host.profile.measures).reader_name)

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        """Write the campaign's page as one self-contained file, its Markdown, its JSON and its aspect's bundle, and the
        judged answers to read blind.

        :param out_dir: The directory; created when missing.
        :return: Each file written, by role, as :meth:`~threetears.evals.analysis.CampaignReport.write` names them
            (``html``, ``markdown``, ``json``, ``bundle:<aspect>``), and, for a judged eval, ``blind_read`` (a
            self-contained page: the judged answers in one seeded random order, no arm, run or judge score on it, to
            answer from the top and save as the labels ``Judge(labels=...)`` reads back) and ``blind_read_csv`` (the
            same order as rows, for a spreadsheet).
        """
        written = self.page.write(out_dir)
        for role, name in (("blind_read", BLIND_READ_HTML), ("blind_read_csv", BLIND_READ_CSV)):
            if name in self.blind_read_files:
                path = Path(out_dir) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(self.blind_read_files[name], encoding="utf-8", newline="")
                written[role] = path
        return written

    def exit_code(self) -> int:
        """What a command line exits with for this result: 0, 1 when a run did not complete or no arm had an answer
        graded, 4 when a gate failed.

        A run in which no trial was graded on any arm (every runner raised on every case, or every trial was lost to
        the test setup) measured nothing, so it has not done its job: 1, its first line the reason. An arm with some
        answers graded, or a comparison with one arm that answered nothing, keeps its code. A comparison is gated on
        its verdicts (:meth:`gate`, the default outcomes); one arm has nothing to gate.
        """
        if any(summary.status != "completed" for summary in self.arms.values()):
            return 1
        if all(self._graded_nothing(arm) for arm in self.arms):
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
        :return: One row per arm and reading, keyed ``arm`` (the arm tested, by its key in :attr:`arms`: the name
            you gave it), ``measure_id`` (the key of the measure read, to cite or filter
            on), ``question`` (in the words it was asked), ``reading`` (the measure as the report heads it),
            ``contrast`` (the arm, as the report names it), ``control`` (the control, as the report names it),
            ``control_mean`` and ``arm_mean`` (over the cases the test read), ``cases`` (how many, paired or not,
            and any one side ran that the test left out), ``delta`` (arm minus control), ``interval`` (on the
            delta, simultaneous over the family), ``hedges_g`` (the standardized effect), ``p_adjusted`` (Holm,
            over the campaign's family), ``verdict`` (in words, printed from ``outcome``) and ``outcome`` (the
            typed verdict: ``improved``, ``regressed``, ``equivalent``, ``not_separated`` or ``untested``, which
            code branches on), and ``advice`` (what to do beside that proof, on the first grader's reading: ``switch``
            or ``keep``, which an agent branches on as it does on ``outcome``; None on any other reading, and where no
            bet read it); empty when the report tested nothing.
        """
        rows = contrast_rows(self.report.bundle, forms=self._aspect_forms(), arm_labels=self.arm_labels or None)
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
        :param kind: Only ``"contrast"``, ``"guardrail"`` or ``"bar"`` verdicts; ``None`` keeps every kind.
        :return: The verdicts, guardrails first, then bars, then contrasts, each in its table's row order.
        """
        return [
            verdict
            for verdict in self.report.verdicts
            if (reading is None or reading in (verdict.name, verdict.heading))
            and (kind is None or verdict.kind == kind)
        ]

    def gate(self, fail_on: Iterable[str] = DEFAULT_FAIL_ON, *, readings: Iterable[str] | None = None) -> GateResult:
        """Gate a build on this comparison's verdicts, as ``python -m threetears.evals gate`` does on a campaign.

        :param fail_on: The outcomes that fail it (:data:`~threetears.evals.analysis.gate.GATE_TOKENS`). The default,
            :data:`~threetears.evals.analysis.gate.DEFAULT_FAIL_ON`, fails on a regressed contrast, a breached
            guardrail and an undecided guardrail.
        :param readings: Only the verdicts on these readings, by key or heading; ``None`` gates every verdict.
        :return: The gate's result: ``failed`` when a named outcome occurred, ``passed`` only when every verdict was
            decided and none failed, ``undecided`` otherwise — never a pass.
        :raises ValueError: A ``fail_on`` name that is no outcome, or a reading no verdict is on.
        """
        return gate_verdicts(self.report.verdicts, fail_on=fail_on, readings=readings)

    def strata(self, arm: str | None = None, *, reading: str | None = None) -> list[dict[str, Any]]:
        """The rows of the report's "By stratum" table: each arm's figures over all cases, then per kind of case.

        Present when the cases declare strata (each case's reserved ``stratum`` key). Each arm opens with a
        ``cases`` row saying how many cases and observations each column rests on, a stratum of fewer than
        ``STRATUM_MIN_CASES`` (10) cases marked too few to read alone, then one row per reading.

        :param arm: Only this arm's rows, by its key in :attr:`arms`; ``None`` keeps every arm's.
        :param reading: Only the rows on this reading, as the report heads it (``"Accuracy"``), or ``"cases"``;
            ``None`` keeps every row.
        :return: One row per arm and reading, keyed ``arm`` (the arm, by its key in :attr:`arms`), ``label`` (the arm as
            the report names it), ``reading``, ``all`` (the figure over every case) and ``strata`` (the figure
            over each stratum's cases alone, by stratum: its declared name, else its id (``edge_case``), as the
            case declared it, never the name the page prints (``edge case``); the cases declaring none under
            ``"(no stratum)"``; a stratum the arm has no case in is absent); empty when no case declares a stratum.
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
        :return: One row per guardrail and arm, keyed ``arm`` (the arm checked, by its key in :attr:`arms`),
            ``measure_id`` (the guardrail's key: a scorer's name or a judged dimension's), ``outcome`` (``held``,
            ``breached`` or ``undecided``), ``contrast`` (the arm, as the report names it), and the table's other
            columns: ``guardrail`` (as the report heads it), ``control_mean`` and ``arm_mean``, ``cases``,
            ``delta`` (arm minus control), ``interval`` (on the delta), ``margin`` and ``decision`` (the outcome in
            words, with why when undecided); empty when no reading is a guardrail or no control resolved.
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
        :return: The guardrails the arm breached, those it is undecided on, and those it held, each by key; all empty
            for the control, which no guardrail is checked against itself, and when no reading is a guardrail.
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
        undo: Literal["easy", "hard"] | None = None,
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
        :param name: The new campaign's name; ``None`` names it after this one and the new control.
        :param created_by: Who the campaign and its control are recorded as created by.
        :param question: The question the new campaign asks; ``None`` asks this result's own, with its margin.
        :param margin: The question's margin, in the measure's units, as :class:`~threetears.evals.Eval`'s ``margin``;
            ``None`` keeps this result's own where no new question is given, and declares none with a new one.
        :param undo: How hard switching would be to undo, as :class:`~threetears.evals.Eval`'s ``undo``; ``None`` keeps
            this result's own. Declared here it is declared after the results: ``"hard"`` is honoured (raising the bar
            can only make the advice say switch less often), while ``"easy"`` on a result declared ``"hard"`` is
            ignored, the earlier ``"hard"`` kept, and the print and the page say so
            (:data:`~threetears.evals.analysis.style.UNDO_LOWERING_IGNORED_SENTENCE`).
        :return: A result over the same arms and runs, against ``control``.
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
        if undo is not None:
            refuse_an_unusable_undo(undo)

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
                name=name or f"{self.name}, against {_shown_arm_name(control)}",
                behavior=campaign.behavior,
                repetitions=design.intended_repetitions if design is not None else None,
                created_by=created_by,
                guardrail_margins=design.guardrail_margins if design is not None else (),
                measure_latency=design.measure_latency if design is not None else False,
                question=asked,
                margin=held,
                measure_order=self.grader_readings,
                runs=self._arm_runs(),
                # A bar is never lowered once the results are in: an earlier "hard" stands, and an "easy" asked over it
                # is recorded as ignored.
                undo="hard" if campaign.aspects[0].undo == "hard" or undo == "hard" else "easy",
                undo_ignored=campaign.aspects[0].undo == "hard" and undo == "easy",
            )
            return with_judge_examples(
                replace(
                    again,
                    store=self.store,
                    fields=self.fields,
                    judge=self.judge,
                    disclosures=self.disclosures,
                    calibrations=self.calibrations,
                    blind_read=self.blind_read,
                    blind_read_files=self.blind_read_files,
                    judge_read=self.judge_read,
                    reviews_kept=self.reviews_kept,
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
    :param question: The question it would be read under; ``None`` asks none.
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


def refuse_an_unusable_undo(undo: object) -> None:
    """Refuse an ``undo`` no advice could read: one that is neither ``"easy"`` nor ``"hard"``.

    :param undo: How hard the switch would be to undo, as given.
    :raises ValueError: Naming what it may be.
    """
    if undo not in ("easy", "hard"):
        raise ValueError(f'undo= is "easy" (the default) or "hard", how hard a switch would be to undo, not {undo!r}')


def _refuse_an_unknown_arm(arms: Mapping[str, Any], arm: str) -> None:
    if arm not in arms:
        raise ValueError(f"{arm!r} names no arm; the arms are {', '.join(map(repr, arms))}")


#: What a measure of an arm that graded no answer reads, in place of a figure (:meth:`Result._answered_nothing`): a rate
#: over no answer is no rate, so every measure of it, carried or not, reads this one way, never 0%.
_NOTHING_GRADED = "none (no answer was graded)"

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


def _results_of(
    host: EvalHost,
    arms: Mapping[str, EvalSummary],
    scope_id: str,
    runs: Mapping[str, Sequence[str]] | None = None,
) -> list[EvalResult]:
    """Every arm's results, arm by arm: each arm's runs in ``runs`` (a continued comparison's every run), else its
    summary's one run. Reads the store."""
    return [
        result
        for arm, summary in arms.items()
        for run_id in (runs or {}).get(arm, (summary.run_id,))
        for result in list_results(host.storage, run_id, scope_id)
    ]


def read_blind(
    judge: Judge | None,
    host: EvalHost,
    arms: Mapping[str, EvalSummary],
    control: str,
    scope_id: str,
    name: str,
    *,
    runs: Mapping[str, Sequence[str]] | None = None,
) -> tuple[BlindClose | None, tuple[CalibrationRating, ...]]:
    """Draw the run's blind read and read the labels ``judge`` names against it. Reads the store.

    One arm is read as singles; several as the control against the first candidate, in pairs. The draw is not
    stored: the same answers draw the same order, so a run replayed from its recordings reads the labels saved from
    an earlier run's page as answers to its own draw. A label that is not on this draw (an older ``to_label.csv``,
    another run's answers) pairs with the judge's score of the byte-identical answer and counts as not drawn at random.

    :return: The read's close over the labels (the read with its report, and the readings the print states), and the
        people's labels that are not on its draw; ``(None, ())`` with no judge or no judged answer to show.
    :raises ValueError: A labels file with a missing column, or a label not on its dimension's scale.
    """
    if judge is None:
        return None, ()
    results = _results_of(host, arms, scope_id, runs)
    evidence = {}
    for result in results:
        trace = get_result_trace(host.storage, result) if result.has_trace else None
        if trace is not None and trace.judge_evidence is not None:
            evidence[result.id] = trace.judge_evidence
    chosen = [control, *(arm for arm in arms if arm != control)][:2]
    read = draw_blind_read(
        results,
        evidence,
        judge.dims,
        [
            BlindArm(
                key=arm,
                role="control" if arm == control else "candidate",
                run_ids=list((runs or {}).get(arm, (arms[arm].run_id,))),
            )
            for arm in chosen
        ],
        scope_id=scope_id,
        created_by=CREATED_BY,
        name=f"{name}: blind read",
    )
    if not read.items:
        return None, ()
    labels = None
    if judge.labels is not None and judge.labels.is_file():
        labels = read_labels(judge.labels, read, results)
        read = read.model_copy(update={"answers": labels.answers})
    loose = tuple(labels.loose) if labels is not None else ()
    closed: BlindClose = close_blind_read(
        read,
        results,
        other_ratings=loose,
        unread=labels.unread if labels is not None else (),
        judge_models=[judge.model],
    )
    return closed, loose


def model_slug(model: str | CompletionClient) -> str:
    """A file-name-safe digest of a reviewer's model id: what a kept answers file is named by, never read back."""
    name = model if isinstance(model, str) else str(getattr(model, "model_name", ""))
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


def blind_reading(
    judge: Judge | None, closed: BlindClose | None, arms: Mapping[str, EvalSummary], kept: Path | None = None
) -> dict[str, Any]:
    """What a result carries of its blind read, as plain values: the files it writes, the lines it prints, and the
    judge's agreement with people. The read itself stays the engine's.

    :param judge: The eval's judge, or ``None``.
    :param closed: The run's blind read's close over its labels, or ``None`` where there is no judged answer to show.
    :param arms: Each arm's summary, by name: an arm is printed by the name every surface prints it by.
    :param kept: The file a model reviewer's answers are kept in, or ``None``.
    :return: The result's ``blind_read``, ``blind_read_files``, ``judge_read`` and ``calibrations``.
    """
    if judge is None:
        return {"blind_read": None, "blind_read_files": {}, "judge_read": (), "calibrations": ()}
    read = closed.read if closed is not None else None
    report = read.report if read is not None else None
    lines = judge_lines(
        closed,
        # The file by its name, never the path it was read from: a path is the machine's, not the reading's.
        labels_name=None if judge.labels is None else judge.labels.name,
        labels_there=judge.labels is not None and judge.labels.is_file(),
        loose=report.other_ratings if report is not None else 0,
        unread=report.unread if report is not None else (),
        shown={arm: _shown_arm_name(arm) for arm in arms},
        kept=None if kept is None else kept.name,
    )
    files = (
        {}
        if read is None
        else {BLIND_READ_HTML: blind_read_html(blind_sheet(read)), BLIND_READ_CSV: blind_read_csv(read)}
    )
    return {
        "blind_read": report,
        "blind_read_files": files,
        "judge_read": tuple(lines),
        "calibrations": calibrations_of(closed),
    }


def calibrations_of(closed: BlindClose | None) -> tuple[JudgeCalibration, ...]:
    """How the judge agrees with people's labels, per dimension, off a blind read's close.

    A model's answers are never here: this is agreement with people.

    :param closed: The close, or ``None``.
    :return: One per dimension a person's label pairs with.
    """
    if closed is None:
        return ()
    return tuple(
        JudgeCalibration(
            dimension=reading.dim,
            labels=reading.pairs,
            kappa=reading.kappa,
            lower=reading.lower,
            upper=reading.upper,
            calibration=reading.calibration,
            tier=reading.tier,
        )
        for reading in closed.agreement
        if reading.anchor == "person"
    )


#: The id the page's line on how the judge graded its examples carries.
_JUDGE_EXAMPLES_ANCHOR = "judge-examples"


def with_judge_examples(result: Result, examples: JudgeExamples | None) -> Result:
    """``result`` carrying how its judge graded its examples, its page saying so; ``result`` itself for ``None``.

    The page says it in Design & provenance, where it states how far the judge's scores can be read (the judge's
    calibration against people's labels), after that table: one sentence, then the misses. It is the eval's reading,
    taken in its call and not stored with the runs, so the line is set on this result's page; a page read again from
    the store has none, as it has no blind read.

    :param result: The result.
    :param examples: How the judge graded its examples, or ``None`` where it has none.
    :return: The result, its page with the line.
    """
    if examples is None:
        return result
    line = DisclosureBlock(
        section="provenance", source="calibration", text=examples.sentence(), anchor=_JUDGE_EXAMPLES_ANCHOR
    )
    return replace(
        result, judge_examples=examples, page=result.page.model_copy(update={"blocks": [*result.page.blocks, line]})
    )


def with_said(result: Result, line: str, *, anchor: str) -> Result:
    """``result`` saying ``line`` beside its readings: printed with its disclosures, and on its pages after their
    caveats (the aspect page's under its scoreboard, the campaign page's under its warnings).

    For what the eval knows of how it was graded and the runs do not store, so the line is set on this result's
    pages, as its judge's examples are; a page read again from the store has none.

    :param result: The result.
    :param line: The sentence.
    :param anchor: The id the line carries on each page.
    :return: The result, saying it.
    """
    report = result.report.model_copy(
        update={
            "blocks": _placed(
                result.report.blocks, DisclosureBlock(section="scoreboard", source="caveat", text=line, anchor=anchor)
            )
        }
    )
    page = result.page.model_copy(
        update={
            "blocks": _placed(
                result.page.blocks, DisclosureBlock(section="warnings", source="caveat", text=line, anchor=anchor)
            )
        }
    )
    return replace(result, disclosures=(*result.disclosures, line), report=report, page=page)


def _placed(blocks: Sequence[Any], line: DisclosureBlock) -> list[Any]:
    """``blocks`` with ``line`` after the last block of its section, else at the end."""
    held = list(blocks)
    at = max((index for index, block in enumerate(held) if block.section == line.section), default=len(held) - 1)
    return [*held[: at + 1], line, *held[at + 1 :]]


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
    runs: Mapping[str, Sequence[str]] | None = None,
    undo: Literal["easy", "hard"] = "easy",
    undo_ignored: bool = False,
) -> Result:
    """File the arms' runs as one campaign, one axis per factor, designate ``control``, and read its report.

    ``question`` is the one sentence the comparison answers, filed as the campaign's question: every reading on the
    aspect is then read as asked rather than as a lead. ``margin`` (a share of 0 to 1, the measure's units) is the
    author's own, filed on the aspect in points: a non-inferiority question, the pass rate's drop read against it.
    ``None`` files none, and nothing is assumed: a question with no margin is a superiority question, its drop read
    against zero. ``question`` ``None`` asks none, and a margin without one is refused. ``undo`` ``"hard"`` is filed on
    the aspect (``Aspect.undo``), raising the evidence its advice switches at; ``"easy"`` files nothing; ``undo_ignored``
    records an ``"easy"`` asked after the results over an earlier ``"hard"``, ignored (``Aspect.undo_ignored``). ``measure_order`` is each
    grader's measure in the order the eval lists them (:attr:`Result.grader_readings`), filed on the aspect's design so
    the campaign's page leads on the first of them, as the printed line does; empty files none.

    ``runs`` is each arm's runs, oldest first, where an arm reads more than its summary's one: a comparison continued
    over cases appended to it (:func:`~threetears.evals.quick.one_call.launch_arms`), whose every run the campaign
    holds. ``None`` files each summary's one run.

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
    # An arm named as factors is labelled by them on the campaign, every arm, so no page names one by its variant's
    # levers (`callable.effort=high, callable.prompt=v1`, sorted, the engine's kind in each lever's name).
    shown = {arm: _shown_arm_name(arm) for arm in ordered if coordinates(arm) is not None}
    axes = []
    for factor in factors:
        values = list(dict.fromkeys(levels[arm][factor] for arm in ordered))
        axes.append(
            {
                "axis_id": factor if factor in _UNPREFIXED else f"{prefix}.{factor}",
                # What every page calls the lever: the factor as the eval's author wrote it, never the engine's key.
                "reader_name": factor,
                "values": [{"content": value, "display": value} for value in values],
                "rationale": (
                    f"does any arm do better than {shown.get(control, control)}"
                    if len(factors) == 1
                    else f"does moving {factor} change what the arms score, against {shown.get(control, control)}"
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
        "run_ids": [run_id for arm, summary in arms.items() for run_id in (runs or {}).get(arm, (summary.run_id,))],
        "design": design,
    }
    refuse_an_unusable_margin(margin, question)
    refuse_an_unusable_undo(undo)
    if margin is not None:
        aspect["margin"] = round(100 * margin, 10)
    if undo == "hard":
        aspect["undo"] = "hard"
    if undo_ignored:
        aspect["undo_ignored"] = "easy"
    if arms[control].template_id is not None:
        aspect["template_id"] = arms[control].template_id
    # The control arm is named as the eval names it, so every page reads `hasty (control) vs careful`; an arm named as
    # factors by its factors, and every other such arm with it (`Prompt v1, effort low (control) vs Prompt v2, ...`).
    declared_arms = [
        {
            "label": shown.get(arm, arm),
            "role": "control" if arm == control else "candidate",
            "from_run_ids": list((runs or {}).get(arm, (arms[arm].run_id,))),
        }
        for arm in ordered
        if arm == control or arm in shown
    ]
    # The campaign's name is held to the cap every write of one is (a qualifier beside its title): an eval's name that
    # runs past it is cut at a word there, and kept whole in the campaign's description. The result keeps it whole.
    stored = _campaign_name(name)
    defined: dict[str, Any] = {
        "name": stored,
        "subject_id": name,
        "behavior": behavior,
        "aspects": [aspect],
        "arms": declared_arms,
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
    # An arm named as factors by the name the campaign labels it by, on this report as on the page.
    arm_labels = {arm_variants[arm]: name for arm, name in shown.items() if arm in arm_variants}
    report = build_code_only_report_in_forms(
        bundle,
        assembled_at=fold.report.source.generated_at,
        campaign_name=campaign.name,
        arm_labels=arm_labels or None,
        arm_tags=campaign_arm_tags([bundle], control_key, arms=list(arm_variants.values())),
        measure_order=list(dict.fromkeys(measure_order)),
        # In the campaign page's forms, so the report reads every figure as the page's fold of it does.
        forms=_aspect_forms(page, bundle),
    )
    arm_of_variant = {variant: arm for arm, variant in arm_variants.items()}
    surface = bundle_decision_surface(bundle)
    labels = report_cell_labels(bundle, arm_labels=arm_labels or None)
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
        arm_labels=arm_labels,
        grader_readings=tuple(measure_order),
        question=question,
        margin=margin,
    )


__all__ = [
    "CREATED_BY",
    "QUIET_MISSES",
    "JudgeCalibration",
    "Result",
    "blind_reading",
    "model_slug",
    "coordinates",
    "declare",
    "read_blind",
    "refuse_an_unusable_margin",
    "refuse_an_unusable_undo",
    "with_judge_examples",
    "with_said",
]
