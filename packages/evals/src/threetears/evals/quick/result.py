"""``Result``: what an eval ran, for one arm or a comparison of many, and what its campaign's report says.

Every eval's run ends in one :class:`Result`, whatever its runner: one arm, or a mapping of arms whose first key is
the control. Everything here is the engine's own path, composed:

- **Each arm is one run**, labelled by its arm name, all into one host and one scope, over one content-addressed case
  set. Two arms with different names are two variants. Named plainly (``{"baseline": a, "terse": b}``), an arm's name
  is the run's level of the host's arm lever (:data:`~threetears.evals.quick.one_call.ARM_LEVER`), every arm at one
  shared candidate model, so the report calls the arm ``candidate=<name>``. Named as ``k=v`` pairs
  (``"prompt=v2, effort=low"``), each pair is one factor of a factorial design: ``model`` is the run's candidate model
  and every other factor is a lever the engine names ``callable.<factor>`` and keys into the variant.
- **Every arm is started in one launch**, so they are measured side by side rather than one after another, and a
  refusal on the last arm leaves none run.
- **The campaign declares its design** — one axis per factor, a controlled stimulus, a commissioned apparatus and the
  repeats each arm ran — through :func:`~threetears.evals.analysis.create_campaign`, which gates the declaration
  against the host's vocabulary as it gates any other. One arm is a campaign of one run, its own control.
- **The control is designated from the control arm's run** through
  :func:`~threetears.evals.analysis.set_campaign_control`, which reads the variant key that run's observations carry.
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
    Report,
    TableBlock,
    assemble_context_bundle,
    build_code_only_report,
    create_campaign,
    get_campaign,
    judge_agreement,
    judge_self_agreement,
    report_markdown,
    set_campaign_control,
    variant_key_of_run,
)
from threetears.evals.kernel import (
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
    separation_criterion,
    tier_of,
)
from threetears.evals.schema import CalibrationRating, EvalResult, RubricScale, utc_now_iso
from threetears.evals.kernel.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.kernel.text_block import one_line
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.bundle.trials import trial_order
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.confusion import confusion_matrix, label_statistics
from threetears.evals.analysis.summary import CaseResult, EvalSummary
from threetears.evals.analysis.report.build import contrast_rows, report_cell_labels
from threetears.evals.quick.graders import UNUSABLE_ANSWER, expected_label, predicted_label
from threetears.evals.quick.judged import Judge
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

#: What an uncalibrated judge's line ends with: the step that calibrates it, at the calibrated tier's floor of labels.
_NEXT_STEP = (
    f"; next step: rate about {CALIBRATION_MIN_RESULTS} outputs in the {TO_LABEL} the report writes, without "
    "looking at the judge's verdict"
)

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
            figure = f"kappa {self.kappa:.2f}"
            if self.lower is not None:
                figure += f" (lower bound {self.lower:.2f})"
        return f"{self.dimension}: {self.labels} label(s), {figure}, calibration {self.calibration}; tier {self.tier}"


@dataclass(frozen=True)
class Result:
    """What an eval ran — one arm, or every arm of a comparison — and what its campaign's report says.

    ``print(result)`` is the quiet summary (:meth:`render`); :meth:`write` writes the report.

    :ivar name: The eval's name, which titles its campaign and report.
    :ivar scope_id: The scope the campaign and its runs are stored in.
    :ivar control: The arm every other arm is tested against: the runner mapping's first key, or the one arm.
    :ivar arms: Each arm's run summary, by arm name, in the order the arms were given.
    :ivar report: The campaign's report, as :func:`~threetears.evals.analysis.campaign_report` reads it.
    :ivar host: The host the runs and the campaign are stored in, for reading them further.
    :ivar campaign_id: The campaign holding every arm's run.
    :ivar factors: The factors each arm name names a level of, in order: ``("candidate",)`` when the arms are named
        plainly on the arm lever, ``("model",)`` when the one arm or the arms are named as models.
    :ivar store: Where the runs are stored, in words: the file, or that they are held in memory.
    :ivar fields: The eval's field graders, the classifier first.
    :ivar judge: The eval's judge, or ``None``.
    :ivar disclosures: What the run must say about how it was run, beside its readings: a tool recording replayed or
        made, a logged runner's file.
    :ivar calibrations: How the judge agrees with the labels ``Judge(labels=...)`` names, per dimension
        (:func:`calibrate`); empty with no judge, no labels, or no label that pairs.
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
    factors: tuple[str, ...] = (CANDIDATE_MODEL_LEVER,)
    store: str = ""
    fields: tuple[str, ...] = ()
    judge: Judge | None = field(default=None, repr=False, compare=False)
    disclosures: tuple[str, ...] = ()
    calibrations: tuple[JudgeCalibration, ...] = ()
    to_label: tuple[dict[str, str], ...] = field(default=(), repr=False, compare=False)
    contrast_arms: tuple[str | None, ...] = field(default=(), repr=False, compare=False)
    contrast_measures: tuple[str | None, ...] = field(default=(), repr=False, compare=False)
    kind: str = field(default=CALLABLE_KIND, repr=False, compare=False)
    guardrail_readings: GuardrailReadings = field(default_factory=GuardrailReadings, repr=False, compare=False)
    arm_variants: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    labelled_arms: dict[str, str | None] = field(default_factory=dict, repr=False, compare=False)

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

    # --- labels and calibration --------------------------------------------------------------------------------------

    def labels(
        self, grader: str | None = None, arm: str | None = None
    ) -> tuple[list[ConfusionCount], list[LabelStatistics]]:
        """One field grader's confusion matrix and each label's precision, recall and F1.

        The classifier's (the first field) are the run's own, as its summary carries them; any other field's are
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
        name = self.fields[0] if grader is None else grader
        if name not in self.fields:
            raise ValueError(f"{name!r} is no field grader of this eval; its fields are {', '.join(self.fields)}")
        if name == self.fields[0] and summary.confusion:
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
            return [f"judge: 0 labels: judge uncalibrated (Judge(labels=...) names a person's labels){_NEXT_STEP}"]
        if not judge.labels.is_file():
            return [f"judge: 0 labels: judge uncalibrated ({judge.labels} is not there yet){_NEXT_STEP}"]
        read = self.calibrations
        if not read:
            return [f"judge: 0 labels paired from {judge.labels}: judge uncalibrated{_NEXT_STEP}"]
        return [f"judge against {judge.labels}: {entry.line()}" for entry in read]

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
        if full:
            return "\n\n".join([*(summary.render() for summary in self.arms.values()), report_markdown(self.report)])
        one_arm = len(self.arms) == 1
        lines = [self._verdict_line()]
        if one_arm:
            # One arm's verdict is how its cases came out, and how far its judge can be trusted is read with it.
            lines += self.calibration_lines()
        for arm, summary in self.arms.items():
            label = "" if len(self.arms) == 1 else f"{arm}{' (control)' if arm == self.control else ''}: "
            lines.append(f"{label}{summary.results_text}")
            lines += [f"  {line}" for line in (summary.stopped_text,) if line is not None]
            lines += [f"  {line}" for line in self._measure_texts(summary, arm)]
            if summary.confusion:
                lines.append("  per label:")
                lines += [f"    {line}" for line in summary.label_lines]
            lines += [f"  {line}" for line in summary.dimension_lines]
            lines += [f"  {line}" for line in summary.goal_check_lines]
            # What it cost, priced or not, and the warnings no quiet print drops.
            for line in (summary.total_spend_text, summary.self_judging_warning, summary.model_identity_warning):
                if line is not None:
                    lines.append(f"  {line}")
            lines += [f"  error: {one_line(error)}" for error in summary.errors]
            lines += [f"  excluded: {cause}" for cause in _exclusion_causes(summary)]
        if not one_arm:
            lines += self.calibration_lines()
        order = self._page_order()
        misses = sorted(
            ((arm, miss) for arm, summary in self.arms.items() for miss in summary.misses()),
            key=lambda pair: order(pair[0], pair[1]),
        )
        if misses:
            lines.append(f"top misses ({min(len(misses), QUIET_MISSES)} of {len(misses)}):")
            for arm, miss in misses[:QUIET_MISSES]:
                who = "" if len(self.arms) == 1 else f"{arm}, "
                lines.append(f"  case {miss.case} ({who}repeat {miss.repeat}): {'; '.join(miss.missed_because)}")
        lines += self.disclosures
        lines.append(f"stored in {self.store}, scope {self.scope_id!r}")
        return "\n".join(lines)

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

    def _measure_texts(self, summary: EvalSummary, arm: str) -> list[str]:
        """Each measure's line, the classifier's ``match`` under its field's name and its matrix's count left out.

        In a comparison a guardrail's line says what the comparison decided on it for ``arm``; alone, that a guardrail
        is decided only in one.
        """
        texts: list[str] = []
        lines = (
            summary.measure_lines if len(self.arms) == 1 else summary.compared_measure_lines(self._guardrail_said(arm))
        )
        for measure in lines:
            if measure.name == CONFUSION_CELL_MEASURE or measure.state == "not_carried":
                continue
            text = measure.text
            if measure.name == MATCH_MEASURE and self.fields:
                text = f"{self.fields[0]}{text.removeprefix(MATCH_MEASURE)}"
            texts.append(text)
        if (omitted := summary.uncarried_text) is not None:
            texts.append(omitted)
        return texts

    def _verdict_line(self) -> str:
        """The first line: what the comparison decided, or how the one arm's cases came out (:func:`_cases_verdict`)."""
        if len(self.arms) == 1:
            (arm,) = self.arms
            order = self._page_order()
            return f"{self.name}: {_cases_verdict(self.summary, order=lambda result: order(arm, result))}"
        decided = [row for row in self.contrasts() if row["outcome"] in ("improved", "regressed", "equivalent")]
        if not decided:
            others = " or ".join(arm for arm in self.arms if arm != self.control)
            return f"{self.name}: nothing separated {others} from {self.control}"
        # The report's words, which name both arms: ``Accuracy: v2 better than v1``.
        return f"{self.name}: " + "; ".join(f"{row['reading']}: {row['verdict']}" for row in decided)

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        """Write the report as one self-contained page, its Markdown and its bundle, and the judged answers to label.

        :param out_dir: The directory; created when missing.
        :ptype out_dir: Path | str
        :return: Each file written, by role: ``html`` (``report.html``), ``bundle`` (``bundle.json``) and ``markdown``
            (``report.md``), and, for a judged eval, ``to_label`` (:data:`TO_LABEL`): each judged answer on each
            dimension, the judge's verdict withheld, for a person to label and ``Judge(labels=...)`` to read back.
        :rtype: dict[str, Path]
        """
        written: dict[str, Path] = {str(role): path for role, path in self.report.write(out_dir).items()}
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
        rows = contrast_rows(self.report.bundle)
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
            over each stratum's cases alone, by stratum name, the cases declaring none under ``"(no stratum)"``;
            a stratum the arm has no case in is absent); empty when no case declares a stratum.
        :rtype: list[dict[str, Any]]
        :raises ValueError: ``arm`` names no arm.
        """
        if arm is not None:
            _refuse_an_unknown_arm(self.arms, arm)
        tables = [block for block in self.report.blocks if isinstance(block, TableBlock) and block.name == "strata"]
        rows: list[dict[str, Any]] = []
        for table in tables:
            headers = {column.key: column.header for column in table.columns}
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

    async def against(self, control: str, *, name: str | None = None, created_by: str = CREATED_BY) -> Result:
        """The same runs read against another control: a second campaign over them, with no run repeated.

        The engine tests every arm against one control, so in a factorial the effect of one factor at the control's
        level of the other is in :attr:`report`, and its effect at another level is read here — the prompt's effect on
        the second model is the contrast against the first prompt on that model. Each campaign Holm-corrects its own
        contrasts, so the two reports are two families, not one.

        :param control: The arm to test every other against, by name.
        :ptype control: str
        :param name: The new campaign's name; ``None`` names it after this one and the new control.
        :ptype name: str | None
        :param created_by: Who the campaign and its control are recorded as created by.
        :ptype created_by: str
        :return: A result over the same arms and runs, against ``control``.
        :rtype: Result
        :raises ValueError: ``control`` names no arm.
        :raises ValidationFailedError: The host refuses the campaign's declaration.
        """
        _refuse_an_unknown_arm(self.arms, control)

        def declared() -> Result:
            campaign = get_campaign(self.host.storage, self.campaign_id, self.scope_id)
            design = campaign.declared_design
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
            )
            return replace(
                again,
                store=self.store,
                fields=self.fields,
                judge=self.judge,
                disclosures=self.disclosures,
                calibrations=self.calibrations,
                to_label=self.to_label,
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


def _refuse_an_unknown_arm(arms: Mapping[str, Any], arm: str) -> None:
    if arm not in arms:
        raise ValueError(f"{arm!r} names no arm; the arms are {', '.join(map(repr, arms))}")


def _cases_verdict(
    summary: EvalSummary, *, order: Callable[[CaseResult], tuple[str, int, str, int, int]] | None = None
) -> str:
    """How one arm's cases came out, in plain words: passed of usable, the cases missed, the cases excluded by cause.

    The cases missed are listed in ``order``, a case by its first result's place (the page's, from
    :meth:`Result._page_order`), else in the summary's.

    Counted as the page counts them, off each repeat's :attr:`~threetears.evals.analysis.summary.CaseResult.state`,
    which the page's own trial reader decided: a case is usable when any of its repeats was not lost, and passed when
    none of those failed — every check the template declares passed on each. A case every repeat of which was lost is
    counted under the cause its first repeat records.
    """
    repeats: dict[str, list[CaseResult]] = {}
    for result in summary.results():
        repeats.setdefault(result.case, []).append(result)
    usable = 0
    missed: list[str] = []
    excluded: dict[str, int] = {}
    for case, results in repeats.items():
        kept = [result for result in results if result.state != "lost"]
        if not kept:
            cause = (results[0].errors or ["no reason recorded"])[0]
            excluded[cause] = excluded.get(cause, 0) + 1
            continue
        usable += 1
        if any(result.state == "failed" for result in kept):
            missed.append(case)
    if order is not None:
        missed.sort(key=lambda case: min(order(result) for result in repeats[case]))
    text = f"{usable - len(missed)}/{usable} cases passed; {len(missed)} {'miss' if len(missed) == 1 else 'misses'}"
    if missed:
        text += f": {', '.join(missed[:QUIET_MISSES])}"
        if len(missed) > QUIET_MISSES:
            text += f", and {len(missed) - QUIET_MISSES} more"
    return text + "".join(f"; {count} excluded: {cause}" for cause, count in excluded.items())


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
) -> Result:
    """File the arms' runs as one campaign, one axis per factor, designate ``control``, and read its report.

    The report is the campaign's code-only report, as :func:`~threetears.evals.analysis.campaign_report` reads
    a campaign no analysis has been generated for — which a campaign filed a moment ago is — laid out here from
    the evidence bundle it is assembled from, so each contrast the report tests is matched to the arm whose run
    carries its variant key rather than to the words the report names it by. ``levels`` is each arm's level of
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
    filed: dict[str, Any] = {
        "name": name,
        "subject_id": name,
        "behavior": behavior,
        "run_ids": [summary.run_id for summary in arms.values()],
        "declared_design": design,
    }
    # Every arm ran the one template the eval authored, so the campaign names it: a judged guardrail's margin is
    # declared on a dimension of its rubric, which the declaration gate reads from it.
    if arms[control].template_id is not None:
        filed["template_id"] = arms[control].template_id
    campaign = create_campaign(
        host.storage,
        filed,
        scope_id=scope_id,
        created_by=created_by,
        profile=host.profile,
    )
    campaign = set_campaign_control(
        host.storage, campaign.id, scope_id, arms[control].run_id, set_by=created_by, profile=host.profile
    )
    bundle = assemble_context_bundle(campaign, storage=host.storage, profile=host.profile)
    report = build_code_only_report(bundle, assembled_at=utc_now_iso(), campaign_name=campaign.name)
    arm_variants = {
        arm: variant
        for arm, summary in arms.items()
        if (variant := variant_key_of_run(list_results(host.storage, summary.run_id, scope_id))) is not None
    }
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
    )


__all__ = [
    "CREATED_BY",
    "LABEL_COLUMNS",
    "QUIET_MISSES",
    "TO_LABEL",
    "JudgeCalibration",
    "Result",
    "calibrate",
    "coordinates",
    "declare",
    "judged_rows",
]
