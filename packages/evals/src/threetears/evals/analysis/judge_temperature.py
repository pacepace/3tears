"""Read a judge temperature comparison's answers: per-dimension score variance and self-agreement at each setting (#633).

:func:`~threetears.evals.run.judge_at_two_temperatures` re-judges a finished run's borderline cases at the pinned
temperature and at the provider's default and hands back every answer
(:class:`~threetears.evals.kernel.JudgeTemperatureAnswers`). :func:`read_judge_temperatures` reads them, by code that
spends nothing:

- **Per case,** the scores answered at each setting, their sample variance, and whether every answer was the same (a
  "can't tell" counted as an answer of its own).
- **Per setting,** each case's later answers paired with its first answer there and read by the code every
  self-agreement is read by (:func:`~threetears.evals.analysis.agreement.self_agreement_of_repeats`, the core of
  :func:`~threetears.evals.analysis.judge_self_agreement`): exact agreement and kappa, a "can't tell" a disagreement.
- **The temperature sent is checked, never assumed.** A score recorded at anything other than its setting's
  temperature — a client that dropped it, or reports none — is left out of that side's figures and counted, and the
  comparison is marked not comparable, with the reason.
"""

from __future__ import annotations

import statistics

from pydantic import Field

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.summary import dollars_text
from threetears.evals.analysis.agreement import JudgeSelfAgreement, self_agreement_of_repeats
from threetears.evals.kernel.judge_temperature import (
    TEMPERATURE_SETTINGS,
    JudgeTemperatureAnswers,
    TemperatureAnswer,
    TemperatureCaseAnswers,
    TemperatureSelection,
    TemperatureSetting,
    TemperatureSkip,
)
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import JudgeRepeat, JudgeTemperature, RepeatedScore, RubricScale


class TemperatureCase(EvalBaseModel):
    """One result's dimension, judged ``repeats`` times at one setting: its answers and their spread."""

    result_id: str
    rubric_dim: str
    scale: RubricScale
    stored_score: int = Field(description="The score the run's judge stored, which made the case borderline or not.")
    scores: list[int] = Field(description="The scores answered at this setting's temperature, in call order.")
    cannot_tell: int = Field(ge=0, description="Answers that said the judge could not tell.")
    failed: int = Field(ge=0, description="Calls that failed — an infrastructure fault, saying nothing of the judge.")
    off_setting: int = Field(
        ge=0, description="Scores recorded at a temperature other than this setting's, left out of `scores`."
    )
    variance: float | None = Field(
        description="The sample variance of `scores`; None under two scores. 0 = the judge gave one score every time."
    )
    stable: bool | None = Field(
        description=(
            'Whether every answer was the same, a "can\'t tell" counted as an answer of its own; None under two answers.'
        )
    )


class TemperatureSide(EvalBaseModel):
    """One dimension at one setting: how much its cases' scores moved across repeats, and the judge's self-agreement."""

    cases: int = Field(ge=0, description="Cases with at least two answers at this setting.")
    mean_variance: float | None = Field(
        description="The mean over cases of each case's score variance; None when no case has two scores."
    )
    max_variance: float | None = Field(description="The largest case variance; None when no case has two scores.")
    unstable_cases: int = Field(ge=0, description="Cases whose answers were not all the same.")
    exact_agreement: float | None = Field(
        description=(
            "The share of repeats answering what the case's first answer at this setting did, as "
            "`judge_self_agreement` reads it; None when it read no pair, or split the dim across judges."
        )
    )
    kappa: float | None = Field(description="Cohen's kappa of the same pairs; None when undefined or not read.")
    weighted_kappa: float | None = Field(description="Quadratic-weighted kappa on 1-5; None on pass/fail or undefined.")


class TemperatureDimension(EvalBaseModel):
    """One dimension, the two settings side by side."""

    rubric_dim: str
    scale: RubricScale
    pinned: TemperatureSide
    provider_default: TemperatureSide


class TemperatureSettingRead(EvalBaseModel):
    """Everything one setting's answers say, case by case."""

    setting: TemperatureSetting
    requested: float | None = Field(description="The temperature each call was requested at; None = sent none.")
    recorded: list[str] = Field(
        description=(
            "The temperatures the answers recorded being sent at, sorted, as text: a number, 'model_default' (sent "
            "none), or 'unrecorded' (the client reports nothing)."
        )
    )
    off_setting: int = Field(ge=0, description="Scores recorded at a temperature other than `requested`.")
    self_agreement: JudgeSelfAgreement = Field(
        description="Each case's later answers paired with its first at this setting, read by judge_self_agreement."
    )
    cases: list[TemperatureCase]


class JudgeTemperatureComparison(EvalBaseModel):
    """The measurement: a run's borderline cases re-judged at the pinned temperature and at the provider's default.

    Attributes:
        run_id: The run.
        judge_model: The run's judge pin.
        selection: Which scored dims were re-judged.
        repeats: Calls per dim per setting.
        results: The results re-judged.
        cases: The (result, dim) pairs re-judged.
        dimensions: Per dimension, the two settings side by side.
        settings: Per setting, every case and the full self-agreement read.
        comparable: Whether every score on each side was recorded at that side's temperature. False means the
            figures do not compare the two settings as named, and ``incomparable`` says why.
        incomparable: Why not, when not.
        skipped: The run's results not re-judged, each with why — decided before anything was spent.
        stopped: Why the comparison stopped before its last result, when it did.
        calls_made: Judge calls made, as the ledger recorded them under purpose ``judge``.
        cost_usd: What they cost together; ``None`` when any went unpriced.
        cap_usd: The out-of-run cap they were admitted under; ``None`` when the host enforces none.
    """

    run_id: str
    judge_model: str
    selection: TemperatureSelection
    repeats: int
    results: int
    cases: int
    dimensions: list[TemperatureDimension]
    settings: list[TemperatureSettingRead]
    comparable: bool
    incomparable: str | None = None
    skipped: list[TemperatureSkip]
    stopped: str | None = None
    calls_made: int
    cost_usd: float | None
    cap_usd: float | None

    def render(self) -> str:
        """The comparison as text: what was asked and spent, then each dimension's two settings side by side."""
        cost = "unpriced" if self.cost_usd is None else dollars_text(self.cost_usd)
        requested = {read.setting: read for read in self.settings}
        pinned = requested["pinned"].requested
        lines = [
            f"judge temperature comparison on run {self.run_id} (judge {self.judge_model}): {self.cases} "
            f"{self.selection} case(s) over {self.results} result(s), {self.repeats} repeat(s) at temperature "
            f"{format_number(pinned)} and at the provider default; {self.calls_made} call(s), {cost} — measurement cost, "
            "ledgered under judge, never the candidate's",
        ]
        if not self.comparable:
            lines.append(f"NOT COMPARABLE: {self.incomparable}")
        if self.stopped:
            lines.append(f"stopped: {self.stopped}")
        lines += [f"skipped {skip.result_id}: {skip.reason}" for skip in self.skipped]
        for read in self.settings:
            lines.append(f"{read.setting}: answers recorded at {', '.join(read.recorded) or 'nothing'}")
        lines.append(
            "per dimension — cases, mean / max score variance across repeats, unstable cases, exact agreement, kappa"
        )
        for row in self.dimensions:
            lines.append(f"- {row.rubric_dim} ({row.scale})")
            lines.append(f"    temperature {format_number(pinned)}:      {_side_text(row.pinned)}")
            lines.append(f"    provider default:   {_side_text(row.provider_default)}")
        return "\n".join(lines)


def _side_text(side: TemperatureSide) -> str:
    """One setting of one dimension, in a line."""
    agreement = "n/a" if side.exact_agreement is None else f"{side.exact_agreement:.0%}"
    kappa = side.weighted_kappa if side.weighted_kappa is not None else side.kappa
    return (
        f"{side.cases} case(s), variance {format_number(side.mean_variance)} / {format_number(side.max_variance)}, "
        f"{side.unstable_cases} unstable, exact agreement {agreement}, kappa {format_number(kappa)}"
    )


def _recorded_text(temperature: JudgeTemperature | None) -> str:
    """A recorded temperature as the report spells it."""
    if temperature is None:
        return "unrecorded"
    return temperature if isinstance(temperature, str) else format_number(temperature)


def _case(answered: TemperatureCaseAnswers, expected: JudgeTemperature) -> tuple[TemperatureCase, set[str]]:
    """One case's answers at one setting, and the temperatures they recorded."""
    scores: list[int] = []
    answers: list[int | None] = []  # None = "can't tell", an answer of its own
    recorded: set[str] = set()
    cannot_tell = failed = off = 0
    for answer in answered.answers:
        if answer.score is not None:
            recorded.add(_recorded_text(answer.score.judge_temperature))
            if answer.score.judge_temperature != expected:
                off += 1
                continue
            scores.append(answer.score.score)
            answers.append(answer.score.score)
        elif answer.cannot_tell is not None:
            cannot_tell += 1
            answers.append(None)
        else:
            failed += 1
    case = TemperatureCase(
        result_id=answered.result_id,
        rubric_dim=answered.rubric_dim,
        scale=answered.scale,
        stored_score=answered.stored_score,
        scores=scores,
        cannot_tell=cannot_tell,
        failed=failed,
        off_setting=off,
        variance=statistics.variance(scores) if len(scores) >= 2 else None,
        stable=len(set(answers)) == 1 if len(answers) >= 2 else None,
    )
    return case, recorded


def _repeats_of(answered: TemperatureCaseAnswers, judge_model: str) -> list[JudgeRepeat]:
    """One case's answers at one setting as repeats of its first scored answer there, for the self-agreement read.

    The anchor is the first answer that is a score; every later answer is a repeat of it, so the rounds are
    ``repeat 1`` (the next answer) onward. Never stored. Empty when no answer is a score.
    """
    anchor_at = next((index for index, answer in enumerate(answered.answers) if answer.score is not None), None)
    if anchor_at is None:
        return []
    anchor: TemperatureAnswer = answered.answers[anchor_at]
    assert anchor.score is not None
    repeats: list[JudgeRepeat] = []
    for answer in answered.answers[anchor_at + 1 :]:
        entry = RepeatedScore(
            dim=answered.rubric_dim,
            scale=anchor.score.scale,
            first_score=anchor.score.score,
            first_served_model=anchor.score.served_model,
            first_judge_config_id=anchor.config_id,
            first_judge_temperature=anchor.score.judge_temperature,
            repeat=answer.score,
            error=(answer.error or "no answer") if answer.score is None and answer.cannot_tell is None else None,
            cannot_tell=answer.cannot_tell,
        )
        repeats.append(
            JudgeRepeat(
                judge_model=judge_model,
                scores=[entry],
                judge_config_ids={answered.rubric_dim: answer.config_id} if answer.config_id is not None else {},
            )
        )
    return repeats


def _side(cases: list[TemperatureCase], agreement: JudgeSelfAgreement, dim: str) -> TemperatureSide:
    """One dimension at one setting, from its cases and that setting's self-agreement."""
    answered = [case for case in cases if case.stable is not None]
    variances = [case.variance for case in cases if case.variance is not None]
    groups = [group for group in agreement.dimensions if group.rubric_dim == dim]
    # One judge (model, config, temperature) answered every pair, or the figures would pool two judges: none then.
    group = groups[0] if len(groups) == 1 else None
    return TemperatureSide(
        cases=len(answered),
        mean_variance=statistics.fmean(variances) if variances else None,
        max_variance=max(variances) if variances else None,
        unstable_cases=sum(1 for case in answered if not case.stable),
        exact_agreement=None if group is None else group.exact_agreement,
        kappa=None if group is None else group.kappa,
        weighted_kappa=None if group is None else group.weighted_kappa,
    )


def read_judge_temperatures(answers: JudgeTemperatureAnswers) -> JudgeTemperatureComparison:
    """Read a judge temperature comparison's answers into per-dimension variance and self-agreement, side by side.

    Spends nothing and reads nothing but ``answers``, so the same answers always read the same.

    Args:
        answers: What :func:`~threetears.evals.run.judge_at_two_temperatures` asked and was answered.

    Returns:
        The comparison: per dimension the two settings side by side, per setting every case, whether the two sides
        compare the settings as named, and the case count and spend carried over.
    """
    reads: list[TemperatureSettingRead] = []
    cases_by: dict[TemperatureSetting, list[TemperatureCase]] = {}
    agreement_by: dict[TemperatureSetting, JudgeSelfAgreement] = {}
    problems: list[str] = []
    for spec in TEMPERATURE_SETTINGS:
        cases: list[TemperatureCase] = []
        recorded: set[str] = set()
        repeats: dict[str, list[JudgeRepeat]] = {}
        for answered in answers.answers:
            if answered.setting != spec.setting:
                continue
            case, seen = _case(answered, spec.recorded)
            cases.append(case)
            recorded |= seen
            repeats.setdefault(answered.result_id, []).extend(_repeats_of(answered, answers.judge_model))
        agreement = self_agreement_of_repeats((result_id, judged) for result_id, judged in repeats.items() if judged)
        cases_by[spec.setting], agreement_by[spec.setting] = cases, agreement
        off_setting = sum(case.off_setting for case in cases)
        if off_setting:
            others = sorted(recorded - {_recorded_text(spec.recorded)})
            problems.append(
                f"{off_setting} score(s) at the {spec.setting} setting were recorded at "
                f"{', '.join(others) or 'another temperature'}, not {_recorded_text(spec.recorded)}"
            )
        reads.append(
            TemperatureSettingRead(
                setting=spec.setting,
                requested=spec.requested,
                recorded=sorted(recorded),
                off_setting=off_setting,
                self_agreement=agreement,
                cases=cases,
            )
        )
    scales: dict[str, RubricScale] = {}
    for answered in answers.answers:
        scales.setdefault(answered.rubric_dim, answered.scale)
    dimensions = [
        TemperatureDimension(
            rubric_dim=dim,
            scale=scale,
            pinned=_side([c for c in cases_by["pinned"] if c.rubric_dim == dim], agreement_by["pinned"], dim),
            provider_default=_side(
                [c for c in cases_by["provider_default"] if c.rubric_dim == dim], agreement_by["provider_default"], dim
            ),
        )
        for dim, scale in sorted(scales.items())
    ]
    incomparable = None
    if problems:
        incomparable = (
            "; ".join(problems)
            + " — the judge's client did not send (or does not report sending) the temperature each side names, so "
            "those scores are left out and the two sides do not compare the settings as named"
        )
    return JudgeTemperatureComparison(
        run_id=answers.run_id,
        judge_model=answers.judge_model,
        selection=answers.selection,
        repeats=answers.repeats,
        results=answers.results,
        cases=answers.cases,
        dimensions=dimensions,
        settings=reads,
        comparable=incomparable is None,
        incomparable=incomparable,
        skipped=answers.skipped,
        stopped=answers.stopped,
        calls_made=answers.calls_made,
        cost_usd=answers.cost_usd,
        cap_usd=answers.cap_usd,
    )


__all__ = [
    "JudgeTemperatureComparison",
    "TemperatureCase",
    "TemperatureDimension",
    "TemperatureSettingRead",
    "TemperatureSide",
    "read_judge_temperatures",
]
