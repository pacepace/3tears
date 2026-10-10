"""A judge campaign's readout: per judge and criterion, agreement with the labels, with itself, and parse validity (#628).

A judge campaign's cells are judge trials (:class:`~threetears.evals.kernel.JudgeTrial`, on each result's
``kind_payload``): one judge configuration asked one frozen case's criterion. This module pools them into the three
measures a judge is held to, **per (judge, criterion)** — the judge as :class:`~threetears.evals.analysis.JudgeKey`
keys one (the criterion's dim and scale, the served model, the config that asked, the temperature the call was sent
at), and the criterion as the case worded it (:meth:`~threetears.evals.kernel.JudgeCase.criterion_digest`):

- **agreement with the labels** — the trial's score against the person labels its case froze, read by
  :func:`~threetears.evals.analysis.judge_agreement`;
- **self-agreement** — the judge's later answers to one case against its first, read by
  :func:`~threetears.evals.analysis.judge_self_agreement`;
- **parse validity** — the share of replies that kept the protocol (a score on the scale, or the offered "can't
  tell") among every reply that came back.

**Nothing here computes a statistic of its own.** Each (judge, criterion, case) is restated as the shape the two
agreement reads already take — one result holding the judge's first score on the case, its later answers recorded
beside it as repeats, and the case's labels as ratings of that result — and handed to them. So a judge measured as a
subject and a judge measured inside the campaign it scored are read by one computation, their figures compare, and
the evidence tiers' floors (which count distinct results) count distinct CASES here: a case repeated k times weighs
one case, in self-agreement as in calibration.

**The first scored trial is the reading.** Agreement with the labels pairs each case's first scored trial (by run,
then repeat) with its labels, as a run's judge is paired with ratings of its one score; every later trial of the
case is a repeat. A case none of whose trials scored still counts in ``cases`` and in parse validity, and in no
agreement.

Pure: the caller hands in the results; nothing here reads storage.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, TypeVar

from pydantic import Field

from threetears.evals.analysis.agreement import (
    DimensionAgreement,
    JudgeKey,
    SelfAgreementDimension,
    judge_agreement,
    judge_self_agreement,
)
from threetears.evals.kernel.judge_cases import JudgeTrial, judge_trial_of
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import (
    CalibrationRating,
    JudgeRepeat,
    JudgeTemperature,
    RepeatedScore,
    RubricScale,
    RubricScore,
)

if TYPE_CHECKING:
    from threetears.evals.schema.models import EvalResult

__all__ = [
    "JudgeKindReading",
    "JudgeKindReadings",
    "JudgeParseValidity",
    "UnreadJudgeTrial",
    "judge_kind_readings",
]


class JudgeParseValidity(EvalDocumentModel):
    """How often one judge's replies on one criterion kept the protocol."""

    replies: int = Field(ge=0, description="Trials the judge replied to: scored, can't tell and invalid.")
    scored: int = Field(ge=0, description="Replies carrying a score on the criterion's scale.")
    cannot_tell: int = Field(
        ge=0, description="Replies answering, in the protocol, that the evidence does not decide it."
    )
    invalid: int = Field(ge=0, description="Replies that broke the protocol: unreadable, off the scale, or cut short.")
    rate: float | None = Field(
        ge=0.0, le=1.0, description="(scored + can't tell) / replies; None when there was no reply to read."
    )


class JudgeKindReading(EvalDocumentModel):
    """One judge's three measures on one criterion, over the frozen cases it was asked."""

    rubric_dim: str = Field(min_length=1, description="The criterion's dim.")
    scale: RubricScale = Field(description="The scale it was asked on.")
    judge_model: str | None = Field(description="The model the judge's responses named; None when none named one.")
    judge_config_id: str | None = Field(description="The versioned JudgeConfig that asked; None = the built-in prompt.")
    judge_temperature: JudgeTemperature | None = Field(
        description="The temperature the calls were sent at, as the client reported it; None = not reported."
    )
    criterion_digest: str = Field(min_length=1, description="The criterion as the cases worded it (its digest).")
    case_set_fingerprint: str = Field(
        min_length=1,
        description=(
            "A digest of the cases the reading is over — each case's id and content digest, sorted — so two readings "
            "over the same frozen cases say so, and a reading over a changed case set does not pass for one."
        ),
    )
    cases: int = Field(ge=0, description="Distinct cases the judge replied to on this criterion.")
    trials: int = Field(ge=0, description="Trials read: every reply, a case's repeats included.")
    run_ids: list[str] = Field(description="The runs the trials came from, sorted.")
    measured_at: str = Field(description="When the measurement was taken: the latest scored_at among its trials.")
    parse_validity: JudgeParseValidity
    label_agreement: DimensionAgreement | None = Field(
        description=(
            "Agreement with the cases' person labels, by judge_agreement; None when no labelled case was scored. Its "
            "`results` counts distinct cases."
        ),
    )
    self_agreement: SelfAgreementDimension | None = Field(
        description=(
            "Agreement of each case's later answers with its first, by judge_self_agreement; None when no case was "
            "answered twice after a score. Its `results` counts distinct cases."
        ),
    )

    @property
    def key(self) -> JudgeKey:
        """The judge this reading measured, as every judge reading is keyed."""
        return JudgeKey(self.rubric_dim, self.scale, self.judge_model, self.judge_config_id, self.judge_temperature)


class UnreadJudgeTrial(EvalDocumentModel):
    """A result the readout left out, and why."""

    result_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class JudgeKindReadings(EvalDocumentModel):
    """Every judge and criterion a judge campaign's results measured, and every result left out."""

    readings: list[JudgeKindReading] = Field(
        default_factory=list,
        description="One per (judge, criterion), ordered by dim, scale, judge, config, temperature.",
    )
    unread: list[UnreadJudgeTrial] = Field(
        default_factory=list,
        description=(
            "Results left out: no judge trial on them, a harness fault (excluded, as every lens excludes it), or a "
            "judge call that failed before any reply (the candidate's failure, with nothing to read)."
        ),
    )

    def reading(self, key: JudgeKey, criterion_digest: str | None = None) -> JudgeKindReading | None:
        """The reading for one judge (and, when given, one criterion wording), or ``None`` when none was measured."""
        return next(
            (
                reading
                for reading in self.readings
                if reading.key == key and (criterion_digest is None or reading.criterion_digest == criterion_digest)
            ),
            None,
        )


#: (judge, criterion digest) — what one reading is about.
_GroupKey = tuple[JudgeKey, str]


def _key_of(trial: JudgeTrial) -> JudgeKey:
    """The judge behind one trial's reply."""
    return JudgeKey(trial.dim, trial.scale, trial.served_model, trial.judge_config_id, trial.judge_temperature)


def _order(entry: tuple[EvalResult, JudgeTrial]) -> tuple[str, int, str]:
    """A case's trials in the order they are read: run, then repeat, then id."""
    result, _ = entry
    return (result.eval_run_id, result.k_iteration, result.id)


def _score_of(trial: JudgeTrial) -> RubricScore:
    """A scored trial's answer as the score a judged result would hold."""
    assert trial.score is not None
    return RubricScore(
        dim=trial.dim,
        scale=trial.scale,
        score=trial.score,
        reasoning=trial.reasoning or "",
        served_model=trial.served_model,
        judge_temperature=trial.judge_temperature,
    )


def _reading_of_case(
    case_id: str, entries: Sequence[tuple[EvalResult, JudgeTrial]]
) -> tuple[EvalResult, list[CalibrationRating]] | None:
    """One case's trials as the shape the agreement reads take: a result holding the first score, the rest as repeats.

    Returns:
        The restated result and its labels as person ratings of it, or ``None`` when no trial of the case scored.
    """
    first_index = next((i for i, (_, trial) in enumerate(entries) if trial.outcome == "scored"), None)
    if first_index is None:
        return None
    base, first = entries[first_index]
    first_score = _score_of(first)
    configs = {first.dim: first.judge_config_id} if first.judge_config_id is not None else {}
    repeats = []
    for index, (_, trial) in enumerate(entries):
        if index == first_index:
            continue
        repeated = RepeatedScore(
            dim=first.dim,
            scale=first.scale,
            first_score=first_score.score,
            first_served_model=first.served_model,
            first_judge_config_id=first.judge_config_id,
            first_judge_temperature=first.judge_temperature,
            repeat=_score_of(trial) if trial.outcome == "scored" else None,
            cannot_tell=(trial.cannot_tell or "the judge gave no reason") if trial.outcome == "cannot_tell" else None,
            error=(trial.error or "the reply broke the protocol") if trial.outcome == "invalid" else None,
        )
        repeats.append(
            JudgeRepeat(
                judge_model=trial.served_model or base.model,
                judge_config_ids={trial.dim: trial.judge_config_id} if trial.judge_config_id is not None else {},
                scores=[repeated],
            )
        )
    # The trial's own result, restated: the case is the unit both agreements count, so the reading carries its id.
    reading = base.model_copy(
        update={
            "id": case_id,
            "rubric_scores": [first_score],
            "transcript_score": None,
            "outcome_score": None,
            "judge_config_ids": configs,
            "judge_repeats": repeats,
        }
    )
    ratings = [
        CalibrationRating(
            scope_id=base.scope_id,
            run_id=base.eval_run_id,
            result_id=case_id,
            rubric_dim=first.dim,
            rater=label.rater,
            rater_kind="person",
            scale=first.scale,
            score=label.score,
            reason=label.reason,
        )
        for label in first.labels
    ]
    return reading, ratings


#: The agreement row either read returns.
_Row = TypeVar("_Row", DimensionAgreement, SelfAgreementDimension)


def _matching(rows: Iterable[_Row], key: JudgeKey) -> _Row | None:
    """The one agreement row measured for ``key``, or ``None``."""
    for row in rows:
        if JudgeKey(row.rubric_dim, row.scale, row.judge_model, row.judge_config_id, row.judge_temperature) == key:
            return row
    return None


def judge_kind_readings(results: Iterable[EvalResult]) -> JudgeKindReadings:
    """Pool a judge campaign's trials into agreement with the labels, self-agreement and parse validity, per judge and criterion.

    Args:
        results: The judge campaign's results — every result of its runs. A result carrying no judge trial is listed
            as unread, never guessed at.

    Returns:
        The readings, and every result left out with why.
    """
    groups: dict[_GroupKey, list[tuple[EvalResult, JudgeTrial]]] = {}
    unread: list[UnreadJudgeTrial] = []
    for result in results:
        trial = judge_trial_of(result.kind_payload)
        if trial is None:
            unread.append(UnreadJudgeTrial(result_id=result.id, reason="it carries no judge trial"))
            continue
        if result.infra_error is not None:
            unread.append(UnreadJudgeTrial(result_id=result.id, reason=f"harness fault: {result.infra_error}"))
            continue
        if trial.outcome == "no_reply":
            unread.append(
                UnreadJudgeTrial(result_id=result.id, reason=f"the judge's call failed: {trial.error or 'no reply'}")
            )
            continue
        groups.setdefault((_key_of(trial), trial.criterion_digest), []).append((result, trial))
    readings: list[JudgeKindReading] = []
    for (key, criterion), entries in groups.items():
        by_case: dict[str, list[tuple[EvalResult, JudgeTrial]]] = {}
        for entry in sorted(entries, key=_order):
            by_case.setdefault(entry[1].test_case_id, []).append(entry)
        restated: list[EvalResult] = []
        ratings: list[CalibrationRating] = []
        for case_id, case_entries in sorted(by_case.items()):
            read = _reading_of_case(case_id, case_entries)
            if read is not None:
                restated.append(read[0])
                ratings.extend(read[1])
        trials = [trial for _, trial in entries]
        scored = sum(1 for trial in trials if trial.outcome == "scored")
        cannot_tell = sum(1 for trial in trials if trial.outcome == "cannot_tell")
        invalid = sum(1 for trial in trials if trial.outcome == "invalid")
        label = _matching(judge_agreement(ratings, restated).dimensions, key)
        repeat = _matching(judge_self_agreement(restated).dimensions, key)
        readings.append(
            JudgeKindReading(
                rubric_dim=key.rubric_dim,
                scale=key.scale,
                judge_model=key.judge_model,
                judge_config_id=key.judge_config_id,
                judge_temperature=key.judge_temperature,
                criterion_digest=criterion,
                case_set_fingerprint=canonical_digest(
                    sorted({(trial.test_case_id, trial.case_digest) for trial in trials})
                ),
                cases=len(by_case),
                trials=len(trials),
                run_ids=sorted({result.eval_run_id for result, _ in entries}),
                measured_at=max(result.scored_at for result, _ in entries),
                parse_validity=JudgeParseValidity(
                    replies=len(trials),
                    scored=scored,
                    cannot_tell=cannot_tell,
                    invalid=invalid,
                    rate=(scored + cannot_tell) / len(trials) if trials else None,
                ),
                label_agreement=label,
                self_agreement=repeat,
            )
        )
    readings.sort(
        key=lambda r: (
            r.rubric_dim,
            r.scale,
            r.judge_model or "",
            r.judge_config_id or "",
            "" if r.judge_temperature is None else str(r.judge_temperature),
            r.criterion_digest,
        )
    )
    return JudgeKindReadings(readings=readings, unread=unread)
