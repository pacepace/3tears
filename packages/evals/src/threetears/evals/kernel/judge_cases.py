"""The judge as a subject: what a judge case freezes, what a judge trial records, and the kind's declaration (#628).

A campaign's judged scores come from one judge, and the engine measures that judge only inside the campaign it
scored — its agreement with people's ratings of those results, and with its own repeats of them. The **judge kind**
makes a judge the subject of a campaign of its own: the candidate is a judge configuration (a model, the prompt per
criterion, the temperature it is sampled at — what :class:`~threetears.evals.analysis.JudgeKey` keys a judge by),
each case is one stored output and the criterion it was judged on, frozen with whatever people said about it, and
a trial asks the judge under test that one question again. Nothing re-buys the candidate whose output it was: the
output is replayed from what its run stored.

This module is the part of that kind every side of the engine reads, so it lives in the kernel:

- :class:`JudgeCase` — one frozen case, carried on its test case's ``host_payload`` under :data:`JUDGE_CASE_KEY`.
  It is written by the freeze (:func:`~threetears.evals.run.freeze_judge_cases`) and read by the kind
  (:class:`~threetears.evals.run.JudgeKind`).
- :class:`JudgeTrial` — what one trial recorded, carried on its result's ``kind_payload``. It is written by the
  kind and read by the readout (:func:`~threetears.evals.analysis.judge_kind_readings`), which pools trials into
  agreement with the labels, agreement with the judge's own repeats, and parse validity, per judge and criterion.
- :data:`JUDGE_KIND_CONTRACT` — the kind's contract: the overlays an arm turns (:class:`JudgeKindOverlays`, the
  prompt per criterion and the temperature; the judge's model is the run's candidate model), so two arms that
  differ in their judge are two variants every lens can tell apart.
- :data:`JUDGE_KIND_MEASURES` — the two code-graded measures a trial lands on ``host_measures``, for a host to
  declare on its measure registry.

**Why the grade is code.** A judge kind's cell is graded against what is already known about the case — the labels
people gave the output — and against the judge's own other answers to the same question. Neither needs a second
model, so the kind declares itself unjudged and no judge phase runs over its cells: the only model a trial pays
for is the judge under test, and its spend is recorded under the ``judge`` role (:data:`JUDGE_KIND_ROLE`).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from threetears.evals.kernel.host.kinds import KindContract
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import (
    DimName,
    EvalTestCase,
    GoalStateOutcome,
    JudgedArtifact,
    JudgeEvidence,
    JudgeTemperature,
    RubricDim,
    RubricScale,
    UsageRole,
)

__all__ = [
    "JUDGE_CASE_KEY",
    "JUDGE_KIND",
    "JUDGE_KIND_CONTRACT",
    "JUDGE_KIND_MEASURES",
    "JUDGE_KIND_ROLE",
    "JUDGE_LABEL_AGREEMENT_MEASURE",
    "JUDGE_PARSE_VALID_MEASURE",
    "JudgeCase",
    "JudgeCaseLabel",
    "JudgeCaseSource",
    "JudgeKindOverlays",
    "JudgeTrial",
    "JudgeTrialOutcome",
    "judge_case_of",
    "judge_case_payload",
    "judge_criterion_digest",
    "judge_trial_of",
]

#: The ``candidate_kind`` a judge template declares.
JUDGE_KIND = "judge"

#: The key a judge case lives under inside ``EvalTestCase.host_payload``.
JUDGE_CASE_KEY = "judge_case"

#: The role every call a judge trial makes is metered under. The judge under test is the cell's subject, but what
#: it spends is judge spend: no candidate model runs in a judge campaign, so no ``candidate`` row is ever written.
JUDGE_KIND_ROLE: UsageRole = "judge"

#: The measure a trial lands when the judge under test answered: whether its reply kept the protocol — a score on
#: the criterion's scale, or the offered "can't tell" — rather than a reply that could not be read.
JUDGE_PARSE_VALID_MEASURE = "judge_parse_valid"

#: The measure a scored trial of a labelled case lands: the share of the case's person labels the score equals.
JUDGE_LABEL_AGREEMENT_MEASURE = "judge_label_agreement"

#: How a trial ended, as far as its readout is concerned.
#:
#:   ``scored``       the judge answered with a score on the criterion's scale.
#:   ``cannot_tell``  the judge answered, in the protocol, that the evidence does not let it score the criterion.
#:   ``invalid``      the judge answered, and the reply broke the protocol (unparseable, off the scale, or cut
#:                    short by the provider) — a parse failure, which is what parse validity counts.
#:   ``no_reply``     the judge's call itself failed, so there was no reply to read; the cell is the candidate's
#:                    failure and no measure reads it.
JudgeTrialOutcome = Literal["scored", "cannot_tell", "invalid", "no_reply"]


class JudgeCaseLabel(EvalBaseModel):
    """One person's rating of the frozen output on the case's criterion, as it stood when the case was frozen."""

    rater: str = Field(min_length=1, description="Who rated, as the host names them.")
    score: int = Field(strict=True, description="1-5 on the ordinal scale; 1 (pass) or 0 (fail) on pass/fail.")
    reason: str = Field(min_length=1, description="The rater's own words for the score.")
    rating_id: str = Field(min_length=1, description="The CalibrationRating the label was read from.")


class JudgeCaseSource(EvalBaseModel):
    """Where a judge case's output came from: the stored result, and what its own judge answered."""

    run_id: str = Field(min_length=1, description="The run the judged result belongs to.")
    result_id: str = Field(min_length=1, description="The judged result whose stored evidence the case replays.")
    test_case_id: str = Field(min_length=1, description="The source run's test case the result ran.")
    first_score: int | None = Field(
        default=None, description="The score the result's own judge gave the criterion; None when it gave none."
    )
    first_served_model: str | None = Field(default=None, description="The model that served that score.")
    first_judge_config_id: str | None = Field(
        default=None, description="The versioned JudgeConfig that asked for it; None = the built-in prompt."
    )
    first_judge_temperature: JudgeTemperature | None = Field(
        default=None, description="The temperature that call was sent at; None = not recorded."
    )


class JudgeCase(EvalBaseModel):
    """One output and the criterion it is judged on, frozen from a stored result — and what people said about it.

    Everything the judge read when it first scored the output is here, as the cell recorded it: the template's
    intent, the case's variation, the goal-state outcomes, and the evidence the cell's kind rendered for its judge.
    A trial rebuilds the same judge context from these and asks the judge under test the criterion through the
    engine's own judge service, so the evidence block it reads is the one the first judge read; only the judge —
    its model, its prompt and its sampling — is the arm's.

    Frozen because everything it is read from moves: a template is edited, a rating is replaced, a result is
    re-judged. A case that re-read them at each trial would change under every arm scored on it.
    """

    dim: DimName = Field(description="The criterion: a template rubric dim's name, or a reserved axis id.")
    scale: RubricScale = Field(description="The scale the criterion is answered on.")
    criterion: RubricDim | None = Field(
        default=None,
        description=(
            "The rubric dimension as the source template worded it — its description and scoring guide are the "
            "built-in prompt's criterion. None for a reserved axis, whose criterion is the engine's own."
        ),
    )
    intent: str = Field(description="The source template's intent, which the judge reads as the scenario.")
    variation: dict[str, str] = Field(default_factory=dict, description="The source case's variation parameters.")
    goal_outcomes: list[GoalStateOutcome] = Field(
        default_factory=list, description="The source cell's goal-state outcomes, which the judge reads as evidence."
    )
    judged_artifact: JudgedArtifact = Field(description="The source kind's declaration, which words the evidence.")
    evidence: JudgeEvidence = Field(description="What the source cell's kind rendered for its judge, verbatim.")
    labels: list[JudgeCaseLabel] = Field(
        default_factory=list, description="Person ratings of the output on the criterion; empty for an unlabelled case."
    )
    source: JudgeCaseSource

    def criterion_digest(self) -> str:
        """A digest of the criterion as the case asks it: its id, scale and wording.

        Two cases of one dim worded differently are two criteria, so a readout never pools a judge's answers to
        two questions under one name.

        Returns:
            The digest.
        """
        return judge_criterion_digest(self.dim, self.scale, self.criterion)

    def content_digest(self) -> str:
        """A digest of everything the case freezes, its labels included — what makes two freezes the same case.

        Returns:
            The digest.
        """
        return canonical_digest(self.model_dump(mode="json"))


def judge_criterion_digest(dim: str, scale: RubricScale, criterion: RubricDim | None) -> str:
    """A digest of one criterion as a judge is asked it: its id, scale and wording (#628).

    The one derivation behind :meth:`JudgeCase.criterion_digest` and the stored judge profile's criterion, so a
    campaign whose judge read a template's rubric dimension finds the profile measured on cases frozen from it, and
    a dimension reworded since is another criterion whose profile it never reads.

    Args:
        dim: The criterion's id: a rubric dim's name, or a reserved axis id.
        scale: The scale it is answered on.
        criterion: The rubric dimension as the template words it; None for a reserved axis, whose criterion is
            the engine's own.

    Returns:
        The digest.
    """
    wording = None if criterion is None else criterion.model_dump(mode="json")
    return canonical_digest({"dim": dim, "scale": scale, "criterion": wording})


def judge_case_payload(case: JudgeCase) -> dict[str, Any]:
    """The ``host_payload`` a judge ``EvalTestCase`` carries: this module's own schema, under its own key.

    The one writer of the payload :func:`judge_case_of` reads back, so the judge kind reads no key it does not own
    (the self-keyed lane of the engine's opaque-payload gate, which the reporter kind's case takes too).

    Args:
        case: The case to store.

    Returns:
        The payload.
    """
    return {JUDGE_CASE_KEY: case.model_dump(mode="json")}


def judge_case_of(test_case: EvalTestCase) -> JudgeCase | None:
    """The judge case a test case carries, or ``None`` when it carries none.

    Args:
        test_case: A stored case.

    Returns:
        The case, or ``None`` when its ``host_payload`` holds nothing under :data:`JUDGE_CASE_KEY`.

    Raises:
        ValueError: It holds something there that is not a judge case this build can read.
    """
    payload = test_case.host_payload
    if not isinstance(payload, dict) or JUDGE_CASE_KEY not in payload:
        return None
    try:
        return JudgeCase.model_validate(payload[JUDGE_CASE_KEY])
    except ValidationError as exc:
        raise ValueError(f"test case {test_case.id!r} carries an unreadable judge case: {exc}") from exc


class JudgeTrial(EvalBaseModel):
    """What one judge trial recorded: the case it asked about, the judge that answered, and the answer.

    Carried on the trial's result as its ``kind_payload``, so the readout reads trials off the results alone. The
    case's labels ride along, frozen as the case froze them, so what a score is compared with is what the case
    said when the trial ran.
    """

    test_case_id: str = Field(min_length=1, description="The judge case the trial asked about.")
    dim: DimName = Field(description="The criterion asked.")
    scale: RubricScale = Field(description="The scale it was asked on.")
    criterion_digest: str = Field(min_length=1, description="The case's criterion digest (JudgeCase.criterion_digest).")
    case_digest: str = Field(min_length=1, description="The case's content digest (JudgeCase.content_digest).")
    source_result_id: str = Field(min_length=1, description="The stored result whose output the case replays.")
    labels: list[JudgeCaseLabel] = Field(default_factory=list, description="The case's person labels.")
    outcome: JudgeTrialOutcome
    score: int | None = Field(default=None, description="The judge's score, when it scored.")
    reasoning: str | None = Field(default=None, description="The judge's reasoning, when it answered.")
    cannot_tell: str | None = Field(default=None, description="The judge's reason, when it could not tell.")
    error: str | None = Field(default=None, description="Why there is no valid answer, for an invalid or failed call.")
    served_model: str | None = Field(
        default=None, description="The model the response named as having answered; None when it named none."
    )
    judge_config_id: str | None = Field(
        default=None, description="The versioned JudgeConfig that asked; None = the built-in prompt."
    )
    judge_temperature: JudgeTemperature | None = Field(
        default=None, description="What the call was sent at, as its client reported it; None = not reported."
    )


def judge_trial_of(kind_payload: object) -> JudgeTrial | None:
    """The judge trial a result's ``kind_payload`` carries, or ``None`` when it carries none this build can read.

    Args:
        kind_payload: A result's ``kind_payload``.

    Returns:
        The trial, or ``None``.
    """
    if not isinstance(kind_payload, dict):
        return None
    try:
        return JudgeTrial.model_validate(kind_payload)
    except ValidationError:
        # NOSILENT: None IS the answer -- a payload this build cannot read as a trial is not one, as documented
        return None


class JudgeKindOverlays(BaseModel):
    """What one arm of a judge campaign turns besides its model: the prompt per criterion, and the temperature.

    The judge's model is the arm's candidate model (``EvalRun.candidate_model``), which the engine already keys a
    variant by. These two are the rest of a judge's identity, declared as the kind's overlays so each is a lever
    (``judge.config_ids``, ``judge.config_ids.<criterion>``, ``judge.temperature``) that keys the arm's variant and
    reads as an axis of the campaign.
    """

    config_ids: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "criterion -> the versioned JudgeConfig whose prompt asks it; a criterion absent from the map is asked "
            "with the built-in prompt"
        ),
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "the temperature every call is requested at; None = what each criterion's prompt asks for (its "
            "config's temperature, else the default every unconfigured criterion is judged at)"
        ),
    )


#: The judge kind's contract: its overlays, and the rig it fills. It fills no seat — the judge under test is the
#: candidate, not the run's judge, so the engine's judge and simulator dimensions are not blanks on its runs but
#: things they do not have.
JUDGE_KIND_CONTRACT = KindContract(JUDGE_KIND, overlays=JudgeKindOverlays, seats=frozenset())

#: The measures a judge trial lands on ``host_measures``, for a host to declare on its measure registry. Both are
#: graded by code against what the case froze; agreement with the judge's own repeats is read across trials, by the
#: readout (:func:`~threetears.evals.analysis.judge_kind_readings`), since no one trial carries it.
JUDGE_KIND_MEASURES: tuple[MetricDescriptor, ...] = (
    MetricDescriptor(
        name=JUDGE_PARSE_VALID_MEASURE,
        reader_name="Judge reply parsed",
        data_type="boolean",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=True,
        description=(
            "Whether the judge under test answered in its protocol: a score on the criterion's scale, or the offered "
            "'can't tell'. False for a reply that could not be read or was cut short."
        ),
    ),
    MetricDescriptor(
        name=JUDGE_LABEL_AGREEMENT_MEASURE,
        reader_name="Judge agreed with the label",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=True,
        merit_axis="quality",
        population="scored",
        value_range=(0.0, 1.0),
        description="The share of the case's person labels the judge's score equals, on a scored trial of a labelled case.",
    ),
)
