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
- :class:`AuthoredJudgeCase` — a case written by hand rather than frozen from a stored result: a hard negative
  that must fail its criterion, or an output that looks wrong and must pass. It freezes into a :class:`JudgeCase`
  (:meth:`JudgeCase.from_authored`) whose one label is its expected verdict, so everything that reads a labelled
  case reads it unchanged.
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

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, ValidationError, model_validator

from threetears.evals.kernel.host.kinds import KindContract
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import (
    PASS_FAIL_SCORES,
    SCALES,
    ClientRequestSettings,
    DimName,
    EvalTestCase,
    GoalStateOutcome,
    JudgedArtifact,
    JudgeEvidence,
    JudgeTemperature,
    RubricAxis,
    RubricDim,
    RubricScale,
    KnownUsageRole,
    RatedOutput,
)

__all__ = [
    "JUDGE_CASE_KEY",
    "JUDGE_KIND",
    "JUDGE_KIND_CONTRACT",
    "JUDGE_KIND_MEASURES",
    "JUDGE_KIND_ROLE",
    "JUDGE_LABEL_AGREEMENT_MEASURE",
    "JUDGE_PARSE_VALID_MEASURE",
    "AuthoredJudgeCase",
    "AuthoredVerdict",
    "JudgeCase",
    "JudgeCaseAuthoring",
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
JUDGE_KIND_ROLE: KnownUsageRole = "judge"

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
    """One person's rating of the frozen output on the case's criterion, as it stood when the case was frozen.

    On an authored case, the one label is its author's expected verdict, written with the case: no rating is read.
    """

    rater: str = Field(min_length=1, description="Who rated, as the host names them.")
    score: int = Field(strict=True, description="1-5 on the ordinal scale; 1 (pass) or 0 (fail) on pass/fail.")
    reason: str = Field(min_length=1, description="The rater's own words for the score.")
    rating_id: str | None = Field(
        min_length=1,
        description=(
            "The CalibrationRating the label was read from; None = an authored case's expected verdict, which its "
            "author wrote with the case rather than as a rating of a stored result."
        ),
    )


class JudgeCaseAuthoring(EvalBaseModel):
    """Who wrote an authored judge case, and why its verdict is what it is: the case's provenance in place of a result."""

    author: str = Field(
        min_length=1, description="Who wrote the case and its expected verdict, as the host names them."
    )
    note: str = Field(
        min_length=1,
        description="The author's own words for the verdict: why the output must pass or fail, or earns its score.",
    )


class JudgeCaseSource(EvalBaseModel):
    """Where a judge case's output came from: the stored result and what its own judge answered, or its author.

    A case frozen from a durable label (``freeze_judge_cases(from_labels=True)``) names the run and result the label
    was given on as provenance, which a core baseline reset may since have deleted; it records no test case and no
    first judge (every ``first_*`` is None): the label carries what was rated, never who first judged it.

    An **authored** case (:class:`AuthoredJudgeCase`) was written by hand rather than produced by any run: it names
    its author (``authored``) and no run, result, test case or first judge, every one of which is None. Exactly one
    of the two is set: a stored result's run and result, or an author.
    """

    run_id: str | None = Field(
        default=None,
        min_length=1,
        description="The run the judged result belongs to; None = an authored case, which no run produced.",
    )
    result_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The judged result whose stored evidence the case replays; None = an authored case, which replays none."
        ),
    )
    test_case_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The source run's test case the result ran; None = frozen from a durable label, which records the run "
            "and result it was given on but not their test case."
        ),
    )
    first_score: int | None = Field(
        default=None,
        description=(
            "The score the result's own judge gave the criterion; None when it gave none, or when the case was frozen "
            "from a durable label, where the first judge is not recorded."
        ),
    )
    first_served_model: str | None = Field(default=None, description="The model that served that score.")
    first_judge_config_id: str | None = Field(
        default=None,
        description=(
            "The versioned JudgeConfig that asked for it, or a ``builtin:`` id = the built-in prompt, by its wording; "
            "None = no judge identity recorded (a score made outside a judge phase)."
        ),
    )
    first_judge_temperature: JudgeTemperature | None = Field(
        default=None, description="The temperature that call was sent at; None = not recorded."
    )
    first_judge_sent_settings: ClientRequestSettings | None = Field(
        default=None,
        description=(
            "The request settings that call was actually sent with, as the score recorded them; None = unverified. "
            "Left out of the case's content digest while None, so a case frozen before the field existed keeps its "
            "digest."
        ),
    )
    authored: JudgeCaseAuthoring | None = Field(
        default=None,
        description=(
            "Who wrote the case, for an authored case; None = a case frozen from a stored result or a label. Left "
            "out of the case's content digest while None, so a case frozen before the field existed keeps its digest."
        ),
    )

    @model_validator(mode="after")
    def _a_result_or_an_author(self) -> Self:
        """Refuse a source naming both a stored result and an author, or neither, or an authored one claiming a judge.

        Raises:
            ValueError: The source is not exactly one of a stored result (its run and result) or an author, or an
                authored source records a test case or a first judge, which no hand-written case has.
        """
        stored = self.run_id is not None and self.result_id is not None
        if self.authored is None:
            if not stored:
                raise ValueError(
                    "a judge case's source names the run and result its output was stored by, or its author"
                )
            return self
        recorded = {
            name: value
            for name, value in (
                ("run_id", self.run_id),
                ("result_id", self.result_id),
                ("test_case_id", self.test_case_id),
                ("first_score", self.first_score),
                ("first_served_model", self.first_served_model),
                ("first_judge_config_id", self.first_judge_config_id),
                ("first_judge_temperature", self.first_judge_temperature),
                ("first_judge_sent_settings", self.first_judge_sent_settings),
            )
            if value is not None
        }
        if recorded:
            raise ValueError(
                f"an authored judge case was produced by no run and judged by no judge, and this source records "
                f"{sorted(recorded)}"
            )
        return self


#: An expected verdict as its author states it: a score on the criterion's scale (1-5 ordinal; 1 pass, 0 fail), or
#: ``"pass"`` / ``"fail"`` on a pass/fail criterion.
AuthoredVerdict = Annotated[int, Field(strict=True)] | Literal["pass", "fail"]


def _verdict_score(dim: str, scale: RubricScale, verdict: int | str) -> int:
    """``verdict`` as the score it stands for on ``scale``, refused when the scale cannot hold it.

    Raises:
        ValueError: The verdict is ``"pass"`` / ``"fail"`` on an ordinal criterion, or a score outside the scale's
            bounds — named with the criterion and its scale, never moved onto the nearest value the scale holds.
    """
    spec = SCALES[scale]
    low, high = spec.scores
    if isinstance(verdict, str):
        if scale != "pass_fail" or verdict not in PASS_FAIL_SCORES:
            raise ValueError(
                f"criterion {dim!r} is answered on the {scale} scale ({spec.reads_as}), which cannot hold the "
                f"verdict {verdict!r}; give a score from {low} to {high}"
            )
        return PASS_FAIL_SCORES[verdict]
    if not low <= verdict <= high:
        raise ValueError(
            f"criterion {dim!r} is answered on the {scale} scale ({spec.reads_as}), which cannot hold the verdict "
            f"{verdict}; give a score from {low} to {high}" + (" ('pass' or 'fail')" if scale == "pass_fail" else "")
        )
    return verdict


class AuthoredJudgeCase(EvalBaseModel):
    """A calibration case written by hand: an output, the criterion it is judged on, and the verdict it must get.

    What a person writes when no stored result holds the output they need — a **hard negative** that must fail
    the criterion, or an output that looks wrong and must pass it. It carries what a judge reads when it judges a
    stored result, at its least: the criterion as the subject's template words it, the scenario (the template's
    intent), what the output is judged against, and the output itself. Frozen into a judge template
    (:func:`~threetears.evals.run.freeze_judge_cases`, ``authored=``), it is a :class:`JudgeCase` whose one label
    is its expected verdict, so the judge kind, the readout and a judge profile treat it as a labelled case.

    Refused at authoring when the criterion's scale cannot hold the verdict, naming both: a hard negative whose
    verdict is quietly moved is a calibration item that measures nothing.
    """

    criterion: RubricDim = Field(
        description=(
            "The rubric dimension the output is judged on, worded exactly as the subject's template words it: its "
            "wording is the criterion a judge profile is keyed by, so a reworded copy calibrates another criterion."
        )
    )
    output: str = Field(min_length=1, description="The output under review, as the judge reads it.")
    case_material: str = Field(min_length=1, description="What the output is judged against: its source, or its facts.")
    intent: str = Field(
        min_length=1, description="The scenario the judge reads: the subject template's intent, what each case asks."
    )
    expected: AuthoredVerdict = Field(
        description=(
            "The verdict the output must get: 'pass' or 'fail' (or 1 / 0) on a pass/fail criterion, a score from 1 to "
            "5 on an ordinal one."
        )
    )
    author: str = Field(min_length=1, description="Who wrote the case, as the host names them: the label's rater.")
    note: str = Field(min_length=1, description="Why the output must get that verdict: the label's reason.")
    subject: str | None = Field(
        default=None,
        description="Who or what produced the output, as the judge should know it; None when it speaks for itself.",
    )
    judged_artifact: JudgedArtifact = Field(
        default=JudgedArtifact.DOCUMENT,
        description="Whether the output is a document judged against its material, or a conversation's transcript.",
    )

    @model_validator(mode="after")
    def _a_verdict_its_scale_holds(self) -> Self:
        """Refuse a verdict the criterion's scale cannot hold, and an output no judge reads.

        Raises:
            ValueError: The verdict is off the criterion's scale, naming both; or the artifact is declared unjudged.
        """
        _verdict_score(self.criterion.name, self.criterion.scale, self.expected)
        if self.judged_artifact is JudgedArtifact.UNJUDGED:
            raise ValueError("an authored judge case's output is one a judge reads: a document or a transcript")
        return self

    @property
    def expected_score(self) -> int:
        """The verdict as the stored score it stands for: 1 (pass) or 0 (fail), or the 1-5 score."""
        return _verdict_score(self.criterion.name, self.criterion.scale, self.expected)


class JudgeCase(EvalBaseModel):
    """One output and the criterion it is judged on, frozen from a stored result — and what people said about it.

    Everything the judge read when it first scored the output is here, as the cell recorded it: the template's
    intent, the case's variation, the goal-state outcomes, and the evidence the cell's kind rendered for its judge.
    A trial rebuilds the same judge context from these and asks the judge under test the criterion through the
    engine's own judge service, so the evidence block it reads is the one the first judge read; only the judge —
    its model, its prompt and its sampling — is the arm's.

    Frozen because everything it is read from moves: a template is edited, a rating is replaced, a result is
    re-judged. A case that re-read them at each trial would change under every arm scored on it.

    **An authored case** (:meth:`from_authored`) is one a person wrote rather than one a run stored: a hard negative
    that must fail its criterion, or an output that looks wrong and must pass. Its expected verdict IS its label —
    one label, by its author, carrying no rating id — so a trial, the readout and a judge profile read it exactly as
    they read a labelled case.
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

    @model_validator(mode="after")
    def _an_authored_case_is_labelled_by_its_verdict(self) -> Self:
        """Refuse an authored case whose labels are not its one verdict, and a stored one with a label read from no rating.

        Raises:
            ValueError: An authored case carries other than exactly one label, by its author and with no rating id,
                on the criterion's scale; or a case frozen from a stored result carries a label with no rating id.
        """
        authored = self.source.authored
        if authored is None:
            if unrated := [label.rater for label in self.labels if label.rating_id is None]:
                raise ValueError(
                    f"a judge case frozen from a stored result carries labels read from ratings, and the label(s) by "
                    f"{unrated} name none"
                )
            return self
        if len(self.labels) != 1 or self.labels[0].rating_id is not None or self.labels[0].rater != authored.author:
            raise ValueError(
                f"an authored judge case carries exactly one label, its expected verdict by its author "
                f"{authored.author!r} with no rating id; this one carries {len(self.labels)} label(s)"
            )
        _verdict_score(self.dim, self.scale, self.labels[0].score)
        return self

    @classmethod
    def from_authored(cls, authored: AuthoredJudgeCase) -> Self:
        """The case a hand-written calibration case freezes into, read from no run.

        Args:
            authored: The case as its author wrote it, its verdict already held to its criterion's scale.

        Returns:
            The case: the criterion and scenario as written, the output judged as the authored artifact declares,
            no variation and no goal outcomes, and the expected verdict as its one label.
        """
        return cls(
            dim=authored.criterion.name,
            scale=authored.criterion.scale,
            criterion=authored.criterion,
            intent=authored.intent,
            judged_artifact=authored.judged_artifact,
            evidence=JudgeEvidence(
                subject=authored.subject, case_material=authored.case_material, artifact=authored.output
            ),
            labels=[
                JudgeCaseLabel(
                    rater=authored.author, score=authored.expected_score, reason=authored.note, rating_id=None
                )
            ],
            source=JudgeCaseSource(authored=JudgeCaseAuthoring(author=authored.author, note=authored.note)),
        )

    @classmethod
    def from_rated(cls, rated: RatedOutput, labels: list[JudgeCaseLabel], source: JudgeCaseSource) -> Self:
        """The case a durable label's rated content freezes into, read from no result.

        Args:
            rated: What the label's rater read (``CalibrationRating.rated``), proven against its label key when
                the rating was written.
            labels: The person labels on that output and criterion.
            source: Where the content came from, as provenance (a label's run and result, which may be gone).

        Returns:
            The case, its content field for field the rated content's.
        """
        return cls(
            dim=rated.dim,
            scale=rated.scale,
            criterion=rated.criterion,
            intent=rated.intent,
            variation=dict(rated.variation),
            goal_outcomes=list(rated.goal_outcomes),
            judged_artifact=rated.judged_artifact,
            evidence=rated.evidence,
            labels=labels,
            source=source,
        )

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

        ``source.first_judge_sent_settings`` and ``source.authored`` are left out of the pre-image while they are
        None, so a case whose first score recorded no sent settings, and every case frozen from a stored result,
        digests as it did before the field existed (the digest-default rule of
        :mod:`~threetears.evals.schema.versioning`).

        Returns:
            The digest.
        """
        frozen = self.model_dump(mode="json")
        for added in ("first_judge_sent_settings", "authored"):
            if frozen["source"][added] is None:
                del frozen["source"][added]
        return canonical_digest(frozen)


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
    axis: RubricAxis = Field(description="The axis the criterion was asked on, as the judge request stated it.")
    criterion_digest: str = Field(min_length=1, description="The case's criterion digest (JudgeCase.criterion_digest).")
    case_digest: str = Field(min_length=1, description="The case's content digest (JudgeCase.content_digest).")
    source_result_id: str | None = Field(
        min_length=1,
        description="The stored result whose output the case replays; None = an authored case, which replays none.",
    )
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
        default=None,
        description=(
            "The versioned JudgeConfig that asked, or a ``builtin:`` id = the built-in prompt, by its wording; None = "
            "no judge identity was recorded: an unknown judge, never the same as a recorded one."
        ),
    )
    judge_temperature: JudgeTemperature | None = Field(
        default=None,
        description=(
            "What the call was sent at, as its client reported it; None only for a call that returned no completion "
            "to read it from (a trial with no reply). A scored trial always carries it."
        ),
    )
    judge_sent_settings: ClientRequestSettings | None = Field(
        default=None,
        description=(
            "The request settings the call was actually sent with, as its client reported them "
            "(``sent_settings_of``); None = unverified: the client did not report them, or the call returned no "
            "completion to read them from. Part of the judge's identity beside its temperature."
        ),
    )

    @model_validator(mode="after")
    def _a_score_says_what_it_was_sent_at(self) -> Self:
        """Refuse a scored trial with no temperature: its score could not be told from another judge's."""
        if self.outcome == "scored" and self.judge_temperature is None:
            raise ValueError("a scored judge trial records the temperature its call was sent at")
        return self


def judge_trial_of(kind_payload: object) -> JudgeTrial | None:
    """The judge trial a result's ``kind_payload`` carries, or ``None`` when it carries none.

    Args:
        kind_payload: A result's ``kind_payload``.

    Returns:
        The trial, or ``None`` when the result carries no ``kind_payload`` object at all.

    Raises:
        ValueError: It carries one that is not a judge trial this build can read — named apart from carrying none,
            as :func:`judge_case_of` names an unreadable case, so a reader can say why it left the result out.
    """
    if not isinstance(kind_payload, dict):
        return None
    try:
        return JudgeTrial.model_validate(kind_payload)
    except ValidationError as exc:
        raise ValueError(f"its kind_payload is not a judge trial this build can read: {exc}") from exc


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
