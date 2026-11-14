"""A blind read: a campaign's judged outputs in a seeded random order, a rater's answers to it, and what it found.

One idea carries the whole shape: **the population in a seeded random order.** Every case where each chosen arm has
a judged, usable answer is one *item*; the items are a seeded permutation, and a rater answers them from the top. A
prefix of a random permutation is a simple random sample, so nobody picks a sample size first, and "is this sample
random" is a structural fact (:class:`BlindStanding`): the answered items are the first ``m`` of the order, or they
are not.

**There is no key to hide.** What a rater is shown is a :class:`BlindSheet`: the outputs, the case each answers and
the criteria as worded. It names no arm, run or result, and carries no judge score. Unblinding is a lookup the
analysis already performs, by the label key of each output (its fingerprint and the criterion's); what must be kept
is the draw (:class:`BlindDraw`), which lives on the :class:`BlindRead` and never on the sheet.

**Who answered is part of every answer** (:data:`BlindRaterKind`). A ``person`` is a person. A ``reference`` is a
model that answered the same drawn sheet, named by its model id: a reference reviewer. A reference reviewer's answer
is never a person's, on any surface: it is read apart, and where both exist the person's decides.

The shapes here are plain documents with no behaviour. The mechanism (drawing, the sheet, the standing, closing) is
:mod:`threetears.evals.analysis.blind_read`.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field, model_validator

from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.models import DimName, JudgeEvidence, RubricDim, RubricScale, utc_now_iso

#: The slot an output is shown in. A single is always ``A``; a pair's two outputs are ``A`` and ``B`` as drawn.
BlindSlot = Literal["A", "B"]

#: Who answered a blind read: a ``person``, or a ``reference`` reviewer (a model that answered the drawn sheet).
BlindRaterKind = Literal["person", "reference"]

#: What a rater preferred of a pair: the output in a slot, or neither.
BlindPreferenceAnswer = Literal["A", "B", "tie"]

#: What a blind read asks of each item: a verdict per criterion per output, and, of a pair, which output is better.
BlindAsk = Literal["verdicts", "preference"]

#: Whether a rater's answered items are a random sample of the population: ``random`` (the first ``m`` of the drawn
#: order), ``selected`` (any other set), or ``none`` (nothing answered).
BlindSampleStanding = Literal["random", "selected", "none"]

#: How a preference read: the candidate shown better or worse, not told apart, too few non-tie items to test, or
#: ``withheld``: the read has not reached its planned length, so no direction, count or p is given. A preference
#: read after every item and stopped when it separates is wrong far more often than its level says, and a running
#: count tells the rater which arm is ahead while they are still reading.
BlindPreferenceVerdict = Literal["improved", "regressed", "not_separated", "too_few", "withheld"]

#: The test a preference was read by: the exact sign test (one answer per item), or the bounded test on shares.
BlindPreferenceTest = Literal["exact_sign", "bounded_share"]

#: The prefix a reference reviewer's rater name carries on every surface, before its model id.
REFERENCE_RATER_PREFIX: Final = "reference:"


class BlindArm(EvalDocumentModel):
    """One arm a blind read was drawn over: what names it, its role, and the runs that hold its answers."""

    key: str = Field(min_length=1, description="What names the arm where the read was drawn: its variant key.")
    role: Literal["control", "candidate"] = Field(description="The control, or the candidate read against it.")
    run_ids: list[str] = Field(min_length=1, description="The runs whose results are this arm's answers.")


class BlindDraw(EvalDocumentModel):
    """The draw: what the order was drawn from and by which seed. Kept on the read, never on the sheet."""

    seed: int = Field(description="The seed the order, the pick among repeats and the slots were drawn by.")
    population: int = Field(ge=0, description="The cases every chosen arm gave a judged, usable answer on.")
    population_digest: str = Field(
        min_length=1,
        description=(
            "A digest of the population's label keys, sorted: the same answers under the same criteria draw the same "
            "order, and labels drawn over other answers are seen to be."
        ),
    )
    planned_items: int = Field(
        ge=0,
        description=(
            "The length the preference is read at, fixed when the read was drawn: the first this-many items of the "
            "order, and no others. Before a rater has answered all of them the preference is withheld."
        ),
    )
    excluded: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Cases left out of the order or of its pairs, counted by reason: an arm gave no judged answer; no stored "
            "evidence; identical outputs (shown once, as a single, with no preference asked)."
        ),
    )


class BlindOutput(EvalDocumentModel):
    """One output of an item: what a rater reads, and (off the sheet) which result it is."""

    slot: BlindSlot = Field(description="The slot it is shown in.")
    arm: str = Field(min_length=1, description="The arm it came from (BlindArm.key). Never on the sheet.")
    run_id: str = Field(min_length=1, description="The run that holds it. Provenance; may dangle. Never on the sheet.")
    result_id: str = Field(min_length=1, description="The result it is. Provenance; never on the sheet.")
    output_fingerprint: str = Field(
        min_length=1, description="The fingerprint of the evidence, which with a criterion's is its label key."
    )
    criterion_fingerprints: dict[str, str] = Field(
        description="Each asked criterion's fingerprint as the judge's score of this output carries it, by dimension."
    )
    evidence: JudgeEvidence = Field(description="What the judge read, verbatim: the case material and the output.")


class BlindItem(EvalDocumentModel):
    """One case in the drawn order: a single output, or a pair in slots A and B."""

    item: str = Field(min_length=1, description="An opaque id: a digest of the seed, the population and the position.")
    case_id: str = Field(min_length=1, description="The test case. Never on the sheet.")
    outputs: list[BlindOutput] = Field(min_length=1, max_length=2, description="One for a single; A then B for a pair.")


class BlindVerdict(EvalDocumentModel):
    """A rater's verdict on one criterion of one output."""

    slot: BlindSlot = Field(description="The output it is of.")
    dim: DimName = Field(description="The criterion.")
    score: int = Field(strict=True, description="1 (pass) or 0 (fail) on pass/fail; 1 to 5 on the ordinal scale.")
    reason: str = Field(default="", description="The rater's own words, where given.")


class BlindAnswer(EvalDocumentModel):
    """One rater's answer to one item: its verdicts, its preference of a pair, or that it was skipped."""

    item: str = Field(min_length=1, description="The item answered (BlindItem.item).")
    rater: str = Field(min_length=1, description="Who answered: a person as the host names them, or a model id.")
    rater_kind: BlindRaterKind = Field(
        default="person",
        description="A person, or a reference reviewer (a model). A reference reviewer's answer is never a person's.",
    )
    reviewer_served_model: str | None = Field(
        default=None,
        min_length=1,
        description="The model that served a reference reviewer's answer, as the provider named it. None for a person.",
    )
    reviewer_effort: str | None = Field(
        default=None, min_length=1, description="The effort a reference reviewer was asked for, where it was."
    )
    collected_by: str | None = Field(
        default=None, min_length=1, description="Who relayed a person's answer into the read, where someone did."
    )
    answered_at: str = Field(default_factory=utc_now_iso)
    verdicts: list[BlindVerdict] = Field(default_factory=list)
    prefer: BlindPreferenceAnswer | None = Field(default=None, description="Of a pair: the better output, or a tie.")
    skipped: bool = Field(default=False, description="The rater passed on the item: counted, in no figure.")

    @model_validator(mode="after")
    def _a_skip_answers_nothing_and_only_a_reference_names_a_served_model(self) -> BlindAnswer:
        """Refuse a skip that carries an answer, and a person's answer that names a model.

        :raises ValueError: Naming which.
        """
        if self.skipped and (self.verdicts or self.prefer is not None):
            raise ValueError(f"the answer to item {self.item!r} is skipped, so it carries no verdict or preference")
        if self.rater_kind == "person" and (self.reviewer_served_model or self.reviewer_effort):
            raise ValueError("a person's answer names no served model or effort; those are a reference reviewer's")
        return self


class BlindStanding(EvalDocumentModel):
    """Whether one rater's answered items are a random sample: a structural fact, read off the order."""

    rater: str = Field(min_length=1)
    rater_kind: BlindRaterKind
    standing: BlindSampleStanding = Field(
        description="`random`: the first `answered` items of the order. `selected`: any other set. `none`: nothing."
    )
    answered: int = Field(ge=0, description="The items answered, skips included.")
    skipped: int = Field(
        default=0,
        ge=0,
        description=(
            "How many of them were skipped. A skip follows content, so a sample with skips is random over the cases "
            "this rater answered, and every reading of it says how many were skipped."
        ),
    )
    items: int = Field(ge=0, description="The items in the order.")


class BlindVerdictTable(EvalDocumentModel):
    """A pass/fail criterion's two-by-two: the rater against the judge, and how many rater-failed answers the judge
    failed too."""

    both_pass: int = Field(ge=0)
    rater_pass_judge_fail: int = Field(ge=0)
    rater_fail_judge_pass: int = Field(ge=0)
    both_fail: int = Field(ge=0)
    detection: float | None = Field(
        description="The share of rater-failed answers the judge failed too; None where the rater failed none."
    )
    detection_interval: tuple[float, float] | None = Field(
        default=None, description="Its Wilson interval; None where the rater failed none."
    )


class BlindAgreement(EvalDocumentModel):
    """How one judge's scores on one criterion agree with one anchor's answers, and the tier that bears."""

    dim: DimName = Field(description="The criterion.")
    scale: RubricScale
    judge_model: str | None = Field(description="The model that served the judge's scores; None where unnamed.")
    anchor: BlindRaterKind = Field(description="Whose answers the judge is read against: people's, or a reference's.")
    anchor_names: list[str] = Field(
        min_length=1, description="The people, sorted; or the one reference reviewer's model id."
    )
    decides: bool = Field(
        description=(
            "Whether this reading decides the criterion's tier. A person reading does once it is itself decided (met "
            "or not met at the tier's floor), or where no reference reading exists; until then a reference reading "
            "stands and the person reading is shown beside it."
        )
    )
    shares_family: bool = Field(
        default=False,
        description=(
            "A reference reading only: the reviewer's model id and the judge's name one provider, so the two may "
            "share a lean their agreement cannot show."
        ),
    )
    pairs: int = Field(ge=1, description="The (answer, judge score) pairs read.")
    results: int = Field(ge=0, description="The distinct results behind the kappa: what the tier's floor counts.")
    cases: int = Field(
        ge=0,
        description=(
            "The distinct cases those pairs' results answer. A pair item gives two results from one case, whose "
            "judge errors are unlikely to be independent, so the cases are stated beside the results."
        ),
    )
    blind_pairs: int = Field(ge=0, description="How many of the pairs were given on this blind read.")
    random_pairs: int = Field(ge=0, description="How many of those came from a rater whose sample is random.")
    exact_agreement: float = Field(ge=0.0, le=1.0)
    kappa: float | None = Field(description="Kappa on pass/fail, quadratic-weighted on 1-5; None where undefined.")
    lower: float | None = Field(default=None, description="Its one-sided 95% lower bound.")
    upper: float | None = Field(default=None, description="Its one-sided 97.5% upper bound.")
    calibration: str = Field(description="How the calibration criterion reads: met, not_met, undecided, insufficient.")
    results_needed: int | None = Field(
        default=None, description="About how many more results would decide the criterion; None once decided."
    )
    tier_before: str = Field(description="The tier the judge's readings bore before these answers.")
    tier: str = Field(description="The tier they bear with them, on this anchor.")
    table: BlindVerdictTable | None = Field(default=None, description="The two-by-two, on a pass/fail criterion.")


class BlindAnchorCheck(EvalDocumentModel):
    """How a reference reviewer agrees with people on the outputs both answered: the anchor, itself measured."""

    dim: DimName
    reviewer: str = Field(min_length=1, description="The reference reviewer's model id.")
    outputs: int = Field(ge=1, description="The outputs both a person and the reviewer answered on this criterion.")
    exact_agreement: float = Field(ge=0.0, le=1.0)
    kappa: float | None = Field(description="Kappa on pass/fail, quadratic-weighted on 1-5; None where undefined.")


class BlindRefusedAnchor(EvalDocumentModel):
    """A reference reviewer whose answers are not read as calibration, and why."""

    reviewer: str = Field(min_length=1)
    dim: DimName
    reason: str = Field(min_length=1)


class BlindPreference(EvalDocumentModel):
    """Which of a pair one kind of rater preferred, per case, and whether that separates the arms."""

    anchor: BlindRaterKind
    anchor_names: list[str] = Field(min_length=1)
    random: bool = Field(description="Whether every rater behind it answered a random sample.")
    control: str = Field(min_length=1, description="The control arm (BlindArm.key).")
    candidate: str = Field(min_length=1, description="The candidate arm.")
    planned: int = Field(ge=0, description="The planned length: the first this-many items are what is read.")
    answered: int = Field(ge=0, description="How many of those items the rater furthest along has answered.")
    skipped: int = Field(default=0, ge=0, description="How many of the planned items were skipped by a rater read.")
    items: int = Field(ge=0, description="The pair items with a preference; 0 while withheld.")
    prefer_candidate: int = Field(ge=0, description="Items where the candidate's share is above a half.")
    prefer_control: int = Field(ge=0, description="Items where it is below a half.")
    ties: int = Field(ge=0, description="Items at exactly a half: counted, never split.")
    test: BlindPreferenceTest
    p: float | None = Field(description="The two-sided p; None where too few items carry a preference to test.")
    verdict: BlindPreferenceVerdict
    slot_a_share: float | None = Field(
        description="The share of non-tie preferences given to slot A, which a rater with no lean gives a half."
    )
    slot_a: int = Field(default=0, ge=0, description="The non-tie preferences given to slot A.")
    slot_choices: int = Field(ge=0, description="The non-tie preferences that share is over.")

    @model_validator(mode="after")
    def _a_withheld_preference_gives_nothing_away(self) -> BlindPreference:
        """Refuse a withheld preference that carries a count, a share or a p.

        :raises ValueError: Naming the rule.
        """
        if self.verdict == "withheld" and (
            self.items
            or self.prefer_candidate
            or self.prefer_control
            or self.ties
            or self.p is not None
            or self.slot_a_share is not None
            or self.slot_choices
        ):
            raise ValueError("a withheld preference states how far the read is, and no count, share or p")
        return self


class BlindReadReport(EvalDocumentModel):
    """What a blind read found, as its close computed it."""

    standings: list[BlindStanding] = Field(default_factory=list)
    agreement: list[BlindAgreement] = Field(default_factory=list)
    anchor_checks: list[BlindAnchorCheck] = Field(default_factory=list)
    refused_anchors: list[BlindRefusedAnchor] = Field(default_factory=list)
    preferences: list[BlindPreference] = Field(default_factory=list)
    other_ratings: int = Field(
        default=0, ge=0, description="Ratings read beside the read's own answers, which count as selected."
    )
    unread: list[str] = Field(
        default_factory=list, description="Answers offered that could not be read against the read, each with why."
    )


class BlindRead(EvalDocumentModel):
    """A drawn, blinded order of judged outputs, the answers given to it, and its close."""

    id: str = Field(min_length=1)
    scope_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    created_at: str = Field(default_factory=utc_now_iso)
    created_by: str = Field(min_length=1)
    campaign_id: str | None = Field(default=None, description="The campaign it was drawn over. Provenance.")
    arms: list[BlindArm] = Field(min_length=1, max_length=2, description="One (singles) or a control and a candidate.")
    criteria: list[RubricDim] = Field(min_length=1, description="The criteria as worded when the read was drawn.")
    ask: list[BlindAsk] = Field(min_length=1, description="What each item asks. Preference only with two arms.")
    draw: BlindDraw
    items: list[BlindItem] = Field(default_factory=list, description="The items, in the drawn order.")
    answers: list[BlindAnswer] = Field(default_factory=list)
    closed_at: str | None = None
    closed_by: str | None = None
    report: BlindReadReport | None = None


class SheetLevel(EvalDocumentModel):
    """One answer a criterion admits, and what it means where the criterion's scoring guide says."""

    value: str = Field(min_length=1)
    meaning: str = ""


class SheetCriterion(EvalDocumentModel):
    """A criterion as the rater reads it."""

    dim: DimName
    question: str = Field(min_length=1)
    scale: RubricScale
    levels: list[SheetLevel]


class SheetOutput(EvalDocumentModel):
    """An output as the rater reads it: its slot and its text, and nothing else."""

    slot: BlindSlot
    text: str


class SheetAsk(EvalDocumentModel):
    """What one item asks."""

    verdicts: list[DimName] = Field(default_factory=list)
    preference: bool = False


class SheetItem(EvalDocumentModel):
    """An item as the rater reads it."""

    item: str = Field(min_length=1)
    case_material: str
    outputs: list[SheetOutput]
    ask: SheetAsk


class BlindSheet(EvalDocumentModel):
    """The blinded view of a read: all a renderer may show.

    A renderer keeps four rules. Show the items in order and offer ``next`` first (skipping ahead is allowed, and
    makes the sample ``selected``). Show nothing but what the sheet carries. Label the slots ``A`` and ``B`` as given,
    never re-sorted. A skip is an answer.
    """

    read_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    criteria: list[SheetCriterion]
    items: list[SheetItem]
    answered: list[str] = Field(default_factory=list, description="The items the rater named has answered.")
    next: str | None = Field(default=None, description="The first unanswered item in order; None when all are.")


__all__ = [
    "REFERENCE_RATER_PREFIX",
    "BlindAgreement",
    "BlindAnchorCheck",
    "BlindAnswer",
    "BlindArm",
    "BlindAsk",
    "BlindDraw",
    "BlindItem",
    "BlindOutput",
    "BlindPreference",
    "BlindPreferenceAnswer",
    "BlindPreferenceTest",
    "BlindPreferenceVerdict",
    "BlindRaterKind",
    "BlindRead",
    "BlindReadReport",
    "BlindRefusedAnchor",
    "BlindSampleStanding",
    "BlindSheet",
    "BlindSlot",
    "BlindStanding",
    "BlindVerdict",
    "BlindVerdictTable",
    "SheetAsk",
    "SheetCriterion",
    "SheetItem",
    "SheetLevel",
    "SheetOutput",
]
