"""A blind read: a campaign's judged outputs in a seeded random order, a rater's answers to it, and what it found.

One idea carries the whole shape: **the population in a seeded random order.** Every case where each chosen arm has
a judged, usable answer is one *item*; the items are a seeded permutation, and a rater answers them from the top. A
prefix of a random permutation is a simple random sample, so nobody picks a sample size first, and "is this sample
random" is a structural fact (:class:`BlindSample`): the answered items are the first ``m`` of the order, or they
are not.

**There is no key to hide.** What a rater is shown is a :class:`BlindSheet`: the outputs, the case each answers and
the criteria as worded. It names no arm, run or result, and carries no judge score. Unblinding is a lookup the
analysis already performs, by the label key of each output (its fingerprint and the criterion's); what must be kept
is the draw (:class:`BlindDraw`), which lives on the :class:`BlindRead` and never on the sheet.

**Who answered is part of every answer** (``made_by``, a :data:`~threetears.evals.schema.reading_states.Maker`, the
one type a recorded call, an answer and a reading share). A ``person`` is a person. A ``model`` is a model that
answered the same drawn sheet, named by its model id: a reference reviewer. A model's answer is never a person's, on
any surface: it is read apart, and where both exist the person's decides.

The shapes here are plain documents with no behaviour. The mechanism (drawing, the sheet, the standing, closing) is
:mod:`threetears.evals.analysis.blind_read`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from threetears.evals.schema.base import CoreShapeModel, DurableDocumentModel, EvalDocumentModel
from threetears.evals.schema.reading_states import Maker, SampleKind, StateOf
from threetears.evals.schema.versioning import DURABLE_SCHEMA_VERSION
from threetears.evals.schema.models import (
    DimName,
    DurableSchemaVersion,
    JudgeEvidence,
    RubricDim,
    RubricScale,
    utc_now_iso,
)

#: The slot an output is shown in. A single is always ``A``; a pair's two outputs are ``A`` and ``B`` as drawn.
BlindSlot = Literal["A", "B"]

#: What a rater preferred of a pair: the output in a slot, or neither.
BlindPreferenceAnswer = Literal["A", "B", "tie"]

#: What a blind read asks of each item: a verdict per criterion per output, and, of a pair, which output is better.
BlindAsk = Literal["verdicts", "preference"]

#: The test a preference was read by: the exact sign test (one answer per item), the bounded test on shares (several
#: raters), or, for a read closed before its planned length, ``anytime``: the bounded test on shares in the drawn
#: order with its stakes fixed for the planned length, whose p holds at whatever item the read was stopped.
BlindPreferenceTest = Literal["exact_sign", "bounded_share", "anytime"]


class BlindArm(EvalDocumentModel, CoreShapeModel):
    """One arm a blind read was drawn over: what names it, its role, and the runs that hold its answers."""

    key: str = Field(min_length=1, description="What names the arm where the read was drawn: its variant key.")
    role: Literal["control", "candidate"] = Field(description="The control, or the candidate read against it.")
    run_ids: list[str] = Field(min_length=1, description="The runs whose results are this arm's answers.")


class BlindDraw(EvalDocumentModel, CoreShapeModel):
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
            "evidence; judged on other wording of a criterion than the read asks (a template edited since); stored "
            "evidence differs from what was judged; identical outputs (shown once, as a single, with no preference "
            "asked)."
        ),
    )


class BlindOutput(EvalDocumentModel, CoreShapeModel):
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


class BlindItem(EvalDocumentModel, CoreShapeModel):
    """One case in the drawn order: a single output, or a pair in slots A and B."""

    item: str = Field(min_length=1, description="An opaque id: a digest of the seed, the population and the position.")
    case_id: str = Field(min_length=1, description="The test case. Never on the sheet.")
    outputs: list[BlindOutput] = Field(min_length=1, max_length=2, description="One for a single; A then B for a pair.")


class BlindVerdict(EvalDocumentModel, CoreShapeModel):
    """A rater's verdict on one criterion of one output."""

    slot: BlindSlot = Field(description="The output it is of.")
    dim: DimName = Field(description="The criterion.")
    score: int = Field(strict=True, description="1 (pass) or 0 (fail) on pass/fail; 1 to 5 on the ordinal scale.")
    reason: str = Field(default="", description="The rater's own words, where given.")


class BlindAnswer(EvalDocumentModel, CoreShapeModel):
    """One rater's answer to one item: its verdicts, its preference of a pair, or that it was skipped."""

    item: str = Field(min_length=1, description="The item answered (BlindItem.item).")
    rater: str = Field(min_length=1, description="Who answered: a person as the host names them, or a model id.")
    made_by: Maker = Field(
        default="person",
        description="A person, or a model (the read's reviewer). A model's answer is never a person's.",
    )
    reviewer_served_model: str | None = Field(
        default=None,
        min_length=1,
        description="The model that served a model's answer, as the provider named it. None for a person.",
    )
    reviewer_effort: str | None = Field(
        default=None, min_length=1, description="The effort a model was asked for, where it was."
    )
    collected_by: str | None = Field(
        default=None, min_length=1, description="Who relayed a person's answer into the read, where someone did."
    )
    answered_at: str = Field(default_factory=utc_now_iso)
    verdicts: list[BlindVerdict] = Field(default_factory=list)
    prefer: BlindPreferenceAnswer | None = Field(default=None, description="Of a pair: the better output, or a tie.")
    skipped: bool = Field(default=False, description="The rater passed on the item: counted, in no figure.")

    @model_validator(mode="after")
    def _a_skip_answers_nothing_and_only_a_model_names_a_served_model(self) -> BlindAnswer:
        """Refuse a skip that carries an answer, and a person's answer that names a model.

        :raises ValueError: Naming which.
        """
        if self.skipped and (self.verdicts or self.prefer is not None):
            raise ValueError(f"the answer to item {self.item!r} is skipped, so it carries no verdict or preference")
        if self.made_by == "person" and (self.reviewer_served_model or self.reviewer_effort):
            raise ValueError("a person's answer names no served model or effort; those are a model's")
        return self


class BlindSample(EvalDocumentModel, CoreShapeModel):
    """Whether one rater's answered items are a random sample: a structural fact, read off the order."""

    rater: str = Field(min_length=1)
    made_by: Maker
    sample: SampleKind = Field(
        description="`random`: the first `answered` items of the order. `selected`: any other set. `none`: nothing."
    )
    answered: int = Field(ge=0, description="The items answered, skipped and partly answered ones included.")
    partial: int = Field(
        default=0,
        ge=0,
        description=(
            "How many of them were answered in part: a verdict or the preference the item asks is missing. What "
            "was given counts; what was left out follows content, as a skip does, and is said with the skips."
        ),
    )
    skipped: int = Field(
        default=0,
        ge=0,
        description=(
            "How many of them were skipped. A skip follows content, so a sample with skips is random over the cases "
            "this rater answered, and every reading of it says how many were skipped."
        ),
    )
    items: int = Field(ge=0, description="The items in the order.")


class BlindVerdictTable(EvalDocumentModel, CoreShapeModel):
    """Every (reader's score, judge's score) pair one check read, counted: on pass/fail the confusion counts, on 1-5 the
    full cross-table. Only the counts are kept; an agreement share, a catch rate and its interval are read off them
    when the check is read, never stored."""

    scores: list[int] = Field(
        min_length=2,
        description=(
            "The criterion's scores, low to high, which name both the rows and the columns: 0 (fail) and 1 (pass) on "
            "pass/fail, 1 to 5 on 1-5."
        ),
    )
    counts: list[list[int]] = Field(
        description=(
            "One row per score the reader gave, in `scores` order; in each, one count per score the judge gave, in "
            "the same order. On pass/fail: [[both failed, the reader failed and the judge passed], [the reader "
            "passed and the judge failed, both passed]]."
        )
    )

    @model_validator(mode="after")
    def _a_count_per_pair_of_scores(self) -> BlindVerdictTable:
        """Refuse a table that is not square over its scores, or holds a negative count.

        :raises ValueError: Naming which.
        """
        width = len(self.scores)
        if len(self.counts) != width or any(len(row) != width for row in self.counts):
            raise ValueError(f"a verdict table holds one count per pair of its {width} scores")
        if any(count < 0 for row in self.counts for count in row):
            raise ValueError("a verdict table's counts are none below 0")
        return self


class BlindAgreement(EvalDocumentModel, CoreShapeModel):
    """How one judge's scores on one criterion stood against one reader configuration's answers, as counts.

    A reader configuration is the people (together), or one model reviewer as one served model. Nothing derived is
    stored: no kappa, no bar, no tier and no estimate of results still needed. What a reading of the judge says is
    read off these counts when it is read.
    """

    dim: DimName = Field(description="The criterion.")
    scale: RubricScale
    judge_model: str | None = Field(description="The model that served the judge's scores; None where unnamed.")
    anchor: Maker = Field(description="Whose answers the judge is read against: people's, or a model's.")
    anchor_names: list[str] = Field(
        min_length=1, description="The people, sorted; or the one reference reviewer's model id."
    )
    served_model: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "A model's reading only: the model that served every answer counted here, as the provider named it. A "
            "reviewer whose answers two models served is read once per served model, never pooled. None for people."
        ),
    )
    pairs: int = Field(ge=1, description="The (answer, judge score) pairs read.")
    results: int = Field(ge=0, description="The distinct results those pairs are of.")
    cases: int = Field(
        ge=0,
        description=(
            "The distinct cases those pairs' results answer. A pair item gives two results from one case, whose "
            "judge errors are unlikely to be independent, so the cases are stated beside the results."
        ),
    )
    blind_pairs: int = Field(ge=0, description="How many of the pairs were given on this blind read.")
    random_pairs: int = Field(ge=0, description="How many of those came from a rater whose sample is random.")
    table: BlindVerdictTable = Field(description="Every pair, counted by the reader's score and the judge's.")


class BlindReviewerCheck(EvalDocumentModel, CoreShapeModel):
    """How a reference reviewer stood against people on the outputs both answered, as counts: the anchor, itself
    measured. People are the reader here, and the reviewer sits where a judge would."""

    dim: DimName
    reviewer: str = Field(min_length=1, description="The reference reviewer's model id.")
    served_model: str | None = Field(
        min_length=1,
        description=(
            "The model that served every reviewer answer counted here, as the provider named it; None for the answers "
            "whose served model was not recorded. Answers two models served are counted apart, never pooled."
        ),
    )
    outputs: int = Field(ge=1, description="The (person's answer, reviewer's answer) pairs on this criterion.")
    table: BlindVerdictTable = Field(
        description="Every pair, counted by the person's score (rows) and the reviewer's (columns)."
    )


class BlindPreference(EvalDocumentModel, CoreShapeModel):
    """Which of a pair one kind of rater preferred, per case, and whether that separates the arms."""

    anchor: Maker
    anchor_names: list[str] = Field(min_length=1)
    random: bool = Field(description="Whether every rater behind it answered a random sample.")
    control: str = Field(min_length=1, description="The control arm (BlindArm.key).")
    candidate: str = Field(min_length=1, description="The candidate arm.")
    planned: int = Field(ge=0, description="The planned length: the first this-many items are what is read.")
    answered: int = Field(ge=0, description="How many of those items the rater furthest along has answered.")
    missing: int = Field(
        default=0,
        ge=0,
        description="The planned pair items a rater read gave no preference on: skipped, or left blank.",
    )
    items: int = Field(ge=0, description="The pair items with a preference; 0 while withheld.")
    prefer_candidate: int = Field(ge=0, description="Items where the candidate's share is above a half.")
    prefer_control: int = Field(ge=0, description="Items where it is below a half.")
    ties: int = Field(ge=0, description="Items at exactly a half: counted, never split.")
    test: BlindPreferenceTest
    p: float | None = Field(description="The two-sided p; None where too few items carry a preference to test.")
    verdict: Literal["improved", "regressed", "not_separated"] | None = Field(
        description=(
            "How it read, in a contrast's own verdict words: the candidate shown better or worse, or the two not "
            "told apart; None where it was not read, and `state` says why."
        )
    )
    state: StateOf | None = Field(
        default=None,
        description=(
            "Why the preference was not read, where it was not. `withheld`, `unfinished`: the read has not reached "
            "its planned length, so no direction, count or p is given (a preference read after every item and "
            "stopped when it separates is wrong far more often than its level says, and a running count tells the "
            "rater which arm is ahead while they are still reading). `not_available`, `too_few`: too few items "
            "carry a preference to test. None where it was read."
        ),
    )
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
        if (self.verdict is None) == (self.state is None):
            raise ValueError("a preference has a verdict, or the state that says why it has none, and never both")
        if self.state is not None and (self.state.state, self.state.reason) not in _UNREAD:
            raise ValueError("a preference that was not read is withheld as unfinished, or not available as too few")
        if self.withheld and (
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

    @property
    def withheld(self) -> bool:
        """Whether the read has not reached its planned length, so the preference gives nothing away."""
        return self.state is not None and self.state.state == "withheld"

    @property
    def too_few(self) -> bool:
        """Whether too few items carry a preference to test."""
        return self.state is not None and self.state.reason == "too_few"


#: The two ways a preference is not read: the planned read is unfinished, or too few items carry a preference.
_UNREAD = frozenset({("withheld", "unfinished"), ("not_available", "too_few")})


class BlindReview(EvalDocumentModel, CoreShapeModel):
    """One model's reading of a blind read's sheet: the calls made for it, what they cost, and what went unanswered.

    The spend is measurement spend: what it cost to check the judge, never part of what a candidate costs.
    """

    reviewer: str = Field(min_length=1, description="The model's id, which names its answers (BlindAnswer.rater).")
    served_models: list[str] = Field(
        default_factory=list,
        description="Every model that served one of its answers, as the provider named it, sorted.",
    )
    asked: int = Field(ge=0, description="The items a call was made for in this reading.")
    reused: int = Field(default=0, ge=0, description="The items read from answers recorded earlier, with no call made.")
    calls: int = Field(default=0, ge=0, description="The calls made.")
    cost_usd: float | None = Field(default=None, description="What the priced calls cost; None when none was priced.")
    unpriced_calls: int = Field(default=0, ge=0, description="The calls that carry no dollars.")
    unanswered: list[str] = Field(
        default_factory=list,
        description="Why each item the model was asked and gave no usable answer on has none, in order.",
    )
    stopped: str | None = Field(
        default=None, description="Why the reading stopped before every item was asked; None where nothing stopped it."
    )


class BlindReadReport(EvalDocumentModel, CoreShapeModel):
    """What a blind read found, as its close counted it: how much was read and how it was sampled, how the judge's
    scores stood against whoever read (counts only), what was preferred, and what a model's reading cost."""

    cases: int = Field(default=0, ge=0, description="The cases in the read's drawn order.")
    reviews: list[BlindReview] = Field(
        default_factory=list,
        description="Each model's reading of the sheet: its calls, their cost, what went unanswered.",
    )
    samples: list[BlindSample] = Field(default_factory=list)
    agreement: list[BlindAgreement] = Field(default_factory=list)
    reviewer_checks: list[BlindReviewerCheck] = Field(default_factory=list)
    not_checked: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Each model whose answers are not read as a check of the judge, by its id, with why: it is the judge's "
            "own model, or which model served either side was not recorded."
        ),
    )
    preferences: list[BlindPreference] = Field(default_factory=list)
    other_ratings: int = Field(
        default=0, ge=0, description="Ratings read beside the read's own answers, which count as selected."
    )
    unread: list[str] = Field(
        default_factory=list, description="Answers offered that could not be read against the read, each with why."
    )


class BlindRead(DurableDocumentModel):
    """A drawn, blinded order of judged outputs, the answers given to it, and its close.

    **Stored, and durable** (durable v17.4): what a person answered is authored, so a read outlives a core baseline
    reset as the ratings its close writes do; the runs and results it was drawn over are provenance and may dangle.
    The quick path recomputes its read from the eval's files and never stores one; a host's actions store one per
    draw (:mod:`threetears.evals.ops.blind_reads`).
    """

    doc_type: Literal["blind_read"] = "blind_read"
    schema_version: DurableSchemaVersion = DURABLE_SCHEMA_VERSION
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
    reviews: list[BlindReview] = Field(
        default_factory=list,
        description=(
            "Each model's reading of the sheet as it was made: what was asked, what it cost and what was not "
            "answered. An answer is a model's because a review here made it, never because of how its rater is named."
        ),
    )
    reviewer: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
        description=(
            "The model declared, when the read was drawn, to answer it as a reference reviewer, by its id: fixed "
            "before any answer exists, and one per read. None: no model reads it."
        ),
    )
    review_job_id: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
        description=(
            "The job the declared reviewer reads the sheet in, by its record's id: a model's answers are taken into "
            "the read only from inside that job. None: no review was started."
        ),
    )
    closed_at: str | None = None
    closed_by: str | None = None
    closed_early: bool = Field(
        default=False,
        exclude_if=lambda value: not value,
        description=(
            "Closed before any rater had answered every planned item: its preference, where one is asked, is read by "
            "the anytime test over the items answered in the drawn order, which holds its level at any stop."
        ),
    )
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
    """An output as the rater reads it: its slot and its text, and the fingerprints a label of it is bound to.

    The fingerprints are digests of the text shown and of each criterion's wording. They name no arm, run or result,
    and they are what lets a saved label pair with the same output under another draw. A host's door leaves them out:
    its answers name an item and a slot, and a judge score carries the same fingerprint.
    """

    slot: BlindSlot
    text: str
    output_fingerprint: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)
    criterion_fingerprints: dict[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)


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
    "BlindAgreement",
    "BlindReviewerCheck",
    "BlindAnswer",
    "BlindArm",
    "BlindAsk",
    "BlindDraw",
    "BlindItem",
    "BlindOutput",
    "BlindPreference",
    "BlindPreferenceAnswer",
    "BlindPreferenceTest",
    "BlindRead",
    "BlindReadReport",
    "BlindReview",
    "BlindSheet",
    "BlindSlot",
    "BlindSample",
    "BlindVerdict",
    "BlindVerdictTable",
    "SheetAsk",
    "SheetCriterion",
    "SheetItem",
    "SheetLevel",
    "SheetOutput",
]
