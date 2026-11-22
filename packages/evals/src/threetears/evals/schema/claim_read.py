"""A claim read: one claim criterion's graded claims drawn by stratum, each shown as the instrument read it, and what
people answered of each. The human side of a claim instrument's calibration.

**One item is one claim, as the instrument saw it** (:class:`ClaimView`): the claim, its excerpt with offsets, the
segment it was said in, and every piece of evidence the instrument numbered for the cell and every web window it read
for the claim. No verdict, no citation and no arm is on the view, so a rater cannot read the instrument's answer off
it. Its digest (:func:`claim_view_digest`) is the label's key: two items whose views are byte-identical carry one
label, whichever read drew them.

**The draw is stratified by the instrument's own answer** (:class:`ClaimDraw`): its truth verdict, or whether it
grounded the claim. The instrument's errors hide in the strata where it said nothing was wrong, so a simple random
draw would spend most labels confirming what it already got right. Each stratum has its population, its planned
draws and the draws it got, and at most :data:`CLAIMS_PER_CELL` claims come from one cell: claims of one output share
its subject and its sources, so two of them are not twice the evidence one is. The figures are read through
inverse-probability weights (stratum population over stratum draws) when the read is read, never stored: what is
stored is who answered what.

**Who answered is part of every answer** (``made_by``, a :data:`~threetears.evals.schema.reading_states.Maker`): only a
person's answer calibrates the instrument.

The shapes here are plain documents. Drawing, viewing and reading are
:mod:`threetears.evals.analysis.claim_read`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal, Self

from pydantic import Field, model_validator

from threetears.evals.schema.base import CoreShapeModel, DurableDocumentModel, EvalDocumentModel
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import DimName, DurableSchemaVersion, utc_now_iso
from threetears.evals.schema.reading_states import Maker
from threetears.evals.schema.versioning import DURABLE_SCHEMA_VERSION

#: What a claim read asks of an item, each pass/fail. ``true``: the claim is not false (the rater may search).
#: ``grounded``: a document the candidate itself was given, among those shown, supports it. ``is_claim``: it is a
#: checkable claim the text makes, the extractor's precision.
ClaimQuestion = Literal["true", "grounded", "is_claim"]

#: The instrument answer a draw is stratified by: its truth verdict, or whether it grounded the claim.
ClaimStratifier = Literal["truth", "grounded"]

#: One stratum of a draw: a truth verdict, or ``grounded`` / ``ungrounded``.
ClaimStratum = Literal["contradicted", "true_ungrounded", "unverifiable", "supported", "grounded", "ungrounded"]

#: A truth verdict, as a graded claim stores it.
ClaimTruthVerdict = Literal["contradicted", "true_ungrounded", "unverifiable", "supported"]

#: The strata of each stratifier, in the order a draw lists them.
STRATA_OF: Final[Mapping[str, tuple[ClaimStratum, ...]]] = MappingProxyType(
    {
        "truth": ("contradicted", "true_ungrounded", "unverifiable", "supported"),
        "grounded": ("grounded", "ungrounded"),
    }
)

#: The most claims a draw takes from one cell.
CLAIMS_PER_CELL: Final = 2


def claim_view_digest(view: ClaimView) -> str:
    """The label key of a view: the sha256 of its canonical JSON.

    :param view: The view.
    :return: Its digest.
    """
    return canonical_digest({"claim_view": view.model_dump(mode="json")})


class ClaimReadArm(EvalDocumentModel, CoreShapeModel):
    """One arm a claim read was drawn over: what names it, and the runs that hold its graded cells."""

    key: str = Field(min_length=1, description="What names the arm where the read was drawn: its variant key.")
    run_ids: list[str] = Field(min_length=1, description="The runs whose cells are this arm's.")


class ClaimStratumDraw(EvalDocumentModel, CoreShapeModel):
    """One stratum of a draw: how many claims it holds, how many were planned of it, and how many it gave."""

    stratum: ClaimStratum
    population: int = Field(ge=0, description="The claims in the stratum, over every arm.")
    cells: int = Field(ge=0, description="The distinct cells those claims come from.")
    planned: int = Field(ge=0, description="The draws planned of it, fixed before any was drawn.")
    drawn: int = Field(
        ge=0,
        description="The draws it gave: the planned count, or fewer where the per-cell cap left too few claims.",
    )

    @model_validator(mode="after")
    def _drawn_within_planned_within_population(self) -> Self:
        """:raises ValueError: More drawn than planned, or more planned than the stratum holds."""
        if not self.drawn <= self.planned <= self.population:
            raise ValueError(
                f"stratum {self.stratum!r} draws {self.drawn} of {self.planned} planned of {self.population} claims; "
                "a stratum draws at most what was planned, and plans at most what it holds"
            )
        return self


class ClaimDraw(EvalDocumentModel, CoreShapeModel):
    """How a claim read was drawn: the seed, the stratifier, the cap per cell, each stratum, and what was left out."""

    seed: int = Field(description="The seed each stratum's order and the items' order were drawn by.")
    stratify_by: ClaimStratifier
    per_cell: int = Field(ge=1, description="The most claims drawn from one cell.")
    population_digest: str = Field(
        min_length=1,
        description="A digest of the population's claims and their strata, sorted: the same grades under the same "
        "criterion draw the same items.",
    )
    strata: list[ClaimStratumDraw] = Field(description="Every stratum of the stratifier, in its order.")
    excluded: dict[str, int] = Field(
        default_factory=dict,
        description="Claims left out of the population, counted by reason: a grade made under another definition of "
        "the criterion, stored evidence other than what was graded, a grade whose web results a recording served "
        "(replayed), a restatement (it takes its original's verdict), no groundedness reading.",
    )

    @model_validator(mode="after")
    def _every_stratum_once_in_order(self) -> Self:
        """:raises ValueError: The strata are not the stratifier's, each once, in order."""
        named = tuple(stratum.stratum for stratum in self.strata)
        if named != STRATA_OF[self.stratify_by]:
            raise ValueError(
                f"a draw by {self.stratify_by} lists the strata {list(STRATA_OF[self.stratify_by])}, each once and in "
                f"that order; this one lists {list(named)}"
            )
        return self


class ShownEvidence(EvalDocumentModel, CoreShapeModel):
    """One numbered piece of evidence a rater is shown, as the instrument was shown it."""

    ref: str = Field(min_length=1, description="Its number as the instrument read it (`K1`, `S2`, `W1`).")
    provenance: Literal["case", "run", "web"] = Field(
        description="`case`, the same for every arm; `run`, a document the candidate itself was given; `web`, a "
        "window of a page the web check read."
    )
    title: str = Field(min_length=1)
    text: str = Field(min_length=1, description="The passage, key fact or window, verbatim.")


class ClaimView(EvalDocumentModel, CoreShapeModel):
    """One claim as a rater reads it: what was said and what the instrument read, and nothing it concluded."""

    claim: str = Field(min_length=1, description="The proposition, stated on its own.")
    excerpt: str = Field(min_length=1, description="The words it stands in.")
    start: int = Field(ge=0, description="Where the excerpt starts in its segment.")
    end: int = Field(ge=1, description="Where it ends.")
    segment: str = Field(min_length=1, description="The segment it was said in, whole.")
    evidence: list[ShownEvidence] = Field(
        default_factory=list,
        description="Every piece of evidence the instrument numbered for the cell, then the windows its web check of "
        "this claim read, in the order read.",
    )

    @model_validator(mode="after")
    def _excerpt_is_its_segments_slice(self) -> Self:
        """:raises ValueError: The excerpt is not the segment's text at its offsets."""
        if self.segment[self.start : self.end] != self.excerpt:
            raise ValueError(f"the excerpt of claim {self.claim!r} is not its segment's text at {self.start}..{self.end}")
        return self


class ClaimItem(EvalDocumentModel, CoreShapeModel):
    """One drawn claim: its view, and (never shown) where it came from and what the instrument answered."""

    item: str = Field(min_length=1, description="An opaque id: a digest of the seed, the population and the claim.")
    stratum: ClaimStratum
    arm: str = Field(min_length=1, description="The arm it came from (ClaimReadArm.key). Never shown.")
    run_id: str = Field(min_length=1, description="Provenance; may dangle. Never shown.")
    result_id: str = Field(min_length=1, description="The cell it was graded in. Provenance; never shown.")
    case_id: str = Field(min_length=1, description="The test case. Never shown.")
    claim: int = Field(ge=0, description="Its place in the cell's grade. Never shown.")
    verdict: ClaimTruthVerdict = Field(description="The instrument's truth verdict. Never shown.")
    grounded: bool | None = Field(
        default=None, description="Whether the instrument grounded it; None where no ground step read it. Never shown."
    )
    view_digest: str = Field(pattern=r"^[0-9a-f]{64}$", description="claim_view_digest of the view: the label key.")
    view: ClaimView

    @model_validator(mode="after")
    def _digest_is_the_views(self) -> Self:
        """:raises ValueError: ``view_digest`` is not the view's."""
        if claim_view_digest(self.view) != self.view_digest:
            raise ValueError(f"item {self.item!r}'s view_digest is not its view's digest")
        return self


class ClaimVerdict(EvalDocumentModel, CoreShapeModel):
    """A rater's answer to one question of one item: 1 passes, 0 fails."""

    question: ClaimQuestion
    score: int = Field(strict=True, ge=0, le=1, description="1 (pass) or 0 (fail). A bool is not a score.")
    reason: str = Field(default="", description="The rater's own words, where given.")


class ClaimAnswer(EvalDocumentModel, CoreShapeModel):
    """One rater's answer to one item: a verdict per question, or that it was skipped."""

    item: str = Field(min_length=1, description="The item answered (ClaimItem.item).")
    rater: str = Field(min_length=1, description="Who answered: a person as the host names them, or a model id.")
    made_by: Maker = Field(default="person", description="A person, or a model. A model's answer is never a person's.")
    collected_by: str | None = Field(
        default=None, min_length=1, description="Who relayed a person's answer into the read, where someone did."
    )
    answered_at: str = Field(default_factory=utc_now_iso)
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    skipped: bool = Field(default=False, description="The rater passed on the item: counted, in no figure.")

    @model_validator(mode="after")
    def _a_skip_answers_nothing_and_a_question_once(self) -> Self:
        """:raises ValueError: A skip carries a verdict, or a question is answered twice."""
        if self.skipped and self.verdicts:
            raise ValueError(f"the answer to item {self.item!r} is skipped, so it carries no verdict")
        if twice := sorted(q for q, n in Counter(v.question for v in self.verdicts).items() if n > 1):
            raise ValueError(f"the answer to item {self.item!r} answers {twice} more than once")
        return self


class ClaimRead(DurableDocumentModel):
    """A stratified draw of one claim criterion's graded claims, and the answers given to it.

    **Stored, and durable** (durable v17.10): what a person answered is authored, so a read outlives a core baseline
    reset; the runs and results it was drawn over are provenance and may dangle. Every item carries the view it was
    answered on and the instrument's answer, so every figure is read from the read alone
    (:func:`threetears.evals.analysis.claim_read.read_claim_calibration`), and none is stored.
    """

    doc_type: Literal["claim_read"] = "claim_read"
    schema_version: DurableSchemaVersion = DURABLE_SCHEMA_VERSION
    id: str = Field(min_length=1)
    scope_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    created_at: str = Field(default_factory=utc_now_iso)
    created_by: str = Field(min_length=1)
    campaign_id: str | None = Field(default=None, description="The campaign it was drawn over. Provenance.")
    criterion: DimName = Field(description="The claim criterion whose grades were drawn.")
    criterion_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="The criterion's definition every drawn grade was made under."
    )
    questions: list[ClaimQuestion] = Field(min_length=1, description="What each item asks, each once.")
    arms: list[ClaimReadArm] = Field(min_length=1)
    draw: ClaimDraw
    items: list[ClaimItem] = Field(default_factory=list, description="The items, in a seeded order.")
    answers: list[ClaimAnswer] = Field(default_factory=list)

    @model_validator(mode="after")
    def _items_agree_with_the_draw(self) -> Self:
        """Refuse a read whose items do not add up to its draw, or whose answers ask what it does not.

        :raises ValueError: A question asked twice; a run named by two arms; an item of no arm, twice, in a stratum
            its instrument answer does not give, or past the cap of its cell; a stratum whose drawn count is not its
            items; an item asked groundedness with no groundedness reading; or an answer to no item or to a question
            the read does not ask.
        """
        if len(set(self.questions)) != len(self.questions):
            raise ValueError(f"a claim read asks each question once, and this one asks {self.questions}")
        runs = [run_id for arm in self.arms for run_id in arm.run_ids]
        if twice := sorted({run_id for run_id in runs if runs.count(run_id) > 1}):
            raise ValueError(f"runs {twice} are named by more than one arm of the claim read")
        keys = {arm.key for arm in self.arms}
        ids = [item.item for item in self.items]
        if twice := sorted({item for item in ids if ids.count(item) > 1}):
            raise ValueError(f"items {twice} appear more than once")
        per_stratum = Counter(item.stratum for item in self.items)
        per_cell = Counter(item.result_id for item in self.items)
        for item in self.items:
            if item.arm not in keys:
                raise ValueError(f"item {item.item!r} comes from arm {item.arm!r}, which is no arm of the read")
            stratum = item.verdict if self.draw.stratify_by == "truth" else _ground_stratum(item.grounded)
            if stratum != item.stratum:
                raise ValueError(
                    f"item {item.item!r} sits in stratum {item.stratum!r}, and its instrument answer gives {stratum!r}"
                )
            if "grounded" in self.questions and item.grounded is None:
                raise ValueError(f"item {item.item!r} is asked groundedness, and no ground step read its claim")
        if over := sorted(cell for cell, count in per_cell.items() if count > self.draw.per_cell):
            raise ValueError(f"cells {over} give more than {self.draw.per_cell} items")
        for stratum in self.draw.strata:
            if per_stratum.get(stratum.stratum, 0) != stratum.drawn:
                raise ValueError(
                    f"stratum {stratum.stratum!r} drew {stratum.drawn}, and the read holds "
                    f"{per_stratum.get(stratum.stratum, 0)} items of it"
                )
        known = set(ids)
        for answer in self.answers:
            if answer.item not in known:
                raise ValueError(f"an answer names item {answer.item!r}, which is no item of claim read {self.id!r}")
            if unasked := sorted({v.question for v in answer.verdicts} - set(self.questions)):
                raise ValueError(f"the answer to item {answer.item!r} answers {unasked}, which the read does not ask")
        return self


def _ground_stratum(grounded: bool | None) -> ClaimStratum | None:
    """The groundedness stratum an instrument answer gives; None where it gave none."""
    if grounded is None:
        return None
    return "grounded" if grounded else "ungrounded"


__all__ = [
    "CLAIMS_PER_CELL",
    "STRATA_OF",
    "ClaimAnswer",
    "ClaimDraw",
    "ClaimItem",
    "ClaimQuestion",
    "ClaimRead",
    "ClaimReadArm",
    "ClaimStratifier",
    "ClaimStratum",
    "ClaimStratumDraw",
    "ClaimTruthVerdict",
    "ClaimVerdict",
    "ClaimView",
    "ShownEvidence",
    "claim_view_digest",
]
