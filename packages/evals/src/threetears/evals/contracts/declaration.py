"""What a campaign set out to learn — the declaration, as distinct from what it measured.

A campaign has always been curated membership with no structure, so an analysis wanting to
compare arms had to infer one. Inferring *factorial* over a star design is the expensive
mistake: it pools cells that moved a different lever into the comparison arm, manufactures a
difference inside that arm, and then correctly reports the arm as confounded. The campaign that
produced this model had its one decision deferred on exactly that invented confound, over data
that isolated the lever cleanly.

**Declared and realized are two designs and they must not collapse into one.** The declaration
here is operator input — *what we set out to learn*. The realized design is derived from
observations and lives on the bundle — *what we actually measured*. **The delta between them is
the coverage story**, and merging them destroys it: a declared value with no observations is
`unswept` **and named**, where an inferred design can only report what happened to run. Today
`unswept` means "we did not see it" rather than "we meant to and did not".

**Nothing here names a host concept.** An axis id is whatever the host ACCEPTS as an axis — a
registered sweepable LEVER, or a member of an open family the host's own membership test
recognises, which is what lets an ad-hoc knob be swept without a registration per knob.
Registration alone is not enough either, since the measuring rig and a bare label are registered too;
a behavior is a string the engine never interprets; `template_id` stays on the campaign as an
optional host reference rather than moving inside the declaration — which is what lets a consumer
with no template declare a design at all.

**The word is `intended_repetitions`, not `intended_k`.** `k` is not shared vocabulary: one
consumer means iterations per cell, another means the candidate window and calls the repeat
count something else. An engine-owned field named for one host's meaning is the failure
this whole layer exists to avoid — the slot is shared, the word is not.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, Literal, get_args, get_origin

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.covariates import (
    COVARIATE_KEYS,
)
from threetears.evals.contracts.host.bars import Bar, BarRegistrationError, contradicts_descriptor, no_better_end
from threetears.evals.contracts.host.values import Scale, SweepableValue
from threetears.evals.contracts.metrics import (
    DERIVED_PER_RESULT_MEASURES,
    FRONTIER_RANKING_MEASURE,
    METRIC_DESCRIPTORS,
    MeritAxis,
    MetricDescriptor,
    describe_goal_check_rate,
    describe_rubric_dim,
    goal_check_of,
    is_code_graded,
    list_metrics,
)
from threetears.evals.contracts.models import (
    RESERVED_DIM_IDS,
    ApparatusProvenance,
    EvalResult,
    EvalTemplate,
    RubricScale,
    utc_now_iso,
)

if TYPE_CHECKING:
    from threetears.evals.contracts.host.measures import MeasureRegistry
    from threetears.evals.contracts.host.profile import HostProfile

#: Characters in a variant key — a sha256 hex digest, which is what
#: :func:`~threetears.evals.contracts.identity.compute_variant_key` produces.
_VARIANT_KEY_LENGTH = 64


class SweptAxis(EvalDocumentModel):
    """One axis this campaign set out to vary, and the levels it meant to compare.

    ``axis_id`` must be something the campaign's host will accept as an axis — checked at campaign
    creation rather than at analysis time, which is the difference between a gate and a page. A
    campaign declaring an axis its host cannot vary is not an experiment that will read badly
    later; it is one that cannot run, and finding that out before the money is spent is the whole
    point.

    **Acceptance is not the same as "has a declaration."** A fixed ``lever`` declaration qualifies;
    so does a member of an open family that the host's own ``owns_member`` test recognises, which is
    what makes an ad-hoc knob a first-class axis without a registration ticket per knob. A family's
    own container name does not, because it identifies no knob and no resolution ever emits it.

    Registration on its own is not the bar either. An ``apparatus`` input is the measuring rig and
    a ``label`` identifies rather than determines; both are registered, and neither can become an
    arm, because the variant key is built from levers alone. Declaring one is refused here.
    """

    axis_id: str = Field(
        min_length=1,
        description=(
            "A name this host will accept as a swept axis. That is a fixed LEVER declaration, or a member "
            "of an open family the host's own membership test recognises — a family's container name is "
            "NOT one, since it identifies no knob. Refused at authoring time — on create and on every "
            "update that supplies a design — when the host recognises neither, and equally when the name "
            "is registered as apparatus or as a label, because the variant key is built from levers alone "
            "and neither can become an arm. The refusal carries the host's own vocabulary as its remedy."
        ),
    )
    values: list[SweepableValue] = Field(
        min_length=1,
        description=(
            "The levels this campaign meant to compare, each carrying identity, rendering and scale. "
            "At least one: an axis declared with no values states an intention nobody can check."
        ),
    )
    rationale: str = Field(
        default="", description="Why this axis is in the campaign. Optional, and the first thing a reader wants."
    )

    @field_validator("values", mode="before")
    @classmethod
    def _content_address_authored_levels(cls, values: Any) -> Any:
        """Let an author declare a level by its CONTENT and address it here.

        A swept value's identity is a sha256 over its content (R9), and a declaration is operator
        input — so requiring the author to supply the digest would mean asking a person to hash a
        canonical JSON encoding by hand. Worse than inconvenient: a digest typed wrong is not
        rejected by anything, it silently produces a level that joins to no observation, and the
        campaign reports the axis as unswept while the runs are sitting right there.

        So two authored shapes, and they are not a fallback pair — they are "I hold the identity"
        and "I hold the content". ``content_hash`` present means a value that already exists: a
        declaration read back and amended, or one a host computed over content the engine must
        not see. ``content`` present means the level itself, addressed here. Neither guesses at
        the other, and a value carrying neither is refused rather than defaulted.

        ``keep_raw`` is on, unlike the snapshot paths, so a reader can check the declaration
        against the runs rather than against three hashes. **The bound is what the operator typed,
        not the matrix** — stated exactly because the obvious phrasing ("small by construction")
        would be a claim nothing enforces: ``content`` is arbitrary authored JSON, and a lever an
        operator is likely to sweep is a prompt override, content-addressed by the resolved prompt
        BODIES — so declaring that axis by content persists full prompt text on the campaign
        document. That is not the forbidden axis — retention that scales with
        what was *observed* — it is operator input, and an operator who types a megabyte gets a
        megabyte. A path that needs a hard bound drops ``raw`` and keeps the hash, which the model
        already supports and every snapshot path already does.

        Args:
            values: The authored levels, before validation.

        Returns:
            The levels, with any content-authored one replaced by its addressed value.

        Raises:
            ValueError: A level carries neither ``content_hash`` nor ``content``, so there is
                nothing to identify it by.
        """
        if not isinstance(values, list):
            return values
        addressed = []
        for level in values:
            if not isinstance(level, dict):
                addressed.append(level)
                continue
            if "content_hash" in level:
                if "content" in level:
                    # Refused rather than resolved by precedence: the two disagreeing is exactly
                    # the silent-wrong this whole form exists to prevent, and nothing here can
                    # tell which one the author meant.
                    raise ValueError(
                        "a swept value carries both `content_hash` and `content` — send one. If they "
                        "disagree, nothing can tell which level you meant"
                    )
                addressed.append(level)
                continue
            if "content" not in level:
                raise ValueError(
                    "a swept value needs either `content_hash` (an identity you already hold) or `content` "
                    "(the level itself, which is content-addressed here) — this one carries neither"
                )
            authored = dict(level)
            content = authored.pop("content")
            scale = authored.get("scale")
            addressed.append(
                SweepableValue.of(
                    content,
                    display=authored.get("display"),
                    scale=TypeAdapter(Scale).validate_python(scale) if scale is not None else None,
                    keep_raw=True,
                )
            )
        return addressed


class BarOverride(EvalDocumentModel):
    """A standard this campaign holds itself to, tighter than the registered one.

    Tighter only. A campaign that could lower the bar it is judged against is not being judged;
    :meth:`~threetears.evals.contracts.host.bars.BarRegistry.check_override` refuses a looser proposal and
    quotes the registered value, so the refusal names what it is protecting rather than merely
    saying no.
    """

    measure_id: str = Field(
        min_length=1,
        description=(
            "What this bar is read on — a code-graded measure a single result carries with a better end "
            "(the engine's or the host's), a judged dimension (a rubric dimension the campaign's template "
            "declares, or a reserved dual-score axis), or a goal-state check the template declares, spelled "
            "exactly as the template writes it. Anything else — a run-level statistic, a judge-mediated "
            "summary, a composite, a categorical or directionless measure, an undescribed name — is refused "
            "at authoring time, because no verdict could ever be given on it. The one composite admitted is "
            "`pass_hat_k`, the measure the frontier ranks on, with a threshold in [0, 1]: no cell carries it, and "
            "the analysis bundle passes the bar to the frontier, which reads it on each contestant's pass^k "
            "interval."
        ),
    )
    threshold: float = Field(description="The value the measure must reach for this campaign.")
    direction: Literal["higher_is_better", "lower_is_better"] = Field(
        description=(
            "Which way clearing runs. Declared rather than inferred so a reader of the campaign can "
            "check the threshold without resolving the measure registry — but NEVER trusted against "
            "the incumbent: the registry's direction governs an override check, because a proposal "
            "that restates the rule its own threshold is judged by can flip one boolean and register "
            "a looser bar as tighter. It is not trusted against the measure DESCRIPTOR either: "
            "`refuse_an_undeclarable_design` refuses a bar whose direction contradicts the one the "
            "host declared for that measure, which is the check that fires when no incumbent exists."
        )
    )


class Question(EvalDocumentModel):
    """Something this campaign is trying to find out.

    Questions live on the campaign; **resolutions live on the analysis**, because a resolution
    is that analysis's claim over that evidence and a later analysis over more runs may resolve
    differently. Collapsing the two would make a question's answer look like a property of the
    question.

    **The text is immutable once resolved.** An edit mints a new question carrying
    ``supersedes``, and the lineage renders. Editing in place would silently restate what an
    existing analysis had already answered — the resolution would still point here, now
    attached to a question nobody asked when it was written.

    A question is **retired, never deleted**, because analyses reference it by id. A retired
    question stops requiring a resolution from future analyses and keeps its history.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid7()),
        min_length=1,
        description=(
            "Stable id. Analyses reference it, which is why a question is retired rather than deleted. "
            "Minted here when the author does not supply one — an operator writing a question should "
            "not have to produce a uuid, and the supersession path has to mint one regardless."
        ),
    )
    text: str = Field(
        min_length=1, description="The question, in the operator's words. Immutable once any analysis has resolved it."
    )
    merit_axes: list[MeritAxis] = Field(
        default_factory=list,
        description=(
            "Which axes an answer would move, when the question is about one. The bundle lists the bars on "
            "these axes per question (`verdict_order.questions`). Empty means unscoped."
        ),
    )

    @field_validator("merit_axes")
    @classmethod
    def _each_axis_once(cls, axes: list[MeritAxis]) -> list[MeritAxis]:
        """Refuse an axis named twice: the list is a set of axes an answer moves, and a repeat says nothing."""
        if duplicates := sorted({axis for axis in axes if axes.count(axis) > 1}):
            raise ValueError(f"merit_axes names {', '.join(duplicates)} more than once; name each axis once")
        return axes

    asked_at: str = Field(default_factory=utc_now_iso, description="When it was asked (ISO-8601).")
    retired_at: str | None = Field(
        default=None,
        description="When it stopped requiring resolutions. None = live. Retiring keeps history; deleting would orphan every analysis that answered it.",
    )
    supersedes: str | None = Field(
        default=None,
        description="The question id this one replaces, when an edit minted it. None = originally asked.",
    )


#: The merit axis a judged dimension serves. A judge scores how good an output is, so a capability dimension is
#: always a quality reading; a boundary dimension is a guardrail and serves no axis.
JUDGED_MERIT_AXIS: MeritAxis = "quality"


def axis_in_question_scope(axis: MeritAxis | None, question_axes: Sequence[MeritAxis]) -> bool:
    """Whether a reading on ``axis`` is one a question naming ``question_axes`` asks about.

    The one rule for which readings a declared question covers: a reading on one of the axes it names, or on
    any axis when it names none (an unscoped question asks about every axis — every axis, not every reading).
    A reading on no axis — a guardrail, a diagnostic, the rig's own figures — is asked about by no question.
    The bundle's comparison families read it, and so does the exploratory label, so the two cannot disagree
    about which readings a question covers.

    Args:
        axis: The reading's merit axis, or None when it serves none.
        question_axes: The axes the question names; empty for an unscoped question.

    Returns:
        True when the question covers the reading.
    """
    return axis is not None and (not question_axes or axis in question_axes)


def exploratory_reading(axis: MeritAxis | None, questions: Sequence[Question]) -> bool:
    """Whether a reading on ``axis`` lies outside every one of ``questions`` — exploratory, not confirmatory.

    Only meaningful where the campaign declares questions: with none, every reading is exploratory, and that
    is said once for the campaign rather than on every row (a label that fires on every row is one readers
    learn to skip). A guardrail is never exploratory: it is held because it was declared one, not because
    something happened to move.

    Args:
        axis: The reading's merit axis, or None when it serves none.
        questions: The campaign's live questions.

    Returns:
        True when no question covers the reading.
    """
    return not any(axis_in_question_scope(axis, question.merit_axes) for question in questions)


class ControlDeclaration(EvalDocumentModel):
    """What held still, stated — because an absent control is a fact, not a null.

    Two independent axes, and the engine must not collapse them. ``stimulus`` says whether a
    battery held what the subject was asked; ``apparatus`` says whether the observations were
    commissioned deliberately or found after the fact. A consumer reading production traffic is
    ``uncontrolled`` and ``witnessed`` and is not thereby doing worse science — it is doing
    different science, and a memo that cannot tell the two apart will describe one as the other.
    """

    stimulus: Literal["controlled", "uncontrolled"] = Field(
        description="controlled = a battery held the stimulus fixed. uncontrolled = nothing did, and `stimulus_reason` says what that means here."
    )
    stimulus_reason: str = Field(
        default="",
        description="Required when stimulus is uncontrolled: what varied instead. An uncontrolled stimulus with no reason is a gap wearing a label.",
    )
    apparatus: ApparatusProvenance = Field(
        description=(
            "commissioned = these observations were gathered deliberately under a declared rig. "
            "witnessed = they were found. The difference between an experiment and a log, and cells "
            "never pool across it. The same words every run records (`EvalRun.apparatus_provenance`), so the "
            "bundle compares this declaration with what the runs say value for value (`held_fixed_reading`)."
        )
    )

    @model_validator(mode="after")
    def _an_uncontrolled_stimulus_states_what_varied(self) -> ControlDeclaration:
        """An uncontrolled stimulus must say what that means, or it is a label with no content.

        Returns:
            The validated declaration.

        Raises:
            ValueError: The stimulus is uncontrolled and no reason was given, so a reader learns
                that something varied but not what — which is worse than not asking, because it
                reads as disclosed.
        """
        if self.stimulus == "uncontrolled" and not self.stimulus_reason.strip():
            raise ValueError(
                "an uncontrolled stimulus must state what varied instead — a bare 'uncontrolled' discloses nothing"
            )
        return self


class CampaignDesign(EvalDocumentModel):
    """The declaration: what this campaign set out to learn, before it learned anything.

    Operator input throughout. Every field here is a statement of intent, and none of it is
    derived from observations — that is the realized design's job, and keeping them apart is
    what makes coverage a comparison against intent rather than a description of whatever ran.

    ``control`` is a **variant key**, not a run id. Curating the control *run* out of a campaign
    no longer destroys the design if another observation carries the same variant — a state that
    previously needed its own field because it was a live failure.

    ``held_fixed`` says what held still while the campaign ran. It was named ``controls`` until it
    was renamed within schema v8, because one letter apart from ``control`` it named a different
    thing; a stored campaign or analysis carrying ``controls`` reads it as ``held_fixed``.
    """

    __retired_fields__: ClassVar[dict[str, str | None]] = {"controls": "held_fixed"}

    axes: list[SweptAxis] = Field(
        min_length=1,
        description=(
            "What this campaign set out to vary. At least one: a declaration with no axes declares "
            "nothing, and the typed-answer validator needs these as its denominator — 'the answer "
            "addressed every declared axis' is uncheckable without them."
        ),
    )
    questions: list[Question] = Field(
        default_factory=list,
        description=(
            "What it is trying to find out. Empty is a real state — a passive campaign measures without "
            "asking. UNBOUNDED, and the bound is the operator's: retire-never-delete is load-bearing (an "
            "analysis resolves a question by id, so removing one orphans its resolution), a text edit MINTS "
            "a superseding question rather than rewriting, and omitting a stored question from an update is "
            "refused — so nothing can ever shrink this list. An operator who iterates one question's wording "
            "ten times carries ten rows forever, and every later update echoes all ten. `live_questions()` "
            "keeps the LIFECYCLE complete; nothing keeps the SIZE bounded, and that is the cost of resolving "
            "by id rather than an oversight."
        ),
    )
    bars: list[BarOverride] = Field(
        default_factory=list,
        description="Standards this campaign holds itself to, tighter than the registered ones. Empty = the registry's bars apply unchanged.",
    )
    control: str | None = Field(
        default=None,
        description=(
            "The variant key every other cell is read against — the reference point, not a run id. "
            "A 64-hex digest over the resolved contestant stack, which is why it is ADDRESSED from an "
            "observation rather than typed: `campaign_set_control` takes a run (and a candidate model "
            "where the run carries more than one) and resolves the key here, the same shape a swept "
            "level is authored in. None = no control declared, and the analysis says so rather than "
            "electing one. NOT 'baseline', which is temporal: a control is contemporaneous, same "
            "apparatus and same campaign. Not `held_fixed` below: this is WHICH CELL is the "
            "reference; that is WHAT HELD STILL while the campaign ran."
        ),
    )
    intended_repetitions: int | None = Field(
        default=None,
        ge=1,
        description=(
            "How many repetitions this design intends per cell, counted per case: a cell's least-repeated "
            "case is what is compared, because it is the cell's weakest replication. "
            "Named for the slot rather than for one host's word: `k` means iterations per cell to one "
            "consumer and something else to the next, and an engine-owned field cannot carry one host's "
            "meaning. The bundle names every cell that falls short (`short_cells`). None = unstated, "
            "which makes a shortfall undetectable rather than zero."
        ),
    )
    held_fixed: ControlDeclaration = Field(
        description=(
            "What held still while the campaign ran — the stimulus, stated `controlled` or "
            "`uncontrolled`, and the apparatus, stated `commissioned` or `witnessed`; the two take "
            "different words and neither takes all four. Declared because an absent control is a fact "
            "rather than a null. Not `control` above, which names WHICH CELL is the reference point; "
            "this names WHAT WAS HELD STILL. A campaign can declare either without the other."
        )
    )
    merit_priority: list[MeritAxis] = Field(
        default_factory=list,
        description=(
            "Tie-break order when no bar picks a winner, strongest first. The bundle ranks the adjudicated bars "
            "by it (`verdict_order`). Empty = no stated preference, and the analysis must not invent one."
        ),
    )
    declared_at: str = Field(default_factory=utc_now_iso, description="When the declaration was made (ISO-8601).")
    declared_by: str = Field(default="", description="Who declared it.")

    @model_validator(mode="after")
    def _questions_do_not_collide_or_dangle(self) -> CampaignDesign:
        """Question ids are unique and every ``supersedes`` names a question in this design.

        Returns:
            The validated design.

        Raises:
            ValueError: Two questions share an id, so an analysis resolving one cannot say which
                it answered; or a question supersedes an id nothing here declares, which renders
                a lineage with a hole in it.
        """
        ids = [q.id for q in self.questions]
        if duplicates := sorted({qid for qid in ids if ids.count(qid) > 1}):
            raise ValueError(
                f"duplicate question ids: {', '.join(duplicates)} — an analysis could not say which it resolved"
            )
        known = set(ids)
        if dangling := sorted({q.supersedes for q in self.questions if q.supersedes and q.supersedes not in known}):
            raise ValueError(f"questions supersede ids this design does not declare: {', '.join(dangling)}")
        return self

    @field_validator("control")
    @classmethod
    def _a_control_is_a_variant_key(cls, value: str | None) -> str | None:
        """Refuse a control that is not the shape of a variant key.

        The failure this prevents is silent and expensive: a key that is not a digest joins no
        observation, so the analysis reports no control resolved while the runs that carry one
        sit right there — indistinguishable from a campaign that never declared a control at
        all. It is the same shape as the mistyped swept level
        :meth:`SweptAxis._content_address_authored_levels` exists to prevent, and it is caught
        the same way: at authoring time, where the author is still holding the thing they meant.

        A run id is the mistake this most expects, and it is named, because ``control`` held one
        for the whole of its previous life.

        Args:
            value: The authored control, or None.

        Returns:
            The validated key, or None.

        Raises:
            ValueError: The value is not a 64-character lowercase hex digest.
        """
        if value is None:
            return None
        if len(value) != _VARIANT_KEY_LENGTH or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(
                f"`control` is a variant key — a {_VARIANT_KEY_LENGTH}-character lowercase hex digest over the "
                f"resolved contestant stack — and '{value}' is not one. A run id is not a control any more: use "
                "`campaign_set_control` with the run (and its candidate model, where it carries more than one) "
                "and the key is resolved for you"
            )
        return value

    @model_validator(mode="after")
    def _axes_and_bars_are_keyed_too(self) -> CampaignDesign:
        """One entry per axis, per measure and per ranked merit axis — the same rule the questions already carry.

        A declared axis is the denominator of every coverage number and of the typed-answer
        validator's "did the answer address every axis"; a duplicate double-counts it. Two bars
        on one measure is worse: both persist, both read as this campaign's standard, and
        nothing decides which governs — while the R10 gate's own dict comprehension silently
        dedupes, so the refusal an operator might expect never comes.

        Returns:
            The validated design.

        Raises:
            ValueError: An axis id, a measure id or a merit-priority axis appears twice.
        """
        axes = [axis.axis_id for axis in self.axes]
        if duplicates := sorted({axis for axis in axes if axes.count(axis) > 1}):
            raise ValueError(
                f"duplicate axis ids: {', '.join(duplicates)} — coverage and the typed-answer validator "
                f"count declared axes, so a repeat is counted twice and neither entry's levels govern"
            )
        measures = [bar.measure_id for bar in self.bars]
        if duplicates := sorted({m for m in measures if measures.count(m) > 1}):
            raise ValueError(
                f"more than one bar on: {', '.join(duplicates)} — both would read as this campaign's "
                f"standard on that measure and nothing decides which one it is held to"
            )
        priority = list(self.merit_priority)
        if duplicates := sorted({axis for axis in priority if priority.count(axis) > 1}):
            raise ValueError(
                f"merit_priority names {', '.join(duplicates)} more than once — it is a strongest-first ranking of "
                "axes, and an axis ranked twice puts its bars in two tiers; name each axis once"
            )
        return self

    def live_questions(self) -> list[Question]:
        """The questions a future analysis still owes a resolution for.

        A retired question keeps its history and its resolutions; what it stops doing is
        requiring new ones. Superseded questions drop out too — the replacement carries the
        obligation, and demanding both would have every analysis answer the same thing twice.

        Returns:
            The live questions, in declaration order.
        """
        superseded = {q.supersedes for q in self.questions if q.supersedes}
        return [q for q in self.questions if q.retired_at is None and q.id not in superseded]


def reconcile_question_edits(stored: list[Question], incoming: list[Question]) -> list[Question]:
    """Fold an authored question list onto the stored one without ever rewriting a question.

    A question's text does not change. An analysis resolves a question BY ID, so editing the
    text in place would leave that resolution pointing at a sentence nobody asked when it was
    written — the answer would silently re-attach to a different question. An edit therefore
    MINTS a replacement carrying ``supersedes``, the original stays exactly as asked, and
    :meth:`CampaignDesign.live_questions` drops the superseded one so nothing is answered twice.

    **Minting is unconditional, where the direction says "immutable once resolved".** The
    condition is not evaluable: a resolution is a fact about an analysis, and no analysis records
    one yet. The choices were to mint always or to allow in-place edits until the resolution
    record exists — and the second one writes a rule that starts silently mutating history the
    day it lands. Minting always is the conservative half of the same rule, costs a superseded
    row nobody has to read, and narrows cleanly if the resolution record ever makes the
    condition checkable.

    A stored question missing from ``incoming`` is REFUSED rather than dropped, because a
    question is retired and never deleted — an analysis holds its id. Retiring is an edit like
    any other: send it back with ``retired_at`` set.

    Args:
        stored: The questions the campaign already carries, in declaration order.
        incoming: The questions the author supplied, which must name every stored id.

    Returns:
        The reconciled list — every stored question verbatim or with its non-historical fields
        updated, the newly authored ones, and one minted replacement per edited text, appended
        after the originals they supersede.

    Raises:
        ValueError: ``incoming`` omits a stored question id, which would delete a question some
            analysis may already reference. The message names the ids and the way to retire one.
    """
    by_id = {q.id: q for q in stored}
    if dropped := sorted(set(by_id) - {q.id for q in incoming}):
        raise ValueError(
            f"questions are retired, never deleted — this update omits {', '.join(dropped)}, "
            f"and an analysis may already resolve them by id. Send each one back with `retired_at` set instead"
        )

    reconciled: list[Question] = []
    minted: list[Question] = []
    for question in incoming:
        prior = by_id.get(question.id)
        if prior is None:
            reconciled.append(question)
        elif question.text == prior.text:
            # `asked_at` and `supersedes` are the question's history, not its content: taking
            # them from the author would let an edit re-date a question or rewrite its lineage.
            # Everything else on the row — merit_axes, retired_at — is theirs to change.
            reconciled.append(question.model_copy(update={"asked_at": prior.asked_at, "supersedes": prior.supersedes}))
        else:
            # The original keeps its text, its date and its lineage; `retired_at` is the one
            # incoming field it still takes, because retiring while rephrasing is one intent and
            # dropping half of it would discard operator input without saying so. (The superseded
            # question already falls out of `live_questions`, so this changes no obligation — it
            # keeps the record saying what the author said.)
            reconciled.append(prior.model_copy(update={"retired_at": question.retired_at}))
            minted.append(Question(text=question.text, merit_axes=question.merit_axes, supersedes=prior.id))
    return reconciled + minted


# ---------------------------------------------------------------------------
# What a bar names, and where a result carries it — the one answer both the declaration gate and
# the bundle's bar adjudication read.
# ---------------------------------------------------------------------------

#: The three kinds of name a bar may carry, each read from a different place on a result.
#:
#: ``measure`` — a described, code-graded measure, read off the result's measure walk (its carried
#: fields, the measures it implies, its covariates and the host's own measures). ``judged`` — a
#: judged dimension, read off the judge's scores by dimension name: the template's rubric
#: dimensions plus the two reserved dual-score axes every run scores. ``goal_state`` — a goal-state
#: check, read off the result's goal-state outcomes by the check's verbatim text.
BarNameKind = Literal["measure", "judged", "goal_state"]

#: Why a bar name has no verdict to give. ``not_numeric`` — it names a categorical measure, which a
#: threshold has nothing to compare against. ``no_better_end`` — it names a numeric measure that
#: declares no direction (a raw count, or a diagnostic), so clearing it would mean nothing.
#: ``not_carried`` — no result can carry a value under it as any of the three kinds.
BarNameRefusal = Literal["not_numeric", "no_better_end", "not_carried"]

#: The covariate keys a result can carry — every key ``derive_covariates`` writes. A covariate is a
#: per-result observation keyed by a described name, so a bar may name one.
_COVARIATE_MEASURES: frozenset[str] = COVARIATE_KEYS


@dataclass(frozen=True)
class BarName:
    """A bar name a result can carry: which kind it is, and the descriptor that says which way it runs.

    Attributes:
        name: The name, exactly as the bar spells it.
        kind: Which of the three kinds it resolved as — which is also where it is read; see
            :data:`BarNameKind`.
        descriptor: The measure's descriptor. Its ``higher_is_better`` is never ``None`` here, since
            a name with no better end is refused rather than resolved.
    """

    name: str
    kind: BarNameKind
    descriptor: MetricDescriptor

    @property
    def higher_is_better(self) -> bool:
        """Which way the descriptor says clearing runs."""
        return bool(self.descriptor.higher_is_better)


@dataclass(frozen=True)
class UnreadableBarName:
    """A bar name no verdict can be given on, and why — the text an author or a reader is shown.

    Attributes:
        name: The name, exactly as the bar spells it.
        refusal: Which of the three reasons applies; see :data:`BarNameRefusal`.
        reason: One sentence saying why, naming what the name is where anything describes it.
    """

    name: str
    refusal: BarNameRefusal
    reason: str


def _carriers_of(annotation: object) -> Iterator[type[BaseModel]]:
    """Yield every pydantic model inside a field annotation, through ``Optional`` and collections."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
    for argument in get_args(annotation):
        yield from _carriers_of(argument)


@cache
def _carried_field_names() -> frozenset[str]:
    """Every name a result's own fields can carry a measure under — its scalars and its carriers' fields.

    The type-level mirror of the bundle's measure walk, which reads a result's top-level scalars and
    the scalar leaves of every sub-model it holds, single or listed. Derived from ``EvalResult`` by
    reflection so a carrier the result gains is counted with no edit here. Cached because the
    answer is a property of the model class and never of a run.
    """
    names: set[str] = set()
    for field_name, field in EvalResult.model_fields.items():
        carriers = list(_carriers_of(field.annotation))
        if not carriers:
            names.add(field_name)
        for carrier in carriers:
            names.update(carrier.model_fields)
    return frozenset(names)


def _host_measure_names(measures: MeasureRegistry) -> frozenset[str]:
    """The names only the host declares — each one a key its kinds may land on ``host_measures``.

    The one open door of the per-result set, and the gate is honest about what it cannot see
    through it: a host may declare a run-level aggregate beside its per-cell measures, and nothing
    in a descriptor tells the two apart, so every name the host declares is admitted. A bar on one
    its kind never emits is admitted here and adjudicated as carried by no result.

    Args:
        measures: The host's measure registry.
    """
    names = measures.names
    # A host catalogue may re-declare the engine's core, and the core wins that tie wherever a name
    # is described. A re-declared core name is carried where the core carries it or nowhere, so it
    # opens no door here.
    return frozenset(names) - METRIC_DESCRIPTORS.keys()


def _per_result_measure_names(measures: MeasureRegistry) -> frozenset[str]:
    """Every described name a single result can carry a code-graded value under.

    Four sources, which are exactly the four the bundle's measure walk reads: the result's own
    fields, the measures it implies, its covariates, and the host's own measures. A core
    measure outside them — a run-summary statistic such as ``mean_total_ms`` or ``p95_total_ms``,
    a report-level accuracy — is described and never carried by any one result, so a bar on it is
    never read.

    Args:
        measures: The host's measure registry.
    """
    return _carried_field_names() | DERIVED_PER_RESULT_MEASURES | _COVARIATE_MEASURES | _host_measure_names(measures)


def _listed(annotation: object) -> bool:
    """Whether a field annotation holds a LIST of carriers (one observation per element), through ``Optional``."""
    if get_origin(annotation) in (list, tuple):
        return True
    return any(_listed(argument) for argument in get_args(annotation) if argument is not type(None))


@cache
def _single_carried_field_names() -> frozenset[str]:
    """Every name a result's own fields carry ONCE — its scalars and the fields of its single sub-models.

    :func:`_carried_field_names` without the listed carriers: a usage row per role or a delivery per
    async call carries its fields once per element, so a result holds several values under such a name
    and no one of them is the result's. Derived by reflection, cached, for the same reasons.
    """
    names: set[str] = set()
    for field_name, field in EvalResult.model_fields.items():
        carriers = list(_carriers_of(field.annotation))
        if not carriers:
            names.add(field_name)
        elif not _listed(field.annotation):
            for carrier in carriers:
                names.update(carrier.model_fields)
    return frozenset(names)


def mechanism_measure_names(measures: MeasureRegistry) -> frozenset[str]:
    """Every described name one result carries a single value under — what a lever may declare it acts on.

    The per-result set :func:`_per_result_measure_names` reads, narrowed to the values a result holds
    once: its own scalars and single sub-models, the measures it implies, its covariates and the host's
    own measures. A field of a listed carrier is left out — ``reasoning_tokens`` is recorded per usage
    row, one per role — because comparing a lever's levels needs one value per result, and pooling one
    role's count with another's describes no mechanism.

    Args:
        measures: The host's measure registry.
    """
    return (
        _single_carried_field_names()
        | DERIVED_PER_RESULT_MEASURES
        | _COVARIATE_MEASURES
        | _host_measure_names(measures)
    )


def resolve_bar_name(
    name: str,
    *,
    rubric_dimensions: Mapping[str, RubricScale],
    goal_state_checks: Collection[str],
    measures: MeasureRegistry,
) -> BarName | UnreadableBarName:
    """Say what a bar names and where a result carries it, or why no result can.

    **The one answer to that question.** The declaration gate asks it with the campaign's template's
    names, and the bundle's bar adjudication asks it with the names the campaign's results actually
    carry, so a name the gate admits is read the way the gate resolved it, and a name the
    adjudicator could never read is refused at authoring time rather than reported afterwards as
    carried by nobody.

    Resolution order, most specific first, so a name two kinds share is read one way everywhere:

    1. A described measure a result can carry (:func:`_per_result_measure_names`). Admitted when it
       is code-graded (:func:`~threetears.evals.contracts.metrics.is_code_graded`), numeric and declares
       a better end. A non-numeric one (categorical, boolean) is refused as ``not_numeric`` and a
       directionless one as ``no_better_end``.
    2. A judged dimension — one of ``rubric_dimensions``, or a reserved dual-score axis, which every
       run scores whatever its template says.
    3. One of ``goal_state_checks``, by its verbatim text or by the measure name the bundle publishes
       for it (:func:`~threetears.evals.contracts.metrics.goal_check_measure`), which reads as the check.

    Anything else is ``not_carried``, and the reason says what the name IS where anything describes
    it: a judge-mediated summary such as ``mean_score``, a composite such as ``pass_hat_k``, or a
    run-level statistic such as ``mean_total_ms`` is described and still never lands on a result.

    Args:
        name: What the bar names.
        rubric_dimensions: The judged dimensions in play and the scale each is answered on — the
            template's rubric at authoring time, the dimensions the results were scored on at
            adjudication.
        goal_state_checks: The goal-state check texts in play, on the same two readings.
        measures: The host's measure registry, whose declarations join the engine's core.

    Returns:
        A :class:`BarName` when a result can carry the name with a direction, else an
        :class:`UnreadableBarName`.
    """
    # A check's published measure name reads as the check itself: the bundle and a memo cite a check
    # as `goal_state:<check>`, and a bar declared by that name holds the same check to the threshold.
    if (check := goal_check_of(name)) is not None:
        name = check
    described = {descriptor.name: descriptor for descriptor in list_metrics(measures)}
    carried = _per_result_measure_names(measures)
    descriptor = described.get(name)
    if descriptor is not None and name in carried and is_code_graded(descriptor, measures):
        if descriptor.data_type != "numeric":
            return UnreadableBarName(
                name, "not_numeric", f"{name} is {descriptor.data_type}, so a threshold has nothing to compare against"
            )
        if (what := no_better_end(descriptor)) is not None:
            return UnreadableBarName(
                name,
                "no_better_end",
                f"{name} declares no better end — {what} — so clearing a threshold on it means nothing",
            )
        return BarName(name, "measure", descriptor)
    if name in RESERVED_DIM_IDS or name in rubric_dimensions:
        return BarName(
            name,
            "judged",
            describe_rubric_dim(name, scale="ordinal" if name in RESERVED_DIM_IDS else rubric_dimensions[name]),
        )
    if name in goal_state_checks:
        # The check's RATE, which is what a bar on it reads — not the check's boolean observation.
        return BarName(name, "goal_state", describe_goal_check_rate(name))
    if descriptor is None:
        return UnreadableBarName(
            name,
            "not_carried",
            f"nothing describes {name}, and it is neither a judged dimension nor a goal-state check here",
        )
    if name in METRIC_DESCRIPTORS and name not in carried:
        what = f"a {descriptor.family} measure computed over many results" if descriptor.family else "a measure"
        return UnreadableBarName(
            name,
            "not_carried",
            f"{name} is described as {what}, and no single result carries it, so no bar on it is ever read",
        )
    return UnreadableBarName(
        name,
        "not_carried",
        f"{name} is a {descriptor.family} measure, which no result carries as a code-graded value — "
        "a judged or composite figure is read by its dimension, never by a summary's name",
    )


def refuse_an_undeclarable_design(
    design: CampaignDesign,
    *,
    behavior: str,
    template: EvalTemplate | None,
    profile: HostProfile,
) -> None:
    """Refuse a declaration this host cannot honour — at authoring time, not at analysis time.

    R10's "a gate, not a page". Four things are checked and all four are cheap, which is the
    point: a campaign declaring an axis its host will not accept — neither a registered lever nor a
    member of an open family its own membership test recognises — cannot
    vary that axis — and an apparatus input or a label is registered without being one; a bar
    naming something no result of the campaign will carry can never be read, so it is not a
    standard but a sentence; a campaign proposing a bar looser than the registered standard is not
    being held to a standard at all; and a bar declaring the opposite better-direction from the
    measure's own descriptor is cleared by exactly the values it should fail. Discovering any of
    them after the runs have executed means the money is already spent and the report is one
    nobody can act on.

    **A bar may name exactly what :func:`resolve_bar_name` resolves**, which is exactly what the
    bundle's bar adjudication can read, because the adjudication asks the same function: a
    code-graded measure a single result carries with a better end (its own fields, the measures it
    implies, its covariates, or one the host declares); a judged dimension — a rubric
    dimension the campaign's template declares, or one of the two reserved axes every run scores;
    or a goal-state check the template declares, by its verbatim text. The template-minted names
    are the open half of the measure name space — no registry enumerates them — but they are not
    unenumerable: the template that MINTS them lists them, so once the campaign's template is read
    every name a bar may carry is enumerable and this can refuse rather than guess. A name the
    registry describes is still refused when no result carries it with a direction: a run-level
    statistic (``mean_total_ms``), a judge-mediated summary (``mean_score``), a composite
    (``mean_composite``), a categorical, a raw count or a diagnostic. **The one exception is**
    :data:`~threetears.evals.contracts.metrics.FRONTIER_RANKING_MEASURE` (``pass_hat_k``): no cell
    carries it, but the bundle passes a bar on it to the frontier, which reads it on each contestant's
    pass^k interval, so it is admitted with a threshold in pass^k's range ``[0, 1]`` and its direction
    checked against the engine's descriptor. A phase-timing key is carried too,
    but no catalogue describes one, so a bar on it has no descriptor to be read against and is
    refused with the rest. **What the gate cannot see**: a host measure is admitted on the host's
    declaration alone, since nothing in a descriptor says whether the host's kind lands it on a
    result — so a bar on one the kind never emits passes here and is adjudicated as carried by no
    result.

    **The last two checks are one evaluability axis and one standard**, and they are not one check
    twice. The looser-bar rule is the campaign against the INCUMBENT and can only fire where a
    bar is registered; the direction rule is the campaign against the measure DESCRIPTOR and
    fires whether or not one is. A host with no registered bars has only the second.

    Every check reads ``profile``'s registries, so the same declaration is legal for one
    consumer and refused for another — which is correct, and is why this cannot live as a
    validator on the model. A model validator would have to name a registry, and the whole
    contract is that the engine does not know whose registry it is holding.

    Args:
        design: The declaration to check.
        behavior: The campaign's behavior, which is the scope a registered bar is keyed under.
        template: The campaign's template, whose rubric dimensions and goal-state checks are the
            judged and checked names its results will carry. Required, so no caller can reach
            this gate without deciding which template it is authoring against. ``None`` means the
            campaign has no template the caller could read — none named, or a reference that does
            not resolve — so no name of either kind is known and a bar on one is refused, saying
            that rather than calling the rubric empty.
        profile: The host whose levers, measures and bars the declaration is checked against.

    Raises:
        ValueError: An axis the host will not accept — neither a registered lever nor a
            recognised open-family member — naming the
            axis and carrying the registry's own ``axis_remedy`` as the vocabulary to pick from; a
            bar names nothing a verdict can be given on, naming the bar, why, and every set it
            could have named; a
            bar is looser than the registered incumbent, quoting the
            registered value; or a bar contradicts the declared better-direction of what it names,
            naming it and which way its descriptor runs.
    """
    # ASK the profile rather than re-deriving from its registry. `HostProfile.controllable` is
    # R10's evaluability map and the design names this gate as its one use — re-deriving
    # `{d.name for d in profile.sweepables.declarations}` here would make two implementations of
    # "is this axis controllable", free to disagree, with the authoritative one left caller-less.
    # It also throws away the map's reason string, which is the half an operator acts on.
    uncontrollable = {
        axis.axis_id: coverage
        for axis in design.axes
        if (coverage := profile.controllable(axis.axis_id)).state != "covered"
    }
    if uncontrollable:
        reasons = "; ".join(f"{axis}: {coverage.reason}" for axis, coverage in sorted(uncontrollable.items()))
        # The reason ALREADY carries the remedy — every `refuse_as_axis` refusal ends with the
        # registry's own `axis_remedy` — so no list is appended here. Appending one built from
        # `lever_names` is what left an author reading a vocabulary that omitted every open-family
        # member this same gate accepts: `lever_names` cannot express a family, and `axis_remedy`
        # can. One derivation, not two free to disagree.
        raise ValueError(f"host '{profile.host_id}' cannot vary every axis this campaign declares — {reasons}")

    # The naming check and the direction check read ONE resolver, and the bundle's bar adjudication
    # reads the same one — so what this gate admits is exactly what a verdict can be given on, and
    # the three cannot come to disagree about what a bar names. The template supplies the judged and
    # goal-state names; with no template only the two reserved axes every run scores are known.
    rubric = {dim.name: dim.scale for dim in template.rubric} if template is not None else {}
    checks = list(template.goal_state_checks) if template is not None else []
    resolved = {
        override.measure_id: resolve_bar_name(
            override.measure_id, rubric_dimensions=rubric, goal_state_checks=checks, measures=profile.measures
        )
        for override in design.bars
    }

    # Before either bar-against-a-standard check, because both presume the bar names something:
    # a threshold on a name no result carries is never compared with anything, and the campaign
    # reads as held to a standard it cannot be held to.
    # A bar on the frontier's ranking measure is the one bar no cell carries and something still reads: the
    # bundle passes it to the frontier, which decides each contestant's pass^k interval against it. It is held
    # to pass^k's own range here, since the frontier refuses a threshold outside it.
    unreadable = [
        (override, reading)
        for override in design.bars
        if override.measure_id != FRONTIER_RANKING_MEASURE
        and isinstance(reading := resolved[override.measure_id], UnreadableBarName)
    ]
    out_of_range = [
        override.threshold
        for override in design.bars
        if override.measure_id == FRONTIER_RANKING_MEASURE and not 0.0 <= override.threshold <= 1.0
    ]
    if out_of_range:
        raise ValueError(
            f"a bar on {FRONTIER_RANKING_MEASURE} is read by the frontier against pass^k, a probability, so its "
            f"threshold must lie in [0, 1]; this campaign declares {', '.join(repr(t) for t in out_of_range)}"
        )
    if unreadable:
        # The author's own declared value, echoed as written: a refusal quotes its input rather than
        # restating it under the reader-facing number rule, which lives outside the contracts set.
        named = "; ".join(
            f"the bar at {o.threshold!r} names '{o.measure_id}': {reading.reason}" for o, reading in unreadable
        )
        if template is None:
            minted = (
                "The campaign's template: none — this campaign names no template that could be read, so no "
                "rubric dimension or goal-state check is known beyond the reserved axes every run scores"
            )
        else:
            dims = ", ".join(sorted(rubric)) or "none — the campaign's template declares no rubric"
            goals = "; ".join(repr(check) for check in sorted(checks)) or "none — the campaign's template declares none"
            minted = f"Rubric dimensions of the template: {dims}. Goal-state checks of the template: {goals}"
        readable = sorted(
            descriptor.name
            for descriptor in list_metrics(profile.measures)
            if isinstance(
                resolve_bar_name(
                    descriptor.name, rubric_dimensions={}, goal_state_checks=(), measures=profile.measures
                ),
                BarName,
            )
        )
        raise ValueError(
            f"a bar must name something this campaign's results will carry and a verdict can be given on — "
            f"{named}. {minted}. Reserved judged axes: {', '.join(sorted(RESERVED_DIM_IDS))}. "
            f"Measures a result carries with a direction (the engine's core and host '{profile.host_id}''s "
            f"catalogue): {', '.join(readable)}. And {FRONTIER_RANKING_MEASURE}, which the frontier reads"
        )

    for override in design.bars:
        try:
            profile.bars.check_override(
                Bar(
                    behavior=behavior,
                    measure=override.measure_id,
                    threshold=override.threshold,
                    # The DECLARED direction goes in, and `check_override` deliberately judges
                    # against the incumbent's rather than trusting it — a proposal that restates
                    # the rule its own threshold is judged by can flip one boolean and register a
                    # looser bar as tighter. Passing it anyway is what lets that check fire.
                    higher_is_better=override.direction == "higher_is_better",
                    rationale=f"campaign override for {behavior}",
                )
            )
        except BarRegistrationError as e:
            raise ValueError(str(e)) from e

    # The fourth check, and the one `check_override` above cannot make: it judges a proposed
    # direction against an INCUMBENT, and a host that has registered no bars leaves
    # the campaign's own boolean as the only statement of which way clearing runs. The DESCRIPTOR
    # owns that fact whether or not a bar exists, which is what `BarRegistry.validate_against`
    # already enforces for the bars a host registers; a campaign proposal reaching the same
    # registry unchecked is cleared by exactly the values it should fail, and reads as a standard
    # the whole way.
    #
    # It speaks for every bar the naming check admitted, the template-minted half included: a
    # judged rubric dimension is scored 1-5 with higher better, and a goal-state check is a
    # pass/fail whose better end is passing, and `describe_rubric_dim` / `describe_goal_check_rate` say
    # so by construction. A bar on either read the other way round is the same defect as one on a
    # catalogue measure.
    #
    # There is deliberately no observability predicate on `HostProfile` to ask before this lookup:
    # the resolver returns the descriptor this check needs anyway, so a membership boolean asked
    # first could not change a verdict and no test could catch its removal. The rest of the Coverage
    # family (`controllable`, `representable`, `presumable`, `addressable`) exists for callers that
    # want a REASON without the record behind it; a gate that needs the record is not one of them.
    # The PREDICATE is `bars.contradicts_descriptor`, not a third hand-written copy of it: the
    # descriptor owns this rule, `validate_against` is its other enforcement site, and two copies of
    # one boolean is how they come to disagree. (`check_override` is NOT a third site — it compares
    # a proposal against the incumbent BAR, never against a descriptor.) What is local here is only
    # which bars to ask about and what to say when the answer is yes.
    def contradiction(reading: BarName) -> str:
        """Say which way a bar's name runs and who declared it — the host, or the engine for a template's name."""
        way = "higher" if reading.higher_is_better else "lower"
        if reading.kind == "measure":
            return f"{reading.name} is declared {way}-is-better by this host and this campaign declares the opposite"
        return (
            f"{reading.name} is declared {way}-is-better by the engine for every {reading.descriptor.family} measure "
            "and this campaign declares the opposite"
        )

    contradicted = [
        contradiction(reading)
        for override in design.bars
        if isinstance(reading := resolved[override.measure_id], BarName)
        and contradicts_descriptor(reading.descriptor, override.direction == "higher_is_better")
    ]
    contradicted.extend(
        f"{FRONTIER_RANKING_MEASURE} is declared higher-is-better by the engine and this campaign declares the opposite"
        for override in design.bars
        if override.measure_id == FRONTIER_RANKING_MEASURE
        and contradicts_descriptor(
            METRIC_DESCRIPTORS[FRONTIER_RANKING_MEASURE], override.direction == "higher_is_better"
        )
    )
    if contradicted:
        raise ValueError(
            f"host '{profile.host_id}' describes these measures differently from the bars this campaign "
            f"declares — {'; '.join(contradicted)}. The descriptor owns the direction, so a bar read the "
            "other way round is not a tighter standard — it is cleared by exactly the values it should fail"
        )


__all__ = [
    "BarName",
    "BarNameKind",
    "BarNameRefusal",
    "BarOverride",
    "CampaignDesign",
    "ControlDeclaration",
    "Question",
    "SweptAxis",
    "UnreadableBarName",
    "JUDGED_MERIT_AXIS",
    "axis_in_question_scope",
    "exploratory_reading",
    "mechanism_measure_names",
    "reconcile_question_edits",
    "refuse_an_undeclarable_design",
    "resolve_bar_name",
]
