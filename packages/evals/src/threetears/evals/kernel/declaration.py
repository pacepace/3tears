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
`unswept` **and named**, where an inferred design can only report what happened to run. A level
is per axis; a CELL is a combination of declared levels, and a design may say which combinations
it meant to run (``crossing``, ``skipped_cells``), so a cell left out on purpose reads as skipped by
design and one that was meant to run and did not reads as a gap. A design that says nothing about
crossings has no cell coverage: an unrun combination there is neither.

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

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

from pydantic import Field, TypeAdapter, computed_field, field_validator, model_validator

from threetears.evals.schema.campaign_upgrade import aware_instant
from threetears.evals.schema.base import CoreShapeModel, EvalBaseModel, EvalDocumentModel
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.identity import IDENTITY_VERSION, compute_variant_key
from threetears.evals.kernel.carried_measures import mechanism_measure_names, per_result_measure_names
from threetears.evals.kernel.host.bars import Bar, BarRegistrationError, contradicts_descriptor, no_better_end
from threetears.evals.schema.values import Scale, SweepableValue
from threetears.evals.kernel.metrics import (
    FRONTIER_RANKING_MEASURE,
    METRIC_DESCRIPTORS,
    MeritAxis,
    MetricDescriptor,
    describe_goal_check_rate,
    describe_rubric_dim,
    goal_check_of,
    is_code_graded,
    is_latency_measure,
    list_metrics,
    rule_rates,
)
from threetears.evals.schema.models import (
    DECLARED_NAME_CHARS,
    RESERVED_DIM_IDS,
    SCALES,
    ApparatusProvenance,
    EvalTemplate,
    RubricScale,
    declared_name,
    utc_now_iso,
)

if TYPE_CHECKING:
    from threetears.evals.kernel.host.measures import MeasureRegistry
    from threetears.evals.kernel.host.profile import HostProfile

#: Characters in a variant key — a sha256 hex digest, which is what
#: :func:`~threetears.evals.kernel.identity.compute_variant_key` produces.
_VARIANT_KEY_LENGTH = 64


class SweptAxis(EvalDocumentModel, CoreShapeModel):
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

        :param values: The authored levels, before validation.
        :ptype values: Any
        :return: The levels, with any content-authored one replaced by its addressed value.
        :rtype: Any
        :raises ValueError: A level carries neither ``content_hash`` nor ``content``, so there is
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


class BarOverride(EvalDocumentModel, CoreShapeModel):
    """A standard this campaign holds itself to, tighter than the registered one.

    Tighter only. A campaign that could lower the bar it is judged against is not being judged;
    :meth:`~threetears.evals.kernel.host.bars.BarRegistry.check_override` refuses a looser proposal and
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


class GuardrailMargin(EvalDocumentModel, CoreShapeModel):
    """How much worse than the control an arm may be on one judged guardrail, and still hold it.

    A measure's margin is its descriptor's ``materiality_threshold``, declared by the host. A judged dimension has
    no descriptor of its own to carry one — the engine describes every rubric dimension alike — so a boundary
    dimension's margin is declared here, by the campaign that decides it. With none, the dimension is held at zero
    change: ``held`` then needs the arm shown no worse at all.
    """

    dimension: str = Field(
        min_length=1,
        description=(
            "The judged guardrail: a rubric dimension of the campaign's template on the `boundary` axis, spelled "
            "exactly as the template writes it. A capability dimension, a measure or an undeclared name is refused "
            "at authoring time."
        ),
    )
    margin: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "The worsening tolerated before `breached`, in the dimension's own units (points of a 1-5 scale, a share "
            "of a pass/fail one): positive, and narrower than the scale."
        ),
    )


class RateDrop(EvalDocumentModel, CoreShapeModel):
    """A clause of a floor: no pass rate may drop by more than ``max_points`` against the control.

    ``reading`` None reads every pass/fail reading the arms share: the overall pass rate and each check's. A name
    reads that one alone: a goal-state check by its verbatim text, a pass/fail judged dimension by its name, or a
    pass/fail measure by its key; a name the campaign does not measure is refused when the design is declared
    (:func:`decision_readings`).
    Each rate it reads settles on its own, and only over every arm at the case floor: shown to break the clause where
    the bounded interval on its drop lies wholly past the margin, shown to hold it where that interval lies within the
    margin, and unsettled otherwise, a drop past the margin on its point estimate alone included. The clause is keep
    when one rate breaks it, switch when every rate holds it, and otherwise unsettled. Through 0.70 an observed drop
    past the margin settled keep on any number of cases.
    """

    kind: Literal["rate_drop"] = Field(
        default="rate_drop", description="The clause's kind, which a stored rule is read back by. Never set by hand."
    )
    max_points: float = Field(
        gt=0.0,
        lt=100.0,
        allow_inf_nan=False,
        description="The most a pass rate may drop, in points of a 0-100 percent: positive, and narrower than the scale.",
    )
    reading: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The one pass/fail reading this clause reads: a goal-state check's verbatim text, or a measure or judged "
            "dimension's key. None reads every pass/fail reading the arms share, the overall pass rate included."
        ),
    )


class CostLower(EvalDocumentModel, CoreShapeModel):
    """A clause of a floor: the arm must cost less per reply than the control.

    A cost observed no lower settles keep on any number of cases, since an arm's price per reply is not a sample;
    switch needs every arm at the case floor and the cost contrast's interval wholly below zero.
    """

    kind: Literal["cost_lower"] = Field(
        default="cost_lower", description="The clause's kind, which a stored rule is read back by. Never set by hand."
    )


#: One clause of a floor, read back by its ``kind``.
DecisionClause = Annotated[RateDrop | CostLower, Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# The rules a declaration is held to, one implementation each: the campaign's design and an aspect's
# design (and a question's axes, and an arm's key) validate through these, so the two cannot drift.
# ---------------------------------------------------------------------------


def _each_merit_axis_once(axes: list[MeritAxis]) -> list[MeritAxis]:
    """Refuse a merit axis named twice: the list is a set of axes an answer moves, and a repeat says nothing."""
    if duplicates := sorted({axis for axis in axes if axes.count(axis) > 1}):
        raise ValueError(f"merit_axes names {', '.join(duplicates)} more than once; name each axis once")
    return axes


def _is_variant_key(value: str) -> bool:
    """Whether ``value`` has the shape of a variant key: a 64-character lowercase hex digest."""
    return len(value) == _VARIANT_KEY_LENGTH and all(c in "0123456789abcdef" for c in value)


def _date_an_undated_key(data: Any, key: str, version: str) -> Any:
    """Stamp this build's identity version beside a stored key that arrives without one.

    A key typed by hand states no predicate; the only one it can have been read under is the build accepting it.

    :param data: The authored or stored fields, before validation.
    :ptype data: Any
    :param key: The field holding the variant key.
    :ptype key: str
    :param version: The field holding the predicate it was computed under.
    :ptype version: str
    :return: The fields, with ``version`` stamped where ``key`` is set and ``version`` is not.
    :rtype: Any
    """
    if isinstance(data, Mapping) and data.get(key) is not None and data.get(version) is None:
        return {**data, version: IDENTITY_VERSION}
    return data


def _pre_image_is_the_key_s(key: str, levers: Mapping[str, SweepableValue] | None, version: int | None) -> bool:
    """Whether a lever map is the pre-image of ``key``, checked only under this build's predicate.

    A map recorded under another predicate digested to the key by THAT predicate's arithmetic, so holding it to
    this one would refuse every key declared before a bump: the exact document the map is kept to rescue.
    """
    return levers is None or version != IDENTITY_VERSION or compute_variant_key(levers) == key


def _refuse_repeated_design_entries(
    axes: Sequence[SweptAxis],
    bars: Sequence[BarOverride],
    guardrail_margins: Sequence[GuardrailMargin],
    merit_priority: Sequence[MeritAxis],
) -> None:
    """One entry per axis, per measure, per guardrail dimension and per ranked merit axis.

    A declared axis is the denominator of every coverage number and of the typed-answer validator's "did the
    answer address every axis"; a duplicate double-counts it. Two bars on one measure is worse: both persist,
    both read as the standard, and nothing decides which governs.

    :raises ValueError: An axis id, a measure id, a guardrail margin's dimension or a merit-priority axis appears
        twice.
    """
    ids = [axis.axis_id for axis in axes]
    if duplicates := sorted({axis for axis in ids if ids.count(axis) > 1}):
        raise ValueError(
            f"duplicate axis ids: {', '.join(duplicates)} — coverage and the typed-answer validator "
            f"count declared axes, so a repeat is counted twice and neither entry's levels govern"
        )
    measures = [bar.measure_id for bar in bars]
    if duplicates := sorted({m for m in measures if measures.count(m) > 1}):
        raise ValueError(
            f"more than one bar on: {', '.join(duplicates)} — both would read as this campaign's "
            f"standard on that measure and nothing decides which one it is held to"
        )
    guarded = [entry.dimension for entry in guardrail_margins]
    if duplicates := sorted({name for name in guarded if guarded.count(name) > 1}):
        raise ValueError(
            f"more than one guardrail margin on: {', '.join(duplicates)} — nothing would decide which one the "
            "dimension is held to"
        )
    priority = list(merit_priority)
    if duplicates := sorted({axis for axis in priority if priority.count(axis) > 1}):
        raise ValueError(
            f"merit_priority names {', '.join(duplicates)} more than once — it is a strongest-first ranking of "
            "axes, and an axis ranked twice puts its bars in two tiers; name each axis once"
        )


def _cell_identity(axes: Sequence[SweptAxis], cell: Mapping[str, str]) -> tuple[str, ...]:
    """A cell, authored as ``{axis_id: level content_hash or display}``, as its levels' content hashes in axis order.

    :raises ValueError: The cell does not name exactly the declared axes, or names a level its axis does not
        declare, or one that matches more than one declared level.
    """
    ids = [axis.axis_id for axis in axes]
    if sorted(cell) != sorted(ids):
        raise ValueError(
            f"a skipped cell names axes {sorted(cell)}, and a cell is one level of EVERY declared axis: {sorted(ids)}"
        )
    identity = []
    for axis in axes:
        named = cell[axis.axis_id]
        matches = [level for level in axis.values if named in (level.content_hash, level.display)]
        if len(matches) != 1:
            raise ValueError(
                f"a skipped cell names {named!r} on axis {axis.axis_id!r}, which "
                + ("is none of its declared levels" if not matches else "matches more than one declared level")
                + f"; name one of: {', '.join(level.display for level in axis.values)}"
            )
        identity.append(matches[0].content_hash)
    return tuple(identity)


def _skipped_by_design(
    axes: Sequence[SweptAxis],
    skipped_cells: Sequence[Mapping[str, str]],
    crossing: Literal["full", "star"] | None,
    cell: tuple[str, ...],
) -> bool:
    """Whether a design left ``cell`` (one content hash per axis, in axis order) out on purpose.

    True when ``skipped_cells`` names it, or ``crossing`` is ``star`` and it differs from the centre (each axis's
    first declared level) on more than one axis.
    """
    if cell in {_cell_identity(axes, skipped) for skipped in skipped_cells}:
        return True
    if crossing == "star":
        centre = [axis.values[0].content_hash for axis in axes]
        return sum(1 for level, middle in zip(cell, centre, strict=True) if level != middle) > 1
    return False


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

    :param axis: The reading's merit axis, or None when it serves none.
    :ptype axis: MeritAxis | None
    :param question_axes: The axes the question names; empty for an unscoped question.
    :ptype question_axes: Sequence[MeritAxis]
    :return: True when the question covers the reading.
    :rtype: bool
    """
    return axis is not None and (not question_axes or axis in question_axes)


class ControlDeclaration(EvalDocumentModel, CoreShapeModel):
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

        :return: The validated declaration.
        :rtype: ControlDeclaration
        :raises ValueError: The stimulus is uncontrolled and no reason was given, so a reader learns
            that something varied but not what — which is worse than not asking, because it
            reads as disclosed.
        """
        if self.stimulus == "uncontrolled" and not self.stimulus_reason.strip():
            raise ValueError(
                "an uncontrolled stimulus must state what varied instead — a bare 'uncontrolled' discloses nothing"
            )
        return self


#: How a declared control's keys were arrived at. ``undeclared`` — the design names no control. ``current`` — the
#: key was computed under this build's predicate, and it is the one key. ``rekeyed`` — it was computed under another
#: predicate and the stored lever map re-keys it, so the control matches the recomputed key and the stored one.
#: ``unrekeyable`` — it was computed under another predicate with no lever map beside it, so the stored key is the
#: only one, and no run this build stamps can carry it.
ControlSource = Literal["undeclared", "current", "rekeyed", "unrekeyable"]


class ResolvedControl(EvalBaseModel):
    """A declared control as the SET of keys that name its arm under this build — a value, never written back.

    A set, not one key, because stamped keys are never re-derived: a cell keeps the key it was stamped with
    under the predicate it ran under. A control declared under an older predicate therefore matches TWO keys
    once the predicate moves — the stored key, which the cells stamped under the older predicate still carry,
    and the key its lever map re-keys to, which the cells stamped under this one carry. Matching only the
    re-keyed one would read a campaign holding only old cells as stale. Which of them is the control ARM in a
    given campaign is a question about its cells, so it is answered where the cells are (:meth:`matched`,
    from ``_campaign_design``): the first one any cell carries, in :attr:`variant_keys` order. A campaign
    carrying both keeps ONE control arm, the re-keyed cells, and names the others as the control under the
    earlier predicate (``RealizedDesign.earlier_control``) — never pooled into the control's numbers, which
    would mix two predicates' keys, and never contrasted with it, which would read the control against
    itself.

    Returned rather than assigned for the reason every identity read in
    :mod:`~threetears.evals.kernel.identity` is: a key recomputed at read and saved onto the campaign
    would be indistinguishable from the one the operator declared, and the next save would turn a
    read into a backfill.
    """

    variant_keys: list[str] = Field(
        description=(
            "Every key that names the control's arm under this build, preferred first: the key its lever map "
            "re-keys to under this build's predicate (when the stored key was computed under another and the "
            "recomputation differs), then the stored key as declared — always, since cells stamped under its "
            "predicate carry it for good. Empty when the design declares no control."
        )
    )
    identity_version: int | None = Field(
        description=(
            "The predicate the stored key was computed under, which is what says how far behind it is. None "
            "when no control is declared."
        )
    )
    source: ControlSource = Field(description="How the keys were arrived at — see `ControlSource`.")
    stale_disclosure: str | None = Field(
        default=None,
        description=(
            "Set exactly when the control is `unrekeyable`: what a reader is told if no cell of the campaign "
            "carries the stored key — that the control cannot be resolved, the two predicates that say why, and "
            "the remedy. Never said while a cell does carry it: such a control resolves, to the cells stamped "
            "under its own predicate."
        ),
    )

    def matched(self, carried: Collection[str]) -> str | None:
        """The key that names the control's arm among ``carried`` — the first of :attr:`variant_keys` one carries.

        :param carried: The variant keys the campaign's cells carry.
        :ptype carried: Collection[str]
        :return: The control's key in this campaign, or None when no cell carries any key that names it.
        :rtype: str | None
        """
        return next((key for key in self.variant_keys if key in carried), None)


def _resolve_control_key(key: str, levers: Mapping[str, SweepableValue] | None, version: int | None) -> ResolvedControl:
    """The keys naming a control declared as ``key``, its pre-image ``levers`` and its predicate ``version``."""
    stored = version if version is not None else IDENTITY_VERSION
    if stored == IDENTITY_VERSION:
        return ResolvedControl(variant_keys=[key], identity_version=stored, source="current")
    if levers is not None:
        rekeyed = compute_variant_key(levers)
        keys = [key] if rekeyed == key else [rekeyed, key]
        return ResolvedControl(variant_keys=keys, identity_version=stored, source="rekeyed")
    return ResolvedControl(
        variant_keys=[key],
        identity_version=stored,
        source="unrekeyable",
        stale_disclosure=(
            f"The declared control {key} matches no observation here and cannot be resolved: it is a "
            f"bare variant key computed under identity predicate v{stored}, this build keys variants under "
            f"v{IDENTITY_VERSION}, and no lever map was recorded beside it to re-key it from, so no run this build "
            "stamps can carry it and every comparison here is marginal. Re-address the control from a run of its "
            "arm (control_from_run_id on campaign_update), which records the lever map beside the key."
        ),
    )


# ---------------------------------------------------------------------------
# A campaign of aspects: one question, its arms, and per aspect the design it is read under. Beside
# today's design and unused by it; the campaign that holds them is the next stored shape.
# ---------------------------------------------------------------------------


class AspectDesign(EvalDocumentModel, CoreShapeModel):
    """What one aspect of a campaign set out to vary and hold itself to — the declaration, one aspect's worth.

    An aspect is one part of what a campaign asks, measured on its own runs. The question (a campaign asks one), the
    control (the campaign's control :class:`CampaignArm`) and when it was declared (:attr:`Aspect.declared_at`) belong
    to the campaign as a whole, not here. The merit axes are the axes an answer would move on this aspect: None when
    the campaign asks no question, so nothing read here confirms anything, and an empty list for an unscoped question,
    which asks about every axis.
    """

    axes: list[SweptAxis] = Field(
        min_length=1,
        description=(
            "What this aspect set out to vary. At least one: a declaration with no axes declares nothing, and "
            "coverage and the typed-answer validator count these as their denominator."
        ),
    )
    bars: list[BarOverride] = Field(
        default_factory=list,
        description="Standards this aspect holds itself to, tighter than the registered ones. Empty = the registry's bars apply unchanged.",
    )
    guardrail_margins: list[GuardrailMargin] = Field(
        default_factory=list,
        description=(
            "The margin of each judged guardrail (a boundary rubric dimension) this aspect decides, one per "
            "dimension. Empty = none declared, and every judged guardrail is held at zero change. A measure's "
            "margin is its descriptor's `materiality_threshold`, never declared here."
        ),
    )
    intended_repetitions: int | None = Field(
        default=None,
        ge=1,
        description=(
            "How many repetitions this design intends per cell, counted per case: a cell's least-repeated case is "
            "what is compared. The bundle names every cell that falls short. None = unstated, which makes a "
            "shortfall undetectable rather than zero."
        ),
    )
    held_fixed: ControlDeclaration = Field(
        description=(
            "What held still while this aspect's runs ran — the stimulus, `controlled` or `uncontrolled`, and the "
            "apparatus, `commissioned` or `witnessed`. Not the control arm, which names WHICH CELL is the "
            "reference; this names WHAT WAS HELD STILL."
        )
    )
    merit_axes: list[MeritAxis] | None = Field(
        default=None,
        description=(
            "Which axes the campaign's question would move on this aspect. The bundle lists the bars on these axes "
            "(`verdict_order`). None = the campaign asks no question, so every reading here is exploratory and none "
            "confirms anything. Empty = an unscoped question, which asks about every axis."
        ),
    )
    merit_priority: list[MeritAxis] = Field(
        default_factory=list,
        description=(
            "Tie-break order when no bar picks a winner, strongest first. Empty = no stated preference, and the "
            "analysis must not invent one."
        ),
    )
    crossing: Literal["full", "star"] | None = Field(
        default=None,
        description=(
            "Which combinations of the declared levels (cells) this design meant to run. full = every combination; "
            "star = the centre (each axis's FIRST declared level) and every cell differing from it on one axis "
            "only, the rest skipped by design. None = the design says nothing about combinations."
        ),
    )
    skipped_cells: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Cells left out on purpose, each naming every declared axis once with one of its declared levels, by "
            "the level's content_hash or its display. A skipped cell that did not run is skipped by design, never "
            "a gap. Empty = none beyond what crossing skips."
        ),
    )
    measure_latency: bool = Field(
        default=False,
        description=(
            "Whether latency is under test on this aspect. Its runs must then be launched with "
            "`measure_latency=True`, one cell at a time; latency read under concurrency is withheld from every "
            "reading. False: a bar on a latency measure, a merit axis of latency or `latency` in `merit_priority` "
            "is refused when the design is declared."
        ),
    )

    @field_validator("merit_axes")
    @classmethod
    def _each_axis_once(cls, axes: list[MeritAxis] | None) -> list[MeritAxis] | None:
        """Refuse an axis named twice: the list is a set of axes an answer moves, and a repeat says nothing."""
        return None if axes is None else _each_merit_axis_once(axes)

    @model_validator(mode="after")
    def _axes_and_bars_are_keyed_too(self) -> AspectDesign:
        """One entry per axis, per measure, per guardrail dimension and per ranked merit axis.

        :return: The validated design.
        :rtype: AspectDesign
        :raises ValueError: One of them appears twice.
        """
        _refuse_repeated_design_entries(self.axes, self.bars, self.guardrail_margins, self.merit_priority)
        return self

    @model_validator(mode="after")
    def _skipped_cells_name_declared_levels(self) -> AspectDesign:
        """Each skipped cell names every declared axis once, at a level that axis declares — or it is refused.

        :return: The validated design.
        :rtype: AspectDesign
        :raises ValueError: A skipped cell names an undeclared axis, leaves one out, or names an undeclared level.
        """
        for cell in self.skipped_cells:
            self.cell_identity(cell)
        return self

    def cell_identity(self, cell: Mapping[str, str]) -> tuple[str, ...]:
        """A cell, authored as ``{axis_id: level content_hash or display}``, as its levels' content hashes in axis
        order.

        :param cell: The cell.
        :ptype cell: Mapping[str, str]
        :return: One content hash per declared axis, in the order the axes are declared.
        :rtype: tuple[str, ...]
        :raises ValueError: The cell does not name exactly the declared axes, or names a level its axis does not
            declare, or one that matches more than one declared level.
        """
        return _cell_identity(self.axes, cell)

    def declares_cells(self) -> bool:
        """Whether this design says which combinations of its levels it meant to run."""
        return self.crossing is not None or bool(self.skipped_cells)

    def skipped_by_design(self, cell: tuple[str, ...]) -> bool:
        """Whether the design left ``cell`` (one content hash per axis, in axis order) out on purpose.

        :param cell: The combination.
        :ptype cell: tuple[str, ...]
        :return: True when ``skipped_cells`` names it, or ``crossing`` is ``star`` and it differs from the centre on
            more than one axis.
        :rtype: bool
        """
        return _skipped_by_design(self.axes, self.skipped_cells, self.crossing, cell)


#: A margin as an aspect declares one: points of a 0-100 pass rate, strictly between the ends and finite.
Margin = Annotated[float, Field(gt=0.0, lt=100.0, allow_inf_nan=False)]


#: What an arm is to its campaign: the ``control`` every other arm is read against, or a ``candidate`` for it.
ArmRole = Literal["control", "candidate"]


class CampaignArm(EvalDocumentModel, CoreShapeModel):
    """One arm of a campaign: a contestant stack, named, and whether it is the control or a candidate.

    An arm is a **variant key**, not a run, so curating a run out of a campaign never takes an arm with it while
    another observation carries the same stack. The key is stored beside its pre-image, as every stored variant key
    is: ``levers``, the map it was digested from, and ``identity_version``, the predicate that digested it — so a
    change to how a map is digested re-keys the arm instead of orphaning it (:func:`resolve_control_arm`).

    **The key and its pre-image are the server's, never the caller's.** An authoring surface addresses an arm from a
    run and records the run's own key and lever map; a caller names the run, not the digest.
    """

    label: str = Field(
        min_length=1,
        description="What a reader calls this arm: unique within its campaign, and the word every page names it by.",
    )
    role: ArmRole = Field(
        description="`control`: the arm every other arm is read against, one per campaign at most. `candidate`: an arm read against it."
    )
    variant_key: str = Field(
        description=(
            "The contestant stack this arm is: a 64-hex variant key, ADDRESSED from a run rather than typed. Read "
            "through `identity_version` and `levers` beside it, never alone."
        )
    )
    levers: dict[str, SweepableValue] | None = Field(
        default=None,
        description=(
            "The arm's PRE-IMAGE: the resolved lever map `variant_key` was digested from, recorded from the run the "
            "arm was addressed from. Under a later identity predicate the arm matches the key this map re-keys to as "
            "well as the stored one. Server-owned. None = the key was typed, so it names only the cells stamped under "
            "its own predicate."
        ),
    )
    identity_version: int = Field(
        ge=1,
        description=(
            "Which key-derivation predicate (`IDENTITY_VERSION`) `variant_key` was computed under. A key that arrives "
            "without one is dated by the build accepting it, the only predicate it can have been read under."
        ),
    )
    named_at: str | None = Field(
        default=None,
        description=(
            "When the arm was addressed from its run (ISO-8601). A control named after an aspect's first result is "
            "disclosed on that aspect's pages as a criterion set after its readings were visible: what its comparisons "
            "are read against was chosen once the evidence existed. Named after its runs were launched and before any "
            "result landed, it was named before anything was seen. None = not recorded (an arm upgraded from a "
            "campaign stored before arms existed), which the pages say."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _an_undated_key_is_dated_by_the_build_accepting_it(cls, data: Any) -> Any:
        """Stamp this build's identity version beside a key that arrives without one, as a design's control is.

        :param data: The authored or stored fields, before validation.
        :ptype data: Any
        :return: The fields, with ``identity_version`` stamped where it is absent.
        :rtype: Any
        """
        return _date_an_undated_key(data, "variant_key", "identity_version")

    @field_validator("variant_key")
    @classmethod
    def _an_arm_is_a_variant_key(cls, value: str) -> str:
        """Refuse a key that is not the shape of a variant key: one that is not a digest joins no observation.

        :param value: The arm's key.
        :ptype value: str
        :return: The validated key.
        :rtype: str
        :raises ValueError: The value is not a 64-character lowercase hex digest.
        """
        if not _is_variant_key(value):
            raise ValueError(
                f"an arm's `variant_key` is a {_VARIANT_KEY_LENGTH}-character lowercase hex digest over the resolved "
                f"contestant stack, and '{value}' is not one; address the arm from a run of it, and the key is "
                "resolved for you"
            )
        return value

    @model_validator(mode="after")
    def _the_pre_image_is_the_arm_s_own(self) -> CampaignArm:
        """The lever map, under this build's predicate, digests to the arm's key.

        :return: The validated arm.
        :rtype: CampaignArm
        :raises ValueError: The map does not digest to ``variant_key``, so it is some other arm's pre-image and
            re-keying from it would move this arm onto that one.
        """
        if not _pre_image_is_the_key_s(self.variant_key, self.levers, self.identity_version):
            raise ValueError(
                f"arm {self.label!r}: `levers` does not digest to `variant_key` under this build's identity predicate, "
                "so it is not the arm's pre-image; address the arm from a run, which records the two together"
            )
        return self


def resolve_control_arm(arm: CampaignArm | None) -> ResolvedControl:
    """The keys that name a campaign's control arm under this build — the one place that is decided.

    The control is a variant key, and keys are never re-derived at read: a cell keeps the key it was stamped with, and
    a run survives an :data:`~threetears.evals.kernel.identity.IDENTITY_VERSION` bump because it keeps its lever map
    beside it. The arm keeps its map too (:attr:`CampaignArm.levers`), and this is where it is read. **The same
    predicate:** the key as stored, alone. **Another predicate, with a map:** the key recomputed from the map AND the
    stored key, which the arm's cells stamped under the older predicate still carry. **Another predicate, without
    one:** the stored key alone (``unrekeyable``), with the disclosure a reader is told when no cell carries it.

    :param arm: The control arm, or None for a campaign that declares none.
    :ptype arm: CampaignArm | None
    :return: The resolved control.
    :rtype: ResolvedControl
    """
    if arm is None:
        return ResolvedControl(variant_keys=[], identity_version=None, source="undeclared")
    return _resolve_control_key(arm.variant_key, arm.levers, arm.identity_version)


#: The fields of an aspect that say only how a page prints it: the one list every reader that must not see them reads.
#: A bundle carries them under the same names and leaves them out of its fingerprint, and a campaign bundle's
#: fingerprint leaves them out of its declaration's aspects.
ASPECT_PRESENTATION_FIELDS: Final[frozenset[str]] = frozenset({"short_name"})


def printed_aspect_name(label: str, short_name: str | None) -> str:
    """What a page prints for an aspect: its declared short name, else its whole label, never a shortening of it.

    The one statement of the rule: the campaign's uniqueness check and every page part
    (:func:`~threetears.evals.analysis.report.campaign.names.aspect_name`) read it.

    :param label: The aspect's label.
    :ptype label: str
    :param short_name: Its declared short name, or None.
    :ptype short_name: str | None
    :return: The name a page prints.
    :rtype: str
    """
    return short_name if short_name is not None else label


class Aspect(EvalDocumentModel, CoreShapeModel):
    """One part of what a campaign asks, read on its own runs: its template, its margin, its design and its runs.

    A campaign asks one question of its arms, and an aspect is where that question is measured: one behavior's
    template, the runs gathered on it, and the design they were gathered under. Each aspect is read on its own
    (shown better, shown worse, no worse than the margin or unclear against the control) and the campaign's
    recommendation reads them all.
    """

    label: str = Field(
        min_length=1,
        description=(
            "What a reader calls this aspect: unique within its campaign, and never holding a `/`, which separates a "
            "floor key's parts. A campaign stored before aspects existed reads as one aspect labelled with its behavior."
        ),
    )
    short_name: str | None = Field(
        default=None,
        # Presentation only: written only when declared, so an aspect naming none is stored as it was before the field.
        exclude_if=lambda name: name is None,
        description=(
            f"What a page prints for this aspect in table cells and sentences, at most {DECLARED_NAME_CHARS} "
            "characters; the label is printed once beside it. It is no other aspect's label or short name. None = the "
            "label is printed whole. No part of any identity, key or fingerprint, so naming an aspect changes no "
            "reading."
        ),
    )
    template_id: str | None = Field(
        default=None,
        description="Reference to an existing eval_template this aspect's runs were gathered on; None names none.",
    )
    stratum: None = Field(
        default=None,
        description="Which stratum of the template's cases this aspect reads. Always None: every aspect reads all of them.",
    )
    margin: float | None = Field(
        default=None,
        gt=0.0,
        lt=100.0,
        allow_inf_nan=False,
        description=(
            "How far, in points of a 0-100 pass rate, a candidate may fall below the control here before it is "
            "shown worse. None = no margin declared, which only an aspect upgraded from a campaign stored without "
            "one holds: its readings are shown better or unclear, never shown worse, and the page says no margin "
            "was declared."
        ),
    )
    design: AspectDesign | None = Field(
        default=None,
        description="What this aspect set out to vary and hold itself to. None = exploratory: its readings confirm nothing, and the page says so once.",
    )
    run_ids: list[str] = Field(
        default_factory=list,
        description="Curated membership: the runs this aspect reads. A run belongs to one aspect of a campaign at most.",
    )
    declared_at: str | None = Field(
        default=None, description="When this aspect's design was declared (ISO-8601); None when it declares none."
    )
    asked_at: str | None = Field(
        default=None,
        description=(
            "When this aspect was put to the campaign's question (ISO-8601), and so when its bars, and a margin or "
            "floor declared with it, were set: its pre-registration time. Each such criterion is disclosed on the "
            "aspect's pages where it is later than the aspect's first stored result, without changing any reading. "
            "An adopted aspect keeps its source's, so adopting it never re-dates it. None = no time recorded, which "
            "the pages say."
        ),
    )
    margin_declared_at: str | None = Field(
        default=None,
        description=(
            "When `margin` was declared (ISO-8601), where that was after `asked_at`: a margin added when the aspect "
            "was adopted is dated then, and is disclosed as set after its readings were visible where that is later "
            "than the aspect's first stored result. None = declared with the aspect (`asked_at`)."
        ),
    )

    @field_validator("label")
    @classmethod
    def _a_label_holds_no_key_separator(cls, label: str) -> str:
        """Refuse a label holding ``/``: a floor's key joins its aspect, kind and reading with it.

        :param label: The aspect's label.
        :ptype label: str
        :return: The label.
        :rtype: str
        :raises ValueError: The label holds ``/``, so a floor key on it could be read as another aspect's.
        """
        if "/" in label:
            raise ValueError(
                f"aspect label {label!r} holds '/', which separates a floor key's aspect, kind and reading; name it "
                "without one"
            )
        return label

    @model_validator(mode="after")
    def _a_short_name_is_short(self) -> Aspect:
        """Refuse a short name that is blank or longer than :data:`~threetears.evals.schema.models.DECLARED_NAME_CHARS`.

        :return: The validated aspect.
        :rtype: Aspect
        :raises ValueError: The short name is blank or too long.
        """
        if self.short_name is not None:
            declared_name(self.short_name, what=f"aspect {self.label!r}")
        return self


class Floor(EvalDocumentModel, CoreShapeModel):
    """A line a candidate is warned for crossing on one aspect: a typed clause, read on that aspect's readings.

    A floor only WARNS, and loudly: a broken floor is the first thing on the campaign's page, the writer must address
    it, and a recorded call that adopts the arm anyway names it as overridden (:attr:`CampaignCall.floors_overridden`).
    It never blocks. A stored decision rule reads back as floors on its one aspect, one per clause. A floor's
    :attr:`key` is unique within its campaign, which the campaign holding the floors enforces.
    """

    aspect: str = Field(min_length=1, description="The label of the aspect whose readings this floor is read on.")
    clause: DecisionClause = Field(
        description="The line: a `RateDrop` (no pass rate may drop by more than `max_points`) or a `CostLower` (the arm must cost less per reply)."
    )
    declared_at: str | None = Field(
        default=None,
        description=(
            "When this floor was declared (ISO-8601), where that was after its aspect was put to the question: a "
            "floor added when the aspect was adopted is dated then, and is disclosed as set after its readings were "
            "visible where that is later than the aspect's first stored result. None = declared with the aspect "
            "(`asked_at`)."
        ),
    )

    @model_validator(mode="after")
    def _a_reading_is_not_the_wildcard(self) -> Floor:
        """Refuse a rate floor whose reading is ``*``, which its key spells for every pass rate.

        :return: The validated floor.
        :rtype: Floor
        :raises ValueError: The clause names the reading ``*``, so its key could not be told from the all-rates floor's.
        """
        if isinstance(self.clause, RateDrop) and self.clause.reading == "*":
            raise ValueError(
                "a floor's reading '*' is what its key writes for every pass rate; name the reading, or leave it out "
                "to read every pass rate"
            )
        return self

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "`{aspect}/{kind}/{reading}`, with `*` for a clause that reads every pass rate or reads none: what a "
            "recorded call names an overridden floor by. Derived on every read, never stored as a choice."
        )
    )
    @property
    def key(self) -> str:
        """The floor's key: its aspect, its clause's kind and the reading it reads (``*`` for all or none)."""
        reading = self.clause.reading if isinstance(self.clause, RateDrop) else None
        return f"{self.aspect}/{self.clause.kind}/{reading or '*'}"


class OverriddenBar(EvalDocumentModel, CoreShapeModel):
    """An absolute bar a call's arm missed on one aspect, named as the writer's ``bars_missed`` names one: the aspect
    and the bar's measure."""

    aspect: str = Field(min_length=1, description="The aspect's label.")
    measure: str = Field(min_length=1, description="The bar's measure, spelled as the bundle spells it.")


class CampaignCall(EvalDocumentModel, CoreShapeModel):
    """What a person decided over a campaign's readings: the arm, why, and every broken floor and missed bar they chose
    to override.

    Code reports, the writer recommends, a person decides. The call is the person's, recorded beside what the
    readings said, so a later reader sees both — and a broken floor or missed bar the call overrode is named here,
    never silent.
    """

    arm: str = Field(
        min_length=1,
        description="The decided arm's label, as the page showed it; the control's label keeps the control.",
    )
    variant_key: str = Field(
        description=(
            "The decided arm's variant key, which is what the call decided: a label names an arm for a reader, and "
            "this says which contestant stack it was."
        )
    )
    reason: str = Field(
        default="", description="Why, in the decider's words. Optional, and the first thing a later reader wants."
    )
    floors_overridden: list[str] = Field(
        default_factory=list,
        description=(
            "The key (`Floor.key`) of every floor the decided arm broke and the decider overrode, each once. A call "
            "on an arm with a broken floor names it here or is refused; a key for a floor not broken is refused."
        ),
    )
    bars_overridden: list[OverriddenBar] = Field(
        default_factory=list,
        description=(
            "Every absolute bar the decided arm missed and the decider overrode, by aspect and measure, each once. A "
            "call on an arm that missed a bar names it here or is refused; a bar the arm did not miss is refused."
        ),
    )
    analysis_id: str | None = Field(
        default=None,
        description="The analysis whose memo the call was made over; None when it was made on code's readings alone.",
    )
    run_ids: list[str] = Field(
        default_factory=list,
        description=(
            "The campaign's runs when the call was made, so a page can say how many have joined since. A run curated "
            "away, or a result scored late on a run already held, is not told apart."
        ),
    )
    called_at: str = Field(default_factory=utc_now_iso, description="When the call was recorded (ISO-8601).")
    called_by: str = Field(
        min_length=1,
        description="Who made the call, as the recording surface knows them. Server-owned, as `created_by` is.",
    )

    @field_validator("variant_key")
    @classmethod
    def _a_call_is_on_a_variant_key(cls, value: str) -> str:
        """Refuse a key that is not the shape of a variant key.

        :param value: The decided arm's key.
        :ptype value: str
        :return: The key.
        :rtype: str
        :raises ValueError: The value is not a 64-character lowercase hex digest.
        """
        if not _is_variant_key(value):
            raise ValueError(
                f"a call's `variant_key` is a {_VARIANT_KEY_LENGTH}-character lowercase hex digest, and '{value}' is "
                "not one; record the decided arm's own key"
            )
        return value

    @field_validator("floors_overridden")
    @classmethod
    def _each_floor_once(cls, keys: list[str]) -> list[str]:
        """Refuse a floor named twice: overriding it once is the whole record.

        :param keys: The overridden floors' keys.
        :ptype keys: list[str]
        :return: The keys.
        :rtype: list[str]
        :raises ValueError: A key appears twice.
        """
        if duplicates := sorted({key for key in keys if keys.count(key) > 1}):
            raise ValueError(f"floors_overridden names {', '.join(duplicates)} more than once; name each floor once")
        return keys

    @field_validator("bars_overridden")
    @classmethod
    def _each_bar_once(cls, bars: list[OverriddenBar]) -> list[OverriddenBar]:
        """Refuse a bar named twice: overriding it once is the whole record.

        :param bars: The overridden bars.
        :ptype bars: list[OverriddenBar]
        :return: The bars.
        :rtype: list[OverriddenBar]
        :raises ValueError: An aspect and measure appear twice.
        """
        named = [(bar.aspect, bar.measure) for bar in bars]
        if duplicates := sorted({pair for pair in named if named.count(pair) > 1}):
            spelled = ", ".join(f"{measure} on {aspect}" for aspect, measure in duplicates)
            raise ValueError(f"bars_overridden names {spelled} more than once; name each bar once")
        return bars


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


@dataclass(frozen=True)
class BarName:
    """A bar name a result can carry: which kind it is, and the descriptor that says which way it runs.

    :ivar name: The name, exactly as the bar spells it.
    :ivar kind: Which of the three kinds it resolved as — which is also where it is read; see
        :data:`BarNameKind`.
    :ivar descriptor: The measure's descriptor. Its ``higher_is_better`` is never ``None`` here, since
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

    :ivar name: The name, exactly as the bar spells it.
    :ivar refusal: Which of the three reasons applies; see :data:`BarNameRefusal`.
    :ivar reason: One sentence saying why, naming what the name is where anything describes it.
    """

    name: str
    refusal: BarNameRefusal
    reason: str


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

    1. A described measure a result can carry (:func:`~threetears.evals.kernel.carried_measures.per_result_measure_names`). Admitted when it
       is code-graded (:func:`~threetears.evals.kernel.metrics.is_code_graded`), numeric and declares
       a better end. A non-numeric one (categorical, boolean) is refused as ``not_numeric`` and a
       directionless one as ``no_better_end``.
    2. A judged dimension — one of ``rubric_dimensions``, or a reserved dual-score axis, which every
       run scores whatever its template says.
    3. One of ``goal_state_checks``, by its verbatim text or by the measure name the bundle publishes
       for it (:func:`~threetears.evals.kernel.metrics.goal_check_measure`), which reads as the check.

    Anything else is ``not_carried``, and the reason says what the name IS where anything describes
    it: a judge-mediated summary such as ``mean_score``, a composite such as ``pass_hat_k``, or a
    run-level statistic such as ``mean_total_ms`` is described and still never lands on a result.

    :param name: What the bar names.
    :ptype name: str
    :param rubric_dimensions: The judged dimensions in play and the scale each is answered on — the
        template's rubric at authoring time, the dimensions the results were scored on at
        adjudication.
    :ptype rubric_dimensions: Mapping[str, RubricScale]
    :param goal_state_checks: The goal-state check texts in play, on the same two readings.
    :ptype goal_state_checks: Collection[str]
    :param measures: The host's measure registry, whose declarations join the engine's core.
    :ptype measures: MeasureRegistry
    :return: A :class:`BarName` when a result can carry the name with a direction, else an
        :class:`UnreadableBarName`.
    :rtype: BarName | UnreadableBarName
    """
    # A check's published measure name reads as the check itself: the bundle and a memo cite a check
    # as `goal_state:<check>`, and a bar declared by that name holds the same check to the threshold.
    if (check := goal_check_of(name)) is not None:
        name = check
    described = {descriptor.name: descriptor for descriptor in list_metrics(measures)}
    carried = per_result_measure_names(measures)
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


def _registered_judged_bar_reading(
    name: str, *, rubric: Mapping[str, RubricScale], checks: Collection[str], profile: HostProfile
) -> UnreadableBarName | None:
    """Why a registered judged bar's dimension is one this campaign's results cannot carry, or None when they can.

    Resolved by :func:`resolve_bar_name`, the path a campaign's own bars take, so the refusal reads as theirs does.
    A name the template carries as something other than a judged dimension (a goal-state check written the same
    way) is refused too: the bar was registered as judged, and would otherwise be read as the check.
    """
    reading = resolve_bar_name(name, rubric_dimensions=rubric, goal_state_checks=checks, measures=profile.measures)
    if isinstance(reading, UnreadableBarName):
        return reading
    if reading.kind != "judged":
        return UnreadableBarName(
            name,
            "not_carried",
            f"{name} is registered as a judged dimension, and this campaign's template carries it as a {reading.kind}",
        )
    return None


class DesignRefused(RefusedInput):
    """A declaration this host cannot honour, as what is wrong with it and what to declare instead.

    What :func:`refuse_an_undeclarable_design` raises; the authoring surface hands
    its two halves on apart, as every refusal states them.
    """


class UndeclarableAxisError(DesignRefused):
    """A declared axis this host will not accept: neither a registered lever nor a recognised open-family member.

    Distinct so the authoring surface can point its author at the one call listing what IS declarable
    (:func:`~threetears.evals.analysis.campaigns.declarable_axes`) — a pointer that would be noise on a refused bar.
    """


#: How a campaign comes to have a template whose rubric dimensions and goal-state checks a bar or a guardrail
#: margin may name — what every refusal of one that found none says. Both ways are reachable at creation:
#: ``campaign_create`` takes ``template_id``, and infers it from runs that all ran one template.
NO_TEMPLATE_REMEDY = (
    "name the template with template_id when creating the campaign (templates_list names this scope's templates), "
    "or create it over runs that all ran one template, which is then the campaign's"
)


def _refuse_unreadable_guardrail_margins(design: AspectDesign, template: EvalTemplate | None) -> None:
    """Refuse a guardrail margin on anything but a boundary dimension of the campaign's template, or wider than its
    scale.

    :raises ValueError: A margin names a dimension the template does not declare, or one on the capability axis (a
        margin there would be read by nothing); or it is as wide as the dimension's scale, so every arm holds.
    """
    if not design.guardrail_margins:
        return
    dims = {dim.name: dim for dim in template.rubric} if template is not None else {}
    boundary = sorted(name for name, dim in dims.items() if dim.axis == "boundary")
    for entry in design.guardrail_margins:
        dim = dims.get(entry.dimension)
        if dim is None or dim.axis != "boundary":
            why = (
                f"this campaign names no template that could be read, so no rubric dimension is known — "
                f"{NO_TEMPLATE_REMEDY}"
                if template is None
                else "the template declares no such dimension"
                if dim is None
                else "it is on the capability axis, which is compared, not held"
            )
            raise DesignRefused(
                f"a guardrail margin names '{entry.dimension}': {why}",
                remedy=(
                    "declare a guardrail margin on a rubric dimension of the campaign's template on the boundary "
                    f"axis: {', '.join(boundary) or 'it declares none'}"
                ),
            )
        low, high = SCALES[dim.scale].value_range
        if entry.margin >= high - low:
            raise DesignRefused(
                f"a guardrail margin of {entry.margin!r} on '{entry.dimension}', scored {low:g} to {high:g}, is as "
                "wide as its scale, so every arm would hold whatever it did",
                remedy=f"declare a margin narrower than {high - low:g}",
            )


def decision_readings(template: EvalTemplate | None, profile: HostProfile) -> list[str]:
    """Every name a ``RateDrop.reading`` may carry on a campaign of ``template``: what the page reads as a pass rate.

    A goal-state check of the template, by its verbatim text; a judged dimension of the template on the pass/fail
    scale and the capability axis, by its name (a boundary dimension is a guardrail, held on its own); and each check
    measure the host's catalogue describes (:func:`~threetears.evals.kernel.metrics.rule_rates`: a boolean with a
    better end, either end, or a pass/fail scorer or grader), by its key. Exactly the rates
    :func:`~threetears.evals.analysis.report.decision.rule_readings` gives a rule, which reads the same function.

    :param template: The campaign's template; None when it names none that could be read.
    :ptype template: EvalTemplate | None
    :param profile: The host whose measures are read.
    :ptype profile: HostProfile
    :return: The names, checks first, then dimensions, then measures, each sorted.
    :rtype: list[str]
    """
    checks = sorted(template.goal_state_checks) if template is not None else []
    dims = (
        sorted(dim.name for dim in template.rubric if dim.scale == "pass_fail" and dim.axis == "capability")
        if template is not None
        else []
    )
    measures = sorted(rule_rates(list_metrics(profile.measures)))
    return [*checks, *dims, *measures]


def _refuse_unreadable_floor_readings(
    floors: Sequence[Floor], template: EvalTemplate | None, profile: HostProfile
) -> None:
    """Refuse a floor's ``RateDrop`` naming a reading this aspect will not measure as a pass rate.

    A misspelled reading would otherwise reach the page as a clause that never settles; refused here, it is fixed
    before a run is paid for. A clause reading every pass rate (``reading`` None) names nothing to check.

    :raises DesignRefused: A clause names a reading :func:`decision_readings` does not offer, naming it and listing what
        is offered.
    """
    offered = decision_readings(template, profile)
    unread = [
        (floor.key, floor.clause.reading)
        for floor in floors
        if isinstance(floor.clause, RateDrop)
        and floor.clause.reading is not None
        and floor.clause.reading not in offered
    ]
    if not unread:
        return
    named = "; ".join(f"floor {key} names {reading!r}" for key, reading in unread)
    where = f"campaign's template and host '{profile.host_id}'" if template is not None else f"host '{profile.host_id}'"
    raise DesignRefused(
        f"a floor's RateDrop must name a pass/fail reading this aspect measures — {named}, which it does not",
        remedy=(
            f"name one the {where} offer: {', '.join(repr(name) for name in offered) or 'none'}; or leave reading "
            "unset to read every pass rate" + ("" if template is not None else f". {NO_TEMPLATE_REMEDY}")
        ),
    )


def refuse_an_undeclarable_design(
    design: AspectDesign,
    *,
    floors: Sequence[Floor] = (),
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
    :data:`~threetears.evals.kernel.metrics.FRONTIER_RANKING_MEASURE` (``pass_hat_k``): no cell
    carries it, but the bundle passes a bar on it to the frontier, which reads it on each contestant's
    pass^k interval, so it is admitted with a threshold in pass^k's range ``[0, 1]`` and its direction
    checked against the engine's descriptor. A phase-timing key is carried too,
    but no catalogue describes one, so a bar on it has no descriptor to be read against and is
    refused with the rest. **What the gate cannot see**: a host measure is admitted on the host's
    declaration alone, since nothing in a descriptor says whether the host's kind lands it on a
    result — so a bar on one the kind never emits passes here and is adjudicated as carried by no
    result.

    **A host's registered bar on a judged dimension is resolved here too**
    (:attr:`~threetears.evals.kernel.host.bars.Bar.judged`). Its registration could not resolve a template-minted
    name, so every campaign of its behavior that does not override it resolves it against the campaign's template
    by the same resolver, and is refused in the same words when the template does not carry it as a judged
    dimension.

    **The last two checks are one evaluability axis and one standard**, and they are not one check
    twice. The looser-bar rule is the campaign against the INCUMBENT and can only fire where a
    bar is registered; the direction rule is the campaign against the measure DESCRIPTOR and
    fires whether or not one is. A host with no registered bars has only the second.

    Every check reads ``profile``'s registries, so the same declaration is legal for one
    consumer and refused for another — which is correct, and is why this cannot live as a
    validator on the model. A model validator would have to name a registry, and the whole
    contract is that the engine does not know whose registry it is holding.

    :param design: One aspect's declaration to check.
    :ptype design: AspectDesign
    :param floors: The floors the campaign declares on that aspect, whose ``RateDrop`` readings are checked; none by
        default.
    :ptype floors: Sequence[Floor]
    :param behavior: The campaign's behavior, which is the scope a registered bar is keyed under.
    :ptype behavior: str
    :param template: The campaign's template, whose rubric dimensions and goal-state checks are the
        judged and checked names its results will carry. Required, so no caller can reach
        this gate without deciding which template it is authoring against. ``None`` means the
        campaign has no template the caller could read — none named, or a reference that does
        not resolve — so no name of either kind is known and a bar on one is refused, saying
        that rather than calling the rubric empty.
    :ptype template: EvalTemplate | None
    :param profile: The host whose levers, measures and bars the declaration is checked against.
    :ptype profile: HostProfile
    :raises UndeclarableAxisError: An axis the host will not accept — neither a registered lever nor a
        recognised open-family member — naming the
        axis and carrying the registry's own ``axis_remedy`` as the vocabulary to pick from.
    :raises ValueError: A bar names nothing a verdict can be given on, naming the bar, why, and every
        set it could have named; a bar is looser than the registered incumbent, quoting the
        registered value; a bar contradicts the declared better-direction of what it names,
        naming it and which way its descriptor runs; or a bar, the aspect's merit axes or the merit
        priority asks about latency in a design that does not declare ``measure_latency``, naming each and
        the setting to turn on; or a floor's ``RateDrop`` names a reading :func:`decision_readings` does
        not offer.
    """
    _refuse_unreadable_floor_readings(floors, template, profile)
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
        raise UndeclarableAxisError(
            f"host '{profile.host_id}' cannot vary every axis this campaign declares — {reasons}",
            remedy=(
                "declare only the axes this refusal lists as declarable — the host's levers and the members of its open "
                "families it recognises, read from the registry this gate decides with"
            ),
        )

    # The naming check and the direction check read ONE resolver, and the bundle's bar adjudication
    # reads the same one — so what this gate admits is exactly what a verdict can be given on, and
    # the three cannot come to disagree about what a bar names. The template supplies the judged and
    # goal-state names; with no template only the two reserved axes every run scores are known.
    rubric = {dim.name: dim.scale for dim in template.rubric} if template is not None else {}
    checks = list(template.goal_state_checks) if template is not None else []
    _refuse_unreadable_guardrail_margins(design, template)
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
        (f"the bar at {override.threshold!r} names '{override.measure_id}'", reading)
        for override in design.bars
        if override.measure_id != FRONTIER_RANKING_MEASURE
        and isinstance(reading := resolved[override.measure_id], UnreadableBarName)
    ]
    # A host's registered bar on a judged dimension names a template-minted name no registry enumerates, so its
    # registration could not resolve it: each campaign of its behavior resolves it here, against its own template,
    # by the same resolver and refused in the same words. One the campaign overrides is replaced by the campaign's
    # own bar, which the resolution above already read.
    overridden = {override.measure_id for override in design.bars}
    for bar in profile.bars.bars:
        if not bar.judged or bar.behavior != behavior or bar.measure in overridden:
            continue
        unread = _registered_judged_bar_reading(bar.measure, rubric=rubric, checks=checks, profile=profile)
        if unread is not None:
            which = f"host '{profile.host_id}''s registered bar at {bar.threshold!r} names '{bar.measure}'"
            unreadable.append((which, unread))
    out_of_range = [
        override.threshold
        for override in design.bars
        if override.measure_id == FRONTIER_RANKING_MEASURE and not 0.0 <= override.threshold <= 1.0
    ]
    if out_of_range:
        raise DesignRefused(
            f"a bar on {FRONTIER_RANKING_MEASURE} is read by the frontier against pass^k, a probability, so its "
            f"threshold must lie in [0, 1]; this campaign declares {', '.join(repr(t) for t in out_of_range)}",
            remedy="declare it in [0, 1]",
        )
    if unreadable:
        # The author's own declared value, echoed as written: a refusal quotes its input rather than
        # restating it under the reader-facing number rule, which lives outside the kernel.
        named = "; ".join(f"{which}: {reading.reason}" for which, reading in unreadable)
        if template is None:
            minted = (
                "The campaign's template: none — this campaign names no template that could be read, so no "
                "rubric dimension or goal-state check is known beyond the reserved axes every run scores; "
                f"{NO_TEMPLATE_REMEDY}"
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
        raise DesignRefused(
            f"a bar must name something this campaign's results will carry and a verdict can be given on — "
            f"{named}. {minted}",
            remedy=(
                f"name one of the reserved judged axes ({', '.join(sorted(RESERVED_DIM_IDS))}), a measure a result "
                f"carries with a direction (the engine's core and host '{profile.host_id}''s catalogue: "
                f"{', '.join(readable)}), {FRONTIER_RANKING_MEASURE}, which the frontier reads, or a rubric dimension "
                "or goal-state check of the campaign's template"
            ),
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
            raise DesignRefused(
                str(e),
                remedy=(
                    "declare the bar at least as tight as the registered one, in the registered direction, or leave "
                    "the registered bar to stand"
                ),
            ) from e

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
        raise DesignRefused(
            f"host '{profile.host_id}' describes these measures differently from the bars this campaign "
            f"declares — {'; '.join(contradicted)}. The descriptor owns the direction, so a bar read the "
            "other way round is not a tighter standard — it is cleared by exactly the values it should fail",
            remedy="declare each bar in the direction its measure is described",
        )

    # Latency is under test only when the design says so (#701), never inferred from a bar or a question: a
    # campaign that asks about latency while its runs execute their cells concurrently has every latency
    # reading withheld, so the declaration that asks must be the declaration that measures. Refused, naming
    # the setting, rather than a silent switch to serial execution behind the author's back.
    if not design.measure_latency:
        latency = [
            f"the bar on {override.measure_id}"
            for override in design.bars
            if isinstance(reading := resolved[override.measure_id], BarName) and is_latency_measure(reading.descriptor)
        ]
        if "latency" in (design.merit_axes or []):
            latency.append("latency among the aspect's merit axes")
        if "latency" in design.merit_priority:
            latency.append("latency in merit_priority")
        if latency:
            raise DesignRefused(
                f"this campaign asks about latency — {'; '.join(latency)} — and does not declare latency under test; "
                "a run launched without it executes its cells concurrently, and the latency it records is never "
                "compared",
                remedy=(
                    "set measure_latency=True on its design and launch its runs with measure_latency=True (run_launch, "
                    "or sweep_launch for a sweep), so their cells run one at a time and their latency is read clean"
                ),
            )


__all__ = [
    "NO_TEMPLATE_REMEDY",
    "Margin",
    "ArmRole",
    "Aspect",
    "AspectDesign",
    "BarName",
    "BarNameKind",
    "BarNameRefusal",
    "BarOverride",
    "CampaignArm",
    "CampaignCall",
    "ControlDeclaration",
    "ControlSource",
    "CostLower",
    "DecisionClause",
    "Floor",
    "GuardrailMargin",
    "OverriddenBar",
    "RateDrop",
    "ResolvedControl",
    "SweptAxis",
    "DesignRefused",
    "UndeclarableAxisError",
    "UnreadableBarName",
    "JUDGED_MERIT_AXIS",
    "aware_instant",
    "axis_in_question_scope",
    "decision_readings",
    "mechanism_measure_names",
    "refuse_an_undeclarable_design",
    "resolve_bar_name",
    "resolve_control_arm",
]
