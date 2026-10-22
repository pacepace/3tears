"""Data models for the eval analysis subsystem.

:class:`EvalCampaign` is the hub — a first-class stored entity that groups eval
runs under one subject×behavior, so a generated ``EvalAnalysis`` has something to
attach to. Subject and Behavior stay *references* (an existing host subject, an
``eval_template``) rather than being reified into new tables; the campaign is the only new hub.
There is no battery pointer: the case set a run froze already enters its apparatus class, so a
second reference to "the scenario suite" would be a second answer to one question.

All three models mirror the shape of the sibling stored eval definitions
(:class:`~threetears.evals.schema.models.JudgeConfig` /
:class:`~threetears.evals.schema.models.CatalogRubricDim`): the strict document base,
``doc_type`` discriminator with a rejecting validator, ``schema_version``, and a
required ``scope_id`` — the opaque partition every stored eval document names, which
storage never stamps. A campaign and everything generated from it live in the scope its
runs live in.

``window`` is deliberately **not** a stored field — a campaign's [start, end] time
span is DERIVED from its member runs' ``created_at`` at read time (:func:`derive_window`),
so it can never drift from the runs it summarises.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, Self, get_args

from pydantic import Field, computed_field, field_validator, model_validator

from threetears.evals.kernel.authored import AuthoredCampaignAnalysis, Confidence
from threetears.evals.kernel.declaration import Aspect, CampaignArm, CampaignCall, Floor
from threetears.evals.kernel.evidence_tiers import JudgedEvidenceTier, JudgedTierRule, weakest_judged_tier
from threetears.evals.schema.values import SweepableValue
from threetears.evals.kernel.identity import compute_variant_key
from threetears.evals.schema.base import (
    CoreDocumentModel,
    CoreShapeModel,
    DurableDocumentModel,
    EvalDocumentModel,
    VerbatimJsonObject,
)
from threetears.evals.schema.job_lease import (
    CancelRequestReasonField,
    CancelRequestedAtField,
    FaultsField,
    LeaseField,
    LostField,
)
from threetears.evals.schema.refusal import RefusalField
from threetears.evals.schema.models import (
    ArmSchedule,
    CoreSchemaVersion,
    DurableSchemaVersion,
    SchemaVersion,
    utc_now_iso,
)
from threetears.evals.schema.versioning import CORE_SCHEMA_VERSION, DURABLE_SCHEMA_VERSION, REGENERABLE_SCHEMA_VERSION
from threetears.evals.schema.prose import ModelProse
from threetears.evals.kernel.surface import DecisionSurface
from threetears.evals.schema.logs import get_logger

log = get_logger(__name__)


#: How firmly the evidence settles a claim, as a qualitative tier — the tier a stored finding or
#: decision carries.
#:
#: A tier, not a probability: the reporter is a language model with no calibration behind a
#: number, and a "0.72" typed by one reads with the authority of a computed figure while nothing
#: checks it. Nothing sits below ``low`` — a claim the evidence contradicts is restated in the
#: direction the evidence supports, never asserted at a sub-coin-toss confidence. The tiers
#: name four affirmative bands (``very_high`` ≈ firm, ``high`` ≈ leaning, ``medium`` ≈ tentative,
#: ``low`` ≈ doubtful).
#:
#: The SAME declaration the writer authors against (:data:`~threetears.evals.kernel.authored.Confidence`),
#: not a second one: what the writer may author and what a stored analysis may hold are one set, so a
#: tier added to one is a tier of the other.
ConfidenceTier = Confidence

#: The tiers, strongest first — the order a surface ranks by.
CONFIDENCE_TIERS: tuple[ConfidenceTier, ...] = get_args(Confidence)


_CONFIDENCE_TIER_DESCRIPTION = "How firmly the evidence settles the claim: very_high | high | medium | low."

#: Which kind of per-cell reading a reference names — a measure (``CellFacts.measures``) or a
#: judged dimension (``CellFacts.judged``). Explicit rather than inferred from the name, because
#: the two namespaces are independent and a name in both would otherwise resolve to whichever one
#: the resolver happened to check first.
ReadingKind = Literal["measure", "judged"]

_READING_DESCRIPTION = (
    "Whether `measure_id` names a measure (`cell_measures[].measures`) or a judged dimension "
    "(`cell_measures[].judged`). Required: the finding's evidence tier is read off it, so a default "
    "would decide the tier."
)

#: What a finding's verdict stands on, as code reads it off the finding's resolved evidence rows —
#: never chosen by the report writer: code assigns the tier. Strongest first:
#:
#: - ``mechanical``: every reading is a measure — checks and metrics from the trace, which need no
#:   judge;
#: - ``calibrated``, ``separation``, ``undetermined``, ``incidental``: at least one reading is a judged
#:   score, and the finding stands on the weakest judged tier among its rows
#:   (:mod:`threetears.evals.kernel.evidence_tiers` defines each and the order they compose in);
#: - ``none``: the finding names no reading, so there is nothing for a tier to stand on.
#:
#: A judged tier is never read off a writer's claim: each judged row carries the tier code resolved
#: for its cell's judges (``EvidenceRow.judged_tier``).
EvidenceTier = Literal["mechanical", "calibrated", "separation", "undetermined", "incidental", "none"]


def evidence_tier_of(rows: list[JudgedEvidenceTier | None]) -> EvidenceTier:
    """The tier a verdict resting on these evidence rows stands on: the weakest among them.

    :param rows: Per evidence row, in any order: ``None`` for a measure, the row's judged tier for a judged score.
    :ptype rows: list[JudgedEvidenceTier | None]
    :return: ``none`` for no rows; the weakest judged tier when any row is judged; otherwise ``mechanical``.
    :rtype: EvidenceTier
    """
    if not rows:
        return "none"
    judged = [tier for tier in rows if tier is not None]
    return weakest_judged_tier(judged) if judged else "mechanical"


class CampaignWindow(EvalDocumentModel):
    """The [start, end] time span a campaign's runs cover — DERIVED, never stored.

    Produced by :func:`derive_window` from the member runs' ``created_at`` values
    and carried on :class:`CampaignView` in the ``campaign_get`` response. Not a
    field on :class:`EvalCampaign`: deriving it at read time is what keeps it from
    drifting as runs join or leave the campaign.
    """

    start: str = Field(description="Earliest member-run created_at (ISO-8601).")
    end: str = Field(description="Latest member-run created_at (ISO-8601).")


class EvalCampaign(DurableDocumentModel):
    """What a person set out to learn, and the runs gathered to learn it: one question, its arms, its aspects.

    **Durable** (:mod:`threetears.evals.schema.versioning`, durable v16): a campaign is authored — the question, the
    arms, the margin each aspect holds a candidate to, the floors — and nothing stored can make it again, so it
    outlives every core baseline reset and a later release upgrades it rather than refusing it. A campaign stored
    before aspects existed (core or durable v15) reads as one aspect labelled with its behavior.

    **A campaign asks ONE question of its arms** (``question``), and an **aspect** is where it is measured: one
    template, the runs gathered on it, the margin a candidate may fall below the control by there, and the design
    those runs were gathered under (:class:`~threetears.evals.kernel.declaration.Aspect`). A campaign has at least one
    aspect; a run belongs to one aspect of a campaign at most, though a run may belong to more than one campaign.
    Every per-aspect reader reads a campaign through :func:`~threetears.evals.analysis.aspects.aspect_view`, never
    through these fields.

    **The arms are variants, not runs** (:class:`~threetears.evals.kernel.declaration.CampaignArm`): each is a
    variant key beside its pre-image, addressed from a run and named by a label, so curating a run out never takes
    an arm with it — the control was once ``control_run_id``, a run on the campaign that archiving it destroyed. At
    most one arm is the control, and it comes first. A campaign that names no candidate arm reads
    each other variant among its runs as one (:func:`~threetears.evals.analysis.campaigns.campaign_arms`).

    **Floors only warn** (:class:`~threetears.evals.kernel.declaration.Floor`): each is a typed clause on one aspect,
    and a broken one is the first thing the page says. **The call** is what a person decided over the readings
    (:class:`~threetears.evals.kernel.declaration.CampaignCall`), None until one is recorded.

    **Exploratory is derived, never stored**: a campaign none of whose aspects declares a design is exploratory, and
    its report says once that its readings confirm nothing.

    Lives in one ``scope_id``, and so do its member runs: a run in another scope is refused at attachment
    (:mod:`threetears.evals.analysis.campaigns`). Subject and Behavior are *references*, not FK-enforced fields.
    ``subject_kind`` is a discriminator (free string) with no default kind. A campaign has no open/closed status:
    ``archived`` is the lifecycle a campaign has.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_campaign"] = "eval_campaign"
    schema_version: DurableSchemaVersion = DURABLE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    name: str = Field(min_length=1, description="Operator-facing campaign name.")
    description: str = Field(default="")
    subject_id: str = Field(
        min_length=1,
        description="Reference to an existing subject the host owns — NOT an FK-enforced field.",
    )
    subject_kind: str = Field(
        default="",
        description=(
            "Discriminator naming what kind of thing the subject is; a free string the host chooses, "
            "data the render/grouping layer keys on and never a code branch. Blank means this "
            "campaign declared no kind — not that it is of the host's usual kind."
        ),
    )
    behavior: str = Field(min_length=1, description="What the campaign is about, e.g. 'tool selection'.")
    question: str = Field(
        default="",
        description=(
            "The one question the campaign asks of its arms, in the operator's words: what would make a candidate "
            "the one to adopt. Empty = it asks none: every reading is exploratory, though a design is declared."
        ),
    )
    arms: list[CampaignArm] = Field(
        default_factory=list,
        description=(
            "The arms a person named, each a variant addressed from a run: the control, when one is named, first, "
            "then the candidates. Empty = none named; every variant among the runs is then read as a candidate."
        ),
    )
    aspects: list[Aspect] = Field(
        min_length=1,
        description=(
            "Where the question is measured, in the order a page lists them: each aspect's template, runs, margin and "
            "design. At least one; labels are unique, and a run belongs to one aspect at most."
        ),
    )
    floors: list[Floor] = Field(
        default_factory=list,
        description=(
            "The lines a candidate is warned for crossing, each on one aspect, in declaration order. Each floor's "
            "key is unique. Empty = none: no reading is warned of."
        ),
    )
    declared_at: str | None = Field(
        default=None,
        description=(
            "When the question, its margins and its floors were declared in full (ISO-8601), which a page discloses "
            "where it is later than the evidence. An aspect's design carries its own time "
            "(`aspects[].declared_at`). None when the campaign declares nothing."
        ),
    )
    declared_by: str = Field(
        default="", description="Who declared the question, as the declaring surface knows them; empty when unrecorded."
    )
    call: CampaignCall | None = Field(
        default=None,
        description="What a person decided over the campaign's readings. None = no call recorded yet.",
    )
    archived: bool = Field(default=False)
    created_at: str = Field(default_factory=utc_now_iso)
    created_by: str = Field(
        min_length=1,
        description=(
            "Who created the campaign, as the CREATING SURFACE knows them — `web:<username>` from a "
            "session, `mcp` from the action seam, which names the surface rather than a person "
            "because an MCP token identifies the surface. Server-owned: a caller-supplied value is "
            "discarded, because authorship a caller types is a claim rather than a record. Required "
            "and non-blank: every campaign has an author, and a later editor is not it."
        ),
    )

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_campaign":
            raise ValueError(f"doc_type must be 'eval_campaign', got '{v}'")
        return v

    @model_validator(mode="after")
    def _aspects_hold_their_runs_alone(self) -> Self:
        """Each aspect's label is unique, and each run is held once, by one aspect.

        :return: The validated campaign.
        :rtype: EvalCampaign
        :raises ValueError: Two aspects share a label, or a run is held twice — by one aspect or by two.
        """
        labels = [aspect.label for aspect in self.aspects]
        if repeated := sorted({label for label in labels if labels.count(label) > 1}):
            raise ValueError(
                f"aspect label(s) {', '.join(map(repr, repeated))} name more than one aspect; label each once"
            )
        holder: dict[str, str] = {}
        for aspect in self.aspects:
            for run_id in aspect.run_ids:
                if run_id in holder:
                    where = (
                        f"twice by aspect {aspect.label!r}"
                        if holder[run_id] == aspect.label
                        else f"by aspects {holder[run_id]!r} and {aspect.label!r}"
                    )
                    raise ValueError(f"run {run_id!r} is held {where}; a run belongs to one aspect of a campaign")
                holder[run_id] = aspect.label
        return self

    @model_validator(mode="after")
    def _floors_are_on_its_aspects(self) -> Self:
        """Every floor names one of the campaign's aspects, and no two floors share a key.

        :return: The validated campaign.
        :rtype: EvalCampaign
        :raises ValueError: A floor names no aspect of the campaign, or two floors have one key.
        """
        labels = {aspect.label for aspect in self.aspects}
        if stray := sorted({floor.aspect for floor in self.floors} - labels):
            raise ValueError(
                f"floor(s) on {', '.join(map(repr, stray))}, which the campaign has no aspect for; its aspects are "
                f"{', '.join(map(repr, sorted(labels)))}"
            )
        keys = [floor.key for floor in self.floors]
        if repeated := sorted({key for key in keys if keys.count(key) > 1}):
            raise ValueError(f"floor(s) {', '.join(repeated)} are declared more than once; declare each floor once")
        return self

    @model_validator(mode="after")
    def _one_control_first(self) -> Self:
        """At most one arm is the control and it comes first; labels and variant keys name one arm each.

        :return: The validated campaign.
        :rtype: EvalCampaign
        :raises ValueError: Two control arms, a control after a candidate, or a label or key on two arms.
        """
        controls = [index for index, arm in enumerate(self.arms) if arm.role == "control"]
        if len(controls) > 1:
            raise ValueError(f"the campaign names {len(controls)} control arms; a campaign has one control at most")
        if controls and controls[0] != 0:
            raise ValueError(f"the control arm {self.arms[controls[0]].label!r} comes after a candidate; list it first")
        for what, values in (
            ("label", [arm.label for arm in self.arms]),
            ("variant key", [arm.variant_key for arm in self.arms]),
        ):
            if repeated := sorted({value for value in values if values.count(value) > 1}):
                raise ValueError(f"arm {what}(s) {', '.join(map(repr, repeated))} name more than one arm")
        return self

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="Every run the campaign holds, aspect by aspect in order. Derived from the aspects, never stored as a choice."
    )
    @property
    def run_ids(self) -> list[str]:
        """Every run the campaign holds, its aspects' in order."""
        return [run_id for aspect in self.aspects for run_id in aspect.run_ids]

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="Whether no aspect declares a design: its readings confirm nothing. Derived, never stored as a choice."
    )
    @property
    def exploratory(self) -> bool:
        """Whether no aspect of the campaign declares a design."""
        return all(aspect.design is None for aspect in self.aspects)

    @property
    def control(self) -> CampaignArm | None:
        """The control arm, or None when the campaign names none."""
        return self.arms[0] if self.arms and self.arms[0].role == "control" else None

    def aspect(self, label: str) -> Aspect | None:
        """The aspect labelled ``label``, or None when the campaign has none.

        :param label: The aspect's label.
        :ptype label: str
        :return: The aspect.
        :rtype: Aspect | None
        """
        return next((aspect for aspect in self.aspects if aspect.label == label), None)

    def aspect_of_run(self, run_id: str) -> Aspect | None:
        """The aspect holding ``run_id``, or None when no aspect does.

        :param run_id: The run.
        :ptype run_id: str
        :return: The aspect.
        :rtype: Aspect | None
        """
        return next((aspect for aspect in self.aspects if run_id in aspect.run_ids), None)


class CampaignView(EvalDocumentModel):
    """A campaign plus its read-time-derived window — the ``campaign_get`` payload.

    Computed once by the host's eval service and serialized by every surface that
    answers ``campaign_get``, so no two of them can answer ``campaign_get`` with different facts. ``window`` is ``None`` when
    no member runs resolved (no scope supplied, or none of the run ids exist in
    it) — an honest absence, not a zero span.

    A member run gets exactly one of three fates, and all three are reported:
    resolved (it feeds the window), archived, or unresolvable. The third used to be
    a bare ``continue``, so a campaign could claim N memberships, resolve none of
    them, and print a bare ``Runs: N``.
    """

    campaign: EvalCampaign
    window: CampaignWindow | None = Field(
        default=None,
        description="Derived from member runs' created_at; None when no runs resolved.",
    )
    archived_run_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Member runs an operator archived — still attached, excluded from the window and from "
            "every cohort a report over this campaign assembles. Empty when no scope_id was "
            "supplied, since membership cannot be resolved without one."
        ),
    )
    unresolved_run_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Member run_ids that do not exist in the supplied scope — an honest gap, not dropped "
            "silently. Reported separately from archived_run_ids for the reason ContextBundle already "
            "records: unresolved is a lookup that failed, archived is a curation that succeeded, and "
            "pooling them makes a deliberate exclusion read as missing data. Empty when no scope_id "
            "was supplied, since membership cannot be resolved without one — NOT because every member "
            "resolved."
        ),
    )


def derive_window(run_created_ats: list[str]) -> CampaignWindow | None:
    """Derive a campaign's [start, end] window from its runs' ``created_at`` values.

    Pure function over the runs' ISO-8601 ``created_at`` strings: the window is the
    min (start) and max (end). All eval runs stamp ``created_at`` via
    :func:`~threetears.evals.schema.models.utc_now_iso` (UTC, ``+00:00`` offset), so a
    lexicographic min/max is a chronological one.

    :param run_created_ats: The member runs' ``created_at`` values.
    :ptype run_created_ats: list[str]
    :return: A :class:`CampaignWindow`, or ``None`` when the list is empty — a campaign
        with no (resolvable) runs has no window.
    :rtype: CampaignWindow | None
    """
    if not run_created_ats:
        return None
    return CampaignWindow(start=min(run_created_ats), end=max(run_created_ats))


# =============================================================================
# EvalAnalysis / EvalInsight — the generated, stored analysis + the
# insights it mints. Value objects are typed sub-models (not open dicts) so the
# shape is validated at the storage boundary; ``viz.payload`` and the
# ``run_index`` config/metrics maps stay open on purpose (see their fields).
# =============================================================================


class GenerationProvenance(EvalDocumentModel):
    """How an :class:`EvalAnalysis` was generated — enables reproducible prompt A/B.

    Records the prompt (id + version), generator model, and a fingerprint of the
    canonical context bundle the generation ran over, so the same bundle can be
    re-analysed under a different prompt and the two outputs compared apples-to-
    apples. ``token_cost`` is the generation cost, totalled across both calls where a
    repair ran (Visible Costs).

    ``repair_attempts`` / ``repaired_refusal`` are what keep the repair round-trip from
    being a fallback tier. An analysis that needed a repair is NOT the same
    artifact as one accepted first time, and an operator comparing generator models has to
    be able to see which needed one — a silent repair would hide exactly the defect rate
    the comparison exists to measure. A ``0``/``None`` pair reads as "accepted as emitted".
    Every field is required: the generator states each one, so none is ever assumed.
    """

    prompt_id: str = Field(description="Id of the prompt template used to generate the analysis.")
    prompt_version: str = Field(description="Version of that prompt template.")
    generator_model: str = Field(description="Model that generated the analysis, as the provider reported it.")
    bundle_fingerprint: str = Field(
        description=(
            "The fingerprint of the campaign bundle the generation ran over: `v2:` and the sha256 of its canonical "
            "JSON with every float rounded to twelve significant figures; unprefixed, recorded before the rounding."
        )
    )
    generated_at: str = Field(description="When the analysis was generated (ISO-8601).")
    bundle_assembled_at: str = Field(
        description=(
            "When the bundle this analysis was generated from was assembled (ISO-8601) — stamped by the "
            "caller immediately before assembly, so it precedes the insight-ledger read. It is the cutoff a "
            "re-assembly for this analysis reads prior insights as of: `generated_at` is stamped after the "
            "provider call (and again after a repair), so an insight another analysis minted in between "
            "would be in a rebuild cut there and was never in this generation's input."
        ),
    )
    token_cost: float = Field(
        ge=0.0,
        description=(
            "Generation token cost in dollars (Visible Costs). Where a repair round-trip ran this is "
            "the TOTAL across both calls, not the successful one alone — both were billed, and a "
            "figure naming only the survivor would under-report what the analysis cost."
        ),
    )
    repair_attempts: int = Field(
        ge=0,
        description=(
            "Soundness-repair round-trips this analysis needed: 0 when the generator's first output "
            "was accepted, 1 when it was refused and regenerated with the refusal fed back. Bounded "
            "at one attempt by the generator, deliberately NOT by this field — a stored analysis must "
            "keep loading if that bound is ever revisited."
        ),
    )
    repaired_refusal: str | None = Field(
        description=(
            "The refusal that triggered the repair, verbatim — a soundness rejection, an unparseable "
            "payload or a missing required field alike; None when nothing was "
            "repaired. Stored as the message rather than a code because it names the offending "
            "finding and the rule it broke, which is what makes a repair rate diagnosable instead of "
            "merely countable."
        ),
    )
    cell_model_version: int = Field(
        ge=1,
        description=(
            "Which definition of a cell produced this analysis' pooling — the bundle's "
            "``cell_model_version``, stamped at generation. An analysis is a claim about which "
            "observations belong together, and that claim is only legible beside the rule that grouped "
            "them — two analyses whose cells were computed under different definitions are not "
            "comparable on n, on k, or on any per-cell number."
        ),
    )
    bundle_schema_version: int = Field(
        ge=1,
        description=(
            "The ``AnalysisContextBundle.schema_version`` the generation ran over — the package's bundle shape. "
            "With ``host_declarations_digest`` it is what lets a re-assembly that no longer reproduces "
            "``bundle_fingerprint`` say whether the package's shape, the host's declarations or the evidence "
            "moved."
        ),
    )
    host_declarations_digest: str = Field(
        min_length=1,
        description=(
            "The ``AnalysisContextBundle.host_declarations_digest`` the generation ran over: a digest of the "
            "host's declared sweepables and world dimensions, derived from its registries at assembly."
        ),
    )
    user_message_digest: str = Field(
        min_length=1,
        description=(
            "sha256 of the user message the generation's FIRST request sent — the bundle as the writer read it, "
            "rendered by that build. A repair resends it with the refusal appended; that addition is "
            "`repaired_refusal`, not part of this digest. It is what lets a later rendering of the same bundle be "
            "checked against what the writer was actually sent: a reporter case freezing a recorded memo's writer "
            "message compares against it."
        ),
    )


class LeverCoverage(EvalDocumentModel):
    """Per-lever coverage summary — a point estimate is invalid without n + dispersion.

    ``n`` and ``dispersion`` are REQUIRED (no default): a point estimate rendered
    without its sample size and spread is a rendering bug, so the model
    refuses to construct one that omits them.

    It carries no confidence: how firmly a reading stands is the evidence tier on the reading itself.
    """

    name: str = Field(min_length=1, description="Lever name, as its host declares it, e.g. 'search.model'.")
    cells: int = Field(ge=0, description="Number of matrix cells measured for this lever.")
    k: int = Field(
        ge=0,
        description=(
            "The repeat FLOOR binding this lever's comparison: for each level, the `k_runs` of its "
            "best-replicated run, then the lowest of those across the lever's levels. It says how thin the "
            "lever's weakest level is, and is never any one arm's repeat count — a control run three times "
            "beside arms run once reads 1 here."
        ),
    )
    n: int = Field(ge=0, description="Total samples behind the estimate (REQUIRED — no point estimate without it).")
    dispersion: str = Field(description="Spread of the estimate (REQUIRED — no point estimate without it).")
    status: Literal["measured", "thin", "unswept"] = Field(description="Coverage status for this lever.")


class CoverageLens(EvalDocumentModel):
    """The coverage lens: per-lever measurement summaries."""

    levers: list[LeverCoverage] = Field(default_factory=list, description="Per-lever coverage summaries.")


# The caveat kinds the ENGINE owns, available to every host.
#
# Deliberately not a closed enum on the field. R8 gives the lever vocabulary to the host for the
# reason that applies here too: a closed engine-owned set on a host-facing field forces a domain
# with a legitimate fifth kind to jam it into ``scope``, and the field then stops meaning anything
# — which is the one thing a required classification exists to prevent. A host registers its own
# kinds on its profile (:attr:`~threetears.evals.kernel.host.profile.HostProfile.caveat_kinds`) and the
# union is CHECKED nowhere yet — that belongs where a profile is in hand and prompt compliance is
# judged on output, which is where the prompt is rebuilt. These four were derived from
# the first host's own caveats and are offered, not imposed.
ENGINE_CAVEAT_KINDS: frozenset[str] = frozenset({"apparatus", "sampling", "instrument", "scope"})


class EvidenceRow(EvalDocumentModel):
    """One reading at one cell, and the number code resolved it to, with its basis.

    **The model names the reading; code fills the number.** The generator authors the cell and the
    reading, and code fills ``value``, ``n`` and ``dispersion`` from the analysis's decision surface,
    so the figure a reader sees is the one computed, never a transcription of it. ``n`` and
    ``dispersion`` are REQUIRED for the reason :class:`LeverCoverage` requires them — a point estimate
    rendered without its sample size and spread is a rendering bug.

    ``aspect`` says which aspect's surface the reading was resolved on: one arm under one rig is the same cell in
    every aspect it ran in, so the ``cell_ref`` alone does not say which aspect's evidence it is.
    """

    aspect: str = Field(min_length=1, description="The label of the aspect whose surface the reading resolved on.")
    cell_ref: str = Field(
        min_length=1,
        description="The cell the reading sits at — `<variant_key>:<apparatus_class_id>`, as `cells.cell_ref` mints it.",
    )
    measure_id: str = Field(min_length=1, description="Which measure (or judged dimension) the value is of.")
    reading: ReadingKind = Field(description=_READING_DESCRIPTION)
    value: float = Field(
        description=(
            "The cell's mean, filled by code. NUMERIC deliberately: a categorical or boolean measure has no "
            "point estimate, and its evidence is a distribution, which this row cannot carry honestly."
        )
    )
    n: int = Field(ge=0, description="Observations behind the value, filled by code: a case judged k times counts k.")
    n_cases: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Distinct test cases behind the value, filled by code: the independent draws its interval is read "
            "over, and the count a reader is shown. None when the summary behind it counted no distinct case, read "
            "as not recorded (never as `n`, which counts observations)."
        ),
    )
    dispersion: str = Field(description="Spread of the value, filled by code.")
    judged_tier: JudgedEvidenceTier | None = Field(
        default=None,
        description=(
            "On a judged row, the evidence tier code resolved for the judges behind the cell's scores "
            "(`cell_measures[].judged[].evidence_tier`); None on a measure row, which needs no judge."
        ),
    )

    @model_validator(mode="after")
    def _a_judged_row_carries_its_tier(self) -> Self:
        """Refuse a judged row without a tier, or a measure row with one.

        The finding's tier is read off this field, so a judged row with none would compose as mechanical,
        and a measure row with one would weaken a finding no judge touched.

        :raises ValueError: ``judged_tier`` is absent on a judged row or present on a measure row.
        """
        if (self.reading == "judged") != (self.judged_tier is not None):
            raise ValueError(
                f"a {self.reading} row carries judged_tier={self.judged_tier!r}; a judged row carries the tier "
                "code resolved for its judges, and a measure row carries none"
            )
        return self


class RunIndexEntry(EvalDocumentModel):
    """One row of the analysis's run index — a run's config + key metrics.

    ``config`` and ``key_metrics`` are open dicts: they mirror whatever the run
    recorded, and the analysis surface must not constrain that shape.
    """

    run_id: str = Field(min_length=1, description="The indexed run's id.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The value each applicable lever actually ran at (open dict); a lever whose value could not be "
            "established is absent. Each model here belongs to a ROLE and they are not comparable with each "
            "other — `model` is the candidate under test, `search.model` the inner agent it called."
        ),
    )
    config_provenance: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "How each lever's value was established: `overridden` (the value was NAMED — by a launch overlay, "
            "by an open family restating a fixed lever, or by a fixed declaration's own reader), `inherited` "
            "(recovered from what the run observably did), or `unknown`. A reader must not describe an "
            "`inherited` OVERLAY lever as one the run chose, nor an `unknown` one as a value at all. `model` "
            "is the deliberate exception: a launch does name its candidate models, and `inherited` there "
            "records only that the value was read off the tokens the candidate actually spent."
        ),
    )
    key_metrics: dict[str, Any] = Field(default_factory=dict, description="Headline metrics for the run (open dict).")


class VariantIndexEntry(EvalDocumentModel):
    """One observed variant, and the resolved lever map its key was digested from.

    The dimension table for the cell algebra's coordinate, and the peer of
    :class:`RunIndexEntry` at the coordinate that pools. A variant key is a digest over the
    whole resolved contestant stack, so it says two observations are the same stack and
    refuses to say what that stack WAS — which leaves every join between the memo's typed
    answer (keyed by axis and level) and a cell (keyed by variant) with nothing to join on.
    This is that missing side, and it carries the FULL map rather than the campaign's declared
    axes: a lever that varied without being declared is exactly the manufactured difference an
    experiment must be able to show, and an index narrowed to the declaration could not.

    **Frozen at generation, never resolved live.** Resolving the map at read time would let
    archiving or curating a run silently change what a stored analysis says its arms were —
    the same failure :attr:`EvalAnalysis.design_snapshot` refuses one level up.

    The entry is self-verifying: the key is recomputable from the levels, so an entry that
    mislabels an arm fails loudly at construction instead of rendering a wrong one. The one
    exception declares itself — an entry carrying ``levels_unavailable`` skips the recompute
    because its key was minted by a predicate this build does not have, and it must then carry no
    levers at all, so there is nothing for the skipped check to have caught.
    """

    variant_key: str = Field(
        min_length=1,
        description="The variant this entry names — the digest every cell and control is keyed by.",
    )
    levers: dict[str, SweepableValue] = Field(
        description=(
            "Lever name → the level this variant carried, as the host resolved it. Every value is "
            "content-addressed, which is what lets an axis point and a variant be compared without "
            "either side resolving a store the other cannot see. Empty in two cases that are not the "
            "same fact: with `levels_unavailable` set, the arm's key was minted by a predicate this "
            "build cannot reproduce and nothing can say what it ran; without it, the host resolved no "
            "levers at all, which describes the arm exactly."
        ),
    )
    levels_unavailable: str | None = Field(
        default=None,
        description=(
            "Why this arm's levels cannot be described, or None when they can. An arm whose key was "
            "stamped by a predicate the current one does not reproduce was really measured and really "
            "pooled, but nothing today can say what it ran — and describing it with today's levels "
            "would label it with a stack it never carried."
        ),
    )
    swept: dict[str, SweepableValue] = Field(
        default_factory=dict,
        description=(
            "The open-family members this arm's runs overlaid — the knobs a campaign actually swept, such "
            "as `search.max_rounds` — each content-addressed from the value the launch named. Filled only "
            "where the surface those members resolve into is in `folded`: the members then say everything "
            "that surface's movement says, and they are what the arm is NAMED by. Not part of the key's "
            "pre-image, which is `levers` alone. Empty for an arm that swept no member — the control of a "
            "one-knob sweep, whose surface is then the campaign's shared residual and names nothing — and "
            "for every arm of a campaign whose surface moved on its own. A surface a fixed knob is written "
            "into adds nothing here: that knob is already one of `levers`, and names the arm itself."
        ),
    )
    folded: list[str] = Field(
        default_factory=list,
        description=(
            "Levers in `levers` that are a RESOLVED SURFACE whose movement across this campaign is "
            "explained by the knob written into it alone — for an open family, the residual left after taking "
            "every swept member back out agreed on every run; for a fixed knob (a kind overlay marked "
            "`ResolvesInto`), the surface held one level within each of the knob's levels. Such a surface is "
            "the knob's change seen a second time: never a second moved lever, never a confound, and not how "
            "the arm is named. It stays in `levers` because the key is digested from it. Absent from this list "
            "wherever that failed, or could not be read — then the surface moved on its own and names the arm "
            "like any other lever. Sorted."
        ),
    )

    @property
    def named_levers(self) -> dict[str, SweepableValue]:
        """What this arm is NAMED by — its levers with each folded surface replaced by the members it swept.

        The coordinate an arm table joins on and renders. ``levers`` is the key's pre-image and must
        keep the resolved surface, because the key is digested from it; a reader naming arms from it
        instead gets one opaque surface hash per arm and no mention of the knob a campaign swept, so
        every arm of a one-key tool-config sweep reads alike.

        :return: Lever name → level, sorted by name.
        :rtype: dict[str, SweepableValue]
        """
        named = {name: level for name, level in self.levers.items() if name not in self.folded}
        named.update(self.swept)
        return dict(sorted(named.items()))

    @model_validator(mode="after")
    def _the_levels_recompute_the_key(self) -> VariantIndexEntry:
        """Reject an entry whose lever map does not digest to the key it claims.

        The check is the whole reason to persist the map beside the key rather than the label an
        author found convenient: a wrong entry places every measurement of that arm under the
        wrong levels, and nothing downstream can tell — a table naming the wrong winner reads
        exactly like one naming the right one.

        ``levels_unavailable`` is the one way past it, and it is an admission rather than an
        exemption: an entry that declares one must carry NO levers at all. That is what keeps the
        escape narrow — the check exists to stop a map mislabelling an arm, and a caller cannot use
        this door to smuggle a half-map through, because the only map it permits is the empty one
        that labels nothing. Skipping the recompute for such an entry is not a weakening: its key
        was minted by a predicate this process does not have, so recomputing would prove nothing
        either way.

        :return: The validated entry.
        :rtype: VariantIndexEntry
        :raises ValueError: The levels digest to some other key, an entry declaring its levels
            unavailable carries levers anyway, or an entry folds a lever it does not carry.
        """
        if stranded := sorted(set(self.folded) - set(self.levers)):
            raise ValueError(
                f"variant index entry {self.variant_key} folds {stranded}, which it does not carry as levers; "
                "only a surface the arm resolved can be explained away"
            )
        if self.levels_unavailable is not None:
            if self.levers:
                raise ValueError(
                    f"variant index entry {self.variant_key} declares its levels unavailable "
                    f"({self.levels_unavailable!r}) but carries {len(self.levers)} lever(s); an entry that "
                    "cannot be described must describe nothing, or the half it does carry mislabels the arm"
                )
            return self
        recomputed = compute_variant_key(self.levers)
        if recomputed != self.variant_key:
            raise ValueError(
                f"variant index entry claims key {self.variant_key} but its levels digest to {recomputed}; "
                "an entry whose map does not recompute its own key mislabels every measurement at that arm"
            )
        return self


class FindingResolution(EvalDocumentModel):
    """What code filled for one authored finding: the numbers its readings resolved to."""

    evidence: list[EvidenceRow] = Field(
        default_factory=list, description="One resolved row per authored evidence reading, in the same order."
    )

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "What the finding's verdict stands on, read off `evidence`: `mechanical` when every row is a "
            "measure; the weakest judged row's `judged_tier` (`calibrated`, `separation`, `undetermined`, "
            "`incidental`) when any is a judged score; `none` when there are no rows. Derived on every read and "
            "never stored as a choice, so it cannot disagree with the rows it came from."
        )
    )
    @property
    def evidence_tier(self) -> EvidenceTier:
        """The finding's evidence tier, derived from its evidence rows."""
        return evidence_tier_of([row.judged_tier for row in self.evidence])


class CampaignDeclaration(EvalDocumentModel):
    """What a campaign declared, as of one generation: its question, arms, aspects and floors.

    A snapshot rather than a reference because a declaration can change after an analysis was written (an aspect
    added, a floor tightened, an arm named), and an analysis written against one declaration must not later read as
    written against another. Every campaign declares at least its aspects, so every analysis carries one; an aspect
    whose ``design`` is None declared nothing about its arms, and its memo reads as exploratory there.
    """

    question: str = Field(description="The campaign's one question, in its words; empty when it asks none.")
    declared_at: str | None = Field(description="When the question's wording was declared (ISO-8601); None = never.")
    declared_by: str | None = Field(description="Who declared it; None when nobody did.")
    arms: list[CampaignArm] = Field(description="The arms a person named, the control first.")
    aspects: list[Aspect] = Field(min_length=1, description="Every aspect, as declared, in the campaign's order.")
    floors: list[Floor] = Field(description="Every floor, on the aspect it names.")

    @classmethod
    def of(cls, campaign: EvalCampaign) -> CampaignDeclaration:
        """Snapshot what ``campaign`` declares now.

        :param campaign: The campaign.
        :ptype campaign: EvalCampaign
        :return: Its declaration.
        :rtype: CampaignDeclaration
        """
        return cls(
            question=campaign.question,
            declared_at=campaign.declared_at,
            declared_by=campaign.declared_by,
            arms=list(campaign.arms),
            aspects=list(campaign.aspects),
            floors=list(campaign.floors),
        )


class AspectAnalysis(EvalDocumentModel):
    """One aspect of an analysed campaign: the evidence bundle the memo was written over there, and what code read.

    Frozen at generation, never resolved at read: an analysis must keep saying what each aspect's arms and numbers
    WERE, whatever anyone archives or curates afterwards.
    """

    label: str = Field(min_length=1, description="The aspect's label.")
    bundle: VerbatimJsonObject = Field(
        description=(
            "The aspect's evidence bundle, frozen at generation as its canonical dump "
            "(`AnalysisContextBundle.to_dict()`): the report renders every count, verdict, chart and disclosure of "
            "this aspect from it. Read back with `AnalysisContextBundle.from_dict`."
        )
    )
    bundle_fingerprint: str = Field(min_length=1, description="The frozen bundle's fingerprint.")
    decision_surface: DecisionSurface = Field(
        description=(
            "The aspect's per-cell numbers, as code computed them, copied from its bundle at generation; every "
            "evidence value resolved on this aspect was filled from it."
        )
    )
    variant_index: list[VariantIndexEntry] = Field(
        default_factory=list,
        description=(
            "One entry per variant observed in this aspect — the coordinate table that places a cell on the declared "
            "axes. The dimension table, never the report: the arm table's rows are derived at render."
        ),
    )
    run_index: list[RunIndexEntry] = Field(default_factory=list, description="Per-run config + key metrics index.")
    coverage: CoverageLens = Field(
        default_factory=CoverageLens, description="The coverage lens: per-lever measurement summaries."
    )


class EvalAnalysis(EvalDocumentModel):
    """A generated, stored analysis of one campaign — the productized "Analysis" lens.

    Lives in its campaign's ``scope_id``, which is also the scope of every run it
    analyses. ``campaign_id`` is a *reference* to an :class:`EvalCampaign`: the model only
    requires it be non-empty — that the campaign actually EXISTS is a service
    concern checked at generate time, not a model validator.

    **One memo per campaign.** Code reports, the writer recommends, a person decides: ``document`` is the writer's one
    memo over every aspect — findings across them, one recommendation, a short summary per aspect and next steps
    (:class:`~threetears.evals.kernel.authored.AuthoredCampaignAnalysis`) — kept as written, except that each figure
    reference in its prose is replaced by the value code read (:mod:`threetears.evals.analysis.prose_refs`).
    **Everything else code filled sits beside it**: ``resolutions`` holds, per finding by position, the numbers its
    readings resolved to (each row naming its aspect); ``aspects`` holds each aspect's frozen bundle and what code read
    off it; and ``campaign_bundle`` what the campaign adds over its aspects — its question, arms, floors and code's
    readings across aspects. Links inside the document are positions, and :meth:`check_positions` refuses one that
    points nowhere or makes the ``invalidates`` relation cycle.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_analysis"] = "eval_analysis"
    schema_version: SchemaVersion = REGENERABLE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    campaign_id: str = Field(
        min_length=1,
        description="Reference → EvalCampaign.id — non-empty here; existence is checked by the service at generate.",
    )
    subject_id: str = Field(min_length=1, description="Reference to the analysed subject.")
    subject_kind: str = Field(
        description=(
            "Discriminator; free string, data not a code branch. REQUIRED and un-defaulted: a default is a host "
            "noun a second consumer would inherit without ever choosing it. '' is a real, reachable value meaning "
            "the campaign declared no kind — which is why it cannot also be what an omission produces."
        ),
    )
    behavior: str = Field(
        min_length=1,
        description="The behaviour under analysis, e.g. 'tool selection' — the spine's own word for it.",
    )
    observation_refs: list[str] = Field(
        default_factory=list,
        description="Observations this analysis summarises: every aspect's runs.",
    )
    model_versions: dict[str, str] = Field(default_factory=dict, description="Model ids in play at analysis time.")
    generation: GenerationProvenance = Field(description="How the analysis was generated (provenance).")
    design_snapshot: CampaignDeclaration = Field(
        description=(
            "What the campaign DECLARED, as of this generation: its question, arms, aspects and floors. A snapshot "
            "rather than a reference because a declaration can change after an analysis was written, and an "
            "analysis written against one declaration must not later read as written against another."
        ),
    )
    document: AuthoredCampaignAnalysis = Field(
        description="What the generator authored, as written, with each figure reference in its prose rendered by code."
    )
    recommended_variant_key: str | None = Field(
        description=(
            "The variant key of the arm the memo recommends, resolved by code from the recommendation's label when the "
            "analysis was saved, so a person's call is matched to the arm by its key whatever later relabels it. None "
            "where the memo recommends nothing (a measurement, or a choice with no clear winner)."
        )
    )
    judged_tier_rule: JudgedTierRule = Field(
        default="interval_lower_bound",
        description=(
            "The rule the judged evidence tiers in `resolutions` and each aspect's `decision_surface` were decided "
            "by. `interval_lower_bound`: each criterion on confidence bounds for its agreement against the bar."
        ),
    )
    resolutions: list[FindingResolution] = Field(
        default_factory=list,
        description="What code filled for each finding, by position in `document.findings`: its numbers.",
    )
    aspects: list[AspectAnalysis] = Field(
        min_length=1, description="Each aspect's frozen evidence and what code read off it, in the campaign's order."
    )
    campaign_bundle: VerbatimJsonObject = Field(
        description=(
            "What the campaign adds over its aspects' bundles, frozen at generation (`CampaignBundle.frame()`): its "
            "question, arms, floors and code's readings across every aspect, and each aspect's bundle fingerprint. "
            "Read back, with `aspects`' bundles, by `CampaignBundle.from_frame`. "
            "`generation.bundle_fingerprint` is the fingerprint of the whole."
        )
    )
    archived: bool = Field(default=False)
    archived_reason: str | None = Field(
        default=None,
        description=(
            "Why this analysis was archived, in the operator's own words — None when it is live, and "
            "None again the moment it is un-archived, since a reason surviving the restore describes a "
            "state the document is no longer in. An archive with no reason is still an archive: the "
            "field is optional so a routine retirement (superseded, duplicated) is not forced to invent "
            "prose, which would dilute the entries that carry real evidence. It exists because an "
            "analysis is archived precisely when it was shown FALSE, and the record of that is the "
            "evidence a closed loop counts — an `archived` flag alone says a human acted and refuses to "
            "say why, which is the state that made a wrong analysis indistinguishable from a stale one."
        ),
    )
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_analysis":
            raise ValueError(f"doc_type must be 'eval_analysis', got '{v}'")
        return v

    def aspect(self, label: str) -> AspectAnalysis:
        """The aspect of this analysis with ``label``.

        :param label: The aspect's label.
        :ptype label: str
        :return: The aspect.
        :rtype: AspectAnalysis
        :raises KeyError: No aspect of this analysis has that label.
        """
        for aspect in self.aspects:
            if aspect.label == label:
                return aspect
        raise KeyError(f"analysis {self.id} has no aspect {label!r}; its aspects are {[a.label for a in self.aspects]}")

    @model_validator(mode="after")
    def check_positions(self) -> EvalAnalysis:
        """Refuse a link that points nowhere, a resolution that does not match its finding, or a gating cycle.

        Also refuses aspects that are not one each, a resolved row or an aspect summary naming an aspect the analysis
        does not hold, and an aspect summarised twice.

        :return: This analysis, unchanged.
        :rtype: EvalAnalysis
        :raises ValueError: A position is out of range, a finding invalidates itself, the ``invalidates``
            relation cycles, ``resolutions`` does not have one entry per finding, a resolution's
            rows are not the readings its finding names, or an aspect is unknown or repeated.
        """
        labels = [aspect.label for aspect in self.aspects]
        if repeated := sorted({label for label in labels if labels.count(label) > 1}):
            raise ValueError(f"aspects {repeated} appear more than once; one entry per aspect")
        findings = self.document.findings
        count = len(findings)
        if self.resolutions and len(self.resolutions) != count:
            raise ValueError(f"resolutions has {len(self.resolutions)} entries for {count} findings; one per finding")
        # A finding's evidence tier is read off its resolved rows, so the rows must be the readings the
        # finding names — one per reading, in order, of the same kind — or the tier describes other evidence.
        for index, (finding, resolution) in enumerate(zip(findings, self.resolutions, strict=False)):
            named = [(ref.cell, ref.measure_id, ref.reading) for ref in finding.evidence]
            resolved = [(row.cell_ref, row.measure_id, row.reading) for row in resolution.evidence]
            if named != resolved:
                raise ValueError(
                    f"resolutions[{index}].evidence resolves {resolved} and findings[{index}] names {named}; "
                    "one resolved row per named reading, in order"
                )
            if stray := sorted({row.aspect for row in resolution.evidence} - set(labels)):
                raise ValueError(f"resolutions[{index}].evidence names aspect(s) {stray}, which this analysis lacks")
        summarised = [summary.aspect for summary in self.document.aspect_summaries]
        if twice := sorted({label for label in summarised if summarised.count(label) > 1}):
            raise ValueError(f"aspect_summaries summarise {twice} more than once; one summary per aspect")
        if unknown := sorted(set(summarised) - set(labels)):
            raise ValueError(f"aspect_summaries name {unknown}, which this analysis's aspects ({labels}) do not hold")

        def in_range(where: str, positions: list[int]) -> None:
            bad = [position for position in positions if not 0 <= position < count]
            if bad:
                raise ValueError(
                    f"{where} names finding position(s) {bad}, and there are {count} findings (0-{count - 1})"
                )

        recommended = self.document.recommended()
        if recommended is not None:
            in_range("recommendation.rests_on", recommended.rests_on)
        if (recommended is None) != (self.recommended_variant_key is None):
            raise ValueError(
                "recommended_variant_key is the key of the arm the memo recommends: set exactly when the memo makes a "
                "recommendation"
            )
        for index, finding in enumerate(findings):
            in_range(f"findings[{index}].invalidates", finding.invalidates)
            if index in finding.invalidates:
                raise ValueError(f"findings[{index}] invalidates itself")
        # A gating cycle has no reading order: each finding would have to be read before the other.
        state: dict[int, int] = {}

        def visit(node: int, trail: list[int]) -> None:
            state[node] = 1
            for nxt in findings[node].invalidates:
                if state.get(nxt) == 1:
                    raise ValueError(f"findings invalidate each other in a cycle: {[*trail, node, nxt]}")
                if state.get(nxt) is None:
                    visit(nxt, [*trail, node])
            state[node] = 2

        for node in range(count):
            if state.get(node) is None:
                visit(node, [])
        return self


#: Where one generation attempt stands. ``running``: started and not yet ended — the record a background
#: generation writes as it starts, so its job is answerable from the store. How one ended: ``stored``: an
#: analysis was produced and persisted.
#: ``refused``: the generator finished and its output was refused, and so was the repair's.
#: ``failed``: anything else that ended the attempt with an error — a call the provider cut short,
#: a precondition refused before any call, a provider call that raised, a failed write.
#: ``cancelled``: the attempt was cancelled mid-flight, by a caller's budget or disconnect, or by a shutdown —
#: or it was lost (its ``lost`` says so: the process running it ended or shut down).
AttemptOutcome = Literal["running", "stored", "refused", "failed", "cancelled", "timed_out"]


class EvalAnalysisAttempt(CoreDocumentModel):
    """One run of the analysis generator for a campaign, whatever it came to.

    A stored :class:`EvalAnalysis` records only a SUCCESS, so a generation that was refused, cut
    short or cancelled left nothing durable, and its spend was visible only in a log line. That made
    the question a writer comparison asks — how often does each writer fail, and what does a failure
    cost — unanswerable from stored data. One of these is written for every generation, success
    included, so a campaign's attempts are the whole denominator in one listing: the stored ones
    point at their analysis, the rest carry what they spent and why they ended.

    Its own record rather than a field of :class:`EvalAnalysis`, because a failure has no analysis to
    hang a field on, and rather than a failed-only record, because a failure rate needs the
    successes counted in the same list.

    A generation started as a background job writes its attempt as it starts (``running``, carrying the
    job's lease) and again as it ends, so the job is answered from this record in any process; one run
    inline writes it once, as it ends.

    **Kept** (:mod:`threetears.evals.schema.versioning`): an attempt records what a generation spent and why it
    ended, and no later generation can say what an earlier one did, so a later release reads it rather than
    refusing it. The analysis a stored attempt points at is regenerable; the attempt outlives it.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_analysis_attempt"] = "eval_analysis_attempt"
    schema_version: CoreSchemaVersion = CORE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    campaign_id: str = Field(
        min_length=1, description="Reference → EvalCampaign.id — the campaign the generation was over."
    )
    outcome: AttemptOutcome = Field(
        description=(
            "Where the attempt stands: running (started, not ended), or how it ended: stored (an analysis was "
            "persisted), refused (the output was refused and so was the repair's), failed (a call cut short, a "
            "precondition, a provider error or a failed write), cancelled (a caller's disconnect or a shutdown ended "
            "it mid-flight; or it was lost, which `lost` records), or timed_out (its own wall-clock budget ran out "
            "mid-flight)."
        ),
    )
    analysis_id: str | None = Field(
        description=(
            "The analysis this attempt stored — set exactly when the outcome is `stored`. Required though nullable: "
            "its absence is not 'not recorded' but a disagreement with the outcome, so a stored attempt always "
            "carries it."
        ),
    )
    generator_model: str = Field(
        min_length=1,
        description=(
            "The generator model the attempt was built for — the writer being compared. What the provider "
            "reported each call ran on is `reported_models`."
        ),
    )
    reported_models: list[str] = Field(
        default_factory=list,
        description="The model each call that returned reported, in call order; a provider may route calls differently.",
    )
    prompt_id: str = Field(min_length=1, description="The registry key the generator prompt was resolved under.")
    prompt_version: str | None = Field(
        default=None,
        description=(
            "Version of the assembled prompt, as a stored analysis records it. None only when the attempt ended "
            "before the prompt was assembled — a precondition refused before any call."
        ),
    )
    bundle_fingerprint: str = Field(
        description=(
            "The fingerprint of the campaign bundle the attempt ran over: `v2:` and the sha256 of its canonical JSON "
            "with every float rounded to twelve significant figures; unprefixed, recorded before the rounding."
        )
    )
    calls: int = Field(
        default=0,
        ge=0,
        description=(
            "Provider calls sent: 0 when a precondition refused first, 1 for the generation, 2 when a refusal "
            "bought its one repair. Counted when a call is SENT, so a call that was cancelled or raised counts."
        ),
    )
    unpriced_calls: int = Field(
        default=0,
        ge=0,
        description=(
            "Calls whose cost was never observed: cancelled or raised before a result came back, or returned with "
            "no reported cost. Each may still have been billed, so `token_cost` is a floor whenever this is non-zero."
        ),
    )
    token_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Dollars the provider reported across every call that returned — the total, not the last call's.",
    )
    refusals: list[str] = Field(
        default_factory=list,
        description=(
            "Every refusal of a finished call's output, as code raised it and in order: the first call's, then the "
            "repair's. A cell is named by its full cell_ref, never by the short alias the writer was shown, because an "
            "alias is minted per bundle and this record keeps only the bundle's fingerprint. Empty when no output was "
            "refused."
        ),
    )
    error: str | None = Field(
        description=(
            "What the caller was told ended the attempt — set exactly when it has ended other than `stored`. "
            "Required though nullable, for the reason `analysis_id` is."
        ),
    )
    started_at: str = Field(description="When the generation was started (ISO-8601).")
    created_at: str = Field(
        default_factory=utc_now_iso,
        description="When the attempt was last recorded (ISO-8601): as it started while running, as it ended after.",
    )
    lease: LeaseField = None
    cancel_requested_at: CancelRequestedAtField = None
    cancel_request_reason: CancelRequestReasonField = None
    lost: LostField = None
    faults: FaultsField = Field(default_factory=list)
    refusal: RefusalField = None

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_analysis_attempt":
            raise ValueError(f"doc_type must be 'eval_analysis_attempt', got '{v}'")
        return v

    @model_validator(mode="after")
    def check_outcome_fields(self) -> EvalAnalysisAttempt:
        """Refuse an attempt whose analysis link or error disagrees with its outcome.

        :return: This attempt, unchanged.
        :rtype: EvalAnalysisAttempt
        :raises ValueError: A stored attempt names no analysis or carries an error, an ended unstored one names an
            analysis or carries no error, a running one names either, or a refused one records no refusal.
        """
        if self.outcome == "running":
            if self.analysis_id is not None or self.error is not None:
                raise ValueError("a running attempt names no analysis and carries no error: it has not ended")
            return self
        stored = self.outcome == "stored"
        if stored != (self.analysis_id is not None):
            raise ValueError(f"analysis_id is set exactly when the outcome is 'stored'; outcome={self.outcome!r}")
        if stored == (self.error is not None):
            raise ValueError(f"error is set exactly when the outcome is not 'stored'; outcome={self.outcome!r}")
        if self.outcome == "refused" and not self.refusals:
            raise ValueError("a refused attempt records the refusals that ended it")
        return self


def generation_slot_id(campaign_id: str) -> str:
    """The id of a campaign's generation slot: one per campaign, so every process racing a start names one document."""
    return f"generation-slot:{campaign_id}"


class EvalGenerationSlot(CoreDocumentModel):
    """Which analysis attempt holds a campaign's one generation: the document two starts race to claim.

    One per campaign and scope, at :func:`generation_slot_id`, so a start in any process writes the same document:
    created with a create-only write when the campaign has none, and taken over with ``if_match`` from an attempt
    that has ended. Of two processes racing a start exactly one write lands; the other re-reads, finds the winner's
    attempt live, and is refused naming it. Never released — the attempt's own record says whether it still runs
    (its lease), so a process that dies holding the slot frees it when its lease lapses.

    **Kept** (:mod:`threetears.evals.schema.versioning`): every build must read the slot. One a later build
    refused would block every generation of its campaign, and one it read as free would let a second generation
    start while an older process, mid rolling upgrade, still holds it live. So it is upgraded, never dropped.
    """

    id: str = Field(min_length=1, description="generation_slot_id(campaign_id): one slot per campaign.")
    doc_type: Literal["eval_generation_slot"] = "eval_generation_slot"
    schema_version: CoreSchemaVersion = CORE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    campaign_id: str = Field(min_length=1, description="The campaign whose generations the slot admits one at a time.")
    attempt_id: str = Field(
        min_length=1, description="The attempt that last claimed the slot (EvalAnalysisAttempt.id)."
    )
    claimed_at: str = Field(default_factory=utc_now_iso, description="When it claimed it (ISO-8601).")

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_generation_slot":
            raise ValueError(f"doc_type must be 'eval_generation_slot', got '{v}'")
        return v

    @model_validator(mode="after")
    def check_slot_id(self) -> EvalGenerationSlot:
        """Refuse a slot stored anywhere but its campaign's one id.

        :raises ValueError: ``id`` is not ``generation_slot_id(campaign_id)``.
        """
        if self.id != generation_slot_id(self.campaign_id):
            raise ValueError(f"a generation slot's id is {generation_slot_id(self.campaign_id)!r}; got {self.id!r}")
        return self


#: Where a sweep stands: ``running`` while its arms are being launched and run; ``completed`` — every arm ran;
#: ``failed`` — an arm was refused at launch, or its run ended ``failed``, so no further arm was launched;
#: ``cancelled`` — a cancel stopped it, cancelling the arms in flight and launching none after; or it was lost
#: (its ``lost`` says so), its process gone with no further arm launched; ``timed_out`` — its own wall-clock budget
#: ran out, stopping the arms in flight and launching none after, as every job kind's budget reads.
SweepOutcome = Literal["running", "completed", "failed", "cancelled", "timed_out"]


class SweepArmRecord(EvalDocumentModel, CoreShapeModel):
    """One arm of a sweep as its record carries it: what it is called, its model, and its run once launched."""

    label: str = Field(min_length=1, description="The arm's name, unique within the sweep.")
    model: str = Field(min_length=1, description="The arm's candidate model.")
    run_id: str | None = Field(default=None, description="The arm's run, once launched; None before, or never.")


class EvalSweep(CoreDocumentModel):
    """A multi-arm launch run arm after arm: its arms, its campaign, and how it ended — the record its job reads.

    Written when the sweep starts, again as each arm's run is created (the run is a member of :attr:`campaign_id`
    from then), and when it ends — by the sweep's own task, conditionally, so a record another process has
    written lost or asked to cancel is never overwritten. It carries the job's lease: a sweep whose record reads
    ``running`` after its lease lapsed was ended by its process going away, and reads lost; its launched runs are
    campaign members either way.

    **Kept** (:mod:`threetears.evals.schema.versioning`): how a sweep ended — which arm failed, what cancelled
    it — is recorded nowhere else, so a later release reads it rather than refusing it.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_sweep"] = "eval_sweep"
    schema_version: CoreSchemaVersion = CORE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    campaign_id: str = Field(min_length=1, description="The campaign every arm's run joins as it is created.")
    template_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    arms: list[SweepArmRecord] = Field(min_length=1, description="Every arm, in launch order.")
    max_concurrent_arms: int = Field(ge=1, description="How many arms run at once; 1 runs them one after another.")
    arm_schedule: ArmSchedule | None = Field(
        default=None,
        description="How the sweep's arms shared time (`ArmSchedule`), as each arm's run records it too; None for a "
        "sweep stored before the field existed.",
    )
    outcome: SweepOutcome = "running"
    detail: str | None = Field(
        default=None,
        description=(
            "Why it ended other than completed; on a completed sweep, each arm that measured nothing (its run failed "
            "alone, and the sweep went on), or None."
        ),
    )
    created_at: str = Field(default_factory=utc_now_iso)
    ended_at: str | None = None
    lease: LeaseField = None
    cancel_requested_at: CancelRequestedAtField = None
    cancel_request_reason: CancelRequestReasonField = None
    lost: LostField = None
    faults: FaultsField = Field(default_factory=list)
    refusal: RefusalField = None

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_sweep":
            raise ValueError(f"doc_type must be 'eval_sweep', got '{v}'")
        return v


class EvalInsight(EvalDocumentModel):
    """A durable, subject-scoped insight extracted from an analysis.

    Lives in the ``scope_id`` of the analysis that minted it. Traces back to the analysis that minted it
    via ``source_campaign_id`` / ``source_analysis_id``; ``invalidation_trigger``
    states the two rules that retire it — its analysis archived, or a later analysis restating it
    (:func:`~threetears.evals.analysis.bundle.insights.superseding_insights`).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_insight"] = "eval_insight"
    schema_version: SchemaVersion = REGENERABLE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    subject_id: str = Field(min_length=1, description="Reference to the subject the insight is about.")
    subject_kind: str = Field(
        description=(
            "Discriminator; free string, data not a code branch. REQUIRED and un-defaulted for the reason "
            "EvalAnalysis.subject_kind is: '' means the source campaign declared no kind, so an omission must not "
            "be able to produce the same value."
        ),
    )
    statement: ModelProse = Field(min_length=1, description="The insight itself.")
    scope: ModelProse = Field(default="", description="Where the insight applies (free text).")
    confidence: ConfidenceTier = Field(description=_CONFIDENCE_TIER_DESCRIPTION)
    evidence_run_ids: list[str] = Field(default_factory=list, description="Run ids the insight rests on.")
    evidence_result_ids: list[str] = Field(default_factory=list, description="Result ids the insight rests on.")
    observed_at: str = Field(default_factory=utc_now_iso)
    model_versions: dict[str, str] = Field(default_factory=dict, description="Model ids in play when observed.")
    invalidation_trigger: ModelProse = Field(
        min_length=1,
        description=(
            "What retires the insight, as the engine carries it out: written at mint, naming the analysis whose "
            "archive retracts it and that a later analysis of the subject restating the claim replaces it in "
            "place."
        ),
    )
    source_campaign_id: str = Field(default="", description="Campaign that produced the analysis this came from.")
    source_analysis_id: str = Field(default="", description="Analysis that minted this insight.")

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_insight":
            raise ValueError(f"doc_type must be 'eval_insight', got '{v}'")
        return v


__all__ = [
    "ENGINE_CAVEAT_KINDS",
    "AttemptOutcome",
    "CampaignView",
    "CampaignWindow",
    "CoverageLens",
    "AspectAnalysis",
    "CampaignDeclaration",
    "EvalAnalysis",
    "EvalAnalysisAttempt",
    "EvalCampaign",
    "EvalInsight",
    "EvalSweep",
    "EvidenceRow",
    "EvidenceTier",
    "FindingResolution",
    "GenerationProvenance",
    "LeverCoverage",
    "RunIndexEntry",
    "SweepArmRecord",
    "SweepOutcome",
    "VariantIndexEntry",
    "derive_window",
    "evidence_tier_of",
]
