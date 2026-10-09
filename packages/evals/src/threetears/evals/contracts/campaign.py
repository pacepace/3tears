"""Data models for the eval analysis subsystem.

:class:`EvalCampaign` is the hub — a first-class stored entity that groups eval
runs under one subject×behavior, so a generated ``EvalAnalysis`` has something to
attach to. Subject and Behavior stay *references* (an existing host subject, an
``eval_template``) rather than being reified into new tables; the campaign is the only new hub.
There is no battery pointer: the case set a run froze already enters its apparatus class, so a
second reference to "the scenario suite" would be a second answer to one question.

All three models mirror the shape of the sibling stored eval definitions
(:class:`~threetears.evals.contracts.models.JudgeConfig` /
:class:`~threetears.evals.contracts.models.CatalogRubricDim`): the strict document base,
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
from typing import Annotated, Any, ClassVar, Literal, Self, TypeVar, get_args

from pydantic import BeforeValidator, Field, ValidationInfo, computed_field, field_validator, model_validator

from threetears.evals.contracts.authored import AuthoredAnalysis, Confidence
from threetears.evals.contracts.declaration import CampaignDesign
from threetears.evals.contracts.evidence_tiers import JudgedEvidenceTier, JudgedTierRule, weakest_judged_tier
from threetears.evals.contracts.host.values import SweepableValue
from threetears.evals.contracts.identity import compute_variant_key
from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.models import EVAL_SCHEMA_VERSION, SchemaVersion, utc_now_iso
from threetears.evals.contracts.prose import ModelProse
from threetears.evals.contracts.surface import DecisionSurface
from threetears.observe import get_logger

log = get_logger(__name__)

# An optional prose field a stored INSIGHT carries, read with ``null`` as "nothing to say".
#
# Insights are kept across the v4 -> v5 analysis drop, and ones minted before it were built from a
# generator that could emit ``null`` for an optional string; this keeps them loadable. New insights
# are minted by code from a finding's ``durable`` claim and never carry a ``null``.
_LLMProse = Annotated[ModelProse, BeforeValidator(lambda v: "" if v is None else v)]

# The list sibling of ``_LLMProse``, for the optional arrays a stored insight carries.
#
# Same reason: an insight minted before v5 may hold an explicit ``null`` array, which
# ``default_factory=list`` does not cover (it supplies a default only when the key is ABSENT).
# GENERIC over the element type, so the element constraint survives the tolerance. The coercion
# WARNS with the field's name, because a normalized ``null`` reads exactly like a genuine empty list.
_T = TypeVar("_T")


# A module-level constant so the test can assert against the template rather than the emitted text.
# Do not read the guard as making a bad reword impossible — that claim was made twice here and was
# false both times. Pinning the emitted line against this constant catches no REWORD on its own
# (the test imports it, so both sides move together) — it is there to catch call-site drift, such
# as a log call that stops interpolating the field name. The token check that does the rest is a
# blocklist, so it only covers phrasings someone has thought of. A wording that avoids every listed
# token can still make a false path claim and stay green. What is mechanised is the shipped mistakes
# and their near neighbours; the rest is a human read at review time.
_NULL_LIST_WARNING = (
    "eval analysis: LLM-filled array %r arrived as null and was read as empty — the field will "
    "read as one the generator had nothing for. Tolerated so a paid analysis is neither discarded "
    "nor left unopenable; not a normal emission."
)


def _coerce_null_list(value: Any, info: ValidationInfo) -> Any:
    """Normalize an explicit ``null`` array to ``[]``, leaving anything else untouched."""
    if value is None:
        # Names the FIELD, which is what lets an operator tell which array to distrust.
        log.warning(_NULL_LIST_WARNING, info.field_name)
        return []
    return value


_LLMList = Annotated[list[_T], BeforeValidator(_coerce_null_list)]

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
#: The SAME declaration the writer authors against (:data:`~threetears.evals.contracts.authored.Confidence`),
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
#:   (:mod:`threetears.evals.contracts.evidence_tiers` defines each and the order they compose in);
#: - ``none``: the finding names no reading, so there is nothing for a tier to stand on.
#:
#: A judged tier is never read off a writer's claim: each judged row carries the tier code resolved
#: for its cell's judges (``EvidenceRow.judged_tier``).
EvidenceTier = Literal["mechanical", "calibrated", "separation", "undetermined", "incidental", "none"]


def evidence_tier_of(rows: list[JudgedEvidenceTier | None]) -> EvidenceTier:
    """The tier a verdict resting on these evidence rows stands on: the weakest among them.

    Args:
        rows: Per evidence row, in any order: ``None`` for a measure, the row's judged tier for a judged score.

    Returns:
        ``none`` for no rows; the weakest judged tier when any row is judged; otherwise ``mechanical``.
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


class EvalCampaign(EvalDocumentModel):
    """A curated set of eval runs under one subject×behavior — the analysis hub.

    Lives in one ``scope_id``, and so do its member runs: a run in another scope is
    refused at attachment (:mod:`threetears.evals.analysis.campaigns`), because every
    read of a campaign's members is a read within one scope. Membership (``run_ids``)
    is curated, not queried: a run may belong to more than one campaign, and a campaign
    holds exactly the runs an operator attached to it.

    Subject and Behavior are *references*, not FK-enforced fields:
    ``subject_id`` points at an existing subject the host owns and
    ``template_id`` at an existing ``eval_template``. ``subject_kind`` is a discriminator (free string) —
    data the render/grouping layer keys on, never a code branch. **It has
    no default kind.** A blank means this campaign declared none; defaulting it to
    whichever kind the host happens to evaluate most would make every other
    consumer's campaign silently claim a kind nobody chose, which is the same
    dishonesty as branching on it.

    **Membership alone does not describe an experiment, which is what
    ``declared_design`` is for.** A bag of runs cannot say whether each cell moved
    one lever off a shared reference or whether the sweep was factorial, so an
    analysis reading it has to guess — and guessing factorial over a star design
    pools cells that share no lever, which manufactures a difference inside the
    comparison arm.

    **The control lives on the declaration and is a VARIANT, not a run.** It was
    ``control_run_id`` here for the whole of its previous life, and the cost was a
    design that any curation could destroy: archive the control run and the campaign
    stopped being an experiment, even where three other runs carried the identical
    contestant stack. ``declared_design.control`` is a variant key, so a control
    resolves through whichever observation carries it — and nothing on this model is
    a pointer at a run whose deletion could dangle it.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_campaign"] = "eval_campaign"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
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
    behavior: str = Field(min_length=1, description="Which aspect is under test, e.g. 'tool selection'.")
    template_id: str | None = Field(default=None, description="Reference to an existing eval_template (the Behavior).")
    run_ids: list[str] = Field(
        default_factory=list,
        description="Curated membership; a run may belong to more than one campaign.",
    )
    declared_design: CampaignDesign | None = Field(
        default=None,
        description=(
            "What this campaign SET OUT to learn — operator input, never derived. The realized "
            "design lives on the bundle and is derived from observations; the delta between the two "
            "is the coverage story, which is why they are two fields and not one. A declared value "
            "with no observations is `unswept` AND NAMED, where an inferred design can only report "
            "what happened to run.\n\n"
            "**Nullable, and that is a real state:** a campaign may legitimately never declare one. "
            "Such a campaign gets a realized design only, the memo says so in one line, and the "
            "axis-completeness validator degrades from a gate to a warning — exploratory campaigns "
            "that declared nothing are still analysable."
        ),
    )
    status: Literal["open", "closed"] = Field(default="open")
    archived: bool = Field(default=False)
    created_at: str = Field(default_factory=utc_now_iso)
    created_by: str = Field(
        min_length=1,
        description=(
            "Who created the campaign, as the CREATING SURFACE knows them — `web:<username>` from a "
            "session, `mcp` from the action seam, which names the surface rather than a person "
            "because an MCP token identifies the surface. Server-owned: a caller-supplied value is "
            "discarded, because authorship a caller types is a claim rather than a record, and this "
            "field spent its whole life advertised as an optional key and written by nobody. Required "
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
    :func:`~threetears.evals.contracts.models.utc_now_iso` (UTC, ``+00:00`` offset), so a
    lexicographic min/max is a chronological one.

    Args:
        run_created_ats: The member runs' ``created_at`` values.

    Returns:
        A :class:`CampaignWindow`, or ``None`` when the list is empty — a campaign
        with no (resolvable) runs has no window.
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
    bundle_fingerprint: str = Field(description="sha256 of the canonical context bundle the generation ran over.")
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

    It carries no confidence. ``confidence`` was retired within schema v8: it was a fixed lookup on
    ``status`` (measured → high, thin → medium, unswept → low), so it said nothing ``status`` does not,
    while reading as a confidence in the lever's estimate — an unswept lever, which nothing measured, read
    as "low confidence" in a measurement that does not exist. How firmly a reading stands is the evidence
    tier on the reading itself. A stored analysis carrying the key reads with it discarded.
    """

    __retired_fields__: ClassVar[dict[str, str | None]] = {"confidence": None}

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


#: Every visualization a stored finding can carry.
VizType = Literal[
    "delta_table",
    "frontier",
    "timeseries",
    "distribution",
    "null_result",
    "breakdown",
    "attribution",
    "sweep_ranking",
]


class Viz(EvalDocumentModel):
    """A visualization spec attached to a finding.

    **The model chooses; code reads and compiles.** The generator authors a chart's type, cells
    and measures; code reads them into the type's reference (``ref``) and builds ``payload`` from
    it, so every number a chart draws was computed. ``payload`` is the stored, rendered shape — a
    reader never needs the reference to draw a chart.

    ``payload`` is deliberately an OPEN dict on the model: a new viz type — or an
    extra payload field — must never break the persisted schema or fail a stored
    read, so the model is not the enforcement point. The per-``type`` payload SHAPE
    is a reader-side contract. CODE produces every stored payload —
    :func:`threetears.evals.analysis.viz_refs.build_viz_payload` compiles it from ``ref``. The
    lockstep set is therefore the ``viz_refs`` chart builders,
    :data:`threetears.evals.analysis.viz.PAYLOAD_MODELS` and any frontend
    client's mirror of the discriminated union. Change one and change the others.

    Types listed in :data:`threetears.evals.analysis.viz.PAYLOAD_MODELS` additionally have a
    typed payload model enforced at GENERATION time, which is where a malformed
    payload can still be fixed by retrying rather than by paying for another
    analysis. Openness here and strictness there are not in tension: this model
    guarantees a stored analysis always loads, and that one guarantees a new one
    is never stored malformed.
    """

    type: VizType = Field(description="Which visualization renders this finding.")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Viz-specific data — open dict; the per-type shape is a reader contract (see the class docstring), "
            "not enforced here. COMPILED BY CODE from `ref`."
        ),
    )
    ref: dict[str, Any] = Field(
        description=(
            "The chart type's reference, as code read it off the authored chart: the cells and readings the "
            "chart draws, plus the model's prose (caption, a null result's mechanism) — never a number. Code "
            "compiles `payload` from it against the decision surface. Open here for the reason `payload` is; "
            "the per-type reference shape is built by code at generation."
        ),
    )


# The caveat kinds the ENGINE owns, available to every host.
#
# Deliberately not a closed enum on the field. R8 gives the lever vocabulary to the host for the
# reason that applies here too: a closed engine-owned set on a host-facing field forces a domain
# with a legitimate fifth kind to jam it into ``scope``, and the field then stops meaning anything
# — which is the one thing a required classification exists to prevent. A host registers its own
# kinds on its profile (:attr:`~threetears.evals.contracts.host.profile.HostProfile.caveat_kinds`) and the
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
    """

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
    n: int = Field(ge=0, description="Samples behind the value, filled by code.")
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

        Raises:
            ValueError: ``judged_tier`` is absent on a judged row or present on a measure row.
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

        Returns:
            Lever name → level, sorted by name.
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

        Returns:
            The validated entry.

        Raises:
            ValueError: The levels digest to some other key, an entry declaring its levels
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
    """What code filled for one authored finding: the numbers its readings resolved to, and its chart."""

    evidence: list[EvidenceRow] = Field(
        default_factory=list, description="One resolved row per authored evidence reading, in the same order."
    )
    chart: Viz | None = Field(
        default=None, description="The chart compiled from the finding's authored chart; None when it drew none."
    )
    chart_note: str = Field(
        default="", description="Why a proposed chart was dropped as undrawable; empty when none was dropped."
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


class EvalAnalysis(EvalDocumentModel):
    """A generated, stored analysis of one campaign — the productized "Analysis" lens.

    Lives in its campaign's ``scope_id``, which is also the scope of every run it
    analyses. ``campaign_id`` is a *reference* to an :class:`EvalCampaign`: the model only
    requires it be non-empty — that the campaign actually EXISTS is a service
    concern checked at generate time, not a model validator.

    **What the generator authored is kept as written in** ``document`` — the headline, the summary,
    the findings, decisions, question answers and next steps, in the shape it was asked for
    (:mod:`threetears.evals.contracts.authored`) — except that each figure reference in its prose is
    replaced by the value code read (:mod:`threetears.evals.analysis.prose_refs`). **Everything else
    code filled sits beside it**: ``resolutions`` holds, per finding by position, the numbers its readings resolved to and the
    chart it compiled to. Links inside the document are positions, and :meth:`check_positions`
    refuses one that points nowhere or makes the ``invalidates`` relation cycle.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_analysis"] = "eval_analysis"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
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
        description="Observations this analysis summarises.",
    )
    model_versions: dict[str, str] = Field(default_factory=dict, description="Model ids in play at analysis time.")
    generation: GenerationProvenance = Field(description="How the analysis was generated (provenance).")
    design_snapshot: CampaignDesign | None = Field(
        default=None,
        description=(
            "What the campaign DECLARED, as of this generation. A snapshot rather than a reference because a "
            "declaration can gain a question after an analysis was written, and an analysis validated against "
            "four axes must not later read as having been validated against five. None for a campaign that "
            "declared nothing, whose memo says so and whose completeness is read against what actually ran."
        ),
    )
    document: AuthoredAnalysis = Field(
        description="What the generator authored, as written, with each figure reference in its prose rendered by code."
    )
    judged_tier_rule: JudgedTierRule | None = Field(
        default=None,
        description=(
            "The rule the judged evidence tiers in `resolutions` and `decision_surface` were decided by. "
            "`interval_lower_bound`: each criterion on confidence bounds for its agreement against the bar. None on an "
            "analysis stored before that rule, whose tiers were the point estimate against the bar — which awarded "
            "`calibrated` to a judge below the bar as much as a third of the time; every surface rendering such a "
            "tier says it was decided on the point estimate. Optional within v8 for that reason: requiring it "
            "would drop every stored analysis to learn a fact the old ones never had."
        ),
    )
    resolutions: list[FindingResolution] = Field(
        default_factory=list,
        description="What code filled for each finding, by position in `document.findings`: its numbers and its chart.",
    )
    coverage: CoverageLens = Field(
        default_factory=CoverageLens, description="The coverage lens: per-lever measurement summaries."
    )
    run_index: list[RunIndexEntry] = Field(default_factory=list, description="Per-run config + key metrics index.")
    variant_index: list[VariantIndexEntry] = Field(
        default_factory=list,
        description=(
            "One entry per variant observed under this campaign — the coordinate table that lets a "
            "reader place a cell on the declared axes. Frozen at generation from the cells the bundle "
            "built, and a peer of `run_index` at the coordinate the cell algebra actually uses. It is "
            "the DIMENSION table and never the report: the arm table's rows, their statuses and their "
            "order are derived at render, because persisting those is what would cost the pivot. Empty "
            "only for an analysis with no observation at all. An empty index yields no rows at all unless a control is "
            "declared, so what the arm "
            "table says in that case is that it derived nothing — and, beside it, every coordinate the "
            "memo's answer named that the join could not place. The disclosures do not ride on there "
            "being rows; an empty table is when the answer is most likely to have stranded."
        ),
    )
    decision_surface: DecisionSurface = Field(
        description=(
            "The campaign's per-cell numbers, as code computed them — each cell's measures, judged scores, "
            "replication and run notes, and every bar adjudicated against every cell. Copied from the "
            "bundle at generation and never authored by the model; every other number on the analysis — its "
            "evidence values and charts — was filled from it."
        ),
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

    @model_validator(mode="after")
    def check_positions(self) -> EvalAnalysis:
        """Refuse a link that points nowhere, a resolution that does not match its finding, or a gating cycle.

        Returns:
            This analysis, unchanged.

        Raises:
            ValueError: A position is out of range, a finding invalidates itself, the ``invalidates``
                relation cycles, ``resolutions`` does not have one entry per finding, or a resolution's
                rows are not the readings its finding names.
        """
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

        def in_range(where: str, positions: list[int]) -> None:
            bad = [position for position in positions if not 0 <= position < count]
            if bad:
                raise ValueError(
                    f"{where} names finding position(s) {bad}, and there are {count} findings (0-{count - 1})"
                )

        for index, decision in enumerate(self.document.decisions):
            in_range(f"decisions[{index}].rests_on", decision.rests_on)
        for index, answer in enumerate(self.document.questions):
            in_range(f"questions[{index}].rests_on", answer.rests_on)
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


#: How one generation attempt ended. ``stored``: an analysis was produced and persisted.
#: ``refused``: the generator finished and its output was refused, and so was the repair's.
#: ``failed``: anything else that ended the attempt with an error — a call the provider cut short,
#: a precondition refused before any call, a provider call that raised, a failed write.
#: ``cancelled``: the attempt was cancelled mid-flight, by a caller's budget or disconnect, or by a shutdown.
AttemptOutcome = Literal["stored", "refused", "failed", "cancelled"]


class EvalAnalysisAttempt(EvalDocumentModel):
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
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_analysis_attempt"] = "eval_analysis_attempt"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    campaign_id: str = Field(
        min_length=1, description="Reference → EvalCampaign.id — the campaign the generation was over."
    )
    outcome: AttemptOutcome = Field(
        description=(
            "How the attempt ended: stored (an analysis was persisted), refused (the output was refused and so was "
            "the repair's), failed (a call cut short, a precondition, a provider error or a failed write), or "
            "cancelled (a caller's budget or disconnect, or a shutdown, ended it mid-flight)."
        ),
    )
    analysis_id: str | None = Field(
        default=None,
        description="The analysis this attempt stored — set exactly when the outcome is `stored`.",
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
    bundle_fingerprint: str = Field(description="sha256 of the canonical context bundle the attempt ran over.")
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
        default=None,
        description="What the caller was told ended the attempt — set exactly when the outcome is not `stored`.",
    )
    started_at: str = Field(description="When the generation was started (ISO-8601).")
    created_at: str = Field(
        default_factory=utc_now_iso, description="When the attempt ended and was recorded (ISO-8601)."
    )

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

        Returns:
            This attempt, unchanged.

        Raises:
            ValueError: A stored attempt names no analysis or carries an error, or an unstored one names an
                analysis or carries no error, or a refused one records no refusal.
        """
        stored = self.outcome == "stored"
        if stored != (self.analysis_id is not None):
            raise ValueError(f"analysis_id is set exactly when the outcome is 'stored'; outcome={self.outcome!r}")
        if stored == (self.error is not None):
            raise ValueError(f"error is set exactly when the outcome is not 'stored'; outcome={self.outcome!r}")
        if self.outcome == "refused" and not self.refusals:
            raise ValueError("a refused attempt records the refusals that ended it")
        return self


class EvalInsight(EvalDocumentModel):
    """A durable, subject-scoped insight extracted from an analysis.

    Lives in the ``scope_id`` of the analysis that minted it. Traces back to the analysis that minted it
    via ``source_campaign_id`` / ``source_analysis_id``; ``invalidation_trigger``
    records the condition under which the insight should be re-checked.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_insight"] = "eval_insight"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
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
    scope: _LLMProse = Field(default="", description="Where the insight applies (free text).")
    confidence: ConfidenceTier = Field(description=_CONFIDENCE_TIER_DESCRIPTION)
    evidence_run_ids: _LLMList[str] = Field(default_factory=list, description="Run ids the insight rests on.")
    evidence_result_ids: _LLMList[str] = Field(default_factory=list, description="Result ids the insight rests on.")
    observed_at: str = Field(default_factory=utc_now_iso)
    model_versions: dict[str, str] = Field(default_factory=dict, description="Model ids in play when observed.")
    invalidation_trigger: _LLMProse = Field(default="", description="Condition under which to re-check the insight.")
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
    "EvalAnalysis",
    "EvalAnalysisAttempt",
    "EvalCampaign",
    "EvalInsight",
    "EvidenceRow",
    "EvidenceTier",
    "FindingResolution",
    "GenerationProvenance",
    "LeverCoverage",
    "RunIndexEntry",
    "VariantIndexEntry",
    "Viz",
    "VizType",
    "derive_window",
    "evidence_tier_of",
]
