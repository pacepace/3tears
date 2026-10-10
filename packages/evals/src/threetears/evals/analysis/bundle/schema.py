"""The context bundle's schema: :class:`AnalysisContextBundle` and every value object it carries.

All are typed sub-models so the bundle shape is validated and JSON-round-trips; none is a stored eval
doc_type (the bundle is an in-memory intermediate — only its fingerprint persists, on
``EvalAnalysis.generation.bundle_fingerprint``). The derivations that fill these models live in the
sibling modules of :mod:`threetears.evals.analysis.bundle`, and
:func:`~threetears.evals.analysis.bundle.assemble.assemble_context_bundle` composes them. This module
imports none of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from threetears.evals.analysis.agreement import (
    JudgeAgreement,
    JudgeSelfAgreement,
    InterJudgeAgreement,
    InterJudgeDimension,
)
from threetears.evals.kernel.evidence_tiers import (
    CALIBRATION_MIN_AGREEMENT,
    CALIBRATION_MIN_RESULTS,
    SEPARATION_MIN_AGREEMENT,
    SEPARATION_MIN_RESULTS,
    JudgedEvidenceTier,
    JudgeEvidenceTier,
)
from threetears.evals.analysis.judge_drift import JudgeDrift
from threetears.evals.analysis.cells import (
    CELL_MODEL_VERSION,
    Cell,
    NextExperiment,
    RefusedMerge,
    SubjectKeyInstability,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.reporting import served_model_state
from threetears.evals.analysis.measurement_windows import MeasurementWindow
from threetears.evals.analysis.lenses.comparison_sets import ComparisonSetsResult
from threetears.evals.analysis.lenses.frontier import FrontierResult
from threetears.evals.analysis.stats import (
    MULTIPLE_COMPARISON_CORRECTION,
    SIGNIFICANCE_ALPHA,
)
from threetears.evals.kernel.analysis_measures import BarAdjudication, MeasureCollection
from threetears.evals.kernel.campaign import CampaignWindow, EvalInsight, ReadingKind, VariantIndexEntry
from threetears.evals.kernel.covariates import REASONING_RATIO_KEY
from threetears.evals.kernel.declaration import CampaignDesign
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.schema.values import PooledProductionFooting, ProductionFooting
from threetears.evals.kernel.metrics import (
    AttributionScope,
    MeasureScale,
    MeritAxis,
    MetricDescriptor,
    Materiality,
)
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.models import (
    ApparatusProvenance,
    GoalCheckProof,
    RubricAxis,
)
from threetears.evals.kernel.surface import (
    CellFacts,
    GuardrailReadings,
    PredictionPoweredReading,
    TimeAxis,
)


# What "could not be decided" adds to a dimension's reason. Kept whole rather than
# interpolated per dimension so the catalog stays one entry per name; the STATE itself is
# structural (``Confound.status``) and nothing has to read this to detect it.
UNDECIDED_CONFOUND_PREFIX = "not recorded on every run in this campaign, so whether it varied cannot be established"


#: How one movement between two levels reads — :func:`~threetears.evals.analysis.stats.level_difference`'s
#: verdict, in the vocabulary of the history read (#592). ``not_separated`` claims nothing about whether the
#: measure moved; only ``equivalent`` says the move is small, and only against a declared margin.
MovementDirection = Literal["improved", "regressed", "equivalent", "not_separated", "untested"]


class MeasureMovement(EvalDocumentModel):
    """How one measure moved between two levels of a lever, and whether that movement separates from noise.

    ``direction`` is the movement tested against its own noise, never the bare sign of ``delta``: the
    engine's between-level test over per-case means (paired over the cases both levels ran, Welch's
    otherwise), read against Student's t at α. A movement that does not separate is ``not_separated``,
    which says the data cannot tell it from noise — never that the measure held still. ``equivalent``
    is the one reading that claims a small move, and it needs the measure's declared margin.
    """

    name: str = Field(min_length=1, description="The measure's registry name — its key into the measure_catalog.")
    scope: AttributionScope = Field(description="The measure's attribution scope.")
    mean_a: float = Field(description="Mean of the per-case means the test read at the first level.")
    mean_b: float = Field(description="Mean of the per-case means the test read at the second level.")
    delta: float = Field(description="mean_b - mean_a, in the measure's own unit.")
    se_of_delta: float | None = Field(
        default=None,
        description=(
            "Standard error of the difference the test read: of the per-case differences when paired, each level's "
            "SEM of its case means in quadrature when not. None when no test ran, and where every case moved by one "
            "amount (the bounded test on the declared range reads that, and has no standard error)."
        ),
    )
    test: Literal["paired", "unpaired"] | None = Field(
        default=None,
        description=(
            "`paired` = over the cases both levels ran; `unpaired` = Welch's t statistic on Hsu's conservative "
            "min(n) − 1 degrees of freedom over each level's cases, when they share fewer than two. None when no "
            "test could run."
        ),
    )
    n_a: int = Field(ge=0, description="Cases read at the first level (each case's repeats averaged first).")
    n_b: int = Field(ge=0, description="Cases read at the second level.")
    direction: MovementDirection = Field(
        description=(
            "improved / regressed = the movement separates from noise at alpha (this movement's own test, not "
            "corrected across the lens). equivalent = shown inside ± the measure's declared materiality threshold by "
            "a paired equivalence test; never read without one. not_separated = the data cannot tell this movement "
            "from noise, which says nothing about whether the measure moved (`not_separated_reason` says why where no "
            "test ran). untested = too few cases to test."
        )
    )
    not_separated_reason: str | None = Field(
        default=None,
        description=(
            "Set when `direction` is not_separated with no test run: every case moved by the same amount on a measure "
            "that declares no value_range, where no test of the mean can call the move (or a value lies outside the "
            "declared range). Names the remedy. None otherwise, and on a bundle assembled before 0.66, which read "
            "such a move by the exact sign-flip test, a test of symmetry rather than of the mean."
        ),
    )
    materiality: Materiality = Field(
        description=(
            "The movement read against the host's declared materiality threshold for this measure: `immaterial` "
            "when the delta is smaller than the threshold — too small to act on however clearly it clears its "
            "noise — and `material` otherwise, including when no threshold is declared. A caveat on an immaterial "
            "movement belongs in no finding."
        )
    )


class Confound(EvalDocumentModel):
    """One dimension that did not hold still, and what state that fact is in.

    A comparison groups runs by one lever. Everything else that varied inside those groups
    is a rival explanation for whatever moved, so the comparison is *marginal* — averaged
    over the rest — rather than controlled. That is the honest thing a campaign of this
    size can offer, and it is only misleading when it goes unsaid.

    **Three states, and all three are structural.** A dimension varied, it held still
    (absent from the list), or it *could not be decided* because some run never recorded it.
    The third is carried by ``status`` rather than by wording inside a sentence: a consumer
    told to detect "cannot be decided" by reading prose is a consumer inferring a fact from
    an unstructured signal, which is the defect this whole surface exists to remove. The
    reason prose would also drift out from under any test that matched it.

    **The reason lives once, in the bundle's ``confound_catalog``, keyed by ``dimension``**
    — the same normalisation :class:`MeasureSummary` uses for descriptors, and for the same
    reason: a coverage map re-states its confounds for every lever, so inlining a
    multi-sentence reason repeats it N times inside a bundle that IS the paid one-shot
    prompt. The reason still travels as a **value** in the same payload, which is what
    matters — the bundle reaches the generator as JSON and field descriptions do not.

    **Disclosure qualifies; it never suppresses.** A confounded comparison is often still
    worth reporting. What it may not do is credit the movement to the named lever alone.
    """

    dimension: str = Field(
        min_length=1,
        description=(
            "What varied — a lever name, a run attribute, an observed mechanism (`observed:<covariate>`), a "
            "resolved surface folded into its knob without a check (`unverified_fold:<surface>`), or the model that "
            "answered one requested model id (`served_model:candidate`); key into confound_catalog."
        ),
    )
    kind: Literal["swept_lever", "apparatus", "observed_mechanism", "unverified_fold", "served_model"] = Field(
        description=(
            "swept_lever = another knob this campaign deliberately tuned. apparatus = the measuring rig moved "
            "under the comparison, which is the more serious of the two because nothing intended it. "
            "observed_mechanism = the comparison is across candidate models, and what the models were measured "
            "doing diverged with no setting to say so: their means of the named covariate are at least "
            "`threshold` apart, so part of the movement may belong to that difference rather than to the model. "
            "unverified_fold = a resolved surface the host records beside the knob written into it moved with "
            "that knob and is reported as the same change, but every level of the knob here was run by one arm "
            "only, so nothing in these runs could have shown the surface moving apart from the knob: the fold is "
            "an assumption these runs did not test, never a checked non-confound. "
            "served_model = the runs asked for one candidate model id and the provider's responses named more than "
            "one model as having answered it (a floating alias that moved, within an arm or between arms), so the "
            "numbers under that id are a mixture of models; undecided when some response named no model, so which "
            "model answered cannot be established."
        )
    )
    status: Literal["varied", "undecided"] = Field(
        default="varied",
        description=(
            "varied = observed at more than one value across these runs. undecided = some run never recorded it, "
            "so whether it varied cannot be established — NOT the same as holding still, which is absence from "
            "the list. Treat undecided as present until a run says otherwise. An observed_mechanism confound is "
            "always varied: one is named only where two levels' measured means diverged. An unverified_fold "
            "confound is always varied: the surface did take more than one value; what is untested is why."
        ),
    )
    level_values: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "observed_mechanism only: the comparison's level -> the covariate's value there, the mean of its "
            "per-case means over the level's results that measured it (repeats of a case averaged first, as "
            "everywhere a level's value is stated). A level that measured none is absent. Empty on every other "
            "kind."
        ),
    )
    threshold: float | None = Field(
        default=None,
        description=(
            "observed_mechanism only: how far apart two levels' means must be for the divergence to be named, "
            "in the covariate's own unit. None on every other kind."
        ),
    )

    @model_validator(mode="after")
    def _observed_mechanism_carries_its_evidence(self) -> Confound:
        """An observed-mechanism confound states its values and threshold; no other kind carries either.

        Raises:
            ValueError: The evidence fields disagree with ``kind``.
        """
        if self.kind == "observed_mechanism":
            if self.threshold is None or len(self.level_values) < 2 or self.status != "varied":
                raise ValueError(
                    "an observed_mechanism confound names the threshold it crossed and at least two levels' "
                    "values, and is always varied"
                )
        elif self.level_values or self.threshold is not None:
            raise ValueError(f"a {self.kind} confound carries no level_values or threshold")
        if self.kind == "unverified_fold" and (
            self.status != "varied" or not self.dimension.startswith(UNVERIFIED_FOLD_PREFIX)
        ):
            raise ValueError(
                f"an unverified_fold confound names its surface as {UNVERIFIED_FOLD_PREFIX}<surface>, and is always "
                "varied"
            )
        if (self.kind == "served_model") != self.dimension.startswith(SERVED_MODEL_PREFIX):
            raise ValueError(f"a served_model confound, and only one, names its role as {SERVED_MODEL_PREFIX}<role>")
        return self


#: The prefix an observed mechanism's confound dimension carries — the covariate it names follows it.
#: Prefixed for the reason world dimensions are: a covariate name and a lever name share
#: ``confound_catalog``, and a bare name meeting a lever's would let one reason overwrite the other.
OBSERVED_MECHANISM_PREFIX = "observed:"

#: The prefix an unverified fold's confound dimension carries — the resolved surface it names follows it.
#: Prefixed for the same reason: the surface is a lever name too, and where some other cohort does not fold
#: it, it is named there as a confound of its own with its own reason, which a bare name would overwrite.
UNVERIFIED_FOLD_PREFIX = "unverified_fold:"

#: The prefix a served-model confound's dimension carries — the role whose served model it names follows it.
#: Prefixed for the reason the two above are: it shares ``confound_catalog`` with lever names, and a host
#: may name a lever anything.
SERVED_MODEL_PREFIX = "served_model:"

#: The one served-model confound the engine raises today: the candidate's. The judge's served model is
#: an apparatus input (``judge_model``) and confounds as one.
CANDIDATE_SERVED_MODEL_CONFOUND = f"{SERVED_MODEL_PREFIX}candidate"

#: Why the candidate's served model moving under one requested id clouds a comparison, in the catalog's words.
_SERVED_MODEL_CONFOUNDS = (
    "the candidate was asked for one model id and the provider's responses named more than one model as having "
    "answered it — a floating alias (a 'latest' pointer) resolves on the provider's side and can move between "
    "runs or within one — so the numbers recorded under that id are a mixture of models, and a difference between "
    "arms may belong to which model answered rather than to anything the arms set"
)

#: How far apart two levels' mean reasoning share (``reasoning_ratio``, absolute) must be before a
#: comparison between them is disclosed as confounded by it. A reasoning effort is sent to a provider
#: as a word, and each vendor maps the word to its own effective budget; two model arms pinned to one
#: word observed at 0.30 and 0.70 were not held at one reasoning budget, and an arm that spends its
#: output cap reasoning truncates and is charged for it as the model's failure. A fifth of the
#: completion is the scale at which that has been seen to decide outcomes; it is stated on every
#: confound it raises, so a reader can disagree with it.
REASONING_SHARE_DIVERGENCE = 0.20


@dataclass(frozen=True)
class _ObservedMechanism:
    """One covariate read as an observed mechanism: when its divergence is named, and why it matters."""

    threshold: float
    reason: str


#: The covariates read as observed mechanisms, each with its threshold and the reason a divergence in it
#: clouds a comparison. One entry today; the check is written over the covariate name, so a second
#: is an entry here rather than a second code path.
_OBSERVED_MECHANISMS: dict[str, _ObservedMechanism] = {
    REASONING_RATIO_KEY: _ObservedMechanism(
        threshold=REASONING_SHARE_DIVERGENCE,
        reason=(
            "the share of the candidate's output spent reasoning differed between these levels by at least the "
            "stated threshold. A reasoning effort is sent as a word that each vendor maps to its own budget, so a "
            "comparison of models at one effort setting does not hold reasoning constant: the level that reasons "
            "more can spend its output cap reasoning and truncate, and that cost is then charged to the level "
            "rather than to the budget it was given"
        ),
    )
}


def observed_mechanism_key(covariate: str) -> str:
    """Name an observed mechanism for the confound maps.

    Args:
        covariate: The covariate's registered name.

    Returns:
        The prefixed key. See :data:`OBSERVED_MECHANISM_PREFIX`.
    """
    return f"{OBSERVED_MECHANISM_PREFIX}{covariate}"


#: Why a swept lever's mechanism could not be checked: ``not_declared`` = the lever names no measure it
#: acts on; ``not_swept`` = it was observed at one level, so there is nothing to compare;
#: ``levels_unobserved`` = some level observed none of the measure, so no pair of levels separated and
#: whether it held still at every level cannot be shown; ``too_few_observations`` = every level observed
#: it, but some pair of levels has too few cases on a side for the separation test to run;
#: ``uniform_move_needs_range`` = no pair separated, and some pair shifted every case alike on a measure that
#: declares no range, where no test of the mean can call the shift (declare ``value_range`` on the measure).
MechanismUncheckedReason = Literal[
    "not_declared", "not_swept", "levels_unobserved", "too_few_observations", "uniform_move_needs_range"
]


class MechanismCheck(EvalDocumentModel):
    """Whether the measure a swept lever declares it acts on measurably moved across the lever's levels.

    A lever that did not move an outcome and a lever that never took effect read alike in every
    outcome measure, and they lead to opposite actions. The lever's own declaration
    (:attr:`~threetears.evals.kernel.host.sweepables.Sweepable.acts_on`) names the measure that
    should have moved; this records whether it did, with the per-level evidence beside the verdict.

    **Read with the engine's own separation test, never by inequality.** Each pair of levels is
    compared on the measure's per-case means exactly as a family comparison compares a contrast with
    the control (paired over shared cases, else Welch's; Holm-corrected across the lever's pairs), so
    noise does not read as a lever taking effect. A gap with no spread — every case shifted alike — is
    read by the bounded test on the measure's declared range, and with no range is never ``moved``: no
    test of the mean can call it, so the check is ``unchecked`` for ``uniform_move_needs_range`` (#597).

    **Three states, none of them a default.** ``moved``: some pair of levels separates on the measure.
    ``inert``: every level observed it, every pair could be tested, and none separates — no measurable
    evidence the lever acted on its mechanism; identical constants are the degenerate case.
    ``unchecked``: the engine could not tell, and ``reason`` says why; an unchecked lever is NOT a lever
    that took effect.
    """

    state: Literal["moved", "inert", "unchecked"] = Field(
        description=(
            "moved = some pair of levels separates on the mechanism measure. inert = every pair was tested and none "
            "separates: no measurable evidence the lever acted on its mechanism, so a null outcome on this lever is "
            "not evidence it does not matter. unchecked = it could not be established either way (see reason)."
        )
    )
    measure: str | None = Field(
        default=None, description="The measure or covariate the lever declares it acts on. None when it declares none."
    )
    level_means: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "The lever's level -> the mean of `measure`'s per-case means over this row's results at that level "
            "(repeats of a case averaged first, the unit the separation test reads). A level that observed none "
            "is in `unobserved_levels` instead."
        ),
    )
    level_n: dict[str, int] = Field(
        default_factory=dict,
        description="The lever's level -> how many cases observed `measure` there: the n each mean and test rests on.",
    )
    unobserved_levels: list[str] = Field(
        default_factory=list, description="Levels at which no result observed `measure`, sorted."
    )
    reason: MechanismUncheckedReason | None = Field(
        default=None,
        description=(
            "Why the check is unchecked; None otherwise. not_declared = the lever names no mechanism. not_swept = "
            "one level only. levels_unobserved = some level observed none of it. too_few_observations = some pair "
            "of levels had fewer than two cases on a side. uniform_move_needs_range = no pair separated, and some "
            "pair shifted every case by the same amount on a measure that declares no value_range, where no test of "
            "the mean can call the shift; declare value_range on the measure for the bounded test to read it. A "
            "bundle assembled before 0.66 read such a shift by the exact sign-flip test and keeps its state."
        ),
    )

    @model_validator(mode="after")
    def _state_carries_its_evidence(self) -> MechanismCheck:
        """Each state carries exactly the evidence that decides it.

        Raises:
            ValueError: The fields contradict ``state``.
        """
        if (self.state == "unchecked") != (self.reason is not None):
            raise ValueError("a mechanism check is unchecked exactly when it states a reason")
        if self.measure is None and self.reason != "not_declared":
            raise ValueError("a lever declaring no mechanism can only be unchecked for that reason")
        if set(self.level_n) != set(self.level_means):
            raise ValueError("every level with a mean states the n it rests on, and no other level does")
        if self.state != "unchecked" and len(self.level_means) < 2:
            raise ValueError("moved and inert compare at least two levels")
        if self.state == "inert" and self.unobserved_levels:
            raise ValueError("inert needs the measure observed at every level")
        return self


class ArmMechanismReading(EvalDocumentModel):
    """One arm's mean of one observed-mechanism covariate, or the statement that it was not measured."""

    variant_key: str = Field(min_length=1, description="The arm, as `arms` and `design` key it.")
    covariate: str = Field(min_length=1, description="The covariate read, e.g. `reasoning_ratio`.")
    mean: float | None = Field(
        default=None,
        description=(
            "The mean of its per-case means over the arm's results that measured it (repeats of a case averaged "
            "first, as in `confounded_by`). None = no result of this arm measured it, which is not zero: a "
            "provider that reports no reasoning split leaves the share unknown."
        ),
    )
    n_measured: int = Field(ge=0, description="The arm's results that measured the covariate.")
    n_results: int = Field(ge=0, description="The arm's results in all.")


class ArmServedModel(EvalDocumentModel):
    """Which model answered one arm's candidate calls, as the provider's responses named it.

    An arm is keyed by the model id its launch ASKED for, and a floating alias is resolved on the
    provider's side, so two runs of one arm months apart can have been answered by different models and
    still pool as one arm. Only the response names the model that answered (``RoleUsage.served_model``),
    so this reads that and nothing else: never the requested id standing in for a response that named
    none.
    """

    variant_key: str = Field(min_length=1, description="The arm, as `arms` and `design` key it.")
    served_models: list[str] = Field(
        default_factory=list,
        description=(
            "Every distinct model the provider's responses named as having answered this arm's candidate calls, "
            "sorted. Never the requested id: a call whose response named no model adds nothing here and is "
            "counted in n_unrecorded."
        ),
    )
    n_results: int = Field(ge=1, description="The arm's results whose candidate calls left a usage row.")
    n_unrecorded: int = Field(
        ge=0,
        description=(
            "Of those, the results with at least one candidate call whose response named no model — or stored "
            "before served models were recorded. Not recorded, never a match with the requested id."
        ),
    )
    state: Literal["one", "pooled", "unrecorded"] = Field(
        description=(
            "one = every candidate call named one and the same model. pooled = the arm pooled observations "
            "answered by two or more models (served_models), so its numbers are a mixture, and every comparison "
            "involving it names the served_model confound. unrecorded = at most one model was named and some call "
            "named none, so whether the arm was answered by one model cannot be established."
        )
    )

    @model_validator(mode="after")
    def _state_follows_the_counts(self) -> ArmServedModel:
        """The state is the one the served models and the unrecorded count imply, never a second opinion.

        Raises:
            ValueError: ``state`` disagrees with ``served_models`` and ``n_unrecorded``.
        """
        expected = served_model_state(self.served_models, self.n_unrecorded)
        if self.state != expected or self.n_unrecorded > self.n_results:
            raise ValueError(
                f"an arm with these served models and unrecorded calls is {expected!r}, not {self.state!r}"
            )
        return self


class DesignArm(EvalDocumentModel):
    """One arm of the campaign, the runs that measured it, and what it moved off the control.

    **An arm is a variant key**, and a run carries exactly one, so every run of an arm belongs to it
    whole. Several runs of one arm are that arm's repeats: they pool into it and raise its ``k``,
    never appearing as a second arm that "moved nothing". Reading runs as arms was the run-keyed
    design's mistake — a re-run of the control became a contrast, and a run carrying several models
    became one arm at a set of models.

    ``moved`` is DERIVED by diffing this arm's effective configuration against the control arm's,
    never declared: a declared lever is a second source of truth for a fact the runs already carry.
    The candidate model is an ordinary lever here. An arm that moved two things derives two, which is
    the honest reading of a design that is not one-factor-at-a-time.
    """

    variant_key: str = Field(
        min_length=1, description="The arm's variant key — the coordinate `cells` and `variant_index` use."
    )
    run_ids: list[str] = Field(
        min_length=1,
        description="Every resolved member run that measured this arm, in bundle order. Several runs are repeats of one arm.",
    )
    k: int = Field(
        ge=0,
        description=(
            "Repeats per case this arm's runs PLANNED, at the case repeated least: runs over the same cases add "
            "their `k_runs`, runs over different cases do not. What was delivered is the arm's cells' "
            "`repeats_per_case_min`/`max`, which fall below this when a run came up short. The arm's planned "
            "replication, which a coverage row's `k` floors across a lever's levels."
        ),
    )
    moved: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Lever → this arm's level, for every lever whose value differs from the control arm's; empty on the "
            "control. Empty on a non-control arm means it moved no LEVER — its key differs from the control's "
            "through something no lever names. The apparatus is not a lever, so an arm measured under a "
            "different judge or template can land here too, and apparatus_confounds is what reports that."
        ),
    )
    mechanism_confounds: list[Confound] = Field(
        default_factory=list,
        description=(
            "What was observed, not set, to differ under this arm and the control arm: each observed mechanism that "
            "diverged when they ran different candidate models (the covariate, the threshold it crossed and both "
            "models' values), and the served_model confound when one requested model id was answered by more than "
            "one model across the two. It qualifies the contrast and suppresses nothing. Empty on the control, and "
            "when neither applies."
        ),
    )


class RealizedDesign(EvalDocumentModel):
    """What kind of experiment this campaign turned out to be — INFERRED from the runs, never declared.

    **The derived twin of the declaration, and the two must not share a name.** The declaration
    (:class:`~threetears.evals.kernel.declaration.CampaignDesign`, on the campaign) says what an
    operator SET OUT to learn; this says what the observations actually show. The delta between
    them is the coverage story — a declared value with no observations is NAMED, ``not_run`` in its
    axis row's ``declared_levels`` (``undetermined`` where some run's level cannot be established),
    where this one can only ever report what happened to run.

    They were briefly both called ``CampaignDesign``, in one package, and the collision silently
    renamed a published OpenAPI component: two live classes, one schema name, and whichever the
    generator reached first won.

    A campaign is curated membership with no structure, so an analysis that wants to compare
    arms has to infer one. Inferring *factorial* is the expensive mistake: it pools every run
    sharing a level of the lever under test, which in a star design puts arms that moved a
    DIFFERENT lever inside the comparison — manufacturing a difference within it and then
    correctly reporting it as confounded. The campaign that produced this model had its one
    decision deferred on exactly that invented confound, over data that isolated the lever cleanly.

    **Keyed on arms, never on runs.** The control is the declared control VARIANT with every run
    that measured it; each other arm is one contrast however many runs measured it.
    """

    control_arm: DesignArm | None = Field(
        default=None,
        description=(
            "The arm every contrast is read against: the campaign's DECLARED control variant, with every resolved "
            "run that measured it. Null when no control resolved."
        ),
    )
    control_excluded: Literal["archived", "unresolved", "unobserved"] | None = Field(
        default=None,
        description=(
            "Set only when the campaign DID declare a control and no resolved observation carries it. "
            "Three causes, three different operator actions, and collapsing any two sends someone the "
            "wrong way. 'archived' = a member run carrying that variant was curated out, so the campaign "
            "gave up its design by choice — un-archive it or declare another. 'unresolved' = some member "
            "run could not be loaded at all, so whether it carried the variant is UNKNOWABLE rather than "
            "false; a membership outliving its run is a recurring state in a real store, and this is the arm that "
            "keeps it from being reported as a coverage gap. 'unobserved' = every member resolved and "
            "none carried it, which is the only one that says the experiment was not run as designed. "
            "Null alongside a null control_arm means no control was ever declared."
        ),
    )
    contrasts: list[DesignArm] = Field(
        default_factory=list,
        description=(
            "Every other arm, with the levers it moved off the control, in order of its first run. One entry "
            "per arm: runs repeating an arm are listed in its `run_ids`, never as another contrast."
        ),
    )
    unplaced_run_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Resolved runs none of whose observations resolved a variant key, so they belong to no arm and "
            "are in no contrast. They still count in every campaign-wide comparison."
        ),
    )
    shape: Literal["one_factor_at_a_time", "multi_factor", "undesignated"] = Field(
        default="undesignated",
        description=(
            "one_factor_at_a_time = a control resolved and every contrast moves at most one lever, so "
            "each one IS a clean contrast against it. multi_factor = a control resolved and some "
            "arm moved more than one, so its movement cannot be attributed to either. "
            "undesignated = no control resolved, and no comparison here is known to be controlled."
        ),
    )


class ScopeDivergence(EvalDocumentModel):
    """Two scopes disagreeing about what one lever change did — a finding, not a caveat.

    The whole run can move by more or less than the part under test accounts for, and a report
    that ranks on either lane alone presents that as a clean result. The concrete failure: turn
    latency roughly halved between two arms while the tuned subsystem's own elapsed time barely
    changed, so the arm was credited for a ~50s improvement that happened somewhere else entirely.

    **What is tested is the divergence itself**: whether the whole's movement and the part's differ.
    Each case's remainder — its whole minus its part — is compared between the two levels by the
    engine's between-level test (paired over shared cases, Welch's otherwise, per-case means so
    repeats are not counted as cases), and a lever's divergence tests are Holm-corrected together. A
    divergence is published only where that corrected test separates. Two movements graded apart and
    set side by side are never the test: a whole that separates beside a part that does not is no
    evidence the two differ.

    The two measures are paired by **unit**, which is what makes this subject-agnostic: an
    end-to-end and a subsystem measure in the same unit are two views of one quantity at
    different scopes, so reading their movements side by side is meaningful. Measures in
    different units are not comparable at all and are never paired, so nothing spurious
    arrives from a campaign that happens to track both money and milliseconds.

    A shared unit is what makes the two **comparable**; it is not what makes one *part of*
    the other, and the difference decides whether they may be SUBTRACTED. Milliseconds spent
    on detached background work are disjoint from the milliseconds a turn span covers, so
    their remainder describes no stretch of wall-clock. Containment is declared per measure
    (``MetricDescriptor.contained_by``) and absent by default; where it is not declared,
    both movements are still published and ``unattributed_withheld`` says why no number is.

    Containment alone is not enough: the part must also EXHAUST the whole. Where the catalog
    partitions a whole into several components, differencing one of them leaves the others'
    movement, which is attributed rather than unattributed — so that too is withheld with a
    sentence naming the components. Both movements are published in every withheld case; only
    the arithmetic between them is refused.

    **Where the catalog partitions the whole, the parts are graded too, and the one carrying the
    movement is named.** A whole that moved by more than a subsystem measure says only that the
    difference was not THERE; left at that, a reader sets the whole beside whatever else is in view
    — a disjoint phase timing, say — and attributes the swing to the lever. The whole's own
    components answer where it went: a ``total_ms`` swing that is almost all ``llm_ms`` is time
    inside model calls, which a provider's load moves as readily as any lever.
    """

    lever: str = Field(min_length=1, description="The lever whose levels are being compared.")
    level_a: str = Field(description="The first level ('—' = ran without the override).")
    level_b: str = Field(description="The second level.")
    unit: str = Field(min_length=1, description="The unit both measures share — why they are comparable.")
    end_to_end: MeasureMovement = Field(description="How the whole-run measure moved.")
    subsystem: MeasureMovement = Field(description="How the isolating measure moved.")
    test: Literal["paired", "unpaired"] = Field(
        description=(
            "The test of the divergence — of each case's whole-minus-part between the two levels: `paired` over "
            "the cases both levels ran, `unpaired` (Welch's t statistic on Hsu's min(n) − 1 degrees of freedom) when "
            "they share fewer than two."
        )
    )
    n_cases_a: int = Field(ge=2, description="Cases carrying both measures that the divergence test read at level_a.")
    n_cases_b: int = Field(ge=2, description="Cases carrying both measures that the divergence test read at level_b.")
    p_raw: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "The divergence test's own two-sided p, before correction. Kept for audit, never the figure the "
            "divergence rests on, and withheld from the analysis writer."
        ),
    )
    p_adjusted: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "The Holm-adjusted p across every divergence test of this lever (`family_size`) — the figure the "
            "divergence rests on. Always below alpha: nothing else is published."
        ),
    )
    family_size: int = Field(
        ge=1, description="How many divergence tests this lever's comparisons carried a p, corrected together."
    )
    unattributed_delta: float | None = Field(
        default=None,
        description=(
            "The whole-run movement the isolating measure does not account for (end_to_end delta minus "
            "subsystem delta), in the shared unit — the swing that happened outside the part under test. "
            "None whenever the subtraction would not be sound, in which case unattributed_withheld says "
            "which way. The DIRECTIONS above stay valid either way (each is a true statement about its "
            "own population); only the arithmetic does not."
        ),
    )
    unattributed_withheld: str | None = Field(
        default=None,
        description=(
            "Why no unattributed swing is stated, as a sentence — the two measures point in opposite "
            "better-directions, they are averaged over different populations, the part is not declared "
            "a component of the whole, or the part is only ONE of the whole's declared components, so "
            "the leftover is the other components' movement rather than anything unattributed. None "
            "when the subtraction was sound and the number is present. Carried as prose because the "
            "reader is owed the reason, not a null to interpret."
        ),
    )
    contained_by: str | None = Field(
        default=None,
        description=(
            "What the measure catalog declares `subsystem.name` to be a component OF "
            "(`MetricDescriptor.contained_by`), carried here so a consumer of this divergence has the fact "
            "that decides the subtraction without a second lookup against a catalog it may not hold. None "
            "means 'not known to be contained', which is where every measure starts. It is the SOURCE of "
            "the containment clause in `unattributed_withheld`, so the two cannot drift: where this does "
            "not name `end_to_end.name`, no swing is stated. The converse does NOT hold — naming it is "
            "necessary and not sufficient, because a part that is one of several declared components of "
            "the whole is contained and still earns no remainder (see `_unsound_subtraction`)."
        ),
    )
    confounded_by: list[Confound] = Field(
        default_factory=list,
        description=(
            "Everything NOT held fixed across the two cohorts being compared — other swept levers, "
            "run attributes that moved on their own, observed mechanisms that diverged, and a requested candidate "
            "model answered by more than one model. This is a marginal comparison, not a controlled "
            "one: some of the movement may belong to these. Empty means the comparison is clean."
        ),
    )
    whole_components: list[MeasureMovement] = Field(
        default_factory=list,
        description=(
            "How each measure the catalog declares a component of `end_to_end.name` moved between the same two "
            "levels, graded against its own noise. Empty when the catalog declares no partition of the whole, "
            "or no component was measured at both levels. Sorted by name."
        ),
    )
    carried_by: str | None = Field(
        default=None,
        description=(
            "The component shown to carry the whole-run movement: the one with the largest delta in the whole's "
            "direction, named only when its own movement separates that way and it is shown to move further than "
            "every other component (each case's difference between the two, tested between the levels, "
            "Holm-adjusted). None when the whole's movement does not separate from its noise, no component moved "
            "its way, or no single component is shown to carry it — read `whole_components` then. When it "
            "is time inside model calls (llm_ms), a provider's load moves it as readily as the lever does, so "
            "read candidate_output_tokens_per_s across the two cohorts before attributing the movement to the lever."
        ),
    )
    carried_share: float | None = Field(
        default=None,
        description="`carried_by`'s delta as a fraction of the whole's delta — how much of the movement it carries. None with it.",
    )

    @model_validator(mode="after")
    def _swing_and_its_absence_are_exclusive(self) -> ScopeDivergence:
        """Refuse a divergence that states both a swing and a reason there is none, or neither.

        The two fields are one fact in two shapes, and every reader treats them that way: a
        number present with a reason beside it reads as a qualified measurement rather than a
        refused one, and a null with no reason is the bare hole this pair exists to close.
        Enforced on the model because the pair travels to the generator as data — the single
        construction site keeping them in step is not something the JSON can be trusted on.

        Raises:
            ValueError: If both are set, or neither is.
        """
        if (self.unattributed_delta is None) == (self.unattributed_withheld is None):
            raise ValueError(
                "exactly one of unattributed_delta / unattributed_withheld must be set — "
                f"got delta={self.unattributed_delta!r}, withheld={self.unattributed_withheld!r}"
            )
        return self


#: Why a judged dimension sits beside the ranking surface rather than on it. One sentence for every
#: dimension, because the reason is a property of how judged scores are produced and not of any
#: one dimension: an arm that delivered nothing has nothing for a judge to score, so it is ABSENT
#: from the dimension rather than scored low, and a ranking on it can reward declining the work.
JUDGED_OFF_RANKING_REASON = (
    "judge-mediated: a judge scores only what an arm delivered, so an arm that delivered nothing is "
    "absent from this dimension rather than scored low on it, and ranking on it can reward declining "
    "the work. It is measured quality — reportable, with its posture stated — and never a ranking measure."
)


class JudgedArm(EvalDocumentModel):
    """One judged dimension's scores in one cell — the arm, measured under one rig."""

    variant_key: str = Field(
        min_length=1, description="The arm's variant — its key into variant_index, and half of its cell."
    )
    apparatus_class_id: str = Field(
        min_length=1, description="The rig it was measured under — the other half of its cell."
    )
    run_ids: list[str] = Field(
        min_length=1,
        description=(
            "The member runs whose observations this cell pooled, sorted — what a finding citing this "
            "judged score puts in `observation_refs`. The variant key is a cell coordinate, never a run id, and "
            "a citation of one is refused."
        ),
    )
    n: int = Field(ge=0, description="Scores contributing to the mean — one per scored observation.")
    n_independent: int = Field(
        ge=0,
        description=(
            "Distinct test cases behind those scores — the independent draws. Below n, the scores are repeats "
            "of the same cases, and `sem` is computed over the cases, exactly as on a telemetry measure."
        ),
    )
    n_infra_excluded: int = Field(
        default=0,
        ge=0,
        description=(
            "Scores the judge gave to observations the harness had already faulted, left out of `n` and the "
            "mean: a judge reading a broken transcript is not measuring the candidate."
        ),
    )
    case_means: list[float] | None = Field(
        default=None,
        description=(
            "Each test case's mean over its observations, ascending, recorded only below 5 cases (the chart band "
            "floor): a chart draws these as points rather than an interval band there. None at 5 cases or more, "
            "and on a summary stored before it, which reads as not recorded."
        ),
    )
    n_cannot_tell: int = Field(
        default=0,
        ge=0,
        description=(
            "Observations on which the judge answered the evidence did not let it score this dimension, "
            "left out of `n` and the mean. Not a fault: the judge worked and declined to guess."
        ),
    )
    mean: float | None = Field(default=None, description="Mean score on the dimension's own scale. None when n is 0.")
    sem: float | None = Field(
        default=None,
        description=(
            "Standard error of that mean, over the test cases (cluster-robust: a case's repeats are not "
            "independent draws), read on `n_independent - 1` degrees of freedom. None below two cases, where "
            "no between-case spread is estimable."
        ),
    )
    evidence_tier: JudgedEvidenceTier = Field(
        description=(
            "What these scores can bear: the weakest `judge_evidence_tiers` tier among the judges that served the "
            "scores counted in `n` — `undetermined` when none was counted or the evidence decides no tier. "
            "Flagged, never a reason the scores are dropped."
        )
    )
    prediction_powered: PredictionPoweredReading | None = Field(
        default=None,
        description=(
            "The prediction-powered estimate (#598): `mean` corrected by people's calibration ratings of some of the "
            "counted observations — the judge's mean plus the rectifier (person minus judge over the rated ones), "
            "with its interval clustered by case. Beside `mean`, never replacing it: `mean` is what the judge said, "
            "this is what people would have said, estimated. Its `mean` is None (with the reason) below "
            "`min_labelled` rated observations. None when no person rated any counted observation here."
        ),
    )


class JudgedMeasure(EvalDocumentModel):
    """A judged dimension, measured — carried so its exclusion from ranking is not read as its absence.

    Judged dimensions are kept off the ranking surface permanently, and that exclusion used to erase
    them: a rubric leaf reached the measure walk as a bare ``score`` with the dimension's name already
    lost, was refused by the ranking filter, and appeared nowhere else, so a campaign whose judge scored
    every result read to the generator as one that measured no quality at all. This is the surface they
    live on instead — named, per cell, with the reason they are not ranked stated once rather than
    inferred from where they are not.
    """

    name: str = Field(min_length=1, description="The dimension, spelled exactly as the judge stamped it.")
    family: str = Field(
        min_length=1,
        description="The registry family — `rubric` for a template dimension, `dual_axis` for a reserved axis.",
    )
    value_range: tuple[float, float] | None = Field(default=None, description="The scale the scores are on.")
    scale: MeasureScale | None = Field(
        default=None,
        description="`interval` for a 1-5 score (only differences mean anything), `ratio` for a pass rate.",
    )
    higher_is_better: bool = Field(default=True, description="Which end of the scale is better.")
    axis: RubricAxis = Field(
        default="capability",
        description=(
            "`boundary` when any score on it was judged as a boundary dimension: a guardrail, decided in "
            "`guardrails` and never in a comparison family or the composite. `capability` otherwise."
        ),
    )
    off_ranking_reason: str = Field(
        default=JUDGED_OFF_RANKING_REASON,
        description="Why this dimension is measured and reportable yet never a ranking measure. Absent from measure_catalog for this reason alone.",
    )
    bar_threshold: float | None = Field(
        default=None,
        description="The threshold of the declared bar naming this dimension, or None when no bar names it. Its verdicts are in bar_adjudications.",
    )
    arms: list[JudgedArm] = Field(
        default_factory=list,
        description="One entry per cell carrying a score on this dimension, ordered by (variant_key, apparatus_class_id).",
    )
    second_judges: list[InterJudgeDimension] = Field(
        default_factory=list,
        description=(
            "How far each second judge asked about the member runs agreed with their judge on this dimension — n, exact "
            "agreement, kappa and its bounds (`inter_judge_agreement`), beside the scores it qualifies. Empty when no "
            "second judge was asked: agreement between judges is then unmeasured, not perfect."
        ),
    )


class JudgeIdentityLevel(EvalDocumentModel):
    """One judge the member runs were scored by, as each run recorded it at launch, and the runs and arms under it."""

    judge_model: str = Field(
        min_length=1, description="The run's judge pin: the model every unconfigured dim was sent to."
    )
    judge_config_ids: dict[str, str] = Field(
        default_factory=dict,
        description="dim -> the versioned JudgeConfig the run pinned; absent = the built-in prompt.",
    )
    judge_temperature: float | None = Field(
        default=None, description="The temperature unconfigured dims were requested at; None = not recorded."
    )
    run_ids: list[str] = Field(min_length=1, description="The member runs judged this way, sorted.")
    variant_keys: list[str] = Field(default_factory=list, description="The arms those runs measured, sorted.")


class JudgeDriftLink(EvalDocumentModel):
    """A drift reading that spans a judge change: evidence of one side re-scored by the other side's judge."""

    from_level: int = Field(ge=0, description="The index in `levels` whose runs' evidence was re-scored.")
    to_level: int = Field(ge=0, description="The index in `levels` whose judge re-scored it.")
    run_ids: list[str] = Field(min_length=1, description="The runs whose stored evidence was re-scored, sorted.")
    pass_ids: list[str] = Field(min_length=1, description="The second-judge passes read, sorted.")
    drift: JudgeDrift = Field(description="How far each dimension moved between the two judges on the same evidence.")


class JudgeChange(EvalDocumentModel):
    """Whether the member runs were judged by more than one judge, which, and the drift readings that span the change."""

    levels: list[JudgeIdentityLevel] = Field(
        default_factory=list,
        description=(
            "Each judge the judged member runs recorded, ordered by model, configs and temperature. More than one is a "
            "judge change: a judged difference between arms on different levels may be the judge, not the subject."
        ),
    )
    drift_links: list[JudgeDriftLink] = Field(
        default_factory=list,
        description=(
            "Drift readings among the member runs that re-scored one level's evidence under another level's judge. "
            "Empty when none exists — then nothing measured how far the judge change alone moves the scores."
        ),
    )
    sentence: str | None = Field(
        default=None,
        description="The sentence to quote about the change; None when every judged member run had one judge.",
    )


class TokenRollup(EvalDocumentModel):
    """Summed token usage across results — visible cost of generation (§Visible Costs).

    A count a provider did not report is left out of its sum, never added as zero — the same rule
    cost follows (``n_cost_unpriced``). Each sum is ``None`` when no row reported that count at all,
    and ``n_results_tokens_unreported`` says how many results' prompt or completion counts are
    missing from the sums, so a partial sum reads as "at least this much".
    """

    prompt_tokens: int | None = Field(
        ge=0,
        description="Summed reported prompt tokens over results carrying usage; None when no row reported one.",
    )
    completion_tokens: int | None = Field(
        ge=0, description="Summed reported completion tokens; None when no row reported one."
    )
    reasoning_tokens: int | None = Field(
        ge=0,
        description="Summed reasoning tokens; None when no counted row reported a split, 0 only when one reported zero.",
    )
    n_results_tokens_unreported: int = Field(
        ge=0,
        description=(
            "Results carrying usage with at least one token-metered row (any role but external) whose prompt or "
            "completion count the provider did not report. Those counts are absent from the sums, not zero: read "
            "the sums as 'at least this much' when this is non-zero. External rows, metered in provider units, "
            "carry no token counts by design and are not counted here."
        ),
    )
    n_results_with_usage: int = Field(
        ge=0,
        description=(
            "Results carrying at least one usage row. Counted on truthiness, so it excludes both "
            "`usage=None` (no observation was made) and `usage=[]` (capture ran and attributed no roles) — "
            "read it as 'results with rows to sum', not as 'results where capture ran'."
        ),
    )


class RunSummary(EvalDocumentModel):
    """One run's compact digest — the levers it RAN at + its key telemetry.

    Feeds ``EvalAnalysis.run_index``. ``config`` carries each lever's effective value under the
    name its HOST declares — ``model``, ``retriever.top_k``, ``chunk_tokens`` — and the
    name's shape means nothing: a dot is part of a name a host chose, not a path. It names the
    same levers the campaign's coverage map does, so the two surfaces cannot disagree about which
    levers exist; ``config_provenance`` says how each was established. Not "what this run tuned":
    a run that tuned nothing still ran at values, and the arm that names no override is the
    control of every sweep — describing it as having no config is what let a report attribute an
    inner agent to the wrong model.

    **Every model this run ran at is named here, per role, or the reader supplies the missing
    one from somewhere else and gets it wrong.** Carrying the inner-agent model alone left a
    single-arm campaign's config naming a model the candidate never was, beside a ``candidate_model``
    naming the candidate — two true facts a reader with no role labels between them can
    only read as a contradiction, which is what one did.
    """

    run_id: str = Field(description="The run's id.")
    status: str = Field(description="EvalRun.status at read time.")
    created_at: str = Field(description="When the run was created (ISO-8601).")
    candidate_model: str = Field(min_length=1, description="The run's candidate model; a run is one arm.")
    k_runs: int = Field(
        ge=0,
        description=(
            "How many times this run was launched to repeat each test case — the run's own k. A property of this "
            "run alone: an arm measured by several runs holds more repeats per case than any one of them, so an "
            "arm's repeats per case are its cells' in `cell_measures`, never this."
        ),
    )
    config: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "The value each lever actually ran at, under the name its host declares — `model` is the "
            "CANDIDATE model, as distinct from a model an inner agent ran on under its own lever: "
            "different roles, never in disagreement. A key is absent for two different reasons, and "
            "`config_provenance` separates them: a lever whose value could not be established is here "
            "as `unknown` there and absent here, so 'we could not establish this' is never readable as "
            "a level; a lever the campaign never engaged with — declared by the host, never moved, never "
            "named by a launch, never recovered — is in neither, and says nothing about the experiment. "
            "The level `null` is a value the launch SET (it named the lever as null, stamped `overridden`), "
            "not a missing one."
        ),
    )
    config_provenance: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "How each lever's value was established: `overridden` (the run record names it, which for "
            "a launch-overlay lever means the launch chose it), `inherited` (nothing named it and the "
            "value was recovered from what the run observably did), or `unknown` (the lever applied but "
            "nothing pins its value). Keys are a superset of `config`'s — `unknown` levers appear here "
            "and not there."
        ),
    )
    n_results: int = Field(ge=0, description="Results the run produced.")
    n_errors: int = Field(ge=0, description="Results carrying any error (runner/judge/candidate/infra).")
    cost_usd: float = Field(
        ge=0.0,
        description=(
            "Summed cost_usd across the run's PRICED results — a FLOOR, not a reconciliation. A result whose "
            "spend went unpriced carries no cost_usd and is left out rather than counted as zero; "
            "`n_cost_unpriced` says how many. Read it as 'at least this much'."
        ),
    )
    n_cost_unpriced: int = Field(
        ge=0,
        description=(
            "Results whose spend could not be priced (a model call whose client reported no price) — real "
            "spend absent from `cost_usd`. Non-zero means `cost_usd` understates the run by an unknown amount."
        ),
    )
    prod_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "TOTAL production-replicating spend over the turns that measured one — candidate + "
            "inner_agent + external, with the judge and simulator measurement scaffolding excluded. A "
            "total, so it scales with how many results a run produced and is NOT comparable across runs "
            "of different size; compare `mean_prod_cost_usd`. It is also not what the configuration "
            "would cost in production: it sums dollars spent under whatever this run substituted or "
            "swept, and that gap has no reliable sign. `None` when no result measured one — unknown "
            "rather than free."
        ),
    )
    mean_prod_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Production-replicating spend per MEASURED turn — the comparable figure, and the one the "
            "measure registry names the reporting default. Its denominator is `n_prod_cost_usd`, never "
            "`n_results`: a result that measured nothing is absent from this mean rather than dragging it "
            "toward a zero nobody observed, which would rank the least-measured configuration cheapest. A "
            "result the harness faulted is absent too, as from every comparison cost."
        ),
    )
    n_prod_cost_usd: int = Field(
        default=0,
        ge=0,
        description=(
            "Results contributing to `prod_cost_usd` — its denominator, and the reason the pair "
            "travels together. A result with no usage decomposition, or one carrying a substituted "
            "delivery, is ABSENT from both rather than entering the sum as a zero: averaging an "
            "unobserved zero in would rank the least-measured config the cheapest. So is a call the model "
            "refused or errored on, which took no turn (`delivered_a_turn`), and a result the harness faulted, "
            "whose cut-short spend would let the rig make an arm look cheaper: every comparison cost leaves both "
            "out, while `cost_usd` (program spend) keeps them."
        ),
    )
    production_footing: ProductionFooting | None = Field(
        default=None,
        description=(
            "Which inputs this run held away from the subject's production configuration, read off the host's "
            "declarations — what `prod_cost_usd` and `mean_prod_cost_usd` were spent under. `moved` names each "
            "input the run set off production with its level, `unchecked` each lever whose departure could not "
            "be decided, `held` the inputs checked and found at production. Only an empty `moved` AND an empty "
            "`unchecked` say the cost was measured at production's configuration; anything in either means it "
            "was not, or may not have been. None when nobody checked: a summary assembled before the "
            "disclosure existed, or one built from a run read without its host payload."
        ),
    )
    measures: MeasureCollection = Field(
        default_factory=MeasureCollection, description="Scope-tagged measures over the run's results."
    )


class DeclaredLevelCoverage(EvalDocumentModel):
    """Whether one level the campaign DECLARED for an axis was run — the declaration's delta, by name."""

    display: str = Field(min_length=1, description="The declared level, as the declaration renders it.")
    content_hash: str = Field(min_length=1, description="The declared level's identity, which runs are joined on.")
    state: Literal["ran", "not_run", "undetermined"] = Field(
        description=(
            "ran = some member run sat at this level. not_run = declared and never run: every member run's "
            "level on the axis is established and none is this one, so the comparison it was declared for "
            "was not made. undetermined = no run is known to sit here, but some run's level on the axis "
            "could not be established (a run that inherited the subject's own setting may be at it), so "
            "'not run' cannot be claimed."
        )
    )


class DeclaredCellCoverage(EvalDocumentModel):
    """One combination of declared levels (a cell), and whether it ran, was skipped on purpose, or is missing."""

    levels: dict[str, str] = Field(
        description="Each declared axis's level in this cell, as the declaration renders it."
    )
    state: Literal["ran", "not_run", "skipped_by_design", "undetermined"] = Field(
        description=(
            "ran = some run sat at every one of these levels. skipped_by_design = the design left it out on purpose "
            "(its crossing, or skipped_cells) and it did not run: no gap. not_run = the design meant to run it and no "
            "run did: a gap. undetermined = meant to run, no run is known to sit here, but some run's level on an "
            "axis could not be established."
        )
    )


class DeclaredCrossing(EvalDocumentModel):
    """Every cell of a design that says which combinations of its levels it meant to run."""

    crossing: Literal["full", "star"] | None = Field(
        description="The design's declared crossing; None where only skipped_cells were named (full less those)."
    )
    n_cells: int = Field(ge=0, description="Every combination of the declared levels.")
    n_ran: int = Field(ge=0)
    n_not_run: int = Field(ge=0, description="Meant to run and never did: the gaps.")
    n_skipped_by_design: int = Field(ge=0)
    n_undetermined: int = Field(ge=0)
    cells: list[DeclaredCellCoverage] = Field(description="The cells, in declaration order; capped, gaps kept first.")
    cells_omitted: int = Field(default=0, ge=0)
    sentence: str = Field(description="What the crossing comes to, gaps and skips counted apart.")


class AliasedFactors(EvalDocumentModel):
    """Factors that moved in lockstep: every one splits the runs into the same groups, so no comparison separates them."""

    factors: list[str] = Field(description="The factors, sorted; two or more.")
    n_runs: int = Field(ge=0, description="The runs whose levels of every one of these factors were read.")
    n_levels: int = Field(ge=2, description="How many groups each of them splits those runs into.")
    sentence: str = Field(description="The one sentence that names the group; quote it rather than a factor apiece.")


class FactorPairCell(EvalDocumentModel):
    """One combination of two factors' levels, and how many runs sat at it."""

    row_level: str
    column_level: str
    n_runs: int = Field(ge=0)
    status: Literal["ran", "not_run"] = Field(
        description="ran = some run sat at both levels; not_run = none did, a hole in the design."
    )


class FactorPairPivot(EvalDocumentModel):
    """Two co-varying factors crossed: every combination of their observed levels, the unrun ones as ``not_run``."""

    row_factor: str
    column_factor: str
    row_aliases: list[str] = Field(
        default_factory=list, description="Factors in the row factor's lockstep group, which this pivot stands for too."
    )
    column_aliases: list[str] = Field(
        default_factory=list,
        description="Factors in the column factor's lockstep group, which this pivot stands for too.",
    )
    cells: list[FactorPairCell] = Field(description="The full cross of observed levels, row-major in level order.")


#: What the pair scan cannot see, stated wherever its result is.
INTERACTION_ALIASING_UNCHECKED = (
    "Only pairs of factors are checked: a factor that moved with a combination of two others (C = A⊕B, an "
    "interaction) is not detected, so it is neither grouped nor shown."
)


class FactorPairScan(EvalDocumentModel):
    """Every pair of varying factors checked for co-varying, with a pivot for each co-varying pair outside a group.

    Two factors co-vary when one never moves while the other holds still: across the runs at any one level of
    either, the other takes a single level. Every pair is checked; only the pivots are bounded, so the count says
    how complete the short list is.
    """

    factors: list[str] = Field(
        description=(
            "Every factor that varied: the swept levers and candidate model the coverage reads, and each apparatus "
            "dimension every run recorded."
        )
    )
    n_pairs_examined: int = Field(ge=0)
    n_covarying: int = Field(ge=0, description="Pairs that co-vary, inside a lockstep group or not.")
    n_covarying_in_groups: int = Field(ge=0, description="Of those, the pairs whose two factors share a group.")
    pivots: list[FactorPairPivot] = Field(default_factory=list)
    pivots_omitted: int = Field(default=0, ge=0, description="Pivots past the cap, fewest holes first.")
    completeness: str = Field(description="The sentence stating what was examined and what is shown.")
    interaction_aliasing: str = Field(default=INTERACTION_ALIASING_UNCHECKED)


class LeverCoverageInput(EvalDocumentModel):
    """Structural coverage of one lever, as the bundle computes it.

    The generator copies it, field for field, into the stored
    :class:`~threetears.evals.kernel.campaign.LeverCoverage`: it reports how finely a
    lever was swept (``cells`` = distinct observed levels), how many samples inform
    it (``n`` = distinct results), the repeat floor (``k``), a scored-signal spread
    (``dispersion``, the composite SEM read via the core ``stats`` helper — never a
    new statistic), and a coarse ``status``. Nothing grades these into a confidence:
    carrying them is what makes coverage the analysis's spine.
    """

    name: str = Field(description="Lever name, as the host registered it.")
    levels: list[str] = Field(
        default_factory=list,
        description="Distinct observed values ('—' = ran without the override; 'null' = the launch set it to null).",
    )
    cells: int = Field(ge=0, description="Number of distinct observed levels — how finely the lever was swept.")
    k: int = Field(
        ge=0,
        description=(
            "Repeat depth binding this lever's comparison — min over levels of the best-replicated arm at that level, "
            "where an arm's repeats per case add up across the runs that measured the same cases."
        ),
    )
    n: int = Field(ge=0, description="Distinct results informing the lever's comparison.")
    dispersion: str = Field(
        description="Within-level composite spread (±mean SEM), or 'unscored' when no composite signal exists."
    )
    status: Literal["measured", "thin", "unswept"] = Field(
        description="measured (swept + k>=3) | thin | unswept (1 level)."
    )
    cohort_scope: Literal["control_referenced", "campaign"] = Field(
        default="campaign",
        description=(
            "Which run set every number on this row was computed over. 'control_referenced' = the "
            "runs of the declared control arm plus those of the arms that moved THIS lever, so "
            "n/k/dispersion/confounded_by describe a contrast. 'campaign' = every run, so they describe a "
            "marginal comparison averaged over whatever else moved — the case when no control resolved, and also when "
            "one did but no arm moved this lever. Structural rather than inferable: the two are the same shape and differ only in "
            "what they may be read as, so a consumer left to work it out from the design will read a "
            "marginal comparison as a controlled one."
        ),
    )
    confounded_by: list[Confound] = Field(
        default_factory=list,
        description=(
            "Everything else that varied across the runs behind this lever — other swept levers, run "
            "attributes that moved on their own, observed mechanisms that diverged between its levels, and a "
            "requested candidate model answered by more than one model. A "
            "comparison on this lever is marginal, not controlled, for each of these. Empty means nothing else "
            "moved."
        ),
    )
    declared_levels: list[DeclaredLevelCoverage] = Field(
        default_factory=list,
        description=(
            "On a DECLARED axis, every level the declaration names, in its order, each marked ran / not_run / "
            "undetermined. `levels` lists only what ran, so this is where a declared level that never ran is "
            "named — a gap in the design as run, distinct from a level nobody declared. Empty on an "
            "undeclared row."
        ),
    )
    cannot_be_an_arm: str | None = Field(
        default=None,
        description=(
            "Set only on a DECLARED axis the host cannot vary on purpose — an apparatus or label input, or a name "
            "it never registered: the host's own reason, with its remedy. Such an input never enters the variant "
            "key, so every run resolves to one arm on it and this row is unswept BY CONSTRUCTION, whatever the "
            "runs did — a design that cannot be met, never a sweep that did not happen. Null on every other row."
        ),
    )
    mechanism: MechanismCheck = Field(
        description=(
            "Whether the measure this lever declares it acts on separated across its levels, over this row's "
            "cohort. inert = no measurable evidence the lever acted on its mechanism, which is a different finding "
            "from a lever that took effect and changed nothing; unchecked = nobody can say, so a null on this lever "
            "is mechanism-unverified."
        )
    )


class TelemetryRollup(EvalDocumentModel):
    """Campaign-wide descriptive telemetry — the trustworthy-signal layer.

    Cost fields come straight from ``reporting.compute_program_budget`` (the authoritative
    spend lens, which excludes nothing); measures and tokens are descriptive
    aggregates this module computes because ``reporting`` provides none.
    """

    n_runs: int = Field(ge=0, description="Runs summarised.")
    n_results: int = Field(ge=0, description="Results across all runs.")
    n_errors: int = Field(ge=0, description="Results carrying any error.")
    total_cost_usd: float = Field(ge=0.0, description="program_budget total (all statuses, nothing excluded).")
    incomplete_cost_usd: float = Field(ge=0.0, description="Spend attributed to incomplete runs (program_budget).")
    unattributed_cost_usd: float = Field(ge=0.0, description="Spend with no run attribution (program_budget).")
    measures: MeasureCollection = Field(
        default_factory=MeasureCollection, description="Campaign-wide scope-tagged measures."
    )
    tokens: TokenRollup | None = Field(
        default=None, description="Campaign-wide token sums, or None if no usage recorded."
    )


class HeldFixedReading(EvalDocumentModel):
    """What the campaign declared held still, beside what its runs say about the apparatus.

    The declaration (:class:`~threetears.evals.kernel.declaration.ControlDeclaration`) is a claim
    about every run the campaign holds, and each run records whether its apparatus was set or found
    (:attr:`~threetears.evals.schema.models.EvalRun.apparatus_provenance`). The two use the same
    words, so they are compared value for value here, once, and a writer quotes the result rather
    than reading a declaration of `commissioned` over a campaign half made of captured sessions.
    """

    declared_stimulus: Literal["controlled", "uncontrolled"] | None = Field(
        default=None,
        description="The declared stimulus control, or None when the campaign declared no design.",
    )
    stimulus_reason: str = Field(
        default="", description="What varied instead, when the stimulus was declared uncontrolled; blank otherwise."
    )
    declared_apparatus: ApparatusProvenance | None = Field(
        default=None,
        description="The declared apparatus control (commissioned | witnessed), or None when no design was declared.",
    )
    run_provenance: dict[str, ApparatusProvenance] = Field(
        default_factory=dict,
        description=(
            "Every resolved member run → the provenance it recorded. Read off the run, never the declaration; "
            "archived and unresolved members are absent because nothing here pools them."
        ),
    )
    contradicting_run_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Resolved member runs whose recorded provenance is not the declared apparatus, sorted. Empty when "
            "they agree or when nothing was declared — the second is a different fact, said in `disclosure`."
        ),
    )
    disclosure: str | None = Field(
        default=None,
        description=(
            "The sentence a writer quotes when the declaration and the runs disagree, the runs mix provenances "
            "with no declaration to say which was meant, or the stimulus was declared uncontrolled. None when "
            "there is nothing a reader needs warning about."
        ),
    )


class ShortCell(EvalDocumentModel):
    """A cell holding fewer repetitions than the declaration intended — a short run, stated per cell.

    ``intended_repetitions`` is per cell, so its shortfall is too: a campaign whose runs each delivered
    their whole matrix can still leave one arm short of the repetitions the design set out to buy,
    because the arm was launched at a smaller ``k`` or measured by fewer runs than its siblings.
    """

    variant_key: str = Field(min_length=1, description="The cell's variant coordinate.")
    apparatus_class_id: str = Field(min_length=1, description="The cell's apparatus coordinate.")
    intended: int = Field(ge=1, description="The declared `intended_repetitions`.")
    observed: int = Field(
        ge=1,
        description=(
            "The fewest times any one case ran in the cell (its `repeats_per_case_min`): the cell's weakest "
            "replication, which pooling more repeats of its other cases does not repair."
        ),
    )
    sentence: str = Field(min_length=1, description="The disclosure a writer quotes about this cell.")


class CellCoordinate(EvalDocumentModel):
    """A cell named by its two coordinates and nothing else — an entry in a list of cells a fact holds for."""

    variant_key: str = Field(min_length=1, description="The cell's variant coordinate.")
    apparatus_class_id: str = Field(min_length=1, description="The cell's apparatus coordinate.")


class MeritTier(EvalDocumentModel):
    """One axis of the declared merit priority, and the bars that give verdicts on it."""

    axis: MeritAxis = Field(description="The merit axis, in the campaign's declared priority order.")
    bar_measure_ids: list[str] = Field(
        default_factory=list,
        description=(
            "The adjudicated bars on this axis, in `bar_adjudications` order. Empty is a real state: the "
            "campaign ranked this axis and holds itself to no bar on it, so no verdict can decide on it."
        ),
    )


class QuestionScope(EvalDocumentModel):
    """Which verdicts bear on one declared question — read off the axes the question names."""

    question_id: str = Field(min_length=1, description="The live question's id.")
    merit_axes: list[MeritAxis] = Field(
        min_length=1,
        description="The axes the question names, ranked by `merit_priority` first and then in the question's own order.",
    )
    bar_measure_ids: list[str] = Field(
        default_factory=list,
        description="The adjudicated bars on those axes, in the order of `merit_axes`.",
    )
    unbarred_axes: list[MeritAxis] = Field(
        default_factory=list,
        description=(
            "Axes the question names that no adjudicated bar is on, in the order of `merit_axes`. An answer on "
            "one of these rests on the cell measures alone, with no verdict to quote."
        ),
    )


class VerdictOrder(EvalDocumentModel):
    """The order verdicts are read in, as the campaign declared it — never as the writer would choose.

    ``merit_priority`` is the tie-break when no bar picks a winner, and an empty one means the campaign
    stated no preference, which the analysis must not invent. So the ranking is arithmetic over the
    declaration and the bars' own axes, done here, and a bar on an axis the priority does not name is
    listed as unranked rather than placed by guess.
    """

    merit_priority: list[MeritAxis] = Field(
        default_factory=list,
        description="The declared priority, strongest first. Empty = no stated preference.",
    )
    tiers: list[MeritTier] = Field(
        default_factory=list, description="One per axis of `merit_priority`, in that order. Empty when it is."
    )
    unranked_bar_measure_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Adjudicated bars whose axis the priority does not name, or which serve no axis, in "
            "`bar_adjudications` order. Every bar when no priority was stated."
        ),
    )
    questions: list[QuestionScope] = Field(
        default_factory=list,
        description="One per live declared question that names at least one merit axis, in declaration order. An unscoped question is absent.",
    )


#: What one comparison in a family came to, read off its ADJUSTED p's. ``equivalent`` = shown inside the
#: measure's declared margin by an equivalence test; ``untested`` = no test could run (fewer than two cases a
#: side), which is neither a separation nor its absence.
ComparisonVerdict = Literal["improved", "regressed", "equivalent", "not_separated", "untested"]


class ComparedCell(EvalDocumentModel):
    """One side of a family comparison: a cell, and the per-case values its test read."""

    variant_key: str = Field(min_length=1, description="The arm's variant — half of the cell.")
    apparatus_class_id: str = Field(min_length=1, description="The rig — the other half of the cell.")
    n_cases: int = Field(
        ge=0,
        description=(
            "The cases the test read on this side, one per-case mean each: the cases both cells ran when the test "
            "is paired, every case carrying the reading otherwise."
        ),
    )
    mean: float | None = Field(
        default=None,
        description=(
            "Mean of those per-case values — over the cases the test read, which is not the cell's own mean when "
            "`n_left_out` is above 0. None when n_cases is 0."
        ),
    )
    n_left_out: int = Field(
        default=0,
        ge=0,
        description=(
            "Cases this side carried the reading on that the test did not read, because the other side did not run "
            "them: a paired test reads only the cases both ran. 0 when the test read every case this side has."
        ),
    )


class FamilyComparison(EvalDocumentModel):
    """One contrast against the control on one reading, tested and corrected within its family."""

    reading: ReadingKind = Field(description="Whether `name` is a measure or a judged dimension.")
    name: str = Field(min_length=1, description="The measure or judged dimension compared, as the bundle spells it.")
    higher_is_better: bool = Field(description="Which way is better on this reading — how `verdict` reads `delta`.")
    control: ComparedCell = Field(description="The control arm's cell under this rig.")
    contrast: ComparedCell = Field(description="The contrast arm's cell under the same rig.")
    delta: float | None = Field(
        default=None,
        description=(
            "contrast mean minus control mean, in the reading's unit, over the cases the test read (each side's "
            "`mean`). None when either side is empty."
        ),
    )
    interval: tuple[float, float] | None = Field(
        default=None,
        description=(
            "The interval on `delta` at the family's `interval_level`, from the same test as `p_raw`: simultaneous "
            "over the family, so every interval in it covers its true difference together at least 95% of the time. "
            "A separation needs it to exclude zero. Where the values have no spread (every shared case moved by one "
            "amount, or each side constant) it is the bounded test's on the declared range (`basis` `bounded`). "
            "None when no test ran. A bundle assembled before 0.66 stated none where the values had no spread."
        ),
    )
    hedges_g: float | None = Field(
        default=None,
        description=(
            "The standardized effect, Hedges' g (bias-corrected Cohen's d): over the SD of the per-case differences "
            "when paired, the pooled SD when not. None when no test ran, at two paired cases, where no unbiased "
            "estimate exists, and where the values have no spread, where no finite effect size exists."
        ),
    )
    test: Literal["paired", "unpaired"] | None = Field(
        default=None,
        description=(
            "`paired` = over the cases both cells ran; `unpaired` = over each cell's per-case values, when they share "
            "fewer than two cases. `basis` says which test. None when no test ran."
        ),
    )
    basis: Literal["t", "bounded", "identical"] | None = Field(
        default=None,
        description=(
            "Which test produced `p_raw`: `t` = a paired t-test, or Welch's t statistic on Hsu's conservative "
            "min(n) − 1 degrees of freedom when unpaired; `bounded` = the bounded test by betting on the reading's "
            "declared range, where the values have no spread (every shared case moved by one amount, or each side "
            "constant), valid for the mean at every n; `identical` = identical values on both sides, p 1. None when "
            "no test ran, and on a bundle assembled before 0.66, whose no-spread rows read the exact sign-flip test, "
            "a test of symmetry rather than of the mean, and keep the verdicts they were assembled with."
        ),
    )
    p_raw: float | None = Field(
        default=None,
        description=(
            "The test's own two-sided p, before correction. Kept for audit; never the figure a verdict rests on, "
            "and withheld from the analysis writer. None when no p exists."
        ),
    )
    p_adjusted: float | None = Field(
        default=None,
        description="The Holm-adjusted p within this comparison's family — the one figure a separation rests on.",
    )
    equivalence_margin: float | None = Field(
        default=None,
        description=(
            "The measure's margin the equivalence test ran against, in its unit: its declared `materiality_threshold`, "
            "or the margin its runs declared (`margin_source` says which). None when it has none, for a judged "
            "dimension, and for an unpaired test: then no equivalence test ran and the verdict cannot be `equivalent`."
        ),
    )
    margin_source: Literal["measure", "run"] | None = Field(
        default=None,
        description=(
            "Where the measure's margin came from: `measure` = its descriptor's `materiality_threshold`, declared "
            "by the host; `run` = a margin every member run declared at launch on a core rate measure "
            "(`run_margins`). It is the margin `materiality` reads, and `equivalence_margin` when an equivalence "
            "test ran. None when the measure has no margin, and on a judged dimension."
        ),
    )
    equivalence_p_raw: float | None = Field(
        default=None,
        description=(
            "The paired TOST p against ± `equivalence_margin` (the larger one-sided p), before correction. Kept for "
            "audit and withheld from the writer, as `p_raw` is. None when no equivalence test ran."
        ),
    )
    equivalence_untested_reason: str | None = Field(
        default=None,
        description=(
            "Set when the measure declares a margin and no `value_range`: no equivalence test ran, because none "
            "holds its error rate on a mean with no declared range, so this comparison can never read "
            "`equivalent`. The sentence names the remedy: declare value_range. None otherwise, including where no "
            "margin is declared."
        ),
    )
    equivalence_p_adjusted: float | None = Field(
        default=None,
        description="The TOST p adjusted within the family — the one figure an `equivalent` verdict rests on.",
    )
    verdict: ComparisonVerdict = Field(
        description=(
            "improved or regressed when `p_adjusted` is below the family's alpha and `interval` excludes zero, in the "
            "direction `delta` moved on this reading; else equivalent when `equivalence_p_adjusted` is below it; else "
            "not_separated, which says nothing about whether the arms differ; untested when no test could run."
        )
    )
    untested_reason: str | None = Field(
        default=None, description="Why no test could run, when `verdict` is untested. None otherwise."
    )
    not_separated_reason: str | None = Field(
        default=None,
        description=(
            "Set when `verdict` is not_separated with no test run: every shared case moved by the same amount (or "
            "each side is constant) on a reading that declares no `value_range`, where no test of the mean can call "
            "the gap, or a value lies outside the declared range. The sentence names the remedy. None otherwise, and "
            "on a bundle assembled before 0.66."
        ),
    )
    materiality: Materiality | None = Field(
        default=None,
        description=(
            "`delta` read against the measure's margin (`margin_source`: its declared materiality threshold, or its "
            "runs' declared margin): `immaterial` when it is smaller than the margin — too small to act on, whatever `verdict` says about its separation — "
            "and `material` otherwise, including when no threshold is declared (a judged dimension declares none). "
            "No finding names a winner on an immaterial delta. None when `delta` is None."
        ),
    )
    mechanism_confounds: list[Confound] = Field(
        default_factory=list,
        description=(
            "What was observed, not set, to differ between the two sides of this contrast: each observed mechanism "
            "that diverged when they ran different candidate models (the covariate, the threshold it crossed and "
            "both models' values), and the served_model confound when one requested model id was answered by more "
            "than one model across the two. It qualifies the contrast and suppresses nothing. Empty when neither "
            "applies."
        ),
    )


class ComparisonFamily(EvalDocumentModel):
    """Every comparison one declared question could draw a verdict from, corrected as one family.

    Ten comparisons at α=0.05 find a "difference" by chance more often than not, so the readings a
    question asks about (each contrast against the control, on every reading on the question's merit
    axes) are adjusted together, and a verdict stands only on the adjusted p.
    """

    question_id: str | None = Field(
        min_length=1,
        description=(
            "The live declared question this family serves. None for the campaign-wide family a campaign declaring no "
            "question gets: every comparison it holds, on every reading on a merit axis, corrected as one — so a "
            "chance difference is no more a finding for a campaign that asked nothing than for one that asked."
        ),
    )
    merit_axes: list[MeritAxis] = Field(
        default_factory=list,
        description="The axes the question names. Empty = an unscoped question, whose family covers every axis.",
    )
    correction: Literal["holm"] = Field(
        default=MULTIPLE_COMPARISON_CORRECTION, description="The family-wise correction applied."
    )
    alpha: float = Field(default=SIGNIFICANCE_ALPHA, description="The family-wise error rate the verdicts hold.")
    family_size: int = Field(
        ge=0,
        description=(
            "How many comparisons carried a separation p and were corrected together — the m the adjustment "
            "divides by, and the most hypotheses that can be true at once when equivalence tests join them."
        ),
    )
    n_equivalence_tests: int = Field(
        default=0,
        ge=0,
        description=(
            "Equivalence tests corrected in the same family, one per paired comparison on a measure that declares "
            "a margin. A comparison's difference is either zero or at least its margin, never both, so the two "
            "tests of one comparison cannot both be wrong and the family's error stays at alpha over every verdict."
        ),
    )
    interval_level: float | None = Field(
        default=None,
        description=(
            "The level of each comparison's `interval`: 1 − alpha / family_size, Bonferroni's, so the family's "
            "intervals hold together at 1 − alpha. None when no comparison carried a p."
        ),
    )
    n_untested: int = Field(ge=0, description="Comparisons in the family that could run no test, so carry no p.")
    comparisons: list[FamilyComparison] = Field(
        default_factory=list,
        description="Ordered by reading, name, rig and contrast variant.",
    )
    disclosure: str = Field(min_length=1, description="The sentence a writer quotes about this family's correction.")


class MultipleComparisons(EvalDocumentModel):
    """The campaign's comparisons, one corrected family per live declared question — or one for the whole campaign."""

    families: list[ComparisonFamily] = Field(
        default_factory=list,
        description=(
            "One per live declared question, in declaration order; with no question declared, one campaign-wide "
            "family over every reading (`question_id` None)."
        ),
    )
    withheld: str | None = Field(
        default=None,
        description=(
            "Why there is no family at all — no control to compare a contrast against, so no separation between "
            "arms is tested and none may be claimed. None when families exist."
        ),
    )


class ReadingScope(EvalDocumentModel):
    """Which readings the campaign's declared questions asked about — and which it reads only exploratorily.

    A reading no question asked about can still be reported, as a lead: it was not looked for, so a
    pattern in it is the kind a reader finds in any data. Labelled where questions are declared, on the
    readings outside them only; where none are, the whole campaign is exploratory and that is said once
    (``disclosure``), never on every row — a label that fires on every row is one readers learn to skip.
    """

    questions_declared: bool = Field(
        default=False, description="Whether the campaign declares at least one live question."
    )
    exploratory_measures: list[str] = Field(
        default_factory=list,
        description=(
            "Measures in `measure_catalog` on no axis a live question names, sorted — exploratory: reportable as "
            "leads, never as a confirmed answer. Empty when no question is declared (see `disclosure`). A "
            "guardrail is never listed: it is held because it was declared one."
        ),
    )
    exploratory_dimensions: list[str] = Field(
        default_factory=list,
        description=(
            "Capability judged dimensions no live question covers (none names `quality`, and none is unscoped), "
            "sorted. Empty when no question is declared."
        ),
    )
    disclosure: str | None = Field(
        default=None,
        description=(
            "Set when the campaign declares no live question: the one sentence saying every finding is "
            "exploratory — and, when it declared no design at all (`declared_design` null), that it is an "
            "exploratory campaign whose design was inferred from the runs. None when questions are declared, "
            "where the two lists above carry the label."
        ),
    )


#: The one sentence a campaign with no live question gets, in place of a label on every reading.
NO_QUESTION_EXPLORATORY = (
    "This campaign declares no live question, so every finding it supports is exploratory: nothing was asked "
    "before the evidence was read, and a pattern found in it is a lead for a campaign that asks, not an answer."
)

#: The same sentence for a campaign that declared no design at all — an exploratory campaign, which is a valid
#: one: it has no question either, and the arms its comparisons read were inferred from the runs.
NO_DESIGN_EXPLORATORY = (
    "This campaign declares no design, so it is exploratory and its readings confirm nothing: nothing was asked "
    "before the evidence was read, a pattern found in it is a lead for a campaign that declares one, not an "
    "answer, and the design its comparisons read was inferred from the runs, not declared."
)


def exploratory_disclosure(declared: CampaignDesign | None) -> str | None:
    """The one sentence saying a whole campaign is exploratory, or None when a live question makes it confirmatory.

    One derivation for the bundle's ``reading_scope`` and the report built from a stored analysis, so the two
    cannot word it differently. An undeclared campaign (``declared is None``) is exploratory by definition — that
    is derived here, never stored as a flag of its own — and says so as such; a declared one with no live
    question gets the no-question line.

    Args:
        declared: The campaign's declaration, or None when it declared none.

    Returns:
        The disclosure, or None when the campaign declares at least one live question.
    """
    if declared is None:
        return NO_DESIGN_EXPLORATORY
    return None if declared.live_questions() else NO_QUESTION_EXPLORATORY


class AnalysisContextBundle(EvalDocumentModel):
    """The closed context bundle a generation prompt runs over.

    An in-memory intermediate — deliberately **not** a stored eval doc_type; only
    its :meth:`fingerprint` persists (on ``EvalAnalysis.generation.bundle_fingerprint``).
    It carries exactly these components: campaign keys, per-run summaries, the
    comparison + frontier lenses, a descriptive telemetry rollup, a per-lever
    coverage map, the campaign's derived design, and the subject's prior insights.
    Subject/task-agnostic: levers and findings are data; nothing branches on
    subject kind.

    **Three fields answer questions the per-lever lenses structurally cannot.** ``design``
    says which runs were meant to be read together — without it a comparison has to infer a
    structure, and inferring factorial over a star design pools cells that share no lever.
    ``apparatus_confounds`` says whether the rig held still at all, which has an answer even
    when the campaign compared nothing and every lever-attached list is therefore empty.
    ``incomplete_runs`` says which members did not finish, because they are pooled like any
    other and nothing else marks a truncated arm as truncated.

    **How much of its matrix a run delivered is a separate question from how it ended**, and
    three fields carry it because a status cannot answer it. A run that reaches the end of its
    matrix loop stamps ``completed`` however many cells actually produced a stored measurement,
    so ``incomplete_runs`` alone would pool a run that measured 11 of 15 exactly like a full arm.
    ``short_runs`` names the ones that came up short, whatever their status; and
    ``completeness_unknown_run_ids`` names the ones that cannot say, because a run carrying no
    completeness record has not finished, or had its record's write refused, rather than having
    delivered everything — absence is not zero, and reporting unknown as complete re-commits the
    error one layer down.

    ``held_fixed_reading`` was ``controls_reading`` until the declaration's ``controls`` was renamed
    ``held_fixed``; a frozen bundle (a reporter case's) carrying the old key reads it under the new one.
    """

    __retired_fields__: ClassVar[dict[str, str | None]] = {"controls_reading": "held_fixed_reading"}

    # The bundle's shape version. It reaches `fingerprint()`, so bump it whenever a fingerprinted
    # field is added, renamed or removed — and whenever a PACKAGE change moves a fingerprinted VALUE
    # over unchanged evidence: a core dimension joining the apparatus partition, a new ordering, a
    # different rendering of what the writer is shown. Otherwise a re-assembled bundle that only
    # changed shape reads as evidence that moved, which is the one thing this number separates. A
    # HOST editing its own declarations is not a reason to bump it, and a host has no way to: that
    # move is carried by `host_declarations_digest`, derived from the declarations themselves. An
    # A/B set spanning a bump must be read as spanning it. Why each earlier version moved is in
    # this file's history.
    schema_version: int = Field(
        default=47, ge=1, description="Bundle-shape version, for future evolution + fingerprint clarity."
    )

    # --- Campaign keys ---
    campaign_id: str = Field(description="The campaign this bundle summarises.")
    subject_id: str = Field(description="The analysed subject's stable id.")
    subject_kind: str = Field(
        default="",
        description=(
            "Discriminator; data, never a code branch. Carried through from the campaign, and "
            "blank when the campaign declared no kind — never defaulted to the host's usual kind."
        ),
    )
    behavior: str = Field(description="Which aspect is under test, e.g. 'extraction'.")
    template_id: str | None = Field(default=None, description="Referenced eval_template (the Behavior), or None.")
    scope_id: str = Field(
        description=(
            "Storage scope the member runs were loaded from. The engine assigns it no meaning — nothing "
            "branches on it and no lens reads it — but it is a field like any other, so its VALUE is in "
            "`fingerprint()`'s pre-image."
        )
    )
    run_ids: list[str] = Field(default_factory=list, description="Resolved member-run ids (sorted).")
    unresolved_run_ids: list[str] = Field(
        default_factory=list,
        description="Campaign run_ids not found in this scope — an honest gap, not dropped silently.",
    )
    archived_run_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Member runs an operator archived, held out of every lens below. Reported separately from "
            "unresolved_run_ids because the two absences mean opposite things: unresolved is a lookup "
            "that failed, archived is a curation that succeeded, and pooling them would make a "
            "deliberate exclusion read as a missing-data problem."
        ),
    )
    model_versions: dict[str, str] = Field(
        default_factory=dict, description="Distinct models by role (candidate/judge/simulator)."
    )
    window: CampaignWindow | None = Field(
        default=None, description="Derived [start, end] from member-run created_at, or None."
    )

    # --- Computed lenses + derived views ---
    run_summaries: list[RunSummary] = Field(default_factory=list, description="Per-run digests (run_id order).")
    comparison: ComparisonSetsResult = Field(description="reporting.compute_comparison_sets over the member runs.")
    frontier: FrontierResult = Field(
        description="reporting.compute_frontier — empty (no subjects) when the data can't seat a ranking yet."
    )
    frontier_bar_withheld: str | None = Field(
        default=None,
        description=(
            "Set when the frontier lens was given no bar: the campaign's effective bar (declared, else registered) "
            "is passed to it only when one names `pass_hat_k`, the measure it ranks on, and `frontier.bar` is then "
            "set and each point's `bar_decision` names the variants below it. Otherwise its verdict is withheld "
            "and each subject's `n_cleared_bar` is a default of 0 rather than a count of arms that failed "
            "anything — quoting it as one reports a comparison nobody made. The sentence ends with why: which bars "
            "exist and on what measure, or why the one on pass^k could not be passed. The bars this campaign is "
            "held to are adjudicated in `bar_adjudications`."
        ),
    )
    telemetry: TelemetryRollup = Field(description="Campaign-wide descriptive telemetry.")
    coverage: list[LeverCoverageInput] = Field(
        default_factory=list, description="Per-lever structural coverage map (the analysis's spine)."
    )
    scope_divergences: list[ScopeDivergence] = Field(
        default_factory=list,
        description=(
            "Lever changes where the whole-run measure moved by a different amount than the isolating measure, the "
            "difference itself tested and Holm-corrected within the lever — each one is a finding. An empty list "
            "claims no agreement: a pair whose test did not separate is not_separated, and one that could not be "
            "tested is counted in divergences_untested."
        ),
    )
    divergences_omitted: int = Field(
        default=0,
        ge=0,
        description="Gated divergences beyond the reporting cap, dropped weakest-first. Stated so a short list is not read as a complete one.",
    )
    divergences_tested: int = Field(
        default=0,
        ge=0,
        description="Whole-and-part pairs across every lever whose divergence test carried a p, published or not.",
    )
    divergences_untested: int = Field(
        default=0,
        ge=0,
        description=(
            "Whole-and-part pairs whose divergence could not be tested: fewer than two cases carrying both measures "
            "on a side, or a remainder (whole minus part) that moved by the same amount on every case where the two "
            "measures do not both declare value_range, so no test of the mean can call it; declare value_range on "
            "both for the bounded test to read it. Nothing is known of them. A bundle assembled before 0.66 read a "
            "remainder with no spread by the exact sign-flip test, a test of symmetry rather than of the mean."
        ),
    )
    declared_design: CampaignDesign | None = Field(
        default=None,
        description=(
            "What the campaign SET OUT to do, carried whole rather than reduced to its control. It is the "
            "denominator every completeness claim needs: 'the answer addressed every declared axis' and 'each "
            "declared question got exactly one resolution' are both uncheckable without it, and inferring the "
            "declaration from `design` below would make coverage a description of whatever ran. None means the "
            "campaign declared nothing — an exploratory campaign, as `reading_scope.disclosure` says — which is a "
            "different fact from declaring nothing to sweep."
        ),
    )
    design: RealizedDesign = Field(
        default_factory=RealizedDesign,
        description=(
            "What kind of experiment this is, inferred from the runs — never a declaration, and never to be "
            "called declared: the run carrying the declared control, each cell's moved levers, and "
            "whether the design is one-factor-at-a-time. Read it before any comparison: it says which "
            "runs were meant to be read together, which a curated bag of run ids cannot. What the campaign "
            "declared is `declared_design`; where that is null, this is the only design there is, and it is "
            "inferred from the runs."
        ),
    )
    incomplete_runs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Run id → status, for every resolved member run that did not reach 'completed'. Those runs "
            "ARE pooled into every lens here — unlike an archived run, nothing curated them out — so a "
            "cancelled or budget-stopped run contributes however many cases it got through, as though "
            "it were a full arm. Empty means every run reached a 'completed' status — NOT that every "
            "run delivered its whole matrix, which is what short_runs answers."
        ),
    )
    short_runs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Run id → the disclosure sentence for every resolved member run that delivered fewer cells "
            "than it promised, whatever status it ended on. A run that reaches the end of its matrix "
            "loop stamps 'completed' however many of its cells produced a stored, non-infra-excluded "
            "measurement, so a degraded run is invisible to incomplete_runs and is pooled into every "
            "lens here as though it were a full arm — its pass^k resting on a denominator its siblings "
            "do not share. The sentence is the same one every run-level surface renders."
        ),
    )
    completeness_unknown_run_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Resolved member runs carrying no completeness record, so whether they came up short is "
            "unknown rather than answered. A record is written on every terminal run, so absence means "
            "every attempt to write it was refused, or the run has not reached a terminal state yet — campaign membership carries no status guard, so a campaign "
            "analysed mid-sweep puts a still-running member here. Reported separately "
            "from short_runs because absence is not zero: treating these as complete would state as "
            "fact the very thing that was not recorded."
        ),
    )
    short_cells: list[ShortCell] = Field(
        default_factory=list,
        description=(
            "Every cell holding fewer repetitions than the declaration's `intended_repetitions`, ordered by "
            "(variant_key, apparatus_class_id), each with the sentence to quote. A per-cell shortfall, which "
            "`short_runs` cannot see: every run can deliver its whole matrix and an arm still hold fewer "
            "repetitions than the design set out to buy. Counted by each cell's least-repeated case. Empty when "
            "every cell met the intention — and empty too when the declaration states none, which makes a "
            "shortfall undetectable rather than zero (`declared_design.intended_repetitions` is null then)."
        ),
    )
    cost_unmeasured_cells: list[CellCoordinate] = Field(
        default_factory=list,
        description=(
            "Every cell where no turn the candidate took observed spend — no usage row in its cost roles carried "
            "dollars — ordered by (variant_key, apparatus_class_id). Each result there stores a `cost_usd` of 0 "
            "that is the sum of nothing, not a measured $0, so the cell carries no `cost_usd` reading: nothing "
            "charts it or tests it against the control. A cell with a measured $0 (a row carrying 0 dollars) is "
            "not listed. A result whose spend went unpriced is not listed either: its cost is unknown, which "
            "`cost_usd` null already says. Nor is a cell where no result took a turn, which `all_failed_cells` lists."
        ),
    )
    cost_unmeasured: str | None = Field(
        default=None,
        description=(
            "The sentence to quote about `cost_unmeasured_cells` — that cost was not measured there, and why a $0 "
            "would mean nothing. None when every cell observed spend somewhere."
        ),
    )
    all_failed_cells: list[CellCoordinate] = Field(
        default_factory=list,
        description=(
            "Every cell where no result the harness did not fault took a turn — the candidate's model refused or "
            "errored on every call — ordered by (variant_key, apparatus_class_id). The cell carries no cost or "
            "latency reading: a refused call's round trip and its spend describe no turn, and read as one they "
            "made a refusing arm the fastest and cheapest. Its failures still count against it in every rate, bar "
            "and judged score. Not 'unmeasured': the arm was measured, and every result failed. A cell whose "
            "every result failed but whose turns ran (its budget ended them) is not listed: it has their cost and "
            "latency, and `n_candidate_failed` says they all failed."
        ),
    )
    all_failed: str | None = Field(
        default=None,
        description=(
            "The sentence to quote about `all_failed_cells` — that every result there failed, so there is no cost "
            "or latency to read. None when every cell delivered a result."
        ),
    )
    latency_contended_cells: list[CellCoordinate] = Field(
        default_factory=list,
        description=(
            "Every cell holding a result whose latency was read while other cells or runs executed beside it "
            "(`execution_mode` `concurrent`: its launch did not declare latency under test, or another run "
            "executed beside it), ordered by (variant_key, apparatus_class_id). That latency is left out of every "
            "reading in this bundle — the cell measures, the contrasts and their Holm family, the bars, the "
            "frontier's latency axis, the mechanism and scope lenses — so a cell's latency, where it has one, is "
            "read only from its results measured serially. Every other measure of those results stands."
        ),
    )
    latency_contended: str | None = Field(
        default=None,
        description=(
            "The sentence to quote about `latency_contended_cells` — that latency read under concurrency is not "
            "compared, and how much was left out. None when every latency the campaign holds was read serially, "
            "or it holds none."
        ),
    )
    held_fixed_reading: HeldFixedReading = Field(
        default_factory=HeldFixedReading,
        description=(
            "What the campaign declared held fixed beside the provenance every resolved run recorded, compared value for value, "
            "with the sentence to quote when they disagree, when runs mix commissioned and witnessed apparatus "
            "with nothing declared, or when the stimulus was declared uncontrolled. Commissioned and witnessed "
            "observations never share a cell, so a mixed campaign has separate cells for them."
        ),
    )
    judge_agreement: JudgeAgreement = Field(
        default_factory=JudgeAgreement,
        description=(
            "How the judge's scores agreed with people's calibration ratings of the same results, per judged "
            "dimension, judge model and judge config: n, distinct results, exact agreement, Cohen's kappa and, on 1-5 "
            "dimensions, quadratic-weighted kappa, each pooled over people by result — over every resolved member run's results. A dimension absent here is uncalibrated: "
            "nobody rated it, so an absolute claim about it rests on the judge alone. Ratings that could not be "
            "paired with a judge score are listed with why."
        ),
    )
    judge_self_agreement: JudgeSelfAgreement = Field(
        default_factory=JudgeSelfAgreement,
        description=(
            "How the judge's repeated scores agreed with its own first scores of the same evidence, per judged "
            "dimension, judge model and judge config, read exactly as `judge_agreement` is (n, distinct results, "
            'exact agreement, kappa, weighted kappa, a "can\'t tell" repeat counted as a disagreement) — over every '
            "resolved member run's results. Empty when nothing was repeated: the judge's "
            "consistency is then unmeasured. Repeats that could not be paired are listed with why."
        ),
    )
    judge_evidence_tiers: list[JudgeEvidenceTier] = Field(
        default_factory=list,
        description=(
            "The evidence tier of each judge's readings on each judged dimension — a judge being a served model "
            "and a judge config — decided by code from `judge_agreement` and `judge_self_agreement`, each criterion on "
            "confidence bounds for its agreement and never the point estimate: `calibrated` (the one-sided 95% lower "
            "bound on agreement with "
            f"people at or above {format_number(CALIBRATION_MIN_AGREEMENT)}, over at least {CALIBRATION_MIN_RESULTS} "
            "distinct results), `separation` (that bound on agreement with its own repeats at or above "
            f"{format_number(SEPARATION_MIN_AGREEMENT)}, over at least {SEPARATION_MIN_RESULTS} distinct results), "
            "`incidental` (both upper bounds below their bars), or `undetermined` (not shown either way: too few "
            "results, or bounds across a bar). Each entry carries both criteria, their bounds, and how many more "
            "results each needs to be decided (`results_needed`). Every "
            "judged reading in `judged_measures` and `cell_measures` carries the tier of the judges behind it; a "
            "finding citing one stands on it. An entry carrying `from_profile` took its tier and criteria from a "
            "stored judge profile — this campaign's own evidence decided no tier, and a judge campaign's measurement "
            "of this very judge and criterion decided one: say so when citing it, with when and on how many frozen "
            "cases the profile was measured, never as this campaign's own measurement."
        ),
    )
    inter_judge_agreement: InterJudgeAgreement = Field(
        default_factory=InterJudgeAgreement,
        description=(
            "How a second judge's scores of the member runs' stored evidence agreed with their judge's, per judged "
            "dimension, first judge and second judge: n, distinct results, exact agreement, and kappa "
            "(quadratic-weighted on 1-5, unweighted on pass/fail) with its confidence bounds — read exactly as "
            "`judge_agreement` is. An undefined kappa says why, never 0. Empty when no second judge was asked. Each "
            "judged dimension's rows are also on its `judged_measures` entry."
        ),
    )
    judge_drift: JudgeDrift = Field(
        default_factory=JudgeDrift,
        description=(
            "How far each judged dimension's scores moved when a second judge re-scored the member runs' stored "
            "evidence: the movement over cases, its interval at 1 − α/m over the dimensions, and separated (exactly when "
            "that interval excludes 0) / not separated / untested. It detects movement between two judges, never which judge is right."
        ),
    )
    judge_change: JudgeChange = Field(
        default_factory=JudgeChange,
        description=(
            "Each judge the judged member runs were scored by (model pin, configs, temperature), the runs and arms "
            "under each, and the drift readings that span a change. When there is more than one, quote `sentence`: a "
            "judged difference between arms judged differently may be the judge's, and where `drift_links` is empty "
            "nothing measured how much."
        ),
    )
    goal_check_proofs: list[GoalCheckProofReading] = Field(
        default_factory=list,
        description=(
            "Per goal check the member runs graded: whether it was shown, at launch, to tell its outcomes apart "
            "(`proven`), or not (`unproven`: no control, or a proof recorded under an earlier rule (`stale`); "
            "`refuted`: a control it does not beat, or a check the grammar refused at launch (`refused`)). The check's pass "
            "rate is the measure `goal_state:<check>`; on any check not `proven` that rate may be what a candidate "
            "that did nothing would score, so it never reads as the behaviour measured."
        ),
    )
    multiple_comparisons: MultipleComparisons = Field(
        default_factory=MultipleComparisons,
        description=(
            "Each contrast tested against the control on every reading a live question asks about, per rig, with "
            "Holm correction inside each question's family: the family's size, each comparison's adjusted p and "
            "the verdict read off it. A separation stands only where its verdict says so."
        ),
    )
    guardrails: GuardrailReadings = Field(
        default_factory=GuardrailReadings,
        description=(
            "The guardrails — boundary judged dimensions and measures declared `guardrail`, what the candidate must "
            "not get worse on — each decided for every arm against the control on its own 95% interval: `held` "
            "(shown no worse than its margin), `breached` (shown worse) or `undecided`. Kept out of every "
            "comparison family and composite, so a capability gain cannot pay for a guardrail loss. An arm with a "
            "breached guardrail is not adopted; an undecided one is never safe, and is stated wherever the arm is "
            "recommended."
        ),
    )
    run_margins: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Margins on core rate measures (`accuracy`) that every member run declared alike at launch "
            "(`EvalRun.declared_margins`), by measure. A core descriptor declares no margin, so this is the one "
            "margin such a measure has here: each comparison on it runs its equivalence test against it, and names "
            "it (`equivalence_margin`, `margin_source` `run`). Read by the contrasts against the control only: a bar "
            "on the measure and the history's movements read its descriptor, which declares none. Empty when no run "
            "declared one."
        ),
    )
    launch_declarations: list[str] = Field(
        default_factory=list,
        description=(
            "One sentence per host measure that is read here otherwise than the reading host now declares it. Each "
            "measure is read on how its member runs were launched to read it (`EvalRun.declared_measures`: "
            "direction, merit axis, guardrail, margin, range), so a stored comparison's verdicts do not depend on who "
            "reads them; a sentence names the declaration read and the reading host's. Where the runs were launched "
            "under different declarations, no margin is read on the measure and the sentence says so. Empty when "
            "every measure is read as the reading host declares it, and for runs stored before launches recorded "
            "their declarations."
        ),
    )
    run_margins_withheld: str | None = Field(
        default=None,
        description=(
            "Why a margin some member runs declared on a core rate measure is read on none of its comparisons: the "
            "runs do not all declare it, or declare different ones, and a contrast between two arms read against a "
            "margin only one of them chose would be read against a margin nobody chose for the pair. None when no "
            "run margin was withheld."
        ),
    )
    reading_scope: ReadingScope = Field(
        default_factory=ReadingScope,
        description=(
            "Which readings no declared question asked about: exploratory, reportable as leads and never as "
            "confirmed answers. Where no question is declared, or no design, one sentence says every finding is "
            "exploratory."
        ),
    )
    verdict_order: VerdictOrder = Field(
        default_factory=VerdictOrder,
        description=(
            "The order verdicts are read in, as declared: the bars on each axis of `merit_priority`, strongest "
            "first, the bars on no ranked axis, and for each live question the bars on the axes it names. An "
            "empty priority means no stated preference — never rank the axes yourself."
        ),
    )
    measurement_windows: list[MeasurementWindow] = Field(
        default_factory=list,
        description=(
            "Each resolved member run's wall-clock measurement span, derived from that run's own "
            "scored_at stamps. The structured half of measurement_window_disclosure below, which "
            "states the same fact as the sentence a reader quotes — and the only form of it that "
            "answers for ONE PAIR, since which pairs overlapped is a question about these spans and "
            "the sentence names at most a capped few. A run that produced no result "
            "has no span and is simply absent: an honest gap, never a zero-length point at some "
            "arbitrary instant, which would compare against the others as though the run had been "
            "measured. Ordered by start, then end — the order the disclosure lists them in."
        ),
    )
    launch_disclosure: str | None = Field(
        default=None,
        description=(
            "Set when the member runs were NOT all started by one campaign launch — some came from "
            "different launches, or were started on their own. Arms started together share one "
            "concurrency slot, so their measurement windows overlap by construction; arms started apart "
            "overlap only by the scheduler's chance, and a difference between them can be when they ran "
            "as well as what they ran. None when every run shares one launch group, or there are fewer "
            "than two runs."
        ),
    )
    measurement_window_disclosure: str | None = Field(
        default=None,
        description=(
            "Set when AT LEAST ONE PAIR of member runs was measured over spans of wall-clock time that "
            "do not overlap — so anything that moved between those spans (a model revision, a provider's "
            "load, a rate limit) moved with the runs, and a difference between the two runs of such a "
            "pair is not attributable to the runs alone. The condition is existential and the sentence "
            "says which quantifier it earned: it reads as a universal only when every pair really is "
            "disjoint, and otherwise counts the disjoint pairs against the total and names the "
            "overlapping remainder. It also names each disjoint pair's gap magnitude, widest first. "
            "None when every pair overlaps or too few runs resolved a span. Built by the same helper "
            "runs_compare discloses from, on the same predicate, and listing every span rather than "
            "collapsing above the inline cap: the reader here cannot go and fetch the omitted ones. "
            "The PAIR list can still truncate — pairs grow quadratically where spans grow linearly — "
            "and says how many it did not name."
        ),
    )
    apparatus_confounds: list[Confound] = Field(
        default_factory=list,
        description=(
            "Apparatus dimensions that varied across the WHOLE campaign, scanned independently of any "
            "lever. The per-lever and per-divergence lists answer 'what else moved under this "
            "comparison'; this one answers 'did the measuring rig hold still at all', which has an "
            "answer even when the campaign compared nothing. Empty means the rig held — a claim only "
            "made about runs that recorded a value, since an unrecorded dimension is 'undecided' here "
            "as everywhere else."
        ),
    )
    aliased_factors: list[AliasedFactors] = Field(
        default_factory=list,
        description=(
            "Factors that moved in lockstep across the campaign: each group's factors split the runs identically, "
            "so no comparison separates them. Name a group once, by its sentence, never one factor at a time. "
            "Pairs only: see factor_pairs.interaction_aliasing."
        ),
    )
    factor_pairs: FactorPairScan | None = Field(
        default=None,
        description=(
            "Every pair of varying factors checked for co-varying, with a pivot (unrun combinations as not_run) "
            "for each co-varying pair outside a lockstep group, and how many pairs were examined. None on a "
            "bundle assembled before the scan existed."
        ),
    )
    declared_crossing: DeclaredCrossing | None = Field(
        default=None,
        description=(
            "Where the declared design says which combinations of its levels it meant to run: every cell, marked "
            "ran, not_run (a gap), skipped_by_design (left out on purpose: never a gap) or undetermined. None when "
            "there is no design or it says nothing about combinations, so no unrun combination is read either way."
        ),
    )
    arm_mechanisms: list[ArmMechanismReading] = Field(
        default_factory=list,
        description=(
            "Each arm's mean of every covariate read as an observed mechanism (today the candidate's reasoning "
            "share, `reasoning_ratio`), sorted by arm then covariate. An arm whose results measured none of it is "
            "listed with a null mean — said, not omitted. A run whose observations resolved no arm is in no entry. "
            "Where two levels of a comparison diverge in it, the comparison's `confounded_by` names it."
        ),
    )
    arm_served_models: list[ArmServedModel] = Field(
        default_factory=list,
        description=(
            "Which model the provider's responses named as having answered each arm's candidate calls, sorted by "
            "arm. An arm is keyed by the model id it ASKED for, and a floating alias can be answered by different "
            "models; a `pooled` arm's numbers are a mixture of models, and an `unrecorded` one cannot be said to "
            "be one model. An arm whose candidate left no usage row is absent. Wherever one requested id was "
            "answered by more than one model across a comparison's runs, the comparison names the "
            "`served_model:candidate` confound."
        ),
    )
    arm_production_footings: dict[str, PooledProductionFooting] = Field(
        default_factory=dict,
        description=(
            "Arm (variant key) -> what each of its runs set away from the subject's production configuration, read "
            "off the host's sweepable declarations: `runs` maps run id -> that run's footing (`moved` with levels, "
            "`unchecked`, `held`; null for a run nobody could check). An arm's production_replicating_cost — on its "
            "cells, contrasts and bars — is what production would spend only where every run moved nothing; where a "
            "run moved or left unchecked an input, it is the cost of those settings, with no reliable sign of "
            "error. Empty on a bundle assembled before it was read: nobody checked, never 'nothing moved'."
        ),
    )
    cell_model_version: int = Field(
        default=CELL_MODEL_VERSION,
        description=(
            "Which definition of a cell produced `cells`. Carried on the bundle so a fingerprint and "
            "the pooling behind it travel together — two bundles computed under different cell models "
            "are not comparable on any per-cell number, and a fingerprint alone cannot say so."
        ),
    )
    host_declarations_digest: str | None = Field(
        default=None,
        description=(
            "sha256 of the host's declared sweepables (each one's name, role and blank rule) and world dimension "
            "names — the declarations that partition observations into apparatus classes and arms "
            "(:func:`host_declarations_digest`). Derived from the declarations at assembly, never kept by hand, "
            "so a host that adds, removes or renames one moves it without bumping anything. Two bundles whose "
            "fingerprints differ while this and `schema_version` agree did not differ in those declarations. "
            "None on a bundle frozen before it was recorded, which says nothing about them."
        ),
    )
    cells: list[Cell] = Field(
        default_factory=list,
        description=(
            "Every (variant, apparatus class) that any observation landed in, with how many observations "
            "pooled there and over how many cases and repeats per case. THE unit of comparison: a cell is "
            "what pools, independent of which run produced it, so a re-run of a setting grows its "
            "observations instead of minting a second thing to compare. An arm measured under two rigs "
            "is two cells; `design` lists each arm once, with the runs that measured it."
        ),
    )
    variant_index: list[VariantIndexEntry] = Field(
        default_factory=list,
        description=(
            "One entry per keyed variant — the variant key and the resolved lever map it was digested "
            "from. The dimension table for `cells`, whose own coordinate is a digest that says two "
            "observations ran the same stack and refuses to say what that stack WAS. Without it nothing "
            "joins a cell, or the campaign's declared control, to a declared axis level, so an arm "
            "table derives every row as unresolved. An entry is described from the run's RECORDED lever "
            "map wherever the run carries one, so an IDENTITY_VERSION bump between a campaign and its "
            "analysis no longer costs a single description. A variant that has none — a run its host "
            "assembled without the launch, read by a build whose predicate no longer reproduces its key — "
            "is PRESENT carrying "
            "`levels_unavailable` rather than described by levels it never carried; it pooled either "
            "way, so leaving it out described it by omission. Where a host registers an open family's "
            "resolved surface as a lever and that surface's movement is explained by the swept members "
            "alone, an entry names the surface in `folded` and the members it swept in `swept`: name and "
            "compare the arm by `swept`, never by the folded surface's hash, which is the same change seen "
            "twice."
        ),
    )
    refused_merges: list[RefusedMerge] = Field(
        default_factory=list,
        description=(
            "Same-variant cells that did NOT pool, and which rule kept them apart. A refusal is "
            "evidence: a reader seeing two cells of k=3 where they expected one of k=6 is owed the "
            "reason, and only one of the three reasons is fixable by recording something."
        ),
    )
    refused_merges_omitted: int = Field(
        default=0,
        ge=0,
        description=(
            "Refused pairs beyond the reporting cap, dropped smallest-first (by the observations the two cells "
            "hold). Stated so a short list is not read as a complete one."
        ),
    )
    next_experiments: list[NextExperiment] = Field(
        default_factory=list,
        description=(
            "What recording one unrecorded apparatus dimension would buy, in units of k. Generated "
            "mechanically — no model is asked — because it is arithmetic over cells that already share "
            "a variant. This is what keeps a conservative merge from being pure refusal."
        ),
    )
    next_experiments_omitted: int = Field(
        default=0,
        ge=0,
        description=(
            "Recordings beyond the reporting cap, dropped least-gain-first (by the observations recording would "
            "add). Stated so a short list is not read as a complete one."
        ),
    )
    subject_key_instabilities: list[SubjectKeyInstability] = Field(
        default_factory=list,
        description=(
            "Subject keys and labels disagreeing about how many subjects there are — one key under two "
            "labels, or one label under two keys. A population-level warning, never a judgement about "
            "any single observation, and it never refuses: a reader groups on the LABEL because that is "
            "what is legible, and this says when the keys do not support that grouping."
        ),
    )
    measure_catalog: dict[str, MetricDescriptor] = Field(
        default_factory=dict,
        description=(
            "What each measure name MEANS — the registry descriptor, carried once per campaign rather "
            "than repeated on every run's summary. Covers every name in a MeasureCollection here (the run "
            "summaries', the telemetry rollup's and each cell's in cell_measures), which is exactly the "
            "RANKING surface. It does not cover the frontier lens's own fields, and it deliberately omits "
            "every judged dimension: absence from this catalog means 'not rankable', never 'not measured' — see judged_measures."
        ),
    )
    judged_measures: list[JudgedMeasure] = Field(
        default_factory=list,
        description=(
            "Every judged dimension any result was scored on, with its per-cell scores. Measured quality, "
            "kept off the ranking surface for the reason each entry states, and carried here so that "
            "exclusion is not read as the dimension never having been measured. Empty means no result "
            "carried a judged score — the one case where 'no quality was measured' is true. Sorted by name."
        ),
    )
    bar_adjudications: list[BarAdjudication] = Field(
        default_factory=list,
        description=(
            "Every bar this campaign is held to — its own declared bars, plus each registered incumbent for "
            "its behavior that no declared bar overrides — with a verdict per cell computed here, or the "
            "reason none exists. A verdict is decided by the cell's interval against the threshold less the "
            "measure's declared margin, never by its mean, and its `decision` is `cleared` (shown on the good "
            "side), `missed` (shown on the bad side), `undecided` (the interval straddles the line: neither a pass "
            "nor a failure), `no_interval` or `no_data`. Read verdicts from this; never recompute them."
        ),
    )
    cell_measures: list[CellFacts] = Field(
        default_factory=list,
        description=(
            "Everything measured in each cell, one entry per cell, the declared control's cells first as the "
            "reference, then every other arm alphabetically by name (an order that is not a ranking): "
            "every measure over the cell's non-faulted observations — the population every bar verdict is "
            "read over, so a value here and a verdict on the same cell describe the same observations — "
            "every judged dimension scored there, its replication, and the notes on its member runs. Read "
            "per-arm numbers from here rather than pooling run summaries, which mix arms a run co-ran and "
            "include the observations the harness faulted. A cost or latency measure is read over the turns the "
            "candidate took (population `delivered`): every failure counts against its arm in every rate, bar and "
            "judged score, but a call its model refused or errored on took no turn, and its round trip and spend "
            "are no turn's. A failure that took a turn — its budget ended it, its output cap cut it, its deadline "
            "struck mid-call — stays in cost and latency: that is what failing cost the arm. `n_candidate_failed` "
            "says how many failed and `n_no_turn` how many of them took no turn; a cell where no result took a "
            "turn carries no cost or latency reading at all and is listed in `all_failed_cells`."
        ),
    )
    time_axis: TimeAxis | None = Field(
        default=None,
        description=(
            "The runs placed in time, when they span two or more builds (the host's release label) or, failing that, "
            "two or more days: each position names its runs and carries every cell measured there, computed exactly "
            "as `cell_measures` is over that position's runs alone. Earliest first. A `date` axis states in "
            "`basis_reason` why it is not builds (naming any runs that recorded no build). A `timeseries` chart "
            "draws one reading across these positions; without a time axis no chart can draw time."
        ),
    )
    time_axis_withheld: str | None = Field(
        default=None,
        description="Why there is no time axis — what every run shared — or None when there is one.",
    )
    confound_catalog: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "What a change in each confound dimension DOES to a measurement, carried once per campaign "
            "rather than repeated on every lever that names it. Keyed by Confound.dimension; covers every "
            "dimension appearing anywhere in this bundle. A dimension's own entry is what makes it "
            "judgeable — 'a different rubric and case set' kills a composite and leaves a latency figure "
            "standing, and only the reason says which."
        ),
    )
    prior_insights: list[EvalInsight] = Field(
        default_factory=list,
        description=(
            "Subject-scoped prior insights (newest first), one per claim and at most the reporting cap of the "
            "newest. Every insight minted by an ARCHIVED analysis is left out and named in `retracted_insights` "
            "instead; what else is left out is counted in `prior_insights_omitted`."
        ),
    )
    prior_insights_omitted: int = Field(
        default=0,
        ge=0,
        description=(
            "Live prior insights the ledger holds that `prior_insights` does not carry: an older insight stating "
            "the same claim as a carried one, and every insight beyond the cap, oldest dropped first. Retracted "
            "insights are not counted here. Stated so a short list is not read as the whole ledger."
        ),
    )
    retracted_insights: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Insight id → the archived analysis that minted it, for every insight the ledger held that is NOT "
            "in `prior_insights` because its analysis was archived as shown false. Ids only, never the "
            "statement: a retracted claim carried here would be the retracted claim read again. Derived from "
            "the analysis's archived state at assembly, so un-archiving the analysis restores its insights. "
            "Empty is the ordinary case."
        ),
    )

    def fingerprint(self) -> str:
        """Return the sha256 of the bundle's canonical JSON — a stable A/B key.

        Routes through :func:`threetears.evals.schema.hashing.canonical_digest` (sorted
        keys, tight separators) so two structurally identical bundles produce
        identical bytes. Because the bundle carries no wall-clock value, identical
        (runs, results, insights) yield an identical fingerprint — the invariant
        the prompt-tuning A/B loop depends on.

        **Under one host profile**, which is a condition rather than a caveat: this assembly reads
        the profile it is handed at several points, ``measure_catalog`` among them (a measure whose
        vocabulary is a host tool's is described by the host that declares it), so the same runs
        assembled for a DIFFERENT host would fingerprint differently. Every path that reaches here
        names its profile — :func:`assemble_context_bundle` takes one — so this cannot vary within
        a host. It is stated because a reader comparing two fingerprints needs to know what
        "identical" is quantified over, and because the A/B loop's premise is that nothing but the
        prompt moved.

        Returns:
            A 64-character lowercase hex digest.
        """
        return canonical_digest(self.to_dict())


class BundleInspection(EvalDocumentModel):
    """A bundle handed to a reader instead of to a model — the read surface's payload.

    The bundle used to be assembled, hashed, sent to a paid generator and dropped,
    so the only post-hoc trace of what it *refused* to publish — a subtraction
    rejected as unsound, a swing withheld by the containment default, a divergence
    dropped by the reporting cap — was whatever the generator chose to write into
    prose. That made the disclosure of a deterministic judgement depend on the
    model's discretion, which is the exact dependency moving those judgements into
    the bundle was meant to remove. This is the projection that closes it.

    **Projected, never stored.** There is no second copy: the surface re-assembles
    from the same :func:`assemble_context_bundle` a generation calls, so no
    fingerprinted field moves, no migration exists, and nothing here has to be
    carried across an extraction. That constraint is load-bearing —
    :meth:`AnalysisContextBundle.fingerprint` is the invariant the prompt-tuning
    A/B loop rests on.

    **Which is why the fingerprint is compared rather than assumed.** A bundle
    re-assembled today is not automatically the bundle a generation ran over
    months ago: a member run can have been archived, or results deleted, since.
    Addressed by analysis id, this therefore carries the recorded
    ``generation.bundle_fingerprint`` beside the freshly computed one and states
    whether they agree. A ``False`` is itself the finding, not a failure of the read.

    **An insight minted after the generation is NOT one of those moves.** The re-assembly
    reads prior insights as of the instant the generation's bundle was assembled
    (``generation.bundle_assembled_at``) —
    the generation hashed its bundle before saving what it minted, and another analysis can mint
    while the provider call runs, so without that cutoff an analysis could carry insights it never
    read into its re-assembled input and fail to reproduce.

    **Archiving an analysis is one of the evidence moves, and it reaches OTHER analyses.** An
    insight minted by an archived analysis is retracted at assembly (see
    :func:`retracted_insights`), and the cutoff above does not undo that: retraction is read from
    the minting analysis's state now, because the archive is the claim that the insight was false
    all along. So archiving one analysis moves the re-assembly of every analysis that READ its
    insights, and un-archiving it moves them back.

    **What remains has three causes, and ``mismatch_cause`` names which.** The package's bundle
    shape moved (:attr:`AnalysisContextBundle.schema_version` differs from the one the generation
    recorded — a rename is enough), the host's declarations moved
    (:attr:`AnalysisContextBundle.host_declarations_digest` differs — a host added, removed or
    renamed a sweepable, which moves the apparatus partition without touching package code), or
    neither did and the evidence moved — a member archived, a result deleted, an insight the
    generation read since deleted or superseded, or the analysis that minted one archived. An
    analysis stored before :class:`~threetears.evals.kernel.campaign.GenerationProvenance`
    recorded both reads ``cannot_say``: a version that was never written is never read as the
    same one.

    Host-agnostic by construction: campaign, scope, fingerprint and bundle are
    all engine vocabulary, and nothing here branches on what the subject is.
    """

    campaign_id: str = Field(description="The campaign the bundle was assembled for.")
    scope_id: str = Field(description="The storage scope its member runs were loaded from.")
    fingerprint: str = Field(description="sha256 of the bundle AS ASSEMBLED NOW — recomputed, never read from a store.")
    analysis_id: str | None = Field(
        default=None,
        description="The analysis this inspection was addressed by, or None when addressed by campaign.",
    )
    recorded_fingerprint: str | None = Field(
        default=None,
        description=(
            "The addressed analysis's stored generation.bundle_fingerprint — what the paid call actually "
            "ran over. None when addressed by campaign, since no generation is being spoken about."
        ),
    )
    reproduces_generation: bool | None = Field(
        default=None,
        description=(
            "Whether the re-assembled bundle IS the one the addressed analysis was generated over "
            "(fingerprint == recorded_fingerprint). Prior insights are read as of the instant the "
            "analysis's bundle was assembled, so an insight minted afterwards — its own included — does not move this. "
            "False means the bundle below explains today's inputs and not that generation's; `mismatch_cause` says "
            "why. None when addressed by campaign. Stated rather than "
            "hedged in prose because a reader diagnosing a generator defect from its input has to "
            "know whether the input is the one that produced the defect."
        ),
    )
    recorded_bundle_schema_version: int | None = Field(
        default=None,
        description=(
            "The bundle schema version the addressed generation ran over, as its provenance recorded it. None "
            "when addressed by campaign, or when the analysis was stored before the version was recorded."
        ),
    )
    recorded_host_declarations_digest: str | None = Field(
        default=None,
        description=(
            "The host declarations digest the addressed generation ran over, as its provenance recorded it. None "
            "when addressed by campaign, or when the analysis was stored before the digest was recorded."
        ),
    )
    mismatch_cause: Literal["package_shape", "host_declarations", "evidence", "cannot_say"] | None = Field(
        default=None,
        description=(
            "Why the re-assembly does not reproduce the generation, set exactly when `reproduces_generation` is "
            "False. `package_shape`: the bundle `schema_version` differs from the recorded one, so this build "
            "digests the same evidence differently (when the host declarations differ too, this still wins: a "
            "package change can move the core declarations the digest covers). `host_declarations`: the version "
            "agrees and `host_declarations_digest` does not — the host changed which sweepables or world "
            "dimensions it declares. `evidence`: both agree, so the inputs moved — a run archived, a result "
            "deleted, an insight the generation read since deleted or superseded, the analysis that minted one "
            "archived — or a host declaration the digest does not cover (a measure, a bar, prose) changed. "
            "`cannot_say`: the stored provenance lacks either recorded value, so no cause can be told apart."
        ),
    )
    bundle: AnalysisContextBundle = Field(description="The assembled bundle itself, whole.")

    @classmethod
    def over(
        cls,
        bundle: AnalysisContextBundle,
        *,
        analysis_id: str | None = None,
        recorded_fingerprint: str | None = None,
        recorded_schema_version: int | None = None,
        recorded_host_declarations_digest: str | None = None,
    ) -> BundleInspection:
        """Wrap an assembled bundle, computing its fingerprint, the comparison and, on a mismatch, its cause.

        Args:
            bundle: The freshly assembled bundle to project.
            analysis_id: The analysis the caller addressed, when they addressed one.
            recorded_fingerprint: That analysis's stored ``bundle_fingerprint``.
            recorded_schema_version: That analysis's stored ``bundle_schema_version``; None when it recorded none.
            recorded_host_declarations_digest: That analysis's stored ``host_declarations_digest``; None when it
                recorded none.

        Returns:
            The inspection payload, with ``reproduces_generation`` set exactly when
            a recorded fingerprint was supplied to compare against, and ``mismatch_cause``
            exactly when that comparison is False.
        """
        fingerprint = bundle.fingerprint()
        reproduces = None if recorded_fingerprint is None else fingerprint == recorded_fingerprint
        cause: Literal["package_shape", "host_declarations", "evidence", "cannot_say"] | None = None
        if reproduces is False:
            if recorded_schema_version is None or recorded_host_declarations_digest is None:
                cause = "cannot_say"
            elif recorded_schema_version != bundle.schema_version:
                cause = "package_shape"
            elif recorded_host_declarations_digest != bundle.host_declarations_digest:
                cause = "host_declarations"
            else:
                cause = "evidence"
        return cls(
            campaign_id=bundle.campaign_id,
            scope_id=bundle.scope_id,
            fingerprint=fingerprint,
            analysis_id=analysis_id,
            recorded_fingerprint=recorded_fingerprint,
            reproduces_generation=reproduces,
            recorded_bundle_schema_version=recorded_schema_version,
            recorded_host_declarations_digest=recorded_host_declarations_digest,
            mismatch_cause=cause,
            bundle=bundle,
        )


class GoalCheckProofReading(EvalDocumentModel):
    """Whether one goal check the campaign's runs graded was shown to beat doing nothing."""

    check: str = Field(min_length=1, description="The goal check, as its template states it.")
    measure_id: str = Field(min_length=1, description="The measure its pass rate is read as: `goal_state:<check>`.")
    proof: GoalCheckProof = Field(
        description=(
            "`proven` only when every run that graded it recorded it proven at launch; `refuted` when any recorded "
            "its control does not discriminate; otherwise `unproven` — including a run that recorded no proof."
        )
    )
    runs: int = Field(ge=1, description="Member runs that graded it, or for a refused check, that refused it.")
    unrecorded: int = Field(
        ge=0, description="Of those, runs launched before proofs were recorded — read as unproven, never as proven."
    )
    stale: int = Field(
        default=0,
        ge=0,
        description=(
            "Of those, runs that recorded it `proven` under an earlier proof rule (before a control's case "
            "parameters were read as a case stores them, #665) — read as unproven, and needing a new launch to "
            "be proven again."
        ),
    )
    refused: str | None = Field(
        default=None,
        description=(
            "Why the grammar refused the check when a member run launched, for a check a template stored before "
            "the rule still carried: those runs graded it on no cell (their cells are not rig faults for it), so "
            "it has no pass rate there and its proof is `refuted`. None for a check every run could grade."
        ),
    )


def host_declarations_digest(profile: HostProfile) -> str:
    """sha256 of the host declarations that partition observations — derived, so no host has a counter to forget.

    A host's apparatus sweepable is a dimension of every observation's apparatus class, and so of the bundle
    fingerprint, and the same holds for a world dimension; a lever is a coordinate of every arm. A host adding,
    removing or renaming one moves every fingerprint over unchanged runs, and the package's own versions
    cannot say so — they are the package's. This digest is computed from the registry the bundle was assembled
    under, so it moves exactly when those declarations do: each sweepable's name, role and blank rule (whether
    a blank means "never recorded", which decides whether a level can be compared at all), and each world
    dimension's name. The core's declarations are in it too, because the registry a host extends carries them;
    a core change also moves :attr:`AnalysisContextBundle.schema_version`, which is read first.

    What it does not cover: a reader callable's behaviour (code, not a declaration), and the host's measure and
    bar registries, which describe readings rather than partitioning them.

    Args:
        profile: The host profile a bundle is assembled under.

    Returns:
        A 64-character lowercase hex digest.
    """
    sweepables = sorted(
        (declared.name, declared.role, declared.indeterminate_when_blank)
        for declared in profile.sweepables.declarations
    )
    world = sorted(profile.world.names) if profile.world is not None else []
    return canonical_digest(
        {
            "sweepables": [
                {"name": name, "role": role, "indeterminate_when_blank": blank} for name, role, blank in sweepables
            ],
            "world": world,
        }
    )


__all__ = [
    "AliasedFactors",
    "AnalysisContextBundle",
    "ArmMechanismReading",
    "ArmServedModel",
    "BundleInspection",
    "CANDIDATE_SERVED_MODEL_CONFOUND",
    "CellCoordinate",
    "ComparedCell",
    "ComparisonFamily",
    "ComparisonVerdict",
    "Confound",
    "DeclaredCellCoverage",
    "DeclaredCrossing",
    "DeclaredLevelCoverage",
    "DesignArm",
    "exploratory_disclosure",
    "FactorPairCell",
    "FactorPairPivot",
    "FactorPairScan",
    "FamilyComparison",
    "GoalCheckProofReading",
    "HeldFixedReading",
    "host_declarations_digest",
    "INTERACTION_ALIASING_UNCHECKED",
    "JudgeChange",
    "JUDGED_OFF_RANKING_REASON",
    "JudgedArm",
    "JudgedMeasure",
    "JudgeDriftLink",
    "JudgeIdentityLevel",
    "LeverCoverageInput",
    "MeasureMovement",
    "MechanismCheck",
    "MechanismUncheckedReason",
    "MeritTier",
    "MovementDirection",
    "MultipleComparisons",
    "NO_DESIGN_EXPLORATORY",
    "NO_QUESTION_EXPLORATORY",
    "observed_mechanism_key",
    "OBSERVED_MECHANISM_PREFIX",
    "QuestionScope",
    "ReadingScope",
    "RealizedDesign",
    "REASONING_SHARE_DIVERGENCE",
    "RunSummary",
    "ScopeDivergence",
    "SERVED_MODEL_PREFIX",
    "ShortCell",
    "TelemetryRollup",
    "TokenRollup",
    "UNDECIDED_CONFOUND_PREFIX",
    "UNVERIFIED_FOLD_PREFIX",
    "VerdictOrder",
]
