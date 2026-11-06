"""The campaign bundle: everything the writer reads to write a campaign's one memo, across every aspect.

Code reports, the writer recommends, a person decides. A campaign asks one question of its arms over one or more
aspects, and each aspect's evidence is its own :class:`~threetears.evals.analysis.bundle.schema.AnalysisContextBundle`.
The writer makes ONE call per campaign, so it reads one input: a :class:`CampaignBundle`, which holds the campaign's
question, arms and floors, code's readings across every aspect (:class:`~threetears.evals.analysis.campaign_readings.CampaignReadings`),
and each aspect's bundle whole, for code.

**One naming across aspects.** A cell is one arm under one rig, and the same arm under the same rig is the same
``cell_ref`` in every aspect it ran in, so a ref alone does not say which aspect's evidence it is. The writer names a
cell by an alias unique across the campaign (``c1`` … ``cN``, :func:`~threetears.evals.analysis.references.campaign_cell_aliases`)
and a trial by an id unique across it (``T1`` … ``TN``); :meth:`CampaignBundle.cell` and :meth:`CampaignBundle.trial`
map each back to its aspect. A campaign of one aspect names both exactly as that aspect's bundle does. An arm is named
by its label (:attr:`CampaignBundle.arms`), never by its variant key.

**The writer reads a curated view, within a budget.** The user message (:func:`build_campaign_user_message`) is not
the bundles dumped: per aspect it carries code's readings, the arms' scoreboard, the design, coverage, the cells'
readings, the separated contrasts first, guardrails, bars, the flagged trials and every soundness disclosure, compact.
Each aspect's part is held to :data:`CAMPAIGN_MESSAGE_CHARS_PER_ASPECT`; over it, code trims in a fixed order —
the flagged trials' excerpts, then the flagged trials, then the contrasts least separated first, then the runs'
configs, then the cells' readings last, since the writer names them in its evidence — and says in the aspect's part
what it left out, since the page shows all of it.
Code's own readings across aspects, the floors, the pair readings and the bars are never trimmed, so a part can stay
over budget; then it says so, and the overshoot is logged as a warning.

**Facts at the altitude the writer claims them, each said once, spelled as the page spells it.** Beside each
aspect's part the view hands the campaign's own: each arm's figures across aspects with their weighting
(``aggregate``, on two or more aspects); each arm's floors, broken, unclear and held, each with the rate it read and,
when unclear, what would settle it (``floors``); and every disclosure the report shares (``disclosures``), each naming
the aspects carrying it: the criteria set after their readings were visible, that each cost figure reflects its
runs' provider routing wherever an arm is priced (``cost_routing``, as the page says it), the latency and
request-settings disclosures wherever they are carried, and any other two or more aspects carry alike, which no aspect's part then
repeats (a confound's explanation likewise, in ``confound_catalog``). A sizing carries its kind on each side
(``to_settle``), a change and every other figure in points reads to the decimal the page prints, an aspect carries
its declared ``short_name``, and a measure its page name, a goal check's its declared name.

**The campaign's own part is bounded by the same budget, and only warned on.** What sits outside every aspect's part
— the arms, floors, bars, a choice's pair readings and a measurement's measures — is code's adjudication and is never
trimmed. It grows with arms and aspects (a choice's pairs with the square of its arms: six arms over four aspects read
60 pairs), so the message is at most one budget per aspect plus that part. When that part alone is over one aspect's
budget, the overshoot is logged as a warning, so a campaign too wide for its writer is seen, not silently sent.

**The writer is told the campaign's kind** (``kind``, derived from its arms): a measurement recommends nothing, a
choice reads its pair readings and any missed bars first, and a change reads its floors and readings.

**Frozen as a frame beside the aspects.** A stored analysis keeps each aspect's bundle in its own
:class:`~threetears.evals.kernel.campaign.AspectAnalysis`, and what the campaign adds over them as the bundle's
:meth:`~CampaignBundle.frame`; :meth:`CampaignBundle.from_frame` puts the two back together.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from pydantic import Field, field_validator, model_validator

from threetears.evals.analysis.arms import arm_names, multi_rig_variants, short_digest, writer_arms
from threetears.evals.analysis.bundle.caps import capped_text
from threetears.evals.analysis.bundle.schema import (
    AnalysisContextBundle,
    evidence_digest,
    fingerprint_digest,
    refuse_a_bundle_of_another_shape,
)
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaign_readings import (
    AggregateFigure,
    CampaignReadings,
    FloorStatus,
    campaign_aggregate,
)
from threetears.evals.analysis.kinds import CampaignKind, campaign_kind
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.depth import REPEATS_FLOOR, Depth, Precision
from threetears.evals.analysis.depth_reading import aspect_depth
from threetears.evals.analysis.lenses.frontier import PASS_HAT_K_CELL_READING, pass_hat_k_cell_lines_bounded
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.references import CellAddress, campaign_cell_aliases
from threetears.evals.analysis.report.campaign.provenance import COST_ROUTING_CAVEAT, priced_aspects
from threetears.evals.analysis.report.decision import (
    CasesNeeded,
    FloorReadings,
    SampleProjection,
    case_cost_usd,
    control_cannot_show,
    floor_projection,
    floor_readings,
    projected_usd_value,
    read_floor,
    reading_words,
)
from threetears.evals.analysis.style import measure_name
from threetears.evals.kernel.campaign import CampaignDeclaration, EvalCampaign
from threetears.evals.kernel.declaration import ASPECT_PRESENTATION_FIELDS, CampaignArm, Floor, resolve_control_arm
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.metrics import goal_check_of
from threetears.evals.kernel.surface import DecisionSurface
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.hashing import round_floats
from threetears.evals.schema.logs import get_logger

log = get_logger(__name__)

#: The campaign bundle's own shape: what it is built at, and what :meth:`CampaignBundle.from_dict` and
#: :meth:`CampaignBundle.from_frame` read a frozen copy back as. 2: each reading and pair states both arms' depth, the
#: paired cases that differ in it, and what limits its interval (`precision`). 3: the readings carry each arm's figures
#: across every aspect (`readings.aggregate`).
CAMPAIGN_BUNDLE_SHAPE = 3

#: The frozen shapes read back as this one: shape 2 differs only by `readings.aggregate`, which is read off its own
#: readings and aspects' bundles (:func:`~threetears.evals.analysis.campaign_readings.campaign_aggregate`), so a case
#: or an analysis frozen at 2 reads, and is fingerprinted, as the same evidence. Any other shape is refused.
_EARLIER_SHAPES = frozenset({2})

#: The shape a fingerprint reads a frame at, and what it leaves out of the readings: shape 3 adds only figures derived
#: from what shape 2 holds, so one campaign's evidence fingerprints alike at either, and a frozen case's labels, an
#: analysis's staleness and a re-freeze's match never move with the shape.
_EVIDENCE_SHAPE = 2
_DERIVED_READINGS = frozenset({"aggregate"})

#: The most characters one aspect's part of the writer's user message may take, as compact JSON. A campaign's message
#: grows with its aspects, at most this much each, and the system prompt is budgeted apart
#: (:data:`~threetears.evals.analysis.gen_prompt.PROMPT_CHAR_BUDGET`). Like that budget, it only falls.
CAMPAIGN_MESSAGE_CHARS_PER_ASPECT = 24_500

#: What opens the writer's user message.
CAMPAIGN_MESSAGE_HEADING = "CAMPAIGN BUNDLE (JSON):\n"

#: How much of a judge's reasoning on one trial the writer reads: enough to name a failure mode, not the transcript.
WRITER_REASON_CHARS = 300

#: How much of what a flagged trial's judge was handed the writer reads: its tail, the part a failure shows in.
WRITER_EXCERPT_CHARS = 400

#: What a measure's meaning is cut to in the writer's view.
_MEANING_CHARS = 200

#: The significant figures a figure in the writer's view keeps: enough to read, not to recompute.
_FIGURES = 4

#: The comparison verdicts that separate an arm from the control, or establish it alike: listed first, so a trim of
#: the contrasts drops the unresolved ones before any of these.
_SEPARATED = frozenset({"improved", "regressed", "equivalent"})

#: What each kind of campaign asks of its memo, as the writer's view says it.
_KIND_MEANS: dict[str, str] = {
    "measurement": "One arm, measured: there is nothing to choose between, so the memo recommends no arm.",
    "choice": (
        "Several arms and no control: the memo picks the arm the evidence favours, or recommends none when no arm "
        "stood out (no clear winner); a pick of an arm shown ahead on no pair rests on cost or simplicity, at low "
        "confidence. A pick names every aspect another arm is shown ahead on, in `behind_on`."
    ),
    "change": "A control and candidates: the memo recommends a candidate the evidence separated, or keeps the control.",
}

#: The bundle fields the writer's view carries as code's disclosures on the aspect's soundness, each only when set.
_DISCLOSURES: tuple[str, ...] = (
    "scope_divergences",
    "whole_swings",
    "incomplete_runs",
    "short_runs",
    "completeness_unknown_run_ids",
    "short_cells",
    "all_failed",
    "cost_unmeasured",
    "latency_contended",
    "launch_disclosure",
    "model_identity_disclosure",
    "request_settings_disclosure",
    "apparatus_confounds",
    "aliased_factors",
    "unexercised_factors",
    "differential_exclusions",
    "arm_served_models",
    "subject_key_instabilities",
    "frontier_bar_withheld",
    "time_axis_withheld",
    "retracted_insights",
)

#: The disclosures that speak for the whole report whichever aspects carry them, each kind said once in the campaign's
#: own ``disclosures`` and never in an aspect's part, with the criteria set after their readings were visible
#: (``late_criteria``). Any other disclosure two or more aspects carry alike is said there once too; one aspect's own
#: stays in its part.
_REPORT_WIDE: frozenset[str] = frozenset({"latency_contended", "request_settings_disclosure"})

#: The floor states in the order the writer's floor rollup lists them, per arm.
_FLOOR_STATES: tuple[str, ...] = ("broken", "unclear", "held")

#: The projection kinds whose count the writer's view states: what a ``more`` needs in all, the run's own cases for
#: ``enough``, the cap a ``beyond`` passed. ``lost`` and ``never`` state no count.
_COUNTED: frozenset[str] = frozenset({"more", "enough", "beyond"})


class CampaignBundle(EvalDocumentModel):
    """A campaign's whole evidence for its one memo: the question, arms and floors, code's readings, every aspect.

    In memory, never stored as itself: a stored analysis keeps each aspect's bundle and this bundle's
    :meth:`frame`. Its :meth:`fingerprint` is the one a generation records.
    """

    schema_version: int = Field(default=CAMPAIGN_BUNDLE_SHAPE, description="The campaign bundle's shape.")
    campaign_id: str = Field(min_length=1, description="The campaign.")
    scope_id: str = Field(min_length=1, description="The scope it lives in.")
    name: str = Field(description="The campaign's name.")
    subject_id: str = Field(description="The analysed subject.")
    subject_kind: str = Field(description="The subject's kind; '' when the campaign declared none.")
    behavior: str = Field(min_length=1, description="The campaign's behavior.")
    declaration: CampaignDeclaration = Field(
        description="What the campaign declares: its question, named arms, aspects and floors, as of assembly."
    )
    arms: list[CampaignArm] = Field(
        description="The campaign's arms as a reader reads them: those a person named, the control first, then each "
        "derived candidate (:func:`~threetears.evals.analysis.campaigns.campaign_arms`)."
    )
    readings: CampaignReadings = Field(description="Code's readings across every aspect and floor.")
    aspects: list[AnalysisContextBundle] = Field(
        min_length=1, description="Each aspect's evidence bundle, in the campaign's order."
    )

    @field_validator("schema_version")
    @classmethod
    def _this_shape(cls, shape: int) -> int:
        """Refuse a campaign bundle of another shape.

        :raises ValueError: ``shape`` is not :data:`CAMPAIGN_BUNDLE_SHAPE`.
        """
        if shape != CAMPAIGN_BUNDLE_SHAPE:
            raise ValueError(f"campaign bundle shape {shape} is not this build's ({CAMPAIGN_BUNDLE_SHAPE})")
        return shape

    @model_validator(mode="after")
    def _one_campaign(self) -> CampaignBundle:
        """Refuse aspects that are not this campaign's, or not one each, and arms whose labels repeat.

        :raises ValueError: An aspect's bundle names another campaign, a label repeats, readings name other aspects,
            or two arms share a label.
        """
        labels = [bundle.aspect for bundle in self.aspects]
        if repeated := sorted({label for label in labels if labels.count(label) > 1}):
            raise ValueError(f"the campaign bundle holds aspect(s) {repeated} more than once")
        if strays := sorted({bundle.campaign_id for bundle in self.aspects} - {self.campaign_id}):
            raise ValueError(f"the campaign bundle of {self.campaign_id} holds aspects of campaign(s) {strays}")
        if self.readings.aspects and self.readings.aspects != labels:
            raise ValueError(f"code's readings are of aspects {self.readings.aspects}, and the bundle holds {labels}")
        names = [arm.label for arm in self.arms]
        if twice := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"the campaign bundle names arm label(s) {twice} more than once")
        return self

    # ------------------------------------------------------------------ identity and storage

    def frame(self, *, like: str | None = None) -> dict[str, Any]:
        """What the campaign adds over its aspects' bundles, as a stored analysis keeps it.

        :param like: A stored campaign fingerprint, to name each aspect's bundle by its fingerprint in that one's
            version (:func:`~threetears.evals.analysis.bundle.schema.fingerprint_digest`); None for the current one,
            which is what a stored analysis keeps.
        :ptype like: str | None
        :return: Every field but the aspects' bundles, which become ``{aspect, fingerprint}`` pairs.
        :rtype: dict[str, Any]
        """
        frame = self.model_dump(mode="json", exclude={"aspects"})
        frame["aspects"] = [
            {"aspect": bundle.aspect, "fingerprint": bundle.fingerprint(like=like)} for bundle in self.aspects
        ]
        return frame

    def fingerprint(self, *, like: str | None = None) -> str:
        """The campaign bundle's fingerprint: its frame's, which names each aspect's bundle by its fingerprint.

        Its floats rounded to twelve significant figures, so one campaign's evidence fingerprints alike across Python
        builds, except, rarely, a value that falls exactly on a 12th-figure rounding boundary (:meth:`AnalysisContextBundle.fingerprint` says why).

        :param like: A fingerprint stored earlier, to compute this one in that one's version so the two compare;
            None for the current version.
        :ptype like: str | None
        :return: ``v2:`` and a 64-character lowercase hex digest, or, ``like`` an unprefixed one, the bare digest.
        :rtype: str
        """
        return fingerprint_digest(_evidence_of(self.frame(like=like)), like=like)

    @staticmethod
    def refuse_another_shape(data: Mapping[str, Any], *, what: str, remedy: str) -> None:
        """Refuse a frozen campaign bundle, or frame, of a shape this build does not read.

        :param data: The frozen document.
        :ptype data: Mapping[str, Any]
        :param what: What holds it, as a sentence names it.
        :ptype what: str
        :param remedy: What to do instead.
        :ptype remedy: str
        :raises RefusedInput: Its ``schema_version`` is neither :data:`CAMPAIGN_BUNDLE_SHAPE` nor an earlier shape this
            build reads it as, or it is not a campaign bundle at all (one aspect's bundle, frozen before one memo
            covered a campaign).
        """
        shape = data.get("schema_version")
        if "readings" not in data or (shape != CAMPAIGN_BUNDLE_SHAPE and shape not in _EARLIER_SHAPES):
            when = (
                "before one memo covered a campaign's every aspect"
                if "readings" not in data
                else "by a build whose readings carried another shape"
            )
            raise RefusedInput(
                f"{what} holds evidence of shape {shape} that is not a campaign bundle of this build's shape "
                f"({CAMPAIGN_BUNDLE_SHAPE}) or one it reads as that ({', '.join(map(str, sorted(_EARLIER_SHAPES)))}): "
                f"it was frozen {when}, so this build cannot read it",
                remedy=remedy,
            )

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, what: str = "this campaign bundle", remedy: str = ""
    ) -> CampaignBundle:
        """Read a whole campaign bundle back, an earlier shape it reads as this one included, refusing any other shape
        with a remedy.

        :param data: A campaign bundle, as ``to_dict`` wrote it (each aspect's bundle whole in its place).
        :ptype data: Mapping[str, Any]
        :param what: What holds it, as a sentence names it, for the refusal.
        :ptype what: str
        :param remedy: What to do instead, for the refusal; empty names re-assembling it.
        :ptype remedy: str
        :return: The bundle.
        :rtype: CampaignBundle
        :raises RefusedInput: It is not a campaign bundle of this build's shape, or an aspect's bundle is from before
            a bundle read one aspect.
        """
        fix = remedy or "assemble the campaign's evidence again, which reads it as it stands"
        cls.refuse_another_shape(data, what=what, remedy=fix)
        bundles = []
        for aspect in data.get("aspects") or []:
            refuse_a_bundle_of_another_shape(aspect, what=what, remedy=fix)
            bundles.append(AnalysisContextBundle.from_dict(dict(aspect)))
        return cls.model_validate({**_as_this_shape(data, bundles), "aspects": bundles})

    @classmethod
    def from_json(cls, json_str: str) -> CampaignBundle:
        """Read ``to_json``'s text back, refusing another shape as :meth:`from_dict` does.

        :param json_str: The JSON text.
        :ptype json_str: str
        :return: The bundle.
        :rtype: CampaignBundle
        :raises RefusedInput: See :meth:`from_dict`.
        """
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_frame(
        cls, frame: Mapping[str, Any], bundles: Sequence[AnalysisContextBundle], *, what: str = "this analysis"
    ) -> CampaignBundle:
        """Put a stored frame and its aspects' bundles back together, an earlier shape it reads as this one included,
        refusing a pair that do not belong together.

        :param frame: The frame, as :meth:`frame` wrote it.
        :ptype frame: Mapping[str, Any]
        :param bundles: The aspects' bundles, in the frame's order.
        :ptype bundles: Sequence[AnalysisContextBundle]
        :param what: What holds the frame, for the refusal.
        :ptype what: str
        :return: The bundle.
        :rtype: CampaignBundle
        :raises RefusedInput: The frame is of another shape.
        :raises ValueError: The bundles are not the ones the frame names, by aspect and fingerprint.
        """
        cls.refuse_another_shape(frame, what=what, remedy="generate the analysis again (analysis_generate)")
        named = [(entry["aspect"], entry["fingerprint"]) for entry in frame.get("aspects") or []]
        # Each bundle in the version its frame entry was stored in, so a frame stored before fingerprints rounded
        # their floats still names its bundles.
        likes = [fingerprint for _, fingerprint in named]
        held = [
            (bundle.aspect, bundle.fingerprint(like=likes[index] if index < len(likes) else None))
            for index, bundle in enumerate(bundles)
        ]
        if named != held:
            raise ValueError(f"{what}'s campaign bundle names aspects {named}, and its aspects hold {held}")
        return cls.model_validate({**_as_this_shape(frame, bundles), "aspects": bundles})

    # ------------------------------------------------------------------ reading it

    @property
    def run_ids(self) -> list[str]:
        """Every aspect's resolved runs, in aspect order."""
        return [run for bundle in self.aspects for run in bundle.run_ids]

    @property
    def labels(self) -> list[str]:
        """The aspects' labels, in the campaign's order."""
        return [bundle.aspect for bundle in self.aspects]

    def aspect(self, label: str) -> AnalysisContextBundle:
        """The bundle of the aspect labelled ``label``.

        :param label: The aspect's label.
        :ptype label: str
        :return: Its bundle.
        :rtype: AnalysisContextBundle
        :raises KeyError: No aspect has that label.
        """
        for bundle in self.aspects:
            if bundle.aspect == label:
                return bundle
        raise KeyError(f"campaign {self.campaign_id} has no aspect {label!r}; its aspects are {self.labels}")

    def surfaces(self) -> dict[str, DecisionSurface]:
        """Each aspect's decision surface, by label: what every reading on that aspect resolves against.

        :return: ``label -> surface``.
        :rtype: dict[str, DecisionSurface]
        """
        return {bundle.aspect: bundle_decision_surface(bundle) for bundle in self.aspects}

    def cell_aliases(self) -> dict[str, CellAddress]:
        """The aliases the writer names cells by, unique across the campaign (``c1`` … ``cN``).

        :return: ``alias -> (aspect, cell_ref)``.
        :rtype: dict[str, CellAddress]
        """
        return campaign_cell_aliases([(bundle.aspect, bundle.cell_measures) for bundle in self.aspects])

    def cell(self, alias: str) -> CellAddress:
        """Resolve the alias a writer named a cell by, or refuse it naming the aliases there are.

        :param alias: The cell as the writer named it.
        :ptype alias: str
        :return: The aspect and the cell's ``cell_ref``.
        :rtype: CellAddress
        :raises UnresolvableReference: No cell carries that alias.
        """
        aliases = self.cell_aliases()
        if (address := aliases.get(alias)) is not None:
            return address
        known = ", ".join(aliases) or "(none — the campaign holds no cell)"
        raise UnresolvableReference(
            f"reference names cell {alias!r}, which is no cell's alias; a cell is named by its `cell` in an aspect's "
            f"`cells` — the cells are: {known}"
        )

    def trial_ids(self) -> dict[str, tuple[str, str]]:
        """The ids the writer names trials by, unique across the campaign: ``T1`` … ``TN``, aspect after aspect.

        A campaign of one aspect names each trial by its bundle's own id.

        :return: ``campaign trial id -> (aspect, the trial's id in that aspect's bundle)``.
        :rtype: dict[str, tuple[str, str]]
        """
        return campaign_trial_ids(
            (bundle.aspect, [trial.trial_id for trial in bundle.trials]) for bundle in self.aspects
        )

    def trial(self, trial_id: str) -> tuple[str, str] | None:
        """The aspect and bundle trial a campaign trial id names; None for an id the campaign does not hold.

        :param trial_id: The id, as the writer named it.
        :ptype trial_id: str
        :return: ``(aspect, bundle trial id)``, or None.
        :rtype: tuple[str, str] | None
        """
        return self.trial_ids().get(trial_id)

    @property
    def kind(self) -> CampaignKind:
        """What kind of campaign this is, derived from its arms: a measurement, a choice or a change.

        A **measurement** has one arm; a **choice** several and no control (nothing is shipped to change from); a
        **change** a control and its candidates. Never declared: read off the arms by
        :func:`~threetears.evals.analysis.kinds.campaign_kind`.
        """
        return campaign_kind(self.arms)

    def control(self) -> CampaignArm | None:
        """The campaign's control arm, or None when it declares none."""
        return next((arm for arm in self.arms if arm.role == "control"), None)

    def arm(self, label: str) -> CampaignArm | None:
        """The arm the writer named by ``label``; None for a label no arm carries.

        :param label: The arm's label.
        :ptype label: str
        :return: The arm, or None.
        :rtype: CampaignArm | None
        """
        return next((arm for arm in self.arms if arm.label == label), None)

    def arm_keys(self, arm: CampaignArm) -> frozenset[str]:
        """Every variant key that names ``arm``'s cells: its own, and the control's under each predicate.

        :param arm: One of this campaign's arms.
        :ptype arm: CampaignArm
        :return: The keys.
        :rtype: frozenset[str]
        """
        if arm.role == "control":
            return frozenset({arm.variant_key, *resolve_control_arm(arm).variant_keys})
        return frozenset({arm.variant_key})

    def label_of(self, variant_key: str) -> str:
        """What a reader calls the arm a variant key names: its arm's label, else the name its levels give it.

        :param variant_key: The variant.
        :ptype variant_key: str
        :return: The label.
        :rtype: str
        """
        for arm in self.arms:
            if variant_key in self.arm_keys(arm):
                return arm.label
        for bundle in self.aspects:
            if variant_key in (names := arm_names(bundle.variant_index)):
                return names[variant_key]
        return short_digest(variant_key)


def campaign_evidence_digest(document: Mapping[str, Any], *, like: str | None = None) -> str:
    """A whole campaign bundle document's fingerprint, as :meth:`CampaignBundle.fingerprint` computes it.

    For a reader holding the stored document rather than the model: a frozen reporter case checks the bundle it
    holds against the fingerprint it recorded, before anything is rebuilt.

    :param document: A campaign bundle, as :meth:`CampaignBundle.to_dict` wrote it.
    :ptype document: Mapping[str, Any]
    :param like: The fingerprint recorded beside it, to compute this one in that one's version; None for the
        current version.
    :ptype like: str | None
    :return: The fingerprint.
    :rtype: str
    """
    frame = {key: value for key, value in document.items() if key != "aspects"}
    frame["aspects"] = [
        {"aspect": aspect.get("aspect"), "fingerprint": evidence_digest(aspect, like=like)}
        for aspect in document.get("aspects") or []
    ]
    return fingerprint_digest(_evidence_of(frame), like=like)


def _as_this_shape(document: Mapping[str, Any], bundles: Sequence[AnalysisContextBundle]) -> dict[str, Any]:
    """A frozen campaign bundle's or frame's fields but its aspects, at :data:`CAMPAIGN_BUNDLE_SHAPE`: one of an earlier
    shape it reads as this one gains what that shape lacked, read off its own readings and its aspects' bundles."""
    held = {key: value for key, value in document.items() if key != "aspects"}
    if held.get("schema_version") not in _EARLIER_SHAPES:
        return held
    readings = CampaignReadings.model_validate(held["readings"])
    aggregate = campaign_aggregate(readings, {bundle.aspect: bundle for bundle in bundles})
    readings = readings.model_copy(update={"aggregate": aggregate})
    return {**held, "schema_version": CAMPAIGN_BUNDLE_SHAPE, "readings": readings}


def _evidence_of(frame: Mapping[str, Any]) -> dict[str, Any]:
    """A campaign bundle's frame as its fingerprint reads it: at :data:`_EVIDENCE_SHAPE`, without the figures derived
    from its readings (:data:`_DERIVED_READINGS`), and without how a page prints each aspect
    (:data:`~threetears.evals.kernel.declaration.ASPECT_PRESENTATION_FIELDS`).

    The figures across aspects are read off the readings and the aspects' bundles the fingerprint already reads, so a
    bundle of this shape and one frozen at the shape before it fingerprint alike over the same evidence. A short name
    is presentation, as an aspect bundle's own is (left out of its fingerprint), so naming an aspect leaves every
    analysis and reporter case of the campaign reproducing. An aspect naming none carries no key to drop.
    """
    frame = dict(frame)
    if frame.get("schema_version") == CAMPAIGN_BUNDLE_SHAPE:
        frame["schema_version"] = _EVIDENCE_SHAPE
        if isinstance(readings := frame.get("readings"), Mapping):
            frame["readings"] = {key: value for key, value in readings.items() if key not in _DERIVED_READINGS}
    declaration = frame.get("declaration")
    if not isinstance(declaration, Mapping) or not isinstance(declaration.get("aspects"), list):
        return frame
    aspects = [
        {key: value for key, value in aspect.items() if key not in ASPECT_PRESENTATION_FIELDS}
        if isinstance(aspect, Mapping)
        else aspect
        for aspect in declaration["aspects"]
    ]
    return {**frame, "declaration": {**declaration, "aspects": aspects}}


def build_campaign_bundle(
    campaign: EvalCampaign,
    bundles: Sequence[AnalysisContextBundle],
    *,
    arms: Sequence[CampaignArm],
    readings: CampaignReadings,
) -> CampaignBundle:
    """Put a campaign's evidence together for its one memo.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param bundles: Each aspect's bundle, in the campaign's aspect order
        (:func:`~threetears.evals.analysis.bundle.assemble.assemble_context_bundle` per aspect).
    :ptype bundles: Sequence[AnalysisContextBundle]
    :param arms: The campaign's arms as a reader reads them (:func:`~threetears.evals.analysis.campaigns.campaign_arms`).
    :ptype arms: Sequence[CampaignArm]
    :param readings: Code's readings across the aspects and floors.
    :ptype readings: CampaignReadings
    :return: The campaign bundle.
    :rtype: CampaignBundle
    :raises ValueError: The bundles are not the campaign's aspects, in its order.
    """
    expected = [aspect.label for aspect in campaign.aspects]
    if [bundle.aspect for bundle in bundles] != expected:
        raise ValueError(
            f"campaign {campaign.id}'s aspects are {expected}, and the bundles handed in are of "
            f"{[bundle.aspect for bundle in bundles]}; one bundle per aspect, in the campaign's order"
        )
    return CampaignBundle(
        campaign_id=campaign.id,
        scope_id=campaign.scope_id,
        name=campaign.name,
        subject_id=campaign.subject_id,
        subject_kind=campaign.subject_kind,
        behavior=campaign.behavior,
        declaration=CampaignDeclaration.of(campaign),
        arms=list(arms),
        readings=readings,
        aspects=list(bundles),
    )


# ---------------------------------------------------------------------- the writer's view


def build_campaign_user_message(bundle: CampaignBundle, *, budget: int = CAMPAIGN_MESSAGE_CHARS_PER_ASPECT) -> str:
    """Render the campaign bundle as the writer's user message — the closed system's whole context.

    Public because it has two readers that must agree to the byte: the generator, which sends it, and the reporter
    eval's judge, which scores a memo against exactly what its generator was given.

    :param bundle: The campaign bundle.
    :ptype bundle: CampaignBundle
    :param budget: The most characters each aspect's part may take, as compact JSON.
    :ptype budget: int
    :return: The user message.
    :rtype: str
    """
    return CAMPAIGN_MESSAGE_HEADING + _compact(writer_view(bundle, budget=budget))


def writer_view(bundle: CampaignBundle, *, budget: int = CAMPAIGN_MESSAGE_CHARS_PER_ASPECT) -> dict[str, Any]:
    """The campaign as the writer reads it: its question, arms, floors and prior insights, then each aspect's part.

    :param bundle: The campaign bundle.
    :ptype bundle: CampaignBundle
    :param budget: The most characters each aspect's part may take, as compact JSON.
    :ptype budget: int
    :return: The view, as the message carries it: every float to twelve significant figures at most
        (:func:`~threetears.evals.schema.hashing.round_floats`), so one campaign renders one message across Python
        builds, whose C maths differ in a computed figure's last bits, except, rarely, a value that falls exactly on a 12th-figure rounding boundary; the figures
        :func:`_figure` keeps are shorter.
    :rtype: dict[str, Any]
    """
    namer = _Namer(bundle)
    floors = sorted(bundle.readings.floors, key=lambda status: status.status != "broken")
    disclosures, catalog, parts = _aspect_parts(bundle, budget, namer)
    bars = sorted(bundle.readings.bars, key=lambda status: status.status != "missed")
    first = bundle.aspects[0]
    kind = bundle.kind
    view: dict[str, Any] = {
        "campaign": {
            "name": bundle.name,
            "behavior": bundle.behavior,
            "question": bundle.declaration.question,
            "aspects": bundle.labels,
        },
        "kind": kind,
        "kind_means": _KIND_MEANS[kind],
    }
    if kind == "choice":
        # A choice reads first which arm code showed ahead of which, on each aspect, and every bar an arm missed.
        view["pairs"] = [
            {
                "aspect": pair.aspect,
                "arms": [pair.first_label, pair.second_label],
                "change": _points(pair.change),
                "interval": _points_pair(pair.interval),
                "confidence": pair.confidence,
                "p_holm": _figure(pair.p_holm),
                "ahead": pair.ahead_label,
                "confirmatory": pair.confirmatory,
                "passed": [
                    f"{pair.first_passed} of {pair.first_trials}",
                    f"{pair.second_passed} of {pair.second_trials}",
                ],
                "why": pair.reason,
                "precision": _precision(pair.precision, namer),
            }
            for pair in bundle.readings.pairs
        ]
    view["bars"] = [
        {
            "aspect": status.aspect,
            "arm": status.arm_label,
            "measure": status.measure,
            "bar": status.bar,
            "direction": status.direction,
            "status": status.status,
            "value": _figure(status.value),
            "interval": _figures(status.interval),
            "why": status.reason,
        }
        for status in bars
    ]
    if missed := [status for status in bars if status.status == "missed"]:
        view["bar_warning"] = (
            f"{len(missed)} bar(s) missed: "
            + "; ".join(f"{status.measure} on {status.aspect} by {status.arm_label}" for status in missed)
            + ". A bar only warns: a recommendation of an arm below a bar names each in `bars_missed`, with why."
        )
    if kind == "measurement":
        view["measured"] = [
            {
                "aspect": measured.aspect,
                "arm": measured.arm_label,
                "measure": measured.measure,
                "name": measured.name,
                "value": _figure(measured.value),
                "interval": _figures(measured.interval),
                "cases": measured.cases,
                "higher_is_better": measured.higher_is_better,
            }
            for measured in bundle.readings.measures
        ]
    view["arms"] = _arms(bundle, namer)
    if (aggregate := _aggregate(bundle)) is not None:
        view["aggregate"] = aggregate
    view |= {
        # Each arm's verdict on each aspect's pass rate first, then each arm's floors: a floor speaks for the one rate
        # it read (a check's, or the pass rate's), never for its whole aspect, and an unclear floor is not a broken one.
        "pass_rate_verdicts": [
            {"aspect": row.aspect, "arm": row.arm_label, "reading": row.reading, "change": _points(row.change)}
            for row in bundle.readings.readings
        ],
        "floors": _floors(bundle, floors),
    }
    if disclosures:
        view["disclosures"] = disclosures
    if any(pair.precision is not None for pair in bundle.readings.pairs) or any(
        reading.precision is not None for reading in bundle.readings.readings
    ):
        # A term defined once: what every comparison's `precision` reads its between-case share by.
        view["precision_means"] = REPEATS_FLOOR
    if catalog:
        view["confound_catalog"] = catalog
    view |= {
        "prior_insights": [
            {"statement": insight.statement, "confidence": insight.confidence} for insight in first.prior_insights
        ],
        # Stated so a short list is not read as the whole ledger.
        "prior_insights_omitted": first.prior_insights_omitted,
        "aspects": parts,
    }
    if broken := [status for status in floors if status.status == "broken"]:
        view["floor_warning"] = (
            f"{len(broken)} floor(s) broken: "
            + "; ".join(f"{status.floor} on {status.arm_label}" for status in broken)
            + ". A floor only warns: it never blocks a recommendation, and a recommendation of an arm with a broken "
            "floor names each such floor in `floors_overridden`, with why breaking it is acceptable."
        )
    if not bundle.declaration.floors:
        view["floors_declared"] = "The campaign declares no floor."
    shared = len(_compact({key: value for key, value in view.items() if key != "aspects"}))
    if shared > budget:
        log.warning(
            "eval.analysis the writer's campaign-level part (arms, floors, bars, pairs, measures) of campaign %s is %d "
            "characters over one aspect's budget of %d: it is code's adjudication and is never trimmed",
            bundle.campaign_id,
            shared - budget,
            budget,
        )
    # One pass over the finished view, every section alike: a figure no section rounds is still not served past
    # the twelfth figure, where two Python builds can disagree.
    rounded: dict[str, Any] = round_floats(view)
    return rounded


def aspect_view(
    bundle: CampaignBundle, index: int, *, budget: int = CAMPAIGN_MESSAGE_CHARS_PER_ASPECT
) -> dict[str, Any]:
    """One aspect's part of the writer's view, trimmed to ``budget`` characters of compact JSON.

    Trimmed in a fixed order, each step only as far as it must: the flagged trials' excerpts, last trial first; then
    the flagged trials themselves; then the contrasts, least separated first; then the runs' configs, last run first;
    then, last, the cells' readings, last cell's last reading first, since the writer's evidence names them. What was
    left out is said in ``trimmed``. Code's
    readings, the scoreboard, the design, coverage and the disclosures are never trimmed, so a part can stay over budget
    when they alone exceed it: ``trimmed`` says so, and the overshoot is logged as a warning.

    A disclosure the whole report shares is not in the part: a report-wide kind, or one two or more aspects carry
    alike, is the campaign's own (:func:`writer_view`'s ``disclosures``), and so is a confound's explanation two or
    more aspects carry alike (its ``confound_catalog``).

    :param bundle: The campaign bundle.
    :ptype bundle: CampaignBundle
    :param index: The aspect's position.
    :ptype index: int
    :param budget: The most characters the part may take.
    :ptype budget: int
    :return: The aspect's part.
    :rtype: dict[str, Any]
    """
    return _aspect_parts(bundle, budget, _Namer(bundle))[2][index]


def _aspect_parts(
    bundle: CampaignBundle, budget: int, namer: _Namer
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Every aspect's part, each trimmed to ``budget`` once what the report shares is taken out of them: the
    campaign's disclosures, the explanations of confounds two or more aspects carry alike (each naming them), and the
    parts."""
    parts = [_untrimmed(bundle, aspect, namer) for aspect in bundle.aspects]
    disclosures = _shared_disclosures(bundle, parts)
    catalog = _shared_catalog(parts)
    for part in parts:
        _trim(part, budget)
    return disclosures, catalog, parts


def _alike(held: Sequence[tuple[str, Any]]) -> list[dict[str, Any]]:
    """``(aspect, value)`` pairs grouped by value, each value once with the aspects carrying it, first seen first."""
    groups: list[dict[str, Any]] = []
    for aspect, value in held:
        group = next((group for group in groups if group["said"] == value), None)
        if group is None:
            groups.append({"aspects": [aspect], "said": value})
        else:
            group["aspects"].append(aspect)
    return groups


def _shared_disclosures(bundle: CampaignBundle, parts: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each disclosure the report shares, taken out of the aspects' parts and said once, naming the aspects that carry
    it: the criteria set after their readings were visible, that every cost figure is its runs' provider routing's
    wherever an arm is priced, each report-wide kind (:data:`_REPORT_WIDE`) whatever carries it, and any other a second
    aspect carries alike. A value one aspect alone carries stays in its part."""
    said: dict[str, list[dict[str, Any]]] = {}
    late: dict[tuple[str, str | None, str | None], list[str]] = {}
    for aspect in bundle.aspects:
        for criterion in aspect.reading_scope.late_criteria:
            late.setdefault((criterion.criterion, criterion.name, criterion.page_date()), []).append(aspect.aspect)
    if late:
        # Dated as the page dates them, so the memo and the page say one date.
        said["late_criteria"] = [
            {"criterion": criterion, "name": name, "set_at": set_at, "aspects": aspects}
            for (criterion, name, set_at), aspects in late.items()
        ]
    if priced := priced_aspects(bundle.aspects):
        # Said as the page says it, once: a cost is the runs' provider routing's, never the arm's alone.
        said["cost_routing"] = [{"aspects": priced, "said": COST_ROUTING_CAVEAT}]
    for key in dict.fromkeys(key for part in parts for key in part["disclosures"]):
        groups = _alike([(part["aspect"], part["disclosures"][key]) for part in parts if key in part["disclosures"]])
        shared = [group for group in groups if key in _REPORT_WIDE or len(group["aspects"]) > 1]
        if not shared:
            continue
        said[key] = shared
        carried = {aspect for group in shared for aspect in group["aspects"]}
        for part in parts:
            if part["aspect"] in carried:
                del part["disclosures"][key]
    return said


def _shared_catalog(parts: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each confound's explanation two or more aspects' parts carry alike, taken out of them and said once, naming the
    aspects that carry it, as :func:`_shared_disclosures` says a disclosure: each wording of one confound apart, so
    two aspects agreeing beside a third worded otherwise say theirs once. A wording one aspect alone carries stays in
    its part; an aspect left with none of its own carries no catalog."""
    dimensions = dict.fromkeys(dimension for part in parts for dimension in part.get("confound_catalog", {}))
    said: dict[str, list[dict[str, Any]]] = {}
    for dimension in dimensions:
        groups = _alike(
            [
                (part["aspect"], part["confound_catalog"][dimension])
                for part in parts
                if dimension in part.get("confound_catalog", {})
            ]
        )
        if shared := [group for group in groups if len(group["aspects"]) > 1]:
            said[dimension] = shared
    for part in parts:
        own = {
            dimension: reason
            for dimension, reason in part.pop("confound_catalog", {}).items()
            if not any(part["aspect"] in group["aspects"] for group in said.get(dimension, []))
        }
        if own:
            part["confound_catalog"] = own
    return said


class _Namer:
    """Names every arm by its label and every cell by its campaign alias, across a campaign's aspects."""

    def __init__(self, bundle: CampaignBundle) -> None:
        self.bundle = bundle
        self.aliases = {(address.aspect, address.cell_ref): alias for alias, address in bundle.cell_aliases().items()}
        self.trials = {where: trial for trial, where in bundle.trial_ids().items()}
        keys = {entry.variant_key for aspect in bundle.aspects for entry in aspect.variant_index}
        keys |= {key for arm in bundle.arms for key in bundle.arm_keys(arm)}
        #: Every variant key the campaign holds, and what a reader calls its arm.
        self.labels: dict[str, str] = {key: bundle.label_of(key) for key in sorted(keys)}

    def arm(self, variant_key: str) -> str:
        """The label of the arm ``variant_key`` names."""
        if variant_key not in self.labels:
            self.labels[variant_key] = self.bundle.label_of(variant_key)
        return self.labels[variant_key]

    def cell(self, aspect: str, variant_key: str, apparatus_class_id: str) -> str | None:
        """The alias of one aspect's cell; None for a cell the aspect does not hold."""
        return self.aliases.get((aspect, cell_ref(variant_key, apparatus_class_id)))

    def walk(self, node: Any, aspect: str, path: str = "") -> Any:
        """``node`` in writer terms: a cell by its alias, an arm (and a key naming one) by its label.

        :raises RuntimeError: A cell is named that the aspect's ``cell_measures`` does not hold — the bundle was built
            inconsistently, and the writer could not name that cell at all.
        """
        if isinstance(node, list):
            return [self.walk(item, aspect, f"{path}[{at}]") for at, item in enumerate(node)]
        if isinstance(node, str):
            return self.labels.get(node, node)
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        if "variant_key" in node and "apparatus_class_id" in node:
            alias = self.cell(aspect, node["variant_key"], node["apparatus_class_id"])
            if alias is None:
                raise RuntimeError(
                    f"aspect {aspect!r} names a cell at {path or '/'} that has no `cell_measures` entry to alias it by"
                )
            out["cell"] = alias
        elif "variant_key" in node:
            out["arm"] = self.arm(node["variant_key"])
        for key, value in node.items():
            if key in ("variant_key", "apparatus_class_id"):
                continue
            out[self.labels.get(str(key), str(key))] = self.walk(value, aspect, f"{path}.{key}")
        return out


def _arms(bundle: CampaignBundle, namer: _Namer) -> dict[str, Any]:
    """Every arm by its label and role, with the levels that tell it apart; the levels every arm shares, once."""
    entries = {entry.variant_key: entry for aspect in bundle.aspects for entry in aspect.variant_index}
    described = writer_arms(list(entries.values()))
    listed: list[dict[str, Any]] = described["arms"]  # type: ignore[assignment]  # writer_arms types its values as object
    by_key = {arm["variant_key"]: arm for arm in listed}
    roles = {key: arm.role for arm in bundle.arms for key in bundle.arm_keys(arm)}
    arms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in [*(arm.variant_key for arm in bundle.arms), *entries]:
        label = namer.arm(key)
        if label in seen:
            continue
        seen.add(label)
        held = by_key.get(key, {})
        arms.append(
            {
                "arm": label,
                "role": roles.get(key, "candidate"),
                "levels": held.get("levels", {}),
                "levels_unavailable": held.get("levels_unavailable"),
            }
        )
    return {"arms": arms, "shared_levels": described["shared_levels"]}


def _aggregate(bundle: CampaignBundle) -> dict[str, Any] | None:
    """Each arm's figures across every aspect, with the weighting they are read under, as the scorecard's equal-weight
    row and chart print them; None on one aspect, whose figures are its own part's, and where none was read."""
    aggregate = bundle.readings.aggregate
    if aggregate is None or len(bundle.aspects) < 2:
        return None
    arms: list[dict[str, Any]] = []
    for arm in aggregate.arms:
        entry: dict[str, Any] = {"arm": arm.arm_label, "pass_rate_pct": _across(arm.pass_rate, _points)}
        if arm.change is not None:
            entry["change_pts"] = _across(arm.change, _points)
        entry["cost_per_reply_usd"] = _across(arm.cost, _figure)
        if arm.cheaper_on is not None:
            entry["cheaper_on"] = arm.cheaper_on
        if arm.dearer_on is not None:
            entry["dearer_on"] = arm.dearer_on
        arms.append(entry)
    weighting = aggregate.weighting
    return {
        "weighting": {
            "basis": weighting.basis,
            "weights": {label: _figure(weight) for label, weight in weighting.weights.items()},
            "intervals": weighting.intervals,
        },
        "level": aggregate.level,
        "arms": arms,
    }


def _across(figure: AggregateFigure, spelled: Callable[[float | None], float | None]) -> dict[str, Any]:
    """One figure across aspects: its value and interval as the page spells them, or the aspects it is withheld for."""
    if figure.value is None or figure.interval is None:
        return {"withheld_for": figure.missing}
    return {"value": spelled(figure.value), "interval": [spelled(end) for end in figure.interval]}


def _floors(bundle: CampaignBundle, floors: Sequence[FloorStatus]) -> list[dict[str, Any]]:
    """Every floor's status, arm by arm (an arm with a broken floor first): the floors it breaks, leaves unclear and
    holds, each with the aspect and the rate it read, and for an unclear one what would settle it.

    :raises RuntimeError: A floor is read with no control — the bundle was built inconsistently.
    :raises KeyError: A floor's status names a floor the campaign's declaration does not hold.
    """
    if not floors:
        return []
    control = bundle.readings.control
    if control is None:
        raise RuntimeError("the campaign's readings hold floors but name no control they were read against")
    declared = {floor.key: floor for floor in bundle.declaration.floors}
    figures: dict[tuple[str, str], FloorReadings] = {}
    by_arm: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for status in floors:
        aspect = bundle.aspect(status.aspect)
        if (status.aspect, status.arm) not in figures:
            # Named as the page names them: the aspect's own page passes its declared check names.
            figures[status.aspect, status.arm] = floor_readings(
                aspect, control, status.arm, check_names=aspect.goal_check_names
            )
        read = figures[status.aspect, status.arm]
        rate = status.margin is not None
        entry: dict[str, Any] = {
            "floor": status.floor,
            "aspect": status.aspect,
            "reads": _read_name(aspect, read, status.reading)
            if status.reading is not None
            else ("the pass rate" if rate else "the cost per reply"),
            "interval": _points_pair(status.interval) if rate else _figures(status.interval),
            "worst_drop": _points(status.worst_drop),
            "margin": status.margin,
            "short_by": _points(status.short_by),
            "why": status.reason,
        }
        if status.status == "unclear":
            entry["to_settle"] = _to_settle(_floor_projection(bundle, status, declared[status.floor], read))
        by_arm.setdefault(status.arm_label, {}).setdefault(status.status, []).append(
            {key: value for key, value in entry.items() if value is not None}
        )
    return [
        {"arm": arm, **{state: held[state] for state in _FLOOR_STATES if state in held}}
        for arm, held in sorted(by_arm.items(), key=lambda item: "broken" not in item[1])
    ]


def _read_name(aspect: AnalysisContextBundle, figures: FloorReadings, reading: str) -> str:
    """The rate a floor read, by the name the page gives the rate its floor table reads (``figures``, built as the
    page builds them): a measure by its reader name, a judged dimension in words; a check by its declared name, else
    its whole expression, as the view's ``measures`` names it. A reading the arms carry no rate for is said as the
    page's floor statement says it (:func:`~threetears.evals.analysis.report.decision.reading_words`).
    """
    rate = next((rate for rate in figures.rates if rate.answers(reading)), None)
    if rate is None or rate.measure is None:
        return reading_words(reading, figures, aspect.goal_check_names)
    if (check := goal_check_of(rate.measure)) is not None:
        return aspect.check_name(rate.measure) or check
    return rate.name


def _floor_projection(
    bundle: CampaignBundle, status: FloorStatus, floor: Floor, figures: FloorReadings
) -> SampleProjection | None:
    """The cases that would show an unclear floor holds, counted as its aspect's page and the scorecard's To settle
    count it: read again from the aspect's evidence (``figures``). None where the floor's reading cannot be projected
    (:func:`~threetears.evals.analysis.report.decision.floor_projection`)."""
    verdict = read_floor(floor, figures, control=bundle.readings.control_label, candidate=status.arm_label)
    return floor_projection(verdict, per_case_usd=case_cost_usd(bundle.aspect(status.aspect)))


def _to_settle(projection: SampleProjection | None) -> dict[str, Any] | None:
    """What would settle an unclear reading, each side with its kind: if the arms do not differ (``if_none``) and if
    they differ as observed (``if_observed``). None where nothing was projected."""
    if projection is None:
        return None
    return {"if_none": _need(projection, projection.if_none), "if_observed": _need(projection, projection.if_observed)}


def _need(projection: SampleProjection, need: CasesNeeded) -> dict[str, Any]:
    """One side of a projection: its kind (:data:`~threetears.evals.analysis.report.decision.NeedKind`), the cases
    it states (the total, never cases to add), and what running them costs, to two figures as the page prices it."""
    said: dict[str, Any] = {"kind": need.kind}
    if need.kind in _COUNTED:
        said["cases"] = need.cases
    if (cost := projection.cost(need)) is not None:
        said["usd"] = projected_usd_value(cost)
    return said


def _untrimmed(bundle: CampaignBundle, aspect: AnalysisContextBundle, namer: _Namer) -> dict[str, Any]:
    """One aspect's part of the view, whole."""
    label = aspect.aspect
    declared = aspect.declared
    design = declared.design if declared is not None else None
    payload = aspect.to_dict()
    view: dict[str, Any] = {"aspect": label}
    if aspect.short_name is not None:
        view["short_name"] = aspect.short_name
    view |= {
        "question": {
            "asked": aspect.question_declared,
            "margin": declared.margin if declared is not None else None,
            "merit_axes": design.merit_axes if design is not None else None,
            "merit_priority": design.merit_priority if design is not None else [],
            "reading_scope": namer.walk(_without(payload["reading_scope"], "late_criteria"), label),
        },
        "readings": [
            {
                "arm": reading.arm_label,
                "reading": reading.reading,
                "change": _points(reading.change),
                "gain_interval": _points_pair(reading.gain_interval),
                "gain_confidence": reading.gain_confidence,
                "drop_interval": _points_pair(reading.drop_interval),
                "drop_confidence": reading.drop_confidence,
                "within_interval": _points_pair(reading.within_interval),
                "margin": reading.margin,
                "control_passed": f"{reading.control_passed} of {reading.control_trials}",
                "arm_passed": f"{reading.arm_passed} of {reading.arm_trials}",
                "cases": [reading.control_cases, reading.arm_cases],
                "to_settle": _to_settle(reading.projection),
                "confirmatory": reading.confirmatory,
                "why": reading.reason,
                "precision": _precision(reading.precision, namer),
            }
            for reading in bundle.readings.on_aspect(label)
        ],
        "depth": _depth(aspect, namer),
        "scoreboard": [_figured(row, _SCOREBOARD_FIGURES) for row in namer.walk(payload["scoreboard"], label)],
        "design": namer.walk(payload["design"], label),
        "runs": [
            {
                "run": summary.run_id,
                "candidate_model": summary.candidate_model,
                "k_runs": summary.k_runs,
                "n_results": summary.n_results,
                "n_errors": summary.n_errors,
                "config": summary.config,
                "config_provenance": summary.config_provenance,
            }
            for summary in aspect.run_summaries
        ],
        "coverage": namer.walk([_without(row, "declared_levels") for row in payload["coverage"]], label),
        "cells": _cells(aspect, namer),
        "measures": _measures(aspect),
        "contrast_families": [
            {
                "merit_axes": family["merit_axes"],
                "correction": family["correction"],
                "alpha": family["alpha"],
                "family_size": family["family_size"],
                "disclosure": family.get("disclosure"),
            }
            for family in payload["multiple_comparisons"]["families"]
        ],
        "contrasts_withheld": payload["multiple_comparisons"].get("withheld"),
        "contrasts": _contrasts(aspect, namer),
        "guardrails": namer.walk([_without(check, "control") for check in payload["guardrails"]["checks"]], label),
        "bars": [
            {**bar, "verdicts": [_figured(verdict, _BAR_VERDICT_FIGURES) for verdict in bar["verdicts"]]}
            for bar in namer.walk(payload["bar_adjudications"], label)
        ],
        "judge_evidence_tiers": namer.walk(payload["judge_evidence_tiers"], label),
        "goal_check_proofs": payload["goal_check_proofs"],
        "arm_mechanisms": namer.walk(payload["arm_mechanisms"], label),
        "verdict_order": namer.walk(payload["verdict_order"], label),
        "time_axis": namer.walk(payload["time_axis"], label),
        "trials": _trials(aspect, namer),
        "disclosures": {
            key: namer.walk(payload[key], label) for key in _DISCLOSURES if payload.get(key) not in (None, "", [], {})
        },
    }
    # Keyed by the arm, as the view names every arm, never by the model id: two arms on one model are two entries.
    unmeasured = {
        namer.arm(point.variant_key): point.pass_hat_k_unmeasured_reason
        for subject in aspect.frontier.subjects
        for point in subject.points
        if point.pass_hat_k_unmeasured_reason
    }
    if unmeasured:
        view["disclosures"]["pass_k_unmeasured"] = unmeasured
    # Every pass^k cell beside an arm's headline, and a headline that is a cell of one run, after the one sentence
    # saying what "kept out" covers (the frontier's pass^k, bar and verdict; every other figure still counts the run).
    # Bounded to a line per reason, since the writer's budget is shared with its evidence and a contestant of many
    # runs that each recorded nothing would otherwise spend it a sentence per run.
    unpooled: dict[str, list[str]] = {}
    for subject in aspect.frontier.subjects:
        for point in subject.points:
            unpooled.setdefault(namer.arm(point.variant_key), []).extend(
                pass_hat_k_cell_lines_bounded(point.pass_hat_k_cells)
            )
    unpooled = {arm: lines for arm, lines in unpooled.items() if lines}
    if unpooled:
        view["disclosures"]["pass_k_unpooled"] = {"reading": PASS_HAT_K_CELL_READING, "by_arm": unpooled}
    if aspect.telemetry.measures.absent_scopes:
        view["disclosures"]["absent_scopes"] = list(aspect.telemetry.measures.absent_scopes)
    if aspect.telemetry.measures.unreported_observations:
        view["disclosures"]["unreported_observations"] = list(aspect.telemetry.measures.unreported_observations)
    if payload["held_fixed_reading"].get("disclosure"):
        view["disclosures"]["held_fixed_reading"] = payload["held_fixed_reading"]["disclosure"]
    confounds = {
        dimension: reason
        for dimension, reason in payload["confound_catalog"].items()
        if dimension in json.dumps(view["coverage"]) or dimension in json.dumps(view["disclosures"])
    }
    if confounds:
        view["confound_catalog"] = confounds
    if (bounds := _control_floor(bundle, aspect)) is not None:
        view["control_floor"] = bounds
    return view


def _control_floor(bundle: CampaignBundle, aspect: AnalysisContextBundle) -> dict[str, Any] | None:
    """What the control's pass rate on ``aspect`` leaves no candidate able to show, as the scorecard marks it
    (:func:`~threetears.evals.analysis.report.decision.control_cannot_show`): a ``drop`` past the margin, a ``gain``;
    None where it can show both, or the campaign has no control."""
    rows = bundle.readings.on_aspect(aspect.aspect)
    control = bundle.readings.control
    if control is None or not rows:
        return None
    first = rows[0]
    bounds = control_cannot_show(
        aspect,
        control,
        [row.arm for row in rows],
        margin=first.margin,
        passed=first.control_passed,
        trials=first.control_trials,
    )
    return {"control": bundle.readings.control_label, "cannot_show": list(bounds)} if bounds else None


def _repeats(depth: Depth) -> dict[str, Any]:
    """One arm's depth as the writer reads it: its cases, trials and repeats per case, fewest, median and most."""
    return {
        "cases": depth.cases,
        "trials": depth.trials,
        "repeats_per_case": [depth.repeats_min, depth.repeats_median, depth.repeats_max],
    }


def _depth(aspect: AnalysisContextBundle, namer: _Namer) -> dict[str, Any]:
    """The aspect's depths: each arm's, the sentence to say once where they differ, and per frontier subject the depth
    pass^k compares at (``common_depth``, the fewest runs any arm gave a shared case), above which no comparison of
    pass^k may rank; ``unmeasured_for`` names the arms that ran no case the planned ``k`` times."""
    depth = aspect_depth(aspect, namer.arm)
    view: dict[str, Any] = {
        "arms": {label: _repeats(held) for label, held in depth.arms.items()},
        "differ": depth.differ,
    }
    if depth.said is not None:
        view["said"] = depth.said
    if depth.compared:
        # One entry per frontier subject, each with the sentences to say: pass^k compares at `common_depth`, nothing
        # where `shares_no_case`, and is unmeasured at the planned `k` for each arm `unmeasured_for` names.
        view["pass_k"] = [
            {
                "where": subject.where,
                "k": subject.k,
                "common_depth": subject.common_depth,
                "shares_no_case": subject.shares_no_case,
                "unmeasured_for": subject.unmeasured_for,
                "said": subject.lines(),
            }
            for subject in depth.subjects
            if subject.compared
        ]
    # Keyed by subject (`where`, as each `pass_k` entry is) then arm: an aspect with two subjects holds an arm modelled
    # on both, and a figure with no subject would be one the writer cannot place.
    modelled = {
        said.where: {
            namer.arm(point.variant_key): {
                "k": point.model_pass_hat_k.k,
                "pass_hat_k": _figure(point.model_pass_hat_k.pass_hat_k),
                "basis": point.model_pass_hat_k.basis,
                "assumes": point.model_pass_hat_k.assumes,
            }
            for point in subject.points
            if point.model_pass_hat_k is not None
        }
        for subject, said in zip([sub for sub in aspect.frontier.subjects if sub.points], depth.subjects, strict=True)
    }
    modelled = {where: arms for where, arms in modelled.items() if arms}
    if modelled:
        # A model, never a measurement: carried labelled, and no comparison may rank on it.
        view["pass_k_model"] = modelled
    return view


def _precision(precision: Precision | None, namer: _Namer) -> dict[str, Any] | None:
    """What limits a comparison's interval, as the writer reads it: its half-width and each arm's own in points, the arm
    limiting it, and the between-case share of its variance. What that share means is defined once, for the whole
    view (``precision_means``), never per comparison."""
    if precision is None:
        return None
    return {
        "half_width": _points(precision.half_width),
        "own_half_width": {namer.arm(key): _points(width) for key, width in precision.half_widths.items()},
        "limited_by": namer.arm(precision.limited_by) if precision.limited_by is not None else None,
        "between_case_share": _figure(precision.between_share),
    }


def _cells(aspect: AnalysisContextBundle, namer: _Namer) -> list[dict[str, Any]]:
    """Every cell's readings, compact: its replication and each measure's and judged dimension's point and spread."""
    multi = multi_rig_variants(aspect.cell_measures)
    cells: list[dict[str, Any]] = []
    for cell in aspect.cell_measures:
        measures: dict[str, Any] = {}
        for summary in cell.measures.measures:
            if summary.texts:
                continue
            if summary.categories:
                measures[summary.name] = {"n": summary.n, "counts": dict(summary.categories)}
                continue
            point = summary.rate if summary.rate is not None else summary.mean
            entry: dict[str, Any] = {"mean": _figure(point), "n": summary.n, "n_cases": summary.n_independent}
            if summary.ci_low is not None and summary.ci_high is not None:
                entry["ci"] = [_figure(summary.ci_low), _figure(summary.ci_high)]
            tail = summary.bad_tail()
            if tail is not None:
                entry["p05" if summary.higher_is_better else "p95"] = _figure(tail)
            if summary.n_zero:
                entry["n_zero"] = summary.n_zero
            measures[summary.name] = entry
        judged = {
            reading.dimension: {
                "mean": _figure(reading.mean),
                "n": reading.n,
                "n_cases": reading.n_independent,
                "tier": reading.evidence_tier,
            }
            for reading in cell.judged
        }
        entry = {
            "cell": namer.cell(aspect.aspect, cell.variant_key, cell.apparatus_class_id),
            "arm": namer.arm(cell.variant_key),
            "runs": len(cell.run_ids),
            "n_observations": cell.n_observations,
            "n_cases": cell.n_cases,
            "repeats_per_case": [cell.repeats_per_case_min, cell.repeats_per_case_max],
            "n_candidate_failed": cell.n_candidate_failed,
            "n_no_turn": cell.n_no_turn,
            "measures": measures,
            "judged": judged,
        }
        if cell.variant_key in multi:
            entry["rig"] = short_digest(cell.apparatus_class_id)
        cells.append(entry)
    return cells


def _measures(aspect: AnalysisContextBundle) -> dict[str, Any]:
    """What each measure the cells carry means, keyed by its key: its name on the page, and what it means, never
    guessed from the key."""
    names = sorted({summary.name for cell in aspect.cell_measures for summary in cell.measures.measures})
    meanings: dict[str, Any] = {}
    for name in names:
        facts = aspect.measure_catalog.get(name)
        if facts is None:
            continue
        check = goal_check_of(name)
        meanings[name] = {
            # As the page names it: a check by its declared name, else its whole expression; any other measure by
            # its reader name, so a cost is never named by its key.
            "name": (aspect.check_name(name) or check) if check is not None else measure_name(name, facts.reader_name),
            "means": capped_text(facts.reader_prose or facts.description or "", _MEANING_CHARS),
            "higher_is_better": facts.higher_is_better,
            "unit": facts.unit,
            "merit_axis": facts.merit_axis,
            "scope": facts.attribution_scope,
            "diagnostic": facts.diagnostic,
        }
    return meanings


def _contrasts(aspect: AnalysisContextBundle, namer: _Namer) -> list[dict[str, Any]]:
    """Every comparison against the control, separated ones first and each family's adjusted p only."""
    rows: list[tuple[tuple[int, float], dict[str, Any]]] = []
    for family in aspect.multiple_comparisons.families:
        for comparison in family.comparisons:
            row = {
                "cell": namer.cell(
                    aspect.aspect, comparison.contrast.variant_key, comparison.contrast.apparatus_class_id
                ),
                "control": namer.cell(
                    aspect.aspect, comparison.control.variant_key, comparison.control.apparatus_class_id
                ),
                "reading": comparison.reading,
                "name": comparison.name,
                "verdict": comparison.verdict,
                "delta": _figure(comparison.delta),
                "interval": _figures(comparison.interval),
                "p_adjusted": _figure(comparison.p_adjusted),
                "family_size": family.family_size,
                "materiality": comparison.materiality,
                "equivalence_margin": _figure(comparison.equivalence_margin),
                "equivalence_p_adjusted": _figure(comparison.equivalence_p_adjusted),
                "untested_reason": comparison.untested_reason,
                "not_separated_reason": comparison.not_separated_reason,
                "mechanism_confounds": namer.walk(
                    [confound.model_dump(mode="json") for confound in comparison.mechanism_confounds], aspect.aspect
                ),
            }
            separated = comparison.verdict in _SEPARATED
            p = comparison.p_adjusted if comparison.p_adjusted is not None else 1.0
            rows.append(
                ((0 if separated else 1, p), {key: value for key, value in row.items() if value not in (None, [])})
            )
    return [row for _, row in sorted(rows, key=lambda pair: pair[0])]


def _trials(aspect: AnalysisContextBundle, namer: _Namer) -> list[dict[str, Any]]:
    """Every trial that did not pass — the candidate's failures and the test setup's losses — by its campaign id."""
    trials: list[dict[str, Any]] = []
    for trial in aspect.trials:
        if trial.state == "passed":
            continue
        row: dict[str, Any] = {
            "trial": namer.trials[(aspect.aspect, trial.trial_id)],
            "arm": namer.arm(trial.arm),
            "case": trial.case_label,
            "repeat": trial.repeat,
            "state": trial.state,
            "cause": trial.cause,
            "failed_checks": [check.name for check in trial.failed_checks],
            "judge": [
                {"dim": judged.dim, "score": judged.score}
                | ({"reason": capped_text(judged.reason, WRITER_REASON_CHARS)} if judged.reason else {})
                for judged in trial.judge
            ],
            "excerpt": _tail(trial.excerpt, WRITER_EXCERPT_CHARS),
        }
        # What a trial does not carry is left out, not sent empty: a flagged trial is billed on every generation.
        trials.append({key: value for key, value in row.items() if value not in (None, "", [])})
    return trials


def _trim(view: dict[str, Any], budget: int) -> None:
    """Trim one aspect's part to ``budget`` characters of compact JSON, in place, saying what was left out.

    The note saying so is counted in the budget: it is written at its longest before anything is trimmed, so the part
    that carries it fits, then rewritten with the counts that were trimmed, which is never longer.
    """
    if len(_compact(view)) <= budget:
        return
    left_out = {"excerpts": 0, "trials": 0, "contrasts": 0, "runs' configs": 0, "cell readings": 0}
    view["trimmed"] = _trimmed_note(dict.fromkeys(left_out, 10**6), budget, over=True)

    def over() -> bool:
        return len(_compact(view)) > budget

    for trial in reversed(view["trials"]):
        if not over():
            break
        if trial.get("excerpt"):
            trial["excerpt"] = ""
            left_out["excerpts"] += 1
    while over() and view["trials"]:
        view["trials"].pop()
        left_out["trials"] += 1
    while over() and view["contrasts"]:
        view["contrasts"].pop()
        left_out["contrasts"] += 1
    for run in reversed(view["runs"]):
        if not over():
            break
        if run.pop("config", None) is not None:
            run.pop("config_provenance", None)
            left_out["runs' configs"] += 1
    # The cells' readings go last: the writer's evidence rows name them.
    for cell in reversed(view["cells"]):
        for group in ("judged", "measures"):
            while over() and cell[group]:
                cell[group].pop(next(reversed(cell[group])))
                left_out["cell readings"] += 1
    still_over = over()
    view["trimmed"] = _trimmed_note(left_out, budget, over=still_over)
    if still_over:
        log.warning(
            "eval.analysis the writer's part for aspect %s is %d characters over its budget of %d after every trim: "
            "what is left (code's readings, the scoreboard, the design, coverage and the disclosures) is never trimmed",
            view["aspect"],
            len(_compact(view)) - budget,
            budget,
        )


def _trimmed_note(left_out: Mapping[str, int], budget: int, *, over: bool) -> str:
    """What an aspect's part says it left out to fit the writer's budget."""
    said = [f"{count} {what}" for what, count in left_out.items() if count]
    return (
        f"To fit the writer's budget of {budget} characters per aspect, this aspect leaves out: "
        + (", ".join(said) or "nothing it could trim")
        + " (the excerpts of flagged trials; flagged trials, last first; contrasts, least separated first; the "
        "runs' configs, last first; then the cells' readings, last first). The page shows all of them."
        + (" It is still over budget: what is left is never trimmed." if over else "")
    )


def _compact(node: Any) -> str:
    """The compact JSON a budget is counted in and the message is sent as."""
    return json.dumps(node, separators=(",", ":"), ensure_ascii=False)


def _without(node: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    """``node`` without ``keys``."""
    return {key: value for key, value in node.items() if key not in keys}


def _figure(value: float | None) -> float | None:
    """A figure to the significant figures the writer's view keeps; None, NaN and infinities as None."""
    if value is None or not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    # Rounded as a number, not spelled: the view is JSON, and json writes the float.
    return round(value, _FIGURES - 1 - math.floor(math.log10(abs(value))))


#: The figures code computes on an arm's scoreboard row, which the writer reads as :func:`_figure` keeps them.
_SCOREBOARD_FIGURES: frozenset[str] = frozenset({"test_cost_usd", "reply_cost_usd"})

#: The figures code computes on a bar's verdict for one cell; its margin is declared, and read as declared.
_BAR_VERDICT_FIGURES: frozenset[str] = frozenset({"value", "sem", "ci_low", "ci_high"})


def _figured(node: dict[str, Any], names: frozenset[str]) -> dict[str, Any]:
    """``node`` with each of ``names`` it holds as :func:`_figure` keeps it."""
    return {key: _figure(value) if key in names else value for key, value in node.items()}


def _figures(values: Sequence[float] | None) -> list[float | None] | None:
    """An interval's ends, each as :func:`_figure` keeps it."""
    return None if values is None else [_figure(value) for value in values]


def _points(value: float | None) -> float | None:
    """A figure in points or percent as the page prints it: to one decimal, where a percent or a point is read; a
    non-zero figure that would round to nought keeps :func:`_figure`'s, so it never reads as no change."""
    if value is None or not math.isfinite(value):
        return None
    rounded = round(value, 1)
    return _figure(value) if rounded == 0 and value != 0 else rounded


def _points_pair(values: Sequence[float] | None) -> list[float | None] | None:
    """An interval in points, each end as :func:`_points` spells it."""
    return None if values is None else [_points(value) for value in values]


def _tail(text: str, limit: int) -> str:
    """The last ``limit`` characters of ``text``, led by an ellipsis when cut."""
    return text if len(text) <= limit else "…" + text[-limit:]


def campaign_trial_ids(aspects: Iterable[tuple[str, Sequence[str]]]) -> dict[str, tuple[str, str]]:
    """The ids the writer names a campaign's trials by: ``T1`` … ``TN``, aspect after aspect, each trial in its bundle's
    order. The one rule: what the writer is shown (:meth:`CampaignBundle.trial_ids`) and every page that reads a
    writer's id back to its trial number them here.

    :param aspects: Each aspect's label and its bundle's trial ids, in the campaign's order.
    :ptype aspects: Iterable[tuple[str, Sequence[str]]]
    :return: ``campaign trial id -> (aspect, the trial's id in that aspect's bundle)``.
    :rtype: dict[str, tuple[str, str]]
    """
    ids: dict[str, tuple[str, str]] = {}
    for aspect, trials in aspects:
        for trial in trials:
            ids[f"T{len(ids) + 1}"] = (aspect, trial)
    return ids


__all__ = [
    "CAMPAIGN_BUNDLE_SHAPE",
    "CAMPAIGN_MESSAGE_CHARS_PER_ASPECT",
    "CAMPAIGN_MESSAGE_HEADING",
    "WRITER_EXCERPT_CHARS",
    "WRITER_REASON_CHARS",
    "CampaignBundle",
    "aspect_view",
    "campaign_evidence_digest",
    "build_campaign_bundle",
    "build_campaign_user_message",
    "campaign_trial_ids",
    "writer_view",
]
