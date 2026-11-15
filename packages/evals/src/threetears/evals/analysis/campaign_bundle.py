"""The campaign bundle: a campaign's whole evidence, across every aspect.

Code reports, the writer recommends, a person decides. A campaign asks one question of its arms over one or more
aspects, and each aspect's evidence is its own :class:`~threetears.evals.analysis.bundle.schema.AnalysisContextBundle`.
A :class:`CampaignBundle` holds the campaign's question, arms and floors, code's readings across every aspect
(:class:`~threetears.evals.analysis.campaign_readings.CampaignReadings`), and each aspect's bundle whole: what code's
page lays out and what the report writer's snapshot freezes.

**One naming across aspects.** A cell is one arm under one rig, and the same arm under the same rig is the same
``cell_ref`` in every aspect it ran in, so a ref alone does not say which aspect's evidence it is. A cell is named by an
alias unique across the campaign (``c1`` … ``cN``, :func:`~threetears.evals.analysis.references.campaign_cell_aliases`)
and a trial by an id unique across it (``T1`` … ``TN``); :meth:`CampaignBundle.cell` and :meth:`CampaignBundle.trial`
map each back to its aspect. A campaign of one aspect names both exactly as that aspect's bundle does. An arm is named
by its label (:attr:`CampaignBundle.arms`), never by its variant key.

**Frozen as a frame beside the aspects.** A stored analysis keeps each aspect's bundle in its own
:class:`~threetears.evals.kernel.campaign.AspectAnalysis`, and what the campaign adds over them as the bundle's
:meth:`~CampaignBundle.frame`; :meth:`CampaignBundle.from_frame` puts the two back together.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from pydantic import Field, field_validator, model_validator

from threetears.evals.analysis.arms import arm_names, short_digest
from threetears.evals.analysis.bundle.schema import (
    AnalysisContextBundle,
    refuse_a_bundle_of_another_shape,
    unnamed_axes,
)
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaign_readings import (
    CampaignReadings,
)
from threetears.evals.analysis.kinds import CampaignKind, campaign_kind
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.references import CellAddress, campaign_cell_aliases
from threetears.evals.analysis.style import lever_spelled
from threetears.evals.kernel.campaign import CampaignDeclaration, EvalCampaign
from threetears.evals.kernel.declaration import (
    ARM_PRESENTATION_FIELDS,
    ASPECT_PRESENTATION_FIELDS,
    CampaignArm,
    resolve_control_arm,
)
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.surface import DecisionSurface
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.hashing import identity_digest
from threetears.evals.schema.logs import get_logger

log = get_logger(__name__)

#: The campaign bundle's own shape: what it is built at, and the one shape :meth:`CampaignBundle.from_dict` and
#: :meth:`CampaignBundle.from_frame` read a frozen copy back as. 2: each reading and pair states both arms' depth, the
#: paired cases that differ in it, and what limits its interval (`precision`). 3: the readings carry each arm's figures
#: across every aspect (`readings.aggregate`). 4: names left the readings, with bundle shape 66 (owner ruling,
#: 2026-11-11): a floor's status holds its decision as facts (`FloorStatus.decision`), a bar's its decision
#: (`BarStatus.decision`), a measure's reading no name, and a derived arm spelled from several levers carries its
#: spelling (`arm_spellings`). No earlier shape is read: a frozen copy of one is refused, and what holds it regenerated.
CAMPAIGN_BUNDLE_SHAPE = 4

#: The campaign's own fields its fingerprint leaves out: what a list prints for it. A campaign renamed is the same
#: evidence, so renaming it moves no fingerprint (its description never reaches the bundle at all).
CAMPAIGN_PRESENTATION_FIELDS: frozenset[str] = frozenset({"name"})

#: What a campaign bundle's fingerprint starts with: version 3, the digest of :func:`~threetears.evals.schema.hashing.
#: identity_digest` (floats rounded) over its evidence without the campaign's name. A fingerprint of another version was
#: taken over a frame of an earlier shape, which no build reads, so it never matches: what holds it is regenerated.
_CAMPAIGN_FINGERPRINT_PREFIX: Final[str] = "v3:"

#: What each kind of campaign asks of its report, as the report input says it (``report.input_build``).
KIND_MEANS: dict[str, str] = {
    "measurement": "One arm, measured: there is nothing to choose between, so the report recommends no arm.",
    "choice": (
        "Several arms and no control: the report picks the arm the evidence favours, or recommends none when no arm "
        "stood out (no clear winner); a pick of an arm shown ahead on no pair rests on cost or simplicity, at low "
        "confidence. A pick names every aspect another arm is shown ahead on."
    ),
    "change": (
        "A control and candidates: the report recommends a candidate the evidence separated, or keeps the control."
    ),
}


class ArmSpelling(EvalDocumentModel):
    """How code spelled a derived arm's label from several levers: each lever by id, its level as the label spells it,
    and what tells the arm from another spelled alike. What a page prints the arm by, each lever by its stored name
    (:meth:`label`), rebuilt from these facts and never by reading the label's text back; the frozen label is the same
    rule with each lever by its id read as words, which the campaign bundle holds to (:meth:`CampaignBundle.
    _one_campaign`).
    """

    levers: list[str] = Field(min_length=2, description="The levers the label is spelled from, by id, in its order.")
    levels: list[str] = Field(
        min_length=2,
        description=(
            "Each lever's level as the label spells it, in the same order: the level alone, cut as a name cuts a level."
        ),
    )
    digest: str | None = Field(
        default=None,
        description=(
            "The digest that tells the arm from another spelled alike, as its label shows it (`0123456789ab`, after "
            "`arm`); None where no other arm reads alike."
        ),
    )
    taken: int | None = Field(
        default=None,
        ge=2,
        description="The number the label carries where its spelling was already taken (`(2)`, `(3)` …); None for none.",
    )

    def suffix(self) -> str:
        """What follows the levers in the label and tells the arm from another: `` (arm <digest>)``, then
        `` (<taken>)``; empty for neither.

        :return: The suffix.
        :rtype: str
        """
        return (f" (arm {self.digest})" if self.digest else "") + (f" ({self.taken})" if self.taken else "")

    def label(self, declared: Mapping[str, str] | None = None) -> str:
        """The arm's name, each lever by its stored name (``declared``), or by its id read as words with none: the
        frozen label.

        :param declared: Each lever's stored reader name, by id (``CampaignBundle.lever_names``); None for none.
        :ptype declared: Mapping[str, str] | None
        :return: The name.
        :rtype: str
        """
        return lever_spelled(self.levers, self.levels, declared) + self.suffix()


class CampaignBundle(EvalDocumentModel):
    """A campaign's whole evidence for its one report: the question, arms and floors, code's readings, every aspect.

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
    arm_spellings: dict[str, ArmSpelling] = Field(
        default_factory=dict,
        description=(
            "How code spelled each derived arm whose label it spelled from several levers, by the arm's variant key "
            "(:func:`~threetears.evals.analysis.campaigns.campaign_arms_spelled`): what a page prints the arm by, each "
            "lever by its stored name. An arm spelled from one level, or one a person named, has none."
        ),
    )
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
            two arms share a label, two aspects name one lever apart, or an arm's spelling is not of a derived arm of
            the campaign or does not spell its label.
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
        levers: dict[str, set[str]] = {}
        for bundle in self.aspects:
            for lever, name in bundle.lever_names.items():
                levers.setdefault(lever, set()).add(name)
        if apart := sorted(lever for lever, held in levers.items() if len(held) > 1):
            raise ValueError(
                f"the campaign bundle's aspects name lever(s) {apart} apart; one campaign names a lever one way"
            )
        given = {key for arm in self.declaration.arms for key in self.arm_keys(arm)}
        derived = {arm.variant_key: arm for arm in self.arms if arm.variant_key not in given}
        for key, spelling in self.arm_spellings.items():
            if key not in derived:
                raise ValueError(f"the campaign bundle spells arm {key}, which is no arm code derived for it")
            if spelling.label() != derived[key].label:
                raise ValueError(
                    f"the campaign bundle spells arm {key} as {spelling.label()!r}, and its label is "
                    f"{derived[key].label!r}; an arm's spelling is what its label was spelled from"
                )
        return self

    # ------------------------------------------------------------------ identity and storage

    def frame(self) -> dict[str, Any]:
        """What the campaign adds over its aspects' bundles, as a stored analysis keeps it.

        :return: Every field but the aspects' bundles, which become ``{aspect, fingerprint}`` pairs.
        :rtype: dict[str, Any]
        """
        frame = self.model_dump(mode="json", exclude={"aspects"})
        frame["aspects"] = [{"aspect": bundle.aspect, "fingerprint": bundle.fingerprint()} for bundle in self.aspects]
        return frame

    def fingerprint(self) -> str:
        """The campaign bundle's fingerprint: its frame's, which names each aspect's bundle by its fingerprint, without
        the campaign's name (:data:`CAMPAIGN_PRESENTATION_FIELDS`) or how a page prints each aspect and arm.

        Its floats rounded to twelve significant figures, so one campaign's evidence fingerprints alike across Python
        builds, except, rarely, a value that falls exactly on a 12th-figure rounding boundary
        (:meth:`AnalysisContextBundle.fingerprint` says why).

        :return: ``v3:`` and a 64-character lowercase hex digest.
        :rtype: str
        """
        return _campaign_digest(self.frame())

    @staticmethod
    def refuse_another_shape(data: Mapping[str, Any], *, what: str, remedy: str) -> None:
        """Refuse a frozen campaign bundle, or frame, of a shape this build does not read.

        :param data: The frozen document.
        :ptype data: Mapping[str, Any]
        :param what: What holds it, as a sentence names it.
        :ptype what: str
        :param remedy: What to do instead.
        :ptype remedy: str
        :raises RefusedInput: Its ``schema_version`` is not :data:`CAMPAIGN_BUNDLE_SHAPE` (one frozen at an earlier
            shape is refused saying it was frozen before names left the readings), or it is not a campaign bundle at
            all (one aspect's bundle, frozen before one memo covered a campaign).
        """
        shape = data.get("schema_version")
        if "readings" in data and shape == CAMPAIGN_BUNDLE_SHAPE:
            return
        if "readings" not in data:
            when = "before one memo covered a campaign's every aspect"
        elif isinstance(shape, int) and shape < CAMPAIGN_BUNDLE_SHAPE:
            when = (
                "before names left the readings: its floors' reasons were worded, and its derived arms spelled, when "
                "it was frozen"
            )
        else:
            when = "by a build whose readings carried another shape"
        raise RefusedInput(
            f"{what} holds evidence of shape {shape} that is not a campaign bundle of this build's shape "
            f"({CAMPAIGN_BUNDLE_SHAPE}): it was frozen {when}, so this build cannot read it",
            remedy=remedy,
        )

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, what: str = "this campaign bundle", remedy: str = ""
    ) -> CampaignBundle:
        """Read a whole campaign bundle back, refusing any shape but this build's with a remedy.

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
        return cls.model_validate({**_held(data), "aspects": bundles})

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
        """Put a stored frame and its aspects' bundles back together, refusing a frame of another shape or a pair that
        do not belong together.

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
        held = [(bundle.aspect, bundle.fingerprint()) for bundle in bundles]
        if named != held:
            raise ValueError(f"{what}'s campaign bundle names aspects {named}, and its aspects hold {held}")
        return cls.model_validate({**_held(frame), "aspects": bundles})

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

    @property
    def lever_names(self) -> dict[str, str]:
        """What every page calls each lever, by id, as the aspects' bundles carry it
        (:attr:`~threetears.evals.analysis.bundle.schema.AnalysisContextBundle.lever_names`): every aspect names a
        lever by the one rule across the campaign, so they never name one apart (:meth:`_one_campaign` refuses
        aspects that do), and this is every lever any of them names.

        :return: Lever id to reader name, sorted by id.
        :rtype: dict[str, str]
        """
        return dict(
            sorted({lever: name for bundle in self.aspects for lever, name in bundle.lever_names.items()}.items())
        )

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


def _campaign_digest(frame: Mapping[str, Any]) -> str:
    """A campaign bundle frame's fingerprint: :data:`_CAMPAIGN_FINGERPRINT_PREFIX` and the digest of its evidence
    without the campaign's name."""
    evidence = _evidence_of(frame)
    unnamed = {key: value for key, value in evidence.items() if key not in CAMPAIGN_PRESENTATION_FIELDS}
    return _CAMPAIGN_FINGERPRINT_PREFIX + identity_digest(unnamed)


def _held(document: Mapping[str, Any]) -> dict[str, Any]:
    """A frozen campaign bundle's or frame's fields but its aspects."""
    return {key: value for key, value in document.items() if key != "aspects"}


def _evidence_of(frame: Mapping[str, Any]) -> dict[str, Any]:
    """A campaign bundle's frame as its fingerprint reads it: without how a page prints each aspect and each arm
    (:data:`~threetears.evals.kernel.declaration.ASPECT_PRESENTATION_FIELDS`,
    :data:`~threetears.evals.kernel.declaration.ARM_PRESENTATION_FIELDS`).

    A short name is presentation, as an aspect bundle's own is (left out of its fingerprint), and so is what a page
    calls a declared lever and each of its declared levels
    (:data:`~threetears.evals.kernel.declaration.AXIS_PRESENTATION_FIELDS`,
    :data:`~threetears.evals.kernel.declaration.LEVEL_PRESENTATION_FIELDS`), so naming an aspect, an arm, a lever or a
    level leaves every analysis and reporter case of the campaign reproducing. One naming none carries no key to drop.
    The campaign's own name stays here, and :func:`_campaign_digest` leaves it out.

    :param frame: The frame.
    :ptype frame: Mapping[str, Any]
    :return: The frame's evidence.
    :rtype: dict[str, Any]
    """
    frame = dict(frame)
    if isinstance(arms := frame.get("arms"), list):
        frame["arms"] = _unpresented(arms, ARM_PRESENTATION_FIELDS)
    declaration = frame.get("declaration")
    if not isinstance(declaration, Mapping):
        return frame
    declaration = dict(declaration)
    if isinstance(declaration.get("aspects"), list):
        declaration["aspects"] = [
            {**aspect, "design": unnamed_axes(aspect["design"])}
            if isinstance(aspect, Mapping) and isinstance(aspect.get("design"), Mapping)
            else aspect
            for aspect in _unpresented(declaration["aspects"], ASPECT_PRESENTATION_FIELDS)
        ]
    if isinstance(declaration.get("arms"), list):
        declaration["arms"] = _unpresented(declaration["arms"], ARM_PRESENTATION_FIELDS)
    return {**frame, "declaration": declaration}


def _unpresented(items: Sequence[Any], fields: frozenset[str]) -> list[Any]:
    """``items`` with each mapping's presentation ``fields`` left out; anything else as it is."""
    return [
        {key: value for key, value in item.items() if key not in fields} if isinstance(item, Mapping) else item
        for item in items
    ]


def build_campaign_bundle(
    campaign: EvalCampaign,
    bundles: Sequence[AnalysisContextBundle],
    *,
    arms: Sequence[CampaignArm],
    readings: CampaignReadings,
    spellings: Mapping[str, ArmSpelling] | None = None,
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
    :param spellings: How code spelled each derived arm it spelled from several levers, by variant key
        (:func:`~threetears.evals.analysis.campaigns.campaign_arms_spelled`); None for none.
    :ptype spellings: Mapping[str, ArmSpelling] | None
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
        arm_spellings=dict(spellings or {}),
        aspects=list(bundles),
    )


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
    "ArmSpelling",
    "CampaignBundle",
    "build_campaign_bundle",
    "campaign_trial_ids",
]
