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
the flagged trials' excerpts, then the flagged trials, then the contrasts least separated first, then the cells'
readings last — and says in the aspect's part what it left out, since the page shows all of it. Code's own readings
across aspects and the floors are never trimmed.

**Frozen as a frame beside the aspects.** A stored analysis keeps each aspect's bundle in its own
:class:`~threetears.evals.kernel.campaign.AspectAnalysis`, and what the campaign adds over them as the bundle's
:meth:`~CampaignBundle.frame`; :meth:`CampaignBundle.from_frame` puts the two back together.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field, field_validator, model_validator

from threetears.evals.analysis.arms import arm_names, multi_rig_variants, short_digest, writer_arms
from threetears.evals.analysis.bundle.caps import capped_text
from threetears.evals.analysis.bundle.schema import (
    AnalysisContextBundle,
    evidence_digest,
    refuse_a_bundle_before_aspects,
)
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.references import CellAddress, campaign_cell_aliases
from threetears.evals.kernel.campaign import CampaignDeclaration, EvalCampaign
from threetears.evals.kernel.declaration import CampaignArm, resolve_control_arm
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.surface import DecisionSurface
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.hashing import canonical_digest

#: The campaign bundle's own shape: what :meth:`CampaignBundle.from_dict` and :meth:`CampaignBundle.from_frame`
#: accept, and the only one. A frozen copy of another shape is refused with a remedy, never read.
CAMPAIGN_BUNDLE_SHAPE = 1

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
    "measurement_window_disclosure",
    "apparatus_confounds",
    "aliased_factors",
    "unexercised_factors",
    "differential_exclusions",
    "arm_served_models",
    "subject_key_instabilities",
    "frontier_bar_withheld",
    "retracted_insights",
)


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

    def frame(self) -> dict[str, Any]:
        """What the campaign adds over its aspects' bundles, as a stored analysis keeps it.

        :return: Every field but the aspects' bundles, which become ``{aspect, fingerprint}`` pairs.
        :rtype: dict[str, Any]
        """
        frame = self.model_dump(mode="json", exclude={"aspects"})
        frame["aspects"] = [{"aspect": bundle.aspect, "fingerprint": bundle.fingerprint()} for bundle in self.aspects]
        return frame

    def fingerprint(self) -> str:
        """The sha256 of the campaign bundle: its frame, which names each aspect's bundle by its fingerprint.

        :return: A 64-character lowercase hex digest.
        :rtype: str
        """
        return canonical_digest(self.frame())

    def to_dict(self) -> dict[str, Any]:
        """The whole campaign bundle as a dict: its frame, each aspect's bundle whole in its place.

        :return: The dict :meth:`from_dict` reads back.
        :rtype: dict[str, Any]
        """
        whole = self.model_dump(mode="json", exclude={"aspects"})
        whole["aspects"] = [bundle.to_dict() for bundle in self.aspects]
        return whole

    def to_json(self, *, indent: int | None = None) -> str:
        """:meth:`to_dict` as JSON.

        :param indent: The indent, or None for compact.
        :ptype indent: int | None
        :return: The JSON text.
        :rtype: str
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @staticmethod
    def refuse_another_shape(data: Mapping[str, Any], *, what: str, remedy: str) -> None:
        """Refuse a frozen campaign bundle, or frame, of a shape this build does not read.

        :param data: The frozen document.
        :ptype data: Mapping[str, Any]
        :param what: What holds it, as a sentence names it.
        :ptype what: str
        :param remedy: What to do instead.
        :ptype remedy: str
        :raises RefusedInput: Its ``schema_version`` is not :data:`CAMPAIGN_BUNDLE_SHAPE`, or it is not a campaign
            bundle at all (one aspect's bundle, frozen before one memo covered a campaign).
        """
        shape = data.get("schema_version")
        if "readings" not in data or shape != CAMPAIGN_BUNDLE_SHAPE:
            raise RefusedInput(
                f"{what} holds evidence of shape {shape} that is not a campaign bundle of this build's shape "
                f"({CAMPAIGN_BUNDLE_SHAPE}): it was frozen before one memo covered a campaign's every aspect, so this "
                "build cannot read it",
                remedy=remedy,
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, what: str = "this campaign bundle", remedy: str = "") -> CampaignBundle:
        """Read a whole campaign bundle back, refusing one of another shape with a remedy.

        :param data: A campaign bundle, as :meth:`to_dict` wrote it.
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
            refuse_a_bundle_before_aspects(aspect, what=what, remedy=fix)
            bundles.append(AnalysisContextBundle.from_dict(dict(aspect)))
        return cls.model_validate({**{key: value for key, value in data.items() if key != "aspects"}, "aspects": bundles})

    @classmethod
    def from_json(cls, text: str) -> CampaignBundle:
        """Read :meth:`to_json`'s text back.

        :param text: The JSON text.
        :ptype text: str
        :return: The bundle.
        :rtype: CampaignBundle
        :raises RefusedInput: See :meth:`from_dict`.
        """
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_frame(
        cls, frame: Mapping[str, Any], bundles: Sequence[AnalysisContextBundle], *, what: str = "this analysis"
    ) -> CampaignBundle:
        """Put a stored frame and its aspects' bundles back together, refusing a pair that do not belong together.

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
        return cls.model_validate({**{key: value for key, value in frame.items() if key != "aspects"}, "aspects": bundles})

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
        ids: dict[str, tuple[str, str]] = {}
        for bundle in self.aspects:
            for trial in bundle.trials:
                ids[f"T{len(ids) + 1}"] = (bundle.aspect, trial.trial_id)
        return ids

    def trial(self, trial_id: str) -> tuple[str, str] | None:
        """The aspect and bundle trial a campaign trial id names; None for an id the campaign does not hold.

        :param trial_id: The id, as the writer named it.
        :ptype trial_id: str
        :return: ``(aspect, bundle trial id)``, or None.
        :rtype: tuple[str, str] | None
        """
        return self.trial_ids().get(trial_id)

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


def campaign_evidence_digest(document: Mapping[str, Any]) -> str:
    """A whole campaign bundle document's fingerprint, as :meth:`CampaignBundle.fingerprint` computes it.

    For a reader holding the stored document rather than the model: a frozen reporter case checks the bundle it
    holds against the fingerprint it recorded, before anything is rebuilt.

    :param document: A campaign bundle, as :meth:`CampaignBundle.to_dict` wrote it.
    :ptype document: Mapping[str, Any]
    :return: A 64-character lowercase hex digest.
    :rtype: str
    """
    frame = {key: value for key, value in document.items() if key != "aspects"}
    frame["aspects"] = [
        {"aspect": aspect.get("aspect"), "fingerprint": evidence_digest(aspect)} for aspect in document.get("aspects") or []
    ]
    return canonical_digest(frame)


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
    :return: The view, as the message carries it.
    :rtype: dict[str, Any]
    """
    namer = _Namer(bundle)
    floors = sorted(bundle.readings.floors, key=lambda status: status.status != "broken")
    first = bundle.aspects[0]
    view: dict[str, Any] = {
        "campaign": {
            "name": bundle.name,
            "behavior": bundle.behavior,
            "question": bundle.declaration.question,
            "aspects": bundle.labels,
        },
        "arms": _arms(bundle, namer),
        "floors": [
            {
                "floor": status.floor,
                "aspect": status.aspect,
                "arm": status.arm_label,
                "status": status.status,
                "reading": status.reading,
                "interval": _figures(status.interval),
                "reason": status.reason,
            }
            for status in floors
        ],
        "prior_insights": [
            {"statement": insight.statement, "confidence": insight.confidence} for insight in first.prior_insights
        ],
        # Stated so a short list is not read as the whole ledger.
        "prior_insights_omitted": first.prior_insights_omitted,
        "aspects": [_aspect_part(bundle, index, budget, namer) for index in range(len(bundle.aspects))],
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
    return view


def aspect_view(bundle: CampaignBundle, index: int, *, budget: int = CAMPAIGN_MESSAGE_CHARS_PER_ASPECT) -> dict[str, Any]:
    """One aspect's part of the writer's view, trimmed to ``budget`` characters of compact JSON.

    Trimmed in a fixed order, each step only as far as it must: the flagged trials' excerpts, last trial first; then
    the flagged trials themselves; then the contrasts, least separated first; then the cells' readings, last cell's
    last reading first. What was left out is said in ``trimmed``. Code's readings, the scoreboard, the design and the
    disclosures are never trimmed, so a part can stay over budget when they alone exceed it, and ``trimmed`` says so.

    :param bundle: The campaign bundle.
    :ptype bundle: CampaignBundle
    :param index: The aspect's position.
    :ptype index: int
    :param budget: The most characters the part may take.
    :ptype budget: int
    :return: The aspect's part.
    :rtype: dict[str, Any]
    """
    return _aspect_part(bundle, index, budget, _Namer(bundle))


def _aspect_part(bundle: CampaignBundle, index: int, budget: int, namer: _Namer) -> dict[str, Any]:
    """:func:`aspect_view`, naming through ``namer``."""
    view = _untrimmed(bundle, bundle.aspects[index], namer)
    _trim(view, budget)
    return view


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
            out[self.labels.get(key, key)] = self.walk(value, aspect, f"{path}.{key}")
        return out


def _arms(bundle: CampaignBundle, namer: _Namer) -> dict[str, Any]:
    """Every arm by its label and role, with the levels that tell it apart; the levels every arm shares, once."""
    entries = {entry.variant_key: entry for aspect in bundle.aspects for entry in aspect.variant_index}
    described = writer_arms(list(entries.values()))
    by_key = {arm["variant_key"]: arm for arm in described["arms"]}  # type: ignore[index, union-attr]
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


def _untrimmed(bundle: CampaignBundle, aspect: AnalysisContextBundle, namer: _Namer) -> dict[str, Any]:
    """One aspect's part of the view, whole."""
    label = aspect.aspect
    declared = aspect.declared
    design = aspect.declared.design if declared is not None else None
    payload = aspect.to_dict()
    view: dict[str, Any] = {
        "aspect": label,
        "question": {
            "asked_in_time": aspect.question_declared,
            "post_hoc": aspect.reading_scope.question_post_hoc,
            "margin": declared.margin if declared is not None else None,
            "merit_axes": design.merit_axes if design is not None else None,
            "merit_priority": design.merit_priority if design is not None else [],
            "reading_scope": namer.walk(payload["reading_scope"], label),
        },
        "readings": [
            {
                "arm": reading.arm_label,
                "reading": reading.reading,
                "change": _figure(reading.change),
                "gain_interval": _figures(reading.gain_interval),
                "gain_confidence": reading.gain_confidence,
                "drop_interval": _figures(reading.drop_interval),
                "drop_confidence": reading.drop_confidence,
                "margin": reading.margin,
                "control_passed": f"{reading.control_passed} of {reading.control_trials}",
                "arm_passed": f"{reading.arm_passed} of {reading.arm_trials}",
                "cases": [reading.control_cases, reading.arm_cases],
                "cases_to_settle": reading.projection.cases if reading.projection is not None else None,
                "confirmatory": reading.confirmatory,
                "why": reading.reason,
            }
            for reading in bundle.readings.on_aspect(label)
        ],
        "scoreboard": namer.walk(payload["scoreboard"], label),
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
        "coverage": namer.walk(
            [_without(row, "declared_levels") for row in payload["coverage"]], label
        ),
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
        "guardrails": namer.walk(
            [_without(check, "control") for check in payload["guardrails"]["checks"]], label
        ),
        "bars": namer.walk(payload["bar_adjudications"], label),
        "judge_evidence_tiers": namer.walk(payload["judge_evidence_tiers"], label),
        "goal_check_proofs": payload["goal_check_proofs"],
        "arm_mechanisms": namer.walk(payload["arm_mechanisms"], label),
        "verdict_order": namer.walk(payload["verdict_order"], label),
        "trials": _trials(aspect, namer),
        "disclosures": {
            key: namer.walk(payload[key], label) for key in _DISCLOSURES if payload.get(key) not in (None, "", [], {})
        },
    }
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
    return view


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
    """What each measure the cells carry means, keyed by name: never guessed from the name."""
    names = sorted({summary.name for cell in aspect.cell_measures for summary in cell.measures.measures})
    meanings: dict[str, Any] = {}
    for name in names:
        facts = aspect.measure_catalog.get(name)
        if facts is None:
            continue
        meanings[name] = {
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
                "cell": namer.cell(aspect.aspect, comparison.contrast.variant_key, comparison.contrast.apparatus_class_id),
                "control": namer.cell(aspect.aspect, comparison.control.variant_key, comparison.control.apparatus_class_id),
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
            rows.append(((0 if separated else 1, p), {key: value for key, value in row.items() if value not in (None, [])}))
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
    left_out = {"excerpts": 0, "trials": 0, "contrasts": 0, "cell readings": 0}
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
    for cell in reversed(view["cells"]):
        for group in ("judged", "measures"):
            while over() and cell[group]:
                cell[group].pop(next(reversed(cell[group])))
                left_out["cell readings"] += 1
    still_over = over()
    view["trimmed"] = _trimmed_note(left_out, budget, over=still_over)


def _trimmed_note(left_out: Mapping[str, int], budget: int, *, over: bool) -> str:
    """What an aspect's part says it left out to fit the writer's budget."""
    said = [f"{count} {what}" for what, count in left_out.items() if count]
    return (
        f"To fit the writer's budget of {budget} characters per aspect, this aspect leaves out: "
        + (", ".join(said) or "nothing it could trim")
        + " (the excerpts of flagged trials; flagged trials, last first; contrasts, least separated first; the "
        "cells' readings, last first). The page shows all of them."
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
    return float(f"{value:.{_FIGURES}g}")


def _figures(values: Sequence[float] | None) -> list[float | None] | None:
    """An interval's ends, each as :func:`_figure` keeps it."""
    return None if values is None else [_figure(value) for value in values]


def _tail(text: str, limit: int) -> str:
    """The last ``limit`` characters of ``text``, led by an ellipsis when cut."""
    return text if len(text) <= limit else "…" + text[-limit:]


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
    "writer_view",
]
