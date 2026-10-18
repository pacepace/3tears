"""The types a campaign of aspects is made of hold the rules today's design holds, and a campaign reads as its aspects.

The aspect's design is today's design less what moved to the campaign, validated by the same rules; an arm is a
variant key kept beside its pre-image, as the design's control is; a floor's key is derived from its aspect and its
clause; and a stored campaign reads, through :func:`aspect_views`, as the one aspect it upgrades to.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.factories import make_aspect_campaign, make_campaign, minimal_aspect_design
from threetears.evals.analysis import AspectReading, CampaignReadings, aspect_view, aspect_views
from threetears.evals.kernel.campaign_upgrade import UPGRADED_CONTROL_LABEL
from threetears.evals.kernel.declaration import (
    Aspect,
    AspectDesign,
    BarOverride,
    CampaignArm,
    CampaignCall,
    CampaignDesign,
    ControlDeclaration,
    CostLower,
    Floor,
    GuardrailMargin,
    RateDrop,
    SweptAxis,
    resolve_control_arm,
    resolve_declared_control,
)
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.identity import IDENTITY_VERSION, compute_variant_key
from threetears.evals.schema.values import SweepableValue

_LEVERS = {"model": SweepableValue.of("glm", display="GLM")}
_KEY = compute_variant_key(_LEVERS)

#: The design fields that belong to the campaign once it has aspects.
_MOVED = {"questions", "control", "control_levers", "control_identity_version", "declared_at", "declared_by"}


def _axis(axis_id: str, *levels: str) -> SweptAxis:
    return SweptAxis(axis_id=axis_id, values=[SweepableValue.of(level, display=level) for level in levels])


def _design_fields(**overrides: Any) -> dict[str, Any]:
    """Fields both designs share, as an author writes them: two axes, a bar, a guardrail margin, a star crossing."""
    fields: dict[str, Any] = dict(
        axes=[_axis("model", "glm", "kimi"), _axis("temperature", "0", "1")],
        bars=[BarOverride(measure_id="pass_rate", threshold=0.8, direction="higher_is_better")],
        guardrail_margins=[GuardrailMargin(dimension="safety", margin=0.5)],
        held_fixed=ControlDeclaration(stimulus="controlled", apparatus="commissioned"),
        merit_priority=["quality", "cost"],
        crossing="star",
        skipped_cells=[{"model": "kimi", "temperature": "1"}],
        intended_repetitions=3,
    )
    fields.update(overrides)
    return fields


class TestTheAspectDesign:
    def test_it_accepts_what_the_campaign_design_accepts_less_what_moved(self) -> None:
        campaign = CampaignDesign(**_design_fields(), control=_KEY, control_levers=_LEVERS)
        stored = {key: value for key, value in campaign.model_dump(mode="json").items() if key not in _MOVED}

        aspect = AspectDesign.model_validate({**stored, "merit_axes": ["quality"]})

        assert aspect.model_dump(mode="json") == {**stored, "merit_axes": ["quality"]}
        assert aspect.declares_cells() == campaign.declares_cells()
        for cell in ({"model": "glm", "temperature": "0"}, {"model": "kimi", "temperature": "1"}):
            identity = aspect.cell_identity(cell)
            assert identity == campaign.cell_identity(cell)
            assert aspect.skipped_by_design(identity) == campaign.skipped_by_design(identity)
        assert aspect.skipped_by_design(aspect.cell_identity({"model": "kimi", "temperature": "1"}))

    @pytest.mark.parametrize("moved", sorted(_MOVED))
    def test_a_field_that_moved_to_the_campaign_is_refused(self, moved: str) -> None:
        campaign = CampaignDesign(**_design_fields(), control=_KEY, control_levers=_LEVERS).model_dump(mode="json")
        stored = {key: value for key, value in campaign.items() if key not in _MOVED}

        with pytest.raises(ValidationError, match=moved):
            AspectDesign.model_validate({**stored, moved: campaign[moved]})

    @pytest.mark.parametrize(
        ("overrides", "refusal"),
        [
            pytest.param({"axes": [_axis("model", "glm"), _axis("model", "kimi")]}, "duplicate axis ids", id="axis"),
            pytest.param(
                {
                    "bars": [
                        BarOverride(measure_id="pass_rate", threshold=t, direction="higher_is_better")
                        for t in (0.7, 0.8)
                    ]
                },
                "more than one bar on: pass_rate",
                id="bar",
            ),
            pytest.param(
                {"guardrail_margins": [GuardrailMargin(dimension="safety", margin=m) for m in (0.5, 1.0)]},
                "more than one guardrail margin on: safety",
                id="guardrail",
            ),
            pytest.param({"merit_priority": ["cost", "cost"]}, "merit_priority names cost", id="priority"),
            pytest.param(
                {"skipped_cells": [{"model": "nowhere", "temperature": "0"}]},
                "is none of its declared levels",
                id="cell",
            ),
        ],
    )
    def test_it_refuses_what_the_campaign_design_refuses_in_the_same_words(
        self, overrides: dict[str, Any], refusal: str
    ) -> None:
        with pytest.raises(ValidationError, match=refusal):
            CampaignDesign(**_design_fields(**overrides))
        with pytest.raises(ValidationError, match=refusal):
            AspectDesign(**_design_fields(**overrides))

    def test_it_refuses_a_merit_axis_named_twice(self) -> None:
        with pytest.raises(ValidationError, match="merit_axes names quality more than once"):
            minimal_aspect_design(merit_axes=["quality", "quality"])


class TestTheArm:
    def test_an_undated_key_is_dated_by_this_build(self) -> None:
        arm = CampaignArm.model_validate({"label": "control", "role": "control", "variant_key": _KEY})

        assert arm.identity_version == IDENTITY_VERSION

    def test_a_key_that_is_not_a_digest_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="64-character lowercase hex digest"):
            CampaignArm(label="control", role="control", variant_key="run-1", identity_version=IDENTITY_VERSION)

    def test_a_lever_map_that_is_not_the_key_s_pre_image_is_refused(self) -> None:
        other = {"model": SweepableValue.of("kimi", display="Kimi")}

        with pytest.raises(ValidationError, match="not the arm's pre-image"):
            CampaignArm(
                label="control", role="control", variant_key=_KEY, levers=other, identity_version=IDENTITY_VERSION
            )

    def test_an_earlier_predicate_s_map_is_not_held_to_this_one(self) -> None:
        other = {"model": SweepableValue.of("kimi", display="Kimi")}

        arm = CampaignArm(label="control", role="control", variant_key=_KEY, levers=other, identity_version=1)

        assert resolve_control_arm(arm).variant_keys == [compute_variant_key(other), _KEY]

    @pytest.mark.parametrize("version", [IDENTITY_VERSION, 1])
    @pytest.mark.parametrize("levers", [_LEVERS, None])
    def test_it_resolves_as_the_design_s_control_does(
        self, version: int, levers: dict[str, SweepableValue] | None
    ) -> None:
        arm = CampaignArm(label="control", role="control", variant_key=_KEY, levers=levers, identity_version=version)
        design = CampaignDesign(
            **_design_fields(), control=_KEY, control_levers=levers, control_identity_version=version
        )

        assert resolve_control_arm(arm) == resolve_declared_control(design)

    def test_no_arm_resolves_as_no_control(self) -> None:
        assert resolve_control_arm(None) == resolve_declared_control(None)


class TestTheFloor:
    @pytest.mark.parametrize(
        ("clause", "key"),
        [
            pytest.param(RateDrop(max_points=5), "tone/rate_drop/*", id="every-rate"),
            pytest.param(
                RateDrop(max_points=5, reading="cites a source"), "tone/rate_drop/cites a source", id="one-rate"
            ),
            pytest.param(CostLower(), "tone/cost_lower/*", id="cost"),
        ],
    )
    def test_its_key_is_its_aspect_kind_and_reading(self, clause: RateDrop | CostLower, key: str) -> None:
        floor = Floor(aspect="tone", clause=clause)

        assert floor.key == key
        assert Floor.model_validate(floor.model_dump()) == floor


class TestTheAspect:
    @pytest.mark.parametrize("margin", [0.0, 100.0, -1.0, float("nan")])
    def test_a_margin_off_the_scale_is_refused(self, margin: float) -> None:
        with pytest.raises(ValidationError, match="margin"):
            Aspect(label="tone", margin=margin)

    def test_it_reads_every_stratum(self) -> None:
        with pytest.raises(ValidationError, match="stratum"):
            Aspect.model_validate({"label": "tone", "stratum": "hard"})


class TestUniqueLabels:
    def test_a_call_names_each_overridden_floor_once(self) -> None:
        with pytest.raises(ValidationError, match="floors_overridden names tone/cost_lower/\\* more than once"):
            CampaignCall(arm="candidate", floors_overridden=["tone/cost_lower/*"] * 2, called_by="web:someone")

    def test_readings_name_each_aspect_once(self) -> None:
        with pytest.raises(ValidationError, match="aspects names tone more than once"):
            CampaignReadings(
                campaign_id="campaign-1",
                alpha=0.05,
                control=None,
                control_label="",
                aspects=["tone", "tone"],
                readings=[],
                floors=[],
            )

    def test_readings_are_read_back_by_aspect(self) -> None:
        row = AspectReading(
            aspect="tone",
            arm=_KEY,
            arm_label="GLM",
            change=4.0,
            interval=(-1.0, 9.0),
            confidence=0.95,
            gain_p=0.2,
            gain_p_holm=0.4,
            drop_p=None,
            drop_p_holm=None,
            reading="unclear",
            margin=None,
            control_cases=10,
            arm_cases=10,
            control_passed=6,
            control_trials=10,
            arm_passed=7,
            arm_trials=10,
            projection=None,
        )
        readings = CampaignReadings(
            campaign_id="campaign-1",
            alpha=0.05,
            control=None,
            control_label="",
            aspects=["tone", "speed"],
            readings=[row],
            floors=[],
        )

        assert (readings.on_aspect("tone"), readings.on_aspect("speed"), readings.broken_floors()) == ([row], [], [])


class TestAspectViews:
    def test_a_stored_campaign_is_one_view_of_its_template_and_runs(self) -> None:
        campaign = make_aspect_campaign(label="tone", template_id="template-1", run_ids=["run-1", "run-2"])

        (view,) = aspect_views(campaign)

        assert (view.campaign_id, view.scope_id, view.behavior) == (campaign.id, campaign.scope_id, campaign.behavior)
        assert (view.aspect, view.template_id, view.run_ids) == ("tone", "template-1", ["run-1", "run-2"])
        assert aspect_view(campaign) == aspect_view(campaign, "tone") == view

    def test_its_declaration_is_what_the_factory_declared(self) -> None:
        control = CampaignArm(
            label=UPGRADED_CONTROL_LABEL,
            role="control",
            variant_key=_KEY,
            levers=_LEVERS,
            identity_version=IDENTITY_VERSION,
        )
        design = minimal_aspect_design(merit_axes=["quality"])
        campaign = make_aspect_campaign(
            label="tone",
            design=design,
            question="Is the candidate kinder?",
            control=control,
            floors=[RateDrop(max_points=5), CostLower()],
        )

        declared = aspect_view(campaign).declared

        assert declared is not None
        assert (declared.question, declared.design, declared.control, declared.margin) == (
            "Is the candidate kinder?",
            design,
            control,
            5.0,
        )
        assert [floor.key for floor in declared.floors] == ["tone/rate_drop/*", "tone/cost_lower/*"]

    def test_a_campaign_with_no_design_is_exploratory(self) -> None:
        assert aspect_view(make_campaign(run_ids=["run-1"])).declared is None

    def test_an_aspect_the_campaign_does_not_hold_is_refused_by_name(self) -> None:
        with pytest.raises(RefusedInput, match="no aspect 'speed'") as refused:
            aspect_view(make_aspect_campaign(label="tone"), "speed")

        assert "tone" in refused.value.remedy
