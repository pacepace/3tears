"""An arm's short name is declared (``CampaignArm.short_name``), never part of what anything keys on, and printed.

A campaign's arm may declare what a page prints for it where space is tight: at most 32 characters, in any script,
never blank, and no other arm's label or short name. It is declared on create (``ArmDefinition.short_name``), carried
by adoption, and set or cleared through ``campaign_update``'s ``arm_short_names``, which stays editable once runs
have joined. It is written only when set, so a campaign naming no arm is stored as durable v16.3 stored it, and a
v16.3 document reads. It is no part of any fingerprint (``test_declared_names`` pins every identity of a fixture
naming its arms against one naming nothing); here, naming an arm on a frozen campaign leaves its bundle's
fingerprint as it was.

Where declared, it is what the page prints: on each outcome bar's short label and the cost/quality chart's point,
ahead of the label the campaign gives the arm and of any short name the page would compute, and in the report input
(``Arm.short_name``), so ``{arm:<id>}`` prints it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.factories import fixture_variant_key
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture
from packages.evals.tests.report_shapes_support import Shape, s1_change
from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis import assemble_context_bundle, campaign_report
from threetears.evals.analysis.campaign_bundle import build_campaign_bundle
from threetears.evals.analysis.campaigns import create_campaign, update_campaign
from threetears.evals.analysis.report.blocks import ChartBlock
from threetears.evals.analysis.report.figure_keys import render
from threetears.evals.analysis.report.model import CampaignReport
from threetears.evals.analysis.report_input_build import planner_view, report_input
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignArm
from threetears.evals.kernel.errors import ValidationFailedError

#: The short names the s1 change campaign's arms declare where a test names them: the control's shorter than its
#: label, a candidate's longer than its own (a declared name wins either way), and one candidate left to the page.
NAMED = {"production": "Prod", "v2": "Second try", "v3": "Third"}


def _arm(label: str, **fields: Any) -> CampaignArm:
    return CampaignArm(label=label, role="candidate", variant_key=fixture_variant_key(label), **fields)


def _named(shape: Shape, names: dict[str, str]) -> Shape:
    """``shape`` with its arms naming ``names``, on the campaign and among the arms its bundle is built with."""
    arms = [arm.model_copy(update={"short_name": names.get(arm.label)}) for arm in shape.arms]
    campaign = shape.campaign.model_copy(
        update={"arms": [arm.model_copy(update={"short_name": names.get(arm.label)}) for arm in shape.campaign.arms]}
    )
    return Shape(
        name=shape.name,
        campaign=campaign,
        bundles=shape.bundles,
        reports=shape.reports,
        readings=shape.readings,
        arms=arms,
        runs=shape.runs,
        results=shape.results,
    )


def _update(fixture: Any, campaign_id: str, updates: dict[str, Any], **keywords: Any) -> EvalCampaign:
    host = fixture.host.eval_host
    return update_campaign(
        host.storage, campaign_id, TOYHOST_SCOPE, updates, updated_by="op", profile=host.profile, **keywords
    )


def _with_a_control(fixture: Any) -> EvalCampaign:
    """The corpus campaign, which holds its runs, with its control named from its first run (allowed once)."""
    return _update(fixture, fixture.campaign.id, {}, control_from_run_id=fixture.campaign.run_ids[0])


# --- the field --------------------------------------------------------------------------------------------------


class TestTheField:
    def test_one_of_at_most_the_cap_in_characters_is_kept_in_any_script(self) -> None:
        assert _arm("x", short_name="ا" * 32).short_name == "ا" * 32
        assert _arm("x", short_name="b" * 32).short_name == "b" * 32

    @pytest.mark.parametrize("name", ["b" * 33, "  ", ""], ids=["over the cap", "blank", "empty"])
    def test_an_over_long_or_blank_one_is_refused(self, name: str) -> None:
        with pytest.raises(ValidationError, match="arm 'x'"):
            _arm("x", short_name=name)

    def test_none_is_written_as_no_key_at_all(self) -> None:
        assert "short_name" not in _arm("x").model_dump()
        assert _arm("x", short_name="X").model_dump()["short_name"] == "X"

    @pytest.mark.parametrize(
        "names",
        [
            pytest.param({"v2": "Fast", "v3": "Fast"}, id="two alike"),
            pytest.param({"v2": "v3"}, id="another arm's label"),
            pytest.param({"production": "v4"}, id="the control printing a candidate's label"),
        ],
    )
    def test_two_arms_printed_alike_are_refused(self, names: dict[str, str]) -> None:
        campaign = s1_change().campaign
        arms = [arm.model_copy(update={"short_name": names.get(arm.label)}).model_dump() for arm in campaign.arms]
        with pytest.raises(ValidationError, match="would name more than one arm"):
            EvalCampaign.model_validate({**campaign.model_dump(), "arms": arms})

    def test_an_arm_may_be_short_named_its_own_label(self) -> None:
        campaign = s1_change().campaign
        arms = [arm.model_copy(update={"short_name": arm.label}).model_dump() for arm in campaign.arms]
        assert [arm.short_name for arm in EvalCampaign.model_validate({**campaign.model_dump(), "arms": arms}).arms]


class TestADurableRead:
    def test_a_v16_3_campaign_naming_no_arm_reads_and_is_written_back_unchanged(self) -> None:
        campaign = s1_change().campaign
        document = {**campaign.to_dict(), "schema_minor": 3}
        assert all("short_name" not in arm for arm in document["arms"])

        read = EvalCampaign.from_dict(document)

        assert [arm.short_name for arm in read.arms] == [None] * len(campaign.arms)
        assert read.to_dict()["arms"] == document["arms"]

    def test_a_named_arm_round_trips(self) -> None:
        campaign = _named(s1_change(), NAMED).campaign
        read = EvalCampaign.from_dict(campaign.to_dict())
        assert {arm.label: arm.short_name for arm in read.arms} == {
            arm.label: NAMED.get(arm.label) for arm in campaign.arms
        }


# --- declared, adopted, updated -----------------------------------------------------------------------------------


class TestThroughEveryWrite:
    def test_it_is_edited_on_a_frozen_campaign_and_moves_no_fingerprint(self) -> None:
        fixture = ops_fixture()
        host = fixture.host.eval_host
        controlled = _with_a_control(fixture)
        (control,) = controlled.arms
        before = assemble_context_bundle(controlled, storage=host.storage, profile=host.profile)

        named = _update(fixture, controlled.id, {"arm_short_names": {control.label: "Base"}})

        assert [arm.short_name for arm in named.arms] == ["Base"]
        assert named.declared_at == controlled.declared_at, "naming is no declaration: nothing is re-dated"
        assert named.arms[0].named_at == control.named_at, "naming an arm does not re-address it"
        after = assemble_context_bundle(named, storage=host.storage, profile=host.profile)
        assert after.declared is not None and after.declared.control is not None
        assert after.declared.control.short_name == "Base"
        assert after.fingerprint() == before.fingerprint()

        cleared = _update(fixture, controlled.id, {"arm_short_names": {control.label: None}})
        assert cleared.arms[0].short_name is None
        assert cleared.to_dict()["arms"] == controlled.to_dict()["arms"], "cleared, it is stored as before"

    @pytest.mark.parametrize(
        ("arm_short_names", "refusal"),
        [
            pytest.param({"no such arm": "x"}, "which the campaign names no arm by", id="unknown label"),
            pytest.param({"control": "b" * 33}, "at most 32", id="over the cap"),
            pytest.param(["control"], "maps an arm's label", id="not a map"),
        ],
    )
    def test_a_bad_one_is_refused_on_update(self, arm_short_names: object, refusal: str) -> None:
        fixture = ops_fixture()
        controlled = _with_a_control(fixture)
        with pytest.raises(ValidationFailedError, match=refusal):
            _update(fixture, controlled.id, {"arm_short_names": arm_short_names})

    def test_a_campaign_naming_no_arm_has_none_to_name(self) -> None:
        fixture = ops_fixture()
        assert not fixture.campaign.arms
        with pytest.raises(ValidationFailedError, match="names no arm by") as refused:
            _update(fixture, fixture.campaign.id, {"arm_short_names": {"extractor-v2": "v2"}})
        assert "name the campaign's arms first" in (refused.value.remedy or "")

    def test_a_create_declares_it_adoption_carries_it_and_an_over_long_one_is_refused(self) -> None:
        fixture = ops_fixture()
        host = fixture.host.eval_host
        source = fixture.campaign
        definition: dict[str, Any] = {
            "name": "armed",
            "subject_id": source.subject_id,
            "behavior": source.behavior,
            "arms": [{"label": "base", "role": "control", "from_run_ids": [source.run_ids[0]], "short_name": "b" * 33}],
        }
        with pytest.raises(ValidationFailedError, match="at most 32"):
            create_campaign(
                host.storage,
                definition,
                scope_id=TOYHOST_SCOPE,
                created_by="op",
                profile=host.profile,
                aspects_from=[source.id],
            )
        definition["arms"][0]["short_name"] = "Base"
        armed = create_campaign(
            host.storage,
            definition,
            scope_id=TOYHOST_SCOPE,
            created_by="op",
            profile=host.profile,
            aspects_from=[source.id],
        )
        assert [arm.short_name for arm in armed.arms] == ["Base"]

        adopted = create_campaign(
            host.storage,
            {"name": "adopted", "subject_id": source.subject_id, "behavior": source.behavior},
            scope_id=TOYHOST_SCOPE,
            created_by="op",
            profile=host.profile,
            aspects_from=[armed.id],
        )
        assert [(arm.label, arm.short_name) for arm in adopted.arms] == [("base", "Base")]

    async def test_the_update_action_sets_a_name_and_clears_it_with_null(self) -> None:
        fixture = off_loop(ops_fixture)
        controlled = off_loop(_with_a_control, fixture)
        label = controlled.arms[0].label
        tool = eval_catalogue().mount_all(standard_tools())[0]

        named = await tool.call(
            {"action": "campaign_update", "campaign_id": controlled.id, "arm_short_names": {label: "Base"}},
            host=fixture.host,
            caller=CALLER,
        )
        assert not named.is_error, named.text
        stored = off_loop(fixture.host.eval_host.storage.load_campaign, controlled.id, TOYHOST_SCOPE)
        assert stored is not None and [arm.short_name for arm in stored.arms] == ["Base"]

        cleared = await tool.call(
            {"action": "campaign_update", "campaign_id": controlled.id, "arm_short_names": {label: None}},
            host=fixture.host,
            caller=CALLER,
        )
        assert not cleared.is_error, cleared.text
        stored = off_loop(fixture.host.eval_host.storage.load_campaign, controlled.id, TOYHOST_SCOPE)
        assert stored is not None and [arm.short_name for arm in stored.arms] == [None]


# --- printed on the page and in the report input ------------------------------------------------------------------


def _outcome_short_names(page: CampaignReport) -> dict[str, set[str | None]]:
    """Each arm's short labels across every aspect's outcome bar, by the arm's name."""
    found: dict[str, set[str | None]] = {}
    for aspect in page.aspects:
        for block in aspect.report.blocks:
            if isinstance(block, ChartBlock) and block.chart == "outcome_bar":
                for bar in block.bars:
                    found.setdefault(bar.arm, set()).add(bar.short_name)
    return found


class TestThePagePrintsIt:
    def test_a_declared_name_is_every_outcome_bar_s_short_label_and_the_cost_chart_s(self) -> None:
        shape = _named(s1_change(), NAMED)
        page = campaign_report(shape.host(), shape.campaign.id, shape.campaign.scope_id)

        shorts = _outcome_short_names(page)
        assert shorts["production"] == {"Prod"}, "the control prints its declared name, not its label"
        assert shorts["v2"] == {"Second try"}, "a declared name wins though the label is shorter"
        assert shorts["v3"] == {"Third"}
        assert shorts["v4"] == {"v4"}, "an arm declaring none keeps the name the page computes"
        (cost,) = [block for block in page.blocks if isinstance(block, ChartBlock) and block.chart == "cost_quality"]
        assert {point.arm: point.short_name for point in cost.points} == {
            "production": "Prod",
            "v2": "Second try",
            "v3": "Third",
            "v4": "v4",
        }

    def test_naming_nothing_prints_what_it_did(self) -> None:
        shape = s1_change()
        shorts = _outcome_short_names(campaign_report(shape.host(), shape.campaign.id, shape.campaign.scope_id))
        assert shorts["production"] == {"production"} and shorts["v2"] == {"v2"}


class TestTheReportInputCarriesIt:
    def test_each_declared_name_reaches_its_arm_and_an_undeclared_one_none(self) -> None:
        shape = _named(s1_change(), NAMED)
        bundle = build_campaign_bundle(
            shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings
        )

        built = report_input(bundle)

        assert {arm.id: arm.short_name for arm in built.arms} == {
            "production": "Prod",
            "v2": "Second try",
            "v3": "Third",
            "v4": None,
        }
        viewed = {arm["id"]: arm for arm in json.loads(planner_view(built))["arms"]}
        assert viewed["v2"]["short_name"] == "Second try"
        assert "short_name" not in viewed["v4"], "the planner's view carries no key for an arm naming none"

    def test_an_arm_key_prints_the_declared_name_ahead_of_a_computed_one(self) -> None:
        shape = _named(s1_change(), NAMED)
        bundle = build_campaign_bundle(
            shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings
        )
        built = report_input(bundle)

        written = render("{arm:v2} against {arm:v4}", built, {"v2": "computed v2", "v4": "computed v4"})

        assert written.startswith("Second try (A)") and "computed v2" not in written
        assert written.endswith("computed v4 (C)")
