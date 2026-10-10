"""``campaign_create`` declares a design: the action and the operation take one, held to the gates every declaration meets.

Driven over the toy host through :meth:`MountedTool.call`, the path every transport takes. Pinned here:

- **A declared design is stored as declared, stamped by the surface**: its axes, what it held fixed and its
  questions, with ``declared_by`` the caller rather than anything the payload typed.
- **The control is addressed from a run**, and resolves to the same variant key ``set_campaign_control`` gives.
- **Every refusal names what was wrong**: an axis the host does not declare (with its vocabulary), a control run
  with no design or outside the campaign, a typed control beside a control run, and the old ``controls`` name.
- **An axis the host does not declare is refused naming ``declarable_axes()``**, the call listing what it does.
- **Without a design the campaign is exploratory** (#685): derived from ``declared_design is None``, never stored as a
  flag, and said once, at the top, by the bundle, the code-only report, the report of a generated analysis and the
  message the writer is sent; the writer's prompt calls the design it reads inferred from the runs, never declared.
"""

from __future__ import annotations

from typing import Any

import pytest

import json

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import (
    NO_DESIGN_EXPLORATORY,
    NO_QUESTION_EXPLORATORY,
    AnalysisContextBundle,
)
from threetears.evals.analysis.campaigns import set_campaign_control
from threetears.evals.analysis.gen_prompt import EVAL_ANALYSIS_GEN_DEFAULT
from threetears.evals.analysis.generator import build_user_message, generate_analysis
from threetears.evals.analysis.report import build_report
from threetears.evals.analysis.report.model import DisclosureBlock, Report
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.ops import CampaignLine
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_AXIS, toyhost_design
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, TOYHOST_SUBJECT, OpsFixture, ops_fixture
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, memo_payload


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


def _design(**changes: Any) -> dict[str, Any]:
    """The toy host's declaration as an author sends it: no authorship, no date — the surface stamps both."""
    design = toyhost_design().model_dump(mode="json", exclude={"declared_at", "declared_by"})
    design.update(changes)
    return design


async def _create(tool: MountedTool, fixture: OpsFixture, caller: Caller = CALLER, **arguments: Any) -> Any:
    call = {
        "action": "campaign_create",
        "name": "declared at birth",
        "subject_id": TOYHOST_SUBJECT.subject_id,
        "behavior": fixture.campaign.behavior,
        "run_ids": list(fixture.campaign.run_ids),
        **arguments,
    }
    return await tool.call(call, host=fixture.host, caller=caller)


async def test_a_campaign_created_with_a_design_carries_it_stamped_by_the_caller(evals: MountedTool) -> None:
    fixture = ops_fixture()

    outcome = await _create(evals, fixture, declared_design=_design())

    assert not outcome.is_error, outcome.text
    stored = fixture.host.eval_host.storage.load_campaign(
        CampaignLine.model_validate(outcome.structured).id, TOYHOST_SCOPE
    )
    assert stored is not None and stored.declared_design is not None
    declared = stored.declared_design
    assert [axis.axis_id for axis in declared.axes] == [TOYHOST_AXIS]
    assert declared.held_fixed == toyhost_design().held_fixed
    assert [q.id for q in declared.questions] == [q.id for q in toyhost_design().questions]
    assert declared.declared_by == CALLER.identity, "authorship is the surface's record, not the payload's"
    assert declared.control is None


async def test_the_control_is_resolved_from_a_run_as_set_campaign_control_resolves_it(evals: MountedTool) -> None:
    fixture = ops_fixture()
    storage = fixture.host.eval_host.storage
    control_run = fixture.campaign.run_ids[0]

    created = await _create(evals, fixture, declared_design=_design(), control_from_run_id=control_run)
    designated_later = await _create(evals, fixture, declared_design=_design())

    assert not created.is_error, created.text
    with_control = storage.load_campaign(created.structured["id"], TOYHOST_SCOPE)
    later = set_campaign_control(
        storage,
        designated_later.structured["id"],
        TOYHOST_SCOPE,
        control_run,
        set_by="agent:test",
        profile=fixture.host.eval_host.profile,
    )
    assert with_control is not None and with_control.declared_design is not None
    assert later.declared_design is not None
    assert with_control.declared_design.control == later.declared_design.control
    assert with_control.declared_design.control is not None


@pytest.mark.parametrize(
    ("arguments", "says"),
    [
        pytest.param(
            {
                "declared_design": _design(
                    axes=[{"axis_id": "no_such_lever", "values": [{"content": "x", "display": "x"}]}]
                )
            },
            "`declarable_axes()` lists every axis this host accepts",
            id="an-axis-the-host-does-not-declare",
        ),
        pytest.param({"control_from_run_id": "{run}"}, "declare a design", id="a-control-run-with-no-design"),
        pytest.param(
            {"declared_design": _design(), "control_from_run_id": "run-not-in-the-campaign"},
            "is not attached",
            id="a-control-run-outside-the-campaign",
        ),
        pytest.param(
            {"declared_design": _design(control="a" * 64), "control_from_run_id": "{run}"},
            "give one",
            id="a-typed-control-beside-a-control-run",
        ),
        pytest.param(
            {"declared_design": {**_design(), "controls": _design()["held_fixed"]}},
            "renamed `held_fixed`",
            id="the-old-name",
        ),
        pytest.param({"declared_design": _design(axes=[])}, "axes", id="no-axes"),
    ],
)
async def test_a_design_the_host_cannot_honour_is_refused_naming_why(
    evals: MountedTool, arguments: dict[str, Any], says: str
) -> None:
    fixture = ops_fixture()
    run = fixture.campaign.run_ids[0]
    resolved = {key: (value.format(run=run) if isinstance(value, str) else value) for key, value in arguments.items()}
    before = fixture.host.eval_host.storage.list_campaigns(TOYHOST_SCOPE)

    outcome = await _create(evals, fixture, **resolved)

    assert outcome.is_error and says in outcome.text, outcome.text
    assert fixture.host.eval_host.storage.list_campaigns(TOYHOST_SCOPE) == before, "a refused create writes nothing"


async def test_an_undeclared_axis_is_named_beside_the_pointer(evals: MountedTool) -> None:
    fixture = ops_fixture()
    lever = {"axis_id": "no_such_lever", "values": [{"content": "x", "display": "x"}]}

    outcome = await _create(evals, fixture, declared_design=_design(axes=[lever]))

    assert outcome.is_error and "no_such_lever" in outcome.text and "declarable_axes()" in outcome.text, outcome.text


# --- with no design the campaign is exploratory (#685) --------------------------------------------------------------


async def _exploratory(tool: MountedTool) -> tuple[OpsFixture, str, AnalysisContextBundle]:
    """A campaign created through the action with no design, and its bundle as an analysis would assemble it."""
    fixture = ops_fixture()
    outcome = await _create(tool, fixture)
    assert not outcome.is_error, outcome.text
    eval_host = fixture.host.eval_host
    stored = eval_host.storage.load_campaign(outcome.structured["id"], TOYHOST_SCOPE)
    assert stored is not None and stored.declared_design is None, "a create with no design declares none"
    bundle = assemble_context_bundle(stored, storage=eval_host.storage, profile=eval_host.profile)
    return fixture, stored.id, bundle


def _scope_said(report: Report) -> list[DisclosureBlock]:
    return [block for block in report.blocks if isinstance(block, DisclosureBlock) and block.source == "scope"]


async def test_the_bundle_calls_an_undeclared_campaign_exploratory_once(evals: MountedTool) -> None:
    _fixture, _campaign_id, bundle = await _exploratory(evals)

    assert bundle.declared_design is None
    assert bundle.reading_scope.disclosure == NO_DESIGN_EXPLORATORY
    assert bundle.reading_scope.exploratory_measures == [] and bundle.reading_scope.exploratory_dimensions == [], (
        "said once for the campaign, never on every reading"
    )
    assert "inferred from the runs" in NO_DESIGN_EXPLORATORY and "confirm nothing" in NO_DESIGN_EXPLORATORY


async def test_the_code_only_report_says_so_once_straight_after_its_opening_line(evals: MountedTool) -> None:
    fixture, campaign_id, _bundle = await _exploratory(evals)

    read = await evals.call(
        {"action": "report_read", "campaign_id": campaign_id, "format": "json"}, host=fixture.host, caller=CALLER
    )

    assert not read.is_error, read.text
    report = Report.model_validate_json(read.structured["body"])
    assert report.basis == "code_only"
    (said,) = _scope_said(report)
    assert said.text == NO_DESIGN_EXPLORATORY
    # At the top: the first block after the one-line opening saying no analysis was generated, which stays one line.
    assert report.blocks.index(said) == 1 and said.section == "questions"


async def test_the_writer_is_told_and_its_report_says_so_once_at_the_top(evals: MountedTool) -> None:
    fixture, _campaign_id, bundle = await _exploratory(evals)

    message = build_user_message(bundle)
    assert message.count(NO_DESIGN_EXPLORATORY) == 1, "the writer's view carries the line once"
    assert "the design inferred from the runs (`design`" in EVAL_ANALYSIS_GEN_DEFAULT
    assert "never called declared" in EVAL_ANALYSIS_GEN_DEFAULT

    memo = memo_payload(bundle)
    memo["questions"] = []  # the toy memo answers the toy campaign's question; this one declares none
    analysis, _insights = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=FixturedClient(json.dumps(memo)),
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=fixture.host.eval_host.profile,
    )
    report = build_report(analysis)

    assert analysis.design_snapshot is None
    (said,) = _scope_said(report)
    assert said.text == NO_DESIGN_EXPLORATORY and said.section == "summary"
    assert all(block.section == "summary" for block in report.blocks[: report.blocks.index(said) + 1])


async def test_a_declared_campaign_with_no_question_keeps_the_no_question_line(evals: MountedTool) -> None:
    fixture = ops_fixture()
    outcome = await _create(evals, fixture, declared_design=_design(questions=[]))
    assert not outcome.is_error, outcome.text
    eval_host = fixture.host.eval_host
    stored = eval_host.storage.load_campaign(outcome.structured["id"], TOYHOST_SCOPE)
    assert stored is not None

    bundle = assemble_context_bundle(stored, storage=eval_host.storage, profile=eval_host.profile)

    assert bundle.reading_scope.disclosure == NO_QUESTION_EXPLORATORY, "a declared design is not called undeclared"
