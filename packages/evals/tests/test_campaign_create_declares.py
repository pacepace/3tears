"""``campaign_create`` declares: the action and the operation take a question, aspects and arms, held to the gates.

Driven over the toy host through :meth:`MountedTool.call`, the path every transport takes. Pinned here:

- **A declared aspect is stored as declared, stamped by the surface**: its axes and what it held fixed, the question,
  and ``declared_by`` the caller rather than anything the payload typed.
- **An arm is addressed from a run**, and the control resolves to the same variant key ``set_campaign_control`` gives.
- **Every refusal names what was wrong**: an axis the host does not declare (with its vocabulary), an arm on an
  exploratory campaign or from a run no aspect holds, a declared aspect with no margin, a design with no axes.
- **An axis the host does not declare is refused naming what is declarable**, the refusal listing the host's axes.
- **Without a design the campaign is exploratory** (#685): derived from no aspect declaring one, never stored as a
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
from threetears.evals.analysis.campaign_bundle import build_campaign_user_message
from threetears.evals.analysis.generator import generate_analysis
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from threetears.evals.analysis.report import build_report
from threetears.evals.analysis.report.model import DisclosureBlock, Report
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.ops import CampaignLine
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_AXIS
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, TOYHOST_SUBJECT, OpsFixture, ops_fixture
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, memo_payload
from packages.evals.tests.off_loop import off_loop

#: The question the declared campaigns here ask.
_QUESTION = "Does the wider chunk extract the fields as well?"


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


def _design(fixture: OpsFixture, **changes: Any) -> dict[str, Any]:
    """The toy host's aspect design as an author sends it."""
    (aspect,) = fixture.campaign.aspects
    assert aspect.design is not None
    design = aspect.design.model_dump(mode="json", exclude={"merit_axes"})
    design.update(changes)
    return design


def _aspect(fixture: OpsFixture, *, design: dict[str, Any] | None = None, **changes: Any) -> dict[str, Any]:
    """The toy campaign's one aspect and its runs, declaring ``design`` (the toy host's by default)."""
    aspect: dict[str, Any] = {
        "label": fixture.campaign.behavior,
        "run_ids": list(fixture.campaign.run_ids),
        "design": _design(fixture) if design is None else design,
    }
    aspect.update(changes)
    return aspect


async def _create(tool: MountedTool, fixture: OpsFixture, caller: Caller = CALLER, **arguments: Any) -> Any:
    call = {
        "action": "campaign_create",
        "name": "declared at birth",
        "subject_id": TOYHOST_SUBJECT.subject_id,
        "behavior": fixture.campaign.behavior,
        **arguments,
    }
    return await tool.call(call, host=fixture.host, caller=caller)


async def test_a_campaign_created_with_a_design_carries_it_stamped_by_the_caller(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)

    outcome = await _create(evals, fixture, question=_QUESTION, aspects=[_aspect(fixture)])

    assert not outcome.is_error, outcome.text
    stored = off_loop(
        fixture.host.eval_host.storage.load_campaign, CampaignLine.model_validate(outcome.structured).id, TOYHOST_SCOPE
    )
    assert stored is not None and not stored.exploratory
    (aspect,) = stored.aspects
    assert aspect.design is not None and aspect.margin is None, "a campaign naming no control declares no margin"
    assert [axis.axis_id for axis in aspect.design.axes] == [TOYHOST_AXIS]
    assert aspect.design.held_fixed == fixture.campaign.aspects[0].design.held_fixed  # type: ignore[union-attr]
    assert aspect.design.merit_axes == [], "a question naming no axes asks about every one"
    assert stored.question == _QUESTION
    assert stored.declared_by == CALLER.identity, "authorship is the surface's record, not the payload's"
    assert stored.declared_at is not None and aspect.declared_at == stored.declared_at
    assert stored.control is None


async def test_the_control_is_resolved_from_a_run_as_set_campaign_control_resolves_it(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    storage = fixture.host.eval_host.storage
    control_run = fixture.campaign.run_ids[0]

    created = await _create(
        evals,
        fixture,
        aspects=[_aspect(fixture)],
        campaign_arms=[{"label": "narrow", "role": "control", "from_run_ids": [control_run]}],
    )
    designated_later = await _create(evals, fixture, aspects=[_aspect(fixture)])

    assert not created.is_error, created.text
    with_control = off_loop(storage.load_campaign, created.structured["id"], TOYHOST_SCOPE)
    later = off_loop(
        set_campaign_control,
        storage,
        designated_later.structured["id"],
        TOYHOST_SCOPE,
        control_run,
        set_by="agent:test",
        profile=fixture.host.eval_host.profile,
    )
    assert with_control is not None and with_control.control is not None and later.control is not None
    assert with_control.control.variant_key == later.control.variant_key
    assert (with_control.control.label, later.control.label) == ("narrow", "control")


@pytest.mark.parametrize(
    ("arguments", "says"),
    [
        pytest.param(
            {"design": {"axes": [{"axis_id": "no_such_lever", "values": [{"content": "x", "display": "x"}]}]}},
            "remedy: declare only the axes this refusal lists as declarable",
            id="an-axis-the-host-does-not-declare",
        ),
        pytest.param(
            {
                "exploratory": True,
                "campaign_arms": [{"label": "control", "role": "control", "from_run_ids": ["{run}"]}],
            },
            "declares no question, arm",
            id="an-arm-on-an-exploratory-campaign",
        ),
        pytest.param(
            {"campaign_arms": [{"label": "control", "role": "control", "from_run_ids": ["run-not-in-the-campaign"]}]},
            "which no aspect of the campaign holds",
            id="an-arm-from-a-run-outside-the-campaign",
        ),
        pytest.param(
            {
                "margin": None,
                "question": _QUESTION,
                "campaign_arms": [{"label": "control", "role": "control", "from_run_ids": ["{run}"]}],
            },
            "declare a design and no margin",
            id="a-declared-aspect-with-no-margin-under-a-question-of-a-control",
        ),
        pytest.param(
            {"margin": 5},
            "declare an absolute bar instead",
            id="a-margin-on-a-campaign-naming-no-control",
        ),
        pytest.param({"design": {"axes": []}}, "axes", id="no-axes"),
    ],
)
async def test_a_design_the_host_cannot_honour_is_refused_naming_why(
    evals: MountedTool, arguments: dict[str, Any], says: str
) -> None:
    fixture = off_loop(ops_fixture)
    run = fixture.campaign.run_ids[0]
    campaign_level = {key: arguments[key] for key in ("exploratory", "campaign_arms", "question") if key in arguments}
    for arm in campaign_level.get("campaign_arms", []):
        arm["from_run_ids"] = [run_id.format(run=run) for run_id in arm["from_run_ids"]]
    changes = {key: value for key, value in arguments.items() if key not in campaign_level}
    if "design" in changes:
        changes["design"] = _design(fixture, **changes["design"])
    aspect = (
        {"label": fixture.campaign.behavior, "run_ids": list(fixture.campaign.run_ids)}
        if campaign_level.get("exploratory")
        else _aspect(fixture, **changes)
    )
    before = off_loop(fixture.host.eval_host.storage.list_campaigns, TOYHOST_SCOPE)

    outcome = await _create(evals, fixture, aspects=[aspect], **campaign_level)

    assert outcome.is_error and says in outcome.text, outcome.text
    assert off_loop(fixture.host.eval_host.storage.list_campaigns, TOYHOST_SCOPE) == before, (
        "a refused create writes nothing"
    )


async def test_an_undeclared_axis_is_named_beside_the_pointer(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    lever = {"axis_id": "no_such_lever", "values": [{"content": "x", "display": "x"}]}

    outcome = await _create(evals, fixture, aspects=[_aspect(fixture, design=_design(fixture, axes=[lever]))])

    assert outcome.is_error and "no_such_lever" in outcome.text and "Declarable axes:" in outcome.text, outcome.text
    assert "remedy: declare only the axes this refusal lists as declarable" in outcome.text, outcome.text


# --- with no design the campaign is exploratory (#685) --------------------------------------------------------------


async def _exploratory(tool: MountedTool) -> tuple[OpsFixture, str, AnalysisContextBundle]:
    """A campaign created through the action with no design, and its bundle as an analysis would assemble it."""
    fixture = off_loop(ops_fixture)
    aspect = {"label": fixture.campaign.behavior, "run_ids": list(fixture.campaign.run_ids)}
    outcome = await _create(tool, fixture, aspects=[aspect], exploratory=True)
    assert not outcome.is_error, outcome.text
    eval_host = fixture.host.eval_host
    stored = off_loop(eval_host.storage.load_campaign, outcome.structured["id"], TOYHOST_SCOPE)
    assert stored is not None and stored.exploratory, "a create with no design declares none"
    bundle = off_loop(assemble_context_bundle, stored, storage=eval_host.storage, profile=eval_host.profile)
    return fixture, stored.id, bundle


#: How line 2 of a report opens when no question was declared before the run.
_EXPLORATORY_BASIS = "Exploratory: "


def _exploratory_said(report: Report) -> list[DisclosureBlock]:
    return [block for block in report.blocks if isinstance(block, DisclosureBlock) and _EXPLORATORY_BASIS in block.text]


async def test_the_bundle_calls_an_undeclared_campaign_exploratory_once(evals: MountedTool) -> None:
    _fixture, _campaign_id, bundle = await _exploratory(evals)

    assert bundle.declared is None
    assert bundle.reading_scope.disclosure == NO_DESIGN_EXPLORATORY
    assert bundle.reading_scope.exploratory_measures == [] and bundle.reading_scope.exploratory_dimensions == [], (
        "said once for the campaign, never on every reading"
    )
    assert "inferred from the runs" in NO_DESIGN_EXPLORATORY and "confirm nothing" in NO_DESIGN_EXPLORATORY


async def test_the_code_only_report_says_so_once_on_its_first_screen(evals: MountedTool) -> None:
    fixture, campaign_id, _bundle = await _exploratory(evals)

    read = await evals.call(
        {"action": "report_read", "campaign_id": campaign_id, "format": "markdown"}, host=fixture.host, caller=CALLER
    )

    assert not read.is_error, read.text
    page = read.structured["body"]
    # On the first screen, under the title and above the aspect's fold, and nowhere else on the page: the fold leaves
    # out the aspect page's own exploratory verdict line, which the campaign's first screen carries once.
    title = next(index for index, line in enumerate(page.splitlines()) if line.startswith("# "))
    first_screen = page.split("\n## ", 1)[0]
    lead = "No question was declared, so every reading is a lead."
    assert page.count(lead) == 1 and first_screen.index(lead) > first_screen.index(page.splitlines()[title])
    assert _EXPLORATORY_BASIS not in page


async def test_the_writer_is_told_and_its_report_says_so_once_at_the_top(evals: MountedTool) -> None:
    fixture, _campaign_id, bundle = await _exploratory(evals)

    message = _writer_message(bundle)
    assert message.count(NO_DESIGN_EXPLORATORY) == 1, "the writer's view carries the line once"
    assert "the design inferred from the runs (`design`" in EVAL_ANALYSIS_GEN_DEFAULT
    assert "never called declared" in EVAL_ANALYSIS_GEN_DEFAULT

    memo = memo_payload(bundle)
    analysis, _insights = await generate_analysis(
        campaign_bundle(bundle),
        prompt=PROMPT,
        model=MODEL,
        client=FixturedClient(json.dumps(memo)),
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=fixture.host.eval_host.profile,
    )
    report = build_report(analysis)

    assert [aspect.design for aspect in analysis.design_snapshot.aspects] == [None], "it declared nothing"
    (said,) = _exploratory_said(report)
    assert said.text.startswith(_EXPLORATORY_BASIS) and said.source == "verdict"
    assert report.blocks.index(said) == 1 and said.section == "header"


async def test_a_declared_campaign_with_no_question_keeps_the_no_question_line(evals: MountedTool) -> None:
    fixture = off_loop(ops_fixture)
    outcome = await _create(evals, fixture, aspects=[_aspect(fixture)])
    assert not outcome.is_error, outcome.text
    eval_host = fixture.host.eval_host
    stored = off_loop(eval_host.storage.load_campaign, outcome.structured["id"], TOYHOST_SCOPE)
    assert stored is not None

    bundle = off_loop(assemble_context_bundle, stored, storage=eval_host.storage, profile=eval_host.profile)

    assert bundle.reading_scope.disclosure == NO_QUESTION_EXPLORATORY, "a declared design is not called undeclared"


def _writer_message(bundle: Any) -> str:
    """What the writer is sent over one aspect's bundle: the campaign bundle of that one aspect, rendered."""
    return build_campaign_user_message(campaign_bundle(bundle))
