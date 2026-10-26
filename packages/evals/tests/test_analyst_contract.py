"""The analyst's tool contract: one frozen snapshot, every figure from its key, nothing that branches on eval kind.

Contracts held here (``threetears.evals.kernel.report_snapshot``, ``analysis.analyst``, ``ops.analyst``,
``actions.analyst``):

- **the snapshot**: a freeze from a campaign, from a stored analysis, and a bare report input each store a snapshot
  whose report input reads back as the one built; its id is the campaign's and the input digest's, so freezing
  unchanged evidence names the stored snapshot (``reused``) and a document whose id disagrees is refused; a snapshot
  whose stored input moved, or no longer reads, is refused with the remedy to freeze again; a freeze naming both or
  neither source, an unknown campaign, and a campaign with no evidence are refused;
- **every analyst action** — the ones present now and every one a later chunk registers, on every fixture snapshot
  (r1 to r5, the misfit inputs, Kairo round 2) — answers the same twice, keeps its text within the budget, and carries
  every number inside a cited figure whose key resolves to it;
- **the gap note** stores nothing, logs one record, and names the same gap for the same snapshot, need and why;
- **the cut**: the analyst's actions are in the catalogue but not on ``evals`` or the read-only tool, whose parameters
  are the engine's alone; the analyst's tool carries them, and the toolsets are read off the workflows;
- **no branch on eval kind**: no analyst module names ``classify``, ``spine``, ``report_kind`` or ``Signals``, or
  reads ``meta.kind`` or ``meta.design_shape``.
"""

from __future__ import annotations

import ast
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from packages.evals.tests.analyst_support import (
    KAIRO,
    TEXT_BUDGET,
    action_calls,
    fixture_inputs,
    fixture_snapshots,
    read_actions,
)
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture, settled
from threetears.evals.actions import (
    ActionCatalogue,
    ActionOutcome,
    MountedTool,
    analyst_actions,
    analyst_tools,
    engine_actions,
    eval_catalogue,
    read_only_tools,
    standard_tools,
)
from threetears.evals.actions.render import render_cited
from threetears.evals.analysis import assemble_campaign_bundle
from threetears.evals.analysis.analyst.cited import CitedFigure, cite, cite_if
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.analysis.analyst.snapshot import ANALYST_EXAMPLES, frozen_input, report_snapshot
from threetears.evals.analysis.planned_analysis import report_input_digest, report_input_document_digest
from threetears.evals.analysis.report.build import analysis_campaign_bundle
from threetears.evals.analysis.report.figure_keys import figure_key_grammar, render, resolve
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel import EvalReportSnapshot, report_snapshot_id
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.ops import analysis_generate
from threetears.evals.storage import InMemoryDocumentStore

SRC = Path(__file__).resolve().parents[1] / "src" / "threetears" / "evals"

#: Every module an analyst tool is built from: its computation, its operations and its actions.
ANALYST_MODULES = sorted(
    [
        *(SRC / "analysis" / "analyst").glob("*.py"),
        *(SRC / "ops").glob("analyst*.py"),
        *(SRC / "actions").glob("analyst*.py"),
    ]
)

#: The fields of a listing that page it, which are counts of rows and never figures, and the next page's call, whose
#: arguments are the caller's own.
PAGING = frozenset({"total", "offset", "next_offset", "limit", "next"})

#: The fields every cited figure carries, by which one is found in a result's JSON.
FIGURE_FIELDS = frozenset({"key", "reads", "value", "kind", "scale", "unit"})


def _next_line(call: dict[str, Any]) -> str:
    """The next page's call as a page spells it on its last line: each argument in order, a string quoted, any other
    value as compact JSON with its keys sorted."""
    spelled = (
        f"{name}='{value}'"
        if isinstance(value, str)
        else f"{name}={json.dumps(value, sort_keys=True, ensure_ascii=False)}"
        for name, value in call.items()
    )
    return "next: " + ", ".join(spelled)


def _analyst_tool() -> MountedTool:
    (spec,) = analyst_tools()
    return eval_catalogue().mount(spec)


def _evals_tool() -> MountedTool:
    return eval_catalogue().mount(standard_tools()[0])


async def _call(tool: MountedTool, fixture: OpsFixture, arguments: dict[str, Any]) -> ActionOutcome:
    return await tool.call(arguments, host=fixture.host, caller=CALLER)


def _stored(fixture: OpsFixture, snapshot_id: str) -> EvalReportSnapshot:
    snapshot = off_loop(fixture.host.eval_host.storage.load_report_snapshot, snapshot_id, TOYHOST_SCOPE)
    assert snapshot is not None, snapshot_id
    return snapshot


def _seeded(store: InMemoryDocumentStore | None = None) -> tuple[OpsFixture, dict[str, EvalReportSnapshot]]:
    """The toy host, with every fixture snapshot stored in its scope."""
    fixture = ops_fixture(store=store)
    snapshots = fixture_snapshots(TOYHOST_SCOPE)
    for snapshot in snapshots.values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return fixture, snapshots


async def _generated(fixture: OpsFixture) -> EvalAnalysis:
    """The one analysis generated of the toy campaign."""
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    assert (await settled(fixture.host, job.job_id)).state == "completed"
    (analysis,) = off_loop(fixture.host.eval_host.storage.list_analyses_by_campaign, fixture.campaign.id, TOYHOST_SCOPE)
    return analysis


# =============================================================================
# The snapshot
# =============================================================================


async def test_a_campaign_freezes_to_the_report_input_its_evidence_builds_and_again_to_the_same_snapshot() -> None:
    fixture = off_loop(ops_fixture)
    tool = _analyst_tool()

    first = await _call(tool, fixture, {"action": "report_snapshot_freeze", "campaign_id": fixture.campaign.id})
    assert first.structured is not None, first.text
    assert first.structured["source"] == "campaign" and first.structured["reused"] is False
    snapshot = _stored(fixture, first.structured["snapshot_id"])

    bundle = off_loop(assemble_campaign_bundle, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE)
    built = report_input(bundle, examples=ANALYST_EXAMPLES)
    assert frozen_input(snapshot) == built
    assert snapshot.id == report_snapshot_id(fixture.campaign.id, report_input_digest(built))
    assert (snapshot.campaign_id, snapshot.analysis_id, snapshot.created_by) == (
        fixture.campaign.id,
        None,
        "agent:test",
    )
    assert snapshot.bundle_fingerprint == bundle.fingerprint() and snapshot.bundle_assembled_at is not None
    shown = [example.trial for example in built.examples if example.trial is not None]
    assert sorted(snapshot.example_result_ids) == sorted(shown)

    again = await _call(tool, fixture, {"action": "report_snapshot_freeze", "campaign_id": fixture.campaign.id})
    once_more = await _call(tool, fixture, {"action": "report_snapshot_freeze", "campaign_id": fixture.campaign.id})
    assert again.structured is not None and again.structured["reused"] is True
    assert {**again.structured, "reused": False} == first.structured
    assert (again.text, again.structured) == (once_more.text, once_more.structured)
    assert len(first.text) <= TEXT_BUDGET
    _held_to_figures(again.structured, frozen_input(snapshot))


async def test_a_stored_analysis_freezes_to_the_report_input_its_frozen_bundle_builds() -> None:
    fixture = off_loop(ops_fixture)
    analysis = await _generated(fixture)

    outcome = await _call(_analyst_tool(), fixture, {"action": "report_snapshot_freeze", "analysis_id": analysis.id})
    assert outcome.structured is not None, outcome.text
    assert (outcome.structured["source"], outcome.structured["analysis_id"]) == ("analysis", analysis.id)
    snapshot = _stored(fixture, outcome.structured["snapshot_id"])

    bundle = analysis_campaign_bundle(analysis)
    built = report_input(bundle, examples=ANALYST_EXAMPLES)
    assert frozen_input(snapshot) == built
    assert snapshot.id == report_snapshot_id(analysis.campaign_id, report_input_digest(built))
    assert snapshot.bundle_assembled_at == analysis.generation.bundle_assembled_at
    assert snapshot.bundle_fingerprint == bundle.fingerprint()


async def test_a_bare_report_input_is_a_snapshot_every_analyst_tool_reads() -> None:
    fixture = off_loop(ops_fixture)
    held = fixture_inputs()["r1"]
    snapshot = report_snapshot(held, scope_id=TOYHOST_SCOPE, campaign_id=None, source="report_input", created_by="me")
    off_loop(fixture.host.eval_host.storage.save_report_snapshot, snapshot)

    assert snapshot.id == report_snapshot_id(None, report_input_digest(held))
    assert (snapshot.bundle, snapshot.bundle_fingerprint, snapshot.example_result_ids) == (None, None, {})
    assert frozen_input(_stored(fixture, snapshot.id)) == held
    outcome = await _call(
        _analyst_tool(),
        fixture,
        {"action": "capability_gap", "snapshot_id": snapshot.id, "need": "n", "why": "w", "workaround": "x"},
    )
    assert outcome.structured is not None, outcome.text


def test_a_snapshot_id_is_its_campaign_and_digest_and_a_document_disagreeing_is_refused() -> None:
    held = fixture_inputs()["r2"]
    one = report_snapshot(held, scope_id="s", campaign_id=None, source="report_input", created_by="a")
    two = report_snapshot(held, scope_id="s", campaign_id=None, source="report_input", created_by="b")
    assert one.id == two.id == report_snapshot_id(None, report_input_digest(held))
    assert report_snapshot_id("campaign-1", one.report_input_digest) != one.id
    assert report_snapshot_id("campaign-1", one.report_input_digest) == report_snapshot_id(
        "campaign-1", one.report_input_digest
    )
    other = fixture_inputs()["r3"]
    assert report_snapshot_id(None, report_input_digest(other)) != one.id

    document = one.to_dict()
    with pytest.raises(ValueError, match="a report snapshot's id is"):
        EvalReportSnapshot.from_dict({**document, "id": report_snapshot_id("campaign-1", one.report_input_digest)})
    with pytest.raises(ValueError, match="names no campaign"):
        EvalReportSnapshot.from_dict(
            {**document, "campaign_id": "c", "id": report_snapshot_id("c", one.report_input_digest)}
        )
    with pytest.raises(ValueError, match="names an analysis exactly when"):
        EvalReportSnapshot.from_dict({**document, "analysis_id": "analysis-1"})


def _tampered(store: InMemoryDocumentStore, snapshot: EvalReportSnapshot, *, redigest: bool, change: str) -> None:
    """Rewrite a stored snapshot's report input under it: a figure moved, or a key the model does not declare."""
    document = store.get(snapshot.id, snapshot.scope_id)
    assert document is not None
    held = json.loads(json.dumps(document["report_input"]))
    if change == "figure":
        held["meta"]["campaign"] = held["meta"]["campaign"] + " (edited)"
    else:
        held["meta"]["unheard_of"] = "a field this build does not read"
    document["report_input"] = held
    if redigest:
        document["report_input_digest"] = report_input_document_digest(held)
        document["id"] = snapshot.id  # the id stays the one the caller holds
    store.upsert(document)


@pytest.mark.parametrize(
    ("change", "refusal"),
    [("figure", "does not hold the input it names"), ("unread", "does not read")],
    ids=["its input moved", "its input no longer reads"],
)
async def test_a_snapshot_whose_input_moved_or_no_longer_reads_is_refused_with_the_remedy_to_freeze_again(
    change: str, refusal: str
) -> None:
    store = InMemoryDocumentStore()
    fixture = off_loop(ops_fixture, store=store)
    snapshot = report_snapshot(
        fixture_inputs()["r4"], scope_id=TOYHOST_SCOPE, campaign_id=None, source="report_input", created_by="me"
    )
    off_loop(fixture.host.eval_host.storage.save_report_snapshot, snapshot)
    off_loop(_tampered, store, snapshot, redigest=change == "unread", change=change)

    outcome = await _call(
        _analyst_tool(),
        fixture,
        {"action": "capability_gap", "snapshot_id": snapshot.id, "need": "n", "why": "w", "workaround": "x"},
    )
    assert outcome.ending == "refused" and outcome.refusal is not None
    assert outcome.refusal.kind == "validation_failed"
    assert (
        refusal in outcome.text
        and "remedy: freeze the campaign or the analysis again with report_snapshot_freeze" in (outcome.text)
    )


async def test_an_unknown_snapshot_is_refused_naming_how_to_get_one() -> None:
    fixture = off_loop(ops_fixture)
    outcome = await _call(
        _analyst_tool(),
        fixture,
        {"action": "capability_gap", "snapshot_id": "report-snapshot:nope", "need": "n", "why": "w", "workaround": "x"},
    )
    assert outcome.refusal is not None and outcome.refusal.kind == "not_found"
    assert "report_snapshot_freeze returned" in outcome.text


@pytest.mark.parametrize(
    "arguments",
    [{}, {"campaign_id": "c", "analysis_id": "a"}],
    ids=["neither", "both"],
)
async def test_a_freeze_naming_neither_or_both_sources_is_refused_as_the_call(arguments: dict[str, Any]) -> None:
    fixture = off_loop(ops_fixture)
    outcome = await _call(_analyst_tool(), fixture, {"action": "report_snapshot_freeze", **arguments})
    assert outcome.refusal is not None and outcome.refusal.kind == "validation_failed"
    assert "name a campaign_id or an analysis_id, exactly one" in outcome.text


async def test_a_freeze_of_an_unknown_campaign_or_one_with_no_evidence_is_refused_and_stores_nothing() -> None:
    store = InMemoryDocumentStore()
    fixture = off_loop(ops_fixture, store=store)
    unknown = await _call(_analyst_tool(), fixture, {"action": "report_snapshot_freeze", "campaign_id": "no-such"})
    assert unknown.refusal is not None and unknown.refusal.kind == "not_found"

    evals = _evals_tool()
    for run_id in fixture.campaign.run_ids:
        archived = await _call(evals, fixture, {"action": "run_archive", "run_id": run_id, "archived": True})
        assert archived.structured is not None, archived.text
    empty = await _call(
        _analyst_tool(), fixture, {"action": "report_snapshot_freeze", "campaign_id": fixture.campaign.id}
    )
    assert empty.refusal is not None and empty.refusal.kind == "validation_failed"
    assert "every one is archived" in empty.text
    assert off_loop(lambda: list(store.iter_by_doc_type("eval_report_snapshot", TOYHOST_SCOPE))) == []


# =============================================================================
# Every analyst action, on every fixture snapshot
# =============================================================================


def _numbers(node: Any, path: str = "") -> Iterator[tuple[str, Any, dict[str, Any] | None]]:
    """Every number in a result's JSON, with where it sits and the cited figure holding it, if one does."""
    if isinstance(node, dict):
        if FIGURE_FIELDS <= node.keys():
            value = node["value"]
            ends = value if isinstance(value, list) else [value]
            for end in ends:
                if isinstance(end, int | float) and not isinstance(end, bool):
                    yield f"{path}.value", end, node
            return
        for key, item in node.items():
            if key in PAGING:
                continue
            yield from _numbers(item, f"{path}.{key}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _numbers(item, f"{path}[{index}]")
    elif isinstance(node, int | float) and not isinstance(node, bool):
        yield path, node, None


def _held_to_figures(structured: dict[str, Any], held: ReportInput) -> None:
    """Every number in a result sits in a cited figure, and every cited figure's key resolves to its value."""
    for where, _number, figure in _numbers(structured):
        assert figure is not None, f"{where} is a number no figure key names"
    figures = [node for node in _figure_nodes(structured)]
    for figure in figures:
        resolved = resolve(figure["key"], held)
        expected = list(resolved.value) if isinstance(resolved.value, tuple) else resolved.value
        assert figure["value"] == expected, f"{figure['key']} holds {figure['value']}, and the key reads {expected}"


def _figure_nodes(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if FIGURE_FIELDS <= node.keys():
            yield node
            return
        for item in node.values():
            yield from _figure_nodes(item)
    elif isinstance(node, list):
        for item in node:
            yield from _figure_nodes(item)


def test_the_number_walk_finds_a_bare_number_and_a_figure_that_does_not_resolve() -> None:
    """The control for the walk below: a number outside a figure, and a figure its key does not read, both fail."""
    held = fixture_inputs()["r5"]
    with pytest.raises(AssertionError, match="no figure key names"):
        _held_to_figures({"rows": [{"n_cases": 12}]}, held)
    figure = {"key": "n_candidates", "reads": "x", "value": 999.0, "kind": "number", "scale": "count", "unit": ""}
    with pytest.raises(AssertionError, match="holds 999.0"):
        _held_to_figures({"figure": figure}, held)
    _held_to_figures({"total": 3, "offset": 0, "next_offset": None}, held)


async def test_every_analyst_read_action_answers_alike_twice_within_its_budget_on_every_figure_by_its_key() -> None:
    fixture, snapshots = off_loop(_seeded)
    tool = _analyst_tool()
    actions = read_actions()
    assert {action.name for action in actions} >= {"capability_gap"}
    for name, snapshot in snapshots.items():
        held = frozen_input(snapshot)
        for action in actions:
            for arguments in action_calls(action, held, snapshot.id):
                first = await _call(tool, fixture, arguments)
                second = await _call(tool, fixture, arguments)
                where = f"{action.name} on {name}"
                assert first.ending == "ok", f"{where}: {first.text}"
                assert (first.text, first.structured) == (second.text, second.structured), where
                assert len(first.text) <= TEXT_BUDGET, f"{where}: {len(first.text)} characters"
                assert first.structured is not None
                _held_to_figures(first.structured, held)


def test_the_fixture_snapshots_hold_the_kairo_bundle_and_its_examples_results() -> None:
    kairo = fixture_snapshots("s")[KAIRO]
    assert kairo.source == "campaign" and kairo.bundle is not None
    held = frozen_input(kairo)
    assert len(held.examples) > 12, "the analyst's pool is wider than the planner's"
    assert set(kairo.example_result_ids) == {example.trial for example in held.examples if example.trial is not None}


class _Row(BaseModel):
    """A row as a fact tool returns one: plain fields and cited figures."""

    arm: str
    pass_rate: CitedFigure
    cost: CitedFigure | None


class _Listing(BaseModel):
    """A paged listing of rows, as a fact tool returns one."""

    rows: list[_Row]
    total: int
    offset: int
    next_offset: int | None


def test_a_result_reads_back_as_each_figures_key_and_reading_under_its_row() -> None:
    held = fixture_inputs()["r5"]
    rows = [
        _Row(
            arm=f"arm:{arm.id}",
            pass_rate=cite(f"pass_rate.{arm.id}", held),
            cost=cite_if(f"cost_per_1k.{arm.id}", held),
        )
        for arm in held.arms
    ]
    listing = _Listing(rows=rows, total=len(rows), offset=0, next_offset=None)
    text = render_cited(listing)

    first = held.arms[0].id
    reads = render(f"{{pass_rate.{first}}}", held, {})
    assert f"  - arm: arm:{first}\n    pass_rate.{first} = {reads}\n" in text
    assert text.splitlines()[0] == "_listing" and "total: 3" in text
    assert "next_offset" not in text, "a page's next call is the tool's to write"
    _held_to_figures(listing.model_dump(mode="json"), held)


def test_a_cited_figure_says_whether_code_derived_it_as_its_key_resolves() -> None:
    held = fixture_inputs()["r5"]
    aspect = next(aspect for aspect in held.aspects if aspect.readings)
    reading = aspect.readings[0]
    derived = cite(f"aspects.{aspect.id}.flip_margin.{reading.arm}", held)
    stated = cite(f"pass_rate.{held.arms[0].id}", held)
    assert derived.derived is True
    assert derived.derived == resolve(derived.key, held).derived
    assert stated.derived is False
    assert stated.derived == resolve(stated.key, held).derived


def test_a_figure_the_input_leaves_unstated_is_absent_and_a_key_naming_nothing_is_raised() -> None:
    held = fixture_inputs()["r1"]
    assert held.meta.question is None, "the control: r1 states no question"
    assert cite_if("meta.question", held) is None
    with pytest.raises(UnstatedFigure, match="does not state"):
        cite("meta.question", held)
    with pytest.raises(UnresolvableReference, match="names no arm") as refused:
        cite_if("arm:no-such-arm-here", held)
    assert not isinstance(refused.value, UnstatedFigure), "a key naming nothing is wrong, not an unstated figure"
    campaign = cite_if("meta.campaign", held)
    assert campaign is not None and campaign.value == held.meta.campaign and campaign.kind == "text"


# =============================================================================
# The gap note
# =============================================================================


async def test_a_gap_is_noted_alike_for_the_same_need_logged_once_and_stored_nowhere(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryDocumentStore()
    fixture, snapshots = off_loop(_seeded, store)
    snapshot_id = snapshots["r3"].id
    stored = off_loop(lambda: sorted(store.iter_by_doc_type("eval_report_snapshot", TOYHOST_SCOPE)))
    tool = _analyst_tool()
    noted = {"action": "capability_gap", "snapshot_id": snapshot_id, "need": "a settle tool", "why": "to size it"}

    with caplog.at_level(logging.INFO, logger="threetears.evals.analyst"):
        one = await _call(tool, fixture, {**noted, "workaround": "read the intervals", "claim_ids": ["c2"]})
    two = await _call(tool, fixture, {**noted, "workaround": "said so in words"})
    other = await _call(tool, fixture, {**noted, "need": "a confound tool", "workaround": "x"})

    assert one.structured is not None and two.structured is not None and other.structured is not None
    assert one.structured["gap_id"] == two.structured["gap_id"] != other.structured["gap_id"]
    assert one.structured["recorded_in"] == "transcript"
    assert "read the intervals" in one.text and "c2" in one.text
    records = [record for record in caplog.records if record.name == "threetears.evals.analyst"]
    assert len(records) == 1 and records[0].levelno == logging.INFO
    assert one.structured["gap_id"] in records[0].getMessage()
    assert off_loop(lambda: sorted(store.iter_by_doc_type("eval_report_snapshot", TOYHOST_SCOPE))) == stored


# =============================================================================
# The cut
# =============================================================================


def test_the_evals_and_read_only_tools_carry_the_engines_actions_and_parameters_alone() -> None:
    engine = ActionCatalogue(engine_actions())
    for spec in (*standard_tools(), *read_only_tools()):
        mounted = eval_catalogue().mount(spec)
        assert mounted.input_schema() == engine.mount(spec).input_schema(), spec.name
        assert not {action.name for action in analyst_actions()} & {action.name for action in mounted.actions}


def test_the_analyst_tool_carries_every_analyst_action_and_read_only_it_carries_no_freeze() -> None:
    names = [action.name for action in analyst_actions()]
    assert [action.name for action in _analyst_tool().actions] == names
    assert _analyst_tool().hints.idempotent, "the freeze, its one write, names the snapshot it stored on a repeat"
    (reading,) = analyst_tools(writes=False)
    read_only = eval_catalogue().mount(reading)
    assert read_only.hints.read_only and "report_snapshot_freeze" not in {a.name for a in read_only.actions}
    assert read_only.hints.idempotent
    assert {action.permission for action in analyst_actions()} <= {"read", "write"}
    assert [action.name for action in analyst_actions() if action.permission == "write"] == ["report_snapshot_freeze"]


async def test_an_analyst_action_called_on_evals_is_refused_naming_the_analyst_tools() -> None:
    fixture = off_loop(ops_fixture)
    outcome = await _call(_evals_tool(), fixture, {"action": "capability_gap", "snapshot_id": "s"})
    assert outcome.refusal is not None
    assert "capability_gap is not an action of tool evals: it is a read action of the analyst tools" in outcome.text


def test_the_editors_cut_is_the_analyst_tool_read_off_the_workflows_with_no_blocks_and_no_freeze() -> None:
    (analyst_spec,) = analyst_tools()
    (editor_spec,) = analyst_tools(role="editor")
    analyst, editor = eval_catalogue().mount(analyst_spec), eval_catalogue().mount(editor_spec)
    assert editor.name == analyst.name == "evals_analyst" and editor.spec.description != analyst.spec.description
    by_name = {action.name: action for action in analyst_actions()}
    blocks = {name for name, action in by_name.items() if action.workflow == "Investigate: blocks"}
    assert blocks, "the control: the analyst's cut holds block actions for the editor's to leave out"
    enum = editor.input_schema()["properties"]["action"]["enum"]
    assert enum == ["help", *(a.name for a in analyst_actions() if a.name not in blocks | {"report_snapshot_freeze"})]
    assert not blocks & set(enum) and "capability_gap" in enum
    assert {by_name[name].workflow for name in enum[1:]} == {
        "Investigate: facts",
        "Investigate: analyses",
        "Investigate: gaps",
    }
    assert editor.hints.read_only and editor.hints.idempotent
    # One meaning per parameter: every parameter the editor's cut carries reads exactly as the analyst's tool says it.
    editor_params, analyst_params = editor.input_schema()["properties"], analyst.input_schema()["properties"]
    assert all(editor_params[name] == analyst_params[name] for name in editor_params if name != "action")
    assert set(editor_params) < set(analyst_params), "the block parameters are the analyst's alone"


#: Every analyst listing, the fixture snapshot it is read on, and a call whose rows run to more than one small page.
_PAGED_CALLS: dict[str, tuple[str, dict[str, Any]]] = {
    "readings_list": (KAIRO, {"limit": 2}),
    "contrasts_list": (KAIRO, {"limit": 3}),
    "measure_read": (KAIRO, {"measure_key": "persona.holds_character", "by": "stratum", "limit": 3}),
    "examples_list": (KAIRO, {"limit": 2}),
    "decision_sensitivity": ("r2", {"parameter": "threshold:pass_rate", "limit": 1}),
    "measure_breakdown": (KAIRO, {"measure_key": "persona.holds_character", "limit": 2}),
    "price_tradeoff": (KAIRO, {"gain_key": "persona.holds_character", "limit": 1}),
    "block_render": (KAIRO, {"block_kind": "table", "query": {"source": "contrasts"}, "limit": 2}),
}


def test_every_analyst_listing_is_paged() -> None:
    paged = {action.name for action in analyst_actions() if action.paged}
    assert paged == set(_PAGED_CALLS)
    assert paged == {action.name for action in analyst_actions() if "next_offset" in action.result.model_fields}


@pytest.mark.parametrize("action", sorted(_PAGED_CALLS))
async def test_following_next_reads_every_row_once_and_the_last_page_names_no_next(action: str) -> None:
    fixture, snapshots = off_loop(_seeded)
    name, call = _PAGED_CALLS[action]
    snapshot = snapshots[name]
    tool = _analyst_tool()
    first = {"action": action, "snapshot_id": snapshot.id, **call}
    whole = await _call(tool, fixture, {**first, "limit": 200})
    assert whole.structured is not None, whole.text
    assert "next_offset" not in whole.structured and "next" not in whole.structured
    assert "next_offset" not in whole.text and "\nnext: " not in whole.text

    seen: list[Any] = []
    arguments: dict[str, Any] | None = first
    pages = 0
    while arguments is not None:
        page = await _call(tool, fixture, arguments)
        assert page.ending == "ok" and page.structured is not None, page.text
        seen += _page_rows(action, page.structured)
        following = page.structured.get("next")
        if following is not None:
            # The call is the arguments sent, as validated, the offset advanced: as data and as the text's last line.
            held = tool.action(action)
            assert held is not None
            params = held.params
            assert set(following) == {*arguments, "offset"}
            assert list(following)[0] == "action" and list(following)[-1] == "offset"
            sent = {key: value for key, value in arguments.items() if key != "action"}
            asked = {key: value for key, value in following.items() if key != "action"}
            assert params.model_validate(asked) == params.model_validate(
                {**sent, "offset": page.structured["next_offset"]}
            )
            assert following["action"] == action
            assert page.text.splitlines()[-1] == _next_line(following)
            assert page.text.splitlines()[-1].startswith(f"next: action='{action}', snapshot_id='{snapshot.id}'")
        else:
            assert "next_offset" not in page.structured and not page.text.splitlines()[-1].startswith("next: ")
        arguments = following
        pages += 1
    assert pages > 1, "the control: more than one page"
    assert sorted(seen, key=_tag) == sorted(_page_rows(action, whole.structured), key=_tag)


#: The field each listing's rows are in; the readings are grouped by aspect, each group's rows by kind.
_ROWS = {
    "contrasts_list": "contrasts",
    "measure_read": "rows",
    "examples_list": "examples",
    "decision_sensitivity": "flips",
    "measure_breakdown": "slices",
    "block_render": "rows",
}


def _tag(row: Any) -> str:
    """A price row's tag, so the pages' rows compare by kind and each kind in order; every other row has none."""
    return row[0] if isinstance(row, tuple) else ""


def _page_rows(action: str, structured: dict[str, Any]) -> list[Any]:
    """A page's rows, in order."""
    if action in _ROWS:
        return list(structured[_ROWS[action]])
    if action == "price_tradeoff":
        # A page covers options in arm order, each priced against one or listed off the frontier: tagged by which.
        return sorted(
            [("against", row) for row in structured["against"]] + [("off", row) for row in structured["off_frontier"]],
            key=lambda tagged: tagged[0],
        )
    kinds = ("readings", "pairs", "floors", "bars")
    return [row for group in structured["aspects"] for kind in kinds for row in group[kind]]


def _grammar_parts(grammar: str) -> tuple[list[str], list[str]]:
    """The grammar's path lines (its families, between ``Paths:`` and the filters), and every other line of it."""
    lines = grammar.splitlines()
    start = lines.index("Paths:") + 1
    end = next(at for at, line in enumerate(lines) if line.startswith("Filters"))
    return lines[start:end], [*lines[: start - 1], *lines[end:]]


async def test_help_topic_figure_keys_is_an_index_and_its_pages_hold_the_whole_grammar_each_within_budget() -> None:
    fixture, snapshots = off_loop(_seeded)
    grammar = figure_key_grammar()
    paths, rest = _grammar_parts(grammar)
    assert len(grammar) > TEXT_BUDGET, "the control: the whole grammar is past one result's budget"
    for role in ("analyst", "editor"):
        (spec,) = analyst_tools(role=role)
        tool = eval_catalogue().mount(spec)
        index = await _call(tool, fixture, {"action": "help", "topic": "figure_keys"})
        assert index.ending == "ok" and len(index.text) <= TEXT_BUDGET, (role, len(index.text))
        index_lines = index.text.splitlines()
        assert all(line in index_lines for line in rest), "the index holds the grammar's rules and filters whole"
        named = [line.split(":")[0].removeprefix("- ") for line in index_lines if line.startswith("- figure_keys.")]
        assert named, "the index names a page per group"
        held: list[str] = []
        for topic in named:
            page = await _call(tool, fixture, {"action": "help", "topic": topic})
            assert page.ending == "ok" and len(page.text) <= TEXT_BUDGET, (role, topic, len(page.text))
            lines = [line for line in page.text.splitlines() if line.startswith("- `")]
            held += lines
            group = next(line for line in index_lines if line.startswith(f"- {topic}: "))
            assert all(line.split("`")[1] in group.split(". ", 1)[1] for line in lines), "the index names its forms"
        assert sorted(held) == sorted(paths), "every path line of the grammar is on exactly one page"
        missing = await _call(tool, fixture, {"action": "help", "topic": "figure_keys.nothing"})
        assert missing.is_error and missing.recovery is not None
        assert missing.recovery.valid_values == tuple(named)
        assert missing.text.splitlines()[-1].startswith("recovery: ")
        assert "- figure_keys: " in tool.help_index() and "topic='figure_keys.<page>'" in tool.help_index(), role
    catalog = await _call(_analyst_tool(), fixture, {"action": "block_catalog", "snapshot_id": snapshots["r1"].id})
    assert catalog.structured is not None
    forms = catalog.structured["figure_key_forms"]
    assert forms and all(f"- `{form}`" in grammar for form in forms), "the catalog's paths are the grammar's"
    detail = next(action.detail for action in analyst_actions() if action.name == "block_catalog")
    assert "help topic='figure_keys'" in detail


# =============================================================================
# No branch on the kind of eval
# =============================================================================

#: The names an analyst module never reaches for: the planner's classifier, its spine, the reporter kind, its signals.
BANNED_NAMES = frozenset({"classify", "spine", "report_kind", "Signals"})

#: The meta fields an analyst module never reads: what kind of eval it is, and its design's shape.
BANNED_META = frozenset({"kind", "design_shape"})


def _kind_branches(source: str) -> list[str]:
    """Every banned name an analyst module's source reaches for, and every read of ``meta.kind`` or ``meta.design_shape``."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            found.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES:
            found.append(node.attr)
        elif isinstance(node, ast.alias) and {node.name.rsplit(".", 1)[-1], node.asname} & BANNED_NAMES:
            found.append(node.name)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").rsplit(".", 1)[-1] in BANNED_NAMES:
            found.append(node.module or "")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_META and _is_meta(node.value):
            found.append(f"meta.{node.attr}")
        elif isinstance(node, ast.Subscript) and _is_meta(node.value) and _literal(node.slice) in BANNED_META:
            found.append(f"meta[{_literal(node.slice)!r}]")
    return found


def _is_meta(node: ast.expr) -> bool:
    """Whether an expression is a report input's ``meta``: an attribute or a key of that name."""
    return (isinstance(node, ast.Attribute) and node.attr == "meta") or (
        isinstance(node, ast.Subscript) and _literal(node.slice) == "meta"
    )


def _literal(node: ast.expr) -> object:
    return node.value if isinstance(node, ast.Constant) else None


@pytest.mark.parametrize(
    "source",
    [
        "from threetears.evals.analysis.report_plan_checks import classify",
        "from threetears.evals.analysis.report_decision import report_kind as kind_of",
        "x = checks.spine(report_input)",
        "if report_input.meta.kind == 'change':\n    pass",
        "shape = snapshot.report_input['meta']['design_shape']",
        "def f(signals: Signals) -> None: ...",
    ],
)
def test_the_walk_finds_each_way_a_module_could_branch_on_eval_kind(source: str) -> None:
    """The control: each banned reach, written the ways a module could write it, is found."""
    assert _kind_branches(source)


def test_the_walk_covers_every_analyst_module() -> None:
    names = {path.name for path in ANALYST_MODULES}
    assert {"snapshot.py", "cited.py", "analyst.py", "analyst_facts.py", "analyst_render.py"} <= names


@pytest.mark.parametrize("path", ANALYST_MODULES, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_no_analyst_module_branches_on_eval_kind(path: Path) -> None:
    assert _kind_branches(path.read_text(encoding="utf-8")) == []
