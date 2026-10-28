"""The writing loop through its entry point: the rounds, the endings, the editor's isolation, the store gate.

Every turn is scripted (:class:`~packages.evals.tests.report_writer_support.FakeToolTurnClient`, one per named
session) and every tool call runs live on Kairo's fixture snapshot through ``evals_analyst`` as ``write_report``
mounts it. The analyst submits the third prototype's converted Kairo document, which passes validity and every
invariant on that snapshot; the editor answers scripted reviews.

Contracts held here:

- **rounds**: a review that clears ends ``stored`` in round 1; a failed review or a ``must_fix`` sends the analyst back
  in its own conversation, at most twice, and three failed reviews end ``stored_with_findings``;
- **endings** (plan §4.4): ``over_budget`` stores the last reviewed document with its review's findings, or nothing
  when no review was read; ``refused``, ``review_failed`` and ``failed`` store nothing, and the tally keeps every turn;
- **the editor**: an unreadable answer is asked for once more, and a second ends ``review_failed``; its request
  carries nothing of the analyst's session, and its function cannot be handed any; code adds its scores up;
- **budgets** refuse a turn before it is sent;
- **the store gate** re-runs the invariants and refuses a document that fails one.
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import pytest

from packages.evals.tests.fixtures.report_plans.analyst_cli import seeded_host
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, ops_fixture, OpsFixture, planned_generation
from packages.evals.tests.report_doc_support import snapshot_of
from packages.evals.tests.report_writer_support import (
    DOC,
    MODEL,
    FakeToolTurnClient,
    FakeTurnModel,
    answer,
    call,
    kairo_id,
    reply,
    review,
    submitted,
    thinking,
    written,
)
from threetears.evals.actions import Caller, MountedTool, analyst_tools, eval_catalogue
from threetears.evals.actions.report_writing import ReportNotStored, write_report
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.text import doc_page_text
from threetears.evals.analysis.report_doc_store import doc_verdicts, generation_record, made_line, store_report_doc
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel import StorageError
from threetears.evals.kernel.report_doc import (
    EvalReportDoc,
    EvalReportSession,
    ReportDoc,
    ReportFinding,
    new_report_doc_id,
    report_session_id,
)
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.editor import EditorLimits, Review, RubricScore, review_report
from threetears.evals.ops.report_writer.loop import WriterLimits
from threetears.evals.ops.report_writer.session import ToolOutcome, ToolSession
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.schema.turns import TurnText

#: A finding that sends a report back however it scored.
MUST_FIX: Final = {"claim": "c2", "severity": "must_fix", "finding": "Say what clearing the margin would buy."}
#: A finding that does not.
SHOULD_FIX: Final = {"claim": "c1", "severity": "should_fix", "finding": "Name the margin in the first claim."}
#: Scores that fail: one item at 0.
FAILING: Final = (2, 2, 0, 2, 2, 2, 2)

#: What the analyst's first reply holds that no editor may read.
PRIVATE_THOUGHT: Final = "ANALYST-PRIVATE-THOUGHT: the cost gap is the whole story"
PRIVATE_NOTE: Final = "ANALYST-PRIVATE-NOTE: reading the dominance before I write"


def analyst(*replies: Any, ceiling: float | None = 0.01) -> FakeToolTurnClient:
    """The analyst's session: a tool call, then each scripted reply (by default, the Kairo document)."""
    snap = kairo_id()
    return FakeToolTurnClient(
        reply(thinking(PRIVATE_THOUGHT), TurnText(PRIVATE_NOTE), call(1, "options_dominance", snap)),
        *(replies or (reply(submitted()),)),
        ceiling=ceiling,
    )


def editor(*replies: Any) -> FakeToolTurnClient:
    """An editor round's session: a tool call, then each scripted reply."""
    return FakeToolTurnClient(reply(call(9, "options_dominance", kairo_id())), *replies)


def revised() -> dict[str, Any]:
    """A revision of the Kairo document that still passes every check: one more item left out."""
    doc = copy.deepcopy(DOC)
    doc["omitted"].append({"what": "The per-task breakdown", "why": "It moves no arm across the margin."})
    return doc


def _texts(request: dict[str, Any]) -> str:
    return "\n".join(
        block["text"] for message in request["messages"] for block in message["content"] if block["type"] == "text"
    )


def _stored(host: OpsHost) -> list[Any]:
    return off_loop(host.eval_host.storage.list_report_docs, CALLER.scope_id)


# --- rounds -------------------------------------------------------------------------------------------------------


async def test_a_review_that_clears_in_round_one_stores_the_report_with_its_should_fix_findings() -> None:
    host = off_loop(seeded_host)
    model = FakeTurnModel(analyst=analyst(), editor_1=editor(reply(review(findings=[SHOULD_FIX]))))

    result = await written(model, host=host)

    assert result.ending == "stored", result.outcome.reason
    assert model.asked == ["analyst", "editor-1"]
    assert (result.outcome.analyst_rounds, result.outcome.editor_rounds) == (1, 1)
    (stored,) = _stored(host)
    assert stored.id == result.stored.id and stored.ending == "stored"
    assert ReportDoc.model_validate(stored.doc) == ReportDoc.model_validate(DOC)
    assert stored.findings == [SHOULD_FIX]
    generation = stored.generation
    assert generation["model"] == model.model_name
    assert set(generation["tally"]["by_round"]) == {"analyst:1", "editor:1"}
    assert generation["tally"]["total"]["tool_calls"] == 2
    assert [item["total"] for item in generation["reviews"]] == [13]
    assert set(generation["sessions"]) == {"analyst", "editor-1"}
    assert {entry["ref"] for entry in generation["sessions"].values()} == {
        report_session_id(stored.id, "analyst"),
        report_session_id(stored.id, "editor-1"),
    }
    sessions = off_loop(host.eval_host.storage.load_report_sessions, stored.id, CALLER.scope_id)
    assert sessions["analyst"].session["session"]["name"] == "analyst"
    assert sessions["editor-1"].session["name"] == "editor-1"
    assert {result["stored"] for result in sessions["editor-1"].session["results"]} == {"rederive"}
    assert result.page is not None and result.page.made.startswith(f"Written by {model.model_name}")
    assert [finding.finding for finding in result.page.findings] == [SHOULD_FIX["finding"]]


async def test_a_review_that_fails_sends_the_analyst_back_and_a_review_of_the_revision_clears_it() -> None:
    first = editor(reply(review(FAILING, findings=[SHOULD_FIX])))
    second = FakeToolTurnClient(reply(review()))
    analyst_client = analyst(reply(submitted()), reply(submitted(revised())))
    model = FakeTurnModel(analyst=analyst_client, editor_1=first, editor_2=second)

    result = await written(model)

    assert result.ending == "stored", result.outcome.reason
    assert (result.outcome.analyst_rounds, result.outcome.editor_rounds) == (2, 2)
    assert result.stored is not None and len(result.stored.doc["omitted"]) == len(revised()["omitted"])
    assert result.stored.findings == []
    revision = analyst_client.requests[-1]["messages"][-1]
    asked = _texts({"messages": [revision]})
    assert "does not pass" in asked and "3. Honest emphasis: 0 of 2" in asked and SHOULD_FIX["finding"] in asked
    assert "This revision has 30 tool calls" in asked
    assert SHOULD_FIX["finding"] in _texts(second.requests[0]), "round 2 is given its earlier findings"
    assert set(result.generation["sessions"]) == {"analyst", "editor-1", "editor-2"}


async def test_a_must_fix_sends_a_review_that_passes_on_its_scores_back() -> None:
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted())),
        editor_1=editor(reply(review(findings=[MUST_FIX]))),
        editor_2=FakeToolTurnClient(reply(review())),
    )

    result = await written(model)

    assert result.ending == "stored" and result.outcome.editor_rounds == 2
    assert result.outcome.reviews[0].passes and result.outcome.reviews[0].must_fix


async def test_three_failing_reviews_end_stored_with_findings_and_no_fourth_round_is_begun() -> None:
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted()), reply(submitted(revised()))),
        editor_1=editor(reply(review(FAILING))),
        editor_2=FakeToolTurnClient(reply(review(findings=[MUST_FIX]))),
        editor_3=FakeToolTurnClient(reply(review(FAILING, findings=[MUST_FIX, SHOULD_FIX]))),
    )

    result = await written(model)

    assert result.ending == "stored_with_findings"
    assert model.asked == ["analyst", "editor-1", "editor-2", "editor-3"]
    assert (result.outcome.analyst_rounds, result.outcome.editor_rounds) == (3, 3)
    assert result.stored is not None and result.stored.ending == "stored_with_findings"
    assert result.stored.findings == [MUST_FIX, SHOULD_FIX]
    assert len(result.stored.doc["omitted"]) == len(revised()["omitted"]), "the last document is the one stored"
    assert result.page is not None
    assert {shown.finding for shown in result.page.findings} == {MUST_FIX["finding"], SHOULD_FIX["finding"]}


async def test_fewer_revisions_end_sooner() -> None:
    model = FakeTurnModel(analyst=analyst(), editor_1=editor(reply(review(FAILING))))

    result = await written(model, limits=WriterLimits(revisions=0))

    assert result.ending == "stored_with_findings" and model.asked == ["analyst", "editor-1"]


def test_more_than_two_revisions_are_refused() -> None:
    with pytest.raises(ValueError, match="0 to 2 revisions"):
        WriterLimits(revisions=3)


# --- budgets and the endings that store nothing ---------------------------------------------------------------------


async def test_the_budget_running_out_before_any_review_stores_nothing_and_sends_the_editor_nothing() -> None:
    host = off_loop(seeded_host)
    first = editor(reply(review()))
    first.ceilings[0] = 1000.0
    model = FakeTurnModel(analyst=analyst(), editor_1=first)

    result = await written(model, host=host)

    assert result.ending == "over_budget" and "budget refused" in result.outcome.reason
    assert first.requests == [], "the refused turn was never sent"
    assert result.stored is None and result.page is None and _stored(host) == []
    assert result.generation["tally"]["total"]["turns"] == 2, "the analyst's turns are kept"


async def test_the_budget_running_out_after_a_review_stores_the_reviewed_draft_with_its_findings() -> None:
    host = off_loop(seeded_host)
    analyst_client = analyst(reply(submitted()), reply(submitted(revised())))
    analyst_client.ceilings[2] = 1000.0
    model = FakeTurnModel(analyst=analyst_client, editor_1=editor(reply(review(FAILING, findings=[MUST_FIX]))))

    result = await written(model, host=host)

    assert result.ending == "over_budget"
    assert len(analyst_client.requests) == 2, "the revision's turn was refused before it was sent"
    (stored,) = _stored(host)
    assert stored.ending == "over_budget" and stored.findings == [MUST_FIX]
    assert ReportDoc.model_validate(stored.doc) == ReportDoc.model_validate(DOC)


async def test_over_budget_stores_the_last_reviewed_draft_not_a_later_unreviewed_one() -> None:
    second = FakeToolTurnClient(reply(review()))
    second.ceilings[0] = 1000.0
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted(revised()))),
        editor_1=editor(reply(review(FAILING))),
        editor_2=second,
    )

    result = await written(model)

    assert result.ending == "over_budget" and second.requests == []
    assert result.stored is not None and result.stored.doc["omitted"] == DOC["omitted"]


async def test_a_submission_that_still_fails_after_its_bounces_ends_refused_and_stores_nothing() -> None:
    broken = copy.deepcopy(DOC)
    broken["method"] = []
    model = FakeTurnModel(analyst=analyst(*(reply(submitted(broken)) for _ in range(4))))

    result = await written(model)

    assert result.ending == "refused" and result.stored is None
    assert model.asked == ["analyst"], "a refused submission costs no editor call"
    assert result.outcome.faults and result.generation["tally"]["total"]["bounces"] == 3


async def test_a_client_failure_ends_failed_and_keeps_the_tally() -> None:
    failing = editor(ConnectionError("the provider refused the connection after its retries"))
    model = FakeTurnModel(analyst=analyst(), editor_1=failing)

    result = await written(model)

    assert result.ending == "failed" and result.stored is None
    assert "ConnectionError" in result.outcome.reason
    assert result.generation["tally"]["total"]["turns"] == 3
    assert result.generation["sessions"]["analyst"]["session"]["session"]["messages"]
    assert result.generation["sessions"]["editor-1"]["session"]["messages"], "the failed round's session is kept"


# --- the editor ---------------------------------------------------------------------------------------------------


async def test_an_unreadable_review_is_asked_for_once_more() -> None:
    first = editor(reply(TurnText("The report reads well; I would pass it.")), reply(review()))
    model = FakeTurnModel(analyst=analyst(), editor_1=first)

    result = await written(model)

    assert result.ending == "stored"
    asked_again = _texts({"messages": [first.requests[-1]["messages"][-1]]})
    assert "could not be read" in asked_again and "no fenced block" in asked_again


async def test_a_review_unreadable_twice_ends_review_failed_and_stores_nothing() -> None:
    bad_name = answer()
    bad_name["items"][0]["name"] = "Headline"
    first = editor(
        reply(TurnText('```json\n{"items": []}\n```')), reply(TurnText(f"```json\n{json.dumps(bad_name)}\n```"))
    )
    model = FakeTurnModel(analyst=analyst(), editor_1=first)

    result = await written(model)

    assert result.ending == "review_failed" and result.stored is None
    assert "item 1 is named 'First screen'" in result.outcome.reason


async def test_the_editor_request_carries_nothing_of_the_analysts_session() -> None:
    first = editor(reply(review()))
    analyst_client = analyst()
    model = FakeTurnModel(analyst=analyst_client, editor_1=first)

    await written(model)

    sent = json.dumps(first.requests)
    analyst_sent = json.dumps(analyst_client.requests)
    for private in (PRIVATE_THOUGHT, PRIVATE_NOTE, "toolu_001", "Write the report on snapshot", "sig-"):
        assert private in analyst_sent, f"the analyst's session holds {private!r}"
        assert private not in sent, f"the editor was sent {private!r}"
    opening = first.requests[0]
    assert len(opening["messages"]) == 1 and opening["system"].startswith('<file path="editor.md">')
    said = _texts(opening)
    assert "The page, as a reader sees it" in said and "Computed facts, for you to weigh" in said
    assert DOC["omitted"][0]["what"] in said
    draft_page = doc_page_text(build(ReportDoc.model_validate(DOC), _kairo_input(), strict=True))
    assert f"The page, as a reader sees it:\n\n{draft_page}" in said, "the draft's page, no made line, line for line"
    assert "Written by" not in said
    assert opening["tools"][0]["name"] == "evals_analyst"
    actions = opening["tools"][0]["input_schema"]["properties"]["action"]["enum"]
    assert "options_dominance" in actions and "block_render" not in actions, "the editor's cut carries no blocks"


def test_the_editor_function_takes_nothing_an_analyst_session_could_be_passed_through() -> None:
    """Isolation is structural: every parameter is the document, its snapshot, earlier findings or a session's makings."""
    parameters = inspect.signature(review_report).parameters
    assert list(parameters) == [
        "doc",
        "report_input",
        "round",
        "snapshot",
        "earlier",
        "model",
        "client",
        "mount",
        "skill",
        "tally",
        "budget",
        "limits",
    ]
    annotations = " ".join(str(parameter.annotation) for parameter in parameters.values())
    for leak in ("AnalystSession", "TurnMessage", "TurnBlock", "SessionStop", "AnalystOutcome"):
        assert leak not in annotations


@pytest.mark.parametrize(
    ("scores", "total", "passes"),
    [
        pytest.param((2, 2, 2, 2, 2, 1, 2), 13, True, id="13 with no 0"),
        pytest.param((2, 2, 2, 1, 1, 2, 1), 11, True, id="exactly 11"),
        pytest.param((2, 1, 1, 2, 1, 2, 1), 10, False, id="10"),
        pytest.param((2, 2, 2, 2, 2, 2, 0), 12, False, id="12 with a 0"),
    ],
)
def test_code_adds_the_scores_up_and_says_whether_they_pass(scores: tuple[int, ...], total: int, passes: bool) -> None:
    review_ = Review(
        round=1,
        items=tuple(
            RubricScore(n=n, name=item["name"], score=item["score"], reason="r")
            for n, item in enumerate(answer(scores)["items"], start=1)
        ),
        findings=(),
    )
    assert (review_.total, review_.passes, review_.clears) == (total, passes, passes)


async def test_a_total_and_a_pass_the_editor_writes_are_never_read() -> None:
    claimed = editor(reply(review(FAILING, total=14, passes=True)))
    model = FakeTurnModel(analyst=analyst(), editor_1=claimed)

    result = await written(model, limits=WriterLimits(revisions=0))

    (read,) = result.outcome.reviews
    assert (read.total, read.passes) == (12, False)
    assert result.ending == "stored_with_findings"
    assert result.generation["reviews"][0]["total"] == 12


# --- the store gate -------------------------------------------------------------------------------------------------


def _kairo_input() -> ReportInput:
    return ReportInput.model_validate(snapshot_of("prototype3/kairo").report_input)


def _store(doc: dict[str, Any], storage: Any) -> Any:
    snapshot = snapshot_of("prototype3/kairo")
    report_doc_id = new_report_doc_id()
    return store_report_doc(
        storage,
        report_doc_id=report_doc_id,
        scope_id=CALLER.scope_id,
        snapshot_id=snapshot.id,
        campaign_id=snapshot.campaign_id,
        doc=ReportDoc.model_validate(doc),
        report_input=_kairo_input(),
        ending="stored",
        findings=[ReportFinding.model_validate(SHOULD_FIX)],
        generation=generation_record(
            report_doc_id=report_doc_id,
            model="m",
            tally={"turns": [], "total": {"cost_usd": None, "unpriced_turns": 0}},
            skill={"package": "f" * 64},
            reviews=[],
            sessions={},
        ),
        sessions={},
        created_by=CALLER.identity,
    )


def test_the_store_gate_stores_a_document_that_passes_with_its_made_line_checked() -> None:
    storage = seeded_host().eval_host.storage

    record = _store(DOC, storage)

    assert storage.load_report_doc(record.id, CALLER.scope_id) is not None


def test_the_store_gate_reruns_the_invariants_and_refuses_a_document_that_fails_one() -> None:
    storage = seeded_host().eval_host.storage
    typed = copy.deepcopy(DOC)
    typed["headline"] += " It costs 40 percent more."

    with pytest.raises(ValueError, match=r"is not stored: it has \d+ verdicts") as refused:
        _store(typed, storage)

    assert "\n- [1] headline: " in str(refused.value)

    assert storage.list_report_docs(CALLER.scope_id) == []


def test_a_page_that_cannot_be_built_strict_is_a_verdict_not_a_crash() -> None:
    unresolved = copy.deepcopy(DOC)
    unresolved["lede"] += " {cost.no_such_figure}"

    verdicts = doc_verdicts(ReportDoc.model_validate(unresolved), _kairo_input())

    assert verdicts and verdicts[-1].path == "lede" and "does not resolve" in verdicts[-1].fault


# --- the editor's other endings, and its sessions stored and resumed -------------------------------------------------


async def test_an_editor_reply_the_model_stopped_short_ends_failed_and_keeps_the_session() -> None:
    first = editor(reply(TurnText("I will not review this."), stop="content_filter"))
    model = FakeTurnModel(analyst=analyst(), editor_1=first)

    result = await written(model)

    assert result.ending == "failed" and result.stored is None
    assert "content_filter" in result.outcome.reason
    assert result.generation["sessions"]["editor-1"]["session"]["state"]["turns"] == 2


async def test_an_editor_reply_cut_at_the_output_cap_is_asked_for_once_more() -> None:
    first = editor(reply(TurnText('```json\n{"items": ['), stop="max_tokens"), reply(review()))
    model = FakeTurnModel(analyst=analyst(), editor_1=first)

    result = await written(model)

    assert result.ending == "stored"
    asked_again = _texts({"messages": [first.requests[-1]["messages"][-1]]})
    assert "output cap" in asked_again and "could not be read" in asked_again


async def test_the_editor_is_stopped_at_40_tool_calls_and_asked_for_its_review() -> None:
    snap = kairo_id()
    assert EditorLimits().calls == 40
    first = FakeToolTurnClient(
        reply(*(call(n, "help", snap) for n in range(1, 42))),
        reply(call(99, "help", snap)),
        reply(review()),
    )
    model = FakeTurnModel(analyst=analyst(), editor_1=first)

    result = await written(model)

    assert result.ending == "stored"
    assert result.generation["tally"]["by_round"]["editor:1"]["tool_calls"] == 40
    answered = first.requests[1]["messages"][-1]["content"]
    assert answered[40]["content"].startswith("not run: this round's 40 tool calls are used up")
    assert "Answer now" in answered[-1]["text"]
    asked_again = _texts({"messages": [first.requests[2]["messages"][-1]]})
    assert "this review's 40 tool calls are used up, and you called a tool again" in asked_again


@dataclass(frozen=True)
class BoundTool:
    """The editor's tool bound to a host and caller, as ``write_report`` mounts it."""

    tool: MountedTool
    host: OpsHost
    caller: Caller

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def description(self) -> str:
        return self.tool.description

    @property
    def rederivable(self) -> bool:
        return True

    def input_schema(self) -> dict[str, Any]:
        return self.tool.input_schema()

    async def call(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        return await self.tool.call(arguments, host=self.host, caller=self.caller)


async def test_a_stored_editor_session_resumes_sending_what_the_uninterrupted_session_would() -> None:
    (spec,) = analyst_tools(writes=False, role="editor")
    mount = BoundTool(eval_catalogue().mount(spec), off_loop(seeded_host), CALLER)
    skill, snap = load_skill().editor(), kairo_id()
    final = reply(review(findings=[SHOULD_FIX]))
    first = FakeToolTurnClient(reply(call(9, "options_dominance", snap)), final)

    reviewed = await review_report(
        ReportDoc.model_validate(DOC),
        _kairo_input(),
        round=1,
        snapshot=snap,
        earlier=(),
        model=MODEL,
        client=first,
        mount=mount,
        skill=skill,
        tally=Tally(),
        budget=ReportBudget(usd=10),
    )

    record = json.loads(json.dumps(reviewed.session))
    assert reviewed.ending == "reviewed" and record["name"] == "editor-1"
    assert [result["stored"] for result in record["results"]] == ["rederive"]
    resumed_client = FakeToolTurnClient(reply(TurnText("Item 6 names no experiment.")))
    resumed = await ToolSession.from_json(
        record,
        model=MODEL,
        client=resumed_client,
        mount=mount,
        system=skill,
        tally=Tally(),
        budget=ReportBudget(usd=10),
    )
    resumed.open_round(2, calls=5)
    await resumed.send("Why did item 6 score 1?")

    (sent,) = resumed_client.requests
    uninterrupted = [
        *first.requests[-1]["messages"],
        final.message().to_json(),
        {"role": "user", "content": [{"type": "text", "text": "Why did item 6 score 1?"}]},
    ]
    assert sent["messages"] == uninterrupted
    assert (sent["system"], sent["tools"]) == (first.requests[0]["system"], first.requests[0]["tools"])


# --- the freeze path ------------------------------------------------------------------------------------------------


def _unsubmitted() -> FakeToolTurnClient:
    """An analyst that never submits a document: its round ends refused after its bounces, with no editor call."""
    return FakeToolTurnClient(*(reply(TurnText("Still reading.")) for _ in range(4)))


async def _written_from(fixture: OpsFixture, model: FakeTurnModel, **source: str) -> Any:
    return await write_report(fixture.host, CALLER, model=model, budget=ReportBudget(usd=10), **source)


def _assert_frozen_and_read_only(result: Any, model: FakeTurnModel, client: FakeToolTurnClient) -> None:
    assert result.ending == "refused" and model.snapshots == [result.snapshot_id]
    assert f"Write the report on snapshot {result.snapshot_id}" in _texts(client.requests[0])
    actions = client.requests[0]["tools"][0]["input_schema"]["properties"]["action"]["enum"]
    assert "campaign_overview" in actions and "report_snapshot_freeze" not in actions, "the mount carries no write"


async def test_a_report_written_from_a_campaign_freezes_it_first_and_mounts_no_write() -> None:
    fixture = off_loop(ops_fixture, generation=False)
    client = _unsubmitted()
    model = FakeTurnModel(analyst=client)

    result = await _written_from(fixture, model, campaign_id=fixture.campaign.id)

    snapshot = off_loop(fixture.host.eval_host.storage.load_report_snapshot, result.snapshot_id, CALLER.scope_id)
    assert snapshot is not None and (snapshot.campaign_id, snapshot.analysis_id) == (fixture.campaign.id, None)
    _assert_frozen_and_read_only(result, model, client)


async def test_a_report_written_from_an_analysis_freezes_its_bundle_first_and_mounts_no_write() -> None:
    fixture = off_loop(ops_fixture)
    await planned_generation(fixture)
    (analysis,) = off_loop(
        fixture.host.eval_host.storage.list_analyses_by_campaign, fixture.campaign.id, CALLER.scope_id
    )
    client = _unsubmitted()
    model = FakeTurnModel(analyst=client)

    result = await _written_from(fixture, model, analysis_id=analysis.id)

    snapshot = off_loop(fixture.host.eval_host.storage.load_report_snapshot, result.snapshot_id, CALLER.scope_id)
    assert snapshot is not None and snapshot.analysis_id == analysis.id
    _assert_frozen_and_read_only(result, model, client)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param({}, id="none"),
        pytest.param({"snapshot_id": "report-snapshot:any", "campaign_id": "any-campaign"}, id="two"),
        pytest.param({"campaign_id": "any-campaign", "analysis_id": "any-analysis"}, id="a campaign and an analysis"),
    ],
)
async def test_a_report_is_written_from_exactly_one_source(source: dict[str, str]) -> None:
    model = FakeTurnModel()

    with pytest.raises(ValueError, match="a report is written from one source"):
        await write_report(
            off_loop(seeded_host),
            CALLER,
            model=model,
            budget=ReportBudget(usd=10),
            **source,  # type: ignore[arg-type]
        )

    assert model.asked == [], "nothing is frozen or sent"


# --- the made line ------------------------------------------------------------------------------------------------


def _record(generation: dict[str, Any]) -> EvalReportDoc:
    snapshot = snapshot_of("prototype3/kairo")
    return EvalReportDoc(
        scope_id=CALLER.scope_id,
        snapshot_id=snapshot.id,
        campaign_id=snapshot.campaign_id,
        doc=DOC,
        ending="stored",
        generation=generation,
        created_by=CALLER.identity,
        created_at="2026-10-28T09:30:00+00:00",
    )


def _generation(cost: float | None, unpriced: int) -> dict[str, Any]:
    turns = [
        {"model": "writer-a", "served_model": "writer-a-20261001"},
        {"model": "writer-a", "served_model": None},
        {"model": "writer-b", "served_model": "writer-b-20260901"},
    ]
    return generation_record(
        report_doc_id="report-1",
        model="writer-a",
        tally={"turns": turns, "total": {"cost_usd": cost, "unpriced_turns": unpriced}},
        skill={"analyst": "a" * 64, "editor": "e" * 64, "package": "0123456789abcdef" * 4},
        reviews=[],
        sessions={},
    )


def test_the_made_line_of_a_priced_report() -> None:
    assert made_line(_record(_generation(1.234, 0))) == (
        "Written by writer-a, writer-b, served as writer-a-20261001, writer-b-20260901, for $1.23, with analyst skill "
        "0123456789ab, on 2026-10-28."
    )


def test_the_made_line_of_a_report_with_unpriced_turns() -> None:
    assert made_line(_record(_generation(None, 2))) == (
        "Written by writer-a, writer-b, served as writer-a-20261001, writer-b-20260901, at a cost not known (2 turns "
        "unpriced), with analyst skill 0123456789ab, on 2026-10-28."
    )


def test_the_made_line_refuses_a_generation_record_of_another_format() -> None:
    with pytest.raises(ValueError, match="this build reads format 2"):
        made_line(_record({**_generation(1.0, 0), "format": 1}))


# --- a paid run never loses its record ------------------------------------------------------------------------------


async def test_a_store_that_fails_after_the_run_raises_with_the_runs_record_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = off_loop(seeded_host)
    failure = StorageError("the document is larger than the store takes")

    def refuse(report: EvalReportDoc, *, sessions: Sequence[EvalReportSession] = ()) -> None:
        raise failure

    monkeypatch.setattr(host.eval_host.storage, "save_report_doc", refuse)
    model = FakeTurnModel(analyst=analyst(), editor_1=editor(reply(review(findings=[SHOULD_FIX]))))

    with pytest.raises(ReportNotStored, match="the report ended stored and was not stored") as refused:
        await written(model, host=host)

    assert refused.value.__cause__ is failure
    kept = refused.value.written
    assert kept.ending == "stored" and kept.stored is None and kept.page is None
    assert kept.generation["tally"]["total"]["turns"] == 4
    assert set(kept.generation["sessions"]) == {"analyst", "editor-1"}
    assert [review["total"] for review in kept.generation["reviews"]] == [13]
