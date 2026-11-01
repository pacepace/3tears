"""The writing loop through its entry point: the rounds, the endings, the editor's isolation, the store gate.

Every turn is scripted (:class:`~packages.evals.tests.report_writer_support.FakeToolTurnClient`, one per named
session) and every tool call runs live on Kairo's fixture snapshot through ``evals_analyst`` as ``write_report``
mounts it. The analyst submits the third prototype's converted Kairo document, which passes validity and every
invariant on that snapshot; the editor answers scripted reviews.

Contracts held here:

- **rounds**: a review that passes with no finding open ends ``stored`` in round 1, finished; a failed review, a
  ``must_fix`` or a ``should_fix`` sends the analyst back in its own conversation, at most twice, and three reviews
  that do not clear end ``stored_with_findings``, a draft whose page says so first (owner ruling 2026-10-29); a
  ``should_fix`` first raised after round 1 is sent back while a revision remains and never makes a draft, and a
  revision it sends that ends short stores the report it cleared, finished (owner ruling 2026-10-31);
- **endings** (plan §4.4): ``over_budget`` stores the last reviewed document with its review's findings, or nothing
  when no review was read; ``refused`` and ``review_failed`` store nothing, and ``failed`` nothing ever; the tally
  keeps every turn. A revision a cleared review sent that ends ``over_budget``, ``refused`` or ``review_failed`` ends
  ``stored`` instead, as the rounds say;
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
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
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
from threetears.evals.ops.report_writer.rubric import editor_rubric
from threetears.evals.ops.report_writer.session import ToolOutcome, ToolSession
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.schema.turns import TurnText

#: A finding that sends a report back however it scored.
MUST_FIX: Final = {"claim": "c2", "severity": "must_fix", "finding": "Say what clearing the margin would buy."}
#: A finding that does not fail a review's scores, and still sends the report back while revisions remain.
SHOULD_FIX: Final = {"claim": "c1", "severity": "should_fix", "finding": "Name the margin in the first claim."}
#: Scores that fail: one item at 0.
FAILING: Final = (2, 2, 0, 2, 2, 2, 2)


def marked(finding: Mapping[str, str], continues: str) -> dict[str, str]:
    """A finding as a later review writes it: marked with the earlier finding's id it continues, or ``new``."""
    return {**finding, "continues": continues}


def kept(finding: Mapping[str, str], id: str, since: int) -> dict[str, Any]:
    """A finding as the store keeps it: its id and the round it was first raised, a page-wide one naming no claim."""
    claim = "" if finding["claim"] == "report" else finding["claim"]
    return {"claim": claim, "severity": finding["severity"], "finding": finding["finding"], "id": id, "since": since}


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


async def test_a_review_that_clears_in_round_one_stores_the_report_finished() -> None:
    host = off_loop(seeded_host)
    model = FakeTurnModel(analyst=analyst(), editor_1=editor(reply(review())))

    result = await written(model, host=host)

    assert result.ending == "stored", result.outcome.reason
    assert model.asked == ["analyst", "editor-1"]
    assert (result.outcome.analyst_rounds, result.outcome.editor_rounds) == (1, 1)
    (stored,) = _stored(host)
    assert stored.id == result.stored.id and stored.ending == "stored"
    assert not stored.draft and stored.findings == []
    assert ReportDoc.model_validate(stored.doc) == ReportDoc.model_validate(DOC)
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
    assert not result.page.draft and result.page.findings == []
    assert "Draft" not in doc_page_text(result.page)


async def test_a_should_fix_sends_a_review_that_passes_on_its_scores_back_and_a_clean_review_stores_it() -> None:
    analyst_client = analyst(reply(submitted()), reply(submitted(revised())))
    model = FakeTurnModel(
        analyst=analyst_client,
        editor_1=editor(reply(review(findings=[SHOULD_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(resolved=["f1.1"]))),
    )

    result = await written(model)

    assert result.ending == "stored", result.outcome.reason
    assert model.asked == ["analyst", "editor-1", "editor-2"]
    first = result.outcome.reviews[0]
    assert first.passes and not first.must_fix and not first.clears, "a passing review with a should_fix is not clean"
    asked = _texts({"messages": [analyst_client.requests[-1]["messages"][-1]]})
    assert "holds findings that block it" in asked and "f1.1 [should_fix, blocking]" in asked
    assert SHOULD_FIX["finding"] in asked
    assert "must_fix and should_fix alike" in asked
    assert result.stored is not None and result.stored.findings == [] and not result.stored.draft
    assert result.page is not None and not result.page.draft


async def test_a_review_that_fails_sends_the_analyst_back_and_a_review_of_the_revision_clears_it() -> None:
    first = editor(reply(review(FAILING, findings=[SHOULD_FIX])))
    second = FakeToolTurnClient(reply(review(resolved=["f1.1"])))
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
    assert f"- f1.1 [should_fix] c1: {SHOULD_FIX['finding']}" in _texts(second.requests[0]), (
        "round 2 is given its earlier findings, each by its id"
    )
    assert result.outcome.reviews[1].resolved == ("f1.1",)
    assert set(result.generation["sessions"]) == {"analyst", "editor-1", "editor-2"}


async def test_a_must_fix_sends_a_review_that_passes_on_its_scores_back() -> None:
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted())),
        editor_1=editor(reply(review(findings=[MUST_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(resolved=["f1.1"]))),
    )

    result = await written(model)

    assert result.ending == "stored" and result.outcome.editor_rounds == 2
    assert result.outcome.reviews[0].passes and result.outcome.reviews[0].must_fix


async def test_three_failing_reviews_end_stored_with_findings_and_no_fourth_round_is_begun() -> None:
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted()), reply(submitted(revised()))),
        editor_1=editor(reply(review(FAILING))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(MUST_FIX, "new")]))),
        editor_3=FakeToolTurnClient(
            reply(review(FAILING, findings=[marked(MUST_FIX, "f2.1"), marked(SHOULD_FIX, "new")]))
        ),
    )

    result = await written(model)

    assert result.ending == "stored_with_findings"
    assert model.asked == ["analyst", "editor-1", "editor-2", "editor-3"]
    assert (result.outcome.analyst_rounds, result.outcome.editor_rounds) == (3, 3)
    assert result.stored is not None and result.stored.ending == "stored_with_findings"
    assert result.stored.findings == [kept(MUST_FIX, "f2.1", 2), kept(SHOULD_FIX, "f3.1", 3)]
    assert len(result.stored.doc["omitted"]) == len(revised()["omitted"]), "the last document is the one stored"
    assert result.page is not None
    assert {shown.finding for shown in result.page.findings} == {MUST_FIX["finding"], SHOULD_FIX["finding"]}
    assert result.stored.draft and result.page.draft


async def test_a_loop_out_of_revisions_with_should_fix_findings_open_stores_a_draft_that_says_so_first() -> None:
    """The Discodon incident's shape: every review passes on its scores, and its should_fix findings, raised in round 1,
    stay open through every revision."""
    other = {
        "claim": "report",
        "severity": "should_fix",
        "finding": "Give the options table's cost column its dollars.",
    }
    host = off_loop(seeded_host)
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted()), reply(submitted(revised()))),
        editor_1=editor(reply(review(findings=[SHOULD_FIX, other]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(SHOULD_FIX, "f1.1"), marked(other, "f1.2")]))),
        editor_3=FakeToolTurnClient(reply(review(findings=[marked(SHOULD_FIX, "f1.1"), marked(other, "f1.2")]))),
    )

    result = await written(model, host=host)

    assert result.ending == "stored_with_findings", result.outcome.reason
    assert all(item.passes and not item.must_fix for item in result.outcome.reviews)
    (stored,) = _stored(host)
    assert stored.ending == "stored_with_findings" and stored.draft
    assert stored.findings == [kept(SHOULD_FIX, "f1.1", 1), kept(other, "f1.2", 1)], "carried, they keep their ids"
    assert [len(review.blocking) for review in result.outcome.reviews] == [2, 2, 2]
    page = result.page
    assert page is not None and page.draft
    lines = [line for line in doc_page_text(page).splitlines() if line]
    assert lines[1] == "Draft: its editor's review left 2 findings open.", "first, right under the masthead"
    assert [finding.finding for finding in page.findings] == [SHOULD_FIX["finding"], other["finding"]]


# --- carried and new findings after round 1 (owner rulings 2026-10-29, option 3, and 2026-10-31) --------------------

#: A should_fix a later review raises for the first time.
LATE: Final = {"claim": "report", "severity": "should_fix", "finding": "Say the run's date in the lede."}


async def test_a_new_should_fix_in_round_two_gets_one_more_revision_and_stays_open_on_a_finished_report() -> None:
    """The Kairo pilot's shape: a should_fix new in round 2, with a revision left, is sent back once more; still open
    after it, it blocks nothing, and the report is stored finished, the finding listed open and counted at the top."""
    host = off_loop(seeded_host)
    analyst_client = analyst(reply(submitted()), reply(submitted()), reply(submitted(revised())))
    model = FakeTurnModel(
        analyst=analyst_client,
        editor_1=editor(reply(review(findings=[SHOULD_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(LATE, "new")], resolved=["f1.1"]))),
        editor_3=FakeToolTurnClient(reply(review(findings=[marked(LATE, "f2.1")]))),
    )

    result = await written(model, host=host)

    assert result.ending == "stored", result.outcome.reason
    assert model.asked == ["analyst", "editor-1", "editor-2", "editor-3"], "one more revision, and no fourth review"
    second, third = result.outcome.reviews[1:]
    assert second.clears and not second.settles and second.blocking == ()
    asked = _texts({"messages": [analyst_client.requests[-1]["messages"][-1]]})
    assert "holds findings that do not block it" in asked and f"- f2.1 [should_fix] report: {LATE['finding']}" in asked
    assert third.clears and third.blocking == () and third.findings[0].since == 2, "carried, it still blocks nothing"
    (stored,) = _stored(host)
    assert stored.ending == "stored" and not stored.draft
    assert stored.findings == [kept(LATE, "f2.1", 2)], "counted in the stored record"
    assert len(stored.doc["omitted"]) == len(revised()["omitted"]), "the revised document is the one stored"
    page = result.page
    assert page is not None and not page.draft
    assert [finding.finding for finding in page.findings] == [LATE["finding"]], "listed open where open findings are"
    text = doc_page_text(page)
    assert "Draft" not in text
    assert [line for line in text.splitlines() if line][1] == "1 review finding open, listed at the end."


async def test_a_new_should_fix_in_round_two_that_its_revision_resolves_stores_a_clean_report() -> None:
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted()), reply(submitted(revised()))),
        editor_1=editor(reply(review(findings=[SHOULD_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(LATE, "new")], resolved=["f1.1"]))),
        editor_3=FakeToolTurnClient(reply(review(resolved=["f2.1"]))),
    )

    result = await written(model)

    assert result.ending == "stored" and result.outcome.editor_rounds == 3, result.outcome.reason
    assert result.stored is not None and result.stored.findings == [] and not result.stored.draft
    assert result.page is not None and result.page.findings == []
    assert "open, listed at the end" not in doc_page_text(result.page)


async def test_a_new_should_fix_in_the_last_round_stores_a_finished_report_at_once() -> None:
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted()), reply(submitted(revised()))),
        editor_1=editor(reply(review(findings=[MUST_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(MUST_FIX, "f1.1")]))),
        editor_3=FakeToolTurnClient(reply(review(findings=[marked(LATE, "new")], resolved=["f1.1"]))),
    )

    result = await written(model)

    assert result.ending == "stored", result.outcome.reason
    assert model.asked == ["analyst", "editor-1", "editor-2", "editor-3"], "no revision past the two"
    assert result.outcome.analyst_rounds == 3
    assert result.stored is not None and not result.stored.draft
    assert result.stored.findings == [kept(LATE, "f3.1", 3)]


async def test_the_budget_refusing_the_revision_a_cleared_review_sent_stores_that_report_finished() -> None:
    host = off_loop(seeded_host)
    analyst_client = analyst(reply(submitted()), reply(submitted()), reply(submitted(revised())))
    analyst_client.ceilings[3] = 1000.0
    model = FakeTurnModel(
        analyst=analyst_client,
        editor_1=editor(reply(review(findings=[SHOULD_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(LATE, "new")], resolved=["f1.1"]))),
    )

    result = await written(model, host=host)

    assert result.ending == "stored", result.outcome.reason
    assert "cleared the report, ended over_budget" in result.outcome.reason
    assert len(analyst_client.requests) == 3, "the revision's turn was refused before it was sent"
    (stored,) = _stored(host)
    assert stored.ending == "stored" and not stored.draft and stored.findings == [kept(LATE, "f2.1", 2)]
    assert ReportDoc.model_validate(stored.doc) == ReportDoc.model_validate(DOC), "the reviewed report, not a later one"


async def test_a_refused_revision_after_a_cleared_review_stores_the_cleared_report_finished() -> None:
    broken = copy.deepcopy(DOC)
    broken["method"] = []
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted()), *(reply(submitted(broken)) for _ in range(4))),
        editor_1=editor(reply(review(findings=[SHOULD_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(LATE, "new")], resolved=["f1.1"]))),
    )

    result = await written(model)

    assert result.ending == "stored" and "ended refused" in result.outcome.reason
    assert model.asked == ["analyst", "editor-1", "editor-2"], "a refused submission costs no editor call"
    assert result.stored is not None and not result.stored.draft and result.stored.doc["method"]


async def test_a_review_of_the_revision_a_cleared_review_sent_unreadable_twice_stores_the_cleared_report() -> None:
    host = off_loop(seeded_host)
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted()), reply(submitted(revised()))),
        editor_1=editor(reply(review(findings=[SHOULD_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(LATE, "new")], resolved=["f1.1"]))),
        editor_3=FakeToolTurnClient(
            reply(TurnText('```json\n{"items": []}\n```')), reply(TurnText('```json\n{"items": []}\n```'))
        ),
    )

    result = await written(model, host=host)

    assert result.ending == "stored", result.outcome.reason
    assert "cleared the report, ended review_failed" in result.outcome.reason
    assert model.asked == ["analyst", "editor-1", "editor-2", "editor-3"]
    (stored,) = _stored(host)
    assert stored.ending == "stored" and not stored.draft and stored.findings == [kept(LATE, "f2.1", 2)]
    assert ReportDoc.model_validate(stored.doc) == ReportDoc.model_validate(DOC), "the reviewed report, not a later one"


async def test_the_budget_refusing_the_review_of_the_revision_a_cleared_review_sent_stores_the_cleared_report() -> None:
    host = off_loop(seeded_host)
    third = FakeToolTurnClient(reply(review()))
    third.ceilings[0] = 1000.0
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted()), reply(submitted(revised()))),
        editor_1=editor(reply(review(findings=[SHOULD_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(LATE, "new")], resolved=["f1.1"]))),
        editor_3=third,
    )

    result = await written(model, host=host)

    assert result.ending == "stored", result.outcome.reason
    assert "cleared the report, ended over_budget" in result.outcome.reason
    assert third.requests == [], "the editor's turn was refused before it was sent"
    (stored,) = _stored(host)
    assert stored.ending == "stored" and not stored.draft and stored.findings == [kept(LATE, "f2.1", 2)]
    assert ReportDoc.model_validate(stored.doc) == ReportDoc.model_validate(DOC), (
        "the reviewed report, not the revision"
    )


async def test_a_client_failure_in_the_revision_a_cleared_review_sent_still_ends_failed() -> None:
    """A failure reaches the caller: the cleared report is not stored over it."""
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted()), ConnectionError("the provider refused the connection")),
        editor_1=editor(reply(review(findings=[SHOULD_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(LATE, "new")], resolved=["f1.1"]))),
    )

    result = await written(model)

    assert result.ending == "failed" and result.stored is None
    assert "ConnectionError" in result.outcome.reason and result.outcome.failure is not None


async def test_a_round_two_review_that_keeps_an_earlier_should_fix_open_revises() -> None:
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted()), reply(submitted(revised()))),
        editor_1=editor(reply(review(findings=[SHOULD_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(SHOULD_FIX, "f1.1")]))),
        editor_3=FakeToolTurnClient(reply(review(resolved=["f1.1"]))),
    )

    result = await written(model)

    assert result.ending == "stored", result.outcome.reason
    assert model.asked == ["analyst", "editor-1", "editor-2", "editor-3"]
    second = result.outcome.reviews[1]
    assert not second.clears and [finding.id for finding in second.blocking] == ["f1.1"]


async def test_a_new_must_fix_in_round_two_revises() -> None:
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted()), reply(submitted(revised()))),
        editor_1=editor(reply(review(findings=[SHOULD_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(MUST_FIX, "new")], resolved=["f1.1"]))),
        editor_3=FakeToolTurnClient(reply(review(resolved=["f2.1"]))),
    )

    result = await written(model)

    assert result.ending == "stored" and result.outcome.editor_rounds == 3, result.outcome.reason
    assert [finding.id for finding in result.outcome.reviews[1].blocking] == ["f2.1"]


async def test_out_of_revisions_with_only_new_should_fix_findings_is_no_draft() -> None:
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted(revised()))),
        editor_1=editor(reply(review(findings=[MUST_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(LATE, "new")], resolved=["f1.1"]))),
    )

    result = await written(model, limits=WriterLimits(revisions=1))

    assert result.ending == "stored", result.outcome.reason
    assert model.asked == ["analyst", "editor-1", "editor-2"], "with no revision left, no third round"
    assert result.stored is not None and not result.stored.draft and len(result.stored.findings) == 1
    assert result.page is not None and not result.page.draft


async def test_out_of_revisions_with_a_carried_finding_open_stores_a_draft() -> None:
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted(revised()))),
        editor_1=editor(reply(review(findings=[SHOULD_FIX]))),
        editor_2=FakeToolTurnClient(reply(review(findings=[marked(SHOULD_FIX, "f1.1"), marked(LATE, "new")]))),
    )

    result = await written(model, limits=WriterLimits(revisions=1))

    assert result.ending == "stored_with_findings", result.outcome.reason
    assert result.stored is not None and result.stored.draft
    assert result.stored.findings == [kept(SHOULD_FIX, "f1.1", 1), kept(LATE, "f2.1", 2)]
    assert result.page is not None and result.page.draft
    lines = [line for line in doc_page_text(result.page).splitlines() if line]
    assert lines[1] == "Draft: its editor's review left 2 findings open."


@pytest.mark.parametrize(
    ("findings", "resolved", "fault"),
    [
        ([SHOULD_FIX], [], "findings[0].continues is None"),
        ([marked(SHOULD_FIX, "f9.9")], [], "findings[0].continues is 'f9.9'"),
        ([], [], "earlier findings ['f1.1'] are neither continued nor resolved"),
        ([marked(SHOULD_FIX, "f1.1")], ["f1.1"], "a finding of this review continues"),
        ([marked(SHOULD_FIX, "f1.1"), marked(LATE, "f1.1")], [], "which another finding already continues"),
        ([], ["f1.1", "f7.1"], "resolved names ['f7.1']"),
    ],
)
async def test_a_later_review_that_does_not_account_for_each_earlier_finding_once_is_asked_again(
    findings: list[dict[str, str]], resolved: list[str], fault: str
) -> None:
    second = FakeToolTurnClient(reply(review(findings=findings, resolved=resolved)), reply(review(resolved=["f1.1"])))
    model = FakeTurnModel(
        analyst=analyst(reply(submitted()), reply(submitted())),
        editor_1=editor(reply(review(findings=[SHOULD_FIX]))),
        editor_2=second,
    )

    result = await written(model)

    assert result.ending == "stored", result.outcome.reason
    assert fault in _texts({"messages": [second.requests[1]["messages"][-1]]})


async def test_a_first_review_that_names_an_earlier_finding_is_asked_again() -> None:
    first = editor(reply(review(findings=[marked(SHOULD_FIX, "f0.1")])), reply(review()))
    model = FakeTurnModel(analyst=analyst(), editor_1=first)

    result = await written(model)

    assert result.ending == "stored", result.outcome.reason
    assert "there are no earlier findings" in _texts({"messages": [first.requests[-1]["messages"][-1]]})


async def test_a_draft_failed_on_its_scores_alone_says_its_review_did_not_pass_it() -> None:
    model = FakeTurnModel(analyst=analyst(), editor_1=editor(reply(review(FAILING))))

    result = await written(model, limits=WriterLimits(revisions=0))

    assert result.ending == "stored_with_findings"
    assert result.stored is not None and result.stored.draft and result.stored.findings == []
    assert result.page is not None
    lines = [line for line in doc_page_text(result.page).splitlines() if line]
    assert lines[1] == "Draft: its editor's review did not pass it."


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
    assert stored.ending == "over_budget" and stored.findings == [kept(MUST_FIX, "f1.1", 1)] and stored.draft
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


async def test_a_validity_fault_and_an_invariant_fault_bounce_once_together_each_with_its_remedy() -> None:
    """One bounce carries every verdict the store gate's checker finds, validity's included: a draft naming an arm the
    snapshot does not hold and typing a digit is sent back once, listing both, each with its remedy, and the next
    submission goes to the editor. A session that checked validity on its own first would bounce twice, the first time
    with validity's fault alone and no remedy."""
    broken = copy.deepcopy(DOC)
    broken["recommends"] = ["October"]
    broken["headline"] += " It costs 40 percent more."
    first = analyst(reply(submitted(broken)), reply(submitted()))
    model = FakeTurnModel(analyst=first, editor_1=editor(reply(review())))

    result = await written(model)

    assert result.ending == "stored", result.outcome.reason
    bounces = [_texts({"messages": [request["messages"][-1]]}) for request in first.requests[2:]]
    assert len(bounces) == 1 and "bounce 1 of 3" in bounces[0]
    (bounce,) = bounces
    validity = [line for line in bounce.splitlines() if line.startswith("- [validity] recommends")]
    invariant = [line for line in bounce.splitlines() if line.startswith("- [invariant] headline")]
    assert validity and "October" in validity[0] and "campaign_overview lists its arms" in validity[0]
    assert invariant and "figure_keys_check" in invariant[0]
    assert result.generation["tally"]["by_round"]["analyst:1"]["bounce_kinds"] == {"validity": 1}


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


def _edited_skill() -> AnalystSkill:
    """The shipped skill with an ``editor.md`` an arm under test might run: item 3 renamed, and an eighth item added."""
    shipped = load_skill()
    text = shipped.files["editor.md"]
    assert "3. **Honest emphasis.**" in text and "\n\nA report passes when" in text
    text = text.replace("3. **Honest emphasis.**", "3. **Fair emphasis.**").replace(
        "\n\nA report passes when",
        "\n8. **Plain words.** Every sentence reads without the campaign's own jargon.\n\nA report passes when",
    )
    return AnalystSkill({**shipped.files, "editor.md": text})


@pytest.mark.parametrize(
    ("scores", "ending"),
    [
        pytest.param((2, 2, 2, 2, 1, 1, 1, 2), "stored", id="13 of 16 passes"),
        pytest.param((2, 2, 1, 2, 1, 1, 1, 2), "stored_with_findings", id="12 of 16 does not"),
    ],
)
async def test_an_arm_whose_editor_renames_and_adds_an_item_is_reviewed_on_its_own_rubric(
    scores: tuple[int, ...], ending: str
) -> None:
    """Its editor scores the eight items its editor.md states, by their names there, and the review is read on them,
    never refused against the shipped seven. It passes at the shipped bar's share of its top, rounded up: 13 of 16."""
    skill = _edited_skill()
    names = [item.name for item in editor_rubric(skill.editor().text).items]
    assert names[2] == "Fair emphasis" and len(names) == 8
    items = [
        {"n": n, "name": name, "score": score, "reason": f"item {n} reads so"}
        for n, (name, score) in enumerate(zip(names, scores, strict=True), start=1)
    ]
    answer_ = TurnText(f"My review.\n```json\n{json.dumps({'items': items, 'findings': []})}\n```")
    model = FakeTurnModel(analyst=analyst(), editor_1=editor(reply(answer_)))

    result = await written(model, skill=skill, limits=WriterLimits(revisions=0))

    (read,) = result.outcome.reviews
    assert [item.name for item in read.items] == names
    assert (read.total, read.pass_total) == (sum(scores), 13)
    assert result.ending == ending


async def test_a_skill_whose_editor_states_no_rubric_is_refused_before_a_turn_is_sent() -> None:
    shipped = load_skill()
    broken = AnalystSkill(
        {**shipped.files, "editor.md": shipped.files["editor.md"].replace("## The rubric", "## Rubric")}
    )
    model = FakeTurnModel(analyst=analyst(), editor_1=editor(reply(review())))

    with pytest.raises(ValueError, match="no '## The rubric' section"):
        await written(model, skill=broken)

    assert model.sessions["analyst"].requests == []


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
    model = FakeTurnModel(analyst=analyst(), editor_1=editor(reply(review())))

    with pytest.raises(ReportNotStored, match="the report ended stored and was not stored") as refused:
        await written(model, host=host)

    assert refused.value.__cause__ is failure
    kept = refused.value.written
    assert kept.ending == "stored" and kept.stored is None and kept.page is None
    assert kept.generation["tally"]["total"]["turns"] == 4
    assert set(kept.generation["sessions"]) == {"analyst", "editor-1"}
    assert [review["total"] for review in kept.generation["reviews"]] == [13]
