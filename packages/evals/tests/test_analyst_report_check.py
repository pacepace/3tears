"""``report_check``: a draft report document read on its snapshot as the store and the editor read it.

Contracts held here (``ops.analyst.report_check``, ``actions.analyst_write``):

- **a clean draft** (each converted document of the second and third prototypes, and a plain draft on every fixture
  snapshot) comes back with no verdict, and its page, read page by page, is the editor's page text
  (``doc_page_text(build(doc, input, strict=True))``) line for line: the same function, never a second rendering;
- **each invariant's planted negative** comes back as a verdict under its number, and a page that does not build is
  one more verdict with no page to read; a draft with no Method item is no ``report-doc/1`` document, so invariant 6
  is the call's refusal;
- **a draft that does not read** (not JSON, or not a document) is refused as the call, naming the field at fault,
  with a recovery line that carries a draft that validates;
- **paging**: a result stays within one result's budget whatever its verdicts: they come first, grouped so each remedy
  is stated once, as many as fit, and those that do not are read by ``verdict_offset``, so every verdict is reachable
  and none is cut unsaid; the page's lines are paged by ``offset``, never by a next call that writes the draft back;
- **read-only and idempotent**: the same draft twice is byte-identical, a draft that differs only in ``story`` reads
  the same, and nothing is written to the store;
- **the cut**: the action is the analyst tool's, in the blocks workflow; the editor's cut does not carry it, and the
  ``evals`` tool, the editor's cut and the analyst tool's other actions serve the bytes they served before it.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

import pytest

from packages.evals.tests.analyst_support import KAIRO, TEXT_BUDGET, action_calls, fixture_inputs, fixture_snapshots
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import OpsFixture, ops_fixture
from packages.evals.tests.report_doc_support import LATER_DOCS, converted_doc, snapshot_of
from threetears.evals.actions import (
    ActionOutcome,
    Caller,
    MountedTool,
    analyst_actions,
    analyst_tools,
    catalogue_primer,
    eval_catalogue,
    standard_tools,
)
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.text import doc_page_text
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_doc_checks import check_report_doc
from threetears.evals.analysis.report_doc_store import doc_verdicts
from threetears.evals.kernel.report_doc import ReportDoc, report_doc_prose
from threetears.evals.storage import InMemoryDocumentStore

#: The scope the converted documents' snapshots are frozen in, and a caller in it.
_SCOPE = "uni-1"
_CALLER = Caller(scope_id=_SCOPE, identity="agent:test")

#: The Kairo document of the third prototype, which the negatives are planted in.
_KAIRO = "prototype3/kairo"
#: Where Kairo's recommendation reads.
_RECOMMENDATION = ("decision", 0, "text")


class _WritesCounted(InMemoryDocumentStore):
    """The in-memory store, counting every write it is asked to make."""

    def __init__(self) -> None:
        """Start with no write counted."""
        super().__init__()
        self.writes = 0

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Count, then store."""
        self.writes += 1
        super().upsert(document, if_match=if_match, create_only=create_only)

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        """Count, then merge."""
        self.writes += 1
        return super().merge_fields(doc_id, scope_id, fields)

    def delete(self, doc_id: str, scope_id: str) -> bool:
        """Count, then delete."""
        self.writes += 1
        return super().delete(doc_id, scope_id)


def _seeded(store: InMemoryDocumentStore | None = None) -> OpsFixture:
    """The toy host, with the snapshot of every later converted document stored in its scope."""
    fixture = ops_fixture(store=store)
    for name in LATER_DOCS:
        fixture.host.eval_host.storage.save_report_snapshot(snapshot_of(name))
    return fixture


def _analyst() -> MountedTool:
    (spec,) = analyst_tools()
    return eval_catalogue().mount(spec)


def _editor() -> MountedTool:
    (spec,) = analyst_tools(role="editor")
    return eval_catalogue().mount(spec)


def _input(name: str) -> ReportInput:
    return ReportInput.model_validate(snapshot_of(name).report_input)


async def _check(fixture: OpsFixture, name: str, doc: Any, **more: Any) -> ActionOutcome:
    arguments = {"action": "report_check", "snapshot_id": snapshot_of(name).id, "doc": doc, **more}
    return await _analyst().call(arguments, host=fixture.host, caller=_CALLER)


async def _read_whole(fixture: OpsFixture, name: str, doc: Any, **more: Any) -> tuple[list[ActionOutcome], list[str]]:
    """Every result of a check, following ``unread_from`` from the first, and the page's lines they hold."""
    results: list[ActionOutcome] = []
    lines: list[str] = []
    offset: int | None = 0
    while offset is not None:
        outcome = await _check(fixture, name, doc, offset=offset, **more)
        assert outcome.ending == "ok" and outcome.structured is not None, outcome.text
        results.append(outcome)
        structured = outcome.structured
        offset = structured["unread_from"]
        count = (offset if offset is not None else structured["total"]) - structured["offset"]
        lines += structured["page"].split("\n") if count > 0 else []
    return results, lines


def _flat(structured: dict[str, Any]) -> list[tuple[int, str, str, str]]:
    """A result's verdicts, out of their groups: each as its invariant, path, fault and remedy, in order."""
    return [
        (group["invariant"], fault["path"], fault["fault"], group["remedy"])
        for group in structured["verdicts"]
        for fault in group["faults"]
    ]


def _change(doc: dict[str, Any], where: tuple[str | int, ...], value: Any) -> dict[str, Any]:
    node: Any = doc
    for step in where[:-1]:
        node = node[step]
    node[where[-1]] = value
    return doc


# --- a clean draft ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", LATER_DOCS)
async def test_a_clean_converted_draft_has_no_verdict_and_its_page_is_the_editors_page_text(name: str) -> None:
    """Read page by page, the page is ``doc_page_text`` of the strict build, line for line, and nothing more."""
    fixture = off_loop(_seeded)
    raw = converted_doc(name)
    results, lines = await _read_whole(fixture, name, raw)
    expected = doc_page_text(build(ReportDoc.model_validate(raw), _input(name), strict=True)).splitlines()

    assert lines == expected
    for outcome in results:
        assert outcome.structured is not None
        assert outcome.structured["verdicts"] == [] and outcome.structured["passes"] is True
        assert outcome.structured["total"] == len(expected)
        assert len(outcome.text) <= TEXT_BUDGET, len(outcome.text)


@pytest.mark.parametrize("name", [*fixture_inputs(), KAIRO])
async def test_a_plain_draft_reads_cleanly_on_every_fixture_snapshot(name: str) -> None:
    """The contract walk's draft (no call, every caveat open, one block drawn whole) passes on every snapshot."""
    store = InMemoryDocumentStore()
    fixture = off_loop(ops_fixture, store=store)
    snapshot = fixture_snapshots(_SCOPE)[name]
    off_loop(fixture.host.eval_host.storage.save_report_snapshot, snapshot)
    (action,) = [action for action in analyst_actions() if action.name == "report_check"]
    calls = action_calls(action, frozen_input(snapshot), snapshot.id)
    assert calls, "the control: every fixture draws a block"
    for arguments in calls:
        outcome = await _analyst().call(arguments, host=fixture.host, caller=_CALLER)
        assert outcome.ending == "ok" and outcome.structured is not None, outcome.text
        assert outcome.structured["verdicts"] == [], outcome.text
        assert outcome.structured["page"], outcome.text
    first = await _analyst().call(calls[0], host=fixture.host, caller=_CALLER)
    assert first.structured is not None
    total, unread = first.structured["total"], first.structured["unread_from"]
    assert first.structured["progress"] == (
        f"all {total} lines of the page shown"
        if unread is None
        else f"lines 1 to {unread} of {total} of the page shown; {total - unread} unread: send the same draft again "
        f"with offset={unread}"
    )


# --- the invariants' verdicts ------------------------------------------------------------------------------------


def _unknown_arm(doc: dict[str, Any]) -> dict[str, Any]:
    return _change(doc, ("recommends",), ["no-such-arm"])


def _typed_digit(doc: dict[str, Any]) -> dict[str, Any]:
    return _change(doc, _RECOMMENDATION, doc["decision"][0]["text"] + " It is 12 percent cheaper.")


def _query_naming_nothing(doc: dict[str, Any]) -> dict[str, Any]:
    return _change(doc, ("claims", 0, "evidence", 0, "query", "measures"), ["nope"])


def _caveat_left_out(doc: dict[str, Any]) -> dict[str, Any]:
    caveats = doc["confidence"]["caveats"]
    return _change(doc, ("confidence", "caveats"), [entry for entry in caveats if entry["id"] != "judge_uncalibrated"])


def _plain_high(doc: dict[str, Any]) -> dict[str, Any]:
    return _change(doc, ("confidence", "form"), "high")


def _slot_word(doc: dict[str, Any]) -> dict[str, Any]:
    return _change(doc, _RECOMMENDATION, doc["decision"][0]["text"] + " See why_not.")


@pytest.mark.parametrize(
    ("invariant", "plant", "builds"),
    [
        (0, _unknown_arm, False),
        (1, _typed_digit, True),
        (2, _query_naming_nothing, False),
        (3, _caveat_left_out, True),
        (4, _plain_high, True),
        (5, _slot_word, True),
    ],
    ids=["validity", "a typed digit", "a block that does not draw", "a caveat left out", "a plain high", "a slot word"],
)
async def test_each_invariants_planted_negative_comes_back_as_its_verdict(
    invariant: int, plant: Any, builds: bool
) -> None:
    """The verdict is the one checker's (``doc_verdicts``), whole, ahead of the page; a page that does not build is one
    more verdict, at the field the strict build stopped at, and leaves no page to read."""
    fixture = off_loop(_seeded)
    doc = plant(converted_doc(_KAIRO))
    outcome = await _check(fixture, _KAIRO, doc)
    draft = ReportDoc.model_validate(doc)
    checked = [(v.invariant, v.path, v.fault, v.remedy) for v in doc_verdicts(draft, _input(_KAIRO))]
    checks = [(v.invariant, v.path, v.fault, v.remedy) for v in check_report_doc(draft, _input(_KAIRO))]

    assert outcome.ending == "ok" and outcome.structured is not None, outcome.text
    verdicts = _flat(outcome.structured)
    assert invariant in {verdict[0] for verdict in verdicts}, outcome.text
    assert invariant in {check[0] for check in checks}, (
        "the plant is found by its own check, not only by the page build"
    )
    assert outcome.structured["passes"] is False
    assert outcome.structured["verdicts_total"] == len(verdicts)
    assert outcome.structured["verdicts_unread_from"] is None
    assert sorted(verdicts) == sorted(checked), "every verdict the one checker finds, and no other"
    if builds:
        assert outcome.structured["page"] and outcome.structured["total"] > 0
    else:
        unbuilt = checked[len(checks) :]
        assert checked[: len(checks)] == checks and len(unbuilt) == 1, "a page that does not build is one more verdict"
        assert verdicts[-1] == unbuilt[0], "the page's verdict follows the checks'"
        assert outcome.structured["page"] == "" and outcome.structured["total"] == 0
        assert outcome.structured["unread_from"] is None
        assert outcome.structured["progress"].startswith("no page: the draft does not build")
    assert outcome.text.index("verdicts (") < outcome.text.index("progress: "), "verdicts come first"


async def test_a_draft_with_no_method_item_is_refused_as_the_call_since_it_is_no_document() -> None:
    """Invariant 6 holds a document built without validation; a draft sent to the tool is validated first."""
    fixture = off_loop(_seeded)
    outcome = await _check(fixture, _KAIRO, _change(converted_doc(_KAIRO), ("method",), []))

    assert outcome.ending == "refused" and outcome.is_error
    assert "doc is not a report-doc/1 document: method: List should have at least 1 item" in outcome.text


# --- a draft that does not read ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("doc", "fault"),
    [
        ('{"schema": "report-doc/1", "story": ', "doc is not JSON (Expecting value, line 1 column 37): "),
        ({"schema": "report-doc/1"}, "doc is not a report-doc/1 document: story: Field required; headline: "),
        (
            _change(converted_doc(_KAIRO), ("claims", 0, "evidence"), []),
            "doc is not a report-doc/1 document: claims[0].evidence: List should have at least 1 item",
        ),
    ],
    ids=["not JSON", "not a document", "a field out of bounds"],
)
async def test_a_draft_that_does_not_read_is_refused_naming_the_field_with_a_recovery_that_validates(
    doc: Any, fault: str
) -> None:
    fixture = off_loop(_seeded)
    outcome = await _check(fixture, _KAIRO, doc)

    assert outcome.ending == "refused" and outcome.is_error and outcome.structured is None
    lines = outcome.text.splitlines()
    assert lines[0] == "refused: report_check was called with values it cannot take:"
    assert lines[1].startswith("- doc: Value error, " + fault), lines[1]
    assert "send the draft as one report-doc/1 object (or its JSON text whole)" in lines[1]
    assert outcome.recovery is not None
    recovery = outcome.recovery.as_data()
    assert lines[-1] == "recovery: " + json.dumps(recovery, separators=(",", ":"), ensure_ascii=False)
    assert recovery["parameters"] == ["doc"]
    assert recovery["accepted"] == ["snapshot_id", "doc", "limit", "offset", "verdict_offset"]
    assert recovery["see"] == {"action": "help", "topic": "report_check"}
    example = recovery["example"]
    assert example["action"] == "report_check"
    ReportDoc.model_validate(example["doc"])


async def test_a_draft_sent_as_its_json_text_reads_as_the_object() -> None:
    fixture = off_loop(_seeded)
    raw = converted_doc(_KAIRO)
    as_object = await _check(fixture, _KAIRO, raw, limit=4)
    as_text = await _check(fixture, _KAIRO, json.dumps(raw), limit=4)

    assert (as_text.text, as_text.structured) == (as_object.text, as_object.structured)


# --- paging ------------------------------------------------------------------------------------------------------


async def test_every_result_of_a_few_verdicts_carries_them_all_and_pages_by_offset_never_the_draft_written_back() -> (
    None
):
    fixture = off_loop(_seeded)
    doc = _plain_high(converted_doc(_KAIRO))
    results, lines = await _read_whole(fixture, _KAIRO, doc, limit=5)
    whole = await _check(fixture, _KAIRO, doc, limit=200)
    assert whole.structured is not None

    assert len(results) > 2, "the control: more than two pages"
    assert lines == doc_page_text(build(ReportDoc.model_validate(doc), _input(_KAIRO), strict=True)).splitlines()
    total = whole.structured["total"]
    assert total == len(lines)
    for index, outcome in enumerate(results):
        structured = outcome.structured
        assert structured is not None
        assert structured["verdicts"] == whole.structured["verdicts"] != []
        assert structured["offset"] == index * 5
        assert "next" not in structured and "\nnext: " not in outcome.text
        assert doc["headline"] not in outcome.text, "the draft is never written back"
    assert results[0].structured is not None
    assert results[0].structured["progress"] == (
        f"lines 1 to 5 of {total} of the page shown; {total - 5} unread: send the same draft again with offset=5"
    )
    last = results[-1].structured
    assert last is not None and last["unread_from"] is None and last["progress"].endswith(", the last of them")
    past = await _check(fixture, _KAIRO, doc, offset=total)
    assert past.structured is not None and past.structured["page"] == ""
    assert past.structured["progress"] == f"no line from offset={total}: the page has {total} lines, read from offset=0"


async def test_a_default_result_is_sized_to_one_result_and_its_progress_names_the_next_offset() -> None:
    fixture = off_loop(_seeded)
    first = await _check(fixture, _KAIRO, converted_doc(_KAIRO))
    assert first.structured is not None

    assert len(first.text) <= TEXT_BUDGET, len(first.text)
    unread = first.structured["unread_from"]
    assert unread is not None, "the control: Kairo's page runs past one result"
    assert first.structured["progress"].endswith(f"send the same draft again with offset={unread}")


# --- read-only and idempotent -------------------------------------------------------------------------------------


async def test_the_same_draft_twice_is_byte_identical_and_nothing_is_written() -> None:
    store = _WritesCounted()
    fixture = off_loop(_seeded, store)
    before = store.writes
    for doc in (converted_doc(_KAIRO), _plain_high(converted_doc(_KAIRO)), _unknown_arm(converted_doc(_KAIRO))):
        first = await _check(fixture, _KAIRO, doc)
        second = await _check(fixture, _KAIRO, doc)
        assert (first.text, first.structured) == (second.text, second.structured)

    assert store.writes == before


async def test_a_draft_that_differs_only_in_its_story_reads_the_same() -> None:
    fixture = off_loop(_seeded)
    for doc in (converted_doc(_KAIRO), _plain_high(converted_doc(_KAIRO))):
        told = await _check(fixture, _KAIRO, doc, limit=200)
        other = "decision" if doc["story"] != "decision" else "exploratory"
        retold = await _check(fixture, _KAIRO, _change(dict(doc), ("story",), other), limit=200)
        assert doc["story"] != other, "the control"
        assert (told.text, told.structured) == (retold.text, retold.structured)


# --- the cut -------------------------------------------------------------------------------------------------------


def _digest(texts: list[str]) -> str:
    return sha256("\n\x00\n".join(texts).encode()).hexdigest()


def _served(tool: MountedTool) -> list[str]:
    """Every text a tool serves without a call to an action: its description, input, help, topics and primer."""
    texts = [tool.description, json.dumps(tool.input_schema(), sort_keys=True), tool.help_index()]
    texts += [tool.help_page(action.name) for action in tool.actions]
    for topic in tool.spec.topics:
        texts.append(topic.text())
        texts += list((topic.pages() if topic.pages is not None else {}).values())
    return [*texts, catalogue_primer([tool])]


#: What the ``evals`` tool and the editor's cut of the analyst tool served before ``report_check``, as digested by
#: :func:`_digest` over :func:`_served`. The ``evals`` digest moved once since, at the develop merge that brought in the
#: read bounds (#1000: narrowed-read disclosures in pivot, history and frontier, and an export paged inside a run), and
#: again when a run's summary gained ``cap_checked_on_estimates`` (W4 PRICE), the one field its help page now lists,
#: and again when ``analysis_generate`` moved to the report writer (W4 GEN): its summary and detail, and
#: ``analysis_estimate``'s, describe the report budget, and the estimate gains ``bound`` and ``unchecked``.
#: The editor's cut and the analyst tool moved at #993: the figure-key grammar they teach words its ``usd`` filter for
#: money's one form per family (and its ``usd_per_1k`` filter its ruler moved per 1,000,000 under a cent), and its
#: examples render money in it. All three moved at W4 GAPS: the evals tool gained ``analyst_gaps_list`` (and its
#: ``since``), and both analyst cuts' ``capability_gap`` gained ``kind``, ``figure_keys`` and ``calls`` and says who
#: stores the gap.
_EVALS_BYTES = "415a22621c9c4ce9bd9d57aecd436fdfe298f89532309496f43265cc4ae42f4b"
_EDITOR_BYTES = "886eaa0e5e17613066f2d731cb9e89cad9563e8d8779d51d1f59028cbd11f335"

#: The analyst tool's actions before ``report_check``, in help's order, and the digest of their help pages, the tool's
#: description and its topics.
_ANALYST_ACTIONS_BEFORE = [
    "report_snapshot_freeze",
    "campaign_overview",
    "readings_list",
    "contrasts_list",
    "measure_read",
    "examples_list",
    "caveats_list",
    "figure_keys_check",
    "options_dominance",
    "decision_sensitivity",
    "measure_breakdown",
    "arms_compare",
    "settle_cases",
    "price_tradeoff",
    "levers_effects",
    "block_catalog",
    "block_render",
    "capability_gap",
]
_ANALYST_BYTES = "b992b82f5379e421eea28fb58cd51654c08e31c386a5c4d0a0e83c3c5e1e1f2b"


def test_the_evals_tool_serves_the_bytes_it_served_before() -> None:
    assert _digest(_served(eval_catalogue().mount(standard_tools()[0]))) == _EVALS_BYTES


def test_the_editors_cut_lacks_report_check_and_serves_the_bytes_it_served_before() -> None:
    editor = _editor()
    assert "report_check" not in {action.name for action in editor.actions}
    assert _digest(_served(editor)) == _EDITOR_BYTES


def test_report_check_is_an_action_of_the_analyst_tool_in_the_blocks_workflow_and_the_others_are_unchanged() -> None:
    analyst = _analyst()
    names = [action.name for action in analyst.actions]
    assert names == [*_ANALYST_ACTIONS_BEFORE[:-1], "report_check", _ANALYST_ACTIONS_BEFORE[-1]]
    assert [spec.name for spec in analyst_tools()] == ["evals_analyst"], "an action, not a new tool"
    action = analyst.action("report_check")
    assert action is not None and action.workflow == "Investigate: blocks" and action.permission == "read"
    texts = [analyst.description, *(analyst.help_page(name) for name in _ANALYST_ACTIONS_BEFORE)]
    texts += [topic.text() for topic in analyst.spec.topics]
    assert _digest(texts) == _ANALYST_BYTES


async def test_the_editor_calling_report_check_is_refused_as_an_action_its_cut_goes_without() -> None:
    fixture = off_loop(_seeded)
    arguments = {"action": "report_check", "snapshot_id": snapshot_of(_KAIRO).id, "doc": converted_doc(_KAIRO)}
    outcome = await _editor().call(arguments, host=fixture.host, caller=_CALLER)

    assert outcome.ending == "refused"
    assert outcome.text.startswith("refused: report_check is not on tool evals_analyst: ")


# --- many verdicts -----------------------------------------------------------------------------------------------


def _steps(path: str) -> list[str | int]:
    """A prose field's path as the steps into the document: ``claims[2].text`` is ``claims``, ``2``, ``text``."""
    steps: list[str | int] = []
    for part in path.split("."):
        name, _, rest = part.partition("[")
        steps.append(name)
        steps += [int(index.rstrip("]")) for index in rest.split("[") if index] if rest else []
    return steps


#: The prose fields with a tight length bound, which a planted sentence could push out of shape.
_BOUNDED = frozenset({"headline", "lede", "flag"})


#: What is planted in every prose field: a typed number, and a key naming nothing, which also stops the page building.
_UNBUILT_PLANT = " It is 12 percent off {scoreboard.no-such-arm.reply_cost_usd}."
#: A typed number alone and a longer sentence, so the page still builds and follows the verdicts.
_BUILT_PLANT = " It is 12 percent cheaper, and 3 of 4 tasks agree, by 7 points at the 95th percentile."


def _faulted_everywhere(doc: dict[str, Any], plant: str = _UNBUILT_PLANT) -> dict[str, Any]:
    """The draft with ``plant`` added to every prose field but the bounded ones."""
    for path, _text in report_doc_prose(ReportDoc.model_validate(doc)):
        steps = _steps(path)
        if steps[0] in _BOUNDED:
            continue
        node: Any = doc
        for step in steps[:-1]:
            node = node[step]
        node[steps[-1]] += plant
    return doc


async def test_many_verdicts_stay_within_the_budget_and_every_one_is_reached_by_verdict_offset() -> None:
    """A draft faulted in every prose field: each result fits one result's budget, the verdicts are grouped under their
    remedies, the ones that do not fit are read by ``verdict_offset`` until every one is read, and the page follows
    the last of them."""
    fixture = off_loop(_seeded)
    doc = _faulted_everywhere(converted_doc(_KAIRO))
    whole = check_report_doc(ReportDoc.model_validate(doc), _input(_KAIRO))
    assert len(whole) >= 40, f"the control: {len(whole)} verdicts"
    checked = doc_verdicts(ReportDoc.model_validate(doc), _input(_KAIRO))
    assert checked[:-1] == whole, "the control: the one checker adds the page's refusal to the checks'"

    read: list[tuple[int, str, str, str]] = []
    results: list[dict[str, Any]] = []
    verdict_offset: int | None = 0
    while verdict_offset is not None:
        outcome = await _check(fixture, _KAIRO, doc, verdict_offset=verdict_offset)
        assert outcome.ending == "ok" and outcome.structured is not None, outcome.text
        assert len(outcome.text) <= TEXT_BUDGET, len(outcome.text)
        structured = outcome.structured
        results.append(structured)
        flat = _flat(structured)
        read += flat
        remedies = [(group["invariant"], group["remedy"]) for group in structured["verdicts"]]
        assert len(remedies) == len(set(remedies)), "each remedy is stated once in a result"
        verdict_offset = structured["verdicts_unread_from"]
        if verdict_offset is not None:
            assert structured["page"] == "", "no line of the page while verdicts are unread"
            assert f"unread: send the same draft again with verdict_offset={verdict_offset}" in structured["progress"]
            assert outcome.text.index("verdicts (") < outcome.text.index("progress: ")

    assert len(results) > 1, "the control: the verdicts run past one result"
    # The checks' verdicts, in order, then the one a page that does not build adds, at the field it stopped at: its
    # keys name nothing.
    assert [verdict[:3] for verdict in read[:-1]] == [(v.invariant, v.path, v.fault) for v in whole]
    refused = checked[-1]
    assert read[-1] == (refused.invariant, refused.path, refused.fault, refused.remedy)
    assert all(structured["verdicts_total"] == len(whole) + 1 for structured in results)
    assert results[-1]["progress"].endswith(
        "no page: the draft does not build, so there is none to read (the "
        "last verdict names the field it stopped at); mend the verdicts and "
        "check the draft again"
    )
    typed = [group for group in results[0]["verdicts"] if group["invariant"] == 1]
    assert typed and len(typed[0]["faults"]) > 1, "the control: verdicts share a remedy"


async def test_many_verdicts_on_a_page_that_builds_are_followed_by_the_page() -> None:
    """Typed numbers in every prose field, the page still building: each result fits, every verdict is reached, and
    the page's lines follow the last verdict, read from where the verdicts end."""
    fixture = off_loop(_seeded)
    doc = _faulted_everywhere(converted_doc(_KAIRO), _BUILT_PLANT)
    whole = check_report_doc(ReportDoc.model_validate(doc), _input(_KAIRO))
    assert len(whole) >= 40, f"the control: {len(whole)} verdicts"

    read: list[tuple[int, str, str, str]] = []
    verdict_offset: int | None = 0
    last: dict[str, Any] = {}
    while verdict_offset is not None:
        outcome = await _check(fixture, _KAIRO, doc, verdict_offset=verdict_offset)
        assert outcome.ending == "ok" and outcome.structured is not None, outcome.text
        assert len(outcome.text) <= TEXT_BUDGET, len(outcome.text)
        last = outcome.structured
        read += _flat(last)
        verdict_offset = last["verdicts_unread_from"]

    assert [verdict[:3] for verdict in read] == [(v.invariant, v.path, v.fault) for v in whole]
    expected = doc_page_text(build(ReportDoc.model_validate(doc), _input(_KAIRO), strict=True)).splitlines()
    if not last["page"]:
        assert "the page follows the last verdict" in last["progress"], last["progress"]
    following = await _check(fixture, _KAIRO, doc, verdict_offset=len(whole))
    assert following.structured is not None and len(following.text) <= TEXT_BUDGET
    assert following.structured["verdicts"] == [] and following.structured["page"]
    held = following.structured["unread_from"] or following.structured["total"]
    assert following.structured["page"].split("\n") == expected[:held]
    assert following.structured["progress"].startswith(
        f"verdicts 1 to {len(whole)} of {len(whole)} passed over (verdict_offset={len(whole)})"
    )
