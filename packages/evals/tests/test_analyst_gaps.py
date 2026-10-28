"""The gap loop: every capability gap and code notice stored durably with its provenance, and the digest that lists them.

A report write's sessions call ``capability_gap`` through the write's own mount, which stores nothing; the write stores
each call, and code's two notices (a figure key that did not resolve and sent a draft back, a rubric item scored below
2), when it ends, on every ending. A call through any other mount stores at once. ``analyst_gaps_list`` groups the
stored records by kind and normalized need, ranks them (suspect, then must_fix-linked, then the rest) and writes each
group as issue text. Every turn here is scripted (``report_writer_support``); every tool call runs live on Kairo's
fixture snapshot.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import pytest

from packages.evals.tests.fixtures.report_plans.analyst_cli import seeded_host
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.report_writer_support import (
    DOC,
    MODEL,
    FakeToolTurnClient,
    FakeTurnModel,
    call,
    kairo_id,
    reply,
    review,
    submitted,
    written,
)
from threetears.evals.actions import MountedTool, analyst_tools, eval_catalogue, standard_tools
from threetears.evals.actions.report_writing import ReportNotStored
from threetears.evals.analysis.analyst_gaps import gap_digest
from threetears.evals.kernel import StorageError
from threetears.evals.kernel.analyst_gap import EvalAnalystGap, GapToolCall
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.analyst import AnalystLimits
from threetears.evals.ops.report_writer.loop import WriterLimits
from threetears.evals.schema.doc_types import DOC_TYPE_FACTS

#: A finding that sends a report back however it scored, about claim c2.
MUST_FIX: Final = {"claim": "c2", "severity": "must_fix", "finding": "Say what clearing the margin would buy."}

#: A missing capability, as the analyst notes one.
MISSING: Final = {
    "need": "the cases needed to separate two arms' readings",
    "why": "to say what would settle the asked question",
    "workaround": "named the case count from the interval widths, in words",
}

#: A suspect result, with what reproduces it.
SUSPECT: Final = {
    "kind": "suspect",
    "need": "the cost per reply of qwen3.8-flash reads lower than its trials' own costs",
    "why": "its trials each cost more than the mean the key reads",
    "workaround": "left the cost claim at the arms' order and named no figure",
    "claim_ids": ["c2"],
    "figure_keys": ["{cost_per_1k.qwen3.8-flash}"],
    "calls": [{"action": "measure_read", "measure_key": "cost", "by": "arm"}],
}

#: A key no snapshot resolves.
BAD_KEY: Final = "{cost.uplift_nowhere}"


def _analyst(*replies: Any) -> FakeToolTurnClient:
    """The analyst: one gap of each kind, then each scripted reply (by default, the Kairo document)."""
    snap = kairo_id()
    return FakeToolTurnClient(
        reply(call(1, "capability_gap", snap, **MISSING), call(2, "capability_gap", snap, **SUSPECT)),
        *(replies or (reply(submitted()),)),
    )


def _editor(*replies: Any) -> FakeToolTurnClient:
    return FakeToolTurnClient(*replies)


def _gaps(host: OpsHost, *, since: datetime | None = None) -> list[EvalAnalystGap]:
    return off_loop(lambda: host.eval_host.storage.list_analyst_gaps(CALLER.scope_id, since=since))


def _by_kind(gaps: list[EvalAnalystGap]) -> dict[str, list[EvalAnalystGap]]:
    found: dict[str, list[EvalAnalystGap]] = {}
    for gap in gaps:
        found.setdefault(gap.kind, []).append(gap)
    return found


def _with_bad_key() -> dict[str, Any]:
    doc = copy.deepcopy(DOC)
    assert "{cost.uplift_range}" in doc["headline"], "the control: the headline cites the key the test breaks"
    doc["headline"] = doc["headline"].replace("{cost.uplift_range}", BAD_KEY)
    return doc


# --- both kinds, stored with their provenance ---------------------------------------------------------------------


async def test_a_write_stores_each_gap_call_once_with_its_snapshot_report_role_model_and_transcript() -> None:
    host = off_loop(seeded_host)
    model = FakeTurnModel(analyst=_analyst(), editor_1=_editor(reply(review(scores=(2, 2, 2, 2, 2, 2, 2)))))

    result = await written(model, host=host)

    assert result.ending == "stored" and result.stored is not None, result.outcome.reason
    stored = _gaps(host)
    assert [gap.kind for gap in stored] == ["missing", "suspect"], "each call is stored once, never twice"
    transcript = result.generation["tally"]["transcripts"]["analyst"]
    for gap in stored:
        assert (gap.snapshot_id, gap.report_doc_id) == (kairo_id(), result.stored.id)
        assert (gap.role, gap.session, gap.model, gap.transcript_digest) == ("analyst", "analyst", MODEL, transcript)
        assert gap.created_by == CALLER.identity
    missing, suspect = stored
    assert (missing.need, missing.why, missing.workaround) == (MISSING["need"], MISSING["why"], MISSING["workaround"])
    assert suspect.figure_keys == SUSPECT["figure_keys"] and suspect.claim_ids == ["c2"]
    assert suspect.calls == [GapToolCall(action="measure_read", arguments={"measure_key": "cost", "by": "arm"})]
    assert list(result.gaps) == stored and result.gaps_not_stored is None


async def test_an_in_write_call_is_acknowledged_as_kept_by_the_write_under_the_id_it_is_stored_under() -> None:
    host = off_loop(seeded_host)
    analyst = _analyst()
    model = FakeTurnModel(analyst=analyst, editor_1=_editor(reply(review(scores=(2, 2, 2, 2, 2, 2, 2)))))

    await written(model, host=host)

    answered = [block for block in analyst.requests[1]["messages"][-1]["content"] if block["type"] == "tool_result"]
    assert [block.get("is_error", False) for block in answered] == [False, False]
    assert all("recorded_in: write" in block["content"] for block in answered), answered
    assert all(gap.id in block["content"] for gap, block in zip(_gaps(host), answered, strict=True))


async def test_a_write_that_stores_no_report_still_stores_its_gaps_naming_no_report() -> None:
    host = off_loop(seeded_host)
    broken = copy.deepcopy(DOC)
    broken["method"] = []
    model = FakeTurnModel(analyst=_analyst(reply(submitted(broken))))
    limits = WriterLimits(analyst=AnalystLimits(bounces=0))

    result = await written(model, host=host, limits=limits)

    assert result.ending == "refused" and result.stored is None
    stored = _gaps(host)
    assert [gap.kind for gap in stored] == ["missing", "suspect"]
    assert all(gap.report_doc_id is None and gap.session == "analyst" for gap in stored)
    assert list(result.gaps) == stored


async def test_a_store_that_refuses_the_report_still_stores_its_gaps_naming_no_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = off_loop(seeded_host)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the store is full")

    monkeypatch.setattr(host.eval_host.storage, "save_report_doc", refuse)
    model = FakeTurnModel(analyst=_analyst(), editor_1=_editor(reply(review(scores=(2, 2, 2, 2, 2, 2, 2)))))

    with pytest.raises(ReportNotStored, match="with its gaps, stored naming no report") as refused:
        await written(model, host=host)

    assert [gap.report_doc_id for gap in _gaps(host)] == [None, None]
    assert [gap.kind for gap in refused.value.written.gaps] == ["missing", "suspect"]


async def test_a_store_that_refuses_the_gaps_fails_neither_the_report_nor_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = off_loop(seeded_host)

    def refuse(gap: EvalAnalystGap) -> bool:
        raise StorageError("the gaps table is read-only")

    monkeypatch.setattr(host.eval_host.storage, "save_analyst_gap", refuse)
    model = FakeTurnModel(analyst=_analyst(), editor_1=_editor(reply(review(scores=(2, 2, 2, 2, 2, 2, 2)))))

    result = await written(model, host=host)

    assert result.ending == "stored" and result.stored is not None
    assert result.gaps_not_stored is not None and "the gaps table is read-only" in result.gaps_not_stored
    assert [gap.kind for gap in result.gaps] == ["missing", "suspect"], "the records ride on the result"


async def test_a_replayed_write_stores_nothing_twice() -> None:
    host = off_loop(seeded_host)
    for _ in range(2):
        model = FakeTurnModel(analyst=_analyst(), editor_1=_editor(reply(review(scores=(2, 2, 2, 2, 2, 2, 2)))))
        await written(model, host=host)

    assert [gap.kind for gap in _gaps(host)] == ["missing", "suspect"]


async def test_a_call_outside_a_write_stores_at_call_time_naming_its_caller_and_no_report() -> None:
    host = off_loop(seeded_host)
    (spec,) = analyst_tools()
    tool: MountedTool = eval_catalogue().mount(spec)

    outcome = await tool.call(
        {"action": "capability_gap", "snapshot_id": kairo_id(), **SUSPECT}, host=host, caller=CALLER
    )

    assert outcome.refusal is None, outcome.text
    assert "recorded_in: store" in outcome.text
    (gap,) = _gaps(host)
    assert gap.kind == "suspect" and gap.id in outcome.text
    assert (gap.snapshot_id, gap.created_by) == (kairo_id(), CALLER.identity)
    assert (gap.report_doc_id, gap.role, gap.session, gap.model, gap.transcript_digest) == (None,) * 5


async def test_a_suspect_gap_that_names_nothing_to_reproduce_it_is_refused() -> None:
    host = off_loop(seeded_host)
    (spec,) = analyst_tools()
    tool = eval_catalogue().mount(spec)
    bare = {key: value for key, value in SUSPECT.items() if key not in {"figure_keys", "calls"}}

    outcome = await tool.call({"action": "capability_gap", "snapshot_id": kairo_id(), **bare}, host=host, caller=CALLER)

    assert outcome.refusal is not None and "names what reproduces it" in outcome.text
    assert _gaps(host) == []


async def test_a_missing_gap_keeps_the_id_it_had_before_kinds_and_a_suspect_one_of_the_same_words_is_another() -> None:
    """A missing call's id is the sha256 of its snapshot, need and why, as capability_gap always named it."""
    host = off_loop(seeded_host)
    (spec,) = analyst_tools()
    tool = eval_catalogue().mount(spec)
    asked = {"action": "capability_gap", "snapshot_id": kairo_id(), **MISSING}

    await tool.call(asked, host=host, caller=CALLER)
    await tool.call({**asked, "kind": "suspect", "figure_keys": ["{k}"]}, host=host, caller=CALLER)

    before = "gap:" + hashlib.sha256(f"{kairo_id()}\n{MISSING['need']}\n{MISSING['why']}".encode()).hexdigest()[:16]
    missing, suspect = sorted(_gaps(host), key=lambda gap: gap.kind)
    assert missing.id == before and suspect.id != before


# --- code's notices ------------------------------------------------------------------------------------------------


async def test_an_unresolved_key_that_bounced_a_draft_and_a_low_score_are_noticed() -> None:
    host = off_loop(seeded_host)
    model = FakeTurnModel(
        analyst=FakeToolTurnClient(reply(submitted(_with_bad_key())), reply(submitted()), reply(submitted())),
        editor_1=_editor(reply(review(findings=[MUST_FIX]))),
        editor_2=_editor(reply(review(scores=(2, 2, 2, 2, 2, 2, 2)))),
    )

    result = await written(model, host=host)

    assert result.ending == "stored", result.outcome.reason
    kinds = _by_kind(_gaps(host))
    (key,) = kinds["unresolved_key"]
    assert key.figure_keys == [BAD_KEY] and key.bounce is not None
    assert (key.bounce.submission, key.bounce.path, key.bounce.key) == (1, "headline", BAD_KEY)
    assert key.bounce.verdicts and all(verdict.path == "headline" for verdict in key.bounce.verdicts)
    assert BAD_KEY in key.need and key.workaround is None
    assert (key.session, key.role, key.report_doc_id) == ("analyst", "analyst", result.stored.id)  # type: ignore[union-attr]
    (low,) = kinds["editor_score"]
    assert low.editor_item is not None
    assert (low.editor_item.round, low.editor_item.n, low.editor_item.score) == (1, 6, 1)
    assert low.editor_item.reason == "item 6 reads so" and low.editor_item.must_fix and low.must_fix
    assert (low.session, low.role) == ("editor-1", "editor")
    assert low.transcript_digest == result.generation["tally"]["transcripts"].get("editor-1")


async def test_a_gap_touching_a_claim_a_must_fix_finding_names_is_linked_to_it() -> None:
    host = off_loop(seeded_host)
    model = FakeTurnModel(
        analyst=_analyst(reply(submitted()), reply(submitted())),
        editor_1=_editor(reply(review(scores=(2, 2, 2, 2, 2, 2, 2), findings=[MUST_FIX]))),
        editor_2=_editor(reply(review(scores=(2, 2, 2, 2, 2, 2, 2)))),
    )

    await written(model, host=host)

    kinds = _by_kind(_gaps(host))
    assert [gap.must_fix for gap in kinds["suspect"]] == [True], "it touches c2, which the must_fix names"
    assert [gap.must_fix for gap in kinds["missing"]] == [False]


# --- durable ------------------------------------------------------------------------------------------------------


async def test_drop_derived_leaves_every_gap_in_place() -> None:
    host = off_loop(seeded_host)
    model = FakeTurnModel(analyst=_analyst(), editor_1=_editor(reply(review())))
    await written(model, host=host)
    before = _gaps(host)

    dropped = off_loop(host.eval_host.storage.drop_derived, [CALLER.scope_id])

    assert "eval_analyst_gap" not in dropped.counts and dropped.counts["eval_report_doc"] == 1
    assert _gaps(host) == before and len(before) == 3
    assert (DOC_TYPE_FACTS["eval_analyst_gap"].origin, DOC_TYPE_FACTS["eval_analyst_gap"].lifecycle) == (
        "record",
        "durable",
    )


# --- the digest ---------------------------------------------------------------------------------------------------


def _record(n: int, need: str, *, kind: str = "missing", must_fix: bool = False, at: int = 0, **fields: Any) -> EvalAnalystGap:
    return EvalAnalystGap(
        id=f"gap:{n:016d}",
        scope_id=CALLER.scope_id,
        kind=kind,
        need=need,
        why=fields.pop("why", "to say what would settle it"),
        workaround=fields.pop("workaround", "said it in words"),
        must_fix=must_fix,
        snapshot_id=fields.pop("snapshot_id", "report-snapshot:aaaa1111"),
        created_by=CALLER.identity,
        created_at=(datetime(2026, 10, 28, tzinfo=UTC) + timedelta(hours=at)).isoformat(),
        **fields,
    )


def test_needs_that_differ_only_in_case_spacing_punctuation_ids_and_keys_group_and_other_words_stay_apart() -> None:
    records = [
        _record(1, "The cases needed to separate `arm.a` from arm.b", at=0),
        _record(2, "the  CASES needed to separate {compare.a.b.cost} from x.y!", at=1, snapshot_id="report-snapshot:bbbb2222"),
        _record(3, "the cases needed to separate report-snapshot:0193ab12 from arm.c", at=2),
        _record(4, "the trials needed to separate arm.a from arm.b", at=3),
        _record(5, "the cases needed to separate arm.a from arm.b", kind="suspect", figure_keys=["{k}"], at=4),
    ]

    digest = gap_digest(records)

    groups = {(group.kind, group.normalized): group for group in digest.groups}
    near = groups[("missing", "the cases needed to separate _key_ from _key_")]
    assert near.count == 2 and near.gap_ids == ["gap:0000000000000001", "gap:0000000000000002"]
    assert near.need == "The cases needed to separate `arm.a` from arm.b"
    assert near.snapshots == ["report-snapshot:aaaa1111", "report-snapshot:bbbb2222"]
    assert (near.first_seen, near.last_seen) == (records[0].created_at, records[1].created_at)
    assert ("missing", "the cases needed to separate _id_ from _key_") in groups
    assert ("missing", "the trials needed to separate _key_ from _key_") in groups, "other words stay apart"
    assert ("suspect", "the cases needed to separate _key_ from _key_") in groups, "another kind stays apart"
    assert len(digest.groups) == 4 and digest.records == 5


def test_suspect_groups_rank_first_then_must_fix_linked_then_the_rest_by_count_then_last_seen() -> None:
    records = [
        _record(1, "a capability", at=0),
        _record(2, "a capability", at=1),
        _record(3, "a capability", at=2),
        _record(4, "another capability", at=3),
        _record(5, "a third capability", at=4),
        _record(6, "a linked capability", must_fix=True, at=5),
        _record(7, "a wrong result", kind="suspect", figure_keys=["{k}"], at=6),
    ]

    ranked = [group.need for group in gap_digest(records).groups]

    assert ranked == ["a wrong result", "a linked capability", "a capability", "a third capability", "another capability"]


def test_a_suspect_groups_issue_text_carries_its_reproducing_keys_and_calls() -> None:
    record = _record(
        1,
        "the cost per reply reads low",
        kind="suspect",
        figure_keys=["{cost_per_1k.qwen3.8-flash}"],
        calls=[GapToolCall(action="measure_read", arguments={"measure_key": "cost", "by": "arm"})],
        report_doc_id="01a1-report",
        session="analyst",
        role="analyst",
        model=MODEL,
        transcript_digest="9" * 64,
    )

    (group,) = gap_digest([record]).groups

    assert group.title == "evals analyst gap (suspect): the cost per reply reads low"
    assert "`{cost_per_1k.qwen3.8-flash}`" in group.body
    assert 'action `measure_read` with `{"by": "arm", "measure_key": "cost"}`' in group.body
    assert "report 01a1-report" in group.body and "session analyst (analyst)" in group.body
    assert f"model {MODEL}" in group.body and "9" * 64 in group.body
    assert "**Workarounds:**\n- said it in words" in group.body


async def test_the_digest_action_reads_the_store_since_a_time_by_each_records_own_time() -> None:
    host = off_loop(seeded_host)
    for record in (_record(1, "an early need", at=0), _record(2, "a later need", at=5), _record(3, "an early need", at=6)):
        off_loop(host.eval_host.storage.save_analyst_gap, record)
    tool = eval_catalogue().mount(_evals_spec())

    every = await tool.call({"action": "analyst_gaps_list"}, host=host, caller=CALLER)
    since = await tool.call(
        {"action": "analyst_gaps_list", "since": "2026-10-28T04:00:00+00:00"}, host=host, caller=CALLER
    )

    assert every.refusal is None and since.refusal is None, (every.text, since.text)
    assert "3 records in 2 groups" in every.text
    assert "since 2026-10-28T04:00:00+00:00): 2 records in 2 groups" in since.text
    assert "Noted once" in since.text and "Noted 2 times" not in since.text, "a group shows only its records since then"
    assert [gap.id for gap in _gaps(host, since=datetime(2026, 10, 28, 4, tzinfo=UTC))] == [
        "gap:0000000000000002",
        "gap:0000000000000003",
    ]


async def test_the_digest_refuses_a_time_with_no_zone() -> None:
    host = off_loop(seeded_host)
    tool = eval_catalogue().mount(_evals_spec())

    outcome = await tool.call({"action": "analyst_gaps_list", "since": "2026-10-28T04:00:00"}, host=host, caller=CALLER)

    assert outcome.refusal is not None


def _evals_spec() -> Any:
    return standard_tools()[0]


# --- the editor's cut ---------------------------------------------------------------------------------------------


def test_the_editors_cut_carries_the_gap_note_with_its_kind() -> None:
    (spec,) = analyst_tools(role="editor")
    editor = eval_catalogue().mount(spec)

    assert "capability_gap" in editor.input_schema()["properties"]["action"]["enum"]
    assert {"kind", "figure_keys", "calls"} <= set(editor.input_schema()["properties"])
