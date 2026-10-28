"""A report's sessions in transcript storage (``eval_report_session``): stored beside their report, named by ref.

Contracts held here:

- **round trip**: a stored report's sessions load from storage, all of them or one by name, each exactly as its
  session's ``to_json()`` gave it;
- **the record holds refs and digests only**: the stored ``generation`` names each session by ``{digest, turns, ref}``
  and holds no message, and the Kairo record falls from 253,272 bytes to under the plan's 75 KB;
- **a paid run never loses its sessions**: an ending that stores nothing, and a store that refuses, keep each session
  inline on the in-memory record;
- **resume through storage is byte-identical**: an analyst session and an editor session reloaded from storage and
  resumed send the request body an uninterrupted session sends, byte for byte, on a store that hands keys back in
  another order too (jsonb's, or reversed), with a tool call's arguments and every thinking block's signature intact;
- **W3's elision is kept**: a re-derivable result body is stored elided and recorded so, and resuming re-derives it;
- **a listing reads no session**: ``list_report_docs`` asks the store for report documents only;
- **the sweep**: ``drop_derived`` and ``delete_report_doc`` remove the sessions with their report;
- **the bump**: a report document or session written under another report version is refused by its version alone,
  with the regenerate wording, never field by field.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pytest
from pydantic import ValidationError

from packages.evals.tests.factories import memory_storage
from packages.evals.tests.fixtures.report_plans.analyst_cli import seeded_host
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, ops_fixture
from packages.evals.tests.report_doc_support import snapshot_of
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
    thinking,
    written,
)
from threetears.evals.actions import MountedTool, analyst_tools, eval_catalogue
from threetears.evals.actions.catalogue import Caller, catalogue_primer
from threetears.evals.actions.report_writing import ReportNotStored
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.report_decision import every_caveat
from threetears.evals.analysis.report_doc_store import doc_verdicts, generation_record, store_report_doc
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
from threetears.evals.kernel import storage as storage_module
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.analyst import AnalystSession
from threetears.evals.ops.report_writer.loop import WriterLimits
from threetears.evals.ops.report_writer.session import ToolOutcome, ToolSession
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.schema.doc_types import DOC_TYPE_FACTS
from threetears.evals.schema.store_port import DocumentStore
from threetears.evals.schema.turns import TurnText, TurnThinking, TurnToolUse
from threetears.evals.schema.versioning import REPORT_DOC_SCHEMA_VERSION
from threetears.evals.storage import InMemoryDocumentStore

#: The Kairo record of the fourth prototype, as the writing loop stored it under report version 13.
KAIRO_RECORD: Final = Path(__file__).parent / "fixtures" / "report_docs" / "prototype4" / "kairo" / "record.json"

#: What the editor is asked after its review, in a resumed session.
QUESTION: Final = "Why did item 6 score 1?"

#: What the analyst is asked to revise by, in both the uninterrupted and the resumed session.
REVISION: Final = "The report editor found claim c1's evidence thin. Revise the report."

#: Scores that fail the review: one item at 0.
FAILING: Final = (2, 2, 0, 2, 2, 2, 2)


@dataclass(frozen=True)
class HostTool:
    """A catalogue tool bound to a host and caller and described with the catalogue's primer, as ``write_report``
    mounts the analyst's and the editor's; it counts each call, so a test sees each re-derived body asked for."""

    tool: MountedTool
    host: OpsHost
    caller: Caller
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def description(self) -> str:
        return f"{self.tool.description}\n\n{catalogue_primer([self.tool])}"

    @property
    def rederivable(self) -> bool:
        return self.tool.hints.read_only

    def input_schema(self) -> dict[str, Any]:
        return self.tool.input_schema()

    async def call(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        self.calls.append(dict(arguments))
        return await self.tool.call(arguments, host=self.host, caller=self.caller)


def mounted(host: OpsHost, role: str) -> HostTool:
    (spec,) = analyst_tools(writes=False, role="editor" if role == "editor" else "analyst")
    return HostTool(eval_catalogue().mount(spec), host, CALLER)


def analyst_script(*more: Any) -> list[Any]:
    """The analyst's draft: a thought and a tool call, then the Kairo document; then each reply after it."""
    return [
        reply(thinking("reading the dominance"), call(1, "options_dominance", kairo_id())),
        reply(submitted()),
        *more,
    ]


def editor_script() -> list[Any]:
    """A review round: a tool call, then a review that clears."""
    return [reply(call(9, "options_dominance", kairo_id())), reply(review())]


async def stored_run() -> tuple[OpsHost, Any, FakeToolTurnClient, FakeToolTurnClient]:
    """A report written and stored in round 1, on a fresh seeded host: the host, the result, and both clients."""
    host = off_loop(seeded_host)
    analyst, editor = FakeToolTurnClient(*analyst_script()), FakeToolTurnClient(*editor_script())
    result = await written(FakeTurnModel(analyst=analyst, editor_1=editor), host=host)
    assert result.ending == "stored", result.outcome.reason
    return host, result, analyst, editor


def host_over(store: DocumentStore) -> OpsHost:
    """The toy host over ``store``, holding Kairo round 2's snapshot as the seeded host does. Blocking."""
    host = ops_fixture(generation=False, store=store).host
    snapshot = seeded_host().eval_host.storage.load_report_snapshot(kairo_id(), CALLER.scope_id)
    assert snapshot is not None
    host.eval_host.storage.save_report_snapshot(snapshot)
    return host


def sessions_of(host: OpsHost, report_doc_id: str, **name: str) -> dict[str, EvalReportSession]:
    return off_loop(host.eval_host.storage.load_report_sessions, report_doc_id, CALLER.scope_id, **name)


def kairo_input(host: OpsHost) -> ReportInput:
    snapshot = off_loop(host.eval_host.storage.load_report_snapshot, kairo_id(), CALLER.scope_id)
    assert snapshot is not None
    return ReportInput.model_validate(snapshot.report_input)


def resume_options(host: OpsHost, mount: HostTool) -> dict[str, Any]:
    """What resuming the analyst takes beside its record and client: what the writing loop started it with."""
    report_input = kairo_input(host)
    return {
        "model": MODEL,
        "mount": mount,
        "skill": load_skill().analyst(),
        "arms": [arm.id for arm in report_input.arms],
        "caveats": [caveat.id for caveat in every_caveat(report_input)],
        "check": lambda doc: doc_verdicts(doc, report_input),
        "tally": Tally(),
        "budget": ReportBudget(usd=100),
    }


# --- round trip ------------------------------------------------------------------------------------------------------


def test_a_report_session_is_derived_and_regenerable() -> None:
    facts = DOC_TYPE_FACTS["eval_report_session"]
    assert (facts.origin, facts.lifecycle) == ("derived", "regenerable")


async def test_a_stored_report_s_sessions_load_from_storage_each_exactly_as_its_to_json_gave_it() -> None:
    host, result, _, _ = await stored_run()

    loaded = sessions_of(host, result.stored.id)

    assert list(loaded) == ["analyst", "editor-1"]
    for name, session in loaded.items():
        assert session.session == json.loads(json.dumps(result.outcome.sessions[name]))
        assert (session.id, session.report_doc_id, session.scope_id) == (
            report_session_id(result.stored.id, name),
            result.stored.id,
            CALLER.scope_id,
        )
    assert sessions_of(host, result.stored.id, name="editor-1") == {"editor-1": loaded["editor-1"]}
    assert sessions_of(host, result.stored.id, name="editor-2") == {}
    assert sessions_of(host, "no-such-report") == {}


async def test_the_stored_record_names_each_session_by_digest_turns_and_ref_and_holds_no_message() -> None:
    host, result, analyst, editor = await stored_run()
    stored = result.stored
    tally = stored.generation["tally"]

    assert stored.generation["format"] == 2
    assert stored.generation["sessions"] == {
        "analyst": {
            "digest": tally["transcripts"]["analyst"],
            "turns": len(analyst.requests),
            "ref": report_session_id(stored.id, "analyst"),
        },
        "editor-1": {
            "digest": tally["transcripts"]["editor-1"],
            "turns": len(editor.requests),
            "ref": report_session_id(stored.id, "editor-1"),
        },
    }
    held = json.dumps(stored.to_dict())
    assert '"messages"' not in held and "reading the dominance" not in held, "the record holds no message"
    for name, session in sessions_of(host, stored.id).items():
        entry = stored.generation["sessions"][name]
        assert (session.digest, session.turns) == (entry["digest"], entry["turns"])
    (listed,) = off_loop(host.eval_host.storage.list_report_docs, CALLER.scope_id)
    assert listed.generation == stored.generation


def test_the_kairo_record_falls_from_253_kb_to_under_75_kb() -> None:
    """The fourth prototype's Kairo record, its sessions moved to their own documents: 253,272 bytes to 38,120."""
    before = KAIRO_RECORD.read_text()
    record = json.loads(before)
    old = record["generation"]
    generation = generation_record(
        report_doc_id=record["id"],
        model=old["model"],
        tally=old["tally"],
        skill=old["skill"],
        reviews=old["reviews"],
        sessions=old["sessions"],
    )
    after = json.dumps({**record, "generation": generation}, indent=2, ensure_ascii=False) + "\n"

    assert len(before.encode()) == 253_272
    assert len(after.encode()) < 75_000
    assert len(json.dumps(old["sessions"]).encode()) > 180_000, "the sessions were most of the record"


# --- a paid run never loses its sessions --------------------------------------------------------------------------------


async def test_an_ending_that_stores_nothing_keeps_every_session_inline_in_memory() -> None:
    analyst = FakeToolTurnClient(*analyst_script())
    editor = FakeToolTurnClient(reply(TurnText("I will not review this."), stop="content_filter"))
    host = off_loop(seeded_host)

    result = await written(FakeTurnModel(analyst=analyst, editor_1=editor), host=host)

    assert result.ending == "failed" and result.stored is None
    for name, entry in result.generation["sessions"].items():
        assert entry["ref"] is None and entry["session"] == result.outcome.sessions[name]
    assert set(result.generation["sessions"]) == {"analyst", "editor-1"}
    assert off_loop(host.eval_host.storage.list_report_docs, CALLER.scope_id) == []


async def test_a_store_that_refuses_the_document_keeps_every_session_inline_on_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = off_loop(seeded_host)

    def refuse(report: EvalReportDoc, *, sessions: Sequence[EvalReportSession] = ()) -> None:
        raise StorageError("the document is larger than the store takes")

    monkeypatch.setattr(host.eval_host.storage, "save_report_doc", refuse)
    model = FakeTurnModel(analyst=FakeToolTurnClient(*analyst_script()), editor_1=FakeToolTurnClient(*editor_script()))

    with pytest.raises(ReportNotStored) as refused:
        await written(model, host=host)

    kept = refused.value.written
    assert {name: entry["session"] for name, entry in kept.generation["sessions"].items()} == kept.outcome.sessions
    assert all(entry["ref"] is None for entry in kept.generation["sessions"].values())


async def test_a_refused_store_names_the_minted_id_so_its_orphaned_sessions_are_deleted_by_it_alone() -> None:
    """The store wrote the sessions and refused the report: the refusal names the id they are under, and deleting
    that report removes them and nothing else in the scope."""
    host = off_loop(host_over, _RefusesReportDocs())
    model = FakeTurnModel(analyst=FakeToolTurnClient(*analyst_script()), editor_1=FakeToolTurnClient(*editor_script()))

    with pytest.raises(ReportNotStored) as refused:
        await written(model, host=host)

    report_doc_id = refused.value.report_doc_id
    assert report_doc_id in str(refused.value)
    assert set(sessions_of(host, report_doc_id)) == {"analyst", "editor-1"}, "the sessions were written first"
    storage = host.eval_host.storage
    assert off_loop(storage.delete_report_doc, report_doc_id, CALLER.scope_id) is False, "no report was stored"
    assert sessions_of(host, report_doc_id) == {}
    assert off_loop(storage.load_report_snapshot, kairo_id(), CALLER.scope_id) is not None, "nothing else is touched"


# --- the store gate's order -----------------------------------------------------------------------------------------


class _RefusesReportDocs(InMemoryDocumentStore):
    """The reference store, refusing every write of a report document; every other write lands."""

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        if document.get("doc_type") == "eval_report_doc":
            raise RuntimeError("the store is full")
        super().upsert(document, if_match=if_match, create_only=create_only)


def _gate(storage: EvalStorage, sessions: Mapping[str, Mapping[str, Any]], report_doc_id: str) -> EvalReportDoc:
    """The store gate over Kairo's snapshot, with the sessions and the record naming them."""
    snapshot = snapshot_of("prototype3/kairo")
    tally: dict[str, Any] = {
        "turns": [{"session": name, "model": "m"} for name in sessions],
        "total": {"cost_usd": None, "unpriced_turns": len(sessions)},
        "transcripts": {name: f"{name}-digest" for name in sessions},
    }
    return store_report_doc(
        storage,
        report_doc_id=report_doc_id,
        scope_id=CALLER.scope_id,
        snapshot_id=snapshot.id,
        campaign_id=snapshot.campaign_id,
        doc=ReportDoc.model_validate(DOC),
        report_input=ReportInput.model_validate(snapshot.report_input),
        ending="stored",
        findings=[],
        generation=generation_record(
            report_doc_id=report_doc_id,
            model="m",
            tally=tally,
            skill={"package": "f" * 64},
            reviews=[],
            sessions=sessions,
        ),
        sessions=sessions,
        created_by=CALLER.identity,
    )


SESSIONS: Final = {"analyst": {"format": 2, "said": "the analyst's"}, "editor-1": {"format": 2, "said": "a review"}}


def test_the_store_gate_writes_the_sessions_then_the_document_so_a_failed_write_leaves_only_unnamed_sessions() -> None:
    """A stored document never names a session that was not written; the sessions a failure leaves are swept."""
    storage = EvalStorage(_RefusesReportDocs())
    report_doc_id = new_report_doc_id()

    with pytest.raises(StorageError):
        _gate(storage, SESSIONS, report_doc_id)

    assert storage.list_report_docs(CALLER.scope_id) == []
    assert set(storage.load_report_sessions(report_doc_id, CALLER.scope_id)) == {"analyst", "editor-1"}
    assert storage.drop_derived([CALLER.scope_id]).counts["eval_report_session"] == 2


def test_the_store_gate_refuses_a_record_that_names_other_sessions_than_it_is_handed_and_stores_nothing() -> None:
    storage, _ = memory_storage()

    with pytest.raises(ValueError, match="its generation record names the sessions"):
        store_report_doc(
            storage,
            report_doc_id=new_report_doc_id(),
            scope_id=CALLER.scope_id,
            snapshot_id=snapshot_of("prototype3/kairo").id,
            campaign_id=None,
            doc=ReportDoc.model_validate(DOC),
            report_input=ReportInput.model_validate(snapshot_of("prototype3/kairo").report_input),
            ending="stored",
            findings=[ReportFinding(claim="", severity="should_fix", finding="Tighten the lede.")],
            generation=generation_record(
                report_doc_id="another-report",
                model="m",
                tally={"turns": [], "total": {"cost_usd": None, "unpriced_turns": 0}},
                skill={"package": "f" * 64},
                reviews=[],
                sessions=SESSIONS,
            ),
            sessions=SESSIONS,
            created_by=CALLER.identity,
        )

    assert storage.list_report_docs(CALLER.scope_id) == []
    assert storage.drop_derived([CALLER.scope_id]).counts["eval_report_session"] == 0


def test_a_session_that_kept_no_reply_goes_through_the_gate_with_no_digest() -> None:
    """A session the budget stopped before its first reply has no transcript digest: None on the record and the session.

    Its turns are counted from the tally all the same, so a session refused before it sent counts 0.
    """
    storage, _ = memory_storage()
    snapshot = snapshot_of("prototype3/kairo")
    report_doc_id = new_report_doc_id()
    tally: dict[str, Any] = {
        "turns": [{"session": "analyst", "model": "m"}, {"session": "analyst", "model": "m"}],
        "total": {"cost_usd": None, "unpriced_turns": 2},
        "transcripts": {"analyst": "a" * 64},
    }
    generation = generation_record(
        report_doc_id=report_doc_id, model="m", tally=tally, skill={"package": "f" * 64}, reviews=[], sessions=SESSIONS
    )

    record = store_report_doc(
        storage,
        report_doc_id=report_doc_id,
        scope_id=CALLER.scope_id,
        snapshot_id=snapshot.id,
        campaign_id=snapshot.campaign_id,
        doc=ReportDoc.model_validate(DOC),
        report_input=ReportInput.model_validate(snapshot.report_input),
        ending="over_budget",
        findings=[],
        generation=generation,
        sessions=SESSIONS,
        created_by=CALLER.identity,
    )

    entries = record.generation["sessions"]
    assert entries["editor-1"] == {"digest": None, "turns": 0, "ref": report_session_id(report_doc_id, "editor-1")}
    assert entries["analyst"]["digest"] == "a" * 64 and entries["analyst"]["turns"] == 2
    loaded = storage.load_report_sessions(report_doc_id, CALLER.scope_id)
    assert (loaded["editor-1"].digest, loaded["editor-1"].turns) == (None, 0)
    assert loaded["editor-1"].session == SESSIONS["editor-1"]


def test_a_session_of_another_report_is_never_stored_with_this_one() -> None:
    storage, _ = memory_storage()
    record = _gate(storage, SESSIONS, new_report_doc_id())
    stray = EvalReportSession(
        id=report_session_id("another-report", "analyst"),
        scope_id=CALLER.scope_id,
        report_doc_id="another-report",
        name="analyst",
        digest=None,
        turns=0,
        session={},
    )

    with pytest.raises(ValueError, match="is stored with its own sessions only"):
        storage.save_report_doc(record.model_copy(update={"id": new_report_doc_id()}), sessions=[stray])


# --- resume through storage ------------------------------------------------------------------------------------------


async def test_an_analyst_session_reloaded_from_storage_resumes_with_the_bytes_an_uninterrupted_one_sends() -> None:
    host, result, interrupted, _ = await stored_run()
    uninterrupted = FakeToolTurnClient(*analyst_script(reply(submitted())))
    whole = AnalystSession.start(
        client=uninterrupted,
        snapshot=kairo_id(),
        limits=WriterLimits().analyst,
        **resume_options(host, mounted(host, "analyst")),
    )
    assert (await whole.draft()).ending == "submitted"
    await whole.revise(REVISION)

    stored = sessions_of(host, result.stored.id, name="analyst")["analyst"]
    later, mount = FakeToolTurnClient(reply(submitted())), mounted(host, "analyst")
    resumed = await AnalystSession.resume(stored.session, client=later, **resume_options(host, mount))
    outcome = await resumed.revise(REVISION)

    assert outcome.ending == "submitted" and outcome.round == 2
    (sent,) = later.requests
    assert json.dumps(sent).encode() == json.dumps(uninterrupted.requests[len(interrupted.requests)]).encode()


async def test_an_editor_session_reloaded_from_storage_resumes_with_the_bytes_an_uninterrupted_one_sends() -> None:
    host, result, _, editor = await stored_run()
    final = editor_script()[-1]
    stored = sessions_of(host, result.stored.id, name="editor-1")["editor-1"]
    later = FakeToolTurnClient(reply(TurnText("Item 6 names no experiment.")))

    resumed = await ToolSession.from_json(
        stored.session,
        model=MODEL,
        client=later,
        mount=mounted(host, "editor"),
        system=load_skill().editor(),
        tally=Tally(),
        budget=ReportBudget(usd=10),
    )
    resumed.open_round(2, calls=5)
    await resumed.send(QUESTION)

    (sent,) = later.requests
    uninterrupted = {
        **editor.requests[-1],
        "messages": [
            *editor.requests[-1]["messages"],
            final.message().to_json(),
            {"role": "user", "content": [{"type": "text", "text": QUESTION}]},
        ],
    }
    assert json.dumps(sent).encode() == json.dumps(uninterrupted).encode()


async def test_a_stored_session_keeps_w3_s_elision_and_resuming_re_derives_each_elided_body() -> None:
    host, result, interrupted, _ = await stored_run()
    stored = sessions_of(host, result.stored.id, name="analyst")["analyst"].session["session"]

    elided = [
        block for text in stored["messages"] for block in json.loads(text)["content"] if block["type"] == "tool_result"
    ]
    assert [item["stored"] for item in stored["results"]] == ["rederive"]
    assert [block["content"] for block in elided] == [""], "a re-derivable body is stored elided"

    mount, later = mounted(host, "analyst"), FakeToolTurnClient(reply(submitted()))
    record = sessions_of(host, result.stored.id, name="analyst")["analyst"].session
    resumed = await AnalystSession.resume(record, client=later, **resume_options(host, mount))
    await resumed.revise(REVISION)

    (rederived,) = [
        block
        for message in later.requests[0]["messages"]
        for block in message["content"]
        if block["type"] == "tool_result"
    ]
    (sent_whole,) = [
        block
        for message in interrupted.requests[-1]["messages"]
        for block in message["content"]
        if block["type"] == "tool_result"
    ]
    assert mount.calls[0] == {"action": "options_dominance", "snapshot_id": kairo_id()}, "the body is called again"
    assert rederived == sent_whole and rederived["content"], "and the request carries it whole, as it was sent"


# --- resume across a store that keeps no key order ------------------------------------------------------------------

KeyOrder = Callable[[list[str]], list[str]]


def jsonb_order(keys: list[str]) -> list[str]:
    """The order a Postgres ``jsonb`` value holds an object's keys in: shorter keys first, then bytewise."""
    return sorted(keys, key=lambda key: (len(key.encode("utf-8")), key.encode("utf-8")))


def reversed_order(keys: list[str]) -> list[str]:
    """Every object's keys the other way round: what a store keeping no order may hand back."""
    return keys[::-1]


def reordered(node: Any, order: KeyOrder) -> Any:
    """``node`` with every object's keys, at every depth, in ``order``; its values and every list's order kept."""
    if isinstance(node, dict):
        return {key: reordered(node[key], order) for key in order(list(node))}
    if isinstance(node, list):
        return [reordered(value, order) for value in node]
    return node


class _StoreKeepingNoKeyOrder(InMemoryDocumentStore):
    """The reference store, holding every document with each object's keys, at every depth, in ``order``."""

    def __init__(self, order: KeyOrder) -> None:
        super().__init__()
        self.order = order

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        held = reordered(json.loads(json.dumps(document)), self.order)
        super().upsert(held, if_match=if_match, create_only=create_only)

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        return super().merge_fields(doc_id, scope_id, reordered(json.loads(json.dumps(fields)), self.order))


#: A tool call whose arguments are in an order no reordering store keeps: neither sorted, nor jsonb's, nor reversed.
UNORDERED_INPUT: Final = {"snapshot_id": "", "action": "options_dominance", "arms": ["September"]}


def unordered_script() -> list[Any]:
    """The analyst's draft with a signed thinking block, a redacted one ({type, data}, which jsonb turns round) and a
    tool call whose arguments no reordering store keeps in order; then the Kairo document."""
    arguments = {**UNORDERED_INPUT, "snapshot_id": kairo_id()}
    return [
        reply(
            TurnThinking({"type": "thinking", "thinking": "the cost gap reads first", "signature": "sig-signed-once"}),
            TurnThinking({"type": "redacted_thinking", "data": "opaque-redacted-payload"}),
            TurnToolUse("toolu_001", "evals_analyst", arguments),
        ),
        reply(submitted()),
    ]


@pytest.mark.parametrize("order", [jsonb_order, reversed_order], ids=["jsonb", "reversed"])
async def test_a_session_resumed_from_a_store_that_reorders_keys_sends_the_bytes_an_uninterrupted_one_sends(
    order: KeyOrder,
) -> None:
    store = _StoreKeepingNoKeyOrder(order)
    host = off_loop(host_over, store)
    draft = FakeToolTurnClient(*unordered_script())
    result = await written(FakeTurnModel(analyst=draft, editor_1=FakeToolTurnClient(*editor_script())), host=host)
    assert result.ending == "stored", result.outcome.reason

    uninterrupted = FakeToolTurnClient(*unordered_script(), reply(submitted()))
    whole = AnalystSession.start(
        client=uninterrupted,
        snapshot=kairo_id(),
        limits=WriterLimits().analyst,
        **resume_options(host, mounted(host, "analyst")),
    )
    await whole.draft()
    await whole.revise(REVISION)
    stored = sessions_of(host, result.stored.id, name="analyst")["analyst"]
    held = off_loop(store.get, stored.id, CALLER.scope_id)
    fields = list(stored.to_dict())
    assert held is not None and [key for key in held if key in fields] == order(fields) != fields, (
        "the store hands the session back in its own key order"
    )
    later = FakeToolTurnClient(reply(submitted()))
    resumed = await AnalystSession.resume(
        stored.session, client=later, **resume_options(host, mounted(host, "analyst"))
    )
    await resumed.revise(REVISION)

    (sent,) = later.requests
    assert json.dumps(sent).encode() == json.dumps(uninterrupted.requests[len(draft.requests)]).encode()
    blocks = [block for message in sent["messages"] for block in message["content"]]
    (use,) = [block for block in blocks if block["type"] == "tool_use"]
    assert list(use["input"]) == list(UNORDERED_INPUT), "the call's arguments are resent in the order they were sent"
    assert [block for block in blocks if block["type"] in ("thinking", "redacted_thinking")] == [
        {"type": "thinking", "thinking": "the cost gap reads first", "signature": "sig-signed-once"},
        {"type": "redacted_thinking", "data": "opaque-redacted-payload"},
    ]
    assert [list(block) for block in blocks if block["type"] == "redacted_thinking"] == [["type", "data"]]
    assert '"signature": "sig-signed-once"' in json.dumps(sent), "the thinking block's signature survives"


# --- a listing reads no session --------------------------------------------------------------------------------------


class _ReadsCounted(InMemoryDocumentStore):
    """The reference store, keeping the doc type of every bulk read and every id read by point."""

    def __init__(self) -> None:
        super().__init__()
        self.read: list[str] = []

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **options: Any,
    ) -> list[dict[str, Any]]:
        self.read.append(doc_type)
        return super().by_doc_type(doc_type, scope_id, field_in=field_in, **options)

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        self.read.append(f"get:{doc_id}")
        return super().get(doc_id, scope_id)

    def get_many(self, doc_type: str, doc_ids: Sequence[str], scope_id: str, **options: Any) -> list[dict[str, Any]]:
        self.read.append(doc_type)
        return super().get_many(doc_type, doc_ids, scope_id, **options)


def test_listing_report_documents_reads_report_documents_and_no_session() -> None:
    store = _ReadsCounted()
    storage = EvalStorage(store)
    first, again = _gate(storage, SESSIONS, new_report_doc_id()), _gate(storage, SESSIONS, new_report_doc_id())
    store.read.clear()

    listed = storage.list_report_docs(CALLER.scope_id)

    assert {record.id for record in listed} == {first.id, again.id}
    assert store.read == ["eval_report_doc"]


# --- the sweep -------------------------------------------------------------------------------------------------------


def test_drop_derived_sweeps_the_sessions_with_their_report() -> None:
    storage, store = memory_storage()
    record = _gate(storage, SESSIONS, new_report_doc_id())

    dropped = storage.drop_derived([CALLER.scope_id])

    assert (dropped.counts["eval_report_doc"], dropped.counts["eval_report_session"]) == (1, 2)
    assert storage.load_report_sessions(record.id, CALLER.scope_id) == {}
    assert list(store.iter_by_doc_type("eval_report_session", CALLER.scope_id)) == []


def test_deleting_a_report_document_deletes_its_sessions_and_no_other_report_s() -> None:
    storage, store = memory_storage()
    gone, kept = _gate(storage, SESSIONS, new_report_doc_id()), _gate(storage, SESSIONS, new_report_doc_id())

    assert storage.delete_report_doc(gone.id, CALLER.scope_id) is True

    assert storage.load_report_doc(gone.id, CALLER.scope_id) is None
    assert storage.load_report_sessions(gone.id, CALLER.scope_id) == {}
    assert set(storage.load_report_sessions(kept.id, CALLER.scope_id)) == {"analyst", "editor-1"}
    assert len(list(store.iter_by_doc_type("eval_report_session", CALLER.scope_id))) == 2
    assert storage.delete_report_doc(gone.id, CALLER.scope_id) is False


# --- the bump ----------------------------------------------------------------------------------------------------------


def _one_refusal(refused: pytest.ExceptionInfo[ValidationError]) -> str:
    errors = refused.value.errors()
    assert len(errors) == 1, "a stale document is refused by its version alone, never field by field"
    return str(errors[0]["msg"])


def test_a_report_document_in_the_old_record_shape_is_refused_by_its_version_with_the_regenerate_wording() -> None:
    """The fourth prototype's Kairo record: report version 13, its sessions whole in a format-1 generation record."""
    storage, store = memory_storage()
    old = json.loads(KAIRO_RECORD.read_text()) | {"scope_id": CALLER.scope_id}
    assert (old["schema_version"], old["generation"]["format"]) == (REPORT_DOC_SCHEMA_VERSION - 1, 1)
    store.upsert(old)

    with pytest.raises(ValidationError) as refused:
        storage.load_report_doc(old["id"], CALLER.scope_id)

    said = _one_refusal(refused)
    assert f"written under report document schema v{REPORT_DOC_SCHEMA_VERSION - 1}" in said
    assert f"reads report document schema v{REPORT_DOC_SCHEMA_VERSION} only" in said
    assert "EvalStorage.drop_derived(scopes), then writes each report again from a fresh snapshot" in said
    assert storage.drop_derived([CALLER.scope_id]).counts["eval_report_doc"] == 1


def test_a_report_session_of_another_version_is_refused_by_its_version_alone() -> None:
    storage, store = memory_storage()
    record = _gate(storage, SESSIONS, new_report_doc_id())
    stale = {
        **storage.load_report_sessions(record.id, CALLER.scope_id, name="analyst")["analyst"].to_dict(),
        "schema_version": REPORT_DOC_SCHEMA_VERSION - 1,
        "turns": "not a count",
    }
    store.upsert(stale)

    with pytest.raises(ValidationError) as refused:
        storage.load_report_sessions(record.id, CALLER.scope_id, name="analyst")

    assert f"this report session was written under report document schema v{REPORT_DOC_SCHEMA_VERSION - 1}" in (
        _one_refusal(refused)
    )


def test_a_report_session_is_stored_under_the_id_its_report_and_name_give_and_no_other() -> None:
    session = EvalReportSession(
        id="report-1:session:editor-2",
        scope_id=CALLER.scope_id,
        report_doc_id="report-1",
        name="editor-2",
        digest=None,
        turns=0,
        session={},
    )
    assert session.id == report_session_id("report-1", "editor-2")

    with pytest.raises(ValidationError, match="the one its report and name give"):
        EvalReportSession.model_validate({**session.to_dict(), "id": "report-1:session:analyst"})


# --- awaiting a caller ---------------------------------------------------------------------------------------------

#: Every workspace member's ``src`` tree.
_REPO: Final = Path(__file__).resolve().parents[3]
_SOURCES: Final = sorted([*_REPO.glob("packages/*/src"), *_REPO.glob("packages/agent/*/src")])

#: The module that defines ``EvalStorage.load_report_sessions``, the one ``src`` file allowed to name it until VIEW.
_DEFINED_IN: Final = Path(storage_module.__file__ or "").resolve()


def test_loading_a_report_s_sessions_awaits_its_production_caller() -> None:
    """``EvalStorage.load_report_sessions`` is the audit read's (``report_read(include_sessions=true)``, wave 2's VIEW).

    Until VIEW calls it, the store is its only ``src`` reader; the change that gives it a caller removes this pin.
    """
    callers = sorted(
        str(path)
        for root in _SOURCES
        for path in root.rglob("*.py")
        if "load_report_sessions" in path.read_text() and path.resolve() != _DEFINED_IN
    )
    assert callers == [], f"load_report_sessions has a production caller now: drop this pin. Callers: {callers}"
