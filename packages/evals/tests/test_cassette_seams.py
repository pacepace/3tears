"""The cassette lane wires only the seams a kind supplies, and replays a recorded session from them.

The candidate here is a tabletop game-master session, chosen because it has both shapes a kind can
declare: synchronous tools (``roll``, ``rules_lookup``) that answer from ``act()``, and an asynchronous
one (``scout_ahead``) whose ``act()`` only acknowledges and whose report arrives later. A capture runs
them live and records; a replay of the same calls must hand the session the same answers without
running any of them live — while the session's live tools are deliberately built to answer
differently, so a replay that went live would be visible.

Storage is the production ``EvalStorage`` over the in-memory document store, so what is recorded is
what a later lane reads back through the real cassette queries.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.factories import memory_storage
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.kernel.cassettes import (
    ActionSeam,
    CassetteCorrupt,
    CassetteExhausted,
    CassetteMiss,
    CassetteSeams,
    DeliveryRecorder,
    DeliveryReplay,
    DeliverySeam,
    DeliveryTicket,
    ReplayedDelivery,
    ToolLike,
    ToolWrap,
)
from threetears.evals.kernel.host.apparatus import ApparatusError
from threetears.evals.schema.models import CassetteKey, EvalCassette
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.run.cassette_proxy import (
    DELIVERY_ACTION,
    CassetteCell,
    CassetteLane,
    CassetteProxy,
    params_hash,
)

_SCOPE = "table-1"
_TEMPLATE = "tpl-goblin-ambush"
_CASE = "case-1"
_MODEL = "gm-model-a"
_CORPUS = "run-capture-1"


# =============================================================================
# The candidate's own types — pydantic models, which is all a Recordable is
# =============================================================================


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RollResult(_Strict):
    """What ``roll`` answers."""

    ok: bool
    total: int | None = None
    error: str | None = None


class RulesAnswer(_Strict):
    """What ``rules_lookup`` answers."""

    ok: bool
    text: str


class Ack(_Strict):
    """What ``narrate`` and ``scout_ahead``'s ``act()`` answer."""

    ok: bool
    text: str


class ScoutReport(_Strict):
    """What ``scout_ahead`` delivers, later."""

    area: str
    findings: str


# =============================================================================
# The session's tools
# =============================================================================


# parity-with: threetears.evals.kernel.cassettes.ToolLike
class FakeTool:
    """A synchronous tool whose live answers come from ``answer``, logging every live call."""

    description = "a tool the session's own machinery reads through any wrapper"

    def __init__(self, name: str, actions: frozenset[str], answer: Callable[[dict[str, Any]], BaseModel]) -> None:
        self._name = name
        self._actions = actions
        self._answer = answer
        self.live_calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return self._name

    def can_dispatch(self, action: str) -> bool:
        return action in self._actions

    async def act(self, action: str, parameters: dict[str, Any]) -> Any:
        self.live_calls.append((action, dict(parameters)))
        if action not in self._actions:
            return RollResult(ok=False, error=f"Unknown action: {action}")
        return self._answer(parameters)


# parity-with: threetears.evals.kernel.cassettes.DeliverySeam
class FakeScoutTool(DeliverySeam):
    """An asynchronous tool: ``act()`` acknowledges, and the report reaches the session afterwards.

    Implements the delivery seam the way a real asynchronous tool would: capture starts a ticket where
    the scout is SENT and settles it where the report arrives; replay takes the recording of the same
    request where it would have started live work. ``latency`` says how many event-loop yields each
    area's live scout takes, so two scouts can finish in the opposite order to the one they were sent
    in; ``failing`` areas end in an error, and ``lost`` areas never come back.
    """

    def __init__(
        self,
        reports: Mapping[str, str],
        *,
        latency: Mapping[str, int] | None = None,
        failing: frozenset[str] = frozenset(),
        lost: frozenset[str] = frozenset(),
    ) -> None:
        self._reports = dict(reports)
        self._latency = dict(latency or {})
        self._failing = failing
        self._lost = lost
        self.live_scouts: list[str] = []
        self.delivered: list[ScoutReport] = []
        self.failures: list[str] = []
        self.replayed: list[ReplayedDelivery[Any]] = []
        self.pending: list[asyncio.Task[None]] = []
        self.recorder: DeliveryRecorder | None = None
        self.replay: DeliveryReplay[Any] | None = None

    @property
    def name(self) -> str:
        return "scout_ahead"

    @property
    def payload_type(self) -> type[ScoutReport]:
        return ScoutReport

    def can_dispatch(self, action: str) -> bool:
        return action == "scout"

    def arm_capture(self, recorder: DeliveryRecorder) -> None:
        self.recorder = recorder

    def arm_replay(self, replay: DeliveryReplay[Any]) -> None:
        self.replay = replay

    async def act(self, action: str, parameters: dict[str, Any]) -> Ack:
        area = parameters["area"]
        if self.replay is not None:
            delivery = self.replay.next({"area": area})
            self.replayed.append(delivery)
            if delivery.outcome == "delivered":
                assert delivery.payload is not None
                self.pending.append(asyncio.create_task(self._arrive(delivery.payload)))
            elif delivery.outcome == "failed":
                assert delivery.error is not None
                self.failures.append(delivery.error)
        else:
            self.live_scouts.append(area)
            ticket = self.recorder.started({"area": area}) if self.recorder is not None else None
            self.pending.append(asyncio.create_task(self._scout_live(area, ticket)))
        return Ack(ok=True, text=f"scouting {area}")

    async def _scout_live(self, area: str, ticket: DeliveryTicket | None) -> None:
        for _ in range(self._latency.get(area, 1)):
            await asyncio.sleep(0)
        if area in self._lost:
            return
        if area in self._failing:
            self.failures.append(f"the scout to {area} was ambushed")
            if ticket is not None:
                ticket.failed(f"the scout to {area} was ambushed")
            return
        report = ScoutReport(area=area, findings=self._reports[area])
        await self._arrive(report)
        if ticket is not None:
            ticket.delivered(report)

    async def _arrive(self, report: ScoutReport) -> None:
        await asyncio.sleep(0)
        self.delivered.append(report)


# parity-with: threetears.evals.kernel.cassettes.ActionSeam
class FakeGameSession(CassetteSeams):
    """The kind's prepared candidate: its tools, and the seams it hands the lane."""

    def __init__(
        self,
        *,
        rolls: list[int],
        rules: Mapping[str, str],
        reports: Mapping[str, str],
        recorded: Mapping[str, type[BaseModel]] | None = None,
        scout: FakeScoutTool | None = None,
    ) -> None:
        totals = iter(rolls)
        self.roll = FakeTool("roll", frozenset({"roll"}), lambda p: RollResult(ok=True, total=next(totals)))
        self.rules = FakeTool(
            "rules_lookup", frozenset({"lookup"}), lambda p: RulesAnswer(ok=True, text=rules[p["topic"]])
        )
        self.narrate = FakeTool("narrate", frozenset({"say"}), lambda p: Ack(ok=True, text=p["line"]))
        self.scout = scout if scout is not None else FakeScoutTool(reports)
        self.tools: dict[str, Any] = {
            "roll": self.roll,
            "rules_lookup": self.rules,
            "narrate": self.narrate,
            "scout_ahead": self.scout,
        }
        self._recorded = dict(recorded if recorded is not None else {"roll": RollResult, "rules_lookup": RulesAnswer})

    @property
    def action_seam(self) -> ActionSeam | None:
        return self

    @property
    def delivery_seams(self) -> Mapping[str, DeliverySeam]:
        return {"scout_ahead": self.scout}

    @property
    def recorded_tools(self) -> Mapping[str, type[BaseModel]]:
        return self._recorded

    def arm_tools(self, wrap: ToolWrap) -> None:
        self.tools = wrap(self.tools)

    async def call(self, tool: str, action: str, **parameters: Any) -> Any:
        return await self.tools[tool].act(action, parameters)

    async def settle(self) -> None:
        await asyncio.gather(*self.scout.pending)


# parity-with: threetears.evals.kernel.cassettes.CassetteSeams
class FakeSeamlessSession:
    """A candidate whose kind declares no seam at all."""

    @property
    def action_seam(self) -> ActionSeam | None:
        return None

    @property
    def delivery_seams(self) -> Mapping[str, DeliverySeam]:
        return {}


# =============================================================================
# Helpers
# =============================================================================


_REPORTS = {"north corridor": "three goblins behind a barricade", "chapel": "an empty altar"}


def _capture_session(**scout: Any) -> FakeGameSession:
    return FakeGameSession(
        rolls=[14, 7, 3],
        rules={"grapple": "Contested Athletics check."},
        reports=_REPORTS,
        scout=FakeScoutTool(_REPORTS, **scout) if scout else None,
    )


def _replay_session() -> FakeGameSession:
    """The same session, built so every live answer differs from the captured one."""
    return FakeGameSession(
        rolls=[1, 1, 1],
        rules={"grapple": "WRONG — a live lookup"},
        reports={"north corridor": "WRONG — a live scout", "chapel": "WRONG — a live scout"},
    )


def _lane(storage: EvalStorage, mode: str, *, scope_id: str = _SCOPE, corpus_id: str = _CORPUS) -> CassetteLane:
    return CassetteLane(mode=mode, store=storage, scope_id=scope_id, corpus_id=corpus_id)  # type: ignore[arg-type]


def _cell(lane: CassetteLane, *, test_case_id: str = _CASE) -> CassetteCell:
    return lane.cell(template_id=_TEMPLATE, test_case_id=test_case_id, model=_MODEL)


def _wire(lane: CassetteLane, seams: Any, *, test_case_id: str = _CASE) -> CassetteCell:
    cell = _cell(lane, test_case_id=test_case_id)
    cell.wire(seams)
    return cell


async def _play(session: FakeGameSession) -> list[Any]:
    """One GM turn sequence: two rolls, a rules lookup, a line of narration and two scouting trips."""
    answers = [
        await session.call("roll", "roll", dice="1d20", modifier=3),
        await session.call("roll", "roll", dice="2d6"),
        await session.call("rules_lookup", "lookup", topic="grapple"),
        await session.call("narrate", "say", line="The torches gutter."),
        await session.call("scout_ahead", "scout", area="north corridor"),
        await session.call("scout_ahead", "scout", area="chapel"),
    ]
    await session.settle()
    return answers


async def _captured(storage: EvalStorage, **scout: Any) -> tuple[FakeGameSession, list[Any]]:
    session = _capture_session(**scout)
    cell = _wire(_lane(storage, "capture"), session)
    answers = await _play(session)
    cell.close()
    return session, answers


def _recordings(storage: EvalStorage, *, corpus_id: str = _CORPUS, test_case_id: str = _CASE) -> list[EvalCassette]:
    return storage.list_case_cassettes(
        corpus_id=corpus_id, template_id=_TEMPLATE, test_case_id=test_case_id, scope_id=_SCOPE
    )


# =============================================================================
# Capture, then replay
# =============================================================================


async def test_a_replay_serves_the_captured_session_and_runs_no_recorded_tool_live() -> None:
    storage, _ = memory_storage()
    captured, captured_answers = await _captured(storage)

    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)
    replayed_answers = await _play(replay)

    assert replayed_answers == captured_answers
    assert replay.scout.delivered == captured.scout.delivered
    assert [r.findings for r in replay.scout.delivered] == ["three goblins behind a barricade", "an empty altar"]
    # Nothing recorded ran live under replay, and nothing went missing from the capture either.
    assert replay.roll.live_calls == [] and replay.rules.live_calls == [] and replay.scout.live_scouts == []
    assert len(captured.roll.live_calls) == 2 and captured.scout.live_scouts == ["north corridor", "chapel"]


async def test_a_tool_the_kind_did_not_declare_is_left_unwrapped_and_runs_live() -> None:
    storage, _ = memory_storage()
    await _captured(storage)
    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)

    assert isinstance(replay.tools["roll"], CassetteProxy)
    assert replay.tools["narrate"] is replay.narrate
    # The asynchronous tool is on the delivery seam, never wrapped at the action seam.
    assert replay.tools["scout_ahead"] is replay.scout
    await replay.call("narrate", "say", line="A door creaks.")
    assert replay.narrate.live_calls == [("say", {"line": "A door creaks."})]


async def test_the_proxy_is_the_tool_to_the_session_s_own_machinery() -> None:
    storage, _ = memory_storage()
    session = _capture_session()
    _wire(_lane(storage, "capture"), session)

    proxy = session.tools["roll"]
    assert isinstance(proxy, ToolLike)
    assert proxy.name == "roll"
    assert proxy.description == FakeTool.description


async def test_the_action_key_ignores_parameter_order() -> None:
    storage, _ = memory_storage()
    _, captured_answers = await _captured(storage)
    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)

    assert await replay.call("roll", "roll", modifier=3, dice="1d20") == captured_answers[0]
    assert params_hash({"a": 1, "b": {"x": 1, "y": 2}}) == params_hash({"b": {"y": 2, "x": 1}, "a": 1})
    assert params_hash({"dice": [1, 2]}) != params_hash({"dice": [2, 1]})


# =============================================================================
# The same ask twice is two recordings, served in the order asked
# =============================================================================


async def test_a_session_that_rolls_the_same_dice_twice_replays_both_rolls_in_order() -> None:
    """Rolling 1d20 twice records 14 then 3 — a replay must serve 14 then 3, not 3 twice."""
    storage, _ = memory_storage()
    capture = FakeGameSession(rolls=[14, 3], rules={}, reports={})
    cell = _wire(_lane(storage, "capture"), capture)
    first = await capture.call("roll", "roll", dice="1d20")
    second = await capture.call("roll", "roll", dice="1d20")
    cell.close()
    assert (first.total, second.total) == (14, 3)

    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)
    assert [(await replay.call("roll", "roll", dice="1d20")).total for _ in range(2)] == [14, 3]
    assert replay.roll.live_calls == []


async def test_asking_the_same_thing_once_more_than_the_capture_did_is_exhausted_not_reused() -> None:
    storage, _ = memory_storage()
    capture = FakeGameSession(rolls=[14, 3], rules={}, reports={})
    _wire(_lane(storage, "capture"), capture)
    await capture.call("roll", "roll", dice="1d20")
    await capture.call("roll", "roll", dice="1d20")

    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)
    await replay.call("roll", "roll", dice="1d20")
    await replay.call("roll", "roll", dice="1d20")
    with pytest.raises(CassetteExhausted) as caught:
        await replay.call("roll", "roll", dice="1d20")
    assert isinstance(caught.value, ApparatusError)
    assert (caught.value.key.tool, caught.value.key.occurrence, caught.value.recorded) == ("roll", 2, 2)
    assert replay.roll.live_calls == []


async def test_each_occurrence_of_an_ask_is_its_own_recording() -> None:
    storage, _ = memory_storage()
    capture = FakeGameSession(rolls=[14, 3], rules={}, reports={})
    _wire(_lane(storage, "capture"), capture)
    await capture.call("roll", "roll", dice="1d20")
    await capture.call("roll", "roll", dice="1d20")

    rolls = sorted((c.occurrence, c.response) for c in _recordings(storage) if c.tool == "roll")
    assert rolls == [(0, {"ok": True, "total": 14, "error": None}), (1, {"ok": True, "total": 3, "error": None})]


# =============================================================================
# Background work is paired with the request that started it
# =============================================================================


async def test_a_slow_scout_and_a_fast_one_each_get_their_own_report_on_replay() -> None:
    """Captured: north sent first and slow, chapel second and fast, so chapel's report lands first."""
    storage, _ = memory_storage()
    captured, _ = await _captured(storage, latency={"north corridor": 6, "chapel": 1})
    assert [r.area for r in captured.scout.delivered] == ["chapel", "north corridor"]

    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)
    await replay.call("scout_ahead", "scout", area="north corridor")
    await replay.call("scout_ahead", "scout", area="chapel")
    await replay.settle()

    assert {r.area: r.findings for r in replay.scout.delivered} == _REPORTS
    assert [r.payload.area for r in replay.scout.replayed if r.payload is not None] == ["north corridor", "chapel"]


async def test_a_request_the_capture_never_made_stops_the_cell_rather_than_serving_another_report() -> None:
    storage, _ = memory_storage()
    await _captured(storage)
    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)

    with pytest.raises(CassetteMiss) as caught:
        await replay.call("scout_ahead", "scout", area="the well")
    assert isinstance(caught.value, ApparatusError)
    assert caught.value.key.params_hash == params_hash({"area": "the well"})
    assert replay.scout.delivered == [] and replay.scout.live_scouts == []


async def test_work_that_failed_or_never_came_back_in_the_capture_replays_the_same_way() -> None:
    storage, _ = memory_storage()
    await _captured(storage, failing=frozenset({"north corridor"}), lost=frozenset({"chapel"}))

    outcomes = {c.params_hash: (c.outcome, c.error) for c in _recordings(storage) if c.seam == "delivery"}
    assert outcomes == {
        params_hash({"area": "north corridor"}): ("failed", "the scout to north corridor was ambushed"),
        params_hash({"area": "chapel"}): ("undelivered", None),
    }

    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)
    await replay.call("scout_ahead", "scout", area="north corridor")
    await replay.call("scout_ahead", "scout", area="chapel")
    await replay.settle()
    assert [d.outcome for d in replay.scout.replayed] == ["failed", "undelivered"]
    assert replay.scout.failures == ["the scout to north corridor was ambushed"]
    assert replay.scout.delivered == [] and replay.scout.live_scouts == []


async def test_deliveries_are_stored_under_the_engine_s_delivery_action_and_their_request() -> None:
    storage, _ = memory_storage()
    await _captured(storage)

    recorded = sorted(
        ((c.action, c.params_hash, c.occurrence, c.outcome) for c in _recordings(storage) if c.seam == "delivery"),
    )
    assert recorded == sorted(
        [
            (DELIVERY_ACTION, params_hash({"area": "north corridor"}), 0, "delivered"),
            (DELIVERY_ACTION, params_hash({"area": "chapel"}), 0, "delivered"),
        ]
    )
    assert {c.captured_model for c in _recordings(storage)} == {_MODEL}


async def test_a_ticket_is_settled_once() -> None:
    storage, _ = memory_storage()
    session = _capture_session()
    _wire(_lane(storage, "capture"), session)
    assert session.scout.recorder is not None
    ticket = session.scout.recorder.started({"area": "chapel"})
    ticket.delivered(ScoutReport(area="chapel", findings="x"))
    with pytest.raises(ValueError, match="already settled"):
        ticket.failed("twice")


async def test_work_settling_after_its_cell_ended_keeps_the_session_s_undelivered_record() -> None:
    storage, _ = memory_storage()
    session = _capture_session()
    cell = _wire(_lane(storage, "capture"), session)
    assert session.scout.recorder is not None
    ticket = session.scout.recorder.started({"area": "chapel"})
    cell.close()
    ticket.delivered(ScoutReport(area="chapel", findings="arrived too late"))

    [recording] = _recordings(storage)
    assert (recording.outcome, recording.response) == ("undelivered", None)


# =============================================================================
# Misses are loud, and they are the rig's
# =============================================================================


async def test_an_unrecorded_call_is_a_miss_naming_its_key_and_never_runs_live() -> None:
    storage, _ = memory_storage()
    await _captured(storage)
    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)

    with pytest.raises(CassetteMiss) as caught:
        await replay.call("roll", "roll", dice="8d6")
    assert isinstance(caught.value, ApparatusError)
    assert caught.value.key == CassetteKey(
        corpus_id=_CORPUS,
        template_id=_TEMPLATE,
        test_case_id=_CASE,
        tool="roll",
        action="roll",
        params_hash=params_hash({"dice": "8d6"}),
        occurrence=0,
    )
    assert replay.roll.live_calls == []


@pytest.mark.parametrize(
    ("scope_id", "corpus_id"), [("table-2", _CORPUS), (_SCOPE, "run-capture-2")], ids=["other-scope", "other-corpus"]
)
async def test_a_replay_of_another_scope_or_corpus_reads_nothing_of_this_one(scope_id: str, corpus_id: str) -> None:
    storage, _ = memory_storage()
    await _captured(storage)
    replay = _replay_session()
    _wire(_lane(storage, "replay", scope_id=scope_id, corpus_id=corpus_id), replay)

    with pytest.raises(CassetteMiss):
        await replay.call("roll", "roll", dice="1d20", modifier=3)
    with pytest.raises(CassetteMiss):
        await replay.call("scout_ahead", "scout", area="north corridor")


async def test_an_action_the_tool_cannot_dispatch_gets_the_tool_s_own_answer_not_a_miss() -> None:
    storage, _ = memory_storage()
    await _captured(storage)
    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)

    answer = await replay.call("roll", "reroll_everything")
    assert answer == RollResult(ok=False, error="Unknown action: reroll_everything")
    assert replay.roll.live_calls == [("reroll_everything", {})]


async def test_asking_for_more_scouting_than_was_captured_exhausts_rather_than_going_live() -> None:
    storage, _ = memory_storage()
    await _captured(storage)
    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)
    await _play(replay)

    with pytest.raises(CassetteExhausted) as caught:
        await replay.call("scout_ahead", "scout", area="chapel")
    assert isinstance(caught.value, ApparatusError)
    assert (caught.value.key.tool, caught.value.recorded) == ("scout_ahead", 1)
    assert replay.scout.live_scouts == []


async def test_a_case_that_recorded_nothing_wires_and_fails_only_if_asked() -> None:
    storage, _ = memory_storage()
    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)

    with pytest.raises(CassetteMiss):
        await replay.call("scout_ahead", "scout", area="north corridor")
    assert replay.scout.live_scouts == []


def _corrupt(store: InMemoryDocumentStore, *, seam: str, change: Callable[[dict[str, Any]], None]) -> None:
    for document in store.documents.values():
        if document.get("doc_type") == "eval_cassette" and document["seam"] == seam:
            change(document)


def _misfit_payload(document: dict[str, Any]) -> None:
    document["response"] = {"total": "not a number", "findings": 7}


def _unknown_field(document: dict[str, Any]) -> None:
    document["a_field_this_build_never_wrote"] = True


def _other_schema(document: dict[str, Any]) -> None:
    document["schema_version"] = 5


_CALLS: dict[str, Callable[[FakeGameSession], Any]] = {
    "action": lambda s: s.call("roll", "roll", dice="1d20", modifier=3),
    "delivery": lambda s: s.call("scout_ahead", "scout", area="north corridor"),
}


@pytest.mark.parametrize("seam", ["action", "delivery"])
@pytest.mark.parametrize("change", [_misfit_payload, _unknown_field, _other_schema], ids=lambda c: c.__name__)
async def test_a_recording_that_no_longer_reads_back_is_the_rig_s_failure(
    seam: str, change: Callable[[dict[str, Any]], None]
) -> None:
    """A stale row — another schema, an unknown key, a payload that no longer fits — is CassetteCorrupt."""
    storage, store = memory_storage()
    await _captured(storage)
    _corrupt(store, seam=seam, change=change)
    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)

    with pytest.raises(CassetteCorrupt) as caught:
        await _CALLS[seam](replay)
    assert isinstance(caught.value, ApparatusError)


# parity-with: threetears.evals.schema.store_port.DocumentStore
class FakeUnreadableStore(InMemoryDocumentStore):
    """A store whose reads fail, as a backend outage would."""

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        raise ConnectionError("the database went away")


@pytest.mark.parametrize("seam", ["action", "delivery"])
async def test_a_store_that_cannot_answer_is_the_rig_s_failure(seam: str) -> None:
    store = FakeUnreadableStore()
    replay = _replay_session()
    _wire(_lane(EvalStorage(store), "replay"), replay)

    with pytest.raises(CassetteCorrupt, match="the database went away"):
        await _CALLS[seam](replay)


async def test_a_stale_row_in_another_case_does_not_break_this_case() -> None:
    """Reads are per case: one unreadable row elsewhere in the template leaves every other case replayable."""
    storage, store = memory_storage()
    await _captured(storage)
    other = _capture_session()
    _wire(_lane(storage, "capture"), other, test_case_id="case-2")
    await other.call("scout_ahead", "scout", area="chapel")
    await other.settle()
    for document in store.documents.values():
        if document.get("test_case_id") == "case-2":
            document["schema_version"] = 5

    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)
    assert await _play(replay)
    assert [r.area for r in replay.scout.delivered] == ["north corridor", "chapel"]


async def test_a_recording_asked_for_at_the_other_seam_is_corrupt() -> None:
    storage, _ = memory_storage()
    key = CassetteKey(
        corpus_id=_CORPUS,
        template_id=_TEMPLATE,
        test_case_id=_CASE,
        tool="roll",
        action="roll",
        params_hash=params_hash({"dice": "1d20"}),
        occurrence=0,
    )
    storage.save_cassette(
        EvalCassette.build(key, scope_id=_SCOPE, seam="delivery", captured_model=_MODEL, outcome="undelivered")
    )
    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)
    with pytest.raises(CassetteCorrupt, match="delivery seam"):
        await replay.call("roll", "roll", dice="1d20")


# =============================================================================
# Capture keeps the corpus whole
# =============================================================================


class _FailingWritesStore(InMemoryDocumentStore):
    """The reference store; while ``fail_writes`` is set, every ``upsert`` raises."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_writes = False

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None) -> None:
        if self.fail_writes:
            raise RuntimeError("simulated write failure")
        super().upsert(document, if_match=if_match)


def _failing_storage() -> tuple[EvalStorage, _FailingWritesStore]:
    store = _FailingWritesStore()
    return EvalStorage(store), store


async def test_a_failed_action_write_still_answers_the_session() -> None:
    storage, store = _failing_storage()
    session = _capture_session()
    _wire(_lane(storage, "capture"), session)
    store.fail_writes = True

    assert await session.call("roll", "roll", dice="1d20") == RollResult(ok=True, total=14)


async def test_a_failed_write_leaves_a_hole_a_replay_reports_rather_than_a_neighbour_in_its_place() -> None:
    """The first roll's recording is lost and the second's lands: the first ask must not get the second roll."""
    storage, store = _failing_storage()
    session = _capture_session()
    cell = _wire(_lane(storage, "capture"), session)

    store.fail_writes = True
    await session.call("roll", "roll", dice="1d20")
    store.fail_writes = False
    await session.call("roll", "roll", dice="1d20")
    cell.close()

    replay = _replay_session()
    _wire(_lane(storage, "replay"), replay)
    with pytest.raises(CassetteMiss):
        await replay.call("roll", "roll", dice="1d20")


async def test_a_recapture_replaces_the_case_s_whole_recording() -> None:
    storage, _ = memory_storage()
    await _captured(storage)

    shorter = _capture_session()
    cell = _wire(_lane(storage, "capture"), shorter)
    await shorter.call("scout_ahead", "scout", area="chapel")
    await shorter.settle()
    cell.close()

    recorded = _recordings(storage)
    assert [(c.tool, c.params_hash) for c in recorded] == [("scout_ahead", params_hash({"area": "chapel"}))]


async def test_a_recapture_clears_rows_this_build_can_no_longer_read() -> None:
    storage, store = memory_storage()
    await _captured(storage)
    for document in store.documents.values():
        document["schema_version"] = 5

    _wire(_lane(storage, "capture"), _capture_session())
    assert _recordings(storage) == []


async def test_a_capture_that_cannot_clear_the_case_refuses_to_mix_two_sessions() -> None:
    storage, store = memory_storage()
    await _captured(storage)

    def refuse(doc_id: str, scope_id: str) -> bool:
        raise ConnectionError("delete refused")

    store.delete = refuse  # type: ignore[method-assign]
    session = _capture_session()
    tools = session.tools
    with pytest.raises(CassetteCorrupt, match="mixed"):
        _wire(_lane(storage, "capture"), session)
    assert session.tools is tools and session.scout.recorder is None


async def test_two_capture_runs_at_once_never_write_into_each_other_s_corpus() -> None:
    """A launch over two models runs two capture runs concurrently, over the same case."""
    storage, _ = memory_storage()
    run_a = make_eval_run(cassette_mode="capture", scope_id=_SCOPE, template_id=_TEMPLATE, candidate_model="model-a")
    run_b = make_eval_run(cassette_mode="capture", scope_id=_SCOPE, template_id=_TEMPLATE, candidate_model="model-b")
    lane_a, lane_b = CassetteLane.for_run(run_a, storage), CassetteLane.for_run(run_b, storage)
    assert lane_a is not None and lane_b is not None
    session_a = FakeGameSession(rolls=[11, 12], rules={}, reports={"chapel": "A's chapel"})
    session_b = FakeGameSession(rolls=[21, 22], rules={}, reports={"chapel": "B's chapel"})
    cell_a = lane_a.cell(template_id=_TEMPLATE, test_case_id=_CASE, model="model-a")
    cell_b = lane_b.cell(template_id=_TEMPLATE, test_case_id=_CASE, model="model-b")

    # Interleaved exactly as the review's scenario: A wires and records, B wires (and clears) and
    # records, A records again.
    cell_a.wire(session_a)
    await session_a.call("roll", "roll", dice="1d20")
    cell_b.wire(session_b)
    await session_b.call("roll", "roll", dice="1d20")
    await session_b.call("scout_ahead", "scout", area="chapel")
    await session_a.call("roll", "roll", dice="1d20")
    await session_a.call("scout_ahead", "scout", area="chapel")
    await asyncio.gather(session_a.settle(), session_b.settle())
    cell_a.close()
    cell_b.close()

    for run, rolls, findings in ((run_a, [11, 12], "A's chapel"), (run_b, [21], "B's chapel")):
        assert {c.captured_model for c in _recordings(storage, corpus_id=run.id)} == {run.candidate_model}
        replay = _replay_session()
        replay_run = make_eval_run(
            cassette_mode="replay", cassette_corpus_id=run.id, scope_id=_SCOPE, template_id=_TEMPLATE
        )
        lane = CassetteLane.for_run(replay_run, storage)
        assert lane is not None
        _wire(lane, replay)
        assert [(await replay.call("roll", "roll", dice="1d20")).total for _ in rolls] == rolls
        await replay.call("scout_ahead", "scout", area="chapel")
        await replay.settle()
        assert [r.findings for r in replay.scout.delivered] == [findings]


# =============================================================================
# What wiring refuses — each before anything is armed
# =============================================================================


@pytest.mark.parametrize("mode", ["capture", "replay"])
def test_a_candidate_with_no_seams_is_refused(mode: str) -> None:
    storage, _ = memory_storage()
    with pytest.raises(ValueError, match="supplies no cassette seam"):
        _wire(_lane(storage, mode), FakeSeamlessSession())


def test_an_object_that_is_not_cassette_seams_is_refused() -> None:
    storage, _ = memory_storage()
    with pytest.raises(TypeError, match="is not a CassetteSeams"):
        _wire(_lane(storage, "replay"), object())


@pytest.mark.parametrize("mode", ["capture", "replay"])
def test_a_cell_is_wired_once(mode: str) -> None:
    storage, _ = memory_storage()
    cell = _wire(_lane(storage, mode), _capture_session())
    with pytest.raises(ValueError, match="already wired"):
        cell.wire(_capture_session())


def test_an_action_seam_naming_no_tool_is_refused_and_arms_nothing() -> None:
    storage, _ = memory_storage()
    session = FakeGameSession(rolls=[], rules={}, reports={}, recorded={})
    tools = session.tools
    with pytest.raises(ValueError, match="names no tool to record"):
        _wire(_lane(storage, "replay"), session)
    assert session.tools is tools and session.scout.replay is None


def test_a_tool_on_both_seams_is_refused_and_arms_nothing() -> None:
    storage, _ = memory_storage()
    session = FakeGameSession(rolls=[], rules={}, reports={}, recorded={"roll": RollResult, "scout_ahead": Ack})
    tools = session.tools
    with pytest.raises(ValueError, match="'scout_ahead' declared on both"):
        _wire(_lane(storage, "replay"), session)
    assert session.tools is tools and session.scout.replay is None


def test_a_declared_tool_the_candidate_does_not_have_is_refused() -> None:
    storage, _ = memory_storage()
    session = FakeGameSession(rolls=[], rules={}, reports={}, recorded={"roll": RollResult, "rool": RollResult})
    tools = session.tools
    with pytest.raises(ValueError, match="declares 'rool' as recorded"):
        _wire(_lane(storage, "replay"), session)
    assert session.tools is tools


class _WrapCountingSession(FakeGameSession):
    def __init__(self, applications: int) -> None:
        super().__init__(rolls=[], rules={}, reports={})
        self._applications = applications

    def arm_tools(self, wrap: ToolWrap) -> None:
        for _ in range(self._applications):
            self.tools = wrap(self.tools)


@pytest.mark.parametrize("applications", [0, 2])
def test_an_action_seam_must_apply_the_wrap_exactly_once(applications: int) -> None:
    storage, _ = memory_storage()
    with pytest.raises(ValueError, match=f"applied the cassette wrap {applications} times"):
        _wire(_lane(storage, "replay"), _WrapCountingSession(applications))


# =============================================================================
# The lane a run calls for
# =============================================================================


def test_a_run_with_cassettes_off_has_no_lane_and_a_lane_cannot_be_off() -> None:
    storage, _ = memory_storage()
    assert CassetteLane.for_run(make_eval_run(cassette_mode="off"), storage) is None
    with pytest.raises(ValueError, match="requires mode 'capture' or 'replay'"):
        _lane(storage, "off")


def test_a_capture_run_records_into_its_own_corpus() -> None:
    storage, _ = memory_storage()
    run = make_eval_run(cassette_mode="capture", scope_id="table-9")
    assert CassetteLane.for_run(run, storage) == CassetteLane(
        mode="capture", store=storage, scope_id="table-9", corpus_id=run.id
    )


def test_a_replay_run_replays_the_corpus_it_names() -> None:
    storage, _ = memory_storage()
    run = make_eval_run(cassette_mode="replay", cassette_corpus_id="run-capture-7", scope_id="table-9")
    assert CassetteLane.for_run(run, storage) == CassetteLane(
        mode="replay", store=storage, scope_id="table-9", corpus_id="run-capture-7"
    )


@pytest.mark.parametrize(("mode", "corpus"), [("replay", None), ("capture", "run-capture-7"), ("off", "run-capture-7")])
def test_a_corpus_is_named_exactly_by_a_replay(mode: str, corpus: str | None) -> None:
    with pytest.raises(ValidationError, match="cassette_corpus_id is set exactly when"):
        make_eval_run(cassette_mode=mode, cassette_corpus_id=corpus)


# =============================================================================
# The stored recording tells one story
# =============================================================================


def _key(**overrides: Any) -> CassetteKey:
    fields: dict[str, Any] = {
        "corpus_id": _CORPUS,
        "template_id": _TEMPLATE,
        "test_case_id": _CASE,
        "tool": "scout_ahead",
        "action": DELIVERY_ACTION,
        "params_hash": params_hash({"area": "chapel"}),
        "occurrence": 0,
    }
    fields.update(overrides)
    return CassetteKey(**fields)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"seam": "action"}, "records the action's result"),
        ({"seam": "action", "response": {}, "outcome": "delivered"}, "carries no delivery outcome"),
        ({"seam": "delivery"}, "records how the work ended"),
        ({"seam": "delivery", "outcome": "delivered"}, "response exactly when"),
        ({"seam": "delivery", "outcome": "undelivered", "response": {}}, "response exactly when"),
        ({"seam": "delivery", "outcome": "failed"}, "error exactly when"),
        ({"seam": "delivery", "outcome": "undelivered", "error": "x"}, "error exactly when"),
    ],
)
def test_a_recording_whose_fields_tell_two_stories_is_refused(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EvalCassette.build(_key(), scope_id=_SCOPE, captured_model=_MODEL, **fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"seam": "action", "response": {"ok": True}},
        {"seam": "delivery", "outcome": "delivered", "response": {"area": "chapel", "findings": "x"}},
        {"seam": "delivery", "outcome": "failed", "error": "ambushed"},
        {"seam": "delivery", "outcome": "undelivered"},
    ],
)
def test_a_recording_telling_one_story_is_accepted(fields: dict[str, Any]) -> None:
    assert EvalCassette.build(_key(), scope_id=_SCOPE, captured_model=_MODEL, **fields).key == _key()


def test_a_recording_whose_id_is_not_its_key_s_is_refused() -> None:
    recording = EvalCassette.build(
        _key(), scope_id=_SCOPE, captured_model=_MODEL, seam="delivery", outcome="undelivered"
    )
    with pytest.raises(ValidationError, match="is not its key's"):
        EvalCassette.model_validate({**recording.model_dump(), "occurrence": 1})
