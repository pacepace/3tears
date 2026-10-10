"""The action seam records and replays the same way whether the candidate awaits its tools or blocks on them.

A host whose tools run as plain blocking calls inside its own turn loop declares a
:class:`~threetears.evals.kernel.cassettes.SyncActionSeam` and drives its tools through ``act_sync``;
one that awaits declares an :class:`~threetears.evals.kernel.cassettes.ActionSeam` and calls ``act``.
Every behaviour below runs on BOTH paths over the same fixture — the same tools, the same answers, the
same session — so a rule that holds on one and not the other fails here. The tools answer both ways
from one answer function, and the replay table is built so every live answer differs from the captured
one, so a replay that went live would be visible.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from packages.evals.tests.factories import memory_storage
from threetears.evals.kernel import SyncActionSeam, SyncToolLike, SyncToolWrap, ToolWrap
from threetears.evals.kernel.cassettes import (
    ActionSeam,
    CassetteCorrupt,
    CassetteExhausted,
    CassetteMiss,
    CassetteSeams,
    DeliverySeam,
)
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.run.cassette_proxy import CassetteCell, CassetteLane, CassetteProxy

_SCOPE = "table-1"
_TEMPLATE = "tpl-goblin-ambush"
_CASE = "case-1"
_MODEL = "gm-model-a"

FLAVOURS = ("awaited", "blocking")


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


# parity-with: threetears.evals.kernel.cassettes.SyncToolLike
class FakeDualTool:
    """A tool answering both ways from one answer function, logging every live call on either."""

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

    def act_sync(self, action: str, parameters: dict[str, Any]) -> Any:
        self.live_calls.append((action, dict(parameters)))
        if action not in self._actions:
            return RollResult(ok=False, error=f"Unknown action: {action}")
        return self._answer(parameters)

    async def act(self, action: str, parameters: dict[str, Any]) -> Any:
        return self.act_sync(action, parameters)


class _Table:
    """A game-master session's tools and the seams it hands the lane — everything but the arming method."""

    blocking: bool

    def __init__(
        self, *, rolls: list[int], rules: Mapping[str, str], recorded: Mapping[str, type[BaseModel]] | None = None
    ) -> None:
        totals = iter(rolls)
        self.roll = FakeDualTool("roll", frozenset({"roll"}), lambda p: RollResult(ok=True, total=next(totals)))
        self.rules = FakeDualTool(
            "rules_lookup", frozenset({"lookup"}), lambda p: RulesAnswer(ok=True, text=rules[p["topic"]])
        )
        self.tools: dict[str, Any] = {"roll": self.roll, "rules_lookup": self.rules}
        self._recorded = dict(recorded if recorded is not None else {"roll": RollResult, "rules_lookup": RulesAnswer})

    @property
    def action_seam(self) -> Any:
        return self

    @property
    def delivery_seams(self) -> Mapping[str, DeliverySeam]:
        return {}

    @property
    def recorded_tools(self) -> Mapping[str, type[BaseModel]]:
        return self._recorded

    async def call(self, tool: str, action: str, **parameters: Any) -> Any:
        """One call, the way this table's candidate makes it — awaited, or blocking with no await at all."""
        if self.blocking:
            return self.tools[tool].act_sync(action, parameters)
        return await self.tools[tool].act(action, parameters)


class FakeAwaitedTable(_Table, ActionSeam, CassetteSeams):
    """A candidate that awaits its tools."""

    blocking = False

    def arm_tools(self, wrap: ToolWrap) -> None:
        self.tools = wrap(self.tools)


class FakeBlockingTable(_Table, SyncActionSeam, CassetteSeams):
    """A candidate whose turn loop calls its tools as plain blocking calls."""

    blocking = True

    def arm_sync_tools(self, wrap: SyncToolWrap) -> None:
        self.tools = wrap(self.tools)


_TABLES: dict[str, type[_Table]] = {"awaited": FakeAwaitedTable, "blocking": FakeBlockingTable}


def _capture_table(flavour: str, **overrides: Any) -> _Table:
    return _TABLES[flavour](rolls=[14, 7, 3], rules={"grapple": "Contested Athletics check."}, **overrides)


def _replay_table(flavour: str, **overrides: Any) -> _Table:
    """The same table, built so every live answer differs from the captured one."""
    return _TABLES[flavour](rolls=[1, 1, 1], rules={"grapple": "WRONG — a live lookup"}, **overrides)


def _lane(storage: EvalStorage, mode: str, *, corpus_id: str = "run-capture-1") -> CassetteLane:
    return CassetteLane(mode=mode, store=storage, scope_id=_SCOPE, corpus_id=corpus_id)  # type: ignore[arg-type]


def _wire(lane: CassetteLane, seams: Any) -> CassetteCell:
    cell = lane.cell(template_id=_TEMPLATE, test_case_id=_CASE, model=_MODEL)
    cell.wire(seams)
    return cell


async def _play(table: _Table) -> list[Any]:
    """Two rolls of the same dice, one of different dice, and a rules lookup."""
    return [
        await table.call("roll", "roll", dice="1d20"),
        await table.call("roll", "roll", dice="1d20"),
        await table.call("roll", "roll", dice="2d6"),
        await table.call("rules_lookup", "lookup", topic="grapple"),
    ]


async def _captured(storage: EvalStorage, flavour: str, *, corpus_id: str = "run-capture-1") -> list[Any]:
    table = _capture_table(flavour)
    cell = _wire(_lane(storage, "capture", corpus_id=corpus_id), table)
    answers = await _play(table)
    cell.close()
    return answers


# =============================================================================
# Capture and replay behave identically on both paths
# =============================================================================


@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_a_replay_serves_the_capture_and_runs_nothing_live(flavour: str) -> None:
    storage, _ = memory_storage()
    captured = await _captured(storage, flavour)

    replay = _replay_table(flavour)
    _wire(_lane(storage, "replay"), replay)

    assert await _play(replay) == captured
    assert [answer.total for answer in captured[:3]] == [14, 7, 3]
    assert replay.roll.live_calls == [] and replay.rules.live_calls == []


@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_asking_once_more_than_the_capture_did_is_exhausted_and_never_live(flavour: str) -> None:
    storage, _ = memory_storage()
    await _captured(storage, flavour)
    replay = _replay_table(flavour)
    _wire(_lane(storage, "replay"), replay)
    await _play(replay)

    with pytest.raises(CassetteExhausted) as caught:
        await replay.call("roll", "roll", dice="1d20")
    assert caught.value.recorded == 2
    assert replay.roll.live_calls == []


@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_an_unrecorded_call_is_a_miss_and_never_runs_live(flavour: str) -> None:
    storage, _ = memory_storage()
    await _captured(storage, flavour)
    replay = _replay_table(flavour)
    _wire(_lane(storage, "replay"), replay)

    with pytest.raises(CassetteMiss) as caught:
        await replay.call("rules_lookup", "lookup", topic="flanking")
    assert (caught.value.key.tool, caught.value.key.occurrence) == ("rules_lookup", 0)
    assert replay.rules.live_calls == []


@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_an_undispatchable_action_gets_the_tool_s_own_answer_and_is_never_recorded(flavour: str) -> None:
    storage, _ = memory_storage()
    table = _capture_table(flavour)
    _wire(_lane(storage, "capture"), table)

    answer = await table.call("roll", "reroll", dice="1d20")

    assert answer == RollResult(ok=False, error="Unknown action: reroll")
    assert (
        storage.list_case_cassettes(
            corpus_id="run-capture-1", template_id=_TEMPLATE, test_case_id=_CASE, scope_id=_SCOPE
        )
        == []
    )


@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_a_recording_that_no_longer_rebuilds_as_its_type_is_corrupt(flavour: str) -> None:
    storage, _ = memory_storage()
    await _captured(storage, flavour)
    replay = _replay_table(flavour, recorded={"roll": RulesAnswer, "rules_lookup": RulesAnswer})
    _wire(_lane(storage, "replay"), replay)

    with pytest.raises(CassetteCorrupt, match="does not rebuild as RulesAnswer"):
        await replay.call("roll", "roll", dice="1d20")


async def test_both_paths_record_the_same_corpus() -> None:
    storage, _ = memory_storage()
    await _captured(storage, "awaited", corpus_id="corpus-awaited")
    await _captured(storage, "blocking", corpus_id="corpus-blocking")

    def recorded(corpus_id: str) -> list[tuple[Any, ...]]:
        rows = storage.list_case_cassettes(
            corpus_id=corpus_id, template_id=_TEMPLATE, test_case_id=_CASE, scope_id=_SCOPE
        )
        return sorted(
            (row.tool, row.action, row.params_hash, row.occurrence, row.seam, str(row.response)) for row in rows
        )

    assert len(recorded("corpus-awaited")) == 4
    assert recorded("corpus-awaited") == recorded("corpus-blocking")


@pytest.mark.parametrize(("captured_on", "replayed_on"), [("awaited", "blocking"), ("blocking", "awaited")])
async def test_a_corpus_captured_on_one_path_replays_on_the_other(captured_on: str, replayed_on: str) -> None:
    storage, _ = memory_storage()
    captured = await _captured(storage, captured_on)
    replay = _replay_table(replayed_on)
    _wire(_lane(storage, "replay"), replay)

    assert await _play(replay) == captured
    assert replay.roll.live_calls == []


def test_the_blocking_path_needs_no_event_loop() -> None:
    """A turn loop with no event loop running captures and replays through act_sync alone."""
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()
    storage, _ = memory_storage()
    table = _capture_table("blocking")
    cell = _wire(_lane(storage, "capture"), table)
    captured = [table.tools["roll"].act_sync("roll", {"dice": "1d20"}) for _ in range(2)]
    cell.close()

    replay = _replay_table("blocking")
    _wire(_lane(storage, "replay"), replay)
    assert [replay.tools["roll"].act_sync("roll", {"dice": "1d20"}) for _ in range(2)] == captured
    assert replay.roll.live_calls == []


def test_the_blocking_proxy_is_the_tool_to_the_session_s_own_machinery() -> None:
    storage, _ = memory_storage()
    table = _capture_table("blocking")
    _wire(_lane(storage, "capture"), table)

    proxy = table.tools["roll"]
    assert isinstance(proxy, CassetteProxy) and isinstance(proxy, SyncToolLike)
    assert proxy.name == "roll"
    assert proxy.description == FakeDualTool.description


# =============================================================================
# Refusals
# =============================================================================


class FakeBothFlavoursTable(_Table, ActionSeam, SyncActionSeam, CassetteSeams):
    """A seam declaring both arming methods: the cell could not know which entry point the candidate calls."""

    blocking = True

    def arm_tools(self, wrap: ToolWrap) -> None:
        self.tools = wrap(self.tools)

    def arm_sync_tools(self, wrap: SyncToolWrap) -> None:
        self.tools = wrap(self.tools)


class FakeNeitherFlavourTable(_Table, CassetteSeams):
    """A seam declaring recorded tools and no way to arm them."""

    blocking = True


@pytest.mark.parametrize("mode", ["capture", "replay"])
def test_a_seam_declaring_both_flavours_is_refused_and_arms_nothing(mode: str) -> None:
    storage, _ = memory_storage()
    table = FakeBothFlavoursTable(rolls=[], rules={})
    tools = table.tools
    with pytest.raises(ValueError, match="both an ActionSeam .* and a SyncActionSeam"):
        _wire(_lane(storage, mode), table)
    assert table.tools is tools


def test_a_seam_declaring_neither_flavour_is_refused() -> None:
    storage, _ = memory_storage()
    table = FakeNeitherFlavourTable(rolls=[], rules={})
    with pytest.raises(TypeError, match="neither an ActionSeam nor a SyncActionSeam"):
        _wire(_lane(storage, "replay"), table)


@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_the_entry_point_the_seam_did_not_bind_is_refused_before_anything_is_keyed(flavour: str) -> None:
    """Calling the other entry point never falls through to the live tool, and consumes no occurrence."""
    storage, _ = memory_storage()
    captured = await _captured(storage, flavour)
    replay = _replay_table(flavour)
    _wire(_lane(storage, "replay"), replay)
    proxy = replay.tools["roll"]

    if flavour == "blocking":
        with pytest.raises(TypeError, match="wired on a SyncActionSeam; its calls go through act_sync"):
            await proxy.act("roll", {"dice": "1d20"})
    else:
        with pytest.raises(TypeError, match="wired on an ActionSeam; its calls go through act"):
            proxy.act_sync("roll", {"dice": "1d20"})

    assert replay.roll.live_calls == []
    # The refused call took no occurrence: the first real call is still served the first recording.
    assert await replay.call("roll", "roll", dice="1d20") == captured[0]


@pytest.mark.parametrize("bound", [(), ("act", "act_sync")])
def test_a_proxy_is_bound_to_exactly_one_entry_point(bound: tuple[str, ...]) -> None:
    async def act(action: str, parameters: dict[str, Any]) -> Any:
        return None

    def act_sync(action: str, parameters: dict[str, Any]) -> Any:
        return None

    entry_points: dict[str, Any] = {"act": act, "act_sync": act_sync}
    tool = FakeDualTool("roll", frozenset({"roll"}), lambda p: RollResult(ok=True))
    with pytest.raises(ValueError, match="exactly one entry point"):
        CassetteProxy(tool, **{name: entry_points[name] for name in bound})
    # One of them, either one, is a proxy.
    assert CassetteProxy(tool, act=act).name == CassetteProxy(tool, act_sync=act_sync).name == "roll"
