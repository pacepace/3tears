"""``drive_conversation``: the speaker-round loop a conversing kind calls, run against a scripted table."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from packages.evals.tests.scripted_table import DONE, Raw, ScriptedTable, actor
from threetears.evals.contracts.models import ROUND_DONE, ConversationSpec, ConversationStopCause
from threetears.evals.contracts.usage_capture import RoleUsageLedger
from threetears.evals.run import drive_conversation
from threetears.evals.run.simulator import CandidateTurn, SimulatorTurn, TurnDriver


class _Kind:
    """The two things a conversing kind supplies: delivery and the candidate's answer, both recorded."""

    def __init__(self) -> None:
        self.posted: list[SimulatorTurn] = []
        self.rounds: list[list[str]] = []
        self.sessions_started = 1

    async def post(self, turn: SimulatorTurn) -> None:
        if turn.session_break:
            self.sessions_started += 1
        self.posted.append(turn)

    async def answer(self, round_turns: Sequence[SimulatorTurn]) -> CandidateTurn:
        self.rounds.append([turn.actor_id for turn in round_turns])
        return CandidateTurn(content=f"answer {len(self.rounds)}")


def _driver(**fields) -> TurnDriver:
    return TurnDriver(conversation=ConversationSpec(**fields), variation={})


async def test_a_three_actor_table_runs_in_the_schedulers_order_across_three_sessions():
    """The Done-when table: three scripted actors, an ``llm_decided`` order no rotation produces, two breaks.

    Round-robin would give [a], [b], [c], ... one actor per round; the scheduler here fills rounds
    of different sizes in its own order. Reinstating the round-robin fallback makes every round
    assertion below go red.
    """
    driver = _driver(
        actors=[
            actor("a", initial_utterance_template="We enter the hall."),
            actor("b"),
            actor("c"),
        ],
        turn_scheduler="llm_decided",
        max_speakers_per_round=3,
        max_turns=6,
        sessions=3,
    )
    table = ScriptedTable(
        picks=[
            *["c", ROUND_DONE],  # round 0: a (the opener), c
            *["b", "a", "b"],  # round 1: b, a, b — full at three, so no round_done pick
            *["c", ROUND_DONE],  # round 2: c — the first line of session 1
            *["a", "c", ROUND_DONE],  # round 3: a, c
            *["b", ROUND_DONE],  # round 4: b — the first line of session 2
            *["c", "b", ROUND_DONE],  # round 5: c, b
        ],
        lines={"a": ["a1", "a2", "a3"], "b": ["b1", "b2", "b3", "b4", "b5"], "c": ["c1", "c2", "c3", "c4", "c5"]},
    )
    kind = _Kind()

    cause = await drive_conversation(driver, kind.answer, kind.post, llm=table)

    assert cause is ConversationStopCause.MAX_TURNS
    assert kind.rounds == [["a", "c"], ["b", "a", "b"], ["c"], ["a", "c"], ["b"], ["c", "b"]]
    assert [turn.round_index for turn in kind.posted if turn.session_break] == [2, 4]
    assert kind.sessions_started == 3
    assert table.picks == [], "every scripted pick was asked for"
    # Spend: one call per pick and per spoken line, all of it the simulator's.
    # 15 picks (round 1 filled its three slots, so it asked no round_done) and 10 spoken lines (the
    # opener is a template and calls nothing).
    assert sum(call.purpose == "schedule" for call in driver.calls) == 15
    assert sum(call.purpose == "utterance" for call in driver.calls) == 10
    ledger = RoleUsageLedger(role="simulator")
    driver.fold_usage(ledger)
    assert ledger.rows()[0].call_count == 25


async def test_round_robin_keeps_one_utterance_and_one_answer_per_round():
    driver = _driver(actors=[actor("a"), actor("b")], max_turns=3)
    table = ScriptedTable(lines={"a": ["a1", "a2"], "b": ["b1"]})
    kind = _Kind()

    assert await drive_conversation(driver, kind.answer, kind.post, llm=table) is ConversationStopCause.MAX_TURNS
    assert kind.rounds == [["a"], ["b"], ["a"]]


async def test_a_departure_is_not_delivered_and_an_emptied_round_starts_over():
    """a leaves on round 2's only slot, so that round holds nothing to answer; b fills it instead."""
    driver = _driver(actors=[actor("a"), actor("b")], max_turns=3)
    table = ScriptedTable(lines={"a": ["a1", DONE], "b": ["b1", "b2"]})
    kind = _Kind()

    await drive_conversation(driver, kind.answer, kind.post, llm=table)

    assert kind.rounds == [["a"], ["b"], ["b"]]
    assert [turn.content for turn in kind.posted] == ["a1", "b1", "b2"]
    assert driver.departed == {"a"}


async def test_everyone_leaving_ends_the_conversation_unanswered():
    driver = _driver(actors=[actor("a"), actor("b")], turn_scheduler="llm_decided", max_speakers_per_round=2)
    table = ScriptedTable(picks=["a", ROUND_DONE, "b"], lines={"a": ["a1", DONE], "b": [DONE]})
    kind = _Kind()

    assert await drive_conversation(driver, kind.answer, kind.post, llm=table) is ConversationStopCause.USER_DONE
    assert kind.rounds == [["a"]]
    assert [turn.content for turn in kind.posted] == ["a1"]


async def _raising(*_args) -> None:
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    ("side", "cause"),
    [
        ("simulator", ConversationStopCause.SIMULATOR_ERROR),
        ("post", ConversationStopCause.APPARATUS_ERROR),
        ("candidate", ConversationStopCause.CANDIDATE_ERROR),
    ],
)
async def test_the_side_that_failed_is_recorded_and_its_exception_propagates(side, cause):
    driver = _driver(actors=[actor("a")], max_turns=2)
    table = ScriptedTable(lines={"a": [Raw("not json") if side == "simulator" else "a1"]})
    kind = _Kind()
    post = _raising if side == "post" else kind.post
    answer = _raising if side == "candidate" else kind.answer

    with pytest.raises(Exception) as caught:
        await drive_conversation(driver, answer, post, llm=table)

    assert driver.stop_cause is cause
    assert type(caught.value).__name__ == ("SimulatorReplyInvalid" if side == "simulator" else "RuntimeError")
    # The simulator's spend is complete whichever side failed.
    assert [(call.purpose, call.actor_id) for call in driver.calls] == [("utterance", "a")]


async def test_a_driver_that_has_run_is_refused():
    driver = _driver(actors=[actor("a")])
    driver.record_candidate_turn(CandidateTurn(content="already"))
    kind = _Kind()
    with pytest.raises(ValueError, match="from its start"):
        await drive_conversation(driver, kind.answer, kind.post, llm=ScriptedTable())
