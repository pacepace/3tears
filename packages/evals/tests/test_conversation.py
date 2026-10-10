"""``drive_conversation``: the speaker-round loop a conversing kind calls, run against a scripted table."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from packages.evals.tests.scripted_table import DONE, FakeCellSink, Raw, ScriptedTable, actor
from threetears.evals.schema.models import ROUND_DONE, ConversationSpec, ConversationStopCause
from threetears.evals.kernel.usage_capture import RoleUsageLedger
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

    cause = await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=FakeCellSink())

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
    # Stored per actor and purpose: each pick under the actor it chose (round_done under none), each line
    # under its speaker.
    assert {(row.actor_id, row.purpose): row.call_count for row in ledger.rows()} == {
        ("c", "schedule"): 4,
        ("b", "schedule"): 4,
        ("a", "schedule"): 2,
        (None, "schedule"): 5,
        ("c", "utterance"): 4,
        ("b", "utterance"): 4,
        ("a", "utterance"): 2,
    }
    assert sum(row.call_count or 0 for row in ledger.rows()) == 25


async def test_round_robin_keeps_one_utterance_and_one_answer_per_round():
    driver = _driver(actors=[actor("a"), actor("b")], max_turns=3)
    table = ScriptedTable(lines={"a": ["a1", "a2"], "b": ["b1"]})
    kind = _Kind()

    assert (
        await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=FakeCellSink())
        is ConversationStopCause.MAX_TURNS
    )
    assert kind.rounds == [["a"], ["b"], ["a"]]


async def test_a_departure_is_not_delivered_and_an_emptied_round_starts_over():
    """a leaves on round 2's only slot, so that round holds nothing to answer; b fills it instead."""
    driver = _driver(actors=[actor("a"), actor("b")], max_turns=3)
    table = ScriptedTable(lines={"a": ["a1", DONE], "b": ["b1", "b2"]})
    kind = _Kind()

    await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=FakeCellSink())

    assert kind.rounds == [["a"], ["b"], ["b"]]
    assert [turn.content for turn in kind.posted] == ["a1", "b1", "b2"]
    assert driver.departed == {"a"}


async def test_everyone_leaving_in_a_round_that_delivered_nothing_leaves_nothing_to_answer():
    """Round 1: b and then a leave before anyone speaks, so the conversation ends with round 0 answered and no more."""
    driver = _driver(actors=[actor("a"), actor("b")], turn_scheduler="llm_decided", max_speakers_per_round=2)
    table = ScriptedTable(picks=["a", ROUND_DONE, "b"], lines={"a": ["a1", DONE], "b": [DONE]})
    kind = _Kind()

    cause = await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=FakeCellSink())
    assert cause is ConversationStopCause.USER_DONE
    assert kind.rounds == [["a"]]
    assert [turn.content for turn in kind.posted] == ["a1"]


async def test_the_last_actor_leaving_mid_round_is_answered_before_the_conversation_ends():
    """One actor, two slots a round: it says a1 and then leaves in the same round, so a1 is answered first.

    Without the answer the transcript a judge reads would end on a1, a line the candidate was handed and
    never asked to answer — and here the conversation would hold no candidate turn at all.
    """
    driver = _driver(actors=[actor("a")], max_speakers_per_round=2)
    table = ScriptedTable(lines={"a": ["a1", DONE]})
    kind = _Kind()

    cause = await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=FakeCellSink())

    assert cause is ConversationStopCause.USER_DONE
    assert kind.rounds == [["a"]]
    assert driver.candidate_turns == 1
    assert driver.transcript[-1] == ("__candidate__", "answer 1")


async def test_the_opener_is_answered_when_its_actor_leaves_at_once():
    driver = _driver(actors=[actor("a", initial_utterance_template="We enter the hall.")], max_speakers_per_round=2)
    table = ScriptedTable(lines={"a": [DONE]})
    kind = _Kind()

    assert await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=FakeCellSink()) is (
        ConversationStopCause.USER_DONE
    )
    assert kind.rounds == [["a"]]
    assert [turn.content for turn in kind.posted] == ["We enter the hall."]


async def test_a_last_departure_after_others_spoke_under_the_scheduler_is_answered():
    """b speaks, a (the only other actor) left earlier, then b leaves in the same round: b1 is answered."""
    driver = _driver(actors=[actor("a"), actor("b")], turn_scheduler="llm_decided", max_speakers_per_round=3)
    # a leaves on the first pick; b is then the only legal speaker, so the second line needs no pick,
    # and the scheduler's next call picks b over round_done.
    table = ScriptedTable(picks=["a", "b"], lines={"a": [DONE], "b": ["b1", DONE]})
    kind = _Kind()

    cause = await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=FakeCellSink())

    assert cause is ConversationStopCause.USER_DONE
    assert kind.rounds == [["b"]]
    assert table.picks == []


# --- the run's cost cap, asked before every paid call ----------------------------------------------


async def test_every_paid_call_is_preceded_by_asking_the_cap_with_the_simulators_spend_so_far():
    driver = _driver(actors=[actor("a")], max_turns=2)
    table = ScriptedTable(lines={"a": ["a1", "a2"]})
    sink = FakeCellSink()
    kind = _Kind()

    await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=sink)

    # Per round: before the pick (free under round-robin, asked anyway), before the line, before the pick
    # that finds the round full, before the answer. Spend moves only with a line, a tenth of a cent each.
    assert sink.asked == pytest.approx([0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002])


async def test_the_conversation_stops_before_the_call_that_would_follow_crossing_the_cap():
    """A cap of two and a half calls: three utterances are bought, then nothing — not even the answer."""
    driver = _driver(actors=[actor("a")], max_turns=5)
    table = ScriptedTable(lines={"a": ["a1", "a2", "a3", "a4", "a5"]})
    kind = _Kind()

    cause = await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=FakeCellSink(cap_usd=0.0025))

    assert cause is ConversationStopCause.BUDGET_STOPPED
    assert table.utterance_calls == ["a", "a", "a"]
    assert kind.rounds == [["a"], ["a"]], "the third line is left unanswered: answering it is a paid call past the cap"
    assert driver.cost_usd == pytest.approx(0.003)


async def test_the_scheduler_s_picks_count_against_the_cap_too():
    driver = _driver(actors=[actor("a"), actor("b")], turn_scheduler="llm_decided", max_speakers_per_round=2)
    table = ScriptedTable(picks=["a", "b"], lines={"a": ["a1"], "b": ["b1"]})
    kind = _Kind()

    cause = await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=FakeCellSink(cap_usd=0.0015))

    assert cause is ConversationStopCause.BUDGET_STOPPED
    assert [call.purpose for call in driver.calls] == ["schedule", "utterance"]
    assert kind.rounds == []


async def test_one_scheduling_decision_can_overshoot_the_cap_by_its_repair_call_and_no_more():
    """The documented bound: the cap is asked per decision, and a refused pick buys one repair before it is asked again.

    Half a call's worth of cap, crossed by the first call: the refused pick's repair is still made (two calls),
    and nothing after it — no utterance — is bought.
    """
    driver = _driver(actors=[actor("a"), actor("b")], turn_scheduler="llm_decided", max_speakers_per_round=2)
    table = ScriptedTable(picks=[Raw("not a pick"), "a"], lines={"a": ["a1"]})
    kind = _Kind()

    cause = await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=FakeCellSink(cap_usd=0.0005))

    assert cause is ConversationStopCause.BUDGET_STOPPED
    assert [call.purpose for call in driver.calls] == ["schedule", "schedule"]
    assert table.utterance_calls == []
    assert driver.cost_usd == pytest.approx(0.002)


async def test_an_unpriced_simulator_call_stops_the_conversation_under_a_cap():
    driver = _driver(actors=[actor("a")], max_turns=3)
    table = ScriptedTable(lines={"a": ["a1", "a2"]}, cost_usd=None)
    kind = _Kind()

    cause = await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=FakeCellSink(cap_usd=100.0))

    assert cause is ConversationStopCause.BUDGET_STOPPED
    assert table.utterance_calls == ["a"]
    assert driver.cost_usd is None


async def test_a_departure_on_the_cap_leaves_the_round_unanswered_and_records_user_done():
    """The cap reached just as the last actor leaves: the answer is not bought, the stop stays user_done."""
    driver = _driver(actors=[actor("a")], max_speakers_per_round=2)
    table = ScriptedTable(lines={"a": ["a1", DONE]})
    sink = FakeCellSink(cap_usd=0.0015)
    kind = _Kind()

    cause = await drive_conversation(driver, kind.answer, kind.post, llm=table, sink=sink)

    assert cause is ConversationStopCause.USER_DONE
    assert kind.rounds == []
    assert sink.asked[-1] == pytest.approx(0.002)


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
        await drive_conversation(driver, answer, post, llm=table, sink=FakeCellSink())

    assert driver.stop_cause is cause
    assert type(caught.value).__name__ == ("SimulatorReplyInvalid" if side == "simulator" else "RuntimeError")
    # The simulator's spend is complete whichever side failed.
    assert [(call.purpose, call.actor_id) for call in driver.calls] == [("utterance", "a")]


async def test_a_driver_that_has_run_is_refused():
    driver = _driver(actors=[actor("a")])
    driver.record_candidate_turn(CandidateTurn(content="already"))
    kind = _Kind()
    with pytest.raises(ValueError, match="from its start"):
        await drive_conversation(driver, kind.answer, kind.post, llm=ScriptedTable(), sink=FakeCellSink())
