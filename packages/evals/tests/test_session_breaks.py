"""Session breaks: ``conversation.sessions`` sittings, each boundary marked by the driver itself."""

from __future__ import annotations

import pytest

from packages.evals.tests.scripted_table import DONE, ScriptedTable, actor
from threetears.evals.schema.models import ConversationSpec
from threetears.evals.run.simulator import CandidateTurn, SimulatorTurn, TurnDriver


def _driver(*, max_turns: int, sessions: int, actor_ids: tuple[str, ...] = ("a",)) -> TurnDriver:
    conversation = ConversationSpec(
        actors=[actor(actor_id) for actor_id in actor_ids], max_turns=max_turns, sessions=sessions
    )
    return TurnDriver(conversation=conversation, variation={})


async def _rounds(driver: TurnDriver, table: ScriptedTable) -> list[SimulatorTurn]:
    """Every delivered turn of a round-robin conversation run to its turn budget, one per round."""
    delivered = []
    while driver.should_continue():
        while (speaker := await driver.next_speaker(table)) is not None:
            turn = await driver.next_user_turn(table, speaker)
            if not turn.done:
                delivered.append(turn)
        driver.record_candidate_turn(CandidateTurn(content="ok"))
    return delivered


@pytest.mark.parametrize(
    ("max_turns", "sessions", "breaks_before_round"),
    [(6, 3, [2, 4]), (5, 2, [2]), (4, 4, [1, 2, 3]), (6, 1, [])],
    ids=["even", "uneven", "one-turn-sessions", "one-session"],
)
async def test_the_breaks_fall_where_sessions_places_them(max_turns, sessions, breaks_before_round):
    driver = _driver(max_turns=max_turns, sessions=sessions)
    table = ScriptedTable(lines={"a": [f"line {index}" for index in range(max_turns)]})

    turns = await _rounds(driver, table)

    assert [turn.round_index for turn in turns if turn.session_break] == breaks_before_round
    assert driver.session_index == sessions - 1
    # Each turn's session is the number of breaks at or before it.
    assert [turn.session_index for turn in turns] == [
        sum(1 for start in breaks_before_round if start <= turn.round_index) for turn in turns
    ]


async def test_a_break_rides_the_first_delivered_turn_not_a_departure():
    """An actor leaving right after a boundary says nothing to the candidate, so the break waits for one who speaks."""
    driver = _driver(max_turns=4, sessions=2, actor_ids=("a", "b"))
    table = ScriptedTable(lines={"a": ["a1", DONE], "b": ["b1", "b2", "b3"]})

    turns = await _rounds(driver, table)

    # Round 2 opens session 1: a's slot ends in its departure, so the round starts over and b's line
    # is the first the candidate hears in the new session.
    assert [(turn.actor_id, turn.content, turn.session_break) for turn in turns] == [
        ("a", "a1", False),
        ("b", "b1", False),
        ("b", "b2", True),
        ("b", "b3", False),
    ]
    assert driver.departed == {"a"}


async def test_the_simulator_transcript_spans_the_break():
    """Only the candidate's history is the kind's to clear; the actors stay coherent across the break."""
    driver = _driver(max_turns=2, sessions=2)
    table = ScriptedTable(lines={"a": ["pre-break turn", "post-break turn"]})

    turns = await _rounds(driver, table)

    assert driver.transcript == [
        ("a", "pre-break turn"),
        ("__candidate__", "ok"),
        ("a", "post-break turn"),
        ("__candidate__", "ok"),
    ]
    assert [turn.session_break for turn in turns] == [False, True]


def test_nothing_outside_the_driver_can_mark_a_break():
    """The breaks a run holds are the ones its template declared: there is no public way to add one."""
    assert not hasattr(TurnDriver, "mark_session_break")


def test_more_sessions_than_turns_is_refused():
    with pytest.raises(ValueError, match="sessions=3 exceeds max_turns=2"):
        ConversationSpec(actors=[actor("a")], max_turns=2, sessions=3)
