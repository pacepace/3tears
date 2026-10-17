"""The ``llm_decided`` turn scheduler: a simulator-role pick of the next speaker, or that the round is done."""

from __future__ import annotations


import pytest

from packages.evals.tests.scripted_table import DONE, Raw, ScriptedTable, actor
from threetears.evals.schema.models import ROUND_DONE, ConversationSpec, ConversationStopCause
from threetears.evals.run.simulator import (
    SCHEDULER_CALL_ATTEMPTS,
    CandidateTurn,
    SimulatorReplyInvalid,
    TurnDriver,
    next_speaker_response_format,
)


def _driver(actor_ids: list[str] = ["a", "b", "c"], **overrides) -> TurnDriver:  # noqa: B006 - read only
    fields = dict(
        actors=[actor(actor_id) for actor_id in actor_ids],
        turn_scheduler="llm_decided",
        max_speakers_per_round=3,
        max_turns=5,
    )
    fields.update(overrides)
    return TurnDriver(conversation=ConversationSpec(**fields), variation={})


async def _round(driver: TurnDriver, table: ScriptedTable) -> list[str]:
    """Run one round to its end and return who spoke, delivered or not."""
    speakers = []
    while (speaker := await driver.next_speaker(table)) is not None:
        await driver.next_user_turn(table, speaker)
        speakers.append(speaker.id)
    return speakers


async def test_the_scheduler_chooses_the_order_and_ends_the_round():
    """The model's picks decide who speaks: c then a, then the round is done — no round-robin order.

    Reinstating the round-robin fallback for ``llm_decided`` makes this a, b, c with no scheduling
    call at all, and both assertions go red.
    """
    driver = _driver()
    table = ScriptedTable(picks=["c", "a", ROUND_DONE], lines={"a": ["from a"], "c": ["from c"]})

    assert await _round(driver, table) == ["c", "a"]
    assert [legal for _prompt, legal in table.schedule_calls] == [
        ["a", "b", "c"],
        ["a", "b", "c", ROUND_DONE],
        ["a", "b", "c", ROUND_DONE],
    ]


async def test_round_done_is_not_offered_before_anyone_has_spoken():
    """A round the candidate answers holds at least one utterance, so the first pick must name an actor."""
    driver = _driver()
    table = ScriptedTable(picks=["b"], lines={"b": ["hi"]})
    await driver.next_user_turn(table, await driver.next_speaker(table))

    first_prompt, first_legal = table.schedule_calls[0]
    assert ROUND_DONE not in first_legal
    assert "an actor must speak" in first_prompt
    assert next_speaker_response_format(first_legal)["json_schema"]["strict"] is True


async def test_a_full_round_ends_without_asking():
    """At ``max_speakers_per_round`` the round is over; there is nothing to decide and nothing to pay for."""
    driver = _driver(max_speakers_per_round=2)
    table = ScriptedTable(picks=["b", "b"], lines={"b": ["one", "two"]})

    assert await _round(driver, table) == ["b", "b"]
    assert len(table.schedule_calls) == 2


async def test_one_legal_answer_makes_no_call():
    """One actor left and nothing said yet: the only legal answer is that actor, so no model is asked."""
    driver = _driver(["solo"], max_speakers_per_round=1)
    table = ScriptedTable(lines={"solo": ["hello"]})

    assert await _round(driver, table) == ["solo"]
    assert table.schedule_calls == []


async def test_a_bad_reply_gets_one_repair_naming_what_was_wrong():
    driver = _driver()
    table = ScriptedTable(picks=[Raw("the rules lawyer, obviously"), "b"], lines={"b": ["hi"]})

    speaker = await driver.next_speaker(table)

    assert speaker is not None and speaker.id == "b"
    (first, _), (repair, _) = table.schedule_calls
    assert "refused" not in first
    assert "Your previous reply was refused: it was not a JSON object" in repair
    # Both calls were billed; the refused one chose nobody, the repair chose b.
    assert [(call.purpose, call.actor_id) for call in driver.calls] == [("schedule", None), ("schedule", "b")]


@pytest.mark.parametrize("illegal", ["zed", ROUND_DONE], ids=["unknown-actor", "round-done-on-first-pick"])
async def test_an_answer_outside_the_legal_set_is_refused_and_repaired(illegal):
    driver = _driver()
    table = ScriptedTable(picks=[illegal, "a"], lines={})

    speaker = await driver.next_speaker(table)

    assert speaker is not None and speaker.id == "a"
    _, (repair, _) = table.schedule_calls
    assert f'`next` was "{illegal}", which is not one of' in repair


async def test_a_second_bad_reply_is_a_simulator_fault_with_both_calls_recorded():
    driver = _driver()
    table = ScriptedTable(picks=[Raw("{}"), "zed"])

    with pytest.raises(SimulatorReplyInvalid, match="refused 2 times"):
        await driver.next_speaker(table)

    assert SCHEDULER_CALL_ATTEMPTS == 2
    assert len(table.schedule_calls) == SCHEDULER_CALL_ATTEMPTS
    assert [(call.purpose, call.actor_id) for call in driver.calls] == [("schedule", None), ("schedule", None)]


async def test_every_call_is_ledgered_against_the_actor_it_produced_or_chose():
    driver = _driver()
    table = ScriptedTable(picks=["c", ROUND_DONE], lines={"c": ["from c"]})

    await _round(driver, table)

    assert [(call.purpose, call.actor_id, call.round_index) for call in driver.calls] == [
        ("schedule", "c", 0),
        ("utterance", "c", 0),
        ("schedule", None, 0),
    ]
    assert all(call.usage.cost_usd == 0.001 for call in driver.calls)


async def test_an_actor_that_leaves_is_no_longer_offered():
    driver = _driver()
    table = ScriptedTable(picks=["b", "a", "c"], lines={"b": [DONE], "a": ["still here"]})

    await driver.next_user_turn(table, await driver.next_speaker(table))
    await driver.next_user_turn(table, await driver.next_speaker(table))
    await driver.next_speaker(table)

    assert driver.departed == {"b"}
    assert [legal for _prompt, legal in table.schedule_calls] == [
        ["a", "b", "c"],
        ["a", "c"],
        ["a", "c", ROUND_DONE],
    ]


async def test_the_last_actor_leaving_ends_the_conversation():
    driver = _driver(["a", "b"])
    table = ScriptedTable(picks=["a", "b"], lines={"a": [DONE], "b": [DONE]})

    await driver.next_user_turn(table, await driver.next_speaker(table))
    assert driver.should_continue() is True
    await driver.next_user_turn(table, await driver.next_speaker(table))

    assert driver.stop_cause is ConversationStopCause.USER_DONE
    assert driver.transcript == []


# =============================================================================
# Refusals
# =============================================================================


async def test_a_stopped_conversation_names_no_next_speaker():
    driver = _driver()
    driver.stop(ConversationStopCause.USER_DONE)
    with pytest.raises(ValueError, match="nobody speaks next"):
        await driver.next_speaker(ScriptedTable())


async def test_an_actor_from_another_conversation_cannot_speak():
    with pytest.raises(ValueError, match="not one of this conversation's actors"):
        await _driver().next_user_turn(ScriptedTable(), actor("a"))


async def test_an_actor_that_left_cannot_speak():
    driver = _driver()
    table = ScriptedTable(picks=["a"], lines={"a": [DONE]})
    leaver = await driver.next_speaker(table)
    await driver.next_user_turn(table, leaver)

    with pytest.raises(ValueError, match="has left the conversation"):
        await driver.next_user_turn(table, leaver)


async def test_the_opener_opens_or_not_at_all():
    driver = _driver(actors=[actor("a", initial_utterance_template="hi"), actor("b")])
    driver.record_candidate_turn(CandidateTurn(content="hello?"))
    with pytest.raises(ValueError, match="already taken turns"):
        driver.initial_utterance()


@pytest.mark.parametrize("reserved", [ROUND_DONE, "__candidate__"])
def test_a_reserved_actor_id_is_refused(reserved):
    with pytest.raises(ValueError, match="is reserved"):
        ConversationSpec(actors=[actor("a"), actor(reserved)])


def test_two_actors_with_one_id_are_refused():
    with pytest.raises(ValueError, match="appears twice"):
        ConversationSpec(actors=[actor("a"), actor("a", policy="the other a")])
