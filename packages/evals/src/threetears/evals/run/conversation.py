"""``drive_conversation``: the speaker-round loop every conversing kind runs, written once.

A conversing kind has two things to supply and nothing else: how a simulated utterance reaches its
candidate's world (``post_user_turn``), and how its candidate answers a round (``candidate_turn``).
The loop between them — the opener, who speaks next, who has left, when the round is done, when a
session breaks, when the conversation stops — is the template's ``conversation`` block run by a
:class:`~threetears.evals.run.simulator.TurnDriver`, and a kind that wrote its own copy of it would be
a second answer to what the template declared.

**Which side failed is recorded, not inferred.** Each call out of the loop is one of three sides —
the simulator, the kind's delivery, the candidate — and an exception from one stops the driver with
that side's :class:`~threetears.evals.contracts.models.ConversationStopCause` (``simulator_error``,
``apparatus_error``, ``candidate_error``) before it propagates unchanged. The kind reads
``driver.stop_cause`` to know which side it was, and classifies the exception itself (only it knows
its candidate's error types); it reads ``driver.calls`` for the simulator's spend, which is complete
whichever way the loop ended. Cancellation is not a fault and records nothing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from threetears.evals.contracts.models import ConversationStopCause
from threetears.evals.contracts.provider import SimulatorLLM
from threetears.evals.run.simulator import CandidateTurn, SimulatorTurn, TurnDriver


async def drive_conversation(
    driver: TurnDriver,
    candidate_turn: Callable[[Sequence[SimulatorTurn]], Awaitable[CandidateTurn]],
    post_user_turn: Callable[[SimulatorTurn], Awaitable[None]],
    *,
    llm: SimulatorLLM,
) -> ConversationStopCause:
    """Run ``driver``'s conversation from its first turn until it stops on a structural signal.

    Each round: the driver names speakers until the round is done; each delivered utterance goes to
    ``post_user_turn`` in order (one whose ``session_break`` is set is the first of a new session, and
    the kind starts one before delivering it); then ``candidate_turn`` answers the round's utterances
    and its answer is recorded. A reply in which an actor says ``done`` is not delivered.

    Args:
        driver: A fresh driver over the template's ``conversation`` block.
        candidate_turn: The candidate's answer to one round, handed that round's delivered utterances.
        post_user_turn: Delivers one simulated utterance into the candidate's world.
        llm: The simulator-role client, for utterances and ``llm_decided`` picks alike.

    Returns:
        Why the conversation stopped: ``max_turns`` or ``user_done``.

    Raises:
        ValueError: ``driver`` has already taken a turn; this drives a conversation from its start.
    """
    if driver.user_turns or driver.candidate_turns or driver.stop_cause is not None:
        raise ValueError("drive_conversation runs a conversation from its start, and this driver has already run")
    round_turns: list[SimulatorTurn] = []
    opener = driver.initial_utterance()
    if opener is not None:
        await _side(driver, ConversationStopCause.APPARATUS_ERROR, post_user_turn(opener))
        round_turns.append(opener)
    while driver.should_continue():
        while (
            actor := await _side(driver, ConversationStopCause.SIMULATOR_ERROR, driver.next_speaker(llm))
        ) is not None:
            turn = await _side(driver, ConversationStopCause.SIMULATOR_ERROR, driver.next_user_turn(llm, actor))
            if turn.done:
                if not driver.should_continue():
                    break
                continue
            await _side(driver, ConversationStopCause.APPARATUS_ERROR, post_user_turn(turn))
            round_turns.append(turn)
        if not driver.should_continue():
            break
        answer = await _side(driver, ConversationStopCause.CANDIDATE_ERROR, candidate_turn(tuple(round_turns)))
        driver.record_candidate_turn(answer)
        round_turns = []
    cause = driver.stop_cause
    assert cause is not None, "should_continue() records a cause whenever it returns False"
    return cause


async def _side[T](driver: TurnDriver, cause: ConversationStopCause, call: Awaitable[T]) -> T:
    """Await one side's call; if it raises, stop the driver under that side's cause and re-raise."""
    try:
        return await call
    except Exception:
        driver.stop(cause)
        raise


__all__ = ["drive_conversation"]
