"""``drive_conversation``: the speaker-round loop every conversing kind runs, written once.

A conversing kind has two things to supply, and hands over its cell's sink: how a simulated utterance
reaches its candidate's world (``post_user_turn``), and how its candidate answers a round (``candidate_turn``).
The loop between them — the opener, who speaks next, who has left, when the round is done, when a
session breaks, when the conversation stops — is the template's ``conversation`` block run by a
:class:`~threetears.evals.run.simulator.TurnDriver`, and a kind that wrote its own copy of it would be
a second answer to what the template declared.

**Which side failed is recorded, not inferred.** Each call out of the loop is one of three sides —
the simulator, the kind's delivery, the candidate — and an exception from one stops the driver with
that side's :class:`~threetears.evals.schema.models.ConversationStopCause` (``simulator_error``,
``apparatus_error``, ``candidate_error``) before it propagates unchanged. The kind reads
``driver.stop_cause`` to know which side it was, and classifies the exception itself (only it knows
its candidate's error types); it reads ``driver.calls`` for the simulator's spend, which is complete
whichever way the loop ended. Cancellation is not a fault and records nothing.

**No ceiling is asked inside the loop.** The run's cost cap and its metered-call ceiling are read between
cells, against what the recorded cells landed, and a cell already running finishes whole
(:func:`~threetears.evals.run.runner.execute_run`). So the loop makes no decision on a cap: every simulator call
goes through the cell's sink (:class:`~threetears.evals.kernel.capped_completion.CappedCompletion`), which settles
what it cost to the cell's spend, and a conversation that runs long is bounded by its turn budget and the cell
timeout, not by a ceiling. The simulator's own spend bound is in :mod:`~threetears.evals.run.simulator`'s *Spend*.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from threetears.evals.kernel.candidate_kind import CellSink
from threetears.evals.kernel.capped_completion import CappedCompletion
from threetears.evals.schema.models import ConversationStopCause
from threetears.evals.kernel.world_session import WorldSession
from threetears.evals.schema.completion import SimulatorLLM
from threetears.evals.run.simulator import CandidateTurn, SimulatorTurn, TurnDriver


async def drive_conversation(
    driver: TurnDriver,
    candidate_turn: Callable[[Sequence[SimulatorTurn]], Awaitable[CandidateTurn]],
    post_user_turn: Callable[[SimulatorTurn], Awaitable[None]],
    *,
    llm: SimulatorLLM,
    sink: CellSink,
    world: WorldSession | None = None,
) -> ConversationStopCause:
    """Run ``driver``'s conversation from its first turn until it stops on a structural signal.

    Each round: the driver names speakers until the round is done; each delivered utterance goes to
    ``post_user_turn`` in order (one whose ``session_break`` is set is the first of a new session, and
    the kind starts one before delivering it); then ``candidate_turn`` answers the round's utterances
    and its answer is recorded. A reply in which an actor says ``done`` is not delivered.

    **Every delivered line is answered, the last actor's departure included.** When the last actor present leaves
    after others have spoken in the same round, the conversation ends ``user_done`` only once the candidate has
    answered what that round delivered, so a transcript ending on ``user_done`` does not end on a simulated line the
    candidate was handed and not asked to answer — unless the run's cost cap is reached just then, when the answer is
    not bought and the runner excludes the cell for the cap. The stop cause then stays ``user_done``, deliberately:
    the departure ended the conversation before the cap was asked, and a driver's first stop stands. The cell is
    excluded and the run stopped all the same, through the breach the sink recorded. A round that delivered nothing is
    not answered. The turn budget cannot be spent by that answer: the round began with ``candidate_turns <
    max_turns``.

    **A world round has no speaker.** On a round the template declares a world round
    (``ConversationSpec.world_rounds``), the loop fires its triggered dimension through ``world`` — the cell's
    world session, whose seed armed it — and hands ``candidate_turn`` one turn spoken by
    :data:`~threetears.evals.schema.models.WORLD_SPEAKER` that names what fired, instead of any actor's line.
    Nothing is posted through ``post_user_turn``: the host's fire handle moved the candidate's world, and the
    kind decides how its candidate perceives it. A conversation of world rounds alone runs no simulator call.

    :param driver: A fresh driver over the template's ``conversation`` block.
    :ptype driver: TurnDriver
    :param candidate_turn: The candidate's answer to one round, handed that round's delivered utterances.
    :ptype candidate_turn: Callable[[Sequence[SimulatorTurn]], Awaitable[CandidateTurn]]
    :param post_user_turn: Delivers one simulated utterance into the candidate's world.
    :ptype post_user_turn: Callable[[SimulatorTurn], Awaitable[None]]
    :param llm: The simulator-role client, for utterances and ``llm_decided`` picks alike. Every call it makes goes
        through ``sink``, which counts what it cost.
    :ptype llm: SimulatorLLM
    :param sink: The cell's sink, which counts what every simulator call cost (see the module docstring).
    :ptype sink: CellSink
    :param world: The cell's world session, which fires each world round's dimension; required when the
        conversation declares world rounds.
    :ptype world: WorldSession | None
    :return: Why the conversation stopped: ``max_turns`` or ``user_done``.
    :rtype: ConversationStopCause
    :raises ValueError: ``driver`` has already taken a turn; this drives a conversation from its start. Or the
        conversation declares world rounds and no world session was handed in to fire them.
    """
    if driver.user_turns or driver.candidate_turns or driver.stop_cause is not None:
        raise ValueError("drive_conversation runs a conversation from its start, and this driver has already run")
    if driver.conversation.world_rounds and world is None:
        raise ValueError(
            "the conversation declares world rounds, and no world session was handed in to fire them; pass the "
            "cell's world session as world="
        )

    capped = CappedCompletion(llm, sink=sink, source="simulator")
    return await _converse(driver, candidate_turn, post_user_turn, capped, world)


async def _converse(
    driver: TurnDriver,
    candidate_turn: Callable[[Sequence[SimulatorTurn]], Awaitable[CandidateTurn]],
    post_user_turn: Callable[[SimulatorTurn], Awaitable[None]],
    llm: SimulatorLLM,
    world: WorldSession | None,
) -> ConversationStopCause:
    """The loop :func:`drive_conversation` runs, over the simulator client whose spend the sink counts."""
    round_turns: list[SimulatorTurn] = []
    opener = driver.initial_utterance()
    if opener is not None:
        await _side(driver, ConversationStopCause.APPARATUS_ERROR, post_user_turn(opener))
        round_turns.append(opener)
    while driver.should_continue():
        if (world_round := driver.world_round()) is not None:
            assert world is not None  # refused above for a conversation with world rounds
            event = await _side(
                driver,
                ConversationStopCause.APPARATUS_ERROR,
                world.fire(world_round.dimension, turn=driver.candidate_turns + 1),
            )
            round_turns = [driver.record_world_event(world_round, event)]
        while (
            world_round is None
            and (actor := await _side(driver, ConversationStopCause.SIMULATOR_ERROR, driver.next_speaker(llm)))
            is not None
        ):
            turn = await _side(driver, ConversationStopCause.SIMULATOR_ERROR, driver.next_user_turn(llm, actor))
            if turn.done:
                if not driver.should_continue():
                    break
                continue
            await _side(driver, ConversationStopCause.APPARATUS_ERROR, post_user_turn(turn))
            round_turns.append(turn)
        # The last actor leaving ends the conversation, but not before the lines this round already
        # delivered are answered; any other stop (a fault) makes no further call.
        if not driver.should_continue() and not (driver.stop_cause is ConversationStopCause.USER_DONE and round_turns):
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
