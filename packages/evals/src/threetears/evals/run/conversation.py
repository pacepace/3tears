"""``drive_conversation``: the speaker-round loop every conversing kind runs, written once.

A conversing kind has two things to supply, and hands over its cell's sink: how a simulated utterance
reaches its candidate's world (``post_user_turn``), and how its candidate answers a round (``candidate_turn``).
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

**The run's cost cap is asked before every paid call.** The cap otherwise counts a cell's spend only
once its result lands, and one conversation can make thousands of simulator calls (the bound is in
:mod:`~threetears.evals.run.simulator`'s *Spend*). So before each call the loop makes — a scheduling
pick, an utterance, the candidate's answer — it asks the cell's sink whether the cap is reached
counting the simulator's spend so far (:meth:`~threetears.evals.contracts.candidate_kind.CellSink.cost_cap_reached`),
and when it is, stops ``budget_stopped`` without making the call. A round already delivered is then left
unanswered: answering it is a paid call past the cap. The runner excludes such a cell and stops the run.
The candidate's own spend inside the cell is not in that count — the loop cannot see it — and is
counted when the cell's result lands. The cap is asked once per DECISION, and an ``llm_decided``
scheduling decision may take two calls (a refused reply buys one repair,
:data:`~threetears.evals.run.simulator.SCHEDULER_CALL_ATTEMPTS`), so a run overshoots its cap by at most
two simulator calls plus the stopped cell's candidate spend — one call under ``round_robin``, whose only
paid calls are utterances.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from threetears.evals.contracts.candidate_kind import CellSink
from threetears.evals.contracts.models import ConversationStopCause
from threetears.evals.contracts.provider import SimulatorLLM
from threetears.evals.run.simulator import CandidateTurn, SimulatorTurn, TurnDriver


async def drive_conversation(
    driver: TurnDriver,
    candidate_turn: Callable[[Sequence[SimulatorTurn]], Awaitable[CandidateTurn]],
    post_user_turn: Callable[[SimulatorTurn], Awaitable[None]],
    *,
    llm: SimulatorLLM,
    sink: CellSink,
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

    Args:
        driver: A fresh driver over the template's ``conversation`` block.
        candidate_turn: The candidate's answer to one round, handed that round's delivered utterances.
        post_user_turn: Delivers one simulated utterance into the candidate's world.
        llm: The simulator-role client, for utterances and ``llm_decided`` picks alike.
        sink: The cell's sink, asked before every paid call whether the run's cost cap is reached
            counting the simulator's spend so far (see the module docstring).

    Returns:
        Why the conversation stopped: ``max_turns``, ``user_done`` or ``budget_stopped``.

    Raises:
        ValueError: ``driver`` has already taken a turn; this drives a conversation from its start.
    """
    if driver.user_turns or driver.candidate_turns or driver.stop_cause is not None:
        raise ValueError("drive_conversation runs a conversation from its start, and this driver has already run")

    def affordable() -> bool:
        """Whether the run's cap leaves room for another paid call; stops the driver when it does not."""
        if sink.cost_cap_reached(driver.cost_usd):
            driver.stop(ConversationStopCause.BUDGET_STOPPED)
            return False
        return True

    round_turns: list[SimulatorTurn] = []
    opener = driver.initial_utterance()
    if opener is not None:
        await _side(driver, ConversationStopCause.APPARATUS_ERROR, post_user_turn(opener))
        round_turns.append(opener)
    while driver.should_continue():
        while (
            affordable()
            and (actor := await _side(driver, ConversationStopCause.SIMULATOR_ERROR, driver.next_speaker(llm)))
            is not None
        ):
            if not affordable():
                break
            turn = await _side(driver, ConversationStopCause.SIMULATOR_ERROR, driver.next_user_turn(llm, actor))
            if turn.done:
                if not driver.should_continue():
                    break
                continue
            await _side(driver, ConversationStopCause.APPARATUS_ERROR, post_user_turn(turn))
            round_turns.append(turn)
        # The last actor leaving ends the conversation, but not before the lines this round already
        # delivered are answered; any other stop (the cap, a fault) makes no further call.
        if not driver.should_continue() and not (driver.stop_cause is ConversationStopCause.USER_DONE and round_turns):
            break
        if not affordable():
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
