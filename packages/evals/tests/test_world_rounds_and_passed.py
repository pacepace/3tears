"""A turn driven by a world event alone, and a named check for a deliberate pass (#578).

A conversing template can declare rounds whose stimulus is a triggered world dimension firing rather than an
actor's line, and can have no simulator actors at all when the world supplies every round. The engine owns the
notion of a deliberate pass — one reserved ledger entry (``CallLedger.record_pass``) — and the goal language
reads it through ``passed()``, the same for every host; the call builtins do not see it.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from threetears.evals.schema import CallLedger, DSLError, EvalTemplate, WorldSeed
from threetears.evals.kernel.dsl import evaluate
from threetears.evals.schema.goal_grammar import reads_call_ledger
from threetears.evals.schema.models import WORLD_SPEAKER, ConversationSpec, ConversationStopCause, WorldRound
from threetears.evals.kernel.world_session import WorldSession
from threetears.evals.run import drive_conversation
from threetears.evals.run.runner import grade_goal_checks
from threetears.evals.run.simulator import CandidateTurn, SimulatorTurn, TurnDriver
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.fixtures.toyhost.world import toyhost_world
from packages.evals.tests.scripted_table import FakeCellSink, ScriptedTable, actor

#: The toy world's carriers, and a seed arming its triggered ``payment_hold``.
CARRIERS = ("page_reader", "console")
ARMING_SEED = WorldSeed(namespaces={"console": {"payment_hold": "held"}})
HOLD = "payment_hold"


# =============================================================================
# The conversation block
# =============================================================================


def test_a_conversation_with_no_actors_needs_a_world_round_for_every_turn() -> None:
    with pytest.raises(ValidationError, match=r"turn\(s\) \[2\] have no stimulus"):
        ConversationSpec(actors=[], world_rounds=[WorldRound(turn=1, dimension=HOLD)], max_turns=2)
    spec = ConversationSpec(actors=[], world_rounds=[WorldRound(turn=1, dimension=HOLD)], max_turns=1)
    assert spec.world_round(1) == WorldRound(turn=1, dimension=HOLD)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"world_rounds": [WorldRound(turn=1, dimension=HOLD), WorldRound(turn=1, dimension=HOLD)]}, "twice"),
        ({"world_rounds": [WorldRound(turn=4, dimension=HOLD)]}, "past max_turns"),
        (
            {
                "actors": [actor("a", initial_utterance_template="Hello.")],
                "world_rounds": [WorldRound(turn=1, dimension=HOLD)],
            },
            "opens round 1, which is a world round",
        ),
        ({"actors": [actor(WORLD_SPEAKER)]}, "reserved"),
    ],
)
def test_a_block_no_driver_could_run_is_refused(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ConversationSpec(**{"actors": [actor("a")], "max_turns": 3, **fields})


# =============================================================================
# The Done-when: no actors, a world event drives the turn, and passed() reads the pass
# =============================================================================


class _Candidate:
    """A candidate that either passes on the world's event or acts on it, recording into its cell's ledger."""

    def __init__(self, *, acts: bool) -> None:
        self.acts = acts
        self.ledger = CallLedger()
        self.rounds: list[list[str]] = []

    async def answer(self, round_turns: Sequence[SimulatorTurn]) -> CandidateTurn:
        self.rounds.append([turn.actor_id for turn in round_turns])
        if self.acts:
            self.ledger.record("console", "release_payment", {"reason": "hold noticed"})
        else:
            self.ledger.record_pass(reason="the hold needs no action from me")
        return CandidateTurn(content="noted")

    async def post(self, turn: SimulatorTurn) -> None:
        raise AssertionError(f"nothing is posted on a world round; got {turn.actor_id}")


def _world_only_template() -> EvalTemplate:
    """The toy template, conversing with nobody: one round, its stimulus the payment hold firing."""
    document = toyhost_template().model_dump()
    document["conversation"] = ConversationSpec(
        actors=[], world_rounds=[WorldRound(turn=1, dimension=HOLD)], max_turns=1
    ).model_dump()
    document["world_seed"] = ARMING_SEED.model_dump()
    return EvalTemplate.model_validate(document)


async def _drive(*, acts: bool) -> tuple[_Candidate, WorldSession, TurnDriver, ConversationStopCause]:
    template = _world_only_template()
    assert template.conversation is not None and template.conversation.actors == []
    registry, _state = toyhost_world()
    session = WorldSession(registry, provenance="commissioned")
    await session.seed(template.world_seed, attached=CARRIERS)
    driver = TurnDriver(conversation=template.conversation, variation={})
    candidate = _Candidate(acts=acts)
    table = ScriptedTable()
    cause = await drive_conversation(
        driver, candidate.answer, candidate.post, llm=table, sink=FakeCellSink(), world=session
    )
    assert table.utterance_calls == [] and table.schedule_calls == [] and driver.calls == [], "no simulator call"
    return candidate, session, driver, cause


@pytest.mark.parametrize(("acts", "passed"), [(False, True), (True, False)])
async def test_a_world_event_alone_drives_the_turn_and_passed_reads_a_deliberate_pass(acts: bool, passed: bool) -> None:
    candidate, session, driver, cause = await _drive(acts=acts)

    assert cause is ConversationStopCause.MAX_TURNS
    assert candidate.rounds == [[WORLD_SPEAKER]], "the candidate answered one round whose only stimulus was the world"
    assert [(event.dimension, event.caused_by, event.turn) for event in session.events] == [(HOLD, "rig", 1)]
    assert driver.transcript[0][0] == WORLD_SPEAKER
    end_state = await session.end_state()
    outcomes = grade_goal_checks(
        ["passed()", f'fired("{HOLD}")'],
        ledger=candidate.ledger,
        end_state=end_state,
        fired=session.fired,
        variation={},
        world=session.registry,
    )
    assert [outcome.passed for outcome in outcomes] == [passed, True]


async def test_world_rounds_need_the_cells_world_session() -> None:
    template = _world_only_template()
    assert template.conversation is not None
    driver = TurnDriver(conversation=template.conversation, variation={})
    candidate = _Candidate(acts=False)
    with pytest.raises(ValueError, match="no world session was handed in"):
        await drive_conversation(driver, candidate.answer, candidate.post, llm=ScriptedTable(), sink=FakeCellSink())


# =============================================================================
# passed() and the call builtins
# =============================================================================


def _ledger(*entries: str) -> CallLedger:
    ledger = CallLedger()
    for entry in entries:
        if entry == "pass":
            ledger.record_pass()
        else:
            tool, action = entry.split(".")
            ledger.record(tool, action)
    return ledger


def _holds(expression: str, ledger: CallLedger) -> bool:
    return evaluate(expression, end_state={}, ledger=ledger, world=None)


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ((), False),  # doing nothing without saying so is not a deliberate pass
        (("pass",), True),
        (("pass", "pass"), True),
        (("inv.place", "pass"), False),  # acted, then passed
        (("pass", "inv.place"), False),
    ],
)
def test_passed_holds_for_a_pass_and_no_call(entries: tuple[str, ...], expected: bool) -> None:
    assert _holds("passed()", _ledger(*entries)) is expected


def test_the_call_builtins_do_not_see_the_engines_pass() -> None:
    acted_then_passed = _ledger("inv.place", "pass")
    assert _holds('last_call_was("inv.place")', acted_then_passed), "a pass is not the last call"
    assert _holds('call_count("inv.place") == 1', acted_then_passed)
    assert _holds('call_count("__engine__.pass") == 0', acted_then_passed)
    assert _holds('calls("__engine__.pass").length == 0', acted_then_passed)
    assert not _holds('called_before("inv.place", "inv.cancel")', _ledger()), "absent actions read False"


def test_passed_reads_the_ledger_and_takes_no_argument() -> None:
    assert reads_call_ledger("passed()")
    with pytest.raises(DSLError, match="takes 0 argument"):
        _holds('passed("x")', _ledger())
