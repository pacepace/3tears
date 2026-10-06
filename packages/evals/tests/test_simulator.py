"""Unit tests for the multi-actor user simulator."""

from __future__ import annotations

import json

import pytest

from threetears.evals.contracts.models import ActorPolicy, ConversationSpec, ConversationStopCause, RoleUsage
from threetears.evals.contracts.usage_capture import CallUsage, RoleUsageLedger
from threetears.evals.run.simulator import (
    SIMULATED_USER_RESPONSE_FORMAT,
    SIMULATOR_ANSWER_BUDGET_TOKENS,
    SIMULATOR_MAX_TOKENS,
    SIMULATOR_REASONING_ALLOWANCE_TOKENS,
    SIMULATOR_REASONING_EFFORT,
    SIMULATOR_REQUEST_SETTINGS,
    CandidateTurn,
    SimulatorReplyInvalid,
    SimulatorTurn,
    TurnDriver,
)

# =============================================================================
# Helpers
# =============================================================================


def _reply(utterance: str, *, done: bool = False) -> str:
    """The structured reply a schema-enforcing provider returns for one simulated-user turn."""
    return json.dumps({"utterance": utterance, "done": done})


class _RecordingLLM:
    """Fake LLM that returns canned responses and records every call.

    A plain string is an utterance the user is not done after; anything else is returned as the
    completion's raw text, for the tests that hand the driver a reply of their own shape.
    """

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []
        self.response_formats: list[dict | None] = []

    async def generate(self, *, system: str, user: str, response_format: dict | None = None):
        self.calls.append((system, user))
        self.response_formats.append(response_format)
        if not self.responses:
            raise AssertionError("Simulator made more LLM calls than expected.")
        content = self.responses.pop(0)
        return _Response(content if isinstance(content, _Raw) else _reply(content))


class _Raw(str):
    """A completion text handed to the driver verbatim, not wrapped as an utterance."""


class _Response:
    """Minimal response shape matching ``getattr(resp, 'content', '')``."""

    def __init__(self, content: str):
        self.content = content


def _conversation(actors: list[ActorPolicy], **overrides) -> ConversationSpec:
    """A round-robin conversation whose rounds hold up to five utterances, so a test can take several in a row."""
    kwargs = dict(actors=actors, max_turns=5, max_speakers_per_round=5)
    kwargs.update(overrides)
    return ConversationSpec(**kwargs)


async def _speak(driver: TurnDriver, llm) -> SimulatorTurn:
    """The next utterance of the current round: the scheduler's pick, then that actor's call."""
    actor = await driver.next_speaker(llm)
    assert actor is not None, "the round has room for another speaker"
    return await driver.next_user_turn(llm, actor)


def _actor(actor_id: str, **overrides) -> ActorPolicy:
    kwargs = dict(
        id=actor_id,
        policy=f"You speak as {actor_id}.",
        intent="Engage the candidate.",
    )
    kwargs.update(overrides)
    return ActorPolicy(**kwargs)


# =============================================================================
# Construction guards
# =============================================================================


def test_a_conversation_requires_at_least_one_actor():
    """A conversation with nobody on the other side is unrepresentable, so no driver can be built over one."""
    with pytest.raises(ValueError, match="at least 1 item"):
        _conversation(actors=[])


def test_driver_rejects_unknown_turn_scheduler():
    actors = [_actor("shopper")]
    conversation = _conversation(actors)
    # Bypass Pydantic so we can verify the runtime check.
    object.__setattr__(conversation, "turn_scheduler", "round_robin_xyz")
    with pytest.raises(ValueError, match="Unknown turn_scheduler"):
        TurnDriver(conversation=conversation, variation={})


# =============================================================================
# Initial utterance with variation-placeholder rendering
# =============================================================================


def test_initial_utterance_renders_variation_placeholders():
    actor = _actor(
        "curious_shopper",
        initial_utterance_template="Can you put together a bundle of {variation.category_pair}?",
    )
    conversation = _conversation([actor])
    driver = TurnDriver(conversation=conversation, variation={"category_pair": "kitchen and garden"})
    turn = driver.initial_utterance()
    assert turn is not None
    assert turn.actor_id == "curious_shopper"
    assert turn.content == "Can you put together a bundle of kitchen and garden?"
    assert turn.session_break is False
    # Transcript records this turn.
    assert driver.transcript == [("curious_shopper", turn.content)]
    assert driver.user_turns == 1


def test_a_templated_opener_cannot_reach_an_llm_at_all():
    """The opener is a string substitution, so no model choice can affect it.

    This is a signature-level guarantee, not an observation about one call:
    ``initial_utterance`` takes no LLM, so there is no argument by which a
    simulator model could influence a templated opener. Making the opener
    LLM-backed would fail here rather than quietly begin spending on whichever
    model the simulator role happens to resolve.
    """
    import inspect

    assert "llm" not in inspect.signature(TurnDriver.initial_utterance).parameters
    # The entry points that take one are the follow-up turn and the scheduler's pick.
    assert "llm" in inspect.signature(TurnDriver.next_user_turn).parameters
    assert "llm" in inspect.signature(TurnDriver.next_speaker).parameters


def test_max_turns_one_stops_before_any_follow_up_is_requested():
    """At ``max_turns=1`` the driver stops before ``next_user_turn`` is reachable.

    ``next_user_turn`` is the sole consumer of the simulator LLM, and the runner
    only calls it while ``should_continue()`` holds. So a single-turn template
    makes zero simulator calls — which is what makes the simulator's model choice
    inert for the templates in use today, and why that model could be moved to
    its own role without disturbing live data.

    Note the scope: this pins the *driver's* behaviour at ``max_turns=1``. It
    cannot pin that every stored template is single-turn — those live in the
    database, not the repo — so a template growing a second turn is a live-data
    change this test will not catch.
    """
    actor = _actor("shopper", initial_utterance_template="{variation.query}")
    conversation = _conversation([actor], max_turns=1)
    driver = TurnDriver(conversation=conversation, variation={"query": "find me something new"})

    opener = driver.initial_utterance()
    assert opener is not None
    assert opener.content == "find me something new"

    driver.record_candidate_turn(CandidateTurn(content="here you go"))
    assert driver.should_continue() is False
    assert driver.stop_cause is ConversationStopCause.MAX_TURNS

    # Contrast: raise the cap and the driver would ask for a follow-up, so the
    # stop above is a consequence of max_turns rather than of the driver never
    # continuing at all.
    roomier = TurnDriver(conversation=_conversation([actor], max_turns=2), variation={"query": "q"})
    roomier.initial_utterance()
    roomier.record_candidate_turn(CandidateTurn(content="here you go"))
    assert roomier.should_continue() is True


def test_initial_utterance_returns_none_when_no_template():
    actor = _actor("speak_first_via_llm")  # no initial_utterance_template
    conversation = _conversation([actor])
    driver = TurnDriver(conversation=conversation, variation={})
    assert driver.initial_utterance() is None
    assert driver.user_turns == 0


def test_unknown_placeholder_left_in_text():
    actor = _actor(
        "x",
        initial_utterance_template="Hello {variation.missing_field}!",
    )
    conversation = _conversation([actor])
    driver = TurnDriver(conversation=conversation, variation={})
    turn = driver.initial_utterance()
    # Verbatim — surfaces the templating error visibly rather than silently dropping.
    assert turn is not None
    assert "{variation.missing_field}" in turn.content


# =============================================================================
# Round-robin actor scheduling — N=3
# =============================================================================


async def test_round_robin_with_three_actors_cycles_in_order():
    actors = [_actor("a"), _actor("b"), _actor("c")]
    conversation = _conversation(actors)
    driver = TurnDriver(conversation=conversation, variation={})
    llm = _RecordingLLM(["from a", "from b", "from c", "from a again"])

    turn1 = await _speak(driver, llm)
    turn2 = await _speak(driver, llm)
    turn3 = await _speak(driver, llm)
    turn4 = await _speak(driver, llm)

    assert [t.actor_id for t in (turn1, turn2, turn3, turn4)] == ["a", "b", "c", "a"]


async def test_initial_utterance_advances_scheduler_past_first_actor():
    """After initial_utterance(), the next LLM-driven turn comes from actor 2."""
    actors = [_actor("a", initial_utterance_template="hi"), _actor("b"), _actor("c")]
    conversation = _conversation(actors)
    driver = TurnDriver(conversation=conversation, variation={})
    initial = driver.initial_utterance()
    assert initial is not None and initial.actor_id == "a"
    llm = _RecordingLLM(["from b"])
    turn = await _speak(driver, llm)
    assert turn.actor_id == "b"


# =============================================================================
# Transcript building
# =============================================================================


async def test_transcript_captures_both_sides_in_order():
    actors = [_actor("shopper")]
    conversation = _conversation(actors)
    driver = TurnDriver(conversation=conversation, variation={})
    llm = _RecordingLLM(["First shopper turn", "Second shopper turn"])

    await _speak(driver, llm)
    driver.record_candidate_turn(CandidateTurn(content="Candidate response 1"))
    await _speak(driver, llm)
    driver.record_candidate_turn(CandidateTurn(content="Candidate response 2"))

    assert driver.transcript == [
        ("shopper", "First shopper turn"),
        ("__candidate__", "Candidate response 1"),
        ("shopper", "Second shopper turn"),
        ("__candidate__", "Candidate response 2"),
    ]
    assert driver.user_turns == 2
    assert driver.candidate_turns == 2


# =============================================================================
# Stop conditions
# =============================================================================


def test_should_continue_initially_true():
    actors = [_actor("a")]
    conversation = _conversation(actors, max_turns=3)
    driver = TurnDriver(conversation=conversation, variation={})
    assert driver.should_continue() is True


def test_max_turns_stops_after_candidate_count():
    actors = [_actor("a")]
    conversation = _conversation(actors, max_turns=2)
    driver = TurnDriver(conversation=conversation, variation={})

    driver.record_candidate_turn(CandidateTurn(content="ok"))
    assert driver.should_continue() is True

    driver.record_candidate_turn(CandidateTurn(content="ok"))
    assert driver.should_continue() is False
    assert driver.stop_cause is ConversationStopCause.MAX_TURNS


def test_explicit_stop_sets_cause():
    actors = [_actor("a")]
    conversation = _conversation(actors)
    driver = TurnDriver(conversation=conversation, variation={})
    assert driver.stop_cause is None
    driver.stop(ConversationStopCause.USER_DONE)
    assert driver.should_continue() is False
    assert driver.stop_cause is ConversationStopCause.USER_DONE


@pytest.mark.parametrize(
    "refusal",
    ["I can't help with that.", "I'm unable to do that.", "I refuse to order that.", ""],
)
def test_a_refusal_does_not_end_the_conversation(refusal):
    """What the candidate says is never a stop signal — a refusal's quality is the judge's to grade.

    Stopping on the words of a refusal truncated exactly the conversations whose refusals a
    boundary rubric exists to grade, and recorded no reason for doing so.
    """
    driver = TurnDriver(conversation=_conversation([_actor("a")], max_turns=3), variation={})
    driver.record_candidate_turn(CandidateTurn(content=refusal))
    assert driver.should_continue() is True
    assert driver.stop_cause is None


def test_stop_is_idempotent():
    actors = [_actor("a")]
    conversation = _conversation(actors)
    driver = TurnDriver(conversation=conversation, variation={})
    driver.stop(ConversationStopCause.SIMULATOR_ERROR)
    driver.stop(ConversationStopCause.USER_DONE)
    assert driver.stop_cause is ConversationStopCause.SIMULATOR_ERROR


# =============================================================================
# The simulated user's structured reply — the "done" stop and the schema
# =============================================================================


async def test_every_user_turn_sends_the_strict_reply_schema():
    llm = _RecordingLLM(["hi"])
    await _speak(TurnDriver(conversation=_conversation([_actor("a")]), variation={}), llm)

    assert llm.response_formats == [SIMULATED_USER_RESPONSE_FORMAT]
    schema = SIMULATED_USER_RESPONSE_FORMAT["json_schema"]["schema"]
    assert SIMULATED_USER_RESPONSE_FORMAT["json_schema"]["strict"] is True
    assert schema["required"] == ["utterance", "done"]
    assert schema["additionalProperties"] is False


async def test_a_user_that_says_it_is_done_stops_the_conversation():
    driver = TurnDriver(conversation=_conversation([_actor("a")], max_turns=5), variation={})
    turn = await _speak(driver, _RecordingLLM([_Raw(_reply("thanks, bye", done=True))]))

    assert turn.done is True
    assert driver.should_continue() is False
    assert driver.stop_cause is ConversationStopCause.USER_DONE
    # The closing reply is the user leaving, not a line the candidate is asked to answer.
    assert driver.transcript == []


async def test_a_user_that_is_not_done_keeps_the_conversation_going():
    driver = TurnDriver(conversation=_conversation([_actor("a")], max_turns=5), variation={})
    turn = await _speak(driver, _RecordingLLM([_Raw(_reply("and another thing", done=False))]))

    assert turn.done is False
    assert turn.content == "and another thing"
    assert driver.should_continue() is True
    assert driver.stop_cause is None
    assert driver.transcript == [("a", "and another thing")]


@pytest.mark.parametrize(
    "raw",
    [
        "just some prose, no JSON",
        json.dumps({"utterance": "hi"}),
        json.dumps({"utterance": "hi", "done": "yes"}),
        json.dumps({"utterance": "hi", "done": False, "mood": "curious"}),
        "",
    ],
    ids=["prose", "missing-done", "done-not-bool", "extra-field", "empty"],
)
async def test_a_reply_that_breaks_the_schema_is_refused_with_its_spend(raw):
    driver = TurnDriver(conversation=_conversation([_actor("a")]), variation={})

    class _Billed:
        content = raw
        model = "sim-model"
        input_tokens = 9
        output_tokens = 3
        cost_usd = 0.001

    class _LLM:
        async def generate(self, *, system, user, response_format=None):
            return _Billed()

    with pytest.raises(SimulatorReplyInvalid):
        await _speak(driver, _LLM())
    # The malformed reply was billed, so its call is on the driver's record before the refusal.
    [call] = driver.calls
    assert (call.purpose, call.actor_id) == ("utterance", "a")
    assert call.usage.input_tokens == 9
    assert call.usage.cost_usd == 0.001
    assert driver.transcript == []
    assert driver.stop_cause is None


# =============================================================================
# Prompt assembly — system prompt carries the actor's policy + variation block
# =============================================================================


async def test_system_prompt_carries_the_actor_policy_and_variation():
    actor = _actor("shopper", policy="You are casual and curious.")
    conversation = _conversation([actor])
    driver = TurnDriver(conversation=conversation, variation={"tone": "casual", "category": "kitchen"})
    llm = _RecordingLLM(["ok"])

    await _speak(driver, llm)

    system, _user = llm.calls[0]
    assert "You are casual and curious." in system
    assert "tone: casual" in system
    assert "category: kitchen" in system
    # The simulated user talks to a candidate, whatever kind of thing the candidate is.
    assert "interacting with a candidate under evaluation" in system
    assert "persona" not in system.lower()


async def test_user_prompt_carries_intent_and_transcript():
    actor = _actor("shopper", intent="Get them to add the Blue Kettle.")
    conversation = _conversation([actor])
    driver = TurnDriver(conversation=conversation, variation={})
    llm = _RecordingLLM(["initial"])

    await _speak(driver, llm)
    driver.record_candidate_turn(CandidateTurn(content="Sure thing!"))
    llm.responses.append("ok then")

    await _speak(driver, llm)

    _system, user = llm.calls[1]
    assert "Get them to add the Blue Kettle." in user
    # Prior transcript shows both speakers labeled.
    assert "Candidate: Sure thing!" in user
    assert "You (shopper): initial" in user


async def test_multi_actor_transcript_labels_other_actors():
    """In multi-actor scenarios, the user prompt distinguishes 'You (...)' from 'Actor (...)'."""
    actors = [_actor("alice"), _actor("bob")]
    conversation = _conversation(actors)
    driver = TurnDriver(conversation=conversation, variation={})
    llm = _RecordingLLM(["alice speaks", "bob speaks"])

    await _speak(driver, llm)  # alice
    await _speak(driver, llm)  # bob

    # When the third turn renders (alice's turn again), Bob's prior turn shows as "Actor (bob)"
    llm.responses.append("alice replies to bob")
    await _speak(driver, llm)
    _system, user = llm.calls[2]
    assert "You (alice): alice speaks" in user
    assert "Actor (bob): bob speaks" in user


# =============================================================================
# Variation placeholder substitution, read off the opening turn that renders it
# =============================================================================


def _opening(template_text: str, variation: dict[str, str]) -> str:
    """The first actor's templated opener as the driver renders it."""
    driver = TurnDriver(
        conversation=_conversation([_actor("shopper", initial_utterance_template=template_text)]), variation=variation
    )
    turn = driver.initial_utterance()
    assert turn is not None, "a templated first actor opens with its rendered template"
    return turn.content


def test_variation_placeholder_substitution_simple():
    assert _opening("find {variation.category}", {"category": "kitchen"}) == "find kitchen"


def test_variation_placeholder_unknown_left_in_place():
    assert _opening("find {variation.missing}", {"category": "kitchen"}) == "find {variation.missing}"


def test_variation_placeholder_multiple_substitutions():
    out = _opening(
        "{variation.a} and {variation.b}",
        {"a": "AA", "b": "BB"},
    )
    assert out == "AA and BB"


# =============================================================================
# Simulator usage capture — the simulator RoleUsage row's source
# =============================================================================


class _UsageResponse:
    """Response shape carrying the token/cost fields an LLMResult would."""

    def __init__(
        self, content="hi", *, model="sim-model", inp=12, out=34, reasoning=None, cost=0.004, price_source="rate_card"
    ):
        self.content = _reply(content)
        self.model = model
        self.input_tokens = inp
        self.output_tokens = out
        self.reasoning_tokens = reasoning
        self.cost_usd = cost
        self.price_source = price_source


class _UsageLLM:
    def __init__(self, response):
        self._response = response

    async def generate(self, *, system: str, user: str, response_format: dict | None = None):
        return self._response


def _usage_driver() -> TurnDriver:
    """A driver with no initial_utterance_template, so the first turn is LLM-driven."""
    return TurnDriver(
        conversation=_conversation([ActorPolicy(id="a", policy="p", intent="i")]),
        variation={},
    )


async def _usage_of(llm) -> CallUsage:
    """The usage the driver recorded for one LLM-driven turn."""
    driver = _usage_driver()
    await _speak(driver, llm)
    [call] = driver.calls
    assert (call.purpose, call.actor_id, call.round_index) == ("utterance", "a", 0)
    return call.usage


async def test_llm_turn_carries_the_calls_usage():
    usage = await _usage_of(_UsageLLM(_UsageResponse()))

    assert usage.model == "sim-model"
    assert usage.input_tokens == 12
    assert usage.output_tokens == 34
    assert usage.cost_usd == 0.004
    assert usage.price_source == "rate_card", "where the dollars came from is the client's to say"


async def test_llm_turn_reasoning_unreported_stays_none():
    assert (await _usage_of(_UsageLLM(_UsageResponse(reasoning=None)))).reasoning_tokens is None


async def test_llm_turn_reasoning_zero_is_kept():
    assert (await _usage_of(_UsageLLM(_UsageResponse(reasoning=0)))).reasoning_tokens == 0


async def test_llm_turn_from_a_client_reporting_nothing_reads_as_unobserved():
    """A minimal client (content only) reports no spend — not a spend of zero dollars."""
    usage = await _usage_of(_RecordingLLM(["hi"]))

    assert usage.model is None
    # Never measured is absent, not a measured zero.
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.cost_usd is None
    assert usage.price_source is None


def test_scripted_initial_utterance_records_no_call():
    """A templated opener runs no LLM at all — distinct from an LLM call that reported nothing."""
    driver = TurnDriver(
        conversation=_conversation([ActorPolicy(id="a", policy="p", intent="i", initial_utterance_template="Hello")]),
        variation={},
    )
    assert driver.initial_utterance() is not None
    assert driver.calls == []


async def test_fold_usage_lands_every_call_on_the_simulator_ledger():
    driver = _usage_driver()
    await _speak(driver, _UsageLLM(_UsageResponse()))
    await _speak(driver, _UsageLLM(_UsageResponse()))
    ledger = RoleUsageLedger(role="simulator")

    driver.fold_usage(ledger)

    [row] = ledger.rows()
    assert row.call_count == 2
    assert row.cost_usd == pytest.approx(0.008)
    assert (row.actor_id, row.purpose) == (driver.conversation.actors[0].id, "utterance")
    assert driver.cost_usd == pytest.approx(0.008)


def test_an_actor_or_purpose_belongs_on_a_simulator_row_only():
    """The attribution names a simulated actor, which only the simulator role has; refused on every other row."""
    assert RoleUsage(role="simulator", actor_id="a", purpose="schedule").actor_id == "a"
    for fields in ({"actor_id": "a"}, {"purpose": "utterance"}):
        with pytest.raises(ValueError, match="attribute simulator calls"):
            RoleUsage(role="candidate", **fields)
        with pytest.raises(ValueError, match="attribute simulator calls"):
            RoleUsageLedger(role="judge").add(
                model="m", prompt_tokens=None, completion_tokens=None, reasoning_tokens=None, cost_usd=None, **fields
            )


@pytest.mark.parametrize("role", ["candidate", "judge"])
def test_fold_usage_refuses_another_roles_ledger(role):
    """Simulator spend is the program's cost; landing it on the candidate's row would misstate what it costs."""
    with pytest.raises(ValueError, match="simulator ledger"):
        _usage_driver().fold_usage(RoleUsageLedger(role=role))


def test_the_simulator_asks_for_its_reasoning_by_effort_under_a_derived_cap() -> None:
    """``openai/gpt-5-nano`` is effort-only: a token budget was mapped to an effort by its share of the cap and still
    reasoned through the whole cap, so the simulator names the lowest effort that reasons and sends no token budget —
    and its cap is still derived as room for that reasoning plus room for the reply."""
    assert SIMULATOR_REQUEST_SETTINGS.reasoning_effort == SIMULATOR_REASONING_EFFORT == "minimal"
    assert SIMULATOR_REQUEST_SETTINGS.reasoning_max_tokens is None
    assert SIMULATOR_REQUEST_SETTINGS.max_tokens == SIMULATOR_MAX_TOKENS
    assert SIMULATOR_MAX_TOKENS == SIMULATOR_REASONING_ALLOWANCE_TOKENS + SIMULATOR_ANSWER_BUDGET_TOKENS
