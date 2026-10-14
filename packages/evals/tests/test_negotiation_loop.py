"""The shared negotiation loop and its refusal readers: what every completion client sends, resends and learns.

``negotiate`` is the one implementation of "a model that lacks a setting never fails the request" (owner ruling,
2026-10-14), behind the shipped adapter and a host's own client alike. What this holds, with a scripted ``send`` and
no provider: the fixed order (a temperature first, keeping enforcement, whatever the facts say, and only then
enforcement, for that call alone, with the temperature put back), the bounds (once per setting, four resends per
call, at most two effort steps), that only a refusal of a setting the request carried is answered and anything else
propagates as it was raised, that only what a served request proved is learned (never enforcement, a 404's
temperature only when dropping it alone was served, a temperature refused beside reasoning only beside reasoning),
that what is learned is shared by every caller of one store and by no other, that concurrent calls refused together
each resend and the fact is logged once, and that the result reports what the SERVED request carried. Each shipped
reader is held to recorded-shape bodies, OpenRouter's production 404 and Anthropic's live effort list among them.
The pure choice of what to send is ``test_closest_settings.py``; the adapter over it, ``test_models_adapter*.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.negotiation import (
    EFFORT_LADDER,
    MAX_RESENDS,
    LearnedSettings,
    SettingRefusal,
    SettingFacts,
    WireSettings,
    negotiate,
    read_anthropic_refusal,
    read_openai_refusal,
    read_router_refusal,
)
from threetears.evals.schema import ClientRequestSettings

#: The logger the loop writes a learned fact to.
LOOP_LOGGER = "threetears.evals.negotiation.loop"

#: OpenRouter's answer when ``require_parameters`` leaves no endpoint, exactly as Discodon's G2 campaign met it on
#: ``.24`` (every judge call 404'd for the judge temperature).
NO_ENDPOINT = {"error": {"code": 404, "message": "No endpoints found that can handle the requested parameters."}}

#: The judge's asked settings, with the judge temperature, under enforcement.
JUDGE = WireSettings(max_tokens=10240, temperature=0.0, reasoning_effort="high", strict_output=True)


class Answered(Exception):
    """A scripted provider's answer to a request it did not serve: its status and JSON body."""

    def __init__(self, status: int, body: object) -> None:
        super().__init__(f"answered {status}")
        self.status = status
        self.body = body


def router_reader(exc: Exception) -> SettingRefusal | None:
    return read_router_refusal(exc.status, exc.body) if isinstance(exc, Answered) else None


class ScriptedRouter:
    """Refuses (404, no endpoint) any request carrying a temperature or enforcement; serves the rest."""

    def __init__(self, *, refuse_temperature: bool = True, refuse_enforcement: bool = True) -> None:
        self.sent: list[WireSettings] = []
        self.refuse_temperature = refuse_temperature
        self.refuse_enforcement = refuse_enforcement

    async def __call__(self, wire: WireSettings, facts: SettingFacts) -> str:
        self.sent.append(wire)
        await asyncio.sleep(0)
        if (self.refuse_temperature and wire.temperature is not None) or (
            self.refuse_enforcement and wire.strict_output
        ):
            raise Answered(404, NO_ENDPOINT)
        return f"served {len(self.sent)}"


async def _negotiate(send: Any, learned: LearnedSettings, **kwargs: Any) -> Any:
    asked = kwargs.pop("asked", JUDGE)
    return await negotiate(
        asked,
        provider_family=kwargs.pop("provider_family", "router"),
        model=kwargs.pop("model", "openai/gpt-6-luna"),
        send=send,
        read_refusal=kwargs.pop("read_refusal", router_reader),
        learned=learned,
        **kwargs,
    )


# --- the order, and what is reported ------------------------------------------------------------------------------


async def test_a_temperature_goes_first_keeping_enforcement_then_enforcement_and_the_served_one_is_reported() -> None:
    """No endpoint under enforcement: the temperature goes first, then enforcement, with the temperature put back."""
    router = ScriptedRouter(refuse_temperature=False)
    learned = LearnedSettings()
    done = await _negotiate(router, learned)
    first, without_temperature, unenforced = router.sent
    assert first == JUDGE
    assert without_temperature.temperature is None and without_temperature.strict_output is True
    assert unenforced.temperature == 0.0 and unenforced.strict_output is False, "dropping it alone placed nothing"
    assert done.reply == "served 3" and done.wire == unenforced
    assert done.temperature == 0.0
    assert done.sent_settings == ClientRequestSettings(max_tokens=10240, reasoning_effort="high", strict_output=False)
    assert [refusal.setting for refusal in done.refusals] == ["enforcement"]
    assert learned.facts("router", "openai/gpt-6-luna") == SettingFacts(), "nothing proved, so nothing learned"


async def test_a_temperature_still_refused_when_enforcement_goes_is_answered_without_it() -> None:
    """Neither placed enforced, and the temperature refused unenforced too: the third request goes without both."""
    router = ScriptedRouter()
    with pytest.raises(Answered):
        await _negotiate(router, LearnedSettings())
    sent = [(wire.temperature, wire.strict_output) for wire in router.sent]
    assert sent == [(0.0, True), (None, True), (0.0, False)], "a 404 on an unenforced request is not a lack: raised"


async def test_dropping_the_temperature_alone_is_kept_when_that_is_served() -> None:
    """Discodon's production sequence: no endpoint takes a temperature, and every one takes the rest enforced."""
    router = ScriptedRouter(refuse_enforcement=False)
    done = await _negotiate(router, LearnedSettings())
    assert len(router.sent) == 2 and done.wire.strict_output is True and done.temperature is None
    assert done.sent_settings == ClientRequestSettings(max_tokens=10240, reasoning_effort="high", strict_output=True)


@pytest.mark.parametrize("declared", [SettingFacts(accepts_temperature=True), SettingFacts()], ids=["taken", "unknown"])
async def test_a_temperature_declared_taken_still_goes_first_on_a_404(declared: SettingFacts) -> None:
    """A catalogue lists what SOME endpoint takes, never that one takes it beside the rest: it cannot rule it out."""
    router = ScriptedRouter(refuse_enforcement=False)
    learned = LearnedSettings()
    done = await _negotiate(router, learned, declared=declared)
    assert [wire.temperature for wire in router.sent] == [0.0, None] and done.wire.strict_output is True
    assert learned.facts("router", "openai/gpt-6-luna").temperature_beside_reasoning is False


async def test_a_request_served_at_once_reports_what_it_carried_and_learns_nothing() -> None:
    router = ScriptedRouter(refuse_temperature=False, refuse_enforcement=False)
    learned = LearnedSettings()
    done = await _negotiate(router, learned)
    assert router.sent == [JUDGE] and done.refusals == () and done.notes == ()
    assert done.temperature == 0.0 and done.sent_settings == JUDGE.reported()
    assert learned.facts("router", "openai/gpt-6-luna") == SettingFacts()


async def test_the_served_settings_are_the_closest_ones_and_the_notes_say_why() -> None:
    """A budget on a router is sent as the word its share maps to, and the result reports the word."""
    router = ScriptedRouter(refuse_temperature=False, refuse_enforcement=False)
    asked = WireSettings(max_tokens=10240, reasoning_max_tokens=8192, strict_output=True)
    done = await _negotiate(router, LearnedSettings(), asked=asked)
    assert done.sent_settings == ClientRequestSettings(max_tokens=10240, reasoning_effort="high", strict_output=True)
    assert len(done.notes) == 1 and "8192" in done.notes[0]


# --- the bounds --------------------------------------------------------------------------------------------------


async def test_a_call_is_resent_at_most_max_resends_times() -> None:
    """A temperature beside reasoning, two effort steps, the temperature again without reasoning: the fifth is raised.

    The temperature was refused while the request reasoned, so it came back once reasoning went, and was refused
    again: a new fact, and the fourth resend. The format's refusal after it is the fifth, and is raised. The bound is
    the public ``MAX_RESENDS``, which a host sizes its wall-clock ceiling from, so the two cannot drift apart.
    """

    async def send(wire: WireSettings, facts: SettingFacts) -> str:
        sent.append(wire)
        if wire.temperature is not None:
            raise Answered(400, {"message": "no", "param": "temperature", "code": "unsupported_parameter"})
        if wire.reasoning_effort is not None:
            raise Answered(400, {"message": "no", "param": "reasoning_effort", "code": "unsupported_value"})
        if wire.strict_output:
            raise Answered(400, {"message": "no", "param": "response_format", "code": "unsupported_value"})
        raise Answered(400, {"message": "no", "param": "max_tokens", "code": "unsupported_value"})

    sent: list[WireSettings] = []
    with pytest.raises(Answered):
        await negotiate(
            WireSettings(max_tokens=1000, temperature=0.2, reasoning_effort="max", strict_output=True),
            provider_family="openai",
            model="m",
            send=send,
            read_refusal=lambda exc: read_openai_refusal(exc.status, exc.body) if isinstance(exc, Answered) else None,
            learned=LearnedSettings(),
        )
    shapes = [(wire.temperature, wire.reasoning_effort, wire.strict_output) for wire in sent]
    assert shapes == [
        (0.2, "xhigh", True),
        (None, "xhigh", True),
        (None, "high", True),
        (0.2, None, True),
        (None, None, True),
    ]
    assert MAX_RESENDS == 4 and len(sent) == MAX_RESENDS + 1, "one first send, then MAX_RESENDS resends, all refused"


async def test_an_effort_refused_word_after_word_never_walks_the_ladder_into_a_failure() -> None:
    """A model that takes budgets only and refuses each word by name: one step, then no word at all, as a budget.

    The reviewers' trace: before, it walked high, medium, xhigh, low and failed on the fifth refusal.
    """

    async def send(wire: WireSettings, facts: SettingFacts) -> str:
        sent.append(wire)
        if wire.reasoning_max_tokens is not None:
            return "served"
        if facts.adaptive_reasoning is not False:
            raise Answered(400, _anthropic('"thinking.type.adaptive" is not supported for this model.'))
        raise Answered(400, _anthropic(f"effort '{wire.reasoning_effort}' is not supported for this model."))

    sent: list[WireSettings] = []
    learned = LearnedSettings()
    done = await negotiate(
        WireSettings(max_tokens=10240, reasoning_effort="high", strict_output=True),
        provider_family="anthropic",
        model="m",
        send=send,
        read_refusal=lambda exc: read_anthropic_refusal(exc.status, exc.body) if isinstance(exc, Answered) else None,
        learned=learned,
    )
    assert [(wire.reasoning_effort, wire.reasoning_max_tokens) for wire in sent] == [
        ("high", None),
        ("high", None),
        ("medium", None),
        (None, 8192),
    ]
    assert done.sent_settings == ClientRequestSettings(max_tokens=10240, reasoning_max_tokens=8192, strict_output=True)
    assert learned.facts("anthropic", "m").reasoning_efforts == (), "learned to take no effort word"


async def test_an_effort_refused_with_the_words_it_takes_steps_once_to_the_closest_and_then_to_none() -> None:
    """OpenAI's listed words are learned and the closest sent; that one refused too, no effort word is sent."""

    async def send(wire: WireSettings, facts: SettingFacts) -> str:
        sent.append(wire.reasoning_effort)
        if wire.reasoning_effort is None:
            return "served"
        raise Answered(
            400,
            {
                "message": "Unsupported value: 'reasoning_effort' does not support 'xhigh' with this model. Supported "
                "values are: 'low', 'medium', and 'high'.",
                "param": "reasoning_effort",
                "code": "unsupported_value",
            },
        )

    sent: list[str | None] = []
    learned = LearnedSettings()
    await negotiate(
        WireSettings(max_tokens=1000, reasoning_effort="xhigh"),
        provider_family="openai",
        model="m",
        send=send,
        read_refusal=lambda exc: read_openai_refusal(exc.status, exc.body) if isinstance(exc, Answered) else None,
        learned=learned,
    )
    assert sent == ["xhigh", "high", None]
    assert learned.facts("openai", "m").reasoning_efforts == ()


#: Anthropic's answer, seen live on ``claude-haiku-5-5``, to an effort word the model does not take.
HAIKU_EFFORTS = "output_config.effort: Input should be 'low', 'medium', 'high', 'xhigh' or 'max'"


async def test_anthropic_s_listed_effort_words_are_learned_and_the_closest_is_sent() -> None:
    """The live ``claude-haiku-5-5`` message, verbatim: the words it lists are learned; ``minimal`` goes as ``low``."""

    async def send(wire: WireSettings, facts: SettingFacts) -> str:
        sent.append(wire.reasoning_effort)
        if wire.reasoning_effort == "minimal":
            raise Answered(400, _anthropic(HAIKU_EFFORTS))
        return "served"

    sent: list[str | None] = []
    learned = LearnedSettings()
    done = await negotiate(
        WireSettings(max_tokens=5120, reasoning_effort="minimal"),
        provider_family="anthropic",
        model="a-model-taking-five-effort-words",
        send=send,
        read_refusal=lambda exc: read_anthropic_refusal(exc.status, exc.body) if isinstance(exc, Answered) else None,
        learned=learned,
        declared=SettingFacts(reasoning_efforts=EFFORT_LADDER),  # a wrong declaration, which the refusal corrects
    )
    assert sent == ["minimal", "low"] and done.wire.reasoning_effort == "low"
    facts = learned.facts("anthropic", "a-model-taking-five-effort-words")
    assert facts.reasoning_efforts == ("low", "medium", "high", "xhigh", "max")


async def test_an_output_cap_above_the_model_s_own_is_resent_at_the_cap_the_provider_named() -> None:
    cap_refusal = "max_tokens: 64001 > 64000, which is the maximum allowed number of output tokens for claude-x"

    async def send(wire: WireSettings, facts: SettingFacts) -> str:
        sent.append(wire.max_tokens)
        if wire.max_tokens is not None and wire.max_tokens > 64000:
            raise Answered(400, _anthropic(cap_refusal))
        return "served"

    sent: list[int | None] = []
    learned = LearnedSettings()
    done = await negotiate(
        WireSettings(max_tokens=64001),
        provider_family="anthropic",
        model="m",
        send=send,
        read_refusal=lambda exc: read_anthropic_refusal(exc.status, exc.body) if isinstance(exc, Answered) else None,
        learned=learned,
    )
    assert sent == [64001, 64000] and done.sent_settings == ClientRequestSettings(max_tokens=64000)
    assert learned.facts("anthropic", "m").max_output_tokens == 64000


async def test_a_refusal_of_a_setting_already_dropped_is_raised() -> None:
    """The temperature refused, and the resend without it refused for the temperature again: raised, not looped."""
    refusal = {"type": "error", "error": {"type": "invalid_request_error", "message": "`temperature` is deprecated."}}

    async def send(wire: WireSettings, facts: SettingFacts) -> str:
        sent.append(wire)
        raise Answered(400, refusal)

    sent: list[WireSettings] = []
    learned = LearnedSettings()
    with pytest.raises(Answered):
        await negotiate(
            WireSettings(temperature=0.2),
            provider_family="anthropic",
            model="m",
            send=send,
            read_refusal=lambda exc: (
                read_anthropic_refusal(exc.status, exc.body) if isinstance(exc, Answered) else None
            ),
            learned=learned,
        )
    assert [wire.temperature for wire in sent] == [0.2, None]
    assert learned.facts("anthropic", "m") == SettingFacts(), "nothing is learned from a call never served"


async def test_a_router_refusal_of_a_request_that_carried_no_enforcement_is_raised() -> None:
    router = ScriptedRouter()
    with pytest.raises(Answered):
        await _negotiate(router, LearnedSettings(), asked=WireSettings(max_tokens=100, temperature=0.0))
    assert len(router.sent) == 1


async def test_any_other_error_propagates_unchanged_and_teaches_nothing() -> None:
    failure = ValueError("the transport broke")

    async def send(wire: WireSettings, facts: SettingFacts) -> str:
        sent.append(wire)
        raise failure

    sent: list[WireSettings] = []
    learned = LearnedSettings()
    with pytest.raises(ValueError) as raised:
        await _negotiate(send, learned)
    assert raised.value is failure and len(sent) == 1
    assert learned.facts("router", "openai/gpt-6-luna") == SettingFacts()


async def test_an_error_the_reader_does_not_recognise_is_raised_as_it_was() -> None:
    router_payment = Answered(402, {"error": {"code": 402, "message": "Insufficient credits"}})

    async def send(wire: WireSettings, facts: SettingFacts) -> str:
        raise router_payment

    with pytest.raises(Answered) as raised:
        await _negotiate(send, LearnedSettings())
    assert raised.value is router_payment


# --- what is learned, and who shares it ---------------------------------------------------------------------------


async def test_what_one_client_learned_every_client_of_the_store_sends_at_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    learned = LearnedSettings()
    first, second, stranger = (ScriptedRouter(refuse_enforcement=False) for _ in range(3))
    with caplog.at_level(logging.WARNING, logger=LOOP_LOGGER):
        await _negotiate(first, learned, provider="openrouter")
        later = await _negotiate(second, learned, provider="openrouter")
        await _negotiate(stranger, LearnedSettings(), provider="openrouter")
    assert len(first.sent) == 2 and len(second.sent) == 1 and later.refusals == ()
    assert second.sent[0].temperature is None and second.sent[0].strict_output is True
    assert len(stranger.sent) == 2, "a store of its own learned nothing from the others"
    logged = [r.getMessage() for r in caplog.records if r.name == LOOP_LOGGER]
    assert len(logged) == 2, "one fact, once per store"
    assert logged[0].startswith("eval.negotiation model refuses a temperature beside reasoning provider=openrouter")
    assert "declared=unknown" in logged[0] and "No endpoints found" in logged[0]
    assert "for the rest of this store's life" in logged[0]


async def test_enforcement_is_never_learned_and_each_call_starts_enforced() -> None:
    """A 404 can be an outage: dropping enforcement holds for the call it was refused on, and no other."""
    learned = LearnedSettings()
    router = ScriptedRouter(refuse_temperature=False)
    await _negotiate(router, learned)
    await _negotiate(router, learned)
    assert [wire.strict_output for wire in router.sent] == [True, True, False, True, True, False]
    learned.learn("router", "m", [SettingRefusal("enforcement", "no")], declared=SettingFacts(), sent=JUDGE)
    assert learned.facts("router", "m") == SettingFacts(), "not even when handed one directly"


async def test_a_404_s_temperature_is_not_learned_when_a_later_refusal_was_answered_before_it_was_served() -> None:
    """Dropping the temperature drew an effort refusal, not a placement: the served request proved the effort only."""

    async def send(wire: WireSettings, facts: SettingFacts) -> str:
        sent.append(wire)
        if wire.temperature is not None:
            raise Answered(404, NO_ENDPOINT)
        if wire.reasoning_effort == "high":
            upstream = {"message": "no", "param": "reasoning_effort", "code": "unsupported_value"}
            raise Answered(400, _relayed({"error": upstream}, "OpenAI"))
        return "served"

    sent: list[WireSettings] = []
    learned = LearnedSettings()
    done = await _negotiate(send, learned)
    assert [(wire.temperature, wire.reasoning_effort) for wire in sent] == [
        (0.0, "high"),
        (None, "high"),
        (None, "medium"),
    ]
    assert done.temperature is None, "the served request carried none, and says so"
    facts = learned.facts("router", "openai/gpt-6-luna")
    assert facts.refused_efforts == {"high"}
    assert facts.accepts_temperature is None and facts.temperature_beside_reasoning is None


async def test_a_temperature_refused_beside_reasoning_is_still_sent_to_a_call_that_does_not_reason() -> None:
    """A temperature refused beside an effort is learned with that condition: a call asking ``none`` still sends it."""
    learned = LearnedSettings()
    await _negotiate(ScriptedRouter(refuse_enforcement=False), learned)
    for effort in ("none", None):
        plain = ScriptedRouter(refuse_temperature=False, refuse_enforcement=False)
        done = await _negotiate(plain, learned, asked=replace(JUDGE, reasoning_effort=effort))
        assert plain.sent[0].temperature == 0.0 and done.temperature == 0.0
    reasoning = ScriptedRouter(refuse_temperature=False, refuse_enforcement=False)
    await _negotiate(reasoning, learned, asked=replace(JUDGE, reasoning_effort="low"))
    assert reasoning.sent[0].temperature is None, "beside any reasoning word, still dropped"


async def test_concurrent_calls_refused_together_each_resend_and_the_fact_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    learned = LearnedSettings()
    router = ScriptedRouter(refuse_enforcement=False)
    with caplog.at_level(logging.WARNING, logger=LOOP_LOGGER):
        done = await asyncio.gather(*(_negotiate(router, learned) for _ in range(5)))
    assert all(result.temperature is None for result in done)
    assert sum(wire.temperature is not None for wire in router.sent) == 5, "each was refused once"
    assert len(router.sent) == 10, "and each resent once"
    assert len([r for r in caplog.records if r.name == LOOP_LOGGER]) == 1
    assert learned.facts("router", "openai/gpt-6-luna").temperature_beside_reasoning is False


async def test_a_refusal_that_contradicts_a_declared_fact_wins_and_the_log_says_where_to_correct_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    refusal = {"type": "error", "error": {"type": "invalid_request_error", "message": "`temperature` is deprecated."}}

    async def send(wire: WireSettings, facts: SettingFacts) -> str:
        if wire.temperature is not None:
            raise Answered(400, refusal)
        return "served"

    learned = LearnedSettings(client="host.client", declared_by="catalogue", contradiction="fix the catalogue")
    with caplog.at_level(logging.WARNING, logger=LOOP_LOGGER):
        done = await negotiate(
            WireSettings(temperature=0.2),
            provider_family="anthropic",
            model="m",
            send=send,
            read_refusal=lambda exc: (
                read_anthropic_refusal(exc.status, exc.body) if isinstance(exc, Answered) else None
            ),
            learned=learned,
            declared=SettingFacts(accepts_temperature=True),
        )
    assert done.temperature is None and learned.facts("anthropic", "m").accepts_temperature is False
    assert "host.client model refuses a temperature" in caplog.text
    assert "catalogue=accepts (fix the catalogue)" in caplog.text


# --- the readers, over recorded-shape bodies ----------------------------------------------------------------------


def _relayed(upstream: dict[str, Any], provider_name: str) -> dict[str, Any]:
    """OpenRouter's 400 relaying an upstream's own error, its body as text in ``metadata.raw``."""
    return {
        "error": {
            "code": 400,
            "message": "Provider returned error",
            "metadata": {"raw": json.dumps(upstream), "provider_name": provider_name},
        }
    }


OPENAI_TEMPERATURE = {
    "error": {
        "message": "Unsupported parameter: 'temperature' is not supported with this model.",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "unsupported_parameter",
    }
}

ANTHROPIC_BUDGET = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": '"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" and '
        '"output_config.effort" to control thinking behavior.',
    },
}


@pytest.mark.parametrize(
    ("status", "body", "setting"),
    [
        (404, NO_ENDPOINT, "enforcement"),
        (404, json.dumps(NO_ENDPOINT), "enforcement"),  # the SDK keeps the body as text
        (404, json.dumps(NO_ENDPOINT).encode(), "enforcement"),
        (400, _relayed(OPENAI_TEMPERATURE, "OpenAI"), "temperature"),
        (400, _relayed(ANTHROPIC_BUDGET, "Anthropic"), "budget"),
        (400, {"error": {"code": 400, "message": "temperature is not supported"}}, None),  # its own prose: not read
        (400, {"error": {"code": 400, "message": "bad request"}}, None),
        (402, {"error": {"code": 402, "message": "Insufficient credits"}}, None),
        (404, "<html>not found</html>", None),
        (404, {"detail": "no error object"}, None),
    ],
)
def test_the_router_reader(status: int, body: object, setting: str | None) -> None:
    refusal = read_router_refusal(status, body)
    assert (refusal.setting if refusal else None) == setting
    if status == 404 and refusal is not None:
        assert refusal.message == "No endpoints found that can handle the requested parameters."


def _anthropic(message: str, kind: str = "invalid_request_error") -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (400, _anthropic("`temperature` is deprecated for this model."), ("temperature", None)),
        (400, ANTHROPIC_BUDGET, ("budget", None)),
        (400, _anthropic('"thinking.type.adaptive" is not supported for this model.'), ("adaptive", None)),
        (400, _anthropic('"thinking.type.disabled" is not supported for this model.'), ("disable", None)),
        (400, _anthropic("output_config.effort: this model does not support effort."), ("effort", None)),
        (400, _anthropic("output_config.effort: 'xhigh' is not supported for this model."), ("effort_word", "xhigh")),
        (400, _anthropic("output_config.format: structured output is not supported."), ("format", None)),
        (400, _anthropic(HAIKU_EFFORTS), ("effort_word", None)),  # live on claude-haiku-5-5: the word is the loop's
        (
            400,
            _anthropic("max_tokens: 64001 > 64000, which is the maximum allowed number of output tokens for claude-x"),
            ("max_tokens", None),
        ),
        (400, json.dumps(_anthropic("`temperature` is deprecated for this model.")), ("temperature", None)),
        # The caller's errors and the credit notice are not a model's lack.
        (400, _anthropic("thinking.budget_tokens: Input should be greater than or equal to 1024"), None),
        (400, _anthropic("Your credit balance is too low to access the Anthropic API."), None),
        (400, _anthropic("`temperature` is deprecated.", kind="authentication_error"), None),
        (500, _anthropic("`temperature` is deprecated for this model."), None),
    ],
)
def test_the_anthropic_reader(status: int, body: object, expected: tuple[str, str | None] | None) -> None:
    refusal = read_anthropic_refusal(status, body)
    assert ((refusal.setting, refusal.word) if refusal else None) == expected


def test_the_anthropic_reader_takes_the_listed_words_and_the_named_cap() -> None:
    listed = read_anthropic_refusal(400, _anthropic(HAIKU_EFFORTS))
    assert listed is not None and listed.accepted == ("low", "medium", "high", "xhigh", "max")
    capped = read_anthropic_refusal(
        400, _anthropic("max_tokens: 64001 > 64000, which is the maximum allowed number of output tokens for claude-x")
    )
    assert capped is not None and capped.limit == 64000


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (OPENAI_TEMPERATURE, ("temperature", None)),
        (OPENAI_TEMPERATURE["error"], ("temperature", None)),  # the SDK keeps the inner error object
        (
            {
                "message": "Unsupported value: 'reasoning_effort' does not support 'xhigh' with this model. Supported "
                "values are: 'low', 'medium', and 'high'.",
                "param": "reasoning_effort",
                "code": "unsupported_value",
            },
            ("effort_word", ("low", "medium", "high")),
        ),
        (
            {"message": "Unsupported parameter: 'reasoning_effort'.", "param": "reasoning_effort", "code": None},
            None,
        ),
        (
            {
                "message": "Unsupported parameter: 'reasoning_effort'.",
                "param": "reasoning_effort",
                "code": "unsupported_parameter",
            },
            ("effort", None),
        ),
        (
            {
                "message": "Invalid parameter: 'response_format' of type 'json_schema' is not supported with this "
                "model.",
                "param": "response_format",
                "code": None,
            },
            ("format", None),
        ),
        ({"message": "Unsupported parameter: 'top_k'.", "param": "top_k", "code": "unsupported_parameter"}, None),
        (
            {
                "message": "max_tokens is too large: 200000. This model supports at most 128000 completion tokens, "
                "whereas you provided 200000.",
                "param": "max_tokens",
                "code": "invalid_value",
            },
            ("max_tokens", None),
        ),
        ({"message": "max_tokens is too large.", "param": "max_tokens", "code": "invalid_value"}, None),
    ],
)
def test_the_openai_reader(body: object, expected: tuple[str, object] | None) -> None:
    refusal = read_openai_refusal(400, body)
    assert ((refusal.setting, refusal.accepted) if refusal else None) == expected
    assert read_openai_refusal(429, body) is None
    if refusal is not None and refusal.setting == "max_tokens":
        assert refusal.limit == 128000


async def test_an_effort_word_the_reader_left_unnamed_is_the_one_the_request_carried() -> None:
    """OpenAI's ``unsupported_value`` names the parameter; the loop fills the word from what was sent."""

    async def send(wire: WireSettings, facts: SettingFacts) -> str:
        if wire.reasoning_effort == "xhigh":
            raise Answered(400, {"message": "nope", "param": "reasoning_effort", "code": "unsupported_value"})
        return "served"

    done = await negotiate(
        WireSettings(max_tokens=1000, reasoning_effort="xhigh"),
        provider_family="openai",
        model="m",
        send=send,
        read_refusal=lambda exc: read_openai_refusal(exc.status, exc.body) if isinstance(exc, Answered) else None,
        learned=LearnedSettings(),
    )
    assert done.refusals[0].word == "xhigh" and done.wire.reasoning_effort == "high"
