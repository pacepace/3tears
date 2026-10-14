"""``check_refusal_negotiation``: it passes a client that negotiates, and names each mistake of one that does not.

The check scripts a router that refuses a temperature, then enforcement, and watches a host's own client: that it
resends in that order, reports what the served request carried, and sends the next call that way at once. Each wrong
client below makes one mistake a host's client could make (it never resends, drops enforcement first, echoes what it
was asked, reports no sent settings, forgets what it learned, or was built without a temperature), and the failure
must name that mistake; the guide's own client passing it is ``test_doc_blocks.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import pytest

from threetears.evals.negotiation import LearnedSettings, SettingFacts, WireSettings, negotiate, read_router_refusal
from threetears.evals.schema import ClientRequestSettings, StopReason
from threetears.evals.testing import NegotiationConformanceFailure, RouterTransport, check_refusal_negotiation

#: What the scripted client is asked for: an output cap, a temperature and an effort word.
ASKED = WireSettings(max_tokens=1024, temperature=0.0, reasoning_effort="low")


class Refused(Exception):
    def __init__(self, status: int, body: object) -> None:
        super().__init__(f"answered {status}")
        self.status = status
        self.body = body


@dataclass(frozen=True)
class Completion:
    content: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    price_source: str | None
    model: str
    served_model: str | None
    temperature: float | None
    stop_reason: StopReason
    sent_settings: ClientRequestSettings | None


@dataclass(frozen=True)
class Unreporting:
    """A completion type with no ``sent_settings`` member at all."""

    content: str
    temperature: float | None


def _body(wire: WireSettings, response_format: object) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "m", "messages": []}
    if wire.max_tokens is not None:
        body["max_tokens"] = wire.max_tokens
    if wire.temperature is not None:
        body["temperature"] = wire.temperature
    if wire.reasoning_effort is not None:
        body["reasoning"] = {"effort": wire.reasoning_effort}
    if response_format is not None:
        body["response_format"] = response_format
        body["provider"] = {"require_parameters": wire.strict_output}
    return body


class Client:
    """A host's client over the shared loop, with one mistake switched on by name, or none."""

    def __init__(self, post: RouterTransport, *, mistake: str | None = None, asked: WireSettings = ASKED) -> None:
        self.post, self.mistake, self.asked = post, mistake, asked
        self.learned = LearnedSettings()

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> object:
        if self.mistake == "forgets":
            self.learned = LearnedSettings()
        asked = replace(self.asked, strict_output=response_format is not None)
        if self.mistake == "enforcement_first":
            return await self._enforcement_first(asked, response_format)

        async def send(wire: WireSettings, facts: SettingFacts) -> Mapping[str, object]:
            status, answer = await self.post(_body(wire, response_format))
            if status != 200:
                raise Refused(status, answer)
            return answer

        def read(exc: Exception) -> Any:
            if self.mistake == "no_resend" or not isinstance(exc, Refused):
                return None
            return read_router_refusal(exc.status, exc.body)

        done = await negotiate(
            asked, provider_family="router", model="m", send=send, read_refusal=read, learned=self.learned
        )
        if self.mistake == "no_sent_settings":
            return Unreporting(content="{}", temperature=done.temperature)
        echoed = self.mistake == "echoes_asked"
        return Completion(
            content="{}",
            input_tokens=None,
            output_tokens=None,
            reasoning_tokens=None,
            cost_usd=None,
            price_source=None,
            model="m",
            served_model=None,
            temperature=asked.temperature if echoed else done.temperature,
            stop_reason="end_turn",
            sent_settings=asked.reported() if echoed else done.sent_settings,
        )

    async def _enforcement_first(self, asked: WireSettings, response_format: object) -> object:
        """A hand-rolled order: enforcement goes before the temperature."""
        for wire in (asked, replace(asked, strict_output=False), replace(asked, strict_output=False, temperature=None)):
            status, _ = await self.post(_body(wire, response_format))
            if status == 200:
                return Completion("{}", None, None, None, None, None, "m", None, None, "end_turn", wire.reported())
        raise AssertionError("the scripted router served nothing")

    async def aclose(self) -> None:
        return None


async def test_a_client_that_negotiates_passes() -> None:
    await check_refusal_negotiation(lambda transport: Client(transport))


@pytest.mark.parametrize(
    ("mistake", "asked", "named"),
    [
        ("no_resend", ASKED, ["the call raised Refused (answered 404) after 1 request(s)"]),
        (
            "enforcement_first",
            ASKED,
            [
                "request 2 still carried the temperature",
                "request 2 dropped enforcement before the temperature",
                "the next call sent 3 request(s)",
            ],
        ),
        (
            "echoes_asked",
            ASKED,
            ["reports temperature 0.0, and the served request carried none", "reports strict_output=True"],
        ),
        ("no_sent_settings", ASKED, ["the completion reports no sent_settings"]),
        ("forgets", ASKED, ["the next call sent 3 request(s)"]),
        (None, replace(ASKED, temperature=None), ["the first request carried no temperature"]),
    ],
)
async def test_each_mistake_is_named(mistake: str | None, asked: WireSettings, named: list[str]) -> None:
    with pytest.raises(NegotiationConformanceFailure) as failed:
        await check_refusal_negotiation(lambda transport: Client(transport, mistake=mistake, asked=asked))
    found = failed.value.mistakes
    for phrase in named:
        assert any(phrase in sentence for sentence in found), (phrase, found)
    assert len(found) == len(named), found


async def test_a_call_that_raised_is_chained_to_the_failure() -> None:
    with pytest.raises(NegotiationConformanceFailure) as failed:
        await check_refusal_negotiation(lambda transport: Client(transport, mistake="no_resend"))
    assert isinstance(failed.value.__cause__, Refused)
