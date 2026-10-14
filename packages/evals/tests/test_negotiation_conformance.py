"""``check_refusal_negotiation``: it passes a client that negotiates, and names each mistake of one that does not.

The check scripts three routers (the temperature refused under enforcement; no endpoint under enforcement at all;
an upstream refusing the output cap and the effort word, naming what it takes) and watches a host's own client: that
it resends in order, reports what the served request carried, and sends the next call as the first proved, at once.
Each wrong client below makes one mistake a host's client could make (it never resends, drops enforcement first,
reports what it was asked or always unenforced, reports no sent settings, forgets what it learned, keeps enforcement
dropped or a temperature nothing proved, or was built without a temperature or an effort word), and the failure must
name that mistake, against the router that shows it; the guide's own client passing it is ``test_doc_blocks.py``.
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
ASKED = WireSettings(max_tokens=1024, temperature=0.0, reasoning_effort="high")


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
        self.declared = SettingFacts()

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
            asked,
            provider_family="router",
            model="m",
            send=send,
            read_refusal=read,
            learned=self.learned,
            declared=self.declared,
        )
        if self.mistake == "keeps_enforcement_dropped" and not done.wire.strict_output:
            self.declared = replace(self.declared, enforcement=False)
        if self.mistake == "keeps_unproven_temperature" and any(r.setting == "enforcement" for r in done.refusals):
            self.declared = replace(self.declared, accepts_temperature=False)
        if self.mistake == "no_sent_settings":
            return Unreporting(content="{}", temperature=done.temperature)
        sent = done.sent_settings
        if self.mistake == "reports_asked":
            sent = self.asked.reported()
        elif self.mistake == "always_unenforced":
            sent = replace(done.wire, strict_output=False).reported()
        return Completion(
            content="{}",
            input_tokens=None,
            output_tokens=None,
            reasoning_tokens=None,
            cost_usd=None,
            price_source=None,
            model="m",
            served_model=None,
            temperature=self.asked.temperature if self.mistake == "reports_asked" else done.temperature,
            stop_reason="end_turn",
            sent_settings=sent,
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


#: The three scripted routers, as each mistake names the one it showed against.
TEMPERATURE, NO_ENDPOINT, EFFORT_AND_CAP = (
    "[temperature refused]",
    "[no endpoint under enforcement]",
    ("[effort word and cap refused]"),
)


@pytest.mark.parametrize(
    ("mistake", "asked", "named"),
    [
        (
            "no_resend",
            ASKED,
            [
                f"{TEMPERATURE} the call raised Refused (answered 404) after 1 request(s)",
                f"{NO_ENDPOINT} the call raised Refused (answered 404) after 1 request(s)",
                f"{EFFORT_AND_CAP} the call raised Refused (answered 400) after 1 request(s)",
            ],
        ),
        (
            "enforcement_first",
            ASKED,
            [
                f"{TEMPERATURE} request 2 still carried the temperature",
                f"{TEMPERATURE} request 2 dropped enforcement before the temperature",
                f"{TEMPERATURE} the completion reports no temperature, and the served request carried 0.0",
                f"{TEMPERATURE} the next call sent 2 request(s)",
                f"{NO_ENDPOINT} request 2 still carried the temperature",
                f"{NO_ENDPOINT} request 2 dropped enforcement before the temperature",
                f"{NO_ENDPOINT} the completion reports no temperature, and the served request carried 0.0",
                f"{EFFORT_AND_CAP} the call raised AssertionError",
                f"{EFFORT_AND_CAP} request 2 still carried max_tokens 1024, above the 512 the upstream named",
                f"{EFFORT_AND_CAP} request 3 still carried max_tokens 1024, above the 512 the upstream named",
                f"{EFFORT_AND_CAP} request 3 still carried effort 'high', which the upstream refused",
            ],
        ),
        (
            "reports_asked",
            ASKED,
            [
                f"{TEMPERATURE} the completion reports temperature 0.0, and the served request carried none",
                f"{TEMPERATURE} the completion reports strict_output=False, and the served request was enforced",
                f"{EFFORT_AND_CAP} the completion reports strict_output=False, and the served request was enforced",
                f"{EFFORT_AND_CAP} the completion reports max_tokens=1024, and the served request carried 512",
                f"{EFFORT_AND_CAP} the completion reports reasoning_effort='high', and the served request carried "
                "'medium'",
            ],
        ),
        (
            "always_unenforced",
            ASKED,
            [
                f"{TEMPERATURE} the completion reports strict_output=False, and the served request was enforced",
                f"{EFFORT_AND_CAP} the completion reports strict_output=False, and the served request was enforced",
            ],
        ),
        (
            "no_sent_settings",
            ASKED,
            [
                f"{router} the completion reports no sent_settings"
                for router in (TEMPERATURE, NO_ENDPOINT, EFFORT_AND_CAP)
            ],
        ),
        (
            "forgets",
            ASKED,
            [f"{TEMPERATURE} the next call sent 2 request(s)", f"{EFFORT_AND_CAP} the next call sent 3"],
        ),
        ("keeps_enforcement_dropped", ASKED, [f"{NO_ENDPOINT} the next call's first request was not enforced"]),
        ("keeps_unproven_temperature", ASKED, [f"{NO_ENDPOINT} the next call's first request carried no temperature"]),
        (None, replace(ASKED, temperature=None), [f"{TEMPERATURE} the first request carried no temperature"]),
        (None, replace(ASKED, reasoning_effort=None), [f"{TEMPERATURE} the first request carried no reasoning effort"]),
    ],
)
async def test_each_mistake_is_named(mistake: str | None, asked: WireSettings, named: list[str]) -> None:
    with pytest.raises(NegotiationConformanceFailure) as failed:
        await check_refusal_negotiation(lambda transport: Client(transport, mistake=mistake, asked=asked))
    found = failed.value.mistakes
    for phrase in named:
        assert any(sentence.startswith(phrase) for sentence in found), (phrase, found)
    assert len(found) == len(named), found


async def test_a_call_that_raised_is_chained_to_the_failure() -> None:
    with pytest.raises(NegotiationConformanceFailure) as failed:
        await check_refusal_negotiation(lambda transport: Client(transport, mistake="no_resend"))
    assert isinstance(failed.value.__cause__, Refused)
