"""The refusal negotiation check: whether a host's own client answers a router's refusals in order, and says so.

A client that never fails a request because a model lacks a setting (:mod:`threetears.evals.negotiation`) has three
things to get right that no completion conformance check can see: it resends when the router refuses, in the fixed
order (the temperature first, keeping enforcement, and only then enforcement, for that call alone); it reports what
the SERVED request carried, not what it was asked; and it remembers what a served request proved, and nothing it did
not, so the next call goes out as it should at once. This check scripts the router and watches. Under pytest::

    from threetears.evals.testing import check_refusal_negotiation

    async def test_my_client_negotiates() -> None:
        await check_refusal_negotiation(lambda transport: MyClient(..., post=transport))

**What the host wires.** ``build_client`` gets a :data:`RouterTransport` and returns the host's client built to send
each request body through it instead of the network, asking for a temperature, an output cap and a reasoning effort
word (:func:`~threetears.evals.negotiation.wire_settings_for` over
:data:`~threetears.evals.run.JUDGE_REQUEST_SETTINGS` asks for all three), and with a store of learned settings of
its own (a fresh :class:`~threetears.evals.negotiation.LearnedSettings`), so nothing learned elsewhere decides the
first request. The transport takes the JSON body the client would POST to a router's chat completions endpoint and
returns the status and JSON body the router answers with. The check builds one client per scripted router.

**What is scripted.** Every call asks for a strict reply schema, so its first request is enforced
(``provider.require_parameters``). Three routers, two calls each:

* **The temperature refused.** Any enforced request carrying a ``temperature`` is refused with the router's 404
  ("No endpoints found that can handle the requested parameters."); any other is served. A correct client sends two
  requests (the second without the temperature, still enforced), reports ``temperature=None`` and
  ``strict_output=True``, and sends its next call at once without the temperature, enforced.
* **No endpoint under enforcement.** Any enforced request is refused with the 404. A correct client sends three
  requests: without the temperature and still enforced, then with the temperature back and without enforcement
  (dropping the temperature alone placed nothing, so it proved nothing); it reports ``strict_output=False``, and its
  next call starts enforced and with the temperature again, since enforcement is dropped per call and never kept.
* **The effort word and the cap refused.** An upstream behind the router refuses (a 400 the router relays, in
  OpenAI's shape) any output cap above half the first request's, naming its own, and then the first request's effort
  word, listing the words it takes. A correct client sends three requests, reports the cap and the closest listed
  word it was served with, and sends its next call that way at once.

**What it cannot see.** Any refusal shape but these, and any order but this one: a client that answers these
correctly may still mishandle another. It reads a request's settings off the router's body (``temperature``,
``max_tokens`` or ``max_completion_tokens``, ``reasoning.effort``, ``provider.require_parameters``), so a client that
speaks another shape cannot be checked with it.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Final

from threetears.evals.schema import REASONING_EFFORTS, CompletionClient, sent_settings_of

__all__ = ["NegotiationConformanceFailure", "RouterTransport", "check_refusal_negotiation"]

#: A scripted router: the JSON body a client would POST in, the status and JSON body the router answers with out.
RouterTransport = Callable[[Mapping[str, object]], Awaitable[tuple[int, Mapping[str, object]]]]

#: The router's answer when ``require_parameters`` leaves no endpoint for a request, as production met it.
_NO_ENDPOINT: Final[Mapping[str, object]] = {
    "error": {"code": 404, "message": "No endpoints found that can handle the requested parameters."}
}

#: The strict reply schema every scripted call asks for, which asks the router for enforcement.
_REPLY_FORMAT: Final[dict[str, object]] = {
    "type": "json_schema",
    "json_schema": {
        "name": "negotiation_check",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"],
            "additionalProperties": False,
        },
    },
}

#: The served completion's content, which satisfies :data:`_REPLY_FORMAT`.
_CONTENT: Final = json.dumps({"verdict": "served"})

#: Where a mistake is pointed: what a host's client calls to get each of these right.
_REMEMBER: Final = "keep one LearnedSettings for the client's life, or the factory's"


class NegotiationConformanceFailure(AssertionError):
    """A host's client mishandled the router's refusals: each mistake it made, named.

    An :class:`AssertionError`, so a test runner reports it as a failed assertion. The message lists every mistake,
    one per line, each beginning with the scripted router it was made against, and :attr:`mistakes` holds them.

    Attributes:
        mistakes: Each mistake, as a sentence naming what the client did and what it should have done.
    """

    def __init__(self, mistakes: list[str]) -> None:
        """Name every mistake.

        Args:
            mistakes: The mistakes, in the order the check met them.
        """
        self.mistakes = tuple(mistakes)
        super().__init__("the client does not negotiate a router's refusals as it must:\n  " + "\n  ".join(mistakes))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _temperature(body: Mapping[str, object]) -> bool:
    """Whether the request carried a temperature."""
    return body.get("temperature") is not None


def _enforced(body: Mapping[str, object]) -> bool:
    """Whether the request carried ``provider.require_parameters``."""
    return _mapping(body.get("provider")).get("require_parameters") is True


def _cap(body: Mapping[str, object]) -> int | None:
    """The request's output cap, under either name a router takes, or ``None`` when it carried no integer one."""
    cap = body.get("max_tokens", body.get("max_completion_tokens"))
    return cap if isinstance(cap, int) and not isinstance(cap, bool) else None


def _effort(body: Mapping[str, object]) -> object:
    """The request's reasoning effort word."""
    return _mapping(body.get("reasoning")).get("effort")


def _served(body: Mapping[str, object]) -> tuple[int, Mapping[str, object]]:
    """The router serving ``body``: a completion whose content satisfies :data:`_REPLY_FORMAT`."""
    return 200, {
        "id": "gen-negotiation-check",
        "object": "chat.completion",
        "model": body.get("model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _CONTENT},
                "finish_reason": "stop",
                "native_finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }


def _relayed(upstream: Mapping[str, object]) -> tuple[int, Mapping[str, object]]:
    """The router relaying an upstream's 400, its body as text in ``error.metadata.raw``."""
    metadata = {"raw": json.dumps(upstream), "provider_name": "OpenAI"}
    return 400, {"error": {"code": 400, "message": "Provider returned error", "metadata": metadata}}


class _Router:
    """A scripted router: keeps every body it is sent and answers by its script; each case says what to expect."""

    name = ""

    def __init__(self) -> None:
        self.bodies: list[Mapping[str, object]] = []

    async def __call__(self, body: Mapping[str, object]) -> tuple[int, Mapping[str, object]]:
        self.bodies.append(body)
        return self.answer(body)

    def answer(self, body: Mapping[str, object]) -> tuple[int, Mapping[str, object]]:
        """The router's answer to ``body``."""
        raise NotImplementedError

    def first_call(self, bodies: list[Mapping[str, object]], mistakes: list[str]) -> None:
        """The first call's requests, as this router's script needs them."""
        raise NotImplementedError

    def report(self, completion: object, served: Mapping[str, object], mistakes: list[str]) -> None:
        """What the completion must report beyond the served body's settings: nothing, unless a case says."""

    def next_call(self, bodies: list[Mapping[str, object]], mistakes: list[str]) -> None:
        """The second call's requests, as what the first call proved needs them."""
        raise NotImplementedError

    def mistake(self, mistakes: list[str], sentence: str) -> None:
        mistakes.append(f"[{self.name}] {sentence}")


def _temperature_first(router: _Router, bodies: list[Mapping[str, object]], mistakes: list[str]) -> None:
    """Request 2 drops the temperature and keeps enforcement, whichever router refused request 1."""
    if len(bodies) >= 2:
        if _temperature(bodies[1]):
            router.mistake(mistakes, "request 2 still carried the temperature the router refused")
        if not _enforced(bodies[1]):
            router.mistake(
                mistakes,
                "request 2 dropped enforcement before the temperature: the temperature goes first, keeping enforcement",
            )


class _TemperatureRefused(_Router):
    """No endpoint takes the temperature under enforcement; every one takes the rest."""

    name = "temperature refused"

    def answer(self, body: Mapping[str, object]) -> tuple[int, Mapping[str, object]]:
        return (404, _NO_ENDPOINT) if _temperature(body) and _enforced(body) else _served(body)

    def first_call(self, bodies: list[Mapping[str, object]], mistakes: list[str]) -> None:
        _temperature_first(self, bodies, mistakes)
        if len(bodies) > 2:
            self.mistake(mistakes, f"the call sent {len(bodies)} requests, more than the two the order needs")

    def next_call(self, bodies: list[Mapping[str, object]], mistakes: list[str]) -> None:
        if len(bodies) != 1 or _temperature(bodies[0]) or not _enforced(bodies[0]):
            self.mistake(
                mistakes,
                f"the next call sent {len(bodies)} request(s), the first "
                f"{'with' if bodies and _temperature(bodies[0]) else 'without'} a temperature and "
                f"{'enforced' if bodies and _enforced(bodies[0]) else 'unenforced'}: it goes at once without the "
                f"temperature that dropping it alone proved refused, still enforced ({_REMEMBER})",
            )


class _NoEndpoint(_Router):
    """No endpoint takes the request under enforcement at all: an outage, as a client cannot tell it from a lack."""

    name = "no endpoint under enforcement"

    def answer(self, body: Mapping[str, object]) -> tuple[int, Mapping[str, object]]:
        return (404, _NO_ENDPOINT) if _enforced(body) else _served(body)

    def first_call(self, bodies: list[Mapping[str, object]], mistakes: list[str]) -> None:
        _temperature_first(self, bodies, mistakes)
        if len(bodies) >= 3 and _temperature(bodies[0]) and not _temperature(bodies[2]):
            self.mistake(
                mistakes,
                "request 3 went without the temperature: dropping it alone placed nothing, so it proved nothing and "
                "goes back when enforcement goes (negotiate puts it back)",
            )
        if len(bodies) > 3:
            self.mistake(mistakes, f"the call sent {len(bodies)} requests, more than the three the order needs")

    def next_call(self, bodies: list[Mapping[str, object]], mistakes: list[str]) -> None:
        if not bodies:
            self.mistake(mistakes, "the next call sent nothing through the transport")
            return
        if not _enforced(bodies[0]):
            self.mistake(
                mistakes,
                "the next call's first request was not enforced: enforcement was kept dropped beyond the call the "
                "router refused (a 404 can be an outage; each call starts enforced and drops it for itself alone)",
            )
        if not _temperature(bodies[0]):
            self.mistake(
                mistakes,
                "the next call's first request carried no temperature: a temperature refusal was kept that dropping "
                "it alone never proved (learn only what a served request proved: negotiate does)",
            )


class _EffortAndCap(_Router):
    """An upstream that takes half the first request's cap and every effort word but its first one, and says so."""

    name = "effort word and cap refused"

    def __init__(self) -> None:
        super().__init__()
        self.limit: int | None = None
        self.refused_word: str | None = None

    def answer(self, body: Mapping[str, object]) -> tuple[int, Mapping[str, object]]:
        cap, effort = _cap(body), _effort(body)
        if self.limit is None and cap is not None and isinstance(effort, str):
            self.limit, self.refused_word = max(cap // 2, 1), effort
        param = "max_tokens" if "max_tokens" in body else "max_completion_tokens"
        if self.limit is not None and cap is not None and cap > self.limit:
            message = (
                f"{param} is too large: {cap}. This model supports at most {self.limit} completion tokens, whereas "
                f"you provided {cap}."
            )
            return _relayed({"error": {"message": message, "type": "invalid_request_error", "param": param}})
        if self.refused_word is not None and effort == self.refused_word:
            taken = [word for word in reversed(REASONING_EFFORTS) if word not in ("none", self.refused_word)]
            listing = ", ".join(f"'{word}'" for word in taken[:-1]) + f", and '{taken[-1]}'"
            message = (
                f"Unsupported value: 'reasoning_effort' does not support '{effort}' with this model. Supported values "
                f"are: {listing}."
            )
            return _relayed(
                {
                    "error": {
                        "message": message,
                        "type": "invalid_request_error",
                        "param": "reasoning_effort",
                        "code": "unsupported_value",
                    }
                }
            )
        return _served(body)

    def first_call(self, bodies: list[Mapping[str, object]], mistakes: list[str]) -> None:
        for number, body in enumerate(bodies[1:], start=2):
            if self.limit is not None and (cap := _cap(body)) is not None and cap > self.limit:
                self.mistake(
                    mistakes,
                    f"request {number} still carried max_tokens {cap}, above the {self.limit} the upstream named",
                )
            if number > 2 and _effort(body) == self.refused_word:
                self.mistake(
                    mistakes, f"request {number} still carried effort {self.refused_word!r}, which the upstream refused"
                )
        if len(bodies) > 3:
            self.mistake(mistakes, f"the call sent {len(bodies)} requests, more than the three the refusals need")

    def next_call(self, bodies: list[Mapping[str, object]], mistakes: list[str]) -> None:
        if len(bodies) != 1:
            self.mistake(
                mistakes,
                f"the next call sent {len(bodies)} request(s): the output cap and the effort words the upstream named "
                f"are not kept ({_REMEMBER})",
            )


async def check_refusal_negotiation(build_client: Callable[[RouterTransport], CompletionClient]) -> None:
    """Fail, naming each mistake, unless the client answers a router's refusals in order and reports what was served.

    Three scripted routers (the module docstring says what each refuses and what a correct client does), a client
    built over each, and two calls on it, each with a strict reply schema: the first is refused and must be resent
    in order and reported as served; the second must go out as what the first proved, at once.

    Args:
        build_client: Builds the host's client over a scripted router (see the module docstring for how).

    Raises:
        NegotiationConformanceFailure: Naming every mistake: the setup's (a first request without a temperature, an
            output cap, an effort word or enforcement), the resending's (none, out of order, too many, a refused
            setting sent again), the reporting's (a temperature, cap, effort word or enforcement reported that the
            served request did not carry, or no ``sent_settings``), and the remembering's (what a served request
            proved forgotten, or what it did not prove kept: a temperature, or enforcement dropped beyond its call).
            Chained to the first exception a call raised, if any did.
    """
    mistakes: list[str] = []
    raised: Exception | None = None
    for router in (_TemperatureRefused(), _NoEndpoint(), _EffortAndCap()):
        failure, readable = await _run(router, build_client, mistakes)
        raised = raised if raised is not None else failure
        if not readable:
            break  # the client sends nothing, or is not built as the check needs: nothing after can be read
    if mistakes:
        raise NegotiationConformanceFailure(mistakes) from raised


async def _run(
    router: _Router, build_client: Callable[[RouterTransport], CompletionClient], mistakes: list[str]
) -> tuple[Exception | None, bool]:
    """Two calls on a client built over ``router``, each mistake added to ``mistakes``.

    Returns:
        What a call raised first, or ``None``; and whether the client's requests can be read at all (it sent one,
        and the first asked for everything the routers refuse).
    """
    client = build_client(router)
    raised: Exception | None = None
    try:
        try:
            completion = await client.generate(system="You answer.", user="Answer.", response_format=_REPLY_FORMAT)
        except Exception as exc:  # the check's finding: raised by the caller, chained, as the mistake it is
            completion, raised = None, exc
            router.mistake(
                mistakes,
                f"the call raised {type(exc).__name__} ({exc}) after {len(router.bodies)} request(s): a refusal the "
                "client should have answered by resending (hand its errors to a refusal reader through negotiate)",
            )
        first = list(router.bodies)
        if not first:
            router.mistake(
                mistakes, "the client sent nothing through the transport: build it to POST each body through it"
            )
            return raised, False
        if not _setup(router, first[0], mistakes):
            return raised, False
        router.first_call(first, mistakes)
        if completion is None:
            return raised, True
        _check_report(router, completion, first[-1], mistakes)
        try:
            await client.generate(system="You answer.", user="Answer.", response_format=_REPLY_FORMAT)
        except Exception as exc:  # the check's finding, as above
            raised = raised if raised is not None else exc
            router.mistake(mistakes, f"the next call raised {type(exc).__name__} ({exc})")
        router.next_call(router.bodies[len(first) :], mistakes)
    finally:
        await client.aclose()
    return raised, True


def _setup(router: _Router, body: Mapping[str, object], mistakes: list[str]) -> bool:
    """Whether the client's first request asks for everything the scripted routers refuse; each lack named."""
    before = len(mistakes)
    if not _temperature(body):
        router.mistake(
            mistakes,
            "the first request carried no temperature: build the client asking for one, with a fresh LearnedSettings",
        )
    if (cap := _cap(body)) is None or cap < 2:
        router.mistake(mistakes, "the first request carried no output cap: build the client asking for max_tokens")
    if not isinstance(_effort(body), str):
        router.mistake(
            mistakes,
            "the first request carried no reasoning effort word: build the client asking for one "
            "(wire_settings_for(JUDGE_REQUEST_SETTINGS, ...) asks for all three)",
        )
    if not _enforced(body):
        router.mistake(
            mistakes,
            "the first request was not enforced: a strict reply schema is sent under provider.require_parameters",
        )
    return len(mistakes) == before


def _check_report(router: _Router, completion: object, served: Mapping[str, object], mistakes: list[str]) -> None:
    """The completion reports what the served request carried: its temperature, cap, effort word and enforcement."""
    temperature = getattr(completion, "temperature", None)
    if temperature is not None and not _temperature(served):
        router.mistake(
            mistakes,
            f"the completion reports temperature {temperature}, and the served request carried none: report what "
            "was served (Negotiated.temperature)",
        )
    elif temperature is None and _temperature(served):
        router.mistake(
            mistakes,
            f"the completion reports no temperature, and the served request carried {served.get('temperature')}: "
            "report what was served (Negotiated.temperature)",
        )
    sent = sent_settings_of(completion)
    if sent is None:
        router.mistake(mistakes, "the completion reports no sent_settings: report Negotiated.sent_settings")
        return
    if sent.strict_output != _enforced(served):
        router.mistake(
            mistakes,
            f"the completion reports strict_output={sent.strict_output}, and the served request was "
            f"{'enforced' if _enforced(served) else 'not enforced'}: report what was served (Negotiated.sent_settings)",
        )
    if (cap := _cap(served)) != sent.max_tokens:
        router.mistake(
            mistakes, f"the completion reports max_tokens={sent.max_tokens}, and the served request carried {cap}"
        )
    if (effort := _effort(served)) != sent.reasoning_effort:
        router.mistake(
            mistakes,
            f"the completion reports reasoning_effort={sent.reasoning_effort!r}, and the served request carried "
            f"{effort!r}",
        )
