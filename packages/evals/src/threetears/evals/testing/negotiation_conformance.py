"""The refusal negotiation check: whether a host's own client answers a router's refusals in order, and says so.

A client that never fails a request because a model lacks a setting (:mod:`threetears.evals.negotiation`) has three
things to get right that no completion conformance check can see: it resends when the router refuses, in the fixed
order (the temperature first, keeping enforcement, and only then enforcement); it reports what the SERVED request
carried, not what it was asked; and it remembers, so the next call goes out as served at once. This check scripts
the router and watches. Under pytest::

    from threetears.evals.testing import check_refusal_negotiation

    async def test_my_client_negotiates() -> None:
        await check_refusal_negotiation(lambda transport: MyClient(..., post=transport))

**What the host wires.** ``build_client`` gets a :data:`RouterTransport` and returns the host's client built to send
each request body through it instead of the network, asking for a temperature and an output cap, and with a store of
learned settings of its own (a fresh :class:`~threetears.evals.negotiation.LearnedSettings`), so nothing learned
elsewhere decides the first request. The transport takes the JSON body the client would POST to a router's chat
completions endpoint and returns the status and JSON body the router answers with.

**What is scripted.** Any request carrying a ``temperature`` is refused with the router's 404 ("No endpoints found
that can handle the requested parameters."), as is any request under ``provider.require_parameters``; any other is
served a completion whose content satisfies the check's strict reply schema. So a correct client sends three
requests for the first call (temperature and enforcement, then enforcement alone, then neither), reports
``temperature=None`` and ``sent_settings.strict_output=False``, and sends one request for the second call, already
without either.

**What it cannot see.** Any refusal shape but the router's, and any order but this one: a client that answers
these two refusals correctly may still mishandle another. It reads a request's settings off the router's body
(``temperature``, ``max_tokens``, ``reasoning.effort``, ``provider.require_parameters``), so a client that speaks
another shape cannot be checked with it.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Final

from threetears.evals.schema import CompletionClient, sent_settings_of

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


class NegotiationConformanceFailure(AssertionError):
    """A host's client mishandled the router's refusals: each mistake it made, named.

    An :class:`AssertionError`, so a test runner reports it as a failed assertion. The message lists every mistake,
    one per line, and :attr:`mistakes` holds them.

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


class _ScriptedRouter:
    """The router the check scripts: refuses a temperature and enforcement, serves anything else, keeps every body."""

    def __init__(self) -> None:
        self.bodies: list[Mapping[str, object]] = []

    async def __call__(self, body: Mapping[str, object]) -> tuple[int, Mapping[str, object]]:
        self.bodies.append(body)
        if _temperature(body) or _enforced(body):
            return 404, _NO_ENDPOINT
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


async def check_refusal_negotiation(build_client: Callable[[RouterTransport], CompletionClient]) -> None:
    """Fail, naming each mistake, unless the client answers the router's refusals in order and reports what was served.

    Two calls, each with a strict reply schema. The first is refused for its temperature, then for its enforcement;
    the second must go out as the first was served, at once.

    Args:
        build_client: Builds the host's client over the scripted router (see the module docstring for how).

    Raises:
        NegotiationConformanceFailure: Naming every mistake: the setup's (a first request without a temperature or
            without enforcement), the resending's (none, out of order, too many), the reporting's (a temperature or
            enforcement reported that the served request did not carry, no ``sent_settings``, a cap or effort that
            differs from the served body's), and the remembering's (a second call that is refused again).
    """
    router = _ScriptedRouter()
    client = build_client(router)
    mistakes: list[str] = []
    raised: Exception | None = None
    try:
        try:
            completion = await client.generate(system="You answer.", user="Answer.", response_format=_REPLY_FORMAT)
        except Exception as exc:  # the check's finding: raised below, chained, as the mistake it is
            completion, raised = None, exc
            mistakes.append(
                f"the call raised {type(exc).__name__} ({exc}) after {len(router.bodies)} request(s): a refusal the "
                "client should have answered by resending (hand its errors to a refusal reader through negotiate)"
            )
        first = list(router.bodies)
        _check_first_call(first, mistakes)
        if completion is not None and first:
            _check_report(completion, first[-1], mistakes)
        if completion is not None:
            await client.generate(system="You answer.", user="Answer.", response_format=_REPLY_FORMAT)
            later = router.bodies[len(first) :]
            if len(later) != 1 or _temperature(later[0]) or _enforced(later[0]):
                mistakes.append(
                    f"the next call sent {len(later)} request(s), the first "
                    f"{'with' if later and _temperature(later[0]) else 'without'} a temperature and "
                    f"{'enforced' if later and _enforced(later[0]) else 'unenforced'}: what was learned is not kept "
                    "(keep one LearnedSettings for the client's life, or the factory's)"
                )
    finally:
        await client.aclose()
    if mistakes:
        raise NegotiationConformanceFailure(mistakes) from raised


def _check_first_call(bodies: list[Mapping[str, object]], mistakes: list[str]) -> None:
    """The first call's requests: asked with both, then without the temperature, then without enforcement."""
    if not bodies:
        mistakes.append("the client sent nothing through the transport: build it to POST each body through it")
        return
    if not _temperature(bodies[0]):
        mistakes.append(
            "the first request carried no temperature: build the client asking for one, with a fresh LearnedSettings"
        )
    if not _enforced(bodies[0]):
        mistakes.append(
            "the first request was not enforced: a strict reply schema is sent under provider.require_parameters"
        )
    for number, body in enumerate(bodies[1:], start=2):
        if _temperature(body):
            mistakes.append(f"request {number} still carried the temperature the router refused")
    if len(bodies) >= 2 and _temperature(bodies[0]) and not _enforced(bodies[1]):
        mistakes.append(
            "request 2 dropped enforcement before the temperature: the temperature goes first, keeping enforcement"
        )
    if len(bodies) > 3:
        mistakes.append(f"the call sent {len(bodies)} requests, more than the three the order needs")


def _check_report(completion: object, served: Mapping[str, object], mistakes: list[str]) -> None:
    """The completion reports what the served request carried."""
    temperature = getattr(completion, "temperature", None)
    if temperature is not None and not _temperature(served):
        mistakes.append(
            f"the completion reports temperature {temperature}, and the served request carried none: report what "
            "was served (Negotiated.temperature)"
        )
    sent = sent_settings_of(completion)
    if sent is None:
        mistakes.append("the completion reports no sent_settings: report Negotiated.sent_settings")
        return
    if sent.strict_output and not _enforced(served):
        mistakes.append(
            "the completion reports strict_output=True, and the served request was not enforced: report what was "
            "served (Negotiated.sent_settings)"
        )
    cap = served.get("max_tokens", served.get("max_completion_tokens"))
    if cap != sent.max_tokens:
        mistakes.append(f"the completion reports max_tokens={sent.max_tokens}, and the served request carried {cap}")
    effort = _mapping(served.get("reasoning")).get("effort")
    if effort != sent.reasoning_effort:
        mistakes.append(
            f"the completion reports reasoning_effort={sent.reasoning_effort!r}, and the served request carried "
            f"{effort!r}"
        )
