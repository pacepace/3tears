"""Mocked provider transports and canned replies for the models adapter's tests: no provider is ever called.

Every chat model the adapter builds talks HTTP through httpx. :func:`mocked_providers` replaces the transport of
every ``httpx.AsyncClient`` and ``httpx.Client`` constructed while a test runs with an ``httpx.MockTransport``
that serves the replies a test planned per host and records every request body, which is how the investigation
behind the adapter probed LangChain without a key. Three details keep that honest:

* **Anthropic and OpenAI share one httpx client per base URL**, cached process-wide by LangChain. A client cached
  by an earlier test would carry an earlier transport, so every model a test builds gets a base URL of its own
  (:meth:`ProviderTransport.base_url`), on a ``.invalid`` host that no resolver answers: a client that somehow
  escaped the mock could not reach a provider either.
* **OpenRouter's SDK builds its own httpx clients per model**, inside the patched constructor, so they are mocked
  and owned by the SDK, which is what the adapter's ``aclose`` closes.
* **The price catalogue is never fetched**: every test hands the adapter a :class:`FakeCatalogue`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from threetears.evals.providers import ModelsCompletionClient, models_client
from threetears.models import DEFAULT_CHAT_MODEL

__all__ = [
    "ANTHROPIC_HOST",
    "CATALOGUE",
    "OPENAI_HOST",
    "OPENROUTER_HOST",
    "HOSTS",
    "KEYS",
    "MODELS",
    "FakeCatalogue",
    "ProviderTransport",
    "anthropic_reply",
    "build",
    "mocked_providers",
    "ok",
    "openai_reply",
    "openrouter_stream",
    "plan_reply",
    "status",
]

ANTHROPIC_HOST = "anthropic"
OPENAI_HOST = "openai"
OPENROUTER_HOST = "openrouter.ai"

#: A reply the transport serves: a status, a JSON body or an event-stream text, and headers.
Planned = tuple[int, dict[str, Any] | str, dict[str, str]]

#: Response headers that make an SDK's retry wait (if it ever retried) next to nothing.
_FAST_RETRY = {"retry-after-ms": "1", "retry-after": "0"}


@dataclass
class ProviderTransport:
    """The mocked network: what each host answers, in order, and every request each host received.

    Attributes:
        plans: Host key -> the replies it serves in order; the last one repeats.
        requests: ``(host key, JSON body)`` for every request, in arrival order.
    """

    plans: dict[str, list[Planned]] = field(default_factory=dict)
    requests: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def plan(self, host: str, *replies: Planned) -> None:
        """Serve ``replies`` to ``host``, in order, the last repeating."""
        self.plans[host] = list(replies)

    def base_url(self, host: str) -> str:
        """A base URL of its own for one model on ``host``, so no cached client from another test answers it."""
        return f"https://{host}.{uuid.uuid4().hex}.invalid"

    def sent(self, host: str) -> list[dict[str, Any]]:
        """Every request body ``host`` received."""
        return [body for key, body in self.requests if key == host]

    def last(self, host: str) -> dict[str, Any]:
        """The last request body ``host`` received."""
        bodies = self.sent(host)
        assert bodies, f"{host} received no request"
        return bodies[-1]

    def _host_key(self, request: httpx.Request) -> str:
        host = request.url.host
        if host == OPENROUTER_HOST or host.endswith(f".{OPENROUTER_HOST}"):
            return OPENROUTER_HOST
        return host.split(".", 1)[0]

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Answer one request from its host's plan, recording its body."""
        key = self._host_key(request)
        body = json.loads(request.content) if request.content else {}
        self.requests.append((key, body))
        plan = self.plans.get(key)
        assert plan, f"no reply planned for {key} ({request.url})"
        status, payload, headers = plan.pop(0) if len(plan) > 1 else plan[0]
        if isinstance(payload, str):
            return httpx.Response(status, text=payload, headers={"content-type": "text/event-stream", **headers})
        return httpx.Response(status, json=payload, headers=headers)


@contextmanager
def _patched_httpx(monkeypatch: pytest.MonkeyPatch, transport: ProviderTransport) -> Iterator[None]:
    original_async: Callable[..., None] = httpx.AsyncClient.__init__
    original_sync: Callable[..., None] = httpx.Client.__init__

    def async_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(transport.handle)
        original_async(self, *args, **kwargs)

    def sync_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(transport.handle)
        original_sync(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", async_init)
    monkeypatch.setattr(httpx.Client, "__init__", sync_init)
    yield


def mocked_providers(monkeypatch: pytest.MonkeyPatch) -> Iterator[ProviderTransport]:
    """Every httpx client built while this generator is suspended answers from a :class:`ProviderTransport`.

    A test module's own fixture yields from it: ``yield from mocked_providers(monkeypatch)``.
    """
    transport = ProviderTransport()
    with _patched_httpx(monkeypatch, transport):
        yield transport


# --- canned replies --------------------------------------------------------------------------------------


def anthropic_reply(
    *,
    text: str = '{"score": 4}',
    model: str = f"{DEFAULT_CHAT_MODEL}-20260201",
    stop_reason: str | None = "end_turn",
    thinking_tokens: int | None = 5,
    thinking_block: bool = True,
) -> dict[str, Any]:
    """An Anthropic Messages reply: a thinking block (by default), one text block, usage with a thinking split."""
    content: list[dict[str, Any]] = []
    if thinking_block:
        content.append({"type": "thinking", "thinking": "weighing it", "signature": "sig"})
    content.append({"type": "text", "text": text})
    usage: dict[str, Any] = {
        "input_tokens": 11,
        "output_tokens": 22,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 0,
    }
    if thinking_tokens is not None:
        usage["output_tokens_details"] = {"thinking_tokens": thinking_tokens}
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def openai_reply(
    *,
    text: str | None = '{"score": 4}',
    model: str | None = "gpt-5-mini-2025-08-07",
    finish_reason: str | None = "stop",
    reasoning_tokens: int | None = 7,
    refusal: str | None = None,
    usage: bool = True,
) -> dict[str, Any]:
    """An OpenAI chat-completions reply."""
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if refusal is not None:
        message["refusal"] = refusal
    reply: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
    }
    if usage:
        details = {"reasoning_tokens": reasoning_tokens} if reasoning_tokens is not None else {}
        reply["usage"] = {
            "prompt_tokens": 11,
            "completion_tokens": 22,
            "total_tokens": 33,
            "completion_tokens_details": details,
        }
    return reply


def _sse(*chunks: dict[str, Any]) -> str:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def openrouter_stream(
    *,
    text: tuple[str, ...] = ('{"score":', " 4}"),
    model: str = "anthropic/claude-4.6-sonnet-20260201",
    upstream: str | None = "Anthropic",
    finish_reason: str | None = "stop",
    cost: float | None = 0.00123,
    upstream_cost: float | None = None,
    reasoning_tokens: int | None = 7,
    usage: bool = True,
) -> str:
    """An OpenRouter chat stream (models always takes the streamed path there): content chunks, then usage."""
    base: dict[str, Any] = {"id": "gen-1", "object": "chat.completion.chunk", "created": 1, "model": model}
    if upstream is not None:
        base["provider"] = upstream
    chunks: list[dict[str, Any]] = []
    for index, part in enumerate(text):
        last = index == len(text) - 1
        delta: dict[str, Any] = {"content": part}
        if index == 0:
            delta["role"] = "assistant"
        chunks.append(
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason if last else None}]}
        )
    if usage:
        stats: dict[str, Any] = {
            "prompt_tokens": 11,
            "completion_tokens": 22,
            "total_tokens": 33,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        }
        if cost is not None:
            stats["cost"] = cost
        if upstream_cost is not None:
            stats["cost_details"] = {
                "upstream_inference_cost": upstream_cost,
                "upstream_inference_prompt_cost": upstream_cost / 2,
                "upstream_inference_completions_cost": upstream_cost / 2,
            }
        chunks.append({**base, "choices": [], "usage": stats})
    return _sse(*chunks)


def status(code: int, payload: dict[str, Any] | str, *, fast: bool = True) -> Planned:
    """A planned error reply."""
    return (code, payload, dict(_FAST_RETRY) if fast else {})


def ok(payload: dict[str, Any] | str) -> Planned:
    """A planned 200."""
    return (200, payload, {})


# --- the price catalogue ---------------------------------------------------------------------------------

#: A slice of OpenRouter's public ``/models`` catalogue, per-token prices as strings, as it serves them.
CATALOGUE: dict[str, Any] = {
    "data": [
        {"id": f"anthropic/{DEFAULT_CHAT_MODEL}", "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
        {"id": "anthropic/claude-sonnet-4.6", "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
        {"id": "openai/gpt-5-mini", "pricing": {"prompt": "0.00000025", "completion": "0.000002"}},
        {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
        {"id": "anthropic/unpriced", "pricing": {"prompt": "0", "completion": "0"}},
    ]
}


# parity-with: threetears.models.price_lookup.HttpClient
class FakeCatalogue:
    """The price catalogue a test hands the adapter: serves :data:`CATALOGUE`, or fails as told, and counts fetches.

    Attributes:
        fetches: Every URL fetched, in order.
        failure: An exception to raise instead of answering, or ``None``.
        payload: What the catalogue serves.
    """

    def __init__(self, *, failure: BaseException | None = None, payload: dict[str, Any] | None = None) -> None:
        self.fetches: list[str] = []
        self.failure = failure
        self.payload = CATALOGUE if payload is None else payload

    async def get(self, url: str, /) -> httpx.Response:
        """Answer one catalogue read."""
        self.fetches.append(url)
        if self.failure is not None:
            raise self.failure
        return httpx.Response(200, json=self.payload, request=httpx.Request("GET", url))


# --- clients on the mocked network -----------------------------------------------------------------------

#: The model each provider's client is built for, unless a test names another: ids models' registry places on
#: their own for Anthropic and OpenAI, and an alias, which OpenRouter needs ``provider=`` for.
MODELS: dict[str, str] = {
    "anthropic": DEFAULT_CHAT_MODEL,
    "openai": "gpt-4o",
    "openrouter": "~anthropic/claude-sonnet-latest",
}

#: A key shaped as each provider's, never a real one: the transport answers before anything leaves the process.
KEYS: dict[str, str] = {"anthropic": "sk-ant-api03-test", "openai": "sk-test", "openrouter": "sk-or-v1-test"}

#: The host key each provider's requests arrive under.
HOSTS: dict[str, str] = {"anthropic": ANTHROPIC_HOST, "openai": OPENAI_HOST, "openrouter": OPENROUTER_HOST}


def build(
    transport: ProviderTransport,
    provider: str,
    *,
    model: str | None = None,
    prices: Any = None,
    **kwargs: Any,
) -> ModelsCompletionClient:
    """A client for ``provider`` on the mocked network, with a :class:`FakeCatalogue` unless ``prices`` is given."""
    if provider != "openrouter":
        kwargs.setdefault("base_url", transport.base_url(HOSTS[provider]))
    return models_client(
        model or MODELS[provider],
        api_key=kwargs.pop("api_key", KEYS[provider]),
        provider=kwargs.pop("provider", provider),
        prices=FakeCatalogue() if prices is None else prices,
        **kwargs,
    )


def plan_reply(transport: ProviderTransport, provider: str, **reply: Any) -> None:
    """Have ``provider`` answer every request with its canned reply, ``reply`` overriding its defaults."""
    if provider == "anthropic":
        transport.plan(ANTHROPIC_HOST, ok(anthropic_reply(**reply)))
    elif provider == "openai":
        transport.plan(OPENAI_HOST, ok(openai_reply(**reply)))
    else:
        transport.plan(OPENROUTER_HOST, ok(openrouter_stream(**reply)))
