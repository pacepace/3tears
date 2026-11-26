"""Shared builders for the pooled chat model's unit tests: a priced test model, and one Anthropic Messages
API that answers each key's requests from its own script.

The pooled model is built through ``create_chat_model(key_pool=...)``, the one front door, and every
key's model is the real provider model pointed at a served wire through the public ``base_url``, so a
refusal reaches the pool as the exception the SDK and LangChain really raise for it, from the recorded
responses in :mod:`key_error_fixtures`. No network: the wire is on loopback.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx
from langchain_core.callbacks import BaseCallbackHandler

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.factory import create_chat_model
from threetears.models.key_pool import KeyPool, KeyStatus
from threetears.models.tracking import UsageAuditSink, UsageRecord, UsageTracker

from . import key_error_fixtures as fx
from .key_pool_support import secret_for
from .provider_wire import serve_http_handler

#: A model only these tests register: priced, with a bounded output, served by anthropic.
MODEL_ID = "pool-test-claude"

CAPABILITIES = ModelCapabilities(
    model_name=MODEL_ID,
    model_type=ModelType.CHAT,
    model_tier=ModelTier.SMALL,
    provider_name="anthropic",
    max_output_tokens=1000,
    cost_per_input_token=Decimal("0.000003"),
    cost_per_output_token=Decimal("0.000015"),
    cost_per_cache_read_token=Decimal("0.0000003"),
    cost_per_cache_write_token=Decimal("0.00000375"),
)


# registered once, on import: the id is these tests' own, so no other test sees it
register_capabilities(MODEL_ID, CAPABILITIES)


#: An answer the wire gives one request: the request body in, the response out.
Answer = Callable[[dict[str, Any]], httpx.Response]


def _message(content: list[dict[str, Any]], stop_reason: str | None, usage: dict[str, int]) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL_ID,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def _events(events: list[dict[str, Any]]) -> httpx.Response:
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def reply(text: str = "ok", *, input_tokens: int = 3, output_tokens: int = 2, **cache: int) -> Answer:
    """A text answer, as JSON or as a stream, whichever the request asked for.

    :param cache: ``cache_read_input_tokens`` / ``cache_creation_input_tokens``, as the API reports them
    """
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, **cache}

    def answer(body: dict[str, Any]) -> httpx.Response:
        if not body.get("stream"):
            return httpx.Response(200, json=_message([{"type": "text", "text": text}], "end_turn", usage))
        return _events(
            [
                {"type": "message_start", "message": _message([], None, {**usage, "output_tokens": 0})},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    # cumulative, as the API reports it on message_delta, input and cache included
                    "usage": usage,
                },
                {"type": "message_stop"},
            ]
        )

    return answer


def tool_call(name: str, arguments: dict[str, Any]) -> Answer:
    """A non-streamed answer calling the tool ``name``."""

    def answer(body: dict[str, Any]) -> httpx.Response:
        del body
        block = {"type": "tool_use", "id": "toolu_1", "name": name, "input": arguments}
        return httpx.Response(200, json=_message([block], "tool_use", {"input_tokens": 5, "output_tokens": 4}))

    return answer


def refusal(recorded: fx.RecordedError) -> Answer:
    """A provider error response, as recorded."""

    def answer(body: dict[str, Any]) -> httpx.Response:
        del body
        return recorded.response()

    return answer


def partial_then(error_type: str, message: str) -> Answer:
    """A stream that hands over one text chunk, then fails with an ``error`` event of ``error_type``."""

    def answer(body: dict[str, Any]) -> httpx.Response:
        del body
        return _events(
            [
                {"type": "message_start", "message": _message([], None, {"input_tokens": 3, "output_tokens": 0})},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "par"}},
                {"type": "error", "error": {"type": error_type, "message": message}},
            ]
        )

    return answer


@dataclass(frozen=True)
class Exchange:
    """One request a key sent, and its body."""

    label: str
    body: dict[str, Any]


class KeyedWire:
    """The Anthropic Messages API, answering each key's requests from that key's script, in order; a
    key whose script has run out gets :func:`reply`. Every request is kept, by the key that sent it."""

    def __init__(self, *labels: str) -> None:
        self._labels = {secret_for(label): label for label in labels}
        self._scripts: dict[str, list[Answer]] = {label: [] for label in labels}
        self._lock = threading.Lock()
        self.exchanges: list[Exchange] = []

    def script(self, label: str, *answers: Answer) -> None:
        with self._lock:
            self._scripts[label].extend(answers)

    def labels_sent(self) -> list[str]:
        with self._lock:
            return [exchange.label for exchange in self.exchanges]

    def handle(self, request: httpx.Request) -> httpx.Response:
        label = self._labels[request.headers["x-api-key"]]
        body = json.loads(request.content)
        with self._lock:
            self.exchanges.append(Exchange(label, body))
            script = self._scripts[label]
            answer = script.pop(0) if script else reply()
        return answer(body)

    @contextmanager
    def serve(self) -> Iterator[str]:
        with serve_http_handler(self.handle) as base_url:
            yield base_url


class RecordingSink(UsageAuditSink):
    """Keeps every usage record the tracker hands over."""

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    async def record(self, record: UsageRecord) -> None:
        self.records.append(record)


class EventLog(BaseCallbackHandler):
    """A caller's callback: each model run's start, end and error, in order."""

    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        self.events.append("start")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ANN401 -- LangChain's LLMResult
        self.events.append("end")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.events.append(f"error:{type(error).__name__}")


def pooled_model(pool: KeyPool, base_url: str, *, tracker: UsageTracker | None = None, **settings: Any) -> Any:  # noqa: ANN401
    """The pooled model ``create_chat_model`` builds for ``pool``, every key's model on the wire."""
    return create_chat_model(
        MODEL_ID,
        key_pool=pool,
        tracker=tracker if tracker is not None else UsageTracker(),
        base_url=base_url,
        max_retries=0,
        **settings,
    )


async def key_status(pool: KeyPool, label: str) -> KeyStatus:
    """The pool's status for one key."""
    status = await pool.status()
    return next(key for key in status.keys if key.label == label)
