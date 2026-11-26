"""A factory-built model's usage is priced at every rate its capabilities carry.

The tracker used to be handed only the input and output prices, so a prompt-cache read or write
was charged at the full input rate whatever the model's cache prices were. Driven at the transport:
the Anthropic API route against a scripted Messages API on loopback.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any

import httpx
from langchain_core.messages import HumanMessage

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.factory import create_chat_model
from threetears.models.tracking import UsageAuditSink, UsageRecord, UsageTracker

from .provider_wire import serve_http_handler

_MODEL = "test-factory-prices-every-rate"
_INPUT = Decimal("0.000003")
_OUTPUT = Decimal("0.000015")
_CACHE_READ = Decimal("0.0000003")
_CACHE_WRITE = Decimal("0.00000375")
_PER_REQUEST = Decimal("0.001")


class _Sink(UsageAuditSink):
    """keeps every usage record the tracker hands over."""

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    async def record(self, record: UsageRecord) -> None:
        self.records.append(record)


def _answer() -> dict[str, Any]:
    """a Messages API answer whose prompt was mostly read from, and partly written to, the cache.

    :return: the response body
    :rtype: dict[str, Any]
    """
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": _MODEL,
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        # Anthropic reports the three input parts separately; uncached is input_tokens alone
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 20,
            "output_tokens": 7,
        },
    }


async def test_cache_reads_writes_and_the_request_are_priced_at_their_own_rates() -> None:
    register_capabilities(
        _MODEL,
        ModelCapabilities(
            model_name=_MODEL,
            model_type=ModelType.CHAT,
            model_tier=ModelTier.SMALL,
            provider_name="anthropic",
            context_window=200_000,
            cost_per_input_token=_INPUT,
            cost_per_output_token=_OUTPUT,
            cost_per_cache_read_token=_CACHE_READ,
            cost_per_cache_write_token=_CACHE_WRITE,
            cost_per_request=_PER_REQUEST,
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        json.loads(request.content)
        return httpx.Response(200, json=_answer())

    sink = _Sink()
    with serve_http_handler(handler) as base_url:
        model = create_chat_model(
            _MODEL,
            api_key="sk-ant-api03-faketestkey",
            tracker=UsageTracker(audit_sink=sink),
            base_url=base_url,
            max_retries=0,
        )
        await model.ainvoke([HumanMessage(content="hi")])
    for _ in range(5):
        await asyncio.sleep(0)

    assert len(sink.records) == 1
    assert sink.records[0].cost_usd == (
        _INPUT * 10 + _CACHE_READ * 100 + _CACHE_WRITE * 20 + _OUTPUT * 7 + _PER_REQUEST
    ), "a factory-built model charged the cache at the input rate, or dropped the per-request price"
