"""tests for the one cost function and the public token estimate."""

from __future__ import annotations

from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from threetears.models.capabilities import ModelCapabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.pricing import TokenRates, UnpricedModelError, cost_of_usage
from threetears.models.tracking import (
    UsageRecord,
    UsageTracker,
    estimate_input_tokens,
)

_IN = Decimal("0.000003")
_OUT = Decimal("0.000015")
_READ = Decimal("0.0000003")
_WRITE = Decimal("0.00000375")


def _capabilities(**costs: Decimal | None) -> ModelCapabilities:
    """a chat model's capabilities carrying ``costs``."""
    return ModelCapabilities(model_name="m", model_type=ModelType.CHAT, model_tier=ModelTier.LARGE, **costs)


class TestCostOfUsage:
    """each price component, the fallback, and no double counting."""

    def test_input_and_output_are_priced_at_their_rates(self) -> None:
        caps = _capabilities(cost_per_input_token=_IN, cost_per_output_token=_OUT)
        assert cost_of_usage(caps, input_tokens=10, output_tokens=20) == _IN * 10 + _OUT * 20

    def test_cache_reads_and_writes_are_priced_at_their_own_rates(self) -> None:
        caps = _capabilities(
            cost_per_input_token=_IN,
            cost_per_output_token=_OUT,
            cost_per_cache_read_token=_READ,
            cost_per_cache_write_token=_WRITE,
        )
        cost = cost_of_usage(caps, input_tokens=1000, output_tokens=0, cache_read_tokens=600, cache_write_tokens=100)
        assert cost == _IN * 300 + _READ * 600 + _WRITE * 100

    def test_unset_cache_rates_fall_back_to_the_input_rate(self) -> None:
        caps = _capabilities(cost_per_input_token=_IN, cost_per_output_token=_OUT)
        cost = cost_of_usage(caps, input_tokens=1000, output_tokens=5, cache_read_tokens=600, cache_write_tokens=100)
        assert cost == _IN * 1000 + _OUT * 5

    def test_cost_per_request_is_added_once(self) -> None:
        caps = _capabilities(cost_per_input_token=_IN, cost_per_output_token=_OUT, cost_per_request=Decimal("0.01"))
        assert cost_of_usage(caps, input_tokens=0, output_tokens=0) == Decimal("0.01")
        assert cost_of_usage(caps, input_tokens=10, output_tokens=0) == Decimal("0.01") + _IN * 10

    def test_cache_tokens_are_not_counted_twice(self) -> None:
        """input_tokens already contains the cache tokens: all cached means no uncached input."""
        caps = _capabilities(cost_per_input_token=_IN, cost_per_output_token=_OUT, cost_per_cache_read_token=_READ)
        assert cost_of_usage(caps, input_tokens=500, output_tokens=0, cache_read_tokens=500) == _READ * 500

    def test_cache_tokens_beyond_the_input_never_make_the_uncached_part_negative(self) -> None:
        caps = _capabilities(cost_per_input_token=_IN, cost_per_output_token=_OUT)
        assert cost_of_usage(caps, input_tokens=5, output_tokens=0, cache_read_tokens=9) == _IN * 9

    def test_a_float_price_is_converted_through_its_string(self) -> None:
        rates = TokenRates(cost_per_input_token=0.1, cost_per_output_token=0.2)  # type: ignore[arg-type]
        assert cost_of_usage(rates, input_tokens=1, output_tokens=1) == Decimal("0.1") + Decimal("0.2")

    @pytest.mark.parametrize("costs", [{}, {"cost_per_input_token": _IN}, {"cost_per_output_token": _OUT}])
    def test_a_model_without_an_input_or_output_price_is_refused(self, costs: dict[str, Decimal]) -> None:
        with pytest.raises(UnpricedModelError):
            cost_of_usage(_capabilities(**costs), input_tokens=1, output_tokens=1)


class _Capture(UsageTracker):
    """a tracker that keeps what it records."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[UsageRecord] = []

    def record(self, usage: UsageRecord) -> None:
        """keeps the record."""
        self.captured.append(usage)


def _cache_heavy_result() -> LLMResult:
    """an Anthropic-shaped result: 1000 prompt tokens, 800 read from cache, 100 written."""
    message = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 50,
            "total_tokens": 1050,
            "input_token_details": {"cache_read": 800, "cache_creation": 100},
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class TestTrackerUsesTheCostFunction:
    """the tracker's ``cost_usd`` is ``cost_of_usage`` of the extracted usage."""

    def test_cost_usd_prices_cache_at_its_own_rates(self) -> None:
        tracker = _Capture()
        callback = tracker.make_callback(
            model_name="m",
            provider_name="anthropic",
            cost_per_input_token=_IN,
            cost_per_output_token=_OUT,
            cost_per_cache_read_token=_READ,
            cost_per_cache_write_token=_WRITE,
            cost_per_request=Decimal("0.001"),
        )
        callback.on_llm_end(_cache_heavy_result(), run_id=None)
        record = tracker.captured[0]
        assert record.cache_read_tokens == 800
        assert record.cache_creation_tokens == 100
        assert record.cost_usd == _IN * 100 + _READ * 800 + _WRITE * 100 + _OUT * 50 + Decimal("0.001")

    def test_without_cache_rates_the_cost_is_the_input_rate_on_the_whole_prompt(self) -> None:
        tracker = _Capture()
        callback = tracker.make_callback(
            model_name="m", provider_name="anthropic", cost_per_input_token=_IN, cost_per_output_token=_OUT
        )
        callback.on_llm_end(_cache_heavy_result(), run_id=None)
        assert tracker.captured[0].cost_usd == _IN * 1000 + _OUT * 50

    def test_a_model_without_prices_records_no_cost(self) -> None:
        tracker = _Capture()
        callback = tracker.make_callback(model_name="m", provider_name="anthropic")
        callback.on_llm_end(_cache_heavy_result(), run_id=None)
        assert tracker.captured[0].cost_usd is None


class TestEstimate:
    """the estimate keeps the old private function's behaviour."""

    @pytest.mark.parametrize(("text", "tokens"), [("", 0), ("a", 1), ("abcd", 1), ("abcdefgh", 2), ("x" * 40, 10)])
    def test_text_is_about_four_characters_a_token(self, text: str, tokens: int) -> None:
        assert estimate_input_tokens([HumanMessage(content=text)]) == tokens

    def test_a_prompt_is_the_sum_of_its_messages(self) -> None:
        messages = [HumanMessage(content="x" * 40), AIMessage(content=[{"type": "text", "text": "y" * 8}])]
        assert estimate_input_tokens(messages) == 10 + 2

    def test_no_messages_is_zero(self) -> None:
        assert estimate_input_tokens([]) == 0
