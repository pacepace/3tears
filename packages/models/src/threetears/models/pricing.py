"""the one place a call's usage is priced.

:func:`cost_of_usage` turns the token counts of one model call into USD. The usage tracker's
``cost_usd`` and the key pool's settlement both call it, so the two never disagree on what a call cost.

How the providers report input tokens, as LangChain surfaces them (``usage_metadata``):

- ``input_tokens`` is the WHOLE prompt, cache tokens included. ``langchain-anthropic`` sums Anthropic's
  uncached ``input_tokens``, ``cache_read_input_tokens`` and ``cache_creation_input_tokens`` into it;
  OpenAI's ``prompt_tokens`` already contains ``prompt_tokens_details.cached_tokens``; the Claude CLI
  provider sums the three the same way.
- ``input_token_details`` then names the cached parts: ``cache_read`` and ``cache_creation`` (a write).
  They are subsets of ``input_tokens``, not additions to it.

So the function takes ``input_tokens`` as that inclusive total, prices only the uncached remainder at
the input rate, and prices the cache parts at their own rates. Charging ``input_tokens`` at the input
rate and the cache tokens on top would count them twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

__all__ = ["TokenRates", "UnpricedModelError", "cost_of_usage"]


class UnpricedModelError(ValueError):
    """a call cannot be priced because the model has no input or no output price."""


class _Rates(Protocol):
    """anything that carries a model's per-token prices; ``ModelCapabilities`` does.

    The prices are USD per token.
    """

    @property
    def cost_per_input_token(self) -> Decimal | None:
        """USD per uncached input token."""
        ...

    @property
    def cost_per_output_token(self) -> Decimal | None:
        """USD per output token."""
        ...

    @property
    def cost_per_cache_read_token(self) -> Decimal | None:
        """USD per cache-read token; ``None`` prices it at the input rate."""
        ...

    @property
    def cost_per_cache_write_token(self) -> Decimal | None:
        """USD per cache-write token; ``None`` prices it at the input rate."""
        ...

    @property
    def cost_per_request(self) -> Decimal | None:
        """USD per call, flat; ``None`` is none."""
        ...


@dataclass(frozen=True, slots=True)
class TokenRates:
    """a model's prices, for a caller that holds the prices and not a full ``ModelCapabilities``.

    :param cost_per_input_token: USD per uncached input token
    :param cost_per_output_token: USD per output token
    :param cost_per_cache_read_token: USD per cache-read token; ``None`` prices it at the input rate
    :param cost_per_cache_write_token: USD per cache-write token; ``None`` prices it at the input rate
    :param cost_per_request: USD per call, flat; ``None`` is none
    """

    cost_per_input_token: Decimal | None = None
    cost_per_output_token: Decimal | None = None
    cost_per_cache_read_token: Decimal | None = None
    cost_per_cache_write_token: Decimal | None = None
    cost_per_request: Decimal | None = None


def _price(value: Decimal | float | None, fallback: Decimal) -> Decimal:
    """one price as a ``Decimal``, or ``fallback`` when unset.

    :param value: the price, if any
    :ptype value: Decimal | float | None
    :param fallback: what an unset price is
    :ptype fallback: Decimal
    :return: the price
    :rtype: Decimal
    """
    if value is None:
        return fallback
    return value if isinstance(value, Decimal) else Decimal(str(value))


def cost_of_usage(
    rates: _Rates,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal:
    """the USD cost of one call's usage.

    ``input_tokens`` is the whole prompt as the provider reports it, cache tokens included (see the
    module docstring). The uncached remainder, ``input_tokens - cache_read_tokens - cache_write_tokens``
    (never below zero), is priced at the input rate; cache reads and writes at their own rates, each
    falling back to the input rate when unset; output at the output rate; and ``cost_per_request`` once.

    :param rates: the model's prices (``ModelCapabilities``, or :class:`TokenRates`)
    :ptype rates: _Rates
    :param input_tokens: the whole prompt, cache tokens included
    :ptype input_tokens: int
    :param output_tokens: tokens generated
    :ptype output_tokens: int
    :param cache_read_tokens: the part of ``input_tokens`` read from the prompt cache
    :ptype cache_read_tokens: int
    :param cache_write_tokens: the part of ``input_tokens`` written to the prompt cache
    :ptype cache_write_tokens: int
    :return: the cost in USD
    :rtype: Decimal
    :raises UnpricedModelError: when the model has no input price or no output price
    """
    input_rate = rates.cost_per_input_token
    output_rate = rates.cost_per_output_token
    if input_rate is None or output_rate is None:
        raise UnpricedModelError("a call cannot be priced without both an input and an output price")
    input_price = _price(input_rate, Decimal(0))
    read_price = _price(rates.cost_per_cache_read_token, input_price)
    write_price = _price(rates.cost_per_cache_write_token, input_price)
    uncached = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
    return (
        input_price * uncached
        + read_price * cache_read_tokens
        + write_price * cache_write_tokens
        + _price(output_rate, Decimal(0)) * output_tokens
        + _price(rates.cost_per_request, Decimal(0))
    )
