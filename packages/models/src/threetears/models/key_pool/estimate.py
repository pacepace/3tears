"""What a key pool reserves for one call before it runs (KP-E1).

The estimate is the call's estimated input tokens at the input rate, plus its ``max_tokens`` -- or the
model's ``max_output_tokens`` when the call sets none -- at the output rate, and the model's flat
``cost_per_request``. It is priced by :func:`~threetears.models.pricing.cost_of_usage`, the one place a
call's usage is priced, so a reservation and the settlement that replaces it agree on every rate.

The input estimate is the caller's: it counts the call's messages (``estimate_input_tokens``) and
passes the count. Nothing here reads a message or the network.
"""

from __future__ import annotations

from decimal import Decimal

from threetears.models.capabilities import ModelCapabilities
from threetears.models.key_pool.errors import NoOutputLimitError
from threetears.models.pricing import UnpricedModelError, cost_of_usage

__all__ = ["reservation_estimate"]


def reservation_estimate(
    capabilities: ModelCapabilities, *, input_tokens: int, max_tokens: int | None = None
) -> Decimal:
    """The money to reserve for one call to the model ``capabilities`` describes.

    :param capabilities: the model's capabilities, carrying its prices and ``max_output_tokens``
    :ptype capabilities: ModelCapabilities
    :param input_tokens: the call's estimated input tokens, 0 or more
    :ptype input_tokens: int
    :param max_tokens: the call's ``max_tokens``; ``None`` uses the model's ``max_output_tokens``
    :ptype max_tokens: int | None
    :return: the reservation, in USD
    :rtype: Decimal
    :raises UnpricedModelError: when the model has no input or no output price (KP-A4): a call the pool
        can't price is one it can't budget, and treating it as free would be a silent zero
    :raises NoOutputLimitError: when neither ``max_tokens`` nor the model's ``max_output_tokens`` is set
    :raises ValueError: when a token count is negative or not an int
    """
    model = capabilities.model_name
    if capabilities.cost_per_input_token is None or capabilities.cost_per_output_token is None:
        raise UnpricedModelError(
            f"model {model!r} has no {'input' if capabilities.cost_per_input_token is None else 'output'} price "
            "registered, so a key pool can't budget its calls; register its prices first"
        )
    output_tokens = max_tokens if max_tokens is not None else capabilities.max_output_tokens
    if output_tokens is None:
        raise NoOutputLimitError(model)
    for what, count in (("input_tokens", input_tokens), ("max_tokens", output_tokens)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{what} must be an int of 0 or more, got {count!r}")
    return cost_of_usage(capabilities, input_tokens=input_tokens, output_tokens=output_tokens)
