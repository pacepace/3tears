"""the current models are registered on the package's import path, priced, and sent no temperature.

A host prices each call from ``get_capabilities(<id>)`` (``cost_per_input_token`` / ``cost_per_output_token``);
an id with no price records no cost at all. The prices are OpenRouter's catalogue of 2026-11-20, per 1M tokens;
every one of these models refused a temperature on the live API that day.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

import threetears.models  # noqa: F401 - the import path a host takes, which registers the providers' models
from threetears.models.capabilities import get_capabilities

#: id -> (USD per 1M input tokens, USD per 1M output tokens)
_PRICES: dict[str, tuple[str, str]] = {
    "gpt-6.2-sol": ("2", "10"),
    "gpt-6.2-astra": ("8", "40"),
    "gpt-6.1-terra": ("1", "5"),
    "gpt-6-luna": ("0.10", "0.50"),
    "claude-sonnet-5-5": ("2", "10"),
    "claude-haiku-5-5": ("0.10", "0.50"),
    "claude-opus-5-5": ("4", "20"),
}


@pytest.mark.parametrize("model_id", sorted(_PRICES))
def test_each_current_model_resolves_with_its_prices_and_no_temperature(model_id: str) -> None:
    caps = get_capabilities(model_id)
    assert caps is not None, f"{model_id} is not registered on threetears.models' import path"
    per_input, per_output = _PRICES[model_id]
    assert caps.cost_per_input_token == Decimal(per_input) / Decimal(1_000_000)
    assert caps.cost_per_output_token == Decimal(per_output) / Decimal(1_000_000)
    assert caps.accepts_temperature is False
