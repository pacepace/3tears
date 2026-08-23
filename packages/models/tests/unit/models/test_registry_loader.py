"""tests for ``register_model_capabilities_bulk``.

the loader maps a consuming product's own model rows into the shared
capabilities registry, which is what :func:`compute_cost_usd`-style callers
read to price a call. a row whose cost is legitimately ZERO must survive that
mapping -- see :class:`TestAZeroCostSurvivesTheMapping`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from threetears.models.capabilities import clear_capability_overrides, get_capabilities
from threetears.models.enums import ModelType
from threetears.models.registry_loader import register_model_capabilities_bulk


@pytest.fixture(autouse=True)
def _isolate_registry() -> object:
    """drop registrations between tests so global registry state does not leak.

    :return: nothing
    :rtype: object
    """
    clear_capability_overrides()
    yield
    clear_capability_overrides()


class TestAZeroCostSurvivesTheMapping:
    """a zero cost is a VALUE, not an absent one.

    The loader coalesced the canonical and legacy column names with ``or``,
    which treats ``Decimal("0")`` as absent and falls through to the legacy key
    that is not there -- yielding ``None``. Every EMBEDDING model hit this,
    because an embedding has no output tokens and so a genuinely zero output
    cost: the registry ended up with ``cost_per_output_token=None``, and the
    cost calculator's "capability row missing cost rates" branch returned
    ``None`` for the whole call. The effect was that embedding usage recorded no
    USD cost at all while the admin-entered price sat in the product's own
    table, unread.
    """

    def test_zero_output_cost_is_kept(self) -> None:
        """an embedding's zero output cost registers as zero, not None.

        :return: nothing
        :rtype: None
        """
        count = register_model_capabilities_bulk(
            [
                {
                    "model_name": "test-embedding-zero-out",
                    "provider_name": "voyageai",
                    "model_type": ModelType.EMBEDDING,
                    "cost_per_1m_input_tokens": Decimal("0.18"),
                    "cost_per_1m_output_tokens": Decimal("0"),
                }
            ]
        )

        assert count == 1
        caps = get_capabilities("test-embedding-zero-out")
        assert caps is not None
        assert caps.cost_per_output_token == Decimal("0")
        assert caps.cost_per_input_token is not None

    def test_zero_input_cost_is_kept(self) -> None:
        """the same coalescing bug on the input side -- a free model prices at zero.

        :return: nothing
        :rtype: None
        """
        register_model_capabilities_bulk(
            [
                {
                    "model_name": "test-free-model",
                    "cost_per_1m_input_tokens": Decimal("0"),
                    "cost_per_1m_output_tokens": Decimal("0"),
                }
            ]
        )

        caps = get_capabilities("test-free-model")
        assert caps is not None
        assert caps.cost_per_input_token == Decimal("0")
        assert caps.cost_per_output_token == Decimal("0")

    def test_a_zero_context_window_is_kept(self) -> None:
        """same idiom, same failure, different column.

        :return: nothing
        :rtype: None
        """
        register_model_capabilities_bulk([{"model_name": "test-zero-context", "context_window": 0}])

        caps = get_capabilities("test-zero-context")
        assert caps is not None
        assert caps.context_window == 0

    def test_an_absent_cost_is_still_none(self) -> None:
        """the fix must not turn "not supplied" into zero -- that would invent a price.

        :return: nothing
        :rtype: None
        """
        register_model_capabilities_bulk([{"model_name": "test-no-cost"}])

        caps = get_capabilities("test-no-cost")
        assert caps is not None
        assert caps.cost_per_input_token is None
        assert caps.cost_per_output_token is None


class TestLegacyColumnNamesStillResolve:
    """the loader accepts both the canonical and the legacy spellings."""

    def test_legacy_names_are_read(self) -> None:
        """a product on the older column names keeps working.

        :return: nothing
        :rtype: None
        """
        register_model_capabilities_bulk(
            [
                {
                    "model_name": "test-legacy-names",
                    "cost_per_1m_prompt_tokens": Decimal("3"),
                    "cost_per_1m_completion_tokens": Decimal("15"),
                    "context_window_tokens": 200000,
                }
            ]
        )

        caps = get_capabilities("test-legacy-names")
        assert caps is not None
        assert caps.cost_per_input_token is not None
        assert caps.cost_per_output_token is not None
        assert caps.context_window == 200000

    def test_the_canonical_name_wins_over_the_legacy_one(self) -> None:
        """a row carrying both spellings resolves to the canonical value.

        :return: nothing
        :rtype: None
        """
        register_model_capabilities_bulk(
            [
                {
                    "model_name": "test-both-names",
                    "cost_per_1m_input_tokens": Decimal("1"),
                    "cost_per_1m_prompt_tokens": Decimal("999"),
                }
            ]
        )

        caps = get_capabilities("test-both-names")
        assert caps is not None
        # 1 per MTok -> 0.000001 per token; the 999 legacy value must not win
        assert caps.cost_per_input_token == Decimal("1") / Decimal(1_000_000)
