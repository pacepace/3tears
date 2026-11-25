"""tests for ``register_model_capabilities_bulk``.

the loader maps a consuming product's own model rows into the shared
capabilities registry, which is what :func:`compute_cost_usd`-style callers
read to price a call. a row whose cost is legitimately ZERO must survive that
mapping -- see :class:`TestAZeroCostSurvivesTheMapping`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from decimal import Decimal

import pytest
from pydantic import ValidationError

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


class TestRequestSettingsKeys:
    """the loader carries the request-settings facts from a product row."""

    def test_every_request_settings_key_is_read(self) -> None:
        """each key reaches the registry entry unchanged.

        :return: nothing
        :rtype: None
        """
        register_model_capabilities_bulk(
            [
                {
                    "model_name": "test-request-settings",
                    "accepts_temperature": False,
                    "reasoning_efforts": ["low", "medium", "high"],
                    "reasoning_budget": False,
                    "adaptive_reasoning": True,
                    "reasoning_disable": True,
                    "structured_output": True,
                }
            ]
        )

        caps = get_capabilities("test-request-settings")
        assert caps is not None
        assert caps.accepts_temperature is False
        assert caps.reasoning_efforts == ["low", "medium", "high"]
        assert caps.reasoning_budget is False
        assert caps.adaptive_reasoning is True
        assert caps.reasoning_disable is True
        assert caps.structured_output is True

    def test_absent_and_null_keys_stay_not_declared(self) -> None:
        """a row without the keys, or with NULL columns, declares nothing.

        an empty effort list is a value, not an absence, and survives.

        :return: nothing
        :rtype: None
        """
        register_model_capabilities_bulk(
            [
                {"model_name": "test-request-settings-absent"},
                {"model_name": "test-request-settings-null", "accepts_temperature": None, "reasoning_efforts": []},
            ]
        )

        absent = get_capabilities("test-request-settings-absent")
        assert absent is not None
        assert absent.accepts_temperature is None
        assert absent.reasoning_efforts is None
        null = get_capabilities("test-request-settings-null")
        assert null is not None
        assert null.accepts_temperature is None
        assert null.reasoning_efforts == []

    def test_an_unreadable_effort_list_is_refused(self) -> None:
        """a bare string is not a list of words; the loader refuses rather than splitting it.

        :return: nothing
        :rtype: None
        """
        with pytest.raises(ValidationError):
            register_model_capabilities_bulk([{"model_name": "test-request-settings-bad", "reasoning_efforts": "low"}])


#: a fresh interpreter, so its 1,500 rows never reach this process's registry; counts the read-only snapshots
#: (each a copy of a registry layer) the registry module builds while the loader runs.
_LOADER_PROBE = """
import json

import threetears.models.capabilities as capabilities
from threetears.models.registry_loader import register_model_capabilities_bulk

built = capabilities.MappingProxyType
copies = []


def counting(mapping):
    copies.append(len(mapping))
    return built(mapping)


capabilities.MappingProxyType = counting
count = register_model_capabilities_bulk([{"model_name": f"loader-probe-{index}"} for index in range(1500)])
capabilities.MappingProxyType = built
print(json.dumps({"count": count, "copies": len(copies)}))
"""


class TestACatalogueIsOneRegistryCopy:
    """the loader registers its rows as one batch: a copy of the registry a row made a catalogue quadratic."""

    def test_1500_rows_copy_the_registry_once(self) -> None:
        """1,500 rows, one copy (it was 1,500, one a row).

        :return: nothing
        :rtype: None
        """
        result = subprocess.run([sys.executable, "-c", _LOADER_PROBE], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        probe = json.loads(result.stdout.strip().splitlines()[-1])
        assert probe == {"count": 1500, "copies": 1}

    def test_the_rows_before_a_bad_one_still_register(self) -> None:
        """a row that raises stops the load, as it did a row at a time; the rows before it are registered.

        :return: nothing
        :rtype: None
        """
        with pytest.raises(ValidationError):
            register_model_capabilities_bulk(
                [
                    {"model_name": "test-loader-before-bad"},
                    {"model_name": "test-loader-bad", "reasoning_efforts": "low"},
                ]
            )
        assert get_capabilities("test-loader-before-bad") is not None
