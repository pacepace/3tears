"""Building a key pool: the redacted secret, the key spec, the refusals at build, and the reservation
estimate (KP-A1 to A5, KP-B1, KP-E1)."""

from __future__ import annotations

import copy
import pickle
from decimal import Decimal

import pytest

from threetears.models import friendly_api_error, is_provider_error
from threetears.models.capabilities import ModelCapabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.errors import ModelProviderError
from threetears.models.key_pool import (
    KeyPool,
    KeySpec,
    KeyState,
    KeyStatus,
    NoKeyAvailableError,
    NoOutputLimitError,
    Secret,
    reservation_estimate,
)
from threetears.models.pricing import UnpricedModelError

from .key_pool_support import PROVIDER, make_pool, spec

_SENTINEL = "sk-ant-api03-SENTINEL-do-not-show"


class TestSecret:
    def test_every_rendering_is_redacted(self) -> None:
        secret = Secret(_SENTINEL)
        for shown in (str(secret), repr(secret), f"{secret}", f"{secret!r}", f"{secret:>40}", format(secret, "s")):
            assert _SENTINEL not in shown
            assert "redacted" in shown

    def test_the_value_is_reached_only_by_reveal(self) -> None:
        assert Secret(_SENTINEL).reveal() == _SENTINEL

    def test_it_is_not_its_plain_string(self) -> None:
        assert Secret(_SENTINEL) != _SENTINEL
        assert _SENTINEL != Secret(_SENTINEL)

    def test_equal_values_are_equal_and_hash_alike(self) -> None:
        assert Secret(_SENTINEL) == Secret(_SENTINEL)
        assert hash(Secret(_SENTINEL)) == hash(Secret(_SENTINEL))
        assert Secret(_SENTINEL) != Secret(_SENTINEL + "x")

    def test_its_hash_is_not_the_strings(self) -> None:
        assert hash(Secret(_SENTINEL)) != hash(_SENTINEL)

    def test_it_refuses_pickling(self) -> None:
        with pytest.raises(TypeError):
            pickle.dumps(Secret(_SENTINEL))

    def test_it_has_no_dict_to_dump(self) -> None:
        assert not hasattr(Secret(_SENTINEL), "__dict__")

    def test_a_dataclass_holding_one_shows_no_value(self) -> None:
        key = KeySpec(Secret(_SENTINEL), "primary", Decimal(5))
        assert _SENTINEL not in repr(key)
        assert _SENTINEL not in repr(copy.deepcopy(key))

    @pytest.mark.parametrize("value", ["", "   "])
    def test_an_empty_secret_is_refused(self, value: str) -> None:
        with pytest.raises(ValueError):
            Secret(value)


class TestKeySpec:
    @pytest.mark.parametrize("label", ["", "has space", "slash/label", "-leading", "x" * 33, "ünicode"])
    def test_an_unsafe_label_is_refused(self, label: str) -> None:
        with pytest.raises(ValueError, match="label"):
            spec(label)

    @pytest.mark.parametrize("label", ["a", "primary", "key-2", "org_a.k1", "K" * 32])
    def test_a_safe_label_is_accepted(self, label: str) -> None:
        assert spec(label).label == label

    @pytest.mark.parametrize("budget", ["0", "-1", "NaN", "Infinity"])
    def test_a_budget_that_is_not_positive_and_finite_is_refused(self, budget: str) -> None:
        with pytest.raises(ValueError, match="budget"):
            spec("primary", budget)

    def test_a_float_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            KeySpec(Secret(_SENTINEL), "primary", 10.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("currency", ["EUR", "usd", "GBP"])
    def test_a_budget_not_in_usd_is_refused_not_converted(self, currency: str) -> None:
        with pytest.raises(ValueError, match="USD"):
            KeySpec(Secret(_SENTINEL), "primary", Decimal(10), currency=currency)

    def test_a_plain_string_secret_is_refused_without_echoing_it(self) -> None:
        with pytest.raises(TypeError) as caught:
            KeySpec(_SENTINEL, "primary", Decimal(10))  # type: ignore[arg-type]
        assert _SENTINEL not in str(caught.value)

    def test_a_bool_priority_is_refused(self) -> None:
        with pytest.raises(ValueError, match="priority"):
            KeySpec(Secret(_SENTINEL), "primary", Decimal(10), priority=True)

    def test_a_key_with_no_account_is_its_own(self) -> None:
        assert spec("a").account_tag != spec("b").account_tag
        assert spec("a", account="org").account_tag == spec("b", account="org").account_tag


class TestBuildRefusals:
    def test_zero_keys(self) -> None:
        with pytest.raises(ValueError, match="at least one key"):
            KeyPool(PROVIDER, "pool", [])

    def test_duplicate_labels(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            make_pool(spec("a"), spec("a", secret="sk-ant-api03-other"))

    def test_duplicate_secrets_without_naming_the_secret(self) -> None:
        with pytest.raises(ValueError, match="same secret") as caught:
            make_pool(spec("a", secret=_SENTINEL), spec("b", secret=_SENTINEL))
        assert _SENTINEL not in str(caught.value)

    def test_a_subscription_token(self) -> None:
        token = "sk-ant-oat01-SENTINEL"
        with pytest.raises(ValueError, match="subscription") as caught:
            make_pool(spec("a", secret=token))
        assert token not in str(caught.value)

    def test_a_provider_with_no_classifier(self) -> None:
        with pytest.raises(ValueError, match="classifier"):
            KeyPool("bedrock", "pool", [spec("a")])

    @pytest.mark.parametrize("name", ["", "has space", "a/b", "x" * 65])
    def test_an_unsafe_pool_name(self, name: str) -> None:
        with pytest.raises(ValueError, match="pool name"):
            KeyPool(PROVIDER, name, [spec("a")])

    @pytest.mark.parametrize("anchor_day", [0, 29, True])
    def test_an_anchor_day_outside_1_to_28(self, anchor_day: int) -> None:
        with pytest.raises(ValueError, match="anchor_day"):
            make_pool(spec("a"), anchor_day=anchor_day)

    @pytest.mark.parametrize("headroom", [Decimal(-1), Decimal(1), Decimal("NaN")])
    def test_a_headroom_that_is_not_a_fraction(self, headroom: Decimal) -> None:
        with pytest.raises(ValueError, match="min_headroom"):
            make_pool(spec("a"), min_headroom=headroom)

    def test_one_key_is_a_normal_pool(self) -> None:
        pool = make_pool(spec("only"))
        assert pool.labels == ("only",)
        assert pool.provider == PROVIDER

    def test_the_pool_repr_shows_labels_only(self) -> None:
        pool = make_pool(spec("a", secret=_SENTINEL))
        assert _SENTINEL not in repr(pool)
        assert "'a'" in repr(pool)

    def test_preference_order_is_priority_then_configuration(self) -> None:
        pool = make_pool(spec("c", priority=2), spec("a"), spec("b", priority=1), spec("d", priority=1))
        assert pool.labels == ("b", "d", "c", "a")


def _capabilities(**fields: object) -> ModelCapabilities:
    base: dict[str, object] = {
        "model_name": "test-model",
        "model_type": ModelType.CHAT,
        "model_tier": ModelTier.MEDIUM,
        "provider_name": PROVIDER,
        "cost_per_input_token": Decimal("0.000003"),
        "cost_per_output_token": Decimal("0.000015"),
        "max_output_tokens": 1000,
    }
    base.update(fields)
    return ModelCapabilities(**base)  # type: ignore[arg-type]


class TestReservationEstimate:
    def test_input_at_the_input_rate_plus_max_tokens_at_the_output_rate(self) -> None:
        caps = _capabilities()
        assert (
            reservation_estimate(caps, input_tokens=2000, max_tokens=500)
            == Decimal("0.000003") * 2000 + Decimal("0.000015") * 500
        )

    def test_the_models_max_output_tokens_when_the_call_sets_none(self) -> None:
        caps = _capabilities()
        assert reservation_estimate(caps, input_tokens=0) == Decimal("0.000015") * 1000

    def test_the_flat_request_price_is_reserved_too(self) -> None:
        caps = _capabilities(cost_per_request=Decimal("0.01"))
        assert reservation_estimate(caps, input_tokens=0, max_tokens=0) == Decimal("0.01")

    @pytest.mark.parametrize("missing", ["cost_per_input_token", "cost_per_output_token"])
    def test_a_model_with_no_price_is_refused_naming_the_model(self, missing: str) -> None:
        with pytest.raises(UnpricedModelError, match="test-model"):
            reservation_estimate(_capabilities(**{missing: None}), input_tokens=10, max_tokens=10)

    def test_a_call_with_no_output_bound_is_refused(self) -> None:
        with pytest.raises(NoOutputLimitError, match="max_tokens"):
            reservation_estimate(_capabilities(max_output_tokens=None), input_tokens=10)

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError):
            reservation_estimate(_capabilities(), input_tokens=-1)


class TestNoKeyAvailableErrorType:
    def _error(self) -> NoKeyAvailableError:
        key = KeyStatus(
            label="a",
            state=KeyState.EXHAUSTED,
            until=None,
            spent=Decimal(10),
            reserved=Decimal(0),
            remaining=Decimal(0),
            budget=Decimal(10),
            active=True,
        )
        return NoKeyAvailableError(pool="p", provider=PROVIDER, model_id="m", keys=(key,), earliest=None)

    def test_it_is_a_provider_error(self) -> None:
        error = self._error()
        assert isinstance(error, ModelProviderError)
        assert is_provider_error(error)

    def test_its_friendly_message_is_fixed(self) -> None:
        message = friendly_api_error(self._error())
        assert message == (
            f"No {PROVIDER} key can take this request right now. Please retry later or contact an administrator."
        )
        assert "exhausted" not in message
