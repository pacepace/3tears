"""a rate limit opens one model's breaker, never its provider's other models.

metallm dev e2e, 2026-10-15: 26 OpenRouter 429s, then three models refused: one OpenRouter model's 429s opened the breaker for
every model on OpenRouter. A 429 belongs to one model's route; a provider outage
is still the provider's.
"""

from __future__ import annotations

import pytest

from threetears.models.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError, CircuitState
from threetears.models.errors import ModelRateLimitError, is_rate_limit


def _callback(registry: CircuitBreakerRegistry, model: str):  # noqa: ANN202 - a callback handler
    provider = registry.get("openrouter", credential="k")
    return provider.make_callback(rate_limit=registry.get("openrouter", credential="k", model=model))


def _limited() -> Exception:
    return type("TooManyRequests", (Exception,), {"status_code": 429})()


class TestWhatIsARateLimit:
    @pytest.mark.parametrize(
        ("error", "limited"),
        [
            (ModelRateLimitError("slow down", provider="Anthropic"), True),
            (type("TooManyRequests", (Exception,), {"status_code": 429})(), True),
            (ValueError("OpenRouter API returned an error during streaming: busy (code: 429)"), True),
            (ValueError("OpenRouter API returned an error during streaming: down (code: 502)"), False),
            (type("ServerError", (Exception,), {"status_code": 500})(), False),
            (ValueError("(code: 429) but not from OpenRouter"), False),
        ],
    )
    def test_each_shape_a_provider_reports_it_in(self, error: Exception, limited: bool) -> None:
        assert is_rate_limit(error) is limited


class TestOneModelsLimit:
    def test_one_models_rate_limits_open_its_own_breaker_and_no_other(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=2)
        flash, sonnet = _callback(registry, "deepseek/deepseek-v4.1-flash"), _callback(registry, "anthropic/claude")
        for _ in range(2):
            flash.on_llm_error(_limited())
        with pytest.raises(CircuitOpenError) as opened:
            flash.on_chat_model_start({}, [[]])
        assert opened.value.model_name == "deepseek/deepseek-v4.1-flash"
        sonnet.on_chat_model_start({}, [[]])  # the provider's other model still runs
        assert registry.status() == {"openrouter": CircuitState.CLOSED}
        assert registry.model_status() == {"openrouter/deepseek/deepseek-v4.1-flash": CircuitState.OPEN}

    def test_a_provider_failure_still_opens_the_provider_for_every_model(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=2)
        flash, sonnet = _callback(registry, "deepseek/deepseek-v4.1-flash"), _callback(registry, "anthropic/claude")
        for _ in range(2):
            flash.on_llm_error(type("ServerError", (Exception,), {"status_code": 502})())
        with pytest.raises(CircuitOpenError):
            sonnet.on_chat_model_start({}, [[]])
        assert registry.status() == {"openrouter": CircuitState.OPEN}

    def test_with_no_model_breaker_a_rate_limit_counts_against_the_provider_as_before(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=2)
        callback = registry.get("openrouter").make_callback()
        for _ in range(2):
            callback.on_llm_error(_limited())
        assert registry.status() == {"openrouter": CircuitState.OPEN}

    def test_a_success_closes_the_models_breaker_too(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=2)
        flash = _callback(registry, "m")
        flash.on_llm_error(_limited())
        flash.on_llm_end(None)
        flash.on_llm_error(_limited())
        flash.on_chat_model_start({}, [[]])  # one failure since the success: still closed


class TestTheFactoryGivesEachModelItsOwn:
    def test_two_models_on_one_provider_and_key_get_two_rate_limit_breakers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from threetears.models import factory
        from threetears.models.circuit_breaker import CircuitBreakerCallback

        registry = CircuitBreakerRegistry(failure_threshold=2)
        monkeypatch.setattr(factory, "_DEFAULT_BREAKER_REGISTRY", registry)

        def _breaker_callback(model_id: str) -> CircuitBreakerCallback:
            model = factory.create_chat_model(model_id=model_id, api_key="k", provider="openrouter")
            return next(c for c in model.config["callbacks"] if isinstance(c, CircuitBreakerCallback))

        flash, sonnet = _breaker_callback("deepseek/deepseek-v4.1-flash"), _breaker_callback("anthropic/claude")
        for _ in range(2):
            flash.on_llm_error(_limited())
        sonnet.on_chat_model_start({}, [[]])
        with pytest.raises(CircuitOpenError):
            flash.on_chat_model_start({}, [[]])
