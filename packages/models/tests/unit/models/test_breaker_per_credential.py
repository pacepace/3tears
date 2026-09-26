"""the default circuit breaker is scoped to a credential as well as a provider.

a multi-tenant consumer calls one provider with many customers' keys. one
customer's revoked, rate-limited or out-of-credit key failing five times in a
row must not fast-fail every other customer on that provider -- which is what a
breaker keyed by provider alone does. provider granularity stays (a breaker is
never per model); only the credential is separated.

the key itself must never surface: not as the registry's key, not in a log
line, not in anything a caller could export as a metric label.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import Any

import pytest

from threetears.models import DEFAULT_CHAT_MODEL
from threetears.models.circuit_breaker import (
    CircuitBreakerCallback,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)
from threetears.models.factory import create_chat_model

_THRESHOLD = 5  # the default registry's failure_threshold


def _key() -> str:
    """a fresh credential per test, so the process-wide default registry never carries state between tests.

    :return: an api-key-shaped secret
    :rtype: str
    """
    return f"sk-ant-{secrets.token_hex(24)}"


def _breaker_callback(model: Any) -> CircuitBreakerCallback:
    """the breaker callback the factory attached to ``model``.

    :param model: the configured model the factory returned
    :ptype model: Any
    :return: its circuit-breaker callback
    :rtype: CircuitBreakerCallback
    """
    (callback,) = [cb for cb in model.config["callbacks"] if isinstance(cb, CircuitBreakerCallback)]
    return callback


def _trip(callback: CircuitBreakerCallback) -> None:
    """fail the provider call behind ``callback`` until its breaker opens.

    :param callback: the breaker callback to fail through
    :ptype callback: CircuitBreakerCallback
    :return: nothing
    :rtype: None
    """
    for _ in range(_THRESHOLD):
        callback.on_llm_error(RuntimeError("401 invalid x-api-key"))


class TestTheFactoryDefault:
    def test_one_credential_tripping_leaves_another_on_the_same_provider_closed(self) -> None:
        revoked = _breaker_callback(create_chat_model(DEFAULT_CHAT_MODEL, api_key=_key()))
        healthy = _breaker_callback(create_chat_model(DEFAULT_CHAT_MODEL, api_key=_key()))

        _trip(revoked)

        with pytest.raises(CircuitOpenError):
            revoked.on_chat_model_start({}, [[]])
        healthy.on_chat_model_start({}, [[]])  # must not raise

    def test_one_credential_behaves_as_it_always_did(self) -> None:
        """two models on one key share one breaker: tripping through either fast-fails both."""
        key = _key()
        first = _breaker_callback(create_chat_model(DEFAULT_CHAT_MODEL, api_key=key))
        second = _breaker_callback(create_chat_model(DEFAULT_CHAT_MODEL, api_key=key))

        _trip(first)

        with pytest.raises(CircuitOpenError):
            second.on_chat_model_start({}, [[]])

    def test_an_explicit_breaker_is_still_the_one_used(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=1)
        mine = registry.get("anthropic")
        callback = _breaker_callback(create_chat_model(DEFAULT_CHAT_MODEL, api_key=_key(), breaker=mine))

        callback.on_llm_error(RuntimeError("boom"))

        assert mine.state == CircuitState.OPEN

    def test_the_key_appears_in_no_log_line_and_no_error(self, caplog: pytest.LogCaptureFixture) -> None:
        key = _key()
        callback = _breaker_callback(create_chat_model(DEFAULT_CHAT_MODEL, api_key=key))

        with caplog.at_level(logging.DEBUG):
            _trip(callback)
            with pytest.raises(CircuitOpenError) as refused:
                callback.on_chat_model_start({}, [[]])

        assert any("opening" in record.getMessage() for record in caplog.records)
        assert all(key not in record.getMessage() for record in caplog.records)
        assert key not in str(refused.value)
        assert refused.value.provider_name == "anthropic"


class TestTheRegistry:
    def test_credentials_on_one_provider_get_separate_breakers(self) -> None:
        registry = CircuitBreakerRegistry()
        a, b = _key(), _key()

        assert registry.get("anthropic", credential=a) is registry.get("anthropic", credential=a)
        assert registry.get("anthropic", credential=a) is not registry.get("anthropic", credential=b)
        assert registry.get("anthropic", credential=a) is not registry.get("anthropic")

    def test_no_credential_or_its_fingerprint_is_held_or_reported(self) -> None:
        """the registry is keyed by a keyed hash, and status stays keyed by provider alone.

        ``status()`` is what a caller would turn into metric labels, so it must not grow
        one label per customer key; it reports each provider's worst state instead.
        """
        registry = CircuitBreakerRegistry(failure_threshold=1)
        revoked, healthy = _key(), _key()
        registry.get("anthropic", credential=revoked).record_failure()
        registry.get("anthropic", credential=healthy)
        registry.get("openai")

        assert registry.status() == {"anthropic": CircuitState.OPEN, "openai": CircuitState.CLOSED}
        held = repr(vars(registry))
        assert revoked not in held
        assert healthy not in held

    def test_the_fingerprint_is_not_a_plain_hash_anyone_could_recompute(self) -> None:
        """a plain digest of a key is a stable identifier for it; a keyed one is meaningless outside the registry."""
        key = _key()
        registry = CircuitBreakerRegistry()
        registry.get("anthropic", credential=key)

        held = repr(vars(registry))
        plain_digests = [hashlib.blake2b(key.encode(), digest_size=size).hexdigest() for size in (8, 16, 32, 64)]
        plain_digests.append(hashlib.sha256(key.encode()).hexdigest())
        for plain in plain_digests:
            assert plain[:16] not in held

    def test_reset_by_provider_resets_every_credential_and_by_credential_only_that_one(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=1)
        a, b = _key(), _key()
        breaker_a = registry.get("anthropic", credential=a)
        breaker_b = registry.get("anthropic", credential=b)
        breaker_a.record_failure()
        breaker_b.record_failure()

        registry.reset("anthropic", credential=a)
        assert (breaker_a.state, breaker_b.state) == (CircuitState.CLOSED, CircuitState.OPEN)

        registry.reset("anthropic")
        assert breaker_b.state == CircuitState.CLOSED
