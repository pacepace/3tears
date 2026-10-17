"""A caller can set the CLI's own API retries and its output cap, on a subscription model.

The bundled CLI retries a failed API call itself (``CLAUDE_CODE_MAX_RETRIES``, ten by default) and
caps a reply at its own per-model default (``CLAUDE_CODE_MAX_OUTPUT_TOKENS``). Both were out of a
caller's reach: the anthropic factory took ``max_retries`` for the HTTP route and dropped it for a
subscription token, and the subscription model forwarded neither. So a caller that turned the API
route's retries off still had up to ten hidden attempts behind each subscription call, and no way
to bound what one could cost. What reaches the CLI is read at its door (:mod:`.claude_cli_recorder`).
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from langchain_core.messages import HumanMessage  # noqa: E402

from threetears.models import DEFAULT_CHAT_MODEL  # noqa: E402
from threetears.models.factory import create_chat_model  # noqa: E402

from .claude_cli_recorder import TOKEN, pooled_launch, sent_to_cli, subscription_model  # noqa: E402

_RETRIES_ENV = "CLAUDE_CODE_MAX_RETRIES"
_OUTPUT_ENV = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"
_MESSAGES = [HumanMessage(content="hi")]


def _through_the_factory(**provider_kwargs: Any) -> Any:
    """a subscription model built the way most callers build one: ``create_chat_model`` with a token.

    :param provider_kwargs: what the caller passes to ``create_chat_model``
    :ptype provider_kwargs: Any
    :return: a builder for :func:`sent_to_cli`
    :rtype: Any
    """
    return lambda: create_chat_model(DEFAULT_CHAT_MODEL, api_key=TOKEN, **provider_kwargs)


class TestTheCliRetries:
    def test_max_retries_on_the_factory_reaches_the_cli(self) -> None:
        call = sent_to_cli(_MESSAGES, build=_through_the_factory(max_retries=0))

        assert call.options.env[_RETRIES_ENV] == "0"

    def test_max_retries_on_the_subscription_model_reaches_the_cli(self) -> None:
        call = sent_to_cli(_MESSAGES, build=lambda: subscription_model(max_retries=3))

        assert call.options.env[_RETRIES_ENV] == "3"

    def test_max_retries_bound_per_call_reaches_the_cli(self) -> None:
        call = sent_to_cli(_MESSAGES, build=lambda: subscription_model().bind(max_retries=1))

        assert call.options.env[_RETRIES_ENV] == "1"

    def test_left_unset_the_cli_keeps_its_own_default(self) -> None:
        call = sent_to_cli(_MESSAGES, build=_through_the_factory())

        assert _RETRIES_ENV not in call.options.env

    def test_a_pooled_cli_launches_with_it(self) -> None:
        launch = pooled_launch(_MESSAGES, build=lambda: subscription_model(max_retries=0))

        assert launch.options.env[_RETRIES_ENV] == "0"
        assert launch.fallback.env[_RETRIES_ENV] == "0"

    @pytest.mark.parametrize("bad", [-1, True, 1.5, "2"])
    def test_a_count_that_is_not_a_whole_number_of_retries_is_refused(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            sent_to_cli(_MESSAGES, build=lambda: subscription_model().bind(max_retries=bad))


class TestTheCliOutputCap:
    def test_max_output_tokens_on_the_factory_reaches_the_cli(self) -> None:
        call = sent_to_cli(_MESSAGES, build=_through_the_factory(max_output_tokens=4096))

        assert call.options.env[_OUTPUT_ENV] == "4096"

    def test_max_output_tokens_bound_per_call_reaches_the_cli(self) -> None:
        call = sent_to_cli(
            _MESSAGES, build=lambda: subscription_model(max_output_tokens=4096).bind(max_output_tokens=512)
        )

        assert call.options.env[_OUTPUT_ENV] == "512", "a per-call value outranks the model's"

    def test_left_unset_the_cli_keeps_its_own_default(self) -> None:
        call = sent_to_cli(_MESSAGES, build=_through_the_factory())

        assert _OUTPUT_ENV not in call.options.env

    @pytest.mark.parametrize("bad", [0, -5, False, 2.0])
    def test_a_cap_that_is_not_a_positive_whole_number_is_refused(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="max_output_tokens"):
            sent_to_cli(_MESSAGES, build=lambda: subscription_model().bind(max_output_tokens=bad))


class TestNeitherIsDroppedAsAnUnknownOption:
    def test_no_dropped_option_warning_names_them(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="threetears.models.providers.claude_cli"):
            sent_to_cli(_MESSAGES, build=lambda: subscription_model().bind(max_retries=0, max_output_tokens=100))

        dropped = [r for r in caplog.records if "they were not sent" in r.getMessage()]
        assert dropped == [], [r.extra_data for r in dropped]  # type: ignore[attr-defined]


class TestTheHttpRouteKeepsItsOwnRetries:
    """``max_retries`` now defaults to ``None`` on the anthropic factory so a subscription token can tell
    a caller's value from the default; an API key still gets the SDK's two retries, or the caller's."""

    def test_left_unset_an_api_key_model_retries_twice(self) -> None:
        from threetears.models.providers.anthropic import create_anthropic_chat  # noqa: PLC0415

        assert create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-ant-api03-faketestkey").max_retries == 2  # type: ignore[attr-defined]

    def test_a_callers_count_reaches_an_api_key_model(self) -> None:
        from threetears.models.providers.anthropic import create_anthropic_chat  # noqa: PLC0415

        model = create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-ant-api03-faketestkey", max_retries=0)
        assert model.max_retries == 0  # type: ignore[attr-defined]
