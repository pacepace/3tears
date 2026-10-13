"""The models adapter's failure describer (``ModelsFailureDescriber``), over exceptions built as each SDK builds them.

What the port asks of a describer (:class:`~threetears.evals.schema.ProviderFailureDescriber`): total over
``BaseException``; a description carrying the provider, the status and the provider's own message and nothing
else; ``payload_withheld`` set exactly where the exception's own text, or one it chains, carries more than the
description; and ``account_refused`` set where the provider refused the calling account rather than the request.
Every exception here is constructed the way its SDK constructs it (a status error from an ``httpx.Response`` and
its parsed body, OpenRouter's typed errors from their parsed ``data``), each body carrying an account field the
description must never repeat.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
import httpx
import openai
import pydantic
import pytest
from openrouter import errors as openrouter_errors

from threetears.evals.providers import ModelsClientRefused, ModelsFailureDescriber
from threetears.models import CircuitOpenError, ModelCallTimeout, ModelProviderError, ModelRateLimitError

_REQUEST = httpx.Request("POST", "https://provider.invalid/v1/messages")


def _response(code: int, body: Any) -> httpx.Response:
    return httpx.Response(code, json=body, request=_REQUEST)


def _anthropic(cls: type[anthropic.APIStatusError], code: int, message: str) -> anthropic.APIStatusError:
    body = {"type": "error", "error": {"type": "x", "message": message}, "account_id": "acct_SECRET"}
    return cls(f"Error code: {code} - {body}", response=_response(code, body), body=body)


def _openai(
    cls: type[openai.APIStatusError], code: int, message: str, error_code: str | None = None
) -> openai.APIStatusError:
    body = {"message": message, "type": "x", "code": error_code, "account_id": "acct_SECRET"}
    return cls(f"Error code: {code} - {body}", response=_response(code, body), body=body)


def _openrouter(cls: type[Any], code: int, message: str) -> Any:
    body = json.dumps({"error": {"code": code, "message": message}, "user_id": "user_SECRET"})
    raw = httpx.Response(code, text=body, request=_REQUEST, headers={"content-type": "application/json"})
    model = {
        openrouter_errors.PaymentRequiredResponseError: openrouter_errors.PaymentRequiredResponseErrorData,
        openrouter_errors.UnauthorizedResponseError: openrouter_errors.UnauthorizedResponseErrorData,
        openrouter_errors.ForbiddenResponseError: openrouter_errors.ForbiddenResponseErrorData,
        openrouter_errors.TooManyRequestsResponseError: openrouter_errors.TooManyRequestsResponseErrorData,
    }[cls]
    return cls(model.model_validate(json.loads(body)), raw, body)


describe = ModelsFailureDescriber()


@pytest.mark.parametrize(
    ("exc", "says", "withheld", "account"),
    [
        (_anthropic(anthropic.RateLimitError, 429, "slow down"), "anthropic 429: slow down", True, False),
        (_anthropic(anthropic.AuthenticationError, 401, "invalid x-api-key"), "anthropic 401", True, True),
        (_anthropic(anthropic.PermissionDeniedError, 403, "forbidden"), "anthropic 403", True, True),
        (_anthropic(anthropic.APIStatusError, 402, "pay up"), "anthropic 402: pay up", True, True),
        (
            _anthropic(anthropic.BadRequestError, 400, "Your credit balance is too low to access the Anthropic API"),
            "credit balance",
            True,
            True,
        ),
        (_anthropic(anthropic.BadRequestError, 400, "bad field"), "anthropic 400: bad field", True, False),
        (_anthropic(anthropic.InternalServerError, 529, "Overloaded"), "anthropic 529: Overloaded", True, False),
        (_openai(openai.RateLimitError, 429, "slow down", "rate_limit_exceeded"), "openai 429: slow down", True, False),
        (
            _openai(openai.RateLimitError, 429, "You exceeded your current quota", "insufficient_quota"),
            "quota",
            True,
            True,
        ),
        (_openai(openai.AuthenticationError, 401, "bad key"), "openai 401: bad key", True, True),
        (_openai(openai.BadRequestError, 400, "Unsupported value: 'temperature'"), "openai 400", True, False),
        (
            _openrouter(openrouter_errors.PaymentRequiredResponseError, 402, "Insufficient credits"),
            "openrouter 402: Insufficient credits",
            False,
            True,
        ),
        (
            _openrouter(openrouter_errors.UnauthorizedResponseError, 401, "No auth"),
            "openrouter 401: No auth",
            False,
            True,
        ),
        (
            _openrouter(openrouter_errors.ForbiddenResponseError, 403, "flagged by moderation"),
            "openrouter 403",
            False,
            False,
        ),
        (
            _openrouter(openrouter_errors.TooManyRequestsResponseError, 429, "Rate limited"),
            "openrouter 429",
            False,
            False,
        ),
        (
            ValueError("OpenRouter API returned an error during streaming: upstream died (code: 502)"),
            "openrouter 502 (mid-stream): upstream died",
            False,
            False,
        ),
        (
            ValueError("OpenRouter API returned an error during streaming: Insufficient credits (code: 402)"),
            "openrouter 402 (mid-stream): Insufficient credits",
            False,
            True,
        ),
        (
            ValueError("OpenRouter API returned an error: User not found. (code: 401)"),
            "openrouter 401: User not found.",
            False,
            True,
        ),
        (
            ValueError("OpenRouter API returned an error: Provider returned error (code: 502)"),
            "openrouter 502: Provider returned error",
            False,
            False,
        ),
        (ValueError({"message": "in-body error", "account_id": "acct_SECRET"}), "ValueError", True, False),
        (anthropic.APIConnectionError(request=_REQUEST), "anthropic: the request could not connect", False, False),
        (openai.APITimeoutError(request=_REQUEST), "openai: the request timed out", False, False),
        (CircuitOpenError("anthropic", 12.0), "Circuit open for anthropic", False, False),
        (CircuitOpenError("anthropic", 12.0, credential_scoped=True), "one credential", False, True),
        (ModelCallTimeout("no chunk within 120s"), "past its deadline", False, False),
        (TimeoutError(), "timed out", False, False),
        (ModelsClientRefused("reasoning_effort", "not a word"), "reasoning_effort refused", False, False),
        (
            ModelProviderError("bad key", provider="Claude", reason="authentication_failed"),
            "Claude (authentication_failed)",
            False,
            True,
        ),
        (ModelRateLimitError("limit", provider="Claude", reason="rate_limit", resets="5pm"), "resets 5pm", False, True),
        (ModelRateLimitError("limit", provider="Claude", reason="rate_limit"), "Claude (rate_limit)", False, False),
        (KeyError("acct_SECRET"), "KeyError", True, False),
        (asyncio.CancelledError(), "CancelledError", True, False),
    ],
)
def test_each_failure_reads_as_the_port_says(exc: BaseException, says: str, withheld: bool, account: bool) -> None:
    failure = describe(exc)
    assert says in failure.description
    assert (failure.payload_withheld, failure.account_refused) == (withheld, account)
    assert "SECRET" not in failure.description, "a payload was repeated"


def test_a_context_overflow_says_so() -> None:
    from langchain_anthropic.chat_models import AnthropicContextOverflowError

    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long: 300000 tokens"}}
    exc = AnthropicContextOverflowError("Error code: 400", response=_response(400, body), body=body)
    assert "context window" in describe(exc).description


def test_a_malformed_reply_chunk_is_withheld() -> None:
    class Chunk(pydantic.BaseModel):
        model: str

    with pytest.raises(pydantic.ValidationError) as caught:
        Chunk.model_validate({"secret": "acct_SECRET"})
    failure = describe(caught.value)
    assert failure.payload_withheld and "SECRET" not in failure.description


def test_openrouter_s_default_error_withholds_the_body_its_text_appends() -> None:
    body = json.dumps({"error": {"code": 418, "message": "teapot"}, "user_id": "user_SECRET"})
    raw = httpx.Response(418, text=body, request=_REQUEST, headers={"content-type": "application/json"})
    exc = openrouter_errors.OpenRouterDefaultError("API error occurred", raw, body)
    assert "SECRET" in str(exc)
    failure = describe(exc)
    assert failure.payload_withheld and failure.description == "openrouter 418: teapot"


def test_openrouter_s_non_streamed_in_body_error_is_not_read_as_mid_stream() -> None:
    """``_create_chat_result`` raises the same bare ``ValueError`` without "during streaming": it is no stream's error."""
    failure = describe(ValueError("OpenRouter API returned an error: Insufficient credits (code: 402)"))
    assert failure.description == "openrouter 402: Insufficient credits"
    assert failure.account_refused and not failure.payload_withheld
