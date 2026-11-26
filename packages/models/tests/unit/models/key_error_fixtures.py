"""Recorded provider error responses, for the key-error classifiers and the 429-unretried mode.

Each fixture is what a provider's API sends for one failure: status, headers and body, served
through a scripted wire so the SDK and LangChain turn it into the exception a caller of
``model.invoke`` really sees. The comment beside each names where its shape comes from. "Inferred"
marks the parts no source showed verbatim -- usually a message's exact wording, which no rule reads
unless the comment says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

__all__ = [
    "ANTHROPIC_API_ERROR_500",
    "ANTHROPIC_AUTHENTICATION_401",
    "ANTHROPIC_BILLING_402",
    "ANTHROPIC_CONTRACT_SPEND_LIMIT_429",
    "ANTHROPIC_CREDIT_TOO_LOW_400",
    "ANTHROPIC_INVALID_REQUEST_400",
    "ANTHROPIC_MID_STREAM_OVERLOADED",
    "ANTHROPIC_MID_STREAM_RATE_LIMIT",
    "ANTHROPIC_OVERLOADED_529",
    "ANTHROPIC_PERMISSION_403",
    "ANTHROPIC_RAMP_LIMIT_429",
    "ANTHROPIC_RATE_LIMIT_429",
    "ANTHROPIC_RATE_LIMIT_429_LONG_WAIT",
    "ANTHROPIC_RATE_LIMIT_429_RESET_ONLY",
    "ANTHROPIC_SPEND_CAP_429",
    "ANTHROPIC_SPEND_LIMIT_400",
    "ANTHROPIC_WORKSPACE_SPEND_LIMIT_400",
    "OPENAI_BILLING_HARD_LIMIT_429",
    "OPENAI_COMPATIBLE_BALANCE_402",
    "OPENAI_CREDIT_BALANCE_EXHAUSTED_429",
    "OPENAI_INSUFFICIENT_QUOTA_429",
    "OPENAI_INVALID_API_KEY_401",
    "OPENAI_MID_STREAM_QUOTA",
    "OPENAI_MID_STREAM_RATE_LIMIT",
    "OPENAI_ORGANIZATION_SPEND_LIMIT_429",
    "OPENAI_ORGANIZATION_USAGE_LIMIT_429",
    "OPENAI_PROJECT_SPEND_LIMIT_429",
    "OPENAI_RATE_LIMIT_429",
    "OPENAI_RATE_LIMIT_429_RESET_ONLY",
    "OPENAI_SLOW_DOWN_429",
    "OPENAI_REGION_403",
    "OPENAI_SERVER_500",
    "OPENROUTER_BARE_402",
    "OPENROUTER_CONTENT_POLICY_403",
    "OPENROUTER_CREDITS_402",
    "OPENROUTER_FORBIDDEN_403",
    "OPENROUTER_GUARDRAIL_BUDGET_403",
    "OPENROUTER_IN_FLIGHT_402",
    "OPENROUTER_IN_FLIGHT_402_NO_RETRY_AFTER",
    "OPENROUTER_INVALID_KEY_401",
    "OPENROUTER_KEY_BUDGET_403",
    "OPENROUTER_KEY_LIMIT_402",
    "OPENROUTER_MID_STREAM_RATE_LIMIT",
    "OPENROUTER_MODERATION_403",
    "OPENROUTER_RATE_LIMIT_429",
    "OPENROUTER_RATE_LIMIT_429_RESET_ONLY",
    "OPENROUTER_REFUSAL_403",
    "OPENROUTER_UPSTREAM_502",
    "OPENROUTER_WEIGHT_EXCEEDS_BUDGET_402",
    "RecordedError",
    "RecordedStream",
]


@dataclass(frozen=True)
class RecordedError:
    """one error response, as the provider's API sends it.

    :ivar status: the HTTP status
    :ivar body: the JSON body
    :ivar headers: the response headers beyond ``content-type``
    """

    status: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)

    def response(self, *, extra_headers: dict[str, str] | None = None) -> httpx.Response:
        """the response, for an ``httpx`` handler to return.

        :param extra_headers: headers to add, such as a short ``retry-after-ms`` that keeps an SDK's
            retry sleep out of a test
        :ptype extra_headers: dict[str, str] | None
        :return: the response
        :rtype: httpx.Response
        """
        return httpx.Response(self.status, json=self.body, headers={**self.headers, **(extra_headers or {})})


@dataclass(frozen=True)
class RecordedStream:
    """a streamed answer that fails after it began: the 200's server-sent events, the error last.

    :ivar events: the event-stream body
    """

    events: bytes

    def response(self) -> httpx.Response:
        """the response, for an ``httpx`` handler to return.

        :return: the response
        :rtype: httpx.Response
        """
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=self.events)


def _anthropic(error_type: str, message: str, **extra: Any) -> dict[str, Any]:
    # the Claude API errors page, "Error shapes", and anthropic.types.shared.ErrorResponse
    return {"type": "error", "error": {"type": error_type, "message": message, **extra}, "request_id": "req_test"}


# -- anthropic -----------------------------------------------------------------------------------

#: Claude API errors page: "402 - billing_error". Message: inferred.
ANTHROPIC_BILLING_402 = RecordedError(
    402, _anthropic("billing_error", "There is an issue with your billing or payment information.")
)

#: Message: verbatim from errors.py's ``_extract_provider_body_message`` docstring, which quotes it from
#: a production 400 (2026-05-13). Type ``invalid_request_error``: the errors page's type for a 400;
#: inferred for this case. The classifier's message rule for credit names this fixture.
ANTHROPIC_CREDIT_TOO_LOW_400 = RecordedError(
    400,
    _anthropic(
        "invalid_request_error",
        "Your credit balance is too low to access the Anthropic API. "
        "Please go to Plans & Billing to upgrade or purchase credits.",
    ),
)

#: Claude API rate-limits page, "Setting your own spend limit": a 400 ``invalid_request_error`` whose
#: message starts "You have reached your specified API usage limits" and states when access resumes.
#: The tail is inferred. The classifier's spend-limit message rule names this fixture.
ANTHROPIC_SPEND_LIMIT_400 = RecordedError(
    400,
    _anthropic(
        "invalid_request_error",
        "You have reached your specified API usage limits. You will regain access on 2026-12-01 at 00:00 UTC.",
    ),
)

#: As above, for a workspace limit: "You have reached your specified workspace API usage limits".
#: The tail is inferred.
ANTHROPIC_WORKSPACE_SPEND_LIMIT_400 = RecordedError(
    400,
    _anthropic(
        "invalid_request_error",
        "You have reached your specified workspace API usage limits. "
        "You will regain access on 2026-12-01 at 00:00 UTC.",
    ),
)

#: Verbatim from the Claude API rate-limits page, "Reaching your spend cap": a 429 with
#: ``details.error_code = enforced_spend_limit_reached`` and no ``retry-after``.
ANTHROPIC_SPEND_CAP_429 = RecordedError(
    429,
    _anthropic(
        "rate_limit_error",
        "You have reached your API usage limits: your organization has crossed its monthly API usage threshold, "
        "set based on your organization's API tier. You will regain access on 2026-09-01 at 00:00 UTC.",
        details={"error_code": "enforced_spend_limit_reached"},
    ),
)

#: Claude API errors page, "Error codes": ``contract_spend_limit_reached`` on a 429 -- a contract's spend
#: limit is used up. Message: inferred.
ANTHROPIC_CONTRACT_SPEND_LIMIT_429 = RecordedError(
    429,
    _anthropic(
        "rate_limit_error",
        "Your organization has reached its contracted spend limit.",
        details={"error_code": "contract_spend_limit_reached"},
    ),
)

#: Claude API errors page, "Error codes": ``ramp_limit_exceeded`` on a 429 -- traffic rose faster than
#: the organisation's ramp allows. A rate, not a quota. Message and ``retry-after``: inferred.
ANTHROPIC_RAMP_LIMIT_429 = RecordedError(
    429,
    _anthropic(
        "rate_limit_error",
        "Your organization's usage increased too quickly; please ramp up gradually.",
        details={"error_code": "ramp_limit_exceeded"},
    ),
    headers={"retry-after": "5"},
)

#: Claude API errors page: "401 - authentication_error". Message: inferred.
ANTHROPIC_AUTHENTICATION_401 = RecordedError(401, _anthropic("authentication_error", "invalid x-api-key"))

#: Claude API errors page: "403 - permission_error". Message: inferred.
ANTHROPIC_PERMISSION_403 = RecordedError(
    403, _anthropic("permission_error", "Your API key does not have permission to use the specified resource.")
)

#: Claude API errors page: "429 - rate_limit_error"; rate-limits page, "Response headers": ``retry-after``
#: in seconds and the ``anthropic-ratelimit-*`` headers, resets in RFC 3339. Message and values: inferred.
ANTHROPIC_RATE_LIMIT_429 = RecordedError(
    429,
    _anthropic("rate_limit_error", "Number of request tokens has exceeded your per-minute rate limit"),
    headers={
        "retry-after": "17",
        "anthropic-ratelimit-requests-remaining": "41",
        "anthropic-ratelimit-requests-reset": "2026-11-26T12:00:05Z",
        "anthropic-ratelimit-input-tokens-remaining": "0",
        "anthropic-ratelimit-input-tokens-reset": "2026-11-26T12:00:40Z",
    },
)

#: The same 429 asking for an hour's wait: past the pool's two-minute threshold, so it counts as
#: exhaustion. That such a wait comes: inferred (the owner doubts it is real; a guard, not a case).
ANTHROPIC_RATE_LIMIT_429_LONG_WAIT = RecordedError(
    429,
    _anthropic("rate_limit_error", "Number of request tokens has exceeded your per-minute rate limit"),
    headers={"retry-after": "3600"},
)

#: The same 429 with no ``retry-after`` (the rate-limits page: "Not every 429 carries it"): two limits at
#: zero, so the later reset is when the key can serve again. Values: inferred.
ANTHROPIC_RATE_LIMIT_429_RESET_ONLY = RecordedError(
    429,
    _anthropic("rate_limit_error", "Number of request tokens has exceeded your per-minute rate limit"),
    headers={
        "anthropic-ratelimit-requests-remaining": "12",
        "anthropic-ratelimit-requests-reset": "2026-11-26T12:05:00Z",
        "anthropic-ratelimit-input-tokens-remaining": "0",
        "anthropic-ratelimit-input-tokens-reset": "2026-11-26T12:00:30Z",
        "anthropic-ratelimit-output-tokens-remaining": "0",
        "anthropic-ratelimit-output-tokens-reset": "2026-11-26T12:00:45+00:00",
    },
)

#: Claude API errors page: "529 - overloaded_error". Message: inferred.
ANTHROPIC_OVERLOADED_529 = RecordedError(529, _anthropic("overloaded_error", "Overloaded"))

#: Claude API errors page: "500 - api_error". Message: inferred.
ANTHROPIC_API_ERROR_500 = RecordedError(500, _anthropic("api_error", "Internal server error"))

#: Verbatim from the Claude API errors page, "Prefill not supported".
ANTHROPIC_INVALID_REQUEST_400 = RecordedError(
    400,
    _anthropic(
        "invalid_request_error",
        "This model does not support assistant message prefill. The conversation must end with a user message.",
    ),
)


def _anthropic_stream(error: dict[str, Any]) -> RecordedStream:
    # the Messages streaming protocol (message_start, then an ``error`` event, the Claude API
    # streaming page's "Error events"); anthropic/_streaming.py raises it with the stream's 200
    start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1},
        },
    }
    events = [("message_start", start), ("error", error)]
    return RecordedStream("".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode())


#: An error event inside a 200 stream: the streaming page's own example, ``overloaded_error``.
ANTHROPIC_MID_STREAM_OVERLOADED = _anthropic_stream(_anthropic("overloaded_error", "Overloaded"))

#: The same event carrying ``rate_limit_error``. That the API sends this type mid-stream: inferred.
ANTHROPIC_MID_STREAM_RATE_LIMIT = _anthropic_stream(_anthropic("rate_limit_error", "Rate limited"))


# -- openai --------------------------------------------------------------------------------------


def _openai(message: str, error_type: str | None, code: str | None) -> dict[str, Any]:
    # the wire shape the openai SDK parses: ``{"error": {...}}``, ``code``/``type``/``param`` lifted
    # from the inner object (openai/_exceptions.py APIError, openai/_client.py _make_status_error)
    return {"error": {"message": message, "type": error_type, "param": None, "code": code}}


#: KP-C2's ``insufficient_quota``. Message and the code equal to the type: inferred (the error-codes
#: page names the type, not this body).
OPENAI_INSUFFICIENT_QUOTA_429 = RecordedError(
    429,
    _openai(
        "You exceeded your current quota, please check your plan and billing details.",
        "insufficient_quota",
        "insufficient_quota",
    ),
)

#: OpenAI error-codes page: "429 - Credit balance exhausted" (code ``credit_balance_exhausted``; "the
#: broader error.type can still be insufficient_quota"). Message: inferred.
OPENAI_CREDIT_BALANCE_EXHAUSTED_429 = RecordedError(
    429, _openai("Your credit balance is exhausted.", "insufficient_quota", "credit_balance_exhausted")
)

#: OpenAI error-codes page: "429 - Project spend limit reached" (code ``project_spend_limit_exceeded``).
#: Message and type: inferred.
OPENAI_PROJECT_SPEND_LIMIT_429 = RecordedError(
    429, _openai("Project spend limit reached.", "insufficient_quota", "project_spend_limit_exceeded")
)

#: OpenAI error-codes page's 429 rows: an organisation's spend limit, its usage limit, and a billing
#: hard limit. Codes from the page; messages and the ``insufficient_quota`` type: inferred.
OPENAI_ORGANIZATION_SPEND_LIMIT_429 = RecordedError(
    429, _openai("Organization spend limit reached.", "insufficient_quota", "organization_spend_limit_exceeded")
)
OPENAI_ORGANIZATION_USAGE_LIMIT_429 = RecordedError(
    429, _openai("Organization usage limit reached.", "insufficient_quota", "organization_usage_limit_exceeded")
)
OPENAI_BILLING_HARD_LIMIT_429 = RecordedError(
    429, _openai("Billing hard limit has been reached.", "insufficient_quota", "billing_hard_limit_reached")
)

#: OpenAI error-codes page: "429 - Slow down" (code ``slow_down``): a sudden rise in traffic, a rate.
#: Message and type: inferred.
OPENAI_SLOW_DOWN_429 = RecordedError(
    429,
    _openai("Please reduce your rate of requests.", "rate_limit_error", "slow_down"),
    headers={"retry-after": "2"},
)

#: OpenAI error-codes page: "429 - Rate limit reached for requests"; rate-limits guide: ``Retry-After``
#: and the ``x-ratelimit-*`` headers with their sample values. Code ``rate_limit_exceeded`` and type
#: ``requests``: inferred.
OPENAI_RATE_LIMIT_429 = RecordedError(
    429,
    _openai("Rate limit reached for requests", "requests", "rate_limit_exceeded"),
    headers={
        "retry-after": "56",
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "1s",
        "x-ratelimit-remaining-tokens": "149984",
        "x-ratelimit-reset-tokens": "6m0s",
    },
)

#: The same 429 with no ``Retry-After`` (the guide: it "may be present"): requests and tokens both at
#: zero, so the later reset wins. Values: the guide's sample formats.
OPENAI_RATE_LIMIT_429_RESET_ONLY = RecordedError(
    429,
    _openai("Rate limit reached for requests", "requests", "rate_limit_exceeded"),
    headers={
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "1s",
        "x-ratelimit-remaining-tokens": "0",
        "x-ratelimit-reset-tokens": "6m0s",
        "x-ratelimit-remaining-project-tokens": "57000",
        "x-ratelimit-reset-project-tokens": "1h0m0s",
    },
)

#: OpenAI error-codes page: "401 - Incorrect API key provided". Type and code: inferred.
OPENAI_INVALID_API_KEY_401 = RecordedError(
    401, _openai("Incorrect API key provided: sk-test.", "invalid_request_error", "invalid_api_key")
)

#: OpenAI error-codes page: "403 - Country, region, or territory not supported". Type and code: inferred.
OPENAI_REGION_403 = RecordedError(
    403,
    _openai("Country, region, or territory not supported", "request_forbidden", "unsupported_country_region_territory"),
)

#: A 500. Body: inferred.
OPENAI_SERVER_500 = RecordedError(
    500, _openai("The server had an error processing your request.", "server_error", None)
)

#: An OpenAI-compatible endpoint's out-of-balance 402 (this factory serves them through ``base_url``).
#: Entirely inferred: OpenAI itself sends no 402.
OPENAI_COMPATIBLE_BALANCE_402 = RecordedError(402, _openai("Insufficient Balance", "unknown_error", None))

#: An error inside a 200 stream: a ``data:`` line whose JSON has an ``error`` object, which
#: openai/_streaming.py raises as a bare ``APIError``. That OpenAI sends a quota error this way: inferred.
OPENAI_MID_STREAM_QUOTA = RecordedStream(
    b"data: "
    + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1790000000,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}],
        }
    ).encode()
    + b"\n\ndata: "
    + json.dumps(_openai("You exceeded your current quota.", "insufficient_quota", "insufficient_quota")).encode()
    + b"\n\n"
)


#: A rate-limit error inside a 200 stream, carrying the ``rate_limit_error`` type and no code. That
#: OpenAI sends one this way: inferred.
OPENAI_MID_STREAM_RATE_LIMIT = RecordedStream(
    b"data: "
    + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1790000000,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}],
        }
    ).encode()
    + b"\n\ndata: "
    + json.dumps(_openai("Rate limit reached.", "rate_limit_error", None)).encode()
    + b"\n\n"
)


# -- openrouter ----------------------------------------------------------------------------------


def _openrouter(code: int, message: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    # OpenRouter errors page: ``ErrorResponse = { error: { code, message, metadata? } }``
    error: dict[str, Any] = {"code": code, "message": message}
    if metadata is not None:
        error["metadata"] = metadata
    return {"error": error}


#: OpenRouter errors page: "401: Invalid credentials (OAuth session expired, disabled/invalid API key)".
#: Message: inferred.
OPENROUTER_INVALID_KEY_401 = RecordedError(401, _openrouter(401, "User not found."))

#: OpenRouter limits page: ``limit_source`` ``openrouter_key_limit`` -- "the API key's credit limit is
#: exhausted". Message: inferred.
OPENROUTER_KEY_LIMIT_402 = RecordedError(
    402, _openrouter(402, "Key limit exceeded.", {"limit_source": "openrouter_key_limit"})
)

#: OpenRouter limits page: ``openrouter_credits`` -- "your balance cannot cover the request". Message:
#: inferred.
OPENROUTER_CREDITS_402 = RecordedError(
    402, _openrouter(402, "Insufficient credits.", {"limit_source": "openrouter_credits"})
)

#: Verbatim from the OpenRouter limits page's 402 example, with the ``Retry-After`` its errors page says
#: such a 402 may carry. The header's value: inferred.
OPENROUTER_IN_FLIGHT_402 = RecordedError(
    402,
    _openrouter(
        402,
        "This request would exceed your available credits given your current in-flight requests.",
        {"reason": "in_flight_budget_exhausted", "limit_source": "openrouter_in_flight_budget"},
    ),
    headers={"retry-after": "3"},
)

#: The same 402 without the ``Retry-After`` (errors page: "A 402 without the header is not a
#: wait-and-retry case").
OPENROUTER_IN_FLIGHT_402_NO_RETRY_AFTER = RecordedError(402, OPENROUTER_IN_FLIGHT_402.body)

#: OpenRouter limits page: ``"reason": "weight_exceeds_budget"`` with ``"limit_source":
#: "openrouter_credits"`` -- "this single request's estimated cost is larger than your whole budget, so
#: retrying will not help". The page shows no body; the message: inferred.
OPENROUTER_WEIGHT_EXCEEDS_BUDGET_402 = RecordedError(
    402,
    _openrouter(
        402,
        "This request's estimated cost exceeds your available budget.",
        {"reason": "weight_exceeds_budget", "limit_source": "openrouter_credits"},
    ),
)

#: OpenRouter errors page: "402: Your account or API key has insufficient credits", with no metadata.
OPENROUTER_BARE_402 = RecordedError(
    402, _openrouter(402, "Your account or API key has insufficient credits. Add more credits and retry the request.")
)

#: OpenRouter errors page: "403: Forbidden (... moderation flag)" with ``ModerationErrorMetadata``
#: (``reasons``, ``flagged_input``, ``provider_name``, ``model_slug``). Values: inferred.
OPENROUTER_MODERATION_403 = RecordedError(
    403,
    _openrouter(
        403,
        "Your chosen model requires moderation and your input was flagged",
        {"reasons": ["violence"], "flagged_input": "...", "provider_name": "OpenAI", "model_slug": "openai/gpt-4o"},
    ),
)

#: OpenRouter errors page: "403: Forbidden (insufficient permissions, guardrail block ...)", without
#: moderation metadata. Message: inferred.
OPENROUTER_FORBIDDEN_403 = RecordedError(403, _openrouter(403, "This model is not allowed by your guardrail."))

#: Verbatim from the OpenRouter errors page's guardrail budget 403 (no ``Retry-After``: "the wait is
#: usually hours or days"). Its ``scope`` is a workspace's default member, so the account's.
OPENROUTER_GUARDRAIL_BUDGET_403 = RecordedError(
    403,
    _openrouter(
        403,
        "Budget limit exceeded (weekly limit). Contact your org admin. $52.00 of $50.00 used; "
        "resets 2026-10-12T00:00:00.000Z.",
        {
            "limit_source": "openrouter_guardrail_budget",
            "remedy_hint": "Wait for the budget window to reset (see metadata.resets_at), "
            "or ask your org admin to raise the guardrail budget.",
            "scope": "workspace_default_member",
            "limit_usd": 50,
            "spend_usd": 52,
            "resets_at": "2026-10-12T00:00:00.000Z",
        },
    ),
)

#: The same shape for a key's own budget window: ``limit_source`` ``openrouter_key_limit``, ``scope``
#: ``key_limit`` (both named on the limits page). The values and message: inferred.
OPENROUTER_KEY_BUDGET_403 = RecordedError(
    403,
    _openrouter(
        403,
        "Budget limit exceeded (daily limit). $5.10 of $5.00 used; resets 2026-11-27T00:00:00.000Z.",
        {
            "limit_source": "openrouter_key_limit",
            "scope": "key_limit",
            "limit_usd": 5,
            "spend_usd": 5.1,
            "resets_at": "2026-11-27T00:00:00.000Z",
        },
    ),
)

#: OpenRouter errors page: ``content_policy_violation`` (403, "flagged by a content filter") and
#: ``refusal`` (403, "a model refusal as an error"), each in ``error.metadata.error_type``. The page
#: shows neither body; messages and the other metadata: inferred.
OPENROUTER_CONTENT_POLICY_403 = RecordedError(
    403,
    _openrouter(
        403,
        "The input was flagged by the provider's content filter.",
        {"error_type": "content_policy_violation", "provider_name": "Scripted"},
    ),
)
OPENROUTER_REFUSAL_403 = RecordedError(
    403,
    _openrouter(403, "The model declined to answer.", {"error_type": "refusal", "provider_name": "Scripted"}),
)

#: OpenRouter errors page: "429: You are being rate limited", and ``Retry-After`` on a 429. The header as
#: an HTTP date, and the message: inferred.
OPENROUTER_RATE_LIMIT_429 = RecordedError(
    429,
    _openrouter(429, "Rate limit exceeded", {"error_type": "rate_limit_exceeded"}),
    headers={"retry-after": "Thu, 26 Nov 2026 12:01:30 GMT"},
)

#: An OpenRouter key-management 429 with no ``Retry-After``: the limits page's ``X-RateLimit-*`` headers,
#: ``X-RateLimit-Reset`` "the reset time in milliseconds since the Unix epoch". Values: inferred (the
#: reset is 2026-11-26T12:00:45Z).
OPENROUTER_RATE_LIMIT_429_RESET_ONLY = RecordedError(
    429,
    _openrouter(429, "Rate limit exceeded", {"error_type": "rate_limit_exceeded"}),
    headers={"X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1795694445000"},
)

#: OpenRouter errors page: "502: Your chosen model is down or we received an invalid response from it".
OPENROUTER_UPSTREAM_502 = RecordedError(502, _openrouter(502, "Provider returned error"))

#: OpenRouter errors page, mid-stream errors: an SSE chunk carrying
#: ``"error":{"code":429,"message":"Rate limit exceeded","metadata":{"error_type":"rate_limit_exceeded"}}``
#: (verbatim). The page's chunk also carries a ``choices`` entry with ``finish_reason: "error"``;
#: langchain-openrouter 0.2.9 raises only for a chunk with no choices, so this one has none (inferred).
OPENROUTER_MID_STREAM_RATE_LIMIT = RecordedStream(
    b"data: "
    + json.dumps(
        {
            "id": "gen-1",
            "provider": "Scripted",
            "model": "deepseek/deepseek-chat-v3-0324",
            "created": 1790000000,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}],
        }
    ).encode()
    + b"\n\ndata: "
    + json.dumps(
        {
            "id": "gen-1",
            "provider": "Scripted",
            "model": "deepseek/deepseek-chat-v3-0324",
            "created": 1790000000,
            "object": "chat.completion.chunk",
            "error": {"code": 429, "message": "Rate limit exceeded", "metadata": {"error_type": "rate_limit_exceeded"}},
            "choices": [],
        }
    ).encode()
    + b"\n\ndata: [DONE]\n\n"
)
