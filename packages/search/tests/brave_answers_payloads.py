"""Brave Answers streams and error bodies: the live captures, and synthetic edges in their shape.

The captures are the fixture (AR-I2): what Brave really sent on 2026-10-24,
read through :func:`packages.search.tests.recorded_streams.recording`. The facts each pin
relies on are stated here once, beside the capture they describe, so a test
that reads "eighty-three" reads it from the capture's own record.

The synthetic builders write streams in exactly the captured framing --
``data: {chunk}`` events with ``choices[0].delta.content``, a final event with
an OpenAI ``usage`` object, ``data: [DONE]`` -- for the edges no capture
exhibits: a tag split mid-character, an unknown tag, a missing usage report.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from packages.search.tests.recorded_streams import recording

__all__ = [
    "CITATION_FAVICON",
    "INVALID_TOKEN_BODY",
    "NON_ASCII",
    "NON_ASCII_CITATIONS",
    "NON_ASCII_POINT",
    "NON_ASCII_UTF8_LENGTH",
    "RATE_LIMITED_BODY",
    "SINGLE",
    "SINGLE_CITATIONS",
    "SINGLE_POINTS",
    "SINGLE_TEXT_LENGTH",
    "SINGLE_TOTAL_COST",
    "USAGE",
    "VALIDATION_BODY",
    "citation",
    "envelope",
    "stream",
    "usage_tag",
]

#: the twelve-citation capybara answer.
SINGLE = recording("brave_answers_single.sse")
#: citations in :data:`SINGLE`, numbered 1 to 12.
SINGLE_CITATIONS = 12
#: the two points :data:`SINGLE`'s citations share: ten at 89, two at 311.
SINGLE_POINTS = (89,) * 10 + (311,) * 2
#: :data:`SINGLE`'s stripped text, in code points (ASCII, so in bytes too).
SINGLE_TEXT_LENGTH = 311
#: :data:`SINGLE`'s ``X-Request-Total-Cost``, exactly as the provider wrote it.
SINGLE_TOTAL_COST = Decimal("0.05675000000000001")

#: the nine-citation Schrödinger answer, whose text is not ASCII.
NON_ASCII = recording("brave_answers_non_ascii.sse")
#: citations in :data:`NON_ASCII`.
NON_ASCII_CITATIONS = 9
#: the one point every :data:`NON_ASCII` citation shares: the stripped text's
#: length in code points.
NON_ASCII_POINT = 83
#: the same text's length in UTF-8 bytes -- what a byte-unit reader would get.
NON_ASCII_UTF8_LENGTH = 94

#: the live 422 for a key Brave does not recognise.
INVALID_TOKEN_BODY = recording("brave_answers_invalid_token.json")
#: the live 422 for a request body Brave could not validate.
VALIDATION_BODY = recording("brave_answers_validation.json")

#: a favicon URL in the captured shape.
CITATION_FAVICON = "https://imgs.search.brave.com/favicon.png"

#: a usage report in the captured key set, costs as JSON numbers.
USAGE: dict[str, Any] = {
    "X-Request-Requests": 1,
    "X-Request-Queries": 2,
    "X-Request-Tokens-In": 1234,
    "X-Request-Tokens-Out": 300,
    "X-Request-Requests-Cost": 0.0,
    "X-Request-Queries-Cost": 0.008,
    "X-Request-Tokens-In-Cost": 0.00617,
    "X-Request-Tokens-Out-Cost": 0.0015,
    "X-Request-Total-Cost": 0.01567,
}


def envelope(code: str, status: int, detail: str = "refused") -> bytes:
    """An error body in Brave's captured ``ErrorResponse`` envelope.

    :param code: ``error.code``
    :ptype code: str
    :param status: ``error.status``
    :ptype status: int
    :param detail: ``error.detail``
    :ptype detail: str
    :return: the JSON body
    :rtype: bytes
    """
    error = {"code": code, "detail": detail, "meta": {}, "status": status}
    return json.dumps({"type": "ErrorResponse", "error": error}).encode()


#: a 429 in the envelope, with the published code (no 429 was captured).
RATE_LIMITED_BODY = envelope("RATE_LIMITED", 429, "Request rate limit exceeded for plan.")


def citation(number: int, point: int, url: str | None = None, **extra: object) -> str:
    """One ``<citation>`` tag in the captured shape, a zero-width point.

    :param number: the citation's ``number``
    :ptype number: int
    :param point: ``start_index`` and ``end_index``
    :ptype point: int
    :param url: the cited URL; one is made from ``number`` when None
    :ptype url: str | None
    :param extra: fields to add or override (``snippet``, ``start_index``)
    :ptype extra: object
    :return: the tag
    :rtype: str
    """
    payload: dict[str, object] = {
        "start_index": point,
        "end_index": point,
        "number": number,
        "url": url or f"https://example.org/source-{number}",
        "favicon": CITATION_FAVICON,
        "snippet": f"page text of source {number}",
    }
    payload.update(extra)
    return f"<citation>{json.dumps(payload)}</citation>"


def usage_tag(report: dict[str, Any] | None = None) -> str:
    """The ``<usage>`` tag that closes a captured answer.

    :param report: the keys; :data:`USAGE` when None
    :ptype report: dict[str, Any] | None
    :return: the tag
    :rtype: str
    """
    return f"<usage>{json.dumps(USAGE if report is None else report)}</usage>"


def stream(*contents: str, final_usage: bool = True, done: bool = True) -> bytes:
    """A stream in the captured framing, one event per content piece.

    :param contents: each event's ``delta.content``, in order
    :ptype contents: str
    :param final_usage: whether the last event carries the OpenAI ``usage`` object
    :ptype final_usage: bool
    :param done: whether the stream ends ``data: [DONE]``
    :ptype done: bool
    :return: the SSE body
    :rtype: bytes
    """

    def event(content: str, finish: str | None, usage: dict[str, int] | None) -> str:
        chunk = {
            "model": "brave-pro",
            "choices": [{"delta": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "id": "synthetic-1",
            "object": "chat.completion.chunk",
            "usage": usage,
        }
        return f"data: {json.dumps(chunk)}\n\n"

    usage = {"completion_tokens": 300, "prompt_tokens": 1234, "total_tokens": 1534} if final_usage else None
    body = "".join(event(content, None, None) for content in contents) + event("", "stop", usage)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode()
