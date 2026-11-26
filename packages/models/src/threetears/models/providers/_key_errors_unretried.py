"""Telling a provider SDK not to retry a refusal that belongs to the key, not the moment.

A key pool moves to another key on a 401, 402, 403 or 429 (KP-H4). The Anthropic and OpenAI SDKs
retry a 429 on the same key first, sleeping on its ``retry-after``, so a pooled call would wait out a
rate-limited key before it could fail over. Both SDKs obey a response's ``x-should-retry`` header
ahead of their own status rules (``_should_retry`` in each ``_base_client``; the Claude API errors
page, "Retries"), so a response marked ``false`` reaches the caller on its first occurrence while a
5xx or a dropped connection keeps the SDK's retries. Each provider module sets the mark through its
SDK's own public hook: a middleware for Anthropic, an HTTP client event hook for OpenAI.
"""

from __future__ import annotations

from collections.abc import MutableMapping

__all__ = ["KEY_MOVING_STATUSES", "mark_unretried"]

#: The statuses a key pool moves to another key on: revoked, out of credit, forbidden, rate-limited.
KEY_MOVING_STATUSES = frozenset({401, 402, 403, 429})

#: The response header both Stainless SDKs read before their own retry rules.
_SHOULD_RETRY = "x-should-retry"


def mark_unretried(status_code: int, headers: MutableMapping[str, str]) -> None:
    """Mark a key-moving response so the SDK raises it at once instead of retrying it on the same key.

    :param status_code: the response's HTTP status
    :ptype status_code: int
    :param headers: the response's headers, changed in place
    :ptype headers: MutableMapping[str, str]
    """
    if status_code in KEY_MOVING_STATUSES:
        headers[_SHOULD_RETRY] = "false"
