"""Telling a provider SDK not to retry a refusal that belongs to the key, not the moment.

A key pool moves to another key on a refusal that is the key's: revoked, out of credit, forbidden, or a
quota a 429 reports (KP-H4). The Anthropic and OpenAI SDKs retry every 429 on the same key, sleeping on
its ``retry-after`` however long it is, so a pooled call would wait out a used-up quota before it could
fail over. A 429 that is only a rate limit is the SDK's to retry, with the caller's ``max_retries``:
a rate limit means retry (owner, 2026-11-26).

Both SDKs obey a response's ``x-should-retry`` header ahead of their own status rules (``_should_retry``
in each ``_base_client``; the Claude API errors page, "Retries"), so a response marked ``false`` reaches
the caller on its first occurrence while a rate limit, a 5xx or a dropped connection keeps the SDK's
retries. Each provider module sets the mark through its SDK's own public hook -- a middleware for
Anthropic, an HTTP client event hook for OpenAI -- and asks an :data:`UnretriedRule` which responses
to mark. The rule is the key pool's classifier's (``KeyErrorClassifier.refuses_retry``), so the hook
and the verdict on the error that reaches the pool read a response alike.

A 429's body says whether it is a quota, so the hook reads the body of a 429, and of nothing else: a
2xx may be a stream the caller is still to read. The SDK reads the body it raises from afterwards,
and a read body is cached (``httpx.Response.read``), so the error still carries it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping

__all__ = ["BODY_READ_STATUS", "UnretriedRule", "mark_unretried"]

#: Which responses the SDK must raise at once: ``(status, headers, body) -> bool``, where ``body`` is
#: the response's body for a 429 and ``None`` for every other status.
UnretriedRule = Callable[[int, Mapping[str, str], bytes | None], bool]

#: The one status whose body the hook reads: a rate limit and a quota share it, and only the body
#: tells them apart.
BODY_READ_STATUS = 429

#: The response header both Stainless SDKs read before their own retry rules.
_SHOULD_RETRY = "x-should-retry"


def mark_unretried(
    rule: UnretriedRule, status_code: int, headers: MutableMapping[str, str], body: bytes | None
) -> None:
    """Mark a response the rule says is the key's, so the SDK raises it at once instead of retrying it.

    :param rule: which responses to mark
    :ptype rule: UnretriedRule
    :param status_code: the response's HTTP status
    :ptype status_code: int
    :param headers: the response's headers, changed in place
    :ptype headers: MutableMapping[str, str]
    :param body: the response's body when its status is :data:`BODY_READ_STATUS`, else ``None``
    :ptype body: bytes | None
    """
    if rule(status_code, headers, body):
        headers[_SHOULD_RETRY] = "false"
