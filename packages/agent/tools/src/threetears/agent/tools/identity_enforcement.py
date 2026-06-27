"""Pod-side identity-token enforcement mode for the tool server.

Defense in depth against the direct-internal-subject bypass: a tool pod subscribes
``{ns}.tools.internal.{pod_id}`` and runs the dispatched tool. If anything that can publish on
that subject reaches the pod, the registry proxy + RBAC were never consulted. So the pod RE-VERIFIES
the Hub-issued identity token itself (the same EdDSA JWS the proxy verifies) and confirms the call
context's claimed identity matches the token -- a publisher without a valid Hub-signed token, or
one that forged a different identity onto a captured token, is rejected.

This is a SEPARATE control from the registry proxy's enforcement (independently laddered), so a pod
can fail-closed even while the proxy is still in warn. The ladder mirrors the proxy:

- ``off`` (default): the token is not checked at the pod -- inert, the historical behaviour.
- ``warn``: verify; on failure log and ALLOW (observability while the fleet completes rollout).
- ``enforce``: verify; on failure REJECT the call (fail-closed).

Secure-by-default for the ROLLOUT is ``off``: enforcing before every caller mints a token would
break every dispatch, so pod enforcement turns on only when explicitly laddered up.
"""

from __future__ import annotations

import os
from enum import StrEnum

__all__ = ["ToolIdentityEnforcement", "get_tool_identity_enforcement"]

_TOOL_IDENTITY_ENFORCEMENT_ENV = "THREETEARS_TOOL_IDENTITY_ENFORCEMENT"


class ToolIdentityEnforcement(StrEnum):
    """how the tool pod treats the Hub-issued identity token on an inbound call."""

    OFF = "off"
    WARN = "warn"
    ENFORCE = "enforce"


def get_tool_identity_enforcement() -> ToolIdentityEnforcement:
    """read the pod-side identity-enforcement mode from the environment; default ``OFF``.

    env var: ``THREETEARS_TOOL_IDENTITY_ENFORCEMENT`` (``off`` | ``warn`` | ``enforce``). Unset
    defaults to ``OFF`` so the dispatch path stays inert until pod enforcement is explicitly
    enabled. A PRESENT-but-invalid value raises rather than silently defaulting -- a typo'd
    security control must fail loud at startup, never quietly leave the pod unverified.

    :return: the configured pod enforcement mode
    :rtype: ToolIdentityEnforcement
    :raises ValueError: when the env var is set to an unrecognized value
    """
    raw = os.environ.get(_TOOL_IDENTITY_ENFORCEMENT_ENV)
    if raw is None:
        return ToolIdentityEnforcement.OFF
    try:
        return ToolIdentityEnforcement(raw.strip().lower())
    except ValueError:
        valid = [member.value for member in ToolIdentityEnforcement]
        raise ValueError(
            f"invalid {_TOOL_IDENTITY_ENFORCEMENT_ENV}={raw!r}; expected one of {valid}"
        ) from None
