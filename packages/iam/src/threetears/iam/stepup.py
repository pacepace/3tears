"""Step-up re-authentication: reconfirm credentials before sensitive actions.

Distinct from session lifetime. A session can be perfectly fresh by inactivity
and absolute-lifetime measures and still fail a step-up check, because what
step-up asks is not "is this session still valid" but "how long ago did this
human actually prove who they are". An attacker with a stolen but live session
passes every session check and fails this one.

The category taxonomy is :class:`~threetears.agent.acl.ImpersonationCategory`,
imported rather than redeclared. The same six strings previously existed in two
repositories, each carrying a comment explaining that the copy was there only
because no import path connected them. This package is that path.

The window and the authentication time both ride on the token, so the check
costs nothing at the call site -- no policy lookup, no second round trip. That
denormalization is why it is cheap enough to apply everywhere it should be.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from threetears.agent.acl import ImpersonationCategory

__all__ = ["StepUpRequiredError", "SteppableClaims", "is_step_up_satisfied", "require_fresh_auth"]


@runtime_checkable
class SteppableClaims(Protocol):
    """The two claims a step-up check reads.

    Structural rather than a concrete type, so this module does not depend on any particular
    claims class -- :class:`~threetears.iam.tokens.SessionClaims` satisfies it, and so does a
    caller's own claims object that happens to carry the same two fields.
    """

    @property
    def auth_time(self) -> int:
        """Unix seconds at which the subject last actually authenticated."""
        ...

    @property
    def step_up_window(self) -> int:
        """Seconds after ``auth_time`` during which sensitive actions stay permitted."""
        ...


class StepUpRequiredError(Exception):
    """Authentication is too stale for the requested sensitive action.

    A DISTINCT rejection shape from a generic auth failure, on purpose: the caller needs to
    tell the user "confirm your password again", not "access denied". Folding this into an
    ordinary denial leaves someone stuck at a wall with no indication that a five-second
    action would clear it.

    :ivar category: the category that triggered the check.
    """

    def __init__(self, category: ImpersonationCategory) -> None:
        super().__init__(f"step-up re-authentication required for {category.value}.")
        self.category = category


def is_step_up_satisfied(*, auth_time: int, step_up_window: int, now: int | None = None) -> bool:
    """Whether ``auth_time`` is still within ``step_up_window`` seconds of ``now``.

    Pure, no I/O, injectable clock -- this is the function a boundary test exercises at
    exactly the window edge and one second past it, without needing a real token.

    A negative elapsed time (a token minted slightly in the future, by clock skew) satisfies
    the check. Rejecting it would turn a benign few seconds of NTP drift into a login
    failure, and a subject who authenticated in the future has, at worst, authenticated.

    :param auth_time: the ``auth_time`` claim, unix seconds.
    :ptype auth_time: int
    :param step_up_window: the ``step_up_window`` claim, seconds.
    :ptype step_up_window: int
    :param now: injectable clock; defaults to the current time.
    :ptype now: int | None
    :return: ``True`` when ``now - auth_time <= step_up_window``.
    :rtype: bool
    """
    current = now if now is not None else int(time.time())
    return (current - auth_time) <= step_up_window


def require_fresh_auth(
    claims: SteppableClaims,
    *,
    category: ImpersonationCategory,
    now: int | None = None,
) -> None:
    """Enforce step-up freshness for ``category`` against the claims' own window.

    :param claims: the caller's verified claims.
    :ptype claims: SteppableClaims
    :param category: the sensitive-action category being guarded.
    :ptype category: ImpersonationCategory
    :param now: injectable clock.
    :ptype now: int | None
    :raises StepUpRequiredError: the authentication is stale for this category.
    """
    if not is_step_up_satisfied(auth_time=claims.auth_time, step_up_window=claims.step_up_window, now=now):
        raise StepUpRequiredError(category)
