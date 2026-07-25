"""The storage Protocols, and the hashing every ticket store shares.

See :mod:`threetears.iam.stores` for why these are Protocols rather than a
schema this package owns.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final, Protocol, runtime_checkable

__all__ = [
    "AttemptLimiter",
    "AttemptWindow",
    "SingleUseTicketStore",
    "StateStore",
    "TicketIssue",
    "hash_ticket",
    "new_ticket_secret",
]

#: 32 bytes of entropy. A reset ticket or an OAuth state is guessable-once material; at this
#: width, guessing is not an attack anyone can run.
_TICKET_BYTES: Final[int] = 32


def new_ticket_secret() -> str:
    """Generate a single-use ticket secret -- a reset token, handoff code, or state value."""
    return secrets.token_urlsafe(_TICKET_BYTES)


def hash_ticket(secret: str) -> str:
    """SHA-256 hex digest of a ticket secret -- the form a store keeps.

    SHA-256 rather than argon2id for the same reason as an API key: the secret is 256 bits
    of generated randomness, so there is no dictionary to slow an attacker down against, and
    the store needs an equality lookup rather than a per-candidate KDF run.
    """
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TicketIssue:
    """A freshly issued single-use ticket.

    :ivar secret: the value to hand to the holder. This is the ONLY time it exists in
        recoverable form -- the store keeps the hash.
    :ivar hashed: the stored form, exposed so a caller that keeps its own index (an audit
        row, say) can reference the ticket without holding the secret.
    """

    secret: str
    hashed: str


@dataclass(frozen=True, slots=True)
class AttemptWindow:
    """A rate-limit verdict.

    :ivar count: failures recorded in the current window.
    :ivar limited: whether the subject is currently locked out.
    :ivar retry_after: how long until the window rolls over. ``None`` when not limited.
    """

    count: int
    limited: bool
    retry_after: timedelta | None = None


@runtime_checkable
class SingleUseTicketStore(Protocol):
    """Issue and redeem short-lived, single-use secrets.

    Backs password-reset tickets, token handoffs, email-change confirmations, authorization
    codes, and anything else that must work exactly once.
    """

    async def issue(self, payload: Mapping[str, Any], *, ttl: timedelta) -> TicketIssue:
        """Store ``payload`` against a fresh secret and return it.

        :param payload: JSON-serializable data to return on redemption.
        :ptype payload: Mapping[str, Any]
        :param ttl: how long the ticket stays redeemable.
        :ptype ttl: timedelta
        :return: the issued ticket.
        :rtype: TicketIssue
        """
        ...

    async def redeem(self, secret: str) -> Mapping[str, Any] | None:
        """Atomically consume ``secret`` and return its payload.

        Returns ``None`` if the ticket is unknown, expired, or ALREADY REDEEMED. Concurrent
        redemptions of one ticket must produce exactly one non-``None`` result.
        """
        ...


@runtime_checkable
class StateStore(Protocol):
    """Park a value under a caller-chosen key for a bounded time.

    Distinct from :class:`SingleUseTicketStore` in that the key is supplied rather than
    generated -- for correlating the two legs of a redirect flow, where the key IS the
    protocol's ``state`` parameter.
    """

    async def put(self, key: str, payload: Mapping[str, Any], *, ttl: timedelta) -> None:
        """Store ``payload`` under ``key`` for ``ttl``."""
        ...

    async def take(self, key: str) -> Mapping[str, Any] | None:
        """Atomically remove and return the value under ``key``, or ``None``.

        Removing on read is what makes an OAuth ``state`` single-use, which is what stops a
        captured callback URL from being replayed.
        """
        ...


@runtime_checkable
class AttemptLimiter(Protocol):
    """Count failures against a key and report when it is over the line.

    The key is opaque -- a hashed username, a client IP, a tenant. Implementations must not
    assume it is any of those, and callers should hash anything identifying before it
    becomes a key, since keys are far more likely than values to end up in an operator's
    terminal.
    """

    async def record_failure(self, key: str) -> AttemptWindow:
        """Record one failure against ``key`` and return the resulting window."""
        ...

    async def check(self, key: str) -> AttemptWindow:
        """Report ``key``'s current window WITHOUT recording anything."""
        ...

    async def clear(self, key: str) -> None:
        """Reset ``key``'s counter -- called after a successful authentication."""
        ...
