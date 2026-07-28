"""WebAuthn/passkey helpers: the decisions, not the ceremony.

Registration and assertion verification are the ``webauthn`` library's job and
this module does not wrap them -- a thin wrapper around a well-designed library
is a layer that adds a version to track and nothing else. What is here is the
part every deployment gets to decide for itself and therefore gets wrong
differently: the signature-counter clone check, and challenge handling.

Deliberately dependency-free. The ``webauthn`` extra exists for consumers that
want the verification library alongside this; nothing in this module imports it,
so importing this module never costs anything.
"""

from __future__ import annotations

import secrets
from typing import Final

__all__ = [
    "CHALLENGE_BYTES",
    "generate_challenge",
    "is_signature_counter_regression",
    "origin_allowed",
]

#: WebAuthn requires at least 16 bytes of challenge entropy; 32 is the common practice.
CHALLENGE_BYTES: Final[int] = 32


def generate_challenge() -> bytes:
    """Generate a registration or assertion challenge.

    Must be stored server-side and matched on the way back. A challenge the server does not
    remember issuing is a challenge an attacker chose, which defeats the whole ceremony.
    """
    return secrets.token_bytes(CHALLENGE_BYTES)


def is_signature_counter_regression(*, stored_sign_count: int, new_sign_count: int) -> bool:
    """Whether an assertion's signature counter indicates a cloned authenticator.

    A hardware authenticator increments a counter on every assertion. If one comes back at or
    below the last value seen, either the credential has been copied out of the device -- the
    thing the counter exists to detect -- or something is badly wrong. Either way, lock it.

    **The constant-zero case is exempt, and only that case.** Synced passkeys (iCloud
    Keychain, Google Password Manager) do not maintain a counter and legitimately report zero
    forever. Treating that as a clone would lock out every passkey user on the platform.

    A counter that was previously NONZERO and now reports zero is NOT exempt: a real
    authenticator's counter does not reset, so that transition is exactly the clone signal.
    This matches the reference implementation's own condition rather than inventing a looser
    rule.

    :param stored_sign_count: the credential's last known counter.
    :ptype stored_sign_count: int
    :param new_sign_count: the counter this assertion reported.
    :ptype new_sign_count: int
    :return: ``True`` if this is a clone signal and the credential must be locked.
    :rtype: bool
    """
    if stored_sign_count == 0 and new_sign_count == 0:
        return False
    return new_sign_count <= stored_sign_count


def origin_allowed(presented: str, allowed: frozenset[str]) -> bool:
    """Whether an assertion's origin is one this relying party accepts.

    Exact match against a closed set. Not a prefix or suffix test: ``https://acme.com`` must
    not match ``https://acme.com.attacker.net``, and a suffix check is precisely how that
    hole gets opened. An empty allow-list accepts nothing, which is the correct behaviour for
    an unconfigured relying party -- failing closed rather than accepting everything.
    """
    return presented in allowed
