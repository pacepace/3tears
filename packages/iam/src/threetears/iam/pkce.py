"""PKCE (RFC 7636) for the authorization-code flow.

Structural validation only. Every malformed or unsupported input raises
:class:`PkceError`, and callers must treat that as a hard rejection -- never a
soft fallback to some other verification strategy, which is precisely the
downgrade PKCE exists to prevent.

**S256 only, always.** RFC 7636 SS4.2 also defines a ``plain`` transform where
the challenge simply *is* the verifier, unhashed, for clients that cannot do
SHA-256. Accepting it would mean an attacker who intercepts the authorization
request also holds everything needed to redeem the code, which is the entire
attack PKCE was written to stop. This module never accepts it.

The pin is written as a positive allow-list of one rather than a rejection of
``plain`` specifically: a request naming any other method fails identically, so
a future transform cannot arrive as a silently-accepted default.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from base64 import urlsafe_b64encode
from dataclasses import dataclass
from typing import Final

__all__ = [
    "S256_METHOD",
    "PkceChallenge",
    "PkceError",
    "compute_s256_challenge",
    "create_challenge",
    "generate_code_verifier",
    "validate_code_challenge_method",
    "validate_code_verifier",
    "verify",
]

#: The one ``code_challenge_method`` this module accepts.
S256_METHOD: Final[str] = "S256"

# RFC 7636 SS4.1: 43-128 characters from the unreserved charset.
_CODE_VERIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9\-._~]{43,128}$")

# 32 random bytes -> 43 unreserved characters, the RFC's minimum length at full entropy.
_VERIFIER_BYTES: Final[int] = 32


class PkceError(Exception):
    """A PKCE check failed: a malformed ``code_verifier``, an unsupported
    ``code_challenge_method``, or a verifier/challenge mismatch.

    Carries only a structural reason -- never the raw verifier or challenge -- so it is
    safe to log at a verification boundary.
    """


@dataclass(frozen=True, slots=True)
class PkceChallenge:
    """A generated verifier/challenge pair, for callers acting as an OAuth2 CLIENT.

    :ivar code_verifier: the secret to hold until token exchange and send as
        ``code_verifier``. Never put this in the authorization request.
    :ivar code_challenge: the value to send as ``code_challenge`` in the authorization
        request.
    :ivar code_challenge_method: always :data:`S256_METHOD`.
    """

    code_verifier: str
    code_challenge: str
    code_challenge_method: str = S256_METHOD


def generate_code_verifier() -> str:
    """Generate a cryptographically random ``code_verifier`` (43 unreserved characters)."""
    return secrets.token_urlsafe(_VERIFIER_BYTES)


def create_challenge() -> PkceChallenge:
    """Generate a fresh :class:`PkceChallenge` for an outbound authorization request."""
    verifier = generate_code_verifier()
    return PkceChallenge(code_verifier=verifier, code_challenge=compute_s256_challenge(verifier))


def validate_code_verifier(code_verifier: str) -> None:
    """:raises PkceError: ``code_verifier`` is not 43-128 characters of the RFC 7636 SS4.1
    unreserved charset (``[A-Za-z0-9-._~]``)."""
    if not _CODE_VERIFIER_RE.fullmatch(code_verifier):
        raise PkceError("code_verifier must be 43-128 characters from the RFC 7636 unreserved charset.")


def validate_code_challenge_method(code_challenge_method: str) -> None:
    """:raises PkceError: anything other than ``"S256"`` -- notably ``"plain"``, which this
    module never accepts (module docstring)."""
    if code_challenge_method != S256_METHOD:
        raise PkceError(f"unsupported code_challenge_method {code_challenge_method!r}; only 'S256' is accepted.")


def compute_s256_challenge(code_verifier: str) -> str:
    """RFC 7636 SS4.2: ``BASE64URL(SHA256(ASCII(code_verifier)))``, unpadded.

    :param code_verifier: the verifier to derive a challenge from.
    :ptype code_verifier: str
    :return: the unpadded base64url SHA-256 digest.
    :rtype: str
    """
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def verify(*, code_verifier: str, code_challenge: str, code_challenge_method: str) -> None:
    """Verify a presented ``code_verifier`` against the stored ``code_challenge`` (RFC 7636 SS4.6).

    Checks run in order -- method pin, then structural validation, then the comparison -- so
    a malformed verifier is rejected before it is hashed.

    :param code_verifier: the verifier presented at token exchange.
    :ptype code_verifier: str
    :param code_challenge: the challenge stored when the authorization code was issued.
    :ptype code_challenge: str
    :param code_challenge_method: the method the challenge was issued under; must be ``"S256"``.
    :ptype code_challenge_method: str
    :raises PkceError: on an unsupported method, a malformed verifier, or a mismatch.
    """
    validate_code_challenge_method(code_challenge_method)
    validate_code_verifier(code_verifier)
    if not secrets.compare_digest(compute_s256_challenge(code_verifier), code_challenge):
        raise PkceError("code_verifier does not match the stored code_challenge.")
