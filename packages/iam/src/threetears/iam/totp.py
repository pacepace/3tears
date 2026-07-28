"""TOTP (RFC 6238) second factors, and their one-time backup codes.

Pure functions and sealing. No database, no transport: enrolment records,
credential lookup, and the hash write-back are the caller's, because every
service stores them differently and none of that is protocol.

**The seed is a bearer secret and is sealed at rest.** Unlike a password hash,
a TOTP seed is symmetric -- anyone holding it can generate valid codes forever,
so a database dump of raw seeds is a silent, permanent bypass of the second
factor for every enrolled user. Sealing uses
:func:`threetears.core.security.seal`, which authenticates a version header as
AAD and returns typed errors, rather than a hand-rolled AES-GCM wrapper.

**Backup codes are hashed, not sealed.** They are high-entropy generated tokens
rather than human-chosen secrets, so SHA-256 is the right tool for the same
reason it is for API keys: there is no dictionary to slow an attacker against,
and the verifier needs an equality lookup. Comparison is constant-time per
candidate.

**A one-step skew window each side.** Roughly thirty seconds either way, which
is what authenticator apps and verifiers conventionally allow. It is not a
meaningfully wide replay window, and narrowing it further mostly produces
support tickets from users with drifting phone clocks.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Final

import pyotp
from pydantic import SecretStr

from threetears.core.security import DecryptionError, open_secret, seal

__all__ = [
    "BACKUP_CODE_COUNT",
    "DecryptionError",
    "generate_backup_codes",
    "generate_seed",
    "hash_backup_code",
    "match_backup_code",
    "provisioning_uri",
    "seal_seed",
    "unseal_seed",
    "verify_code",
]

#: How many one-time backup codes an enrolment issues.
BACKUP_CODE_COUNT: Final[int] = 10

#: Steps of clock skew tolerated either side of now (one step is 30 seconds).
_VALID_WINDOW: Final[int] = 1


def generate_seed() -> str:
    """Generate a fresh base32 TOTP secret (160 bits, RFC 4226's recommended default)."""
    return pyotp.random_base32()


def provisioning_uri(secret_b32: str, *, account_name: str, issuer_name: str) -> str:
    """The ``otpauth://`` URI an authenticator app scans as a QR code or accepts by hand.

    :param secret_b32: the base32 seed.
    :ptype secret_b32: str
    :param account_name: what the app shows as the account, usually an email address.
    :ptype account_name: str
    :param issuer_name: what the app shows as the issuer. Required and not defaulted: it is
        what tells a user WHICH service a code belongs to when they have thirty of them, and
        a generic default makes every deployment look identical in the list.
    :ptype issuer_name: str
    :return: the provisioning URI.
    :rtype: str
    """
    return pyotp.TOTP(secret_b32).provisioning_uri(name=account_name, issuer_name=issuer_name)


def verify_code(secret_b32: str, code: str) -> bool:
    """Verify a live TOTP code.

    Never raises: a malformed code -- non-numeric, wrong length, empty -- is an
    authentication failure, not an exception, so no rejection reason is distinguishable by
    exception shape.
    """
    if not code:
        return False
    try:
        return bool(pyotp.TOTP(secret_b32).verify(code, valid_window=_VALID_WINDOW))
    except TypeError, ValueError:
        return False


def generate_backup_codes(count: int = BACKUP_CODE_COUNT) -> list[str]:
    """Generate one-time backup codes, shown to the user exactly once at enrolment.

    Only their hashes are ever persisted. The hyphenated shape is for humans copying them off
    a screen under stress, which is the only circumstance in which they are ever used.
    """
    return [f"{secrets.token_hex(4)}-{secrets.token_hex(4)}" for _ in range(count)]


def hash_backup_code(code: str) -> str:
    """SHA-256 hex digest of a backup code, whitespace-stripped.

    Stripping matters: these are read off a screen and pasted, and a trailing space that
    silently invalidates a user's last backup code is a lockout, not a security control.
    """
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()


def match_backup_code(code: str, hashes: tuple[str, ...]) -> str | None:
    """The stored hash matching ``code``, or ``None``.

    Returns the matched hash rather than a boolean so the caller can remove exactly that
    entry -- a backup code is single-use, and "which one was it" is needed to burn it.

    Comparison is constant-time per candidate. The loop's LENGTH is not hidden, which is
    fine: how many backup codes remain is not a secret worth protecting, while the codes
    themselves are.
    """
    target = hash_backup_code(code)
    matched: str | None = None
    for stored_hash in hashes:
        if secrets.compare_digest(stored_hash, target):
            matched = stored_hash
    return matched


def seal_seed(seed_b32: str, key: SecretStr) -> str:
    """Seal a TOTP seed for storage. See the module docstring on why this is not optional."""
    return seal(seed_b32, key)


def unseal_seed(sealed: str, key: SecretStr) -> str:
    """Recover a sealed TOTP seed.

    :raises DecryptionError: the token is malformed, of an unknown version, sealed under a
        different key, or tampered with.
    """
    return open_secret(sealed, key).get_secret_value()
