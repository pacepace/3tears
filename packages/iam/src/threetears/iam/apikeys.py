"""API-key secret generation, hashing, and presentation.

**SHA-256, not argon2id, and that is deliberate.** An API key is a generated
256-bit random token, not a human-chosen password: there is no dictionary to
attack and no low-entropy guess worth making, so argon2id's deliberate slowness
would buy nothing. What it would cost is the lookup: a fast deterministic hash
lets a verifier resolve a key with ``WHERE key_hash = $1``, while a salted KDF
would force it to load every stored key and run the KDF against each one on
every request. Do not "upgrade" this to argon2id.

The generated secret carries a short cleartext prefix so an operator can
identify a key in a log or an admin list without the full secret existing
anywhere it could leak. The prefix is derived from the secret rather than
stored separately -- one value, no chance of the two drifting.
"""

from __future__ import annotations

import secrets
from typing import Final

from threetears.iam._digest import sha256_hex

__all__ = [
    "DEFAULT_KEY_PREFIX",
    "KEY_PREFIX_LEN",
    "generate_api_key_secret",
    "hash_api_key_secret",
    "key_prefix",
    "secrets_match",
]

#: The marker every generated secret starts with. Makes a leaked key greppable in logs and
#: recognizable to secret scanners.
DEFAULT_KEY_PREFIX: Final[str] = "tti_"

#: How many leading characters of the secret are retained in cleartext for identification.
#: Long enough to disambiguate keys in an admin list, far too short to help an attacker.
KEY_PREFIX_LEN: Final[int] = 8

#: 32 bytes = 256 bits of entropy.
_SECRET_BYTES: Final[int] = 32


def generate_api_key_secret(*, prefix: str = DEFAULT_KEY_PREFIX) -> str:
    """Generate a new high-entropy raw API-key secret.

    Shown to the caller exactly once, at mint or rotate time. Only its hash is persisted --
    the raw value is not recoverable afterwards, by design.

    :param prefix: the cleartext marker to prepend.
    :ptype prefix: str
    :return: the raw secret.
    :rtype: str
    """
    return f"{prefix}{secrets.token_urlsafe(_SECRET_BYTES)}"


def hash_api_key_secret(raw_secret: str) -> str:
    """SHA-256 hex digest of a raw API-key secret -- the value to store and look up by.

    Shares its digest with :func:`threetears.iam.stores.base.hash_ticket` via
    :func:`threetears.iam._digest.sha256_hex`; the names stay separate because the two are
    separate contracts, but the hashing is one implementation.
    """
    return sha256_hex(raw_secret)


def key_prefix(raw_secret: str, *, length: int = KEY_PREFIX_LEN) -> str:
    """The cleartext identifying prefix of a raw secret, for display and logging.

    :param raw_secret: the raw secret to derive a prefix from.
    :ptype raw_secret: str
    :param length: how many leading characters to keep.
    :ptype length: int
    :return: the prefix.
    :rtype: str
    """
    return raw_secret[:length]


def secrets_match(presented_hash: str, stored_hash: str) -> bool:
    """Compare two API-key hashes in constant time.

    Hash equality is not secret-dependent in the way a password comparison is -- an attacker
    who can supply a candidate already knows its hash -- but a timing-safe compare costs
    nothing here and removes the question entirely.
    """
    return secrets.compare_digest(presented_hash, stored_hash)
