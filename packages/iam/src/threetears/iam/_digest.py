"""The one place this package turns a high-entropy secret into its stored form.

Two public functions need it -- :func:`threetears.iam.stores.base.hash_ticket` and
:func:`threetears.iam.apikeys.hash_api_key_secret` -- and they had the same body written
out twice. They keep separate public names because they are separate contracts with
separate call sites, and because a future decision to pepper one is a decision about that
one; what they share is the digest, and sharing it means a change to how this package
hashes cannot land in one of them and miss the other.

**SHA-256, not a password KDF, and that is deliberate for both.** Every value passed here
is 256 bits of generated randomness, so there is no dictionary for an attacker to run and
nothing for a slow KDF to slow down. What the stores need instead is an equality lookup,
which a per-candidate KDF run would make impossible. A user-chosen password is the opposite
case in every respect and goes through :mod:`threetears.iam.passwords`, which uses
argon2id. Nothing in this module is appropriate for one.
"""

from __future__ import annotations

import hashlib

__all__ = ["sha256_hex"]


def sha256_hex(secret: str) -> str:
    """SHA-256 hex digest of ``secret``'s UTF-8 bytes.

    :param secret: the raw, high-entropy secret.
    :ptype secret: str
    :return: the lowercase hex digest.
    :rtype: str
    """
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()
