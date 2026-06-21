"""Authenticated encryption at rest: AES-256-GCM under an operator-supplied master key.

For the times an app must *store* a secret (not just reference it) -- e.g. scriob keeps a
per-repo SSH deploy key so any pod can pull the story. :func:`seal` turns plaintext into a
self-describing token safe to persist; :func:`open_secret` recovers it ONLY under the same
master key and rejects any tamper. The master key is itself a resolved secret reference
(see :mod:`threetears.core.security.secret_refs`) -- it never lives in the DB beside the
ciphertext.

Token format (base64url of):

    version (1 byte = 0x01) || nonce (12 bytes) || AES-256-GCM ciphertext+tag

The version byte lets the algorithm / key-derivation evolve without orphaning stored
ciphertext. The 32-byte AES key is derived from the master key via HKDF-SHA256, so the
master may be any high-entropy string (e.g. ``openssl rand -base64 32``), not a raw 32-byte
blob. The opened value is returned inside a :class:`SecretStr` so it cannot leak via repr.
"""

from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import SecretStr

__all__ = ["DecryptionError", "open_secret", "seal"]

_VERSION = 0x01
_NONCE_LEN = 12  # AES-GCM standard nonce length
_KEY_LEN = 32  # AES-256
# HKDF context info — MUST be bumped in lockstep with _VERSION if the key derivation
# ever changes, so a format bump and a KDF change can't silently skew.
_HKDF_INFO = b"threetears.core.security.encryption.v1"


class DecryptionError(Exception):
    """raised when a token cannot be opened: wrong master key, tamper, or malformed token.

    deliberately NOT a ``ValueError`` subclass carrying the value -- the message never
    contains plaintext or key material, only the structural reason.
    """


def _derive_key(master: SecretStr) -> bytes:
    """derive the 32-byte AES key from the master key via HKDF-SHA256."""
    hkdf = HKDF(algorithm=hashes.SHA256(), length=_KEY_LEN, salt=None, info=_HKDF_INFO)
    return hkdf.derive(master.get_secret_value().encode("utf-8"))


def seal(plaintext: str, key: SecretStr) -> str:
    """encrypt ``plaintext`` under ``key``, returning a base64url token safe to persist.

    A fresh random nonce is used per call, so the same plaintext seals to different tokens
    (no deterministic ciphertext that would leak equality of stored secrets).
    """
    aes_key = _derive_key(key)
    nonce = os.urandom(_NONCE_LEN)  # fresh 96-bit GCM nonce per call
    header = bytes([_VERSION])
    # The version header is authenticated as AAD (not encrypted) so it cannot be flipped
    # to steer a future multi-version open_secret toward a weaker path — the tag covers it.
    ciphertext = AESGCM(aes_key).encrypt(nonce, plaintext.encode("utf-8"), header)
    return base64.urlsafe_b64encode(header + nonce + ciphertext).decode("ascii")


def open_secret(token: str, key: SecretStr) -> SecretStr:
    """recover the plaintext sealed by :func:`seal`, inside a :class:`SecretStr`.

    :raises DecryptionError: on a malformed/short token, unknown version, wrong master
        key, or any tamper (the GCM tag fails to verify).
    """
    try:
        btoken = token.encode("ascii")
        # urlsafe alphabet, validate=True → reject out-of-alphabet chars / bad length
        # (urlsafe_b64decode has no validate kwarg; b64decode with altchars does).
        raw = base64.b64decode(btoken, altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise DecryptionError(f"malformed token: not valid base64 ({type(exc).__name__}).") from None
    # Reject non-canonical encodings so a stored token has exactly one string form (no
    # junk-appended variant that maps to the same secret) — defence beyond validate=True.
    if base64.urlsafe_b64encode(raw) != btoken:
        raise DecryptionError("malformed token: non-canonical base64.")
    if len(raw) < 1 + _NONCE_LEN + 16:  # version + nonce + at least the GCM tag
        raise DecryptionError("malformed token: too short to contain version + nonce + tag.")
    header, nonce, ciphertext = raw[0:1], raw[1 : 1 + _NONCE_LEN], raw[1 + _NONCE_LEN :]
    if header[0] != _VERSION:
        raise DecryptionError(f"unsupported token version {header[0]!r}; this build seals v{_VERSION}.")
    try:
        # header passed as AAD — must match what seal() authenticated, or the tag fails.
        plaintext = AESGCM(_derive_key(key)).decrypt(nonce, ciphertext, header)
    except InvalidTag:
        raise DecryptionError("decryption failed: wrong master key or tampered ciphertext.") from None
    return SecretStr(plaintext.decode("utf-8"))
