"""Loads the proxy's assertion-signing key and mints proxy->pod assertions.

The registry proxy signs an assertion per forwarded tool call (see :mod:`proxy_assertion`). Its
Ed25519 key comes from ``secret_refs`` custody; the Hub reads the SAME secret to publish the proxy's
PUBLIC key in its JWKS (alongside the hub key), so tool pods can verify proxy assertions. The kid is
the key's RFC 7638 thumbprint, so the proxy (signing) and the Hub (publishing) always agree on it.
The private key never leaves the signer and is never logged.
"""

from __future__ import annotations

import base64
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import SecretStr

from threetears.core.security.identity_token import IdentityTokenError, build_jwks, jwk_thumbprint
from threetears.core.security.proxy_assertion import mint_proxy_assertion

__all__ = ["ProxyAssertionSigner"]

_ED25519_SEED_LEN = 32
_DEFAULT_TTL_SECONDS = 30


def _b64decode_seed(raw: str) -> bytes:
    """decode a base64 / base64url (padded or unpadded) Ed25519 32-byte private seed."""
    text = raw.strip()
    padded = text + ("=" * (-len(text) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded)
    except ValueError:
        decoded = base64.b64decode(padded)
    return decoded


class ProxyAssertionSigner:
    """signs proxy->pod assertions with the proxy's custody-held Ed25519 key.

    :param signing_key: the proxy's Ed25519 signing key (from ``secret_refs``, never generated live)
    :ptype signing_key: Ed25519PrivateKey
    :param ttl_seconds: assertion lifetime (a short window -- one forwarded hop)
    :ptype ttl_seconds: int
    """

    def __init__(self, *, signing_key: Ed25519PrivateKey, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        if ttl_seconds <= 0:
            raise IdentityTokenError("ttl_seconds must be positive")
        self._key = signing_key
        self._public_key = signing_key.public_key()
        self._kid = jwk_thumbprint(self._public_key)
        self._ttl = ttl_seconds

    @classmethod
    def from_secret(cls, secret: SecretStr, *, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> ProxyAssertionSigner:
        """build the signer from a base64(url) 32-byte Ed25519 seed held in ``secret_refs``.

        Fail closed: a malformed/wrong-length/empty secret raises rather than signing on a junk key.
        The secret value never appears in the raised reason.

        :param secret: the custody-resolved signing-key seed
        :ptype secret: SecretStr
        :return: a ready signer
        :rtype: ProxyAssertionSigner
        :raises IdentityTokenError: when the secret is not a valid Ed25519 seed
        """
        try:
            seed = _b64decode_seed(secret.get_secret_value())
            if len(seed) != _ED25519_SEED_LEN:
                raise ValueError(f"expected a {_ED25519_SEED_LEN}-byte seed, got {len(seed)}")
            key = Ed25519PrivateKey.from_private_bytes(seed)
        except ValueError as exc:
            raise IdentityTokenError(f"invalid proxy assertion signing key ({type(exc).__name__})") from None
        return cls(signing_key=key, ttl_seconds=ttl_seconds)

    @property
    def kid(self) -> str:
        """the JWKS key id (RFC 7638 thumbprint) the proxy signs + the Hub publishes under."""
        return self._kid

    def mint(
        self,
        *,
        pod_id: str,
        agent_id: str,
        customer_id: str,
        body_hash: str,
        nonce: str,
        now: int,
        user_id: str | None = None,
    ) -> str:
        """sign a proxy assertion for a forwarded call.

        :param pod_id: the target pod (``aud``)
        :ptype pod_id: str
        :param agent_id: the VERIFIED agent identity
        :ptype agent_id: str
        :param customer_id: the VERIFIED customer
        :ptype customer_id: str
        :param body_hash: canonical_call_hash of the forwarded call
        :ptype body_hash: str
        :param nonce: a unique single-use value
        :ptype nonce: str
        :param now: unix-seconds issue time (``exp`` = now + ttl)
        :ptype now: int
        :param user_id: the VERIFIED human principal, when one is in the loop
        :ptype user_id: str | None
        :return: a compact EdDSA JWS assertion
        :rtype: str
        """
        return mint_proxy_assertion(
            signing_key=self._key,
            kid=self._kid,
            pod_id=pod_id,
            agent_id=agent_id,
            customer_id=customer_id,
            body_hash=body_hash,
            nonce=nonce,
            iat=now,
            exp=now + self._ttl,
            user_id=user_id,
        )

    def public_jwks(self) -> dict[str, Any]:
        """the proxy's public key as a JWKS, for the Hub to merge into its published JWKS.

        :return: a JWKS carrying this signer's public key under its kid
        :rtype: dict[str, Any]
        """
        return build_jwks({self._kid: self._public_key})
