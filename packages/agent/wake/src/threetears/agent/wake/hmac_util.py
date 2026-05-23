"""Shared HMAC helpers for the webhook receiver + adapter.

The canonical :func:`verify_generic_hmac_sha256` lives here so both
the wake-side adapter (:mod:`threetears.agent.wake.webhook_adapter`)
and the channels-side receiver (:mod:`threetears.channels.webhook`)
share a single implementation. Two parallel inline implementations
were the root of the Critic finding on shard-06: identical algorithm,
double surface area for future divergence.

Placement: ``threetears.agent.wake`` (not channels) because the
package edge runs ``channels -> agent-wake`` (one-way; the channels
``webhook`` extra depends on ``3tears-agent-wake>=0.9.0``). A shared
module on the agent-wake side keeps that direction intact.

Verifier signature
------------------

:func:`verify_generic_hmac_sha256` matches the :data:`~threetears.channels.webhook.Verifier`
protocol:

    ``(secret_bytes, payload_bytes, signature_value) -> bool``

``signature_value`` is the RAW header value (e.g. ``"sha256=<hex>"``)
the receiver already extracted using the configured signature header
name. Verifiers do NOT receive the full headers dict so vendor schemes
that use a non-default header name (e.g. ``X-Hub-Signature-256``)
work uniformly -- the receiver does the header-name resolution once
and hands the verifier just the value.
"""

from __future__ import annotations

import hmac
from hashlib import sha256

__all__ = [
    "compute_generic_hmac_sha256_signature",
    "verify_generic_hmac_sha256",
]


_SIGNATURE_PREFIX = "sha256="


def compute_generic_hmac_sha256_signature(secret: bytes, payload: bytes) -> str:
    """Compute the ``sha256=<hex>`` signature for a payload.

    Used by the receiver fallback path (when ``pre_verified=False`` so
    the adapter still runs the default HMAC inline) and by tests that
    need a known-good signature to POST against the receiver.

    :param secret: HMAC key bytes (typically ``secret_str.encode('utf-8')``)
    :ptype secret: bytes
    :param payload: raw bytes to sign
    :ptype payload: bytes
    :return: ``sha256=`` prefix followed by the lowercase hex digest
    :rtype: str
    """
    return _SIGNATURE_PREFIX + hmac.new(secret, payload, sha256).hexdigest()


def verify_generic_hmac_sha256(
    secret: bytes,
    payload: bytes,
    signature_value: str,
) -> bool:
    """Verify an HMAC-SHA256 signature in the platform-default format.

    The default scheme uses ``sha256=<hex>`` as the signature header
    value. The HMAC is computed over the raw request bytes with
    :func:`hmac.compare_digest` for constant-time comparison
    (timing-attack defence). Returns ``False`` for any structural
    problem (empty value, wrong prefix, length mismatch) rather than
    raising; the receiver maps the boolean to a 403.

    :param secret: subscription's decrypted HMAC secret as bytes
    :ptype secret: bytes
    :param payload: raw HTTP body to verify against
    :ptype payload: bytes
    :param signature_value: header value the receiver extracted (the
        raw string, e.g. ``"sha256=abc..."``); empty / missing values
        return ``False`` rather than raising
    :ptype signature_value: str
    :return: ``True`` when the computed HMAC matches the header
        verbatim, ``False`` otherwise
    :rtype: bool
    """
    if not signature_value or not signature_value.startswith(_SIGNATURE_PREFIX):
        return False
    expected = compute_generic_hmac_sha256_signature(secret, payload)
    return hmac.compare_digest(expected, signature_value)
