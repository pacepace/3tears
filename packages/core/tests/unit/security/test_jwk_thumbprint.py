"""tests for :func:`jwk_thumbprint`: RFC 7638 thumbprints for Ed25519 (OKP) and EC public keys.

The contract this pins:

- the Ed25519 (OKP) path's output is BYTE-IDENTICAL to the pre-EC-support implementation --
  verified against a fixed key vector AND an independent re-derivation of the original
  ``{crv, kty, x}``-only formula, so existing callers (Hub token ``cnf``, proof-of-possession key
  binding) see no behavior change from adding EC support;
- the EC path uses RFC 7638 SS3.2's EC required-member set (``crv, kty, x, y`` -- an EC public
  key carries both coordinates, unlike OKP's single ``x``), is deterministic, and is stable-length
  (SHA-256 -> 43-char unpadded base64url) across the P-256/P-384/P-521 curves this module accepts
  via :class:`EllipticCurvePublicKey`;
- distinct EC keys produce distinct thumbprints (not a constant / degenerate hash).
"""

from __future__ import annotations

import base64
import hashlib
import json

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from hypothesis import assume, given
from hypothesis import strategies as st
from jwt.algorithms import ECAlgorithm, OKPAlgorithm

from threetears.core.security.identity_token import jwk_thumbprint

# (curve, group order) -- bounds the private-scalar draw per curve so derive_private_key gets a
# value in the valid [1, order) range for hypothesis to control directly (real per-example
# variation, not just re-randomizing on every call).
_CURVES_WITH_ORDER = [
    (ec.SECP256R1(), 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551),
    (
        ec.SECP384R1(),
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFC7634D81F4372DDF581A0DB248B0A77AECEC196ACCC52973,
    ),
    (
        ec.SECP521R1(),
        0x01FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFA51868783BF2F966B7FCC0148F709A5D03BB5C9B8899C47AEBB6FB71E91386409,
    ),
]


def test_ed25519_thumbprint_matches_pinned_vector() -> None:
    # a fixed, deterministic Ed25519 key -> a thumbprint literal captured from the
    # PRE-EC-support implementation (see the identical computation this file also performs
    # independently below). A future change to the OKP branch that alters this value is a
    # regression, not an improvement.
    priv = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    assert jwk_thumbprint(priv.public_key()) == "1IG2tMH7J2wbJZnOf8LJzQitKf7LMvoAElsuDMVM54Y"


def test_ed25519_thumbprint_matches_independent_rfc7638_computation() -> None:
    # re-derive the thumbprint via the ORIGINAL {crv, kty, x}-only OKP formula, independently of
    # jwk_thumbprint's own implementation, and confirm they agree for a range of keys -- proof the
    # EC extension did not change the Ed25519 code path's output.
    for seed in range(5):
        priv = Ed25519PrivateKey.from_private_bytes(bytes([seed] * 32))
        pub = priv.public_key()
        jwk = OKPAlgorithm.to_jwk(pub, as_dict=True)
        canonical = json.dumps(
            {"crv": jwk["crv"], "kty": jwk["kty"], "x": jwk["x"]}, separators=(",", ":"), sort_keys=True
        )
        expected = str(
            base64.urlsafe_b64encode(hashlib.sha256(canonical.encode("utf-8")).digest()).rstrip(b"="), "ascii"
        )
        assert jwk_thumbprint(pub) == expected


def test_ec_thumbprint_uses_the_rfc7638_ec_member_set() -> None:
    # independent re-derivation using the EC {crv, kty, x, y} formula (RFC 7638 SS3.2) -- proves
    # jwk_thumbprint's EC branch uses the correct (4-member, not 3-member) required set.
    priv = ec.generate_private_key(ec.SECP256R1())
    pub = priv.public_key()
    jwk = ECAlgorithm.to_jwk(pub, as_dict=True)
    canonical = json.dumps(
        {"crv": jwk["crv"], "kty": jwk["kty"], "x": jwk["x"], "y": jwk["y"]}, separators=(",", ":"), sort_keys=True
    )
    expected = str(base64.urlsafe_b64encode(hashlib.sha256(canonical.encode("utf-8")).digest()).rstrip(b"="), "ascii")
    assert jwk_thumbprint(pub) == expected


def test_ec_thumbprint_is_deterministic() -> None:
    priv = ec.generate_private_key(ec.SECP256R1())
    pub = priv.public_key()
    assert jwk_thumbprint(pub) == jwk_thumbprint(pub)


def test_distinct_ec_keys_have_distinct_thumbprints() -> None:
    pub_a = ec.generate_private_key(ec.SECP256R1()).public_key()
    pub_b = ec.generate_private_key(ec.SECP256R1()).public_key()
    assert jwk_thumbprint(pub_a) != jwk_thumbprint(pub_b)


def test_ec_and_ed25519_thumbprints_of_unrelated_keys_differ() -> None:
    ec_pub = ec.generate_private_key(ec.SECP256R1()).public_key()
    ed_pub = Ed25519PrivateKey.generate().public_key()
    assert jwk_thumbprint(ec_pub) != jwk_thumbprint(ed_pub)


@given(curve_index=st.integers(min_value=0, max_value=len(_CURVES_WITH_ORDER) - 1), draw=st.integers(min_value=1))
def test_ec_thumbprint_property_rfc7638_conformance(curve_index: int, draw: int) -> None:
    """property test: for any P-256/P-384/P-521 EC public key, jwk_thumbprint is deterministic,
    spec-conformant (matches an independent {crv, kty, x, y} re-derivation), and a stable-length
    (43-char, unpadded base64url) SHA-256 digest -- RFC 7638 SS3.2's EC required-member set.

    ``draw`` is reduced modulo each curve's group order to land a hypothesis-controlled private
    scalar in the valid ``[1, order)`` range, so the drawn EC key genuinely varies (and shrinks)
    across examples instead of merely re-randomizing on every call.
    """
    curve, order = _CURVES_WITH_ORDER[curve_index]
    scalar = 1 + (draw % (order - 1))
    assume(1 <= scalar < order)
    pub = ec.derive_private_key(scalar, curve).public_key()

    thumb1 = jwk_thumbprint(pub)
    thumb2 = jwk_thumbprint(pub)
    assert thumb1 == thumb2  # deterministic for the same key

    jwk = ECAlgorithm.to_jwk(pub, as_dict=True)
    canonical = json.dumps(
        {"crv": jwk["crv"], "kty": jwk["kty"], "x": jwk["x"], "y": jwk["y"]}, separators=(",", ":"), sort_keys=True
    )
    expected = str(base64.urlsafe_b64encode(hashlib.sha256(canonical.encode("utf-8")).digest()).rstrip(b"="), "ascii")
    assert thumb1 == expected  # spec-conformant per RFC 7638 SS3.2's EC member set

    assert len(thumb1) == 43  # SHA-256 (32 bytes) -> 43 base64url chars, unpadded
    assert "=" not in thumb1 and "+" not in thumb1 and "/" not in thumb1
