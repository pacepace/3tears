"""Thin shell over the canonical JWT alg-pinning walker.

The identity-token verifier signs and verifies EdDSA only -- it publishes a JWKS, so every
consumer verifies with a public key and there is no symmetric scheme to permit. Anything
else named in this module is a violation.

This file previously carried its own ``ast.walk`` implementation, which was then copied into
a second package. The walker now lives in :mod:`threetears.enforcement.jwt_alg_pinning`, so
both call sites tighten together rather than drifting apart -- which is the failure mode this
kind of test exists to prevent, and which it had itself fallen into.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.jwt_alg_pinning import (
    JwtAlgPinningConfig,
    PinnedModule,
    run_jwt_alg_pinning_enforcement,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_every_jws_verifier_in_this_package_pins_eddsa() -> None:
    """All three, not just the identity token.

    `pop.py` and `proxy_assertion.py` verify JWS on the request path and were outside this
    gate. `_ALG` is pinned by name in each because it is what `jwt.encode` uses -- the walker
    checks decode calls, so without the constant pin the SIGNING algorithm could be repointed
    without tripping anything.
    """
    security = _PACKAGE_ROOT / "src" / "threetears" / "core" / "security"
    run_jwt_alg_pinning_enforcement(
        JwtAlgPinningConfig(
            repo_root=_PACKAGE_ROOT,
            modules=(
                PinnedModule(
                    # A pod-identity token carries no `aud`: `IdentityClaims` has no such
                    # field, and the binding is issuer + kid + the pod/user split.
                    path=security / "identity_token.py",
                    allowed_algorithms=frozenset({"EdDSA"}),
                    pinned_constants={"_ALG": "EdDSA"},
                    require_audience=False,
                ),
                PinnedModule(
                    # A proof-of-possession JWS carries no `aud`; it is bound by ath/bh.
                    path=security / "pop.py",
                    allowed_algorithms=frozenset({"EdDSA"}),
                    pinned_constants={"_ALG": "EdDSA"},
                    require_audience=False,
                ),
                PinnedModule(
                    path=security / "proxy_assertion.py",
                    allowed_algorithms=frozenset({"EdDSA"}),
                    pinned_constants={"_ALG": "EdDSA"},
                    require_audience=True,
                ),
            ),
        )
    )
