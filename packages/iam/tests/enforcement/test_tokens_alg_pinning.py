"""Thin shell over the canonical JWT alg-pinning walker.

Two algorithms are legitimate in :mod:`threetears.iam.tokens`, unlike in a single-scheme
module: EdDSA for tokens other services verify from a JWKS, HS256 for a service that both
mints and verifies its own. What is enforced is that each decode names exactly ONE of them,
from a module constant, so the algorithm can never be chosen by anything the token carries.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.jwt_alg_pinning import (
    JwtAlgPinningConfig,
    PinnedModule,
    run_jwt_alg_pinning_enforcement,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_every_jws_verifier_in_this_package_pins_its_algorithms() -> None:
    """Both modules, not just the session tokens.

    `dpop.py` was outside this gate while the package README claimed "a DPoP proof does not
    get to say which algorithm verifies it" -- which is precisely the property a gate is for.

    `oidc.py` is deliberately absent and stays absent: it verifies through `joserfc`, not
    PyJWT, and its allow-list is per-connection by design (a tenant's IdP chooses RS256 or
    ES256). It rejects anything symmetric or `none` before decoding, in its own code, and
    that check has its own tests -- forcing it into this walker's literal-list rule would be
    a false positive, not a tightening.
    """
    run_jwt_alg_pinning_enforcement(
        JwtAlgPinningConfig(
            repo_root=_PACKAGE_ROOT,
            modules=(
                PinnedModule(
                    path=_PACKAGE_ROOT / "src" / "threetears" / "iam" / "tokens.py",
                    allowed_algorithms=frozenset({"EdDSA", "HS256"}),
                    pinned_constants={"_EDDSA": "EdDSA", "_HS256": "HS256"},
                    require_audience=True,
                ),
                PinnedModule(
                    # A DPoP proof carries no `aud` -- RFC 9449 binds it to htm/htu instead.
                    path=_PACKAGE_ROOT / "src" / "threetears" / "iam" / "dpop.py",
                    allowed_algorithms=frozenset({"ES256"}),
                    pinned_constants={"_ALG": "ES256"},
                    require_audience=False,
                ),
            ),
        )
    )
