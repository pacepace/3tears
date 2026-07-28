"""Thin shell over the canonical JWT alg-pinning walker.

Every module in this workspace that verifies a JWS is listed here. The rule each one keeps is
the two-stage pin: reject on the DECLARED header algorithm before a key is selected, then pass
a literal single-element list to ``decode``. The first stage stops ``alg: none`` and the
asymmetric-confusion downgrade (a token signed with a public key as an HMAC secret verifies
fine if the algorithm is read off the token); the second is what a reviewer can audit without
running anything.

This package SHIPS that walker and, until now, ran it against nothing. The only place it
executed was a consumer repo, guarding the one JWT that repo held locally -- while every
identity token, session token, DPoP proof and proxy assertion this workspace mints sat
outside it. A gate that a downstream product runs on the framework's behalf is not a gate on
the framework.

``oidc.py`` is deliberately absent: it does not decode with a pinned algorithm but validates a
provider's advertised set against :data:`~threetears.iam.oidc.DEFAULT_SIGNING_ALGORITHMS`,
refusing the ``HS``/``none`` families outright. Its own unit tests cover that; there is no
literal for this walker to check.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.jwt_alg_pinning import (
    JwtAlgPinningConfig,
    PinnedModule,
    run_jwt_alg_pinning_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE = _REPO_ROOT / "packages" / "core" / "src" / "threetears" / "core"
_IAM = _REPO_ROOT / "packages" / "iam" / "src" / "threetears" / "iam"

_CONFIG = JwtAlgPinningConfig(
    repo_root=_REPO_ROOT,
    modules=(
        # -- core: the pod-to-pod identity formats, all EdDSA ------------------------------
        PinnedModule(
            path=_CORE / "security" / "identity_token.py",
            allowed_algorithms=frozenset({"EdDSA"}),
            pinned_constants={"_ALG": "EdDSA"},
        ),
        PinnedModule(
            path=_CORE / "security" / "proxy_assertion.py",
            allowed_algorithms=frozenset({"EdDSA"}),
            pinned_constants={"_ALG": "EdDSA"},
            require_audience=True,
        ),
        # A proof of possession is bound to the request it accompanies, not minted for a
        # named consumer, so it carries no `aud` to require.
        PinnedModule(
            path=_CORE / "security" / "pop.py",
            allowed_algorithms=frozenset({"EdDSA"}),
            pinned_constants={"_ALG": "EdDSA"},
        ),
        # -- iam: session, proof-of-possession and OAuth state -----------------------------
        # Two algorithms deliberately: EdDSA for a deployment that publishes a JWKS, HS256 for
        # one that mints and verifies within a single trust boundary. Each has its own header
        # check, so a token signed under one can never be verified under the other.
        PinnedModule(
            path=_IAM / "tokens.py",
            allowed_algorithms=frozenset({"EdDSA", "HS256"}),
            pinned_constants={"_EDDSA": "EdDSA", "_HS256": "HS256"},
            require_audience=True,
        ),
        # DPoP (RFC 9449) proofs carry no `aud` at all.
        PinnedModule(
            path=_IAM / "dpop.py",
            allowed_algorithms=frozenset({"ES256"}),
            pinned_constants={"_ALG": "ES256"},
        ),
        PinnedModule(
            path=_IAM / "oauth_state.py",
            allowed_algorithms=frozenset({"HS256"}),
            pinned_constants={"_ALGORITHM": "HS256"},
            require_audience=True,
        ),
    ),
)


def test_every_jwt_verifying_module_pins_its_algorithm() -> None:
    """No module reads its algorithm off the token it is verifying."""
    run_jwt_alg_pinning_enforcement(_CONFIG)
