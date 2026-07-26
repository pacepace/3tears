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


def test_identity_token_module_pins_eddsa() -> None:
    run_jwt_alg_pinning_enforcement(
        JwtAlgPinningConfig(
            repo_root=_PACKAGE_ROOT,
            modules=(
                PinnedModule(
                    path=_PACKAGE_ROOT / "src" / "threetears" / "core" / "security" / "identity_token.py",
                    allowed_algorithms=frozenset({"EdDSA"}),
                ),
            ),
        )
    )
