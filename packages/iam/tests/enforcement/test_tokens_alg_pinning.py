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


def test_tokens_module_pins_its_algorithms() -> None:
    run_jwt_alg_pinning_enforcement(
        JwtAlgPinningConfig(
            repo_root=_PACKAGE_ROOT,
            modules=(
                PinnedModule(
                    path=_PACKAGE_ROOT / "src" / "threetears" / "iam" / "tokens.py",
                    allowed_algorithms=frozenset({"EdDSA", "HS256"}),
                    pinned_constants={"_EDDSA": "EdDSA", "_HS256": "HS256"},
                ),
            ),
        )
    )
