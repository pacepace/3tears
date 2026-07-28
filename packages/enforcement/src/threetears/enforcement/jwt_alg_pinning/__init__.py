"""JWT algorithm-pinning enforcement domain -- structural checks over JWS verifiers.

Algorithm confusion (``alg=none``, HS/RS substitution) is the canonical JWS forgery.
Behavioural tests prove a verifier resists it today; this domain proves a future edit cannot
quietly remove the defence -- by widening the ``algorithms`` allow-list, disabling a
verification check, repointing a pinned constant, or switching to a decode entry point the
previous check did not know about.

Per-repo configuration goes through :class:`JwtAlgPinningConfig` and :class:`PinnedModule`;
:func:`run_jwt_alg_pinning_enforcement` is the pytest-friendly entry point.
"""

from threetears.enforcement.jwt_alg_pinning.config import (
    DEFAULT_BANNED_ALGORITHMS,
    JwtAlgPinningConfig,
    PinnedModule,
)
from threetears.enforcement.jwt_alg_pinning.runner import (
    run_jwt_alg_pinning_enforcement,
)
from threetears.enforcement.jwt_alg_pinning.walkers import (
    find_alg_pinning_violations,
)

__all__ = [
    "DEFAULT_BANNED_ALGORITHMS",
    "JwtAlgPinningConfig",
    "PinnedModule",
    "find_alg_pinning_violations",
    "run_jwt_alg_pinning_enforcement",
]
