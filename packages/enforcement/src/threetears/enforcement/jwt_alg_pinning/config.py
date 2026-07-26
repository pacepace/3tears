"""configuration dataclass for JWT algorithm-pinning enforcement.

algorithm confusion is the canonical JWS forgery: ``alg=none`` strips
the signature entirely, and an RS256 verifier handed an HS256 token
will happily use the public key as the HMAC secret. The defence is to
never read the algorithm off the token, and behavioural tests prove
that holds *today*. This domain proves a future edit cannot quietly
remove it.

unlike the scanning domains, this one targets NAMED modules rather
than a whole src tree. There are only a handful of places in the
platform that verify a JWS, each is security-critical, and each is
listed explicitly — a module that starts verifying tokens without
being added here is a review failure, not something a tree walk would
catch anyway.

configurable per call site because the legitimate algorithm set
differs: a module that publishes a JWKS signs EdDSA only, while one
that both mints and verifies its own tokens may also use HS256. What
is NOT configurable is the shape of the guarantee — each decode names
exactly one algorithm, from a module-level constant or a literal, and
no decode disables a check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DEFAULT_BANNED_ALGORITHMS", "JwtAlgPinningConfig", "PinnedModule"]

#: Algorithm names that must never appear in a pinned module. A module's own permitted
#: algorithms are subtracted from this set by the walker, so listing HS256 here does not
#: prevent a module that legitimately supports it.
DEFAULT_BANNED_ALGORITHMS: frozenset[str] = frozenset(
    {
        "none",
        "HS256",
        "HS384",
        "HS512",
        "RS256",
        "RS384",
        "RS512",
        "ES256",
        "ES256K",
        "ES384",
        "ES512",
        "PS256",
        "PS384",
        "PS512",
    }
)


@dataclass(frozen=True)
class PinnedModule:
    """one module whose JWS algorithm handling is pinned.

    :ivar path: absolute path to the module.
    :ivar allowed_algorithms: the algorithm names this module may name and decode with.
        Usually one; two only where a module deliberately supports both an asymmetric and a
        symmetric scheme.
    :ivar require_audience: require an ``audience=`` argument on every decode. PyJWT skips
        audience validation entirely when none is supplied, so omitting it is equivalent to
        ``verify_aud=False``.
    :ivar pinned_constants: module-level constant name -> the literal it must hold, e.g.
        ``{"_EDDSA": "EdDSA"}``. Read from the AST rather than imported, so the check cannot
        be satisfied by anything computed at runtime. Empty means the module pins inline
        string literals instead of constants, which is equally acceptable.
    """

    path: Path
    allowed_algorithms: frozenset[str]
    pinned_constants: dict[str, str] = field(default_factory=dict)
    #: whether every decode in this module must pass ``audience=``. True for anything
    #: verifying a token minted for a named consumer; False for proof-of-possession formats
    #: (DPoP, RFC 9449) which carry no ``aud`` at all.
    require_audience: bool = False


@dataclass(frozen=True)
class JwtAlgPinningConfig:
    """per-repo config for the JWT algorithm-pinning enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root.
    :ivar modules: the modules to check. Empty is itself a violation -- a shell that
        resolves no modules has silently stopped enforcing anything.
    :ivar banned_algorithms: names no pinned module may mention, minus that module's own
        allowed set. Defaults to :data:`DEFAULT_BANNED_ALGORITHMS`.
    :ivar mode_env_var: environment variable controlling strict vs report mode.
    """

    repo_root: Path
    modules: tuple[PinnedModule, ...]
    banned_algorithms: frozenset[str] = DEFAULT_BANNED_ALGORITHMS
    mode_env_var: str = "JWT_ALG_PINNING_ENFORCEMENT_MODE"
