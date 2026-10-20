"""policy-driven access-control + secret primitives.

public surface:

- :class:`Sandbox` / :class:`PathSandbox` / :class:`SandboxDecision` /
  :class:`SandboxDenied` — policy-driven access-control (see ``sandbox``).
- secret references (``secret_refs``): :func:`resolve_secret` / :func:`validate_ref` /
  :func:`parse_ref` / :func:`register_scheme` / :class:`SecretResolutionError` — a
  ``scheme://locator`` reference resolved to a ``SecretStr`` at use time; apps add
  schemes via :func:`register_scheme`.
- encryption at rest (``encryption``): :func:`seal` / :func:`open_secret` /
  :class:`DecryptionError` — AES-256-GCM under a master key, for the times a secret
  must be *stored* rather than referenced.
- identity tokens (``identity_token``): :class:`IdentityClaims` / :class:`IdentityTokenError` /
  :class:`IdentityKeyNotFoundError` (the recoverable kid-not-in-cache signal a verifier reacts to
  with a reactive JWKS refresh) /
  :func:`sign_identity_token` / :func:`verify_identity_token` / :func:`build_jwks` /
  :func:`generate_signing_keypair` / :data:`IDENTITY_REFUSED` (the one code every door answers a
  forwarded identity that does not verify) — Hub-issued EdDSA-signed JWS asserting a VERIFIED caller
  identity, verified against the Hub JWKS before RBAC (platform-auth Option B).
  :class:`~threetears.core.security.jwks_provider.CachedHubJwksProvider` fetches + caches that
  JWKS over NATS so a verifier's ``jwks_provider()`` returns it with no hot-path IO.
- proxy assertions (``proxy_assertion``): :func:`mint_proxy_assertion` /
  :func:`verify_proxy_assertion` / :data:`TOOL_PROXY_ASSERTION_UNVERIFIED` (the code a tool pod
  answers a call that could not show it came through the registry for this body and this pod) /
  :data:`TOOL_POP_LEDGER_UNAVAILABLE` (the code the registry and a tool pod both answer when the
  replay ledger their single-use check depends on cannot be reached).
- proof freshness (``freshness``): :data:`ISSUE_TIME_FUTURE_TOLERANCE` (how far ahead of a
  verifier's clock a POD-signed issue time is accepted) / :data:`CLIENT_ISSUE_TIME_FUTURE_TOLERANCE`
  (the same for a proof a user's device signs) -- the numbers a proof verifier's replay guard is
  sized for / :data:`DEFAULT_PROOF_MAX_AGE` (how old one may be) /
  :func:`issue_time_is_fresh` (the check, each direction on its own bound).
- service envelopes (``service_envelope``): :func:`mint_service_envelope` /
  :func:`verify_service_envelope` / :func:`verify_and_record` / :func:`body_hash` /
  :class:`ServiceEnvelopeError` / :class:`ServiceEnvelopeDoor` (one door's acceptance inside its dated
  window) -- a service with no caller token to forward signs "from me, for
  this door and operation, over this body, now, once" with its own JWKS key.
- delegated tokens (``delegated_token``): :func:`mint_delegated_token` / :func:`verify_delegated_token`
  / :func:`verify_door_caller` -- a tool call's own token for its pod, proven with the pod's key, and
  the door check on whoever presents a token.
- door requests (``door_request``): :class:`DoorCredential` / :func:`verify_door_request` /
  :func:`door_request_hash` -- a door request's credential and its proof of possession over the body.
- turn assertions (``turn_assertion``): :func:`mint_turn_assertion` / :func:`verify_turn_assertion` /
  :func:`new_turn_id` -- the hub's signed statement of whose turn an agent runs, kept out of
  ``CallContext``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from threetears.core._extras import import_extras_module

# lazy public API (PEP 562), as in ``threetears.core``: importing this package costs only this file,
# and each name resolves its defining module on first access, so a caller that needs one light
# module here does not load the extras-backed ones beside it. the TYPE_CHECKING block carries the
# real imports for mypy and IDEs; ``_LAZY`` is the runtime equivalent.
if TYPE_CHECKING:
    from threetears.core.security.encryption import DecryptionError, open_secret, seal
    from threetears.core.security.freshness import (
        CLIENT_ISSUE_TIME_FUTURE_TOLERANCE,
        DEFAULT_PROOF_MAX_AGE,
        ISSUE_TIME_FUTURE_TOLERANCE,
        issue_time_is_fresh,
    )
    from threetears.core.security.identity_minter import (
        DEFAULT_IDENTITY_TTL_SECONDS,
        IdentityMinter,
        static_token_provider,
    )
    from threetears.core.security.identity_token import (
        IDENTITY_REFUSED,
        IDENTITY_REFUSED_MESSAGE,
        PLATFORM_CUSTOMER_SENTINEL,
        IdentityClaims,
        IdentityKeyNotFoundError,
        IdentityTokenError,
        VerifiedPrincipal,
        build_jwks,
        canonical_call_hash,
        generate_signing_keypair,
        jwk_thumbprint,
        principal_from,
        principal_from_claims,
        sign_identity_token,
        verify_identity_token,
    )
    from threetears.core.security.jwks_provider import CachedHubJwksProvider
    from threetears.core.security.pop import access_token_hash, make_pop_proof, verify_pop_proof
    from threetears.core.security.proxy_assertion import (
        TOOL_POP_LEDGER_UNAVAILABLE,
        TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE,
        TOOL_PROXY_ASSERTION_UNVERIFIED,
        TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE,
        ProxyAssertionClaims,
        mint_proxy_assertion,
        verify_proxy_assertion,
    )
    from threetears.core.security.proxy_signer import ProxyAssertionSigner
    from threetears.core.security.sandbox import (
        PathSandbox,
        Sandbox,
        SandboxDecision,
        SandboxDenied,
    )
    from threetears.core.security.service_envelope import (
        SERVICE_ENVELOPE_FIELD,
        SERVICE_ENVELOPE_LEEWAY,
        SERVICE_ENVELOPE_MAX_LIFETIME,
        SERVICE_ENVELOPE_REPLAY_TTL,
        EnvelopeJwksSource,
        EnvelopeLedger,
        HubJwksEnvelopeSource,
        InProcessEnvelopeLedger,
        PinnedEnvelopeJwks,
        ServiceEnvelopeClaims,
        ServiceEnvelopeDoor,
        ServiceEnvelopeError,
        ServiceEnvelopeSigner,
        body_hash,
        mint_service_envelope,
        verify_and_record,
        verify_service_envelope,
    )
    from threetears.core.security.delegated_token import (
        DEFAULT_DELEGATED_DOORS,
        DELEGATED_TOKEN_ISSUER,
        DELEGATED_TOKEN_MAX_LIFETIME,
        DELEGATED_TOKEN_TYPE,
        DelegatedClaims,
        DelegatedKeyNotFoundError,
        DelegatedTokenError,
        DelegatedTokenSigner,
        DoorCaller,
        ProofAbsentError,
        is_delegated_token,
        mint_delegated_token,
        verify_delegated_token,
        verify_delegation_at_pod,
        verify_door_caller,
    )
    from threetears.core.security.door_request import (
        DOOR_PROOF_FIELD,
        DOOR_PROOF_REPLAY_TTL,
        DOOR_SENDER_FIELDS,
        DOOR_TOKEN_FIELD,
        DoorCredential,
        DoorProofLedger,
        DoorRequestDoor,
        DoorVerdict,
        PopKey,
        RegistrationKey,
        datasource_query_op,
        claimed_sender_of,
        door_request_hash,
        l3_request_op,
        verify_door_request,
    )
    from threetears.core.security.turn_assertion import (
        TURN_ASSERTION_AUDIENCE,
        TURN_ASSERTION_LIFETIME,
        TurnAssertionClaims,
        TurnAssertionError,
        mint_turn_assertion,
        new_turn_id,
        verify_turn_assertion,
    )
    from threetears.core.security.secret_refs import (
        Resolver,
        SecretResolutionError,
        parse_ref,
        register_scheme,
        resolve_secret,
        validate_ref,
    )

_LAZY: dict[str, tuple[str, str]] = {
    "DecryptionError": ("threetears.core.security.encryption", "DecryptionError"),
    "open_secret": ("threetears.core.security.encryption", "open_secret"),
    "seal": ("threetears.core.security.encryption", "seal"),
    "CLIENT_ISSUE_TIME_FUTURE_TOLERANCE": ("threetears.core.security.freshness", "CLIENT_ISSUE_TIME_FUTURE_TOLERANCE"),
    "DEFAULT_PROOF_MAX_AGE": ("threetears.core.security.freshness", "DEFAULT_PROOF_MAX_AGE"),
    "ISSUE_TIME_FUTURE_TOLERANCE": ("threetears.core.security.freshness", "ISSUE_TIME_FUTURE_TOLERANCE"),
    "issue_time_is_fresh": ("threetears.core.security.freshness", "issue_time_is_fresh"),
    "DEFAULT_IDENTITY_TTL_SECONDS": ("threetears.core.security.identity_minter", "DEFAULT_IDENTITY_TTL_SECONDS"),
    "IdentityMinter": ("threetears.core.security.identity_minter", "IdentityMinter"),
    "static_token_provider": ("threetears.core.security.identity_minter", "static_token_provider"),
    "IDENTITY_REFUSED": ("threetears.core.security.identity_token", "IDENTITY_REFUSED"),
    "IDENTITY_REFUSED_MESSAGE": ("threetears.core.security.identity_token", "IDENTITY_REFUSED_MESSAGE"),
    "PLATFORM_CUSTOMER_SENTINEL": ("threetears.core.security.identity_token", "PLATFORM_CUSTOMER_SENTINEL"),
    "IdentityClaims": ("threetears.core.security.identity_token", "IdentityClaims"),
    "IdentityKeyNotFoundError": ("threetears.core.security.identity_token", "IdentityKeyNotFoundError"),
    "IdentityTokenError": ("threetears.core.security.identity_token", "IdentityTokenError"),
    "VerifiedPrincipal": ("threetears.core.security.identity_token", "VerifiedPrincipal"),
    "build_jwks": ("threetears.core.security.identity_token", "build_jwks"),
    "canonical_call_hash": ("threetears.core.security.identity_token", "canonical_call_hash"),
    "generate_signing_keypair": ("threetears.core.security.identity_token", "generate_signing_keypair"),
    "jwk_thumbprint": ("threetears.core.security.identity_token", "jwk_thumbprint"),
    "principal_from": ("threetears.core.security.identity_token", "principal_from"),
    "principal_from_claims": ("threetears.core.security.identity_token", "principal_from_claims"),
    "sign_identity_token": ("threetears.core.security.identity_token", "sign_identity_token"),
    "verify_identity_token": ("threetears.core.security.identity_token", "verify_identity_token"),
    "CachedHubJwksProvider": ("threetears.core.security.jwks_provider", "CachedHubJwksProvider"),
    "access_token_hash": ("threetears.core.security.pop", "access_token_hash"),
    "make_pop_proof": ("threetears.core.security.pop", "make_pop_proof"),
    "verify_pop_proof": ("threetears.core.security.pop", "verify_pop_proof"),
    "TOOL_POP_LEDGER_UNAVAILABLE": ("threetears.core.security.proxy_assertion", "TOOL_POP_LEDGER_UNAVAILABLE"),
    "TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE": (
        "threetears.core.security.proxy_assertion",
        "TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE",
    ),
    "TOOL_PROXY_ASSERTION_UNVERIFIED": ("threetears.core.security.proxy_assertion", "TOOL_PROXY_ASSERTION_UNVERIFIED"),
    "TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE": (
        "threetears.core.security.proxy_assertion",
        "TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE",
    ),
    "ProxyAssertionClaims": ("threetears.core.security.proxy_assertion", "ProxyAssertionClaims"),
    "mint_proxy_assertion": ("threetears.core.security.proxy_assertion", "mint_proxy_assertion"),
    "verify_proxy_assertion": ("threetears.core.security.proxy_assertion", "verify_proxy_assertion"),
    "ProxyAssertionSigner": ("threetears.core.security.proxy_signer", "ProxyAssertionSigner"),
    "PathSandbox": ("threetears.core.security.sandbox", "PathSandbox"),
    "Sandbox": ("threetears.core.security.sandbox", "Sandbox"),
    "SandboxDecision": ("threetears.core.security.sandbox", "SandboxDecision"),
    "SandboxDenied": ("threetears.core.security.sandbox", "SandboxDenied"),
    "SERVICE_ENVELOPE_FIELD": ("threetears.core.security.service_envelope", "SERVICE_ENVELOPE_FIELD"),
    "SERVICE_ENVELOPE_LEEWAY": ("threetears.core.security.service_envelope", "SERVICE_ENVELOPE_LEEWAY"),
    "SERVICE_ENVELOPE_MAX_LIFETIME": ("threetears.core.security.service_envelope", "SERVICE_ENVELOPE_MAX_LIFETIME"),
    "SERVICE_ENVELOPE_REPLAY_TTL": ("threetears.core.security.service_envelope", "SERVICE_ENVELOPE_REPLAY_TTL"),
    "EnvelopeJwksSource": ("threetears.core.security.service_envelope", "EnvelopeJwksSource"),
    "EnvelopeLedger": ("threetears.core.security.service_envelope", "EnvelopeLedger"),
    "HubJwksEnvelopeSource": ("threetears.core.security.service_envelope", "HubJwksEnvelopeSource"),
    "InProcessEnvelopeLedger": ("threetears.core.security.service_envelope", "InProcessEnvelopeLedger"),
    "PinnedEnvelopeJwks": ("threetears.core.security.service_envelope", "PinnedEnvelopeJwks"),
    "ServiceEnvelopeClaims": ("threetears.core.security.service_envelope", "ServiceEnvelopeClaims"),
    "ServiceEnvelopeDoor": ("threetears.core.security.service_envelope", "ServiceEnvelopeDoor"),
    "ServiceEnvelopeError": ("threetears.core.security.service_envelope", "ServiceEnvelopeError"),
    "ServiceEnvelopeSigner": ("threetears.core.security.service_envelope", "ServiceEnvelopeSigner"),
    "body_hash": ("threetears.core.security.service_envelope", "body_hash"),
    "mint_service_envelope": ("threetears.core.security.service_envelope", "mint_service_envelope"),
    "verify_and_record": ("threetears.core.security.service_envelope", "verify_and_record"),
    "verify_service_envelope": ("threetears.core.security.service_envelope", "verify_service_envelope"),
    "DEFAULT_DELEGATED_DOORS": ("threetears.core.security.delegated_token", "DEFAULT_DELEGATED_DOORS"),
    "DELEGATED_TOKEN_ISSUER": ("threetears.core.security.delegated_token", "DELEGATED_TOKEN_ISSUER"),
    "DELEGATED_TOKEN_MAX_LIFETIME": ("threetears.core.security.delegated_token", "DELEGATED_TOKEN_MAX_LIFETIME"),
    "DELEGATED_TOKEN_TYPE": ("threetears.core.security.delegated_token", "DELEGATED_TOKEN_TYPE"),
    "DelegatedClaims": ("threetears.core.security.delegated_token", "DelegatedClaims"),
    "DelegatedKeyNotFoundError": ("threetears.core.security.delegated_token", "DelegatedKeyNotFoundError"),
    "DelegatedTokenError": ("threetears.core.security.delegated_token", "DelegatedTokenError"),
    "DelegatedTokenSigner": ("threetears.core.security.delegated_token", "DelegatedTokenSigner"),
    "DoorCaller": ("threetears.core.security.delegated_token", "DoorCaller"),
    "ProofAbsentError": ("threetears.core.security.delegated_token", "ProofAbsentError"),
    "is_delegated_token": ("threetears.core.security.delegated_token", "is_delegated_token"),
    "mint_delegated_token": ("threetears.core.security.delegated_token", "mint_delegated_token"),
    "verify_delegated_token": ("threetears.core.security.delegated_token", "verify_delegated_token"),
    "verify_delegation_at_pod": ("threetears.core.security.delegated_token", "verify_delegation_at_pod"),
    "verify_door_caller": ("threetears.core.security.delegated_token", "verify_door_caller"),
    "DOOR_PROOF_FIELD": ("threetears.core.security.door_request", "DOOR_PROOF_FIELD"),
    "DOOR_PROOF_REPLAY_TTL": ("threetears.core.security.door_request", "DOOR_PROOF_REPLAY_TTL"),
    "DOOR_SENDER_FIELDS": ("threetears.core.security.door_request", "DOOR_SENDER_FIELDS"),
    "DOOR_TOKEN_FIELD": ("threetears.core.security.door_request", "DOOR_TOKEN_FIELD"),
    "DoorCredential": ("threetears.core.security.door_request", "DoorCredential"),
    "DoorProofLedger": ("threetears.core.security.door_request", "DoorProofLedger"),
    "DoorRequestDoor": ("threetears.core.security.door_request", "DoorRequestDoor"),
    "DoorVerdict": ("threetears.core.security.door_request", "DoorVerdict"),
    "PopKey": ("threetears.core.security.door_request", "PopKey"),
    "RegistrationKey": ("threetears.core.security.door_request", "RegistrationKey"),
    "datasource_query_op": ("threetears.core.security.door_request", "datasource_query_op"),
    "claimed_sender_of": ("threetears.core.security.door_request", "claimed_sender_of"),
    "door_request_hash": ("threetears.core.security.door_request", "door_request_hash"),
    "l3_request_op": ("threetears.core.security.door_request", "l3_request_op"),
    "verify_door_request": ("threetears.core.security.door_request", "verify_door_request"),
    "TURN_ASSERTION_AUDIENCE": ("threetears.core.security.turn_assertion", "TURN_ASSERTION_AUDIENCE"),
    "TURN_ASSERTION_LIFETIME": ("threetears.core.security.turn_assertion", "TURN_ASSERTION_LIFETIME"),
    "TurnAssertionClaims": ("threetears.core.security.turn_assertion", "TurnAssertionClaims"),
    "TurnAssertionError": ("threetears.core.security.turn_assertion", "TurnAssertionError"),
    "mint_turn_assertion": ("threetears.core.security.turn_assertion", "mint_turn_assertion"),
    "new_turn_id": ("threetears.core.security.turn_assertion", "new_turn_id"),
    "verify_turn_assertion": ("threetears.core.security.turn_assertion", "verify_turn_assertion"),
    "Resolver": ("threetears.core.security.secret_refs", "Resolver"),
    "SecretResolutionError": ("threetears.core.security.secret_refs", "SecretResolutionError"),
    "parse_ref": ("threetears.core.security.secret_refs", "parse_ref"),
    "register_scheme": ("threetears.core.security.secret_refs", "register_scheme"),
    "resolve_secret": ("threetears.core.security.secret_refs", "resolve_secret"),
    "validate_ref": ("threetears.core.security.secret_refs", "validate_ref"),
}

__all__ = [
    # access control
    "PathSandbox",
    "Sandbox",
    "SandboxDecision",
    "SandboxDenied",
    # secret references
    "Resolver",
    "SecretResolutionError",
    "parse_ref",
    "register_scheme",
    "resolve_secret",
    "validate_ref",
    # encryption at rest
    "DecryptionError",
    "open_secret",
    "seal",
    # proof freshness
    "CLIENT_ISSUE_TIME_FUTURE_TOLERANCE",
    "DEFAULT_PROOF_MAX_AGE",
    "ISSUE_TIME_FUTURE_TOLERANCE",
    "issue_time_is_fresh",
    # delegated tokens
    "DEFAULT_DELEGATED_DOORS",
    "DELEGATED_TOKEN_ISSUER",
    "DELEGATED_TOKEN_MAX_LIFETIME",
    "DELEGATED_TOKEN_TYPE",
    "DelegatedClaims",
    "DelegatedKeyNotFoundError",
    "DelegatedTokenError",
    "DelegatedTokenSigner",
    "DoorCaller",
    "ProofAbsentError",
    "is_delegated_token",
    "mint_delegated_token",
    "verify_delegated_token",
    "verify_delegation_at_pod",
    "verify_door_caller",
    # door requests
    "DOOR_PROOF_FIELD",
    "DOOR_PROOF_REPLAY_TTL",
    "DOOR_SENDER_FIELDS",
    "DOOR_TOKEN_FIELD",
    "DoorCredential",
    "DoorProofLedger",
    "DoorRequestDoor",
    "DoorVerdict",
    "PopKey",
    "RegistrationKey",
    "datasource_query_op",
    "claimed_sender_of",
    "door_request_hash",
    "l3_request_op",
    "verify_door_request",
    # turn assertions
    "TURN_ASSERTION_AUDIENCE",
    "TURN_ASSERTION_LIFETIME",
    "TurnAssertionClaims",
    "TurnAssertionError",
    "mint_turn_assertion",
    "new_turn_id",
    "verify_turn_assertion",
    # service envelopes
    "SERVICE_ENVELOPE_FIELD",
    "SERVICE_ENVELOPE_LEEWAY",
    "SERVICE_ENVELOPE_MAX_LIFETIME",
    "SERVICE_ENVELOPE_REPLAY_TTL",
    "EnvelopeJwksSource",
    "EnvelopeLedger",
    "HubJwksEnvelopeSource",
    "InProcessEnvelopeLedger",
    "PinnedEnvelopeJwks",
    "ServiceEnvelopeClaims",
    "ServiceEnvelopeDoor",
    "ServiceEnvelopeError",
    "ServiceEnvelopeSigner",
    "body_hash",
    "mint_service_envelope",
    "verify_and_record",
    "verify_service_envelope",
    # identity tokens
    "DEFAULT_IDENTITY_TTL_SECONDS",
    "IDENTITY_REFUSED",
    "IDENTITY_REFUSED_MESSAGE",
    "PLATFORM_CUSTOMER_SENTINEL",
    "TOOL_POP_LEDGER_UNAVAILABLE",
    "TOOL_POP_LEDGER_UNAVAILABLE_MESSAGE",
    "TOOL_PROXY_ASSERTION_UNVERIFIED",
    "TOOL_PROXY_ASSERTION_UNVERIFIED_MESSAGE",
    "CachedHubJwksProvider",
    "IdentityClaims",
    "IdentityKeyNotFoundError",
    "IdentityMinter",
    "IdentityTokenError",
    "ProxyAssertionClaims",
    "ProxyAssertionSigner",
    "VerifiedPrincipal",
    "access_token_hash",
    "build_jwks",
    "canonical_call_hash",
    "generate_signing_keypair",
    "jwk_thumbprint",
    "make_pop_proof",
    "mint_proxy_assertion",
    "principal_from",
    "principal_from_claims",
    "sign_identity_token",
    "static_token_provider",
    "verify_identity_token",
    "verify_pop_proof",
    "verify_proxy_assertion",
]


def __getattr__(name: str) -> object:
    """resolve a public attribute from its defining module on first access.

    :param name: attribute name being resolved
    :ptype name: str
    :return: the resolved attribute (also cached in module globals so
        ``__getattr__`` fires at most once per name)
    :rtype: object
    :raises AttributeError: when ``name`` is not part of the public API
    :raises ImportError: when the defining module needs an extra that is not installed,
        naming the extra
    """
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = entry
    value: object = getattr(import_extras_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """include lazy attributes in ``dir()`` output.

    :return: sorted union of materialized globals and lazy names
    :rtype: list[str]
    """
    return sorted(set(globals()) | set(_LAZY))
