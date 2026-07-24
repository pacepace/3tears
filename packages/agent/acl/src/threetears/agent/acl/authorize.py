"""canonical rbac authorization primitive shared across every 3tears app.

every resource-typed authorize helper (memory, datasource, channel,
customer, audit, api_key, model, conversation, workspace,
workspace_file, shared_agent, ...) collapses to a 3-line wrapper
around :func:`authorize` that:

1. resolves resource identity to a canonical namespace name
2. picks the action vocabulary specific to the resource
3. catches :class:`AccessDenied` and re-raises a typed
   resource-specific subclass

the primitive itself is resource-agnostic: it takes a
:class:`NamespaceCollection` handle, a namespace name, an action
string, the calling user + agent ids, and a shared :class:`AclCache`.
it looks up the namespace, builds an :class:`EvaluationContext`,
calls :func:`evaluate_decision` (which serves from the cache's
membership and per-namespace layers), and either returns the
:class:`EvaluationResult` or raises :class:`AccessDenied` on a deny.

generalization rationale: per the 3tears platform vision, RBAC is a
cross-cutting concern the SDK owns. one canonical path keeps every
consumer's behavior identical — same cache layers, same denial shape,
same trace span — so a fix landed here propagates without per-app
audit. resource-specific helpers exist only to pin (a) the action
vocabulary and (b) the typed exception class their callers catch
on; they do not re-implement the lookup or the evaluator call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

from threetears.agent.acl.cache import AclCache
from threetears.agent.acl.evaluator import evaluate_with_trail
from threetears.agent.acl.types import (
    EvaluationContext,
    EvaluationResult,
    Namespace as AclNamespace,
)
from threetears.observe import get_logger, traced

__all__ = [
    "INTERNAL_AUDIENCE",
    "AccessDenied",
    "ClaimsForAuthorization",
    "ExternalAudienceNotSupported",
    "ImpersonationCategory",
    "NamespaceNotFound",
    "authorize",
    "authorize_from_claims",
    "authorize_on_entity",
    "authorize_with_trail",
]

# The literal value `authorize_from_claims` deny-lists against -- matches
# `UserTokenClaims.act_reason`'s value space in the identity-core repo's
# `identity_core/tokens/claims.py` exactly (docs/design.md: "`act_reason`
# (optional): `impersonation` or `delegation`"). Duplicated here rather than
# imported: identity-core depends on this package, never the reverse, and a
# platform token's `act_reason` string is effectively a small, stable wire
# vocabulary rather than a type this package should own. A value mismatch
# between the two repos fails closed (the overlay simply never triggers,
# same "no compile-time check across this boundary" posture
# `identity_core/rbac_rpc.py`'s module docstring already documents for the
# claim-grant subject strings), never silently misapplies.
_IMPERSONATION_ACT_REASON = "impersonation"

# The `aud` claim value denoting a token minted for a consumer INSIDE the
# platform trust boundary -- the only audience whose `sub` is a real internal
# `principal_id`. Duplicated from identity-core's own `TokenAudience.INTERNAL`
# for the identical reason `_IMPERSONATION_ACT_REASON` above is: identity-core
# depends on this package, never the reverse, and this is a small, stable wire
# vocabulary rather than a type this package should own. A value mismatch
# between the two repos fails CLOSED here too -- an unrecognized audience
# yields `is_internal_audience=False` and `authorize_from_claims` refuses,
# rather than admitting a token it cannot vouch for.
#
# Public (unlike `_IMPERSONATION_ACT_REASON`) because consuming apps building
# a `ClaimsForAuthorization` need to name the same value; exported so they do
# not each hardcode the string.
INTERNAL_AUDIENCE = "aibots:internal"

log = get_logger(__name__)


class AccessDenied(Exception):
    """raised when the unified evaluator denies an access request.

    carries the action, namespace name, and caller identity so a
    resource-specific wrapper can preserve the contextual fields when
    re-raising as a typed subclass. callers that catch this generic
    base catch every per-resource denial transparently; callers that
    need to dispatch on resource type catch the typed subclass.

    :ivar action: action string evaluated (e.g. ``"memory.read"``)
    :ivar namespace_name: canonical name of namespace evaluated against
    :ivar user_id: invoking user UUID, or ``None`` for agent-only
        evaluations
    :ivar agent_id: invoking agent UUID, or ``None`` for user-only
        evaluations
    :ivar reason: short classification string for log / audit fan-out
    """

    def __init__(
        self,
        message: str,
        *,
        action: str | None = None,
        namespace_name: str | None = None,
        user_id: UUID | None = None,
        agent_id: UUID | None = None,
        reason: str | None = None,
    ) -> None:
        """initialize the denial exception.

        :param message: human-readable denial message
        :ptype message: str
        :param action: action string evaluated
        :ptype action: str | None
        :param namespace_name: namespace name evaluated against
        :ptype namespace_name: str | None
        :param user_id: invoking user UUID
        :ptype user_id: UUID | None
        :param agent_id: invoking agent UUID
        :ptype agent_id: UUID | None
        :param reason: short classification string
        :ptype reason: str | None
        """
        super().__init__(message)
        self.action = action
        self.namespace_name = namespace_name
        self.user_id = user_id
        self.agent_id = agent_id
        self.reason = reason


class NamespaceNotFound(AccessDenied):
    """raised when the authorize primitive cannot resolve namespace by name.

    distinct subclass of :class:`AccessDenied` so resource-specific
    wrappers can surface "namespace row missing" as a wiring-gap
    diagnostic separately from a "user lacks grant" denial. the
    typed subclass keeps callers that catch :class:`AccessDenied`
    backwards-compatible: every namespace-not-found is still an
    access denial.
    """


class ExternalAudienceNotSupported(AccessDenied):
    """raised by :func:`authorize_from_claims` when the caller's claims are
    not internal-audience.

    security-model.md's Authorization section: "`authorize_from_claims`
    resolves `sub` to a real `principal_id` correctly only on internal-
    audience tokens. Doesn't bite v1 (RBAC-checking traffic is entirely
    internal-audience...) but is a real open item before external-audience
    tokens can ever reach an RBAC check -- track it as a launch blocker for
    any external-audience-token feature, not something to discover at
    integration time." Fails closed here rather than silently resolving
    `sub` (a pairwise/opaque id on an external-audience token, per
    docs/design.md's Claims table) against a `principal_id`-keyed group
    membership that will simply never match -- a loud, typed refusal is
    easier to diagnose at integration time than a decision that always
    happens to come back "denied" for the wrong reason.
    """


class ImpersonationCategory(StrEnum):
    """The fixed, non-tenant-configurable deny-list category set --
    security-model.md's Impersonation paragraph names these six categories
    verbatim: "credential, passkey, and MFA management; account deletion;
    API-key mint/rotate for the target; email change; RBAC grant changes;
    audit editing."

    Every member is unconditionally deny-listed under impersonation --
    :func:`authorize_from_claims` denies whenever ``act_reason ==
    "impersonation"`` and the caller names ANY category (there is no
    partial/tenant-configurable subset, per security-model.md: "a fixed,
    non-tenant-configurable category set").

    The identity-core repo defines the SAME six string values independently
    in its own `identity_core/auth/step_up.py` (shared taxonomy: "Same fixed
    taxonomy as the impersonation deny-list, enforced by freshness instead
    of subtraction" -- docs/design.md's Step-Up Re-Authentication section) --
    identity-core's own self-service endpoints never call this cross-repo
    function directly (they check `act_reason` locally, per security-model.md's
    "second, independent enforcement point"), so there is no import
    dependency to share the enum through; the string values are the actual
    shared contract, mirroring the existing `IdentitySubjects`/Hub-subject
    duplication precedent (`identity_core/rbac_rpc.py`'s module docstring).
    """

    CREDENTIAL_PASSKEY_MFA_MANAGEMENT = "credential_passkey_mfa_management"
    ACCOUNT_DELETION = "account_deletion"
    APIKEY_MINT_ROTATE_FOR_TARGET = "apikey_mint_rotate_for_target"
    EMAIL_CHANGE = "email_change"
    RBAC_GRANT_CHANGE = "rbac_grant_change"
    AUDIT_EDITING = "audit_editing"


@dataclass(frozen=True, slots=True)
class ClaimsForAuthorization:
    """The minimal shape :func:`authorize_from_claims` needs off an ALREADY-
    VERIFIED token. This package never verifies a token itself (signature/
    issuer/audience/expiry checking is each consuming app's own concern --
    identity-core's is `identity_core/tokens/sign.py`'s `verify_user_token`);
    the caller reads these fields off its own verified claims object and
    passes them here.

    :ivar sub: the token's `sub` claim -- on an internal-audience token this
        is the real `principal_id` (str, converted to `UUID` internally);
        MUST NOT be passed from an external-audience token (see
        `is_internal_audience`)
    :ivar is_internal_audience: whether the token this claims object came
        from is internal-audience. `authorize_from_claims` raises
        :class:`ExternalAudienceNotSupported` when `False` -- see that
        exception's docstring.
    :ivar act_reason: the token's `act_reason` claim (`"impersonation"`,
        `"delegation"`, or `None`) -- only `"impersonation"` triggers the
        deny-list overlay; `"delegation"` and `None` are unaffected.
    """

    sub: str
    is_internal_audience: bool
    act_reason: str | None = None

    @classmethod
    def from_verified_claims(
        cls, *, sub: str, aud: str | None, act_reason: str | None = None
    ) -> ClaimsForAuthorization:
        """Build from a verified token's own claims, deriving
        `is_internal_audience` from `aud` instead of having the caller assert it.

        Prefer this over the plain constructor. `is_internal_audience` decides
        whether `sub` may be treated as a real internal `principal_id`, so a
        caller that computes the boolean itself is one typo away from handing an
        external token's subject to RBAC as though it were internal. Passing the
        `aud` claim through and deriving here makes the token itself the
        authority, which is the property `authorize_from_claims`'s
        :class:`ExternalAudienceNotSupported` guard assumes it already has.

        Fails CLOSED: any `aud` that is not exactly :data:`INTERNAL_AUDIENCE` --
        including `None`, for a token predating the claim -- yields
        `is_internal_audience=False`, and `authorize_from_claims` then refuses.
        Deliberately NOT tolerant of a missing `aud` the way a token-verifying
        consumer may need to be during rollout: this is the authorization
        boundary rather than the authentication one, and treating an unlabelled
        token as internal here is precisely the mistake worth making impossible.

        :param sub: the token's `sub` claim
        :ptype sub: str
        :param aud: the token's `aud` claim, or `None` when absent
        :ptype aud: str | None
        :param act_reason: the token's `act_reason` claim
        :ptype act_reason: str | None
        :return: claims with `is_internal_audience` derived from `aud`
        :rtype: ClaimsForAuthorization
        """
        return cls(sub=sub, is_internal_audience=aud == INTERNAL_AUDIENCE, act_reason=act_reason)


@traced
async def authorize_on_entity(
    *,
    ns_entity: Any,
    action: str,
    user_id: UUID | None,
    agent_id: UUID | None,
    cache: AclCache,
    namespace_name: str | None = None,
) -> EvaluationResult:
    """canonical rbac authorization primitive over a pre-resolved namespace.

    every resource-typed helper that resolves its namespace through a
    bespoke path (``get_by_owner_and_customer`` for memory + conversation,
    pre-attached entity for workspace, ...) calls this primitive after
    materializing the namespace entity. the surface complements
    :func:`authorize` (lookup-by-name) and :func:`authorize_with_trail`
    (lookup-by-name returning the entity for downstream audit envelopes)
    by removing the lookup step entirely, so the full machinery is
    callable from any helper regardless of how its namespace identity
    was discovered.

    :param ns_entity: pre-resolved namespace entity exposing ``id``,
        ``customer_id``, ``namespace_type``, ``owner_agent_id``
        attributes; typed ``Any`` because concrete Collection entity
        class lives in consumer apps' layers (hub, agent pod) above
        this package
    :ptype ns_entity: Any
    :param action: canonical action string (e.g. ``"memory.read"``,
        ``"workspace.read"``)
    :ptype action: str
    :param user_id: invoking user UUID, or ``None`` for agent-only
        evaluation
    :ptype user_id: UUID | None
    :param agent_id: invoking agent UUID, or ``None`` for user-only
        evaluation
    :ptype agent_id: UUID | None
    :param cache: shared :class:`AclCache` carrying loaders + ttl
        layers
    :ptype cache: AclCache
    :param namespace_name: canonical namespace name for log + denial
        messages; helpers that have it threaded through pass it for
        clearer diagnostics, helpers that build the namespace from
        a workspace / memory pair pass ``None`` and the denial message
        falls back to the entity id
    :ptype namespace_name: str | None
    :return: full evaluation result on allow (carries effective
        actions, contributing trails, limiting side)
    :rtype: EvaluationResult
    :raises AccessDenied: when the evaluator denies the action
    """
    acl_namespace = AclNamespace(
        id=ns_entity.id,
        customer_id=ns_entity.customer_id,
        namespace_type=ns_entity.namespace_type,
        owner_agent_id=ns_entity.owner_agent_id,
    )
    eval_ctx = EvaluationContext(
        namespace=acl_namespace,
        action=action,
        user_id=user_id,
        agent_id=agent_id,
    )
    result = await evaluate_with_trail(eval_ctx, cache=cache)
    if not result.decision:
        ns_label = namespace_name if namespace_name is not None else str(ns_entity.id)
        # convert at border: authorize-denied log extra_data fields
        log_user_id = str(user_id) if user_id else None
        log_agent_id = str(agent_id) if agent_id else None
        log.info(
            "authorize: denied",
            extra={
                "extra_data": {
                    "action": action,
                    "namespace_name": namespace_name,
                    "namespace_id": str(ns_entity.id),  # convert at border: authorize-denied log extra_data field
                    "user_id": log_user_id,
                    "agent_id": log_agent_id,
                },
            },
        )
        raise AccessDenied(
            f"access denied: {action} on namespace {ns_label}",
            action=action,
            namespace_name=namespace_name,
            user_id=user_id,
            agent_id=agent_id,
            reason="evaluator_deny",
        )
    return result


@traced
async def authorize(
    *,
    namespace_collection: Any,
    namespace_name: str,
    action: str,
    user_id: UUID | None,
    agent_id: UUID | None,
    cache: AclCache,
) -> EvaluationResult:
    """canonical rbac authorization primitive.

    looks up namespace by name via ``namespace_collection.get_by_name``,
    then delegates to :func:`authorize_on_entity` for the evaluator
    call + denial machinery. raises :class:`NamespaceNotFound` when
    the namespace row is absent and :class:`AccessDenied` when the
    evaluator denies; returns the full :class:`EvaluationResult` on
    allow so callers that need the effective action set or contributing
    trails do not pay for a second evaluation.

    :param namespace_collection: a Collection exposing
        ``async def get_by_name(name: str) -> entity | None``;
        typed ``Any`` because concrete Collection class lives in
        consumer apps' layers (hub, agent pod) above this package
    :ptype namespace_collection: Any
    :param namespace_name: canonical namespace name to evaluate
        against (e.g. ``"datasources.my_warehouse"``,
        ``"memories.<agent_id_hex>.<customer_id_hex>"``)
    :ptype namespace_name: str
    :param action: canonical action string (e.g. ``"memory.read"``,
        ``"datasource.write"``)
    :ptype action: str
    :param user_id: invoking user UUID, or ``None`` for agent-only
        evaluation
    :ptype user_id: UUID | None
    :param agent_id: invoking agent UUID, or ``None`` for user-only
        evaluation
    :ptype agent_id: UUID | None
    :param cache: shared :class:`AclCache` carrying loaders + ttl
        layers
    :ptype cache: AclCache
    :return: full evaluation result on allow
    :rtype: EvaluationResult
    :raises NamespaceNotFound: when ``namespace_collection.get_by_name``
        returns None for ``namespace_name``
    :raises AccessDenied: when the evaluator denies the action
    """
    ns_entity = await namespace_collection.get_by_name(namespace_name)
    if ns_entity is None:
        # convert at border: authorize namespace-missing log extra_data fields
        log_user_id = str(user_id) if user_id else None
        log_agent_id = str(agent_id) if agent_id else None
        log.warning(
            "authorize: namespace row missing",
            extra={
                "extra_data": {
                    "action": action,
                    "namespace_name": namespace_name,
                    "user_id": log_user_id,
                    "agent_id": log_agent_id,
                },
            },
        )
        raise NamespaceNotFound(
            f"access denied: namespace {namespace_name} not found",
            action=action,
            namespace_name=namespace_name,
            user_id=user_id,
            agent_id=agent_id,
            reason="namespace_not_found",
        )
    return await authorize_on_entity(
        ns_entity=ns_entity,
        action=action,
        user_id=user_id,
        agent_id=agent_id,
        cache=cache,
        namespace_name=namespace_name,
    )


@traced
async def authorize_with_trail(
    *,
    namespace_collection: Any,
    namespace_name: str,
    action: str,
    user_id: UUID | None,
    agent_id: UUID | None,
    cache: AclCache,
) -> tuple[EvaluationResult, Any]:
    """authorize variant that also returns resolved namespace entity.

    several resource wrappers (datasource, customer, memory) need the
    entity itself for downstream audit envelopes or assignment-ensure
    paths. this variant performs the same lookup + evaluator call as
    :func:`authorize` and returns ``(result, ns_entity)`` so callers
    do not pay for a second namespace lookup.

    :param namespace_collection: a Collection exposing
        ``async def get_by_name(name: str) -> entity | None``
    :ptype namespace_collection: Any
    :param namespace_name: canonical namespace name to evaluate against
    :ptype namespace_name: str
    :param action: canonical action string
    :ptype action: str
    :param user_id: invoking user UUID, or ``None``
    :ptype user_id: UUID | None
    :param agent_id: invoking agent UUID, or ``None``
    :ptype agent_id: UUID | None
    :param cache: shared :class:`AclCache`
    :ptype cache: AclCache
    :return: ``(result, ns_entity)`` pair
    :rtype: tuple[EvaluationResult, Any]
    :raises NamespaceNotFound: when ``namespace_collection.get_by_name``
        returns None for ``namespace_name``
    :raises AccessDenied: when the evaluator denies the action
    """
    ns_entity = await namespace_collection.get_by_name(namespace_name)
    if ns_entity is None:
        # convert at border: authorize_with_trail namespace-missing log extra_data fields
        log_user_id = str(user_id) if user_id else None
        log_agent_id = str(agent_id) if agent_id else None
        log.warning(
            "authorize_with_trail: namespace row missing",
            extra={
                "extra_data": {
                    "action": action,
                    "namespace_name": namespace_name,
                    "user_id": log_user_id,
                    "agent_id": log_agent_id,
                },
            },
        )
        raise NamespaceNotFound(
            f"access denied: namespace {namespace_name} not found",
            action=action,
            namespace_name=namespace_name,
            user_id=user_id,
            agent_id=agent_id,
            reason="namespace_not_found",
        )
    result = await authorize_on_entity(
        ns_entity=ns_entity,
        action=action,
        user_id=user_id,
        agent_id=agent_id,
        cache=cache,
        namespace_name=namespace_name,
    )
    return result, ns_entity


@traced
async def authorize_from_claims(
    *,
    namespace_collection: Any,
    namespace_name: str,
    action: str,
    claims: ClaimsForAuthorization,
    cache: AclCache,
    sensitive_category: ImpersonationCategory | None = None,
) -> EvaluationResult:
    """claims-aware authorization primitive -- the impersonation deny-list
    entry point.

    security-model.md's Impersonation paragraph: "Applied via a new
    claims-aware `authorize_from_claims` entry point that takes a verified
    token directly (the existing `authorize()` signature has no path for
    `act_reason` to reach it at all)." This wraps :func:`authorize` with
    exactly two additions the plain `user_id`/`agent_id` signature cannot
    express:

    1. resolves the caller identity from a verified token's claims
       (`claims.sub`) rather than a caller-supplied `user_id`, so the
       identity checked is provably the token's own subject -- refusing
       outright (:class:`ExternalAudienceNotSupported`) on a non-internal-
       audience token rather than silently resolving `sub` against the
       wrong identity space (see that exception's docstring).
    2. applies the impersonation deny-list overlay: when
       ``claims.act_reason == "impersonation"`` AND the caller names a
       `sensitive_category` (:class:`ImpersonationCategory`), the action is
       denied UNCONDITIONALLY -- before the underlying `authorize()` call
       even runs -- regardless of what the target's own permissions would
       otherwise allow. Every other combination (ordinary `act_reason=None`
       traffic, `act_reason="delegation"`, or an impersonation session
       requesting a NON-sensitive action) defers entirely to the ordinary
       `authorize()` evaluation.

    Callers that already have `sensitive_category=None` traffic and no
    `act_reason` to thread through get byte-identical behavior to calling
    :func:`authorize` directly with `user_id=UUID(claims.sub)` -- this
    function is additive, not a second evaluation path with its own bugs to
    diverge from `authorize`'s.

    :param namespace_collection: same as :func:`authorize`
    :ptype namespace_collection: Any
    :param namespace_name: same as :func:`authorize`
    :ptype namespace_name: str
    :param action: same as :func:`authorize`
    :ptype action: str
    :param claims: the caller's verified-token claims (see
        :class:`ClaimsForAuthorization`)
    :ptype claims: ClaimsForAuthorization
    :param cache: same as :func:`authorize`
    :ptype cache: AclCache
    :param sensitive_category: which deny-list category `action` belongs
        to, or ``None`` if it belongs to none -- the caller (a resource-
        specific wrapper one layer up, which already knows its own action
        vocabulary) makes this determination; this package has no way to
        infer it from the bare `action` string alone since that vocabulary
        is app-specific
    :ptype sensitive_category: ImpersonationCategory | None
    :return: full evaluation result on allow
    :rtype: EvaluationResult
    :raises ExternalAudienceNotSupported: `claims.is_internal_audience` is
        `False`
    :raises AccessDenied: `claims.act_reason == "impersonation"` and
        `sensitive_category` is set (reason ``"impersonation_deny_list"``),
        or the underlying :func:`authorize` call denies
    :raises NamespaceNotFound: `namespace_collection.get_by_name` returns
        `None` for `namespace_name`
    """
    if not claims.is_internal_audience:
        log.warning(
            "authorize_from_claims: external-audience token rejected",
            extra={"extra_data": {"action": action, "namespace_name": namespace_name}},
        )
        raise ExternalAudienceNotSupported(
            "authorize_from_claims: external-audience tokens are not supported "
            "(sub cannot be resolved to a real principal_id)",
            action=action,
            namespace_name=namespace_name,
            reason="external_audience_not_supported",
        )
    try:
        user_id = UUID(claims.sub)
    except ValueError:
        raise AccessDenied(
            f"authorize_from_claims: claims.sub {claims.sub!r} is not a valid principal id",
            action=action,
            namespace_name=namespace_name,
            reason="invalid_sub",
        ) from None

    if claims.act_reason == _IMPERSONATION_ACT_REASON and sensitive_category is not None:
        log.info(
            "authorize_from_claims: denied by impersonation deny-list",
            extra={
                "extra_data": {
                    "action": action,
                    "namespace_name": namespace_name,
                    "user_id": str(user_id),  # convert at border: structured logging (extra_data)
                    "sensitive_category": sensitive_category.value,
                },
            },
        )
        raise AccessDenied(
            f"access denied: {action} on namespace {namespace_name} is deny-listed "
            f"under impersonation (category {sensitive_category.value})",
            action=action,
            namespace_name=namespace_name,
            user_id=user_id,
            agent_id=None,
            reason="impersonation_deny_list",
        )

    return await authorize(
        namespace_collection=namespace_collection,
        namespace_name=namespace_name,
        action=action,
        user_id=user_id,
        agent_id=None,
        cache=cache,
    )


# evaluate_decision is intentionally not re-exported here; callers
# that want the bool-only fast path import from the evaluator module
# directly. the canonical user-facing surface for application code
# is :func:`authorize` / :func:`authorize_with_trail`.
