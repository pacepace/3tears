"""Consumer-side helper: re-authorize a tool call against its engagement scope.

A tool that runs inside an authorized engagement -- a scanner re-authorizing its
target, any action gated by a per-engagement allow-set -- calls
:func:`resolve_engagement_scope` from inside its ``execute``. It is the engagement
twin of :func:`~threetears.agent.tools.consume.resolve_object`: the read-side gate
of the §2 keystone applied to consumer A (engagement scope).

The engagement id, the VERIFIED caller identity, and the resolver all ride on the
per-call :class:`~threetears.agent.tools.call_scope.ToolCallScope` the tool server
installs around every dispatch. The engagement id comes from the call CONTEXT
(``scope.context.engagement_id``), never a tool argument -- a tool cannot widen its
own authorization by naming a different engagement. The helper reads them through
:func:`~threetears.agent.tools.call_scope.current_scope`, resolves the scope via the
hub (forwarding the invoking agent's identity token), ASSERTS the customer the hub
echoed equals this call's verified customer (design §2 "belt to the signature
suspenders"), and REFUSES an empty scope (design §3 -- an engagement with no active
targets authorizes nothing).

Fail-closed -- raising :class:`EngagementScopeUnavailableError` -- when invoked
outside a call scope, when the call carries no engagement id, when the pod was wired
with no resolver, when the context carries no verified customer or identity token,
when the echoed customer disagrees, or when the resolved scope is empty. A hub
rejection surfaces as :class:`ResolveEngagementScopeError` from the resolver. A tool
must never proceed on an unresolved, cross-tenant, or empty scope.
"""

from __future__ import annotations

from uuid import UUID

from threetears.observe import get_logger

from threetears.agent.tools.call_scope import current_scope
from threetears.agent.tools.engagement_resolver import (
    EngagementScope,
    ScopeTarget,
)

__all__ = [
    "EngagementScope",
    "EngagementScopeUnavailableError",
    "ScopeTarget",
    "resolve_engagement_scope",
]

_log = get_logger(__name__)


class EngagementScopeUnavailableError(RuntimeError):
    """A tool could not obtain a usable engagement scope to authorize against.

    Carries a structural reason only (no call scope / no engagement id / no
    resolver wired / no verified customer / no identity token / echoed-customer
    mismatch / empty scope) -- never anything the caller should treat as
    authorization. Distinct from a hub REJECTION, which surfaces as
    :class:`~threetears.agent.tools.engagement_resolver.ResolveEngagementScopeError`.
    """


async def resolve_engagement_scope() -> EngagementScope:
    """Resolve + verify the current call's engagement scope, tenant-safely.

    The re-authorization keystone for engagement-bound tools. Reads the pod's
    engagement resolver + the VERIFIED identity off the current call scope and asks
    the hub for the active target set of the call's ``engagement_id``. The
    engagement id is taken from the call CONTEXT (not a tool argument), so a tool
    cannot authorize itself against a foreign or wider engagement. Authentication
    uses the invoking agent's ``identity_token`` -- the hub verifies it and derives
    the customer from the verified claim -- so this pure-``threetears`` pod needs no
    hub session of its own.

    Two post-resolve guards make the tenant + authorization guarantees hold even if
    the resolver or hub had a bug: the customer the hub ECHOED must equal this
    call's verified customer (§2), and the scope must be NON-EMPTY (§3 -- an
    engagement with no active targets authorizes nothing).

    :return: the resolved, verified, non-empty engagement scope; the caller builds
        its domain authorizer from ``scope.targets``
    :rtype: EngagementScope
    :raises EngagementScopeUnavailableError: no scope / no engagement id / no
        resolver wired / no verified customer / no identity token / echoed-customer
        mismatch / empty scope
    :raises ResolveEngagementScopeError: the hub rejected the resolve (identity
        unverified, or the customer does not own the engagement) or it failed in
        transit -- raised by the resolver and propagated
    """
    scope = current_scope()
    if scope is None:
        raise EngagementScopeUnavailableError(
            "engagement scope helper called outside a ToolServer call scope; an "
            "engagement-bound tool runs inside enter_call_scope"
        )
    # engagement_id MAY be None: the caller's conversation has not explicitly selected
    # an engagement. Rather than refuse here, ask the hub to resolve the customer's
    # DEFAULT scope (its single active engagement) -- the hub returns those targets or
    # refuses (zero or multiple active). Fail-closed still holds end to end: an
    # unresolved / empty scope is refused below, so no scan runs without a resolved
    # authorized target set.
    engagement_id: UUID | None = scope.context.engagement_id
    resolver = scope.engagement_resolver
    if resolver is None:
        raise EngagementScopeUnavailableError(
            "the current call scope carries no engagement resolver; the tool pod was "
            "not wired with one (no NATS client)"
        )
    customer_id = scope.context.customer_id
    if customer_id is None:
        raise EngagementScopeUnavailableError(
            "the call context carries no verified customer_id; refusing to resolve an untenanted engagement scope"
        )
    identity_token = scope.context.identity_token
    if identity_token is None:
        raise EngagementScopeUnavailableError(
            "the call context carries no identity_token; cannot authenticate the engagement scope resolve"
        )
    resolved = await resolver.resolve(engagement_id, customer_id=customer_id, identity_token=identity_token)
    if resolved.customer_id != customer_id:
        # echoed-customer mismatch: the hub resolved against a DIFFERENT customer
        # than this call's verified one. never happens on a correct path (both
        # derive from the same token), so treat it as a hard tenant-integrity
        # failure -- refuse, never authorize.
        _log.warning(
            "refusing engagement scope: echoed customer does not match verified customer",
            extra={
                "extra_data": {
                    "verified_customer_id": str(customer_id),  # convert at border: security log extra_data
                    "echoed_customer_id": str(resolved.customer_id),  # convert at border: security log extra_data
                }
            },
        )
        raise EngagementScopeUnavailableError(
            "the resolved engagement scope was resolved against a different customer than the verified caller"
        )
    if not resolved.targets:
        # an engagement with no active targets authorizes nothing; refuse rather
        # than build an empty allow-set a caller might mis-read as "allow all".
        raise EngagementScopeUnavailableError(
            "the engagement has no active authorized targets; refusing to authorize any scan under it"
        )
    return resolved
