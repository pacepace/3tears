"""Consumer-side hub caller: resolve an engagement id to its authorized scope.

Engagement scope is consumer **A** of the §2 keystone (the read-side, tenant-safe
pod-side resolver), the twin of :class:`~threetears.agent.tools.object_resolver.HubObjectResolver`.
A tool running inside an authorized engagement holds the ``engagement_id`` on its
call context but not the engagement's authorized target set; :class:`HubEngagementScopeResolver`
asks the hub to resolve it -- tenant safely -- to the active targets the tool then
re-authorizes each scan against, replacing the pod's global env allowlist.

Authentication rides the per-call ``identity_token`` the invoking agent already
carries on the call context (a hub-minted, EdDSA-signed assertion the hub verifies
in-process). The hub derives the owning customer from the VERIFIED claim -- never an
unauthenticated request field -- so a tool can never resolve an engagement outside
its tenant, and this pod (a pure-``threetears`` tool server) needs no hub session of
its own. The reply ECHOES the customer it resolved against so the caller can assert
it equals this call's verified customer (design §2, "belt to the signature
suspenders").

Shape mirrors :class:`HubObjectResolver` -- a pod-side hub caller, fail-closed -- but
DELIBERATELY WITHOUT its cache. An object's id -> key mapping is immutable once
committed, so the object resolver caches with no refresh; an engagement's target set
is MUTABLE (targets are added and deactivated over the engagement's life), so caching
a resolved scope would serve STALE authorization -- a deactivated target would keep
authorizing. This is a security control, so it fails toward FRESH: every resolve is a
live round-trip. The cost is negligible (a scan runs for seconds to minutes; the
resolve is one sub-ms request), so resolve-per-scan is not a hot path. A short-TTL
cache is a possible future optimisation ONLY with an explicit, documented staleness
bound -- intentionally not built here.

Fail-closed: a transport error, a hub error reply (identity unverified / engagement
not found), or a malformed success reply raises :class:`ResolveEngagementScopeError`.
A tool must never proceed on an unresolved or cross-tenant scope.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Protocol
from uuid import UUID, uuid7

from pydantic import BaseModel
from threetears.nats import NatsClient, RequestError, Subjects
from threetears.observe import get_logger

__all__ = [
    "EngagementScope",
    "EngagementScopeRequestModel",
    "EngagementScopeResolver",
    "EngagementScopeResponseModel",
    "HubEngagementScopeResolver",
    "ResolveEngagementScopeError",
    "ScopeTarget",
]

_log = get_logger(__name__)


class ResolveEngagementScopeError(RuntimeError):
    """A tool could not resolve its engagement id to an authorized target scope.

    Carries a structural reason only (transport failure / hub rejection /
    malformed reply) -- never anything the caller should treat as authorization.
    """


class ScopeTarget(BaseModel):
    """one authorized target within an engagement's scope.

    a framework-general descriptor: the ``target_type`` is an OPAQUE string here
    (the consuming domain interprets it -- e.g. the pentest ``TargetAuthorizer``
    reads ``ip``/``cidr``/``hostname``/``url``). the framework never parses it.

    :param target_type: the domain-defined kind of target (opaque to the framework)
    :ptype target_type: str
    :param value: the target value (an address, range, hostname, or url)
    :ptype value: str
    :param label: an optional human label for the target
    :ptype label: str | None
    """

    model_config = {"frozen": True}

    target_type: str
    value: str
    label: str | None = None


class EngagementScope(BaseModel):
    """the resolved, verified authorization scope of one engagement.

    what a consuming tool re-authorizes each action against: the engagement id,
    the VERIFIED owning customer the hub resolved against (echoed so the caller
    can re-assert it equals this call's verified customer), and the engagement's
    active target set.

    :param engagement_id: the engagement this scope belongs to
    :ptype engagement_id: UUID
    :param customer_id: the verified customer the hub resolved the scope against
    :ptype customer_id: UUID
    :param targets: the active authorized targets (may be empty -- an empty scope
        is a valid hub answer that the re-auth helper refuses as unauthorized)
    :ptype targets: tuple[ScopeTarget, ...]
    """

    model_config = {"frozen": True}

    engagement_id: UUID
    customer_id: UUID
    targets: tuple[ScopeTarget, ...]


class EngagementScopeRequestModel(BaseModel):
    """outbound engagement-scope request to the hub.

    carries the invoking agent's ``identity_token`` as the caller proof. the
    owning customer is NOT sent: the hub derives it server-side from the VERIFIED
    token claim (never an unauthenticated request field), so a tool can never
    resolve an engagement under a customer it does not own. hand-mirrors the
    hub's ``EngagementScopeRequest`` (each side owns its model; they agree on the
    wire, like the object resolve pair).

    :param identity_token: the hub-minted identity assertion the invoking agent
        carries on the call context; the hub verifies it + derives the customer
    :ptype identity_token: str
    :param correlation_id: transport correlation id echoed in the reply (NOT a
        replay control -- replay is bounded by token expiry + session liveness
        hub-side)
    :ptype correlation_id: UUID
    :param engagement_id: the engagement id whose scope to resolve
    :ptype engagement_id: UUID
    """

    identity_token: str
    correlation_id: UUID
    engagement_id: UUID


class EngagementScopeResponseModel(BaseModel):
    """inbound engagement-scope response from the hub.

    one permissive model absorbs both the success and the error reply (the fields
    present differ), matching the object resolve pattern: check ``success`` then
    read the success fields.

    :param success: whether the engagement scope was resolved
    :ptype success: bool
    :param correlation_id: correlation id matching the request
    :ptype correlation_id: UUID | None
    :param customer_id: the verified customer the hub resolved against (on success;
        echoed so the caller re-asserts tenant agreement)
    :ptype customer_id: UUID | None
    :param targets: the active authorized targets (on success; may be empty)
    :ptype targets: list[ScopeTarget] | None
    :param error_code: machine-readable error code (on failure)
    :ptype error_code: str | None
    :param error_message: human-readable error description (on failure)
    :ptype error_message: str | None
    """

    success: bool
    correlation_id: UUID | None = None
    customer_id: UUID | None = None
    targets: list[ScopeTarget] | None = None
    error_code: str | None = None
    error_message: str | None = None


class EngagementScopeResolver(Protocol):
    """resolves an engagement id to its authorized scope under the verified tenant.

    the abstraction the tool server installs on every call scope so consuming
    tools reach a resolver through ``current_scope`` -- the same way they reach
    the object store + object resolver -- without per-tool constructor plumbing.
    tests inject a fake; production wires :class:`HubEngagementScopeResolver`.
    """

    async def resolve(self, engagement_id: UUID, *, customer_id: UUID, identity_token: str) -> EngagementScope:
        """resolve ``engagement_id`` to its authorized scope, or raise.

        :param engagement_id: the engagement id to resolve
        :ptype engagement_id: UUID
        :param customer_id: the VERIFIED owning customer (the caller re-asserts the
            echoed customer against it; NOT sent to the hub -- the hub derives the
            customer from the token)
        :ptype customer_id: UUID
        :param identity_token: the caller proof forwarded to the hub
        :ptype identity_token: str
        :return: the resolved engagement scope (verified customer + active targets)
        :rtype: EngagementScope
        :raises ResolveEngagementScopeError: transport failure, hub rejection, or a
            malformed success reply
        """
        ...


class HubEngagementScopeResolver:
    """resolves engagement ids to authorized target scopes over NATS, tenant-safely.

    a per-pod resolver the tool server self-provisions from its NATS client (it
    needs no S3 creds, only NATS) and installs on every call scope. holds ONLY the
    NATS client + the request timeout -- deliberately NO cache: an engagement's
    target set is mutable, so a resolved scope must never be reused across calls
    (see the module docstring). every resolve is a live round-trip.

    :param nats_client: connected canonical NATS wrapper client
    :ptype nats_client: NatsClient
    :param request_timeout_seconds: the scope request/reply timeout in seconds
    :ptype request_timeout_seconds: float
    """

    def __init__(
        self,
        nats_client: NatsClient,
        *,
        request_timeout_seconds: float,
    ) -> None:
        self._nc = nats_client
        self._timeout = request_timeout_seconds

    async def resolve(self, engagement_id: UUID, *, customer_id: UUID, identity_token: str) -> EngagementScope:
        """resolve ``engagement_id`` to its authorized scope, or raise.

        Sends a scope request carrying the ``identity_token`` (never a
        self-asserted customer/agent/session id) and returns the resolved scope.
        Fail-closed on transport error, an error reply, or a malformed success
        reply (missing echoed customer or target list). No caching -- authorization
        is always resolved fresh.

        :param engagement_id: the engagement id to resolve
        :ptype engagement_id: UUID
        :param customer_id: the VERIFIED owning customer, for the echoed-customer
            re-assertion done by the caller; NOT sent to the hub
        :ptype customer_id: UUID
        :param identity_token: the caller proof forwarded to the hub
        :ptype identity_token: str
        :return: the resolved engagement scope
        :rtype: EngagementScope
        :raises ResolveEngagementScopeError: transport failure, hub rejection, or a
            malformed success reply
        """
        request = EngagementScopeRequestModel(
            identity_token=identity_token,
            correlation_id=uuid7(),
            engagement_id=engagement_id,
        )
        try:
            response = await self._nc.request(
                subject=Subjects.hub_engagement_scope(),
                message=request,
                response_type=EngagementScopeResponseModel,
                timeout=timedelta(seconds=self._timeout),
            )
        except RequestError as exc:
            raise ResolveEngagementScopeError(f"engagement scope request failed: {exc}") from exc
        if not response.success:
            raise ResolveEngagementScopeError(
                f"engagement scope rejected: {response.error_code}: {response.error_message}"
            )
        if response.customer_id is None or response.targets is None:
            raise ResolveEngagementScopeError(
                "engagement scope reply reported success but carried no customer_id / targets"
            )
        scope = EngagementScope(
            engagement_id=engagement_id,
            customer_id=response.customer_id,
            targets=tuple(response.targets),
        )
        _log.debug(
            "resolved engagement scope",
            # engagement_id is the safe handle; target values are NOT logged.
            extra={
                "extra_data": {
                    "engagement_id": str(engagement_id),  # convert at border: resolve log extra_data
                    "target_count": len(scope.targets),
                }
            },
        )
        return scope
