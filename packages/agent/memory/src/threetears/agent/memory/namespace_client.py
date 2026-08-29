"""agent-side client for the HUB-OWNED memory-namespace ensure.

A memory namespace is a row in the hub's ``namespaces`` table, and that table
is the platform control plane. Until this module
existed, :func:`~threetears.agent.memory.authorize._resolve_or_create_memory_namespace`
built the row in-process and pushed it through ``save_entity`` -- an AGENT
writing the control plane, permitted only because the L3 broker's
platform-write gate carved out ``namespace_type='memory'`` for exactly that
call. Removing the last such writer is what lets that carve-out close.

The create moved to the hub, and this module is the caller's half:

- :class:`MemoryNamespaceProvisioner` is what
  :class:`~threetears.agent.memory.authorize.MemoryAuthorizerDependencies`
  holds. It is a Protocol rather than a concrete class so the memory package
  stays transport-agnostic and a test can substitute a provisioner without a
  bus.
- :class:`HubMemoryNamespaceProvisioner` is the NATS implementation. It sends
  ``{ns}.hub.memory.namespace.ensure`` as request/REPLY, never publish: the
  caller cannot proceed without the resolved row, because an authorization
  decision is evaluated against it on the very next statement, so a
  fire-and-forget subject that can return neither a row nor a refusal is the
  wrong primitive here.
- :class:`MemoryNamespaceRef` is what both halves of
  :func:`~threetears.agent.memory.authorize.authorize_memory_access` resolve
  to -- the five fields
  :func:`~threetears.agent.acl.authorize_on_entity` reads, and nothing else.
  The owner path builds one deterministically in-process; the non-owner path
  gets one back from the hub.

**Identity is the forwarded token, never the request body.** The request
carries ``agent_id`` and ``customer_id``, and they are NOT how the hub decides
whose namespace to make: the hub reads that off the verified token and REFUSES
a body stating a different pair, the same judgement
``aibots.hub.tools.manifest_identity`` applies to a registration manifest. The
body's copy exists so the two can be COMPARED -- silently creating whatever
the token says, for a caller that asked about something else, would hand back
a namespace an authorization decision is then made against. :meth:`
HubMemoryNamespaceProvisioner.ensure` re-checks the same equality on the reply
for the same reason, so a hub answering about another pair is a failure here
rather than a wrong allow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from pydantic import BaseModel
from threetears.nats import RequestError, Subjects
from threetears.observe import get_logger
from uuid_utils import uuid7

if TYPE_CHECKING:
    from collections.abc import Callable

    from threetears.nats import NatsClient

__all__ = [
    "DEFAULT_ENSURE_TIMEOUT_SECONDS",
    "HubMemoryNamespaceProvisioner",
    "MemoryNamespaceEnsureReply",
    "MemoryNamespaceEnsureRequest",
    "MemoryNamespaceProvisioner",
    "MemoryNamespaceRef",
    "MemoryNamespaceUnavailableError",
]

log = get_logger(__name__)


#: seconds to wait for the hub's reply. the ensure sits inside a respondent-
#: facing tool call, so it is bounded well below a user-visible timeout; a hub
#: that cannot answer in this window is reported as unavailable rather than
#: waited on.
DEFAULT_ENSURE_TIMEOUT_SECONDS: float = 10.0


class MemoryNamespaceUnavailableError(Exception):
    """raised when the memory namespace could not be resolved hub-side.

    covers every non-answer uniformly: no identity token to present, transport
    failure, hub refusal, malformed reply, and a reply describing a different
    ``(agent, customer)`` pair than was asked about. the caller
    (:func:`~threetears.agent.memory.authorize._resolve_or_create_memory_namespace`)
    converts it to :class:`~threetears.agent.memory.authorize.MemoryAccessDenied`,
    so a namespace that cannot be resolved fails CLOSED rather than admitting
    an unauthorized access.
    """


@dataclass(frozen=True)
class MemoryNamespaceRef:
    """resolved memory namespace, carrying exactly what the evaluator reads.

    frozen and deliberately narrow: :func:`~threetears.agent.acl.authorize_on_entity`
    reads ``id`` / ``customer_id`` / ``namespace_type`` / ``owner_agent_id`` /
    ``owner_namespace`` and nothing else, so a ref cannot be mistaken for a
    persisted entity and cannot be saved by anything.

    :param id: namespace UUID
    :ptype id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param owner_agent_id: owning agent UUID
    :ptype owner_agent_id: UUID
    :param namespace_type: namespace type discriminator (``memory``)
    :ptype namespace_type: str
    :param owner_namespace: owning agent's own namespace name, which is the key
        the evaluator's ownership short-circuit reads; ``None`` when the row
        records no owner
    :ptype owner_namespace: str | None
    """

    id: UUID
    customer_id: UUID
    owner_agent_id: UUID
    namespace_type: str
    owner_namespace: str | None = None


class MemoryNamespaceProvisioner(Protocol):
    """what the authorizer bundle holds to materialize a memory namespace.

    a Protocol rather than a class so the memory package declares the shape it
    needs without importing a transport, and so a caller that provisions
    namespaces some other way (an in-process hub, a test double) satisfies it
    by structure.
    """

    async def ensure(self, *, agent_id: UUID, customer_id: UUID) -> MemoryNamespaceRef:
        """resolve-or-create the memory namespace for one (agent, customer) pair.

        :param agent_id: owning agent UUID
        :ptype agent_id: UUID
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :return: resolved namespace reference
        :rtype: MemoryNamespaceRef
        :raises MemoryNamespaceUnavailableError: when it could not be resolved
        """
        ...  # pragma: no cover - protocol declaration


class MemoryNamespaceEnsureRequest(BaseModel):
    """request asking the hub to materialize one memory namespace row.

    ``agent_id`` and ``customer_id`` are what the caller BELIEVES it is asking
    about; they never decide what the hub writes. The hub derives the pair from
    the verified ``identity_token`` and refuses a request whose body disagrees,
    so these two fields are a coherence check that turns an impersonation
    attempt into a refusal instead of a silent substitution.

    :param identity_token: forwarded hub identity token; the hub verifies it and
        derives the owning agent + customer from its signed claims
    :ptype identity_token: str
    :param correlation_id: correlation id echoed on the reply
    :ptype correlation_id: UUID
    :param agent_id: owning agent UUID the caller is asking about
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID the caller is asking about
    :ptype customer_id: UUID
    """

    identity_token: str
    correlation_id: UUID
    agent_id: UUID
    customer_id: UUID


class MemoryNamespaceEnsureReply(BaseModel):
    """hub reply describing the resolved memory namespace, or the refusal.

    one model rather than a success / error pair because it is also the
    ``response_type`` the NATS client decodes into, and a caller must be able
    to read a refusal off the same shape it reads a success off.

    :param success: whether the namespace was resolved
    :ptype success: bool
    :param correlation_id: correlation id matching the request
    :ptype correlation_id: UUID | None
    :param namespace_id: resolved namespace UUID (on success)
    :ptype namespace_id: UUID | None
    :param name: resolved namespace name (on success)
    :ptype name: str | None
    :param namespace_type: resolved namespace type (on success)
    :ptype namespace_type: str | None
    :param owner_agent_id: owning agent UUID as the hub VERIFIED it (on success)
    :ptype owner_agent_id: UUID | None
    :param customer_id: owning customer UUID as the hub DERIVED it (on success)
    :ptype customer_id: UUID | None
    :param owner_namespace: owning agent's namespace name (on success)
    :ptype owner_namespace: str | None
    :param error_code: machine-readable refusal code (on failure)
    :ptype error_code: str | None
    :param error_message: human-readable refusal description (on failure)
    :ptype error_message: str | None
    """

    success: bool
    correlation_id: UUID | None = None
    namespace_id: UUID | None = None
    name: str | None = None
    namespace_type: str | None = None
    owner_agent_id: UUID | None = None
    customer_id: UUID | None = None
    owner_namespace: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class HubMemoryNamespaceProvisioner:
    """asks the hub to materialize a memory namespace, over NATS request/reply.

    holds a LIVE identity-token provider rather than a captured token: an
    agent's periodic identity refresh re-handshakes and replaces the token, and
    the hub accepts only the current one, so a provisioner that pinned the
    bootstrap token would be refused on every ensure after the first refresh.
    the same reason :class:`aibots_agents.runtime.catalog.HubObjectCataloger`
    reads its session id live.

    :param nats_client: connected canonical NATS wrapper client
    :ptype nats_client: NatsClient
    :param identity_token_provider: zero-arg callable returning this pod's
        CURRENT hub identity token, or ``None`` when no handshake has completed
    :ptype identity_token_provider: Callable[[], str | None]
    :param timeout: seconds to wait for the hub's reply
    :ptype timeout: float
    """

    def __init__(
        self,
        *,
        nats_client: NatsClient,
        identity_token_provider: Callable[[], str | None],
        timeout: float = DEFAULT_ENSURE_TIMEOUT_SECONDS,
    ) -> None:
        """store the transport + identity provider.

        :param nats_client: connected NATS wrapper client
        :ptype nats_client: NatsClient
        :param identity_token_provider: zero-arg CURRENT-token provider
        :ptype identity_token_provider: Callable[[], str | None]
        :param timeout: seconds to wait for the hub's reply
        :ptype timeout: float
        :return: nothing
        :rtype: None
        """
        self._nats_client = nats_client
        self._identity_token_provider = identity_token_provider
        self._timeout = timeout

    def _current_token(self) -> str:
        """read the CURRENT identity token, or fail closed.

        the provider may return ``None`` (no handshake yet) or raise (the L3
        backend's provider raises when the handshake has not completed); both
        mean the same thing to a caller and are reported the same way.

        :return: current hub identity token
        :rtype: str
        :raises MemoryNamespaceUnavailableError: when no token is available
        """
        try:
            token = self._identity_token_provider()
        except Exception as exc:
            raise MemoryNamespaceUnavailableError(
                "memory namespace ensure has no identity token to present",
            ) from exc
        if not token:
            raise MemoryNamespaceUnavailableError(
                "memory namespace ensure has no identity token to present",
            )
        return token

    async def ensure(self, *, agent_id: UUID, customer_id: UUID) -> MemoryNamespaceRef:
        """ask the hub for the ``(agent, customer)`` memory namespace row.

        :param agent_id: owning agent UUID
        :ptype agent_id: UUID
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :return: resolved namespace reference
        :rtype: MemoryNamespaceRef
        :raises MemoryNamespaceUnavailableError: on a missing token, a transport
            failure, a hub refusal, a malformed reply, or a reply describing a
            different ``(agent, customer)`` pair than was asked about
        """
        request = MemoryNamespaceEnsureRequest(
            identity_token=self._current_token(),
            correlation_id=UUID(str(uuid7())),
            agent_id=agent_id,
            customer_id=customer_id,
        )
        try:
            reply: MemoryNamespaceEnsureReply = await self._nats_client.request(
                subject=Subjects.hub_memory_namespace_ensure(),
                message=request,
                response_type=MemoryNamespaceEnsureReply,
                timeout=timedelta(seconds=self._timeout),
            )
        except RequestError as exc:
            raise MemoryNamespaceUnavailableError(
                f"memory namespace ensure request failed: {exc}",
            ) from exc
        return _ref_from_reply(reply, agent_id=agent_id, customer_id=customer_id)


def _ref_from_reply(
    reply: MemoryNamespaceEnsureReply,
    *,
    agent_id: UUID,
    customer_id: UUID,
) -> MemoryNamespaceRef:
    """validate one reply and convert it into a reference, or raise.

    the pair equality check is the load-bearing half. the caller is about to
    evaluate an authorization decision against whatever comes back, so a reply
    naming a namespace owned by a different agent or scoped to a different
    customer must not become that decision's subject -- it is refused here even
    though the hub already refuses the mirror-image case, because the two
    checks defend different sides of the same wire.

    :param reply: decoded hub reply
    :ptype reply: MemoryNamespaceEnsureReply
    :param agent_id: owning agent UUID that was asked about
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID that was asked about
    :ptype customer_id: UUID
    :return: resolved namespace reference
    :rtype: MemoryNamespaceRef
    :raises MemoryNamespaceUnavailableError: when the reply refuses, omits a
        field the evaluator reads, or names a different pair
    """
    if not reply.success:
        raise MemoryNamespaceUnavailableError(
            f"memory namespace ensure refused: {reply.error_code}: {reply.error_message}",
        )
    if reply.namespace_id is None or reply.namespace_type is None:
        raise MemoryNamespaceUnavailableError(
            "memory namespace ensure returned a success carrying no namespace",
        )
    if reply.owner_agent_id != agent_id or reply.customer_id != customer_id:
        raise MemoryNamespaceUnavailableError(
            "memory namespace ensure returned a namespace for a different agent or customer",
        )
    log.debug(
        "memory namespace ensured hub-side",
        extra={
            "extra_data": {
                # convert at border: log extra_data field
                "namespace_id": str(reply.namespace_id),
                "namespace_name": reply.name,
            }
        },
    )
    return MemoryNamespaceRef(
        id=reply.namespace_id,
        customer_id=customer_id,
        owner_agent_id=agent_id,
        namespace_type=reply.namespace_type,
        owner_namespace=reply.owner_namespace,
    )
