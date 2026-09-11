"""call a registered tool through the registry, on the caller's own identity.

**Why this exists.** A process that serves tools can also need to CALL one: a tool
pod working on bulk data reaches the platform's dataset verbs -- ``define``,
``validate``, ``build``, ``run_status``, ``inspect`` -- the way an agent does, over
``{ns}.tools.call``, and the registry decides. It verifies the forwarded hub-minted
identity token at the door, verifies the per-call proof of possession against that
token's holder binding, pins the principal from the SIGNED ``sub``, evaluates that
principal's ``tool.call`` grant on the tool's namespace, and forwards or refuses.
Nothing on this wire names a principal, so a caller can be authorized only as
whoever the hub minted the token for.

**Identity is a PROVIDER, never a value.** The token is short-lived and re-minted
in place by the holder's refresh loop, so a string captured at construction is
expired within the hour and every call after that is refused by a client that
looks correctly wired. The provider is read on EVERY call, exactly as
:class:`threetears.datasources.query_client.DatasourceQueryClient` reads its own.
An empty or missing token is refused HERE, before the bus sees the request,
because the registry refuses it identically and a refusal that names the wiring
gap beats one that reads as a permission problem.

**The proof of possession is signed by the caller's holder key.** The registry
verifies it unconditionally: a call without a valid proof is
``TOOL_POP_UNVERIFIED`` however good the token is. The client takes a signer
rather than a key so the one minter of proofs a process already holds is reused;
the shape is :class:`PopSignerProtocol`, which the SDK's ``PopSigner`` satisfies
without knowing this module exists.

**One wire model, both ends.** The request is the registry's own
:class:`~threetears.registry.proxy.ProxyCallRequest` and the reply its
:class:`~threetears.registry.proxy.ProxyCallResponse`, so this client cannot drift
from the door it knocks on the way a hand-copied envelope does.

**Errors carry the registry's code.** A caller branches on
:attr:`ToolCallError.error_code` and never on the message. ``TOOL_NOT_AUTHORIZED``
is the one a consumer most needs to tell apart from a transport fault: the first
is a grant to fix, the second is a bus to look at. A TOOL's own refusal
(``success=False`` from the pod) raises through the same type with the tool's
code, so one ``except`` covers the whole call.

**Synchronous, on the caller's own inbox.** No durable reply is requested: a tool
pod holds no grant on the result family, and the verbs this client exists for
answer in milliseconds. A tool that runs longer than the registry's forward
budget comes back as the registry's ``TOOL_TIMEOUT``, not as a transport fault,
because the client's own deadline sits above that budget on purpose.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID, uuid7

from threetears.agent.tools.context_envelope import CallContext
from threetears.nats.errors import RequestError
from threetears.nats.subjects import Subjects
from threetears.observe import get_logger, traced

from threetears.registry.proxy import ProxyCallRequest, ProxyCallResponse

if TYPE_CHECKING:
    from threetears.nats import NatsClient

__all__ = [
    "CALL_TIMEOUT_MARGIN_SECONDS",
    "DEFAULT_CALL_TIMEOUT_SECONDS",
    "PopSignerProtocol",
    "ToolCallClient",
    "ToolCallError",
]

log = get_logger(__name__)


#: how much longer than the registry's own forward budget the client waits.
#:
#: the registry forwards a call under a per-tool or default timeout and answers a
#: typed ``TOOL_TIMEOUT`` when it trips. a client deadline BELOW that budget would
#: turn every slow tool into the client's own ``REQUEST_FAILED``, which cannot tell
#: a slow tool from a dead bus and steers a retry that stacks a second call on
#: the first. the margin is what the registry's refusal envelope needs to travel
#: back while the caller is still listening.
CALL_TIMEOUT_MARGIN_SECONDS: float = 10.0

#: how long one call may take, end to end, before the client gives up.
#:
#: the registry's platform default forward budget plus the margin above; pinned
#: above :func:`threetears.registry.config.get_call_timeout` by a test so the two
#: cannot drift into the wrong order.
DEFAULT_CALL_TIMEOUT_SECONDS: float = 120.0 + CALL_TIMEOUT_MARGIN_SECONDS


class PopSignerProtocol(Protocol):
    """what the client needs from a proof-of-possession minter.

    the SDK's ``PopSigner`` has exactly this method, so a tool pod hands its
    existing signer in and nothing here mints a second kind of proof. the
    proof binds THIS token to THIS call body, fresh and single-use, and the
    registry recomputes the body hash from the request it receives.
    """

    def sign(
        self,
        *,
        access_token: str,
        tool_name: str,
        arguments: dict[str, Any],
        correlation_id: str | None,
    ) -> str:
        """mint a proof binding the identity token to one call body.

        :param access_token: the identity token the proof is presented with
        :ptype access_token: str
        :param tool_name: the call's full tool name
        :ptype tool_name: str
        :param arguments: the call arguments
        :ptype arguments: dict[str, Any]
        :param correlation_id: the call correlation id, as the registry will read it
        :ptype correlation_id: str | None
        :return: a compact proof-of-possession JWS
        :rtype: str
        """
        ...


class ToolCallError(RuntimeError):
    """a call did not return the tool's reply, and this is why.

    one exception type for every non-success outcome -- the registry's refusal,
    the tool's own failure, a missing identity, a transport fault -- so a caller
    writes one ``except`` and branches on :attr:`error_code`. the registry's and
    the tool's codes ride through unchanged; the four the client mints itself are
    ``INVALID_TOOL_NAME`` and ``INVALID_TOOL_VERSION`` (refused before any
    publish), ``NO_IDENTITY_TOKEN`` (refused before any publish) and
    ``REQUEST_FAILED`` (the bus, not the registry, said no).

    :param error_code: the machine-readable code
    :ptype error_code: str
    :param message: the human-readable explanation
    :ptype message: str
    """

    def __init__(self, error_code: str, message: str) -> None:
        """bind the code beside the message.

        :param error_code: the machine-readable code
        :ptype error_code: str
        :param message: the human-readable explanation
        :ptype message: str
        """
        super().__init__(f"{error_code}: {message}")
        self.error_code = error_code


class ToolCallClient:
    """publishes tool calls on the registry's subject and returns the tool's reply.

    built once per process over the connected canonical NATS client, the
    process's principal id, its identity token PROVIDER and its proof signer,
    then called per tool call. the subject is built through
    :meth:`threetears.nats.Subjects.tools_call`, which reads the namespace the
    NATS client bound at connect, so the client and the registry derive the
    identical subject from the identical source.

    :param nats_client: the connected canonical NATS wrapper client
    :ptype nats_client: NatsClient
    :param principal_id: this process's principal id, the ``sub`` the hub
        minted its token for; a tool pod's ``tool_pods.id``. carried on the
        envelope because the registry routes on it before verifying, and
        overwritten by the registry with the token's ``sub`` after
    :ptype principal_id: UUID
    :param identity_token: zero-arg provider returning this process's CURRENT
        hub-minted identity token, read on every call; the holder's bound
        ``get``, never a captured string
    :ptype identity_token: Callable[[], str | None]
    :param pop_signer: minter of the per-call proof of possession over the
        holder key the token's ``cnf`` is bound to
    :ptype pop_signer: PopSignerProtocol
    :param timeout: how long one call may take before the client gives up;
        keep it above the registry's forward budget
    :ptype timeout: timedelta | float
    """

    def __init__(
        self,
        nats_client: NatsClient,
        *,
        principal_id: UUID,
        identity_token: Callable[[], str | None],
        pop_signer: PopSignerProtocol,
        timeout: timedelta | float = DEFAULT_CALL_TIMEOUT_SECONDS,
    ) -> None:
        """bind the transport, the principal, the identity provider, the signer and the deadline.

        :param nats_client: the connected canonical NATS wrapper client
        :ptype nats_client: NatsClient
        :param principal_id: this process's principal id
        :ptype principal_id: UUID
        :param identity_token: zero-arg provider returning the current token
        :ptype identity_token: Callable[[], str | None]
        :param pop_signer: minter of the per-call proof of possession
        :ptype pop_signer: PopSignerProtocol
        :param timeout: per-call deadline; a number is seconds
        :ptype timeout: timedelta | float
        """
        self._nats_client = nats_client
        self._principal_id = principal_id
        self._identity_token = identity_token
        self._pop_signer = pop_signer
        self._timeout = timeout if isinstance(timeout, timedelta) else timedelta(seconds=float(timeout))

    def forwarded_identity_token(self) -> str:
        """the CURRENT hub-minted identity token to forward on a call.

        read through the provider on every call, never cached: the token is
        short-lived and re-minted by the refresh loop, so a value captured once
        is expired within the hour.

        :return: the caller's current identity token
        :rtype: str
        :raises ToolCallError: with code ``NO_IDENTITY_TOKEN`` when the
            provider returns nothing; the registry refuses an empty token
            exactly as a missing one, so failing here names the wiring gap
            instead of sending a request that cannot be authorized
        """
        token = self._identity_token()
        if not token:
            raise ToolCallError(
                "NO_IDENTITY_TOKEN",
                "the identity_token provider returned no token. The handshake has not "
                "completed or its result was not threaded through; the registry refuses an "
                "empty token exactly as a missing one, so the request is not sent.",
            )
        return token

    @traced
    async def call(
        self,
        tool_name: str,
        tool_version: str,
        arguments: dict[str, Any] | None = None,
        *,
        correlation_id: UUID | None = None,
    ) -> ProxyCallResponse:
        """call one tool through the registry and return its reply.

        :param tool_name: the tool's full ``mcp_name``, as it registered
        :ptype tool_name: str
        :param tool_version: the tool's ``mcp_version``
        :ptype tool_version: str
        :param arguments: the tool's input, JSON-native values only; ``None``
            is an empty input
        :ptype arguments: dict[str, Any] | None
        :param correlation_id: trace id for this call; minted when omitted
        :ptype correlation_id: UUID | None
        :return: the reply for a call the tool answered successfully, carrying
            the tool's ``content`` and ``metadata``; its ``context`` is whatever
            the answering side stamped, not a re-statement of the caller's
            verified identity
        :rtype: ProxyCallResponse
        :raises ToolCallError: when the tool name or version is empty, when no
            identity token is available, when the registry refuses (its code
            rides on the exception), when the tool itself answers a failure
            (the tool's code rides on the exception), or when the bus fails to
            deliver a decodable reply
        """
        if not tool_name:
            raise ToolCallError(
                "INVALID_TOOL_NAME",
                "a tool call needs the tool's mcp_name; an empty name names no tool",
            )
        if not tool_version:
            raise ToolCallError(
                "INVALID_TOOL_VERSION",
                "a tool call needs the tool's mcp_version; an empty version names no tool",
            )
        token = self.forwarded_identity_token()
        effective_correlation_id = correlation_id if correlation_id is not None else uuid7()
        effective_arguments = dict(arguments) if arguments else {}
        # the proof binds the body EXACTLY as the registry will recompute it: the tool name, the
        # arguments as sent, and the correlation id as the string the registry derives from the
        # deserialized context. a mismatch on any of the three is a spliced proof and is refused.
        pop = self._pop_signer.sign(
            access_token=token,
            tool_name=tool_name,
            arguments=effective_arguments,
            correlation_id=str(effective_correlation_id),
        )
        request = ProxyCallRequest(
            tool_name=tool_name,
            tool_version=tool_version,
            arguments=effective_arguments,
            context=CallContext(
                agent_id=self._principal_id,
                correlation_id=effective_correlation_id,
                identity_token=token,
            ),
            pop=pop,
        )
        log.info(
            "tool call sent",
            extra={
                "extra_data": {
                    "tool_name": tool_name,
                    "tool_version": tool_version,
                    "correlation_id": f"{effective_correlation_id}",
                    "principal_id": f"{self._principal_id}",
                }
            },
        )
        try:
            response: ProxyCallResponse = await self._nats_client.request(
                subject=Subjects.tools_call(),
                message=request,
                response_type=ProxyCallResponse,
                timeout=self._timeout,
            )
        except RequestError as exc:
            log.warning(
                "tool call did not complete",
                extra={
                    "extra_data": {
                        "tool_name": tool_name,
                        "tool_version": tool_version,
                        "correlation_id": f"{effective_correlation_id}",
                        "error": str(exc),
                    }
                },
            )
            raise ToolCallError("REQUEST_FAILED", f"tool call {tool_name}@{tool_version}: {exc}") from exc

        if not response.success:
            error_code = response.error_code or "UNKNOWN"
            log.info(
                "tool call refused",
                extra={
                    "extra_data": {
                        "tool_name": tool_name,
                        "tool_version": tool_version,
                        "correlation_id": f"{effective_correlation_id}",
                        "error_code": error_code,
                    }
                },
            )
            raise ToolCallError(
                error_code,
                response.error or f"tool call {tool_name}@{tool_version} refused",
            )
        return response
