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

**Synchronous, on the caller's own inbox.** No durable reply is requested: the
caller-side reply family the registry's delivery gate admits is
``Subjects.tools_reply_agent_subtree``, which a tool pod is granted neither to
publish nor to subscribe, and the verbs this client exists for answer in
milliseconds. The client's default deadline sits above the registry's DEFAULT
forward budget on purpose, so a tool that runs past that budget comes back as
the registry's ``TOOL_TIMEOUT`` rather than as a transport fault; a tool whose
own declared ``timeout_seconds`` exceeds the default needs ``timeout`` raised to
match, or its slow answer reads as ``REQUEST_FAILED`` here.

**Updates while the call runs.** ``call(updates=sink)`` asks for the tool's progress
and log entries as it runs, and hands each to ``sink`` -- a
``threetears.mcp.CallContext`` is one, so an MCP face serving the call passes its own
context and its client gets ``notifications/progress`` for its own progress token.
The client subscribes its own ``Subjects.tools_updates`` subject for the call (the
registry relays there, and refuses one naming anyone else), asks for progress only
when the sink holds a progress token and for log entries only at the levels it
takes, drops what arrives out of order, and stops listening when the call answers.
Updates are best effort: one lost or late loses only itself.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID, uuid7

from pydantic import ValidationError

from threetears.core.tool_wire import UPDATE_LOG_LEVELS, CallContext, CallUpdate, CallUpdates, UpdateLogLevel
from threetears.nats.errors import NatsClientError, RequestError
from threetears.nats.subjects import Subject, Subjects
from threetears.observe import contain, get_logger, traced

from threetears.registry.config import PLATFORM_DEFAULT_CALL_TIMEOUT
from threetears.registry.proxy import ProxyCallRequest, ProxyCallResponse

if TYPE_CHECKING:
    from threetears.nats.transport import IncomingMessage, RawMessageCallback

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
#: back while the caller is still listening. a literal by design: it is not a
#: timeout of anything, it is the slack between two, and the hardcoded-timeout
#: gate allowlists it by name for that reason.
CALL_TIMEOUT_MARGIN_SECONDS: float = 10.0

#: how long one call may take, end to end, before the client gives up.
#:
#: the registry's platform default forward budget, taken from the config layer
#: that owns it, plus the margin above; a test pins the sum above
#: :func:`threetears.registry.config.get_call_timeout` so the two cannot drift
#: into the wrong order.
DEFAULT_CALL_TIMEOUT_SECONDS: float = PLATFORM_DEFAULT_CALL_TIMEOUT + CALL_TIMEOUT_MARGIN_SECONDS


class ProxyCallRequester(Protocol):
    """the one call the client makes of its NATS client: a typed request to the registry's call subject.

    :class:`~threetears.nats.NatsClient` satisfies it, and so does a scripted test double.
    """

    async def request(
        self,
        *,
        subject: Subject,
        message: ProxyCallRequest,
        response_type: type[ProxyCallResponse],
        timeout: timedelta,
    ) -> ProxyCallResponse:
        """send ``message`` and return the registry's typed reply."""
        ...


@runtime_checkable
class UpdatesSubscriber(Protocol):
    """what a call that asks for updates needs of the NATS client besides its request: a subscription for them.

    :class:`~threetears.nats.NatsClient` satisfies it.
    """

    async def subscribe(self, subject: Subject | str | None = None, *, cb: RawMessageCallback | None = None) -> Any:
        """subscribe ``cb`` to ``subject``; returns the handle :meth:`unsubscribe` takes."""
        ...

    async def unsubscribe(self, sub: Any) -> None:
        """end a subscription."""
        ...


class _CallUpdateSink(Protocol):
    """where a call's updates go: a ``threetears.mcp.CallContext`` is one.

    the sink says what its own client takes -- a progress token, log levels -- and the call asks the tool for that
    and no more.
    """

    @property
    def progress_token(self) -> str | int | None:
        """the client's progress token; ``None`` when it asked for no progress."""
        ...

    @property
    def log_levels(self) -> frozenset[UpdateLogLevel]:
        """the log levels the client takes; empty when it takes none."""
        ...

    async def report_progress(self, progress: float, total: float | None = None, message: str | None = None) -> None:
        """hand on one progress report."""
        ...

    async def log(self, level: UpdateLogLevel, data: Any, *, logger: str | None = None) -> None:
        """hand on one log entry."""
        ...


class _UpdateForwarder:
    """hands one call's updates to its sink, in order, never raising into the subscription.

    :param sink: the call's sink
    :ptype sink: _CallUpdateSink
    :param tool_name: the call's tool, for the log
    :ptype tool_name: str
    """

    def __init__(self, sink: _CallUpdateSink, *, tool_name: str) -> None:
        """bind the forwarder to one call.

        :param sink: the call's sink
        :ptype sink: _CallUpdateSink
        :param tool_name: the call's tool
        :ptype tool_name: str
        :return: nothing
        :rtype: None
        """
        self._sink = sink
        self._tool_name = tool_name
        self._last_seq = 0
        self._last_progress: float | None = None

    async def receive(self, msg: IncomingMessage) -> None:
        """hand one update to the sink, or drop it when it is unreadable, late or not asked for.

        :param msg: the relayed update
        :ptype msg: IncomingMessage
        :return: nothing
        :rtype: None
        """
        try:
            update = CallUpdate.model_validate_json(msg.data)
        except ValidationError:
            # NOSILENT: an update is best effort, and one a newer sender shaped differently loses only itself
            return
        if update.seq <= self._last_seq:
            return
        self._last_seq = update.seq
        try:
            if update.kind == "progress":
                await self._progress(update)
            elif update.level is not None and update.level in self._sink.log_levels:
                await self._sink.log(update.level, update.data, logger=update.logger)
        except Exception as exc:  # noqa: BLE001 -- the sink's failure (its client gone) loses this update only
            contain(
                exc,
                log=log,
                what="a call's update could not be handed on to its client",
                extra={"tool_name": self._tool_name, "seq": update.seq, "kind": update.kind},
            )

    async def _progress(self, update: CallUpdate) -> None:
        """hand on a progress report that exceeds the last, when the sink asked for progress.

        :param update: the progress update
        :ptype update: CallUpdate
        :return: nothing
        :rtype: None
        """
        progress = update.progress
        if self._sink.progress_token is None or progress is None:
            return
        if self._last_progress is not None and progress <= self._last_progress:
            return
        self._last_progress = progress
        await self._sink.report_progress(progress, update.total, update.message)


def _least_level(levels: frozenset[UpdateLogLevel]) -> UpdateLogLevel | None:
    """the least severe of ``levels``, or ``None`` when there are none.

    :param levels: the levels a sink takes
    :ptype levels: frozenset[UpdateLogLevel]
    :return: the least severe
    :rtype: UpdateLogLevel | None
    """
    for level in UPDATE_LOG_LEVELS:
        if level in levels:
            return level
    return None


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
    :ptype nats_client: ProxyCallRequester
    :param principal_id: this process's principal id, the ``sub`` the hub
        minted its token for; a tool pod's ``tool_pods.id``. carried on the
        envelope because the registry refuses a request whose context names no
        principal before it verifies anything, and overwritten by the registry
        with the token's ``sub`` once it has
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
        nats_client: ProxyCallRequester,
        *,
        principal_id: UUID,
        identity_token: Callable[[], str | None],
        pop_signer: PopSignerProtocol,
        timeout: timedelta | float = DEFAULT_CALL_TIMEOUT_SECONDS,
    ) -> None:
        """bind the transport, the principal, the identity provider, the signer and the deadline.

        :param nats_client: the connected canonical NATS wrapper client, or anything that makes its typed request
        :ptype nats_client: ProxyCallRequester
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

    def _forwarded_identity_token(self) -> str:
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
        updates: _CallUpdateSink | None = None,
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
        :param updates: where the tool's progress and log entries go while it runs; ``None`` asks for none, and so
            does a sink with no progress token and no log level. The NATS client must also subscribe
            (:class:`UpdatesSubscriber`) when one asks for anything
        :ptype updates: _CallUpdateSink | None
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
        token = self._forwarded_identity_token()
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
        subscription = await self._subscribe_updates(updates, tool_name)
        if subscription is not None:
            request = request.model_copy(update={"updates": subscription[1]})
        try:
            return await self._send(request, effective_correlation_id)
        finally:
            if subscription is not None:
                await self._unsubscribe_updates(subscription[0])

    async def _subscribe_updates(self, sink: _CallUpdateSink | None, tool_name: str) -> tuple[Any, CallUpdates] | None:
        """subscribe this call's own updates subject when ``sink`` asks for anything.

        :param sink: the call's sink
        :ptype sink: _CallUpdateSink | None
        :param tool_name: the call's tool
        :ptype tool_name: str
        :return: the subscription's handle and the request's ``updates``, or ``None`` when nothing is asked for or the
            subscription could not be made (the call then runs without updates, which are best effort)
        :rtype: tuple[Any, CallUpdates] | None
        :raises TypeError: the NATS client cannot subscribe
        """
        if sink is None:
            return None
        log_level = _least_level(sink.log_levels)
        if sink.progress_token is None and log_level is None:
            return None
        subscriber = self._nats_client
        if not isinstance(subscriber, UpdatesSubscriber):
            raise TypeError(f"a call that asks for updates needs a NATS client that subscribes; got {type(subscriber)}")
        subject = Subjects.tools_updates(self._principal_id, uuid7())
        forwarder = _UpdateForwarder(sink, tool_name=tool_name)
        try:
            # sender-posture: unauthenticated open N2 -- only the registry is granted the subject, whose per-call id
            # only this caller and the registry hold; an update is checked for shape and order and acts on nothing
            handle = await subscriber.subscribe(subject=subject, cb=forwarder.receive)
        except NatsClientError as exc:
            contain(
                exc,
                log=log,
                what="a call's updates could not be subscribed; it runs without them",
                extra={"tool_name": tool_name, "subject": subject.path},
            )
            return None
        return handle, CallUpdates(subject=subject.path, progress_token=sink.progress_token, log_level=log_level)

    async def _unsubscribe_updates(self, handle: Any) -> None:
        """end a call's updates subscription; a failure is logged, the call's answer kept.

        :param handle: the subscription
        :ptype handle: Any
        :return: nothing
        :rtype: None
        """
        subscriber = self._nats_client
        assert isinstance(subscriber, UpdatesSubscriber)  # held by _subscribe_updates
        try:
            await subscriber.unsubscribe(handle)
        except NatsClientError as exc:
            contain(exc, log=log, what="a call's updates subscription could not be ended", extra={})

    async def _send(self, request: ProxyCallRequest, effective_correlation_id: UUID) -> ProxyCallResponse:
        """send the call and read the registry's reply.

        :param request: the call
        :ptype request: ProxyCallRequest
        :param effective_correlation_id: its correlation id
        :ptype effective_correlation_id: UUID
        :return: the reply for a call the tool answered successfully
        :rtype: ProxyCallResponse
        :raises ToolCallError: the registry or the tool refused, or the bus failed
        """
        tool_name = request.tool_name
        tool_version = request.tool_version
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
