"""query a hub-served datasource over NATS, on the caller's own identity.

**Why this exists.** A process that holds no warehouse credential -- a tool pod
above all -- reaches a datasource the way it reaches L3: it publishes a request
on a subject the hub answers, carrying the hub-minted identity token it obtained
at handshake, and the hub decides. The hub verifies the token at the door, pins
the principal from the SIGNED ``sub``, evaluates that principal's grant on the
datasource's namespace, runs the query on the driver it already holds, and
replies with rows or a typed refusal. Nothing on this wire names a principal, so
a caller can be authorized only as whoever the hub minted the token for.

**Identity is a PROVIDER, never a value.** The token is short-lived and re-minted
in place by the holder's refresh loop, so a string captured at construction is
expired within the hour and every query after that is refused by a client that
looks correctly wired. The provider is read on EVERY call, exactly as
:class:`threetears.core.backends.nats_proxy.NatsProxyL3Backend` reads its own.
An empty or missing token is refused HERE, before the bus sees the request,
because the hub refuses it identically and a refusal that names the wiring gap
beats one that reads as a permission problem.

**One wire model, both ends.** The hub's responder imports these same classes,
so the two sides cannot drift apart the way a hand-copied mirror does. The
request forbids unknown fields for the reason the L3 request models do: a
stale client sending ``agent_id`` beside a valid token must be refused at the
border, not silently authorized as the token's principal.

**Errors carry the hub's code.** A caller branches on
:attr:`DatasourceQueryError.error_code` and never on the message, which is for
a human and may be reworded at any time. ``ACCESS_DENIED`` is the one a consumer
most needs to tell apart from a transport fault: the first is a grant to fix,
the second is a bus to look at.

**No markdown, no honesty imperatives.** The datasource TOOL renders rows for a
model and annotates what is missing so the model cannot invent it. This client
serves a program, which gets the rows and nothing else. A program that wants the
model-facing rendering calls the tool.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid7

from pydantic import BaseModel, ConfigDict, Field
from threetears.nats.errors import RequestError
from threetears.nats.subjects import Subjects
from threetears.observe import get_logger, traced

if TYPE_CHECKING:
    from threetears.nats import NatsClient

__all__ = [
    "DEFAULT_QUERY_TIMEOUT_SECONDS",
    "DatasourceQueryClient",
    "DatasourceQueryError",
    "DatasourceQueryRequest",
    "DatasourceQueryResponse",
]

log = get_logger(__name__)


#: how long one query may take, end to end, before the client gives up.
#:
#: longer than the wrapper's default request deadline because a warehouse read
#: is not a control-plane round trip: the hub's own read tool budgets minutes,
#: and a client that times out first leaves the hub finishing a query nobody
#: will collect. shorter than the hub's read statement timeout so the client
#: still reports a stuck warehouse as a timeout rather than hanging with it.
DEFAULT_QUERY_TIMEOUT_SECONDS: float = 120.0


class DatasourceQueryRequest(BaseModel):
    """one datasource query, carrying the caller's identity as a forwarded token.

    extras are FORBIDDEN, and that is a security property rather than a style
    choice. ``agent_id`` and ``user_id`` are deliberately absent: a request that
    carried them would be one whose ACL check a caller could steer. Pydantic's
    default of ignoring an unknown field would let a stale client send one,
    have it dropped, and be authorized as the token's principal with no error
    anywhere -- broader authority than it asked for, silently. Refusing turns
    that into a border failure the caller sees.

    :param correlation_id: trace id echoed back on the reply and bound into the
        hub's logs, so one query can be followed from the caller to the driver
    :ptype correlation_id: UUID
    :param identity_token: the caller's hub-minted identity token, forwarded
        verbatim; the hub verifies it and takes the principal off the signed
        claims
    :ptype identity_token: str
    :param user_identity_token: the per-turn hub-minted user assertion when a
        human is in the loop; ``None`` for a call with nobody's behalf to act
        on, in which case the hub evaluates the principal's own grants alone
    :ptype user_identity_token: str | None
    :param query: the sql to run; the hub admits SELECT and nothing else
    :ptype query: str
    :param params: positional query parameters, JSON-native values only
    :ptype params: list[Any]
    """

    model_config = ConfigDict(extra="forbid")

    correlation_id: UUID
    identity_token: str
    user_identity_token: str | None = None
    query: str
    params: list[Any] = Field(default_factory=list)


class DatasourceQueryResponse(BaseModel):
    """the hub's answer: rows, or a refusal with a code a caller branches on.

    :param success: whether the query ran
    :ptype success: bool
    :param rows: result rows, one dict per row, empty on refusal
    :ptype rows: list[dict[str, Any]]
    :param row_count: how many rows ``rows`` carries
    :ptype row_count: int
    :param error_code: the machine-readable refusal code, ``None`` on success
    :ptype error_code: str | None
    :param error_message: the human-readable refusal, never parsed
    :ptype error_message: str | None
    :param correlation_id: the request's id, echoed
    :ptype correlation_id: UUID
    """

    success: bool
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    correlation_id: UUID


class DatasourceQueryError(RuntimeError):
    """a query did not return rows, and this is why.

    one exception type for every non-row outcome -- the hub's refusal, a
    missing identity, a transport fault -- so a caller writes one ``except`` and
    branches on :attr:`error_code`. the hub's own codes ride through unchanged;
    the two the client mints itself are ``NO_IDENTITY_TOKEN`` (refused before
    any publish) and ``REQUEST_FAILED`` (the bus, not the hub, said no).

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


class DatasourceQueryClient:
    """publishes datasource queries on the hub's subject and returns the rows.

    built once per process over the connected canonical NATS client and the
    process's identity token PROVIDER, then called per query. the subject is
    built through :meth:`threetears.nats.Subjects.datasource_query`, which reads
    the namespace the NATS client bound at connect, so the client and the hub's
    responder derive the identical subject from the identical source.

    :param nats_client: the connected canonical NATS wrapper client
    :ptype nats_client: NatsClient
    :param identity_token: zero-arg provider returning this process's CURRENT
        hub-minted identity token, read on every call; the holder's bound
        ``get``, never a captured string
    :ptype identity_token: Callable[[], str | None]
    :param timeout: how long one query may take before the client gives up
    :ptype timeout: timedelta | float
    """

    def __init__(
        self,
        nats_client: NatsClient,
        *,
        identity_token: Callable[[], str | None],
        timeout: timedelta | float = DEFAULT_QUERY_TIMEOUT_SECONDS,
    ) -> None:
        """bind the transport, the identity provider and the deadline.

        :param nats_client: the connected canonical NATS wrapper client
        :ptype nats_client: NatsClient
        :param identity_token: zero-arg provider returning the current token
        :ptype identity_token: Callable[[], str | None]
        :param timeout: per-query deadline; a number is seconds
        :ptype timeout: timedelta | float
        """
        self._nats_client = nats_client
        self._identity_token = identity_token
        self._timeout = timeout if isinstance(timeout, timedelta) else timedelta(seconds=float(timeout))

    def forwarded_identity_token(self) -> str:
        """the CURRENT hub-minted identity token to forward on a query.

        read through the provider on every call, never cached: the token is
        short-lived and re-minted by the refresh loop, so a value captured once
        is expired within the hour.

        :return: the caller's current identity token
        :rtype: str
        :raises DatasourceQueryError: with code ``NO_IDENTITY_TOKEN`` when the
            provider returns nothing; the hub refuses an empty token exactly as
            a missing one, so failing here names the wiring gap instead of
            sending a request that cannot be authorized
        """
        token = self._identity_token()
        if not token:
            raise DatasourceQueryError(
                "NO_IDENTITY_TOKEN",
                "the identity_token provider returned no token. The handshake has not "
                "completed or its result was not threaded through; the hub refuses an "
                "empty token exactly as a missing one, so the request is not sent.",
            )
        return token

    @traced
    async def query(
        self,
        datasource_name: str,
        query: str,
        params: list[Any] | None = None,
        *,
        user_identity_token: str | None = None,
        correlation_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """run one read query against a datasource and return its rows.

        :param datasource_name: the datasource's name as the hub's ``datasources``
            table holds it
        :ptype datasource_name: str
        :param query: the sql to run; the hub admits SELECT and nothing else
        :ptype query: str
        :param params: positional query parameters, JSON-native values only
        :ptype params: list[Any] | None
        :param user_identity_token: the per-turn user assertion when a human is
            in the loop; ``None`` evaluates the principal's own grants alone
        :ptype user_identity_token: str | None
        :param correlation_id: trace id for this query; minted when omitted
        :ptype correlation_id: UUID | None
        :return: result rows, one dict per row
        :rtype: list[dict[str, Any]]
        :raises DatasourceQueryError: when no identity token is available, when
            the hub refuses (its code rides on the exception), or when the bus
            fails to deliver a decodable reply
        """
        request = DatasourceQueryRequest(
            correlation_id=correlation_id if correlation_id is not None else uuid7(),
            identity_token=self.forwarded_identity_token(),
            user_identity_token=user_identity_token,
            query=query,
            params=list(params) if params is not None else [],
        )
        subject = Subjects.datasource_query(datasource_name)
        log.info(
            "datasource query sent",
            extra={
                "extra_data": {
                    "datasource": datasource_name,
                    "correlation_id": f"{request.correlation_id}",
                    "user_in_loop": user_identity_token is not None,
                }
            },
        )
        try:
            response: DatasourceQueryResponse = await self._nats_client.request(
                subject=subject,
                message=request,
                response_type=DatasourceQueryResponse,
                timeout=self._timeout,
            )
        except RequestError as exc:
            log.warning(
                "datasource query did not complete",
                extra={
                    "extra_data": {
                        "datasource": datasource_name,
                        "correlation_id": f"{request.correlation_id}",
                        "error": str(exc),
                    }
                },
            )
            raise DatasourceQueryError("REQUEST_FAILED", f"datasource query on {datasource_name!r}: {exc}") from exc

        result: list[dict[str, Any]]
        if response.success:
            result = response.rows
        else:
            error_code = response.error_code or "UNKNOWN"
            log.info(
                "datasource query refused",
                extra={
                    "extra_data": {
                        "datasource": datasource_name,
                        "correlation_id": f"{request.correlation_id}",
                        "error_code": error_code,
                    }
                },
            )
            raise DatasourceQueryError(
                error_code,
                response.error_message or f"datasource query on {datasource_name!r} refused",
            )
        return result
