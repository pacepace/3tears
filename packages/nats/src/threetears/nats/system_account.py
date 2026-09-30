"""close one NATS connection on demand, through the server's system account.

NATS has no way to take authority away from a live connection: a user JWT's permissions are fixed at
connect, and the server closes the connection only when that JWT's ``exp`` passes. The one
server-side lever is to close the connection outright -- ``$SYS.REQ.SERVER.<server_id>.KICK`` with
``{"cid": <client id>}`` -- after which the client reconnects and is authorized afresh, where an
auth-callout fence can refuse it. That request is served only to a user of the SYSTEM account, and
only by the one server the connection is attached to, so it names that server.

Both halves of the address arrive with every auth-callout request: the requesting server's id and
the connecting client's id (:attr:`threetears.nats.AuthCalloutRequest.connection`). A server id is a
fresh nkey on every server start and a client id comes from a per-server counter, so the pair names
exactly one connection, once.

Verified against nats-server 2.12.6 and 2.14.2 (``events.go`` ``kickClient`` ->
``server.go`` ``DisconnectClientByID``): a kicked connection is closed at once, sending the client no
``-ERR``; the reply to a kick of a client id the server does not hold is ``error.code`` 500 with
``"no such client or leafnode id"``, the only error ``DisconnectClientByID`` returns; and a server id
that no running server holds has no responder at all. ``packages/nats/tests/integration/
test_connection_kick_live.py`` pins all three on both versions.
"""

from __future__ import annotations

import re
from datetime import timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from threetears.nats.errors import NatsClientError, NoRespondersError, RequestError
from threetears.nats.subjects import Subject

if TYPE_CHECKING:
    from threetears.nats.client import NatsClient

__all__ = [
    "DEFAULT_KICK_TIMEOUT",
    "NO_SUCH_CLIENT_DESCRIPTION",
    "SERVER_PING_SUBJECT",
    "ConnectionKickError",
    "KickClientRequest",
    "KickOutcome",
    "NatsConnectionRef",
    "ServerApiError",
    "ServerApiResponse",
    "ServerIdentity",
    "SystemAccountUnavailableError",
    "kick_connection",
    "kick_subject",
    "require_system_account",
]

#: how long a kick waits for its server's answer. A kick is one in-memory lookup and a socket close
#: on the server that holds the connection; this bounds a server too loaded, or too partitioned, to
#: answer, so the caller can retry rather than wait on it.
DEFAULT_KICK_TIMEOUT: Final[timedelta] = timedelta(seconds=2)

#: the description nats-server answers a kick of a client id it does not hold with. It is the only
#: error ``DisconnectClientByID`` returns, so it means "that connection is already gone".
NO_SUCH_CLIENT_DESCRIPTION: Final[str] = "no such client or leafnode id"

#: the subject every server of the cluster answers for a user of the system account, and for no
#: other: a client of any other account finds no responder there.
SERVER_PING_SUBJECT: Final[str] = "$SYS.REQ.SERVER.PING"

#: a server id is a server nkey: ``N`` then base32. Checked before it is spliced into a subject, so a
#: value that is not one can never name a different subject than the kick of one server.
_SERVER_ID_GRAMMAR: Final[re.Pattern[str]] = re.compile(r"\AN[A-Z2-7]{55}\Z")


class NatsConnectionRef(BaseModel):
    """one live NATS connection, as the server that holds it names it.

    :param server_id: the id of the server the connection is attached to (a server nkey)
    :ptype server_id: str
    :param client_id: that server's id for the connection (``cid``)
    :ptype client_id: int
    """

    model_config = ConfigDict(frozen=True)

    server_id: str
    client_id: int = Field(ge=1)

    @field_validator("server_id")
    @classmethod
    def _server_id_is_a_server_nkey(cls, value: str) -> str:
        """refuse a server id that is not a server nkey.

        :param value: the candidate server id
        :ptype value: str
        :return: the server id unchanged
        :rtype: str
        :raises ValueError: when it is not ``N`` followed by 55 base32 characters
        """
        if not _SERVER_ID_GRAMMAR.match(value):
            raise ValueError(f"{value!r} is not a NATS server id (a server nkey, N followed by 55 base32 characters)")
        return value


class KickClientRequest(BaseModel):
    """the body of a ``$SYS.REQ.SERVER.<server_id>.KICK`` request.

    :param cid: the client id to close, on the server the request is addressed to
    :ptype cid: int
    """

    cid: int = Field(ge=1)


class ServerApiError(BaseModel):
    """the ``error`` a system-account server API reply carries when the request failed.

    :param code: an HTTP-style status the server chose
    :ptype code: int
    :param description: the server's text for it
    :ptype description: str
    """

    model_config = ConfigDict(extra="ignore")

    code: int
    description: str = ""


class ServerIdentity(BaseModel):
    """the ``server`` block every system-account reply opens with, read as far as its id.

    :param id: the answering server's id (a server nkey)
    :ptype id: str
    :param name: the server's configured name
    :ptype name: str
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str = ""


class ServerApiResponse(BaseModel):
    """a system-account server API reply, read only as far as a kick needs.

    :param server: the server that answered, when the reply names it
    :ptype server: ServerIdentity | None
    :param error: the failure, or ``None`` when the request succeeded
    :ptype error: ServerApiError | None
    """

    model_config = ConfigDict(extra="ignore")

    server: ServerIdentity | None = None
    error: ServerApiError | None = None


class KickOutcome(StrEnum):
    """what became of a kick. Every member means the connection no longer holds its credential."""

    #: the server held the connection and closed it.
    KICKED = "kicked"
    #: the server does not hold that client id: the connection had already closed.
    NOT_CONNECTED = "not_connected"
    #: no running server has that id. A server id is fresh on every start, so the server that held
    #: the connection has restarted or left, and the connection died with it.
    SERVER_GONE = "server_gone"


class SystemAccountUnavailableError(NatsClientError):
    """the client that should reach the system account does not.

    Raised by :func:`require_system_account`. A client logged in to any other account finds no
    responder on any ``$SYS.REQ.SERVER`` subject -- which a kick reads as "the server is gone" -- so a
    misconfigured login would make every kick look done while closing nothing. This is how that is
    caught before the first one.
    """


class ConnectionKickError(NatsClientError):
    """a kick whose outcome is unknown -- no answer in time, or an answer that is not one of ours.

    The connection may still be open, so the caller keeps the kick and tries it again.
    """


async def require_system_account(
    system_client: NatsClient,
    *,
    timeout: timedelta = DEFAULT_KICK_TIMEOUT,
) -> ServerIdentity:
    """prove ``system_client`` is logged in to the system account, by asking a server to answer.

    :param system_client: the client that kicks connections
    :ptype system_client: NatsClient
    :param timeout: how long to wait for a server to answer
    :ptype timeout: timedelta
    :return: the server that answered
    :rtype: ServerIdentity
    :raises SystemAccountUnavailableError: when no server answers for this client, or the answer
        names no server
    """
    subject = Subject.raw(SERVER_PING_SUBJECT)
    try:
        reply = await system_client.request_raw(subject=subject, payload=b"", timeout=timeout)
    except RequestError as exc:
        raise SystemAccountUnavailableError(
            f"no NATS server answered {SERVER_PING_SUBJECT} for this client ({exc}). Only a user of the "
            "server's SYSTEM account is answered there: check that the user it logs in as is in the "
            "system account's users and listed in auth_callout.auth_users, and that the server config "
            "names system_account. Until it is, no connection can be kicked."
        ) from exc
    try:
        response = ServerApiResponse.model_validate_json(reply)
    except ValidationError as exc:
        raise SystemAccountUnavailableError(
            f"{SERVER_PING_SUBJECT} answered with something that is not a server API response: {exc}"
        ) from exc
    if response.server is None:
        raise SystemAccountUnavailableError(f"{SERVER_PING_SUBJECT} answered without naming its server")
    return response.server


def kick_subject(server_id: str) -> Subject:
    """the system-account subject that kicks a connection on ``server_id``.

    :param server_id: the server that holds the connection
    :ptype server_id: str
    :return: ``$SYS.REQ.SERVER.<server_id>.KICK``
    :rtype: Subject
    :raises ValueError: when ``server_id`` is not a server nkey
    """
    ref = NatsConnectionRef(server_id=server_id, client_id=1)
    return Subject.raw(f"$SYS.REQ.SERVER.{ref.server_id}.KICK")


async def kick_connection(
    system_client: NatsClient,
    connection: NatsConnectionRef,
    *,
    timeout: timedelta = DEFAULT_KICK_TIMEOUT,
) -> KickOutcome:
    """close ``connection`` now, through ``system_client``'s system-account login.

    A kicked client sees its socket close and takes its ordinary reconnect path, which re-runs the
    server's authentication -- a kick removes a credential, and it is whatever authorizes the
    reconnect that decides whether the client comes back.

    :param system_client: a client connected as a user of the server's SYSTEM account; no other
        account can reach the kick subject, which then has no responder
    :ptype system_client: NatsClient
    :param connection: the connection to close
    :ptype connection: NatsConnectionRef
    :param timeout: how long to wait for the server's answer
    :ptype timeout: timedelta
    :return: :attr:`KickOutcome.KICKED`, or why there was nothing to kick
    :rtype: KickOutcome
    :raises ConnectionKickError: when the server did not answer in time, the request failed on the
        wire, or the answer is not one a kick produces -- the connection may still be open
    """
    subject = kick_subject(connection.server_id)
    payload = KickClientRequest(cid=connection.client_id).model_dump_json().encode("utf-8")
    reply: bytes | None = None
    try:
        reply = await system_client.request_raw(subject=subject, payload=payload, timeout=timeout)
    except NoRespondersError:
        # NOSILENT: no server holds this id any more, which is itself the answer (SERVER_GONE below)
        reply = None
    except RequestError as exc:
        raise ConnectionKickError(
            f"kick of client {connection.client_id} on server {connection.server_id} has no answer: {exc}"
        ) from exc
    outcome = KickOutcome.SERVER_GONE if reply is None else _read_kick_reply(reply, connection)
    return outcome


def _read_kick_reply(reply: bytes, connection: NatsConnectionRef) -> KickOutcome:
    """what a server's answer to a kick says became of the connection.

    :param reply: the server's answer
    :ptype reply: bytes
    :param connection: the connection the kick named
    :ptype connection: NatsConnectionRef
    :return: :attr:`KickOutcome.KICKED`, or :attr:`KickOutcome.NOT_CONNECTED`
    :rtype: KickOutcome
    :raises ConnectionKickError: when the answer is not a server API response, or reports a failure
        other than the connection being gone
    """
    try:
        response = ServerApiResponse.model_validate_json(reply)
    except ValidationError as exc:
        raise ConnectionKickError(
            f"kick of client {connection.client_id} on server {connection.server_id} answered with a reply "
            f"that is not a server API response: {exc}"
        ) from exc
    outcome = KickOutcome.KICKED
    if response.error is not None:
        if response.error.description != NO_SUCH_CLIENT_DESCRIPTION:
            raise ConnectionKickError(
                f"kick of client {connection.client_id} on server {connection.server_id} was refused: "
                f"code={response.error.code} description={response.error.description!r}"
            )
        outcome = KickOutcome.NOT_CONNECTED
    return outcome
