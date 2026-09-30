"""closing one connection through the system account: what each server answer means.

The answers are the ones nats-server 2.12.6 and 2.14.2 give (``test_connection_kick_live.py`` pins
them against both); here they are fed to :func:`kick_connection` directly, so every branch -- the
kick, the connection already gone, the server gone, and the answers that leave the outcome unknown
-- is held to its meaning without a server.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from pydantic import ValidationError

from threetears.nats import (
    DEFAULT_KICK_TIMEOUT,
    NO_SUCH_CLIENT_DESCRIPTION,
    ConnectionKickError,
    KickOutcome,
    NatsConnectionRef,
    NoRespondersError,
    RequestTimeoutError,
    Subject,
    kick_connection,
    kick_subject,
)

_SERVER = "NCSS55HWYWEVBLVURKBUGDYA6LYFUEKJ7W3SESLZEHJ5XH3IXXVKYTJD"
_SERVER_INFO = {"name": _SERVER, "host": "0.0.0.0", "id": _SERVER, "ver": "2.14.2"}  # noqa: S104 - a server's own report of its bind address


# parity-exempt: stands in for NatsClient.request_raw alone, the one call a kick makes; it records the request and returns the scripted answer
class _FakeSystemClient:
    """answers the kick with a scripted reply, or raises a scripted error."""

    def __init__(self, *, reply: bytes | None = None, error: Exception | None = None) -> None:
        self._reply = reply
        self._error = error
        self.requests: list[tuple[str, bytes, timedelta]] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        self.requests.append((subject.path, payload, timeout))
        if self._error is not None:
            raise self._error
        assert self._reply is not None
        return self._reply


def _connection() -> NatsConnectionRef:
    return NatsConnectionRef(server_id=_SERVER, client_id=7)


class TestAConnectionIsNamedByItsServer:
    def test_the_kick_goes_to_the_server_that_holds_the_connection(self) -> None:
        assert kick_subject(_SERVER).path == f"$SYS.REQ.SERVER.{_SERVER}.KICK"

    @pytest.mark.parametrize("server_id", ["", "NSERVER", f"{_SERVER}.X", "*", ">", _SERVER.lower()])
    def test_a_value_that_is_not_a_server_id_names_no_subject(self, server_id: str) -> None:
        """it is spliced into a subject, so anything but a server nkey is refused before it is."""
        with pytest.raises(ValidationError):
            kick_subject(server_id)

    def test_a_client_id_is_positive(self) -> None:
        with pytest.raises(ValidationError):
            NatsConnectionRef(server_id=_SERVER, client_id=0)


class TestWhatAKickAnswerMeans:
    async def test_a_kick_the_server_carried_out(self) -> None:
        client = _FakeSystemClient(reply=json.dumps({"server": _SERVER_INFO}).encode())
        outcome = await kick_connection(client, _connection())  # type: ignore[arg-type]
        assert outcome is KickOutcome.KICKED
        subject, payload, timeout = client.requests[0]
        assert subject == f"$SYS.REQ.SERVER.{_SERVER}.KICK"
        assert json.loads(payload) == {"cid": 7}
        assert timeout == DEFAULT_KICK_TIMEOUT

    async def test_a_connection_the_server_no_longer_holds(self) -> None:
        reply = {"server": _SERVER_INFO, "error": {"code": 500, "description": NO_SUCH_CLIENT_DESCRIPTION}}
        client = _FakeSystemClient(reply=json.dumps(reply).encode())
        assert await kick_connection(client, _connection()) is KickOutcome.NOT_CONNECTED  # type: ignore[arg-type]

    async def test_a_server_that_is_gone(self) -> None:
        """a server id is fresh on every start: no responder means the connection died with it."""
        client = _FakeSystemClient(error=NoRespondersError("no responders"))
        assert await kick_connection(client, _connection()) is KickOutcome.SERVER_GONE  # type: ignore[arg-type]

    async def test_no_answer_in_time_leaves_the_outcome_unknown(self) -> None:
        client = _FakeSystemClient(error=RequestTimeoutError("timed out"))
        with pytest.raises(ConnectionKickError, match="has no answer"):
            await kick_connection(client, _connection())  # type: ignore[arg-type]

    async def test_any_other_refusal_leaves_the_outcome_unknown(self) -> None:
        """only the one error meaning "already gone" is read as gone; anything else is retried."""
        reply = {"server": _SERVER_INFO, "error": {"code": 400, "description": "invalid json"}}
        client = _FakeSystemClient(reply=json.dumps(reply).encode())
        with pytest.raises(ConnectionKickError, match="was refused"):
            await kick_connection(client, _connection())  # type: ignore[arg-type]

    async def test_an_answer_that_is_not_a_server_reply_leaves_the_outcome_unknown(self) -> None:
        client = _FakeSystemClient(reply=b"not json")
        with pytest.raises(ConnectionKickError, match="not a server API response"):
            await kick_connection(client, _connection())  # type: ignore[arg-type]
