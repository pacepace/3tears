"""closing one connection, or asking after it, through the system account: what each answer means.

The answers are the ones nats-server 2.12.6 and 2.14.2 give (``test_connection_kick_live.py`` pins
them against both); here they are fed to :func:`kick_connection` directly, so every branch -- the
kick, the connection already gone, the server gone, and the answers that leave the outcome unknown
-- is held to its meaning without a server. The same holds for :func:`probe_connection`'s CONNZ
answers: held, not held, server gone, and the answers that prove nothing.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from pydantic import ValidationError

from threetears.nats import (
    DEFAULT_KICK_TIMEOUT,
    DEFAULT_PROBE_TIMEOUT,
    NO_SUCH_CLIENT_DESCRIPTION,
    ConnectionKickError,
    ConnectionProbeError,
    KickOutcome,
    NatsConnectionRef,
    NoRespondersError,
    ProbeOutcome,
    RequestTimeoutError,
    Subject,
    connz_subject,
    kick_connection,
    kick_subject,
    probe_connection,
)

_SERVER = "NCSS55HWYWEVBLVURKBUGDYA6LYFUEKJ7W3SESLZEHJ5XH3IXXVKYTJD"
_SERVER_INFO = {"name": _SERVER, "host": "0.0.0.0", "id": _SERVER, "ver": "2.14.2"}  # noqa: S104 - a server's own report of its bind address


# parity-exempt: stands in for NatsClient.request_raw alone, the one call a kick and a probe make; it records the request and returns the scripted answer
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


def _connz(*client_ids: int) -> bytes:
    """a CONNZ reply listing ``client_ids`` open, shaped as nats-server answers it.

    :param client_ids: the open connections the reply lists
    :ptype client_ids: int
    :return: the reply bytes
    :rtype: bytes
    """
    held = [{"cid": cid, "kind": "Client", "ip": "10.0.0.1", "port": 50000 + cid} for cid in client_ids]
    return json.dumps(
        {
            "server": _SERVER_INFO,
            "data": {"server_id": _SERVER, "num_connections": len(held), "total": len(held), "connections": held},
        }
    ).encode()


class TestWhatAProbeAnswerMeans:
    def test_the_probe_goes_to_the_server_that_holds_the_connection(self) -> None:
        assert connz_subject(_SERVER).path == f"$SYS.REQ.SERVER.{_SERVER}.CONNZ"

    @pytest.mark.parametrize("server_id", ["", "NSERVER", f"{_SERVER}.X", "*", ">", _SERVER.lower()])
    def test_a_value_that_is_not_a_server_id_names_no_subject(self, server_id: str) -> None:
        with pytest.raises(ValidationError):
            connz_subject(server_id)

    async def test_a_connection_the_server_lists_open_is_held(self) -> None:
        client = _FakeSystemClient(reply=_connz(7))
        outcome = await probe_connection(client, _connection())  # type: ignore[arg-type]
        assert outcome is ProbeOutcome.HELD
        subject, payload, timeout = client.requests[0]
        assert subject == f"$SYS.REQ.SERVER.{_SERVER}.CONNZ"
        assert json.loads(payload) == {"cid": 7}
        assert timeout == DEFAULT_PROBE_TIMEOUT

    async def test_a_connection_the_server_does_not_list_is_not_held(self) -> None:
        client = _FakeSystemClient(reply=_connz())
        assert await probe_connection(client, _connection()) is ProbeOutcome.NOT_HELD  # type: ignore[arg-type]

    async def test_a_listing_of_other_connections_only_is_not_held(self) -> None:
        """the filter is the server's; the client id is still checked, so a server that ignored it lies to no one."""
        client = _FakeSystemClient(reply=_connz(8, 9))
        assert await probe_connection(client, _connection()) is ProbeOutcome.NOT_HELD  # type: ignore[arg-type]

    async def test_a_server_that_is_gone(self) -> None:
        client = _FakeSystemClient(error=NoRespondersError("no responders"))
        assert await probe_connection(client, _connection()) is ProbeOutcome.SERVER_GONE  # type: ignore[arg-type]

    async def test_no_answer_in_time_proves_nothing(self) -> None:
        client = _FakeSystemClient(error=RequestTimeoutError("timed out"))
        with pytest.raises(ConnectionProbeError, match="has no answer"):
            await probe_connection(client, _connection())  # type: ignore[arg-type]

    async def test_a_refusal_proves_nothing(self) -> None:
        reply = {"server": _SERVER_INFO, "error": {"code": 500, "description": "boom"}}
        client = _FakeSystemClient(reply=json.dumps(reply).encode())
        with pytest.raises(ConnectionProbeError, match="was refused"):
            await probe_connection(client, _connection())  # type: ignore[arg-type]

    async def test_an_answer_without_data_proves_nothing(self) -> None:
        client = _FakeSystemClient(reply=json.dumps({"server": _SERVER_INFO}).encode())
        with pytest.raises(ConnectionProbeError, match="without data"):
            await probe_connection(client, _connection())  # type: ignore[arg-type]

    async def test_an_answer_that_is_not_a_connz_reply_proves_nothing(self) -> None:
        client = _FakeSystemClient(reply=b"not json")
        with pytest.raises(ConnectionProbeError, match="not a CONNZ response"):
            await probe_connection(client, _connection())  # type: ignore[arg-type]
