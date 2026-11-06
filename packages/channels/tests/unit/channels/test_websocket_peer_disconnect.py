"""a peer closing its socket is how a connection ends, not a failure this server absorbed.

``contain()`` records a contained error: a WARNING with a traceback and a count on
``contained_errors_total``. a browser tab closing, a laptop sleeping or a network dropping
ends every connection sooner or later, so counting those buried the real contained errors on
the dashboard under a steady trickle of ordinary departures. an exception that is not the
peer leaving is still contained.
"""

from __future__ import annotations

import json
import logging

import pytest
from prometheus_client import REGISTRY
from starlette.websockets import WebSocketDisconnect

from threetears.channels.websocket import WebSocketHandler

from .websocket_support import EchoRouter, MockWebSocket, valid_auth

_LOGGER = "threetears.channels.websocket"


class _LeavingWebSocket(MockWebSocket):
    """a socket whose peer leaves, with ``failure``, once its queued frames are read.

    :param failure: what ``receive_text`` raises once the queue is empty
    :ptype failure: Exception
    :param messages: frames read before the peer leaves
    :ptype messages: list[str] | None
    :param query_params: the connection's query parameters
    :ptype query_params: dict[str, str] | None
    """

    def __init__(
        self,
        failure: Exception,
        messages: list[str] | None = None,
        query_params: dict[str, str] | None = None,
    ) -> None:
        super().__init__(messages=messages, query_params=query_params)
        self._failure = failure

    async def receive_text(self) -> str:
        """the next queued frame, or the peer's departure.

        :return: the next frame
        :rtype: str
        :raises Exception: ``failure``, once no frame is left
        """
        if not self.messages:
            raise self._failure
        return self.messages.pop(0)


def _contained(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """the records ``contain()`` wrote from the websocket module.

    :param caplog: the captured records
    :ptype caplog: pytest.LogCaptureFixture
    :return: every contained-error record from the module
    :rtype: list[logging.LogRecord]
    """
    return [
        record
        for record in caplog.records
        if record.name == _LOGGER and "contained_site" in (getattr(record, "extra_data", None) or {})
    ]


def _tracebacks(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """the module's records that carry a traceback.

    :param caplog: the captured records
    :ptype caplog: pytest.LogCaptureFixture
    :return: every record from the module with ``exc_info``
    :rtype: list[logging.LogRecord]
    """
    return [record for record in caplog.records if record.name == _LOGGER and record.exc_info]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [1000, 1001, 1005, 1006])
async def test_a_peer_closing_after_it_connected_is_no_contained_error(
    caplog: pytest.LogCaptureFixture, code: int
) -> None:
    """a peer that leaves mid-session, by any close, leaves no contained error and no traceback."""
    ws = _LeavingWebSocket(
        WebSocketDisconnect(code=code),
        messages=[json.dumps({"type": "message", "content": "hello", "metadata": {}})],
        query_params={"token": "valid-token"},
    )
    handler = WebSocketHandler(router=EchoRouter(), auth_validator=valid_auth)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await handler.handle_connection(ws)

    assert _contained(caplog) == []
    assert _tracebacks(caplog) == []
    assert ws not in handler.registry.get_connections("user-123")


@pytest.mark.asyncio
async def test_a_peer_closing_before_it_authenticated_is_no_contained_error(caplog: pytest.LogCaptureFixture) -> None:
    """a peer that leaves before sending its token is refused as before, but counted as nothing."""
    ws = _LeavingWebSocket(WebSocketDisconnect(code=1001))
    handler = WebSocketHandler(router=EchoRouter(), auth_validator=valid_auth)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await handler.handle_connection(ws)

    assert _contained(caplog) == []
    assert _tracebacks(caplog) == []
    assert ws.close_code == 1008


@pytest.mark.asyncio
async def test_a_failed_read_that_is_not_the_peer_leaving_is_still_contained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """a read that fails for any other reason is still a contained error, with its traceback."""
    ws = _LeavingWebSocket(RuntimeError("the transport broke"), query_params={"token": "valid-token"})
    handler = WebSocketHandler(router=EchoRouter(), auth_validator=valid_auth)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await handler.handle_connection(ws)

    contained = _contained(caplog)
    assert len(contained) == 1
    assert contained[0].levelno == logging.WARNING
    assert contained[0].exc_info is not None
    assert isinstance(contained[0].exc_info[1], RuntimeError)
    assert ws not in handler.registry.get_connections("user-123")


@pytest.mark.asyncio
async def test_a_failed_read_before_authentication_is_still_contained(caplog: pytest.LogCaptureFixture) -> None:
    """a read during authentication that fails for any other reason is still contained."""
    ws = _LeavingWebSocket(RuntimeError("the transport broke"))
    handler = WebSocketHandler(router=EchoRouter(), auth_validator=valid_auth)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await handler.handle_connection(ws)

    contained = _contained(caplog)
    assert len(contained) == 1
    assert contained[0].exc_info is not None
    assert isinstance(contained[0].exc_info[1], RuntimeError)
    assert ws.close_code == 1008


@pytest.mark.asyncio
async def test_a_peer_closing_with_a_protocol_error_is_still_contained(caplog: pytest.LogCaptureFixture) -> None:
    """a close that reports something went wrong (1002, a protocol error) is not an ordinary departure."""
    ws = _LeavingWebSocket(WebSocketDisconnect(code=1002), query_params={"token": "valid-token"})
    handler = WebSocketHandler(router=EchoRouter(), auth_validator=valid_auth)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await handler.handle_connection(ws)

    assert len(_contained(caplog)) == 1


_DISCONNECTS = "threetears_websocket_peer_disconnects_total"


def _disconnects(close_code: str, phase: str) -> float:
    """how many peer disconnects the process has counted with these labels.

    :param close_code: the ``close_code`` label
    :ptype close_code: str
    :param phase: the ``phase`` label
    :ptype phase: str
    :return: the counter's value, 0 before its first count
    :rtype: float
    """
    value = REGISTRY.get_sample_value(_DISCONNECTS, {"close_code": close_code, "phase": phase})
    return value if value is not None else 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [1000, 1006, 1002])
async def test_a_peer_disconnect_is_counted_by_its_close_code(code: int) -> None:
    """every disconnect is counted by close code, so a mass abnormal drop (1006) shows on a dashboard."""
    before = _disconnects(str(code), "connected")
    ws = _LeavingWebSocket(WebSocketDisconnect(code=code), query_params={"token": "valid-token"})
    await WebSocketHandler(router=EchoRouter(), auth_validator=valid_auth).handle_connection(ws)

    assert _disconnects(str(code), "connected") == before + 1


@pytest.mark.asyncio
async def test_a_peer_disconnect_before_authenticating_is_counted_in_its_phase() -> None:
    """a peer gone before it sent its token is counted under the authenticating phase."""
    before = _disconnects("1001", "authenticating")
    await WebSocketHandler(router=EchoRouter(), auth_validator=valid_auth).handle_connection(
        _LeavingWebSocket(WebSocketDisconnect(code=1001))
    )

    assert _disconnects("1001", "authenticating") == before + 1


@pytest.mark.asyncio
async def test_an_application_close_code_is_counted_as_one_bounded_value() -> None:
    """a code a client application chose (4000-4999) is counted as ``4xxx``, keeping the series bounded."""
    before = _disconnects("4xxx", "connected")
    ws = _LeavingWebSocket(WebSocketDisconnect(code=4321), query_params={"token": "valid-token"})
    await WebSocketHandler(router=EchoRouter(), auth_validator=valid_auth).handle_connection(ws)

    assert _disconnects("4xxx", "connected") == before + 1
    assert _disconnects("4321", "connected") == 0.0
