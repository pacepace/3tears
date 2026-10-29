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
