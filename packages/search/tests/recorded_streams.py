"""Recorded streams for the answer pins: a :class:`StreamingTransport` that replays the live captures.

An answers adapter is tested against what the provider really sent
(``answers-requirements-design.md`` AR-I2): :func:`recording` reads one of the
captures shipped as package data under ``threetears/search/testing/recordings/``
-- Brave answer streams and error envelopes, recorded once with a real key --
and :class:`RecordedStreamingTransport` replays a :class:`RecordedStream` chunk
by chunk, under the whole-stream cap the real transport keeps. A recording can
be re-cut at any chunk size, so a parser is pinned against boundaries that fall
mid-tag and mid-character, which the network chooses and a fixture must not
hide.

Test support, not public API: nothing in ``src`` replays a stream, so this
lives with the package's tests, the place the streaming-transport pins are
kept for the same reason.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from types import TracebackType
from typing import Self

from pydantic import JsonValue

from threetears.search.contracts import (
    EGRESS_DIRECT,
    LocalCapExceeded,
    Spend,
    StreamedResponse,
    StreamingTransport,
    TransportFailed,
)
from threetears.search.standalone import RESPONSE_BYTES_SCOPE

__all__ = ["RecordedStream", "RecordedStreamingTransport", "recording"]


def recording(name: str) -> bytes:
    """Read one shipped capture, as the provider sent it.

    Read through :mod:`importlib.resources` from the package data the wheel
    ships, so these pins read exactly the bytes an installed package holds.

    :param name: the capture's file name under ``recordings/``
        (``brave_answers_single.sse``)
    :ptype name: str
    :return: the recorded bytes
    :rtype: bytes
    :raises FileNotFoundError: when no capture has that name
    """
    return files("threetears.search.testing").joinpath("recordings", name).read_bytes()


@dataclass(frozen=True, slots=True)
class RecordedStream:
    """One recorded streamed answer: a status, headers, the body in chunks -- or a failure."""

    #: HTTP status.
    status_code: int = 200
    #: the body, in the chunks it is replayed as.
    chunks: tuple[bytes, ...] = ()
    #: response headers, lower-cased keys.
    headers: Mapping[str, str] = field(default_factory=lambda: {"content-type": "text/event-stream"})
    #: raised by ``stream`` instead of opening -- a failure before the first byte.
    failure: BaseException | None = None
    #: raised by the body once every chunk is read -- a stream broken after it began.
    broken: BaseException | None = None

    @classmethod
    def of(
        cls,
        body: bytes,
        *,
        chunk_size: int | None = None,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        broken: BaseException | None = None,
    ) -> RecordedStream:
        """Cut a recorded body into the chunks it is replayed as.

        :param body: the whole body
        :ptype body: bytes
        :param chunk_size: bytes per chunk; None replays one SSE event per
            chunk, as the provider sent them
        :ptype chunk_size: int | None
        :param status_code: HTTP status
        :ptype status_code: int
        :param headers: response headers; None is an event stream's
        :ptype headers: Mapping[str, str] | None
        :param broken: raised once the chunks are read, to break the stream
        :ptype broken: BaseException | None
        :return: the recorded stream
        :rtype: RecordedStream
        :raises ValueError: when ``chunk_size`` is not positive
        """
        if chunk_size is None:
            events = body.split(b"\n\n")
            chunks = tuple(event + b"\n\n" for event in events[:-1]) + ((events[-1],) if events[-1] else ())
        elif chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        else:
            chunks = tuple(body[at : at + chunk_size] for at in range(0, len(body), chunk_size))
        response_headers = dict(headers) if headers is not None else {"content-type": "text/event-stream"}
        return cls(status_code=status_code, chunks=chunks, headers=response_headers, broken=broken)


class RecordedResponse:
    """:class:`StreamedResponse` over one recording, under the whole-stream cap."""

    def __init__(self, recorded: RecordedStream, *, url: str, egress: str, max_bytes: int) -> None:
        """Hold the recording a stream replays.

        :param recorded: the recording
        :ptype recorded: RecordedStream
        :param url: the URL asked for
        :ptype url: str
        :param egress: the exit name the stream reports
        :ptype egress: str
        :param max_bytes: this call's cap on the whole body
        :ptype max_bytes: int
        """
        self._recorded = recorded
        self._url = url
        self._egress = egress
        self._max_bytes = max_bytes
        #: whether the stream has been released.
        self.closed = False

    @property
    def status_code(self) -> int:
        """The recorded status.

        :return: the status
        :rtype: int
        """
        return self._recorded.status_code

    @property
    def headers(self) -> Mapping[str, str]:
        """The recorded headers, lower-cased keys.

        :return: the headers
        :rtype: Mapping[str, str]
        """
        return {key.lower(): value for key, value in self._recorded.headers.items()}

    @property
    def final_url(self) -> str:
        """The URL asked for; a recording follows no redirect.

        :return: the URL
        :rtype: str
        """
        return self._url

    @property
    def egress(self) -> str:
        """The configured exit name.

        :return: the egress
        :rtype: str
        """
        return self._egress

    @property
    def attempts(self) -> int:
        """One attempt: a recording is never retried.

        :return: 1
        :rtype: int
        """
        return 1

    @property
    def elapsed_seconds(self) -> float:
        """A token figure.

        :return: elapsed seconds
        :rtype: float
        """
        return 0.01

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Replay the body.

        :return: the chunks
        :rtype: AsyncIterator[bytes]
        """
        return self._replay()

    async def _replay(self) -> AsyncIterator[bytes]:
        """Yield each chunk, refusing the one that would cross the cap, then break if recorded to.

        :return: the chunks
        :rtype: AsyncIterator[bytes]
        :raises LocalCapExceeded: before the chunk that would take the stream past its cap
        :raises TransportFailed: when the stream was closed before its body ended
        """
        received = 0
        for chunk in self._recorded.chunks:
            if self.closed:
                raise TransportFailed(f"stream from {self._url} was closed before its body ended", spend=Spend())
            if received + len(chunk) > self._max_bytes:
                raise LocalCapExceeded(
                    f"stream from {self._url} ran past this read's {self._max_bytes}-byte cap",
                    spend=Spend(bytes_transferred=received),
                    scope=RESPONSE_BYTES_SCOPE,
                )
            received += len(chunk)
            yield chunk
        if self._recorded.broken is not None:
            raise self._recorded.broken

    async def aclose(self) -> None:
        """Release the stream. Idempotent."""
        self.closed = True

    async def __aenter__(self) -> Self:
        """Enter the owning block.

        :return: this stream
        :rtype: Self
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release on leaving the block.

        :param exc_type: exception type, if the block raised
        :ptype exc_type: type[BaseException] | None
        :param exc: the exception, if the block raised
        :ptype exc: BaseException | None
        :param traceback: the traceback, if the block raised
        :ptype traceback: TracebackType | None
        """
        await self.aclose()


class RecordedStreamingTransport(StreamingTransport):  # parity-with: threetears.search.contracts.StreamingTransport
    """Replays recorded streams in order, under the real transport's whole-stream cap."""

    def __init__(self, streams: tuple[RecordedStream, ...], *, egress_name: str = EGRESS_DIRECT) -> None:
        """Load the recordings, answered in order; the last repeats once the rest are spent.

        :param streams: the recorded answers
        :ptype streams: tuple[RecordedStream, ...]
        :param egress_name: the exit name every stream reports
        :ptype egress_name: str
        :raises ValueError: when there is no recording to answer with
        """
        if not streams:
            raise ValueError("a recorded streaming transport needs at least one recording to answer with")
        self._streams = streams
        self._egress_name = egress_name
        #: every stream opened, in order: ``{"method", "url", "headers", "json_body",
        #: "timeout_seconds", "max_bytes"}``.
        self.calls: list[dict[str, object]] = []
        #: every stream handed out, in order, so a test can see each was released.
        self.opened: list[RecordedResponse] = []

    @property
    def egress_name(self) -> str:
        """The exit name every stream reports.

        :return: the configured name
        :rtype: str
        """
        return self._egress_name

    async def stream(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, JsonValue] | None = None,
        timeout_seconds: float | None = None,
        max_bytes: int,
    ) -> StreamedResponse:
        """Open the next recording, refusing what the real transport refuses.

        :param method: HTTP method, recorded
        :ptype method: str
        :param url: the URL asked for
        :ptype url: str
        :param headers: request headers, recorded
        :ptype headers: Mapping[str, str] | None
        :param json_body: the request body, recorded
        :ptype json_body: Mapping[str, JsonValue] | None
        :param timeout_seconds: this call's bound, recorded
        :ptype timeout_seconds: float | None
        :param max_bytes: this call's cap on the whole body
        :ptype max_bytes: int
        :return: the open recorded stream
        :rtype: StreamedResponse
        :raises ValueError: when ``max_bytes`` is not positive, as the real transport refuses it
        """
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers or {}),
                "json_body": dict(json_body or {}),
                "timeout_seconds": timeout_seconds,
                "max_bytes": max_bytes,
            }
        )
        recorded = self._streams[min(len(self.calls), len(self._streams)) - 1]
        if recorded.failure is not None:
            raise recorded.failure
        response = RecordedResponse(recorded, url=url, egress=self._egress_name, max_bytes=max_bytes)
        self.opened.append(response)
        return response
