"""The streaming transport's pins (AD11, AR-F1): what the seam owes, against a real socket.

The obligations of :class:`~threetears.search.contracts.StreamingTransport`
that hold for any implementation -- the head before the body, chunks as they
arrive, the whole-stream cap, no retry once the body has begun, the connection
released on an early close -- pinned against a loopback
:class:`~threetears.search.testing.LocalHttpServer`, because they are
properties of what an implementation does with a socket. The standalone
transport is the one implementation; its own policies (retry counts, the
deadline, the address guard) are pinned in ``test_standalone.py``.

Kept here rather than shipped in ``threetears.search.testing`` beside the
provider suite: that suite exists because consumers write providers, and no
consumer implements a streaming transport. A shipped suite with no caller would
be public API that exists only for a test; it moves to ``testing/`` when a host
writes a second implementation.
"""

from __future__ import annotations

import asyncio

from threetears.search.contracts import LocalCapExceeded, StreamedResponse, StreamingTransport
from threetears.search.standalone import RESPONSE_BYTES_SCOPE, StandaloneTransport
from threetears.search.testing import LocalHttpServer, Reply

#: how long a pin waits for a chunk or a hang-up before naming the failure, in seconds.
_HANGUP_WAIT_SECONDS = 5.0


async def _drain(response: StreamedResponse) -> int:
    """Read a stream to its end.

    :param response: the open stream
    :ptype response: StreamedResponse
    :return: the body bytes read
    :rtype: int
    """
    total = 0
    async for chunk in response:
        total += len(chunk)
    return total


class TestStreamingTransportObligations:
    """The seam's obligations, held by the standalone transport."""

    def streaming_transport(self) -> StreamingTransport:
        """A standalone transport that may reach the pins' loopback server.

        :return: the transport under test
        :rtype: StreamingTransport
        """
        return StandaloneTransport(allow_private_addresses=True, initial_backoff=0.001, max_backoff=0.002)

    async def test_the_seam_is_satisfied(self) -> None:
        """The transport is a streaming transport by shape (P9)."""
        assert isinstance(self.streaming_transport(), StreamingTransport)

    async def test_the_head_arrives_before_the_body(self) -> None:
        """Status, lower-cased headers, final URL, egress and attempts are in hand with no body sent."""
        gate = asyncio.Semaphore(0)
        transport = self.streaming_transport()
        reply = Reply(status=200, chunks=(b"data: one\n\n",), gate=gate, headers={"X-Request-Id": "abc"})
        async with LocalHttpServer((reply,)) as server:
            url = f"{server.base_url}/stream"
            async with await transport.stream("POST", url, json_body={"q": "x"}, max_bytes=1024) as response:
                assert response.status_code == 200
                assert response.headers.get("x-request-id") == "abc", "header keys come back lower-cased"
                assert response.final_url == url
                assert response.egress == transport.egress_name
                assert response.attempts >= 1
                gate.release()
                assert [chunk async for chunk in response] == [b"data: one\n\n"]

    async def test_chunks_arrive_as_they_are_sent(self) -> None:
        """Each chunk is yielded before the next one exists -- nothing is buffered to the end.

        The server sends a chunk only when the pin releases it, so reading a
        chunk in full before releasing the next proves it reached the caller
        while the rest of the stream did not yet exist. An implementation that
        buffers would wait for a chunk that is never sent, which the bounded
        wait turns into a named failure rather than a hang.
        """
        gate = asyncio.Semaphore(0)
        chunks = (b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n")
        async with LocalHttpServer((Reply(chunks=chunks, gate=gate),)) as server:
            response = await self.streaming_transport().stream("POST", f"{server.base_url}/stream", max_bytes=1024)
            async with response:
                iterator = aiter(response)
                for index, chunk in enumerate(chunks):
                    gate.release()
                    pending = bytearray()
                    try:
                        async with asyncio.timeout(_HANGUP_WAIT_SECONDS):
                            while len(pending) < len(chunk):
                                pending += await anext(iterator)
                    except TimeoutError as exc:
                        raise AssertionError(
                            f"chunk {index} never reached the caller before chunk {index + 1} was sent"
                        ) from exc
                    assert bytes(pending) == chunk

    async def test_the_cap_counts_the_whole_stream_and_never_yields_past_it(self) -> None:
        """A stream that runs past ``max_bytes`` is refused, and the caller holds no more than the cap."""
        async with LocalHttpServer((Reply(chunks=(b"a" * 40, b"b" * 40, b"c" * 40)),)) as server:
            response = await self.streaming_transport().stream("GET", f"{server.base_url}/stream", max_bytes=64)
            received = 0
            refusal: LocalCapExceeded | None = None
            async with response:
                try:
                    async for chunk in response:
                        received += len(chunk)
                except LocalCapExceeded as raised:
                    refusal = raised
        assert refusal is not None, "a stream past its cap was read to the end"
        assert refusal.scope == RESPONSE_BYTES_SCOPE
        assert received <= 64, f"the caller was handed {received} bytes under a 64-byte cap"

    async def test_a_stream_broken_after_its_first_byte_is_not_retried(self) -> None:
        """Once body bytes exist the provider has begun answering; a second attempt would be a second bill."""
        async with LocalHttpServer((Reply(chunks=(b"data: partial",), truncate=True), Reply())) as server:
            response = await self.streaming_transport().stream("POST", f"{server.base_url}/stream", max_bytes=1024)
            async with response:
                # the failure's class is each implementation's own; that it
                # surfaced, and was not retried, is what this pin holds.
                (outcome,) = await asyncio.gather(_drain(response), return_exceptions=True)
            assert isinstance(outcome, Exception), "a truncated stream read as complete"
            assert len(server.requests) == 1, "the stream was retried after its body began"

    async def test_an_early_close_releases_the_connection(self) -> None:
        """A reader that stops part-way does not hold the socket open."""
        gate = asyncio.Semaphore(0)
        async with LocalHttpServer((Reply(chunks=(b"one", b"two", b"three"), gate=gate),)) as server:
            response = await self.streaming_transport().stream("GET", f"{server.base_url}/stream", max_bytes=1024)
            gate.release()
            assert await anext(aiter(response)) == b"one"
            await response.aclose()
            await response.aclose()  # idempotent
            async with asyncio.timeout(_HANGUP_WAIT_SECONDS):
                while server.hangups == 0:
                    await asyncio.sleep(0.01)
            assert server.hangups == 1
