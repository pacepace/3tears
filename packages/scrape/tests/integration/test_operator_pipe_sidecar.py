"""The display bridge against the real sidecar container, a real ``x11vnc`` and a real broker.

What the unit suite cannot prove is here. There the display is a loopback socket that greets
whoever connects; here it is the container this repository ships, running the ``x11vnc`` its own
code starts, in front of the Xvfb its entrypoint brings up. The bytes that cross the pipe are an
actual RFB conversation, and the last of them is a full framebuffer of a real screen.

**Why a forwarder is exec'd into the container, and what it does not weaken.** ``x11vnc`` binds
``127.0.0.1`` and the container publishes nothing on the display path, deliberately: that binding
IS the access control, and in the deployment this exists for the pod side runs INSIDE the network
namespace where loopback already reaches it. A test on the host is outside that namespace, so it
puts a plain TCP forwarder in the container and reaches the display through it. Everything on the
far side of that hop is the real thing; nothing about the sidecar's binding is changed, and the
forwarder exists for the duration of one test.

**What this still does not prove, and only a person can.** That an operator's browser renders the
framebuffer and that clicks and keystrokes land where they look like they land. This gets real
pixels of a real screen out of a real pod; a human at a noVNC client is what closes the rest, and
that is recorded in ``.prawduct/operator-verification.md`` rather than asserted here.

Guarded by ``@pytest.mark.integration``, so the workspace sweep (``-m "not integration"``)
deselects it. Run it explicitly::

    uv run pytest packages/scrape/tests/integration/test_operator_pipe_sidecar.py -m integration -s

It skips when docker is absent or when the sidecar image has not been built, and names the build
command in the skip rather than passing quietly.
"""

from __future__ import annotations

import asyncio
import os
import struct
import time
from collections.abc import AsyncIterator
from datetime import timedelta

import httpx
import pytest
from testcontainers.core.container import DockerContainer
from threetears.nats import NatsClient, Subjects, attach_pipe, open_pipe, set_default_namespace

from threetears.scrape.operator_pipe import serve_display
from threetears.scrape.operator_session import SessionClaim

pytestmark = pytest.mark.integration

#: The image the sidecar's own compose file and bake target both write. Overridable, because a
#: verification run against a stale image is a verification run against code that is not the code
#: under test.
_IMAGE = os.environ.get("SCRAPE_SIDECAR_IMAGE", "aibots/nodriver-sidecar:latest")

#: The sidecar's API port, which is what mints sessions and starts the display.
_API_PORT = 8088

#: Where the in-container forwarder listens. Not 5900: that is x11vnc's own loopback port and
#: publishing it is exactly what the sidecar declines to do.
_FORWARD_PORT = 5901

_NS = "sidecarpipe"
#: the tool-name NODE the serving pod owns, as its ``tool_pods`` row holds it and as
#: its registration reply names it. NOT a tool leaf: the grant is minted from the node.
_OWNED_NODE = "tools.scrape"
_POD_ID = "pod-7f3c"
_SESSION = "operator-session-1"

#: A plain TCP forwarder, run inside the container so a host-side test can reach a display bound
#: to the container's own loopback. Written as a string because it is exec'd into an image that
#: has python and no test dependencies.
_FORWARDER = """
import asyncio

async def pump(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()

async def handle(reader, writer):
    display_reader, display_writer = await asyncio.open_connection("127.0.0.1", 5900)
    await asyncio.gather(
        pump(reader, display_writer),
        pump(display_reader, writer),
        return_exceptions=True,
    )

async def main():
    server = await asyncio.start_server(handle, "0.0.0.0", 5901)
    async with server:
        await server.serve_forever()

asyncio.run(main())
"""


def _docker_or_skip() -> None:
    """Skip unless docker is up and the sidecar image is present."""
    try:
        import docker  # noqa: PLC0415
    except ImportError:  # pragma: no cover - the dev extra always carries it
        pytest.skip("the docker client library is not installed")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- any failure to reach a daemon means the same thing to a test: there is no docker here. Reported in the skip
        pytest.skip(f"docker is not available: {exc}")
    try:
        client.images.get(_IMAGE)
    except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- the client raises ImageNotFound or an API error and both mean the image cannot be used; the message says which
        pytest.skip(
            f"the sidecar image {_IMAGE!r} is not present ({exc}). Build it with "
            f"`docker compose -f packages/scrape/sidecar/docker-compose.yml build`, or point "
            f"SCRAPE_SIDECAR_IMAGE at one."
        )


@pytest.fixture
async def display_address() -> AsyncIterator[tuple[str, int]]:
    """A running sidecar with its display started, reachable at the yielded address.

    Starts the container, waits for its health endpoint, asks it to bring the display up through
    the same API a platform uses, and puts the forwarder in place. Everything is torn down
    afterwards including the container.
    """
    _docker_or_skip()
    container = DockerContainer(_IMAGE).with_exposed_ports(_API_PORT, _FORWARD_PORT)
    # Chromium is started by the entrypoint whatever this test does with it, and the Docker
    # default of 64 MB of shared memory crashes its renderer on a real page.
    container.with_kwargs(shm_size="1g")
    container.start()
    try:
        api = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(_API_PORT)}"
        await _await_health(api)
        async with httpx.AsyncClient(timeout=30.0) as http:
            started = await http.post(f"{api}/v1/hitl/vnc")
            assert started.status_code == 200, f"the sidecar refused to start its display: {started.text}"
        container.get_wrapped_container().exec_run(["python3", "-c", _FORWARDER], detach=True)
        address = (container.get_container_host_ip(), int(container.get_exposed_port(_FORWARD_PORT)))
        await _await_tcp(address)
        yield address
    finally:
        container.stop()


async def _await_health(api: str, *, timeout: float = 120.0) -> None:
    """Wait for the sidecar to report itself healthy, or say what it last answered."""
    deadline = time.monotonic() + timeout
    last = "nothing yet"
    async with httpx.AsyncClient(timeout=5.0) as http:
        while time.monotonic() < deadline:
            try:
                response = await http.get(f"{api}/healthz")
                if response.status_code == 200:
                    return
                last = f"HTTP {response.status_code}: {response.text}"
            except httpx.HTTPError as exc:
                last = str(exc)
            await asyncio.sleep(1.0)
    raise AssertionError(f"the sidecar never became healthy within {timeout}s; last answer was {last}")


async def _await_tcp(address: tuple[str, int], *, timeout: float = 30.0) -> None:
    """Wait until the forwarded display GREETS a connection, not merely accepts one.

    Accepting is not evidence here and the difference cost a debugging round. Docker's published
    port is fronted by a proxy on the host that accepts a connection and then closes it when
    nothing inside the container is listening yet, so a readiness check that connected and hung up
    passed against a forwarder which had not finished binding. The pod side then opened a
    connection that ended immediately, and the failure surfaced as an RFB stream ending twelve
    bytes short. Reading the banner is the check that can only succeed for the right reason.
    """
    deadline = time.monotonic() + timeout
    last: str = "no attempt completed"
    while time.monotonic() < deadline:
        try:
            reader, writer = await asyncio.open_connection(*address)
        except OSError as exc:
            last = str(exc)
            await asyncio.sleep(0.5)
            continue
        try:
            greeting = await asyncio.wait_for(reader.read(12), timeout=5.0)
            if greeting.startswith(b"RFB "):
                return
            last = f"connected but was greeted with {greeting!r}"
        except (TimeoutError, OSError) as exc:
            last = f"connected but heard nothing: {exc}"
        finally:
            writer.close()
        await asyncio.sleep(0.5)
    raise AssertionError(f"the forwarded display at {address} never greeted a client: {last}")


class _PipeReader:
    """Reads exactly what an RFB step needs out of a stream that arrives in chunks."""

    def __init__(self, stream: object) -> None:
        self._stream = stream
        self._buffer = bytearray()
        #: Total bytes taken off the pipe, which is what the measurement below reports.
        self.total = 0

    async def read_exactly(self, count: int) -> bytes:
        """Return the next *count* bytes, waiting for as many frames as that takes."""
        while len(self._buffer) < count:
            chunk = await self._stream.receive()  # type: ignore[attr-defined]
            if chunk is None:
                raise AssertionError(f"the display's stream ended {count - len(self._buffer)} bytes short")
            self._buffer += chunk
            self.total += len(chunk)
        taken = bytes(self._buffer[:count])
        del self._buffer[:count]
        return taken


async def _connect(url: str, name: str) -> NatsClient:
    """Connect a client bound to this test's subject namespace."""
    set_default_namespace(_NS)
    return await NatsClient.connect(nats_url=url, nats_subject_namespace=_NS, client_name=name)


async def test_a_real_display_is_driven_through_the_pipe(nats_container: str, display_address: tuple[str, int]) -> None:
    """An RFB conversation with the container's own ``x11vnc``, entirely over the pipe.

    The handshake is what proves both directions are live against something that is not a test
    double: every server step here is a reply to a client message that had to arrive intact and in
    order, and ``x11vnc`` closes the connection rather than answering if it does not.

    The framebuffer request at the end is deliberate rather than thorough. It is the first real
    traffic this transport has carried, so it is where the design's bandwidth assumption stops
    being an assumption -- and at a full screen it is several megabytes, which puts it well past
    the credit window and makes this the first time the flow control runs against a real producer.
    """
    # The DISPLAY STREAM family, not the control plane's. A session is owner-routed twice on the
    # same key, so the two derive different families deliberately -- sharing one would put both
    # on a single subject and let the queue group split a pod's messages between two handlers.
    family = Subjects.hitl_pipe_family(_OWNED_NODE)
    claim = SessionClaim(session_id=_SESSION)

    async def _where(session_id: str) -> tuple[str, int]:
        del session_id
        return display_address

    async with await _connect(nats_container, "pod") as pod, await _connect(nats_container, "hub") as hub:
        async with serve_display(pod, claim, owned_node=_OWNED_NODE, pod_id=_POD_ID, display=_where):
            endpoint = await attach_pipe(hub, _SESSION, family=family, timeout=timedelta(seconds=10))
            async with open_pipe(hub, endpoint) as stream:
                rfb = _PipeReader(stream)

                banner = await asyncio.wait_for(rfb.read_exactly(12), timeout=30)
                assert banner.startswith(b"RFB 003."), f"the display did not greet as an RFB server: {banner!r}"

                # Speak the version back. Everything after this point only happens if the display
                # received these bytes, so each step is evidence the up direction works.
                await stream.send(banner)
                security_count = (await asyncio.wait_for(rfb.read_exactly(1), timeout=30))[0]
                assert security_count > 0, "the display refused the version this client offered"
                types = await rfb.read_exactly(security_count)
                assert 1 in types, f"the display offered no unauthenticated security type: {types!r}"

                # `-nopw`: the sidecar authenticates nobody and says so, because deciding who may
                # see a display happens in front of the relay rather than on this socket.
                await stream.send(bytes([1]))
                result = struct.unpack("!I", await asyncio.wait_for(rfb.read_exactly(4), timeout=30))[0]
                assert result == 0, f"the display rejected the security handshake: {result}"

                await stream.send(bytes([1]))  # ClientInit, shared
                server_init = await asyncio.wait_for(rfb.read_exactly(24), timeout=30)
                width, height = struct.unpack("!HH", server_init[:4])
                bits_per_pixel = server_init[4]
                name_length = struct.unpack("!I", server_init[20:24])[0]
                name = await rfb.read_exactly(name_length)
                assert width > 0 and height > 0, f"the display reported no screen: {width}x{height}"

                # Raw encoding only, so what comes back is the framebuffer itself and its size is
                # arithmetic rather than a guess about what a codec produced.
                await stream.send(struct.pack("!BBHi", 2, 0, 1, 0))
                await stream.send(struct.pack("!BBHHHH", 3, 0, 0, 0, width, height))

                started = time.monotonic()
                header = await asyncio.wait_for(rfb.read_exactly(4), timeout=120)
                assert header[0] == 0, f"the display sent message type {header[0]}, not a framebuffer update"
                rectangles = struct.unpack("!H", header[2:4])[0]
                pixels = 0
                for _ in range(rectangles):
                    rect = await asyncio.wait_for(rfb.read_exactly(12), timeout=120)
                    rect_width, rect_height = struct.unpack("!HH", rect[4:8])
                    encoding = struct.unpack("!i", rect[8:12])[0]
                    assert encoding == 0, f"the display used encoding {encoding} after raw was the only one offered"
                    body = rect_width * rect_height * (bits_per_pixel // 8)
                    await asyncio.wait_for(rfb.read_exactly(body), timeout=120)
                    pixels += body
                elapsed = time.monotonic() - started

        assert pixels > 0, "the display sent an update carrying no pixels at all"
        print(  # noqa: T201 -- the measured numbers are this test's other output, and a value asserted to be "sane" is not one worth pinning
            f"\nreal display through the pipe: {name.decode('utf-8', 'replace')} {width}x{height} "
            f"@{bits_per_pixel}bpp, one full framebuffer = {pixels / 1e6:.2f} MB in {elapsed:.2f}s "
            f"({pixels / 1e6 / max(elapsed, 1e-9):.2f} MB/s), {rfb.total / 1e6:.2f} MB off the pipe in total"
        )
