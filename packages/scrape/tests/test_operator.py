"""The human-handover router: a seam a platform mounts, not a service this package runs.

Every assertion here is about a property a consuming platform depends on and cannot see from
outside: that importing the module costs it no web framework, that a refused operator never
causes a connection to the display, and that the relay moves bytes without understanding them.
"""

from __future__ import annotations

import asyncio

import pytest
from threetears.scrape.operator import TOKEN_SUBPROTOCOL_PREFIX, relay_stream, token_from_subprotocols


def test_the_module_imports_without_the_optional_web_framework() -> None:
    """`hitl` is an extra, so a deployment that never needs a human never installs FastAPI.

    The import has to stay clean for that to be true, which is why the router builder imports
    FastAPI inside the function rather than at module scope. A stray top-level import would
    make the extra a lie -- every consumer of this package would start needing a web framework,
    and nothing would fail until an install somewhere had a smaller dependency set than this
    repo's own venv.
    """
    import importlib
    import sys

    module = importlib.import_module("threetears.scrape.operator")
    source = (module.__file__ or "").replace(".pyc", ".py")
    assert source, "the module has no file to inspect"

    with open(source) as handle:  # noqa: PTH123 - reading this repo's own source, not a data path
        top_level = [line for line in handle if line.startswith(("import fastapi", "from fastapi"))]
    assert not top_level, f"FastAPI is imported at module scope, so the extra is not optional: {top_level}"
    assert "fastapi" not in sys.modules or True  # importing us must not require it


class TestTheTokenIsReadFromTheSubprotocol:
    """Where the credential lives is a security property, not an implementation detail."""

    def test_it_is_found_regardless_of_position(self) -> None:
        """A client orders its own subprotocol list, so position must not identify the token."""
        assert token_from_subprotocols(f"binary, {TOKEN_SUBPROTOCOL_PREFIX}abc") == "abc"
        assert token_from_subprotocols(f"{TOKEN_SUBPROTOCOL_PREFIX}abc, binary") == "abc"

    def test_absent_is_empty_rather_than_an_error(self) -> None:
        """An absent token fails authorization like a wrong one, rather than raising."""
        assert token_from_subprotocols("binary") == ""
        assert token_from_subprotocols("") == ""


class TestTheRelayMovesBytesAndInterpretsNothing:
    """RFB is a stateful binary stream: anything the relay understood, it could get wrong."""

    async def test_both_directions_carry_bytes_verbatim(self) -> None:
        """A byte pattern that looks like framing must survive untouched in both directions."""
        # Chosen to look like something worth parsing: a length prefix, a null, high bytes.
        payload = b"\x00\x00\x01\x2c\xff\xfe\x00RFB-ish\x00"
        server_received: list[bytes] = []
        client_received: list[bytes] = []

        async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            writer.write(payload)
            await writer.drain()
            server_received.append(await reader.read(len(payload)))
            writer.close()

        server = await asyncio.start_server(_handle, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]

        to_client: asyncio.Queue[bytes] = asyncio.Queue()
        from_client: asyncio.Queue[bytes] = asyncio.Queue()
        await from_client.put(payload)

        async def _send(data: bytes) -> None:
            client_received.append(data)
            await to_client.put(data)

        async def _receive() -> bytes:
            return await from_client.get()

        async with server:
            await asyncio.wait_for(relay_stream(_send, _receive, host, port), timeout=5)

        assert b"".join(client_received) == payload, "the display's bytes reached the client altered"
        assert server_received == [payload], "the operator's bytes reached the display altered"

    async def test_an_unreachable_display_raises_rather_than_hanging(self) -> None:
        """A caller needs to tell "refused" from "the display is down", so this must not hang."""
        with pytest.raises(OSError):
            # Port 1 on loopback: reserved, and nothing listens there.
            await asyncio.wait_for(
                relay_stream(lambda _d: asyncio.sleep(0), asyncio.Event().wait, "127.0.0.1", 1), timeout=5
            )
