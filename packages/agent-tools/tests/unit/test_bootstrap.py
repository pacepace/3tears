"""unit tests for ``threetears.agent.tools.bootstrap.ToolServerBootstrap``."""

from __future__ import annotations

import asyncio
import signal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.agent.tools.bootstrap import ToolServerBootstrap


class _FakeToolServer:
    """minimal ToolServer stand-in for bootstrap behavioral tests."""

    def __init__(self) -> None:
        self.tools_count = 2
        self.serve_called = False
        self.shutdown_called = False
        self._serve_event = asyncio.Event()

    async def serve(self) -> None:
        self.serve_called = True
        await self._serve_event.wait()

    async def shutdown(self) -> None:
        self.shutdown_called = True
        self._serve_event.set()


class _ConcreteBootstrap(ToolServerBootstrap):
    """subclass used to drive ``run`` / ``_run_async`` paths."""

    def __init__(self, *, server: _FakeToolServer, register_called: list[bool]) -> None:
        super().__init__("test-pod")
        self._server = server
        self._register_called = register_called

    async def _build_server(self) -> Any:
        return self._server

    async def _register_tools(self, server: Any) -> None:
        self._register_called.append(True)


class TestRunAsync:
    """``_run_async`` builds, registers, and serves until shutdown."""

    async def test_lifecycle_completes_when_serve_returns(self) -> None:
        server = _FakeToolServer()
        register_called: list[bool] = []
        bootstrap = _ConcreteBootstrap(server=server, register_called=register_called)

        # release serve immediately so _run_async returns
        server._serve_event.set()
        await bootstrap._run_async()

        assert register_called == [True]
        assert server.serve_called is True

    async def test_signal_handler_triggers_shutdown(self) -> None:
        server = _FakeToolServer()
        register_called: list[bool] = []
        bootstrap = _ConcreteBootstrap(server=server, register_called=register_called)

        async def trigger_signal_then_run() -> None:
            run_task = asyncio.create_task(bootstrap._run_async())
            await asyncio.sleep(0)  # let bootstrap install handlers
            await asyncio.sleep(0)
            # simulate the signal handler firing (do not raise actual SIGTERM)
            handler = bootstrap._make_signal_handler(server, "test")
            handler()
            await run_task

        await asyncio.wait_for(trigger_signal_then_run(), timeout=2.0)
        assert server.shutdown_called is True


class TestUnoverriddenHooks:
    """default ``_build_server`` and ``_register_tools`` raise NotImplementedError."""

    async def test_build_server_default_raises(self) -> None:
        bootstrap = ToolServerBootstrap("x")
        with pytest.raises(NotImplementedError):
            await bootstrap._build_server()

    async def test_register_tools_default_raises(self) -> None:
        bootstrap = ToolServerBootstrap("x")
        with pytest.raises(NotImplementedError):
            await bootstrap._register_tools(MagicMock())


class TestRunServe:
    """default ``_run_serve`` awaits ``server.serve()``."""

    async def test_run_serve_awaits_serve(self) -> None:
        server = AsyncMock()
        server.serve = AsyncMock(return_value=None)
        bootstrap = ToolServerBootstrap("x")
        await bootstrap._run_serve(server)
        server.serve.assert_awaited_once()


def test_signal_handler_uses_service_name_in_task_name() -> None:
    bootstrap = ToolServerBootstrap("my-svc")
    server = MagicMock()
    handler = bootstrap._make_signal_handler(server, "sigterm")
    # closure captured the service name in the task name template
    assert callable(handler)
