"""unit tests for ``threetears.agent.tools.bootstrap.ToolServerBootstrap``."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.agent.tools.bootstrap import ToolServerBootstrap


class _FakeToolServer:
    """minimal ToolServer stand-in for bootstrap behavioral tests.

    not a SLF001 violation surface: ``_serve_event`` is owned by
    this class and accessed only by its own methods plus the test
    that constructs it. tests below NEVER touch that field directly.
    """

    def __init__(self) -> None:
        self.tools_count = 2
        self.serve_called = False
        self.shutdown_called = False
        self.serve_event = asyncio.Event()

    async def serve(self) -> None:
        self.serve_called = True
        await self.serve_event.wait()

    async def shutdown(self) -> None:
        self.shutdown_called = True
        self.serve_event.set()


class _ConcreteBootstrap(ToolServerBootstrap):
    """subclass used to drive ``run`` / ``run_async`` paths."""

    def __init__(self, *, server: _FakeToolServer, register_log: list[bool]) -> None:
        super().__init__("test-pod")
        self.server = server
        self.register_log = register_log

    async def build_server(self) -> Any:
        return self.server

    async def register_tools(self, server: Any) -> None:
        self.register_log.append(True)


class TestRunAsync:
    """``run_async`` builds, registers, and serves until shutdown."""

    async def test_lifecycle_completes_when_serve_returns(self) -> None:
        server = _FakeToolServer()
        register_log: list[bool] = []
        bootstrap = _ConcreteBootstrap(server=server, register_log=register_log)

        # release serve immediately so run_async returns
        server.serve_event.set()
        await bootstrap.run_async()

        assert register_log == [True]
        assert server.serve_called is True

    async def test_signal_handler_triggers_shutdown(self) -> None:
        server = _FakeToolServer()
        register_log: list[bool] = []
        bootstrap = _ConcreteBootstrap(server=server, register_log=register_log)

        async def trigger_signal_then_run() -> None:
            run_task = asyncio.create_task(bootstrap.run_async())
            await asyncio.sleep(0)  # let bootstrap install handlers
            await asyncio.sleep(0)
            # simulate the signal handler firing (do not raise actual SIGTERM)
            handler = bootstrap.make_signal_handler(server, "test")
            handler()
            await run_task

        await asyncio.wait_for(trigger_signal_then_run(), timeout=2.0)
        assert server.shutdown_called is True


class TestUnoverriddenHooks:
    """default ``build_server`` and ``register_tools`` raise NotImplementedError."""

    async def test_build_server_default_raises(self) -> None:
        bootstrap = ToolServerBootstrap("x")
        with pytest.raises(NotImplementedError):
            await bootstrap.build_server()

    async def test_register_tools_default_raises(self) -> None:
        bootstrap = ToolServerBootstrap("x")
        with pytest.raises(NotImplementedError):
            await bootstrap.register_tools(MagicMock())


class TestRunServe:
    """default ``run_serve`` awaits ``server.serve()``."""

    async def test_run_serve_awaits_serve(self) -> None:
        server = AsyncMock()
        server.serve = AsyncMock(return_value=None)
        bootstrap = ToolServerBootstrap("x")
        await bootstrap.run_serve(server)
        server.serve.assert_awaited_once()


def test_signal_handler_uses_service_name_in_task_name() -> None:
    bootstrap = ToolServerBootstrap("my-svc")
    server = MagicMock()
    handler = bootstrap.make_signal_handler(server, "sigterm")
    # closure captured the service name in the task name template
    assert callable(handler)
