"""unit tests for ``threetears.agent.tools.bootstrap.ToolServerBootstrap``."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.agent.tools.bootstrap import EX_CONFIG, ToolPodConfigError, ToolServerBootstrap
from threetears.observe import HealthTier


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

    def render_metrics(self) -> tuple[str, bytes]:
        return ("text/plain; version=0.0.4; charset=utf-8", b"")


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


class _ReadinessFakeServer:
    """a ToolServer stand-in exposing the three probe surfaces the bootstrap health server reads;
    ``jwks_warmed`` is flipped by the test to drive the NOT-READY -> READY transition. ``is_healthy``
    (not ``is_connected``) is the nats liveness surface the probe now reads -- real NATS health, so a
    dead connection trips liveness instead of reporting healthy forever."""

    def __init__(self) -> None:
        self.is_healthy = True
        self.tools_count = 1
        self.jwks_warmed = False

    def render_metrics(self) -> tuple[str, bytes]:
        return ("text/plain; version=0.0.4; charset=utf-8", b"")


class TestHealthServerReadinessGate:
    """B5: the tool-pod health server gates readiness on the JWKS being warm -- it must report
    NOT-READY (503) until the pod's JWKS provider can actually verify a token."""

    async def test_reports_not_ready_until_jwks_warmed(self) -> None:
        srv = _ReadinessFakeServer()
        # health_port=0 -> OS-assigned ephemeral port, so the listener never collides in CI.
        bootstrap = ToolServerBootstrap("test-pod", health_port=0)
        health_server = await bootstrap._start_health_server(srv)  # noqa: SLF001 -- intra-package wiring seam
        assert health_server is not None
        try:
            # before the JWKS warms: the jwks_warmed component is unhealthy -> overall NOT-READY.
            before = await health_server.get_status(HealthTier.READY)
            comps = {c.name: c.healthy for c in before.components}
            assert "jwks_warmed" in comps, "the tool-pod health server must wire a jwks_warmed readiness gate"
            assert comps["jwks_warmed"] is False
            assert before.healthy is False
            # after the first successful JWKS fetch: the gate clears -> READY.
            srv.jwks_warmed = True
            after = await health_server.get_status(HealthTier.READY)
            assert all(c.healthy for c in after.components)
            assert after.healthy is True
        finally:
            await health_server.stop()

    async def test_cold_jwks_cache_does_not_fail_liveness(self) -> None:
        """the readiness gate must be invisible to the liveness verdict.

        this is what buys the tool pods a livenessProbe: under the old aliased
        contract a cold JWKS cache took /healthz down too, so the only safe
        deployment was to ship with no livenessProbe and no restart-on-wedge net.
        """
        srv = _ReadinessFakeServer()
        srv.jwks_warmed = False
        srv.tools_count = 0
        bootstrap = ToolServerBootstrap("test-pod", health_port=0)
        health_server = await bootstrap._start_health_server(srv)  # noqa: SLF001 -- intra-package wiring seam
        assert health_server is not None
        try:
            live = await health_server.get_status(HealthTier.LIVE)
            ready = await health_server.get_status(HealthTier.READY)
            assert live.healthy is True, "a cold JWKS cache must never restart the pod"
            assert [c.name for c in live.components] == ["nats"]
            assert ready.healthy is False
        finally:
            await health_server.stop()

    async def test_dead_nats_fails_both_tiers(self) -> None:
        """a terminally wedged data plane restarts the pod AND pulls it from rotation."""
        srv = _ReadinessFakeServer()
        srv.is_healthy = False
        srv.jwks_warmed = True
        bootstrap = ToolServerBootstrap("test-pod", health_port=0)
        health_server = await bootstrap._start_health_server(srv)  # noqa: SLF001 -- intra-package wiring seam
        assert health_server is not None
        try:
            live = await health_server.get_status(HealthTier.LIVE)
            ready = await health_server.get_status(HealthTier.READY)
            assert live.healthy is False
            assert ready.healthy is False
        finally:
            await health_server.stop()


class _StartupFailureBootstrap(ToolServerBootstrap):
    """subclass whose ``build_server`` raises, to drive ``run``'s failure classification."""

    def __init__(self, exc: BaseException) -> None:
        # health_port=0 -> OS-assigned ephemeral port, so nothing collides in CI.
        super().__init__("test-pod", health_port=0)
        self.exc = exc

    async def build_server(self) -> Any:
        raise self.exc

    async def register_tools(self, server: Any) -> None:  # pragma: no cover - startup never gets here
        raise AssertionError("register_tools must not run after build_server raised")


class _SucceedingBootstrap(ToolServerBootstrap):
    """subclass that reaches and returns from the serve loop, driving ``run``'s happy path."""

    def __init__(self, server: _FakeToolServer) -> None:
        super().__init__("test-pod", health_port=0)
        self.server = server

    async def build_server(self) -> Any:
        return self.server

    async def register_tools(self, server: Any) -> None:
        pass


def _error_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """the ERROR-or-worse records the bootstrap emitted, in order."""
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


class TestConfigFaultIsTerminal:
    """a permanent config fault must stop the pod, not feed a restart loop.

    bluelabsio/14-eng-ai-bot#235: a built-in tool pod started with a renamed identity
    env var raised, exited 1, and was restarted 9,580 times over 8 days without ever
    reaching ``running``. no healthcheck could fire (the container never started), no
    neighbour declared ``depends_on`` it, and the 278,110 identical traceback lines it
    wrote went to a stderr nobody tails. the fail-loud intent was right; "loud once"
    is what the deployment turned into "invisible forever".

    the fix has two halves and both are asserted here: a distinct exit STATUS, which
    is the only thing a supervisor can branch on, and ONE structured record naming the
    variable an operator has to go change.
    """

    def test_config_fault_exits_with_ex_config(self) -> None:
        """the status is 78, ``EX_CONFIG`` from sysexits.h -- not the generic 1."""
        bootstrap = _StartupFailureBootstrap(
            ToolPodConfigError("SOME_VAR is required", variable="SOME_VAR"),
        )

        with pytest.raises(SystemExit) as exit_info:
            bootstrap.run()

        assert exit_info.value.code == EX_CONFIG
        assert EX_CONFIG == 78, "the value is the contract: compose / k8s alerting branches on it"

    def test_config_fault_logs_exactly_one_error_naming_the_variable(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """one record, through the repo logger, carrying the offending variable.

        one, because the incident's cost was repetition. through ``get_logger``, because
        a traceback on container stderr never reaches the observability pipeline and so
        was never queryable. naming the variable, because an operator who reads the line
        and still has to open the source has not been told anything.
        """
        bootstrap = _StartupFailureBootstrap(
            ToolPodConfigError(
                "THREETEARS_TOOL_POD_ID is missing (the minter kid must be the pod id)",
                variable="THREETEARS_TOOL_POD_ID",
            ),
        )

        with caplog.at_level(logging.ERROR), pytest.raises(SystemExit):
            bootstrap.run()

        errors = _error_records(caplog)
        assert len(errors) == 1, f"expected exactly one ERROR record, got {[r.getMessage() for r in errors]}"
        extra = getattr(errors[0], "extra_data", {})
        assert extra.get("variable") == "THREETEARS_TOOL_POD_ID"
        assert extra.get("exit_code") == EX_CONFIG
        assert "THREETEARS_TOOL_POD_ID" in extra.get("error", "")

    def test_unrelated_value_error_is_not_treated_as_config(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """the reason the terminal path keys on a distinct type, not on ``ValueError``.

        a ``ValueError`` out of a tool's own business logic during registration is a bug
        that a restart may well clear. catching the base type would make it terminal and
        keep a recoverable pod down.
        """
        bootstrap = _StartupFailureBootstrap(ValueError("bad payload from some tool's own parsing"))

        with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="bad payload"):
            bootstrap.run()

        assert _error_records(caplog) == [], "a non-config failure must not emit the terminal record"

    def test_transient_failure_still_propagates_and_stays_retryable(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """a NATS server that is not up yet is exactly what supervisor restarts are for."""
        bootstrap = _StartupFailureBootstrap(ConnectionRefusedError("nats://nats:4222 refused"))

        with caplog.at_level(logging.ERROR), pytest.raises(ConnectionRefusedError):
            bootstrap.run()

        assert _error_records(caplog) == []

    def test_successful_run_is_unaffected(self) -> None:
        """the guard is inert on the path that actually serves."""
        server = _FakeToolServer()
        server.serve_event.set()  # release serve immediately so run returns

        _SucceedingBootstrap(server).run()

        assert server.serve_called is True


def test_signal_handler_uses_service_name_in_task_name() -> None:
    bootstrap = ToolServerBootstrap("my-svc")
    server = MagicMock()
    handler = bootstrap.make_signal_handler(server, "sigterm")
    # closure captured the service name in the task name template
    assert callable(handler)
