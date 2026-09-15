"""The Claude CLI pool: reuse, per-call tools, correlation, and no orphans.

A subscription credential is spent by driving the Claude Code CLI, and the package that does it
starts a fresh subprocess per call. These are the contracts that make reusing one safe: a session
serves one caller at a time, carries that caller's tools and no one else's, is cleared between
callers, is thrown away rather than re-pooled the moment its stream is abandoned, and never
outlives the process that started it.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.models import DEFAULT_CHAT_MODEL, DEFAULT_FAST_MODEL, claude_cli_pool
from threetears.models.claude_cli_pool import (
    POOL_MARKER_ENV,
    ClaudeCliPool,
    ClaudeCliPoolExhausted,
    ClaudeCliSessionError,
    kill_process_tree,
    launch_key,
    sweep_orphaned_claude_clis,
)

TOKEN = "sk-ant-oat01-aaaaaaaaaaaaaaaa"
OTHER_TOKEN = "sk-ant-oat01-bbbbbbbbbbbbbbbb"


@dataclass
class Options:
    """The launch-time fields the pool reads, without importing the SDK."""

    system_prompt: str | None = "stable persona"
    model: str | None = DEFAULT_FAST_MODEL
    tools: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=lambda: ["mcp__langchain-tools"])
    disallowed_tools: list[str] = field(default_factory=list)
    permission_mode: str | None = "dontAsk"
    max_turns: int | None = 8
    max_budget_usd: float | None = None
    fallback_model: str | None = None
    cwd: str | None = "/tmp/isolated"
    extra_args: dict[str, str | None] = field(default_factory=lambda: {"strict-mcp-config": None})
    include_partial_messages: bool = True
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


# parity-with: threetears.models.claude_cli_pool.PooledCliSession
class FakeSession:
    """A session that records what the pool did to it, and never spawns."""

    instances: list[FakeSession] = []

    def __init__(self, options: Any, key: str) -> None:
        self.options = options
        self.key = key
        self.client = object()
        self.pid: int | None = 4242
        self.reusable = True
        self.closed = False
        self.prepared: list[tuple[str | None, Any]] = []
        self.clears = 0
        self.disposals = 0
        self.fail_clear = False
        self.fail_prepare = False
        self.clear_raises: BaseException | None = None
        self.dispose_delay = 0.0
        self.released = 0
        self.start_ticks: int | None = None
        self.contexts: list[Any] = []
        FakeSession.instances.append(self)

    @classmethod
    async def start(cls, options: Any, *, key: str) -> FakeSession:
        """The production entry point the pool calls; builds a fake instead of a subprocess."""
        return cls(options, key)

    async def prepare(self, *, model: str | None, tool_server: Any | None, call_context: Any = None) -> None:
        if self.closed:
            raise ClaudeCliSessionError("this Claude CLI session has been stopped")
        if self.fail_prepare:
            raise ClaudeCliSessionError("the CLI refused the tool server")
        self.prepared.append((model, tool_server))
        self.contexts.append(call_context)

    async def release_tools(self, *, timeout: float) -> None:
        del timeout
        if self.closed:
            raise ClaudeCliSessionError("this Claude CLI session has been stopped")
        self.released += 1

    async def clear(self, *, timeout: float) -> None:
        del timeout
        if self.closed:
            raise ClaudeCliSessionError("this Claude CLI session has been stopped")
        if self.clear_raises is not None:
            raise self.clear_raises
        if self.fail_clear:
            raise ClaudeCliSessionError("the CLI would not clear")
        self.clears += 1

    async def dispose(self, *, grace_seconds: float) -> None:
        del grace_seconds
        if self.closed:
            return
        self.closed = True
        if self.dispose_delay:
            await asyncio.sleep(self.dispose_delay)
        self.disposals += 1


@pytest.fixture(autouse=True)
def _fresh_sessions() -> Any:
    FakeSession.instances = []
    yield
    FakeSession.instances = []


async def _factory(options: Any, *, key: str) -> Any:
    return await FakeSession.start(options, key=key)


def _pool(**kwargs: Any) -> ClaudeCliPool:
    settings: dict[str, Any] = {"session_factory": _factory, "checkout_timeout_seconds": 0.05, "idle_ttl_seconds": 0.0}
    settings.update(kwargs)
    return ClaudeCliPool(**settings)


class TestTheCliIsReused:
    async def test_a_second_call_gets_the_same_cli(self) -> None:
        pool = _pool()
        async with pool.checkout(Options(), token=TOKEN, tool_server=None) as first:
            pass
        async with pool.checkout(Options(), token=TOKEN, tool_server=None) as second:
            pass
        assert first is second, "the pool started a second CLI for an identical launch"
        assert len(FakeSession.instances) == 1
        await pool.aclose()

    async def test_the_session_is_cleared_between_callers(self) -> None:
        pool = _pool()
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            pass
        assert FakeSession.instances[0].clears == 1, "one caller's conversation would reach the next"
        await pool.aclose()

    async def test_a_different_model_reuses_the_cli_and_switches_it(self) -> None:
        """The model is a per-call setting (``set_model``), not a reason for a second CLI."""
        pool = _pool()
        async with pool.checkout(Options(model=DEFAULT_FAST_MODEL), token=TOKEN, tool_server=None):
            pass
        async with pool.checkout(Options(model=DEFAULT_CHAT_MODEL), token=TOKEN, tool_server=None):
            pass
        assert len(FakeSession.instances) == 1
        assert [m for m, _ in FakeSession.instances[0].prepared] == [DEFAULT_FAST_MODEL, DEFAULT_CHAT_MODEL]
        await pool.aclose()

    async def test_a_different_credential_gets_a_different_cli(self) -> None:
        pool = _pool()
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            pass
        async with pool.checkout(Options(), token=OTHER_TOKEN, tool_server=None):
            pass
        assert len(FakeSession.instances) == 2
        await pool.aclose()

    async def test_a_different_system_prompt_gets_a_different_cli(self) -> None:
        """A running CLI's system prompt cannot change, so a different one cannot share it."""
        pool = _pool()
        async with pool.checkout(Options(system_prompt="persona A"), token=TOKEN, tool_server=None):
            pass
        async with pool.checkout(Options(system_prompt="persona B"), token=TOKEN, tool_server=None):
            pass
        assert len(FakeSession.instances) == 2
        await pool.aclose()


class TestToolsBelongToTheCall:
    async def test_every_checkout_installs_that_calls_own_tool_server(self) -> None:
        """A tool server's handlers close over one conversation's tool objects; a stale one would
        run one conversation's tools on another's behalf."""
        pool = _pool()
        first_tools, second_tools = object(), object()
        async with pool.checkout(Options(), token=TOKEN, tool_server=first_tools):
            pass
        async with pool.checkout(Options(), token=TOKEN, tool_server=second_tools):
            pass
        session = FakeSession.instances[0]
        assert [server for _, server in session.prepared] == [first_tools, second_tools]
        await pool.aclose()

    async def test_a_call_with_no_tools_removes_the_previous_callers_tools(self) -> None:
        pool = _pool()
        async with pool.checkout(Options(), token=TOKEN, tool_server=object()):
            pass
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            pass
        assert FakeSession.instances[0].prepared[-1] == (DEFAULT_FAST_MODEL, None)
        await pool.aclose()

    async def test_a_session_that_refuses_the_tool_server_is_disposed(self) -> None:
        pool = _pool()
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            pass
        FakeSession.instances[0].fail_prepare = True
        with pytest.raises(ClaudeCliSessionError):
            async with pool.checkout(Options(), token=TOKEN, tool_server=object()):
                pass
        assert FakeSession.instances[0].disposals == 1
        assert pool.live_count == 0
        await pool.aclose()


class TestResponsesCannotCross:
    async def test_a_borrowed_session_is_not_handed_to_a_second_caller(self) -> None:
        pool = _pool(per_key=2)
        async with pool.checkout(Options(), token=TOKEN, tool_server=None) as first:
            async with pool.checkout(Options(), token=TOKEN, tool_server=None) as second:
                assert first is not second
        await pool.aclose()

    async def test_a_call_that_raises_disposes_the_session(self) -> None:
        pool = _pool()
        with pytest.raises(RuntimeError):
            async with pool.checkout(Options(), token=TOKEN, tool_server=None):
                raise RuntimeError("the stream broke mid-answer")
        assert FakeSession.instances[0].disposals == 1
        assert FakeSession.instances[0].clears == 0
        assert pool.live_count == 0
        await pool.aclose()

    async def test_a_cancelled_call_disposes_the_session(self) -> None:
        pool = _pool()
        entered = asyncio.Event()

        async def caller() -> None:
            async with pool.checkout(Options(), token=TOKEN, tool_server=None):
                entered.set()
                await asyncio.sleep(3600)

        task = asyncio.create_task(caller())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert FakeSession.instances[0].disposals == 1, "a cancelled turn's half-read session went back in the pool"
        assert pool.live_count == 0
        await pool.aclose()

    async def test_a_session_that_will_not_clear_is_disposed(self) -> None:
        pool = _pool()
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            FakeSession.instances[0].fail_clear = True
        assert FakeSession.instances[0].disposals == 1
        assert pool.live_count == 0
        await pool.aclose()

    async def test_a_session_that_cannot_clear_is_single_use(self) -> None:
        pool = _pool()
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            FakeSession.instances[0].reusable = False
        assert FakeSession.instances[0].disposals == 1
        await pool.aclose()


class TestTheCapsHold:
    async def test_the_per_key_cap_holds(self) -> None:
        pool = _pool(per_key=1)
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            with pytest.raises(ClaudeCliPoolExhausted):
                async with pool.checkout(Options(), token=TOKEN, tool_server=None):
                    pass
        await pool.aclose()

    async def test_the_process_cap_holds_across_keys(self) -> None:
        pool = _pool(max_sessions=1, per_key=5)
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            with pytest.raises(ClaudeCliPoolExhausted):
                async with pool.checkout(Options(), token=OTHER_TOKEN, tool_server=None):
                    pass
        await pool.aclose()

    async def test_a_waiting_caller_is_served_when_a_session_comes_back(self) -> None:
        pool = _pool(per_key=1, checkout_timeout_seconds=2.0)
        release = asyncio.Event()

        async def holder() -> None:
            async with pool.checkout(Options(), token=TOKEN, tool_server=None):
                await release.wait()

        task = asyncio.create_task(holder())
        await asyncio.sleep(0.01)
        waiter = asyncio.create_task(self._borrow_once(pool))
        await asyncio.sleep(0.01)
        release.set()
        await task
        await waiter
        assert len(FakeSession.instances) == 1
        await pool.aclose()

    @staticmethod
    async def _borrow_once(pool: ClaudeCliPool) -> None:
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            pass

    async def test_a_call_cancelled_while_its_cli_starts_frees_the_slot(self) -> None:
        """A turn stopped during the seconds a CLI takes to start must not strand its slot:
        a slot lost there is lost for the life of the process."""
        starting = asyncio.Event()

        async def slow_factory(options: Any, *, key: str) -> Any:
            starting.set()
            await asyncio.sleep(3600)
            return FakeSession(options, key)

        pool = _pool(per_key=1, session_factory=slow_factory)
        task = asyncio.create_task(self._borrow_once(pool))
        await starting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.live_count == 0
        pool._start = _factory  # noqa: SLF001 -- the next start should succeed
        await self._borrow_once(pool)
        await pool.aclose()

    async def test_a_cli_that_will_not_start_frees_the_slot(self) -> None:
        async def broken_factory(options: Any, *, key: str) -> Any:
            raise ClaudeCliSessionError("could not start a Claude CLI")

        pool = _pool(per_key=1, session_factory=broken_factory)
        with pytest.raises(ClaudeCliSessionError):
            await self._borrow_once(pool)
        assert pool.live_count == 0
        await pool.aclose()


class TestNothingOutlivesTheProcess:
    async def test_closing_the_pool_disposes_every_idle_session(self) -> None:
        pool = _pool()
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            pass
        await pool.aclose()
        assert FakeSession.instances[0].disposals == 1

    async def test_shutdown_during_a_call_disposes_it_exactly_once(self) -> None:
        pool = _pool()
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            await pool.aclose()
        assert FakeSession.instances[0].disposals == 1, "the borrower's cleanup signalled a dead session again"

    async def test_an_idle_session_past_its_ttl_is_evicted(self) -> None:
        pool = _pool(idle_ttl_seconds=0.01)
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            pass
        await asyncio.sleep(0.02)
        await pool._reap_once()  # noqa: SLF001
        assert FakeSession.instances[0].disposals == 1
        assert pool.live_count == 0
        await pool.aclose()

    async def test_the_reaper_does_not_inherit_the_request_that_started_it(self) -> None:
        probe: contextvars.ContextVar[str] = contextvars.ContextVar("request_probe", default="none")
        probe.set("the-request-that-started-it")
        pool = _pool(idle_ttl_seconds=600.0)
        try:
            async with pool.checkout(Options(), token=TOKEN, tool_server=None):
                pass
            assert pool._reaper is not None  # noqa: SLF001
            assert pool._reaper.get_context().get(probe, "none") == "none"  # noqa: SLF001
        finally:
            await pool.aclose()


class TestTheLaunchKey:
    def test_the_model_and_the_tool_server_are_not_part_of_it(self) -> None:
        a = Options(model=DEFAULT_FAST_MODEL, mcp_servers={"langchain-tools": object()})
        b = Options(model=DEFAULT_CHAT_MODEL, mcp_servers={})
        assert launch_key(a, TOKEN) == launch_key(b, TOKEN)

    @pytest.mark.parametrize(
        "change",
        [
            {"system_prompt": "another persona"},
            {"tools": ["WebSearch"]},
            {"permission_mode": "default"},
            {"max_turns": 99},
            {"cwd": "/elsewhere"},
            {"extra_args": {}},
        ],
    )
    def test_every_launch_time_option_is(self, change: dict[str, Any]) -> None:
        assert launch_key(Options(), TOKEN) != launch_key(Options(**change), TOKEN)

    def test_the_credential_is_part_of_it_and_the_token_is_not_in_it(self) -> None:
        assert launch_key(Options(), TOKEN) != launch_key(Options(), OTHER_TOKEN)
        assert "aaaaaaaa" not in launch_key(Options(), TOKEN)


class TestTheOrphanSweep:
    """Real processes, because the sweep's whole job is killing real ones."""

    @staticmethod
    def _dead_pid() -> int:
        for candidate in range(4_000_000, 4_000_200):
            if not os.path.exists(f"/proc/{candidate}"):
                return candidate
        pytest.skip("no free pid to stand in for a dead owner")
        raise AssertionError("unreachable")

    @staticmethod
    def _marked_child(marker: str) -> subprocess.Popen[bytes]:
        env = dict(os.environ)
        env[POOL_MARKER_ENV] = marker
        return subprocess.Popen(["sleep", "60"], env=env)

    def test_it_kills_a_cli_whose_owner_is_gone(self) -> None:
        if not os.path.isdir("/proc"):
            pytest.skip("the sweep needs /proc")
        orphan = self._marked_child(f"{self._dead_pid()}:0:orphaned")
        try:
            assert sweep_orphaned_claude_clis(grace_seconds=1.0) >= 1
            orphan.wait(timeout=5)
            assert orphan.poll() is not None, "an orphaned CLI survived the sweep"
        finally:
            if orphan.poll() is None:
                orphan.kill()
                orphan.wait(timeout=5)

    def test_it_leaves_a_live_owners_cli_alone(self) -> None:
        if not os.path.isdir("/proc"):
            pytest.skip("the sweep needs /proc")
        own = f"{os.getpid()}:{claude_cli_pool._process_start_ticks(os.getpid()) or 0}:mine"  # noqa: SLF001
        child = self._marked_child(own)
        try:
            sweep_orphaned_claude_clis(grace_seconds=1.0)
            time.sleep(0.2)
            assert child.poll() is None, "the sweep killed a live process's CLI"
        finally:
            child.kill()
            child.wait(timeout=5)

    def test_it_leaves_unmarked_processes_alone(self) -> None:
        if not os.path.isdir("/proc"):
            pytest.skip("the sweep needs /proc")
        env = {k: v for k, v in os.environ.items() if k != POOL_MARKER_ENV}
        bystander = subprocess.Popen(["sleep", "60"], env=env)
        try:
            sweep_orphaned_claude_clis(grace_seconds=1.0)
            time.sleep(0.2)
            assert bystander.poll() is None, "the sweep signalled something it did not start"
        finally:
            bystander.kill()
            bystander.wait(timeout=5)

    def test_a_recycled_pid_does_not_make_a_stale_cli_look_owned(self) -> None:
        if not os.path.isdir("/proc"):
            pytest.skip("the sweep needs /proc")
        assert claude_cli_pool._owner_is_alive(f"{os.getpid()}:1:x") is False  # noqa: SLF001
        real_ticks = claude_cli_pool._process_start_ticks(os.getpid())  # noqa: SLF001
        assert claude_cli_pool._owner_is_alive(f"{os.getpid()}:{real_ticks}:x") is True  # noqa: SLF001

    def test_killing_a_cli_takes_its_children_with_it(self) -> None:
        if not os.path.isdir("/proc"):
            pytest.skip("the descendant walk needs /proc")
        parent = subprocess.Popen(["sh", "-c", "sleep 60 & wait"])
        try:
            deadline = time.monotonic() + 5
            children: list[int] = []
            while time.monotonic() < deadline:
                children = claude_cli_pool._descendants(parent.pid)  # noqa: SLF001
                if children:
                    break
                time.sleep(0.05)
            assert children, "the test's own child never started"
            kill_process_tree(parent.pid, grace_seconds=1.0)
            parent.wait(timeout=5)
            time.sleep(0.2)
            assert all(not os.path.exists(f"/proc/{pid}") for pid in children), "a CLI's child outlived the CLI"
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)


class TestIdleCapacityIsNotHoarded:
    async def test_an_idle_session_of_another_key_makes_room_at_the_cap(self) -> None:
        """Found live: four keys each left one idle session in a four-session pool, and the fifth
        key's call ran on its own CLI "because every session is busy" -- none was."""
        pool = _pool(max_sessions=2, per_key=2)
        async with pool.checkout(Options(system_prompt="router"), token=TOKEN, tool_server=None):
            pass
        async with pool.checkout(Options(system_prompt="conversation"), token=TOKEN, tool_server=None):
            pass
        async with pool.checkout(Options(system_prompt="arguments"), token=TOKEN, tool_server=None):
            pass
        router, conversation, arguments = FakeSession.instances
        assert router.disposals == 1, "the longest-idle session was not the one that made room"
        assert conversation.disposals == 0
        assert arguments.disposals == 0
        assert pool.live_count == 2
        await pool.aclose()

    async def test_a_busy_session_is_never_evicted(self) -> None:
        pool = _pool(max_sessions=1, per_key=1)
        async with pool.checkout(Options(system_prompt="router"), token=TOKEN, tool_server=None):
            with pytest.raises(ClaudeCliPoolExhausted):
                async with pool.checkout(Options(system_prompt="conversation"), token=TOKEN, tool_server=None):
                    pass
        assert FakeSession.instances[0].disposals == 0
        await pool.aclose()

    def test_a_started_session_logs_which_launch_fields_it_was_keyed_on(self) -> None:
        from threetears.models.claude_cli_pool import launch_fingerprint

        a = launch_fingerprint(Options(system_prompt="persona A"))
        b = launch_fingerprint(Options(system_prompt="persona B"))
        differing = {name for name in a if a[name] != b[name]}
        assert differing == {"system_prompt"}
        assert "persona" not in str(a), "a system prompt leaked into the log"


class TestFailuresTheReviewFound:
    async def test_a_bare_exception_during_clear_frees_the_slot_and_the_call_still_succeeds(self) -> None:
        """The SDK surfaces a CLI that died mid-clear as a bare ``Exception``. It escaped ``_return``:
        the slot was never freed and a call that had succeeded raised out of its cleanup."""
        pool = _pool(per_key=1)
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            FakeSession.instances[0].clear_raises = Exception("the CLI exited during /clear")
        assert FakeSession.instances[0].disposals == 1
        assert pool.live_count == 0, "the slot leaked, and every later call would find the pool busy"
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            pass
        await pool.aclose()

    async def test_a_stop_during_an_eviction_does_not_strand_the_victims_slot(self) -> None:
        pool = _pool(max_sessions=1, per_key=1, checkout_timeout_seconds=5.0)
        async with pool.checkout(Options(system_prompt="router"), token=TOKEN, tool_server=None):
            pass
        victim = FakeSession.instances[0]
        victim.dispose_delay = 0.2

        async def caller() -> None:
            async with pool.checkout(Options(system_prompt="conversation"), token=TOKEN, tool_server=None):
                pass

        task = asyncio.create_task(caller())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.3)
        assert victim.disposals == 1, "the evicted session's disposal was abandoned"
        assert pool.live_count == 0, "the victim's slot leaked"
        await pool.aclose()

    async def test_an_idle_session_holds_none_of_the_last_callers_tools(self) -> None:
        pool = _pool()
        async with pool.checkout(Options(), token=TOKEN, tool_server=object()):
            pass
        assert FakeSession.instances[0].released == 1
        await pool.aclose()

    async def test_the_borrowers_context_reaches_prepare(self) -> None:
        pool = _pool()
        marker = contextvars.copy_context()
        async with pool.checkout(Options(), token=TOKEN, tool_server=object(), call_context=marker):
            pass
        assert FakeSession.instances[0].contexts == [marker]
        await pool.aclose()


class TestAToolCallRunsInItsBorrowersContext:
    """Found by review, reproduced: a reused CLI ran every tool call in the context of the caller that
    STARTED it, because the SDK spawns tool calls from a reader task created at connect time. A second
    conversation's interrupt landed in the first conversation's list and its graph never paused."""

    async def test_a_tool_call_from_the_readers_context_sees_the_borrowers_values(self) -> None:
        pytest.importorskip("claude_agent_sdk")
        from claude_agent_sdk import create_sdk_mcp_server, tool
        from mcp.types import CallToolRequest, CallToolRequestParams

        from threetears.models.claude_cli_pool import bind_tool_server_to_context

        whose: contextvars.ContextVar[str] = contextvars.ContextVar("whose_call", default="nobody")
        captured: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar("captured", default=None)
        seen: list[str] = []

        @tool("probe", "Report whose call this is.", {})
        async def probe(args: dict[str, Any]) -> dict[str, Any]:
            seen.append(whose.get())
            bucket = captured.get()
            if bucket is not None:
                bucket.append("interrupt")
            return {"content": [{"type": "text", "text": "ok"}]}

        server = create_sdk_mcp_server(name="langchain-tools", tools=[probe])["instance"]

        # Conversation A started the CLI: the SDK's reader task, and so every tool call it spawns,
        # carries A's context.
        reader_context = contextvars.Context()
        reader_context.run(whose.set, "conversation-A")

        # Conversation B borrows it.
        b_bucket: list[str] = []
        whose.set("conversation-B")
        captured.set(b_bucket)
        bound = bind_tool_server_to_context(server, contextvars.copy_context())

        handler = bound.request_handlers[CallToolRequest]
        request = CallToolRequest(method="tools/call", params=CallToolRequestParams(name="probe", arguments={}))
        await asyncio.create_task(handler(request), context=reader_context)

        assert seen == ["conversation-B"], "the tool ran in the conversation that started the CLI"
        assert b_bucket == ["interrupt"], "the borrower's interrupt was captured somewhere it will never be read"

    async def test_concurrent_tool_calls_in_one_turn_do_not_collide(self) -> None:
        pytest.importorskip("claude_agent_sdk")
        from claude_agent_sdk import create_sdk_mcp_server, tool
        from mcp.types import CallToolRequest, CallToolRequestParams

        from threetears.models.claude_cli_pool import bind_tool_server_to_context

        @tool("slow", "Wait briefly.", {})
        async def slow(args: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.01)
            return {"content": [{"type": "text", "text": "ok"}]}

        server = create_sdk_mcp_server(name="langchain-tools", tools=[slow])["instance"]
        bound = bind_tool_server_to_context(server, contextvars.copy_context())
        handler = bound.request_handlers[CallToolRequest]
        request = CallToolRequest(method="tools/call", params=CallToolRequestParams(name="slow", arguments={}))
        await asyncio.gather(handler(request), handler(request), handler(request))


class TestWhatMayShareACli:
    def test_a_call_carrying_callables_is_not_pooled(self) -> None:
        from threetears.models.claude_cli_pool import poolable

        @dataclass
        class WithHooks(Options):
            hooks: Any = None

        assert poolable(Options())
        assert not poolable(WithHooks(hooks={"PreToolUse": [object()]}))

    async def test_a_call_that_cannot_be_pooled_is_refused_so_it_falls_back(self) -> None:
        @dataclass
        class Resumed(Options):
            resume: str | None = None

        pool = _pool()
        with pytest.raises(ClaudeCliPoolExhausted, match="cannot share"):
            async with pool.checkout(Resumed(resume="session-1"), token=TOKEN, tool_server=None):
                pass
        assert FakeSession.instances == []
        await pool.aclose()

    def test_an_option_a_call_can_set_per_call_changes_the_key(self) -> None:
        """The base package accepts any option as a per-call override; every one must reach the key."""

        @dataclass
        class WithThinking(Options):
            max_thinking_tokens: int | None = None

        assert launch_key(WithThinking(), TOKEN) != launch_key(WithThinking(max_thinking_tokens=8000), TOKEN)

    def test_the_credential_in_the_environment_is_keyed_by_digest(self) -> None:
        a = Options(env={"CLAUDE_CODE_OAUTH_TOKEN": TOKEN})
        b = Options(env={"CLAUDE_CODE_OAUTH_TOKEN": OTHER_TOKEN})
        assert launch_key(a, None) != launch_key(b, None)
        assert "aaaaaaaa" not in str(claude_cli_pool.launch_fingerprint(a))

    def test_the_pool_marker_does_not_change_the_key(self) -> None:
        a = Options(env={POOL_MARKER_ENV: "1:2:one"})
        b = Options(env={POOL_MARKER_ENV: "1:2:two"})
        assert launch_key(a, TOKEN) == launch_key(b, TOKEN)


class TestTheSdkSurfaceMoving:
    async def test_repeated_structural_failures_turn_pooling_off(self) -> None:
        async def factory(options: Any, *, key: str) -> Any:
            session = FakeSession(options, key)

            async def moved(**kwargs: Any) -> None:
                error = ClaudeCliSessionError("could not prepare: '_query' has no attribute")
                error.structural = True  # type: ignore[attr-defined]
                raise error

            session.prepare = moved  # type: ignore[method-assign]
            return session

        pool = _pool(session_factory=factory, per_key=5, max_sessions=5)
        for _ in range(3):
            with pytest.raises(ClaudeCliSessionError):
                async with pool.checkout(Options(), token=TOKEN, tool_server=None):
                    pass
        with pytest.raises(ClaudeCliPoolExhausted, match="pooling is off"):
            async with pool.checkout(Options(), token=TOKEN, tool_server=None):
                pass
        assert len(FakeSession.instances) == 3, "it kept starting CLIs it could not prepare"
        await pool.aclose()

    async def test_an_ordinary_failure_does_not(self) -> None:
        pool = _pool(per_key=5, max_sessions=5)
        for _ in range(4):
            async with pool.checkout(Options(), token=TOKEN, tool_server=None):
                pass
            FakeSession.instances[-1].fail_prepare = True
            with pytest.raises(ClaudeCliSessionError):
                async with pool.checkout(Options(), token=TOKEN, tool_server=None):
                    pass
        assert pool._broken is False  # noqa: SLF001
        await pool.aclose()


class TestAToolCallThroughTheSdksOwnDispatch:
    """The context fix, pinned through the SDK's real ``tools/call`` routing rather than by calling the
    handler directly: the SDK's reader task carries the conversation that STARTED the CLI, while a
    second conversation holds it. A handler on the borrowed session must run in the borrower's."""

    async def test_the_borrowers_config_is_the_one_a_handler_sees(self) -> None:
        pytest.importorskip("claude_agent_sdk")
        from claude_agent_sdk import create_sdk_mcp_server
        from claude_agent_sdk import tool as sdk_tool
        from claude_agent_sdk._internal.query import Query
        from langchain_core.runnables.config import ensure_config, var_child_runnable_config

        from threetears.models.claude_cli_pool import bind_tool_server_to_context

        seen_config: list[str | None] = []

        @sdk_tool("confirm", "Ask a human to confirm.", {"type": "object", "properties": {"x": {"type": "string"}}})
        async def confirm(args: dict[str, Any]) -> dict[str, Any]:
            seen_config.append(ensure_config().get("configurable", {}).get("who"))
            return {"content": [{"type": "text", "text": "ok"}]}

        instance = create_sdk_mcp_server(name="langchain-tools", version="1.0.0", tools=[confirm])["instance"]

        def become(who: str) -> None:
            var_child_runnable_config.set({"configurable": {"who": who}})

        reader_context = contextvars.Context()
        reader_context.run(become, "A")

        async def borrower() -> None:
            become("B")
            query = Query.__new__(Query)
            query.sdk_mcp_servers = {
                "langchain-tools": bind_tool_server_to_context(instance, contextvars.copy_context())
            }
            message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "confirm", "arguments": {"x": "y"}},
            }
            await asyncio.create_task(query._handle_sdk_mcp_request("langchain-tools", message), context=reader_context)  # noqa: SLF001

        await asyncio.create_task(borrower())

        assert seen_config == ["B"], "the handler saw the conversation that started the CLI"


class TestReturningASessionIsBounded:
    async def test_the_tool_release_on_return_uses_the_short_clear_timeout(self) -> None:
        seen: list[float] = []
        pool = _pool(clear_timeout_seconds=1.5)
        async with pool.checkout(Options(), token=TOKEN, tool_server=None):
            session = FakeSession.instances[0]

            async def record(*, timeout: float) -> None:
                seen.append(timeout)

            session.release_tools = record  # type: ignore[method-assign]
        assert seen == [1.5], "a hung CLI could hold a finished call for the control request's default 30 s"
        await pool.aclose()

    def test_a_session_store_is_caller_state_and_not_pooled(self) -> None:
        from threetears.models.claude_cli_pool import poolable

        @dataclass
        class WithStore(Options):
            session_store: Any = None

        assert not poolable(WithStore(session_store=object()))
