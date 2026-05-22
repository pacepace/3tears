"""tests for :class:`AsyncSyncBridge` -- bounded executor + cancel-aware submit.

scope:

- construction: max_workers + name properties match constructor args
- ``to_thread_with_cancel`` runs the callable + returns the result
- cancellation: cancelling the awaiter fires ``cancel_cb`` BEFORE the
  CancelledError reaches the calling code
- async ``cancel_cb`` is awaited
- ``close()`` is idempotent + uses ``shutdown(wait=False)``
- post-close ``to_thread_with_cancel`` raises RuntimeError
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import pytest

from threetears.datasources.drivers._sync_bridge import AsyncSyncBridge


class TestConstruction:
    """bridge exposes the configured executor knobs."""

    @pytest.mark.asyncio
    async def test_max_workers_property(self) -> None:
        bridge = AsyncSyncBridge(max_workers=4, name="test-bridge")
        try:
            assert bridge.max_workers == 4
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_name_property(self) -> None:
        bridge = AsyncSyncBridge(max_workers=2, name="my-bridge")
        try:
            assert bridge.name == "my-bridge"
        finally:
            await bridge.close()


class TestToThreadWithCancel:
    """submit + await; success path returns the callable's result."""

    @pytest.mark.asyncio
    async def test_returns_callable_result(self) -> None:
        bridge = AsyncSyncBridge(max_workers=2, name="test-rt")
        try:
            result = await bridge.to_thread_with_cancel(
                lambda: 42,
                cancel_cb=lambda: None,
            )
            assert result == 42
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_propagates_exception_from_callable(self) -> None:
        bridge = AsyncSyncBridge(max_workers=2, name="test-exc")
        try:

            def boom() -> int:
                raise ValueError("synthetic")

            with pytest.raises(ValueError, match="synthetic"):
                await bridge.to_thread_with_cancel(boom, cancel_cb=lambda: None)
        finally:
            await bridge.close()


class TestCancellation:
    """cancelling the await fires cancel_cb then re-raises CancelledError."""

    @pytest.mark.asyncio
    async def test_cancel_fires_callback_sync(self) -> None:
        bridge = AsyncSyncBridge(max_workers=2, name="test-cancel")
        try:
            cancel_calls: list[str] = []

            def slow() -> int:
                # the test cancels the asyncio caller; the worker
                # thread keeps running but we don't wait on it after.
                time.sleep(0.5)
                return 0

            async def run() -> int:
                return await bridge.to_thread_with_cancel(
                    slow,
                    cancel_cb=lambda: cancel_calls.append("cancelled"),
                )

            task = asyncio.create_task(run())
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancel_calls == ["cancelled"]
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_cancel_awaits_async_callback(self) -> None:
        bridge = AsyncSyncBridge(max_workers=2, name="test-async-cancel")
        try:
            cancel_calls: list[str] = []

            def slow() -> int:
                time.sleep(0.5)
                return 0

            async def async_cancel() -> None:
                cancel_calls.append("async-cancelled")

            async def run() -> int:
                return await bridge.to_thread_with_cancel(
                    slow,
                    cancel_cb=async_cancel,
                )

            task = asyncio.create_task(run())
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancel_calls == ["async-cancelled"]
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_cancel_callback_exception_suppressed(self) -> None:
        """callback failure MUST NOT mask the CancelledError."""
        bridge = AsyncSyncBridge(max_workers=2, name="test-callback-exc")
        try:

            def slow() -> int:
                time.sleep(0.5)
                return 0

            def bad_cb() -> None:
                raise RuntimeError("cancel hook broke")

            async def run() -> int:
                return await bridge.to_thread_with_cancel(slow, cancel_cb=bad_cb)

            task = asyncio.create_task(run())
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await bridge.close()


class TestClose:
    """close() is idempotent and rejects subsequent submits."""

    @pytest.mark.asyncio
    async def test_close_then_submit_raises(self) -> None:
        bridge = AsyncSyncBridge(max_workers=2, name="test-close")
        await bridge.close()
        with pytest.raises(RuntimeError, match="closed"):
            await bridge.to_thread_with_cancel(lambda: 0, cancel_cb=lambda: None)

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        bridge = AsyncSyncBridge(max_workers=2, name="test-close-idempotent")
        await bridge.close()
        await bridge.close()
        # no exception means idempotency holds

    @pytest.mark.asyncio
    async def test_close_does_not_wait_for_workers(self) -> None:
        """close() returns promptly even if workers are mid-call (DS-09-12)."""
        bridge = AsyncSyncBridge(max_workers=2, name="test-close-busy")
        proceed = threading.Event()

        def long_running() -> int:
            # blocks until the test releases the event; if close()
            # waited on workers this future would never settle and
            # the test would time out.
            proceed.wait(timeout=5)
            return 1

        # fire and forget through the public API: schedule the task
        # but don't await it. cancellation in the asyncio caller is
        # irrelevant -- the worker thread is what matters for the
        # close() shutdown semantics under test.
        bg_task = asyncio.create_task(
            bridge.to_thread_with_cancel(long_running, cancel_cb=lambda: None)
        )
        # give the executor a moment to actually pick up the work
        await asyncio.sleep(0.05)
        # close should return promptly even with the worker blocked
        start = time.monotonic()
        await bridge.close()
        elapsed = time.monotonic() - start
        # release the worker so it doesn't hold the executor forever
        proceed.set()
        # if close() blocked on the worker, elapsed would be ~5s
        assert elapsed < 1.0, (
            f"close() blocked on the executor for {elapsed:.2f}s; "
            "should use shutdown(wait=False)"
        )
        # drain the background task so pytest doesn't warn about
        # unawaited tasks; we don't care about its result here.
        with contextlib.suppress(Exception):
            await bg_task


class TestBoundedExecutorSize:
    """the executor is sized to max_workers; verify via thread count."""

    @pytest.mark.asyncio
    async def test_executor_thread_count_bounded(self) -> None:
        bridge = AsyncSyncBridge(max_workers=3, name="test-bound")
        try:
            barrier = threading.Barrier(3, timeout=5)

            def block_until_three() -> int:
                barrier.wait()
                return 1

            tasks = [
                asyncio.create_task(
                    bridge.to_thread_with_cancel(block_until_three, cancel_cb=lambda: None)
                )
                for _ in range(3)
            ]
            results = await asyncio.gather(*tasks)
            assert results == [1, 1, 1]
            # if the executor cap were lower than 3, the barrier
            # would time out and the test would fail above
        finally:
            await bridge.close()
