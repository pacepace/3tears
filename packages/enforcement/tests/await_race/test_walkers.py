"""tests for the await-race walker: a long-lived value written back from a read taken before an ``await``.

Each case is the smallest module showing one shape, with the targets it must flag (empty: it must flag nothing).
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest

from threetears.enforcement.await_race import AwaitRaceConfig, find_await_race_violations, run_await_race_enforcement


def _flagged(source: str) -> list[str]:
    with tempfile.TemporaryDirectory() as directory:
        module = Path(directory) / "mod.py"
        module.write_text(textwrap.dedent(source))
        return [violation.symbol for violation in find_await_race_violations((module,))]


#: reports' refresh_service before its fix of 2026-11-23: the tuple is read before the await inside the same expression, so two
#: loads ending together each rebuilt from the same tuple and one was lost
_REFRESH_BEFORE = """
class RefreshService:
    async def _side_load(self, side, scope, state):
        state.loading += 1
        try:
            report = await self._watched(side, scope)
        except BaseException as failed:
            state.loading -= 1
            await self.scopes.restore_side(scope)
            if not isinstance(failed, Exception):
                raise
            state.failed = with_entry(state.failed, side, failed)
            return False
        else:
            state.loading -= 1
            state.failed = without_entry(state.failed, side)
            state.loads = (*state.loads, await self._after_load(report, side))
        return True

    async def _alone(self, state):
        scope = await self.scopes.take()
        report = await self._watched("both", scope)
        state.loading -= 1
        if scope.clear and report.cleared is not None:
            await self.scopes.cleared(scope)
        state.loads = (*state.loads, await self._after_load(report, "both"))
"""

#: and its fix: the load's end awaited first, the loads read after it
_REFRESH_FIXED = _REFRESH_BEFORE.replace(
    "state.loads = (*state.loads, await self._after_load(report, side))",
    "ended = await self._after_load(report, side)\n            state.loads = (*state.loads, ended)",
).replace(
    'state.loads = (*state.loads, await self._after_load(report, "both"))',
    'ended = await self._after_load(report, "both")\n        state.loads = (*state.loads, ended)',
)


def test_the_refresh_service_shape_is_found() -> None:
    assert _flagged(_REFRESH_BEFORE) == ["_side_load:state.loads", "_alone:state.loads"]


def test_the_refresh_service_fix_is_clean() -> None:
    assert "ended = await" in _REFRESH_FIXED
    assert _flagged(_REFRESH_FIXED) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            """
            async def f(self):
                seen = self._seen
                await self.flush()
                self._seen = seen | {1}
            """,
            ["f:self._seen"],
            id="read-into-a-local-await-write-back",
        ),
        pytest.param(
            """
            async def f(self):
                self._total += await self.count()
            """,
            ["f:self._total"],
            id="augmented-assignment-across-its-own-await",
        ),
        pytest.param(
            """
            async def f(self):
                self._x = self._x + await self.more()
            """,
            ["f:self._x"],
            id="read-left-of-the-await-in-one-expression",
        ),
        pytest.param(
            """
            async def f(self, rows):
                self._index = await asyncio.to_thread(rebuild, self._index, rows)
            """,
            ["f:self._index"],
            id="snapshot-handed-to-a-thread-and-written-back",
        ),
        pytest.param(
            """
            async def f(self, exc):
                self._times = await _on_error(exc, self._times)
            """,
            ["f:self._times"],
            id="module-function-computing-from-the-stale-value",
        ),
        pytest.param(
            """
            async def f(self, cond):
                held = self._held
                if cond:
                    await self.wait()
                self._held = (*held, 1)
            """,
            ["f:self._held"],
            id="await-on-one-path-only",
        ),
        pytest.param(
            """
            async def f(self):
                held = self._held
                while True:
                    self._held = (*held, 1)
                    await self.tick()
            """,
            ["f:self._held"],
            id="read-before-the-loop-write-in-the-next-turn",
        ),
        pytest.param(
            """
            async def f(self):
                held = self._held
                async with self._lock:
                    self._held = (*held, 1)
            """,
            ["f:self._held"],
            id="async-with-entry-is-an-await",
        ),
        pytest.param(
            """
            async def f(self, items):
                held = self._held
                async for item in items:
                    pass
                self._held = (*held, 1)
            """,
            ["f:self._held"],
            id="async-for-step-is-an-await",
        ),
        pytest.param(
            """
            _HOLDER = Holder()
            async def f():
                value = _HOLDER.value
                await refresh()
                _HOLDER.value = value + 1
            """,
            ["f:_HOLDER.value"],
            id="module-holder-attribute",
        ),
        pytest.param(
            """
            _SEEN = frozenset()
            async def f(key):
                global _SEEN
                seen = _SEEN
                await refresh()
                _SEEN = seen | {key}
            """,
            ["f:_SEEN"],
            id="declared-global",
        ),
        pytest.param(
            """
            async def f(self):
                try:
                    held = self._held
                    await self.work()
                except Exception:
                    self._held = held
            """,
            ["f:self._held"],
            id="restored-in-a-handler-after-the-await",
        ),
    ],
)
def test_stale_write_backs_are_found(source: str, expected: list[str]) -> None:
    assert _flagged(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            """
            async def f(self):
                await self.flush()
                seen = self._seen
                self._seen = seen | {1}
            """,
            id="read-after-the-await",
        ),
        pytest.param(
            """
            async def f(self):
                self._x = (await self.more(), *self._x)
            """,
            id="read-right-of-the-await-in-one-expression",
        ),
        pytest.param(
            """
            async def f(self):
                seen = self._seen
                self._seen = seen | {1}
                await self.flush()
            """,
            id="read-and-write-with-no-await-between",
        ),
        pytest.param(
            """
            async def f(self, key):
                picked = [k for k in self._keys if k != key]
                await self.flush()
                for k in picked:
                    self._passed = without(self._passed, k)
            """,
            id="key-picked-before-value-read-fresh",
        ),
        pytest.param(
            """
            async def f(self):
                held = self._held
                await self.wait()
                if self._held is held:
                    self._held = (*held, 1)
            """,
            id="compare-and-set-after-the-await",
        ),
        pytest.param(
            """
            async def f(self):
                held = self._held
                await self.wait()
                if self._held is not held:
                    return
                self._held = (*held, 1)
            """,
            id="compare-and-return-after-the-await",
        ),
        pytest.param(
            """
            async def f(self, cond):
                held = self._held
                if cond:
                    await self.wait()
                    return
                self._held = (*held, 1)
            """,
            id="the-path-with-the-await-returns-first",
        ),
        pytest.param(
            """
            async def f(self, bucket):
                revision = self._revision
                new = await bucket.update(self.key, revision=revision)
                self._revision = new
            """,
            id="another-objects-answer-carries-nothing",
        ),
        pytest.param(
            """
            async def f(self):
                if self._client is None:
                    self._client = await connect()
                return self._client
            """,
            id="lazy-fill-a-condition-carries-no-value",
        ),
        pytest.param(
            """
            async def f(self, other):
                held = self._held
                await self.wait()
                held = other._held
                self._held = (*held, 1)
            """,
            id="local-bound-again-after-the-await",
        ),
        pytest.param(
            """
            async def f(self, states):
                for state in states:
                    loads = state.loads
                    state.loads = (*loads, 1)
                    await self.tick()
            """,
            id="loop-root-bound-again-each-turn",
        ),
        pytest.param(
            """
            def outer(body):
                streamed = 0

                async def counted():
                    nonlocal streamed
                    async for chunk in body:
                        streamed += len(chunk)
                        yield chunk

                return counted
            """,
            id="nonlocal-counted-each-step-with-no-await-between",
        ),
        pytest.param(
            """
            def f(self):
                held = self._held
                self._held = (*held, 1)
            """,
            id="sync-function",
        ),
        pytest.param(
            """
            async def f(self):
                held = self._held
                async def inner():
                    await self.wait()
                self._held = (*held, 1)
            """,
            id="await-in-a-nested-function-is-not-this-ones",
        ),
    ],
)
def test_safe_shapes_are_clean(source: str) -> None:
    assert _flagged(source) == []


def test_the_report_names_the_read_and_the_await(tmp_path: Path) -> None:
    module = tmp_path / "mod.py"
    module.write_text("async def f(self):\n    seen = self._seen\n    await g()\n    self._seen = seen\n")
    (violation,) = find_await_race_violations((module,))
    assert violation.line == 4
    assert "read at line 2" in violation.reason
    assert "await at line 3" in violation.reason


def test_the_runner_fails_on_a_finding_and_skips_non_serving_roots(tmp_path: Path) -> None:
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    racy = "async def f(self):\n    seen = self._seen\n    await g()\n    self._seen = seen\n"
    (src / "tool.py").write_text(racy)
    run_await_race_enforcement(
        AwaitRaceConfig(repo_root=tmp_path, src_roots=(src,), non_serving_roots=(src / "tool.py",))
    )
    (src / "serving.py").write_text(racy)
    with pytest.raises(pytest.fail.Exception, match="1 violation"):
        run_await_race_enforcement(
            AwaitRaceConfig(repo_root=tmp_path, src_roots=(src,), non_serving_roots=(src / "tool.py",))
        )
