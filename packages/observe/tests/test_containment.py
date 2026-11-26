"""unit tests for ``threetears.observe.contain`` and ``threetears.observe.contained``.

a contained failure is recorded the same way wherever it is absorbed: a WARNING with the
traceback, and one count on ``contained_errors_total`` against the function that caught it.
``contained`` absorbs :class:`Exception` and nothing wider, so a cancellation still reaches
the task that was asked to stop.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import traceback
import weakref
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

from threetears.observe import contain, contained, frames_released, get_logger, remove_tree

_log = get_logger(__name__)
_MODULE = __name__


def _count(site: str, exc_type: str) -> float:
    value = REGISTRY.get_sample_value("contained_errors_total", {"site": site, "exc_type": exc_type})
    return value or 0.0


def _catches_and_contains() -> None:
    try:
        raise ValueError("the upstream said no")
    except ValueError as exc:
        contain(exc, log=_log, what="upstream refused", extra={"upstream": "pricing"})


def _raises_from_a_callee() -> None:
    raise LookupError("deep down")


def _catches_a_callees_failure() -> None:
    try:
        _raises_from_a_callee()
    except LookupError as exc:
        contain(exc, log=_log, what="lookup failed")


def _contains_a_block() -> None:
    with contained("scratch file not removed", log=_log):
        raise OSError("read-only file system")


class TestContain:
    def test_logs_a_warning_that_carries_the_traceback(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _catches_and_contains()
        (record,) = [r for r in caplog.records if r.getMessage() == "upstream refused"]
        assert record.levelno == logging.WARNING
        assert record.exc_info is not None and isinstance(record.exc_info[1], ValueError)

    def test_keeps_the_callers_fields_and_names_the_site(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _catches_and_contains()
        (record,) = [r for r in caplog.records if r.getMessage() == "upstream refused"]
        assert getattr(record, "extra_data") == {
            "upstream": "pricing",
            "contained_site": f"{_MODULE}:_catches_and_contains",
            "exc_type": "ValueError",
        }

    def test_counts_against_the_function_that_caught_it(self) -> None:
        site = f"{_MODULE}:_catches_and_contains"
        before = _count(site, "ValueError")
        _catches_and_contains()
        assert _count(site, "ValueError") == before + 1

    def test_the_site_is_the_catcher_not_the_raiser(self) -> None:
        catcher = f"{_MODULE}:_catches_a_callees_failure"
        raiser = f"{_MODULE}:_raises_from_a_callee"
        before = _count(catcher, "LookupError")
        _catches_a_callees_failure()
        assert _count(catcher, "LookupError") == before + 1
        assert _count(raiser, "LookupError") == 0.0

    def test_an_exception_never_raised_is_named_after_the_caller(self) -> None:
        site = f"{_MODULE}:TestContain.test_an_exception_never_raised_is_named_after_the_caller"
        before = _count(site, "RuntimeError")
        contain(RuntimeError("handed over, never raised"), log=_log, what="handed-over failure")
        assert _count(site, "RuntimeError") == before + 1

    def test_logs_at_error_when_asked(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            try:
                raise TimeoutError("gave up")
            except TimeoutError as exc:
                contain(exc, log=_log, what="retries exhausted", level="error")
        (record,) = [r for r in caplog.records if r.getMessage() == "retries exhausted"]
        assert record.levelno == logging.ERROR

    def test_refuses_a_level_it_does_not_name(self) -> None:
        with pytest.raises(ValueError, match="warning, error or critical"):
            contain(RuntimeError("quiet"), log=_log, what="whispered", level="debug")  # type: ignore[arg-type]


class TestContained:
    def test_absorbs_an_exception_and_counts_it_against_the_with_block(self) -> None:
        site = f"{_MODULE}:_contains_a_block"
        before = _count(site, "OSError")
        _contains_a_block()
        assert _count(site, "OSError") == before + 1

    def test_a_clean_block_records_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING), contained("doing nothing", log=_log):
            pass
        assert not [r for r in caplog.records if r.getMessage() == "doing nothing"]

    async def test_a_cancellation_is_not_absorbed(self) -> None:
        async def _stops_when_told() -> None:
            with contained("waiting forever", log=_log):
                await asyncio.Event().wait()

        task = asyncio.create_task(_stops_when_told())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    def test_a_keyboard_interrupt_is_not_absorbed(self) -> None:
        with pytest.raises(KeyboardInterrupt), contained("an interactive step", log=_log):
            raise KeyboardInterrupt


class TestRemoveTree:
    def test_a_tree_is_removed(self, tmp_path: Path) -> None:
        tree = tmp_path / "tree"
        (tree / "inner").mkdir(parents=True)
        (tree / "inner" / "file").write_text("x")
        remove_tree(tree, log=_log, what="a test tree")
        assert not tree.exists()

    def test_a_tree_already_gone_says_nothing(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)
        remove_tree(tmp_path / "never-made", log=_log, what="a test tree")
        assert caplog.records == []

    def test_a_part_that_will_not_go_is_logged_and_the_rest_still_goes(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        tree = tmp_path / "tree"
        locked = tree / "locked"
        locked.mkdir(parents=True)
        (locked / "stuck").write_text("x")
        (tree / "free").write_text("y")
        locked.chmod(0o500)  # nothing in it can be unlinked
        caplog.set_level(logging.WARNING)
        try:
            remove_tree(tree, log=_log, what="a test tree")
            assert "a test tree could not be removed" in caplog.text
            assert (locked / "stuck").exists() and not (tree / "free").exists()
        finally:
            locked.chmod(0o700)


class _Holder:
    """what keeps a caught failure past its except block, as a result keeps the error it hands on."""

    error: BaseException | None = None


class _Rows:
    """a local the raising frames hold: what a reference cycle through them would keep."""


def _raises(rows: _Rows) -> None:
    try:
        raise KeyError("the inner failure")
    except KeyError as inner:
        raise ValueError("the failure kept") from inner


def _keeps(release: bool) -> tuple[_Holder, weakref.ref[_Rows], weakref.ref[_Holder]]:
    holder, rows = _Holder(), _Rows()
    try:
        _raises(rows)
    except ValueError as exc:
        holder.error = frames_released(exc) if release else exc
    return holder, weakref.ref(rows), weakref.ref(holder)


def test_a_failure_kept_with_its_frames_released_makes_no_reference_cycle() -> None:
    """a failure kept on an object its catching frame holds is a cycle (object -> failure -> traceback -> frame ->
    object), freed only by a full collection, with every local of every frame in it. Released, reference counting
    frees it all, and where it was raised -- its cause's too -- stays with it as text."""
    collecting = gc.isenabled()
    gc.disable()
    try:
        holder, rows, held = _keeps(release=False)
        del holder
        assert rows() is not None and held() is not None, (
            "the plain kept failure made no cycle: the test proves nothing"
        )
        gc.collect()
        holder, rows, held = _keeps(release=True)
        assert holder.error is not None and holder.error.__traceback__ is None
        told = "".join(traceback.format_exception(holder.error))
        assert "in _raises" in told and "the inner failure" in told and "in _keeps" in told
        del holder
        assert rows() is None and held() is None, "the released failure still held its frames"
    finally:
        if collecting:
            gc.enable()
