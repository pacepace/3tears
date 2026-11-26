"""contained failures: the one sanctioned way to absorb a broad exception.

a failure either reaches the caller, or it is recorded where an operator will see it --
never neither. a broad ``except Exception`` that neither re-raises nor calls :func:`contain`
does neither: whatever the handler logs, the caller carries on with an answer it cannot tell
from a real one, and nothing counts how often that happens.

:func:`contain` is for a handler at a boundary -- a supervisor loop, a best-effort side
effect, a tool that turns the failure into an error result -- that has decided the failure
stops here. it does the recording the decision owes, the same way every time:

- a WARNING carrying the traceback, its message the caller's own, and ``contained_site`` and
  ``exc_type`` added to its ``extra_data``;
- ``contained_errors_total{site, exc_type}``, so how often each site absorbs something, and
  what, is a query rather than a grep through logs.

``site`` is ``module:qualified.function`` of the frame that caught the exception, read off the
exception's own traceback. it names a function rather than a line so the series survives an
edit above it.

:func:`contained` is the same thing for a side effect whose failure falls through::

    with contained("pool did not close", log=log):
        await pool.close()

it absorbs :class:`Exception` only. ``CancelledError``, ``KeyboardInterrupt`` and
``SystemExit`` pass through: a cancellation is a request from the caller, and absorbing one
leaves the caller running after it was told to stop.

what this does NOT do is make a handler honest. a handler that contains a failure and then
returns ``None`` or ``[]`` still hands its caller a default that reads as an answer; the
metric only makes the lie countable.
"""

from __future__ import annotations

import inspect
import logging
import os
import shutil
import traceback
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from types import FrameType, TracebackType
from typing import Literal

from threetears.observe.metrics import counter

__all__ = ["ContainedLevel", "contain", "contained", "frames_released", "remove_tree"]

#: the levels a contained failure is logged at: nothing below WARNING, since a failure logged where
#: nobody reads it is the silence this exists to end. named rather than numeric so a caller needs no
#: stdlib ``logging`` import to pick one.
type ContainedLevel = Literal["warning", "error", "critical"]

_LEVELS: dict[str, int] = {"warning": logging.WARNING, "error": logging.ERROR, "critical": logging.CRITICAL}

_METRIC = "contained_errors_total"
_METRIC_HELP = "exceptions a boundary absorbed on purpose, by catching function and exception type"


def _site_of(frame: FrameType | None) -> str:
    """name a frame as ``module:qualified.function``.

    :param frame: the frame that caught the exception, or ``None`` when none is known
    :ptype frame: FrameType | None
    :return: the site label
    :rtype: str
    """
    if frame is None:
        return "<unknown>"
    module = frame.f_globals.get("__name__", "<unknown>")
    return f"{module}:{frame.f_code.co_qualname}"


def _record(
    exc: Exception,
    site: str,
    log: logging.Logger,
    what: str,
    extra: Mapping[str, object] | None,
    level: ContainedLevel,
) -> None:
    """log ``exc`` with its traceback and count it against ``site``.

    :param exc: the exception being absorbed
    :ptype exc: Exception
    :param site: the catching function, as ``module:qualified.function``
    :ptype site: str
    :param log: the catching module's logger
    :ptype log: logging.Logger
    :param what: the log message, verbatim
    :ptype what: str
    :param extra: structured fields for the record's ``extra_data``
    :ptype extra: Mapping[str, object] | None
    :param level: the record's level
    :ptype level: ContainedLevel
    :raises ValueError: ``level`` is not one of ``"warning"``, ``"error"``, ``"critical"``
    """
    if level not in _LEVELS:
        raise ValueError(f"a contained failure is logged at warning, error or critical, not {level!r}")
    exc_type = type(exc).__name__
    log.log(
        _LEVELS[level],
        what,
        exc_info=exc,
        extra={"extra_data": {**(extra or {}), "contained_site": site, "exc_type": exc_type}},
    )
    counter(_METRIC, description=_METRIC_HELP, label_names=("site", "exc_type")).labels(
        site=site, exc_type=exc_type
    ).inc()


def contain(
    exc: Exception,
    *,
    log: logging.Logger,
    what: str,
    extra: Mapping[str, object] | None = None,
    level: ContainedLevel = "warning",
) -> None:
    """record ``exc`` as a failure this boundary absorbs on purpose.

    call it from the ``except`` block that decided the failure stops here, in place of the
    log call it would otherwise make. it does not change control flow: the handler still
    returns, continues or falls through as it says.

    :param exc: the caught exception
    :ptype exc: Exception
    :param log: the catching module's logger
    :ptype log: logging.Logger
    :param what: the WARNING's message, verbatim, e.g. ``"grant cache refresh failed"``
    :ptype what: str
    :param extra: structured fields for the record's ``extra_data``; ``contained_site`` and
        ``exc_type`` are added to them
    :ptype extra: Mapping[str, object] | None
    :param level: ``"warning"`` by default, ``"error"`` where the failure is one an operator acts on
    :ptype level: ContainedLevel
    :raises ValueError: ``level`` is not one of ``"warning"``, ``"error"``, ``"critical"``
    """
    traceback = exc.__traceback__
    if traceback is not None:
        frame: FrameType | None = traceback.tb_frame
    else:
        # an exception handed over without having been raised has no traceback; the caller
        # of contain() is then the best name for where it stopped
        here = inspect.currentframe()
        frame = here.f_back if here is not None else None
    _record(exc, _site_of(frame), log, what, extra, level)


class _Contained:
    """the context manager :func:`contained` returns."""

    def __init__(self, what: str, log: logging.Logger, extra: Mapping[str, object] | None) -> None:
        self._what = what
        self._log = log
        self._extra = extra

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if not isinstance(exc, Exception):
            return False
        # the traceback handed to __exit__ starts at the frame holding the `with` block
        frame = traceback.tb_frame if traceback is not None else None
        _record(exc, _site_of(frame), self._log, self._what, self._extra, "warning")
        return True


def contained(
    what: str,
    *,
    log: logging.Logger,
    extra: Mapping[str, object] | None = None,
) -> AbstractContextManager[None]:
    """absorb and record an :class:`Exception` raised in the block; let everything else through.

    :param what: the WARNING's message if the block fails, e.g. ``"scratch directory not removed"``
    :ptype what: str
    :param log: the calling module's logger
    :ptype log: logging.Logger
    :param extra: structured fields for the record's ``extra_data``
    :ptype extra: Mapping[str, object] | None
    :return: a context manager that contains an :class:`Exception` and passes anything else
    :rtype: AbstractContextManager[None]
    """
    return _Contained(what, log, extra)


def remove_tree(path: str | os.PathLike[str], *, log: logging.Logger, what: str) -> None:
    """remove a directory tree, best-effort, recording each part that would not go.

    ``shutil.rmtree(..., ignore_errors=True)`` hides a failed removal from everyone: the tree stays, and whatever
    counts its bytes or retries it carries on blind. Here a path already gone is the outcome wanted and says
    nothing; any other failure is contained (:func:`contain`) -- logged with the path and counted -- and the
    removal carries on with the rest of the tree. Never raises.

    :param path: the tree
    :ptype path: str | os.PathLike[str]
    :param log: the calling module's logger
    :ptype log: logging.Logger
    :param what: what the tree is, for the record, e.g. ``"retired shared Arrow version"``
    :ptype what: str
    :return: nothing
    :rtype: None
    """

    def _failed(_function: Callable[..., object], failed_path: str, exc: BaseException) -> None:
        if isinstance(exc, FileNotFoundError) or not isinstance(exc, Exception):
            return
        contain(exc, log=log, what=f"{what} could not be removed", extra={"path": str(failed_path)})

    shutil.rmtree(path, onexc=_failed)


def frames_released[E: BaseException](exc: E) -> E:
    """``exc``, to be kept past its ``except`` block (on an object, in a list, for a later raise), let go of the frames
    its traceback holds: where each frame was stays with it as a note, and the frames -- with every local they held --
    go.

    WHY: a traceback holds the frame that caught it, and that frame holds its locals, so an exception kept on an
    object the catching frame holds (``result.error = exc``) is a reference cycle: object -> exception -> traceback ->
    frame -> object. Reference counting never frees a cycle; only a full collection does, which walks the whole heap
    with the interpreter lock held. Every local of every frame in the traceback is kept until then: on reports' pods
    a state's exported rows (zero-copy Arrow over shared memory) were kept so (2026-11-01, 2.9 GiB pinned), and a pod
    that runs no full collection (reports, 2026-11-25) would keep them for good. Clearing the frames where it is caught
    does not do: the catching frame is still running, and every frame below it holds it as its caller.

    The exception's cause and context (and an exception group's members) are released alike. Each gets its traceback
    as text in a note (``traceback.format_tb``), so a later raise or log of it still says where it was raised.

    :param exc: the exception caught
    :ptype exc: BaseException
    :return: the same exception, holding no frame
    :rtype: BaseException
    """
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        each = pending.pop()
        if id(each) in seen:
            continue
        seen.add(id(each))
        held = each.__traceback__
        if held is not None:
            each.add_note("raised at (its frames let go):\n" + "".join(traceback.format_tb(held)).rstrip())
            each.__traceback__ = None
        pending.extend(found for found in (each.__cause__, each.__context__) if found is not None)
        if isinstance(each, BaseExceptionGroup):
            pending.extend(each.exceptions)
    return exc
