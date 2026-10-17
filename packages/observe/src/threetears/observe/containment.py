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
from collections.abc import Mapping
from contextlib import AbstractContextManager
from types import FrameType, TracebackType

from threetears.observe.metrics import counter

__all__ = ["contain", "contained"]

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
    level: int,
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
    :param level: the record's level, WARNING or above
    :ptype level: int
    :raises ValueError: ``level`` is below WARNING
    """
    if level < logging.WARNING:
        raise ValueError(f"a contained failure is logged at WARNING or above, not {logging.getLevelName(level)}")
    exc_type = type(exc).__name__
    log.log(
        level,
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
    level: int = logging.WARNING,
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
    :param level: the record's level; WARNING by default, ERROR where the failure is one an
        operator acts on. nothing below WARNING: a failure logged where nobody reads it is
        the silence this exists to end
    :ptype level: int
    :raises ValueError: ``level`` is below WARNING
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
        _record(exc, _site_of(frame), self._log, self._what, self._extra, logging.WARNING)
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
