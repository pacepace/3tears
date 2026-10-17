"""The package's loggers: stdlib :mod:`logging`, under the ``threetears.evals`` name, and nothing else.

3tears-evals depends on no other 3tears package, so it does not take its loggers from
:mod:`threetears.observe`. Every module asks this one for its logger, which keeps stdlib logging in
one place: the repository's convention gate (``tests/enforcement/test_codebase_conventions.py``)
refuses ``logging.getLogger`` everywhere else, and exempts this file by name.

**A host's logging configuration still applies.** Every logger here is named for its module, so it
sits under ``threetears.evals`` and so under ``threetears``: a handler, formatter or level a host sets
on either (``threetears.observe.configure_logging`` included) reaches every record. What it does not
get is the call-site class :mod:`threetears.observe`'s logger class adds; its formatter falls back to
the record's own file, function and line.

**Nothing prints unless the host says so.** One :class:`logging.NullHandler` on ``threetears.evals``,
the standard library's rule for a library: a host that configured no logging gets no output from the
engine, rather than the interpreter's last-resort handler writing warnings to stderr. It is the
behaviour the package had through :mod:`threetears.observe`, which attached the same handler.

It lives in :mod:`threetears.evals.schema` because that is the one package every other may import. It
is not a public root's name: a host configures logging by logger name, never through this module.

**A broad handler records what it absorbs through** :func:`contain` (or :func:`contained` for a block
whose failure falls through). It is :func:`threetears.observe.contain`'s contract without the metric,
kept here because this package imports no other 3tears package: the message verbatim at WARNING or
above, the traceback, and ``contained_site`` (``module:Qualified.name`` of the catching function) and
``exc_type`` in the record's ``extra_data``. The repository's no-silent-swallow gate accepts a broad
``except`` that calls it, and its test tripwire reads the same fields. See ``docs/exceptions.md``.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from contextlib import AbstractContextManager
from types import FrameType, TracebackType

__all__ = ["PACKAGE_LOGGER", "contain", "contained", "get_logger"]

#: The logger every module's logger sits under.
PACKAGE_LOGGER = "threetears.evals"

logging.getLogger(PACKAGE_LOGGER).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a module of this package.

    :param name: The module's ``__name__``.
    :ptype name: str
    :return: The stdlib logger of that name.
    :rtype: logging.Logger
    """
    return logging.getLogger(name)


def _site_of(frame: FrameType | None) -> str:
    """Name a frame as ``module:qualified.function``.

    :param frame: The frame that caught the exception, or ``None`` when none is known.
    :ptype frame: FrameType | None
    :return: The site.
    :rtype: str
    """
    if frame is None:
        return "<unknown>"
    return f"{frame.f_globals.get('__name__', '<unknown>')}:{frame.f_code.co_qualname}"


def _record(
    exc: Exception,
    frame: FrameType | None,
    log: logging.Logger,
    what: str,
    extra: Mapping[str, object] | None,
    level: int,
) -> None:
    """Log ``exc`` with its traceback, naming the function that caught it.

    :param exc: The exception being absorbed.
    :ptype exc: Exception
    :param frame: The catching frame.
    :ptype frame: FrameType | None
    :param log: The catching module's logger.
    :ptype log: logging.Logger
    :param what: The message, verbatim.
    :ptype what: str
    :param extra: Structured fields for the record's ``extra_data``.
    :ptype extra: Mapping[str, object] | None
    :param level: The record's level, WARNING or above.
    :ptype level: int
    :raises ValueError: ``level`` is below WARNING.
    """
    if level < logging.WARNING:
        raise ValueError(f"a contained failure is logged at WARNING or above, not {logging.getLevelName(level)}")
    data = {**(extra or {}), "contained_site": _site_of(frame), "exc_type": type(exc).__name__}
    log.log(level, what, exc_info=exc, extra={"extra_data": data})


def contain(
    exc: Exception,
    *,
    log: logging.Logger,
    what: str,
    extra: Mapping[str, object] | None = None,
    level: int = logging.WARNING,
) -> None:
    """Record ``exc`` as a failure this boundary absorbs on purpose.

    Call it from the ``except`` block that decided the failure stops here, in place of the log call it
    would otherwise make. It does not change control flow.

    :param exc: The caught exception.
    :ptype exc: Exception
    :param log: The catching module's logger.
    :ptype log: logging.Logger
    :param what: The message, verbatim.
    :ptype what: str
    :param extra: Structured fields for the record's ``extra_data``; ``contained_site`` and ``exc_type``
        are added to them.
    :ptype extra: Mapping[str, object] | None
    :param level: WARNING by default; nothing below it.
    :ptype level: int
    :raises ValueError: ``level`` is below WARNING.
    """
    if exc.__traceback__ is not None:
        frame: FrameType | None = exc.__traceback__.tb_frame
    else:
        # Handed over without having been raised: the caller of contain() is where it stopped.
        here = inspect.currentframe()
        frame = here.f_back if here is not None else None
    _record(exc, frame, log, what, extra, level)


class _Contained:
    """The context manager :func:`contained` returns."""

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
        # The traceback handed to __exit__ starts at the frame holding the `with` block.
        _record(
            exc,
            traceback.tb_frame if traceback is not None else None,
            self._log,
            self._what,
            self._extra,
            logging.WARNING,
        )
        return True


def contained(
    what: str,
    *,
    log: logging.Logger,
    extra: Mapping[str, object] | None = None,
) -> AbstractContextManager[None]:
    """Absorb and record an :class:`Exception` raised in the block; let everything else through.

    :param what: The message if the block fails, verbatim.
    :ptype what: str
    :param log: The calling module's logger.
    :ptype log: logging.Logger
    :param extra: Structured fields for the record's ``extra_data``.
    :ptype extra: Mapping[str, object] | None
    :return: A context manager that contains an :class:`Exception` and passes anything else.
    :rtype: AbstractContextManager[None]
    """
    return _Contained(what, log, extra)
