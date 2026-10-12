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
"""

from __future__ import annotations

import logging

__all__ = ["PACKAGE_LOGGER", "get_logger"]

#: The logger every module's logger sits under.
PACKAGE_LOGGER = "threetears.evals"

logging.getLogger(PACKAGE_LOGGER).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a module of this package.

    Args:
        name: The module's ``__name__``.

    Returns:
        The stdlib logger of that name.
    """
    return logging.getLogger(name)
