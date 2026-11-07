"""The one place that decides whether a missing module means the ``core`` extra is absent.

``dpop``, ``stores.attempt_limiter`` and TOTP seed sealing build on ``threetears.core``, which
the base install leaves out: a consumer doing sign-in and sessions should not install the
Postgres and NATS stack core brings. Each of them turns core's absence into an
:class:`ImportError` that names ``3tears-iam[core]``, so the fix is in the message.

Only core's own absence is relabelled. A module missing from *inside* an installed core --
``sqlalchemy`` gone, say -- is a broken environment, not a missing extra, and telling that
operator to install an extra they already have would send them the wrong way. That error is
re-raised as it came.
"""

from __future__ import annotations

from typing import Final

__all__ = ["CORE_EXTRA", "core_extra_missing"]

#: what a consumer adds to get the modules that build on core
CORE_EXTRA: Final[str] = "3tears-iam[core]"

_CORE_ROOT: Final[str] = "threetears.core"


def core_extra_missing(exc: ModuleNotFoundError, *, needed_by: str) -> ImportError:
    """The error naming the ``core`` extra, when ``exc`` is core's own absence.

    Raise the result ``from exc``. Any other missing module is re-raised unchanged.

    :param exc: the failed import of a ``threetears.core`` name
    :ptype exc: ModuleNotFoundError
    :param needed_by: what needs core, as the message should name it
    :ptype needed_by: str
    :return: the error to raise
    :rtype: ImportError
    :raises ModuleNotFoundError: ``exc`` itself, when the missing module is not core's
    """
    missing = exc.name or ""
    if missing != _CORE_ROOT and not missing.startswith(f"{_CORE_ROOT}."):
        raise exc
    return ImportError(
        f"{needed_by} builds on threetears.core, which the core extra installs: pip install '{CORE_EXTRA}'"
    )
