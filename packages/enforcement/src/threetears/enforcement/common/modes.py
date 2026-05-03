"""enforcement-mode resolver shared across every domain.

every domain offers two modes: ``strict`` (default — fail the test on
any violation) and ``report`` (print but never fail; an opt-in
debugging aid for bulk refactors). this module standardises how each
domain reads its mode env var and validates the value.
"""

from __future__ import annotations

import os

__all__ = [
    "MODE_REPORT",
    "MODE_STRICT",
    "ModeError",
    "resolve_mode",
]


class ModeError(Exception):
    """raised when an enforcement-mode env var holds an unknown value."""


MODE_REPORT = "report"
MODE_STRICT = "strict"

_VALID_MODES: frozenset[str] = frozenset({MODE_REPORT, MODE_STRICT})


def resolve_mode(env_var: str, default: str = MODE_STRICT) -> str:
    """read ``env_var`` from the environment, validate, and normalise.

    the value is lowercased and stripped before comparison, so
    ``"Strict"`` / ``"  STRICT  "`` / ``"strict"`` are all accepted as
    :data:`MODE_STRICT`. an unset variable defaults to ``default`` (in
    turn validated against the same set).

    :param env_var: environment variable name to read
        (e.g. ``CACHE_ENFORCEMENT_MODE``)
    :ptype env_var: str
    :param default: fallback value when the env var is unset
    :ptype default: str
    :return: normalised mode (one of :data:`MODE_REPORT` /
        :data:`MODE_STRICT`)
    :rtype: str
    :raises ModeError: ``env_var`` (or ``default``) holds a value
        outside the valid set; the env var name is included in the
        message so the caller can identify the bad source
    """
    raw = os.environ.get(env_var)
    if raw is None:
        candidate = default
    else:
        candidate = raw
    normalised = candidate.strip().lower()
    if normalised not in _VALID_MODES:
        raise ModeError(f"{env_var}: must be one of {sorted(_VALID_MODES)}, got {raw!r} (default={default!r})")
    return normalised
