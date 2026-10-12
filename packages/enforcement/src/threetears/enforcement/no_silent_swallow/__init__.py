"""no-silent-swallow enforcement domain — silent ``except`` / ``suppress`` walker.

every exception handler in production code that turns an error into
an inert answer -- ``pass``, ``continue``, ``return None`` / a
constant / an empty collection, an assigned default -- must re-raise,
return a typed failure, or carry a ``# NOSILENT(<kind>): <reason>``
marker whose kind the walker checks against the code (``expected``,
``attempt``, ``cleanup``, ``optional-import``, ``logged``; see
:mod:`threetears.enforcement.no_silent_swallow.walkers`). logging
first does not excuse the inert answer: the log tells the operator,
the caller still gets the default. a bare ``# NOSILENT: <reason>``
comment is a violation anywhere. bare ``except:``
clauses (which catch ``SystemExit`` / ``KeyboardInterrupt`` and are
never correct in production) are flagged unconditionally.
``contextlib.suppress(...)`` follows the same contract -- every
suppress site needs a valid kinded marker within 2 lines above.

per-repo configuration goes through :class:`NoSilentSwallowConfig`;
:func:`run_no_silent_swallow_enforcement` is the pytest-friendly
entry point that orchestrates the walker, applies exemptions, emits
the report, and fails in strict mode.
"""

from threetears.enforcement.no_silent_swallow.config import (
    NoSilentSwallowConfig,
)
from threetears.enforcement.no_silent_swallow.runner import (
    run_no_silent_swallow_enforcement,
)
from threetears.enforcement.no_silent_swallow.walkers import (
    body_contains_log,
    body_reraises,
    body_silent_category,
    find_silent_swallows,
    has_nosilent_marker,
    suppress_has_nosilent,
)

__all__ = [
    "NoSilentSwallowConfig",
    "body_contains_log",
    "body_reraises",
    "body_silent_category",
    "find_silent_swallows",
    "has_nosilent_marker",
    "run_no_silent_swallow_enforcement",
    "suppress_has_nosilent",
]
