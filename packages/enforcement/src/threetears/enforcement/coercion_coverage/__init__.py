"""coercion-coverage enforcement domain — single ``run`` override walker.

``TearsTool`` implements ``run`` as the dispatch entry point: it calls
``normalize_kwargs(...)`` to coerce inputs, then forwards to
``execute(...)`` which subclasses are expected to override. a subclass
that overrides ``run`` instead of ``execute`` silently bypasses the
coercion step and re-introduces the empty-string / JSON-encoded-string
422 bug class. this domain detects that override pattern statically.

per-repo configuration goes through :class:`CoerceCoverageConfig`;
:func:`run_coercion_enforcement` is the pytest-friendly entry point
that orchestrates the walker, applies exemptions, emits the report,
and fails in strict mode.
"""

from threetears.enforcement.coercion_coverage.config import (
    CoerceCoverageConfig,
)
from threetears.enforcement.coercion_coverage.runner import (
    run_coercion_enforcement,
)
from threetears.enforcement.coercion_coverage.walkers import (
    base_looks_toolish,
    find_run_overrides,
)

__all__ = [
    "CoerceCoverageConfig",
    "base_looks_toolish",
    "find_run_overrides",
    "run_coercion_enforcement",
]
