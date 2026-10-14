"""Finding charts — eval's own chart intent.

**The engine decides what a chart SAYS.** This package ships no charting library:

- :mod:`payloads` is the *generation-time* contract — each chart type's typed data, rejected while a
  generation can still be retried rather than discovered when a reader opens the report. :mod:`intent`
  turns a validated payload into a :class:`ChartIntent` — the order, the units, the values as drawn,
  what each field encodes, what must be said beside it — through one builder per type (:mod:`intents`),
  and :mod:`policy` holds every intent to the presentation rules. :mod:`quantities` is how a quantity
  is stated.

**This module is the package's public root.** A host imports from here and from no module below
it, and only the names in ``__all__``; ``tests/test_package_matrix.py`` holds that. Code inside the
package imports its own modules directly.
"""

from __future__ import annotations

from threetears.evals.analysis.viz.intent import (
    INTENT_VERSION,
    ChartAxis,
    ChartColours,
    ChartColumn,
    ChartEncoding,
    ChartIdentity,
    ChartIntent,
    ChartReference,
    ChartType,
    ChartValue,
    EncodingRole,
    chart_intent,
)
from threetears.evals.analysis.viz.payloads import PayloadError
from threetears.evals.analysis.viz.policy import IntentPolicyError, check_intent, table_disagreements

__all__ = [
    "INTENT_VERSION",
    "ChartValue",
    "ChartAxis",
    "ChartColours",
    "ChartColumn",
    "ChartEncoding",
    "ChartIdentity",
    "ChartIntent",
    "ChartReference",
    "ChartType",
    "EncodingRole",
    "IntentPolicyError",
    "PayloadError",
    "chart_intent",
    "check_intent",
    "table_disagreements",
]
