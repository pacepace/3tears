"""Finding charts — eval's own chart intent, and the seam a renderer sits behind.

**The engine decides what a chart SAYS; a renderer decides how it LOOKS.** This package is the first
half and ships no charting library:

- :mod:`payloads` is the *generation-time* contract — each chart type's typed data, rejected while a
  generation can still be retried rather than discovered when a reader opens the report. :mod:`intent`
  turns a validated payload into a :class:`ChartIntent` — the order, the units, the values as drawn,
  what each field encodes, what must be said beside it — through one builder per type (:mod:`intents`),
  and :mod:`policy` holds every intent to the presentation rules. :mod:`quantities` is how a quantity
  is stated.
- :mod:`renderer` is the seam: :class:`ChartRenderer` is what a renderer implements, and
  :func:`renderer_disagreements` the one conformance check every renderer passes — what it draws
  agrees with the intent's values.

The package's own Vega-Lite renderer is an optional adapter, ``threetears.evals.vega`` (install
``3tears-evals[vega]``); nothing in the core imports it.

**This module is the package's public root.** A host imports from here and from no module below
it, and only the names in ``__all__``; ``tests/test_package_matrix.py`` holds that. Code inside the
package imports its own modules directly.
"""

from __future__ import annotations

from threetears.evals.analysis.viz.intent import (
    INTENT_VERSION,
    Cell,
    ChartAxis,
    ChartColours,
    ChartColumn,
    ChartEncoding,
    ChartIdentity,
    ChartIntent,
    ChartReference,
    ChartType,
    EncodingRole,
    chart_intent,
)
from threetears.evals.analysis.viz.payloads import SERIES_SLOTS, VALIDATED_SLOTS, PayloadError
from threetears.evals.analysis.viz.policy import IntentPolicyError, check_intent
from threetears.evals.analysis.viz.renderer import ChartRenderer, assert_renderer_conforms, renderer_disagreements

__all__ = [
    "INTENT_VERSION",
    "SERIES_SLOTS",
    "VALIDATED_SLOTS",
    "Cell",
    "ChartAxis",
    "ChartColours",
    "ChartColumn",
    "ChartEncoding",
    "ChartIdentity",
    "ChartIntent",
    "ChartReference",
    "ChartRenderer",
    "ChartType",
    "EncodingRole",
    "IntentPolicyError",
    "PayloadError",
    "assert_renderer_conforms",
    "chart_intent",
    "check_intent",
    "renderer_disagreements",
]
