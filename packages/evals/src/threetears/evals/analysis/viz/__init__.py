"""Finding charts — eval's own chart intent, and a Vega-Lite renderer for it.

**Two halves, and the seam between them is the point.** The engine decides what a chart SAYS; a
renderer decides how it LOOKS:

- **The intent (the core).** :mod:`payloads` is the *generation-time* contract — each chart type's
  typed data, rejected while a generation can still be retried rather than discovered when a reader
  opens the report. :mod:`intent` turns a validated payload into a :class:`ChartIntent` — the order,
  the units, the values as drawn, what each field encodes, what must be said beside it — through one
  builder per type (:mod:`intents`), and :mod:`policy` holds every intent to the presentation rules.
  :mod:`quantities` is how a quantity is stated. None of it knows a charting library exists.
- **The Vega-Lite renderer.** :mod:`compiler` draws an intent as a Vega-Lite spec through one arm per
  type (:mod:`arms`), gated by its own spec rules (:mod:`vega_policy`); :mod:`palette`,
  :mod:`text_metrics` and :mod:`render` are its theme, its measurements and its rasteriser. It reads
  intents and adds a picture; it decides nothing a reader is told. This half moves out of the core into
  an optional adapter, which is why nothing in the first half imports it.

**This module is the package's public root.** A host imports from here and from no module below
it, and only the names in ``__all__``; ``tests/test_package_matrix.py`` holds that. A ``# debt:``
comment on an export names what retires it. Code inside the package imports its own modules directly.
"""

from __future__ import annotations

from threetears.evals.analysis.viz.compiler import CompiledChart, CompiledColumn, compile_chart, draw_intent
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
from threetears.evals.analysis.viz.palette import Theme, vega_config
from threetears.evals.analysis.viz.payloads import SERIES_SLOTS, VALIDATED_SLOTS, PayloadError
from threetears.evals.analysis.viz.policy import IntentPolicyError, check_intent
from threetears.evals.analysis.viz.render import render_png
from threetears.evals.analysis.viz.text_metrics import TextMetricsError, write_font_metrics
from threetears.evals.analysis.viz.vega_policy import SpecPolicyError

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
    "ChartType",
    "CompiledColumn",  # debt: the Vega-Lite renderer; leaves the core for its adapter (phase D chunk 24)
    "CompiledChart",  # debt: the Vega-Lite renderer; leaves the core for its adapter (phase D chunk 24)
    "EncodingRole",
    "IntentPolicyError",
    "PayloadError",
    "SpecPolicyError",  # debt: the Vega-Lite renderer; leaves the core for its adapter (phase D chunk 24)
    "TextMetricsError",  # debt: the Vega-Lite renderer; leaves the core for its adapter (phase D chunk 24)
    "Theme",  # debt: the Vega-Lite renderer; leaves the core for its adapter (phase D chunk 24)
    "chart_intent",
    "check_intent",
    "compile_chart",  # debt: the Vega-Lite renderer; leaves the core for its adapter (phase D chunk 24)
    "draw_intent",  # debt: the Vega-Lite renderer; leaves the core for its adapter (phase D chunk 24)
    "render_png",  # debt: the Vega-Lite renderer; leaves the core for its adapter (phase D chunk 24)
    "vega_config",  # debt: the Vega-Lite renderer; leaves the core for its adapter (phase D chunk 24)
    "write_font_metrics",  # debt: the Vega-Lite renderer; leaves the core for its adapter (phase D chunk 24)
]
