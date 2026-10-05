"""Finding visualizations — typed payloads compiled to Vega-Lite specs.

One compiler, two surfaces. A finding's ``viz`` carries a narrow typed payload;
this package turns it into a Vega-Lite spec that the browser renders with
``vega-embed`` and the server rasterises with ``vl-convert`` for MCP. A chart
that exists only in a browser is a reporting capability on one surface, which is
what the cross-surface parity rule exists to prevent.

The layering matters and is deliberate:

- :mod:`payloads` is the *generation-time* contract — a malformed payload is
  rejected while the generation can still be retried, not discovered when a
  reader opens the report. Generation is billed, so a defect that surfaces at
  render costs a regeneration to correct.
- :mod:`compiler` turns a validated payload into a spec that carries **no
  colour**; the palette arrives as a Vega-Lite ``config`` from whichever
  renderer is drawing.
- :mod:`policy` gates the compiler's OUTPUT rather than living inside it. The
  presentation rules a report is held to are properties of a *spec*, and the
  compiler is not the only thing that will ever produce one — so the gate sits
  where any producer's spec must pass through it.
- :mod:`palette` and :mod:`render` are the server-side half: resolved sRGB hex
  and the vl-convert call.

**This module is the package's public root.** A host imports from here and from no module below
it, and only the names in ``__all__``; ``tests/test_package_matrix.py`` holds that. A ``# debt:``
comment on an export names what retires it. Code inside the package imports its own modules directly.
"""

from __future__ import annotations

from threetears.evals.analysis.viz.compiler import ChartColumn, CompiledChart, compile_chart
from threetears.evals.analysis.viz.models import FindingChart
from threetears.evals.analysis.viz.palette import Theme, vega_config
from threetears.evals.analysis.viz.payloads import PayloadError
from threetears.evals.analysis.viz.policy import SpecPolicyError
from threetears.evals.analysis.viz.render import render_png
from threetears.evals.analysis.viz.text_metrics import TextMetricsError, write_font_metrics

__all__ = [
    "ChartColumn",
    "CompiledChart",
    "FindingChart",
    "PayloadError",
    "SpecPolicyError",
    "TextMetricsError",
    "Theme",
    "compile_chart",
    "render_png",
    "vega_config",
    "write_font_metrics",
]
