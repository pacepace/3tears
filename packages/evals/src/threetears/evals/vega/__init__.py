"""The Vega-Lite chart renderer — an optional adapter over eval's chart intent.

Install it as ``3tears-evals[vega]``, which adds ``vl-convert-python`` for rasterising; compiling a spec
for a browser to draw needs nothing past the core. The core
(:mod:`threetears.evals.analysis.viz`) decides what every chart says as a
:class:`~threetears.evals.analysis.viz.ChartIntent` and never imports this package; this package reads
intents and adds a picture, and decides nothing a reader is told.

- :class:`VegaRenderer` is the :class:`~threetears.evals.analysis.viz.ChartRenderer`: built once with a
  theme, it draws an intent (:meth:`~VegaRenderer.draw`), gives the browser its config, rasterises
  (:meth:`~VegaRenderer.png`, :meth:`~VegaRenderer.svg`) and reads its drawing back for the core's
  conformance check (:func:`~threetears.evals.analysis.viz.renderer_disagreements`).
- :mod:`compiler` draws through one arm per chart type (:mod:`arms`); :mod:`spec_policy` is this
  renderer's own gate on the spec; :mod:`palette`, :mod:`text_metrics` and :mod:`render` are its
  theme, its measurements and its rasteriser.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``; ``tests/test_package_matrix.py`` holds that, and holds that nothing
outside this package imports it.
"""

from __future__ import annotations

from threetears.evals.vega.compiler import CompiledChart, CompiledColumn, compile_chart, draw_intent
from threetears.evals.vega.palette import Theme, vega_config
from threetears.evals.vega.render import register_fonts, render_png, render_svg
from threetears.evals.vega.renderer import VegaRenderer
from threetears.evals.vega.spec_policy import SpecPolicyError, check_spec
from threetears.evals.vega.text_metrics import TextMetricsError, write_font_metrics

__all__ = [
    "CompiledChart",
    "CompiledColumn",
    "SpecPolicyError",
    "TextMetricsError",
    "Theme",
    "VegaRenderer",
    "check_spec",
    "compile_chart",
    "draw_intent",
    "register_fonts",
    "render_png",
    "render_svg",
    "vega_config",
    "write_font_metrics",
]
