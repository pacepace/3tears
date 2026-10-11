"""The Vega-Lite chart renderer — an optional adapter over eval's chart intent.

Install it as ``3tears-evals[vega]``, which adds ``vl-convert-python`` for rasterising; compiling a spec
for a browser to draw needs nothing past the core. The core
(:mod:`threetears.evals.analysis.viz`) decides what every chart says as a
:class:`~threetears.evals.analysis.viz.ChartIntent` and never imports this package; this package reads
intents and adds a picture, and decides nothing a reader is told.

- :class:`VegaRenderer` is the :class:`~threetears.evals.analysis.viz.ChartRenderer`: built once for a
  host's style (:meth:`~VegaRenderer.for_style` — its declared palette and font, else the packaged ones), it draws an intent (:meth:`~VegaRenderer.draw`), gives the browser its config, rasterises
  (:meth:`~VegaRenderer.png`, :meth:`~VegaRenderer.svg`) and reads its drawing back for the core's
  conformance check (:func:`~threetears.evals.analysis.viz.renderer_disagreements`).
- :mod:`compiler` draws through one arm per chart type (:mod:`arms`); :mod:`spec_policy` is this
  renderer's own gate on the spec; :mod:`palette`, :mod:`text_metrics` and :mod:`render` are its
  theme, its measurements and its rasteriser. A host drawing in its own typeface measures it with
  ``python -m threetears.evals.vega.measure_font`` (:mod:`measure_font`, a command rather than an import)
  and declares it with :func:`load_chart_font`; :func:`packaged_font` is the default face, Liberation Sans. A
  host drawing in its own colours declares a palette read from an artifact of the packaged one's shape with
  :func:`load_palette`, which holds it to the slot widths the chart vocabulary assigns; :func:`packaged_palette`
  is the default. :func:`check_font_files` is a host's CI gate on a face it ships: whether the files its table was
  measured from are still those bytes. It hashes files and needs no rasteriser, and importing this package imports
  none (``vl-convert`` is imported only inside the calls that rasterise).

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``; ``tests/test_package_matrix.py`` holds that, and holds that nothing
outside this package imports it.
"""

from __future__ import annotations

from threetears.evals.vega.compiler import CompiledChart, CompiledColumn, compile_chart, draw_intent
from threetears.evals.vega.palette import PaletteError, Theme, load_palette, packaged_palette, vega_config
from threetears.evals.vega.render import RasterRefused, register_fonts, render_png, render_svg
from threetears.evals.vega.renderer import VegaRenderer
from threetears.evals.vega.spec_policy import SpecPolicyError, check_spec
from threetears.evals.vega.text_metrics import (
    TextMetricsError,
    check_font_files,
    load_chart_font,
    packaged_font,
    write_font_metrics,
)

__all__ = [
    "CompiledChart",
    "CompiledColumn",
    "PaletteError",
    "RasterRefused",
    "SpecPolicyError",
    "TextMetricsError",
    "Theme",
    "VegaRenderer",
    "check_font_files",
    "check_spec",
    "compile_chart",
    "draw_intent",
    "load_chart_font",
    "load_palette",
    "packaged_font",
    "packaged_palette",
    "register_fonts",
    "render_png",
    "render_svg",
    "vega_config",
    "write_font_metrics",
]
