"""The Vega-Lite renderer, as a :class:`~threetears.evals.analysis.viz.ChartRenderer` with a palette.

**The palette is the host's, bound once.** A host builds one :class:`VegaRenderer` for its style
(:meth:`VegaRenderer.for_style`) — its declared :class:`~threetears.evals.contracts.host.ChartPalette`,
or, when it declares none, the packaged palette in the variant it names — with the font directory its
brand face lives in, and hands it every intent: :meth:`VegaRenderer.draw` returns the colourless spec a
browser embeds with :meth:`VegaRenderer.config`, and :meth:`VegaRenderer.png` / :meth:`VegaRenderer.svg`
rasterise it for a surface that cannot run a browser — which is the one place ``vl_convert`` is needed.

**It reads its own drawing back** (:meth:`VegaRenderer.drawn_data`) for the core's conformance check:
every inline dataset of the spec, at any depth, with each datum's drawn name resolved back to the
identity it stands for. The arms reshape what they draw — a sweep's levels melted to one record per
lever, a distribution's interval as a span and two caps — so the datum is read as drawn and compared by
value, not by field name.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from threetears.evals.analysis.viz.intent import Cell, ChartIntent
from threetears.evals.analysis.viz.quantities import strip_common_prefix
from threetears.evals.contracts.host import ChartPalette, StyleProfile
from threetears.evals.vega.compiler import DISPLAY_FIELD, CompiledChart, draw_intent
from threetears.evals.vega.palette import Theme, packaged_palette, vega_config
from threetears.evals.vega.render import DEFAULT_SCALE, render_png, render_svg


@dataclass(frozen=True)
class VegaRenderer:
    """Draw chart intents as Vega-Lite, in one palette.

    Build it with :meth:`for_style` (a host's renderer) or :meth:`packaged` (the packaged palette, by
    variant); the constructor takes the palette itself.

    Attributes:
        palette: The colours every drawing is configured and rasterised in.
        font_dir: The directory of font files a raster registers before drawing; ``None`` draws in
            whatever the host's font database resolves (warned about once per process).
    """

    palette: ChartPalette
    font_dir: Path | None = None

    @classmethod
    def for_style(cls, style: StyleProfile, *, theme: Theme = "dark", font_dir: Path | None = None) -> VegaRenderer:
        """The renderer a host draws with: its declared palette, or the packaged one when it declares none.

        **The rule.** A style that declares ``chart_palette`` is drawn in exactly that palette, and
        ``theme`` is not read. A style that declares none (``chart_palette is None``) is drawn in the
        packaged palette's ``theme`` variant. That is a default the host chose by not declaring a palette —
        not a substitute for one it declared, which is never replaced.

        Args:
            style: The host's style profile (``HostProfile.style``).
            theme: Which packaged variant to draw in when the style declares no palette.
            font_dir: The font directory, as on the constructor.

        Returns:
            The renderer.
        """
        palette = style.chart_palette if style.chart_palette is not None else packaged_palette(theme)
        return cls(palette=palette, font_dir=font_dir)

    @classmethod
    def packaged(cls, theme: Theme = "dark", *, font_dir: Path | None = None) -> VegaRenderer:
        """A renderer in the packaged palette's ``theme`` variant.

        Args:
            theme: Which variant.
            font_dir: The font directory, as on the constructor.

        Returns:
            The renderer.
        """
        return cls(palette=packaged_palette(theme), font_dir=font_dir)

    def draw(self, intent: ChartIntent) -> CompiledChart:
        """Draw ``intent`` as a colourless Vega-Lite spec, held to this renderer's spec gate.

        Args:
            intent: A decided chart intent.

        Returns:
            The compiled chart: the spec, with the intent's table, caption and disclosures beside it.

        Raises:
            SpecPolicyError: The drawn spec breaks a rendering rule.
        """
        return draw_intent(intent)

    def config(self) -> dict[str, Any]:
        """The Vega-Lite ``config`` that colours and sets every spec this renderer draws.

        Returns:
            The config, for a browser embedding a spec to pass beside it.
        """
        return vega_config(self.palette)

    def png(self, chart: CompiledChart, *, scale: int = DEFAULT_SCALE) -> bytes:
        """Rasterise a drawn chart to PNG in this renderer's palette. Needs ``vl_convert`` (the ``[vega]`` extra).

        Args:
            chart: A chart :meth:`draw` returned.
            scale: Device-pixel multiplier.

        Returns:
            PNG bytes.
        """
        return render_png(chart.spec, palette=self.palette, scale=scale, font_dir=self.font_dir)

    def svg(self, chart: CompiledChart) -> str:
        """Render a drawn chart to SVG in this renderer's palette. Needs ``vl_convert`` (the ``[vega]`` extra).

        Args:
            chart: A chart :meth:`draw` returned.

        Returns:
            The SVG document.
        """
        return render_svg(chart.spec, palette=self.palette, font_dir=self.font_dir)

    def drawn_data(self, drawing: CompiledChart) -> list[dict[str, Cell]]:
        """Every inline datum the spec draws, each drawn name resolved to the identity it stands for.

        A datum on the identity axis may carry only the name drawn beside its mark. The arms draw either
        the identity itself or — on a categorical axis — the identity with the ``/``-prefix every series
        shares stripped (:func:`~threetears.evals.analysis.viz.quantities.strip_common_prefix`), and one
        drawing never mixes the two. So the drawn names are matched against both spellings of the
        intent's stated order, and resolved through the one spelling that holds all of them: the
        stripped form is strictly shorter, so no set of drawn names is both, except where stripping
        changed nothing and the two spellings are one.

        Args:
            drawing: A chart :meth:`draw` returned.

        Returns:
            The data, in the order the spec holds it.

        Raises:
            ValueError: The drawn names are not all one spelling of the intent's identities, so the
                drawing cannot say which row a datum belongs to.
        """
        data = list(_inline_data(drawing.spec))
        identity = drawing.intent.identity
        if identity is None:
            return data
        drawn = {datum[DISPLAY_FIELD] for datum in data if identity.field not in datum and DISPLAY_FIELD in datum}
        if not drawn:
            return data
        spellings = [
            {name: name for name in identity.order},
            {short: full for full, short in strip_common_prefix(identity.order).items()},
        ]
        names = next((spelling for spelling in spellings if drawn <= spelling.keys()), None)
        if names is None:
            raise ValueError(
                f"drawn names {sorted(map(str, drawn))} are not one spelling of the intent's identities {identity.order}"
            )
        return [
            datum | {identity.field: names[str(datum[DISPLAY_FIELD])]}
            if identity.field not in datum and DISPLAY_FIELD in datum
            else datum
            for datum in data
        ]


def _inline_data(node: Any) -> Iterator[dict[str, Cell]]:
    """Every datum of every inline ``data.values`` list under ``node``, depth first."""
    if isinstance(node, dict):
        source = node.get("data")
        if isinstance(source, dict) and isinstance(source.get("values"), list):
            yield from (dict(datum) for datum in source["values"] if isinstance(datum, dict))
        for child in node.values():
            yield from _inline_data(child)
    elif isinstance(node, list):
        for child in node:
            yield from _inline_data(child)


__all__ = [
    "VegaRenderer",
]
