"""The static page's contract, read by parsing it: what every test of a page written with ``static=True`` holds.

A host whose policy lets a page neither run nor load anything (``default-src 'none'``) asks for the static page. Every
writer of one (a report's page, a campaign's, a page built from a report plan) holds it to the same contract:

- it carries no ``<script>`` and no ``<link>``, not even the package's own, no ``on*`` attribute, no ``javascript:``
  value in any attribute (``data-*`` included: it has no chart data either), and no ``url(`` or ``@import`` in its
  styles (:func:`assert_static`);
- every chart is its figure's title, its picture and its table of values, and no rule the page ships hides that
  table: the one that does needs a drawn ``div.chart`` beside it, and the static page writes none, so no empty box
  stands either. The picture is inline SVG in a ``div.chart-drawn``, drawn on the server, or, for a kind not drawn
  there, one line saying the chart is shown as its table; never neither;
- less its drawings and those lines (:func:`less_drawings`), it is the default page less exactly the kit's script,
  the font links and each ``div.chart`` (:func:`less_what_static_leaves_out`), so the default page is the one it
  always was.
"""

from __future__ import annotations

import re

from threetears.evals.analysis.report.chart_svg import AS_TABLE_NOTE, CHART_STYLE
from packages.evals.tests.page_contract_support import FONT_LINK_MARKUP, KIT_JS, parsed, what_runs

__all__ = ["assert_static", "less_drawings", "less_what_static_leaves_out"]

#: The selectors, of every rule the page ships, that reach a chart's values table: its caption, which repeats the
#: figure's own title for a screen reader, and the rule hiding the table beside a drawn chart.
_TABLE_RULES = {".chart-values caption", ".chart:not(:empty) + .chart-values"}

#: A chart the kit draws, as the default page writes it: every attribute double-quoted, its JSON escaped in them.
_CHART_DIV = re.compile(r'<div class="chart" id="[^"]*" data-kind="[^"]*" data-chart="[^"]*"></div>')

#: Where a static page's drawing of a chart starts, and the line a chart not drawn there carries instead.
_DRAWING_START = '<div class="chart-drawn" data-kind="'
_AS_TABLE_LINE = f'<p class="chart-note">{AS_TABLE_NOTE}</p>'

#: A tag in a drawing that opens or closes a ``div``, for finding where the drawing ends.
_DIV_TAG = re.compile(r"<(/?)div\b")


def _classes(attrs: list[tuple[str, str | None]]) -> list[str]:
    return (dict(attrs).get("class") or "").split()


def _selectors(styles: list[str]) -> set[str]:
    """Every selector of every rule in ``styles`` that names ``chart-values``, its whitespace collapsed."""
    css = re.sub(r"/\*.*?\*/", "", "".join(styles), flags=re.DOTALL)
    found = {" ".join(one.split()) for head in re.findall(r"([^{}]+)\{", css) for one in head.split(",")}
    return {selector for selector in found if "chart-values" in selector}


def assert_static(page: str) -> None:
    """``page`` runs, embeds and loads nothing, and shows every chart as its table."""
    read = parsed(page)
    tags = [tag for tag, _ in read.tags]
    assert what_runs(page) == []
    assert "script" not in tags and "link" not in tags, "not even the kit's script or the package's font links"
    assert "<script" not in page and "<link" not in page, "an author's markup is escaped text"
    attributes = [(name, value or "") for _, attrs in read.tags for name, value in attrs]
    assert not [name for name, _ in attributes if name.startswith("on")]
    assert not [name for name, value in attributes if "javascript:" in "".join(value.split()).lower()]
    assert not [name for name, _ in attributes if name == "data-chart"], "no chart data, so nothing for a script"
    for style in read.styles:
        compact = "".join(style.split()).lower()
        assert "url(" not in compact and "@import" not in compact

    figures = [at for at, (tag, attrs) in enumerate(read.tags) if tag == "figure" and "fig" in _classes(attrs)]
    assert figures, "the page must draw charts for this to test anything"
    assert not [tag for tag, attrs in read.tags if "chart" in _classes(attrs)], "no chart box, drawn or empty"
    tables = [at for at, (tag, attrs) in enumerate(read.tags) if "chart-values" in _classes(attrs)]
    assert len(tables) == len(figures), "every figure holds its table of values"
    assert _selectors(read.styles) == _TABLE_RULES, "a rule that hides the table without a drawn chart beside it"
    for start, table in zip(figures, tables, strict=True):
        assert start < table, "a figure's table is its own"
        inside = read.tags[start:table]
        drawn = [at for at, (_, attrs) in enumerate(inside) if "chart-drawn" in _classes(attrs)]
        noted = [at for at, (_, attrs) in enumerate(inside) if "chart-note" in _classes(attrs)]
        assert len(drawn) + len(noted) == 1, "a figure is drawn above its table, or says it is shown as its table"
        if drawn:
            svgs = [attrs for tag, attrs in inside[drawn[0] :] if tag == "svg" and "viz-chart" in _classes(attrs)]
            assert svgs, "a drawing draws"
            assert all(dict(attrs).get("viewbox") and dict(attrs).get("role") == "img" for attrs in svgs)
    assert page.count(_AS_TABLE_LINE) == len([1 for _, attrs in read.tags if "chart-note" in _classes(attrs)])


def less_what_static_leaves_out(page: str) -> str:
    """The default ``page`` less the kit's script, the font links and each chart's ``div.chart``, each asserted
    present first, so the comparison is about something."""
    assert f"\n<script>{KIT_JS}</script>" in page and FONT_LINK_MARKUP + "\n" in page and _CHART_DIV.search(page)
    page = page.replace(f"\n<script>{KIT_JS}</script>", "", 1).replace(FONT_LINK_MARKUP + "\n", "", 1)
    return _CHART_DIV.sub("", page)


def less_drawings(page: str) -> str:
    """The static ``page`` less each chart's drawing, each line saying a chart is shown as its table and the
    drawings' stylesheet: what it shares with the default page."""
    assert page.count(CHART_STYLE) == 1, "the drawings' stylesheet, once"
    out, at = [], 0
    while (start := page.find(_DRAWING_START, at)) != -1:
        out.append(page[at:start])
        depth, end = 0, start
        for found in _DIV_TAG.finditer(page, start):
            depth += -1 if found.group(1) else 1
            if depth == 0:
                end = page.index(">", found.end()) + 1
                break
        assert depth == 0, "a drawing closes"
        at = end
    out.append(page[at:])
    return "".join(out).replace(_AS_TABLE_LINE, "").replace(CHART_STYLE, "", 1)
