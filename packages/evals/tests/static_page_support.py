"""The static page's contract, read by parsing it: what every test of a page written with ``static=True`` holds.

A host whose policy lets a page neither run nor load anything (``default-src 'none'``) asks for the static page. Every
writer of one (a report's page, a campaign's, a page built from a report plan) holds it to the same contract:

- it carries no ``<script>`` and no ``<link>``, not even the package's own, no ``on*`` attribute, no ``javascript:``
  value in any attribute (``data-*`` included: it has no chart data either), and no ``url(`` or ``@import`` in its
  styles (:func:`assert_static`);
- every chart is its figure's title and its table of values, and no rule the page ships hides that table: the one
  that does needs a drawn ``div.chart`` beside it, and the static page writes none, so no empty box stands either;
- it is the default page less exactly the kit's script, the font links and each ``div.chart``
  (:func:`less_what_static_leaves_out`), so the default page is the one it always was.
"""

from __future__ import annotations

import re

from packages.evals.tests.page_contract_support import FONT_LINK_MARKUP, KIT_JS, parsed, what_runs

__all__ = ["assert_static", "less_what_static_leaves_out"]

#: The selectors, of every rule the page ships, that reach a chart's values table: its caption, which repeats the
#: figure's own title for a screen reader, and the rule hiding the table beside a drawn chart.
_TABLE_RULES = {".chart-values caption", ".chart:not(:empty) + .chart-values"}

#: A chart the kit draws, as the default page writes it: every attribute double-quoted, its JSON escaped in them.
_CHART_DIV = re.compile(r'<div class="chart" id="[^"]*" data-kind="[^"]*" data-chart="[^"]*"></div>')


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


def less_what_static_leaves_out(page: str) -> str:
    """The default ``page`` less the kit's script, the font links and each chart's ``div.chart``, each asserted
    present first, so the comparison is about something."""
    assert f"\n<script>{KIT_JS}</script>" in page and FONT_LINK_MARKUP + "\n" in page and _CHART_DIV.search(page)
    page = page.replace(f"\n<script>{KIT_JS}</script>", "", 1).replace(FONT_LINK_MARKUP + "\n", "", 1)
    return _CHART_DIV.sub("", page)
