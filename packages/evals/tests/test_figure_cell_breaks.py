"""A cell of figures breaks between its figure and its bracket, and nowhere else.

A column of figures holds an arm's level with its interval beside it (``4.5 (3.8 to 5.0)``, ``$1.90 ($0.97 to
$2.82)``). Held on one line whole, such a column is as wide as its longest figure and its interval together, and a
table that fits its host's frame with the interval set under its figure scrolls sideways instead. Free to wrap anywhere,
a narrow column reads ``4.5 (3.8`` above ``to 5.0)``. So the cell's markup leaves exactly one place to break: the space
between the figure (with its unit, ``2.42 s``) and the bracket after it, the bracket's parentheses inside the span that
holds it, so no ``(`` is left at a line's end. A lone figure leaves none.

A break opportunity here is a space in the cell's text outside every span the page's stylesheet holds on one line
(``white-space: nowrap``); the cell itself is held by no such rule. The live page and the static page print a table's
cells with the same markup and the same stylesheet, so both are read.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from packages.evals.tests.report_kit_support import report_of
from threetears.evals.analysis import report_html
from threetears.evals.analysis.report.model import TableBlock, TableCell, TableColumn

#: Each figure cell's text, by the arm column it sits in.
_FIGURES = {
    "level": "4.5 (3.8 to 5.0)",
    "money": "$1.90 ($0.97 to $2.82)",
    "spaced": "2.42 s (2.35 to 2.49)",
    "counted": "95% (38 of 40 trials, 20 cases)",
    "lone": "420ms",
    "lone_spaced": "2.42 s",
}


def _page(*, static: bool) -> str:
    """A page holding one table: a reading, then one column of figures per form in :data:`_FIGURES`."""
    columns = [
        TableColumn(key="reading", header="Reading", term="reading"),
        *(TableColumn(key=key, header=key, term="arm mean") for key in _FIGURES),
    ]
    row = {"reading": TableCell(value="Score"), **{key: TableCell(value=text) for key, text in _FIGURES.items()}}
    table = TableBlock(
        section="contrasts", name="comparisons", title="Against base", columns=columns, rows=[row], order="as built",
        total_rows=1,
    )  # fmt: skip
    return report_html(report_of([table]), static=static)


def _rules(page: str) -> list[tuple[list[str], str]]:
    """Every rule of the page's stylesheets outside an at-rule's block, as its selectors and its declarations."""
    css = re.sub(
        r"/\*.*?\*/", "", "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", page, flags=re.DOTALL)), flags=re.DOTALL
    )
    css = re.sub(r"@[^{};]+\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
    return [
        ([selector.strip() for selector in selectors.split(",")], body)
        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)
    ]


def _held_span_classes(page: str) -> set[str]:
    """The classes of a span inside a table cell that the page's stylesheet holds on one line (``td .iv``)."""
    return {
        found[1]
        for selectors, body in _rules(page)
        if re.search(r"white-space:\s*nowrap", body)
        for selector in selectors
        if (found := re.fullmatch(r"td \.([\w-]+)", selector)) is not None
    }


class _Cell(HTMLParser):
    """One cell's markup read as runs of text, each with the classes of the spans it sits inside."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[set[str]] = []
        self.runs: list[tuple[str, frozenset[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = {name for key, value in attrs if key == "class" and value for name in value.split()}
        self.stack.append(classes if tag == "span" else set())

    def handle_endtag(self, tag: str) -> None:
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        self.runs.append((data, frozenset().union(*self.stack)))


def _cells(page: str) -> dict[str, str]:
    """Each figure cell's markup, by its column's key, in the order :data:`_FIGURES` holds them."""
    (body,) = re.findall(r"<tbody>(.*?)</tbody>", page, flags=re.DOTALL)
    found = re.findall(r'<td class="figs">(.*?)</td>', body, flags=re.DOTALL)
    assert len(found) == len(_FIGURES), found
    return dict(zip(_FIGURES, found, strict=True))


def _breaks(markup: str, held: set[str]) -> list[tuple[str, str]]:
    """Each break opportunity in a cell's markup, as the text before it and the text after it."""
    cell = _Cell()
    cell.feed(markup)
    text = "".join(run for run, _ in cell.runs)
    at, places = 0, []
    for run, classes in cell.runs:
        if not classes & held:
            places += [at + offset for offset, char in enumerate(run) if char.isspace()]
        at += len(run)
    return [(text[:place], text[place + 1 :]) for place in places]


def _held_text(markup: str, span: str) -> list[str]:
    """The text of each span of class ``span`` in a cell's markup."""
    cell = _Cell()
    cell.feed(markup)
    return [run for run, classes in cell.runs if span in classes]


@pytest.mark.parametrize("static", [False, True], ids=["live", "static"])
def test_a_figure_cell_breaks_between_its_figure_and_its_bracket_and_nowhere_else(static: bool) -> None:
    page = _page(static=static)
    held = _held_span_classes(page)
    assert {"pt", "iv", "aside"} <= held, held
    breaks = {key: _breaks(markup, held) for key, markup in _cells(page).items()}
    assert breaks == {
        "level": [("4.5", "(3.8 to 5.0)")],
        "money": [("$1.90", "($0.97 to $2.82)")],
        "spaced": [("2.42 s", "(2.35 to 2.49)")],
        "counted": [("95%", "(38 of 40 trials, 20 cases)")],
        "lone": [],
        "lone_spaced": [],
    }


@pytest.mark.parametrize("static", [False, True], ids=["live", "static"])
def test_an_intervals_parentheses_travel_inside_the_span_that_holds_it(static: bool) -> None:
    cells = _cells(_page(static=static))
    assert _held_text(cells["level"], "iv") == ["(3.8 to 5.0)"]
    assert _held_text(cells["money"], "iv") == ["($0.97 to $2.82)"]
    assert _held_text(cells["counted"], "aside") == ["(38 of 40 trials, 20 cases)"]
    assert _held_text(cells["spaced"], "pt") == ["2.42 s"], "the figure's unit stays with it"
    assert cells["level"] == '<span class="pt">4.5</span> <span class="iv">(3.8 to 5.0)</span>'


def test_no_rule_holds_a_whole_figure_cell_on_one_line() -> None:
    held = [
        selector
        for selectors, body in _rules(_page(static=False))
        if re.search(r"white-space:\s*nowrap", body)
        for selector in selectors
        if re.search(r"\.figs(?![\w-])", selector) and not re.search(r"\.figs\s", selector)
    ]
    assert held == [], (
        f"a rule holds a figure cell whole, so its column never sets its interval under its figure: {held}"
    )
