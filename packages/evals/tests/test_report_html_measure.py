"""The HTML report caps the line length of every prose block, and leaves tables and figures the page's width (#683).

The page is bounded at 920px for its tables; at a 16px body that is about a hundred characters a line, which
the headline, summary and caveats all ran at. Read from the stylesheet the page actually serves, rule by rule,
so a cap set on the wrong selector, or in a unit that does not follow the type, fails here.
"""

from __future__ import annotations

import re

from threetears.evals.analysis import report_html
from packages.evals.tests.report_support import minimal_report

#: The prose blocks the page lays out: the headline and a finding's title, paragraphs, list items, a chart's
#: caption, a disclosure and a caveat.
_PROSE = ("h1", "h3", "p", "li", "figcaption", ".disclosure", ".caveat")

#: A readable measure, in characters: wide enough for a sentence, narrow enough to track a line back.
_READABLE = range(45, 81)


def _rules(page: str) -> dict[str, dict[str, str]]:
    """Each selector the page's stylesheet names, with every declaration any rule gives it."""
    style = page.split("<style>", 1)[1].split("</style>", 1)[0]
    style = re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", style)
    rules: dict[str, dict[str, str]] = {}
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", style):
        declarations = dict(
            (name.strip(), value.strip())
            for name, value in (part.split(":", 1) for part in body.split(";") if ":" in part)
        )
        for selector in selectors.split(","):
            rules.setdefault(selector.strip(), {}).update(declarations)
    return rules


def _measure_ch(value: str) -> int | None:
    """A ``max-width`` given in ``ch``, as a number; None for any other unit."""
    match = re.fullmatch(r"(\d+)ch", value)
    return int(match.group(1)) if match else None


def test_every_prose_block_carries_a_readable_line_length_cap() -> None:
    rules = _rules(report_html(minimal_report()))

    for selector in _PROSE:
        cap = rules.get(selector, {}).get("max-width")
        assert cap is not None, f"{selector} carries no line-length cap"
        assert _measure_ch(cap) in _READABLE, f"{selector} is capped at {cap}, not a readable measure in characters"


def test_tables_and_figures_keep_the_pages_width() -> None:
    rules = _rules(report_html(minimal_report()))

    for selector in ("table", ".table-wrap", "figure", "th, td", "th", "td"):
        assert "max-width" not in rules.get(selector, {}), f"{selector} is narrowed like prose"
    assert rules["body"]["max-width"] == "920px", "the page itself stays as wide as before"


def test_the_stylesheet_reader_sees_a_cap_that_is_missing() -> None:
    page = report_html(minimal_report()).replace("max-width: 70ch;", "")
    assert "max-width" not in _rules(page).get("p", {})
