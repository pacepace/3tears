"""The HTML report caps the line length of every prose block, and leaves tables and figures the page's width (#683).

The page is bounded by the kit's wide measure (``.page`` at ``--wide``, 68rem) for its tables and charts; at the
kit's 15px body that is well over a hundred characters a line, which prose once ran at. Read from the stylesheet the
page actually serves, rule by rule, so a cap set on the wrong selector, or in a unit that does not follow the type,
fails here.
"""

from __future__ import annotations

import re

import pytest

from threetears.evals.analysis import report_html
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import ReportDoc
from packages.evals.tests.g2_support import g2_report
from packages.evals.tests.report_doc_support import converted_doc, snapshot_of

#: The prose blocks the page lays out: paragraphs (the basis and verdict lines among them), list items, a disclosure
#: and a caveat.
_PROSE = ("p", "li", ".disclosure", ".caveat")

#: A readable measure, in characters: wide enough for a sentence, narrow enough to track a line back.
_READABLE = range(45, 81)


def _rules(page: str) -> dict[str, dict[str, str]]:
    """Each selector the page's stylesheet names, with every declaration any rule gives it."""
    style = page.split("<style>", 1)[1].split("</style>", 1)[0]
    style = re.sub(r"/\*.*?\*/", "", style, flags=re.S)
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


@pytest.fixture
async def page() -> str:
    return report_html(await g2_report())


def test_every_prose_block_carries_a_readable_line_length_cap(page: str) -> None:
    rules = _rules(page)

    for selector in _PROSE:
        cap = rules.get(selector, {}).get("max-width")
        assert cap is not None, f"{selector} carries no line-length cap"
        assert _measure_ch(cap) in _READABLE, f"{selector} is capped at {cap}, not a readable measure in characters"


def test_tables_and_figures_keep_the_pages_width(page: str) -> None:
    rules = _rules(page)

    for selector in ("table", ".tbl", "figure", ".fig", "th, td", "th", "td", ".chart"):
        assert "max-width" not in rules.get(selector, {}), f"{selector} is narrowed like prose"
    assert rules[".page"]["max-width"] == "var(--wide)", "the page itself is bounded"
    (root,) = [declared for selector, declared in rules.items() if selector.endswith(":root")]
    assert root["--wide"] == "68rem", "at the kit's wide measure"
    assert '<div class="page">' in page


def test_the_stylesheet_reader_sees_a_cap_that_is_missing(page: str) -> None:
    page = page.replace("max-width: 78ch;", "")
    assert "max-width" not in _rules(page).get("p", {})


#: A written report's first screen, read top to bottom: its masthead, the callouts under it (a draft's line among
#: them), its headline, its lede and its confidence, each by the rule that sets its width.
_TOP_BLOCK = (".mast", ".callout", ".bluf>h1", ".bluf .lede", ".flag")


def test_a_written_report_s_first_screen_reads_at_one_measure_and_its_cards_at_the_page_s_width() -> None:
    """Discodon's 0.94.0 pass at 1104 px: the masthead ran the page's width, the draft's line stopped at 78ch, the
    headline ran uncapped, and the lede and the confidence stopped at the prose column, four measures in one block.
    Each takes the prose column (``--col``); the decision cards and the options table keep the page's width."""
    name = "prototype3/kairo"
    doc = build(
        ReportDoc.model_validate(converted_doc(name)),
        ReportInput.model_validate(snapshot_of(name).report_input),
        strict=True,
    )
    # Drawn as a draft, so its page carries the draft's line under the masthead.
    page = doc_page_html(doc.model_copy(update={"draft": True}))
    assert '<p class="callout warn" data-part="draft">' in page and "<h1" in page, "the control: a draft's top block"
    rules = _rules(page)
    for selector in _TOP_BLOCK:
        assert rules.get(selector, {}).get("max-width") == "var(--col)", (selector, rules.get(selector))
    for selector in (".decision", ".decision>div", ".tbl", "table"):
        assert "max-width" not in rules.get(selector, {}), f"{selector} is narrowed like prose"
