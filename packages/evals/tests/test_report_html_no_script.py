"""The report's HTML reads without a script, and nothing an author wrote can make it run one.

Read by parsing the page, not by searching its text: an element is a ``<script>`` because the parser
opened one, and a string that merely contains the word does not count either way. Checked on the toy
host's real report and on a report whose author wrote markup into every field a model fills — the case
the property exists for, since a page that is script-free only while its prose is polite is not.

What "runs" means here: a ``script`` element; any element that embeds or loads another document or
resource (``iframe``, ``object``, ``embed``, ``link``, ``base``, ``img``, ``meta`` with ``http-equiv``);
an event-handler attribute (``on*``); and any attribute value that is a ``javascript:`` URL. The page
carries no URL at all, so the last is checked on every attribute rather than on a list of URL attributes —
except ``data-*``, which a browser never navigates to or runs, and which carries a chart's intent as JSON
(an author's caption may say ``javascript:`` there and remain words).
"""

from __future__ import annotations

import json
from html.parser import HTMLParser

from threetears.evals.analysis import report_html
from threetears.evals.analysis.report import Report
from packages.evals.tests.report_support import minimal_report, toy_report
from packages.evals.tests.chart_examples import EVERY_TYPE
from threetears.evals.analysis.viz import chart_intent

#: Elements that execute, embed or load something — none belongs on a document that reads without a script.
_ACTIVE_ELEMENTS = frozenset({"script", "iframe", "object", "embed", "link", "base", "img", "frame", "frameset", "svg"})

#: What an author might write into a field a model fills, to try to make the page run something.
_HOSTILE = (
    '<script>alert(1)</script><img src=x onerror="alert(2)"><a href="javascript:alert(3)">x</a>'
    "<iframe src=//evil></iframe>"
)


class _Page(HTMLParser):
    """Every start tag and its attributes, as the parser read them."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, list[tuple[str, str | None]]]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, attrs))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, attrs))

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def _parsed(page: str) -> _Page:
    parser = _Page()
    parser.feed(page)
    parser.close()
    return parser


def _what_runs(page: str) -> list[str]:
    """Everything on the page that would execute or load — empty for a page that reads without a script."""
    found: list[str] = []
    for tag, attrs in _parsed(page).tags:
        if tag in _ACTIVE_ELEMENTS:
            found.append(f"<{tag}>")
        if tag == "meta" and any(name == "http-equiv" for name, _ in attrs):
            found.append("<meta http-equiv>")
        for name, value in attrs:
            if name.startswith("on"):
                found.append(f"{tag}[{name}]")
            if not name.startswith("data-") and value is not None and "javascript:" in value.replace(" ", "").lower():
                found.append(f"{tag}[{name}=javascript:]")
    return found


def _hostile_report() -> Report:
    """A report whose every author-written field, and a chart caption, carries markup."""
    intent = chart_intent("breakdown", {**EVERY_TYPE["breakdown"], "caption": _HOSTILE})
    return minimal_report(
        headline=_HOSTILE,
        blocks=[
            {"kind": "text", "section": "summary", "role": "summary", "body": f"- {_HOSTILE}\n- **bold** `code`"},
            {"kind": "text", "section": "findings", "role": "finding_title", "finding": 0, "body": _HOSTILE},
            {"kind": "text", "section": "findings", "role": "finding_body", "finding": 0, "body": _HOSTILE},
            {
                "kind": "text",
                "section": "findings",
                "role": "caveat",
                "finding": 0,
                "body": _HOSTILE,
                "facts": [{"name": "Kind", "value": _HOSTILE}],
            },
            {
                "kind": "chart",
                "section": "findings",
                "finding": 0,
                "viz_type": "breakdown",
                "intent": intent.model_dump(),
            },
            {"kind": "chart", "section": "findings", "finding": 0, "viz_type": "frontier", "error": _HOSTILE},
            {"kind": "disclosure", "section": "methods", "source": "generation", "text": _HOSTILE},
        ],
    )


async def test_the_toy_reports_html_contains_no_script_element() -> None:
    """Done when: the HTML contains no script element."""
    _, _, report = await toy_report()
    page = report_html(report)

    assert not [tag for tag, _ in _parsed(page).tags if tag == "script"]
    assert _what_runs(page) == []


def test_markup_an_author_wrote_is_text_not_markup() -> None:
    page = report_html(_hostile_report())

    assert _what_runs(page) == []
    assert any("<script>alert(1)</script>" in text for text in _parsed(page).text), "the words survive as text"


def test_a_closed_value_that_bypassed_validation_is_escaped_too() -> None:
    """``basis``, a text block's ``role`` and a disclosure's ``source`` are closed vocabularies, and still escaped.

    A report built without validation (``model_construct``, ``model_copy``) can carry anything in them, and the
    page's "every attribute escaped" holds of the page, not of the model that usually guards it.
    """
    breakout = '"><script>alert(1)</script><p x="'
    report = _hostile_report()
    blocks = [
        block.model_copy(update={"role": breakout})
        if block.kind == "text"
        else block.model_copy(update={"source": breakout})
        if block.kind == "disclosure"
        else block
        for block in report.blocks
    ]
    page = report_html(report.model_copy(update={"basis": breakout, "blocks": blocks}))

    assert _what_runs(page) == []
    assert any(dict(attrs).get("data-basis") == breakout for tag, attrs in _parsed(page).tags if tag == "article")


def test_the_markdown_a_memo_uses_still_renders() -> None:
    """Escaping first is not escaping everything: a list, bold and code still become structure."""
    tags = [tag for tag, _ in _parsed(report_html(_hostile_report())).tags]
    assert {"ul", "li", "strong", "code"} <= set(tags)


def test_the_embedded_intent_is_the_charts_intent_even_when_its_caption_is_markup() -> None:
    report = _hostile_report()
    (figure,) = [attrs for tag, attrs in _parsed(report_html(report)).tags if tag == "figure"]
    embedded = dict(figure)["data-chart-intent"]

    assert embedded is not None
    assert json.loads(embedded)["caption"] == _HOSTILE


def test_the_check_sees_a_script_where_one_is() -> None:
    """A detector that found nothing anywhere would pass every page above for nothing."""
    assert _what_runs("<p>x</p><script>1</script>") == ["<script>"]
    assert _what_runs('<p onclick="x">y</p>') == ["p[onclick]"]
    assert _what_runs('<a href=" JavaScript:x">y</a>') == ["a[href=javascript:]"]
    assert _what_runs("<img src=x>") == ["<img>"]
