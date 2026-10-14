"""The report's HTML reads without a script, and nothing an author wrote can make it run one.

Read by parsing the page, not by searching its text: an element is a ``<script>`` because the parser
opened one, and a string that merely contains the word does not count either way. Checked on the toy
host's real report and on a report whose author wrote markup into every field a model fills — the case
the property exists for, since a page that is script-free only while its prose is polite is not.

What "runs" means here: a ``script`` element; any element that embeds or loads another document or
resource (``iframe``, ``object``, ``embed``, ``link``, ``base``, ``img``, ``meta`` with ``http-equiv``, and inside
the inline SVG ``image``, ``use`` and ``foreignObject``); an event-handler attribute (``on*``); and any attribute
value that is a ``javascript:`` URL. The page links only to places on itself (``#F1``, ``#T3``), so the last is
checked on every attribute rather than on a list of URL attributes — except ``data-*``, which a browser never
navigates to or runs. The charts are inline SVG the package draws, and are held to the same rules.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any

from threetears.evals.analysis import report_html
from threetears.evals.analysis.report import Report
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.report_support import toy_report

#: Elements that execute, embed or load something — none belongs on a document that reads without a script.
_ACTIVE_ELEMENTS = frozenset(
    {"script", "iframe", "object", "embed", "link", "base", "img", "frame", "frameset", "image", "use", "foreignobject"}
)

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
    """A report whose every author-written field, every label a chart draws and every trial's text carries markup."""
    fields: dict[str, Any] = {
        "basis": "analysis",
        "title": _HOSTILE,
        "finding_count": 1,
        "source": {
            "analysis_id": "a",
            "campaign_id": "c",
            "scope_id": "s",
            "behavior": "b",
            "generated_at": "2026-10-05T00:00:00+00:00",
            "generator_model": "m",
            "bundle_fingerprint": "f",
        },
        "blocks": [
            {"kind": "disclosure", "section": "header", "source": "verdict", "text": _HOSTILE},
            {"kind": "text", "section": "memo", "role": "finding_title", "finding": 0, "body": _HOSTILE},
            {
                "kind": "text",
                "section": "memo",
                "role": "finding_body",
                "finding": 0,
                "body": f"- {_HOSTILE}\n- **bold** `code`",
            },
            {
                "kind": "text",
                "section": "memo",
                "role": "caveat",
                "finding": 0,
                "body": _HOSTILE,
                "facts": [{"name": "Kind", "value": _HOSTILE}],
            },
            {
                "kind": "chart",
                "section": "scoreboard",
                "chart": "outcome_bar",
                "title": _HOSTILE,
                "bars": [{"arm": _HOSTILE, "segments": [{"state": "lost", "label": _HOSTILE, "count": 2}]}],
            },
            {
                "kind": "chart",
                "section": "memo",
                "finding": 0,
                "chart": "interval_rows",
                "title": _HOSTILE,
                "rows": [{"label": _HOSTILE, "value": 1.0, "low": 0.5, "high": 1.5, "n_cases": 6, "verdict": _HOSTILE}],
                "unit": _HOSTILE,
            },
            {
                "kind": "table",
                "section": "cases",
                "name": "cases",
                "title": _HOSTILE,
                "columns": [
                    {"key": "case", "header": "Case"},
                    {"key": "label", "header": _HOSTILE},
                    {"key": "trials", "header": "Trials"},
                ],
                "rows": [
                    {"case": {"value": "case 1"}, "label": {"value": _HOSTILE}, "trials": {"value": f"T1 {_HOSTILE}"}}
                ],
                "order": _HOSTILE,
                "total_rows": 1,
            },
            {
                "kind": "trial",
                "section": "trials",
                "anchor": "T1",
                "trial_id": "T1",
                "case_label": _HOSTILE,
                "arm": _HOSTILE,
                "repeat": 1,
                "state": "lost",
                "cause_words": _HOSTILE,
                "first_input": _HOSTILE,
                "excerpt": _HOSTILE,
                "failed_checks": [{"name": _HOSTILE, "expression": _HOSTILE}],
                "judge": [
                    {"dim": _HOSTILE, "criterion": _HOSTILE, "score": None, "scale": "ordinal", "reason": _HOSTILE}
                ],
                "error_words": _HOSTILE,
                "result_id": _HOSTILE,
                "open": True,
            },
            {"kind": "disclosure", "section": "methods", "source": "generation", "text": _HOSTILE},
        ],
        "bundle": toyhost_bundle(),
    }
    return Report.model_validate(fields)


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


def test_a_chart_draws_the_labels_it_is_given_as_text() -> None:
    page = report_html(_hostile_report())
    assert page.count("<svg") == 2
    svg_text = [text for text in _parsed(page).text if "<script>alert(1)</script>" in text]
    assert len(svg_text) > 4, "every label, title and description survives as words"


def test_the_check_sees_a_script_where_one_is() -> None:
    """A detector that found nothing anywhere would pass every page above for nothing."""
    assert _what_runs("<p>x</p><script>1</script>") == ["<script>"]
    assert _what_runs('<p onclick="x">y</p>') == ["p[onclick]"]
    assert _what_runs('<a href=" JavaScript:x">y</a>') == ["a[href=javascript:]"]
    assert _what_runs("<img src=x>") == ["<img>"]
    assert _what_runs('<svg><use href="#x"></use></svg>') == ["<use>"]
