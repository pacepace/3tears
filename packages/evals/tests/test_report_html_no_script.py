"""The report's HTML runs no script, and nothing an author wrote can make it run or load anything.

The principle: nothing an author or a model wrote can make the page run or load anything; the package itself may
load the kit's fonts from the Google Fonts hosts, so the page is set in the faces the kit's reference memos use.
That is the whole of what the page loads: three links the package writes into its head, a preconnect to each host
and one stylesheet, matched attribute for attribute (:mod:`~packages.evals.tests.page_contract_support`). Any other
``<link>``, any ``<script>`` (the page has none of its own), any element that embeds or loads, any ``on*``
attribute and any ``javascript:`` URL is refused.

Read by parsing the page, not by searching its text: an element is a ``<script>`` because the parser opened one,
and a string that merely contains the word does not count either way. Checked on the toy host's real report and on
a report whose author wrote markup into every field a model fills — a script, a handler, a ``javascript:`` link, an
iframe, a stylesheet link to another host and a copy of the package's own font link — the case the property exists
for, since a page that loads only its own fonts while its prose is polite does not keep it. The charts are inline
SVG the package draws, and are held to the same rules.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis import report_html
from threetears.evals.analysis.report import Report
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.page_contract_support import FONT_LINK_MARKUP, FONT_LINKS, font_links, parsed, what_runs
from packages.evals.tests.report_support import toy_report

#: What an author might write into a field a model fills, to try to make the page run or load something: a script,
#: a handler, a ``javascript:`` link, a frame, a stylesheet from another host, and the package's own font links,
#: which from a field are an author's, not the package's.
_HOSTILE = (
    '<script>alert(1)</script><img src=x onerror="alert(2)"><a href="javascript:alert(3)">x</a>'
    '<iframe src=//evil></iframe><link rel="stylesheet" href="https://evil.example/x.css">' + FONT_LINK_MARKUP
)


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

    assert not [tag for tag, _ in parsed(page).tags if tag == "script"]
    assert what_runs(page) == []
    assert font_links(page) == list(FONT_LINKS), "the page loads the kit's fonts and nothing else"
    assert page.index(FONT_LINK_MARKUP) < page.index("</head>"), "the font links are in the head, verbatim"


def test_markup_an_author_wrote_is_text_not_markup() -> None:
    page = report_html(_hostile_report())

    assert what_runs(page) == []
    assert font_links(page) == list(FONT_LINKS), "an author's link, even a copy of the package's, is text"
    assert any("<script>alert(1)</script>" in text for text in parsed(page).text), "the words survive as text"
    assert any(FONT_LINK_MARKUP.split(">", 1)[0] in text for text in parsed(page).text), "so is the copied font link"


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

    assert what_runs(page) == []
    assert font_links(page) == list(FONT_LINKS)
    assert any(dict(attrs).get("data-basis") == breakout for tag, attrs in parsed(page).tags if tag == "article")


def test_the_markdown_a_memo_uses_still_renders() -> None:
    """Escaping first is not escaping everything: a list, bold and code still become structure."""
    tags = [tag for tag, _ in parsed(report_html(_hostile_report())).tags]
    assert {"ul", "li", "strong", "code"} <= set(tags)


def test_a_chart_draws_the_labels_it_is_given_as_text() -> None:
    page = report_html(_hostile_report())
    assert page.count("<svg") == 2
    svg_text = [text for text in parsed(page).text if "<script>alert(1)</script>" in text]
    assert len(svg_text) > 4, "every label, title and description survives as words"


def test_the_check_sees_a_script_where_one_is() -> None:
    """A detector that found nothing anywhere would pass every page above for nothing."""
    assert what_runs("<p>x</p><script>1</script>") == ["<script>"]
    assert what_runs('<p onclick="x">y</p>') == ["p[onclick]"]
    assert what_runs('<a href=" JavaScript:x">y</a>') == ["a[href=javascript:]"]
    assert what_runs("<img src=x>") == ["<img>"]
    assert what_runs('<svg><use href="#x"></use></svg>') == ["<use>"]


def test_the_check_passes_the_package_s_font_links_and_no_other_link() -> None:
    """Only the three font links pass, attribute for attribute: another host, another rel, an extra attribute or a
    handler on one of them is a link that loads something the package did not write."""
    assert what_runs(FONT_LINK_MARKUP) == []
    assert what_runs('<link rel="stylesheet" href="https://evil.example/x.css">') == ["<link>"]
    assert what_runs('<link rel="preconnect" href="https://fonts.googleapis.com.evil.example">') == ["<link>"]
    assert what_runs('<link rel="prefetch" href="https://fonts.googleapis.com">') == ["<link>"]
    assert what_runs('<link rel="preconnect" href="https://fonts.googleapis.com" id="x">') == ["<link>"]
    assert what_runs('<link rel="preconnect" href="https://fonts.googleapis.com" onload="x">') == [
        "<link>",
        "link[onload]",
    ]
