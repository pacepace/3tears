"""The report's HTML runs one script, the kit's own, and nothing an author wrote can make it run or load anything.

The principle: nothing an author or a model wrote can make the page run or load anything. The package itself writes
two things that do (:mod:`~packages.evals.tests.page_contract_support`): the kit's fonts, three links in the head, a
preconnect to each Google Fonts host and one stylesheet, matched attribute for attribute; and the kit's script, one
``<script>`` at the end of the body whose text is byte for byte the ``kit.js`` the package ships, which draws each
chart from the JSON in its ``data-chart``. Any other ``<link>``, any other ``<script>`` (a second copy of the kit, one
with a ``src``, one whose text differs), any element that embeds or loads, any ``on*`` attribute and any
``javascript:`` URL outside a ``data-*`` attribute is refused.

Read by parsing the page, not by searching its text: an element is a ``<script>`` because the parser opened one,
and a string that merely contains the word does not count either way. Checked on the toy host's real report and on
a report whose author wrote markup into every field a model fills — a script, a handler, a ``javascript:`` link, an
iframe, a stylesheet link to another host and a copy of the package's own font link — the case the property exists
for, since a page that loads only its own fonts while its prose is polite does not keep it. A campaign's page is held
to the same case: its campaign name, arm labels, aspect labels, fold lines, memo and folded pages all carry the
markup, since it lays out each aspect's page again and rewrites the ids and links in it.

A chart's labels, titles, tooltips and every other string in its data reach the page only as JSON in its
``data-chart`` attribute, escaped there, so markup in them is never markup on the page; and once the kit has drawn the
charts in headless Chromium (``integration``), no element of the injected markup exists in the document and every
tooltip holds nothing but text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis import (
    CampaignReport,
    build_campaign_report,
    build_code_only_report,
    campaign_readings,
    campaign_report_html,
    report_html,
)
from threetears.evals.analysis.report import Report
from threetears.evals.kernel.authored import AspectSummary, AuthoredCampaignAnalysis, NextStep
from threetears.evals.kernel.declaration import CampaignArm
from packages.evals.tests.campaign_page_support import campaign_fixture, two_aspect_memo
from packages.evals.tests.campaign_support import ASSEMBLED_AT
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.page_contract_support import (
    FONT_LINK_MARKUP,
    FONT_LINKS,
    KIT_JS,
    font_links,
    parsed,
    what_runs,
)
from packages.evals.tests.report_kit_support import chromium_unavailable, rendered
from packages.evals.tests.report_support import toy_report

#: What an author might write into a field a model fills, to try to make the page run or load something: a script,
#: a handler, a ``javascript:`` link, a frame, a stylesheet from another host, and the package's own font links,
#: which from a field are an author's, not the package's.
_HOSTILE = (
    '<script>alert(1)</script><img src=x onerror="alert(2)"><a href="javascript:alert(3)">x</a>'
    '<iframe src=//evil></iframe><link rel="stylesheet" href="https://evil.example/x.css">' + FONT_LINK_MARKUP
)


def _hostile_kit_charts() -> list[dict[str, Any]]:
    """A chart of each kind whose every string the kit draws carries markup: titles, labels, notes, tooltips, legend
    entries, panel titles, column headings, pill words, reasons and an example card's every part."""
    h = _HOSTILE
    return [
        {
            "kind": "chart",
            "section": "memo",
            "finding": 0,
            "chart": "bars",
            "title": h,
            "data": {
                "domain": [0, 10],
                "fmt": "int",
                "level": "95%",
                "legend": [{"label": h, "color": "--s1"}],
                "reference": {"value": 5, "label": h, "kind": "floor"},
                "rows": [
                    {"label": h, "short_label": h, "value": 4, "lo": 3, "hi": 6, "note": h, "tip": [h, h], "group": h}
                ],
            },
        },
        {
            "kind": "chart",
            "section": "memo",
            "finding": 0,
            "chart": "scatter_frontier",
            "title": h,
            "data": {
                "x_domain": [0, 10],
                "y_domain": [0, 100],
                "x_fmt": "usd",
                "y_fmt": "pct",
                "x_title": h,
                "y_title": h,
                "price_line": {"value": 2, "label": h, "region_label": h},
                "points": [
                    {"x": 2, "y": 80, "id": "a", "label": h, "narrow_label": h, "tip": [h]},
                    {"x": 6, "y": 90, "id": "b", "label": h, "narrow_label": h, "tip": [h]},
                ],
            },
        },
        {
            "kind": "chart",
            "section": "memo",
            "finding": 0,
            "chart": "gate_matrix",
            "title": h,
            "data": {
                "row_header": h,
                "pick_label": h,
                "cost_fmt": "usd",
                "cost_unit": h,
                "columns": [{"label": h}],
                "rows": [
                    {
                        "label": h,
                        "detail": h,
                        "flag": h,
                        "cells": [{"status": "bad", "label": h, "reasons": [h], "cost": 1, "pick": True}],
                    }
                ],
            },
        },
        {
            "kind": "chart",
            "section": "memo",
            "finding": 0,
            "chart": "heatmap",
            "title": h,
            "data": {
                "domain": [0, 100],
                "fmt": "pct",
                "columns": [{"label": h}],
                "panels": [
                    {"title": h, "subtitle": h, "rows": [{"label": h, "cells": [{"value": 50, "tip": [h]}]}]},
                    {"title": h, "subtitle": h},
                ],
            },
        },
        {
            "kind": "chart",
            "section": "memo",
            "finding": 0,
            "chart": "examples",
            "title": h,
            "data": {
                "cards": [
                    {
                        "meta": [{"text": h, "mono": True}, {"status": "bad", "label": h}, {"tag": h, "label": h}],
                        "question": h,
                        "fields": [{"term": h, "text": h, "tone": "bad"}],
                        "reason": h,
                        "excerpt": h,
                        "note": h,
                    }
                ]
            },
        },
        {
            "kind": "chart",
            "section": "memo",
            "finding": 0,
            "chart": "lines",
            "title": h,
            "data": {
                "categories": [h, h],
                "domain": [0, 100],
                "fmt": "pct",
                "x_title": h,
                "series": [{"name": h, "end_label": h, "values": [40, 60]}],
            },
        },
    ]


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
            *_hostile_kit_charts(),
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


#: What an author might write where a page prints a label or a line: the hostile markup above, an ``onerror`` and a
#: ``javascript:`` URL in an attribute that tries to break out of its quotes, and an element claiming one of the
#: page's own ids.
_HOSTILE_LINE = (
    _HOSTILE + '"><img src=x onerror="alert(4)"><a id="glossary" href="javascript:alert(5)">x</a>'
    '<section id="aspect-billing">'
)


#: An aspect label carrying markup. A label never holds a ``/``, so this one has no closing tag: a handler, an
#: element that loads, a ``javascript:`` link and an id the page already holds, all in opening tags.
_HOSTILE_ASPECT = '<img src=x onerror="alert(6)"><svg onload="alert(7)"><a id="glossary" href="javascript:alert(8)">'


def _hostile_memo(control: str) -> AuthoredCampaignAnalysis:
    """The two-aspect memo with markup in every field its writer fills, recommending ``control``."""
    memo = two_aspect_memo()
    finding = memo.findings[0].model_copy(update={"title": _HOSTILE_LINE, "body": f"- {_HOSTILE_LINE}\n- **bold**"})
    recommended = memo.recommended()
    assert recommended is not None
    return memo.model_copy(
        update={
            "findings": [finding],
            "recommendation": [recommended.model_copy(update={"arm": control, "reasons": [_HOSTILE_LINE]})],
            "aspect_summaries": [
                AspectSummary(aspect=aspect, what_changed=_HOSTILE_LINE, how_sure=_HOSTILE_LINE, means=_HOSTILE_LINE)
                for aspect in ("billing", _HOSTILE_ASPECT)
            ],
            "summary": _HOSTILE_LINE,
            "next": [NextStep(title=_HOSTILE_LINE, why=_HOSTILE_LINE, leverage="high", lever="")],
        }
    )


async def _hostile_campaign_page() -> CampaignReport:
    """A campaign page of two real aspects, one of them labelled with markup, whose campaign name, arm labels, memo,
    fold lines and second folded page carry markup too: everything on it a host or a model supplies."""
    control_label, candidate_label = f"{_HOSTILE_LINE} A", f"{_HOSTILE_LINE} B"
    campaign, measured = await campaign_fixture("billing", _HOSTILE_ASPECT, name=_HOSTILE_LINE)
    control = campaign.control
    assert control is not None
    arms = [
        control.model_copy(update={"label": control_label}),
        CampaignArm(label=candidate_label, role="candidate", variant_key=measured[0].candidate),
    ]
    bundles = {one.bundle.aspect: one.bundle for one in measured}
    read = campaign_readings(campaign, bundles, arms=arms)
    reports = {
        label: build_code_only_report(
            evidence, assembled_at=ASSEMBLED_AT, campaign_name=_HOSTILE_LINE, arm_labels=read.arm_labels()
        )
        for label, evidence in bundles.items()
    }
    memo = _hostile_memo(control_label)
    page = build_campaign_report(campaign, read, reports, memo, recommended_variant_key=control.variant_key)
    first, second = page.aspects
    folds = [
        first.model_copy(update={"shows": _HOSTILE_LINE}),
        second.model_copy(update={"shows": _HOSTILE_LINE, "report": _hostile_report()}),
    ]
    return page.model_copy(update={"aspects": folds})


async def test_markup_a_host_or_a_model_put_on_a_campaign_page_is_text_not_markup() -> None:
    page = campaign_report_html(await _hostile_campaign_page())
    read = parsed(page)

    assert what_runs(page) == []
    assert font_links(page) == list(FONT_LINKS), "an author's link, even a copy of the package's, is text"
    assert any("<script>alert(1)</script>" in text for text in read.text), "the words survive as text"
    assert any('onerror="alert(4)"' in text for text in read.text)
    assert any(_HOSTILE_ASPECT in text for text in read.text), "the aspect's label is words too"
    drawn = [text for _, data in _chart_data(page) for text in _strings(data)]
    assert any(text.startswith(_HOSTILE_LINE) for text in drawn), "an arm's label reaches its chart as words"
    assert any(FONT_LINK_MARKUP.split(">", 1)[0] in text for text in read.text), "so is the copied font link"
    assert page.count("&lt;script&gt;alert(1)&lt;/script&gt;") > 10, "every field's markup is escaped"
    ids = [value for _, attrs in read.tags for name, value in attrs if name == "id"]
    assert len(ids) == len(set(ids)), "an author's label claimed an id the page already holds"
    hrefs = [value or "" for tag, attrs in read.tags for name, value in attrs if name == "href" and tag != "link"]
    assert {href[1:] for href in hrefs} <= set(ids), "a link lands nowhere"


async def test_the_toy_report_s_html_carries_the_kit_s_script_and_no_other() -> None:
    """Done when: the HTML's one script element is the kit's, verbatim, last on the page."""
    _, _, report = await toy_report()
    page = report_html(report)
    read = parsed(page)

    assert [tag for tag, _ in read.tags if tag == "script"] == ["script"]
    assert list(read.scripts.values()) == [KIT_JS], "the script is the shipped kit.js, byte for byte"
    assert read.tags[-1] == ("script", []), "with no attribute, after every other element"
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


def _chart_data(page: str) -> list[tuple[str, Any]]:
    """Every chart the page hands the kit: its kind and its data, read from the attributes as a browser reads them."""
    return [
        (dict(attrs)["data-kind"] or "", json.loads(dict(attrs)["data-chart"] or ""))
        for tag, attrs in parsed(page).tags
        if tag == "div" and "data-chart" in dict(attrs)
    ]


def _strings(value: Any) -> list[str]:
    """Every string in a JSON value."""
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return [value] if isinstance(value, str) else []


def test_a_chart_carries_the_labels_it_is_given_as_escaped_text_in_its_data() -> None:
    page = report_html(_hostile_report())
    charts = _chart_data(page)
    kinds = [kind for kind, _ in charts]
    assert kinds == ["bars", "bars", "scatter_frontier", "gate_matrix", "heatmap", "examples", "lines"]
    for kind, data in charts:
        assert _HOSTILE in _strings(data), f"a {kind} chart's labels survive as words"
    # In the markup the JSON is escaped into its double-quoted attribute: not one raw angle bracket or quote in it.
    raw = re.findall(r'data-chart="([^"]*)"', page)
    assert len(raw) == len(charts) and all(not set("<>'") & set(value) for value in raw)
    assert "<script>alert(1)</script>" not in page.replace(KIT_JS, "")


def test_interval_rows_reach_the_page_as_their_table_alone() -> None:
    """Interval rows name no interval level, and the kit never draws an interval without one: their figure holds
    their table and no chart."""
    page = report_html(_hostile_report())
    figure = page[page.index('data-figure="interval_rows"') :]
    figure = figure[: figure.index("</figure>")]
    assert 'class="chart"' not in figure and 'class="tbl chart-values"' in figure


@pytest.mark.integration
def test_once_the_kit_has_drawn_the_charts_no_injected_markup_is_an_element(tmp_path: Path) -> None:
    """In headless Chromium, after the kit has drawn every chart of the hostile report: the page holds its one
    script, no element the markup in the labels tried to add, every chart drawn with no error, the injected markup
    present as words, and every tooltip nothing but text (the first line bold)."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = tmp_path / "hostile.html"
    page.write_text(report_html(_hostile_report()), encoding="utf-8")
    probe = """() => {
      const charts = [...document.querySelectorAll('div.chart[data-kind]')];
      const tips = [...document.querySelectorAll('[data-tip]')].flatMap(e => {
        const t = document.createElement('template'); t.innerHTML = e.getAttribute('data-tip');
        return [...t.content.querySelectorAll('*')].map(x => x.tagName.toLowerCase()); });
      return {
        scripts: document.scripts.length,
        injected: [...document.querySelectorAll('img, iframe, link, [onerror], [onload], a[href^="javascript"], svg svg')]
          .filter(e => !(e.tagName === 'LINK' && e.parentElement === document.head)).map(e => e.outerHTML.slice(0, 80)),
        charts: charts.length,
        undrawn: charts.filter(c => !c.children.length).map(c => c.dataset.kind),
        words: charts.filter(c => c.textContent.includes('<script>alert(1)</script>')).length,
        tips: [...new Set(tips)].sort(),
        tipCount: document.querySelectorAll('[data-tip]').length,
      };
    }"""
    seen, errors = rendered(page, probe)
    assert errors == []
    assert seen["scripts"] == 1
    assert seen["injected"] == []
    assert seen["charts"] == 7 and seen["undrawn"] == []
    assert seen["words"] >= 5, "the markup is drawn as words"
    assert seen["tipCount"] > 0 and set(seen["tips"]) <= {"b", "br"}


def test_the_check_sees_a_script_where_one_is() -> None:
    """A detector that found nothing anywhere would pass every page above for nothing."""
    assert what_runs("<p>x</p><script>1</script>") == ["<script>"]
    assert what_runs(f"<p>x</p><script>{KIT_JS}</script>") == []
    assert what_runs('<p onclick="x">y</p>') == ["p[onclick]"]
    assert what_runs('<a href=" JavaScript:x">y</a>') == ["a[href]", "a[href=javascript:]"]
    assert what_runs('<a href="java\tscript:x">y</a>') == ["a[href]", "a[href=javascript:]"]
    assert what_runs("<img src=x>") == ["<img>", "img[src]"]
    assert what_runs('<svg><use href="#x"></use></svg>') == ["<use>"]


#: One small page per thing that loads, embeds, submits or navigates, and what the check names on it.
_LOADERS: list[tuple[str, list[str]]] = [
    ('<video src="v.mp4"></video>', ["<video>", "video[src]"]),
    ('<video poster="p.png"></video>', ["<video>", "video[poster]"]),
    ('<audio src="a.mp3"></audio>', ["<audio>", "audio[src]"]),
    ('<audio><source src="a.mp3"></audio>', ["<audio>", "<source>", "source[src]"]),
    ('<video><track src="t.vtt"></video>', ["<video>", "<track>", "track[src]"]),
    ('<picture><source srcset="a.png"></picture>', ["<picture>", "<source>", "source[srcset]"]),
    ('<form action="https://evil.example"></form>', ["<form>", "form[action]"]),
    ('<button formaction="https://evil.example">x</button>', ["button[formaction]"]),
    ('<input type="image" src="x.png">', ["<input type=image>", "input[src]"]),
    ('<input type=" Image ">', ["<input type=image>"]),
    ('<object data="x.swf"></object>', ["<object>", "object[data]"]),
    ('<p data="x">y</p>', ["p[data]"]),
    ('<portal src="https://evil.example"></portal>', ["<portal>", "portal[src]"]),
    ('<div src="x">y</div>', ["div[src]"]),
    ('<div srcset="x 1x">y</div>', ["div[srcset]"]),
    ('<table background="x.png"><tr><td>y</td></tr></table>', ["table[background]"]),
    ('<a href="#x" ping="https://evil.example">y</a>', ["a[ping]"]),
    ('<a href="https://evil.example">y</a>', ["a[href]"]),
    ('<a href="//evil.example">y</a>', ["a[href]"]),
    ('<area href="x.html">', ["area[href]"]),
    ('<svg><a xlink:href="https://evil.example">y</a></svg>', ["a[xlink:href]"]),
    ('<svg><a href="https://evil.example"><text>y</text></a></svg>', ["a[href]"]),
    ("<p style=\"background: URL ('x.png')\">y</p>", ["p[style=url()]"]),
    ("<style>p { background: url(x.png) }</style>", ["<style> url()/@import"]),
    ('<style>@import "https://evil.example/x.css";</style>', ["<style> url()/@import"]),
    ('<meta http-equiv="refresh" content="0;url=https://evil.example">', ["<meta http-equiv>"]),
    ('<iframe srcdoc="x"></iframe>', ["<iframe>"]),
    ("<p>y</p>" + FONT_LINK_MARKUP, ["<link>", "link[href]", "<link>", "link[href]", "<link>", "link[href]"]),
    ("<html><head></head><body>" + FONT_LINK_MARKUP + "</body></html>", ["<link>", "link[href]"] * 3),
]


#: One small page per way a script could pass for the kit's, and what the check names on it: the kit's own passes
#: only alone, attribute-less, in the body, last, byte for byte.
_SCRIPTS: list[tuple[str, list[str]]] = [
    (f"<p>x</p><script>{KIT_JS}</script><script>{KIT_JS}</script>", ["<script>", "<script>"]),
    (f"<p>x</p><script>alert(1)</script><script>{KIT_JS}</script>", ["<script>", "<script>"]),
    (f'<p>x</p><script src="https://evil.example/kit.js">{KIT_JS}</script>', ["<script>", "script[src]"]),
    (f'<p>x</p><script type="module">{KIT_JS}</script>', ["<script>"]),
    (f"<p>x</p><script>{KIT_JS} </script>", ["<script>"]),
    (f"<p>x</p><script>{KIT_JS.replace('drawCharts', 'drawCharts2', 1)}</script>", ["<script>"]),
    (f"<p>x</p><script>{KIT_JS}</script><p>y</p>", ["<script>"]),
    (f"<html><head><script>{KIT_JS}</script></head><body><p>x</p></body></html>", ["<script>"]),
]


def test_the_check_passes_the_kit_s_script_only_alone_verbatim_and_last() -> None:
    for page, refused in _SCRIPTS:
        assert what_runs(page) == refused, page[:60]


def test_the_check_refuses_every_element_and_attribute_that_loads_embeds_or_navigates() -> None:
    """Each page holds one offender, and the check names it; the font links pass only in the head."""
    for page, refused in _LOADERS:
        assert what_runs(page) == refused, page


def test_the_check_passes_a_page_that_loads_only_the_font_links_and_runs_only_the_kit() -> None:
    """A page as the package writes one: the font links in the head, links to places on the page, inline SVG with a
    title, a ``data-*`` attribute, a chart whose JSON carries a ``javascript:`` string as data, a style attribute
    naming no URL, the word ``url(`` in prose, and the kit's script last."""
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>t</title>'
        + FONT_LINK_MARKUP
        + "<style>body { color: var(--ink); }</style></head><body>"
        '<nav><a href="#memo">Memo</a></nav><article data-basis="analysis"><section id="memo">'
        '<p style="--w: 3">Read url(x) and javascript: in prose.</p><a href="#g-interval" title="t">interval</a>'
        '<svg viewBox="0 0 10 10" role="img"><title>bar</title><rect width="5" height="5"></rect></svg>'
        '<div class="chart" data-kind="bars" data-chart="{&quot;note&quot;:&quot;javascript:x&quot;}"></div>'
        f"</section></article><script>{KIT_JS}</script></body></html>"
    )
    assert what_runs(page) == []
    assert font_links(page) == list(FONT_LINKS)


def test_the_check_passes_the_package_s_font_links_and_no_other_link() -> None:
    """Only the three font links pass, attribute for attribute: another host, another rel, an extra attribute or a
    handler on one of them is a link that loads something the package did not write."""
    assert what_runs(FONT_LINK_MARKUP) == []
    assert what_runs('<link rel="stylesheet" href="https://evil.example/x.css">') == ["<link>", "link[href]"]
    assert what_runs('<link rel="preconnect" href="https://fonts.googleapis.com.evil.example">') == [
        "<link>",
        "link[href]",
    ]
    assert what_runs('<link rel="prefetch" href="https://fonts.googleapis.com">') == ["<link>", "link[href]"]
    assert what_runs('<link rel="preconnect" href="https://fonts.googleapis.com" id="x">') == ["<link>", "link[href]"]
    assert what_runs('<link rel="preconnect" href="https://fonts.googleapis.com" onload="x">') == [
        "<link>",
        "link[href]",
        "link[onload]",
    ]
