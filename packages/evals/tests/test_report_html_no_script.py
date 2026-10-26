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

from threetears.evals.analysis import KIT_CHARTS, campaign_report_html, report_html
from packages.evals.tests.hostile_page_support import (
    HOSTILE,
    HOSTILE_ASPECT,
    HOSTILE_LINE,
    hostile_campaign_page,
    hostile_kit_charts,
    hostile_report,
)
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


async def test_markup_a_host_or_a_model_put_on_a_campaign_page_is_text_not_markup() -> None:
    page = campaign_report_html(await hostile_campaign_page())
    read = parsed(page)

    assert what_runs(page) == []
    assert font_links(page) == list(FONT_LINKS), "an author's link, even a copy of the package's, is text"
    assert any("<script>alert(1)</script>" in text for text in read.text), "the words survive as text"
    assert any('onerror="alert(4)"' in text for text in read.text)
    assert any(HOSTILE_ASPECT in text for text in read.text), "the aspect's label is words too"
    drawn = [text for _, data in _chart_data(page) for text in _strings(data)]
    assert any(text.startswith(HOSTILE_LINE) for text in drawn), "an arm's label reaches its chart as words"
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
    page = report_html(hostile_report())

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
    report = hostile_report()
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
    tags = [tag for tag, _ in parsed(report_html(hostile_report())).tags]
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
    page = report_html(hostile_report())
    charts = _chart_data(page)
    kinds = [kind for kind, _ in charts]
    assert kinds == ["bars", *(chart["chart"] for chart in hostile_kit_charts())]
    assert set(kinds) == set(KIT_CHARTS), "every kind the kit draws"
    for kind, data in charts:
        assert HOSTILE in _strings(data), f"a {kind} chart's labels survive as words"
    # In the markup the JSON is escaped into its double-quoted attribute: not one raw angle bracket or quote in it.
    raw = re.findall(r'data-chart="([^"]*)"', page)
    assert len(raw) == len(charts) and all(not set("<>'") & set(value) for value in raw)
    assert "<script>alert(1)</script>" not in page.replace(KIT_JS, "")


def test_interval_rows_reach_the_page_as_their_table_alone() -> None:
    """Interval rows name no interval level, and the kit never draws an interval without one: their figure holds
    their table and no chart."""
    page = report_html(hostile_report())
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
    page.write_text(report_html(hostile_report()), encoding="utf-8")
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
    assert seen["charts"] == 1 + len(KIT_CHARTS) and seen["undrawn"] == []
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
