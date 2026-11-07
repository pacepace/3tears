"""A report page fits the host that frames it.

The host shows the page verbatim in a sandboxed ``srcdoc`` frame, 1104 px wide at a 1440 window and 688 px at 1024.
Held here, on the page as served (no browser runs on the pool, so the page's structure and its stylesheet are read),
and, where headless Chromium launches (the ``integration`` marker), on the page as drawn at both frames:

- **Tables fit a 688 px frame, or say they scroll.** Every table sits in its own ``.tbl`` frame, which scrolls
  sideways with a shadow at an edge while more lies past it and its first column in view; the To settle table leads
  with the arm and both counts, its "Why unclear" in words that wrap; a page lays every arm by measure out a figure
  per row and an arm per column (its text form, which the editor and a judge read, as built); a settle table holds four columns and no note warns a reader off one; the evidence record a written
  report carries below its document is laid out alike.
- **By stratum reads cold.** A cell of one case run once is its count (``1 case``), a count of one is never plural,
  and where every stratum is too small to read alone one line says so instead of every cell.
- **The host's tokens.** The page reads ``--host-font-body``, ``--host-font-mono``, ``--host-color-text``,
  ``--host-color-muted``, ``--host-color-surface``, ``--host-color-link`` and ``--host-color-accent``, each with its
  own value as the fallback, and declares none of them, so a host's value wins in either theme; secondary text, rules
  and tints mix from them; ``data-theme`` on the page's one ``<html>`` tag wins over the reader's colour scheme.
- **Nothing is fetched.** No stylesheet link, no font host, no ``url(``: the faces are a local stack ending in a
  generic family.
- **One html tag.** The page opens ``<!doctype html>`` then exactly one ``<html lang="en">``, which a host rewrites to
  stamp ``data-theme`` and an inline ``style``.
- **A written report's masthead and cards.** A count of one reads singular (``vs 1 candidate``), and the card grid
  leaves no empty cell beside a short card.
- **A pinned column is never crushed.** It holds a floor as well as its cap, and wraps at its words; only an id-like
  cell (an expression, a digest: classed ``unbroken``) breaks anywhere. Drawn, no pinned cell is under its floor and
  no word in one breaks across lines but an id-like cell's, and ``scripts/check.py`` fails a page where one is.
- **The nav paints over the tables.** Each table frame isolates its cells' stacking, so the sticky nav covers a pinned
  cell scrolled beneath it.
- **Headings fit the host's narrow frame.** Under 700 px they are a step smaller and tighter: a headline holds three
  lines at 688 px.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis import (
    DisclosureBlock,
    Report,
    TableBlock,
    assemble_context_bundle,
    build_code_only_report,
    campaign_report_html,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.analyst.blocks import block_render
from threetears.evals.analysis.analyst.settle import settle_cases
from threetears.evals.analysis.report.build import TOO_FEW_CASES
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report.doc_page.text import doc_page_text
from threetears.evals.analysis.report.model import CampaignReport
from threetears.evals.analysis.report.parts.full_results import laid_out
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel import STRATUM_MIN_CASES, EvalStorage
from threetears.evals.kernel.chart_query import ChartQuery
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.host import SHARED_CORE, HostProfile, MeasureRegistry
from threetears.evals.kernel.report_doc import ReportDoc
from threetears.evals.schema import EvalResult, EvalTestCase
from threetears.evals.storage import InMemoryDocumentStore

from packages.evals.tests.analyst_support import fixture_inputs
from packages.evals.tests.campaign_page_support import kairo_page, page_of
from packages.evals.tests.clock_support import frozen_clock
from packages.evals.tests.fixtures.report_inputs.dataset import kairo_bundle
from packages.evals.tests.factories import fixture_variant_key, make_campaign, make_eval_result, make_eval_run
from packages.evals.tests.page_contract_support import links, parsed
from packages.evals.tests.report_doc_support import LATER_DOCS, converted_doc, snapshot_of
from packages.evals.tests.report_kit_support import chromium_unavailable, render_checked, rendered

#: The narrowest frame the host gives the page, and the page's side gutters and its table frame's border.
FRAME_PX = 688
GUTTERS_PX = 2 * 16 + 2

#: The seven tokens a host sets, and no others.
HOST_TOKENS = frozenset(
    {
        "--host-font-body",
        "--host-font-mono",
        "--host-color-text",
        "--host-color-muted",
        "--host-color-surface",
        "--host-color-link",
        "--host-color-accent",
    }
)

#: Each page colour a host token sets, the token, and the page's own value in light and in dark.
HOST_COLOURS: dict[str, tuple[str, str, str]] = {
    "--ink": ("--host-color-text", "#11151b", "#eef1f5"),
    "--muted": ("--host-color-muted", "#7f8896", "#828b99"),
    "--bg": ("--host-color-surface", "#f5f6f8", "#0e1115"),
    "--surface": ("--host-color-surface", "#ffffff", "#161a20"),
    "--accent": ("--host-color-accent", "#1f5fbf", "#6aa2f0"),
}

#: The colours the page mixes from the host's, so a host's palette carries through: secondary text, the sunken tint,
#: rules and the accent's wash.
MIXED = ("--ink-2", "--sunk", "--line", "--accent-wash")

#: The documents that build strict (``test_doc_page.py``'s).
DOCS = ("prototype/kairo", *LATER_DOCS)


@pytest.fixture(autouse=True)
def _frozen_clock() -> Iterator[None]:
    with frozen_clock():
        yield


# =============================================================================
# The pages read here
# =============================================================================


@cache
def _input(name: str) -> ReportInput:
    return ReportInput.model_validate(snapshot_of(name).report_input)


@cache
def _doc_page(name: str) -> DocPage:
    return build(ReportDoc.model_validate(converted_doc(name)), _input(name), strict=True)


@cache
def _settle_page() -> CampaignReport:
    """A campaign whose billing row is left unclear against its 70-point margin: its page has a To settle table."""
    return asyncio.run(page_of("billing", "tone", margins={"billing": 70.0}))


def _pages() -> dict[str, str]:
    """Every kind of page the package serves, as served: a campaign's, static and not; an aspect's; a written report's,
    alone and with code's evidence record below it."""
    campaign = _settle_page()
    report = next(iter(_aspect_reports(campaign)))
    with_record = campaign.model_copy(update={"doc_page": _doc_page("prototype3/kairo"), "doc_trials": True})
    return {
        "campaign": campaign_report_html(campaign),
        "campaign static": campaign_report_html(campaign, static=True),
        "aspect": report_html(report),
        "doc": doc_page_html(_doc_page("prototype3/kairo")),
        "doc static": doc_page_html(_doc_page("prototype3/kairo"), static=True),
        "doc with its record": campaign_report_html(with_record),
    }


def _aspect_reports(campaign: CampaignReport) -> Iterator[Report]:
    for fold in campaign.aspects:
        yield fold.report


@cache
def _served() -> dict[str, str]:
    return _pages()


# =============================================================================
# The stylesheet, read as rules
# =============================================================================


@dataclass(frozen=True)
class _Rule:
    """One style rule: the at-rule it sits in (empty at the top level), its selector and its declarations."""

    at: str
    selector: str
    body: str

    def declarations(self) -> dict[str, str]:
        found: dict[str, str] = {}
        for part in _split_declarations(self.body):
            name, _, value = part.partition(":")
            if value:
                found[name.strip()] = value.strip()
        return found


def _split_declarations(body: str) -> list[str]:
    """``body`` split at its top-level semicolons (one inside a ``var(...)`` fallback never ends a declaration)."""
    parts, depth, start = [], 0, 0
    for at, char in enumerate(body):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if char == ";" and depth == 0:
            parts.append(body[start:at])
            start = at + 1
    parts.append(body[start:])
    return [part for part in parts if part.strip()]


def _rules(css: str) -> list[_Rule]:
    """Every rule of ``css`` in order, each with the at-rule around it."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    rules: list[_Rule] = []

    def walk(text: str, at: str) -> None:
        position = 0
        while (opening := text.find("{", position)) >= 0:
            prelude = text[position:opening].strip()
            depth, at_char = 1, opening + 1
            while depth:
                depth += {"{": 1, "}": -1}.get(text[at_char], 0)
                at_char += 1
            inner = text[opening + 1 : at_char - 1]
            if prelude.startswith("@"):
                walk(inner, prelude)
            else:
                rules.append(_Rule(at=at, selector=prelude, body=inner))
            position = at_char

    walk(css, "")
    return rules


def _style(page: str) -> str:
    return "".join(parsed(page).styles)


def _roots(css: str) -> tuple[_Rule, _Rule, _Rule]:
    """The three blocks declaring the page's tokens: light, dark by the reader's scheme, dark by ``data-theme``."""
    roots = [rule for rule in _rules(css) if rule.selector.startswith(":root")]
    assert [(rule.at, rule.selector) for rule in roots] == [
        ("", ":root"),
        ("@media (prefers-color-scheme:dark)", ':root:not([data-theme="light"])'),
        ("", ':root[data-theme="dark"]'),
    ], [(rule.at, rule.selector) for rule in roots]
    light, by_scheme, by_attribute = roots
    return light, by_scheme, by_attribute


def _specificity(selector: str) -> tuple[int, int, int]:
    """A simple selector chain's specificity, as :root-led selectors here are written (``:not(x)`` counts as x)."""
    plain = re.sub(r":not\(([^)]*)\)", r"\1", selector)
    ids = len(re.findall(r"#[\w-]+", plain))
    classes = len(re.findall(r"\.[\w-]+|\[[^\]]+\]|:(?!:)[\w-]+", plain))
    elements = len(re.findall(r"(?:^|[\s>+~])[a-z][\w-]*", plain))
    return ids, classes, elements


# =============================================================================
# The host's tokens
# =============================================================================


@pytest.mark.parametrize("name", ["campaign", "campaign static", "aspect", "doc", "doc static"])
def test_the_page_reads_each_host_token_and_declares_none(name: str) -> None:
    css = _style(_served()[name])
    declared = re.findall(r"(?<![\w-])(--host-[\w-]+)\s*:", css)
    assert declared == [], f"the page declares a host token, which would beat a host's: {declared}"
    assert set(re.findall(r"var\((--host-[\w-]+)", css)) == HOST_TOKENS


def test_each_theme_reads_text_muted_surface_and_accent_from_the_host_with_its_own_value_after() -> None:
    light, by_scheme, by_attribute = _roots(_style(_served()["campaign"]))
    for theme, rule in (("light", light), ("dark", by_scheme), ("dark", by_attribute)):
        declared = rule.declarations()
        for colour, (token, light_value, dark_value) in HOST_COLOURS.items():
            fallback = light_value if theme == "light" else dark_value
            assert declared[colour].replace(" ", "") == f"var({token},{fallback})", (rule.selector, colour)
    assert light.declarations()["--link"].replace(" ", "") == "var(--host-color-link,var(--accent))"


def test_secondary_text_rules_and_tints_mix_from_the_host_s_colours_in_either_theme() -> None:
    light, by_scheme, by_attribute = _roots(_style(_served()["campaign"]))
    for colour in MIXED:
        value = light.declarations()[colour]
        assert value.startswith("color-mix("), (colour, value)
        assert set(re.findall(r"var\((--[\w-]+)", value)) <= set(HOST_COLOURS), (colour, value)
        # Each dark block mixes it the same way from its own colours: a value of its own would set a colour over the
        # host's.
        assert by_scheme.declarations()[colour] == value == by_attribute.declarations()[colour], colour


def test_every_face_is_the_host_s_else_a_local_stack_ending_in_a_generic_family() -> None:
    light, by_scheme, by_attribute = _roots(_style(_served()["campaign"]))
    faces = light.declarations()
    for variable, token, generic in (
        ("--f-body", "--host-font-body", "sans-serif"),
        ("--f-display", "--host-font-body", "sans-serif"),
        ("--f-mono", "--host-font-mono", "monospace"),
    ):
        value = faces[variable]
        assert value.startswith(f"var({token},") and value.endswith(f"{generic})"), (variable, value)
    assert not any(name.startswith("--f-") for name in (*by_scheme.declarations(), *by_attribute.declarations()))


@pytest.mark.parametrize("name", ["campaign", "doc"])
def test_every_rule_takes_its_face_and_link_colour_from_the_tokens(name: str) -> None:
    rules = [rule for rule in _rules(_style(_served()[name])) if not rule.selector.startswith(":root")]
    for rule in rules:
        declared = rule.declarations()
        if "font-family" in declared:
            assert declared["font-family"] in ("var(--f-body)", "var(--f-display)", "var(--f-mono)", "inherit"), rule
        if "font" in declared and not declared["font"].startswith("inherit"):
            assert "var(--f-" in declared["font"], rule
    (link,) = [rule for rule in rules if rule.selector == "a"]
    assert link.declarations()["color"] == "var(--link)"


def test_data_theme_on_the_html_tag_wins_over_the_reader_s_colour_scheme() -> None:
    """Dark by the reader's scheme applies only where no host stamped ``data-theme="light"`` (a light page under a
    dark OS stays light); ``data-theme="dark"`` sets dark under a light OS. Both dark blocks declare alike, each beats
    the light block by specificity, and the attribute's comes after the scheme's."""
    css = _style(_served()["campaign"])
    light, by_scheme, by_attribute = _roots(css)
    assert by_scheme.declarations() == by_attribute.declarations()
    assert _specificity(by_scheme.selector) > _specificity(light.selector)
    assert _specificity(by_attribute.selector) > _specificity(light.selector)
    rules = _rules(css)
    assert rules.index(light) < rules.index(by_scheme) < rules.index(by_attribute)
    assert [rule for rule in rules if "data-theme" in rule.selector] == [by_scheme, by_attribute]


# =============================================================================
# One html tag, and nothing fetched
# =============================================================================


@pytest.mark.parametrize("name", ["campaign", "campaign static", "aspect", "doc", "doc static", "doc with its record"])
def test_the_page_opens_with_one_html_tag_a_host_can_rewrite(name: str) -> None:
    page = _served()[name]
    assert page.startswith('<!doctype html>\n<html lang="en">\n<head>'), page[:60]
    assert len(re.findall(r"<html\b", page, flags=re.I)) == 1


@pytest.mark.parametrize("name", ["campaign", "campaign static", "aspect", "doc", "doc static", "doc with its record"])
def test_the_page_loads_no_stylesheet_and_no_font(name: str) -> None:
    page = _served()[name]
    css = _style(page)
    assert links(page) == []
    assert not re.search(r"@import|@font-face|url\(|https?://", css), "the stylesheet reaches outside the page"
    assert "fonts.googleapis" not in page and "fonts.gstatic" not in page


# =============================================================================
# Tables at 688 px
# =============================================================================


@pytest.mark.parametrize("name", ["campaign", "campaign static", "aspect", "doc", "doc with its record"])
def test_every_table_scrolls_in_its_own_frame(name: str) -> None:
    page = _served()[name]
    tables = page.count("<table")
    assert tables, "the control: the page holds tables"
    framed = len(re.findall(r'<div class="tbl(?: [\w-]+)*"><table', page))
    assert framed == tables, "a table outside a scrolling frame widens the page"


def test_a_scrolling_frame_shows_its_overflow_and_keeps_its_first_column_in_view() -> None:
    rules = _rules(_style(_served()["campaign"]))
    (frame,) = [rule for rule in rules if rule.selector == ".tbl" and not rule.at]
    declared = frame.declarations()
    assert declared["overflow-x"] == "auto"
    # The covers scroll with the content (local) and the shadows stay with the frame (scroll): a shadow shows at an
    # edge only while more lies past it, on the static page too, since no script draws it.
    assert declared["background"].count(" local") == 2 and declared["background"].count(" scroll") == 2
    (first,) = [rule for rule in rules if rule.selector == ".tbl :is(th,td):first-child"]
    pinned = first.declarations()
    assert (pinned["position"], pinned["left"]) == ("sticky", "0")
    assert pinned["background"].startswith("var("), "a pinned column is opaque, or the scrolled cells show through"
    (header,) = [rule for rule in rules if rule.selector == "th" and not rule.at and "white-space" in rule.body]
    assert header.declarations()["white-space"] == "normal", "a header wraps rather than widen its column"


#: A pinned column's floor and cap, as the stylesheet writes them: ``min(<rem>rem,<vw>vw)``.
_SPAN = re.compile(r"min\((\d+)rem,(\d+)vw\)")


def _px(value: str, frame: int = FRAME_PX) -> float:
    """A pinned column's floor or cap in px in a frame ``frame`` px wide (the root size 16 px)."""
    span = _SPAN.fullmatch(value.replace(" ", ""))
    assert span is not None, value
    return min(int(span.group(1)) * 16, int(span.group(2)) * frame / 100)


def test_a_pinned_first_column_wraps_at_its_words_between_a_floor_and_a_third_of_the_frame() -> None:
    """A long label in the first column (an example's ``T44 · case · stratum · arm · judged · repeat 1``) wraps within
    its cap instead of filling the frame, so the columns past it can be scrolled to at 390 and 688 px; and the columns
    past it never crush it below its floor (production, 0.93: "Reading" set ``Rea/din/g``, an aspect's name one
    letter a line). It wraps at its spaces, never inside a word: only an id-like cell breaks anywhere."""
    rules = _rules(_style(_served()["campaign"]))
    (first,) = [rule for rule in rules if rule.selector == ".tbl :is(th,td):first-child"]
    pinned = first.declarations()
    assert pinned["white-space"] == "normal"
    assert (pinned["overflow-wrap"], pinned["word-break"], pinned["hyphens"]) == ("normal", "normal", "manual")
    # An arm's chip in it wraps too, rather than run over the column beside it.
    (chip,) = [rule for rule in rules if rule.selector == ".tbl :is(th,td):first-child .armchip"]
    assert chip.declarations()["white-space"] == "normal"
    cap, floor = _px(pinned["max-width"]), _px(pinned["min-width"])
    span = _SPAN.fullmatch(pinned["max-width"].replace(" ", ""))
    assert span is not None and cap <= 0.4 * FRAME_PX and int(span.group(2)) <= 40, pinned
    # The floor holds a word of the page's own (``Question``, ``qwen3.8-flash``) whole, at 390 px too.
    assert 6 * 16 <= floor <= cap and _px(pinned["min-width"], 390) >= 100, pinned
    (unbroken,) = [rule for rule in rules if ".unbroken" in rule.selector]
    assert unbroken.declarations() == {"overflow-wrap": "anywhere"}
    assert unbroken.selector == ".tbl :is(th,td):first-child.unbroken,.tbl.keys-2 :is(th,td):nth-child(2).unbroken", (
        "only a pinned column breaks an id-like cell anywhere; a column that scrolls keeps its header whole"
    )


@cache
def _kairo_page() -> str:
    """Kairo's code page as served: three arms, so its contrasts hold a row per reading on each arm."""
    return report_html(build_code_only_report(kairo_bundle().aspects[0], assembled_at="2026-11-08T00:00:00+00:00"))


def _framed(page: str, name: str) -> tuple[str, str]:
    """Table ``name``'s frame's classes on ``page`` and the table's markup."""
    held = page.split(f'data-table="{name}"', 1)[1]
    frame = re.search(r'<div class="(tbl(?: [\w-]+)*)"><table', held)
    assert frame is not None, f"{name} sits in no frame"
    return frame.group(1), held[frame.end() :].split("</table>", 1)[0]


def test_an_unbroken_expression_in_the_pinned_column_wraps_inside_it_on_the_kairo_contrasts() -> None:
    """Kairo's contrasts pin a reading named by its expression (``call_count("discord.send_message") >= 1 or …``),
    a run longer than the column's cap holds on one line. The serializer classes that cell ``unbroken``, which the
    pinned column breaks anywhere, and nothing inside the cell (its code, the expression's fold, its summary) holds it
    whole, so it never runs over the column beside it. A reading of words (``outcome (judged)``) is not classed: it
    wraps at its spaces."""
    page = _kairo_page()
    _frame, table = _framed(page, "comparisons")
    firsts = [
        (css, html.unescape(re.sub(r"<[^>]+>", "", cell)))
        for css, cell in re.findall(r'<tr[^>]*><td(?: class="([^"]*)")?[^>]*>(.*?)</td>', table)
    ]
    unbroken = max((run for _css, text in firsts for run in text.split()), key=len)
    assert 'call_count("discord.send_message")' in unbroken, "the control: the pinned column holds the expression"
    classed = {text: "unbroken" in css.split() for css, text in firsts}
    assert all(classed[text] for text in classed if unbroken in text), "the expression's cell is id-like"
    assert not classed["outcome (judged)"], "a reading of words wraps at its spaces"
    rules = _rules(_style(page))
    (first,) = [rule for rule in rules if rule.selector == ".tbl :is(th,td):first-child"]
    pinned = first.declarations()
    assert _cell_px(unbroken, header=False) > _px(pinned["max-width"]), "it is wider than the cap"
    (anywhere,) = [rule for rule in rules if ".unbroken" in rule.selector]
    assert anywhere.declarations()["overflow-wrap"] == "anywhere", "break-word leaves the run's min-content whole"
    inside = [rule for rule in rules if re.search(r"\b(code|summary|expr)\b", rule.selector)]
    assert inside, "the control: the page styles code and the expression's fold"
    for rule in inside:
        declared = rule.declarations()
        assert declared.get("overflow-wrap", "anywhere") == "anywhere", rule
        assert declared.get("white-space", "normal") not in ("nowrap", "pre"), rule


@pytest.mark.parametrize("name", ["doc", "doc static"])
def test_a_chart_s_values_table_wraps_its_cells_of_words(name: str) -> None:
    page = _served()[name]
    tables = re.findall(r'<div class="tbl chart-values">(.*?)</div>', page, flags=re.S)
    assert tables, "the control: the page's charts carry their values tables"
    cells = [match for table in tables for match in re.findall(r"<td( class=\"wrap\")?>([^<]*)</td>", table)]
    long = [(css, text) for css, text in cells if len(html.unescape(text)) >= 40]
    assert long, "the control: some value cell is a long label"
    assert all(css for css, _text in long), [text for css, text in long if not css][:3]


def _cell_px(text: str, *, header: bool) -> float:
    """A generous width for a cell holding ``text`` on one line: the kit's table text (13 px, headers 12 px) at about
    0.6 em a character, and its padding either side."""
    return len(text) * (7.2 if header else 7.8) + 2 * 11.2


def test_to_settle_leads_with_the_arm_and_both_counts_which_show_unscrolled_at_688_px() -> None:
    page = _settle_page()
    (table,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle"]
    assert [column.key for column in table.columns][:3] == ["arm", "none", "observed"]
    assert table.columns[-1].key == "why", "why it is unclear comes last, in words that wrap"
    lead = 0.0
    for column in table.columns[:3]:
        widest = max(_cell_px(row[column.key].text, header=False) for row in table.rows)
        lead += max(widest, _cell_px(column.header.split()[0], header=True))
    assert lead <= FRAME_PX - GUTTERS_PX, f"the arm and both counts take {lead:.0f} px"
    html = _served()["campaign"]
    rows = html.split('data-table="to_settle"', 1)[1].split("</table>", 1)[0]
    for row in table.rows:
        if len(row["why"].text) >= 40:
            assert f'<td class="wrap">{row["why"].text[:20]}' in rows.replace("&#x27;", "'"), row["why"].text


@pytest.mark.parametrize("name", DOCS)
def test_every_arm_by_measure_is_laid_out_the_way_that_gives_it_fewer_columns(name: str) -> None:
    """On the page (HTML and Markdown) a figure per row and an arm per column where the arms are fewer than the
    figures, else an arm per row as built; every figure the table holds is in the layout as it was. The text the
    editor and a writer eval's judge read keeps the table as built, so a recorded judge's replay reads what it was
    recorded on."""
    page = _doc_page(name)
    (table,) = [one for one in page.full_results if one.name == "arms_by_measure"]
    grouped = any(column.key == "aspect" for column in table.columns)
    arms = len({row["arm"].text for row in table.rows})
    figures = len([column for column in table.columns if column.key not in ("arm", "aspect")])
    shown = laid_out(table)
    assert len(shown.columns) == min(arms, figures) + 1 + grouped, (name, arms, figures)
    if arms >= figures:
        assert shown == table, "as many arms as figures or more: an arm per row, as built"
        header = "Arm"
    else:
        keys = [column.key for column in shown.columns]
        assert keys == [*(["aspect"] if grouped else []), "measure", *(f"arm:{at}" for at in range(arms))]
        assert {str(row["measure"].value) for row in shown.rows} == {
            column.header for column in table.columns if column.key not in ("arm", "aspect")
        }
        for row in table.rows:
            for column in table.columns:
                if column.key not in ("arm", "aspect") and row[column.key].value is not None:
                    assert any(row[column.key] in printed.values() for printed in shown.rows), column.header
        header = "Measure"
    written = doc_page_html(page).split('data-table="arms_by_measure"', 1)[1].split("</tr>", 1)[0]
    assert re.findall(r"<th[^>]*>([^<]+)", written)[int(grouped and header == "Measure")] == header
    markdown = doc_page_markdown(page).split("Every arm by measure", 1)[1].split("\n\n", 2)[1]
    assert f"| {header} |" in markdown, "the Markdown lays it out as the HTML does"
    text = doc_page_text(page).split("## Every arm by measure", 1)[1]
    assert text.split("\n")[1].startswith("| Arm |"), "the editor's and judge's text keeps the table as built"


def test_both_layouts_occur_a_few_arms_turned_and_a_sweep_as_built() -> None:
    """Kairo (three arms, many figures) is turned; r2's eighteen-arm sweep stays an arm per row."""

    def shown(name: str) -> TableBlock:
        (table,) = [one for one in _doc_page(name).full_results if one.name == "arms_by_measure"]
        return laid_out(table)

    assert shown("prototype3/kairo").columns[0].key == "measure"
    sweep = shown("prototype3/r2")
    assert sweep.columns[0].key == "arm" and len(sweep.rows) == 18


#: A chart query's every field at its plain value.
_EMPTY_QUERY: dict[str, Any] = {
    "aspect": "",
    "measures": [],
    "arms": [],
    "strata": [],
    "limit": 0,
    "transform": "none",
    "small_multiples_by": "none",
}


def _settled() -> Iterator[tuple[str, ReportInput, str]]:
    for name, held in fixture_inputs().items():
        for arm in held.arms:
            try:
                if settle_cases(held, [arm.id]).rows:
                    yield name, held, arm.id
            except RefusedInput:
                continue


def test_a_rerun_table_holds_four_columns_and_warns_off_none_of_them() -> None:
    seen = 0
    for name, held, arm in _settled():
        query = ChartQuery.model_validate({**_EMPTY_QUERY, "source": "settle", "arms": [arm]})
        block = block_render(held, block_kind="table", query=query)
        assert 2 < len(block.columns) <= 4, (name, block.columns)
        assert block.columns[:3] == ["Arm", "Against", "Settles it 4 runs in 5"], (name, block.columns)
        for note in block.notes:
            assert '"' not in note and "tests nothing" not in note and "one-sided" not in note, (name, note)
        seen += 1
    assert seen, "the control: some fixture settles an arm"


# =============================================================================
# By stratum
# =============================================================================

_SCOPE = "fit-1"
_PROFILE = HostProfile(host_id="fit", host_sweepables=SHARED_CORE, measures=MeasureRegistry(()))

#: Strata of one or two cases each, as the production campaign held them: none readable alone.
_STRATA = {"about": 1, "machinery": 2, "off_lane": 2, "pushback": 1, "small_talk": 1, "emotional": 1}


def _result(model: str, case: str, *, matched: bool) -> EvalResult:
    return make_eval_result(
        scope_id=_SCOPE,
        eval_run_id=f"run-{model}",
        test_case_id=case,
        model=model,
        k_iteration=1,
        variant_key=fixture_variant_key(model),
        host_measures={"match": matched},
        goal_state_outcomes=[],
    )


@cache
def _thin_strata_report() -> Report:
    """Two arms over cases whose every stratum holds one or two, each case run once."""
    storage = EvalStorage(InMemoryDocumentStore())
    cases = [
        EvalTestCase(id=f"{stratum}-{index}", scope_id=_SCOPE, template_id="tpl-1", stratum=stratum)
        for stratum, count in _STRATA.items()
        for index in range(count)
    ]
    for case in cases:
        storage.save_test_case(case)
    run_ids = []
    for model in ("sonnet", "haiku"):
        run = make_eval_run(
            id=f"run-{model}",
            scope_id=_SCOPE,
            candidate_model=model,
            k_runs=1,
            test_case_ids=[case.id for case in cases],
            status="completed",
        )
        storage.save_eval_run(run)
        for at, case in enumerate(cases):
            storage.save_eval_result(_result(model, case.id, matched=model == "sonnet" or at % 2 == 0))
        run_ids.append(run.id)
    campaign = make_campaign(scope_id=_SCOPE, run_ids=run_ids)
    storage.save_campaign(campaign)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=_PROFILE)
    return build_code_only_report(bundle, assembled_at="2026-11-08T00:00:00+00:00")


def _strata_table(report: Report) -> TableBlock:
    (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "strata"]
    return table


def test_by_stratum_states_a_one_case_cell_as_its_count_and_never_one_trials() -> None:
    report = _thin_strata_report()
    table = _strata_table(report)
    cells = [cell.text for row in table.rows for cell in row.values()]
    case_rows = [row for row in table.rows if row["reading"].text == "cases"]
    assert case_rows, "the control: each arm opens with its cases"
    counts = {cell.text for row in case_rows for key, cell in row.items() if key.startswith("stratum_")}
    assert counts == {"1 case", "2 cases, 2 trials"}, counts
    assert not any(re.search(r"\b1 (trials|cases|results)\b", text) for text in cells)
    for form in (report_markdown(report), report_html(report)):
        assert not re.search(r"\b1 trials\b", form)


def test_strata_all_too_small_to_read_alone_say_so_in_one_line_and_no_cell_repeats_it() -> None:
    report = _thin_strata_report()
    table = _strata_table(report)
    assert table.details, "the table by stratum is folded"
    assert not any(TOO_FEW_CASES in cell.text for row in table.rows for cell in row.values())
    (line,) = [block for block in report.blocks if isinstance(block, DisclosureBlock) and block.source == "strata"]
    assert line.text == f"Every stratum holds 1 to 2 cases; a stratum is read alone from {STRATUM_MIN_CASES} cases."
    assert report.blocks.index(line) < report.blocks.index(table), "the line comes before the table it reads"


def test_strata_past_the_column_budget_are_rows_and_the_arms_columns_on_the_kairo_capture() -> None:
    """Kairo's code page reads 24 strata: a stratum per row, every arm a column, so the table is as wide as its arms,
    not its strata, on every surface that prints the block."""
    bundle = kairo_bundle()
    report = build_code_only_report(bundle.aspects[0], assembled_at="2026-11-08T00:00:00+00:00")
    table = _strata_table(report)
    arms = [column for column in table.columns if column.key.startswith("arm_")]
    assert [column.key for column in table.columns[:2]] == ["stratum", "reading"]
    assert len(table.columns) == len(arms) + 2 and 2 <= len(arms) <= 4
    strata = list(dict.fromkeys(row["stratum"].text for row in table.rows))
    assert strata[0] == "All cases" and len(strata) > 7, strata[:5]
    assert any(column.header.startswith("▸ ") for column in arms), "the control's column is marked"
    markdown = report_markdown(report).split("By stratum", 1)[1]
    assert "| Stratum | Reading |" in markdown


def test_a_table_told_apart_by_two_columns_keeps_both_in_view_on_the_kairo_capture() -> None:
    """Kairo's By stratum (a stratum per row, each reading under it) and its contrasts (a reading's row on each arm)
    tell a row apart by two columns, so both stay in view as the table scrolls: the first held at exactly its cap, the
    second pinned where the first ends. A table told apart by one keeps one: two arms' contrasts, the statistics."""
    page = _kairo_page()
    for name, keys in (("strata", ["Stratum", "Reading"]), ("comparisons", ["Reading", "Arm"])):
        frame, table = _framed(page, name)
        assert frame == "tbl keys-2", (name, frame)
        assert re.findall(r"<th[^>]*>(?:<a[^>]*>)?([^<]+)", table)[:2] == keys, name
    assert _framed(page, "comparison_statistics")[0] == "tbl"
    two_arms, table = _framed(_served()["aspect"], "comparisons")
    assert re.findall(r"<th[^>]*>(?:<a[^>]*>)?([^<]+)", table)[1] != "Arm", "the control: two arms, no Arm column"
    assert two_arms == "tbl", "two arms' contrasts are told apart by the reading alone"
    rules = _rules(_style(page))
    (first,) = [rule for rule in rules if rule.selector == ".tbl :is(th,td):first-child"]
    cap = first.declarations()["max-width"]
    (held,) = [rule for rule in rules if rule.selector == ".tbl.keys-2 :is(th,td):first-child"]
    assert held.declarations()["width"] == held.declarations()["min-width"] == cap, "the first column is its cap"
    (second,) = [rule for rule in rules if rule.selector == ".tbl.keys-2 :is(th,td):nth-child(2)"]
    pinned = second.declarations()
    # The second pins where the first ends: the first's width, its floor and its cap are one value.
    assert (pinned["position"], pinned["left"]) == ("sticky", cap)
    assert pinned["background"].startswith("var("), "a pinned column is opaque, or the scrolled cells show through"
    # It wraps at its words between a floor and its cap, as the first column does.
    assert (pinned["overflow-wrap"], pinned["word-break"], pinned["hyphens"]) == ("normal", "normal", "manual")
    assert 6 * 16 <= _px(pinned["min-width"]) <= _px(pinned["max-width"]), pinned
    # Both pinned columns together leave most of a 688 px frame to the columns that scroll.
    widths = [re.fullmatch(r"min\((\d+)rem,\d+vw\)", value) for value in (cap, pinned["max-width"])]
    assert all(widths) and sum(int(match.group(1)) * 16 for match in widths if match) <= 0.6 * FRAME_PX


# =============================================================================
# A written report's masthead, its cards and its evidence record
# =============================================================================


def test_a_masthead_count_of_one_reads_singular() -> None:
    """Kairo's page with one candidate left reads ``vs 1 candidate``; with two, ``vs 2 candidates``."""
    name = "prototype3/kairo"
    held = _input(name)
    assert re.search(r"\bvs 2 candidates\b", _doc_page(name).masthead.counts)
    dropped = next(arm.id for arm in held.arms if arm.role == "challenger")
    data: dict[str, Any] = held.model_dump(mode="json", by_alias=True, exclude_none=True)
    data["arms"] = [arm for arm in data["arms"] if arm["id"] != dropped]
    doc = ReportDoc.model_validate({**converted_doc(name), "explore": []})
    counts = build(doc, ReportInput.model_validate(data), strict=False).masthead.counts
    assert re.search(r"\bvs 1 candidate\b(?! *s)", counts) and "1 candidates" not in counts, counts
    for each in DOCS:
        assert not re.search(r"\b1 (candidates|repeats|cases|tasks|arms|aspects)\b", _doc_page(each).masthead.counts)


def test_the_card_grid_lets_its_last_row_fill_so_no_cell_is_left_empty() -> None:
    rules = _rules(_style(_served()["doc"]))
    (grid,) = [rule for rule in rules if rule.selector == ".decision"]
    (card,) = [rule for rule in rules if rule.selector == ".decision>div"]
    assert grid.declarations()["display"] == "flex" and grid.declarations()["flex-wrap"] == "wrap"
    assert "grid-template-columns" not in grid.declarations(), "a grid leaves an empty cell beside a short last row"
    grow, _shrink, basis = card.declarations()["flex"].split()
    assert float(grow) > 0 and basis.endswith("rem"), "each card grows to fill its row"
    assert '<div class="decision" data-part="decision">' in _served()["doc"]


def test_a_written_report_s_evidence_record_lays_its_tables_out_as_the_campaign_page_does() -> None:
    page = _served()["doc with its record"]
    record = page.split('data-part="evidence"', 1)[1] if 'data-part="evidence"' in page else page
    assert "<table" in record, "the control: the record holds the campaign's tables"
    assert record.count("<table") == len(re.findall(r'<div class="tbl(?: [\w-]+)*"><table', record))
    # The aspects' contrasts are in it, framed as on the campaign page: the reading first, which stays in view.
    assert 'data-table="comparisons"' in record
    contrasts = record.split('data-table="comparisons"', 1)[1].split("</tr>", 1)[0]
    assert re.findall(r"<th[^>]*>(?:<a[^>]*>)?([^<]+)", contrasts)[0] == "Reading"


# =============================================================================
# Drawn: pinned columns hold their floor, the nav paints over the tables, headings fit a host's frame
# =============================================================================

#: The two frames a host gives the page: 688 px at a 1024 px window, 1104 px at 1440.
FRAMES = (688, 1104)

#: Each pinned cell of the named tables, drawn: its text, its width, its floor (computed min-width), whether it is
#: id-like, and each word of it (split at its spaces, a hyphen or a slash kept on the word before) the browser drew
#: across more than one line.
_PINNED_PROBE = """(names) => {
  const out = [];
  for (const name of names) {
    const block = document.querySelector(`[data-table="${name}"]`);
    if (!block) { out.push({table: name, missing: true}); continue; }
    for (let e = block; e; e = e.parentElement) if (e.tagName === 'DETAILS') e.open = true;
    const frame = block.querySelector('.tbl');
    const keys = frame.classList.contains('keys-2') ? 2 : 1;
    for (const row of frame.querySelectorAll('tr')) {
      for (const cell of [...row.children].slice(0, keys)) {
        const style = getComputedStyle(cell), broken = [];
        const walk = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
        for (let node = walk.nextNode(); node; node = walk.nextNode())
          for (const word of node.data.matchAll(/[^\\s\\-\\/]+[\\-\\/]?/g)) {
            const range = document.createRange();
            range.setStart(node, word.index); range.setEnd(node, word.index + word[0].length);
            const tops = new Set([...range.getClientRects()].filter(r => r.width).map(r => Math.round(r.top)));
            if (tops.size > 1) broken.push(word[0]);
          }
        out.push({table: name, text: cell.textContent.trim(), width: cell.getBoundingClientRect().width,
                  floor: parseFloat(style.minWidth), sticky: style.position === 'sticky',
                  unbroken: cell.classList.contains('unbroken'), broken});
      }
    }
  }
  return out;
}"""


def _browser() -> None:
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")


@cache
def _kairo_campaign_html() -> str:
    """Kairo's campaign page, code-only: its scorecard pins the aspect's name."""
    return campaign_report_html(kairo_page())


@pytest.mark.integration
@pytest.mark.parametrize("width", FRAMES)
def test_a_pinned_column_holds_its_floor_and_wraps_only_at_its_words_on_kairo_s_contrasts_and_scorecard(
    width: int, tmp_path: Path
) -> None:
    """Production, 0.93, in the host's frame: the Contrasts' "Reading" drew ``Rea/din/g``, the statistics'
    "Question" ``Q/u/e/s/ti/o/n``, and a code-only scorecard's aspect name one letter a line at 1104 px as well as
    688, while the table measured as fitting its frame. Drawn here at both frames: every pinned cell is at least its
    floor wide, and no word in one breaks across lines unless the cell is id-like."""
    _browser()
    pages = {
        "aspect": (_kairo_page(), ("comparisons", "comparison_statistics")),
        "campaign": (_kairo_campaign_html(), ("scorecard",)),
    }
    for name, (page, tables) in pages.items():
        path = tmp_path / f"{name}.html"
        path.write_text(page, encoding="utf-8")
        cells, errors = rendered(path, f"() => ({_PINNED_PROBE})({json.dumps(list(tables))})", width=width)
        assert not errors, errors
        assert not [cell for cell in cells if cell.get("missing")], cells
        assert all(cell["sticky"] for cell in cells), "the control: every key column is pinned"
        texts = {cell["text"] for cell in cells}
        assert {"Reading", "Question"} <= texts or name == "campaign", texts
        assert "character adherence (v2)" in texts or name == "aspect", "the control: the scorecard pins the aspect"
        for cell in cells:
            assert cell["width"] >= cell["floor"] - 0.5 and cell["floor"] >= 96, (name, width, cell)
            assert cell["unbroken"] or not cell["broken"], (name, width, cell)
        assert any(cell["unbroken"] for cell in cells) or name == "campaign", "the control: an expression is pinned"


@pytest.mark.integration
@pytest.mark.parametrize("width", FRAMES)
def test_the_section_nav_paints_over_a_pinned_cell_scrolled_beneath_it(width: int, tmp_path: Path) -> None:
    """Production, 0.93: the code-only page's sticky nav drew under the scorecard's pinned Aspect column ("…ecard").
    Each table frame isolates its cells' stacking, so the nav, above the frame, paints over every cell beneath it."""
    _browser()
    path = tmp_path / "campaign.html"
    path.write_text(_kairo_campaign_html(), encoding="utf-8")
    probe = """() => {
      const nav = document.querySelector('nav'), hits = [];
      if (getComputedStyle(nav).position !== 'sticky') return null;
      for (const cell of document.querySelectorAll('.tbl thead tr > :first-child')) {
        if (!cell.offsetParent) continue;
        window.scrollTo(0, scrollY + cell.getBoundingClientRect().top - nav.getBoundingClientRect().height / 2);
        const box = nav.getBoundingClientRect(), r = cell.getBoundingClientRect();
        // a table near the page's end cannot be scrolled up to the nav
        if (r.top >= box.bottom || r.bottom <= box.top) continue;
        const y = (Math.max(box.top, r.top) + Math.min(box.bottom, r.bottom)) / 2;
        const hit = document.elementFromPoint(r.left + 6, y);
        hits.push({cell: cell.textContent.trim(), onNav: !!hit && nav.contains(hit)});
      }
      return hits;
    }"""
    hits, errors = rendered(path, probe, width=width)
    assert not errors, errors
    assert hits is not None and len(hits) >= 5, f"the control: pinned header cells scrolled beneath the nav: {hits}"
    assert all(hit["onNav"] for hit in hits), [hit["cell"] for hit in hits if not hit["onNav"]]


def test_the_nav_stacks_above_every_table_frame_s_pinned_cells() -> None:
    """Read as rules, for the pool (no browser): a table frame isolates its stacking, so a pinned cell's z-index
    ranks it only within its own frame, and the nav is positioned above the page's other content."""
    rules = _rules(_style(_served()["campaign"]))
    (frame,) = [rule for rule in rules if rule.selector == ".tbl" and not rule.at]
    assert frame.declarations()["isolation"] == "isolate"
    (nav,) = [rule for rule in rules if rule.selector == "nav" and not rule.at]
    assert nav.declarations()["position"] == "sticky" and int(nav.declarations()["z-index"]) >= 1


def test_a_frame_under_700_px_sets_its_headings_a_step_smaller() -> None:
    """Production, 0.93: the written report's headline ran five lines at 688 px, in the 36 px hero size. Under 700 px
    the headings are a step smaller and tighter."""
    rules = _rules(_style(_served()["doc"]))
    (h1,) = [rule for rule in rules if rule.selector == "h1" and not rule.at]
    assert h1.declarations()["font-size"] == "var(--t-hero)", "the control: a wide frame's headline is the hero size"
    small = {rule.selector: rule.declarations() for rule in rules if rule.at == "@media (max-width:700px)"}
    assert set(small) >= {"h1", "h2", "h3"}, small
    assert float(small["h1"]["font-size"].removesuffix("rem")) <= 1.5 and float(small["h1"]["line-height"]) <= 1.25
    assert float(small["h2"]["font-size"].removesuffix("rem")) <= 1.25


@pytest.mark.integration
def test_the_written_headline_takes_a_third_fewer_lines_at_688_px(tmp_path: Path) -> None:
    """Kairo's 148-character headline, drawn in a host's 688 px frame: four lines at 24 px, where the hero size took
    six."""
    _browser()
    path = tmp_path / "doc.html"
    path.write_text(_served()["doc"], encoding="utf-8")
    probe = """() => { const h = document.querySelector('h1'), s = getComputedStyle(h), size = s.fontSize;
      const lines = () => Math.round(h.getBoundingClientRect().height / parseFloat(getComputedStyle(h).lineHeight));
      const drawn = lines(); h.style.fontSize = 'var(--t-hero)'; h.style.lineHeight = '1.12';
      return {size, drawn, hero: lines()}; }"""
    drawn, errors = rendered(path, probe, width=FRAME_PX)
    assert not errors, errors
    assert drawn["size"] == "24px" and drawn["drawn"] <= 4 and drawn["drawn"] <= drawn["hero"] * 2 / 3, drawn


@pytest.mark.integration
def test_the_render_check_fails_a_table_that_fits_by_crushing_its_key_column(tmp_path: Path) -> None:
    """``scripts/check.py`` passes Kairo's pages, and fails the same campaign page with its pinned column let shrink
    to nothing (the 0.93 stylesheet's ``min-width:0; overflow-wrap:anywhere``): a table "fits" only where no pinned
    cell is under its floor and no word in one breaks but an id-like cell's."""
    _browser()
    good = tmp_path / "good.html"
    good.write_text(_kairo_campaign_html(), encoding="utf-8")
    checked, runs = render_checked(good, tmp_path / "shots")
    assert checked.returncode == 0 and len(runs) == 6, checked.stdout + checked.stderr
    crushed = tmp_path / "crushed.html"
    crushed.write_text(
        _kairo_campaign_html().replace(
            "</style>", ".tbl :is(th,td):first-child{min-width:0!important;overflow-wrap:anywhere!important}</style>", 1
        ),
        encoding="utf-8",
    )
    checked, runs = render_checked(crushed, tmp_path / "shots")
    assert checked.returncode == 1 and "crushed:" in checked.stdout, checked.stdout
