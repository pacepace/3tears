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
- **A key column is as wide as its text, and an id is one word.** A two-key table's first column is its text's
  width between the floor and the cap, so a short key ("All cases") leaves the frame to the columns past it. A
  hyphenated id in a pinned cell (``deepseek-v4.1-flash``, a case id) never breaks after its hyphen while another column
  has room: where the table scrolls anyway, or where the columns past it can give way; one too long for the cap, or
  for a phone's, still wraps after its hyphens, inside the frame.
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
from threetears.evals.analysis.report.html_parts import table_html
from threetears.evals.analysis.report.model import CampaignReport, ReportBlock, TableCell, TableColumn
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
from packages.evals.tests.report_kit_support import chromium_unavailable, render_checked, rendered, report_page
from packages.evals.tests.ladder_support import load, offline_copy
from packages.evals.tests.report_shapes_support import CORRECT_BAR, ArmTrials, AspectTrials, s3_measurement, shape_of

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
    "--muted": ("--host-color-muted", "#5f6774", "#8a93a1"),
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
    framed = len(re.findall(_FRAME, page))
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


def test_a_frame_fades_its_right_edge_over_every_cell_while_more_of_its_table_lies_past_it() -> None:
    """Discodon's 0.94.0 pass: the Arms table at 688 px and a code-only scorecard at 688 and 1104 px were cut at their
    frame's edge with no cue, since the frame's shadow is painted under its cells and a header's or a highlighted row's
    background hides it. The frame's own scroll position drives a fade of its right edge, a mask over every cell: none
    on a frame that fits (its timeline is inactive, so no frame of the fade applies), the fade until the last of the
    scroll, none at its end. The shadow stays under the cells for a browser without scroll-driven animations."""
    rules = _rules(_style(_served()["doc"]))
    (frame,) = [rule for rule in rules if rule.selector == ".tbl" and not rule.at]
    declared = frame.declarations()
    assert not [name for name in declared if name.startswith(("mask", "-webkit-mask"))], "a frame that fits has no cue"
    assert declared["animation-timeline"] == "scroll(self inline)", "the frame's own scroll drives it, no script"
    name = declared["animation"].split()[0]
    assert {"linear", "both"} <= set(declared["animation"].split()[1:])
    steps = {rule.selector: rule.declarations() for rule in rules if rule.at == f"@keyframes {name}"}
    assert steps, "the control: the fade's keyframes are on the page"
    (end,) = [said for selector, said in steps.items() if "100%" in selector.split(",")]
    assert end == {"mask-image": "none"}, "scrolled to its end, the frame shows all of its table"
    before = [(selector, said) for selector, said in steps.items() if "100%" not in selector.split(",")]
    assert "from" in ",".join(selector for selector, _ in before).split(","), "from the start of the scroll"
    for _, said in before:
        fade = said["mask-image"]
        assert fade.startswith("linear-gradient(to left,transparent"), "the right edge fades out, over every cell"


def test_a_cell_of_words_wraps_at_its_spaces_and_a_number_or_an_id_holds_its_line() -> None:
    """Fit first: a table narrows to its frame by wrapping its cells of words at their spaces before it scrolls (the
    code-only scorecard and every written-report table fit a host's 1104 px frame so). A number column, an id and a
    pill hold their line; a hyphenated word is held whole in every cell, not only a pinned one; inside a frame that
    scrolls anyway a cell keeps its line, as wrapping there would only crush its column, but for prose and a glyph's
    figure. An interval glyph gives way to the frame between a floor and its full width."""
    rules = _rules(_style(_served()["campaign"]))
    by = {(rule.at, rule.selector): rule.declarations() for rule in rules}
    assert by[("", "th,td")].get("white-space", "normal") == "normal", "a cell of words wraps"
    assert by[("", "td.num")] == {"white-space": "nowrap"}, "a number column holds its line"
    assert by[("", "td.r")] == {"white-space": "nowrap"}
    assert by[("", ".pill")]["white-space"] == "nowrap" and by[("", ".tbl .id")] == {"white-space": "nowrap"}
    scrolling = by[("@container scroll-state(scrollable:x)", ".tbl td:not(.wrap,.glyph,:first-child)")]
    assert scrolling == {"white-space": "nowrap"}
    assert "white-space" not in by[("", "td.glyph")], "a glyph's figure wraps under its glyph"
    glyph = by[("", "svg.viz-glyph")]
    assert (glyph["width"], glyph["max-width"], glyph["min-width"]) == ("7.5em", "100%", "4.5em")
    for (_, selector), said in by.items():
        if re.search(r"(^|[\s,])(td|th)\b", selector) and "unbroken" not in selector:
            assert said.get("overflow-wrap", "normal") == "normal", selector
            assert said.get("word-break", "normal") == "normal", selector
    held = table_html(
        TableBlock(
            section="contrasts",
            name="held",
            title="Held",
            columns=[TableColumn(key="key", header="Stratum"), TableColumn(key="reading", header="Reading")],
            rows=[{"key": TableCell(value="All cases"), "reading": TableCell(value="Failed-reply notices")}],
            order="as read",
            total_rows=1,
        )
    )
    assert '<td><span class="id">Failed-reply</span> notices</td>' in held, held


def test_a_row_s_label_heads_its_row_and_reads_as_a_body_cell() -> None:
    """A turned Every arm by measure heads each row with its figure (``<th scope="row">``), which a check's header
    link reaches; it reads as the body cell it was: the header row's look is the header row's alone."""
    page = _served()["doc"]
    rules = _rules(_style(page))
    by = {(rule.at, rule.selector): rule.declarations() for rule in rules}
    body = by[("", "tbody th")]
    assert body == {
        "font-weight": "inherit",
        "color": "inherit",
        "font-size": "inherit",
        "text-transform": "none",
        "letter-spacing": "normal",
        "background": "none",
        "min-width": "0",
    }
    # a pinned header paints its frame's ground, as its column's body cells do, the header row's look its rule's alone
    assert ("", ".tbl th:first-child") not in by and by[("", ".tbl thead th:first-child")][
        "background"
    ] == "var(--ground)"
    assert by[("", ".tbl.keys-2 thead th:nth-child(2)")]["background"] == "var(--ground)"
    turned = page.split('data-table="arms_by_measure"', 1)[1].split("</table>", 1)[0]
    assert '<th scope="row"' in turned and "<td" in turned, "the control: the page's turned table heads its rows"


def test_a_header_row_sets_its_labels_from_the_top_and_a_long_label_takes_two_lines_where_its_frame_scrolls() -> None:
    """Discodon's re-shoot of host pass 5: the written report's Arms table at 688 px drew a header row about 90 px tall,
    its short labels floating in the middle, set by "Share of output spent reasoning" wrapped in five lines at the
    header's floor in a column past the frame's edge, though the table scrolls anyway. A header row sets every label
    from the top; a label the floor would set in more than two lines carries the width that sets it in two
    (``--head``), which the frame holds it at only when it scrolls anyway; a short label and a pinned one carry none."""
    rules = _rules(_style(_served()["doc"]))
    by = {(rule.at, rule.selector): rule.declarations() for rule in rules}
    assert by[("", "thead th")] == {"vertical-align": "top", "border-bottom": "1px solid var(--ink-2)"}
    assert by[("@container scroll-state(scrollable:x)", ".tbl thead th")] == {"min-width": "var(--head,4.5rem)"}
    arms = _served()["doc"].split('data-table="arms_key"', 1)[1].split("</thead>", 1)[0]
    widths = {label: width for width, label in _HEAD.findall(arms)}
    assert widths["Share of output spent reasoning"], arms
    assert 9 < float(widths["Share of output spent reasoning"]) < 11, "the width of two lines, not of one"
    assert widths["Arm"] == widths["Role"] == widths["Served model"] == "", "pinned, or two lines at the floor"


#: A column header as a table writes it: the width it carries (``--head``), empty for none, and its label.
_HEAD = re.compile(r'<th scope="col"[^>]*?(?: style="--head:([\d.]+)rem")?>([^<]*)</th>')


#: A table's scrolling frame: its classes, and the widths its pinned columns are laid out at, when it sets any.
_FRAME = re.compile(r'<div class="(tbl(?: [\w-]+)*)"(?: style="([^"]*)")?><table')

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


#: Three more arms for Kairo's cases table, hyphenated as a model id is.
_MORE_ARMS = ("deepseek-v4.1-flash", "gemini-3.8-flash", "glm-5.1-air")


@cache
def _kairo_six_arm_page() -> str:
    """Kairo's code page with its cases table as six arms draw it: each case's marks on the three arms again under
    three more. Three arms' marks wrap to fit a host's 1104 px frame; six do not, so the table scrolls at both of a
    host's frames with its case ids pinned."""
    report = build_code_only_report(kairo_bundle().aspects[0], assembled_at="2026-11-08T00:00:00+00:00")
    blocks: list[ReportBlock] = []
    for block in report.blocks:
        if isinstance(block, TableBlock) and block.name == "cases":
            arms = [column for column in block.columns if column.key.startswith("arm_")]
            more = [TableColumn(key=f"arm_{len(arms) + at}", header=arm) for at, arm in enumerate(_MORE_ARMS)]
            block = block.model_copy(
                update={
                    "columns": [*block.columns[:-1], *more, block.columns[-1]],
                    "rows": [
                        {**row, **{added.key: row[arm.key] for arm, added in zip(arms, more, strict=True)}}
                        for row in block.rows
                    ],
                }
            )
        blocks.append(block)
    return report_html(report.model_copy(update={"blocks": blocks}))


def _framed(page: str, name: str) -> tuple[str, str]:
    """Table ``name``'s frame's classes on ``page`` and the table's markup."""
    held = page.split(f'data-table="{name}"', 1)[1]
    frame = re.search(_FRAME, held)
    assert frame is not None, f"{name} sits in no frame"
    return frame.group(1), held[frame.end() :].split("</table>", 1)[0]


def _frame_style(page: str, name: str) -> dict[str, str]:
    """The widths table ``name``'s frame lays its pinned columns out at (its ``style``): each custom property's value."""
    held = page.split(f'data-table="{name}"', 1)[1]
    frame = re.search(_FRAME, held)
    assert frame is not None, f"{name} sits in no frame"
    return dict(part.split(":", 1) for part in (frame.group(2) or "").split(";") if part)


#: An id-like run longer than a pinned column's cap holds on one line: a goal check's expression, as a table that pins a
#: check's own text (a trial's record, a host's table) prints it.
_LONG_RUN = 'call_count("discord.send_message")>=1_or_call_count("music.queue_voice")>=1'


def _probe() -> TableBlock:
    """A table pinning :data:`_LONG_RUN` beside a reading of words."""
    return TableBlock(
        section="contrasts",
        name="probe",
        title="Probe",
        columns=[TableColumn(key="reading", header="Reading"), TableColumn(key="value", header="Value")],
        rows=[
            {"reading": TableCell(value=f"`{_LONG_RUN}`"), "value": TableCell(value="1")},
            {"reading": TableCell(value="outcome (judged)"), "value": TableCell(value="2")},
        ],
        order="as written",
        total_rows=2,
    )


def test_an_unbroken_expression_in_the_pinned_column_wraps_inside_it_on_the_kairo_contrasts() -> None:
    """Kairo's contrasts pin each reading by its name: its goal check reads ``Send message or queue voice check`` (the owner's
    ruling of 2026-11-07, two subjects joined by its ``or``; its expression is the glossary's), which wraps at its spaces as ``Outcome (judged, 1–5)`` does, so neither is
    classed. An id-like run longer than the column's cap holds on one line (:data:`_LONG_RUN`) is classed
    ``unbroken``, which the pinned column breaks anywhere, and nothing inside the cell (its code, an expression's fold,
    its summary) holds it whole, so it never runs over the column beside it."""
    page = _kairo_page()
    _frame, table = _framed(page, "comparisons")
    firsts = [
        (css, html.unescape(re.sub(r"<[^>]+>", "", cell)))
        for css, cell in re.findall(r'<tr[^>]*><td(?: class="([^"]*)")?[^>]*>(.*?)</td>', table)
    ]
    classed = {text: "unbroken" in css.split() for css, text in firsts}
    assert "Send message or queue voice check" in classed, "the control: the pinned column names the check"
    assert not [text for text in classed if "call_count(" in text], "no pinned cell prints the expression"
    assert not classed["Send message or queue voice check"] and not classed["Outcome (judged, 1–5)"], (
        "a reading of words wraps at its spaces"
    )
    cells = re.findall(r'<tr[^>]*><td(?: class="([^"]*)")?[^>]*>(.*?)</td>', table_html(_probe()))
    (shown,) = [html.unescape(re.sub(r"<[^>]+>", "", cell)) for css, cell in cells if "unbroken" in css.split()]
    assert _LONG_RUN in shown, "the id-like run's cell, and only it, is classed"
    unbroken = _LONG_RUN
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
    tell a row apart by two columns, so both stay in view as the table scrolls: the first held at exactly its width,
    its text's between the floor and the cap (``--key`` on the frame), the second pinned where the first ends. A table
    told apart by one keeps one: two arms' contrasts, the statistics."""
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
    span = _SPAN.fullmatch(cap.replace(" ", ""))
    assert span is not None
    (held,) = [rule for rule in rules if rule.selector == ".tbl.keys-2 :is(th,td):first-child"]
    # The frame's width, the cap without one, and never past the cap's share of the frame.
    exact = f"min(var(--key,{span.group(1)}rem),{span.group(2)}vw)"
    declared = held.declarations()
    assert declared["width"] == declared["min-width"] == declared["max-width"] == exact, "held at exactly one width"
    (second,) = [rule for rule in rules if rule.selector == ".tbl.keys-2 :is(th,td):nth-child(2)"]
    pinned = second.declarations()
    # The second pins where the first ends: the first's width, its floor and its cap are one value.
    assert (pinned["position"], pinned["left"]) == ("sticky", exact)
    # Each frame's width is its own text's, between the floor (8rem) and the cap: Kairo's strata names are short, so
    # its By stratum holds far less than the cap; its contrasts pin a reading's expression, which reaches the cap.
    widths = {name: float(_frame_style(page, name)["--key"].removesuffix("rem")) for name in ("strata", "comparisons")}
    assert 8 <= widths["strata"] < int(span.group(1)) == widths["comparisons"], widths
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
    assert record.count("<table") == len(re.findall(_FRAME, record))
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
    # Kairo's readings are pinned by name (the owner's ruling of 2026-11-07), so no expression is: the probe table,
    # laid out in the same page, pins one, the id-like cell the stylesheet lets a pinned column break anywhere.
    aspect = _kairo_page().replace("</article>", table_html(_probe()) + "</article>", 1)
    pages = {
        "aspect": (aspect, ("comparisons", "comparison_statistics", "probe")),
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


@pytest.mark.integration
def test_one_arm_s_turned_scorecard_fits_the_688_px_frame_with_no_scroll(tmp_path: Path) -> None:
    """Discodon's SearXNG page drew a glyph column per measure for its one arm, wider than the frame. Turned, a row per
    measure with its value, interval and cases, the scorecard fits the host's 688 px frame without scrolling: its
    frame's content is no wider than the frame. Held on the shared measurement and on rung 1's triage, whose one arm
    reads sixteen measures, among them the long money measures' names."""
    _browser()
    with pytest.MonkeyPatch.context() as monkeypatch:
        copy = offline_copy("1-triage", tmp_path / "rung1", monkeypatch)
        triage = asyncio.run(load(copy / "triage.py", "triage", monkeypatch).triage_eval.run(store=":memory:")).page
    probe = """() => {
      const frame = document.querySelector('[data-table="scorecard"] .tbl');
      if (!frame) return null;
      return {scroll: frame.scrollWidth, client: frame.clientWidth, rows: frame.querySelectorAll('tbody tr').length,
              glyphs: frame.querySelectorAll('svg.viz-glyph').length};
    }"""
    for name, page, rows in (("measurement", s3_measurement().page(), 6), ("triage", triage, 16)):
        path = tmp_path / f"{name}.html"
        path.write_text(campaign_report_html(page), encoding="utf-8")
        drawn, errors = rendered(path, probe, width=FRAME_PX)
        assert not errors, errors
        assert drawn is not None and drawn["rows"] == rows, f"the control: {name}'s turned scorecard is drawn: {drawn}"
        assert drawn["glyphs"] == 0, (name, drawn)
        assert drawn["scroll"] <= drawn["client"] + 1, (name, drawn)


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


#: Each header of the written report's Arms table as drawn: how many lines its label sets in, where its first line
#: stands below the cell's top, and whether the frame scrolls.
_HEAD_PROBE = """() => {
  const table = document.querySelector('[data-table="arms_key"] table'), frame = table.closest('.tbl');
  const heads = [...table.querySelectorAll('thead th')].map(th => {
    const range = document.createRange(); range.selectNodeContents(th);
    const tops = [...new Set([...range.getClientRects()].map(rect => Math.round(rect.top)))].sort((a, b) => a - b);
    return {label: th.textContent.trim(), lines: tops.length, first: tops[0] - Math.round(th.getBoundingClientRect().top)};
  });
  return {scrolls: frame.scrollWidth > frame.clientWidth + 1, heads};
}"""


@pytest.mark.integration
def test_the_arms_table_s_header_row_sets_every_label_from_the_top_in_two_lines_at_most_at_688_px(
    tmp_path: Path,
) -> None:
    """Drawn in a host's 688 px frame, where the written report's Arms table scrolls anyway: no label takes more than
    two lines (``Share of output spent reasoning`` took five, past the frame's edge, and set the row's height), and
    every label's first line stands at one height, from the top."""
    _browser()
    path = tmp_path / "doc.html"
    path.write_text(_served()["doc"], encoding="utf-8")
    drawn, errors = rendered(path, _HEAD_PROBE, width=FRAME_PX)
    assert not errors, errors
    assert drawn["scrolls"], "the control: the Arms table scrolls in a 688 px frame"
    lines = {head["label"]: head["lines"] for head in drawn["heads"]}
    assert lines["Share of output spent reasoning"] == 2, lines
    assert max(lines.values()) <= 2, lines
    assert len({head["first"] for head in drawn["heads"]}) == 1, drawn["heads"]


@pytest.mark.integration
def test_the_written_headline_takes_a_third_fewer_lines_at_688_px(tmp_path: Path) -> None:
    """Kairo's 148-character headline, drawn in a host's 688 px frame: at most four lines at the scale's 20 px step,
    where 0.93's 36 px hero size took six."""
    _browser()
    path = tmp_path / "doc.html"
    path.write_text(_served()["doc"], encoding="utf-8")
    probe = """() => { const h = document.querySelector('h1'), s = getComputedStyle(h), size = s.fontSize;
      const lines = () => Math.round(h.getBoundingClientRect().height / parseFloat(getComputedStyle(h).lineHeight));
      const drawn = lines(); h.style.fontSize = '2.25rem'; h.style.lineHeight = '1.12';
      return {size, drawn, hero: lines()}; }"""
    drawn, errors = rendered(path, probe, width=FRAME_PX)
    assert not errors, errors
    assert drawn["size"] == "20px" and drawn["drawn"] <= 4 and drawn["drawn"] <= drawn["hero"] * 2 / 3, drawn


@pytest.mark.integration
def test_the_render_check_fails_a_table_that_fits_by_crushing_its_key_column(tmp_path: Path) -> None:
    """``scripts/check.py`` passes Kairo's pages, and fails the same campaign page with its pinned column let shrink
    to nothing (the 0.93 stylesheet's ``min-width:0; overflow-wrap:anywhere``): a table "fits" only where no pinned
    cell is under its floor and no word in one breaks but an id-like cell's."""
    _browser()
    good = tmp_path / "good.html"
    good.write_text(_kairo_campaign_html(), encoding="utf-8")
    checked, runs = render_checked(good, tmp_path / "shots")
    assert checked.returncode == 0 and len(runs) == 8, checked.stdout + checked.stderr
    crushed = tmp_path / "crushed.html"
    crushed.write_text(
        _kairo_campaign_html().replace(
            "</style>", ".tbl :is(th,td):first-child{min-width:0!important;overflow-wrap:anywhere!important}</style>", 1
        ),
        encoding="utf-8",
    )
    checked, runs = render_checked(crushed, tmp_path / "shots")
    assert checked.returncode == 1 and "crushed:" in checked.stdout, checked.stdout


@pytest.mark.integration
def test_the_render_check_fails_a_frame_with_no_cue_and_a_first_screen_at_two_measures(tmp_path: Path) -> None:
    """``scripts/check.py`` passes Kairo's written report, and fails it with its frames' fade taken away (a table
    overflows its frame at 688 px with no cue) and with its masthead let run the page's width (the first screen at two
    measures, as at 0.94.0)."""
    _browser()
    good = tmp_path / "good.html"
    good.write_text(_served()["doc"], encoding="utf-8")
    checked, runs = render_checked(good, tmp_path / "shots")
    assert checked.returncode == 0 and len(runs) == 8, checked.stdout + checked.stderr
    for name, rule, said in (
        ("uncued", ".tbl{animation:none!important}", "uncued:"),
        ("wide", ".mast{max-width:none!important}", "measures:"),
    ):
        bad = tmp_path / f"{name}.html"
        bad.write_text(_served()["doc"].replace("</style>", f"{rule}</style>", 1), encoding="utf-8")
        checked, _ = render_checked(bad, tmp_path / "shots")
        assert checked.returncode == 1 and said in checked.stdout, checked.stdout


# =============================================================================
# A key column as wide as its text, and an id as one word
# =============================================================================

#: Ids as a pinned column holds them: one the floor holds whole, one that needs more (``long``), and one past any cap.
SHORT_ID, LONG_ID, TOO_LONG_ID = "qwen3.8-flash", "deepseek-v4.1-flash", "abt-a-case-id-longer-than-any-cap-holds"


def _keyed_table(keys: list[str], *, key_columns: int = 1, figures: int = 6, name: str = "probe") -> TableBlock:
    """A table pinning ``keys`` (a second key column too, for two keys), then ``figures`` columns of numbers."""
    columns = [TableColumn(key="key", header="Stratum")]
    if key_columns == 2:
        columns.append(TableColumn(key="reading", header="Reading"))
    columns += [TableColumn(key=f"f{at}", header=f"Arm {at}") for at in range(figures)]
    rows = [
        {
            "key": TableCell(value=key),
            **({"reading": TableCell(value="pass rate")} if key_columns == 2 else {}),
            **{f"f{at}": TableCell(value=12.5 + at, value_text=f"{12.5 + at}") for at in range(figures)},
        }
        for key in keys
    ]
    return TableBlock(
        section="contrasts",
        name=name,
        title="Probe",
        columns=columns,
        rows=rows,
        order="as built",
        total_rows=len(rows),
        key_columns=2 if key_columns == 2 else 1,
    )


def test_a_hyphenated_id_in_a_pinned_cell_is_one_word_and_only_its_text_is_read() -> None:
    """A pinned cell sets each hyphenated word of its text as one word (``span.id``), ``long`` past what the floor
    holds; one past any cap, a link's ``href`` and a code span are left as they are. A column past the pinned one,
    whose words now wrap at their spaces, holds its ids whole alike, its links as built. The frame gives a column
    holding a long id the width that holds it whole (``--id1``)."""
    table = _keyed_table([SHORT_ID, f"\u25b8 {LONG_ID}", TOO_LONG_ID, "`a-b-c`", "All cases"]).model_copy(
        update={
            "columns": [*_keyed_table([]).columns[:1], TableColumn(key="arm", header="Arm")],
            "rows": [
                {
                    "key": TableCell(value=key),
                    # each links to the first, which carries the id
                    "arm": TableCell(
                        value="deepseek-v4.1-flash",
                        ref="case-deepseek-v4-1-flash",
                        anchor="case-deepseek-v4-1-flash" if at == 0 else None,
                    ),
                }
                for at, key in enumerate((SHORT_ID, f"\u25b8 {LONG_ID}", TOO_LONG_ID, "`a-b-c`", "All cases"))
            ],
        }
    )
    page = report_page([table])
    _frame, markup = _framed(page, "probe")
    firsts = re.findall(r"<tr[^>]*><td[^>]*>(.*?)</td>", markup)
    assert firsts[:2] == [f'<span class="id">{SHORT_ID}</span>', f'\u25b8 <span class="id long">{LONG_ID}</span>']
    assert firsts[2] == TOO_LONG_ID, "past any cap, an id wraps after its hyphens"
    assert firsts[3] == "<code>a-b-c</code>" and firsts[4] == "All cases"
    # The column past the pinned one keeps its link as built, its id held whole inside it.
    assert markup.count('<a href="#case-deepseek-v4-1-flash"><span class="id long">deepseek-v4.1-flash</span></a>') == 5
    widths = _frame_style(page, "probe")
    assert set(widths) == {"--id1"}, widths
    assert 8 <= float(widths["--id1"].removesuffix("rem")) < 14
    # A pinned column of short ids alone needs no width: the floor holds them.
    assert _frame_style(report_page([_keyed_table([SHORT_ID, "All cases"])]), "probe") == {}


def test_a_two_key_table_s_first_column_is_its_text_s_width_between_the_floor_and_the_cap() -> None:
    """Production, 0.93.1: By stratum's first column sat at its 224 px cap for "All cases", which alone made the table
    scroll at 688 px. Its width is its text's: the floor for a short key, the cap for a long one, between for the rest.
    """

    def key(keys: list[str]) -> float:
        page = report_page([_keyed_table(keys, key_columns=2)])
        return float(_frame_style(page, "probe")["--key"].removesuffix("rem"))

    assert key(["All cases"]) == 8.0
    assert key(["All cases", "a stratum named at length, past any cap a key column holds"]) == 14.0
    assert 8.0 < key(["All cases", "the long-tail requests"]) < 14.0
    assert "--key" not in _frame_style(report_page([_keyed_table(["All cases"])]), "probe"), "one key: auto layout"


def test_an_id_is_one_word_where_the_cap_holds_it_and_wraps_where_no_column_has_room() -> None:
    """Read as rules, for the pool: a short id never breaks; a long one is one box (a line moves it whole), held whole
    in a frame that scrolls anyway, and free to wrap after its hyphens where it does not, or on a phone; each pinned
    column is laid out at the frame's width for its ids, which the table gives it before the columns past it grow."""
    rules = _rules(_style(_served()["campaign"]))
    by = {(rule.at, rule.selector): rule.declarations() for rule in rules}
    assert by[("", ".tbl .id")] == {"white-space": "nowrap"}
    assert by[("", ".tbl .id.long")] == {"white-space": "normal", "display": "inline-block"}
    assert by[("@container scroll-state(scrollable:x)", ".tbl .id.long")] == {"white-space": "nowrap"}
    assert by[("@media (max-width:639px)", ".tbl .id.long")] == {"white-space": "normal"}
    order = [(rule.at, rule.selector) for rule in rules if rule.selector == ".tbl .id.long"]
    assert order[-1][0] == "@media (max-width:639px)", "a phone's rule comes last, so it wins in a scrolling frame"
    (frame,) = [rule for rule in rules if rule.selector == ".tbl" and not rule.at]
    assert frame.declarations()["container-type"] == "scroll-state"
    assert by[("", ".tbl :is(th,td):first-child")]["width"] == "min(var(--id1),35vw)"
    # The second key's width never passes its own cap.
    second = by[("", ".tbl.keys-2 :is(th,td):nth-child(2)")]
    assert second["width"] == f"min(var(--id2),{second['max-width'].removeprefix('min(')}", second


#: Every visible table frame: whether it scrolls, its pinned columns' widths and where the second is pinned, and each
#: hyphenated word in a pinned cell that the browser drew across more than one line.
_ID_PROBE = r"""async () => {
  for (const d of document.querySelectorAll('details')) d.open = true;
  // a frame's scroll state is read once the opened folds are laid out
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, 50))));
  const frames = [];
  for (const frame of document.querySelectorAll('.tbl')) {
    if (frame.clientWidth <= 1) continue;
    const keys = frame.classList.contains('keys-2') ? 2 : 1, broken = [];
    const head = frame.querySelector('tr');
    for (const row of frame.querySelectorAll('tr')) for (const cell of [...row.children].slice(0, keys)) {
      const walk = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
      for (let node = walk.nextNode(); node; node = walk.nextNode())
        for (const word of node.data.matchAll(/\S*\w-\w\S*/g)) {
          const range = document.createRange();
          range.setStart(node, word.index); range.setEnd(node, word.index + word[0].length);
          const tops = new Set([...range.getClientRects()].filter(r => r.width).map(r => Math.round(r.top)));
          if (tops.size > 1) broken.push(word[0]);
        }
    }
    const block = frame.closest('[data-table]');
    frames.push({table: block ? block.dataset.table : null, scrolls: frame.scrollWidth > frame.clientWidth + 1,
                 scrollWidth: frame.scrollWidth, clientWidth: frame.clientWidth, broken,
                 keys: head ? [...head.children].slice(0, keys).map(c => c.getBoundingClientRect().width) : [],
                 left: head && keys == 2 ? parseFloat(getComputedStyle(head.children[1]).left) : null});
  }
  return {page: document.documentElement.scrollWidth, frames};
}"""


def _drawn_ids(page: str, width: int, tmp_path: Path, name: str) -> dict[str, Any]:
    path = tmp_path / f"{name}-{width}.html"
    path.write_text(page, encoding="utf-8")
    drawn, errors = rendered(path, _ID_PROBE, width=width)
    assert not errors, errors
    return drawn


@pytest.mark.integration
def test_by_stratum_with_a_short_key_fits_a_688_px_frame_its_second_key_pinned_where_the_first_ends(
    tmp_path: Path,
) -> None:
    """Production, 0.93.1: By stratum, keyed by "All cases" and its reading, was 687 px in a 636 px frame, its first
    column held at the 224 px cap. Sized to its text it is the 128 px floor and the table fits, the second key pinned
    where the first ends. The control: the same table held at the cap scrolls (with five figures held at the cap it
    runs 72 px past the 640 px frame, and sized to its text it fits with 24 to spare)."""
    _browser()
    page = report_page([_keyed_table(["All cases"], key_columns=2, figures=5, name="strata")])
    drawn = _drawn_ids(page, FRAME_PX, tmp_path, "sized")
    (frame,) = [frame for frame in drawn["frames"] if frame["table"] == "strata"]
    assert frame["scrollWidth"] == frame["clientWidth"] and not frame["scrolls"], frame
    first, _second = frame["keys"]
    assert 127.5 <= first <= 128.5 and frame["left"] == pytest.approx(first, abs=0.5), frame
    sized = re.search(r'<div class="tbl keys-2" style="[^"]*">', page)
    assert sized is not None, "the control: the frame sets its key's width"
    capped = _drawn_ids(page.replace(sized.group(0), '<div class="tbl keys-2">'), FRAME_PX, tmp_path, "capped")
    (control,) = [frame for frame in capped["frames"] if frame["table"] == "strata"]
    assert control["keys"][0] == pytest.approx(224, abs=0.5) and control["scrolls"], control


@pytest.mark.integration
@pytest.mark.parametrize("width", FRAMES)
def test_no_pinned_id_breaks_after_its_hyphen_in_a_host_s_frame_and_one_too_long_still_wraps_inside_it(
    width: int, tmp_path: Path
) -> None:
    """Production, 0.93.1: model and case ids in pinned cells broke after a hyphen at the 128 px floor while the
    table scrolled anyway (44 on the written page at 688 px). Drawn here at both of a host's frames, on Kairo's code
    page (its cases table pins ids up to 24 characters) and a written report with its evidence record: no pinned id
    breaks. At a phone's 390 px a long id wraps after its hyphens, and an id past any cap wraps at every width; the
    page never scrolls sideways. Kairo's three-arm cases table fits the 1104 px frame, so the same page with six arms
    holds the guard there: a cases table wider than the frame, its case ids pinned."""
    _browser()
    pages = {
        "aspect": _kairo_page(),
        "aspect, six arms": _kairo_six_arm_page(),
        "doc with its record": _served()["doc with its record"],
    }
    for name, page in pages.items():
        drawn = _drawn_ids(page, width, tmp_path, re.sub(r"\W+", "-", name))
        assert drawn["page"] == width, (name, drawn["page"])
        ids = {frame["table"]: frame for frame in drawn["frames"]}
        # The control: a cases table that scrolls in this frame, pinning a 24-character case id.
        if name == "aspect, six arms" or (name == "aspect" and width == FRAME_PX):
            assert ids["cases"]["scrolls"] and "abt-genre-you-cant-stand" in page, ("the control", name, width)
        held = [(frame["table"], word) for frame in drawn["frames"] for word in frame["broken"] if len(word) <= 28]
        assert held == [], name
        assert '<span class="id' in page, f"the control: {name} pins a hyphenated id"
        # The first key is held at its estimated width, so a cell wider than the estimate would run under the second:
        # in every two-key frame the second starts where the first ends.
        two_keys = [frame for frame in drawn["frames"] if frame["left"] is not None]
        kairo = name.startswith("aspect")
        assert not kairo or len(two_keys) >= 2, "the control: Kairo's contrasts and strata keep two keys"
        for frame in two_keys:
            assert frame["left"] == pytest.approx(frame["keys"][0], abs=0.5), (name, width, frame)
    probe = report_page([_keyed_table([LONG_ID, TOO_LONG_ID], figures=16)])
    for at, broken in ((width, {TOO_LONG_ID}), (390, {LONG_ID, TOO_LONG_ID})):
        drawn = _drawn_ids(probe, at, tmp_path, f"probe-{at}")
        (frame,) = [frame for frame in drawn["frames"] if frame["table"] == "probe"]
        assert frame["scrolls"] and drawn["page"] == at, (at, frame)
        assert set(frame["broken"]) == broken, (at, frame)
        assert frame["keys"][0] <= 224.5, (at, frame)


# =============================================================================
# Fit first, then a cue; a long label in three lines; a nav link on one line
# =============================================================================

#: An aspect's label at length, with no short name declared: Discodon's SearXNG campaigns'.
_LONG_LABEL = "research judgement on SearXNG search with Extract page text"


@cache
def _long_label_page() -> str:
    """A measurement of one arm on one aspect labelled at length, read on its pass rate, two goal checks, a judge's
    two dimensions, cost and turn time: the many-column code-only scorecard Discodon's SearXNG campaigns draw."""
    scores = {
        "answer.correct": [1 if case < 15 else 0 for case in range(20)],
        "answer.clear": [5 if case < 9 else 3 for case in range(20)],
    }
    shape = shape_of(
        "SearXNG research check",
        [
            AspectTrials(
                _LONG_LABEL,
                [ArmTrials("deepseek-v4.1-flash", k=3, passing=14, cost=0.0027, latency_ms=900.0, scores=scores)],
                margin=None,
                bars=(CORRECT_BAR,),
            )
        ],
        control=False,
        question="How well does it research?",
    )
    return campaign_report_html(shape.page())


def test_a_long_pinned_label_gets_the_width_that_sets_it_in_three_lines() -> None:
    """Discodon's 0.94.0 pass: a code-only scorecard's aspect, a 59-character label with no short name, set in five
    lines at 688 px, its column held at the floor. A pinned label the floor would set in more than three lines gives
    its column the width that sets it in three (``--lines1``), up to the cap, as the column's floor, which the columns
    past it never crush; a short one leaves the floor as it is."""
    page = _long_label_page()
    widths = _frame_style(page, "scorecard")
    px = float(widths["--lines1"].removesuffix("rem")) * 16
    assert 8 * 16 < px <= min(14 * 16, 0.35 * FRAME_PX), widths
    # At the kit's table text, a generous 7.8 px a character, the label sets in at most three lines at that width.
    per_line, lines, used = int((px - 2 * 11.2) / 7.8), 0, 0
    for word in _LONG_LABEL.split():
        if lines and used + 1 + len(word) <= per_line:
            used += 1 + len(word)
        else:
            lines, used = lines + 1, len(word)
    assert lines <= 3, (px, per_line, lines)
    rules = _rules(_style(page))
    (floor,) = [rule for rule in rules if "--lines1" in rule.selector]
    assert floor.declarations() == {"min-width": "min(var(--lines1),35vw)"}
    assert _specificity(floor.selector) > _specificity(".tbl :is(th,td):first-child"), "it wins over the floor"
    assert "--lines1" not in _frame_style(report_page([_keyed_table(["All cases", SHORT_ID])]), "probe")


def test_a_nav_link_is_one_line_cut_with_an_ellipsis() -> None:
    """Discodon's 0.94.0 pass: an aspect's full label took a nav line of its own at 688 px. Each nav link is one line,
    cut with an ellipsis where it is longer than its cap; the whole label is the heading of the fold it lands on (a
    label is a host's text, which no attribute carries)."""
    page = _long_label_page()
    by = {(rule.at, rule.selector): rule.declarations() for rule in _rules(_style(page))}
    link = by[("", "nav a")]
    assert (link["white-space"], link["overflow"], link["text-overflow"]) == ("nowrap", "hidden", "ellipsis")
    assert link["max-width"].startswith("min(100%,"), link
    nav = page[page.index("<nav>") : page.index("</nav>")]
    assert f">{html.escape(_LONG_LABEL)}</a>" in nav and "title=" not in nav


def test_a_fold_s_summary_never_ends_a_line_on_its_separator() -> None:
    """Discodon's 0.94.0 pass at 688 px: an aspect's fold heading wrapped after its "·", which ended its line. The
    separator is bound to the words after it by a no-break space, so a line breaks before it, never after."""
    page = _long_label_page()
    counts = re.findall(r'<span class="count">(.*?)</span>', page)
    assert counts, "the control: the page's folds count what they hold"
    assert all(count.startswith("· ") for count in counts), counts


#: Every table frame a page draws in sight once its folds are open: its table's name, how wide its table is, how wide
#: the frame shows it, and whether its right edge fades, at its start and scrolled to its end.
_FRAMES_PROBE = """async (names) => {
  const frame = () => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)));
  document.querySelectorAll('details').forEach(d => { d.open = true; });
  await frame();
  const out = [];
  for (const f of document.querySelectorAll('.tbl')) {
    const name = f.closest('[data-table]')?.dataset.table;
    if (!f.offsetParent || f.clientWidth <= 2 || (names.length && !names.includes(name))) continue;
    const faded = () => getComputedStyle(f).maskImage !== 'none';
    f.scrollLeft = 0; await frame(); const start = faded();
    f.scrollLeft = f.scrollWidth; await frame(); const end = faded();
    f.scrollLeft = 0;
    out.push({table: name, scroll: f.scrollWidth, client: f.clientWidth, start, end});
  }
  return out;
}"""


@pytest.mark.integration
def test_the_scorecard_and_every_written_report_table_fit_a_1104_px_frame(tmp_path: Path) -> None:
    """Discodon's 0.94.0 pass: a code-only scorecard was cut at its frame's edge at 1104 px as well as 688. Fit first:
    at 1104 px the many-column scorecard and every table of a written report fit their frames, their words wrapped."""
    _browser()
    pages = {"scorecard": (_long_label_page(), ["scorecard"]), "doc": (_served()["doc"], [])}
    for name, (page, tables) in pages.items():
        path = tmp_path / f"{name}.html"
        path.write_text(page, encoding="utf-8")
        frames, errors = rendered(path, f"() => ({_FRAMES_PROBE})({json.dumps(tables)})", width=1104)
        assert not errors, errors
        assert frames, f"the control: {name} draws its tables"
        over = [frame for frame in frames if frame["scroll"] > frame["client"] + 1]
        assert over == [], (name, over)
        assert not [frame for frame in frames if frame["start"] or frame["end"]], "a frame that fits shows no cue"


@pytest.mark.integration
def test_a_frame_that_still_overflows_fades_its_edge_until_it_is_scrolled_to_its_end(tmp_path: Path) -> None:
    """At 688 px a written report's arms key and its every corrected comparison still overflow: each fades its right
    edge, over its header and every row, until it is scrolled to its end; a frame that fits shows none."""
    _browser()
    path = tmp_path / "doc.html"
    path.write_text(_served()["doc"], encoding="utf-8")
    frames, errors = rendered(path, f"() => ({_FRAMES_PROBE})([])", width=FRAME_PX)
    assert not errors, errors
    over = [frame for frame in frames if frame["scroll"] > frame["client"] + 1]
    assert over, "the control: a table overflows its frame at 688 px"
    assert all(frame["start"] and not frame["end"] for frame in over), over
    assert not [frame for frame in frames if frame not in over and (frame["start"] or frame["end"])]


@pytest.mark.integration
def test_a_long_aspect_label_sets_in_at_most_three_lines_in_the_scorecard_at_688_px(tmp_path: Path) -> None:
    _browser()
    path = tmp_path / "scorecard.html"
    path.write_text(_long_label_page(), encoding="utf-8")
    probe = """() => { const cell = document.querySelector('[data-table="scorecard"] tbody tr > :first-child');
      const range = document.createRange(); range.selectNodeContents(cell);
      return {text: cell.textContent.trim(),
              lines: new Set([...range.getClientRects()].filter(r => r.width).map(r => Math.round(r.top))).size}; }"""
    drawn, errors = rendered(path, probe, width=FRAME_PX)
    assert not errors, errors
    assert drawn["text"] == _LONG_LABEL and drawn["lines"] <= 3, drawn
