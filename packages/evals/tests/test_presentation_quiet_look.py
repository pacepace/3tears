"""A report page looks quiet and reads at a glance: its colours carry contrast, its type keeps a five-step scale, and
its tables are ruled, never boxed.

Held on the page as served (its stylesheet's tokens and rules, read here, since no browser runs on the pool) and,
where headless Chromium launches (the ``integration`` marker), on the page as drawn, through ``scripts/check.py``:

- **Contrast, both themes.** Quiet text (``--muted``, ``--ink-2``) reads at 4.5:1 or more on every ground the page
  paints (``--bg``, ``--surface``, ``--sunk``); each status colour reads at 4.5:1 on its own wash and on the surface;
  an arm's colour stands at 3:1 or more as a mark, and its tag's letter at 4.5:1 on it (WCAG 1.4.3, 1.4.11).
- **Colour means one thing.** No arm's colour sits within 40 degrees of hue of a status colour (pass green, warn
  amber, fail red), so an arm never reads as a verdict.
- **Five type sizes.** Every font size the page's stylesheets, its drawn charts and the kit's script set is one of
  12, 14, 16, 20 and 28 px, and the scale's tokens are those five steps. A code page's title is a label at 20 px.
- **Ruled tables.** A table frame draws no box and its header row no fill: a rule in ``--ink-2`` under the header
  and quiet rules between rows; headers read in ``--ink-2``, never the faintest text in the table.
- **A term's dotted underline once.** A column header linking a glossary term is underlined dotted only at the
  first header on the page that links it; every later header keeps its link, plain.
- Drawn, ``scripts/check.py`` passes Kairo's campaign and written pages at every frame in both themes, and fails a
  page with faint text, a size off the scale, a boxed table or a filled header row.
"""

from __future__ import annotations

import colorsys
import html
import re
from functools import cache
from pathlib import Path

import pytest

from pydantic import ValidationError

from threetears.evals.analysis import (
    ChartBlock,
    DisclosureBlock,
    TableBlock,
    build_campaign_report,
    campaign_report_html,
)
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.chart_svg import chart_drawing
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report.html_parts import KIT_JS, check_header_links, escape, table_html
from threetears.evals.analysis.report.model import TableCell, TableColumn
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.style import CANNOT_TELL_APART, CHECK_LINK_TITLE
from threetears.evals.kernel.report_doc import ReportDoc

from packages.evals.tests.campaign_page_support import kairo_page
from packages.evals.tests.clock_support import frozen_clock
from packages.evals.tests.page_contract_support import parsed
from packages.evals.tests.report_doc_support import converted_doc, snapshot_of
from packages.evals.tests.report_kit_support import KIT, chromium_unavailable, render_checked, rendered
from packages.evals.tests.report_shapes_support import s1_change, s2_choice, s4_change

#: The type scale's five steps, in px.
SCALE = frozenset({12.0, 14.0, 16.0, 20.0, 28.0})

#: The three blocks declaring the page's tokens, as their selectors open: light, dark by the reader's scheme, dark by
#: the host's ``data-theme``.
_THEMES = {"light": ":root{", "dark": ':root:not([data-theme="light"]){', "dark, stamped": ':root[data-theme="dark"]{'}


# =============================================================================
# The pages read here
# =============================================================================


@cache
def _doc() -> DocPage:
    """Kairo's written report, built from its stored document: its headline runs 148 characters."""
    name = "prototype3/kairo"
    with frozen_clock():
        return build(
            ReportDoc.model_validate(converted_doc(name)),
            ReportInput.model_validate(snapshot_of(name).report_input),
            strict=True,
        )


@cache
def _pages() -> dict[str, str]:
    """A code page, drawn and static, and a written report's page: every stylesheet the package serves."""
    with frozen_clock():
        return {
            "campaign": campaign_report_html(kairo_page()),
            "campaign static": campaign_report_html(kairo_page(), static=True),
            "doc": doc_page_html(_doc()),
        }


def _style(page: str) -> str:
    return re.sub(r"/\*.*?\*/", "", "".join(parsed(page).styles), flags=re.S)


def _top_rules(css: str) -> dict[str, dict[str, str]]:
    """The top-level rules of ``css`` (none inside an at-rule), each selector's declarations, later ones merged in."""
    rules: dict[str, dict[str, str]] = {}
    depth, start, prelude = 0, 0, ""
    for at, char in enumerate(css):
        if char == "{":
            if depth == 0:
                prelude = css[start:at].strip()
                start = at + 1
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                if not prelude.startswith("@"):
                    said = rules.setdefault(prelude, {})
                    for part in css[start:at].split(";"):
                        name, _, value = part.partition(":")
                        if value.strip():
                            said[name.strip()] = value.strip()
                start = at + 1
    return rules


# =============================================================================
# Colour: the tokens of each theme, resolved as a browser mixes them
# =============================================================================


def _tokens(theme: str) -> dict[str, str]:
    """The kit's tokens in ``theme``: the light block's, with a dark block's laid over them."""
    kit = (KIT / "kit" / "kit.css").read_text(encoding="utf-8")

    def block(opening: str) -> dict[str, str]:
        body = kit.split(opening, 1)[1].split("}", 1)[0]
        found: dict[str, str] = {}
        for part in body.split(";"):
            name, _, value = part.partition(":")
            if name.strip().startswith("--"):
                found[name.strip()] = value.strip()
        return found

    tokens = block(_THEMES["light"])
    if theme != "light":
        tokens |= block(_THEMES[theme])
    return tokens


_VAR = re.compile(r"^var\((--[\w-]+)(?:,(.*))?\)$")
_MIX = re.compile(r"^color-mix\(in (srgb|oklab),\s*(.+?) (\d+)%,(.+)\)$")


def _rgb(value: str, tokens: dict[str, str]) -> tuple[float, float, float]:
    """A token's value as sRGB: a hex colour, a ``var()`` (a host token falls back to the page's own value) or a
    ``color-mix`` in srgb, which mixes the gamma-encoded channels."""
    value = value.strip()
    if found := _VAR.match(value):
        name, fallback = found.group(1), found.group(2)
        return _rgb(tokens[name] if name in tokens else str(fallback), tokens)
    if found := _MIX.match(value):
        space, share = found.group(1), int(found.group(3)) / 100
        first, second = _rgb(found.group(2), tokens), _rgb(found.group(4), tokens)
        if space == "oklab":
            a, b = _to_oklab(first), _to_oklab(second)
            return _from_oklab(
                (
                    a[0] * share + b[0] * (1 - share),
                    a[1] * share + b[1] * (1 - share),
                    a[2] * share + b[2] * (1 - share),
                )
            )
        return (
            first[0] * share + second[0] * (1 - share),
            first[1] * share + second[1] * (1 - share),
            first[2] * share + second[2] * (1 - share),
        )
    assert re.fullmatch(r"#[0-9a-fA-F]{6}", value), f"not a colour this test reads: {value}"
    return (int(value[1:3], 16) / 255, int(value[3:5], 16) / 255, int(value[5:7], 16) / 255)


def _to_oklab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """sRGB as Björn Ottosson's oklab, the space CSS ``color-mix(in oklab, ...)`` mixes in."""
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _from_oklab(lab: tuple[float, float, float]) -> tuple[float, float, float]:
    lightness, a, b = lab
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    linear = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )
    encoded = [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in linear]
    return (min(1.0, max(0.0, encoded[0])), min(1.0, max(0.0, encoded[1])), min(1.0, max(0.0, encoded[2])))


def _luminance(rgb: tuple[float, float, float]) -> float:
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(theme: str, ink: str, ground: str) -> float:
    """The WCAG contrast of token ``ink`` on token ``ground`` in ``theme``."""
    return _drawn_contrast(theme, f"var({ink})", f"var({ground})")


def _drawn_contrast(theme: str, ink: str, ground: str) -> float:
    """The WCAG contrast of colour ``ink`` on colour ``ground``, each as a drawing writes it, in ``theme``."""
    tokens = _tokens(theme)
    lighter, darker = sorted((_luminance(_rgb(ink, tokens)), _luminance(_rgb(ground, tokens))), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _hue(theme: str, token: str) -> float:
    return colorsys.rgb_to_hls(*_rgb(f"var({token})", _tokens(theme)))[0] * 360


_ARMS = ("--s1", "--s2", "--s3", "--s4", "--s5")
_STATUS = ("--good", "--warn", "--bad")


@pytest.mark.parametrize("theme", list(_THEMES))
def test_quiet_text_reads_at_four_and_a_half_to_one_on_every_ground(theme: str) -> None:
    """Production, 0.94: the light theme's ``--muted`` read 3.3:1 on the page, and it set the table headers, the tick
    labels, the captions and the verdict words. Quiet text, secondary text and the status words read at 4.5:1 on
    every ground they are drawn on (WCAG 1.4.3)."""
    faint = {
        (ink, ground): round(_contrast(theme, ink, ground), 2)
        for ink in ("--muted", "--ink-2", "--ink")
        for ground in ("--bg", "--surface", "--sunk", "--accent-wash", "--warn-wash")
        if _contrast(theme, ink, ground) < 4.5
    }
    faint |= {
        (status, ground): round(_contrast(theme, status, ground), 2)
        for status in _STATUS
        for ground in (f"{status}-wash", "--surface")
        if _contrast(theme, status, ground) < 4.5
    }
    assert not faint, f"{theme}: under 4.5:1: {faint}"


@pytest.mark.parametrize("theme", list(_THEMES))
def test_an_arm_s_colour_stands_as_a_mark_and_carries_its_tag_s_letter(theme: str) -> None:
    """Production, 0.94: the third arm's green drew at 2.8:1 on the surface. Each arm's colour stands at 3:1 or more
    against the page and its cards (WCAG 1.4.11), and its tag's letter, the kit's on-fill, reads at 4.5:1 on it."""
    weak = {
        (arm, ground): round(_contrast(theme, arm, ground), 2)
        for arm in _ARMS
        for ground in ("--surface", "--bg")
        if _contrast(theme, arm, ground) < 3
    }
    weak |= {
        (arm, "--on-fill"): round(_contrast(theme, "--on-fill", arm), 2)
        for arm in _ARMS
        if _contrast(theme, "--on-fill", arm) < 4.5
    }
    assert not weak, f"{theme}: {weak}"


@pytest.mark.parametrize("theme", list(_THEMES))
def test_no_arm_s_colour_shares_a_hue_family_with_a_status_colour(theme: str) -> None:
    """Production, 0.94: arm C's series green drew beside the outcome bar's status green, and the orange and ochre
    arms sat in the warn and fail families. Every arm's hue stands at least 40 degrees from pass, warn and fail, so a
    colour on a page means one thing: no arm is drawn in a status colour."""
    near = {
        (arm, status): round(gap, 1)
        for arm in _ARMS
        for status in _STATUS
        if (gap := min(abs(_hue(theme, arm) - _hue(theme, status)), 360 - abs(_hue(theme, arm) - _hue(theme, status))))
        < 40
    }
    assert not near, f"{theme}: arm hues within 40 degrees of a status hue: {near}"


# =============================================================================
# Type: five sizes
# =============================================================================


def _px(value: str, steps: dict[str, str]) -> float | None:
    """A ``font-size`` as px, or None where it takes its parent's (``inherit``, ``1em``, ``100%``)."""
    value = value.strip()
    if value in {"inherit", "1em", "100%"}:
        return None
    if found := _VAR.match(value):
        return _px(steps[found.group(1)], steps)
    if value.endswith("rem"):
        return float(value.removesuffix("rem")) * 16
    if value.endswith("px"):
        return float(value.removesuffix("px"))
    return -1.0


def test_every_font_size_is_one_of_five_steps() -> None:
    """Production, 0.94: seven steps from 12 to 36 px plus 11, 12 and 13 px strays, three of them on one table. Every
    size a stylesheet sets (at every width), a drawn chart's text sets in its own style, and the kit's script draws or
    measures a chart's text at, is one of 12, 14, 16, 20 and 28 px; the scale's tokens are exactly those steps."""
    steps = {name: value for name, value in _tokens("light").items() if name.startswith("--t-")}
    assert {_px(value, steps) for value in steps.values()} == SCALE, steps
    off: set[str] = set()
    for name, page in _pages().items():
        css = _style(page)
        said = re.findall(r"font-size\s*:\s*([^;}]+)", css)
        said += [found.split("/")[0].split()[0] for found in re.findall(r"[{;\s]font\s*:\s*([^;}]+)", css)]
        said += re.findall(r'font-size:\s*([^;"]+)', page.split("</style>", 1)[1])
        off |= {f"{name}: {value}" for value in said if (px := _px(value, steps)) is not None and px not in SCALE}
    # textW's second argument is the size a label is measured at; its first is the text, holding no comma
    drawn = re.findall(r"font-size:(\d+(?:\.\d+)?)px", KIT_JS)
    drawn += re.findall(r"textW\((?:[^,()]|\([^()]*\))*,\s*(\d+)", KIT_JS)
    off |= {f"kit.js: {size}px" for size in drawn if float(size) not in SCALE}
    assert not off, sorted(off)


def test_a_code_page_s_title_is_a_label_at_twenty_px() -> None:
    """The code page's title is the campaign's name, a label over the lead: it sets at the scale's 20 px step, where
    a written report's headline keeps the hero size."""
    for name in ("campaign", "campaign static"):
        rules = _top_rules(_style(_pages()[name]))
        assert rules["article > header > h1"].get("font-size") == "var(--t-lg)", name
    assert _top_rules(_style(_pages()["doc"]))["h1"]["font-size"] == "var(--t-hero)"


def test_a_headline_over_ninety_characters_sets_in_at_twenty_px() -> None:
    """Production, 0.94: a 40-word writer headline was set at the hero size and ran six lines at 1104 px, the size a
    one-word campaign id got too. A headline over 90 characters is classed ``long``, which sets at the scale's 20 px
    step; one of 90 keeps the hero size."""
    assert '<h1 class="long">' in _pages()["doc"], "the control: Kairo's 148-character headline"
    for length, opened in ((90, "<h1>"), (91, '<h1 class="long">')):
        page = doc_page_html(_doc().model_copy(update={"headline": "a" * length}))
        assert f"{opened}{'a' * length}</h1>" in page, (length, page.split("<h1", 1)[1][:120])
    long = _top_rules(_style(_pages()["doc"]))["h1.long"]
    assert long["font-size"] == "var(--t-lg)", long


def _diff_heatmap() -> ChartBlock:
    """A difference heatmap with a cell at every strength its drawing takes, on both sides of zero."""
    values = [v / 2 for v in range(-20, 21) if v]
    return ChartBlock.model_validate(
        {
            "section": "memo",
            "title": "Difference",
            "chart": "diff_heatmap",
            "data": {
                "fmt": "+pts",
                "columns": [{"label": str(v)} for v in values],
                "rows": [{"label": "row", "cells": [{"value": v} for v in values]}],
            },
        }
    )


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a_difference_cell_s_figure_reads_on_its_own_fill_in_either_theme(theme: str) -> None:
    """Reference r3, 2026-11-08: the difference heatmap put the on-fill white on every cell past 55% strength, which
    read 3.0:1 on a light red tint. Drawn on the server, every cell's figure takes the ink that reads best on that
    cell's own fill, and reads at 4.5:1 or more in each theme."""
    drawing = chart_drawing(_diff_heatmap())
    assert drawing is not None
    cell = (
        r'<rect[^>]*style="fill:([^";]+)[^"]*"[^>]*>(?:</rect>)?\s*<text[^>]*class="cell-main"[^>]*style="fill:([^"]+)"'
    )
    cells = re.findall(cell, drawing)
    assert len(cells) >= 40, drawing[:600]
    faint = {
        (fill, ink): round(ratio, 2)
        for fill, ink in cells
        if (ratio := _drawn_contrast(theme, html.unescape(ink), html.unescape(fill))) < 4.5
    }
    assert not faint, faint


def test_a_scatter_s_y_title_sits_above_its_plot_never_inside_it() -> None:
    """Written page r2-without-its-gates at 390 px: the y title "Pass rate" sat inside the plot at its bottom left,
    and an arm's point was drawn over it. The title takes a line of its own above the plot's top gridline."""
    block = ChartBlock.model_validate(
        {
            "section": "memo",
            "title": "Cost and pass rate",
            "chart": "scatter_frontier",
            "data": {
                "x_domain": [0, 10],
                "y_domain": [0, 100],
                "x_fmt": "usd",
                "y_fmt": "pct",
                "y_title": "Pass rate",
                "points": [{"x": 1, "y": 2, "id": "a", "label": "a"}, {"x": 6, "y": 90, "id": "b", "label": "b"}],
            },
        }
    )
    drawing = chart_drawing(block)
    assert drawing is not None
    for svg in re.findall(r"<svg\b.*?</svg>", drawing, re.S):
        (title,) = [float(y) for y in re.findall(r'<text[^>]*\by="([\d.]+)"[^>]*>Pass rate</text>', svg)]
        grids = [float(y) for y in re.findall(r'<line[^>]*\by1="([\d.]+)"[^>]*class="grid"', svg)]
        assert grids and title < min(grids), (title, sorted(grids)[:3])


# =============================================================================
# Tables: ruled, never boxed
# =============================================================================


def test_a_table_is_ruled_with_hairlines_never_boxed_or_tinted() -> None:
    """Production, 0.94: every table sat in a white, rounded box with a grey header fill. A frame draws no border and
    sits on the ground it is set on (``--ground``: the page's ``--bg``, a card's or a figure's ``--surface``); a header
    cell has no fill but that ground (a pinned one paints it, so it hides what scrolls beneath it); the header row
    stands on a rule in ``--ink-2`` and reads in ``--ink-2``; rows part on quiet ``--line`` rules (Schwabish 2020;
    booktabs)."""
    for name, page in _pages().items():
        rules = _top_rules(_style(page))
        frame = rules[".tbl"]
        assert frame.get("border", "0") in {"0", "none"}, (name, frame)
        heads = {
            selector: said
            for selector, said in rules.items()
            if re.search(r"(^|[\s,>(])th\b", selector) and not re.search(r"tbody|tr\.hl|:target", selector)
        }
        fills = {
            selector: said["background"]
            for selector, said in heads.items()
            if "background" in said and said["background"] not in {"none", "var(--ground)"}
        }
        assert not fills, (name, fills)
        assert rules[".tbl thead th:first-child"]["background"] == "var(--ground)", name
        assert "--ground:var(--bg);" in _style(page).split("}", 1)[0] and rules[
            ".fig .tbl,.decision .tbl,.excard .tbl,.callout .tbl"
        ] == {"--ground": "var(--surface)"}, name
        assert rules["thead th"]["border-bottom"] == "1px solid var(--ink-2)", name
        assert rules["th"]["color"] == "var(--ink-2)", name
        assert rules["th,td"]["border-bottom"] == "1px solid var(--line)", name


def _cases_table(name: str) -> str:
    return table_html(
        TableBlock(
            section="contrasts",
            name=name,
            title=name,
            columns=[
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="cases", header="Cases", term="cases"),
            ],
            rows=[{"arm": TableCell(value="v1"), "cases": TableCell(value=12, value_text="12")}],
            order="as read",
            total_rows=1,
        )
    )


def test_a_header_term_is_underlined_dotted_at_its_first_header_only() -> None:
    """Production, 0.94: nearly every column header carried a dotted underline, the same term's on every table. The
    first header on the page linking a term keeps its dotted underline; each later header linking the same term is
    marked ``seen`` and drawn plain, its link kept. A link in prose is the page's own (solid) and marks nothing."""
    markup = check_header_links(_cases_table("first") + _cases_table("second"), [])
    links = re.findall(r'<a\b[^>]*href="#g-cases"[^>]*>', markup)
    assert len(links) == 2, markup
    assert 'class="seen"' not in links[0] and 'class="seen"' in links[1], links
    for name in ("campaign", "doc"):
        rules = _top_rules(_style(_pages()[name]))
        assert (
            rules["th a"]["text-decoration"] == "underline dotted" and rules["th a.seen"]["text-decoration"] == "none"
        )
    # A link's target can be moved to land outside a closed fold (the glossary's own heading), so a term's link is
    # told by its title, the term's sentence; a goal check's link carries one title for every check and is left out.
    headed = [
        (found.group(1), 'class="seen"' in anchor)
        for cell in re.findall(r"<th\b[^>]*>.*?</th>", _pages()["campaign"], re.S)
        for anchor in re.findall(r"<a\b[^>]*>", cell)
        if (found := re.search(r'title="([^"]+)"', anchor)) is not None and found.group(1) != escape(CHECK_LINK_TITLE)
    ]
    assert len({title for title, _ in headed}) < len(headed), "the control: the page links a term from two headers"
    first: set[str] = set()
    for title, seen in headed:
        assert seen == (title in first), title
        first.add(title)
    prose = re.findall(r'<a class="term"[^>]*>', _pages()["campaign"])
    assert prose and not any("seen" in anchor for anchor in prose)


# =============================================================================
# The lead's weight: its decisions in bold, its figures regular
# =============================================================================


def _lead(text: str, emphasis: list[str], term: str | None = None) -> DisclosureBlock:
    return DisclosureBlock(section="answer", source="verdict", text=text, tone="open", term=term, emphasis=emphasis)


def _verdict_line(lead: DisclosureBlock) -> str:
    """The verdict line of a campaign page whose lead the caller supplies, as its markup."""
    shape = s4_change()
    page = campaign_report_html(build_campaign_report(shape.campaign, shape.readings, shape.reports, lead=[lead]))
    found = re.search(r'<p class="verdict"[^>]*>(.*?)</p>', page, re.S)
    assert found is not None
    return found.group(1)


@pytest.mark.parametrize(
    ("text", "emphasis", "said"),
    [
        ("Pass rate: v2 shown better. Of 3 trials.", ["v3 shown better"], "does not say there"),
        ("Pass rate: v2 shown better. Of 3 trials.", ["Of 3 trials", "Pass rate"], "does not say there"),
        ("Pass rate: v2 shown better. Of 3 trials.", [""], "does not say there"),
        ("On `check_x`: v2 shown better.", ["check_x`: v2"], "cuts a code span"),
    ],
    ids=["unsaid", "out of order", "empty", "cuts a code span"],
)
def test_a_phrase_set_in_bold_is_said_in_order_whole_and_outside_a_code_span(
    text: str, emphasis: list[str], said: str
) -> None:
    """A host hands a lead to ``build_campaign_report(lead=...)`` as a block, built or read from JSON: a phrase it asks
    to set in bold that the line does not say, says before the one ahead of it, is empty, or would cut a code span in
    two, is refused where the block is made."""
    with pytest.raises(ValidationError, match=said):
        _lead(text, emphasis)
    with pytest.raises(ValidationError, match=said):
        DisclosureBlock.model_validate(
            {"section": "answer", "source": "verdict", "text": text, "tone": "open", "emphasis": emphasis}
        )


def test_a_decision_in_bold_keeps_a_host_s_text_escaped() -> None:
    """A host's lead holding ``<``, ``&`` and ``"`` is escaped inside the bold as outside it: the bold wraps the
    escaped text, never markup the text could open."""
    line = _verdict_line(
        _lead('Keep "a<b" & c: v2 shown better. Of 3 trials, a<b.', ['Keep "a<b" & c: v2 shown better'])
    )
    assert line.startswith("<strong>Keep &quot;a&lt;b&quot; &amp; c: v2 shown better</strong>. Of 3 trials, a&lt;b."), (
        line
    )
    assert "<b" not in line.replace("<strong>", "")


def test_a_phrase_in_bold_is_placed_on_the_text_never_on_its_markup() -> None:
    """Bold is placed where the check found the phrase in the raw text, so a phrase next to ``&`` is never found inside
    its ``&amp;``, and one next to a code span leaves the span whole."""
    assert (
        _verdict_line(_lead("Cost & amp: v2 shown better.", ["amp"]))
        == "Cost &amp; <strong>amp</strong>: v2 shown better."
    )
    assert (
        _verdict_line(_lead("`check_x`: v2 shown better. Of 3.", ["v2 shown better"]))
        == "<code>check_x</code>: <strong>v2 shown better</strong>. Of 3."
    )


def test_the_undecided_reading_is_linked_once_inline_else_after_the_line() -> None:
    """The verdict line links the undecided reading in its sentence form where that form stands whole in one piece of
    the line; where it does not (cut by the bold, or longer than the form allows), the line links the term after its
    words, so it is never left unlinked."""
    bracketed = "(cannot tell apart)</a>"
    inline = _verdict_line(
        _lead(
            "Pass rate: the evidence cannot tell v2 and production apart. Of 3.",
            ["Pass rate: the evidence cannot tell v2 and production apart"],
            CANNOT_TELL_APART,
        )
    )
    assert re.findall(r"<a\b[^>]*>(.*?)</a>", inline) == ["cannot tell v2 and production apart"], inline
    cut = _verdict_line(
        _lead(
            "Pass rate: the evidence cannot tell v2 and production apart. Of 3.",
            ["Pass rate: the evidence cannot tell v2"],
            CANNOT_TELL_APART,
        )
    )
    long = _verdict_line(_lead(f"The evidence cannot tell {'v2, ' * 50}and production apart.", [], CANNOT_TELL_APART))
    for line in (cut, long):
        assert line.endswith(bracketed) and len(re.findall(r"<a\b", line)) == 1, line


def _bold_and_regular(page: str) -> tuple[list[str], str]:
    """A page's verdict line: what it sets in bold, each phrase without its tags, and the rest of it."""
    found = re.search(r'<p class="verdict"[^>]*>(.*?)</p>', page, re.S)
    assert found is not None
    line = found.group(1)
    bold = [re.sub(r"<[^>]+>", "", said) for said in re.findall(r"<strong>(.*?)</strong>", line, re.S)]
    return bold, re.sub(r"<[^>]+>", "", re.sub(r"<strong>.*?</strong>", "", line, flags=re.S))


def test_the_campaign_lead_sets_its_decisions_in_bold_in_its_rows_and_pairs_forms() -> None:
    """The campaign lead leads on an aspect's pass-rate rows where no contrast is read on a lead measure, and on its
    pairs for a choice: each decision bold, the figures after it regular (rung 3's contrast form is held in
    ``test_presentation_one_reading``)."""
    rows = s1_change()
    bold, regular = _bold_and_regular(
        campaign_report_html(build_campaign_report(rows.campaign, rows.readings, rows.reports))
    )
    assert bold[:2] == [
        "Pass rate on billing: v2 against production shown worse",
        "Pass rate on billing: v3 against production unclear",
    ]
    assert len(bold) == 9 and "Of 20 cases each, production passed 18 and v2 6." in regular, (bold, regular)
    pairs = s2_choice()
    bold, regular = _bold_and_regular(
        campaign_report_html(build_campaign_report(pairs.campaign, pairs.readings, pairs.reports))
    )
    assert bold == [
        "The evidence cannot tell alpha and beta apart, alpha ahead of gamma, the evidence cannot tell beta and gamma apart"
    ], bold
    assert regular.strip() == ".", regular


# =============================================================================
# The appendix's last lines, and a name held whole in running text
# =============================================================================


#: The model a candidate of Kairo round 2 ran, which its written page names in the headline, the lede and the cards.
_HYPHENATED_MODEL = "qwen3.8-flash"


@cache
def _doc5() -> DocPage:
    """Kairo round 2's written report (prototype 5): Method terms carrying caveats, a made line, and a hyphenated
    model's name in its headline, lede and decision cards."""
    name = "prototype5/kairo"
    with frozen_clock():
        return build(
            ReportDoc.model_validate(converted_doc(name)),
            ReportInput.model_validate(snapshot_of(name).report_input),
            strict=True,
        )


def test_a_model_s_name_in_the_headline_lede_and_cards_is_held_whole() -> None:
    """Discodon's acceptance page broke ``qwen3.8-`` / ``flash`` across its headline's lines: a table's cell held a
    hyphenated word whole, running text did not. Every place the written page's body prints the name above its
    appendix (headline, lede, cards, claims, captions, code's lines, chart values) holds it as one word
    (``span.id``); the masthead, a row of labelled parts, is left flat."""
    page = doc_page_html(_doc5())
    first = page[page.index("<body") : page.index('data-part="appendix"')]
    # The masthead is a flat row of labelled spans a host reads part by part, not running text.
    first = re.sub(r'<div class="mast">.*?</div>', "", first, count=1, flags=re.S)
    said = [part for part in re.split(r"(<[^>]+>)", first) if not part.startswith("<")]
    held = sum(
        word.count(_HYPHENATED_MODEL) for word in re.findall(r'<span class="id(?: long)?">([^<]*)</span>', first)
    )
    named = sum(part.count(_HYPHENATED_MODEL) for part in said)
    assert named > 3 and held == named, (held, named)
    headline = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)
    assert headline is not None and f'<span class="id">{_HYPHENATED_MODEL}</span>' in headline.group(1)


#: At the drawn width: the headline's held name set after each of sixty runs of filler, and the line fragments it
#: takes held and bare; then the made line's and a Method caveat line's type and spacing beside the appendix's.
_HELD_AND_APPENDIX = (
    """() => {
  const h1 = document.querySelector('.bluf h1, h1');
  const name = '%s';
  const frags = (markup) => {h1.innerHTML = markup; const el = h1.querySelector('span.id');
    if (el) return el.getClientRects().length;
    const text = [...h1.childNodes].find(n => n.nodeType === 3 && n.textContent.includes(name));
    const range = document.createRange(); const at = text.textContent.indexOf(name);
    range.setStart(text, at); range.setEnd(text, at + name.length); return range.getClientRects().length;};
  let held = 0, bare = 0;
  for (let n = 0; n < 60; n++) {
    const lead = 'Keep September: ' + 'more '.repeat(n);
    held = Math.max(held, frags(lead + '<span class="id">' + name + '</span> costs more and holds character no better.'));
    bare = Math.max(bare, frags(lead + name + ' costs more and holds character no better.'));
  }
  const made = document.querySelector('p.cap[data-part="made"]');
  const dd = document.querySelector('.method dd');
  const cap = document.querySelector('.method .cap[data-part="caveats"]');
  const style = (el) => getComputedStyle(el);
  const above = made.previousElementSibling.getBoundingClientRect().bottom;
  return {held, bare,
    made: {size: style(made).fontSize, gap: made.getBoundingClientRect().top - above},
    appendix: {size: style(dd).fontSize, color: style(dd).color},
    caveat: {size: style(cap).fontSize, color: style(cap).color, margin: parseFloat(style(cap).marginTop)}};
}"""
    % _HYPHENATED_MODEL
)


@pytest.mark.integration
def test_at_688_px_a_name_never_breaks_at_its_hyphen_and_the_appendix_ends_quietly(tmp_path: Path) -> None:
    """At 688 px, a long headline set with the model's name at every place a line can end: held, the name is always
    one line fragment, where bare it breaks after its hyphen. The made line (who wrote the report, for what, when)
    sets at the appendix's size with a space above it, where it set at body size flush under the last finding; a
    Method term's caveat line is spaced from its paragraph and quieter, where it ran on from it."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    path = tmp_path / "doc.html"
    made = "Written by claude-opus-5-5, served as claude-opus-5-5, for $2.54, with analyst skill fd74326c6f2e."
    path.write_text(doc_page_html(_doc5().model_copy(update={"made": made})), encoding="utf-8")
    drawn, errors = rendered(path, _HELD_AND_APPENDIX, width=688)
    assert errors == []
    assert drawn["bare"] > 1, "the control: some line end falls inside the bare name"
    assert drawn["held"] == 1, drawn
    assert drawn["made"]["size"] == drawn["appendix"]["size"] == "14px", drawn
    assert drawn["made"]["gap"] >= 16, drawn
    assert drawn["caveat"]["size"] == drawn["appendix"]["size"] and drawn["caveat"]["margin"] > 0, drawn
    assert drawn["caveat"]["color"] != drawn["appendix"]["color"], "the caveat line reads quieter than its term's text"


# =============================================================================
# Drawn: scripts/check.py
# =============================================================================


@pytest.mark.integration
def test_the_render_check_holds_contrast_type_and_tables_in_both_themes(tmp_path: Path) -> None:
    """``scripts/check.py`` passes Kairo's campaign page and written report at 1200, 1104, 688 and 390 px, light and
    dark, and fails each page with one fault put in: faint header text, a size off the scale, a boxed table, a filled
    header row."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    faults = (
        ("faint", "th{color:color-mix(in srgb,var(--muted) 50%,var(--surface))!important}", "contrast:"),
        ("stray", "p{font-size:13px!important}", "sizes:"),
        ("boxed", ".tbl{border:1px solid var(--line)!important}", "tables:"),
        ("filled", "thead th{background:var(--sunk)!important}", "tables:"),
    )
    for name in ("campaign", "doc"):
        good = tmp_path / f"{name}.html"
        good.write_text(_pages()[name], encoding="utf-8")
        checked, runs = render_checked(good, tmp_path / "shots")
        assert checked.returncode == 0 and len(runs) == 8, checked.stdout + checked.stderr
        for fault, rule, said in faults:
            bad = tmp_path / f"{name}-{fault}.html"
            bad.write_text(_pages()[name].replace("</style>", f"{rule}</style>", 1), encoding="utf-8")
            checked, _ = render_checked(bad, tmp_path / "shots")
            assert checked.returncode == 1 and said in checked.stdout, (name, fault, checked.stdout)
