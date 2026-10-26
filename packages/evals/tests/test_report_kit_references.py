"""The reference memos in ``docs/eval-report-kit`` build from their sources and pass the kit's render checks.

Each reference is a page source (``*.src.html``) plus its data (``*.json``); ``scripts/build.py`` inlines the kit
and the data into the built page committed beside them. Two contracts hold for every reference:

- building the source reproduces the committed page byte for byte, so the built page never drifts from its
  source, its data or the kit;
- ``scripts/check.py`` passes on the built page at 1200 and 390 px, light and dark: no console errors, no
  sideways scroll, nothing past the right edge, no overlapping chart labels (GENERATOR.md §8, items 1-3).

- the Kairo page says what its plan says (``references/r5-kairo-round2.plan.json``), so the two cannot drift apart
  silently: each evidence section's eyebrow, and its headline and takeaway where the page source writes them as
  text, are the plan's, read off the source with no browser; and its headline, lede, four decision slots, every
  Method pair and the example cards it draws (the rows the plan's examples query selects), as the browser draws
  them.

The render check drives headless Chromium through playwright, which not every runner has, so it carries the
``integration`` marker and runs under ``./scripts/test-integration.sh evals``. So does the drawn half of the Kairo
parity check: the page writes that prose and the cards from its data in its script, so they exist only once a
browser has run it. Where Chromium
cannot launch either skips and says why; a skip there is a missing browser, never a pass.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.report_kit_support import KIT, chromium_unavailable, render_checked, rendered, run_script
from threetears.evals.analysis.report.figure_keys import render

REFERENCES = sorted(p.name.removesuffix(".src.html") for p in (KIT / "references").glob("*.src.html"))


def _build(name: str, out: Path) -> Path:
    page = out / f"{name}.html"
    built = run_script("build.py", KIT / "references" / f"{name}.src.html", KIT / "references" / f"{name}.json", page)
    assert built.returncode == 0, built.stderr
    return page


def test_the_kit_holds_all_five_references() -> None:
    assert REFERENCES == [
        "r1-model-selection",
        "r2-config-sweep",
        "r3-classifier",
        "r4-tradeoff",
        "r5-kairo-round2",
    ]


@pytest.mark.parametrize("name", REFERENCES)
def test_a_reference_builds_to_its_committed_page(name: str, tmp_path: Path) -> None:
    page = _build(name, tmp_path)
    assert page.read_text() == (KIT / "references" / f"{name}.html").read_text(), (
        f"{name}.html is stale: rebuild it with scripts/build.py from its .src.html and .json"
    )


@pytest.mark.integration
@pytest.mark.parametrize("name", REFERENCES)
def test_a_reference_passes_the_render_checks(name: str, tmp_path: Path) -> None:
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = _build(name, tmp_path)
    checked, runs = render_checked(page, tmp_path / "screenshots")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert len(runs) == 4 and all(line.startswith("ok") for line in runs), checked.stdout


#: Reads the Kairo page's first-screen prose and its Method pairs once its script has drawn them.
_KAIRO_PROSE_PROBE = """() => {
  const text = (id) => document.getElementById(id).innerText;
  return {
    headline: text("h1"), lede: text("lede"), recommendation: text("d-rec"), why_not: text("d-why"),
    watch: text("d-watch"), next: text("d-next"),
    method: [...document.querySelectorAll("#method dt")].map((dt) => [dt.innerText, dt.nextElementSibling.innerText]),
    cards: [...document.querySelectorAll("#ex .excard")].map((card) => card.querySelector(".top span.mono").innerText),
  };
}"""


def _said(text: str) -> str:
    """Prose as a reader reads it: whitespace collapsed, Markdown emphasis dropped."""
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


@pytest.mark.integration
def test_the_kairo_page_says_what_its_plan_says(tmp_path: Path) -> None:
    """The page wears each arm's letter tag as a coloured square beside its name, not in the text, so the plan's
    rendered `name (A)` is read as `name`."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page, errors = rendered(_build("r5-kairo-round2", tmp_path), _KAIRO_PROSE_PROBE)
    assert errors == []
    plan = dataset.load_plan(KIT / "references" / "r5-kairo-round2.plan.json")
    report_input = dataset.report_input_of("r5")

    def written(text: str) -> str:
        return _said(re.sub(r" \([A-Z]\)", "", render(text, report_input, {})))

    slots = {slot.key: slot.text for slot in plan.decision.slots}
    wanted = {
        "headline": written(plan.headline),
        "lede": written(plan.lede),
        **{key: written(slots[key]) for key in ("recommendation", "why_not", "watch", "next")},
        "method": [[written(item.term), written(item.text)] for item in plan.method],
    }
    cards = page.pop("cards")
    drawn = {
        key: [[_said(t), _said(v)] for t, v in value] if key == "method" else _said(value)
        for key, value in page.items()
    }
    assert drawn == wanted
    assert cards == _example_rows(plan, report_input), "the cards are the rows the plan's examples query selects"


def _example_rows(plan: Any, report_input: Any) -> list[str]:
    """The trials the plan's examples section selects: every example of the arms and strata its query names, in the
    input's order, to its limit, as the plan checks read a query."""
    (section,) = [section for section in plan.sections if section.chart.query.source == "examples"]
    query = section.chart.query
    rows = [
        example.trial or example.case or "example"
        for example in report_input.examples
        if (example.arm is None or not query.arms or example.arm in query.arms)
        and (not query.strata or example.stratum in query.strata)
        and (not query.aspect or example.aspect in (None, query.aspect))
    ]
    return rows[: query.limit] if query.limit else rows


class _SectionHeaders(HTMLParser):
    """Each section header's eyebrow, headline and takeaway in a page source, as written: the text, or None where an
    element the page's script fills by id is empty."""

    def __init__(self) -> None:
        super().__init__()
        self.headers: list[dict[str, str | None]] = []
        self._field: str | None = None
        self._in_header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = (dict(attrs).get("class") or "").split()
        if tag == "header":
            self._in_header = True
            self.headers.append({"eyebrow": None, "headline": None, "takeaway": None})
        elif self._in_header and tag == "span" and "eyebrow" in classes:
            self._field = "eyebrow"
        elif self._in_header and tag == "h2":
            self._field = "headline"
        elif self._in_header and tag == "p" and "takeaway" in classes:
            self._field = "takeaway"

    def handle_endtag(self, tag: str) -> None:
        if tag == "header":
            self._in_header = False
        elif tag in ("span", "h2", "p"):
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._field is not None and data.strip():
            header = self.headers[-1]
            header[self._field] = (header[self._field] or "") + data


def test_the_kairo_sources_section_headers_are_its_plans() -> None:
    """Read off the page source, no browser: every evidence section's eyebrow is the plan's, and its headline and
    takeaway are too wherever the source writes them as text (the script draws the rest, checked in the browser)."""
    parsed = _SectionHeaders()
    parsed.feed((KIT / "references" / "r5-kairo-round2.src.html").read_text(encoding="utf-8"))
    by_eyebrow = {_said(header["eyebrow"]): header for header in parsed.headers if header["eyebrow"]}
    plan = dataset.load_plan(KIT / "references" / "r5-kairo-round2.plan.json")
    report_input = dataset.report_input_of("r5")
    compared = 0
    for section in plan.sections:
        eyebrow = _said(render(section.eyebrow, report_input, {}))
        assert eyebrow in by_eyebrow, f"the page has no section headed {eyebrow!r}"
        header = by_eyebrow[eyebrow]
        for field in ("headline", "takeaway"):
            written = header[field]
            if written is not None:
                assert _said(written) == _said(render(getattr(section, field), report_input, {})), (eyebrow, field)
                compared += 1
    assert compared >= 3, "the source writes at least the examples' headline and two takeaways as text"
