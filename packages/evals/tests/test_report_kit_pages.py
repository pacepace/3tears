"""The report's pages are drawn in the eval report kit's visual system, and pass the kit's render checks.

- **One kit.** The package ships the kit's ``kit.css`` and ``kit.js`` as package data beside the serializer, byte for
  byte the documentation copy in ``docs/eval-report-kit/kit/`` that ``scripts/build.py`` inlines into the reference
  memos, so the pages and the references never drift apart. The page's stylesheet opens with the kit's.
- **The references' fonts.** The page's head carries the same three font links as every reference memo's source,
  verbatim.
- **The render checks.** Each of the four shape campaigns' pages (``report_shapes_support.SHAPES``) passes
  ``scripts/check.py`` at 1200 and 390 px, light and dark: no console errors, no sideways scroll, nothing past the
  right edge outside a scrolling frame, no overlapping chart labels (GENERATOR.md §8, items 1-3). That check drives
  headless Chromium, so it carries the ``integration`` marker; where Chromium cannot launch it skips and says why.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest

from threetears.evals.analysis import campaign_report_html
from packages.evals.tests.page_contract_support import FONT_LINK_MARKUP
from packages.evals.tests.report_kit_support import KIT, chromium_unavailable, render_checked
from packages.evals.tests.report_shapes_support import SHAPES


@pytest.mark.parametrize("name", ["kit.css", "kit.js"])
def test_the_package_ships_the_kit_byte_for_byte(name: str) -> None:
    shipped = (files("threetears.evals.analysis.report") / "kit" / name).read_bytes()
    assert shipped == (KIT / "kit" / name).read_bytes(), (
        f"the package's kit/{name} differs from docs/eval-report-kit/kit/{name}: copy the documentation copy over it"
    )


def test_a_page_opens_its_stylesheet_with_the_kit_and_carries_the_references_font_links() -> None:
    page = campaign_report_html(SHAPES["s4"]().page())
    kit = (KIT / "kit" / "kit.css").read_text(encoding="utf-8")
    assert f"<style>{kit}" in page
    assert page.index(FONT_LINK_MARKUP) < page.index("<style>") < page.index("</head>")
    for source in sorted((KIT / "references").glob("*.src.html")):
        assert FONT_LINK_MARKUP in source.read_text(encoding="utf-8"), source.name


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_a_shape_campaign_s_page_passes_the_render_checks(name: str, tmp_path: Path) -> None:
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = tmp_path / f"{name}.html"
    page.write_text(campaign_report_html(SHAPES[name]().page()), encoding="utf-8")
    checked, runs = render_checked(page, tmp_path / "screenshots")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert len(runs) == 4 and all(line.startswith("ok") for line in runs), checked.stdout
