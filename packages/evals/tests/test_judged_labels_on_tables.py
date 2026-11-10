"""A judged dimension's row on every table reads its measure's display name, marked judged.

Discodon's C5b re-shoot (2026-11-10, item 4): the Character v2 page's Contrasts, corrected-comparison statistics and
per-stratum tables, and a SearXNG page's All cases table, printed a judged row in lowercase ("persona holds character
(judged)", "research answers facets (judged)") beside measure rows that start with a capital ("Lost-call rate"), while
the page's charts and kit tables named the same dimension "Persona holds character". A row on a table is a label, and
every label starts with a capital, whatever its source
(:func:`~threetears.evals.analysis.report.parts.charting.measure_words`); so a judged row prints its measure's display
name, then its judged mark (and a score's scale), on each of those tables.

Kairo round 2's committed bundle carries the contrasts, the statistics and the strata; the measurement shape (s3), one
arm on each aspect, carries the All cases table.
"""

from __future__ import annotations

import re

import pytest

from threetears.evals.analysis import build_code_only_report
from threetears.evals.analysis.report.model import Report, TableBlock
from threetears.evals.analysis.report.parts.charting import measure_words
from threetears.evals.analysis.report.words import judged_heading, judged_label
from threetears.evals.analysis.report_input_build import report_input

from packages.evals.tests.analyst_support import shape_bundle
from packages.evals.tests.campaign_support import ASSEMBLED_AT
from packages.evals.tests.fixtures.report_inputs.dataset import kairo_bundle

#: A judged row's mark, after its name: ``(judged)``, or a score's ``(judged, 1–5)``.
_MARK = re.compile(r" \(judged(?:, [^)]+)?\)$")


def _judged_rows(report: Report, table: str) -> list[str]:
    """Every judged row's label on the page's table of that name."""
    (block,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == table]
    return [str(row["reading"].value) for row in block.rows if _MARK.search(str(row["reading"].value))]


def _pages(name: str) -> tuple[list[Report], dict[str, str]]:
    """Each aspect's own page over a campaign's bundle, and the display name its report input gives each judged
    dimension, by the name the dimension's mark follows on a table."""
    bundle = kairo_bundle() if name == "kairo" else shape_bundle(name)
    read = report_input(bundle)
    names = {
        judged_heading(dimension).removesuffix(" (judged)"): measure_words(read, dimension) for dimension in read.judged
    }
    pages = [build_code_only_report(aspect, assembled_at=ASSEMBLED_AT) for aspect in bundle.aspects]
    return pages, names


@pytest.mark.parametrize(
    ("campaign", "table"),
    [
        ("kairo", "comparisons"),
        ("kairo", "comparison_statistics"),
        ("kairo", "strata"),
        ("s3", "readings"),
    ],
)
def test_a_judged_row_reads_its_measure_s_display_name(campaign: str, table: str) -> None:
    pages, names = _pages(campaign)
    rows = [row for page in pages for row in _judged_rows(page, table)]
    assert rows, f"{campaign}'s {table} table holds a judged row"
    for row in rows:
        named = _MARK.sub("", row)
        assert named in names.values(), (row, names)
        assert named[:1].isupper(), row


def test_a_judged_label_is_its_heading_starting_with_a_capital() -> None:
    assert judged_label("persona.holds_character") == "Persona holds character (judged)"
    assert judged_heading("persona.holds_character") == "persona holds character (judged)"
