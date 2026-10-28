"""A stored report document read on its snapshot degrades rather than refuses; generation refuses (plan §2.3).

Each case starts from the third prototype's Kairo document, which builds strict with nothing degraded, and plants
one thing that no longer reads:

- a figure key that no longer resolves prints ``—``, and the page carries one note to regenerate;
- a block that no longer draws as its kind prints as a table of its values, under its caption and code's line;
- a block that draws in no form keeps its caption alone, with code's line;
- a caveat the document says is checked immaterial, which code does not prove, prints ``—`` for its checked line.

Strict refuses each, naming the field path. A stored document that no longer validates never reaches the builder:
its record is refused by its version first, with the remedy to regenerate.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.report_doc_support import converted_doc, snapshot_of
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.blocks import TableBlock
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report.doc_page.text import doc_page_text
from threetears.evals.analysis.report.doc_page.words import (
    DEGRADED_BLOCK,
    DEGRADED_PAGE,
    DEGRADED_TABLE,
    UNRESOLVED,
)
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import EvalReportDoc, ReportDoc
from threetears.evals.schema.versioning import REGENERABLE_SCHEMA_VERSION

#: The document every case plants one fault in.
_KAIRO = "prototype3/kairo"

#: A figure key the snapshot holds nothing for.
_GONE = "{reading.no-such-arm.change}"


def _input() -> ReportInput:
    return ReportInput.model_validate(snapshot_of(_KAIRO).report_input)


def _planted(change: Callable[[dict[str, Any]], None]) -> ReportDoc:
    doc = converted_doc(_KAIRO)
    change(doc)
    return ReportDoc.model_validate(doc)


def _read(change: Callable[[dict[str, Any]], None]) -> DocPage:
    return build(_planted(change), _input(), strict=False)


def _key_gone(doc: dict[str, Any]) -> None:
    doc["headline"] = f"Keep {{arm:September}}: {_GONE} is gone."


def _first_block(query: dict[str, Any] | None = None, **fields: Any) -> Callable[[dict[str, Any]], None]:
    def change(doc: dict[str, Any]) -> None:
        evidence = doc["claims"][0]["evidence"][0]
        evidence.update({**fields, "query": {**evidence["query"], **(query or {})}})

    return change


#: Bars of production cost drawn as a lattice, which draws configurations alone; a table of the same query still draws.
_UNDRAWN_KIND = _first_block(block="lattice")

#: The same block naming an arm the snapshot does not hold: no form of it draws.
_UNDRAWN = _first_block(query={"arms": ["no-such-arm"]})


def _checked_unproved(doc: dict[str, Any]) -> None:
    (judge,) = [caveat for caveat in doc["confidence"]["caveats"] if caveat["id"] == "judge_uncalibrated"]
    judge.update({"material_if": None, "checked": "immaterial"})


def test_the_unplanted_document_reads_with_nothing_degraded() -> None:
    page = build(ReportDoc.model_validate(converted_doc(_KAIRO)), _input(), strict=False)

    assert page.degraded == ""
    assert DEGRADED_PAGE not in doc_page_html(page)


def test_a_key_that_no_longer_resolves_prints_a_dash_and_the_page_one_note() -> None:
    page = _read(_key_gone)

    assert page.headline.startswith("Keep ") and page.headline.endswith(f": {UNRESOLVED} is gone.")
    assert "September" in page.headline and "{" not in page.headline
    assert page.degraded == DEGRADED_PAGE
    for written in (html.unescape(doc_page_html(page)), doc_page_markdown(page), doc_page_text(page)):
        assert written.count(DEGRADED_PAGE) == 1


def test_a_block_that_no_longer_draws_as_its_kind_prints_as_its_values_table() -> None:
    page = _read(_UNDRAWN_KIND)
    (evidence, *_rest) = page.claims[0].evidence

    assert evidence.chart is not None and isinstance(evidence.chart.chart, TableBlock)
    assert evidence.chart.chart.rows
    assert evidence.degraded == DEGRADED_TABLE
    assert evidence.caption.startswith("Production cost per 1,000 replies")
    assert page.degraded == DEGRADED_PAGE
    assert DEGRADED_TABLE in html.unescape(doc_page_html(page)) and DEGRADED_TABLE in doc_page_markdown(page)


def test_a_block_that_draws_in_no_form_keeps_its_caption_alone() -> None:
    page = _read(_UNDRAWN)
    (evidence, *_rest) = page.claims[0].evidence

    assert evidence.chart is None
    assert evidence.degraded == DEGRADED_BLOCK
    assert evidence.caption.startswith("Production cost per 1,000 replies")
    assert page.degraded == DEGRADED_PAGE
    assert evidence.caption in doc_page_text(page)


def test_a_caveat_checked_where_code_proves_nothing_prints_a_dash() -> None:
    page = _read(_checked_unproved)
    (line,) = [line for line in page.confidence.caveats if line.label == "The judge is not calibrated"]

    assert (line.checked, line.text) == (True, UNRESOLVED)
    assert page.degraded == DEGRADED_PAGE


@pytest.mark.parametrize(
    ("change", "path"),
    [
        (_key_gone, "headline"),
        (_UNDRAWN_KIND, "claims[0].evidence[0]"),
        (_UNDRAWN, "claims[0].evidence[0]"),
        (_checked_unproved, "confidence.caveats[0].checked"),
    ],
    ids=["key", "kind", "block", "checked"],
)
def test_generation_refuses_what_a_read_degrades(change: Callable[[dict[str, Any]], None], path: str) -> None:
    with pytest.raises(UnresolvableReference, match=re.escape(path)):
        build(_planted(change), _input(), strict=True)


def test_a_stored_document_of_another_version_is_refused_by_its_version_before_it_is_read() -> None:
    """Plan §2.3's third row: a document this build no longer reads is regenerated; its record never reaches the
    page."""
    with pytest.raises(ValidationError, match="regenerated"):
        EvalReportDoc.model_validate(
            {
                "schema_version": f"{REGENERABLE_SCHEMA_VERSION}.old",
                "scope_id": "uni-1",
                "snapshot_id": "snap",
                "campaign_id": None,
                "doc": converted_doc(_KAIRO),
                "ending": "stored",
                "generation": {},
                "created_by": "tester",
            }
        )
