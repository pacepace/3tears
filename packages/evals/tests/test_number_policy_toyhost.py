"""One measure prints one way across a campaign's pages (number rule 1, owner ruling on set scope): the toy host's
field accuracy, a host's fraction on 0 to 1 (``value_range=(0, 1)``, no unit), is a rate on its code page, so its
written page reads it as a rate too, ``78.0%``, never the bare proportion ``0.780`` (rule 4: a rate carries its ``%``).
"""

from __future__ import annotations

import asyncio
from functools import cache

from packages.evals.tests.hot_path_support import campaign_id_of, toyhost_doc_capture, toyhost_from
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture
from threetears.evals.ops import report_read


@cache
def _written() -> str:
    """The toy campaign's written page, as Markdown: its report generated and stored, then read."""
    documents = asyncio.run(toyhost_doc_capture())
    host, campaign_id = toyhost_from(documents), campaign_id_of(documents)
    return report_read(host, campaign_id, TOYHOST_SCOPE, format="markdown", trials="table").body


@cache
def _code() -> str:
    """The same campaign's code page, as Markdown."""
    fixture = ops_fixture()
    return report_read(fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format="markdown").body


def test_field_accuracy_reads_as_a_rate_on_the_written_page() -> None:
    (row,) = [line for line in _written().splitlines() if line.startswith("| A · 256tok | $20 |")]
    assert "| 78.0% | 75.5 to 80.5% |" in row, row
    assert "0.780" not in _written()


def test_field_accuracy_prints_the_same_text_on_the_written_and_the_code_page() -> None:
    (code,) = [line for line in _code().splitlines() if line.startswith("| Field accuracy | 256tok |")]
    assert "| 78.0% | 75.5 to 80.5% |" in code, code
    (written,) = [line for line in _written().splitlines() if line.startswith("| A · 256tok | $20 |")]
    assert "| 78.0% | 75.5 to 80.5% |" in written, written
