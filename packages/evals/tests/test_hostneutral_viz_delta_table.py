"""A delta table's "not drawn" line is a disclosure of its own, never folded into the author's caption.

The arm's one disclosure is carried on ``disclosures``, not on ``caption``. The toy capture's
delta table draws every row, so the toy host never reaches this line. The substitute is a synthetic
payload, compiled through :func:`~threetears.evals.analysis.viz.compiler.compile_chart`, comparing two builds
of an invoice extractor on three measures:

- one that draws;
- one categorical, which a relative axis cannot place;
- one whose baseline is zero, so it has no relative change.

No host type or vocabulary is in it.
"""

from __future__ import annotations

from typing import Any


from threetears.evals.analysis.viz.compiler import compile_chart


_NOT_DRAWN = (
    "2 of 3 metrics are not drawn (layout_kind, retries) — a relative axis cannot place a non-numeric value "
    "or a change from a zero baseline."
)


def _payload(**overrides: Any) -> dict[str, Any]:
    return {
        "a_label": "build-a",
        "b_label": "build-b",
        "caption": "build-b extracts faster.",
        "rows": [
            {"metric": "extract_ms", "a": 900.0, "b": 720.0, "unit": "ms"},
            {"metric": "layout_kind", "data_type": "categorical", "a": "table", "b": "list"},
            {"metric": "retries", "a": 0.0, "b": 2.0},
        ],
        **overrides,
    }


def test_the_undrawn_rows_are_named_on_a_disclosure_line_of_their_own() -> None:
    chart = compile_chart("delta_table", _payload())

    assert chart.disclosures == [_NOT_DRAWN]


def test_the_authors_caption_is_served_exactly_as_written() -> None:
    chart = compile_chart("delta_table", _payload())

    assert chart.caption == "build-b extracts faster."
    assert "not drawn" not in chart.caption


def test_the_values_table_still_counts_every_row() -> None:
    chart = compile_chart("delta_table", _payload())

    assert [row["metric"] for row in chart.rows] == ["extract_ms", "layout_kind", "retries"]


def test_a_table_whose_every_row_draws_discloses_nothing() -> None:
    chart = compile_chart(
        "delta_table", _payload(rows=[{"metric": "extract_ms", "a": 900.0, "b": 720.0, "unit": "ms"}])
    )

    assert chart.disclosures == []
