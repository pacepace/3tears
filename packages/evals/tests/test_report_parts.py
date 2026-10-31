"""A table cell drawn as the kit's status pill or arm chip, by the parts every page writer shares.

A cell's ``status`` draws its text as the kit's pill (the colour, the icon and the word, never a colour alone); its
``tag`` draws its text as an arm's chip (the letter tag on the arm's identity colour, then the name). Both are
optional and default to nothing, so a cell that sets neither is drawn exactly as before: that half is held by the
memo pages' own tests, which render byte for byte as they did. Markdown has no colour, so a status reads as its word
and a chip as its tag and name.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.report.blocks import TableBlock, TableCell, TableColumn, table_row
from threetears.evals.analysis.report.html_parts import pill_html, table_html
from threetears.evals.analysis.report.md_parts import table_lines


def _options(*rows: dict[str, TableCell]) -> TableBlock:
    return TableBlock(
        section="contrasts",
        name="options",
        title="Options",
        columns=[TableColumn(key="arm", header="Arm"), TableColumn(key="verdict", header="Verdict")],
        rows=list(rows),
        order="as given",
        total_rows=len(rows),
    )


def _row(arm: TableCell, verdict: TableCell) -> dict[str, TableCell]:
    return table_row({"arm": arm, "verdict": verdict})


class TestACellDrawsAPillOrAChip:
    def test_a_status_draws_the_cells_words_as_the_kits_pill(self) -> None:
        page = table_html(_options(_row(TableCell(value="September"), TableCell(value="Worse", status="bad"))))
        assert '<td><span class="pill bad"><span aria-hidden="true">✕</span>Worse</span></td>' in page

    @pytest.mark.parametrize(("status", "icon"), [("good", "✓"), ("warn", "!"), ("bad", "✕"), ("neutral", "–")])
    def test_each_status_carries_the_kits_icon_beside_its_word(self, status: str, icon: str) -> None:
        assert pill_html(status, "word") == (  # type: ignore[arg-type]
            f'<span class="pill {status}"><span aria-hidden="true">{icon}</span>word</span>'
        )

    def test_a_tag_draws_the_cells_words_as_the_arms_chip_on_its_colour(self) -> None:
        page = table_html(_options(_row(TableCell(value="Spark", tag="A", color="--s1"), TableCell(value="Better"))))
        assert (
            '<td><span class="armchip"><span class="tag" style="background:var(--s1);color:var(--on-fill)">A</span>'
            "Spark</span></td>"
        ) in page

    def test_a_chip_with_no_colour_is_the_kits_ink_gray_with_its_letter_on_the_surface(self) -> None:
        page = table_html(_options(_row(TableCell(value="September", tag="P"), TableCell(value="Control"))))
        assert '<span class="tag" style="background:var(--ink-2);color:var(--surface)">P</span>September' in page

    def test_the_words_inside_a_pill_or_a_chip_are_escaped(self) -> None:
        hostile = "<script>alert(1)</script>"
        page = table_html(
            _options(_row(TableCell(value=hostile, tag="A", color="--s1"), TableCell(value=hostile, status="warn")))
        )
        assert "<script>" not in page and page.count("&lt;script&gt;alert(1)&lt;/script&gt;") == 2

    def test_a_cell_setting_neither_draws_as_it_always_did(self) -> None:
        page = table_html(_options(_row(TableCell(value="Spark"), TableCell(value="Better"))))
        assert "<td>Spark</td><td>Better</td>" in page
        assert "pill" not in page and "armchip" not in page


class TestMarkdownSaysItInWords:
    def test_a_status_reads_as_its_word_and_a_chip_as_its_tag_and_name(self) -> None:
        lines = table_lines(
            _options(_row(TableCell(value="Spark", tag="A", color="--s1"), TableCell(value="Worse", status="bad")))
        )
        assert lines[-1] == "| A · Spark | Worse |"

    def test_a_cell_setting_neither_prints_as_it_always_did(self) -> None:
        assert table_lines(_options(_row(TableCell(value="Spark"), TableCell(value="Better"))))[-1] == (
            "| Spark | Better |"
        )


class TestACellRefusesAChipItCannotDraw:
    def test_a_colour_with_no_tag_is_refused(self) -> None:
        with pytest.raises(ValueError, match="colour"):
            TableCell(value="Spark", color="--s1")

    def test_a_pill_and_a_chip_on_one_cell_are_refused(self) -> None:
        with pytest.raises(ValueError, match="pill"):
            TableCell(value="Spark", tag="A", status="good")

    @pytest.mark.parametrize("tag", ["", "a", "A1", "<b>"])
    def test_a_tag_is_capital_letters_alone(self, tag: str) -> None:
        with pytest.raises(ValueError):
            TableCell(value="Spark", tag=tag)

    def test_a_colour_is_a_kit_token(self) -> None:
        with pytest.raises(ValueError):
            TableCell(value="Spark", tag="A", color="red")

    def test_a_status_is_one_of_the_kits(self) -> None:
        with pytest.raises(ValueError):
            TableCell(value="Spark", status="great")  # type: ignore[arg-type]
