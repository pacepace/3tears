"""A sentence read before its page exists is printed in that page's forms, and a table names a refused cell.

A campaign row's reason is frozen as text, which the campaign's fingerprint reads; a page states it again from the same
evidence with its figures as numbers (``reading_reason``), so a row's reason and a floor's beside it read in one form,
and the sentence read again is the one frozen. A frontier cell's own pass^k is a pass^k, which the page prints as it
prints every pass^k. A cell refused while a table is read from data is named by its table, row and column.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from packages.evals.tests.campaign_page_support import campaign_fixture
from packages.evals.tests.campaign_support import arms_of
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.campaign_readings import campaign_readings, reading_reason
from threetears.evals.analysis.lenses.frontier import PassHatKCellReading, pass_hat_k_cell_lines
from threetears.evals.analysis.report.blocks import TableBlock
from threetears.evals.analysis.report.forms import PageForms


async def test_a_row_s_reason_read_again_is_the_sentence_its_readings_froze() -> None:
    campaign, measured = await campaign_fixture("billing", "tone", "steady", margins={"billing": 70.0})
    bundles = {one.bundle.aspect: one.bundle for one in measured}
    read = campaign_readings(campaign, bundles, arms=arms_of(measured[0]))
    assert read.readings
    for row in read.readings:
        said = reading_reason(row, bundles[row.aspect], control_label=read.control_label)
        assert said.stored() == row.reason


async def test_a_row_s_reason_is_not_read_from_another_aspect_s_evidence() -> None:
    campaign, measured = await campaign_fixture("billing", "tone")
    bundles = {one.bundle.aspect: one.bundle for one in measured}
    read = campaign_readings(campaign, bundles, arms=arms_of(measured[0]))
    row = read.on_aspect("billing")[0]
    with pytest.raises(ValueError, match="holds no row of aspect 'billing'"):
        reading_reason(row, bundles["tone"], control_label=read.control_label)


def test_a_frontier_cell_s_own_pass_k_prints_as_the_page_prints_a_pass_k() -> None:
    cell = PassHatKCellReading(
        run_ids=["run-b"], headline=False, reason="other_served_model", detail="model b", pass_hat_k=0.625
    )
    forms = PageForms.off_page()
    [line] = [forms.said(said) for said in pass_hat_k_cell_lines(cell)]
    assert line.endswith(f"its own pass^k {forms.pass_k(TRIAL_PASS_NAME, 0.625, kind='trial').text} over 0 case(s)")
    assert "62.5%" in line


def test_a_cell_refused_in_a_table_read_from_data_names_its_table_row_and_column() -> None:
    with pytest.raises(ValidationError, match=r"table 'to_settle', row 0, column 'figure': the cell holding 0\.5"):
        TableBlock.model_validate(
            {
                "section": "scorecard",
                "name": "to_settle",
                "title": "To settle",
                "columns": [{"key": "figure", "header": "Figure"}],
                "rows": [{"figure": {"value": 0.5, "unit": "%"}}],
                "order": "as read",
                "total_rows": 1,
            }
        )
