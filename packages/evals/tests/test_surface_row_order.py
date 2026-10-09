"""The decision surface's one row order, wherever it is laid out (#645).

A reader short on time takes a table's top row for the recommendation. So the decision surface leads with
the control, marked as the reference every other arm is read against, follows with the other arms in
alphabetical order of their names, and says on the table that row order is not a ranking. The table-level
rule is pinned in ``test_surface_table.py``; this file pins that every other place the surface is laid out
follows it: the code-only report, the strata table, and the writer's ``cell_measures`` (which the stored
surface is a copy of, and whose order mints the writer's cell aliases).

Driven over the toy host's campaign. Its arms are named by chunk width (``chunk_tokens=256tok`` and
``chunk_tokens=1024tok``), so alphabetically the wide width comes first; declaring the narrow width the control is what makes control
first and name order disagree.
"""

from __future__ import annotations

import json

from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import cell_at
from threetears.evals.analysis.arms import arm_names
from threetears.evals.analysis.bundle import AnalysisContextBundle, assemble_context_bundle, bundle_decision_surface
from threetears.evals.analysis.cells import cell_ref, variant_of_cell_ref
from threetears.evals.analysis.generator import build_user_message
from threetears.evals.analysis.report import build_code_only_report, report_markdown
from threetears.evals.analysis.report.model import TableBlock
from threetears.evals.analysis.surface_table import REFERENCE_MARK, SURFACE_ORDER, SURFACE_ORDER_NO_CONTROL


def _bundle(*, control: int | None) -> AnalysisContextBundle:
    """The toy campaign's bundle, with the arm at chunk width ``control`` declared the control, or none."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    assert campaign.declared_design is not None, "the toy campaign declares a design"
    chosen = variant_of_cell_ref(cell_at(bundle, control)) if control is not None else None
    design = campaign.declared_design.model_copy(update={"control": chosen})
    return assemble_context_bundle(
        campaign.model_copy(update={"declared_design": design}), storage=storage, profile=profile
    )


def _surface_table(bundle: AnalysisContextBundle) -> TableBlock:
    report = build_code_only_report(
        bundle, measures=toyhost_profile().measures, assembled_at="2026-10-09T00:00:00+00:00"
    )
    (table,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "surface"]
    assert f"**Decision surface** ({table.order})" in report_markdown(report)
    return table


class TestTheWritersCellsFollowTheRule:
    def test_the_control_leads_cell_measures_and_is_the_first_alias(self) -> None:
        bundle = _bundle(control=TOYHOST_NARROW)
        narrow, wide = cell_at(bundle, TOYHOST_NARROW), cell_at(bundle, TOYHOST_WIDE)

        assert [cell_ref(c.variant_key, c.apparatus_class_id) for c in bundle.cell_measures] == [narrow, wide]
        shown = json.loads(build_user_message(bundle).split("\n", 1)[1])
        assert shown["cell_measures"][0]["cell"] == "c1"
        assert shown["cell_measures"][0]["variant_key"] == variant_of_cell_ref(narrow)

    def test_without_a_control_the_arms_are_in_name_order(self) -> None:
        """The wide width's name sorts before the narrow one's (``1024`` before ``256``), whatever the keys say."""
        bundle = _bundle(control=None)
        assert [cell_ref(c.variant_key, c.apparatus_class_id) for c in bundle.cell_measures] == [
            cell_at(bundle, TOYHOST_WIDE),
            cell_at(bundle, TOYHOST_NARROW),
        ]

    def test_the_frozen_surface_keeps_the_order(self) -> None:
        bundle = _bundle(control=TOYHOST_NARROW)
        assert bundle_decision_surface(bundle).cells == bundle.cell_measures


class TestTheCodeOnlyReportFollowsTheRule:
    def test_the_reference_row_leads_marked_and_the_rule_is_printed(self) -> None:
        bundle = _bundle(control=TOYHOST_NARROW)
        names = arm_names(bundle.variant_index)
        narrow, wide = (variant_of_cell_ref(cell_at(bundle, level)) for level in (TOYHOST_NARROW, TOYHOST_WIDE))
        table = _surface_table(bundle)
        assert table.order == SURFACE_ORDER
        assert [row["arm"] for row in table.rows] == [f"{names[narrow]} {REFERENCE_MARK}", names[wide]]

    def test_with_no_control_no_row_is_marked_and_the_rule_says_so(self) -> None:
        table = _surface_table(_bundle(control=None))
        assert table.order == SURFACE_ORDER_NO_CONTROL
        assert not any(REFERENCE_MARK in str(row["arm"]) for row in table.rows)
