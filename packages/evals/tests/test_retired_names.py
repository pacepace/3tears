"""A name a retired generation left is gone from every root, and reads as unknown as on any module.

The analysis reporter kind is retired with the memo writer and the report planner it measured, and the run's cost cap
is read between cells. Pinned here:

- **Every name they left is gone,** the case bank, its curation and calibration, the kind's contract, launch names and
  kinds, the two generations' prompt constants, the memo writer's entry point and client settings, the host's tone
  register, and the in-cell cost cap's error and handle. No root exports one, and a read of one raises
  ``AttributeError`` as for any unknown name; the catalogue holds no ``reporter_*`` action.
- **The members they left are gone too:** the cell sink has no ``reserve`` or ``cost_cap_reached``, ``Ceilings`` has
  no generation ceiling, and the planner's call count is no name at all.
"""

from __future__ import annotations

import importlib

from threetears.evals.actions import eval_catalogue
from threetears.evals.kernel import CellSink
from threetears.evals.run import Ceilings

import pytest

#: Each root's names the retired generations and the in-cell cost cap took with them: never exported again.
REMOVED = {
    "threetears.evals.analysis": [
        "AS_RECORDED_MODEL",
        "AmbiguousPair",
        "CalibrationCase",
        "CalibrationCell",
        "CalibrationGroup",
        "CriterionDrift",
        "DimensionReading",
        "FrozenReporterCase",
        "LABEL_BANDS",
        "LabelCriterion",
        "LabelDirection",
        "LabelReading",
        "LabelVerdict",
        "PreparedReporter",
        "ReporterCalibration",
        "ReporterCase",
        "ReporterCaseBank",
        "ReporterCaseStore",
        "ReporterLabel",
        "BudgetedGenerator",
        "CallAdmission",
        "PreparedGeneration",
        "WriterMessageCheck",
        "estimate_analysis_generation",
        "first_request",
        "freeze_reporter_case",
        "frozen_case_receipt",
        "live_reporter_cases",
        "prepare_analysis_generation",
        "prompt_content_version",
        "render_memo_as_written",
        "reporter_calibration",
        "reporter_case_bank",
        "run_analysis_generation",
        "set_reporter_case_archived",
        # Removed after one deprecated minor that no final release carried.
        "AsRecordedReporterKind",
        "EVAL_ANALYSIS_GEN_DEFAULT",
        "EVAL_REPORT_PLAN_DEFAULT",
        "REPORTER_KIND",
        "REPORTER_KIND_CONTRACT",
        "ReporterKind",
        "ReporterOverlays",
        "analysis_gen_request_settings_for",
        "generate_analysis",
        "refuse_undeliverable_reporter_template",
        "reporter_cell_timeout_s",
        "reporter_kind_contract",
    ],
    "threetears.evals.ops": [
        "AmbiguousReporterPair",
        "FrozenReporterCase",
        "ReporterCaseEntry",
        "ReporterCaseFreeze",
        "ReporterCaseListing",
        "ReporterPrompts",
        "UnreadableReporterCase",
        "reporter_case_archive",
        "reporter_case_freeze",
        "reporter_cases_list",
        "REPORTER_UNHONOURED_LAUNCH_ARGUMENTS",
        "ReporterAdmitted",
        "ReporterModelResolver",
        "ReporterModelRole",
        "admit_reporter_launch",
        "plan_reporter_arm",
        "price_largest_reporter_call",
    ],
    "threetears.evals.kernel": ["ANALYSIS_GEN_PROMPT", "CellReservation", "CostCapReached"],
    "threetears.evals.kernel.host": ["StyleProfile", "ToneRegister"],
}


@pytest.mark.parametrize(("root", "name"), [(root, name) for root, names in REMOVED.items() for name in names])
def test_a_removed_name_is_exported_by_no_root_and_reads_as_unknown(root: str, name: str) -> None:
    module = importlib.import_module(root)
    assert name not in module.__all__
    with pytest.raises(AttributeError, match=f"has no attribute '{name}'"):
        getattr(module, name)


def test_the_catalogue_holds_no_reporter_action() -> None:
    names = {action.name for action in eval_catalogue().actions}
    assert not {name for name in names if name.startswith("reporter_")}
    assert {"writer_template_create", "writer_case_mint", "writer_case_archive"} <= names


def test_the_cell_sink_has_no_member_that_asked_the_cap_inside_a_cell() -> None:
    assert not hasattr(CellSink, "reserve") and not hasattr(CellSink, "cost_cap_reached")
    assert callable(CellSink.add_spend), "a call's cost is counted once it returns"


def test_the_ceilings_hold_no_generation_ceiling() -> None:
    assert not hasattr(Ceilings(), "generation_ceiling_s")


def test_the_planner_s_call_count_is_no_name() -> None:
    from threetears.evals.kernel import inner_ceilings

    for name in ("MAX_GENERATION_CALLS", "_PLANNER_CALLS", "generation_ceiling_s"):
        assert name not in inner_ceilings.__all__
        assert not hasattr(inner_ceilings, name)
