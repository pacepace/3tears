"""A name a retired generation left keeps importing for one minor, warns on every read, and launches nothing.

The analysis reporter kind is retired with the memo writer and the report planner it measured. Pinned here:

- **The case bank, its curation, its calibration read and its three actions are gone.** No root exports them, a read
  of one raises ``AttributeError`` as for any unknown name, and the catalogue holds no ``reporter_*`` action.
- **The contract and launch names a host binds at import still import**, each read warning with a
  ``DeprecationWarning`` that names the ``report_writer`` kind replacing it and says the next minor removes it,
  attributed to the line that read it.
- **None of them makes the kind launchable.** The contract holds no overlay and seats nothing, whatever seats a host
  passes; the template check and the three launch operations refuse, naming ``writer_template_create``.
"""

from __future__ import annotations

import asyncio
import importlib
import warnings
from types import SimpleNamespace
from typing import Any, cast

import pytest

from threetears.evals.actions import eval_catalogue
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.host import KindContract
from threetears.evals.run import LaunchRequest
from threetears.evals.schema import EvalTemplate

#: Each root's names the reporter cases took with them: never exported again.
REMOVED = {
    "threetears.evals.analysis": [
        "AmbiguousPair",
        "CalibrationCase",
        "CalibrationCell",
        "CalibrationGroup",
        "CriterionDrift",
        "DimensionReading",
        "FrozenReporterCase",
        "LabelReading",
        "ReporterCalibration",
        "ReporterCaseBank",
        "ReporterCaseStore",
        "freeze_reporter_case",
        "frozen_case_receipt",
        "live_reporter_cases",
        "reporter_calibration",
        "reporter_case_bank",
        "set_reporter_case_archived",
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
    ],
}

#: Each root's names kept for one minor: what a host binds at import.
SHIMMED = {
    "threetears.evals.analysis": [
        "REPORTER_KIND_CONTRACT",
        "ReporterOverlays",
        "refuse_undeliverable_reporter_template",
        "reporter_kind_contract",
    ],
    "threetears.evals.ops": [
        "REPORTER_UNHONOURED_LAUNCH_ARGUMENTS",
        "ReporterAdmitted",
        "ReporterModelResolver",
        "ReporterModelRole",
        "admit_reporter_launch",
        "plan_reporter_arm",
        "price_largest_reporter_call",
    ],
}

SHIMS = [(root, name) for root, names in SHIMMED.items() for name in names]


def _read(root: str, name: str) -> Any:
    """``root.name`` as a host reads it, with the one warning the read raises."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = getattr(importlib.import_module(root), name)
    (warning,) = caught
    assert warning.category is DeprecationWarning
    assert warning.filename == __file__, "attributed to the line that read the name, not to the shim"
    return value


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


@pytest.mark.parametrize(("root", "name"), SHIMS)
def test_a_shimmed_name_is_still_exported_and_warns_naming_its_replacement(root: str, name: str) -> None:
    assert name in importlib.import_module(root).__all__
    with pytest.warns(DeprecationWarning) as caught:
        getattr(importlib.import_module(root), name)
    (warning,) = caught
    message = str(warning.message)
    assert message.startswith(f"{root}.{name} is retired and leaves in the next minor: ")
    assert "report_writer kind (writer_template_create" in message


def test_an_import_of_a_shimmed_name_warns_at_the_importing_line() -> None:
    with pytest.warns(DeprecationWarning, match="admit_reporter_launch is retired") as caught:
        from threetears.evals.ops import admit_reporter_launch
    assert callable(admit_reporter_launch)
    assert caught[0].filename == __file__


def test_the_contract_holds_no_overlay_and_seats_nothing_whatever_the_host_passes() -> None:
    contract = _read("threetears.evals.analysis", "REPORTER_KIND_CONTRACT")
    build = _read("threetears.evals.analysis", "reporter_kind_contract")

    assert contract == KindContract("analysis_reporter")
    assert (contract.overlays, contract.spec, contract.seats) == (None, None, None)
    assert build(host_seats=["host.rig"]) == contract


def test_the_template_check_refuses_every_reporter_template_naming_the_writer_kind() -> None:
    refuse = _read("threetears.evals.analysis", "refuse_undeliverable_reporter_template")
    template = EvalTemplate(
        id="memo-reporter",
        scope_id="s",
        name="Memo reporter",
        intent="Judge the memo a campaign got.",
        candidate_kind="analysis_reporter",
    )

    with pytest.raises(
        ValidationFailedError, match="'memo-reporter': the analysis reporter kind is retired"
    ) as refused:
        refuse(template)
    assert refused.value.remedy is not None and "writer_template_create" in refused.value.remedy


def test_each_launch_operation_refuses_naming_the_writer_kind() -> None:
    # The refusal reads only the arm's template id, so the arm is that alone.
    request = cast(LaunchRequest, SimpleNamespace(template=SimpleNamespace(id="memo-reporter")))
    admit = _read("threetears.evals.ops", "admit_reporter_launch")
    plan = _read("threetears.evals.ops", "plan_reporter_arm")
    price = _read("threetears.evals.ops", "price_largest_reporter_call")
    hooks: dict[str, object] = {"prompts": object(), "resolve_model": lambda role, named: "m"}

    calls = [
        lambda: admit(request, storage=object(), **hooks),
        lambda: plan(object(), request, **hooks),
        lambda: asyncio.run(price(object(), request, **hooks)),
    ]
    for call in calls:
        with pytest.raises(ValidationFailedError, match="'memo-reporter': the analysis reporter kind is retired") as e:
            call()
        assert e.value.remedy is not None and "writer_template_create" in e.value.remedy


def test_the_launch_types_keep_their_shapes_for_a_host_s_annotations() -> None:
    unhonoured = _read("threetears.evals.ops", "REPORTER_UNHONOURED_LAUNCH_ARGUMENTS")
    admitted = _read("threetears.evals.ops", "ReporterAdmitted")

    assert unhonoured == frozenset({"simulator_model", "cassette_mode", "n_variations"})
    assert set(admitted.__dataclass_fields__) == {"preset", "prompt", "candidate_model", "judge_model", "cases"}


def test_a_name_neither_kept_nor_exported_is_unknown_as_on_any_module() -> None:
    for root in SHIMMED:
        with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
            getattr(importlib.import_module(root), "no_such_name")
