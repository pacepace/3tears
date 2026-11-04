"""A name a retired generation left keeps importing for one minor, warns on every read, and launches nothing.

The analysis reporter kind is retired with the memo writer and the report planner it measured. Pinned here:

- **The case bank, its curation, its calibration read and its three actions are gone,** and so are the kind's case,
  label and memo-rendering names. No root exports them, a read of one raises ``AttributeError`` as for any unknown
  name, and the catalogue holds no ``reporter_*`` action.
- **The contract and launch names a host binds at import still import**, each read warning with a
  ``DeprecationWarning`` that names the ``report_writer`` kind replacing it and says the next minor removes it,
  attributed to the line that read it.
- **None of them makes the kind launchable.** The contract holds no overlay and seats nothing, whatever seats a host
  passes; the template check, the three launch operations and building either reporter kind refuse, naming
  ``writer_template_create``. ``REPORTER_KIND`` still names the kind a stored template declares, and the cell ceiling
  keeps its answer (``test_wall_clock_ceilings``).

The report planner's generation path is retired too, and the memo writer's prompt and client settings with it:

- **The planner's generation names are gone** (``prepare_analysis_generation``, ``run_analysis_generation``,
  ``estimate_analysis_generation``, ``PreparedGeneration``, ``BudgetedGenerator``).
- **The two prompt constants a host seeds a slot from by name still resolve**, through ``importlib`` as a host's
  registry reads them, each to one line saying the generation it steered is retired and never to its old prompt, and
  each read warns naming the slot to drop. ``EVAL_REPORT_PLAN_DEFAULT``, which no final release exported, is served
  but not in ``__all__``.
- **The memo writer's client settings** still answer what they did, warning that the report writer runs on the host's
  analyst client under ``AnalysisGeneration.limits``.

The memo writer itself is retired:

- **Its prompt, its first request and its prompt version are gone** (``first_request``, ``prompt_content_version``,
  ``CallAdmission``, and the kernel's ``ANALYSIS_GEN_PROMPT`` slot id), and no ``SEED_PROMPTS`` entry seeds
  ``eval_analysis_gen`` (``test_prompt_catalogue``).
- **``generate_analysis`` still imports**, warning that ``analysis_generate`` writes a campaign's report now, and every
  call refuses with ``ValidationFailedError`` naming ``analysis_generate`` before its client is asked anything.

The run's cost cap is read between cells, and the names that asked it inside one are retired:

- **``CostCapReached`` still imports from the kernel** and warns that nothing raises it; a host's ``except`` clause
  still compiles. ``CellSink.reserve`` and ``CellSink.cost_cap_reached`` stay on the protocol and warn when the
  engine's sink is asked (``test_cell_width_and_spend``), naming ``add_spend``; no engine path calls either.
- **``Ceilings.generation_ceiling_s``**, the report planner's ceiling, still answers what it did and warns naming
  ``writer_ceiling_s``. The planner's call count is no public name.
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
from threetears.evals.schema.models import JudgedArtifact

#: Each root's names the reporter cases took with them: never exported again.
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
    "threetears.evals.kernel": ["ANALYSIS_GEN_PROMPT"],
}

#: Each root's names kept for one minor: what a host binds at import.
SHIMMED = {
    "threetears.evals.analysis": [
        "AsRecordedReporterKind",
        "REPORTER_KIND",
        "REPORTER_KIND_CONTRACT",
        "ReporterKind",
        "ReporterOverlays",
        "refuse_undeliverable_reporter_template",
        "reporter_cell_timeout_s",
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


def test_the_kind_name_is_the_one_a_stored_reporter_template_declares() -> None:
    assert _read("threetears.evals.analysis", "REPORTER_KIND") == "analysis_reporter"


def test_building_either_reporter_kind_refuses_naming_the_writer_kind() -> None:
    generating = _read("threetears.evals.analysis", "ReporterKind")
    recorded = _read("threetears.evals.analysis", "AsRecordedReporterKind")
    builds = [
        lambda: generating(
            prompt="p", prompt_id="eval_analysis_gen", prompt_version=None, client=None, model="m", host=None
        ),
        recorded,
    ]
    for build, name in zip(builds, ("ReporterKind", "AsRecordedReporterKind"), strict=True):
        with pytest.raises(ValidationFailedError, match=f"{name}: the analysis reporter kind is retired") as refused:
            build()
        assert refused.value.remedy is not None and "writer_template_create" in refused.value.remedy


def test_the_reporter_kinds_keep_the_shape_a_host_s_annotations_name() -> None:
    for name in ("ReporterKind", "AsRecordedReporterKind"):
        kind = _read("threetears.evals.analysis", name)
        assert kind.judged_artifact is JudgedArtifact.DOCUMENT
        assert kind.max_concurrent_cells is None
        assert callable(kind.prepare) and callable(kind.invoke)


def test_the_launch_types_keep_their_shapes_for_a_host_s_annotations() -> None:
    unhonoured = _read("threetears.evals.ops", "REPORTER_UNHONOURED_LAUNCH_ARGUMENTS")
    admitted = _read("threetears.evals.ops", "ReporterAdmitted")

    assert unhonoured == frozenset({"simulator_model", "cassette_mode", "n_variations"})
    assert set(admitted.__dataclass_fields__) == {"preset", "prompt", "candidate_model", "judge_model", "cases"}


def test_a_name_neither_kept_nor_exported_is_unknown_as_on_any_module() -> None:
    for root in SHIMMED:
        with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
            getattr(importlib.import_module(root), "no_such_name")


# --- the report planner and the memo writer ------------------------------------------------------------------------

#: The analysis root's names the two retired generations left a host, each with the replacement its warning names.
GENERATION_SHIMS = {
    "EVAL_ANALYSIS_GEN_DEFAULT": "drop the eval_analysis_gen slot",
    "EVAL_REPORT_PLAN_DEFAULT": "drop the eval_report_plan slot",
    "analysis_gen_request_settings_for": "AnalysisGeneration.limits (WriterLimits)",
    "generate_analysis": "analysis_generate writes a campaign's report",
}


@pytest.mark.parametrize("name", sorted(GENERATION_SHIMS))
def test_a_retired_generation_s_name_warns_naming_its_replacement(name: str) -> None:
    with pytest.warns(DeprecationWarning) as caught:
        getattr(importlib.import_module("threetears.evals.analysis"), name)
    (warning,) = caught
    message = str(warning.message)
    assert message.startswith(f"threetears.evals.analysis.{name} is retired and leaves in the next minor: ")
    assert GENERATION_SHIMS[name] in message


def test_the_planner_s_prompt_constant_is_served_but_not_exported_and_the_memo_writer_s_stays_exported() -> None:
    root = importlib.import_module("threetears.evals.analysis")
    assert "EVAL_REPORT_PLAN_DEFAULT" not in root.__all__, "no final release exported it"
    assert {"EVAL_ANALYSIS_GEN_DEFAULT", "analysis_gen_request_settings_for", "generate_analysis"} <= set(root.__all__)


@pytest.mark.parametrize(("name", "slot"), [("EVAL_REPORT_PLAN_DEFAULT", "eval_report_plan"),
                                            ("EVAL_ANALYSIS_GEN_DEFAULT", "eval_analysis_gen")])  # fmt: skip
def test_a_prompt_constant_a_host_seeds_by_name_is_one_retired_line(name: str, slot: str) -> None:
    """As a host's registry reads a seed (``importlib`` and ``getattr``): a string a slot can hold, one line saying
    the generation is retired, never the prompt it was."""
    seed = _read("threetears.evals.analysis", name)
    assert isinstance(seed, str) and seed.startswith("Retired: ") and "\n" not in seed
    assert slot in seed and len(seed) < 200


def test_an_import_of_the_planner_s_prompt_constant_warns_at_the_importing_line() -> None:
    with pytest.warns(DeprecationWarning, match="EVAL_REPORT_PLAN_DEFAULT is retired") as caught:
        from threetears.evals.analysis import EVAL_REPORT_PLAN_DEFAULT
    assert EVAL_REPORT_PLAN_DEFAULT.startswith("Retired: ")
    assert caught[0].filename == __file__


def test_the_memo_writer_s_settings_still_answer_what_they_did() -> None:
    settings = _read("threetears.evals.analysis", "analysis_gen_request_settings_for")
    built = settings(answer_budget_tokens=4000, reasoning_headroom_tokens=2000)
    assert (built.max_tokens, built.reasoning_effort, built.strict_output) == (6000, "medium", True)


class _UntouchedClient:
    """A client the retired memo writer is handed, recording any request it is asked to make."""

    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []

    async def generate(self, **request: Any) -> Any:
        self.asked.append(request)
        raise AssertionError("the retired memo writer asked its client for a completion")


def test_the_memo_writer_refuses_every_call_naming_analysis_generate_before_its_client_is_asked() -> None:
    generate_analysis = _read("threetears.evals.analysis", "generate_analysis")
    client = _UntouchedClient()

    with pytest.raises(ValidationFailedError, match="the memo writer is retired") as refused:
        asyncio.run(
            generate_analysis(
                cast(Any, SimpleNamespace()),
                prompt="any prompt",
                model="any-model",
                client=client,
                prompt_id="eval_analysis_gen",
                bundle_assembled_at="2026-11-04T00:00:00+00:00",
                profile=cast(Any, SimpleNamespace()),
            )
        )

    assert "analysis_generate" in str(refused.value) and "analysis_generate" in (refused.value.remedy or "")
    assert client.asked == []


# --- the in-cell cost cap and the report planner's ceiling ---------------------------------------------------------


def test_cost_cap_reached_still_imports_from_the_kernel_and_warns_that_nothing_raises_it() -> None:
    root = importlib.import_module("threetears.evals.kernel")
    assert "CostCapReached" in root.__all__
    with pytest.warns(DeprecationWarning) as caught:
        retired = root.CostCapReached
    (warning,) = caught
    message = str(warning.message)
    assert message.startswith("threetears.evals.kernel.CostCapReached is retired and leaves in the next minor: ")
    assert "read between cells" in message and "drop the except clause" in message
    assert warning.filename == __file__
    assert issubclass(retired, Exception), "a host's except clause still compiles"
    assert retired().detail == "the run's cost cap has no room for another paid call"


def test_an_import_of_cost_cap_reached_warns_at_the_importing_line() -> None:
    with pytest.warns(DeprecationWarning, match="CostCapReached is retired") as caught:
        from threetears.evals.kernel import CostCapReached
    assert issubclass(CostCapReached, Exception)
    assert caught[0].filename == __file__


def test_the_cell_sink_keeps_both_retired_members_for_a_host_s_annotations() -> None:
    """The engine's sink answers them as it did and warns (``test_cell_width_and_spend``); no engine path calls them."""
    from threetears.evals.kernel import CellReservation, CellSink

    assert callable(CellSink.reserve) and callable(CellSink.cost_cap_reached)
    assert callable(CellReservation.settle) and callable(CellReservation.release)


def test_the_generation_ceiling_still_answers_and_warns_naming_the_writer_ceiling() -> None:
    from threetears.evals.run import Ceilings

    with pytest.warns(DeprecationWarning) as caught:
        ceiling = Ceilings().generation_ceiling_s
    (warning,) = caught
    message = str(warning.message)
    assert message.startswith(
        "threetears.evals.run.Ceilings.generation_ceiling_s is retired and leaves in the next minor: "
    )
    assert "Ceilings().writer_ceiling_s" in message
    assert warning.filename == __file__
    assert ceiling(request_s=lambda _max_tokens: 100.0, generator_max_tokens=1) == 400.0, "four planner requests"


def test_the_planner_s_call_count_is_no_public_name() -> None:
    from threetears.evals.kernel import inner_ceilings

    assert "MAX_GENERATION_CALLS" not in inner_ceilings.__all__
    assert not hasattr(inner_ceilings, "MAX_GENERATION_CALLS")
