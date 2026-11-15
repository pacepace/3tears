"""The identifiers a store, a log line or a wire holds by value are pinned, so no move or rename changes one unnoticed.

The re-root by role (#1199) moves public spellings, never a stored or sent identifier, and the structure moves after it
(S6, S8, S9) move modules. Each identifier below is written into a stored document, a job record, an error text, a log
record, a tool listing or an operator's command line, and is read back by a value someone else holds:

- ``doc_type`` and the class each one is read into (all stored document classes, found by their ``doc_type``
  literal), which a host's store and its audit walk key on;
- ``result_model``, the class name an action job stores beside its result (``run/job_work.py``), for every action's
  result type;
- the class name the runner writes into a cell's error text (``run/runner.py``), for every public exception type;
- action names and the MCP tool names a host mounts;
- ``contained_site``, ``module:qualname`` in every contained failure's log record (``schema/logs.py``), by the modules
  that contain a failure;
- the logger names an operator filters on, the package-data paths and the two command paths;
- the digests: no hashing module reads a module path or a qualified name, so no move changes a digest.

Each pin was taken from the tree on 2026-11-15. A deliberate change edits the pin in the same commit, with a
``MIGRATIONS.toml`` note for a stored value and a ``module`` line in the re-root map for a moved module (design-reroot-
cutover.md §6). The cut-over itself changes none of them except the contained sites of the six renamed modules, which
the map's ``module`` lines cover.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from importlib.resources import files
from pathlib import Path
from typing import NoReturn

import threetears.evals
from threetears.evals.actions import definition_drift_actions, eval_catalogue, standard_tools
from threetears.evals.actions.analyst import analyst_tools
from threetears.evals.actions.catalogue import read_only_tools
from threetears.evals.kernel.host import profile
from threetears.evals.ops import analyst
from threetears.evals.schema import EVAL_DOC_TYPES
from threetears.evals.schema.logs import PACKAGE_LOGGER

from packages.evals.tests.script_loader import load_script

role_roots = load_script("role_roots.py")

SOURCE = Path(threetears.evals.__file__).resolve().parent
MAP_PATH = SOURCE.parents[2] / "api" / "reroot-map.txt"

#: Every stored document class: ``doc_type`` → ``(defining module, class qualname)``.
STORED_CLASSES: dict[str, tuple[str, str]] = {
    "calibration_rating": ("threetears.evals.schema.models", "CalibrationRating"),
    "case_set": ("threetears.evals.schema.models", "CaseSet"),
    "eval_action_job": ("threetears.evals.run.job_work", "EvalActionJob"),
    "eval_analysis": ("threetears.evals.kernel.campaign", "EvalAnalysis"),
    "eval_analysis_attempt": ("threetears.evals.kernel.campaign", "EvalAnalysisAttempt"),
    "eval_analyst_gap": ("threetears.evals.kernel.analyst_gap", "EvalAnalystGap"),
    "eval_campaign": ("threetears.evals.kernel.campaign", "EvalCampaign"),
    "eval_cassette": ("threetears.evals.schema.models", "EvalCassette"),
    "eval_generation_slot": ("threetears.evals.kernel.campaign", "EvalGenerationSlot"),
    "eval_insight": ("threetears.evals.kernel.campaign", "EvalInsight"),
    "eval_judge_profile": ("threetears.evals.kernel.judge_profiles", "EvalJudgeProfile"),
    "eval_judge_temperature_check": ("threetears.evals.run.judge_temperature", "EvalJudgeTemperatureCheck"),
    "eval_launch": ("threetears.evals.run.launch_job", "EvalLaunch"),
    "eval_out_of_run_spend": ("threetears.evals.schema.out_of_run_spend", "OutOfRunSpend"),
    "eval_report_doc": ("threetears.evals.kernel.report_doc", "EvalReportDoc"),
    "eval_report_plan": ("threetears.evals.kernel.campaign", "EvalReportPlan"),
    "eval_report_session": ("threetears.evals.kernel.report_doc", "EvalReportSession"),
    "eval_report_snapshot": ("threetears.evals.kernel.report_snapshot", "EvalReportSnapshot"),
    "eval_result": ("threetears.evals.schema.models", "EvalResult"),
    "eval_run": ("threetears.evals.schema.models", "EvalRun"),
    "eval_second_judge_pass": ("threetears.evals.run.judge_second", "EvalSecondJudgePass"),
    "eval_sweep": ("threetears.evals.kernel.campaign", "EvalSweep"),
    "eval_template": ("threetears.evals.schema.models", "EvalTemplate"),
    "eval_test_case": ("threetears.evals.schema.models", "EvalTestCase"),
    "eval_trace": ("threetears.evals.schema.models", "EvalTrace"),
    "judge_config": ("threetears.evals.schema.models", "JudgeConfig"),
    "judge_config_tombstone": ("threetears.evals.schema.models", "JudgeConfigTombstone"),
    "rubric_dim": ("threetears.evals.schema.models", "CatalogRubricDim"),
    "rubric_dim_tombstone": ("threetears.evals.schema.models", "RubricDimTombstone"),
}

#: The ``__name__`` of every action's result type, which an action job stores as ``result_model``.
RESULT_MODELS: frozenset[str] = frozenset(
    """
    AnalysisDeleted AnalysisGenerationEstimate AnalysisLine AnalysisListing ArmsComparison Attested BarProposals
    BlockCatalog Breakdown CampaignCallLine CampaignExport CampaignLine CampaignListing CampaignOverview CaseRenamed
    CaseSetLine CaseSetListing CaveatListing ContrastListing DefinitionDrift DefinitionReset Dominance DraftChecked
    EvalSummary ExampleListing FrontierResult GapDigest GapNoted HistoryResult InsightDeleted InsightDetail
    InsightListing JobsStarted JobStatus JudgeCaseFreezeReport JudgeCaseListing JudgeConfigListing JudgeProfileListing
    JudgeProfileRecording JudgeRepeatReport JudgeTemperatureEstimate KeysChecked LaunchEstimate LeverEffects
    MeasureFigures MintedWriterCase OutOfRunSpendReport PivotTable PriceTradeoff ReadingsListing Regrade Remeasure
    RenderedBlock ReportDocument ResultDetail ResultListing ResultRated ResumableJobs RunDeleted RunLine RunListing
    RunsBisected RunsCompared ScoreExport SecondJudgeEstimate Sensitivity Settle SnapshotFrozen TemplateLine
    TemplateListing UndescribableArmsListing UnreadableRunsListing WriterCaseState WriterPairwiseReport
    WriterPromotion
    """.split()
)

#: The ``__name__`` of every exception type a public root exports, which the runner writes into a cell's error text.
EXCEPTION_NAMES: frozenset[str] = frozenset(
    """
    AccountExhaustedError AdmissionRefusedError ApparatusError BarRegistrationError BudgetStoppedError CallTimedOut
    CandidateKindDefect CandidatePreparationFailed CassetteCorrupt CassetteExhausted CassetteMiss CeilingRaisedError
    CompletionConformanceFailure CompletionOutsideJob CompletionProtocolDefect ConflictError DSLError DesignRefused
    EvalCellTimeout EvalJobTimeout EvalServiceError EveryCellApparatusFailedError GenerationError
    GoalCheckUnevaluable JobAwaitedByItsCall JobFenced JobsConformanceFailure KindContractError LeverCoordinateError
    MapFieldInSchema MeasureRegistrationError MisplacedDocument ModelUnservedError NegotiationConformanceFailure
    NotFoundError NothingMeasuredError PageNotPastAfter ProfileRegistrationError PromptContractError ProviderError
    ProviderRefusedError ReaderConformanceFailure RecordingRefused RefusedInput RegistrationError ReplayMiss
    RouteChanged SeedRefused SimulatorReplyInvalid SoundnessRefusal StorageError StoreCalledOnEventLoop StoreConflict
    StoreConformanceFailure TurnReplyRefused UndeclarableAxisError UnknownCandidateKind UnportableSchema
    UnreadableRecordError UnreadableResultKey UnreadableRunError UnresolvableReference UnsupportedSchemaError
    ValidationFailedError WorldConformanceError WorldRegistrationError WorldSessionError
    """.split()
)

#: Every action's name: what a caller sends as ``action``, and what a job record stores as its action.
ACTION_NAMES: frozenset[str] = frozenset(
    """
    analyses_list analyses_undescribable analysis_archive analysis_delete analysis_estimate analysis_generate
    analyst_gaps_list arms_compare bars_propose block_catalog block_render campaign_archive campaign_create
    campaign_export campaign_overview campaign_record campaign_runs_add campaign_runs_remove campaign_update
    campaigns_list capability_gap case_set_mint case_sets_list cases_rename caveats_list contrasts_list
    decision_sensitivity examples_list figure_keys_check insight_delete insight_get insights_list job_cancel job_poll
    job_resume jobs_resumable judge_case_author judge_cases_freeze judge_cases_list judge_configs_list
    judge_drift_check judge_profiles_list judge_profiles_record judge_second judge_second_estimate judge_temperature
    judge_temperature_estimate judge_template_create launch_estimate levers_effects measure_breakdown measure_read
    options_dominance price_tradeoff readings_list report_check report_read report_snapshot_freeze result_get
    result_rate result_rate_person result_regrade results_list run_archive run_delete run_get run_label run_launch
    run_remeasure runs_attest_definitions runs_bisect runs_compare runs_list runs_unreadable scope_export
    scope_frontier scope_history scope_out_of_run_spend scope_pivot settle_cases sweep_launch templates_list
    writer_case_archive writer_case_mint writer_pairwise writer_promotion writer_template_create
    definitions_drift definitions_reset
    """.split()
)

#: The modules that contain a failure, so ``contained_site`` names them (``module:qualname``) in operators' logs.
CONTAINED_SITE_MODULES: frozenset[str] = frozenset(
    f"threetears.evals.{module}"
    for module in """
    analysis.report.serialize_html harness.__main__ kernel.job_leases kernel.out_of_run kernel.provider ops.analysis
    ops.regressions ops.report_writer.editor ops.report_writer.loop ops.report_writer.write ops.sweeps providers.model
    quick.blind quick.cli quick.files quick.one_call run.job_work run.jobs run.judge run.judge_kind run.judge_service
    run.launch run.lifecycle run.ratings run.run_document run.runner testing.jobs_conformance
    testing.negotiation_conformance testing.store_conformance transports.fastmcp
    """.split()
)

#: The modules that compute a digest a store or a golden holds; none may read a module path or a qualified name.
DIGEST_MODULES = ("schema/hashing.py", "analysis/bundle/evidence.py", "run/definition_digests.py")


def _module_of(path: Path) -> str:
    parts = path.relative_to(SOURCE.parents[1]).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_every_stored_document_class_keeps_its_doc_type_class_and_module() -> None:
    found = {
        doc_type: (klass.__module__, klass.__qualname__) for doc_type, klass in role_roots.stored_classes().items()
    }
    assert found == STORED_CLASSES, (
        "a stored document class changed its doc_type, its name or its module: a doc_type or class name is stored and "
        "keyed on by hosts (a MIGRATIONS.toml note), and a moved module needs a module line in the re-root map and a "
        "host note; then update this pin"
    )
    assert set(EVAL_DOC_TYPES) == set(STORED_CLASSES)


def test_every_action_result_keeps_the_class_name_its_job_stores() -> None:
    found = {klass.__name__ for klass in role_roots.action_results().values()}
    assert found == RESULT_MODELS, (
        "an action's result type was renamed, added or removed: result_model stores its class name on action-job "
        "records, so a rename needs a MIGRATIONS.toml note; then update this pin"
    )


def test_every_public_exception_keeps_the_class_name_error_text_stores() -> None:
    found = set()
    for root in ("threetears.evals", *threetears.evals.PUBLIC_ROOTS):
        module = importlib.import_module(root)
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            if inspect.isclass(value) and issubclass(value, BaseException):
                found.add(value.__name__)
    assert found == EXCEPTION_NAMES, (
        "a public exception type was renamed, added or removed: the runner stores type(fault).__name__ in a cell's "
        "error text; then update this pin"
    )


def _no_store(_key: str) -> NoReturn:
    raise AssertionError("the definition-drift actions are built here only to read their names")


def test_action_and_tool_names_are_unchanged() -> None:
    actions = [*eval_catalogue().actions, *definition_drift_actions(_no_store)]
    assert {action.name for action in actions} == ACTION_NAMES
    assert [tool.name for tool in standard_tools()] == ["evals", "evals_admin"]
    assert [tool.name for tool in read_only_tools()] == ["evals"]
    assert [tool.name for tool in analyst_tools()] == ["evals_analyst"]


def _contains(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name in ("contain", "contained"):
                return True
    return False


def test_the_modules_naming_a_contained_site_are_pinned() -> None:
    found = {_module_of(path) for path in sorted(SOURCE.rglob("*.py")) if _contains(path)}
    assert found == CONTAINED_SITE_MODULES, (
        "a module that contains a failure moved, appeared or stopped containing: contained_site is module:qualname "
        "in operators' logs, so a move needs a module line in the re-root map; then update this pin"
    )


def test_each_renamed_module_naming_a_contained_site_has_a_module_line() -> None:
    lines = [row.split("\t") for row in MAP_PATH.read_text(encoding="utf-8").splitlines() if not row.startswith("#")]
    moved = {old: new for _epoch, record, old, new, _note in lines if record == "module"}
    for module in sorted(CONTAINED_SITE_MODULES):
        for old, new in role_roots.MODULE_RENAMES.items():
            if module == old or module.startswith(f"{old}."):
                assert moved.get(old) == new, f"{module} is renamed with no module line in {MAP_PATH.name}"


def test_the_logger_names_operators_filter_on_are_unchanged() -> None:
    assert PACKAGE_LOGGER == "threetears.evals"
    assert analyst.log.name == "threetears.evals.analyst"
    assert profile.log.name == "threetears.evals.kernel.host.profile"


def test_the_package_data_paths_and_command_paths_resolve() -> None:
    literals = set()
    for path in sorted(SOURCE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "files"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.startswith("threetears.evals")
            ):
                literals.add(node.args[0].value)
    assert literals == {"threetears.evals.analysis.analyst", "threetears.evals.analysis.report"}
    for package in literals:
        assert files(package).is_dir(), package
    for command in ("threetears.evals.__main__", "threetears.evals.harness.__main__"):
        assert importlib.util.find_spec(command) is not None, f"python -m {command.removesuffix('.__main__')} is gone"


def test_no_digest_reads_a_module_path_or_a_qualified_name() -> None:
    for relative in DIGEST_MODULES:
        path = SOURCE / relative
        reads = [
            f"{relative}:{node.lineno}"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
            if isinstance(node, ast.Attribute) and node.attr in ("__module__", "__qualname__")
        ]
        assert not reads, f"a digest module reads a module path or qualified name, which a move would change: {reads}"
