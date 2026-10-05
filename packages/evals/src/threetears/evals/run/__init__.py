"""The engine's run package: launching and executing a run, judging it, metering it, and storing it.

Of the engine it imports only itself and :mod:`threetears.evals.contracts`; what it needs from a host arrives
through the ports a host implements.

**This module is the package's public root.** A host imports from here and from no module below
it, and only the names in ``__all__``; ``tests/test_package_matrix.py`` holds that, and
``tests/test_public_surface_is_closed.py`` holds every engine type a public signature hands a host to
being exported from a public root. A ``# debt:`` comment on an export names what retires it. Code
inside the package imports its own modules directly.
"""

from __future__ import annotations

from threetears.evals.run.authoring import (
    RUBRIC_DIM_SERVER_FIELDS,
    TEMPLATE_SERVER_FIELDS,
    create_judge_config,
    create_rubric_dim,
    create_template,
    delete_judge_config,
    delete_rubric_dim,
    get_judge_config,
    get_rubric_dim,
    get_template,
    list_judge_configs,
    list_rubric_dims,
    list_templates,
    refuse_stale_presumptions,
    update_judge_config,
    update_rubric_dim,
    update_template,
    validated_kind_spec,
)
from threetears.evals.run.curation import (
    delete_analysis,
    delete_insight,
    delete_result,
    delete_run,
    load_run_as_listed,
    require_delete_confirmation,
    set_analysis_archived,
    set_run_archived,
)
from threetears.evals.run.definition_seed import load_seed_corpus, seed_eval_definitions
from threetears.evals.run.fidelity import FidelityContract, callers_missing_the_constructor, resolve_constructor
from threetears.evals.run.jobs import (
    AdmissionTicket,
    EvalJobManager,
    EvalJobTimeout,
    JobTimeoutFactory,
    default_job_timeout,
)
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS, JUDGE_MAX_TOKENS, JUDGE_REQUEST_SETTINGS, run_judge_llm
from threetears.evals.run.judge_service import JudgeRequest, JudgeService
from threetears.evals.run.launch import (
    KindLauncher,
    KindWiring,
    LaunchableKind,
    LaunchArgument,
    LaunchGroup,
    LaunchHost,
    LaunchRequest,
    LaunchSettings,
    RunJudge,
    TemplatePreflight,
    build_judge_service,
    launch_as_group,
    launch_run,
    no_launcher_for,
    start_run,
    start_universal_battery,
)
from threetears.evals.run.lifecycle import (
    AbandonedRunSweepReport,
    cancel_run,
    get_run,
    rejudge_result,
    sweep_abandoned_runs,
)
from threetears.evals.run.metering import MeteredCallLedger, MeteredCallTally
from threetears.evals.run.offload import run_blocking
from threetears.evals.run.reads import get_result, get_result_trace, list_results, list_runs
from threetears.evals.run.rejudge import reproducible_judge_inputs
from threetears.evals.run.runner import (
    CellContext,
    ErrorLedger,
    GoalCheckUnevaluable,
    KindFactory,
    RunnerOptions,
    assert_preconditions,
    build_judge_context,
    evaluate_goal_state,
    execute_run,
    fold_metered_cell,
    grade_goal_checks,
    metered_cell_tally,
    precondition_failure_text,
    sample_concurrent_eval_jobs,
)
from threetears.evals.run.simulator import (
    SIMULATOR_REQUEST_SETTINGS,
    CandidateTurn,
    SimulatorReplyInvalid,
    SimulatorTurn,
    TurnDriver,
)
from threetears.evals.run.budget import AccountExhaustedError, BudgetStoppedError, CapBreach
from threetears.evals.run.curation import CurationStore
from threetears.evals.run.definition_seed import SeedCorpus, SeedOutcome
from threetears.evals.run.jobs import WorkFn
from threetears.evals.run.judge_service import JudgeClientFactory, JudgeContext, JudgeOutcome
from threetears.evals.run.launch import BatteryPreflight
from threetears.evals.run.rejudge import JudgeInputStore, ReproducibleJudgeInputs, RequestSettingsPolicy
from threetears.evals.run.runner import EveryCellApparatusFailedError, RunCallbacks


__all__ = [
    "JUDGE_CALL_ATTEMPTS",
    "JUDGE_MAX_TOKENS",
    "JUDGE_REQUEST_SETTINGS",
    "RUBRIC_DIM_SERVER_FIELDS",
    "SIMULATOR_REQUEST_SETTINGS",
    "TEMPLATE_SERVER_FIELDS",
    "AbandonedRunSweepReport",
    "AccountExhaustedError",
    "AdmissionTicket",
    "BatteryPreflight",
    "BudgetStoppedError",
    "CandidateTurn",
    "CapBreach",
    "CellContext",
    "CurationStore",
    "ErrorLedger",
    "EvalJobManager",
    "EvalJobTimeout",
    "EveryCellApparatusFailedError",
    "FidelityContract",
    "GoalCheckUnevaluable",
    "JobTimeoutFactory",
    "JudgeClientFactory",
    "JudgeContext",
    "JudgeInputStore",
    "JudgeOutcome",
    "JudgeRequest",
    "JudgeService",
    "KindFactory",
    "KindLauncher",
    "KindWiring",
    "LaunchArgument",
    "LaunchGroup",
    "LaunchHost",
    "LaunchRequest",
    "LaunchSettings",
    "LaunchableKind",
    "MeteredCallLedger",
    "MeteredCallTally",
    "ReproducibleJudgeInputs",
    "RequestSettingsPolicy",
    "RunCallbacks",
    "RunJudge",
    "RunnerOptions",
    "SeedCorpus",
    "SeedOutcome",
    "SimulatorReplyInvalid",
    "SimulatorTurn",
    "TemplatePreflight",
    "TurnDriver",
    "WorkFn",
    "assert_preconditions",
    "build_judge_context",
    "build_judge_service",
    "callers_missing_the_constructor",
    "cancel_run",
    "create_judge_config",
    "create_rubric_dim",
    "create_template",
    "default_job_timeout",
    "delete_analysis",
    "delete_insight",
    "delete_judge_config",
    "delete_result",
    "delete_rubric_dim",
    "delete_run",
    "evaluate_goal_state",
    "execute_run",
    "fold_metered_cell",
    "get_judge_config",
    "get_result",
    "get_result_trace",
    "get_rubric_dim",
    "get_run",
    "get_template",
    "grade_goal_checks",
    "launch_as_group",
    "launch_run",
    "list_judge_configs",
    "list_results",
    "list_rubric_dims",
    "list_runs",
    "list_templates",
    "load_run_as_listed",
    "load_seed_corpus",
    "metered_cell_tally",
    "no_launcher_for",
    "precondition_failure_text",
    "refuse_stale_presumptions",
    "rejudge_result",
    "reproducible_judge_inputs",
    "require_delete_confirmation",
    "resolve_constructor",
    "run_blocking",
    "run_judge_llm",
    "sample_concurrent_eval_jobs",
    "seed_eval_definitions",
    "set_analysis_archived",
    "set_run_archived",
    "start_run",
    "start_universal_battery",
    "sweep_abandoned_runs",
    "update_judge_config",
    "update_rubric_dim",
    "update_template",
    "validated_kind_spec",
]
