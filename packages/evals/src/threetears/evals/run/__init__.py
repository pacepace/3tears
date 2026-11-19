"""The engine's run package: launching and executing a run, judging it, metering it, and storing it.

Of the engine it imports only itself, :mod:`threetears.evals.schema` and :mod:`threetears.evals.kernel`; what it needs from a host arrives
through the ports a host implements.

**This module is the package's public root.** A host imports from here and from no module below
it, and only the names in ``__all__``; ``tests/test_package_matrix.py`` holds that, and
``tests/test_public_surface_is_closed.py`` holds every engine type a public signature hands a host to
being exported from a public root. A ``# debt:`` comment on an export names what retires it. Code
inside the package imports its own modules directly.
"""

from __future__ import annotations

from threetears.evals.run.authoring import (
    admit_template,
    create_judge_config,
    create_rubric_dim,
    create_template,
    delete_judge_config,
    delete_rubric_dim,
    list_rubric_dims,
    update_judge_config,
    update_rubric_dim,
    update_template,
)
from threetears.evals.run.conversation import drive_conversation
from threetears.evals.run.curation import (
    delete_result,
    delete_run,
)
from threetears.evals.run.ratings import RatingStore, rate_result
from threetears.evals.run.definition_digests import template_case_content_digest
from threetears.evals.run.definition_seed import load_seed_corpus, seed_eval_definitions
from threetears.evals.run.fidelity import FidelityContract, callers_missing_the_constructor, resolve_constructor
from threetears.evals.run.jobs import (
    AdmissionTicket,
    EvalJobManager,
    RunEndHook,
)
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS, JUDGE_MAX_TOKENS, JUDGE_REQUEST_SETTINGS, run_judge_llm
from threetears.evals.run.judge_service import JudgeRequest, JudgeService
from threetears.evals.run.call_floor import price_subject_call_floor
from threetears.evals.run.launch import (
    AdmittedLaunch,
    admit_launch,
    ArmOutcome,
    ArmPlan,
    ArmPrice,
    ArmQuote,
    ArmVerdict,
    CallCeiling,
    KindLauncher,
    KindWiring,
    LaunchableKind,
    LaunchArgument,
    LaunchGroup,
    LaunchHost,
    TemplateField,
    ServedCheck,
    ServedRole,
    SubjectResolver,
    UnknownSubject,
    UnpricedCall,
    LaunchPricer,
    LaunchQuote,
    LaunchRequest,
    LaunchSettings,
    PlannedJudge,
    RunJudge,
    build_judge_service,
    launch_as_group,
    launch_run,
    plan_judge,
    price_arms,
    quote_launch,
    require_candidate_model,
    resolve_judge_pin,
    start_run,
)
from threetears.evals.run.lifecycle import (
    AbandonedRunSweepReport,
    ReclaimStore,
    get_run,
    rejudge_result,
    sweep_abandoned_runs,
)
from threetears.evals.run.executor import (
    DEFAULT_MAX_CONCURRENT_CELLS,
    CellExecutor,
    CellWork,
)
from threetears.evals.run.metering import CellMeter, MeteredCallLedger, MeteredCallTally
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.run.reads import (
    get_result_trace,
    list_results,
    list_runs,
)
from threetears.evals.run.judge_repeat import (
    JudgeRepeatEstimate,
    JudgeRepeatReport,
    JudgeRepeatSkip,
    estimate_judge_repeat,
    judge_repeat_work,
)
from threetears.evals.run.judge_temperature import (
    JUDGE_TEMPERATURE_JOB,
    EvalJudgeTemperatureCheck,
    JudgeTemperatureEstimate,
    estimate_judge_temperature_comparison,
    judge_temperature_work,
)
from threetears.evals.run.judge_second import (
    SECOND_JUDGE_JOB,
    EvalSecondJudgePass,
    SecondJudgeAskedBy,
    SecondJudgeEstimate,
    SecondJudgeReport,
    SecondJudgeSkip,
    second_judge_work,
)
from threetears.evals.run.launch_job import (
    LAUNCH_JOB,
    EvalLaunch,
    launch_job_work,
)
from threetears.evals.run.job_work import (
    ACTION_JOB,
    EvalActionJob,
    JobOutcome,
    JobRecorder,
    JobWork,
    run_job_work,
)
from threetears.evals.run.rejudge import reproducible_judge_inputs
from threetears.evals.run.regrade import Regrade, RegradedDim, regrade
from threetears.evals.run.remeasure import (
    Attested,
    Remeasure,
    attest_measure_definitions,
    remeasure_run,
)
from threetears.evals.run.judge_kind import (
    FrozenJudgeCase,
    JudgeCaseFreezeReport,
    JudgeCaseSkip,
    JudgeKind,
    PreparedJudge,
    freeze_judge_cases,
    judge_kind,
    launchable_judge_kind,
)
from threetears.evals.run.runner import (
    CellContext,
    GoalCheckUnevaluable,
    KindFactory,
    RunnerOptions,
    assert_preconditions,
    execute_run,
    fold_metered_cell,
    grade_goal_checks,
    metered_cell_tally,
    precondition_failure_text,
)
from threetears.evals.run.simulator import (
    SIMULATOR_REQUEST_SETTINGS,
    CandidateTurn,
    SimulatorCall,
    SimulatorReplyInvalid,
    SimulatorTurn,
    TurnDriver,
)
from threetears.evals.run.witnessed import WitnessedJudging, record_witnessed_cell, stamp_witnessed_judge
from threetears.evals.run.budget import (
    AccountExhaustedError,
    BudgetStoppedError,
    CapBreach,
    EvalRunCostCap,
    ModelUnservedError,
)
from threetears.evals.run.case_sets import mint_case_set
from threetears.evals.run.ceilings import (
    CeilingRaisedError,
)
from threetears.evals.run.curation import CurationStore
from threetears.evals.run.package_ceilings import Ceilings
from threetears.evals.run.definition_seed import SeedCaseSet, SeedCorpus, SeedJudgeCase, SeedOutcome, SeedTemplateCase
from threetears.evals.run.definition_drift import (
    DefinitionDrift,
    DefinitionReading,
    DefinitionReset,
    DefinitionState,
    ResetDefinition,
    corpus_digests,
    definition_drift,
    reset_definitions,
)
from threetears.evals.run.jobs import WorkFn
from threetears.evals.run.judge_service import JudgeClientFactory, JudgeContext, JudgeOutcome
from threetears.evals.run.rejudge import JudgeInputStore, ReproducibleJudgeInputs, RequestSettingsPolicy
from threetears.evals.run.runner import EveryCellApparatusFailedError, NothingMeasuredError, RunCallbacks
from threetears.evals.run.runner import StoredCell, StoredCells
from threetears.evals.run.resume import (
    ResumableEnding,
    ResumableGroup,
    ResumableJob,
    ResumableJobs,
    RunResumption,
    resumable_jobs,
)


__all__ = [
    "ReclaimStore",
    "JUDGE_CALL_ATTEMPTS",
    "JUDGE_MAX_TOKENS",
    "JUDGE_REQUEST_SETTINGS",
    "SIMULATOR_REQUEST_SETTINGS",
    "DEFAULT_MAX_CONCURRENT_CELLS",
    "AbandonedRunSweepReport",
    "AccountExhaustedError",
    "AdmissionTicket",
    "ArmOutcome",
    "ArmPlan",
    "ArmPrice",
    "ArmQuote",
    "ArmVerdict",
    "BudgetStoppedError",
    "CallCeiling",
    "CandidateTurn",
    "Ceilings",
    "CapBreach",
    "CeilingRaisedError",
    "EvalRunCostCap",
    "CellContext",
    "CellExecutor",
    "CellMeter",
    "CellWork",
    "CurationStore",
    "EvalJobManager",
    "EveryCellApparatusFailedError",
    "NothingMeasuredError",
    "FidelityContract",
    "GoalCheckUnevaluable",
    "RunEndHook",
    "FrozenJudgeCase",
    "JudgeCaseFreezeReport",
    "JudgeCaseSkip",
    "JudgeClientFactory",
    "JudgeContext",
    "JudgeInputStore",
    "JudgeKind",
    "PreparedJudge",
    "JudgeOutcome",
    "JudgeRequest",
    "JudgeService",
    "KindFactory",
    "KindLauncher",
    "KindWiring",
    "LaunchArgument",
    "LaunchGroup",
    "LaunchHost",
    "TemplateField",
    "ServedCheck",
    "ServedRole",
    "SubjectResolver",
    "UnknownSubject",
    "UnpricedCall",
    "LaunchPricer",
    "LaunchQuote",
    "LaunchRequest",
    "LaunchSettings",
    "LaunchableKind",
    "MeteredCallLedger",
    "MeteredCallTally",
    "ModelUnservedError",
    "RatingStore",
    "Regrade",
    "Attested",
    "Remeasure",
    "RegradedDim",
    "ReproducibleJudgeInputs",
    "RequestSettingsPolicy",
    "RunCallbacks",
    "PlannedJudge",
    "RunJudge",
    "RunnerOptions",
    "SeedCaseSet",
    "SeedCorpus",
    "SeedJudgeCase",
    "SeedOutcome",
    "SeedTemplateCase",
    "DefinitionDrift",
    "DefinitionReading",
    "DefinitionReset",
    "DefinitionState",
    "ResetDefinition",
    "corpus_digests",
    "definition_drift",
    "reset_definitions",
    "SimulatorCall",
    "SimulatorReplyInvalid",
    "SimulatorTurn",
    "TurnDriver",
    "WitnessedJudging",
    "WorkFn",
    "admit_template",
    "assert_preconditions",
    "freeze_judge_cases",
    "judge_kind",
    "launchable_judge_kind",
    "build_judge_service",
    "callers_missing_the_constructor",
    "create_judge_config",
    "create_rubric_dim",
    "create_template",
    "delete_judge_config",
    "delete_result",
    "delete_rubric_dim",
    "delete_run",
    "drive_conversation",
    "execute_run",
    "fold_metered_cell",
    "get_result_trace",
    "get_run",
    "grade_goal_checks",
    "launch_as_group",
    "launch_run",
    "list_results",
    "list_rubric_dims",
    "list_runs",
    "load_seed_corpus",
    "template_case_content_digest",
    "metered_cell_tally",
    "plan_judge",
    "price_arms",
    "price_subject_call_floor",
    "quote_launch",
    "require_candidate_model",
    "resolve_judge_pin",
    "precondition_failure_text",
    "rate_result",
    "regrade",
    "attest_measure_definitions",
    "remeasure_run",
    "record_witnessed_cell",
    "stamp_witnessed_judge",
    "rejudge_result",
    "judge_repeat_work",
    "reproducible_judge_inputs",
    "resolve_constructor",
    "run_blocking",
    "run_judge_llm",
    "seed_eval_definitions",
    "start_run",
    "mint_case_set",
    "sweep_abandoned_runs",
    "ResumableEnding",
    "ResumableGroup",
    "ResumableJob",
    "ResumableJobs",
    "RunResumption",
    "StoredCell",
    "StoredCells",
    "resumable_jobs",
    "update_judge_config",
    "update_rubric_dim",
    "update_template",
    "JudgeRepeatEstimate",
    "JudgeRepeatReport",
    "JudgeRepeatSkip",
    "estimate_judge_repeat",
    "SecondJudgeEstimate",
    "SecondJudgeReport",
    "SecondJudgeSkip",
    "second_judge_work",
    "SECOND_JUDGE_JOB",
    "EvalSecondJudgePass",
    "SecondJudgeAskedBy",
    "JudgeTemperatureEstimate",
    "estimate_judge_temperature_comparison",
    "judge_temperature_work",
    "JUDGE_TEMPERATURE_JOB",
    "EvalJudgeTemperatureCheck",
    "ACTION_JOB",
    "EvalActionJob",
    "JobOutcome",
    "JobRecorder",
    "JobWork",
    "run_job_work",
    "AdmittedLaunch",
    "admit_launch",
    "LAUNCH_JOB",
    "EvalLaunch",
    "launch_job_work",
]
