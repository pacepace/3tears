"""What each declared public name of ``3tears-evals`` asks of its reader: its kind, and the budget each kind is held to.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed. The snapshot generator
(``generate_api_snapshot.py``) writes each declared name's kind into the tier map the wheel ships, beside its tier,
and ``generate_surface.py`` renders ``docs/surface.md`` and ``llms.txt`` from the same kinds.

A declared name (``host-contract`` or ``stable``) is exactly one of five kinds:

- ``typed``: a newcomer writes it in an eval file. Each one a rung teaches is in :data:`TYPED` with that rung, held
  to what the rung's example and its ladder section import; one no rung teaches is in :data:`UNTAUGHT_TYPED`.
- ``bound``: a host implements it or hands it to the engine: a protocol, a store, a registry, a client, a kind seam,
  a function handed over as a value, a factory that builds one, and a shape a host builds for a seam.
- ``invoked``: an operation called by name to make the engine do something, from host code or as an action of the
  catalogue; and, until an owner decides to demote them, the :data:`HELPERS`.
- ``received``: a value someone is handed and reads: a result or stored-document shape, a literal, a constant, an
  error.
- ``tests``: a conformance kit, a recorder or a guard a host runs in its own tests.

A kind is derived by rule only where the rule is certain (:func:`kind_by_rule`); every other declared name is recorded
in :data:`KIND_ENTRIES`, so a new name never takes a kind by default. ``tests/test_api_tiers.py`` fails on a declared
name with no kind (saying how to decide one), on an entry a rule already decides, and on an entry for a name that is
not declared; and it holds each kind to its ceiling, each ceiling to its count, and each list of findings
(:data:`UNTAUGHT_TYPED`, :data:`HELPERS`, :data:`UNREACHED_SHAPES`, :data:`NEAR_DUPLICATES`) to what is measured.
"""

from __future__ import annotations

import annotationlib
import ast
import dataclasses
import importlib
import inspect
import re
import sys
import textwrap
import typing
import warnings
from collections.abc import Iterator, Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from public_surface import PACKAGE_ROOT, SurfaceEntry, assignment, declared, is_alias, public_names
from pydantic import BaseModel

#: A declared name's kind: what it asks of the person or agent who meets it.
Kind = Literal["typed", "bound", "invoked", "received", "tests"]

#: The kinds, in the order a reader meets them.
KINDS: tuple[Kind, ...] = typing.get_args(Kind)

#: Every ``typed`` name a rung teaches, by ``root.name``, with the first rung that does: the rung whose example under
#: ``examples/`` imports it, or whose section of ``docs/ladder.md`` imports it in a code block. ``tests/test_api_tiers.py``
#: holds this equal to what those sources import (:func:`taught_rungs`), so an import a rung gains fails until it is
#: recorded here and its budget reconsidered. Rungs 0 and 1 are a first eval.
TYPED: Mapping[str, int] = MappingProxyType(
    {
        "threetears.evals.Eval": 1,
        "threetears.evals.Model": 1,
        "threetears.evals.Judge": 2,
        "threetears.evals.Example": 2,
        "threetears.evals.grader": 3,
        "threetears.evals.World": 4,
        "threetears.evals.quick.as_host": 4,
        "threetears.evals.transports.fastmcp.mount_fastmcp": 4,
        "threetears.evals.harness.AnalystSkill": 5,
        "threetears.evals.harness.Writer": 5,
        "threetears.evals.harness.WriterEval": 5,
    }
)

#: Names a user writes in an eval file that no rung teaches: ``typed``, and each a finding. A rung that comes to teach
#: one moves it to :data:`TYPED`; the list may only shrink.
UNTAUGHT_TYPED: frozenset[str] = frozenset(
    {
        # A runner returns it to report its own spend beside its answer; only the reference describes it.
        "threetears.evals.Answer",
        # A runner over an app's own logged outputs; taught in docs/evaluating-agents.md, on no rung.
        "threetears.evals.Log",
        # The decorator a measure of one's own is declared with; taught in docs/adopting-a-host.md, on no rung.
        "threetears.evals.quick.measure",
    }
)

#: The ladder's examples, each directory under ``examples/`` with its rung.
LADDER_EXAMPLES: Mapping[str, int] = MappingProxyType(
    {"1-triage": 1, "2-replies": 2, "3-invoices": 3, "4-booking": 4, "5-writer": 5}
)

#: Declared functions that are ``invoked`` but are no operation: a formatter, a parser, a digest, a clock, a wire
#: helper, called inside the engine and by a host only to repeat what the engine does. Each is a candidate to demote
#: to provisional, an owner's decision; until then they count against ``invoked``, and the list may only shrink.
HELPERS: frozenset[str] = frozenset(
    {
        "threetears.evals.surface_tier",
        "threetears.evals.schema.bind_cell_identity",
        "threetears.evals.schema.check_after",
        "threetears.evals.schema.current_cell_identity",
        "threetears.evals.schema.field_in_values",
        "threetears.evals.schema.is_builtin_judge_config_id",
        "threetears.evals.schema.judge_config_of",
        "threetears.evals.schema.label_key_of",
        "threetears.evals.schema.nested_schemas",
        "threetears.evals.schema.sent_settings_of",
        "threetears.evals.schema.utc_now_iso",
        "threetears.evals.kernel.confusion_cell",
        "threetears.evals.kernel.extract_json",
        "threetears.evals.kernel.extract_json_array",
        "threetears.evals.kernel.fold_phase_timings",
        "threetears.evals.kernel.production_replicating_cost",
        "threetears.evals.kernel.seed_digest",
        "threetears.evals.kernel.strict_response_format",
        "threetears.evals.run.corpus_digests",
        "threetears.evals.run.fold_metered_cell",
        "threetears.evals.run.metered_cell_tally",
        "threetears.evals.run.precondition_failure_text",
        "threetears.evals.run.run_blocking",
        "threetears.evals.analysis.check_names",
        "threetears.evals.analysis.dollars_text",
        "threetears.evals.analysis.format_number",
        "threetears.evals.analysis.host_declarations_digest",
        "threetears.evals.analysis.humanize",
        "threetears.evals.analysis.judge_key",
        "threetears.evals.analysis.lint_prose",
        "threetears.evals.analysis.tier_sentence",
        "threetears.evals.negotiation.anthropic_thinking",
        "threetears.evals.negotiation.facts_from_router_catalogue",
        "threetears.evals.negotiation.read_anthropic_refusal",
        "threetears.evals.negotiation.read_openai_refusal",
        "threetears.evals.negotiation.read_router_refusal",
        "threetears.evals.negotiation.wire_settings_for",
    }
)

#: The kind of each declared name no rule decides (:func:`kind_by_rule` returns ``None`` for it): a function that is no
#: action, a class that is no shape, and a shape a parameter or a seam names or that no declared name names directly.
#: An entry for a name a rule decides is refused, so a rule and this list never disagree.
KIND_ENTRIES: Mapping[str, Kind] = MappingProxyType(
    {
        # Functions a host hands the engine as a value: a failure describer, a timeout factory.
        "threetears.evals.providers.describe_failure": "bound",
        "threetears.evals.kernel.withhold_failure_detail": "bound",
        "threetears.evals.kernel.default_job_timeout": "bound",
        "threetears.evals.kernel.host.default_cell_timeout": "bound",
        # Wiring factories: each builds something a host hands the engine (a catalogue, its tools, a registry, a
        # pricer, a launch entry, a store) or hands an action (a renderer).
        "threetears.evals.actions.eval_catalogue": "bound",
        "threetears.evals.actions.definition_drift_actions": "bound",
        "threetears.evals.actions.standard_tools": "bound",
        "threetears.evals.actions.analyst_tools": "bound",
        "threetears.evals.actions.render_jobs_started": "bound",
        "threetears.evals.kernel.host.load_bar_registry": "bound",
        "threetears.evals.ops.history_launch_pricer": "bound",
        "threetears.evals.run.launchable_judge_kind": "bound",
        "threetears.evals.storage.bridge_async_store": "bound",
        # A conformance check a host runs in its own tests.
        "threetears.evals.kernel.host.check_world_conformance": "tests",
        # Operations: each makes the engine do something (launch, run, judge, read, write, price, render, assemble).
        "threetears.evals.kernel.failed_call_output": "invoked",
        "threetears.evals.kernel.prompt_drift": "invoked",
        "threetears.evals.kernel.host.check_seed": "invoked",
        "threetears.evals.run.admit_template": "invoked",
        "threetears.evals.run.assert_preconditions": "invoked",
        "threetears.evals.run.build_judge_service": "invoked",
        "threetears.evals.run.create_template": "invoked",
        "threetears.evals.run.definition_drift": "invoked",
        "threetears.evals.run.delete_run": "invoked",
        "threetears.evals.run.drive_conversation": "invoked",
        "threetears.evals.run.estimate_judge_repeat": "invoked",
        "threetears.evals.run.execute_run": "invoked",
        "threetears.evals.run.freeze_judge_cases": "invoked",
        "threetears.evals.run.get_result_trace": "invoked",
        "threetears.evals.run.get_run": "invoked",
        "threetears.evals.run.grade_goal_checks": "invoked",
        "threetears.evals.run.judge_repeat_work": "invoked",
        "threetears.evals.run.judge_temperature_work": "invoked",
        "threetears.evals.run.launch_as_group": "invoked",
        "threetears.evals.run.launch_run": "invoked",
        "threetears.evals.run.list_results": "invoked",
        "threetears.evals.run.list_runs": "invoked",
        "threetears.evals.run.load_seed_corpus": "invoked",
        "threetears.evals.run.mint_case_set": "invoked",
        "threetears.evals.run.plan_judge": "invoked",
        "threetears.evals.run.price_subject_call_floor": "invoked",
        "threetears.evals.run.quote_launch": "invoked",
        "threetears.evals.run.rate_result": "invoked",
        "threetears.evals.run.regrade": "invoked",
        "threetears.evals.run.require_candidate_model": "invoked",
        "threetears.evals.run.reset_definitions": "invoked",
        "threetears.evals.run.resolve_judge_pin": "invoked",
        "threetears.evals.run.run_job_work": "invoked",
        "threetears.evals.run.run_judge_llm": "invoked",
        "threetears.evals.run.second_judge_work": "invoked",
        "threetears.evals.run.seed_eval_definitions": "invoked",
        "threetears.evals.run.start_run": "invoked",
        "threetears.evals.run.sweep_abandoned_runs": "invoked",
        "threetears.evals.run.update_template": "invoked",
        "threetears.evals.analysis.assemble_context_bundle": "invoked",
        "threetears.evals.analysis.campaign_report": "invoked",
        "threetears.evals.analysis.create_campaign": "invoked",
        "threetears.evals.analysis.frontier": "invoked",
        "threetears.evals.analysis.gate_verdicts": "invoked",
        "threetears.evals.analysis.inspect_campaign_bundle": "invoked",
        "threetears.evals.analysis.inter_judge_agreement": "invoked",
        "threetears.evals.analysis.judge_agreement": "invoked",
        "threetears.evals.analysis.judge_drift": "invoked",
        "threetears.evals.analysis.judge_evidence_tiers": "invoked",
        "threetears.evals.analysis.judge_kind_readings": "invoked",
        "threetears.evals.analysis.judge_self_agreement": "invoked",
        "threetears.evals.analysis.read_judge_temperatures": "invoked",
        "threetears.evals.analysis.report_html": "invoked",
        "threetears.evals.gen.generate_variations": "invoked",
        "threetears.evals.gen.plan_draft": "invoked",
        "threetears.evals.gen.propose_draft": "invoked",
        "threetears.evals.quick.run_cli": "invoked",
        "threetears.evals.ops.await_job": "invoked",
        "threetears.evals.ops.case_rename": "invoked",
        "threetears.evals.ops.start_action_job": "invoked",
        "threetears.evals.ops.start_paid_call": "invoked",
        "threetears.evals.negotiation.closest_settings": "invoked",
        "threetears.evals.negotiation.negotiate": "invoked",
        # Classes a host builds and hands the engine: stores, registries, services, clients, the catalogue, and what
        # a host's own hook or resolver answers with (a caller, a disclosure, an unknown subject, a command).
        "threetears.evals.kernel.CappedCompletion": "bound",
        "threetears.evals.kernel.EvalStorage": "bound",
        "threetears.evals.kernel.JobLeases": "bound",
        "threetears.evals.kernel.WorldSession": "bound",
        "threetears.evals.kernel.host.BarRegistry": "bound",
        "threetears.evals.kernel.host.MeasureRegistry": "bound",
        "threetears.evals.kernel.host.SweepableRegistry": "bound",
        "threetears.evals.kernel.host.WorldRegistry": "bound",
        "threetears.evals.kernel.host.KindContract": "bound",
        "threetears.evals.run.JudgeService": "bound",
        "threetears.evals.run.LaunchGroup": "bound",
        "threetears.evals.run.MeteredCallLedger": "bound",
        "threetears.evals.run.UnknownSubject": "bound",
        "threetears.evals.storage.InMemoryDocumentStore": "bound",
        "threetears.evals.storage.RoutedDocumentStore": "bound",
        "threetears.evals.storage.SqliteDocumentStore": "bound",
        "threetears.evals.quick.HostCommand": "bound",
        "threetears.evals.ops.AnalysisGeneration": "bound",
        "threetears.evals.ops.ReportBudget": "bound",
        "threetears.evals.actions.ActionCatalogue": "bound",
        "threetears.evals.actions.Caller": "bound",
        "threetears.evals.negotiation.LearnedSettings": "bound",
        "threetears.evals.providers.AnthropicPublishedRates": "bound",
        "threetears.evals.providers.Catalogue": "bound",
        "threetears.evals.providers.Clients": "bound",
        "threetears.evals.providers.Model": "bound",
        "threetears.evals.schema.RunDisclosure": "bound",
        # Shapes a host builds and hands the engine: a kind's trace, a describer's failure, a replay's delivery, a
        # launch registry's entry and what its callables answer, a writer's limits, and the base a host's own stored
        # models extend.
        "threetears.evals.schema.CellTrace": "bound",
        "threetears.evals.schema.EvalBaseModel": "bound",
        "threetears.evals.schema.ProviderFailure": "bound",
        "threetears.evals.kernel.ReplayedDelivery": "bound",
        "threetears.evals.run.LaunchableKind": "bound",
        "threetears.evals.ops.WriterLimits": "bound",
        "threetears.evals.run.ArmPlan": "bound",
        "threetears.evals.run.CallCeiling": "bound",
        "threetears.evals.run.UnpricedCall": "bound",
        "threetears.evals.ops.AnalystLimits": "bound",
        "threetears.evals.ops.EditorLimits": "bound",
        "threetears.evals.ops.SessionLimits": "bound",
        "threetears.evals.actions.ConformanceSample": "tests",
        # What a decorator or a template hands back to be read, and a kit's stage.
        "threetears.evals.quick.Measure": "received",
        "threetears.evals.schema.ConversationSpec": "received",
        "threetears.evals.actions.ConformanceStage": "tests",
        # The host and the seams it wires: what a host builds and hands every entrypoint, its routes, its own
        # actions and tools, and what its resolver answers.
        "threetears.evals.kernel.host.EvalHost": "bound",
        "threetears.evals.kernel.host.HostProfile": "bound",
        "threetears.evals.kernel.host.TemplateChecks": "bound",
        "threetears.evals.run.LaunchHost": "bound",
        "threetears.evals.run.KindWiring": "bound",
        "threetears.evals.ops.OpsHost": "bound",
        "threetears.evals.storage.DocTypeRoute": "bound",
        "threetears.evals.actions.Action": "bound",
        "threetears.evals.transports.fastmcp.CatalogueTool": "bound",
        "threetears.evals.transports.fastmcp.HostResolution": "bound",
        # Engine services a host builds and runs: the conversation scheduler, the regression watch, a role's ledger.
        "threetears.evals.run.TurnDriver": "bound",
        "threetears.evals.ops.RegressionWatch": "bound",
        "threetears.evals.kernel.RoleUsageLedger": "bound",
        # The markers a host writes on its own kind's level fields, which the engine reads off its model.
        "threetears.evals.kernel.host.ActsOn": "bound",
        "threetears.evals.kernel.host.Interval": "bound",
        "threetears.evals.kernel.host.MemberActsOn": "bound",
        "threetears.evals.kernel.host.Ordinal": "bound",
        "threetears.evals.kernel.host.ResolvesInto": "bound",
        "threetears.evals.kernel.host.Triggered": "bound",
        # Shapes: records, declarations and values a callable takes as well as hands back, or that only another
        # shape or a constant names. Each is read, and none is a seam.
        "threetears.evals.RootExtra": "received",
        "threetears.evals.schema.ActorPolicy": "received",
        "threetears.evals.schema.CalibrationRating": "received",
        "threetears.evals.schema.CallLedger": "received",
        "threetears.evals.schema.CaseSetRef": "received",
        "threetears.evals.schema.CatalogRubricDim": "received",
        "threetears.evals.schema.CellIdentity": "received",
        "threetears.evals.schema.ClientRequestSettings": "received",
        "threetears.evals.schema.EvalResult": "received",
        "threetears.evals.schema.EvalRun": "received",
        "threetears.evals.schema.EvalTemplate": "received",
        "threetears.evals.schema.EvalTestCase": "received",
        "threetears.evals.schema.EvalTrace": "received",
        "threetears.evals.schema.ExternalSpend": "received",
        "threetears.evals.schema.IntervalScale": "received",
        "threetears.evals.schema.JudgeConfig": "received",
        "threetears.evals.schema.JudgeEvidence": "received",
        "threetears.evals.schema.OutOfRunSpend": "received",
        "threetears.evals.schema.RecordedCall": "received",
        "threetears.evals.schema.RubricDim": "received",
        "threetears.evals.schema.RubricProposal": "received",
        "threetears.evals.schema.SecondJudge": "received",
        "threetears.evals.schema.SubjectSnapshot": "received",
        "threetears.evals.schema.ThinkingBinding": "received",
        "threetears.evals.schema.ToolResultClearing": "received",
        "threetears.evals.schema.TurnCeiling": "received",
        "threetears.evals.schema.TurnMessage": "received",
        "threetears.evals.schema.TurnReply": "received",
        "threetears.evals.schema.TurnToolSpec": "received",
        "threetears.evals.schema.WorldSeed": "received",
        "threetears.evals.kernel.AdmittedCall": "received",
        "threetears.evals.kernel.AuthoredJudgeCase": "received",
        "threetears.evals.kernel.CandidateOutput": "received",
        "threetears.evals.kernel.CandidateTelemetry": "received",
        "threetears.evals.kernel.CostLower": "received",
        "threetears.evals.kernel.EvalAnalysis": "received",
        "threetears.evals.kernel.EvalCampaign": "received",
        "threetears.evals.kernel.EvalInsight": "received",
        "threetears.evals.kernel.EvalJudgeProfile": "received",
        "threetears.evals.kernel.ExternalRateTable": "received",
        "threetears.evals.kernel.MeasureFamily": "received",
        "threetears.evals.kernel.MetricDescriptor": "received",
        "threetears.evals.kernel.OutOfRunBudget": "received",
        "threetears.evals.kernel.OverriddenBar": "received",
        "threetears.evals.kernel.PlannedCall": "received",
        "threetears.evals.kernel.RateDrop": "received",
        "threetears.evals.kernel.SeedPrompt": "received",
        "threetears.evals.kernel.SeedTemplate": "received",
        "threetears.evals.kernel.VariantConfig": "received",
        "threetears.evals.kernel.host.Bar": "received",
        "threetears.evals.kernel.host.PassThreshold": "received",
        "threetears.evals.kernel.host.RolePins": "received",
        "threetears.evals.kernel.host.Sweepable": "received",
        "threetears.evals.kernel.host.WorldDimension": "received",
        "threetears.evals.run.ArmQuote": "received",
        "threetears.evals.run.CandidateTurn": "received",
        "threetears.evals.run.Ceilings": "received",
        "threetears.evals.run.CellContext": "received",
        "threetears.evals.run.FidelityContract": "received",
        "threetears.evals.run.JudgeRepeatReport": "received",
        "threetears.evals.run.LaunchRequest": "received",
        "threetears.evals.run.LaunchSettings": "received",
        "threetears.evals.run.MeteredCallTally": "received",
        "threetears.evals.run.RunnerOptions": "received",
        "threetears.evals.run.SeedCorpus": "received",
        "threetears.evals.run.SimulatorTurn": "received",
        "threetears.evals.analysis.ChartBlock": "received",
        "threetears.evals.analysis.Verdict": "received",
        "threetears.evals.ops.CampaignDefinition": "received",
        "threetears.evals.ops.CaseSetMint": "received",
        "threetears.evals.ops.JobsStarted": "received",
        "threetears.evals.ops.JudgeCaseAuthor": "received",
        "threetears.evals.ops.JudgeProfilesRecord": "received",
        "threetears.evals.ops.LaunchArguments": "received",
        "threetears.evals.actions.ActionOutcome": "received",
        "threetears.evals.transports.fastmcp.ToolCall": "received",
        "threetears.evals.negotiation.SettingFacts": "received",
        "threetears.evals.negotiation.SettingRefusal": "received",
        "threetears.evals.negotiation.WireSettings": "received",
        "threetears.evals.providers.Completion": "received",
    }
)

#: The most declared names each kind should hold (information-architecture §8.2), which replaced one ceiling on the
#: whole declared surface. ``received`` has no total: its discipline is that every shape is reached from a ``typed``,
#: ``bound`` or ``invoked`` name (:func:`unreached_shapes`), never hidden to protect a count. ``tests`` is held at its
#: count when kinds were first recorded: one kit per port, and a new kit names its port.
KIND_BUDGETS: Mapping[Kind, int | None] = MappingProxyType(
    {"typed": 25, "bound": 100, "invoked": 110, "received": None, "tests": 16}
)

#: The most each kind holds today: its budget, or, where today's count is over it, that count, which may only fall.
#: ``tests/test_api_tiers.py`` holds each to exactly the larger of its budget and its count, so a ceiling over the
#: count fails until it is lowered, and one under it fails until a name leaves.
KIND_CEILINGS: Mapping[Kind, int | None] = MappingProxyType(
    {
        "typed": 25,
        # Over budget when kinds were first recorded (2026-11-15): what a host builds and hands the engine includes
        # the shapes its seams return (a kind's arm plan, a describer's failure, a writer's limits).
        "bound": 107,
        # Over budget when kinds were first recorded: 80 operations and the 37 HELPERS, an owner's decision to demote.
        "invoked": 117,
        "received": None,
        "tests": 16,
    }
)

#: The received shapes no ``typed``, ``bound``, ``invoked`` or ``tests`` name reached when kinds were first recorded
#: (2026-11-15): each is handed to a host only through a private method, a job's untyped result or a docstring's
#: prose, so no declared signature names it. The list may only shrink: a shape newly unreached fails, and one a
#: signature comes to name must leave the list.
UNREACHED_SHAPES: frozenset[str] = frozenset(
    {
        "threetears.evals.schema.StoreCalledOnEventLoop",
        "threetears.evals.kernel.CandidateKindDefect",
        "threetears.evals.kernel.ProviderRefusedError",
        "threetears.evals.run.Ceilings",
        "threetears.evals.run.FidelityContract",
        "threetears.evals.run.JudgeRepeatReport",
        "threetears.evals.providers.Completion",
        "threetears.evals.providers.PriceSource",
    }
)

#: The near-duplicate shapes (:func:`near_duplicate_shapes`) found when kinds were first recorded (2026-11-15). The
#: list may only shrink: a new pair fails, and a pair resolved must leave it.
NEAR_DUPLICATES: frozenset[tuple[str, str]] = frozenset(
    {
        # The stored spend document and the frozen runtime value: a deliberate split, since one is a stored schema.
        ("threetears.evals.schema.AsyncExternalSpend", "threetears.evals.schema.ExternalSpend"),
        # The settings a request was asked or sent with, and the same as a client asks for them, less a temperature.
        ("threetears.evals.schema.ClientRequestSettings", "threetears.evals.negotiation.WireSettings"),
    }
)

#: The most ``typed`` names a first eval (rungs 0 and 1) needs.
FIRST_EVAL_CEILING = 8

#: The most ``typed`` names rungs 0 to 3 need together.
THROUGH_RUNG_3_CEILING = 12

#: The root whose every declared name is a host's test kit.
TESTING_ROOT = "threetears.evals.testing"


# --- the walk: which engine types a public object names, and where ---------------------------------------------

#: Where an engine type sits in what names it.
Position = Literal["parameter", "return", "field", "base", "raises"]

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RAISES = re.compile(r"^:raises ~?([A-Za-z_][A-Za-z0-9_.]*):", re.M)


def _engine_class(value: object) -> bool:
    return inspect.isclass(value) and value.__module__.startswith("threetears.evals")


@cache
def _aliases() -> Mapping[int, object]:
    """Every type alias a public root exports, by ``id``: what an annotation token is matched against."""
    found: dict[int, object] = {}
    for _, _, obj in public_names():
        if is_alias(obj) and not inspect.isclass(obj):
            found[id(obj)] = obj
    return MappingProxyType(found)


@cache
def _plain_aliases() -> frozenset[str]:
    """Each exported type alias that is a bare builtin (``RefusalKind = str``), by name: an annotation names it only in
    its source text, since its value is the builtin."""
    from public_surface import describe

    return frozenset(
        name
        for root, name, obj in public_names()
        if inspect.isclass(obj) and not _engine_class(obj) and describe(root, name, obj).kind == "type alias"
    )


def _namespace(obj: object) -> dict[str, Any]:
    module = sys.modules.get(getattr(obj, "__module__", "") or "")
    return dict(vars(module)) if module else {}


def _types_in(annotation: object, namespace: Mapping[str, Any], _seen: set[int] | None = None) -> Iterator[object]:
    """Each engine class or exported type alias an annotation names, however it is written."""
    _seen = set() if _seen is None else _seen
    if isinstance(annotation, (list, tuple)):  # a ``Callable``'s parameter list
        for item in annotation:
            yield from _types_in(item, namespace, _seen)
        return
    if annotation is inspect.Parameter.empty or annotation is None:
        return
    if isinstance(annotation, str):
        for token in _IDENT.findall(annotation):
            value = namespace.get(token)
            if _engine_class(value) or id(value) in _aliases() or (token in _plain_aliases() and value is not None):
                yield value
            elif value is not None and is_alias(value) and id(value) not in _seen:
                # An alias no root exports (``Scale``) is read through to the types it unions.
                _seen.add(id(value))
                yield from _types_in(value, namespace, _seen)
        return
    if isinstance(annotation, annotationlib.ForwardRef):
        yield from _types_in(annotation.__forward_arg__, namespace, _seen)
        return
    if id(annotation) in _aliases():
        yield annotation
        return
    if _engine_class(annotation):
        yield annotation
        return
    if isinstance(annotation, typing.TypeAliasType):
        if id(annotation) not in _seen:
            _seen.add(id(annotation))
            yield from _types_in(annotation.__value__, namespace, _seen)
        return
    for arg in typing.get_args(annotation):
        yield from _types_in(arg, namespace, _seen)


#: An annotation is read twice: as its source text, whose words name an alias even when its value is a builtin
#: (``RefusalKind = str``), and as its value, which resolves a name imported only for type checking where it can.
_FORMATS = (annotationlib.Format.STRING, annotationlib.Format.FORWARDREF)


def _signatures(function: Any) -> Iterator[inspect.Signature]:
    for annotation_format in _FORMATS:
        try:
            yield inspect.signature(function, annotation_format=annotation_format)
        except TypeError, ValueError, NameError:
            continue


def _callable_edges(function: Any, namespace: Mapping[str, Any]) -> Iterator[tuple[Position, object]]:
    # A function's names resolve where it was written, whatever ``__module__`` its class was given for display.
    namespace = getattr(inspect.unwrap(function), "__globals__", None) or namespace
    for found in _signatures(function):
        for parameter in found.parameters.values():
            if parameter.name in ("self", "cls"):
                continue
            for value in _types_in(parameter.annotation, namespace):
                yield "parameter", value
        for value in _types_in(found.return_annotation, namespace):
            yield "return", value
    yield from _raises_edges(inspect.getdoc(function) or "", namespace)


@cache
def _exceptions_by_name() -> Mapping[str, tuple[type, ...]]:
    found: dict[str, list[type]] = {}
    for _, name, obj in public_names():
        if inspect.isclass(obj) and issubclass(obj, BaseException) and obj not in found.get(name, []):
            found.setdefault(name, []).append(obj)
    return MappingProxyType({name: tuple(each) for name, each in found.items()})


def _raises_edges(doc: str, namespace: Mapping[str, Any]) -> Iterator[tuple[Position, object]]:
    for entry in _RAISES.finditer(doc):
        name = entry.group(1).rsplit(".", 1)[-1]
        value = namespace.get(name)
        for candidate in (value,) if _engine_class(value) else _exceptions_by_name().get(name, ()):
            yield "raises", candidate


def _shape_class(value: type) -> bool:
    """A model, a dataclass or a typed dict: a class whose constructor is its fields."""
    return issubclass(value, BaseModel) or dataclasses.is_dataclass(value) or typing.is_typeddict(value)


def _class_namespace(klass: type) -> dict[str, Any]:
    """Where a class's annotations resolve: the module its ``__module__`` names when that module holds it, else the
    module one of its own methods was written in."""
    module = sys.modules.get(klass.__module__)
    if module is not None and getattr(module, klass.__name__, None) is klass:
        return dict(vars(module))
    for member_name, member in vars(klass).items():
        if inspect.isfunction(member) and not member_name.startswith("__"):
            return dict(member.__globals__)
    return _namespace(klass)


def _class_edges(value: type) -> Iterator[tuple[Position, object]]:
    for base in value.__mro__[1:]:
        if _engine_class(base):
            yield "base", base
    for klass in value.__mro__:
        if not klass.__module__.startswith("threetears.evals"):
            continue
        namespace = _class_namespace(klass)
        for annotation_format in _FORMATS:
            try:
                annotations = annotationlib.get_annotations(klass, format=annotation_format)
            except NameError, TypeError:
                continue
            for field_name, annotation in annotations.items():
                if not field_name.startswith("_"):
                    for found in _types_in(annotation, namespace):
                        yield "field", found
        for member_name, member in vars(klass).items():
            if member_name.startswith("_") and member_name not in ("__init__", "__call__"):
                continue
            if member_name == "__init__" and _shape_class(value):
                continue  # a shape's constructor is its fields, read above
            if isinstance(member, property) and member.fget is not None:
                yield from _callable_edges(member.fget, namespace)
            elif isinstance(member, (staticmethod, classmethod)):
                yield from _callable_edges(member.__func__, namespace)
            elif inspect.isfunction(member):
                yield from _callable_edges(member, namespace)
    yield from _raises_edges(inspect.getdoc(value) or "", _class_namespace(value))


def edges(value: object) -> tuple[tuple[Position, object], ...]:
    """Each engine type a public object names, with where: a parameter or return of it or of a method, a field, or an
    exception its docstring says it raises."""
    if inspect.isclass(value):
        return tuple(_class_edges(value))
    if id(value) in _aliases():
        namespace = _alias_namespace(value)
        parts = (value.__value__,) if isinstance(value, typing.TypeAliasType) else typing.get_args(value)
        return tuple(("field", found) for part in parts for found in _types_in(part, namespace) if found is not value)
    if callable(value):
        return tuple(_callable_edges(value, _namespace(value)))
    return ()


@cache
def _alias_homes() -> Mapping[int, str]:
    """Each exported type alias's defining module, by ``id``: an alias's own ``__module__`` may name its origin's."""
    from public_surface import defining_module

    found: dict[int, str] = {}
    for root, name, obj in public_names():
        if id(obj) in _aliases() and id(obj) not in found:
            home = defining_module(root, name)
            found[id(obj)] = home[0] if home is not None else root
    return MappingProxyType(found)


def _alias_namespace(value: object) -> dict[str, Any]:
    module = sys.modules.get(_alias_homes().get(id(value), ""))
    return dict(vars(module)) if module else {}


# --- kinds -------------------------------------------------------------------------------------------------------


def _objects(names: typing.Iterable[str]) -> dict[str, Any]:
    wanted = set(names)
    found: dict[str, Any] = {}
    for root, name, obj in public_names():
        if f"{root}.{name}" in wanted:
            found[f"{root}.{name}"] = obj
    return found


def constant_edges(qualified: str) -> tuple[tuple[Position, object], ...]:
    """Each engine type a public constant's annotation names: what holding it hands a reader (``ROOT_EXTRAS`` hands
    a ``RootExtra``)."""
    from public_surface import defining_module

    root, name = qualified.rsplit(".", 1)
    found = defining_module(root, name)
    _, node, _ = assignment(root, name)
    if found is None or not isinstance(node, ast.AnnAssign):
        return ()
    module = sys.modules.get(found[0])
    namespace = dict(vars(module)) if module else {}
    return tuple(("field", each) for each in _types_in(ast.unparse(node.annotation), namespace))


#: Who names a type: each declared name that names it, with the position it is named in.
Holders = Mapping[int, set[tuple[Position, str]]]


def _holders(objects: Mapping[str, Any]) -> dict[int, set[tuple[Position, str]]]:
    """Every engine type the declared names name directly, by ``id``, with each position and the name holding it."""
    found: dict[int, set[tuple[Position, str]]] = {}
    for name, obj in objects.items():
        for position, each in edges(obj):
            if each is not obj:  # a class's own method returning the class says nothing of who holds it
                found.setdefault(id(each), set()).add((position, name))
    return found


def _callable_alias(obj: object) -> bool:
    origin = typing.get_origin(obj.__value__ if isinstance(obj, typing.TypeAliasType) else obj)
    return origin is not None and getattr(origin, "__name__", "") == "Callable"


#: The snapshot kinds of a shape: a class whose constructor is its fields.
_SHAPE_KINDS = ("model", "dataclass", "typed dict")

#: What each kind asks of the reader who meets it, as a test failure states it.
KIND_MEANINGS: Mapping[Kind, str] = MappingProxyType(
    {
        "typed": "a newcomer writes it in an eval file",
        "bound": "a host implements it or hands it to the engine",
        "invoked": "an operation called by name to make the engine do something, from host code or as an action",
        "received": "a value someone is handed and reads",
        "tests": "a kit, recorder or guard a host runs in its own tests",
    }
)


def _received_holder(holder: str, surface: Mapping[str, SurfaceEntry]) -> bool:
    """Whether a name holding a shape as a field or base is itself only read: a shape no entry makes a seam."""
    entry = surface.get(holder)
    return (
        entry is not None
        and entry.kind in (*_SHAPE_KINDS, "exception")
        and holder not in TYPED
        and holder not in UNTAUGHT_TYPED
        and KIND_ENTRIES.get(holder, "received") == "received"
    )


def _seam_holder(holder: str, surface: Mapping[str, SurfaceEntry]) -> bool:
    """Whether a name returning a shape is a seam whose return a host builds: a protocol a host implements, a type
    alias over the callable it hands the engine, or a function it hands the engine as a value. An engine class a
    host constructs is no such seam: what its methods return, the engine built."""
    entry = surface.get(holder)
    if entry is None:
        return False
    if entry.kind in ("protocol", "type alias"):
        return entry.kind == "protocol" or KIND_ENTRIES.get(holder) is None and "Callable" in entry.signature
    return entry.kind in ("function", "async function") and KIND_ENTRIES.get(holder) == "bound"


def kind_by_rule(
    qualified: str,
    obj: Any,
    entry: SurfaceEntry,
    holders: Holders,
    surface: Mapping[str, SurfaceEntry] | None = None,
) -> Kind | None:
    """A declared name's kind where a rule decides it with certainty, in this order, or ``None`` where none does.

    1. In :data:`TYPED` or :data:`UNTAUGHT_TYPED`: ``typed``.
    2. Under :data:`TESTING_ROOT`: ``tests``.
    3. A protocol: ``bound``.
    4. A function named as an action of the catalogue (:func:`action_names`), or in :data:`HELPERS`: ``invoked``.
    5. An exception, an enum, a literal or a constant: ``received``.
    6. A type alias over a ``Callable``: ``bound`` (a host hands the engine the callable); any other: ``received``.
    7. A model, dataclass or typed dict that a declared name other than a seam returns, or a received shape holds,
       and that no declared name takes as a parameter, no seam returns and no name but a received shape holds as a
       field: ``received``. A seam is a protocol, a callable alias or a function :data:`KIND_ENTRIES` records
       ``bound``: what it returns, a host builds.

    Every other function and class, and a shape a parameter or a seam names, has no rule: its kind is a judgement,
    recorded in :data:`KIND_ENTRIES`.
    """
    if qualified in TYPED or qualified in UNTAUGHT_TYPED:
        return "typed"
    if qualified.startswith(f"{TESTING_ROOT}."):
        return "tests"
    kind = entry.kind
    if kind == "protocol":
        return "bound"
    if kind in ("function", "async function"):
        return "invoked" if qualified.rsplit(".", 1)[1] in action_names() or qualified in HELPERS else None
    if kind in ("exception", "enum", "literal") or kind.startswith("constant"):
        return "received"
    if kind == "type alias":
        return "bound" if _callable_alias(obj) else "received"
    if kind in _SHAPE_KINDS:
        named = holders.get(id(obj), set())
        surface = surface or {}
        read = any(
            (position == "return" and not _seam_holder(holder, surface))
            or (position in ("field", "base") and _received_holder(holder, surface))
            for position, holder in named
        )
        seam = any(
            position == "parameter"
            or (position == "return" and _seam_holder(holder, surface))
            or (position in ("field", "base") and not _received_holder(holder, surface))
            for position, holder in named
        )
        return "received" if read and not seam else None
    return None


@cache
def action_names() -> frozenset[str]:
    """Every action the engine's catalogue holds, by name."""
    from threetears.evals.actions import eval_catalogue

    return frozenset(action.name for action in eval_catalogue().actions)


def how_to_decide(name: str, entry: SurfaceEntry) -> str:
    """What a declared name with no kind tells its author: the five kinds, and the question that places it."""
    meanings = "; ".join(f"{kind}, {meaning}" for kind, meaning in KIND_MEANINGS.items())
    return (
        f"{name} ({entry.kind}) has no kind, since no rule decides it: record it in KIND_ENTRIES in "
        f"packages/evals/scripts/surface_kinds.py as the one that fits ({meanings}), asking first whether a "
        "newcomer writes it, then whether a host implements it or hands it to the engine, then whether a caller "
        "invokes it to make the engine act, and otherwise it is only read"
    )


@dataclasses.dataclass(frozen=True)
class Kinds:
    """Every declared name's kind, and what is wrong with the record of them.

    :ivar of: ``root.name`` → its kind, for every declared name that has one, in the surface's order.
    :ivar problems: Each declared name with no kind, each :data:`KIND_ENTRIES` entry a rule already decides, and each
        entry for a name that is not declared.
    """

    of: Mapping[str, Kind]
    problems: tuple[str, ...]


def declared_kinds(surface: Mapping[str, SurfaceEntry]) -> Kinds:
    """Every declared name's kind: by rule (:func:`kind_by_rule`), else by :data:`KIND_ENTRIES`."""
    names = [name for name, entry in surface.items() if declared(entry.tier)]
    objects = _objects(names)
    holders = _holders(objects)
    of: dict[str, Kind] = {}
    problems: list[str] = []
    for name in names:
        ruled = kind_by_rule(name, objects.get(name), surface[name], holders, surface)
        entered = KIND_ENTRIES.get(name)
        if ruled is not None and entered is not None:
            problems.append(f"{name} is {ruled} by rule and also entered as {entered}: drop it from KIND_ENTRIES")
        kind = ruled or entered
        if kind is None:
            problems.append(how_to_decide(name, surface[name]))
        else:
            of[name] = kind
    problems += [f"{name} is in KIND_ENTRIES but is not declared" for name in sorted(set(KIND_ENTRIES) - set(names))]
    return Kinds(MappingProxyType(of), tuple(problems))


def counts(kinds: Mapping[str, Kind]) -> dict[Kind, int]:
    """How many declared names each kind holds."""
    return {kind: sum(1 for each in kinds.values() if each == kind) for kind in KINDS}


# --- shapes: what reaches each received name ---------------------------------------------------------------------


def _is_shape(entry: SurfaceEntry) -> bool:
    """A received name that is a type (a constant is a value, and nothing names it in a signature)."""
    return not entry.kind.startswith("constant")


@dataclasses.dataclass(frozen=True)
class Reach:
    """Which ``typed``, ``bound``, ``invoked`` or ``tests`` name reaches each received shape.

    :ivar by: ``root.name`` of a received shape → the declared name nearest it that reaches it: a parameter, return,
        field or raised error of that name, or of a type it reaches, names the shape. ``None`` when nothing does.
    """

    by: Mapping[str, str | None]


def shape_reach(surface: Mapping[str, SurfaceEntry], kinds: Mapping[str, Kind]) -> Reach:
    """Walk outward from every ``typed``, ``bound``, ``invoked`` and ``tests`` name (a kit hands back its report),
    nearest first, through every engine type it names, and record for each received shape the starting name that
    reached it first."""
    starts = [name for kind in ("typed", "bound", "invoked", "tests") for name in kinds if kinds[name] == kind]
    held = [name for name, kind in kinds.items() if kind == "received" and not _is_shape(surface[name])]
    shapes = [name for name, kind in kinds.items() if kind == "received" and _is_shape(surface[name])]
    objects = _objects([*starts, *shapes])
    names_of: dict[int, list[str]] = {}
    for name in shapes:
        names_of.setdefault(id(objects[name]), []).append(name)
    origin: dict[int, str] = {}
    frontier: list[tuple[object, str]] = []
    for name in starts:
        obj = objects[name]
        if id(obj) not in origin:
            origin[id(obj)] = name
            frontier.append((obj, name))
    for name in held:
        for _, found in constant_edges(name):
            if id(found) not in origin:
                origin[id(found)] = name
                frontier.append((found, name))
    while frontier:
        following: list[tuple[object, str]] = []
        for obj, start in frontier:
            for _, found in edges(obj):
                if id(found) not in origin:
                    origin[id(found)] = start
                    following.append((found, start))
        frontier = following
    return Reach(MappingProxyType({name: origin.get(id(objects[name])) for name in shapes}))


def unreached_shapes(reach: Reach) -> list[str]:
    """Every received shape no ``typed``, ``bound``, ``invoked`` or ``tests`` name reaches."""
    return [name for name, by in reach.by.items() if by is None]


def _fields_of(obj: object) -> frozenset[str] | None:
    if not inspect.isclass(obj) or issubclass(obj, BaseException):
        return None
    if issubclass(obj, BaseModel):
        return frozenset(obj.model_fields)
    if dataclasses.is_dataclass(obj):
        return frozenset(f.name for f in dataclasses.fields(obj))
    return None


def near_duplicate_shapes(kinds: Mapping[str, Kind]) -> list[tuple[str, str]]:
    """Pairs of received models or dataclasses (two objects, not one exported twice) whose fields are the same, or
    differ by one field, over at least four fields between them (information-architecture §8.2 (d)): in the surface's
    order. Each is a finding for review; :data:`NEAR_DUPLICATES` records today's, and the list may only shrink."""
    fields: list[tuple[str, int, frozenset[str]]] = []
    seen: set[int] = set()
    for name, obj in _objects(name for name, kind in kinds.items() if kind == "received").items():
        found = _fields_of(obj)
        if found is not None and id(obj) not in seen:
            seen.add(id(obj))
            fields.append((name, id(obj), found))
    return [
        (first, second)
        for index, (first, _, mine) in enumerate(fields)
        for second, _, theirs in fields[index + 1 :]
        if len(mine ^ theirs) <= 1 and len(mine | theirs) >= 4
    ]


# --- the ladder --------------------------------------------------------------------------------------------------

#: The ladder's page, whose ``## Rung N: ...`` headings title each rung.
LADDER_PAGE = PACKAGE_ROOT / "docs" / "ladder.md"

_RUNG_HEADING = re.compile(r"^## Rung (\d)[a-z]?: (.+)$", re.M)


def rung_titles() -> dict[int, str]:
    """Each rung's title, from the first heading the ladder's page gives it."""
    titles: dict[int, str] = {}
    for number, title in _RUNG_HEADING.findall(LADDER_PAGE.read_text(encoding="utf-8")):
        titles.setdefault(int(number), title)
    return titles


def rung_anchors() -> dict[int, str]:
    """Each rung's first heading on the ladder's page, as the anchor a link to it takes."""
    anchors: dict[int, str] = {}
    for heading in re.finditer(r"^## (Rung (\d)[a-z]?: .+)$", LADDER_PAGE.read_text(encoding="utf-8"), re.M):
        slug = re.sub(r"[^\w\- ]", "", heading[1].lower()).replace(" ", "-")
        anchors.setdefault(int(heading[2]), slug)
    return anchors


def example_imports(directory: Path) -> frozenset[str]:
    """Every ``root.name`` an example's Python files import from the package."""
    found: set[str] = set()
    for path in sorted(directory.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("threetears.evals"):
                found |= {f"{node.module}.{alias.name}" for alias in node.names}
    return frozenset(found)


def ladder_imports() -> dict[str, frozenset[str]]:
    """What each ladder example imports, read from its files."""
    return {name: example_imports(PACKAGE_ROOT / "examples" / name) for name in LADDER_EXAMPLES}


#: One ``from threetears.evals… import …`` statement on a page, parenthesised across lines or not.
_PAGE_IMPORT = re.compile(r"^[ \t]*from threetears\.evals(?:\.\w+)* import (?:\([^)]*\)|[^\n]*)", re.M)


def ladder_page_imports() -> dict[int, frozenset[str]]:
    """What each rung's sections of ``docs/ladder.md`` import in their code blocks, by rung."""
    text = LADDER_PAGE.read_text(encoding="utf-8")
    found: dict[int, set[str]] = {}
    headings = list(re.finditer(r"^## .*$", text, re.M))
    for index, heading in enumerate(headings):
        rung = _RUNG_HEADING.match(heading[0])
        if rung is None:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        for statement in _PAGE_IMPORT.finditer(text, heading.end(), end):
            tree = ast.parse(textwrap.dedent(statement[0]))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    found.setdefault(int(rung[1]), set()).update(f"{node.module}.{a.name}" for a in node.names)
    return {rung: frozenset(names) for rung, names in found.items()}


def taught_rungs() -> dict[str, int]:
    """Each name a rung teaches, with the first rung that does: what the rung's example imports, or what its sections
    of ``docs/ladder.md`` import."""
    by_rung: dict[int, set[str]] = {}
    for example, names in ladder_imports().items():
        by_rung.setdefault(LADDER_EXAMPLES[example], set()).update(names)
    for rung, names in ladder_page_imports().items():
        by_rung.setdefault(rung, set()).update(names)
    taught: dict[str, int] = {}
    for rung in sorted(by_rung):
        for name in sorted(by_rung[rung]):
            taught.setdefault(name, rung)
    return taught


def typed_through(rung: int) -> int:
    """How many typed names rungs 0 to ``rung`` teach together."""
    return sum(1 for first in TYPED.values() if first <= rung)


def objects_of(names: typing.Iterable[str]) -> dict[str, Any]:
    """``root.name`` → the object its root exports, for each name given, in walk order. A one-minor shim's warning is
    not the reader's."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return _objects(names)


def module_of(qualified: str) -> Any:
    """The public root a ``root.name`` is exported from."""
    return importlib.import_module(qualified.rsplit(".", 1)[0])
