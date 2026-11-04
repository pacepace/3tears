"""The names the retired generations left that a host binds, kept for one minor after each retired.

**The analysis reporter kind.** A host declared, registered and launched the reporter kind with these, and binds them at import, so each stays
importable from :mod:`threetears.evals.analysis` and warns on every read
(:func:`~threetears.evals.kernel.retired.served_retired`). None of them makes the kind launchable: the contract holds
no overlay and seats nothing, the template check refuses every reporter template, and building either reporter kind
refuses, each naming the writer kind that replaces it. :data:`REPORTER_KIND` still names the kind a stored reporter
template declares, and :func:`reporter_cell_timeout_s` still answers the ceiling it did.

**The report planner and the memo writer.** A host seeds a prompt slot from :data:`EVAL_REPORT_PLAN_DEFAULT` and
:data:`EVAL_ANALYSIS_GEN_DEFAULT` by name, through ``importlib``, so each still resolves: to one line saying the
generation it steered is retired, never to the prompt it was. No generation reads either slot. The memo writer's
client settings (:func:`analysis_gen_request_settings_for`) still answer what they did, for a host that builds a
client from them; nothing in the package builds one. The memo writer itself (:func:`generate_analysis`) keeps its
signature and refuses every call, naming ``analysis_generate``.

The next minor removes them all.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink, CellSpanWindow, VariantConfig
from threetears.evals.kernel.cassettes import CellCassettes
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.kinds import KindContract
from threetears.evals.kernel.inner_ceilings import judge_phase_ceiling_s
from threetears.evals.kernel.retired import REPORTER_RETIRED, Retired
from threetears.evals.schema.completion import RequestCeiling
from threetears.evals.schema.models import (
    ClientRequestSettings,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    KnownReasoningEffort,
)

if TYPE_CHECKING:
    from threetears.evals.analysis.campaign_bundle import CampaignBundle
    from threetears.evals.analysis.generation_tally import GenerationTally
    from threetears.evals.kernel.campaign import EvalAnalysis, EvalInsight
    from threetears.evals.kernel.host.profile import HostProfile
    from threetears.evals.schema.completion import CompletionGenerator

#: Retired: the ``candidate_kind`` a reporter template declares. A stored one still declares it, and no host
#: registers it now, so its launch is refused as a kind the host has no launcher for.
REPORTER_KIND = "analysis_reporter"

#: What building a retired reporter kind, or anything it would have done, is refused with.
_REMEDY = (
    "create a writer template (writer_template_create) and mint its cases from the same campaigns (writer_case_mint)"
)

#: Requests one reporter cell's memo writer could make: the memo and one repair round-trip.
_MEMO_WRITER_CALLS = 2

#: Retired: the analysis reporter kind's contract, now one with no overlay and no seat, which makes nothing
#: launchable. Evaluate the report writer with the ``report_writer`` kind (``writer_template_create``).
REPORTER_KIND_CONTRACT = KindContract(REPORTER_KIND)


def reporter_kind_contract(host_seats: Iterable[str] = ()) -> KindContract:
    """Retired: the analysis reporter kind's contract, the host's seats ignored; :data:`REPORTER_KIND_CONTRACT`.

    :param host_seats: Ignored: the retired kind seats nothing.
    :ptype host_seats: Iterable[str]
    :return: :data:`REPORTER_KIND_CONTRACT`.
    :rtype: KindContract
    """
    del host_seats
    return REPORTER_KIND_CONTRACT


class ReporterOverlays(BaseModel):
    """Retired: the analysis reporter's one overlay, the memo writer's prompt preset; no launch reads it."""

    model_config = ConfigDict(frozen=True)

    prompt_preset: str | None = Field(default=None, min_length=1, description="retired with the memo writer")


def refuse_undeliverable_reporter_template(template: EvalTemplate) -> None:
    """Retired: refuses every analysis reporter template, naming the writer kind that replaces it.

    :param template: The template a host's check is asked about.
    :ptype template: EvalTemplate
    :raises ValidationFailedError: Always: the analysis reporter kind is retired.
    """
    raise ValidationFailedError(f"template {template.id!r}: {REPORTER_RETIRED}", remedy=_REMEDY)


def reporter_cell_timeout_s(
    *,
    judge_dims: int,
    judge_concurrency: int,
    generator_max_tokens: int,
    judge_call_attempts: int,
    judge_max_tokens: int,
    request_s: RequestCeiling,
) -> float:
    """Retired: the wall-clock ceiling one reporter cell had, the memo writer's two requests then its judge phase.

    :param judge_dims: Rubric dimensions the run's judge scores on each cell.
    :ptype judge_dims: int
    :param judge_concurrency: Judge requests in flight at once.
    :ptype judge_concurrency: int
    :param generator_max_tokens: The memo writer's output cap.
    :ptype generator_max_tokens: int
    :param judge_call_attempts: Requests one judge dimension can make, its parse retries included.
    :ptype judge_call_attempts: int
    :param judge_max_tokens: The judge's output cap.
    :ptype judge_max_tokens: int
    :param request_s: The host's ceiling for one request on its client, by output cap.
    :ptype request_s: RequestCeiling
    :return: The cell ceiling in seconds.
    :rtype: float
    """
    judging_s = judge_phase_ceiling_s(
        judge_dims=judge_dims,
        judge_concurrency=judge_concurrency,
        judge_call_attempts=judge_call_attempts,
        judge_max_tokens=judge_max_tokens,
        request_s=request_s,
    )
    return _MEMO_WRITER_CALLS * request_s(generator_max_tokens) + judging_s


def _refused(kind: str) -> ValidationFailedError:
    """The refusal building, preparing or invoking a retired reporter kind raises."""
    return ValidationFailedError(f"{kind}: {REPORTER_RETIRED}", remedy=_REMEDY)


class _RetiredReporterKind:
    """What both retired reporter kinds share: the candidate-kind shape a host's annotations name, and no behaviour."""

    #: The memo, judged as a document, as the kind declared it.
    judged_artifact = JudgedArtifact.DOCUMENT
    #: No limit of the kind's own, as the kind declared it.
    max_concurrent_cells: int | None = None

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
        world: Any,
    ) -> NoReturn:
        """Refuses: the analysis reporter kind is retired.

        :raises ValidationFailedError: Always.
        """
        raise _refused(type(self).__name__)

    async def invoke(self, instance: Any, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Refuses: the analysis reporter kind is retired.

        :raises ValidationFailedError: Always.
        """
        raise _refused(type(self).__name__)


class ReporterKind(_RetiredReporterKind):
    """Retired: the memo writer as a candidate. Building one refuses, naming the writer kind that replaces it."""

    def __init__(
        self, *, prompt: str, prompt_id: str, prompt_version: str | None, client: Any, model: str, host: Any
    ) -> None:
        """Refuses: the memo writer this kind ran is retired.

        :raises ValidationFailedError: Always.
        """
        del prompt, prompt_id, prompt_version, client, model, host
        raise _refused("ReporterKind")


class AsRecordedReporterKind(_RetiredReporterKind):
    """Retired: a case's recorded memo replayed. Building one refuses, naming the writer kind that replaces it."""

    def __init__(self) -> None:
        """Refuses: the reporter cases it replayed are retired.

        :raises ValidationFailedError: Always.
        """
        raise _refused("AsRecordedReporterKind")


#: Why the report planner's prompt constant retired, and what writes a report now, as its warning says it.
PLANNER_PROMPT_RETIRED = (
    "the report planner is retired; analysis_generate writes with the report writer, which reads no prompt slot: "
    "drop the eval_report_plan slot"
)

#: Why the memo writer's prompt constant retired, and what writes a report now, as its warning says it.
MEMO_PROMPT_RETIRED = (
    "the memo writer is retired; analysis_generate writes with the report writer, which reads no prompt slot: drop "
    "the eval_analysis_gen slot"
)

#: Why the memo writer retired, and what writes a report now, as its warning and its refusal say it.
MEMO_WRITER_RETIRED = (
    "the memo writer is retired; analysis_generate writes a campaign's report with the report writer (the analyst and "
    "the report editor)"
)

#: What a call to the retired memo writer is told to do instead.
_MEMO_WRITER_REMEDY = "generate the campaign's report with analysis_generate (threetears.evals.ops.analysis_generate)"

#: Why the memo writer's client settings retired, and what bounds the report writer, as their warning says it.
MEMO_SETTINGS_RETIRED = (
    "the memo writer's client role is retired; the report writer runs on the host's analyst client, its turns bounded "
    "by AnalysisGeneration.limits (WriterLimits)"
)

#: The least reasoning headroom :func:`analysis_gen_request_settings_for` takes, as it took it.
_MIN_MEMO_WRITER_HEADROOM_TOKENS = 1024


def analysis_gen_request_settings_for(
    *,
    answer_budget_tokens: int,
    reasoning_effort: KnownReasoningEffort = "medium",
    reasoning_headroom_tokens: int = 8192,
) -> ClientRequestSettings:
    """Retired: the memo writer's client settings, answered as they were; nothing in the package builds that client.

    The output cap is the answer budget plus the reasoning headroom, the request asks for reasoning by effort, and
    strict output is always set.

    :param answer_budget_tokens: Output room for the memo itself.
    :ptype answer_budget_tokens: int
    :param reasoning_effort: How hard the writer reasoned, as an effort word.
    :ptype reasoning_effort: KnownReasoningEffort
    :param reasoning_headroom_tokens: Output room left for reasoning, which sizes the cap; at least 1,024.
    :ptype reasoning_headroom_tokens: int
    :return: The settings the memo writer's client was built with.
    :rtype: ClientRequestSettings
    :raises ValueError: ``reasoning_headroom_tokens`` is below 1,024.
    """
    if reasoning_headroom_tokens < _MIN_MEMO_WRITER_HEADROOM_TOKENS:
        raise ValueError(
            f"reasoning_headroom_tokens={reasoning_headroom_tokens} leaves the analysis generator too little room to "
            f"think at any effort: the least is {_MIN_MEMO_WRITER_HEADROOM_TOKENS} (a host's old "
            "*_reasoning_budget_tokens setting is this headroom now)"
        )
    return ClientRequestSettings(
        max_tokens=answer_budget_tokens + reasoning_headroom_tokens,
        reasoning_effort=reasoning_effort,
        strict_output=True,
    )


async def generate_analysis(
    bundle: CampaignBundle,
    *,
    prompt: str,
    model: str,
    client: CompletionGenerator,
    prompt_id: str,
    bundle_assembled_at: str,
    prompt_version: str | None = None,
    tally: GenerationTally | None = None,
    admit: Callable[[str, str, dict[str, Any] | None], None] | None = None,
    profile: HostProfile,
    measuring_writers: bool = False,
) -> tuple[EvalAnalysis, list[EvalInsight]]:
    """Retired: the memo writer. Every call refuses before anything is sent, naming what writes a report now.

    :raises ValidationFailedError: Always: the memo writer is retired.
    """
    del bundle, prompt, model, client, prompt_id, bundle_assembled_at, prompt_version, tally, admit, profile
    del measuring_writers
    raise ValidationFailedError(f"generate_analysis: {MEMO_WRITER_RETIRED}", remedy=_MEMO_WRITER_REMEDY)


#: Retired: what the report planner's prompt slot is seeded from now, one line, never the planner's prompt.
EVAL_REPORT_PLAN_DEFAULT = (
    "Retired: the report planner no longer writes analyses, and no generation reads this eval_report_plan slot."
)

#: Retired: what the memo writer's prompt slot is seeded from by name now, one line, never the memo writer's prompt.
EVAL_ANALYSIS_GEN_DEFAULT = (
    "Retired: the memo writer no longer writes analyses, and no generation reads this eval_analysis_gen slot."
)

#: Each retired name :mod:`threetears.evals.analysis` serves, with why it retired.
RETIRED: Mapping[str, Retired] = MappingProxyType(
    {
        "AsRecordedReporterKind": Retired(AsRecordedReporterKind, REPORTER_RETIRED),
        "EVAL_ANALYSIS_GEN_DEFAULT": Retired(EVAL_ANALYSIS_GEN_DEFAULT, MEMO_PROMPT_RETIRED),
        "EVAL_REPORT_PLAN_DEFAULT": Retired(EVAL_REPORT_PLAN_DEFAULT, PLANNER_PROMPT_RETIRED),
        "REPORTER_KIND": Retired(REPORTER_KIND, REPORTER_RETIRED),
        "REPORTER_KIND_CONTRACT": Retired(REPORTER_KIND_CONTRACT, REPORTER_RETIRED),
        "ReporterKind": Retired(ReporterKind, REPORTER_RETIRED),
        "ReporterOverlays": Retired(ReporterOverlays, REPORTER_RETIRED),
        "analysis_gen_request_settings_for": Retired(analysis_gen_request_settings_for, MEMO_SETTINGS_RETIRED),
        "generate_analysis": Retired(generate_analysis, MEMO_WRITER_RETIRED),
        "refuse_undeliverable_reporter_template": Retired(refuse_undeliverable_reporter_template, REPORTER_RETIRED),
        "reporter_cell_timeout_s": Retired(reporter_cell_timeout_s, REPORTER_RETIRED),
        "reporter_kind_contract": Retired(reporter_kind_contract, REPORTER_RETIRED),
    }
)

__all__ = [
    "EVAL_ANALYSIS_GEN_DEFAULT",
    "EVAL_REPORT_PLAN_DEFAULT",
    "MEMO_PROMPT_RETIRED",
    "MEMO_SETTINGS_RETIRED",
    "MEMO_WRITER_RETIRED",
    "PLANNER_PROMPT_RETIRED",
    "REPORTER_KIND",
    "REPORTER_KIND_CONTRACT",
    "RETIRED",
    "AsRecordedReporterKind",
    "ReporterKind",
    "ReporterOverlays",
    "analysis_gen_request_settings_for",
    "generate_analysis",
    "refuse_undeliverable_reporter_template",
    "reporter_cell_timeout_s",
    "reporter_kind_contract",
]
