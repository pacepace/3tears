"""The analysis reporter kind's names a host binds, kept for one minor after the kind retired.

A host declared, registered and launched the reporter kind with these, and binds them at import, so each stays
importable from :mod:`threetears.evals.analysis` and warns on every read
(:func:`~threetears.evals.kernel.retired.served_retired`). None of them makes the kind launchable: the contract holds
no overlay and seats nothing, the template check refuses every reporter template, and building either reporter kind
refuses, each naming the writer kind that replaces it. :data:`REPORTER_KIND` still names the kind a stored reporter
template declares, and :func:`reporter_cell_timeout_s` still answers the ceiling it did. The next minor removes them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, NoReturn

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink, CellSpanWindow, VariantConfig
from threetears.evals.kernel.cassettes import CellCassettes
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.kinds import KindContract
from threetears.evals.kernel.inner_ceilings import judge_phase_ceiling_s
from threetears.evals.kernel.retired import REPORTER_RETIRED, Retired
from threetears.evals.schema.completion import RequestCeiling
from threetears.evals.schema.models import EvalTemplate, EvalTestCase, JudgedArtifact

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


#: Each retired name :mod:`threetears.evals.analysis` serves, with why it retired.
RETIRED: Mapping[str, Retired] = MappingProxyType(
    {
        "AsRecordedReporterKind": Retired(AsRecordedReporterKind, REPORTER_RETIRED),
        "REPORTER_KIND": Retired(REPORTER_KIND, REPORTER_RETIRED),
        "REPORTER_KIND_CONTRACT": Retired(REPORTER_KIND_CONTRACT, REPORTER_RETIRED),
        "ReporterKind": Retired(ReporterKind, REPORTER_RETIRED),
        "ReporterOverlays": Retired(ReporterOverlays, REPORTER_RETIRED),
        "refuse_undeliverable_reporter_template": Retired(refuse_undeliverable_reporter_template, REPORTER_RETIRED),
        "reporter_cell_timeout_s": Retired(reporter_cell_timeout_s, REPORTER_RETIRED),
        "reporter_kind_contract": Retired(reporter_kind_contract, REPORTER_RETIRED),
    }
)

__all__ = [
    "REPORTER_KIND",
    "REPORTER_KIND_CONTRACT",
    "RETIRED",
    "AsRecordedReporterKind",
    "ReporterKind",
    "ReporterOverlays",
    "refuse_undeliverable_reporter_template",
    "reporter_cell_timeout_s",
    "reporter_kind_contract",
]
