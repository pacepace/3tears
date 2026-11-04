"""The analysis reporter kind's contract names, kept for one minor after the reporter cases retired.

A host declared the reporter kind with these, and binds them at import, so each stays importable from
:mod:`threetears.evals.analysis` and warns on every read (:func:`~threetears.evals.kernel.retired.served_retired`).
None of them makes the kind launchable: the contract holds no overlay and seats nothing, and the template check
refuses every reporter template, naming the writer kind that replaces it. The next minor removes them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.kinds import KindContract
from threetears.evals.kernel.retired import REPORTER_RETIRED, Retired
from threetears.evals.schema.models import EvalTemplate

#: Retired: the analysis reporter kind's contract, now one with no overlay and no seat, which makes nothing
#: launchable. Evaluate the report writer with the ``report_writer`` kind (``writer_template_create``).
REPORTER_KIND_CONTRACT = KindContract("analysis_reporter")


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
    raise ValidationFailedError(
        f"template {template.id!r}: {REPORTER_RETIRED}",
        remedy="create a writer template (writer_template_create) and mint its cases from the same campaigns "
        "(writer_case_mint)",
    )


#: Each retired name :mod:`threetears.evals.analysis` serves, with why it retired.
RETIRED: Mapping[str, Retired] = MappingProxyType(
    {
        "REPORTER_KIND_CONTRACT": Retired(REPORTER_KIND_CONTRACT, REPORTER_RETIRED),
        "ReporterOverlays": Retired(ReporterOverlays, REPORTER_RETIRED),
        "refuse_undeliverable_reporter_template": Retired(refuse_undeliverable_reporter_template, REPORTER_RETIRED),
        "reporter_kind_contract": Retired(reporter_kind_contract, REPORTER_RETIRED),
    }
)

__all__ = [
    "REPORTER_KIND_CONTRACT",
    "RETIRED",
    "ReporterOverlays",
    "refuse_undeliverable_reporter_template",
    "reporter_kind_contract",
]
