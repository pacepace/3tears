"""The analysis reporter kind's launch names, kept for one minor after the reporter cases retired.

A host registered the reporter kind's launch entry with these, and binds them at import, so each stays importable
from :mod:`threetears.evals.ops` and warns on every read (:func:`~threetears.evals.kernel.retired.served_retired`).
None of them launches anything: each operation refuses, naming the writer kind that replaces the reporter, and the
types are kept only so a host's annotations still resolve. The next minor removes them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.retired import REPORTER_RETIRED, Retired
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.run.launch import ArmPlan, CallCeiling, LaunchArgument, LaunchRequest
from threetears.evals.schema.models import EvalTestCase

#: Retired: the launch arguments a reporter arm could not honour. No reporter arm launches now.
REPORTER_UNHONOURED_LAUNCH_ARGUMENTS: frozenset[LaunchArgument] = frozenset(
    {"simulator_model", "cassette_mode", "n_variations"}
)

#: Retired: which model a reporter launch asked the host to resolve, the memo writer or its judge.
ReporterModelRole = Literal["generator", "judge"]

#: Retired: the host's model resolution, as a reporter launch asked it.
ReporterModelResolver = Callable[[ReporterModelRole, str | None], str]


@dataclass(frozen=True)
class ReporterAdmitted:
    """Retired: what a reporter arm resolved to; :func:`admit_reporter_launch` now refuses every arm.

    :ivar preset: The memo writer's prompt preset.
    :ivar prompt: Its text.
    :ivar candidate_model: The memo writer's model.
    :ivar judge_model: The judge's model.
    :ivar cases: The cases the arm would have run.
    """

    preset: str | None
    prompt: str
    candidate_model: str
    judge_model: str
    cases: tuple[EvalTestCase, ...]


def _refused(request: LaunchRequest) -> ValidationFailedError:
    """The refusal every retired reporter launch operation raises."""
    return ValidationFailedError(
        f"template {request.template.id!r}: {REPORTER_RETIRED}",
        remedy="create a writer template (writer_template_create), mint its cases from the same campaigns "
        "(writer_case_mint) and launch that",
    )


def admit_reporter_launch(
    request: LaunchRequest, *, storage: EvalStorage, prompts: object, resolve_model: ReporterModelResolver
) -> ReporterAdmitted:
    """Retired: refuses every reporter arm, naming the writer kind that replaces it.

    :raises ValidationFailedError: Always: the analysis reporter kind is retired.
    """
    del storage, prompts, resolve_model
    raise _refused(request)


def plan_reporter_arm(
    host: EvalHost, request: LaunchRequest, *, prompts: object, resolve_model: ReporterModelResolver
) -> ArmPlan:
    """Retired: refuses every reporter arm's plan, naming the writer kind that replaces it.

    :raises ValidationFailedError: Always: the analysis reporter kind is retired.
    """
    del host, prompts, resolve_model
    raise _refused(request)


async def price_largest_reporter_call(
    host: EvalHost, request: LaunchRequest, *, prompts: object, resolve_model: ReporterModelResolver
) -> CallCeiling | None:
    """Retired: refuses every reporter arm's price, naming the writer kind that replaces it.

    :raises ValidationFailedError: Always: the analysis reporter kind is retired.
    """
    del host, prompts, resolve_model
    raise _refused(request)


#: Each retired name :mod:`threetears.evals.ops` serves, with why it retired.
RETIRED: Mapping[str, Retired] = MappingProxyType(
    {
        "REPORTER_UNHONOURED_LAUNCH_ARGUMENTS": Retired(REPORTER_UNHONOURED_LAUNCH_ARGUMENTS, REPORTER_RETIRED),
        "ReporterAdmitted": Retired(ReporterAdmitted, REPORTER_RETIRED),
        "ReporterModelResolver": Retired(ReporterModelResolver, REPORTER_RETIRED),
        "ReporterModelRole": Retired(ReporterModelRole, REPORTER_RETIRED),
        "admit_reporter_launch": Retired(admit_reporter_launch, REPORTER_RETIRED),
        "plan_reporter_arm": Retired(plan_reporter_arm, REPORTER_RETIRED),
        "price_largest_reporter_call": Retired(price_largest_reporter_call, REPORTER_RETIRED),
    }
)

__all__ = [
    "REPORTER_UNHONOURED_LAUNCH_ARGUMENTS",
    "RETIRED",
    "ReporterAdmitted",
    "ReporterModelResolver",
    "ReporterModelRole",
    "admit_reporter_launch",
    "plan_reporter_arm",
    "price_largest_reporter_call",
]
