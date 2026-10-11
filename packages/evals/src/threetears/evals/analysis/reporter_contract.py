"""The analysis reporter's kind contract, and the refusals every reporter launch makes whatever host makes it.

The reporter is the analysis writer evaluated as a candidate (:mod:`~threetears.evals.analysis.reporter_kind`):
each case is one campaign's frozen analysis bundle, and each candidate a (prompt preset, generator model) pair,
or the memo each campaign actually got, replayed. What a host must add to launch one is only what the host
alone knows — which prompt a preset names, and which model a name resolves to. Everything else is the same for
every host, so it is declared here once:

- :data:`REPORTER_KIND_CONTRACT` — the kind's :class:`~threetears.evals.kernel.host.KindContract`, as the judge
  kind and a host's own kinds have one: its one overlay, the prompt preset (:class:`ReporterOverlays`), and the
  seats its runs fill. A host whose profile declares apparatus of its own that a reporter run records builds
  its contract with :func:`reporter_kind_contract` instead, seating those too.
- :func:`refuse_undeliverable_reporter_template` — a reporter template asking for apparatus a memo writer
  cannot honour, or giving its judge nothing to score. For a host's ``refuse_undeliverable_template`` hook
  (:func:`~threetears.evals.run.create_template`) on a reporter template, and asked again at every launch.
- :func:`live_reporter_cases` — the cases a launch over the template's own bank runs, refused when there are
  none (a reporter case is frozen from a campaign, never generated) and when a (campaign, recorded memo) pair
  holds more than one live case, which would judge one memo twice and count it twice in every denominator.

The launch's own refusals — an unknown preset, a model named by alias, a replay over a case with no recorded
memo — need the launch request, which this package cannot read, and are made by
:func:`~threetears.evals.ops.admit_reporter_launch` over a host's prompt and model hooks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.reporter_kind import REPORTER_KIND, ReporterCase
from threetears.evals.analysis.service import AnalysisStore, reporter_case_bank
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.kinds import KindContract
from threetears.evals.schema.models import EvalTemplate, EvalTestCase


class ReporterOverlays(BaseModel):
    """The one knob a reporter launch may turn: which preset of the analysis generator's prompt writes the memos.

    The model is the arm's candidate model, which the engine already keys a variant by; the preset is the rest
    of the candidate, declared as an overlay so it is a lever (``analysis_reporter.prompt_preset``) two arms
    are told apart by. It names the preset; what the preset's text WAS when the run launched is the host's
    to record on the subject, since only the host's registry can say.
    """

    model_config = ConfigDict(frozen=True)

    prompt_preset: str | None = Field(
        default=None,
        min_length=1,
        description="the analysis generator's prompt preset the memos are written with, by name; none runs the "
        "registry's system default",
    )


#: The seats every reporter run fills: it is judged (its memo is scored on the template's rubric) and it spends
#: under a ceiling (each generation is a paid call). It drives no simulated user, so the simulator's
#: dimensions are not blanks on its runs but things they do not have.
_REPORTER_SEATS: frozenset[str] = frozenset({"judge", "max_cost_usd"})


def reporter_kind_contract(host_seats: Iterable[str] = ()) -> KindContract:
    """The reporter's contract, also seating apparatus dimensions of the host's own that its runs record.

    A profile refuses a contract leaving unseated an apparatus dimension whose blank is a real recorded
    level (one without ``indeterminate_when_blank``), since it could never be omitted; a host declaring such
    a dimension that every run records — reporter runs included — names it here.

    Args:
        host_seats: The host's own apparatus dimensions, or pinned roles, a reporter run fills.

    Returns:
        The contract: :class:`ReporterOverlays`, no spec, and the engine's seats with ``host_seats``.
    """
    return KindContract(REPORTER_KIND, overlays=ReporterOverlays, seats=_REPORTER_SEATS | frozenset(host_seats))


#: The reporter kind's contract, for a profile's ``kinds``: its overlay is the prompt preset, and it seats the
#: judge and the spend ceiling. :func:`reporter_kind_contract` builds it with a host's own seats added.
REPORTER_KIND_CONTRACT = reporter_kind_contract()

#: A reporter template's fields no memo writer honours, each with what it would ask of one.
_UNDELIVERABLE: Mapping[str, str] = {
    "conversation": "a simulated conversation (a reporter writes one memo per case)",
    "goal_state_checks": "goal checks (a reporter acts on no world; its memo is judged)",
    "preconditions": "preconditions (a reporter acts on no world)",
    "tools_allowed": "tools (a reporter calls none)",
    "tools_required": "tools (a reporter calls none)",
    "variation_axes": "variation axes (a reporter case is frozen from a campaign, never generated)",
}


def refuse_undeliverable_reporter_template(template: EvalTemplate) -> None:
    """Refuse a reporter template that asks for apparatus a memo writer cannot honour, or gives its judge nothing to score.

    A reporter cell is one memo over a frozen bundle, judged on the template's rubric. So a conversation, goal
    checks, preconditions, tools, variation axes or a world seed would each be declared and silently never
    run; and a template with no rubric would have every memo written and paid for with nothing to judge it on.

    Args:
        template: A template declaring the reporter kind.

    Raises:
        ValidationFailedError: The template declares any of those, or no rubric; every defect is named at once.
    """
    asked = [why for name, why in _UNDELIVERABLE.items() if getattr(template, name)]
    if template.world_seed.namespaces:
        asked.append("a world seed (a reporter acts on no world)")
    if not template.rubric:
        asked.append("no rubric (a reporter's memo is judged, so its template must say on what)")
    if asked:
        raise ValidationFailedError(f"reporter template {template.name!r} declares " + "; ".join(asked))


def live_reporter_cases(
    storage: AnalysisStore, template: EvalTemplate, scope_id: str
) -> list[tuple[EvalTestCase, ReporterCase]]:
    """The cases a launch over the template's own bank runs: its live reporter cases, oldest first.

    Read through the one derivation of liveness (:func:`~threetears.evals.analysis.reporter_case_bank`), the
    one the freeze and the calibration read use, so a launch, its price and a calibration count the same cases
    — and a case this build cannot read is refused there, since it may supersede one that would otherwise
    read as live.

    Args:
        storage: Where the template's cases are stored.
        template: The reporter template.
        scope_id: The scope its cases live in.

    Returns:
        ``(stored case, reporter case)`` per live case, ordered by creation then id, so two launches over one
        bank freeze one list.

    Raises:
        ValidationFailedError: A stored reporter case of the template does not read; a (campaign, recorded
            memo) pair holds more than one live case; or none is live.
    """
    bank = reporter_case_bank(storage, template.id, scope_id)
    if ambiguous := bank.ambiguous:
        pairs = "; ".join(
            f"campaign {pair.campaign_id!r} with recorded memo {pair.recorded_analysis_id!r} "
            f"({', '.join(pair.live_case_ids)})"
            for pair in ambiguous
        )
        raise ValidationFailedError(
            f"template {template.id!r} holds more than one live case for {pairs}. Launching would judge each "
            "campaign's memo twice and count it twice in every denominator, so nothing was launched or priced; "
            "freeze the pair once more with supersedes naming every live case of it"
        )
    cases = bank.live
    if not cases:
        raise ValidationFailedError(
            f"template {template.id!r} has no live reporter case in scope {scope_id!r}, so there is nothing to run; "
            "a reporter case is frozen from a campaign's analysis bundle, never generated — freeze one first"
        )
    return sorted(cases, key=lambda pair: (pair[0].created_at, pair[0].id))


__all__ = [
    "REPORTER_KIND_CONTRACT",
    "ReporterOverlays",
    "live_reporter_cases",
    "refuse_undeliverable_reporter_template",
    "reporter_kind_contract",
]
