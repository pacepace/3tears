"""Campaigns of several aspects, for the tests of a campaign's readings and its page.

Two kinds. :func:`measured_aspect` runs a real eval (the quick path, twenty cases × 3 repeats, production against a
candidate) and reads it as one declared aspect with the label asked for, its bundle assembled as a host's would be:
what a campaign's page folds. :func:`varied` takes such a bundle and replaces the per-case pass shares its readings
read (the overall pass rate's ``floor_contrasts`` values and the scoreboard's counts), so a test sets each aspect's
evidence exactly, many times over, from one run; :func:`campaign_of` holds such aspects together as one campaign. The
two arms are named the same in every run, so their variant keys are the same in every aspect.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals import Answer, Eval
from threetears.evals.analysis import (
    AnalysisContextBundle,
    assemble_context_bundle,
    aspect_view,
    build_code_only_report,
    get_campaign,
)
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.report import Report
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignArm, DecisionClause, Floor

from packages.evals.tests.factories import make_aspect_campaign
from packages.evals.tests.off_loop import off_loop

#: The campaign's question, asked well before any run.
QUESTION = "Does the candidate pass more support requests than production, without losing any aspect?"

#: When the question was asked: before every run, so every aspect's readings are confirmatory.
ASKED_AT = "2000-01-01T00:00:00+00:00"

#: The cases each aspect runs, and the repeats of each.
CASES = 20
REPEATS = 3

#: What a reader calls the candidate, and the control.
CANDIDATE_LABEL = "v2"
CONTROL_LABEL = "production"

#: What a reader calls the second candidate, where one runs: its arm's name in the run too.
THIRD_LABEL = "v3"

#: When every aspect's own page was assembled.
ASSEMBLED_AT = "2026-10-18T00:00:00+00:00"


def _first_cases(right: int) -> Callable[[Mapping[str, Any]], Coroutine[Any, Any, Answer]]:
    """A runner answering right on every trial of the first ``right`` cases and wrong on the rest."""

    async def run(case: Mapping[str, Any]) -> Answer:
        index = int(case["text"].rsplit(" ", 1)[1])
        return Answer(
            case["label"] if index < right else "skipped", model="m", input_tokens=100, output_tokens=10, cost_usd=0.002
        )

    return run


@dataclass(frozen=True)
class MeasuredAspect:
    """One aspect measured for real: its one-aspect campaign, its bundle, its own page and the arms' keys; ``third``
    the second candidate's, where one ran."""

    campaign: EvalCampaign
    bundle: AnalysisContextBundle
    report: Report
    control: str
    candidate: str
    third: str | None = None


async def measured_aspect(
    label: str,
    *,
    production: int,
    candidate: int,
    margin: float | None = 10,
    floors: Sequence[DecisionClause] = (),
    asked_at: str | None = ASKED_AT,
    third: int | None = None,
) -> MeasuredAspect:
    """A real run read as one declared aspect labelled ``label``: production passing its first ``production`` cases
    and the candidate its first ``candidate``, every trial of each, the question asked at ``asked_at``.

    :param label: The aspect's label.
    :ptype label: str
    :param production: The cases production passes.
    :ptype production: int
    :param candidate: The cases the candidate passes.
    :ptype candidate: int
    :param margin: The aspect's margin.
    :ptype margin: float | None
    :param floors: The clauses of the floors on it.
    :ptype floors: Sequence[DecisionClause]
    :param asked_at: When the question was asked; None asks it now, after the runs.
    :ptype asked_at: str | None
    :param third: The cases a second candidate (:data:`THIRD_LABEL`) passes; None runs none.
    :ptype third: int | None
    :return: The aspect.
    :rtype: MeasuredAspect
    """
    cases = [{"text": f"support request {index}", "label": "queued"} for index in range(CASES)]
    arms = {"production": _first_cases(production), "candidate": _first_cases(candidate)}
    if third is not None:
        arms[THIRD_LABEL] = _first_cases(third)
    comparison = await Eval(cases, arms, ["label"], k=REPEATS, name=label).run(store=":memory:")
    storage = comparison.host.storage
    stored = off_loop(get_campaign, storage, comparison.campaign_id, comparison.scope_id)
    view = aspect_view(stored)
    declared = view.declared
    assert declared is not None and declared.control is not None
    when = asked_at if asked_at is not None else None
    campaign = make_aspect_campaign(
        label=label,
        template_id=view.template_id,
        run_ids=view.run_ids,
        design=declared.design.model_copy(update={"merit_axes": []}),
        question=QUESTION,
        asked_at=when,
        control=declared.control.model_copy(update={"named_at": when, "label": CONTROL_LABEL}),
        margin=margin,
        floors=floors,
        id=view.campaign_id,
        scope_id=view.scope_id,
        name=view.name,
        subject_id=view.subject_id,
        subject_kind=view.subject_kind,
        behavior=view.behavior,
    )
    bundle = off_loop(assemble_context_bundle, campaign, storage=storage, profile=comparison.host.profile)
    assert bundle.design.control_arm is not None
    control = bundle.design.control_arm.variant_key
    keys = {entry.levers["candidate"].display: entry.variant_key for entry in bundle.variant_index}
    report = build_code_only_report(bundle, assembled_at=ASSEMBLED_AT, campaign_name="support")
    return MeasuredAspect(
        campaign=campaign,
        bundle=bundle,
        report=report,
        control=control,
        candidate=keys["candidate"],
        third=keys.get(THIRD_LABEL),
    )


def varied(
    bundle: AnalysisContextBundle,
    *,
    aspect: str,
    control: Sequence[float],
    candidate: Sequence[float],
    lost: int = 0,
    also: Mapping[str, Sequence[float]] | None = None,
    paired: bool = True,
) -> AnalysisContextBundle:
    """``bundle`` read as aspect ``aspect`` with the overall pass rate's per-case shares set: ``control`` and
    ``candidate`` paired case by case, each a share of :data:`REPEATS` trials, and ``lost`` of the candidate's trials
    lost to the test setup. The scoreboard's counts follow the shares; nothing else in the bundle moves.

    :param bundle: A measured aspect's bundle.
    :ptype bundle: AnalysisContextBundle
    :param aspect: The aspect it reads as.
    :ptype aspect: str
    :param control: The control's per-case pass shares.
    :ptype control: Sequence[float]
    :param candidate: The candidate's, aligned with ``control``.
    :ptype candidate: Sequence[float]
    :param lost: Trials of the candidate's lost.
    :ptype lost: int
    :param also: Every other candidate's shares, by its variant key; one not named keeps the candidate's.
    :ptype also: Mapping[str, Sequence[float]] | None
    :param paired: Whether the shares are read paired; unpaired, a candidate's may be of another length.
    :ptype paired: bool
    :return: The bundle.
    :rtype: AnalysisContextBundle
    """
    others = dict(also or {})
    assert not paired or all(len(values) == len(control) for values in (candidate, *others.values()))
    control_key = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else None

    def shares_of(key: str) -> list[float]:
        return list(control) if key == control_key else list(others.get(key, candidate))

    def board_of(board: Any) -> Any:
        shares = shares_of(board.variant_key)
        trials = len(shares) * REPEATS
        lose = 0 if board.variant_key == control_key else lost
        usable = trials - lose
        passed = min(usable, round(sum(shares) * REPEATS))
        return board.model_copy(
            update={
                "ran": trials,
                "usable": usable,
                "passed": passed,
                "lost": {"judge_truncated": lose} if lose else {},
                "failed": {"check": usable - passed} if usable > passed else {},
                "cases": len(shares),
                "usable_cases": len(shares),
            }
        )

    contrasts = [
        contrast.model_copy(
            update={
                "case_values": (list(control), shares_of(contrast.contrast.variant_key)),
                "test": "paired" if paired else "unpaired",
            }
        )
        if contrast.reading == "trial" and contrast.name == TRIAL_PASS_NAME
        else contrast
        for contrast in bundle.floor_contrasts
    ]
    return bundle.model_copy(
        update={
            "aspect": aspect,
            "scoreboard": [board_of(board) for board in bundle.scoreboard],
            "floor_contrasts": contrasts,
        }
    )


def campaign_of(
    measured: MeasuredAspect,
    aspects: Sequence[tuple[str, float | None]],
    *,
    floors: Sequence[tuple[str, DecisionClause]] = (),
) -> EvalCampaign:
    """``measured``'s campaign holding ``aspects`` (each a label and its margin) and ``floors`` (each an aspect's
    label and a clause), every aspect a copy of its one aspect; written past the authoring gate.

    :param measured: A measured aspect.
    :ptype measured: MeasuredAspect
    :param aspects: Each aspect's label and margin, in order.
    :ptype aspects: Sequence[tuple[str, float | None]]
    :param floors: Each floor's aspect and clause.
    :ptype floors: Sequence[tuple[str, DecisionClause]]
    :return: The campaign.
    :rtype: EvalCampaign
    """
    (only,) = measured.campaign.aspects
    held = [
        only.model_copy(update={"label": label, "margin": margin, "run_ids": only.run_ids if index == 0 else []})
        for index, (label, margin) in enumerate(aspects)
    ]
    return measured.campaign.model_copy(
        update={"aspects": held, "floors": [Floor(aspect=aspect, clause=clause) for aspect, clause in floors]}
    )


def arms_of(measured: MeasuredAspect) -> list[CampaignArm]:
    """The campaign's arms: its control, then the candidate labelled :data:`CANDIDATE_LABEL`, then the second
    candidate labelled :data:`THIRD_LABEL` where one ran."""
    control = measured.campaign.control
    assert control is not None
    arms = [control, CampaignArm(label=CANDIDATE_LABEL, role="candidate", variant_key=measured.candidate)]
    if measured.third is not None:
        arms.append(CampaignArm(label=THIRD_LABEL, role="candidate", variant_key=measured.third))
    return arms


__all__ = [
    "ASKED_AT",
    "ASSEMBLED_AT",
    "CANDIDATE_LABEL",
    "CONTROL_LABEL",
    "THIRD_LABEL",
    "CASES",
    "QUESTION",
    "REPEATS",
    "MeasuredAspect",
    "arms_of",
    "campaign_of",
    "measured_aspect",
    "varied",
]
