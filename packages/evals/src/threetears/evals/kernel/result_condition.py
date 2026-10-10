"""What condition an :class:`~threetears.evals.schema.models.EvalResult` is in, resolved once.

A result can be in a lot of states — judged, unjudged, judge-failed, dead before its
first turn, cancelled on its deadline — and for a long time nothing on the record named
any of them. Every surface that needed
the answer reconstructed it from a different subset of ``usage``, ``cost_roles``,
``latency``, ``judge_model``, the four score fields and the four error fields — so the
number of predicates grew with the number of consumers, and they disagreed, one of them
knowingly. This module is the one place that question is answered.

**Four axes, not one enum.** The states are not mutually exclusive and collapsing them
would produce a value that cannot describe a result in two of them at once — a cell
cancelled on its deadline whose judge had also failed is in both. Three axes are per-result
and live here:

* **termination** — how the cell's execution ended. Stored, not derived
  (:data:`~threetears.evals.schema.models.CellTermination` carries the reason).
* **judging** — what the judge actually produced, which is not the same question as
  whether a judge was *pinned*: the run-level pin is resolved at launch and stamped even
  on cells that died before judging, so its being set is evidence about the run, not
  about this result.
* **scoring** — how the result participates in aggregates, off the error taxonomy.

The fourth, **per-role capture** (were the rows observed whole, cut short by a deadline, or
never made?), deliberately stays with :func:`~threetears.evals.kernel.usage_capture.resolve_result_usage`, which
already owns it and returns it with its own provenance. Re-carrying it here would create
exactly the second copy this module exists to remove — a surface wanting it calls that
resolver, and both of them read the same stored ``termination``.

**A fourth thing this returns is prose.** ``disclosure`` is the canonical sentence for
whatever a reader must not miss. It is here rather than at the surfaces because three
surfaces phrasing a condition three ways is the same defect as three surfaces deriving
it three ways, and because a surface that renders a value cannot drift from the rule the
way a surface that switches on an enum can. Render it; do not branch on the arms to
rebuild it.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.covariates import TRUNCATED_ROUNDS_KEY, TURN_BUDGET_ENDED_KEY
from threetears.evals.schema.models import (
    RESERVED_DIM_IDS,
    SCALES,
    CellTermination,
    EvalResult,
    GoalStateOutcome,
    RubricScore,
)

__all__ = [
    "CandidateFailureCause",
    "JudgingState",
    "ResultCondition",
    "ResultOutcome",
    "candidate_failure_cause",
    "classify_result",
    "counted_goal_verdicts",
    "counted_rubric_scores",
    "counted_score",
    "delivered_a_turn",
    "resolve_result_condition",
]


class ResultOutcome(str, Enum):
    """How a result participates in scoring, by its error category.

    - ``OK`` — no error; scored normally.
    - ``CANDIDATE_FAIL`` — the candidate's configuration did not deliver a turn: a model it
      runs failed — its own turn model, or a model the configuration sets for a tool it
      drives (a background inner agent), including the cell's deadline striking
      while one of them was pending — or the output cap ended one of its turns, or
      the host's turn budget did (scored by what the end user gets,
      and a cut turn delivers silence or half an answer, an ended one nothing). A hard fail (0.0, not-pass) that lowers pass^k /
      composite and fails every goal-state check. A broken candidate must fail.
    - ``INFRA_EXCLUDE`` — a harness/infra failure (factory, simulator, delivery,
      judge, the cell's deadline striking while the simulator, the judge or the rig was
      pending, a candidate call refused for the calling account); EXCLUDED from the pass^k
      denominator and the composite mean. Excluded ≠ pass (no inflation), excluded ≠ fail (no
      flooring) — an infra failure must never score as candidate quality.
    """

    OK = "ok"
    CANDIDATE_FAIL = "candidate_fail"
    INFRA_EXCLUDE = "infra_exclude"


#: What the judge produced for this result.
#:
#: Four arms because a judge scores per dimension and can therefore half-succeed:
#:
#:   ``scored``         at least one judge output landed and nothing errored.
#:   ``partial``        outputs landed AND something errored — some dimensions scored,
#:                      some did not. Reporting one of these as the other loses whichever
#:                      half the reader needed.
#:   ``failed``         the judge ran and errored, producing nothing.
#:   ``not_attempted``  no judge output and no judge error: no judge service, an empty
#:                      transcript, or a cell that never reached judging.
#:
#: Derived from the judge's own outputs, which is a real evidence trail rather than a
#: correlate — a score exists only because a judge produced it. ``judge_model`` is
#: deliberately not consulted: it is the run-level pin, present on cells that never
#: reached a judge at all, and reading it as evidence of judging is one of the
#: disagreeing predicates this module replaced.
JudgingState = Literal["scored", "partial", "failed", "not_attempted"]

#: Why a result is a candidate failure. ``model_failed``: a model the configuration runs
#: failed, recorded in ``candidate_error``. ``turn_budget``: the host's turn budget ended one of
#: the candidate's turns before it finished. ``output_cap``: the provider cut one of the
#: candidate's turns off at the output cap. Three arms because the reader's remedy differs —
#: a different model or provider for the first, a faster configuration (fewer rounds, less
#: reasoning per round, a quicker model) for the second, a different cap or reasoning setting for
#: the third — and a disclosure naming the wrong one sends them to the wrong lever.
CandidateFailureCause = Literal["model_failed", "turn_budget", "output_cap"]


class ResultCondition(EvalBaseModel):
    """The condition one result is in, on every axis that is a property of the result.

    Returned as a value and never assigned back onto the result — the same rule
    :class:`~threetears.evals.kernel.usage_capture.ResolvedUsage` holds, and for the same reason:
    two of these three axes are derived, and writing a derivation into the document would
    make it indistinguishable from an observation the next time the document is saved.
    ``termination`` is the exception that proves it — it is stored precisely because it
    is *not* derivable, and it is read from the record here rather than recomputed.
    """

    #: How the cell ended, as the runner recorded it — never a value inferred from correlates.
    termination: CellTermination
    #: What the judge produced. Always answerable, from the judge's own outputs.
    judging: JudgingState
    #: How this result participates in aggregates.
    scoring: ResultOutcome
    #: The one sentence a surface must show, or ``None`` when the result is in no
    #: condition a reader needs warning about. Render it verbatim.
    disclosure: str | None = None


def candidate_failure_cause(result: EvalResult) -> CandidateFailureCause | None:
    """Name why ``result`` is a candidate failure, or ``None`` when it is not one.

    The output cap counts because the end user's experience is the measure: a turn the provider cut off delivered silence or half a reply, whatever the
    judge or a restraint check makes of the transcript afterwards — a hold template would
    otherwise score the cap's silence as the candidate holding back. Read from the truncation
    count the runner stores on every result. An absent count means nobody looked and asserts
    nothing; a model failure outranks it as the first-recorded cause.

    The host's turn budget counts for the same reason: a turn it ended delivered nothing, whatever its
    rounds had decided. Read from the count the runner stores when the kind ran turns under a budget
    (:data:`~threetears.evals.kernel.covariates.TURN_BUDGET_ENDED_KEY`). It outranks the output cap,
    because an ended turn left no record of whether its rounds were cut, and a model failure outranks
    it, as it does the cap.

    Args:
        result: The result.

    Returns:
        The cause, or ``None``.
    """
    if result.candidate_error:
        return "model_failed"
    ended = result.covariates.get(TURN_BUDGET_ENDED_KEY)
    if ended is not None and float(ended) > 0:
        return "turn_budget"
    truncated = result.covariates.get(TRUNCATED_ROUNDS_KEY)
    if truncated is not None and float(truncated) > 0:
        return "output_cap"
    return None


def classify_result(result: EvalResult) -> ResultOutcome:
    """Categorize a result for scoring by its error fields and its candidate's delivery.

    Candidate failures take precedence over infra: a broken candidate FAILS even
    if an infra hiccup also occurred, so it can't launder a failure into an
    exclusion. ``judge_error`` is infra (always). Reads only structured
    fields — never re-parses ``runner_error`` (fragile-parse rule).
    """
    if candidate_failure_cause(result) is not None:
        return ResultOutcome.CANDIDATE_FAIL
    if result.infra_error or result.judge_error:
        return ResultOutcome.INFRA_EXCLUDE
    return ResultOutcome.OK


#: How a cell ends when its time ran out under it — its deadline, or a cancel — rather than with the cell's
#: own work done. A model that failed then was cut off mid-call after the clock ran, not refused at once.
_CUT_OFF: frozenset[CellTermination] = frozenset({"cell_timeout", "cancelled"})


def delivered_a_turn(result: EvalResult) -> bool:
    """Whether ``result`` is a turn the candidate took — the one population a cost or latency reading is over.

    The predicate behind :data:`~threetears.evals.kernel.metrics.MeasurePopulation`'s ``delivered``, asked
    by every reader that means a turn's time or spend: the measure walk (cells, strata, bars, comparisons,
    run summaries, the telemetry rollup, the scope divergences), the frontier's ranked latency and cost, the
    trend series and a run's summary. Defined once, here, so no two of them can disagree about which results
    a "mean latency" averages.

    A result is OUT of it when either:

    - **the harness faulted it** (:func:`harness_faulted`) — its clock and its spend measured the rig; or
    - **the candidate's model failed before delivering a turn** — ``candidate_failure_cause`` is
      ``model_failed`` (a refusal, a provider or model error, recorded in ``candidate_error``), the result
      counts no delivered turn (``EvalResult.turns_delivered``), and the cell was not cut off by its deadline
      or a cancel. The call came straight back: its round trip is not a turn's latency, and an arm whose
      every call was refused read as the fastest and cheapest on the surface when it was averaged in.

    Every other candidate failure STAYS in, because it took a turn and the turn's time and spend are the
    arm's real cost of failing: a turn the host's budget ended (``turn_budget``), an answer the output cap
    cut (``output_cap``), a model call the cell's deadline struck while pending, and a model failure after
    the candidate had delivered turns — a conversation whose provider failed on its sixth turn, or whose
    background inner agent's model failed, after five delivered turns' time and spend. Leaving those out
    read an arm whose slowest, costliest conversations all failed late as faster and cheaper than the
    control. All of them still count against the arm wherever it is graded; this predicate decides only what
    a turn's time and spend are averaged over.

    **Where nothing counted the turns** (``turns_delivered`` None — a result stored before the count was kept,
    or a kind that neither reports a count nor stamps turn records) a model failure outside a cut-off reads as
    no turn: the cause alone is all that is left. It is right for a call that failed outright and wrong for
    any result whose candidate delivered something before failing — a multi-turn conversation's earlier
    turns, or a single call that returned and was billed and failed afterwards — whose time and spend it
    drops. That is why every in-tree kind counts: the quick callable kind (one call), the reporter kind (each
    generator call that returned), and a conversing kind through its stamped turn records.

    Args:
        result: The result.

    Returns:
        True when the result's time and spend describe a turn the candidate took.
    """
    if harness_faulted(result):
        return False
    if candidate_failure_cause(result) != "model_failed" or result.termination in _CUT_OFF:
        return True
    return bool(result.turns_delivered)


def harness_faulted(result: EvalResult) -> bool:
    """Whether the harness, not the candidate, spoiled this result — so none of its verdicts count.

    The one answer every reader of a result's goal-state verdicts consults: the bars, the
    analysis bundle's per-check pass rate and the score-record projection behind ``pivot``. A
    slot that never started leaves a check such as ``call_count("notes.write") >= 1``
    reading False, which is a fact about the harness; three readers each deciding for
    themselves is how the pivot and the memo came to report two rates for one check on one
    cell.
    """
    return classify_result(result) is ResultOutcome.INFRA_EXCLUDE


def counted_goal_verdicts(result: EvalResult) -> list[tuple[GoalStateOutcome, bool]] | None:
    """Each goal-state check on ``result`` paired with the verdict every rate counts for it.

    The one answer to "what does this check count as", which every per-check rate reads — the
    score-record projection behind pivot and export, and the analysis bundle — so they cannot
    come to two rates for one check on one cell.

    - A harness-faulted result returns ``None``: its checks read the harness, and it is in no
      rate at all (:func:`harness_faulted`).
    - A candidate failure counts EVERY check as failed. It is the
      rule pass^k already applies — a broken candidate must fail — carried to the per-check
      rates, where it was missing: a restraint check such as "never sent a reminder"
      evaluates True on a conversation the candidate never took part in, and was counted as
      the candidate holding back. A deadline charged to the candidate cancels the cell before its
      checks are graded, so the runner stores the template's checks unevaluated for this rule to
      count (``runner._unevaluated_goal_checks``).
    - Otherwise each check counts as it evaluated.

    Args:
        result: The result.

    Returns:
        ``(outcome, counted_passed)`` per check in stored order, or ``None`` when the result is
        excluded from every rate.
    """
    outcome = classify_result(result)
    if outcome is ResultOutcome.INFRA_EXCLUDE:
        return None
    failed = outcome is ResultOutcome.CANDIDATE_FAIL
    return [(check, False if failed else check.passed) for check in result.goal_state_outcomes]


def counted_score(result: EvalResult, score: RubricScore) -> int | None:
    """What one judged score on ``result`` counts as in every measure — a rubric dim or a reserved axis.

    The rubric sibling of :func:`counted_goal_verdicts`, read wherever a judged score is meaned —
    the dimension summary, the score-record projection behind pivot, export and the bundle's judged
    measures, and the history series of the two reserved axes — so they cannot come to two means
    for one dim.

    - A harness-faulted result counts ``None``: the judge read a transcript the harness broke, so
      its scores measure the rig and enter no mean.
    - A candidate failure counts the score's scale floor, scored by what
      the end user gets. A turn the output cap ended delivered silence or half a reply; the judge
      may still have scored that silence as restraint, and counting its reading would rank a broken
      configuration on a pass. The raw score stays on the result.
    - Otherwise the score counts as judged.

    Args:
        result: The result the score sits on.
        score: One of its rubric scores, or its transcript or outcome axis.

    Returns:
        The counted score on the score's own scale, or ``None`` when it enters no measure.
    """
    outcome = classify_result(result)
    if outcome is ResultOutcome.INFRA_EXCLUDE:
        return None
    if outcome is ResultOutcome.CANDIDATE_FAIL:
        return SCALES[score.scale].scores[0]
    return score.score


def counted_rubric_scores(result: EvalResult) -> list[tuple[RubricScore, int]] | None:
    """Each judged rubric dim on ``result`` paired with what it counts as (:func:`counted_score`).

    Args:
        result: The result.

    Returns:
        ``(score, counted)`` per judged dim in stored order, or ``None`` when the result is in no
        per-dimension measure (a harness fault).
    """
    if classify_result(result) is ResultOutcome.INFRA_EXCLUDE:
        return None
    return [(score, counted) for score in result.rubric_scores if (counted := counted_score(result, score)) is not None]


def _classify_judging(result: EvalResult) -> JudgingState:
    """Decide what the judge produced, from the judge's own outputs.

    Any one of the four outputs is evidence that judging happened: a judge can score
    without narrating, and a template can score a rubric without either reserved axis, so
    requiring a particular one would report a real judging pass as none.
    """
    # A "can't tell" answer is judge output too: the judge worked and said the evidence does not
    # decide the dim, so a result whose every dim was answered that way was judged, not skipped.
    scored = bool(
        result.judge_reasoning
        or result.transcript_score is not None
        or result.outcome_score is not None
        or result.rubric_scores
        or result.judge_cannot_tell
    )
    if result.judge_error:
        return "partial" if scored else "failed"
    return "scored" if scored else "not_attempted"


#: Disclosure text per termination arm. A mapping rather than a chain of ``if``\\ s so
#: that adding an arm to :data:`~threetears.evals.schema.models.CellTermination` without deciding
#: what it discloses is a visible hole here rather than a silent fall-through to "nothing
#: to report" — the failure mode that lets a new condition ship invisible.
_TERMINATION_DISCLOSURE: dict[CellTermination, str | None] = {
    "completed": None,
    "factory_failed": (
        "The candidate factory failed before any candidate turn ran: nothing was spent, and the zero total is a real sum."
    ),
    "cell_timeout": (
        "This cell was cancelled on its deadline. Its total and per-role usage count what it had spent up to "
        "then, but not the call in flight when the deadline struck, so it cost at least this much."
    ),
    "seed_failed": (
        "The eval apparatus failed seeding this cell's world, before any candidate turn ran: the fault is in the "
        "harness or the seed — not in the candidate or its factory — and nothing was spent, so the zero total is a real sum."
    ),
    # Phrased over the WORLD rather than over a fault, because nothing here is broken. The
    # apparatus worked and the template's presumption was not met, which makes the observation
    # invalid rather than the harness faulty — and a reader sent looking for a bug would not find
    # one. ``precondition_outcomes`` on the result names which presumption and what it presumed.
    "precondition_failed": (
        "A precondition this template presumes did not hold when the world was seeded, so the subject was never "
        "placed in the state the probe was written for. The observation is recorded and EXCLUDED rather than "
        "scored; no candidate turn ran and nothing was spent."
    ),
    "apparatus_failed": (
        "The eval apparatus failed while this cell ran — a replay miss, a corrupt recording, a harness fault — so it "
        "is EXCLUDED rather than scored: the fault is the rig's, not the candidate's. Its total and per-role usage "
        "count what it had reported up to then, but not work in flight when the fault struck, so it cost at least "
        "this much."
    ),
    "cancelled": (
        "This cell's run was cancelled while it ran, so it is EXCLUDED rather than scored. Its total and per-role "
        "usage count what it had spent up to then, but not the call in flight when the cancel struck, so it cost "
        "at least this much."
    ),
}

#: Disclosure text per scoring arm. The judge is not skipped when a cell errors — by the
#: time the runner breaks, the transcript already holds turns, so judging runs and the
#: result persists real scores. Without a line here a surface renders those scores beside
#: an error badge and nothing says the result was dropped from the aggregates, which is
#: how an operator ends up citing a score from a cell whose apparatus was broken.
#: Indexed rather than ``.get`` for the same reason as the termination map above.
#: The candidate-failure line when a model the configuration runs failed — shared by the scoring
#: map and the per-cause map below, which key the same sentence differently.
_MODEL_FAILED_DISCLOSURE = (
    "A model this configuration runs — the candidate's own, or one it sets for its background work — "
    "failed, so this result is scored as a hard fail rather than on its content, and any goal-state check "
    "on it counts as failed."
)

_SCORING_DISCLOSURE: dict[ResultOutcome, str | None] = {
    ResultOutcome.OK: None,
    ResultOutcome.CANDIDATE_FAIL: _MODEL_FAILED_DISCLOSURE,
    # Deliberately phrased over what every arm shares — a harness failure, exclusion — and
    # not over "ended" or "truncated". Those describe only the mid-cell arms. A factory or
    # seeding failure produces NO transcript rather than a truncated one, and a judge-only
    # failure leaves the transcript complete; asserting truncation there would send a reader
    # looking for a cut-off conversation that does not exist.
    ResultOutcome.INFRA_EXCLUDE: (
        "A harness failure — not the candidate — produced this result, so it is EXCLUDED from the "
        "aggregates. Any scores shown are the judge's reading of whatever transcript survived, which "
        "may be partial or absent entirely, and are not a measurement of this configuration."
    ),
}

#: The candidate-failure line when the cause is the cap rather than a failed model call —
#: :data:`_SCORING_DISCLOSURE`'s line says a model failed, which would send a reader to swap a
#: model whose every call succeeded.
_OUTPUT_CAP_DISCLOSURE = (
    "The output cap cut off one of the candidate's turns before it finished, so what that turn delivered "
    "was silence or half a reply: this result is scored as a hard fail rather than on its content, and any "
    "goal-state check on it counts as failed."
)

#: The candidate-failure line when the host's turn budget ended a turn — the model may have been
#: answering every call, just not within the budget, so the model-failed line would misdirect too.
_TURN_BUDGET_DISCLOSURE = (
    "The host's turn budget ended one of the candidate's turns before it finished, so that turn delivered "
    "nothing: this result is scored as a hard fail rather than on its content, and any "
    "goal-state check on it counts as failed. The totals count the calls the ended turn completed, but not "
    "one in flight when it was cut, so it cost at least this much."
)

#: The candidate-failure disclosure per cause. Indexed rather than ``.get`` for the reason the
#: termination map is: a cause added to :data:`CandidateFailureCause` without a sentence here fails
#: loudly instead of borrowing another cause's remedy.
_CANDIDATE_FAILURE_DISCLOSURE: dict[CandidateFailureCause, str] = {
    "model_failed": _MODEL_FAILED_DISCLOSURE,
    "turn_budget": _TURN_BUDGET_DISCLOSURE,
    "output_cap": _OUTPUT_CAP_DISCLOSURE,
}

_JUDGING_DISCLOSURE: dict[JudgingState, str | None] = {
    "scored": None,
    # Disclosed rather than left silent: the absence of scores is otherwise indistinguishable
    # from scores a surface simply did not render, and the run-level judge pin is stamped on
    # cells that never reached a judge — so a surface with no disclosure and a judge model in
    # hand will say "judged by X" about a result nothing judged.
    "not_attempted": "No judge produced output for this result, so it carries no scores.",
    "partial": "The judge errored partway: some dimensions were scored and some were not.",
    "failed": "The judge ran and errored, so this result carries no scores.",
}


#: Why a result is left out of a measure. ``infra``: the harness faulted it (``classify_result``
#: says ``INFRA_EXCLUDE``), which leaves it out of every measure. ``judge_cannot_tell``: the judge
#: answered it could not score a dim, which leaves it out of that dim's measure and out of the
#: measures needing every dim (pass^k, the composite), and nowhere else. Counted per cause wherever
#: exclusions are counted, so a shrunken ``n`` always says why.
ExclusionCause = Literal["infra", "judge_cannot_tell"]

#: The ``outcome`` a projected score row carries when the judge could not tell on what the row
#: measures — a dim it named, or the composite that needs every dim. Distinct from every
#: :class:`ResultOutcome` value, so a reader of the row can count it apart from a fault.
JUDGE_CANNOT_TELL_OUTCOME = "judge_cannot_tell"


def trial_exclusion(result: EvalResult) -> ExclusionCause | None:
    """Why ``result`` is left out of the measures that need every rubric dim, or ``None``.

    The one classification pass^k, the composite and their counts read. A candidate failure is
    never excluded: it fails, whatever the judge could or could not tell. pass^k narrows the
    ``judge_cannot_tell`` arm further — a trial a failed check or a sub-bar dim has already decided
    counts as failed there (``scoring._already_failed``) — which needs the bar and so cannot live
    here; the composite keeps the exclusion, since a mean over fewer dims is not the composite.

    Args:
        result: The result.

    Returns:
        The cause, or ``None`` when the result counts.
    """
    outcome = classify_result(result)
    if outcome is ResultOutcome.INFRA_EXCLUDE:
        return "infra"
    if outcome is ResultOutcome.OK and not judged_on_every_dim(result):
        return "judge_cannot_tell"
    return None


def judged_on_every_dim(result: EvalResult) -> bool:
    """Whether the judge scored or failed on every rubric dim, answering "can't tell" on none.

    The one predicate the whole-trial measures read. pass^k and the composite each need every
    rubric dim: a trial whose judge could not tell on one is not measured there, and counting it
    on the dims that remain would pass a trial on fewer criteria than its siblings. A dim's own
    mean needs no such rule — the dim carries no score, so the trial is absent from it — and the
    reserved transcript/outcome axes are in neither whole-trial measure.

    Args:
        result: The result.

    **A boundary dim is not one of them.** pass^k and the composite read capability dims alone
    (:func:`~threetears.evals.kernel.scoring.capability_scores`), so a can't-tell on a guardrail
    (``judge_cannot_tell_boundary``) leaves the trial in both: dropping it would let a guardrail reading
    move the capability pillar.

    Returns:
        ``False`` when the judge answered it could not tell on at least one capability rubric dim.
    """
    boundary = set(result.judge_cannot_tell_boundary)
    return not any(dim not in RESERVED_DIM_IDS and dim not in boundary for dim in result.judge_cannot_tell)


def _cannot_tell_disclosure(result: EvalResult) -> str | None:
    """Name the dims the judge could not score, and what that does to the aggregates.

    Separate from :data:`_JUDGING_DISCLOSURE` because it is not a judging state: it rides on
    ``scored`` and ``partial`` alike, and a result can carry it whatever its other arms say.
    """
    if not result.judge_cannot_tell:
        return None
    dims = ", ".join(sorted(result.judge_cannot_tell))
    said = f"The judge answered it could not tell from the evidence on {dims}, so it is unscored there."
    if not judged_on_every_dim(result):
        if any(not check.passed for check in result.goal_state_outcomes):
            return (
                f"{said} This result is left out of the composite, which needs every rubric dimension; "
                "in pass^k its failed goal-state check fails it."
            )
        return (
            f"{said} This result is left out of pass^k and the composite, which need every rubric dimension, "
            "unless in pass^k a scored dimension below the bar already fails it."
        )
    return said


def resolve_result_condition(result: EvalResult) -> ResultCondition:
    """Return the condition ``result`` is in, on every per-result axis.

    The single place a surface asks this. A surface that re-derives any of these from the
    underlying fields will disagree with the others eventually — that is the history this
    function replaced, not a hypothetical.

    Args:
        result: The result to resolve. Pure function of the stored record: no joins, no
            run lookup, no I/O — so an aggregate can call it per row.

    Returns:
        The resolved condition, including the canonical ``disclosure`` line.
    """
    judging = _classify_judging(result)
    scoring = classify_result(result)
    parts = [
        # Indexed, not ``.get``: a termination arm with no disclosure decision must fail
        # loudly rather than resolve to "nothing to report", which is the shape in which a
        # new condition ships invisible. ``test_every_termination_arm_has_a_disclosure``
        # is what makes that a build-time failure instead of an operator's read.
        _TERMINATION_DISCLOSURE[result.termination],
        _CANDIDATE_FAILURE_DISCLOSURE[cause]
        if (cause := candidate_failure_cause(result)) is not None
        else _SCORING_DISCLOSURE[scoring],
        _JUDGING_DISCLOSURE[judging],
        _cannot_tell_disclosure(result),
    ]
    disclosure = " ".join(part for part in parts if part) or None
    return ResultCondition(
        termination=result.termination,
        judging=judging,
        scoring=scoring,
        disclosure=disclosure,
    )
