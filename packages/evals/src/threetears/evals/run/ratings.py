"""Calibration ratings — the write a rater makes when they score a result the judge already scored.

A rating says whether a person or an agent wrote it (``rater_kind``): only a person's calibrates the judge
against people, so an agent's — a model rating through a tool — is kept out of that agreement and listed.

A judge's agreement with people is only as good as the ratings it is read against, so the write
takes from the caller only what a person decides — who they are, which dimension, the score and
why — and reads everything else off the stored result: its run, the dimension's scale, and whether
the judge scored that dimension at all. A rating of a dimension the judge did not score has
nothing to be compared with, and a scale the caller asserted could disagree with the score it is
compared to; both are refused here rather than surfacing later as an unpaired rating.

Read back through ``AnalysisContextBundle.judge_agreement`` (and a reporter run's
``ReporterCalibration.rating_agreement``), which pair each rating with the result's judge score at
read time. A rater who rates the same dimension of the same result again, as the same kind of rater,
replaces their earlier rating: the document's id is derived from the result, the dimension, the rater
and the rater's kind — so an agent rating under a person's identity stands beside that person's rating
rather than overwriting it.

**A rating is also bound to what the rater read** (#628). The judge stamps each score with its label key —
the fingerprint of the evidence it read and of the dimension's definition — and the write copies that key onto
the rating, from the stored score rather than from the caller. So the label pairs with every judge score of a
byte-identical output on the same criterion, on any result, and a frozen judge case finds it by the key. A
score made outside a judge phase has none, and its rating is read by its result alone.

**A rating carries what was rated** (durable v15.1). Ratings are durable: a core baseline reset deletes the run,
result and trace a rating was given on, and the rating stays. So the write copies the content the rater read onto
the rating (:class:`~threetears.evals.schema.RatedOutput`): the criterion, the scenario and the evidence, rebuilt
from what the run recorded under the one check a judge freeze rebuilds a case under
(:func:`~threetears.evals.run.rejudge.reproducible_judge_inputs`), and proven against the score's label key. When
that check refuses (the template was edited after the run, the run is still being written, the trace stored no
evidence, the score carries no label key), the person's judgment is still recorded: the rating is written with
``rated_withheld`` set to the refusal, and a warning is logged naming it. It is a label that cannot outlive its
evidence, and every reader that rebuilds cases from labels names it.

**Deleting a result does not delete its ratings.** They stay, and every agreement read lists them as
``result_unresolved`` — a person's judgement is not regenerable, and a rating that silently
disappeared would shrink a dimension's n with nothing saying why.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.storage import UnreadableRecordError
from threetears.evals.run.rejudge import JudgeInputStore, reproducible_judge_inputs
from threetears.evals.schema.models import (
    RESERVED_DIM_IDS,
    CalibrationRating,
    EvalResult,
    EvalRun,
    RatedOutput,
    RaterKind,
    RubricScore,
    label_key_of,
)
from threetears.evals.schema.logs import get_logger
from threetears.evals.kernel.remedies import FIND_RESULT, SCHEMA_REMEDY

log = get_logger(__name__)


class RatingStore(JudgeInputStore, Protocol):
    """The calls a rating makes: read the rated result, its run and what its judge read, and write the rating.

    The reads :func:`~threetears.evals.run.rejudge.reproducible_judge_inputs` makes, which rebuild the rated content,
    plus the result, its run and the write. Structural, so :class:`~threetears.evals.kernel.storage.EvalStorage`
    satisfies it by having the methods. Positional parameters are positional-only.
    """

    def load_eval_result(self, result_id: str, scope_id: str, /) -> EvalResult | None:
        """Load one result within a scope, or ``None``."""
        ...

    def load_eval_run(self, run_id: str, scope_id: str, /) -> EvalRun | None:
        """Load one run within a scope, or ``None``."""
        ...

    def save_calibration_rating(self, rating: CalibrationRating, /) -> None:
        """Write a rating; raises ``StorageError`` on failure rather than returning a flag."""
        ...


def rate_result(
    storage: RatingStore,
    *,
    result_id: str,
    scope_id: str,
    rubric_dim: str,
    rater: str,
    rater_kind: RaterKind,
    score: int,
    reason: str,
) -> CalibrationRating:
    """Record one rater's score for one judged dimension of one result — a person's, or an agent's.

    :param storage: Where the result is read and the rating written.
    :ptype storage: RatingStore
    :param result_id: The result rated.
    :ptype result_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param rubric_dim: The judged dimension, spelled as the result's score spells it (a template
        dimension's ``<context>.<dim>``, or a reserved dual-score axis id).
    :ptype rubric_dim: str
    :param rater: Who rated, as the host names them: a person's account, or an agent's identity.
    :ptype rater: str
    :param rater_kind: ``person`` or ``agent``. Required, and the caller's to state from what it knows of who is
        calling — an agent writing through a tool is an ``agent`` whatever account it acts for — because only
        a person's rating is read as judge-versus-human agreement.
    :ptype rater_kind: RaterKind
    :param score: The score, on the dimension's scale: 1-5, or 1 (pass) / 0 (fail).
    :ptype score: int
    :param reason: The rater's own words for the score.
    :ptype reason: str
    :return: The rating as written, carrying the judge score's label key (the fingerprints of the output read
        and of the criterion) when the score has one, and what was rated (``rated``) or why it could not be
        rebuilt (``rated_withheld``). A second rating by the same rater, of the same kind, of the same dimension
        of the same result replaces the first.
    :rtype: CalibrationRating
    :raises NotFoundError: No such result in the scope.
    :raises ValidationFailedError: The judge scored no such dimension on the result; or the score is
        not on its scale, the dimension name is not namespaced, the rater or reason is blank, or ``rater_kind``
        is neither ``person`` nor ``agent``.
    :raises StorageError: The write failed.
    """
    result = storage.load_eval_result(result_id, scope_id)
    if result is None:
        raise NotFoundError("result", result_id, remedy=FIND_RESULT)
    judged = result.judge_score(rubric_dim)
    if judged is None:
        scored = sorted(
            score.dim
            for score in (*result.rubric_scores, result.transcript_score, result.outcome_score)
            if score is not None
        )
        raise ValidationFailedError(
            f"result {result_id!r} carries no judge score on {rubric_dim!r}, so a rating of it has nothing to be "
            "calibrated against",
            details={"result_id": result_id, "rubric_dim": rubric_dim, "judged_dimensions": scored},
            remedy=f"rate a dimension the result was judged on: {scored or 'it was judged on none'}",
        )
    rated = _rated_content(storage, result, judged)
    try:
        rating = CalibrationRating(
            scope_id=result.scope_id,
            run_id=result.eval_run_id,
            result_id=result.id,
            rubric_dim=rubric_dim,
            rater=rater,
            rater_kind=rater_kind,
            scale=judged.scale,
            score=score,
            reason=reason,
            # What the rater read, as the judge stamped it: the label then holds for every judgement of a
            # byte-identical output on the same criterion, not only this result's (#628).
            output_fingerprint=judged.output_fingerprint,
            criterion_fingerprint=judged.criterion_fingerprint,
            rated=rated if isinstance(rated, RatedOutput) else None,
            rated_withheld=rated if isinstance(rated, str) else None,
        )
    except ValidationError as e:
        raise ValidationFailedError(
            f"rating of {rubric_dim!r} on result {result_id!r} refused: {e}", remedy=SCHEMA_REMEDY
        ) from e
    storage.save_calibration_rating(rating)
    if rating.rated_withheld is not None:
        log.warning(
            "eval.rate_result rating=%s result=%s dim=%s is stored without what was rated, so it cannot outlive its "
            "evidence: %s",
            rating.id,
            result.id,
            rubric_dim,
            rating.rated_withheld,
        )
    log.info(
        "eval.rate_result result=%s run=%s scope=%s dim=%s rater=%s rater_kind=%s score=%s",
        result.id,
        result.eval_run_id,
        result.scope_id,
        rubric_dim,
        rater,
        rater_kind,
        score,
    )
    return rating


def _rated_content(storage: RatingStore, result: EvalResult, judged: RubricScore) -> RatedOutput | str:
    """What the rater read on ``judged``, rebuilt from what the result's run recorded, or why it cannot be.

    :param storage: Where the run, template, test case and trace are read.
    :ptype storage: RatingStore
    :param result: The rated result.
    :ptype result: EvalResult
    :param judged: Its judge score on the rated dimension.
    :ptype judged: RubricScore
    :return: The content, its fingerprints the score's label key; or the reason it is withheld, in words a reader of the
        rating is shown.
    :rtype: RatedOutput | str
    :raises StorageError: A read failed: that is not a reason to withhold, and is raised as every read raises it.
    """
    if judged.output_fingerprint is None or judged.criterion_fingerprint is None:
        return (
            f"the judge score on {judged.dim!r} carries no label key (a score made outside a judge phase), so nothing "
            "proves which output and criterion the rater read"
        )
    try:
        run = storage.load_eval_run(result.eval_run_id, result.scope_id)
        if run is None:
            return f"the result's run {result.eval_run_id!r} does not load, so what its judge read cannot be rebuilt"
        inputs = reproducible_judge_inputs(
            storage, result, run, result.scope_id, request_settings="as_recorded", config_dims=()
        )
    except (NotFoundError, ValidationFailedError, UnreadableRecordError) as refused:
        return f"what the judge read cannot be rebuilt from the run: {refused}"
    criterion = next((dim for dim in inputs.template.rubric if dim.name == judged.dim), None)
    if criterion is None and judged.dim not in RESERVED_DIM_IDS:
        return f"the run's template no longer words {judged.dim!r}, so the criterion the rater read cannot be rebuilt"
    rated = RatedOutput(
        dim=judged.dim,
        scale=judged.scale,
        criterion=criterion,
        intent=inputs.template.intent,
        variation=dict(inputs.test_case.variation_params),
        goal_outcomes=list(result.goal_state_outcomes),
        judged_artifact=inputs.judged_artifact,
        evidence=inputs.judge_evidence,
    )
    rebuilt = label_key_of(rated.evidence, criterion if criterion is not None else judged.dim)
    if rebuilt != (judged.output_fingerprint, judged.criterion_fingerprint):
        return (
            "the evidence and criterion rebuilt from the run are not the ones the judge's score was stamped with (its "
            "label key differs), so they are not what the rater read"
        )
    return rated


__all__ = ["RatingStore", "rate_result"]
