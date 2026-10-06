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

**Deleting a result does not delete its ratings.** They stay, and every agreement read lists them as
``result_unresolved`` — a person's judgement is not regenerable, and a rating that silently
disappeared would shrink a dimension's n with nothing saying why.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from threetears.evals.contracts.errors import NotFoundError, ValidationFailedError
from threetears.evals.contracts.models import CalibrationRating, EvalResult, RaterKind
from threetears.observe import get_logger

log = get_logger(__name__)


class RatingStore(Protocol):
    """The two calls a rating makes: read the rated result, write the rating.

    Structural, so :class:`~threetears.evals.contracts.storage.EvalStorage` satisfies it by having
    the methods. Positional parameters are positional-only.
    """

    def load_eval_result(self, result_id: str, scope_id: str, /) -> EvalResult | None:
        """Load one result within a scope, or ``None``."""
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

    Args:
        storage: Where the result is read and the rating written.
        result_id: The result rated.
        scope_id: The scope it lives in.
        rubric_dim: The judged dimension, spelled as the result's score spells it (a template
            dimension's ``<context>.<dim>``, or a reserved dual-score axis id).
        rater: Who rated, as the host names them: a person's account, or an agent's identity.
        rater_kind: ``person`` or ``agent``. Required, and the caller's to state from what it knows of who is
            calling — an agent writing through a tool is an ``agent`` whatever account it acts for — because only
            a person's rating is read as judge-versus-human agreement.
        score: The score, on the dimension's scale: 1-5, or 1 (pass) / 0 (fail).
        reason: The rater's own words for the score.

    Returns:
        The rating as written. A second rating by the same rater, of the same kind, of the same
        dimension of the same result replaces the first.

    Raises:
        NotFoundError: No such result in the scope.
        ValidationFailedError: The judge scored no such dimension on the result; or the score is
            not on its scale, the dimension name is not namespaced, the rater or reason is blank, or ``rater_kind``
            is neither ``person`` nor ``agent``.
        StorageError: The write failed.
    """
    result = storage.load_eval_result(result_id, scope_id)
    if result is None:
        raise NotFoundError("result", result_id)
    judged = result.judge_score(rubric_dim)
    if judged is None:
        scored = sorted(
            score.dim
            for score in (*result.rubric_scores, result.transcript_score, result.outcome_score)
            if score is not None
        )
        raise ValidationFailedError(
            f"result {result_id!r} carries no judge score on {rubric_dim!r}, so a rating of it has nothing to be "
            f"calibrated against; the judged dimensions are {scored or 'none'}",
            details={"result_id": result_id, "rubric_dim": rubric_dim, "judged_dimensions": scored},
        )
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
        )
    except ValidationError as e:
        raise ValidationFailedError(f"rating of {rubric_dim!r} on result {result_id!r} refused: {e}") from e
    storage.save_calibration_rating(rating)
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


__all__ = ["RatingStore", "rate_result"]
