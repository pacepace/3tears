"""A finished run as a short summary: how it ended, how its cells came out, and each measure's mean.

What :func:`~threetears.evals.quick.run_eval` returns, what the CLI prints after a launch and what the
``run_get`` action reads. It is read from the store, never from the job that ran, so a summary of a run finished in another process
says the same thing as one made the moment it ended.

Every count is over the run's stored results, and each result is classified once by
:func:`~threetears.evals.contracts.classify_result`: scored normally, failed by the candidate, or
excluded as a fault of the rig. A measure's mean is over the results that carry it, and its ``n``
says how many did, so a mean over two results of five is never mistaken for one over five.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from threetears.evals.contracts import ResultOutcome, classify_result
from threetears.evals.contracts.host import EvalHost
from threetears.evals.run import get_run, list_results


class MeasureSummary(BaseModel):
    """One measure over a run's results.

    Attributes:
        name: The measure, as the host declares it.
        n: How many results carry it.
        mean: Their mean — a boolean measure's is its rate; ``None`` when none does, or for a text
            measure, whose words are listed by the analysis bundle and never averaged.
        minimum: The lowest value; ``None`` when none does.
        maximum: The highest value; ``None`` when none does.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    n: int
    mean: float | None
    minimum: float | None
    maximum: float | None


class EvalSummary(BaseModel):
    """One run, summarised.

    Attributes:
        run_id: The run.
        scope_id: The scope it is stored in.
        template_id: The template it ran; ``None`` for an ad-hoc run of explicit cases.
        candidate_model: The arm's candidate model.
        status: How the run ended, as stored (``completed``, ``failed``, ``cancelled``, ...).
        k_runs: Repeats per case.
        n_cases: Cases in the run's frozen case set.
        n_results: Results stored.
        n_scored: Results scored normally.
        n_candidate_failed: Results the candidate failed: they count against it.
        n_excluded: Results excluded as a fault of the rig: they count for nothing.
        measures: Each measure the host declares, over the results that carry it.
        errors: Each failed or excluded result's error, prefixed by its case, then the run's own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    scope_id: str
    template_id: str | None
    candidate_model: str
    status: str
    k_runs: int
    n_cases: int
    n_results: int
    n_scored: int
    n_candidate_failed: int
    n_excluded: int
    measures: list[MeasureSummary]
    errors: list[str]

    def render(self) -> str:
        """The summary as a few lines of text for a terminal.

        Returns:
            The text, without a trailing newline.
        """
        lines = [
            f"run {self.run_id} {self.status}: {self.candidate_model} over {self.n_cases} case(s) x k={self.k_runs}",
            f"  {self.n_results} result(s): {self.n_scored} scored, {self.n_candidate_failed} failed by the "
            f"candidate, {self.n_excluded} excluded",
        ]
        for measure in self.measures:
            if measure.mean is None:
                lines.append(f"  {measure.name}: no result carries it")
            else:
                lines.append(
                    f"  {measure.name}: mean {measure.mean:.3g} (n={measure.n}, "
                    f"min {measure.minimum:.3g}, max {measure.maximum:.3g})"
                )
        lines.extend(f"  error: {error}" for error in self.errors)
        return "\n".join(lines)


def summarize_run(host: EvalHost, run_id: str, scope_id: str) -> EvalSummary:
    """Summarise one stored run and its results.

    Args:
        host: The host whose store holds the run, and whose measures are summarised.
        run_id: The run.
        scope_id: The scope it lives in.

    Returns:
        The summary.

    Raises:
        NotFoundError: No run with that id in the scope.
    """
    run = get_run(host.storage, run_id, scope_id)
    results = list_results(host.storage, run_id, scope_id)
    outcomes = [classify_result(result) for result in results]
    measures = []
    for name in host.profile.measures.names:
        carried = [result.host_measures[name] for result in results if name in result.host_measures]
        # A text observation is words, never a number; a boolean counts as 1 or 0, so its mean is its rate.
        values = [float(value) for value in carried if not isinstance(value, str)]
        measures.append(
            MeasureSummary(
                name=name,
                n=len(carried),
                mean=sum(values) / len(values) if values else None,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
            )
        )
    errors = [
        f"case {result.test_case_id}: {result.runner_error}"
        for result, outcome in zip(results, outcomes, strict=True)
        if outcome is not ResultOutcome.OK and result.runner_error
    ]
    errors.extend(run.error_details)
    return EvalSummary(
        run_id=run.id,
        scope_id=scope_id,
        template_id=run.template_id,
        candidate_model=run.candidate_model,
        status=run.status,
        k_runs=run.k_runs,
        n_cases=len(run.test_case_ids),
        n_results=len(results),
        n_scored=outcomes.count(ResultOutcome.OK),
        n_candidate_failed=outcomes.count(ResultOutcome.CANDIDATE_FAIL),
        n_excluded=outcomes.count(ResultOutcome.INFRA_EXCLUDE),
        measures=measures,
        errors=errors,
    )


__all__ = ["EvalSummary", "MeasureSummary", "summarize_run"]
