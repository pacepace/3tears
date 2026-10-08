"""A finished run as a short summary: how it ended, how its cells came out, and each measure's mean.

What :func:`~threetears.evals.quick.run_eval` returns, what the CLI prints after a launch and what the
``run_get`` action reads. It is read from the store, never from the job that ran, so a summary of a run finished in another process
says the same thing as one made the moment it ended.

Every count is over the run's stored results, and each result is classified once by
:func:`~threetears.evals.contracts.classify_result`: scored normally, failed by the candidate, or
excluded as a fault of the rig. A measure's mean is over the results that carry it, and its ``n``
says how many did, so a mean over two results of five is never mistaken for one over five.

**A classifier's run is read as one.** A kind that classifies lands ``match`` and ``confusion_cell``,
core measures no host declares, and the summary reads them wherever a result carries them: ``match``
as a measure whose mean is the share of answers that matched, and the ``confusion_cell`` counts as the
confusion matrix and each label's precision, recall and F1, counted by
:func:`~threetears.evals.analysis.confusion.label_statistics` as the analysis bundle counts them.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict

from threetears.evals.analysis.confusion import ConfusionCount, LabelStatistics, confusion_matrix, label_statistics
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.contracts import CONFUSION_CELL_MEASURE, MATCH_MEASURE, ResultOutcome, classify_result
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
        measures: Each measure the host declares, over the results that carry it, then ``match`` and
            ``confusion_cell`` when a result carries them and the host does not declare them.
        confusion: The confusion matrix of the results' ``confusion_cell`` values, by expected then
            predicted label; empty for a run that classified nothing.
        labels: Each label's precision, recall and F1 from that matrix, by label; empty with it.
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
    confusion: list[ConfusionCount]
    labels: list[LabelStatistics]
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
            if measure.n == 0:
                lines.append(f"  {measure.name}: no result carries it")
            elif measure.name == CONFUSION_CELL_MEASURE and self.confusion:
                lines.append(f"  {measure.name}: n={measure.n}, counted in the confusion matrix below")
            elif measure.mean is None:
                lines.append(f"  {measure.name}: n={measure.n}, not a number")
            else:
                lines.append(
                    f"  {measure.name}: mean {measure.mean:.3g} (n={measure.n}, "
                    f"min {measure.minimum:.3g}, max {measure.maximum:.3g})"
                )
        if self.confusion:
            lines.append("  confusion (expected → predicted):")
            lines.extend(
                f"    {_shown(cell.expected)} → {_shown(cell.predicted)}: {cell.count}" for cell in self.confusion
            )
            lines.append("  per label:")
            lines.extend(f"    {_label_line(statistics)}" for statistics in self.labels)
        lines.extend(f"  error: {error}" for error in self.errors)
        return "\n".join(lines)


def _shown(label: str) -> str:
    """A label as printed: as written, or quoted when whitespace at its ends would otherwise be invisible."""
    return repr(label) if label != label.strip() else label


def _proportion(name: str, rate: float | None, hits: int, n: int, interval: tuple[float, float] | None) -> str:
    if rate is None:
        return f"{name} none (n=0)"
    shown = f"{name} {rate:.3g} ({hits}/{n}"
    if interval is not None:
        shown += f", {INTERVAL_LEVEL:.0%} CI {interval[0]:.2g}-{interval[1]:.2g}"
    return shown + ")"


def _label_line(statistics: LabelStatistics) -> str:
    """One label's line: its precision and recall with their counts and intervals, and its F1."""
    precision = _proportion(
        "precision", statistics.precision, statistics.correct, statistics.predicted, statistics.precision_interval
    )
    recall = _proportion(
        "recall", statistics.recall, statistics.correct, statistics.expected, statistics.recall_interval
    )
    f1 = "f1 none" if statistics.f1 is None else f"f1 {statistics.f1:.3g}"
    return f"{_shown(statistics.label)}: {precision}, {recall}, {f1}"


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
    declared = host.profile.measures.names
    classified = [
        name
        for name in (MATCH_MEASURE, CONFUSION_CELL_MEASURE)
        if name not in declared and any(name in result.host_measures for result in results)
    ]
    measures = []
    for name in (*declared, *classified):
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
    cells = Counter(
        cell for result in results if isinstance(cell := result.host_measures.get(CONFUSION_CELL_MEASURE), str)
    )
    confusion = confusion_matrix(cells)
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
        confusion=confusion,
        labels=label_statistics(confusion),
        errors=errors,
    )


__all__ = ["EvalSummary", "MeasureSummary", "summarize_run"]
