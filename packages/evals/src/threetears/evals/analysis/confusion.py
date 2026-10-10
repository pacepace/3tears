"""A classifier's confusion matrix and the per-label statistics it yields, counted once for every surface.

A classifier lands one ``confusion_cell`` per observation (``expected → predicted``,
:func:`~threetears.evals.kernel.confusion_cell`). Counted, those cells are its confusion matrix
(:func:`confusion_matrix`), and from the matrix each label's precision, recall and F1 follow
(:func:`label_statistics`). The analysis bundle and the run summary both read them from here, so a
label's precision is one computation wherever it is printed.

**A label's interval is over cases.** The statistics are counted from the observations, each beside
its test case, because a case classified k times is one draw repeated, not k draws: precision and
recall take :func:`~threetears.evals.analysis.stats.proportion_interval`, which is the Wilson
interval wherever every case was classified once.

**Labels are kept exactly as given.** These models do not strip their strings, unlike every model
on :class:`~threetears.evals.schema.base.EvalBaseModel`: a label is free text, and ``"positive "``
stripped to ``"positive"`` would count a wrong answer as a right one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from threetears.evals.analysis.stats import proportion_interval
from threetears.evals.kernel.metrics import confusion_of


class ConfusionCount(BaseModel):
    """One cell of a confusion matrix: how often a case expecting one label was given another (or the same).

    Attributes:
        expected: The label the case expected.
        predicted: The label the classifier gave.
        count: Observations in this cell.
    """

    # Not EvalBaseModel: a label's surrounding whitespace is part of it (see the module docstring).
    model_config = ConfigDict(frozen=True, extra="forbid")

    expected: str
    predicted: str
    count: int


class LabelStatistics(BaseModel):
    """One label's counts in a confusion matrix, and the precision, recall and F1 they give.

    A statistic with no denominator is ``None``, never 0: a label never predicted has no precision, one
    never expected has no recall, and either has no F1, rather than a figure stated over no evidence.

    Attributes:
        label: The label.
        expected: Observations whose case expected this label: its support, recall's denominator.
        expected_cases: Distinct cases behind ``expected`` — the independent draws recall rests on. Below
            ``expected`` when cases were repeated.
        predicted: Observations the classifier gave this label: precision's denominator.
        predicted_cases: Distinct cases behind ``predicted`` — the independent draws precision rests on.
        correct: Observations both expected and given this label.
        precision: ``correct / predicted``.
        precision_interval: The interval on ``precision``, over the cases behind it.
        recall: ``correct / expected``.
        recall_interval: The interval on ``recall``, over the cases behind it.
        f1: The harmonic mean of precision and recall, ``2 * correct / (predicted + expected)``. It is
            not a proportion of anything, so it has no interval: read the two it comes from.
    """

    # Not EvalBaseModel: a label's surrounding whitespace is part of it (see the module docstring).
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    expected: int
    expected_cases: int
    predicted: int
    predicted_cases: int
    correct: int
    precision: float | None
    precision_interval: tuple[float, float] | None
    recall: float | None
    recall_interval: tuple[float, float] | None
    f1: float | None


def confusion_matrix(cells: Mapping[str, int]) -> list[ConfusionCount]:
    """The confusion matrix of a ``confusion_cell`` measure's counts, ordered by expected then predicted label.

    Args:
        cells: Each ``confusion_cell`` value and how many observations carried it.

    Returns:
        One count per ``(expected, predicted)`` pair. A value that is not a confusion cell
        (:func:`~threetears.evals.kernel.metrics.confusion_of` reads no two labels from it) is in no cell.
    """
    pairs: dict[tuple[str, str], int] = {}
    for cell, count in cells.items():
        if (labels := confusion_of(cell)) is not None:
            pairs[labels] = pairs.get(labels, 0) + count
    return [
        ConfusionCount(expected=expected, predicted=predicted, count=count)
        for (expected, predicted), count in sorted(pairs.items())
    ]


def label_statistics(observations: Iterable[tuple[str, str]]) -> list[LabelStatistics]:
    """Each label's precision, recall and F1, counted from a classifier's observations, ordered by label.

    Every label the observations name is here, expected or predicted, so a label only ever given (an
    one outside the label set, say) appears with a precision and no recall.

    Args:
        observations: One ``(confusion_cell value, test case id)`` per observation. A value that is not a
            confusion cell (:func:`~threetears.evals.kernel.metrics.confusion_of` reads no two labels
            from it) is in no count, as in :func:`confusion_matrix`.

    Returns:
        One entry per label.
    """
    classified: list[tuple[str, str, str]] = [
        (labels[0], labels[1], case) for cell, case in observations if (labels := confusion_of(cell)) is not None
    ]
    statistics: list[LabelStatistics] = []
    for label in sorted({label for expected, predicted, _ in classified for label in (expected, predicted)}):
        given = [(expected == label, case) for expected, predicted, case in classified if predicted == label]
        met = [(predicted == label, case) for expected, predicted, case in classified if expected == label]
        correct = sum(1 for hit, _ in given if hit)
        predicted = len(given)
        expected = len(met)
        statistics.append(
            LabelStatistics(
                label=label,
                expected=expected,
                expected_cases=len({case for _, case in met}),
                predicted=predicted,
                predicted_cases=len({case for _, case in given}),
                correct=correct,
                precision=correct / predicted if predicted else None,
                precision_interval=_interval(given),
                recall=correct / expected if expected else None,
                recall_interval=_interval(met),
                f1=2 * correct / (predicted + expected) if predicted and expected else None,
            )
        )
    return statistics


def _interval(outcomes: Sequence[tuple[bool, str]]) -> tuple[float, float] | None:
    return proportion_interval([hit for hit, _ in outcomes], [case for _, case in outcomes])


__all__ = ["ConfusionCount", "LabelStatistics", "confusion_matrix", "label_statistics"]
