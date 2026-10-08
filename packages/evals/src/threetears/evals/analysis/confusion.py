"""A classifier's confusion matrix and the per-label statistics it yields, counted once for every surface.

A classifier lands one ``confusion_cell`` per observation (``expected → predicted``,
:func:`~threetears.evals.contracts.confusion_cell`). Counted, those cells are its confusion matrix
(:func:`confusion_matrix`), and from the matrix each label's precision, recall and F1 follow
(:func:`label_statistics`). The analysis bundle and the run summary both read them from here, so a
label's precision is one computation wherever it is printed.

**Labels are kept exactly as given.** These models do not strip their strings, unlike every model
on :class:`~threetears.evals.contracts.base.EvalBaseModel`: a label is free text, and ``"positive "``
stripped to ``"positive"`` would count a wrong answer as a right one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from threetears.evals.analysis.stats import wilson_interval
from threetears.evals.contracts.metrics import confusion_of


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
        predicted: Observations the classifier gave this label: precision's denominator.
        correct: Observations both expected and given this label.
        precision: ``correct / predicted``.
        precision_interval: The Wilson interval on ``precision``.
        recall: ``correct / expected``.
        recall_interval: The Wilson interval on ``recall``.
        f1: The harmonic mean of precision and recall, ``2 * correct / (predicted + expected)``. It is
            not a proportion of anything, so it has no interval: read the two it comes from.
    """

    # Not EvalBaseModel: a label's surrounding whitespace is part of it (see the module docstring).
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    expected: int
    predicted: int
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
        (:func:`~threetears.evals.contracts.metrics.confusion_of` reads no two labels from it) is in no cell.
    """
    pairs: dict[tuple[str, str], int] = {}
    for cell, count in cells.items():
        if (labels := confusion_of(cell)) is not None:
            pairs[labels] = pairs.get(labels, 0) + count
    return [
        ConfusionCount(expected=expected, predicted=predicted, count=count)
        for (expected, predicted), count in sorted(pairs.items())
    ]


def label_statistics(matrix: Iterable[ConfusionCount]) -> list[LabelStatistics]:
    """Each label's precision, recall and F1, counted from a confusion matrix, ordered by label.

    Every label the matrix names is here, expected or predicted, so a label only ever given (an
    one outside the label set, say) appears with a precision and no recall.

    Args:
        matrix: The confusion matrix (:func:`confusion_matrix`).

    Returns:
        One entry per label.
    """
    cells = list(matrix)
    statistics: list[LabelStatistics] = []
    for label in sorted({label for cell in cells for label in (cell.expected, cell.predicted)}):
        correct = sum(cell.count for cell in cells if cell.expected == label and cell.predicted == label)
        predicted = sum(cell.count for cell in cells if cell.predicted == label)
        expected = sum(cell.count for cell in cells if cell.expected == label)
        statistics.append(
            LabelStatistics(
                label=label,
                expected=expected,
                predicted=predicted,
                correct=correct,
                precision=correct / predicted if predicted else None,
                precision_interval=wilson_interval(correct, predicted),
                recall=correct / expected if expected else None,
                recall_interval=wilson_interval(correct, expected),
                f1=2 * correct / (predicted + expected) if predicted and expected else None,
            )
        )
    return statistics


__all__ = ["ConfusionCount", "LabelStatistics", "confusion_matrix", "label_statistics"]
