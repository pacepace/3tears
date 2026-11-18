"""The end-of-term curve for the first-year statistics course: raw final-exam scores to letter grades, per section.

Each section is curved on its own, so a hard paper set for one section doesn't sink it:

1. The section's mean is the mean of the scores of the students who sat the exam. A student with no score (a blank in
   the marks sheet: absent, or deferred to the resit) is not part of the mean.
2. If that mean is below :data:`TARGET_MEAN`, every score in the section moves up by the difference; a curved score
   never goes above 100. A section whose mean is at or above the target is left as it is: the curve never lowers a
   score.
3. A curved score gets its letter from :data:`BANDS`: A at 90 or more, B at 80, C at 70, D at 60, F below 60. A student
   with no score gets ``I`` (incomplete) whatever the curve.

Scores are on the 0 to 100 scale, in steps of a half mark.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

TARGET_MEAN = 72.0
CAP = 100.0
BANDS = ((90.0, "A"), (80.0, "B"), (70.0, "C"), (60.0, "D"))
INCOMPLETE = "I"


@dataclass(frozen=True)
class Result:
    student: str
    raw: float | None
    curved: float | None
    letter: str


def parse_score(text: str) -> float | None:
    """A score from the marks sheet: a number, or ``None`` for a blank cell."""
    text = text.strip()
    return float(text) if text else None


def section_mean(scores: list[float | None]) -> float:
    """The mean raw score of the students in a section who sat the exam."""
    return sum(s for s in scores if s is not None) / sum(s is not None for s in scores)


def shift(scores: list[float | None]) -> float:
    """How far every score in the section moves: up to the target mean, and never down."""
    return max(TARGET_MEAN - section_mean(scores), 0.0)


def letter(score: float | None) -> str:
    """The letter for a curved score."""
    if score is None:
        return INCOMPLETE
    return next((grade for floor, grade in BANDS if score >= floor), "F")


def curve_section(marks: list[tuple[str, float | None]]) -> list[Result]:
    """Every student's result in one section, in the order given."""
    if all(score is None for _, score in marks):
        return [Result(student, None, None, INCOMPLETE) for student, _ in marks]
    moved = shift([score for _, score in marks])
    results = []
    for student, score in marks:
        curved = None if score is None else min(score + moved, CAP)
        results.append(Result(student, score, curved, letter(curved)))
    return results


def read_marks(path: str | Path) -> list[tuple[str, float | None]]:
    """A section's marks sheet: a CSV with ``student`` and ``score`` columns."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [(row["student"].strip(), parse_score(row["score"])) for row in csv.DictReader(handle)]


def curve_file(path: str | Path) -> list[Result]:
    """Every student's result in the section whose marks sheet is at ``path``."""
    return curve_section(read_marks(path))
