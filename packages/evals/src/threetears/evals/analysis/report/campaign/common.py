"""What several parts of a campaign's page share: each arm's name on one aspect, a floor by its key, and how an aspect's
rate of trials is worded (a pass rate, or the share answered where nothing graded can fail a trial)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

from threetears.evals.analysis.arms import arm_names
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.bundle.trials import reads_answered
from threetears.evals.analysis.report.words import PASS_RATE
from threetears.evals.analysis.style import ArmName, arm_name
from threetears.evals.kernel.declaration import Floor

__all__ = [
    "ANSWERED_WORDS",
    "MIXED_WORDS",
    "PASS_WORDS",
    "RateWords",
    "floor_of",
    "named_arms",
    "rate_words",
]


@dataclass(frozen=True)
class RateWords:
    """How a page words an aspect's rate of trials: a pass rate, or where nothing graded can fail a trial and an arm
    failed to answer some, the share answered (:func:`~threetears.evals.analysis.bundle.trials.reads_answered`).

    :ivar heading: The reading's name, as a lead or a row heads it (``Pass rate``, ``Answered``).
    :ivar measure: As a column head names it after an arm (``pass rate``, ``answered``).
    :ivar verb: What an arm did with a trial it counts (``passed``, ``answered``).
    :ivar term: The glossary term a column of the counts carries.
    :ivar noun: As a sentence names the rate (``the pass rate``, ``the share answered``).
    """

    heading: str
    measure: str
    verb: str
    term: str
    noun: str


#: A pass rate's words.
PASS_WORDS: Final = RateWords(
    heading=PASS_RATE, measure=PASS_RATE.lower(), verb="passed", term="passed", noun="the pass rate"
)

#: The share answered's words, where nothing graded can fail a trial.
ANSWERED_WORDS: Final = RateWords(
    heading="Answered", measure="answered", verb="answered", term="answered", noun="the share answered"
)

#: The words across aspects some of which read a pass rate and some the share answered: neither name alone is true of
#: every figure, so the rate is named as both.
MIXED_WORDS: Final = RateWords(
    heading="Pass or answered rate",
    measure="pass or answered rate",
    verb="passed or answered",
    term="answered",
    noun="the pass or answered rate",
)


def rate_words(bundles: Iterable[AnalysisContextBundle]) -> RateWords:
    """The words for the rate of the aspects ``bundles`` hold: the share answered where every one reads it
    (:func:`~threetears.evals.analysis.bundle.trials.reads_answered`), a pass rate where none does, and both names
    where some do, so no figure over an aspect nothing can fail a trial on is called a pass rate.

    :param bundles: The aspects' evidence.
    :return: The words.
    """
    answered = [reads_answered(bundle) for bundle in bundles]
    if answered and all(answered):
        return ANSWERED_WORDS
    return MIXED_WORDS if any(answered) else PASS_WORDS


def named_arms(bundle: AnalysisContextBundle, labels: Mapping[str, str]) -> dict[str, ArmName]:
    """Each arm the bundle holds, named as its variant names it, its label the campaign's where it gives one.

    :param bundle: One aspect's evidence.
    :param labels: The campaign's label for each arm, by variant key.
    :return: Each arm's name, by variant key.
    """
    names = {
        key: arm_name(display, declared=bundle.lever_names) for key, display in arm_names(bundle.variant_index).items()
    }
    return {key: replace(name, label=labels[key], given=True) if key in labels else name for key, name in names.items()}


def floor_of(floors: Sequence[Floor], key: str) -> Floor | None:
    """The floor by its key, among a campaign's floors.

    :param floors: The campaign's floors.
    :param key: The floor's key.
    :return: The floor; None where none has the key.
    """
    return next((floor for floor in floors if floor.key == key), None)
