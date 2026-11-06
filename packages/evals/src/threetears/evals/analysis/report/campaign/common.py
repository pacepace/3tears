"""What several parts of a campaign's page share: the memo's words as the page prints them, each arm's name on one
aspect, a floor by its key, and a change in points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from threetears.evals.analysis.arms import arm_names
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.kernel.numbers import format_number
from threetears.evals.analysis.style import ArmName, arm_name
from threetears.evals.kernel.declaration import Floor

__all__ = ["as_written", "floor_of", "named_arms", "points"]


def as_written(text: str) -> str:
    """The writer's words as the page prints them: as written, on one line. Code reads none of them: whether they
    keep the prompt's rules is for a host's reporter eval to measure, in whatever language they are written.

    :param text: The writer's words.
    :ptype text: str
    :return: The words, each run of whitespace one space.
    :rtype: str
    """
    return " ".join(text.split())


def named_arms(bundle: AnalysisContextBundle, labels: Mapping[str, str]) -> dict[str, ArmName]:
    """Each arm the bundle holds, named as its variant names it, its label the campaign's where it gives one.

    :param bundle: One aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param labels: The campaign's label for each arm, by variant key.
    :ptype labels: Mapping[str, str]
    :return: Each arm's name, by variant key.
    :rtype: dict[str, ArmName]
    """
    names = {key: arm_name(display) for key, display in arm_names(bundle.variant_index).items()}
    return {key: replace(name, label=labels[key], given=True) if key in labels else name for key, name in names.items()}


def floor_of(floors: Sequence[Floor], key: str) -> Floor | None:
    """The floor by its key, among a campaign's floors.

    :param floors: The campaign's floors.
    :ptype floors: Sequence[Floor]
    :param key: The floor's key.
    :ptype key: str
    :return: The floor; None where none has the key.
    :rtype: Floor | None
    """
    return next((floor for floor in floors if floor.key == key), None)


def points(value: float) -> str:
    """A change in points to one decimal, a whole number whole.

    :param value: The change, in points.
    :ptype value: float
    :return: The change as the page spells it.
    :rtype: str
    """
    return format_number(round(value, 1))
