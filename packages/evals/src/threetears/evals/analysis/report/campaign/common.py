"""What several parts of a campaign's page share: each arm's name on one aspect, and a floor by its key."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from threetears.evals.analysis.arms import arm_names
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.style import ArmName, arm_name
from threetears.evals.kernel.declaration import Floor

__all__ = ["floor_of", "named_arms"]


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
