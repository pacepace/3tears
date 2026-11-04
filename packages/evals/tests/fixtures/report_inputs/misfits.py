"""The misfit report inputs: the reference inputs with the criteria or configs a chart or a story leans on taken away.

Each builder reads a reference input (``rN/input.json``) fresh and changes it, so a test can mutate what it gets:

- r2 without its gates, with its gates read in each task, and with one gate;
- r3 without its configs;
- r1 without its gates, and r1 deciding by nothing but one absolute bar on its latency.

They are where a chart a planner or an analyst might name has nothing to draw, and where an exemplar's story no longer
fits: the plan checks are held to refuse or render on them, and every analyst tool to answer on them. :data:`MISFITS`
names each one.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

#: The reference inputs, by name.
NAMES = ("r1", "r2", "r3", "r4", "r5")


def reference_data(name: str) -> dict[str, Any]:
    """One reference input (``r1`` to ``r5``) as its JSON, read fresh."""
    loaded: dict[str, Any] = json.loads((HERE / name / "input.json").read_text(encoding="utf-8"))
    return loaded


def without_gates(name: str) -> dict[str, Any]:
    """A reference input with no declared criteria."""
    data = reference_data(name)
    data["criteria"] = []
    return data


def gates_in_each_task() -> dict[str, Any]:
    """r2 with each of its gates read in each task rather than across them."""
    data = reference_data("r2")
    for criterion in data["criteria"]:
        criterion["scope"] = "each_task"
    return data


def one_gate() -> dict[str, Any]:
    """r2 with only its first gate."""
    data = reference_data("r2")
    data["criteria"] = data["criteria"][:1]
    return data


def without_configs() -> dict[str, Any]:
    """r3 with no arm's level of any factor."""
    data = reference_data("r3")
    data["configs"] = []
    return data


def only_bars() -> dict[str, Any]:
    """r1 deciding by nothing but one absolute bar on its latency, read met on every arm."""
    data = without_gates("r1")
    data["bars"] = [
        {"aspect": "overall", "arm": arm["id"], "measure": "latency_s", "bar": 9.0, "direction": "lower_is_better",
         "status": "met"}
        for arm in data["arms"]
    ]  # fmt: skip
    return data


#: Every misfit input, by what it is, each read fresh.
MISFITS: dict[str, Callable[[], dict[str, Any]]] = {
    "r2 without its gates": lambda: without_gates("r2"),
    "r2 with its gates read in each task": gates_in_each_task,
    "r2 with one gate": one_gate,
    "r3 without its configs": without_configs,
    "r1 without its gates": lambda: without_gates("r1"),
    "r1 reading only bars": only_bars,
}

__all__ = [
    "MISFITS",
    "NAMES",
    "gates_in_each_task",
    "one_gate",
    "only_bars",
    "reference_data",
    "without_configs",
    "without_gates",
]
