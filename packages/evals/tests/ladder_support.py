"""Run a ladder rung under ``examples/<n>-<name>/`` as its reader runs it: from its own directory, offline.

A rung names its cases, its recording and its store by paths relative to the directory it is run from, as a reader
runs it (``cd examples/3-invoices && python extract.py``). A test copies the rung into its own temporary directory
and runs it there, so the run's store, report and ``__pycache__`` never land in the checkout, and takes every
provider key and ``RECORD`` out of the environment, so a model built on a recording can only replay it.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

#: The examples directory.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

#: Every variable a model reads a key from, and the switch that asks for a recording to be made again.
_LIVE_VARIABLES = ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RECORD")


def offline_copy(rung: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy ``examples/<rung>`` into ``tmp_path``, make it the working directory and ``sys.path``'s head, offline.

    Args:
        rung: The rung's directory name, ``"3-invoices"``.
        tmp_path: The test's temporary directory.
        monkeypatch: The test's monkeypatch, which undoes all of it.

    Returns:
        The copy.
    """
    copy = tmp_path / rung
    shutil.copytree(EXAMPLES / rung, copy, ignore=shutil.ignore_patterns(".evals", "report", "__pycache__"))
    for variable in _LIVE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.chdir(copy)
    monkeypatch.syspath_prepend(str(copy))
    return copy


def load(path: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Execute ``path`` as module ``name``, registered so a sibling's ``from <name> import ...`` finds this one.

    Args:
        path: The file.
        name: The module name its siblings import it by.
        monkeypatch: Unregisters the module when the test ends.

    Returns:
        The module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@dataclass
class Served:
    """The calls a rung's model answered while a test ran it: each ``(system, user)``, in the order asked."""

    calls: list[tuple[str, str]] = field(default_factory=list)


def count_calls(model: Any, monkeypatch: pytest.MonkeyPatch) -> Served:
    """Count every call ``model`` (a :class:`~threetears.evals.Model`) serves from here on.

    Args:
        model: The rung's model.
        monkeypatch: Restores the model's own ``generate`` when the test ends.

    Returns:
        The tally, filled as calls are served.
    """
    served = Served()
    generate: Callable[..., Awaitable[Any]] = model.generate

    async def counted(*, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        completion = await generate(system=system, user=user, response_format=response_format)
        served.calls.append((system, user))
        return completion

    monkeypatch.setattr(model, "generate", counted)
    return served


def recorded_spend(recording: Path) -> tuple[int, float]:
    """How many calls a recording holds and what they cost, read from each line's own ``cost_usd``.

    Raises:
        AssertionError: A call the recording holds unpriced, which no docstring could total.
    """
    lines = [json.loads(line) for line in recording.read_text(encoding="utf-8").splitlines() if line.strip()]
    costs = [line["result"]["cost_usd"] for line in lines]
    assert None not in costs, f"{recording} holds an unpriced call"
    return len(lines), sum(costs)


def stated_spend(example: Path) -> tuple[int, float]:
    """The call count and dollars an example's docstring states it was recorded for (``180 calls, $0.0103``)."""
    found = re.search(r"(\d+) calls, \$(\d+\.\d+)", example.read_text(encoding="utf-8"))
    assert found is not None, f"{example.name}'s docstring states no recorded cost"
    return int(found.group(1)), float(found.group(2))


def non_blank_lines(path: Path) -> int:
    """How many lines of ``path`` hold anything."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def secret_markers(path: Path) -> list[str]:
    """Every marker of key material found in a file: a key's prefix, or a provider's key variable named."""
    text = path.read_text(encoding="utf-8")
    return [marker for marker in ("sk-", "or-v1", "OPENROUTER", "ANTHROPIC") if marker in text]
