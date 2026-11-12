"""What the ladder's rung tests share about a rung's files: what its recordings say, and what its program holds.

A rung is copied out to run in by :func:`~packages.evals.tests.ladder_support.offline_copy`, its one owner.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


from threetears.evals.schema import ClientRequestSettings


def recorded(path: Path) -> list[dict[str, object]]:
    """Every line of a recording, as JSON."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recorded_cost(path: Path) -> tuple[int, str]:
    """How many calls a recording holds and what they cost, from its own cost fields, as a docstring states it."""
    lines = recorded(path)
    total = 0.0
    for line in lines:
        result = line["result"]
        assert isinstance(result, dict) and isinstance(result["cost_usd"], float), f"{path} holds an unpriced call"
        total += result["cost_usd"]
    return len(lines), f"${total:.4f}"


def recorded_settings(path: Path) -> ClientRequestSettings:
    """The request settings a recording's client was built with, which a coverage check selects its calls by."""
    settings = {json.dumps(line["settings"], sort_keys=True) for line in recorded(path)}
    assert len(settings) == 1, f"{path} holds calls under {len(settings)} request settings"
    return ClientRequestSettings.model_validate(json.loads(settings.pop()))


def program_lines(path: Path) -> int:
    """A Python file's non-blank lines below its module docstring: the program, without the prose about it."""
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    first = module.body[0]
    docstring = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
    start = first.end_lineno if docstring and first.end_lineno is not None else 0
    return sum(1 for line in source.splitlines()[start:] if line.strip())


def root_only_imports(path: Path) -> list[str]:
    """Every engine import in a file that is not from the package root, nor ``quick.as_host`` or ``quick.measure``."""
    allowed = {"threetears.evals.quick": {"as_host", "measure"}}
    reaching: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("threetears"):
            module = node.module or ""
            if module == "threetears.evals":
                continue
            reaching += [f"{module}.{alias.name}" for alias in node.names if alias.name not in allowed.get(module, ())]
        elif isinstance(node, ast.Import):
            reaching += [alias.name for alias in node.names if alias.name.startswith("threetears")]
    return reaching


__all__ = ["program_lines", "recorded", "recorded_cost", "recorded_settings", "root_only_imports"]
