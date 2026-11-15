"""One module joins an interval's two ends: ``analysis/report/unit_once.py``.

Where an interval's unit is written, and how its ends are joined (``80 to 89``), is one rule (number rule 3), so every
module under ``analysis/report`` joins two ends through :mod:`~threetears.evals.analysis.report.unit_once`
(``interval_beside``, ``interval_alone``) rather than by hand.

**What the walk counts as a hand join.** Read on the syntax tree, so a sentence that says "to" in its words is never
taken for one: an f-string in which the literal ``" to "`` stands alone between two interpolated values
(``f"{low} to {high}"``); a ``+`` with ``" to "`` as an operand; ``" to ".join(...)``; and a string constant holding
``"} to {"``, a template's join (``"{low} to {high}"``).

**What it cannot see.** A join built from a literal that says more than ``" to "`` (``f"{low} to {high} cases"`` is
seen, but ``f"from {low} up to {high}"`` and ``"%s to %s" % ends`` are not); ``"to"`` held in a name and joined
elsewhere; and two values interpolated with ``" to "`` between them that are words, not figures, which is why
:data:`_NOT_INTERVALS` names each such site with its reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPORT = Path(__file__).resolve().parents[1] / "src" / "threetears" / "evals" / "analysis" / "report"

#: The one module that joins an interval's ends.
_OWNER = "unit_once.py"

#: Sites the walk reads as a join that join no interval, keyed by (module relative to ``analysis/report``, the
#: expression as written), each with why. An entry no site matches fails, so the list only shrinks.
_NOT_INTERVALS: dict[tuple[str, str], str] = {
    (
        "aspect/method.py",
        "f'{name} to {level}'",
    ): "a setting moved to a level, in words: a lever's name, then where it was set",
    ("html_parts.py", "f'{_INTERVAL_END} to {_INTERVAL_END}(?: [a-z%]+)?'"): (
        "a regular expression that finds an interval already joined, written as one"
    ),
}


def _joins(source: str) -> list[tuple[int, str]]:
    """Each hand join in ``source``: its line and its expression as written."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.JoinedStr):
            parts = node.values
            if any(
                isinstance(parts[index], ast.Constant)
                and parts[index].value == " to "
                and isinstance(parts[index - 1], ast.FormattedValue)
                and isinstance(parts[index + 1], ast.FormattedValue)
                for index in range(1, len(parts) - 1)
            ):
                found.append((node.lineno, ast.unparse(node)))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if any(isinstance(side, ast.Constant) and side.value == " to " for side in (node.left, node.right)):
                found.append((node.lineno, ast.unparse(node)))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            if isinstance(node.func.value, ast.Constant) and node.func.value.value == " to ":
                found.append((node.lineno, ast.unparse(node)))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "} to {" in node.value:
            found.append((node.lineno, ast.unparse(node)))
    return found


def _sites() -> list[tuple[str, int, str]]:
    """Every hand join under ``analysis/report`` but the owner's: (module, line, expression)."""
    return [
        (path.relative_to(_REPORT).as_posix(), line, expression)
        for path in sorted(_REPORT.rglob("*.py"))
        if path.name != _OWNER
        for line, expression in _joins(path.read_text(encoding="utf-8"))
    ]


def test_no_module_but_unit_once_joins_an_interval_by_hand() -> None:
    found = [
        f"{module}:{line}: {expression}"
        for module, line, expression in _sites()
        if (module, expression) not in _NOT_INTERVALS
    ]
    assert not found, (
        "join an interval's ends through analysis/report/unit_once.py (interval_beside, interval_alone), or name a "
        "site that joins no interval in _NOT_INTERVALS with why:\n  " + "\n  ".join(found)
    )


def test_every_named_exception_still_names_a_site() -> None:
    live = {(module, expression) for module, _line, expression in _sites()}
    assert sorted(set(_NOT_INTERVALS) - live) == []


def test_the_walk_reads_the_report_tree() -> None:
    assert (_REPORT / _OWNER).is_file()
    assert len(list(_REPORT.rglob("*.py"))) > 20


@pytest.mark.parametrize(
    "source",
    [
        'f"{low} to {high}"',
        'f" ({low} to {high}) cases"',
        'low + " to " + high',
        '" to ".join(ends)',
        'TEMPLATE = "{low} to {high}"',
    ],
)
def test_the_walk_sees_each_way_of_joining_by_hand(source: str) -> None:
    assert len(_joins(source)) == 1


@pytest.mark.parametrize(
    "source",
    [
        'f"Switch to {name}"',
        'f"{arm} moved to the control"',
        "interval_beside(low, high)",
        '"from 3 to 5"',
        'f"{low}-to-{high}"',
    ],
)
def test_the_walk_leaves_words_that_say_to(source: str) -> None:
    assert _joins(source) == []
