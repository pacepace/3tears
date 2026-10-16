"""enforcement: the recorded-absence age goes by its date (receiver-trust Chunk 12, Pace 2026-10-15).

``negative_cache_max_age`` is the one timer on a cache the write generations invalidate. It stays
only as the stated cover for writers that predate the write marks
(:mod:`threetears.core.collections.write_marks`): a commit such a writer made, then died before
advancing, is caught by nothing else. Once every writer marks its writes, the timer and "The One
Timer That Stays" (``docs/epoch-task-06-collection-generations-by-default.md``) go together, in the
contract commit. This gate dates that commit: from :data:`CONTRACT_DATE` the build fails while any
collection still sets the age.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path
from typing import Final

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: the contract commit's date, the door windows' (Pace, 2026-10-15)
CONTRACT_DATE: Final[date] = date(2027, 1, 5)


def ages_set(source: str) -> list[str]:
    """the classes in ``source`` that give ``negative_cache_max_age`` a value other than ``None``.

    :param source: a module's text
    :ptype source: str
    :return: the class names
    :rtype: list[str]
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            target = statement.target if isinstance(statement, ast.AnnAssign) else None
            value = statement.value if isinstance(statement, ast.AnnAssign) else None
            if (
                isinstance(target, ast.Name)
                and target.id == "negative_cache_max_age"
                and value is not None
                and not (isinstance(value, ast.Constant) and value.value is None)
            ):
                found.append(node.name)
    return found


def overdue(today: date, sources: dict[str, str]) -> list[str]:
    """the ages still set on or after the contract date.

    :param today: the date the build runs
    :ptype today: date
    :param sources: path to module text
    :ptype sources: dict[str, str]
    :return: ``path::Class`` for each age still set once the date has come; empty before it
    :rtype: list[str]
    """
    if today < CONTRACT_DATE:
        return []
    return [f"{path}::{name}" for path, text in sorted(sources.items()) for name in ages_set(text)]


def _sources() -> dict[str, str]:
    return {
        str(path.relative_to(_REPO_ROOT)): path.read_text(encoding="utf-8")
        for pattern in ("packages/*/src", "packages/agent/*/src")
        for root in _REPO_ROOT.glob(pattern)
        for path in sorted(root.rglob("*.py"))
    }


def test_the_recorded_absence_age_is_gone_by_its_contract_date() -> None:
    sources = _sources()
    assert any(ages_set(text) for text in sources.values()), "no age found at all; the reader reads the wrong source"
    late = overdue(date.today(), sources)
    assert not late, (
        f"the recorded-absence age outlived its contract date {CONTRACT_DATE}: {late}. Delete it, and "
        "'The One Timer That Stays', in the contract commit (receiver-trust plan, Chunk 12)"
    )


def test_an_age_left_past_the_date_is_reported_and_one_before_it_is_not() -> None:
    source = (
        "class Revocations(Base):\n"
        "    negative_cache_max_age: ClassVar[timedelta | None] = timedelta(seconds=60)\n"
        "class Base:\n"
        "    negative_cache_max_age: ClassVar[timedelta | None] = None\n"
    )
    assert ages_set(source) == ["Revocations"]
    assert overdue(CONTRACT_DATE, {"m.py": source}) == ["m.py::Revocations"]
    assert overdue(date(2027, 1, 4), {"m.py": source}) == []
