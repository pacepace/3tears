"""No decision site imports or reads the evidence tier: a decision reads checks, and the tier is kept for old readers.

The tier is still computed and stored (``kernel/evidence_tiers.py``, from the check against people and the check
against itself), and a page one step down may still name it until its section retires. What no longer may is code
that decides: a verdict, a reading's state, a contrast's standing, which check speaks, a caveat that sets something
aside. Each such module is walked twice:

- for an import of a tier name, by any spelling (``from ... import name``, an aliased import, or the module whose names
  they are);
- for a read of a tier value: an attribute ``tier`` or ``judge_tiers`` (the stored tier word, as a report input holds
  it), or the name ``_UNCALIBRATED`` (the tier words a decision once set an improvement aside by).

A decision module the program has planned and not yet built is listed and held as soon as it exists. Two reads are
pinned (:data:`PINNED_TIER_READS`): an input that holds no check (stored before checks were recorded, or written by
hand) is still read by the tier it states, in one function, and listed by its tier map's keys in another. The pin must
shrink: a pinned function that no longer reads the tier fails until it is taken off, and a read anywhere else fails.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "threetears" / "evals"

#: The modules that decide: a verdict, a reading's state, a contrast's standing, which check speaks, a caveat.
DECISION_SITES = (
    "analysis/checks.py",
    "analysis/caveats.py",
    "analysis/first_screen.py",
    "analysis/significance.py",
    "analysis/bundle/comparisons.py",
    "analysis/report/call.py",
    "analysis/report/gates.py",
    "analysis/report/verdict.py",
)

#: Decision modules the program plans (the decision read, the answer, the criteria); held once they exist.
PLANNED_DECISION_SITES = (
    "analysis/answer.py",
    "analysis/criteria.py",
    "analysis/report/decision.py",
)

#: The names that are the tier or decide it.
TIER_NAMES = frozenset(
    {
        "EvidenceTier",
        "JUDGED_TIERS_WEAKEST_FIRST",
        "JUDGED_TIER_RULE",
        "JudgedEvidenceTier",
        "JudgedTierRule",
        "evidence_tier_of",
        "judge_evidence_tiers",
        "tier_for_judges",
        "tier_of",
        "weakest_judged_tier",
    }
)

#: The module whose every name is a tier's: imported whole, it brings the tier in.
TIER_MODULE = "threetears.evals.kernel.evidence_tiers"

#: The attributes that hold a tier value where a decision could read one: a judged dimension's ``tier`` and a report
#: input's ``judge_tiers``.
TIER_ATTRIBUTES = frozenset({"tier", "judge_tiers"})

#: The names that hold tier words a decision once read.
TIER_VALUE_NAMES = frozenset({"_UNCALIBRATED"})

#: The tier reads still allowed, by module and the function that holds them, for an input with no check: it is read by
#: the tier it states (``_stated_standing``), and a dimension only its tier map names is listed by that map's keys
#: (``judged_dimensions``). E1 (the evidence-tier retirement) takes them off.
PINNED_TIER_READS: dict[str, frozenset[str]] = {
    "analysis/answer.py": frozenset({"_stated_standing", "judged_dimensions"}),
}


def _tier_imports(path: Path) -> list[str]:
    """Each import in ``path`` that brings a tier name in, as ``line: what``."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in TIER_NAMES or alias.name == "*" and node.module == TIER_MODULE:
                    found.append(f"{node.lineno}: {alias.name} from {node.module}")
        elif isinstance(node, ast.Import):
            found += [f"{node.lineno}: import {alias.name}" for alias in node.names if alias.name == TIER_MODULE]
    return found


def _tier_reads(path: Path) -> list[tuple[str, str]]:
    """Each read of a tier value in ``path``, as ``(the enclosing function, "line: what")``; a module-level read's
    function is ``""``, and the line that binds a tier-value name counts as a read of it."""
    found: list[tuple[str, str]] = []

    def walk(node: ast.AST, where: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, child.name)
                continue
            if isinstance(child, ast.Attribute) and child.attr in TIER_ATTRIBUTES:
                found.append((where, f"{child.lineno}: .{child.attr}"))
            elif isinstance(child, ast.Name) and child.id in TIER_VALUE_NAMES and isinstance(child.ctx, ast.Load):
                found.append((where, f"{child.lineno}: {child.id}"))
            walk(child, where)

    walk(ast.parse(path.read_text(encoding="utf-8")), "")
    return found


def _decision_sites() -> list[Path]:
    sites = [SOURCE / site for site in DECISION_SITES]
    missing = [str(site.relative_to(SOURCE)) for site in sites if not site.is_file()]
    assert missing == [], f"a decision site moved: name it here where it went ({missing})"
    return sites + [SOURCE / site for site in PLANNED_DECISION_SITES if (SOURCE / site).is_file()]


def test_no_decision_site_reads_a_tier_value_outside_its_pin() -> None:
    offenders: dict[str, list[str]] = {}
    pinned_and_reading: set[tuple[str, str]] = set()
    for site in _decision_sites():
        name = str(site.relative_to(SOURCE))
        allowed = PINNED_TIER_READS.get(name, frozenset())
        for function, read in _tier_reads(site):
            if function in allowed:
                pinned_and_reading.add((name, function))
            else:
                offenders.setdefault(name, []).append(f"{function or '<module>'} {read}")
    assert offenders == {}, "a decision reads a tier value: read the judge's checks instead"
    stale = {(name, function) for name, functions in PINNED_TIER_READS.items() for function in functions}
    assert stale - pinned_and_reading == set(), "a pinned function no longer reads the tier: take it off the pin"


def test_the_walk_sees_a_tier_read_by_attribute_and_by_the_tier_words(tmp_path: Path) -> None:
    # The read the decision made before it read checks (analysis/answer.py at dec50c602, lines 196, 198 and 229).
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def standing(report_input, judged, dimension):\n"
        "    stated = [report_input.caveats.judge_tiers.get(dimension)]\n"
        "    stated += [judged.tier, judged.calibration]\n"
        "    return set(stated) <= _UNCALIBRATED\n"
        "tier = judged.calibration.tier_of\n",
        encoding="utf-8",
    )
    assert _tier_reads(probe) == [
        ("standing", "2: .judge_tiers"),
        ("standing", "3: .tier"),
        ("standing", "4: _UNCALIBRATED"),
    ]


def test_no_decision_site_imports_the_tier() -> None:
    sites = _decision_sites()
    offenders = {str(site.relative_to(SOURCE)): found for site in sites if (found := _tier_imports(site))}
    assert offenders == {}


def test_the_walk_sees_a_tier_import_by_every_spelling(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from threetears.evals.kernel.evidence_tiers import JudgedEvidenceTier as T\n"
        "from threetears.evals.analysis.agreement import tier_for_judges\n"
        "import threetears.evals.kernel.evidence_tiers\n"
        "from threetears.evals.kernel.evidence_tiers import CriterionState\n",
        encoding="utf-8",
    )
    assert [line.split(":")[0] for line in _tier_imports(probe)] == ["1", "2", "3"]
