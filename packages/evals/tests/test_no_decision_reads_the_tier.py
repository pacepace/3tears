"""No decision site imports the evidence tier: a decision reads checks, and the tier is kept for old readers only.

The tier is still computed and stored (``kernel/evidence_tiers.py``, from the check against people and the check
against itself), and a page one step down may still name it until its section retires. What no longer may is code
that decides: a verdict, a reading's state, a contrast's standing, which check speaks, a caveat that sets something
aside. Each such module is walked for an import of a tier name, by any spelling (``from ... import name``, an aliased
import, or the module whose names they are).

A decision module the program has planned and not yet built is listed and held as soon as it exists.
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


def test_no_decision_site_imports_the_tier() -> None:
    sites = [SOURCE / site for site in DECISION_SITES]
    missing = [str(site.relative_to(SOURCE)) for site in sites if not site.is_file()]
    assert missing == [], f"a decision site moved: name it here where it went ({missing})"
    sites += [SOURCE / site for site in PLANNED_DECISION_SITES if (SOURCE / site).is_file()]
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
