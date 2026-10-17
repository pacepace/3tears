"""AST canary: the repair boundary is "did the call finish?", and nothing may drift off it.

The boundary is a CONTRACT expressed as an exception class at 30-odd raise sites, and for a
year nothing checked it. Two sites drifted onto the wrong side and stayed there unnoticed
because `SoundnessRefusal` subclasses `GenerationError`: every `pytest.raises(GenerationError)`
in the suite passes for either class, so a validator raising the terminal class where it meant
the repairable one is invisible to the tests that "cover" it — two validators making the same
argument one field apart can raise different classes and every test of either stays green.

So this asserts the rule structurally rather than per-site:

- The BARE `GenerationError` means no usable call happened, and there are exactly two ways for
  that to be true — nothing was sent, or the provider cut the call short. The allowlist below
  enumerates FUNCTIONS, not causes, and the two counts are not the same number: several
  functions can share the "nothing was sent" cause. So a third cause has to come and argue with
  this file, while a third function only has to justify itself in the allowlist.
- Everything else a finished call can get wrong raises `SoundnessRefusal` and buys its one
  repair round-trip.

An AST walk rather than a runtime probe: it sees a raise on a branch no test reaches, which is
where a drifted site hides.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: The only functions that may raise the bare class, and why each is allowed to.
#:
#: Every entry means "there is no output to correct", which is the whole of what the bare class
#: means on this path. Relational rather than counted: this read "Both mean" and a third entry
#: arrived, so the quantifier went false while the set it describes stayed right. Adding an entry
#: is a change to the repair boundary and needs the reasoning that `SoundnessRefusal`'s docstring
#: carries, not a line here.
_TERMINAL_RAISERS = {
    # The provider reported it cut the call short. Regenerating runs into the identical fixed
    # output cap and bills a second full charge to fail the same way.
    "_reject_incomplete_generation",
    # Nothing was sent: the bundle describes no arm, so the apparatus every arm-bound check joins
    # through is broken before a token is spent. There is no output because there was no call, and
    # no output a model could produce would repair it — the defect is in the evidence handed to it.
    "refuse_an_undescribable_arm_table",
    # Nothing was sent: the host does not allow this writer model (#644). Knowable from the caller's own
    # request before a token is spent, and no output could repair it — a second call is the same model
    # refused the same way.
    "refuse_an_unlisted_writer",
}

#: Every module of the analysis package, DERIVED rather than listed. A hand-maintained list is
#: the same drift this file exists to catch, one level up: a new module raising the bare class
#: would simply not be looked at, and nothing would say so. Only `generator.py` raises today;
#: the glob is what keeps a second one from escaping silently.
_ANALYSIS = Path(__file__).resolve().parents[1] / "src" / "threetears" / "evals" / "analysis"
_MODULES = sorted(p for p in _ANALYSIS.glob("*.py") if p.name != "__init__.py")


def _raises_by_function(path: Path) -> dict[str, set[str]]:
    """Map each function to the generation-contract exception classes it raises directly.

    Nested functions are attributed to their enclosing top-level function, since a closure
    raising inside `generate_analysis` is still that function's behaviour.
    """
    tree = ast.parse(path.read_text())
    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Raise) or inner.exc is None:
                continue
            call = inner.exc
            name = call.func if isinstance(call, ast.Call) else call
            # Both spellings: a bare `GenerationError` and a qualified `errors.GenerationError`.
            # The second is an `ast.Attribute`, and reading only `ast.Name` meant one import style
            # away from this module's own the rule stopped being enforced without failing.
            raised = name.id if isinstance(name, ast.Name) else name.attr if isinstance(name, ast.Attribute) else None
            if raised in {"GenerationError", "SoundnessRefusal"}:
                found.setdefault(node.name, set()).add(raised)
    return found


@pytest.mark.parametrize("path", _MODULES, ids=lambda p: p.name)
def test_only_the_listed_no_output_cases_raise_the_terminal_class(path: Path):
    """A finished call's failure must be repairable; only a missing call is terminal."""
    offenders = sorted(
        fn
        for fn, classes in _raises_by_function(path).items()
        if "GenerationError" in classes and fn not in _TERMINAL_RAISERS
    )
    assert offenders == [], (
        f"{path} raises the bare GenerationError in {offenders}. The bare class means there is no "
        "output to correct — nothing sent, or the provider cut the call short. Anything a FINISHED "
        "call got wrong raises SoundnessRefusal and buys its one repair round-trip. If a genuinely "
        "new no-output case exists, add it to _TERMINAL_RAISERS with its reasoning and say so in "
        "SoundnessRefusal's docstring, which is where the boundary is argued."
    )


def test_the_terminal_raisers_all_still_exist():
    """An allowlist that stops matching anything is a rule that quietly stopped being enforced.

    A renamed or deleted function would leave a stale entry licensing nothing, and the check
    above would keep passing while covering less.
    """
    raising = set()
    for path in _MODULES:
        raising |= {fn for fn, classes in _raises_by_function(path).items() if "GenerationError" in classes}
    stale = sorted(_TERMINAL_RAISERS - raising)
    assert stale == [], f"_TERMINAL_RAISERS names {stale}, which no longer raises the bare class — drop the entry"


def test_the_repairable_class_is_what_the_validators_raise():
    """The population the boundary exists for, asserted as a floor rather than a count.

    A count would go stale the first time a validator landed — which is the drift this file
    exists to catch, so it must not reintroduce it in its own assertion.
    """
    repairing = {
        fn for path in _MODULES for fn, classes in _raises_by_function(path).items() if "SoundnessRefusal" in classes
    }
    # Named rather than counted, and a floor rather than the whole set: the chain itself (a reply off
    # the contract, or a stored-model refusal), the parse step, and the chain's `_reject_*` member.
    # A scan that collapsed loses one of them; a validator added beside them joins without an edit.
    chain_sites = {"_assemble_and_validate", "_parse_payload", "_reject_mismatched_question_answers"}
    assert chain_sites <= repairing, f"the repairable population lost {sorted(chain_sites - repairing)}"
    assert "_reject_incomplete_generation" not in repairing, "the truncation guard must never buy a repair"


def test_the_walk_reads_the_analysis_package():
    """Non-vacuity: the module list is derived from a path, and a wrong path derives nothing.

    Every parametrised case above vanishes when the glob matches nothing, which reads as a pass.
    """
    assert any(path.name == "generator.py" for path in _MODULES), _MODULES
