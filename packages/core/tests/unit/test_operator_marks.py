"""The operator marks: they record their reason and hand the function back unchanged."""

from __future__ import annotations

import pytest

from threetears.core.operator import operator_command, operator_reset


def test_a_mark_returns_the_function_it_was_given_with_its_reason() -> None:
    """Flips if a mark wraps the function (a framework above it would then see a different signature) or drops
    its reason."""

    async def empty(names: list[str]) -> None:
        return None

    marked = operator_reset("an operator empties the tables it names", proof="tests/x.py::test_y")(empty)
    assert marked is empty
    assert getattr(empty, "__operator_reset__") == "an operator empties the tables it names"

    def drill() -> int:
        return 0

    assert operator_command("a drill counts its scratch restore", proof="tests/x.py::test_z")(drill) is drill
    assert getattr(drill, "__operator_command__") == "a drill counts its scratch restore"


@pytest.mark.parametrize(("reason", "proof"), [("", "tests/x.py::test_y"), ("why", "tests/x.py")])
def test_a_mark_without_a_reason_or_a_proof_is_refused(reason: str, proof: str) -> None:
    """Flips if a mark is accepted with no reason, or with a proof that names no test function."""
    with pytest.raises(ValueError, match="reason and a proof"):
        operator_reset(reason, proof=proof)
