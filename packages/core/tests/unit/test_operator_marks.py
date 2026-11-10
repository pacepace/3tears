"""The operator marks: they record their reason and hand the function back unchanged."""

from __future__ import annotations

import pytest

from threetears.core.operator import operator_command, operator_reset, security_backstop, source_change_read


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


def test_a_security_backstop_and_a_source_change_read_return_the_function_they_mark() -> None:
    """Flips if either mark wraps the function, or drops what it records."""

    async def reconcile() -> None:
        return None

    marked = security_backstop(
        "a missed revocation would leave a connection open for a day",
        proof="tests/x.py::test_catches",
        speed="tests/x.py::test_budget",
    )(reconcile)
    assert marked is reconcile
    assert getattr(reconcile, "__security_backstop__") == "a missed revocation would leave a connection open for a day"

    def changes() -> None:
        return None

    assert source_change_read(stamps=("upload_timestamp",), proof="tests/x.py::test_y")(changes) is changes
    assert getattr(changes, "__source_change_read__") == "upload_timestamp"


def test_a_backstop_without_a_speed_test_or_a_source_read_naming_nothing_is_refused() -> None:
    """Flips if a backstop is accepted with no speed test, or a source change read with no stamps, ids or unit, or
    a unit other than a table."""
    with pytest.raises(ValueError, match="speed test"):
        security_backstop("a missed revocation stays open", proof="tests/x.py::test_y", speed="tests/x.py")
    with pytest.raises(ValueError, match="stamps, its ids"):
        source_change_read(proof="tests/x.py::test_y")
    with pytest.raises(ValueError, match="stamps, its ids"):
        source_change_read(unit="row", proof="tests/x.py::test_y")
