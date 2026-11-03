"""What every example under ``examples/`` holds to, read across all of them.

The examples are the ladder: one directory per rung, each run from inside itself against the recordings beside it.
Each rung's own test (``test_rung_one_triage.py``, ``test_rung_two_replies.py``, ``test_invoices_rung.py``,
``test_booking_rung.py``, ``test_rung_five_writer.py``) replays it; what is read here holds across all five. Rung 5's
recordings are a paid step not yet taken, so it holds the command that records them in their place. No test here calls
a model.
"""

from __future__ import annotations

from packages.evals.tests.example_loader import EXAMPLES, RUNGS, UNRECORDED_RUNGS
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations


def test_the_examples_are_the_rungs_and_nothing_beside_them() -> None:
    assert sorted(path.name for path in EXAMPLES.iterdir() if path.is_dir()) == list(RUNGS)
    assert not list(EXAMPLES.glob("*.py")), "every example is a rung's: no single-file example beside them"
    for rung in RUNGS:
        assert list((EXAMPLES / rung).glob("*.py")), f"{rung} holds no Python file"
        if rung in UNRECORDED_RUNGS:
            assert (EXAMPLES / rung / "recordings" / "README.md").is_file(), f"{rung} says nowhere how it is recorded"
        else:
            assert list((EXAMPLES / rung / "recordings").glob("*.jsonl")), f"{rung} holds no recording to replay"


def test_every_example_reaches_the_engine_only_through_public_roots() -> None:
    examples = [(path.relative_to(EXAMPLES).as_posix(), path) for path in sorted(EXAMPLES.rglob("*.py"))]
    assert {name.split("/")[0] for name, _ in examples} == set(RUNGS)
    assert public_root_violations(SOURCE_ROOT, examples, consumer_root=REPO_ROOT) == []
