"""A falsy recorded value on a blank-indeterminate input is a level, not an absence.

``indeterminate_when_blank`` marks an input whose blank means "nobody recorded this". The rule
once read every falsy value as blank, so a host recording ``clean_snapshot=False`` — the snapshot
was NOT clean, an observation — had it stored as undecided, and every finding of the campaign
carried an ``undecided`` apparatus confound the runs never had. Blank is ``None``, ``""`` and an
empty collection; a bool or a number is always a value.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.kernel.host.sweepables import Sweepable, SweepableRegistry

_REGISTRY = SweepableRegistry(
    (
        Sweepable(
            name="clean_snapshot",
            role="apparatus",
            read=lambda _run, _results: None,
            reader_prose="whether the subject started from a clean snapshot",
            confounds="the subject started from a different state, so the runs began from different conversations",
            indeterminate_when_blank=True,
        ),
    )
)


@pytest.mark.parametrize("value", [False, True, 0, 0.0, 3])
def test_a_recorded_bool_or_number_is_never_undecidable(value: Any) -> None:
    assert not _REGISTRY.is_indeterminate("clean_snapshot", value)


@pytest.mark.parametrize("value", [None, "", [], {}, ()])
def test_none_and_empties_are_still_unrecorded(value: Any) -> None:
    assert _REGISTRY.is_indeterminate("clean_snapshot", value)


def test_two_runs_that_both_recorded_false_agree() -> None:
    assert _REGISTRY.comparability("clean_snapshot", [False, False]) == "same"


def test_a_recorded_false_against_a_recorded_true_is_a_difference() -> None:
    assert _REGISTRY.comparability("clean_snapshot", [False, True]) == "differs"


def test_a_recorded_false_against_an_unrecorded_run_cannot_be_decided() -> None:
    assert _REGISTRY.comparability("clean_snapshot", [False, None]) == "unknown"
