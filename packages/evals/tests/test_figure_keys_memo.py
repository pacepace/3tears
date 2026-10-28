"""What reading a figure key twice on one report input holds to: the same answer, and nothing kept past the input.

Contracts held here (``threetears.evals.analysis.report.figure_keys.resolve``, which keeps each input's figures and
refusals once read):

- **a refusal reads alike every time**: a key refused twice on one input raises the same type (an unstated figure
  apart from a key that names nothing, which a listing reads as absent) with the same message, the same cause and the
  same choice of showing its context;
- **nothing keeps an input alive**: an input that read a figure and a refused key is freed once dropped;
- **an input is its own**: an input made where a dropped one was read reads its own figures, never the dropped one's.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any

from packages.evals.tests.analyst_support import fixture_inputs
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report_input import ReportInput

#: A key r2 leaves unstated, refused with a cause: the arm states no spread on the measure.
_UNSTATED = "spread.atlas-m.off.r0.calls_per_task.overall.se"

#: A key that names nothing on any input.
_NOTHING = "no.such.key"


def _copy_of_r2() -> ReportInput:
    """A fresh copy of r2, read from its document: an input nothing else holds."""
    return ReportInput.model_validate(fixture_inputs()["r2"].model_dump(mode="json", by_alias=True))


def _refusal(key: str, report_input: ReportInput) -> tuple[type[BaseException], str, Any, bool]:
    """What reading a refused key raises: its type, message, cause (type and message) and whether its context shows.
    Read inside this function, so no frame of the read outlives it."""
    try:
        resolve(key, report_input)
    except UnresolvableReference as refused:
        cause = refused.__cause__
        return (
            type(refused),
            str(refused),
            None if cause is None else (type(cause), str(cause)),
            refused.__suppress_context__,
        )
    raise AssertionError(f"{key} resolved")


def _first_mean(report_input: ReportInput) -> tuple[str, str]:
    """The first measure and arm whose mean the input states."""
    return next(
        (key, arm)
        for key, measure in report_input.measures.items()
        for arm, stat in measure.per_arm.items()
        if stat.mean is not None
    )


def test_a_key_refused_twice_raises_the_same_refusal_both_times() -> None:
    held = _copy_of_r2()
    for key, kind in ((_UNSTATED, UnstatedFigure), (_NOTHING, UnresolvableReference)):
        first, second = _refusal(key, held), _refusal(key, held)
        assert first[0] is kind, first
        assert first == second, key
    unstated = _refusal(_UNSTATED, held)
    assert unstated[2] is not None and unstated[3], "the control: the unstated key is refused from a cause"


def test_an_input_that_read_a_refused_key_is_freed_once_dropped() -> None:
    held = _copy_of_r2()
    key, arm = _first_mean(held)
    resolve(f"measure.{key}.{arm}.mean", held)
    for refused in (_UNSTATED, _NOTHING):
        _refusal(refused, held)
    gone = weakref.ref(held)
    del held
    gc.collect()
    assert gone() is None, "a figure key read keeps its report input alive"


def test_an_input_made_where_a_dropped_one_was_reads_its_own_figures() -> None:
    document = fixture_inputs()["r2"].model_dump(mode="json", by_alias=True)
    key, arm = _first_mean(fixture_inputs()["r2"])
    path = f"measure.{key}.{arm}.mean"
    seen: set[int] = set()
    reused = 0
    for step in range(200):
        document["measures"][key]["per_arm"][arm]["mean"] = 10.0 + step / 8
        held = ReportInput.model_validate(document)
        reused += id(held) in seen
        seen.add(id(held))
        assert resolve(path, held).value == 10.0 + step / 8, step
        del held
        gc.collect()
    assert reused, "the control: some input was made where a dropped one had been"
