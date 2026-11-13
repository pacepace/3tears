"""What reading a figure key twice on one report input holds to: the same answer, and nothing kept past the input.

Contracts held here (``threetears.evals.analysis.report.figure_keys.resolve``, which keeps each input's figures and
refusals once read, and ``PageForms.of`` and ``page_arm_names``, which keep each input's forms and arm names on the
same values kept per input):

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
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.keys.names import page_arm_names
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
    for step in range(100):
        document["measures"][key]["per_arm"][arm]["mean"] = 10.0 + step / 8
        held = ReportInput.model_validate(document)
        reused += id(held) in seen
        seen.add(id(held))
        assert resolve(path, held).value == 10.0 + step / 8, step
        del held
    assert reused, "the control: some input was made where a dropped one had been"


def test_an_input_whose_forms_and_arm_names_were_read_is_freed_once_dropped() -> None:
    held = _copy_of_r2()
    PageForms.of(held)
    page_arm_names(held)
    gone = weakref.ref(held)
    del held
    gc.collect()
    assert gone() is None, "a page's forms or arm names keep their report input alive"


def test_an_input_made_where_a_dropped_one_was_reads_its_own_forms_and_arm_names() -> None:
    document = fixture_inputs()["r2"].model_dump(mode="json", by_alias=True)
    seen: set[int] = set()
    reused = 0
    for step in range(100):
        document["measures"]["calls_per_task"]["unit"] = f"calls-{step}"
        document["arms"][0]["short_name"] = f"first-{step}"
        held = ReportInput.model_validate(document)
        reused += id(held) in seen
        seen.add(id(held))
        assert PageForms.of(held).measure("calls_per_task").unit == f"calls-{step}", step
        assert page_arm_names(held)[held.arms[0].id] == f"first-{step}", step
        del held
    assert reused, "the control: some input was made where a dropped one had been"


def test_each_page_of_one_input_keeps_its_own_ledger_and_names() -> None:
    held = _copy_of_r2()
    first, second = PageForms.of(held), PageForms.of(held)
    key, arm = _first_mean(held)
    first.level(key, held.measures[key].per_arm[arm].mean)
    assert first.ledger and not second.ledger, "two pages of one input share a ledger"
    names = page_arm_names(held)
    names[held.arms[0].id] = "renamed"
    assert page_arm_names(held)[held.arms[0].id] != "renamed", "a caller's names changed the input's"
