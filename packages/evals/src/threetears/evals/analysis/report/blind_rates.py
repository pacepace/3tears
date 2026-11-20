"""A blind read's catch and false-fail rates, said once for every door that states them (#1208).

A closed read stores counts; the host's response (:mod:`threetears.evals.ops.blind_reads`) and the quick path's
console (:mod:`threetears.evals.quick.blind`) each read the rates off those counts here, so a rate, and the reason no
rate exists, is worded in one place. The console names the side being checked before each rate; the host's response
serves each rate under its own field, ``catch`` or ``false_fail``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from threetears.evals.analysis.blind_read import pass_fail_counts, single_verdicts
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.unit_once import interval_alone
from threetears.evals.analysis.stats import exact_rate_interval
from threetears.evals.schema.blind_read import BlindVerdictTable

#: The words a pass/fail score is said in.
_VERDICT_WORDS: Final = {0: "fail", 1: "pass"}

#: Why a 1-5 criterion has neither rate.
_ORDINAL: Final = "a 1-5 criterion has no fixed pass, so no catch rate: the table counts every pair"


@dataclass(frozen=True)
class StatedRate:
    """A rate read off a check's counts: of ``of`` answers, ``count``, with its exact 95% interval and its words."""

    count: int
    of: int
    interval: tuple[float, float]
    said: str


@dataclass(frozen=True)
class BlindRates:
    """A stored check's catch and false-fail rates, and the plain statements beside them."""

    catch: StatedRate | None
    false_fail: StatedRate | None
    no_catch: str | None
    no_false_fail: str | None
    single: tuple[str, ...]

    @property
    def said(self) -> list[str]:
        """The plain statements: a side that gave a single verdict, then a rate that does not exist, with why."""
        return [*self.single, *(line for line in (self.no_catch, self.no_false_fail) if line is not None)]


def _stated_rate(count: int, of: int, said: str) -> StatedRate:
    """``count`` of ``of`` with its exact interval, said as ``said`` followed by the interval, which stands alone and
    so writes its ``%`` once, after its high end (``0 to 52%``)."""
    interval = exact_rate_interval(count, of)
    assert interval is not None  # of is at least 1
    forms = PageForms.off_page()
    low, high = forms.share(interval[0]), forms.share(interval[1])
    ends = interval_alone(low.digits, high.digits, high.unit)
    return StatedRate(count, of, interval, f"{said} (exact 95% interval {ends})")


def blind_rates(table: BlindVerdictTable, *, reader: str, judge: str) -> BlindRates:
    """The catch rate and the false-fail rate read off a stored table, and the plain statements beside them.

    On pass/fail: of the answers ``reader`` failed, how many the judge failed too (the catch rate), and of those
    ``reader`` passed, how many it failed (the false-fail rate), each with its exact (Clopper-Pearson) 95% interval.
    A 1-5 criterion has no fixed pass, so neither rate. A side that gave every answer one verdict is said, since
    chance alone then accounts for every agreement.

    :param table: A stored table of counts.
    :param reader: How the statements name the side read against (``the reader``, ``people``).
    :param judge: How they name the side being checked (``the judge``, ``the reviewer``).
    :return: The rates and the statements.
    """
    one_reader, one_judge = single_verdicts(table)
    cells = pass_fail_counts(table)

    def verdict(score: int) -> str:
        return _VERDICT_WORDS[score] if cells is not None else f"a {score}"

    single = tuple(
        f"{who} gave every answer one verdict, {verdict(one)}"
        for who, one in ((reader, one_reader), (judge, one_judge))
        if one is not None
    )
    if cells is None:
        return BlindRates(None, None, _ORDINAL, None, single)
    failed = cells.both_fail + cells.reader_fail_judge_pass
    passed = cells.both_pass + cells.reader_pass_judge_fail
    catch = (
        _stated_rate(cells.both_fail, failed, f"caught {cells.both_fail} of the {failed} {reader} failed")
        if failed
        else None
    )
    false_fail = (
        _stated_rate(
            cells.reader_pass_judge_fail,
            passed,
            f"failed {cells.reader_pass_judge_fail} of the {passed} {reader} passed",
        )
        if passed
        else None
    )
    return BlindRates(
        catch,
        false_fail,
        None if catch is not None else f"{reader} failed no answer: no catch rate",
        None if false_fail is not None else f"{reader} passed no answer: no false-fail rate",
        single,
    )


__all__ = ["BlindRates", "StatedRate", "blind_rates"]
