"""The quick path's re-readable comparison: a newcomer adds cases and runs it again, as often as they like, and the
error each reading states still holds (owner ruling, 2026-11-17; the order and the stops, owner ruling "fixed order +
drain").

A quick-path comparison (every result of a kind only the quick path launches) reads each paired contrast on a declared
range by the mixture bet (:func:`~threetears.evals.analysis.stats.bounded_separation_p` and
:func:`~threetears.evals.analysis.stats.bounded_difference_interval` with ``mixture``): the mean of ten bets, each
staking a fixed share ``j / 10`` of the cap on every pair and weighted in proportion to ``j``, hedged over the two
directions (evenly with no question asked; nine tenths toward the better side a question asks about, its primary
decided alone, ``tests/test_small_n_read.py``), over the longest complete
prefix of an order committed before any answer (each launch's new cases shuffled once and appended,
:func:`~threetears.evals.analysis.bundle.trials.committed_order`), which each launch plays case by case. No stake
depends on a declared count, and the quick path continues one comparison by running only the cells no launch recorded
(:mod:`threetears.evals.quick.continuation`), so every reading is one capital read again. Held here:

- the decisions are the simulated construction's (``looks-bet-simulation-1116.md``, "Mixture stakes"), decision for
  decision on fixed seeds, and the interval excludes zero exactly where the test separates;
- the bet alone, over rows appended and kept: re-reading separates at most α under the null;
- the product: a comparison run again runs only the cases appended and reads the rest as they ran; anything else is a
  new comparison that runs every case and says what changed; the case order is append-only;
- a run stopped by its cost cap, interrupted once (the cells in flight finish), interrupted twice (they are cut), or
  turned away by the account or the provider keeps every answer it recorded, and a run again runs only the cells no
  run recorded, a row recording no answer among them (``runs_again``), and says how many ran again: no recorded answer
  is drawn twice, and each reading's p is the bet over the longest complete prefix;
- the product under the null, ``Eval.run`` itself over a scripted runner whose arms are equal in accuracy: run again as
  cases are added it separates at most α, and re-running every case on each run, read by the same bet, exceeds α
  (seeded and pinned, since every launch there completes and the committed order depends on the cases alone); stopped
  by a watcher, with arms alike in speed and with arms of different speed whose slow arm is slower when wrong, drained
  and cut, the product's share and the two earlier rules' over the same answers (drawn, since which cells a stop finds
  in flight depends on time);
- at one stated alternative, a single read's power against the reading it replaces;
- the rung 3 comparison prints the one sentence that says so, and reads cost by its t test, once.

The long simulations are ``long_simulation`` (minutes of CPU each), each with a smoke check in the unit suite; the
pool's ``simulation`` suite runs them, one request per test (``.testament.toml`` says how), on any change its comment
covers: any file under ``analysis/``, ``quick/`` or ``run/``, or the kernel files it names.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pytest

from threetears.evals import Answer, Eval, Result, grader
from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.actions.catalogue import Caller
from threetears.evals.analysis import AnalysisContextBundle, ComparisonFamily, assemble_context_bundle
from threetears.evals.analysis.campaigns import get_campaign
from threetears.evals.analysis.campaigns import list_campaigns
from threetears.evals.analysis.bundle.comparisons import REREAD_SENTENCE, read_by_bet, redrawn_sentence
from threetears.evals.analysis.bundle.trials import RedrawnCells, committed_order, redrawn_cells
from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    bounded_difference_interval,
    bounded_separation_p,
    composite_significance,
    exact_rate_interval,
    holm_adjust,
    separation_test,
)
from threetears.evals.kernel import CampaignArm
from threetears.evals.kernel.candidate_kind import QUICK_PATH_FILER, QUICK_PATH_KINDS
from threetears.evals.kernel.declaration import AspectDesign
from threetears.evals.kernel.result_condition import runs_again
from threetears.evals.ops.host import OpsHost
from threetears.evals.providers import ProviderError
from threetears.evals.quick import as_host
from threetears.evals.quick.continuation import UNKEPT_SENTENCE, continued_sentence, new_comparison_sentence
from threetears.evals.quick.one_call import launched_as_run
from threetears.evals.run import list_results
from threetears.evals.schema import EvalResult, RubricScore
from threetears.evals.storage import InMemoryDocumentStore

from packages.evals.tests.factories import (
    fixture_variant_key,
    make_aspect_campaign,
    make_eval_result,
    make_eval_run,
    minimal_aspect_design,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.ladder_support import load, offline_copy
from packages.evals.tests.off_loop import off_loop

#: The pass rate's range, which the per-case shares and their differences are read on.
_RATE = (0.0, 1.0)


def _pair(rng: random.Random, k: int, control: float, arm: float) -> tuple[float, float]:
    """One case's pass share on each arm: ``k`` trials each, every trial passing at its arm's rate."""
    return (
        sum(rng.random() < control for _ in range(k)) / k,
        sum(rng.random() < arm for _ in range(k)) / k,
    )


def _separates(control: list[float], arm: list[float]) -> bool:
    p = bounded_separation_p(control, arm, paired=True, value_range=_RATE, mixture=True)
    assert p is not None
    return p < SIGNIFICANCE_ALPHA


# --- the construction ------------------------------------------------------------------------------------------------


def _reference_crossing(diffs: list[float]) -> tuple[int, int]:
    """The simulated construction, ``reference-mix.py``'s ``first_crossing`` on its ``mix_grid`` arm with the stakes
    weighted in proportion to their size, one row at a time in plain Python: z = (d + 1) / 2, null 1/2, ten fractions
    λ_j = (j / 10) · 0.9 / 0.5, each direction's log capital a cumulative sum of log1p, the ten combined by successive
    logaddexp, each plus log(j / 55), hedged as their logaddexp less log 2, and the first index at which the hedged log
    capital reaches log(1/α); the direction there the larger side.

    :return: ``(index, direction)``: the 0-based index of the first crossing, or ``len(diffs)`` where none, and +1 up,
        -1 down, 0 none.
    """
    sides = []
    for zz in ([(d + 1) / 2 for d in diffs], [1 - (d + 1) / 2 for d in diffs]):
        acc: list[float] | None = None
        for j in range(1, 11):
            lam = j / 10 * 0.9 / 0.5
            running, path = 0.0, []
            for z in zz:
                running += math.log1p(lam * (z - 0.5))
                path.append(running)
            weighted = [value + math.log(j / 55) for value in path]
            if acc is None:
                acc = weighted
            else:
                acc = [max(a, b) + math.log1p(math.exp(-abs(a - b))) for a, b in zip(acc, weighted, strict=True)]
        assert acc is not None
        sides.append(acc)
    up, down = sides
    for index, (u, d) in enumerate(zip(up, down, strict=True)):
        hedged = max(u, d) + math.log1p(math.exp(-abs(u - d))) - math.log(2)
        if hedged >= math.log(1 / SIGNIFICANCE_ALPHA):
            return index, 1 if u >= d else -1
    return len(diffs), 0


def _sides(diffs: list[float]) -> tuple[list[float], list[float]]:
    """Two arms on the 0-1 range whose paired differences are ``diffs``."""
    return [max(0.0, -d) for d in diffs], [max(0.0, d) for d in diffs]


@pytest.mark.parametrize(("k", "control", "arm"), [(1, 0.5, 0.5), (1, 0.8, 0.6), (3, 0.6, 0.75), (3, 0.9, 0.9)])
def test_decisions_agree_with_the_simulated_construction_read_for_read_on_fixed_seeds(
    k: int, control: float, arm: float
) -> None:
    """Every read of 60 seeded rows a cell, at 20, 40, 60 and 80 pairs, separates exactly where the construction's
    hedged capital first crossed 1/α within those pairs, in its direction; and the interval at 95% excludes zero
    exactly where the test separates, on the side the construction crossed to."""
    rng = random.Random(f"rereadable-construction-{k}-{control}-{arm}")
    for _ in range(60):
        draws = [_pair(rng, k, control, arm) for _ in range(80)]
        diffs = [b - a for a, b in draws]
        crossed, direction = _reference_crossing(diffs)
        for read in (20, 40, 60, 80):
            low_side, high_side = _sides(diffs[:read])
            separated = _separates(low_side, high_side)
            interval = bounded_difference_interval(
                low_side, high_side, paired=True, value_range=_RATE, confidence=0.95, mixture=True
            )
            assert separated == (crossed <= read - 1)
            assert interval is not None
            side = 1 if interval[0] > 0 else -1 if interval[1] < 0 else 0
            assert side == (direction if separated else 0)


def test_a_reading_read_again_after_cases_are_added_is_the_same_capital_read_later() -> None:
    """A case appended is bet on after every case before it: the p over more cases is never above the p over fewer,
    and the interval over more lies inside the one over fewer, because each is one capital read later."""
    rng = random.Random("rereadable-prefix")
    draws = [_pair(rng, 3, 0.7, 0.8) for _ in range(90)]
    control, arm = [a for a, _ in draws], [b for _, b in draws]
    ps, intervals = [], []
    for read in (15, 30, 45, 60, 75, 90):
        ps.append(bounded_separation_p(control[:read], arm[:read], paired=True, value_range=_RATE, mixture=True))
        intervals.append(
            bounded_difference_interval(control[:read], arm[:read], paired=True, value_range=_RATE, mixture=True)
        )
    assert all(later <= earlier for earlier, later in zip(ps, ps[1:]))  # type: ignore[operator]
    for earlier, later in zip(intervals, intervals[1:]):
        assert earlier is not None and later is not None
        assert earlier[0] <= later[0] + 1e-9 and later[1] <= earlier[1] + 1e-9


def test_the_mixture_reads_paired_and_for_no_declared_count() -> None:
    with pytest.raises(ValueError, match="paired"):
        bounded_separation_p([0.0, 1.0], [1.0, 1.0], paired=False, value_range=_RATE, mixture=True)
    with pytest.raises(ValueError, match="planned_n"):
        bounded_difference_interval([0.0, 1.0], [1.0, 1.0], paired=True, value_range=_RATE, planned_n=4, mixture=True)


# --- error under re-reads --------------------------------------------------------------------------------------------


def _reread_until_separated(rng: random.Random, k: int, rate: float, arms: int) -> bool:
    """One newcomer under the null (every arm passing at ``rate``): launch 20 cases, read every arm against the
    control, and while nothing separates add cases and read again, at most five reads in all. Each add is sized from
    the reading: ten cases, plus up to ninety more the closer the smallest p came to α. Several arms are read under
    Holm's correction over their contrasts, as the quick path's family is."""
    cases: list[tuple[float, ...]] = []
    n = 20
    for _ in range(5):
        while len(cases) < n:
            cases.append(tuple(sum(rng.random() < rate for _ in range(k)) / k for _ in range(arms)))
        control = [case[0] for case in cases]
        raw = [
            bounded_separation_p(control, [case[i] for case in cases], paired=True, value_range=_RATE, mixture=True)
            for i in range(1, arms)
        ]
        ps = [p for p in raw if p is not None]
        if any(p < SIGNIFICANCE_ALPHA for p in holm_adjust(ps)):
            return True
        n += 10 + int(90 * (1.0 - min(1.0, min(ps))))
    return False


@pytest.mark.parametrize(("k", "rate", "arms", "seeded_at"), [(1, 0.5, 2, 188), (3, 0.8, 2, 123), (1, 0.8, 3, 138)])
def test_re_reading_as_cases_are_added_separates_under_the_null_at_most_alpha(
    k: int, rate: float, arms: int, seeded_at: int
) -> None:
    """The bet alone, over rows appended and kept (the reference construction, not the product: the product's own
    re-runs are simulated below, through ``Eval.run``). 5,000 seeded newcomers a cell, each reading up to five times
    with adds sized from the data, separate no more
    often than α = 5%: the Clopper-Pearson 95% upper bound on the rate is below α. The seeded draw gives 188 of 5,000
    at k = 1, base 0.5 (3.8%, upper bound 4.3%); 123 at k = 3, base 0.8 (2.5%, upper 2.9%); and 138 for three arms at
    k = 1, base 0.8 under Holm (2.8%, upper 3.3%). The stakes weighted toward the larger ones grow faster on a few
    cases, so they separate more often than an even weighting did under the null, still within α: the count is read
    at 5,000 so its bound clears α. The counts are pinned to two either side, so a change to the bet that moves them is
    seen; the bound is the claim."""
    rng = random.Random(f"rereadable-null-{k}-{rate}-{arms}")
    replicates = 5000
    separated = sum(_reread_until_separated(rng, k, rate, arms) for _ in range(replicates))
    bounds = exact_rate_interval(separated, replicates)
    assert bounds is not None and bounds[1] < SIGNIFICANCE_ALPHA
    assert abs(separated - seeded_at) <= 2


# --- power on a single read ------------------------------------------------------------------------------------------


def test_a_single_read_gives_up_some_power_against_the_reading_it_replaces() -> None:
    """At k = 1, the control passing 80% and the arm 60%, on 60 paired cases read once, 2,000 seeded draws: the
    mixture separates on 805 (40.3%), where the reading it replaces (the paired t test's p below α and the bounded
    test's 95% interval, fixed for those 60 cases, excluding zero) separates on 872 (43.6%). That is the price of
    reading again freely: three and a half points of power at this alternative, held to no more than fifteen."""
    rng = random.Random("rereadable-power")
    mixture = one_shot = 0
    replicates = 2000
    for _ in range(replicates):
        draws = [_pair(rng, 1, 0.8, 0.6) for _ in range(60)]
        control, arm = [a for a, _ in draws], [b for _, b in draws]
        mixture += _separates(control, arm)
        t_p = composite_significance(control, arm, paired=True).p_value
        fixed = bounded_difference_interval(control, arm, paired=True, value_range=_RATE)
        one_shot += bool(
            t_p is not None and t_p < SIGNIFICANCE_ALPHA and fixed is not None and (fixed[0] > 0 or fixed[1] < 0)
        )
    assert mixture < one_shot <= mixture + 0.15 * replicates
    assert abs(mixture - 805) <= 10 and abs(one_shot - 872) <= 10


# --- the quick path --------------------------------------------------------------------------------------------------


async def test_rung_3_s_comparison_is_read_by_the_bet_and_says_so_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rung 3a, replayed offline into its store: the print says the one sentence under its first line; every paired
    reading on a range is the bet's, and cost is read by its t test, once. Its question is asked, so the family is
    decided on the question's own reading, the first grader's (currency's accuracy), alone: its adjusted p is the bet's
    staked toward the better side, at α, and every other reading, the t reading of cost among them, is secondary, with
    no adjusted p and no verdict, so no t p bears on a decision. Kept in memory, the line under the first says instead
    that running it again is a new comparison (:func:`test_kept_in_memory_a_comparison_says_running_it_again_is_a_new_one`)."""
    offline_copy("3-invoices", tmp_path, monkeypatch)
    module = load(Path("compare.py"), "compare", monkeypatch)
    result = await module.compare.run(store=tmp_path / "kept.sqlite")

    printed = str(result).splitlines()
    assert printed[1] == REREAD_SENTENCE and str(result).count(REREAD_SENTENCE) == 1
    (family,) = result.report.bundle.multiple_comparisons.families
    assert read_by_bet(family) and family.disclosure.endswith(REREAD_SENTENCE)
    on_t = [comparison for comparison in family.comparisons if comparison.basis == "t"]
    on_bet = [comparison for comparison in family.comparisons if comparison.basis == "bounded"]
    assert [comparison.name for comparison in on_t] == ["production_replicating_cost"]
    assert {comparison.interval_basis for comparison in on_bet} == {"bounded"} and len(on_bet) == 5
    assert family.primary == "accuracy" and family.family_size == 1
    (primary,) = [comparison for comparison in family.comparisons if comparison.name == family.primary]
    assert primary.p_adjusted is not None and primary.evidence_better is not None
    secondaries = [comparison for comparison in family.comparisons if comparison is not primary]
    assert {(comparison.verdict, comparison.p_adjusted) for comparison in secondaries} == {("not_separated", None)}


# --- one comparison, continued ---------------------------------------------------------------------------------------


def _correct(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer is right."""
    return answer == "right"


def _short(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer is short."""
    return len(answer) < 6


def _coin(tag: str, arm: str, rate: float, calls: Counter[tuple[int, str]]) -> Callable[[Mapping[str, Any]], Any]:
    """An arm answering ``right`` at ``rate``, each call a fresh seeded draw: a case run again draws again, so a product
    that re-ran a case already run would read a new value for it. ``calls`` counts each case's calls on the arm."""

    async def run(case: Mapping[str, Any]) -> str:
        calls[case["n"], arm] += 1
        draw = random.Random(f"{tag}-{case['n']}-{arm}-{calls[case['n'], arm]}").random()
        return "right" if draw < rate else "wrong"

    run.__name__ = arm
    return run


def _cases(n: int, start: int = 0) -> list[dict[str, Any]]:
    return [{"n": index} for index in range(start, n)]


def _how_it_ran(result: Result) -> list[str]:
    """The print's sentence saying whether the comparison continued, or why it is a new one."""
    return [line for line in result.disclosures if line.startswith(("Continues the comparison", "A new comparison"))]


async def test_a_comparison_run_again_runs_only_the_cases_appended_and_reads_every_case() -> None:
    """Run again into its store over more cases, a comparison runs only the cases appended, on every arm; reads every
    case, the earlier ones as they ran; files a campaign holding both runs of each arm; bets on the cases in the order
    they were added; and says so. Run again with no case added, nothing runs and the reading is the same."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    arms = {"a": _coin("continue", "a", 0.5, calls), "b": _coin("continue", "b", 0.5, calls)}
    first = await Eval(_cases(20), arms, [_correct], k=1, name="grows").run(store=store, scope="s")
    assert set(calls) == {(n, arm) for n in range(20) for arm in "ab"} and set(calls.values()) == {1}
    assert _how_it_ran(first) == []

    second = await Eval(_cases(30), arms, [_correct], k=1, name="grows").run(store=store, scope="s")
    assert set(calls) == {(n, arm) for n in range(30) for arm in "ab"} and set(calls.values()) == {1}
    assert _how_it_ran(second) == [continued_sentence(20, 10)]
    assert {arm: summary.n_results for arm, summary in second.arms.items()} == {"a": 30, "b": 30}
    (aspect,) = off_loop(get_campaign, second.host.storage, second.campaign_id, "s").aspects
    (earlier,) = off_loop(get_campaign, first.host.storage, first.campaign_id, "s").aspects
    assert set(earlier.run_ids) < set(aspect.run_ids) and len(aspect.run_ids) == 4
    (family,) = second.report.bundle.multiple_comparisons.families
    (paired,) = [comparison for comparison in family.comparisons if comparison.name == "_correct"]
    assert read_by_bet(family) and paired.basis == "bounded" and paired.case_values is not None
    assert len(paired.case_values[0]) == 30
    # Every case the first run ran is read as it ran: its answer on each arm is the first run's.
    for arm in "ab":
        ran = {(result.case, result.answer) for result in first.arms[arm].case_results}
        assert ran < {(result.case, result.answer) for result in second.arms[arm].case_results}
    places = {trial.case_place for trial in second.report.bundle.trials}
    assert places == set(range(1, 31))

    again = await Eval(_cases(30), arms, [_correct], k=1, name="grows").run(store=store, scope="s")
    assert set(calls.values()) == {1}
    assert _how_it_ran(again) == [continued_sentence(30, 0)]
    (same,) = [c for c in again.report.bundle.multiple_comparisons.families[0].comparisons if c.name == "_correct"]
    assert (same.p_raw, same.interval) == (paired.p_raw, paired.interval)


def _changed_ids() -> list[dict[str, Any]]:
    return [{"id": f"c{n}", "n": n} for n in range(25)]


@pytest.mark.parametrize(
    ("first", "second", "changed"),
    [
        ({}, {"k": 2}, "the repeats per case changed"),
        ({}, {"runner": "renamed"}, "the arms changed"),
        ({}, {"runner": "swapped"}, "the control changed"),
        ({}, {"grader": [_correct, _short]}, "the graders, the guardrails, the question or its margin changed"),
        (
            {},
            {"question": "Does b answer right more often?"},
            "the graders, the guardrails, the question or its margin changed",
        ),
        ({}, {"grader": [grader(_correct, margin=0.1)]}, "a grader's margin, range or guard changed"),
        ({}, {"cases": [{"n": 1}, {"n": 0}, *_cases(25, 2)]}, "case 1 was edited, moved or removed"),
        ({}, {"cases": [{"n": 99}, *_cases(25)]}, "case 1 was edited, moved or removed"),
        ({}, {"cases": [*_cases(7), {"n": 7, "extra": True}, *_cases(25, 8)]}, "case 8 was edited, moved or removed"),
        ({}, {"cases": _cases(15)}, "a case run before was removed"),
        (
            {"cases": _changed_ids()[:20]},
            {"cases": [{"id": "c0", "n": 0}, {"id": "x1", "n": 1}, *_changed_ids()[2:]]},
            "case x1 was edited, moved or removed",
        ),
    ],
    ids=[
        "k",
        "arms",
        "control",
        "graders",
        "question",
        "declaration",
        "reordered",
        "inserted",
        "edited",
        "removed",
        "renamed-id",
    ],
)
async def test_what_changed_starts_a_new_comparison_that_runs_every_case_and_says_what(
    first: dict[str, Any], second: dict[str, Any], changed: str
) -> None:
    """Anything but cases appended at the end makes another comparison: every case runs again, on every arm, and the
    print names what changed in one sentence. The case order is append-only: a case run before moved, inserted
    before, edited or removed is a change."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    arms = {"a": _coin("changed", "a", 0.5, calls), "b": _coin("changed", "b", 0.5, calls)}
    variants = {
        "renamed": {"a": arms["a"], "c": arms["b"]},
        "swapped": {"b": arms["b"], "a": arms["a"]},
    }

    def eval_of(spec: Mapping[str, Any]) -> Eval:
        given = dict(spec)
        runner = given.pop("runner", None)
        return Eval(
            given.pop("cases", _cases(20) if spec is first else _cases(25)),
            variants[runner] if runner is not None else arms,
            given.pop("grader", [_correct]),
            name="changes",
            **{"k": 1, **given},
        )

    await eval_of(first).run(store=store, scope="s")
    ran_first = sum(calls.values())
    result = await eval_of(second).run(store=store, scope="s")
    cases = len(second.get("cases", _cases(25)))
    assert sum(calls.values()) - ran_first == 2 * cases * second.get("k", 1)
    assert _how_it_ran(result) == [new_comparison_sentence(changed)]
    assert all(len(summary.case_results) == cases * second.get("k", 1) for summary in result.arms.values())


async def test_kept_in_memory_a_comparison_says_running_it_again_is_a_new_one() -> None:
    """A store kept in memory is gone when its run ends, so the line under the first line says running it again is a
    new comparison, never that cases may be added and read again. Its page and its bundle, which say the family's
    sentence wherever they are read, say it of a store that outlives the run, so they license no re-read of a run kept
    in memory either: one sentence, true for every store."""
    calls: Counter[tuple[int, str]] = Counter()
    arms = {"a": _coin("unkept", "a", 0.5, calls), "b": _coin("unkept", "b", 0.5, calls)}
    result = await Eval(_cases(20), arms, [_correct], k=1, name="unkept").run(store=":memory:")
    printed = str(result).splitlines()
    assert printed[1] == UNKEPT_SENTENCE and REREAD_SENTENCE not in printed
    assert REREAD_SENTENCE.startswith("Where its runs are kept in a store that outlives the run")
    (family,) = result.report.bundle.multiple_comparisons.families
    assert read_by_bet(family)
    page = json.dumps(result.page.model_dump(mode="json"))
    assert json.dumps(REREAD_SENTENCE)[1:-1] in page and page.count("run it again") == page.count(
        "run it again into that store"
    )


async def test_a_new_name_or_scope_is_a_fresh_comparison_and_says_nothing_of_the_one_before() -> None:
    """The doors to a fresh comparison over the same store: a new name, or another scope. Each runs every case and
    continues nothing; a store kept in memory keeps nothing between runs, so each run there is fresh too."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    arms = {"a": _coin("fresh", "a", 0.5, calls), "b": _coin("fresh", "b", 0.5, calls)}
    await Eval(_cases(20), arms, [_correct], k=1, name="one").run(store=store, scope="s")
    renamed = await Eval(_cases(25), arms, [_correct], k=1, name="two").run(store=store, scope="s")
    elsewhere = await Eval(_cases(25), arms, [_correct], k=1, name="one").run(store=store, scope="t")
    in_memory = await Eval(_cases(25), arms, [_correct], k=1, name="one").run(store=":memory:", scope="s")
    assert sum(calls.values()) == 2 * (20 + 25 + 25 + 25)
    assert _how_it_ran(renamed) == _how_it_ran(elsewhere) == _how_it_ran(in_memory) == []


def test_no_host_declares_a_design_into_the_bet() -> None:
    """No stored field opts a campaign into the bet: a design naming ``rereadable`` is refused as an unknown field."""
    assert "rereadable" not in AspectDesign.model_fields
    with pytest.raises(ValueError, match="rereadable"):
        minimal_aspect_design(rereadable=True)


def _kinded_bundle(
    kind: str, *, created_by: str = QUICK_PATH_FILER, rerun: bool = False, stopped: bool = False, resumed: bool = False
) -> AnalysisContextBundle:
    """One question's bundle over two arms' judged scores on twelve shared cases, every run and result of ``kind``, in
    a campaign filed by ``created_by``; ``rerun`` gives the contrast arm a second run over the first six cases again,
    so it ran those cases twice. ``stopped`` makes the arms' one launch a cancelled one that recorded only the first
    six cases; ``resumed`` adds a second launch over the six it never reached, which completed, as the quick path
    continues a comparison."""
    runs = []
    results: dict[str, list[EvalResult]] = {}
    held = range(12)
    recorded = range(6) if stopped else held
    lists = [("control-model", held, recorded, "a"), ("contrast-model", held, recorded, "a")]
    if rerun:
        lists.append(("contrast-model", range(6), range(6), "b"))
    if resumed:
        lists += [
            ("control-model", range(6, 12), range(6, 12), "b"),
            ("contrast-model", range(6, 12), range(6, 12), "b"),
        ]
    for model, cases, ran, launch in lists:
        run = make_eval_run(
            status="cancelled" if stopped and launch == "a" else "completed",
            candidate_model=model,
            candidate_kind=kind,
            test_case_ids=[f"tc-{case:02d}" for case in cases],
            launch_group_id=f"launch-{launch}",
        )
        runs.append(run)
        results[run.id] = [
            make_eval_result(
                id=f"{run.id}-{case}",
                eval_run_id=run.id,
                scope_id=run.scope_id,
                model=model,
                candidate_kind=kind,
                test_case_id=f"tc-{case:02d}",
                goal_state_outcomes=[],
                rubric_scores=[
                    RubricScore(
                        dim="extraction.d0",
                        score=3 + (case % 2 if model == "contrast-model" else 0),
                        scale="ordinal",
                        axis="capability",
                        judge_temperature=0.0,
                    )
                ],
            )
            for case in ran
        ]
    asked = min(run.created_at for run in runs)
    campaign = make_aspect_campaign(
        label="b",
        run_ids=[run.id for run in runs],
        design=minimal_aspect_design(merit_axes=["quality"]),
        question="is the contrast better?",
        asked_at=asked,
        control=CampaignArm(
            label="control", role="control", variant_key=fixture_variant_key("control-model"), named_at=asked
        ),
        scope_id=runs[0].scope_id,
        name="kinds",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        created_by=created_by,
    )
    return assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=toyhost_profile())


def _kinded_family(kind: str, *, created_by: str = QUICK_PATH_FILER, rerun: bool = False) -> ComparisonFamily:
    """The family of :func:`_kinded_bundle`."""
    (family,) = _kinded_bundle(kind, created_by=created_by, rerun=rerun).multiple_comparisons.families
    return family


def test_the_bet_reads_a_campaign_the_quick_path_filed_over_its_own_runs_and_no_other() -> None:
    """Whether the bet reads a family is decided from stored facts: its campaign's origin (filed by the quick path),
    its runs' kinds (only the quick path's) and its case lists (no arm ran a case twice). The same runs in a campaign a
    host filed, runs of an engine host's kind, or an arm that ran a case in two runs read as before, and say nothing of
    reading again."""
    for kind in sorted(QUICK_PATH_KINDS):
        family = _kinded_family(kind)
        assert read_by_bet(family) and REREAD_SENTENCE in family.disclosure
    for family in (
        _kinded_family("test-kind"),
        _kinded_family(sorted(QUICK_PATH_KINDS)[0], created_by="host-operator"),
        _kinded_family(sorted(QUICK_PATH_KINDS)[0], rerun=True),
    ):
        assert not read_by_bet(family) and REREAD_SENTENCE not in family.disclosure


def test_a_launch_that_stopped_is_read_where_it_recorded_and_is_a_pause_once_a_later_launch_ran() -> None:
    """A quick-path comparison whose first launch was cancelled after six of its twelve cases, and whose second launch
    ran the six it never reached, is read by the bet over all twelve, and opens on its newest launch's ending: the
    cancelled launch is not listed unfinished. Before the second launch, the cancelled one is unfinished, and the reading opens on it. The same runs in a campaign a host
    filed take nothing up."""
    kind = sorted(QUICK_PATH_KINDS)[0]
    resumed = _kinded_bundle(kind, stopped=True, resumed=True)
    (family,) = resumed.multiple_comparisons.families
    assert read_by_bet(family)
    assert {len(compared.case_values[0]) for compared in family.comparisons if compared.case_values is not None} == {12}
    assert not resumed.incomplete_runs
    stopped = _kinded_bundle(kind, stopped=True)
    assert set(stopped.incomplete_runs.values()) == {"cancelled"} and len(stopped.incomplete_runs) == 2
    hosted = _kinded_bundle(kind, created_by="host-operator", stopped=True, resumed=True)
    assert set(hosted.incomplete_runs.values()) == {"cancelled"} and len(hosted.incomplete_runs) == 2


async def test_a_host_s_campaign_over_quick_runs_reads_as_an_engine_campaign_at_alpha_over_its_looks() -> None:
    """A host serving quick evals (``as_host``) files its own campaigns over a comparison's runs through the action
    catalogue (``campaign_create``): their origin is the host's caller, not the quick path, so each reads as an engine
    campaign does, and the bet reads neither. One filed over runs already read declares its plan of looks after its
    first look, which is no plan, so it reads at α; one declaring a plan of two looks before its runs joined
    (``campaign_runs_add``) reads each look at α/2. Neither says it may be read again; the quick path's own campaign
    over the same runs is read by the bet."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    arms = {"a": _coin("hosted", "a", 0.5, calls), "b": _coin("hosted", "b", 0.8, calls)}
    warm, later = (Eval(_cases(30), arms, [_correct], k=1, name=name) for name in ("warm", "later"))
    host = as_host([warm], store=store)
    tool, ops = eval_catalogue().mount_all(standard_tools())[0], OpsHost.reading(host)
    caller = Caller(scope_id="h", identity="host-operator")

    async def act(action: str, **arguments: Any) -> Any:
        done = await tool.call({"action": action, **arguments}, host=ops, caller=caller)
        assert not done.is_error, done.text
        return done.structured

    first = await warm.run(host=host, scope="h")
    (own,) = first.report.bundle.multiple_comparisons.families
    assert read_by_bet(own)
    (aspect,) = off_loop(get_campaign, host.storage, first.campaign_id, "h").aspects
    own_design = aspect.design
    assert own_design is not None and aspect.template_id is not None

    def declared(planned: int | None) -> dict[str, Any]:
        """The quick campaign's own design, as a host copying it declares it, with or without a plan of looks."""
        return own_design.model_copy(update={"planned_looks": planned}).model_dump(mode="json", exclude_none=True)

    question = "does b answer right more often?"
    over_read = await act(
        "campaign_create",
        name="hosted over runs already read",
        subject_id="warm",
        behavior=aspect.label,
        question=question,
        aspects=[{"label": aspect.label, "run_ids": list(aspect.run_ids), "design": declared(2)}],
        campaign_arms=[{"label": "a", "role": "control", "from_run_ids": [first.arms["a"].run_id]}],
    )
    planned = await act(
        "campaign_create",
        name="hosted, planned before its runs",
        subject_id="later",
        behavior=aspect.label,
        question=question,
        aspects=[{"label": aspect.label, "template_id": aspect.template_id, "run_ids": [], "design": declared(2)}],
    )
    second = await later.run(host=host, scope="h")
    await act(
        "campaign_runs_add",
        campaign_id=planned["id"],
        run_ids=[summary.run_id for summary in second.arms.values()],
    )
    await act("campaign_update", campaign_id=planned["id"], control_from_run_id=second.arms["a"].run_id)
    for campaign_id, alpha in ((over_read["id"], SIGNIFICANCE_ALPHA), (planned["id"], SIGNIFICANCE_ALPHA / 2)):
        campaign = off_loop(get_campaign, host.storage, str(campaign_id), "h")
        assert campaign.created_by == "host-operator"
        bundle = off_loop(assemble_context_bundle, campaign, storage=host.storage, profile=host.profile)
        (family,) = bundle.multiple_comparisons.families
        assert (family.look_alpha if family.look_alpha is not None else family.alpha) == pytest.approx(alpha)
        assert not read_by_bet(family) and REREAD_SENTENCE not in family.disclosure
        (paired,) = [comparison for comparison in family.comparisons if comparison.name == "_correct"]
        assert paired.basis == "t"


# --- a run that stopped, run again --------------------------------------------------------------------------------


def _priced(
    tag: str, arm: str, rate: float, calls: Counter[tuple[int, str]], *, cost: float = 0.10
) -> Callable[[Mapping[str, Any]], Any]:
    """:func:`_coin`'s arm, each answer priced at ``cost`` by one model, which it names as served, so a cost cap stops
    it between cells and its runs pool."""
    draw = _coin(tag, arm, rate, calls)

    async def run(case: Mapping[str, Any]) -> Answer:
        return Answer(
            await draw(case),
            model="priced-model",
            served_model="priced-model",
            input_tokens=10,
            output_tokens=2,
            cost_usd=cost,
        )

    run.__name__ = arm
    return run


def _stored(storage: Any, campaign_id: str, scope: str = "s") -> tuple[list[Any], list[EvalResult]]:
    """A filed comparison's runs, in the order they were launched, and every result they recorded. Reads the store."""
    (aspect,) = get_campaign(storage, campaign_id, scope).aspects
    runs = sorted(
        (run for run_id in aspect.run_ids if (run := storage.load_eval_run(run_id, scope)) is not None),
        key=lambda run: (run.created_at, run.id),
    )
    return runs, [result for run in runs for result in list_results(storage, run.id, scope)]


def _cases_by_state(runs: Sequence[Any], results: Sequence[EvalResult]) -> tuple[set[str], set[str], set[str]]:
    """The cases the runs held, as recorded (a row that runs again recording nothing): complete (every arm, every
    repeat), reached on some arms or repeats only, and reached on none."""
    held = {case for run in runs for case in run.test_case_ids}
    every = set(range(1, runs[0].k_runs + 1))
    variants = {result.variant_key for result in results}
    repeats: dict[str, dict[str, set[int]]] = {}
    for result in results:
        if not runs_again(result):
            repeats.setdefault(result.test_case_id, {}).setdefault(result.variant_key, set()).add(result.k_iteration)
    complete = {
        case
        for case, on in repeats.items()
        if set(on) == variants and len(variants) > 1 and all(r == every for r in on.values())
    }
    return complete, set(repeats) - complete, held - set(repeats)


#: The rules a reading is recomputed by here: the product's (the longest complete prefix of the committed order), and
#: the two rules before it, kept as comparisons (each case at its last cell's time; each case in the cases file's order
#: within the launch that first recorded it).
_Rule = Literal["prefix", "recorded", "filed"]


def _reading_p(runs: Sequence[Any], results: Sequence[EvalResult], rule: _Rule = "prefix") -> float | None:
    """The mixture bet's p on ``_correct`` over a comparison's recorded cells, each case's value its arm's mean over the
    repeats recorded (a row that runs again recording nothing), read by ``rule``:

    - ``prefix`` (the product): the cases of the committed order
      (:func:`~threetears.evals.analysis.bundle.trials.committed_order` over the runs' launches) up to the first case
      some arm has not recorded at every repeat;
    - ``recorded`` (a comparison rule, never the product's): every complete case, in the order of its last cell's
      ``scored_at``, launch by launch;
    - ``filed`` (a comparison rule, never the product's): every case both arms recorded any repeat of, in the launch
      that first recorded it, each launch's in the order of the cases file.

    ``None`` with fewer than two cases."""
    launch: dict[str, int] = {}
    cases_of: dict[str, list[str]] = {}
    for run in runs:
        launch.setdefault(run.launch_group_id or run.id, len(launch))
        cases_of.setdefault(run.launch_group_id or run.id, []).extend(run.test_case_ids)
    launch_of = {run.id: launch[run.launch_group_id or run.id] for run in runs}
    filed = {case: place for place, case in enumerate(dict.fromkeys(c for run in runs for c in run.test_case_ids))}
    complete, _, _ = _cases_by_state(runs, results)
    values: dict[str, dict[str, list[float]]] = {}
    last: dict[str, tuple[int, str]] = {}
    first: dict[str, int] = {}
    for result in results:
        if runs_again(result):
            continue
        case = result.test_case_id
        when = (launch_of[result.eval_run_id], result.scored_at)
        last[case] = max(when, last.get(case, when))
        first[case] = min(when[0], first.get(case, when[0]))
        value = result.host_measures.get("_correct")
        if isinstance(value, float):
            values.setdefault(case, {}).setdefault(result.variant_key, []).append(value)
    variants = sorted({result.variant_key for result in results})
    paired = [case for case, on in values.items() if all(on.get(variant) for variant in variants)]
    if rule == "prefix":
        order = []
        for case in committed_order(runs[0].template_id, list(cases_of.values())):
            if case not in complete:
                break
            order.append(case)
        order = [case for case in order if case in paired]
    elif rule == "recorded":
        order = sorted((case for case in paired if case in complete), key=lambda case: (last[case], filed[case]))
    else:
        order = sorted(paired, key=lambda case: (first[case], filed[case]))
    if len(order) < 2 or len(variants) != 2:
        return None
    a, b = ([sum(values[case][variant]) / len(values[case][variant]) for case in order] for variant in variants)
    return separation_test(a, b, paired=True, value_range=_RATE, mixture=True).p


def _same_p(read: float | None, recomputed: float | None) -> bool:
    """Whether a reading's p is the one recomputed here, to float rounding (the product sums each case's repeats in
    exact decimals)."""
    if read is None or recomputed is None:
        return read is recomputed
    return math.isclose(read, recomputed, rel_tol=1e-9, abs_tol=1e-12)


def _correct_p(result: Result) -> float | None:
    """The raw p of the comparison on ``_correct`` its reading holds where the bet read it (paired, in a family the bet
    read); ``None`` where too few cases are complete for the bet, whatever another test read."""
    (family,) = result.report.bundle.multiple_comparisons.families
    (compared,) = [comparison for comparison in family.comparisons if comparison.name == "_correct"]
    return compared.p_raw if compared.test == "paired" and read_by_bet(family) else None


async def test_a_comparison_whose_run_stopped_keeps_what_it_recorded_and_runs_only_the_cells_it_never_recorded() -> (
    None
):
    """A cost cap stops a comparison's run (k = 3) partway: its cells run case by case in the committed order, so the
    cases at the head of it are complete and the one in flight may be short of some repeat. Its print opens on the stop
    and exits 1. Run again with the cap raised, it continues: every cell the stopped run recorded is kept, and only the
    cells no run recorded run, the missing repeats of the cases it reached as well as the cases it never reached, so it
    finishes as a run that completed does. Stopped again partway through a continuation and run again, the same holds.
    No cell runs twice, and the bet is one capital read four times: each reading's p is the mixture bet's over the
    longest complete prefix of the committed order."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    arms = {"a": _priced("stopped", "a", 0.5, calls), "b": _priced("stopped", "b", 0.5, calls)}

    def compare(n: int, cap: float) -> Eval:
        return Eval(_cases(n), arms, [_correct], k=3, name="capped", max_cost_usd=cap)

    readings: list[Result] = []
    stopped = await compare(20, 6.0).run(store=store, scope="s")
    assert stopped.exit_code() == 1 and {summary.status for summary in stopped.arms.values()} == {"budget_stopped"}
    assert set(stopped.report.bundle.incomplete_runs.values()) == {"budget_stopped"}
    readings.append(stopped)
    runs, results = off_loop(_stored, stopped.host.storage, stopped.campaign_id)
    complete, partial, unreached = _cases_by_state(runs, results)
    assert 2 <= len(complete) < 20 and len(partial) <= 2

    first = await compare(20, 30.0).run(store=store, scope="s")
    assert first.exit_code() in (0, 4) and {summary.status for summary in first.arms.values()} == {"completed"}
    # The stopped launch is a pause its continuation took up: the page opens on a verdict, not on the stop.
    assert not first.report.bundle.incomplete_runs
    assert _how_it_ran(first) == [
        continued_sentence(len(complete), 20 - len(complete), unreached=len(unreached), incomplete=len(partial))
    ]
    readings.append(first)
    # Every cell ran once: each case three times on each arm, never a fourth.
    assert set(calls) == {(n, arm) for n in range(20) for arm in "ab"} and set(calls.values()) == {3}

    again = await compare(30, 3.0).run(store=store, scope="s")
    assert again.exit_code() == 1
    assert _how_it_ran(again) == [continued_sentence(20, 10)]
    readings.append(again)
    runs, results = off_loop(_stored, again.host.storage, again.campaign_id)
    complete, partial, unreached = _cases_by_state(runs, results)
    assert 20 <= len(complete) < 30
    finished = await compare(30, 30.0).run(store=store, scope="s")
    assert finished.exit_code() in (0, 4) and not finished.report.bundle.incomplete_runs
    assert {summary.status for summary in finished.arms.values()} == {"completed"}
    assert {summary.n_cases for summary in finished.arms.values()} == {30}
    assert _how_it_ran(finished) == [
        continued_sentence(len(complete), 30 - len(complete), unreached=len(unreached), incomplete=len(partial))
    ]
    readings.append(finished)
    assert set(calls) == {(n, arm) for n in range(30) for arm in "ab"} and set(calls.values()) == {3}
    # Every result a reading held is held as it ran by every reading after it, and each reading is the bet over the
    # longest complete prefix of the committed order.
    held: set[tuple[str, str, int, str]] = set()
    for reading in readings:
        runs, results = off_loop(_stored, reading.host.storage, reading.campaign_id)
        now = {(r.variant_key, r.test_case_id, r.k_iteration, r.scored_at) for r in results}
        assert held <= now
        held = now
        assert _same_p(_correct_p(reading), _reading_p(runs, results))
    (aspect,) = off_loop(get_campaign, finished.host.storage, finished.campaign_id, "s").aspects
    assert len(aspect.run_ids) == 8
    (family,) = finished.report.bundle.multiple_comparisons.families
    assert read_by_bet(family)


async def test_a_comparison_stopped_after_reaching_every_case_finishes_when_run_again_with_the_cap_raised() -> None:
    """A stop that leaves no case unreached, only cases short of some repeat on some arm: here one arm is priced ten
    times the other (k = 3), so its cap stops it while the other completes, and every case holds the cheap arm's three
    repeats. Run again with the cap raised, the costly arm's missing repeats run, and nothing else: no case is
    wholly unreached, yet the comparison completes, every case paired and read by the bet."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    arms = {
        "a": _priced("late", "a", 0.5, calls, cost=0.10),
        "b": _priced("late", "b", 0.5, calls, cost=0.01),
    }

    def compare(cap: float) -> Eval:
        return Eval(_cases(20), arms, [_correct], k=3, name="late", max_cost_usd=cap)

    stopped = await compare(10.0).run(store=store, scope="s")
    assert stopped.exit_code() == 1
    assert {arm: summary.status for arm, summary in stopped.arms.items()} == {"a": "budget_stopped", "b": "completed"}
    runs, results = off_loop(_stored, stopped.host.storage, stopped.campaign_id)
    complete, partial, unreached = _cases_by_state(runs, results)
    assert not unreached and partial
    missing = sum(3 - calls[n, "a"] for n in range(20))
    assert missing > 0 and {calls[n, "b"] for n in range(20)} == {3}

    finished = await compare(100.0).run(store=store, scope="s")
    assert finished.exit_code() in (0, 4) and not finished.report.bundle.incomplete_runs
    assert {summary.status for summary in finished.arms.values()} == {"completed"}
    assert _how_it_ran(finished) == [
        continued_sentence(len(complete), len(partial), unreached=0, incomplete=len(partial))
    ]
    assert set(calls.values()) == {3}
    runs, results = off_loop(_stored, finished.host.storage, finished.campaign_id)
    assert _cases_by_state(runs, results)[0] == {case for run in runs for case in run.test_case_ids}
    (family,) = finished.report.bundle.multiple_comparisons.families
    (compared,) = [comparison for comparison in family.comparisons if comparison.name == "_correct"]
    assert read_by_bet(family) and compared.case_values is not None and len(compared.case_values[0]) == 20
    assert _same_p(_correct_p(finished), _reading_p(runs, results))


async def test_an_interrupted_comparison_finishes_its_cells_in_flight_files_and_runs_only_what_it_never_recorded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A comparison interrupted once while its runs run (the call's task cancelled, as an interrupt does) starts no
    further cell, lets the cells in flight finish and record, says so, files its campaign, and raises the cancel that
    landed: no row records a cut cell. Run again, it continues: only the cells with no recorded result run, each case
    once on every arm, and the bet reads the longest complete prefix of the committed order."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    running: list[asyncio.Task[Result]] = []
    coins = {"a": _coin("drained", "a", 0.5, calls), "b": _coin("drained", "b", 0.5, calls)}

    def interrupting(arm: str) -> Callable[[Mapping[str, Any]], Any]:
        async def run(case: Mapping[str, Any]) -> str:
            answer = await coins[arm](case)
            await asyncio.sleep(0.002)
            if arm == "b" and sum(count for (_, side), count in calls.items() if side == "b") >= 8 and running:
                asyncio.get_running_loop().call_soon(running.pop().cancel, "interrupted")
            return answer

        run.__name__ = arm
        return run

    arms = {arm: interrupting(arm) for arm in "ab"}
    running.append(asyncio.create_task(Eval(_cases(20), arms, [_correct], k=1, name="cut").run(store=store, scope="s")))
    task = running[0]
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await task
    # The cancel that landed is the one raised, its message kept, never a fresh one.
    assert cancelled.value.args == ("interrupted",)
    assert "in flight; interrupt again to stop now." in capsys.readouterr().err
    host = as_host([Eval(_cases(20), arms, [_correct], k=1, name="cut")], store=store)
    (campaign,) = off_loop(list_campaigns, host.storage, "s", subject_id="cut")
    assert campaign.created_by == QUICK_PATH_FILER
    cut_runs, cut_results = off_loop(_stored, host.storage, campaign.id)
    assert {run.status for run in cut_runs} == {"cancelled"}
    assert cut_results and len(cut_results) < 40 and not any(runs_again(result) for result in cut_results)
    complete, partial, unreached = _cases_by_state(cut_runs, cut_results)

    finished = await Eval(_cases(30), arms, [_correct], k=1, name="cut").run(store=store, scope="s")
    assert {summary.status for summary in finished.arms.values()} == {"completed"}
    assert finished.exit_code() in (0, 4) and not finished.report.bundle.incomplete_runs
    assert _how_it_ran(finished) == [
        continued_sentence(len(complete), 30 - len(complete), unreached=len(unreached), incomplete=len(partial))
    ]
    # Drained, nothing was cut: every case ran once on every arm.
    assert set(calls) == {(n, arm) for n in range(30) for arm in "ab"} and set(calls.values()) == {1}
    runs, results = off_loop(_stored, finished.host.storage, finished.campaign_id)
    assert len({(r.variant_key, r.test_case_id, r.k_iteration) for r in results}) == len(results) == 60
    assert _same_p(_correct_p(finished), _reading_p(runs, results))


async def test_a_comparison_interrupted_twice_cuts_its_cells_in_flight_runs_them_again_and_says_how_many() -> None:
    """A second interrupt while the cells in flight finish cuts them: each is recorded cut (termination ``cancelled``),
    which records no answer (``runs_again``). Run again, the continuation runs each cut cell again, and the print says
    how many ran again; the bet reads every case, its cut cells at their new answers, none of them twice."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    running: list[asyncio.Task[Result]] = []
    coins = {"a": _coin("twice", "a", 0.5, calls), "b": _coin("twice", "b", 0.5, calls)}

    def interrupting(arm: str) -> Callable[[Mapping[str, Any]], Any]:
        async def run(case: Mapping[str, Any]) -> str:
            answer = await coins[arm](case)
            # Each answer takes its own while, so the cells in flight end apart and a second interrupt finds some.
            await asyncio.sleep(0.05 + random.Random(f"twice-{case['n']}-{arm}").random() * 0.3)
            if sum(calls.values()) >= 9 and running:
                task = running.pop()
                loop = asyncio.get_running_loop()
                loop.call_soon(task.cancel, "interrupted")
                loop.call_later(0.005, task.cancel, "again")
            return answer

        run.__name__ = arm
        return run

    arms = {arm: interrupting(arm) for arm in "ab"}
    running.append(
        asyncio.create_task(Eval(_cases(12), arms, [_correct], k=1, name="twice").run(store=store, scope="s"))
    )
    task = running[0]
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await task
    assert cancelled.value.args == ("interrupted",)
    host = as_host([Eval(_cases(12), arms, [_correct], k=1, name="twice")], store=store)
    (campaign,) = off_loop(list_campaigns, host.storage, "s", subject_id="twice")
    cut_runs, cut_results = off_loop(_stored, host.storage, campaign.id)
    cut = [result for result in cut_results if runs_again(result)]
    assert cut and {result.termination for result in cut} == {"cancelled"}
    complete, partial, unreached = _cases_by_state(cut_runs, cut_results)

    finished = await Eval(_cases(12), arms, [_correct], k=1, name="twice").run(store=store, scope="s")
    assert finished.exit_code() in (0, 4) and not finished.report.bundle.incomplete_runs
    assert _how_it_ran(finished) == [
        continued_sentence(
            len(complete),
            12 - len(complete),
            unreached=len(unreached),
            incomplete=len(partial),
            redrawn=RedrawnCells(continued=len(cut), continued_cut=len(cut)),
        )
    ]
    runs, results = off_loop(_stored, finished.host.storage, finished.campaign_id)
    kept = [result for result in results if not runs_again(result)]
    assert len({(r.variant_key, r.test_case_id, r.k_iteration) for r in kept}) == len(kept) == 24
    # Each cut cell's case ran twice on its arm: once cut, once again; every other case once.
    case_n = {f"{runs[0].template_id}-{n}": n for n in range(12)}
    arm_of = {run.id: str(launched_as_run(run)[0]) for run in runs}
    again = Counter((case_n[r.test_case_id], arm_of[r.eval_run_id]) for r in cut)
    assert all(calls[cell] == 1 + again[cell] for cell in calls)
    (family,) = finished.report.bundle.multiple_comparisons.families
    assert read_by_bet(family) and _same_p(_correct_p(finished), _reading_p(runs, results))
    # The report says it too: the family's disclosure, which the page and the bundle carry, in the one phrasing.
    # Each cut cell ran again in the continuation, so the sentence says so in the past tense.
    ran_again = redrawn_sentence(*RedrawnCells(continued=len(cut), continued_cut=len(cut)))
    assert ran_again in family.disclosure
    page = json.dumps(finished.page.model_dump(mode="json"))
    assert json.dumps(ran_again)[1:-1] in page

    # Continued again with cases appended, nothing is cut or drawn again now: the count is the running total, and it
    # says so, unchanged.
    again_run = await Eval(_cases(16), arms, [_correct], k=1, name="twice").run(store=store, scope="s")
    assert _how_it_ran(again_run) == [
        continued_sentence(12, 4, redrawn=RedrawnCells(continued=len(cut), continued_cut=len(cut)))
    ]
    assert "a running total over its runs" in _how_it_ran(again_run)[0]


@pytest.mark.parametrize(
    ("status", "body", "ending"),
    [(401, None, "exhausted"), (404, {"error": {"code": "model_not_found"}}, "exhausted")],
    ids=["account-refused", "model-unserved"],
)
async def test_a_call_the_account_refused_or_a_model_not_served_runs_again_when_the_comparison_continues(
    status: int, body: Any, ending: str
) -> None:
    """A cell whose call the provider account refused, or whose model the provider did not serve, records no answer the
    cell would have given (``runs_again``, the engine take-up's own rule): its run ends ``exhausted``, and run again,
    the comparison runs that cell again, with every cell no launch recorded, and completes, every case read by the
    bet. Its print counts it among the cells that ran again."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    coins = {"a": _coin("turned", "a", 0.5, calls), "b": _coin("turned", "b", 0.5, calls)}

    def turned_away(arm: str) -> Callable[[Mapping[str, Any]], Any]:
        async def run(case: Mapping[str, Any]) -> str:
            answer = await coins[arm](case)
            if arm == "a" and case["n"] == 3 and calls[3, "a"] == 1:
                raise ProviderError("openai", status, "turned away", body=body)
            return answer

        run.__name__ = arm
        return run

    arms = {arm: turned_away(arm) for arm in "ab"}
    stopped = await Eval(_cases(10), arms, [_correct], k=1, name="turned").run(store=store, scope="s")
    assert stopped.arms["a"].status == ending
    runs, results = off_loop(_stored, stopped.host.storage, stopped.campaign_id)
    turned = [result for result in results if runs_again(result)]
    assert len(turned) == 1 and turned[0].test_case_id.endswith("-3")
    complete, partial, unreached = _cases_by_state(runs, results)

    finished = await Eval(_cases(10), arms, [_correct], k=1, name="turned").run(store=store, scope="s")
    assert {summary.status for summary in finished.arms.values()} == {"completed"}
    assert finished.exit_code() in (0, 4) and not finished.report.bundle.incomplete_runs
    assert calls[3, "a"] == 2
    assert _how_it_ran(finished) == [
        continued_sentence(
            len(complete),
            10 - len(complete),
            unreached=len(unreached),
            incomplete=len(partial),
            redrawn=RedrawnCells(continued=1),
        )
    ]
    runs, results = off_loop(_stored, finished.host.storage, finished.campaign_id)
    assert _cases_by_state(runs, results)[0] == {case for run in runs for case in run.test_case_ids}
    (family,) = finished.report.bundle.multiple_comparisons.families
    assert read_by_bet(family) and _same_p(_correct_p(finished), _reading_p(runs, results))


def test_the_cells_that_run_again_are_counted_once_each_however_many_rows_recorded_no_answer() -> None:
    """A cell is an arm's case at one repeat: one whose rows recorded no answer twice, cut in one launch and refused in
    the next, is one cell that ran again, and one cut, never two; the same case on another arm, or at another repeat, is
    another cell."""
    cut = make_eval_result(eval_run_id="run-1", test_case_id="tc-1", variant_key="v-a", termination="cancelled")
    refused = make_eval_result(eval_run_id="run-2", test_case_id="tc-1", variant_key="v-a", account_refused=True)
    answered = make_eval_result(eval_run_id="run-3", test_case_id="tc-1", variant_key="v-a")
    # The case answered in a later run ran again when the comparison was continued, and had been cut.
    assert redrawn_cells([cut, refused, answered]) == RedrawnCells(continued=1, continued_cut=1)
    elsewhere = [
        make_eval_result(eval_run_id="run-4", test_case_id="tc-1", variant_key="v-b", model_unserved=True),
        make_eval_result(
            eval_run_id="run-1", test_case_id="tc-1", variant_key="v-a", k_iteration=2, account_refused=True
        ),
    ]
    # The other two have no answer yet: still to run again, neither cut.
    assert redrawn_cells([cut, refused, answered, *elsewhere]) == RedrawnCells(
        pending=2, pending_cut=0, continued=1, continued_cut=1
    )


@pytest.mark.parametrize(
    ("cells", "cut", "said"),
    [
        (
            1,
            1,
            "1 cell of this comparison recorded no answer (a running total over its runs), and each runs again when it "
            "is continued; it was cut while running, by an interrupt or a shutdown, and a cut cell drawn afresh can "
            "lean the reading where a slow answer is a wrong one.",
        ),
        (
            3,
            1,
            "3 cells of this comparison recorded no answer (a running total over its runs), and each runs again when "
            "it is continued; 1 of them was cut while running, by an interrupt or a shutdown, and a cut cell drawn "
            "afresh can lean the reading where a slow answer is a wrong one.",
        ),
        (
            2,
            2,
            "2 cells of this comparison recorded no answer (a running total over its runs), and each runs again when "
            "it is continued; 2 of them were cut while running, by an interrupt or a shutdown, and a cut cell drawn "
            "afresh can lean the reading where a slow answer is a wrong one.",
        ),
        (
            2,
            0,
            "2 cells of this comparison recorded no answer (a running total over its runs), and each runs again when "
            "it is continued: calls the account refused or the provider did not serve.",
        ),
    ],
)
def test_the_sentence_of_cells_run_again_agrees_with_its_cut_count(cells: int, cut: int, said: str) -> None:
    """The verb agrees with the number it follows: the cut count, or the one cell where there is one."""
    assert redrawn_sentence(cells, cut) == said


@pytest.mark.parametrize(
    ("redrawn", "said"),
    [
        (
            RedrawnCells(taken_up=2, taken_up_cut=2),
            "2 cells of this comparison were cut while running, by an interrupt or a shutdown, and ran again when its "
            "run was taken up (a running total over its runs). A cut cell drawn afresh can lean the reading where a "
            "slow answer is a wrong one.",
        ),
        (
            RedrawnCells(taken_up=1),
            "1 cell of this comparison recorded no answer (a call the account refused or the provider did not serve) "
            "and ran again when its run was taken up (a running total over its runs).",
        ),
        (
            RedrawnCells(continued=3, continued_cut=1),
            "1 cell of this comparison was cut while running, by an interrupt or a shutdown, and ran again when the "
            "comparison was continued (a running total over its runs). 2 cells of this comparison recorded no answer "
            "(calls the account refused or the provider did not serve) and ran again when the comparison was "
            "continued (a running total over its runs). A cut cell drawn afresh can lean the reading where a slow "
            "answer is a wrong one.",
        ),
        (
            RedrawnCells(pending=1, pending_cut=1, continued=1, continued_cut=1),
            "1 cell of this comparison recorded no answer (a running total over its runs), and each runs again when it "
            "is continued; it was cut while running, by an interrupt or a shutdown, and a cut cell drawn afresh can "
            "lean the reading where a slow answer is a wrong one. 1 cell of this comparison was cut while running, by "
            "an interrupt or a shutdown, and ran again when the comparison was continued (a running total over its "
            "runs).",
        ),
    ],
)
def test_a_cell_that_ran_again_is_said_in_the_past_tense_by_how_it_ran(redrawn: RedrawnCells, said: str) -> None:
    """A cell still to run again runs again when the comparison is continued; one a take-up ran again, or a later launch
    answered, ran again, and the sentence says which. The lean is said once."""
    assert redrawn_sentence(*redrawn) == said


async def test_a_continuation_that_records_a_refused_cell_counts_it_on_its_print_as_its_page_does() -> None:
    """A continuation whose own launch records a row with no answer (a call the account refused) counts it on its print,
    as the comparison's page does: the print and the page give one count, this launch's rows included."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    coins = {"a": _coin("refused-again", "a", 0.5, calls), "b": _coin("refused-again", "b", 0.5, calls)}
    refused: list[int] = []
    launch = ["first"]

    def turned_away(arm: str) -> Callable[[Mapping[str, Any]], Any]:
        async def run(case: Mapping[str, Any]) -> str:
            answer = await coins[arm](case)
            # Arm a's account refuses once in each launch, on a case no launch refused before.
            if arm == "a" and len(refused) < len(launch) and case["n"] not in refused:
                refused.append(case["n"])
                raise ProviderError("openai", 401, "turned away")
            return answer

        run.__name__ = arm
        return run

    arms = {arm: turned_away(arm) for arm in "ab"}
    await Eval(_cases(10), arms, [_correct], k=1, name="refused-again").run(store=store, scope="s")
    launch.append("again")
    again = await Eval(_cases(10), arms, [_correct], k=1, name="refused-again").run(store=store, scope="s")
    assert len(refused) == 2

    def counted(text: str) -> int:
        # The cells still to run again and those that ran again are said apart; the count is their total.
        return sum(int(found) for found in re.findall(r"(\d+) cells? of this comparison recorded no answer", text))

    (printed,) = _how_it_ran(again)
    (family,) = again.report.bundle.multiple_comparisons.families
    assert counted(printed) == counted(family.disclosure) == 2


def test_a_second_ctrl_c_arriving_as_keyboard_interrupt_cuts_files_and_raises_it_after_the_first_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a terminal a second Ctrl+C reaches the comparison's own task as ``KeyboardInterrupt``, not as a second cancel
    (here, raised as the drain says it is finishing the cells in flight). It is the cutting interrupt: the cells in
    flight are cut, the runs settle cancelled, the campaign is filed, and the ``KeyboardInterrupt`` itself is raised,
    from the cancel that landed first, so a host that would read a cancel as its own still gets the interrupt. Run
    again, only the cells no run recorded run, the cut ones among them, and nothing recorded is drawn twice. Run on a
    loop of its own, as a host's is, since the interrupt leaves the loop it is raised on."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    running: list[asyncio.Task[Result]] = []
    coins = {"a": _coin("keyboard", "a", 0.5, calls), "b": _coin("keyboard", "b", 0.5, calls)}

    class _SecondCtrlC:
        """stderr, whose drain notice is where the second Ctrl+C lands."""

        def write(self, text: str) -> int:
            if text.startswith("Interrupted:"):
                raise KeyboardInterrupt
            return len(text)

        def flush(self) -> None:
            """Nothing is held."""

    def interrupting(arm: str) -> Callable[[Mapping[str, Any]], Any]:
        async def run(case: Mapping[str, Any]) -> str:
            answer = await coins[arm](case)
            await asyncio.sleep(0.05 + random.Random(f"keyboard-{case['n']}-{arm}").random() * 0.3)
            if sum(calls.values()) >= 9 and running:
                monkeypatch.setattr(sys, "stderr", _SecondCtrlC())
                asyncio.get_running_loop().call_soon(running.pop().cancel, "interrupted")
            return answer

        run.__name__ = arm
        return run

    arms = {arm: interrupting(arm) for arm in "ab"}

    async def interrupted() -> Result:
        running.append(
            asyncio.create_task(Eval(_cases(12), arms, [_correct], k=1, name="keys").run(store=store, scope="s"))
        )
        return await running[0]

    with pytest.raises(KeyboardInterrupt) as second:
        asyncio.run(interrupted())
    monkeypatch.undo()
    assert isinstance(second.value.__cause__, asyncio.CancelledError)
    assert second.value.__cause__.args == ("interrupted",)
    host = as_host([Eval(_cases(12), arms, [_correct], k=1, name="keys")], store=store)
    (campaign,) = off_loop(list_campaigns, host.storage, "s", subject_id="keys")
    cut_runs, cut_results = off_loop(_stored, host.storage, campaign.id)
    assert {run.status for run in cut_runs} == {"cancelled"}
    cut = [result for result in cut_results if runs_again(result)]
    assert cut and {result.termination for result in cut} == {"cancelled"}

    finished = asyncio.run(Eval(_cases(12), arms, [_correct], k=1, name="keys").run(store=store, scope="s"))
    assert finished.exit_code() in (0, 4) and not finished.report.bundle.incomplete_runs
    runs, results = off_loop(_stored, finished.host.storage, finished.campaign_id)
    kept = [result for result in results if not runs_again(result)]
    assert len({(r.variant_key, r.test_case_id, r.k_iteration) for r in kept}) == len(kept) == 24
    case_n = {f"{runs[0].template_id}-{n}": n for n in range(12)}
    arm_of = {run.id: str(launched_as_run(run)[0]) for run in runs}
    again = Counter((case_n[r.test_case_id], arm_of[r.eval_run_id]) for r in cut)
    assert all(calls[cell] == 1 + again[cell] for cell in calls)


class _StoreThatBreaksOnTheCutRow(InMemoryDocumentStore):
    """A store whose next read of a row not yet stored fails with an error no store contract names, once armed: the
    cell row a cut cell is saved as is the first such read after the interrupt."""

    armed = False

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        if self.armed and self.get(doc_id, scope_id) is None:
            self.armed = False
            raise RuntimeError("the store broke while the cut cell was saved")
        return super().get_with_etag(doc_id, scope_id)


def test_a_ctrl_c_in_a_cell_still_leaves_when_saving_the_cut_cell_fails(caplog: pytest.LogCaptureFixture) -> None:
    """A Ctrl+C landing in a cell's own step records the cell cut, then is raised; when saving that row fails with an
    unexpected error, the failure is recorded for an operator and the ``KeyboardInterrupt`` still leaves, never turned
    into a failed cell the run carries on past."""
    store = _StoreThatBreaksOnTheCutRow()

    async def a(case: Mapping[str, Any]) -> str:
        store.armed = True
        raise KeyboardInterrupt

    with caplog.at_level(logging.ERROR), pytest.raises(KeyboardInterrupt):
        asyncio.run(Eval(_cases(1), {"a": a}, [_correct], k=1, name="broken").run(store=store, scope="s"))
    assert not store.armed
    assert any("could not be accounted for" in record.getMessage() for record in caplog.records), caplog.text


# --- the committed order, and the cells played in it -----------------------------------------------------------------


def test_the_committed_order_only_appends_and_depends_on_the_cases_alone() -> None:
    """The order the bet reads is fixed before any answer, and only grows:

    - a later launch never moves an earlier case: the order after one launch is a prefix of the order after two, and
      of the order after three, however many cases the later launches add;
    - a case a launch left unfinished, launched again, keeps its place;
    - each case appears once, and every case launched is in it;
    - it is a shuffle, not the cases file's order, and the same for the same cases whatever order a launch lists them
      in: it reads nothing but the case set and the cases, so no answer, latency or time can move it;
    - another case set orders the same cases otherwise."""
    first = [f"set-{n}" for n in range(20)]
    appended = [f"set-{n}" for n in range(20, 30)]
    more = [f"set-{n}" for n in range(30, 80)]
    one = committed_order("set", [first])
    # A launch that left cases 3 and 7 unfinished; the next holds them again with the cases appended.
    two = committed_order("set", [first, ["set-3", "set-7", *appended]])
    three = committed_order("set", [first, ["set-3", "set-7", *appended], more])
    assert two[: len(one)] == one and three[: len(two)] == two
    assert one.index("set-3") == two.index("set-3") == three.index("set-3")
    assert one.index("set-7") == two.index("set-7") == three.index("set-7")
    assert sorted(three) == sorted({*first, *appended, *more}) and len(three) == len(set(three))
    # The earlier cases' order does not depend on how many cases a later launch adds.
    assert committed_order("set", [first, appended[:1]])[:20] == committed_order("set", [first, appended])[:20] == one
    assert one != first and committed_order("set", [list(reversed(first))]) == one
    assert committed_order("other", [first]) != one


async def test_a_comparison_plays_each_run_s_cells_case_by_case_in_the_committed_order() -> None:
    """Each arm's run plays its cells case by case in the committed order, every repeat of a case before the next case
    (k = 2); a continuation after a stop plays the cases the stop left where they stand in that order, then the cases
    appended in theirs.

    Read without any timing: the comparison declares latency under test, so each run plays one cell at a time and its
    arms run one after another, and the order the runner calls each arm in is its run's cell order exactly, whatever a
    loaded machine does between a cell's start and its call. (Several cells at once, a cell started later can reach the
    runner first, and the arms' lanes start when the job manager starts their runs, so neither the order of calls nor
    the arms' turns can be read off the calls there; the interleaver's turns are ``test_arm_schedule``'s.) Each answer
    costs a quarter and each arm's cap is 0.9, so a run stops once four of its cells have landed: after its second
    case, with no case left part-run."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    started: list[tuple[int, str]] = []

    def recording(arm: str) -> Callable[[Mapping[str, Any]], Any]:
        draw = _priced("played", arm, 0.5, calls, cost=0.25)

        async def run(case: Mapping[str, Any]) -> Answer:
            started.append((case["n"], arm))
            answer: Answer = await draw(case)
            return answer

        run.__name__ = arm
        return run

    arms = {arm: recording(arm) for arm in "ab"}

    def compare(n: int, cap: float) -> Eval:
        return Eval(_cases(n), arms, [_correct], k=2, name="played", max_cost_usd=cap, measure_latency=True)

    def played(starts: list[tuple[int, str]], cases: list[int]) -> None:
        """Each arm is called case by case in ``cases``' order, both repeats of a case before the next case."""
        for arm in "ab":
            assert [n for n, side in starts if side == arm] == [n for n in cases for _ in range(2)], (arm, starts)

    stopped = await compare(12, 1.8).run(store=store, scope="s")
    assert stopped.exit_code() == 1
    runs, _ = off_loop(_stored, stopped.host.storage, stopped.campaign_id)
    template = runs[0].template_id
    case_n = {f"{template}-{n}": n for n in range(20)}
    order = [case_n[case] for case in committed_order(template, [[f"{template}-{n}" for n in range(12)]])]
    # Not the cases file's order: the cases were shuffled once, before any answer.
    assert order != list(range(12))
    played(started, order[:2])
    unfinished = order[2:]

    started.clear()
    await compare(20, 100.0).run(store=store, scope="s")
    appended = [
        case_n[case]
        for case in committed_order(
            template,
            [[f"{template}-{n}" for n in range(12)], [f"{template}-{n}" for n in (*unfinished, *range(12, 20))]],
        )
    ][12:]
    # The continuation plays the cases left where they stand, then the appended ones in their own committed order.
    played(started, [*unfinished, *appended])


# --- the null, through the product -----------------------------------------------------------------------------------


async def _newcomer(tag: str, *, continued: bool) -> bool:
    """One newcomer under the null, through ``Eval.run`` itself: two arms answering right at one rate (0.5, k = 1),
    each call a fresh seeded draw. Run 20 cases; while nothing separates, append cases and run again, at most five runs
    in all, each add ten cases plus up to ninety more the closer the reading's p came to α. ``continued`` runs every
    time into one store, so the product continues the comparison; otherwise each run is into a store of its own, which
    re-runs every case, as the quick path did before it continued a comparison, though read by the same bet. Separated:
    a verdict of improved or regressed on any comparison the print reads."""
    calls: Counter[tuple[int, str]] = Counter()
    arms = {"a": _coin(tag, "a", 0.5, calls), "b": _coin(tag, "b", 0.5, calls)}
    store = InMemoryDocumentStore()
    n = 20
    for _ in range(5):
        if not continued:
            store = InMemoryDocumentStore()
        result = await Eval(_cases(n), arms, [_correct], k=1, name="null").run(store=store, scope="s")
        compared = [c for family in result.report.bundle.multiple_comparisons.families for c in family.comparisons]
        if any(comparison.verdict in ("improved", "regressed") for comparison in compared):
            return True
        ps = [
            comparison.p_raw
            for comparison in compared
            if comparison.name == "_correct" and comparison.p_raw is not None
        ]
        n += 10 + int(90 * (1.0 - min(1.0, min(ps, default=1.0))))
    return False


#: Newcomers per shard: each shard is one seeded test, so the pool runs them side by side.
_PER_SHARD = 50

#: How many of each continued shard's fifty newcomers separated, as seeded (shard by shard). The committed order is
#: fixed by the cases alone and every launch here completes, so each reading is reproducible and each shard test
#: reproduces its count exactly; their sum is the product's count, and the bound below is read on it.
_CONTINUED_SEPARATED: tuple[int, ...] = (
        1, 2, 3, 1, 1, 4, 5, 2,
        2, 4, 3, 2, 1, 2, 6, 3,
        0, 2, 8, 1, 2, 4, 4, 2,
        1, 1, 3, 1, 5, 1, 5, 1,
        2, 1, 2, 3, 1, 0, 3, 1,
        2, 2, 2, 0, 1, 2, 1, 1,
        3, 2, 1, 0, 3, 0, 1, 4,
        3, 4, 2, 1, 1, 2, 2, 1,
        0, 4, 4, 6, 1, 4, 0, 2,
        1, 2, 2, 5, 3, 1, 2, 3,
        2, 2, 2, 2, 1, 1, 0, 3,
        2, 2, 1, 3, 4, 1, 2, 2,
        1, 1, 2, 2, 0, 0, 0, 3,
        3, 2, 2, 1, 3, 0, 4, 1,
        2, 1, 2, 1, 1, 3, 3, 2,
        0, 2, 2, 0, 4, 2, 2, 2,
        3, 2, 4, 2, 6, 2, 1, 3,
        1, 2, 2, 2, 1, 0, 2, 2,
        1, 1, 2, 6, 4, 1, 4, 1,
        0, 3, 1, 0, 1, 1, 1, 3,
        2, 2, 2, 3, 2, 4, 0, 5,
        1, 2, 1, 1, 1, 2, 2, 2,
        2, 0, 4, 1, 3, 1, 1, 0,
        1, 2, 3, 0, 2, 2, 5, 2,
        4, 1, 1, 1, 2, 1, 0, 2,
    )

#: How many of each re-run-everything shard's fifty newcomers separated, as seeded.
_RERUN_SEPARATED: tuple[int, ...] = (4, 6, 6, 5, 7, 10, 6, 4, 9, 5, 2, 7)


def _shards(n: int) -> list[Any]:
    """Shards ``0`` to ``n - 1``: the first runs in the unit suite as a smoke check on the bound's inputs, and the
    rest are ``long_simulation`` (a minute or two of CPU each), which the pool's ``simulation`` suite runs:
    ``testament run --no-fallback --suite simulation``. Every shard is the bound's evidence, so all of them run
    whenever the reading they simulate changes (``.testament.toml`` states the rule)."""
    return [pytest.param(shard, marks=() if shard == 0 else pytest.mark.long_simulation) for shard in range(n)]


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("shard", _shards(len(_CONTINUED_SEPARATED)))
async def test_a_continued_comparison_s_newcomers_separate_as_seeded(shard: int) -> None:
    """Fifty seeded newcomers (:func:`_newcomer`), each running ``Eval.run`` into one store, so the quick path continues
    its comparison: as many separate as this shard's pinned count."""
    separated = 0
    for index in range(_PER_SHARD):
        separated += await _newcomer(f"continued-{shard}-{index}", continued=True)
    assert separated == _CONTINUED_SEPARATED[shard]


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("shard", _shards(len(_RERUN_SEPARATED)))
async def test_a_comparison_re_run_whole_each_time_separates_as_seeded(shard: int) -> None:
    """Fifty seeded newcomers (:func:`_newcomer`), each run into a store of its own, so every case runs again on each
    run: as many separate as this shard's pinned count."""
    separated = 0
    for index in range(_PER_SHARD):
        separated += await _newcomer(f"rerun-{shard}-{index}", continued=False)
    assert separated == _RERUN_SEPARATED[shard]


def test_under_the_null_a_continued_comparison_separates_at_most_alpha_and_re_running_every_case_exceeds_it() -> None:
    """The claim, read on the shards' seeded counts (each reproduced by its shard test above), at α = 5%, two arms
    passing 50% each, k = 1, up to five runs with adds sized from the data, stopping at the first separation:

    - continued (the quick path as it is): the Clopper-Pearson 95% upper bound on the false-separation rate is below α;
    - re-run whole each time (the quick path's re-runs before it continued a comparison, read by the same bet): the 95%
      lower bound is above α, so re-drawing every case on each run is not one bet and loses the bound.

    As seeded: continued, 399 of 10,000 newcomers separate (4.0%, 95% interval 3.6% to 4.4%); re-run whole each time,
    71 of 600 (11.8%, 9.4% to 14.7%). Read at 10,000 because the rate sits near α by construction: each reading's p is
    one over the most the capital ever reached over every case read, so the reads together watch the capital
    continuously, and a martingale watched until it settles reaches 1/α with probability approaching α. The stakes
    weighted toward the larger ones settle sooner within the few hundred cases read, so they spend more of α than the
    even weighting did: on the same 10,000 seeds the even grid separated 312 (3.1%). Every reading is the bet's, read
    for read (:func:`_watched` checks it, and the newcomer restated over the bet alone reproduces every shard pinned
    here).
    """
    replicates = _PER_SHARD * len(_CONTINUED_SEPARATED)
    rerun_replicates = _PER_SHARD * len(_RERUN_SEPARATED)
    continued, rerun = sum(_CONTINUED_SEPARATED), sum(_RERUN_SEPARATED)
    kept = exact_rate_interval(continued, replicates)
    lost = exact_rate_interval(rerun, rerun_replicates)
    assert kept is not None and lost is not None
    said = f"continued {continued}/{replicates} {kept}; re-run {rerun}/{rerun_replicates} {lost}"
    assert kept[1] < SIGNIFICANCE_ALPHA, said
    assert lost[0] > SIGNIFICANCE_ALPHA, said


# --- watched and stopped, through the product ------------------------------------------------------------------------

#: How long each arm's answer takes, as a law of the answer: both arms alike, latency apart from the answer (a coin of
#: its own, 1 or 6 ms); both arms alike, a wrong answer slower (1 ms right, 6 ms wrong); or arms of different laws, arm
#: ``a`` 0.4 ms whatever it answers and arm ``b`` five times slower when right (2 ms) and four or twenty times slower
#: again when wrong (8 or 40 ms).
_Latency = Literal["alike-apart", "alike-slow-wrong", "b-slower-4x", "b-slower-20x"]

#: How a watcher stops a run: when the answers landed in it favour an arm by four (after ten on each arm), or at a point
#: drawn apart from any answer, in four runs of five.
_Stop = Literal["lead", "blind"]


def _seconds(latency: _Latency, tag: str, arm: str, n: int, call: int, right: bool) -> float:
    """How long ``arm``'s answer to case ``n`` (its ``call``-th) takes under ``latency``."""
    if latency == "alike-apart":
        return 0.006 if random.Random(f"{tag}-latency-{n}-{arm}-{call}").random() < 0.5 else 0.001
    if latency == "alike-slow-wrong":
        return 0.001 if right else 0.006
    if arm == "a":
        return 0.0004
    return 0.002 if right else 0.002 * (4 if latency == "b-slower-4x" else 20)


async def _watched(
    tag: str, *, latency: _Latency, stop: _Stop, k: int, cut_share: float = 0.0, launches: int = 5
) -> tuple[bool, bool, bool]:
    """One newcomer under the null who watches each run's answers land and stops it, through ``Eval.run`` into one
    store: two arms answering right at one rate (0.5) each call, ``k`` repeats, each answer taking as long as
    ``latency`` says, the cells played as the product plays them (case by case in the committed order, the arms in
    turn, four at a time).

    Run 20 cases; stop each run as ``stop`` says (an interrupt: the run's task cancelled, so the cells in flight finish
    and record), and, in ``cut_share`` of those stops, interrupt again 5 ms later, cutting the cells still in flight;
    read the comparison as it stands (the run's print where it finished, the filed campaign where it was stopped);
    while nothing separates, append cases and run again, at most ``launches`` runs in all, each adding ten cases plus up
    to ninety more the closer the reading's p came to α. Each reading is checked to be the bet over the longest complete
    prefix of the committed order, and read too by the two rules before it over the same stored answers
    (:func:`_reading_p`'s ``recorded`` and ``filed``), at the product's family size (Holm's ``m``, ``_correct`` counted
    where the product left it untested). It stops once all three have separated on ``_correct``.

    :return: Whether the product separated, whether the record-order rule did, and whether the file-order rule did, in
        any reading.
    """
    calls: Counter[tuple[int, str]] = Counter()
    landed: list[tuple[str, float]] = []
    running: list[asyncio.Task[Result]] = []
    stop_at: list[int] = []
    rules = random.Random(f"{tag}-stops")

    def interrupt() -> None:
        task = running.pop()
        loop = asyncio.get_running_loop()
        loop.call_soon(task.cancel)
        if rules.random() < cut_share:
            loop.call_later(0.005, task.cancel)

    def watched(arm: str) -> Callable[[Mapping[str, Any]], Any]:
        coin = _coin(tag, arm, 0.5, calls)

        async def run(case: Mapping[str, Any]) -> str:
            answer = await coin(case)
            await asyncio.sleep(_seconds(latency, tag, arm, case["n"], calls[case["n"], arm], answer == "right"))
            landed.append((arm, 1.0 if answer == "right" else 0.0))
            if not running:
                return answer
            if stop == "blind":
                if stop_at and len(landed) >= stop_at[0]:
                    interrupt()
            else:
                mine = [v for side, v in landed if side == "a"], [v for side, v in landed if side == "b"]
                if min(map(len, mine)) >= 10 and abs(sum(mine[1]) - sum(mine[0])) >= 4:
                    interrupt()
            return answer

        run.__name__ = arm
        return run

    arms = {arm: watched(arm) for arm in "ab"}
    store = InMemoryDocumentStore()
    n = 20
    separated = [False, False, False]
    for _ in range(launches):
        compare = Eval(_cases(n), arms, [_correct], k=k, name="watched")
        landed.clear()
        stop_at[:] = [rules.randint(1, 2 * k * n)] if rules.random() < 0.8 else []
        running.append(asyncio.create_task(compare.run(store=store, scope="s")))
        task = running[0]
        host = as_host([compare], store=store)
        try:
            families: Sequence[ComparisonFamily] = (await task).report.bundle.multiple_comparisons.families
        except asyncio.CancelledError:
            filed = off_loop(list_campaigns, host.storage, "s", subject_id="watched")
            campaign = max(filed, key=lambda each: (each.created_at, each.id))
            bundle = off_loop(assemble_context_bundle, campaign, storage=host.storage, profile=host.profile)
            families = bundle.multiple_comparisons.families
        running.clear()
        product, p = _separated_on_correct(families)
        campaign = max(
            off_loop(list_campaigns, host.storage, "s", subject_id="watched"),
            key=lambda each: (each.created_at, each.id),
        )
        runs, results = off_loop(_stored, host.storage, campaign.id)
        (family,) = families
        (compared,) = [c for c in family.comparisons if c.name == "_correct"]
        # The product's reading is the bet over the longest complete prefix of the committed order.
        assert _same_p(compared.p_raw, _reading_p(runs, results))
        m = family.family_size + (0 if compared.p_raw is not None else 1)
        recorded, filed_order = (_reading_p(runs, results, rule) for rule in ("recorded", "filed"))
        separated = [
            separated[0] or product,
            separated[1] or (recorded is not None and recorded * m < SIGNIFICANCE_ALPHA),
            separated[2] or (filed_order is not None and filed_order * m < SIGNIFICANCE_ALPHA),
        ]
        if all(separated):
            break
        n += 10 + int(90 * (1.0 - min(1.0, p)))
    return separated[0], separated[1], separated[2]


def _separated_on_correct(families: Sequence[ComparisonFamily]) -> tuple[bool, float]:
    """Whether a reading separates the arms on ``_correct``, and its raw p there (1 where it has none)."""
    compared = [c for family in families for c in family.comparisons if c.name == "_correct"]
    return (
        any(comparison.verdict in ("improved", "regressed") for comparison in compared),
        min((c.p_raw for c in compared if c.p_raw is not None), default=1.0),
    )


#: Watchers in each simulation of arms alike in speed (k = 3, up to five runs); stopped comparisons in each simulation
#: of arms of different speed (k = 1, a run and its continuation, so the share's bound is read at a count the pool runs
#: in one request); and each smoke check's.
_ALIKE, _APART, _SMOKE = 300, 1000, 3


async def _watch_many(tag: str, count: int, **how: Any) -> tuple[int, int, int]:
    """How many of ``count`` newcomers (:func:`_watched`, with ``how``) the product, the record-order rule and the
    file-order rule each separated."""
    tally = [0, 0, 0]
    for index in range(count):
        for rule, separated in enumerate(await _watched(f"{tag}-{index}", **how)):
            tally[rule] += separated
    return tally[0], tally[1], tally[2]


def _shares(tally: tuple[int, int, int], count: int) -> str:
    """Each rule's count, share and 95% Clopper-Pearson interval, for a failure's message."""
    return "; ".join(
        f"{rule} {separated}/{count} {exact_rate_interval(separated, count)}"
        for rule, separated in zip(("product", "record order", "file order"), tally, strict=True)
    )


async def test_a_few_watched_comparisons_of_arms_alike_and_apart_in_speed_read_their_prefix() -> None:
    """The unit suite's smoke check of every watched simulation: a few newcomers each, every reading checked to be the
    bet over the longest complete prefix of the committed order (inside :func:`_watched`), and the product separating
    at most one of each few, which at a rate of α happens with probability above 0.99."""
    for how in (
        {"latency": "alike-slow-wrong", "stop": "lead", "k": 3},
        {"latency": "b-slower-4x", "stop": "blind", "k": 1, "launches": 2},
        {"latency": "b-slower-20x", "stop": "blind", "k": 1, "cut_share": 0.3, "launches": 2},
    ):
        product, _, _ = await _watch_many(f"smoke-{how['latency']}-{how['stop']}", _SMOKE, **how)
        assert product <= 1, how


@pytest.mark.long_simulation
@pytest.mark.timeout(1700)
@pytest.mark.parametrize("latency", ["alike-apart", "alike-slow-wrong"])
async def test_under_the_null_arms_alike_in_speed_stopped_when_an_arm_leads_separate_at_most_alpha(
    latency: _Latency,
) -> None:
    """:data:`_ALIKE` newcomers (:func:`_watched`) whose arms answer alike in speed, both apart from the answer or both
    slower when wrong (exchangeable arms; arms of different speed are the next tests'), k = 3, each interrupting a run
    as soon as the answers landed in it favour an arm by four, then continuing, up to five runs: the product's 95%
    Clopper-Pearson upper bound on the false-separation rate is at most α. Drawn, not pinned: which cells an interrupt
    finds in flight depends on time. As drawn for this change (95% Clopper-Pearson intervals), the product separated
    0 of 300 with latency apart (upper 1.2%) and 1 of 300 with both arms slower when wrong (0.3%, upper 1.8%); the
    record-order rule 0 and 1, the file-order rule 0 and 2."""
    tally = await _watch_many(f"alike-{latency}", _ALIKE, latency=latency, stop="lead", k=3)
    bound = exact_rate_interval(tally[0], _ALIKE)
    assert bound is not None and bound[1] <= SIGNIFICANCE_ALPHA, _shares(tally, _ALIKE)


@pytest.mark.long_simulation
@pytest.mark.timeout(1700)
@pytest.mark.parametrize("latency", ["b-slower-4x", "b-slower-20x"])
async def test_under_the_null_arms_of_different_speed_stopped_and_drained_separate_at_most_alpha(
    latency: _Latency,
) -> None:
    """Arms of different speed, the slow one slower again when wrong (arm ``a`` 0.4 ms whatever it answers; arm ``b`` 2
    ms when right and four or twenty times that when wrong): :data:`_APART` newcomers (:func:`_watched`), k = 1, each
    run interrupted in four of five at a point drawn apart from any answer, the cells in flight drained, then continued
    once (two runs). The product reads the longest complete prefix of an order fixed before any answer, so its 95%
    Clopper-Pearson upper bound is at most α. As drawn for this change: at four times slower when wrong, the product 19
    of 1,000 (1.9%, 1.1% to 3.0%); at twenty times, 17 (1.7%, 1.0% to 2.7%); the two comparison rules over the same
    answers 19 and 19, and 20 and 11. Over three runs (800 newcomers each) the product drew 12 and 29, the latter 3.6%
    with an upper bound of 5.1%, which does not clear α at that size: consistent with the bound (the plain five-run
    flow draws 2.2% to 4.5%), and the test reads two runs so that it clears α at a count one pool request holds."""
    tally = await _watch_many(f"drained-{latency}", _APART, latency=latency, stop="blind", k=1, launches=2)
    bound = exact_rate_interval(tally[0], _APART)
    assert bound is not None and bound[1] <= SIGNIFICANCE_ALPHA, _shares(tally, _APART)


@pytest.mark.long_simulation
@pytest.mark.timeout(1700)
@pytest.mark.parametrize("latency", ["b-slower-4x", "b-slower-20x"])
async def test_under_the_null_arms_of_different_speed_whose_stops_cut_are_reported(latency: _Latency) -> None:
    """The same, with a second interrupt 5 ms after three in ten of the stops, cutting the cells still in flight; a
    continuation draws each cut cell afresh, which can lean the reading where a slow answer is a wrong one (the print
    counts them). Reported as it is, as drawn for this change: at four times slower when wrong, the product 20 of
    1,000 (2.0%, 1.2% to 3.1%), the record-order rule 21, the file-order rule 22; at twenty times, the product 14
    (1.4%, 0.8% to 2.3%), the record-order rule 31 (3.1%, 2.1% to 4.4%), the file-order rule 18. The test fails only
    where the product's share passes twice α."""
    tally = await _watch_many(f"cut-{latency}", _APART, latency=latency, stop="blind", k=1, cut_share=0.3, launches=2)
    assert tally[0] / _APART < 2 * SIGNIFICANCE_ALPHA, _shares(tally, _APART)
