"""The quick path's re-readable comparison: a newcomer adds cases and runs it again, as often as they like, and the
error each reading states still holds (owner ruling, 2026-11-17).

A quick-path comparison (every result of a kind only the quick path launches) reads each paired contrast on a declared
range by the mixture bet (:func:`~threetears.evals.analysis.stats.bounded_separation_p` and
:func:`~threetears.evals.analysis.stats.bounded_difference_interval` with ``mixture``): the mean of ten bets, each
staking a fixed share ``j / 10`` of the cap on every pair, over the pairs in the order the cases were added, hedged
over the two directions. No stake depends on a declared count, and the quick path continues one comparison by running
only the cases appended to it (:mod:`threetears.evals.quick.continuation`), so every reading is one capital read again.
Held here:

- the decisions are the simulated construction's (``looks-bet-simulation-1116.md``, "Mixture stakes"), decision for
  decision on fixed seeds, and the interval excludes zero exactly where the test separates;
- the bet alone, over rows appended and kept: re-reading separates at most α under the null;
- the product: a comparison run again runs only the cases appended and reads the rest as they ran; anything else is a
  new comparison that runs every case and says what changed; the case order is append-only;
- the product under the null, ``Eval.run`` itself over a scripted runner whose arms are equal in distribution: run
  again as cases are added it separates at most α, and re-running every case on each run (as the quick path did
  before), read by the same bet, exceeds α;
- at one stated alternative, a single read's power against the reading it replaces;
- the rung 3 comparison prints the one sentence that says so, and reads cost by its t test, once.

Each simulation is seeded, so a run is one fixed draw; its docstring states the rate it holds and the bound it is
checked against. The product null's shards past the first of each are ``long_simulation`` (about two CPU-minutes
each): the unit suite runs the bound and one shard of each, and the pool's ``simulation`` suite runs the rest,
``testament run --no-fallback --suite simulation``, whenever the reading they simulate changes.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from threetears.evals import Answer, Eval, Result, grader
from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.actions.catalogue import Caller
from threetears.evals.analysis import ComparisonFamily, assemble_context_bundle, get_campaign
from threetears.evals.analysis.bundle.comparisons import REREAD_SENTENCE, read_by_bet
from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    bounded_difference_interval,
    bounded_separation_p,
    composite_significance,
    exact_rate_interval,
    holm_adjust,
)
from threetears.evals.kernel import CampaignArm
from threetears.evals.kernel.candidate_kind import QUICK_PATH_FILER, QUICK_PATH_KINDS
from threetears.evals.kernel.declaration import AspectDesign
from threetears.evals.ops.host import OpsHost
from threetears.evals.quick import as_host
from threetears.evals.quick.continuation import UNKEPT_SENTENCE, continued_sentence, new_comparison_sentence
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
    """The simulated construction, ``reference-mix.py``'s ``first_crossing`` on its ``mix_grid`` arm, one row at a time
    in plain Python: z = (d + 1) / 2, null 1/2, ten fractions λ_j = (j / 10) · 0.9 / 0.5, each direction's log capital a
    cumulative sum of log1p, the ten combined by successive logaddexp less log 10, hedged as their logaddexp less log
    2, and the first index at which the hedged log capital reaches log(1/α); the direction there the larger side.

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
            if acc is None:
                acc = path
            else:
                acc = [max(a, b) + math.log1p(math.exp(-abs(a - b))) for a, b in zip(acc, path, strict=True)]
        assert acc is not None
        sides.append([value - math.log(10) for value in acc])
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


@pytest.mark.parametrize(("k", "rate", "arms", "seeded_at"), [(1, 0.5, 2, 66), (3, 0.8, 2, 24), (1, 0.8, 3, 53)])
def test_re_reading_as_cases_are_added_separates_under_the_null_at_most_alpha(
    k: int, rate: float, arms: int, seeded_at: int
) -> None:
    """The bet alone, over rows appended and kept (the reference construction, not the product: the product's own
    re-runs are simulated below, through ``Eval.run``). 2,000 seeded newcomers a cell, each reading up to five times
    with adds sized from the data, separate no more
    often than α = 5%: the Clopper-Pearson 95% upper bound on the rate is below α. The seeded draw gives 66 of 2,000
    at k = 1, base 0.5 (3.3%, upper bound 4.2%); 24 at k = 3, base 0.8 (1.2%, upper 1.8%); and 53 for three arms at
    k = 1, base 0.8 under Holm (2.7%, upper 3.5%). The counts are pinned to two either side, so a change to the bet
    that moves them is seen; the bound is the claim."""
    rng = random.Random(f"rereadable-null-{k}-{rate}-{arms}")
    replicates = 2000
    separated = sum(_reread_until_separated(rng, k, rate, arms) for _ in range(replicates))
    bounds = exact_rate_interval(separated, replicates)
    assert bounds is not None and bounds[1] < SIGNIFICANCE_ALPHA
    assert abs(separated - seeded_at) <= 2


# --- power on a single read ------------------------------------------------------------------------------------------


def test_a_single_read_gives_up_some_power_against_the_reading_it_replaces() -> None:
    """At k = 1, the control passing 80% and the arm 60%, on 60 paired cases read once, 2,000 seeded draws: the
    mixture separates on 742 (37.1%), where the reading it replaces (the paired t test's p below α and the bounded
    test's 95% interval, fixed for those 60 cases, excluding zero) separates on 872 (43.6%). That is the price of
    reading again freely: six and a half points of power at this alternative, held to no more than fifteen."""
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
    assert abs(mixture - 742) <= 10 and abs(one_shot - 872) <= 10


# --- the quick path --------------------------------------------------------------------------------------------------


async def test_rung_3_s_comparison_is_read_by_the_bet_and_says_so_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rung 3a, replayed offline into its store: the print says the one sentence under its first line; every paired
    reading on a range is the bet's, and cost is read by its t test, once; a bet reading's adjusted p is Holm's over the
    family with the t reading's p held at 1, so no t p lets a bet reading step down. Kept in memory, the line under the
    first says instead that running it again is a new comparison (:func:`test_kept_in_memory_a_comparison_says_running_it_again_is_a_new_one`)."""
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
    raw = [comparison.p_raw for comparison in family.comparisons]
    held = holm_adjust([1.0 if comparison.basis == "t" else p for comparison, p in zip(family.comparisons, raw)])
    for comparison, adjusted in zip(family.comparisons, held):
        if comparison.basis == "bounded":
            assert comparison.p_adjusted == pytest.approx(adjusted)


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


def _kinded_family(kind: str, *, created_by: str = QUICK_PATH_FILER, rerun: bool = False) -> ComparisonFamily:
    """One question's family over two arms' judged scores on twelve shared cases, every run and result of ``kind``, in
    a campaign filed by ``created_by``; ``rerun`` gives the contrast arm a second run over the first six cases again,
    so it ran those cases twice."""
    runs = []
    results: dict[str, list[EvalResult]] = {}
    lists = [("control-model", range(12)), ("contrast-model", range(12))]
    if rerun:
        lists.append(("contrast-model", range(6)))
    for model, cases in lists:
        run = make_eval_run(
            status="completed",
            candidate_model=model,
            candidate_kind=kind,
            test_case_ids=[f"tc-{case:02d}" for case in cases],
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
            for case in cases
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
    bundle = assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=toyhost_profile())
    (family,) = bundle.multiple_comparisons.families
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


def _priced(tag: str, arm: str, rate: float, calls: Counter[tuple[int, str]]) -> Callable[[Mapping[str, Any]], Any]:
    """:func:`_coin`'s arm, each answer priced at ten cents by one model, which it names as served, so a cost cap stops
    it between cases and its runs pool."""
    draw = _coin(tag, arm, rate, calls)

    async def run(case: Mapping[str, Any]) -> Answer:
        return Answer(
            await draw(case),
            model="priced-model",
            served_model="priced-model",
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.10,
        )

    run.__name__ = arm
    return run


async def test_a_comparison_whose_run_stopped_finishes_when_run_again_with_the_cap_raised() -> None:
    """A cost cap stops a comparison's run partway, and its print opens on the stop and exits 1. Run again with the cap
    raised, it finishes and exits as a run that completed does. A first run that stopped had nothing read, so the run
    again is a first comparison over every case. A continuation that stopped is set aside: the cases run before it are
    read as they ran, its cases run again after them, and the bet reads every case in the order first run, over the
    first launch's runs and the new ones only, so it is one bet read later."""
    store = InMemoryDocumentStore()
    calls: Counter[tuple[int, str]] = Counter()
    arms = {"a": _priced("stopped", "a", 0.5, calls), "b": _priced("stopped", "b", 0.5, calls)}

    def compare(n: int, cap: float) -> Eval:
        return Eval(_cases(n), arms, [_correct], k=1, name="capped", max_cost_usd=cap, measure_latency=True)

    stopped = await compare(20, 1.0).run(store=store, scope="s")
    assert stopped.exit_code() == 1 and {summary.status for summary in stopped.arms.values()} == {"budget_stopped"}
    first = await compare(20, 10.0).run(store=store, scope="s")
    assert first.exit_code() in (0, 4) and {summary.status for summary in first.arms.values()} == {"completed"}
    assert _how_it_ran(first) == [] and all(summary.n_results == 20 for summary in first.arms.values())

    again = await compare(30, 1.0).run(store=store, scope="s")
    assert again.exit_code() == 1 and _how_it_ran(again) == [continued_sentence(20, 10)]
    before = dict(calls)
    finished = await compare(30, 10.0).run(store=store, scope="s")
    assert finished.exit_code() in (0, 4)
    assert {summary.status for summary in finished.arms.values()} == {"completed"}
    assert _how_it_ran(finished) == [continued_sentence(20, 10, again=10)]
    # The cases the first comparison ran are not run again; the stopped launch's ten are, once more on each arm.
    assert {key: calls[key] - before.get(key, 0) for key in calls if calls[key] != before.get(key, 0)} == {
        (n, arm): 1 for n in range(20, 30) for arm in "ab"
    }
    (aspect,) = off_loop(get_campaign, finished.host.storage, finished.campaign_id, "s").aspects
    (kept,) = off_loop(get_campaign, first.host.storage, first.campaign_id, "s").aspects
    (set_aside,) = off_loop(get_campaign, again.host.storage, again.campaign_id, "s").aspects
    assert set(kept.run_ids) < set(aspect.run_ids) and len(aspect.run_ids) == 4
    assert not set(aspect.run_ids) & (set(set_aside.run_ids) - set(kept.run_ids))
    (family,) = finished.report.bundle.multiple_comparisons.families
    (paired,) = [comparison for comparison in family.comparisons if comparison.name == "_correct"]
    assert read_by_bet(family) and paired.case_values is not None and len(paired.case_values[0]) == 30
    assert {trial.case_place for trial in finished.report.bundle.trials} == set(range(1, 31))
    for arm in "ab":
        ran = {(result.case, result.answer) for result in first.arms[arm].case_results}
        assert ran < {(result.case, result.answer) for result in finished.arms[arm].case_results}


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

#: How many of each continued shard's fifty newcomers separated, as seeded (shard by shard). Each shard test reproduces
#: its count exactly, so their sum is the product's count; the bound below is read on that sum.
_CONTINUED_SEPARATED: tuple[int, ...] = (1, 2, 1, 1, 2, 1, 1, 2, 1, 3, 3, 1, 2, 1, 3, 3, 1, 1, 2, 0)

#: How many of each re-run-everything shard's fifty newcomers separated, as seeded.
_RERUN_SEPARATED: tuple[int, ...] = (5, 3, 3, 5, 3, 3, 6, 8)


def _shards(n: int) -> list[Any]:
    """Shards ``0`` to ``n - 1``: the first runs in the unit suite as a smoke check on the bound's inputs, and the
    rest are ``long_simulation`` (minutes of CPU each), which the pool's ``simulation`` suite runs:
    ``testament run --no-fallback --suite simulation``. Every shard is the bound's evidence, so all of them run
    whenever the reading they simulate changes (``.testament.toml`` names the files)."""
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
    - re-run whole each time (the quick path's re-runs before it continued a comparison, read by the same bet, not by
      the reading the quick path had then): the 95% lower bound is above α, so re-drawing every case on each run is
      not one bet and loses the bound.

    As seeded: continued, 32 of 1,000 newcomers separate (3.2%, 95% interval 2.2% to 4.5%); re-run whole each time,
    36 of 400 (9.0%, 6.4% to 12.2%)."""
    replicates = _PER_SHARD * len(_CONTINUED_SEPARATED)
    rerun_replicates = _PER_SHARD * len(_RERUN_SEPARATED)
    continued, rerun = sum(_CONTINUED_SEPARATED), sum(_RERUN_SEPARATED)
    kept = exact_rate_interval(continued, replicates)
    lost = exact_rate_interval(rerun, rerun_replicates)
    assert kept is not None and lost is not None
    said = f"continued {continued}/{replicates} {kept}; re-run {rerun}/{rerun_replicates} {lost}"
    assert kept[1] < SIGNIFICANCE_ALPHA, said
    assert lost[0] > SIGNIFICANCE_ALPHA, said
