"""``run_eval(tools=...)``: a candidate's tools, called live, recorded, and replayed through the engine's cassettes.

A candidate declaring tools is called with the case and its tools, in every cassette mode alike. A
capture records what each tool answered; a replay of that capture serves the recording and never calls
the tool; an ask the capture never made, or made fewer times, excludes the cell as the rig's failure —
even when the candidate catches the miss itself. A cassette run of a candidate with no tools is refused
before anything is stored, as are tools no candidate could be handed. ``compare`` hands every arm the
same replay.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.contracts import ValidationFailedError
from threetears.evals.quick import CandidateTools, callable_host, compare, run_eval
from threetears.evals.run import get_run, list_runs

SCOPE = "run-eval-tools-tests"
CASES = [{"n": 1}, {"n": 2}, {"n": 3}]


class Dice:
    """A tool whose live answer differs on every call, counting how often it was really called."""

    def __init__(self) -> None:
        self._rolls = itertools.count(100)
        self.calls = 0

    def roll(self, sides: int) -> int:
        self.calls += 1
        return next(self._rolls) % sides + 1


async def adds_a_roll(case: Mapping[str, Any], tools: CandidateTools) -> int:
    return int(case["n"]) + await tools["roll"](sides=20)


def total(case: Mapping[str, Any], answer: Any) -> float:
    return float(answer)


async def test_a_candidate_with_tools_is_handed_them_live_with_cassettes_off_sync_and_async_alike() -> None:
    async def lookup(key: str) -> dict[str, Any]:
        return {"key": key, "found": True}

    async def both(case: Mapping[str, Any], tools: CandidateTools) -> int:
        found = await tools["lookup"](key=str(case["n"]))
        return int(found["key"]) * 10 + await tools["roll"](sides=6)

    dice = Dice()
    summary = await run_eval(CASES, both, [total], scope_id=SCOPE, k=1, tools={"roll": dice.roll, "lookup": lookup})
    assert summary.status == "completed" and summary.n_scored == 3 and dice.calls == 3
    # Keys 1, 2, 3 times ten, plus rolls of 100, 101, 102 on a d6: 5, 6, 1.
    assert summary.measures[0].mean == pytest.approx((15 + 26 + 31) / 3)


async def test_a_replay_serves_the_capture_and_never_calls_the_live_tool() -> None:
    host, dice = callable_host([total]), Dice()
    capture = await run_eval(
        CASES, adds_a_roll, [total], scope_id=SCOPE, host=host, k=1, tools={"roll": dice.roll}, cassette_mode="capture"
    )
    assert capture.status == "completed" and dice.calls == 3
    captured = capture.measures[0].mean

    replays = [
        await run_eval(
            CASES,
            adds_a_roll,
            [total],
            scope_id=SCOPE,
            host=host,
            k=2,
            tools={"roll": dice.roll},
            cassette_mode="replay",
            cassette_corpus_id=capture.run_id,
            model=f"replay-{attempt}",
        )
        for attempt in (1, 2)
    ]
    assert dice.calls == 3, "a replay called the live tool"
    for replay in replays:
        assert (replay.status, replay.n_scored, replay.n_excluded) == ("completed", 6, 0)
        assert replay.measures[0].mean == captured
        run = get_run(host.storage, replay.run_id, SCOPE)
        assert (run.cassette_mode, run.cassette_corpus_id) == ("replay", capture.run_id)


async def test_an_ask_the_capture_never_made_excludes_the_cell_as_the_rigs_failure_and_stays_offline() -> None:
    host, dice = callable_host([total]), Dice()
    capture = await run_eval(
        CASES, adds_a_roll, [total], scope_id=SCOPE, host=host, k=1, tools={"roll": dice.roll}, cassette_mode="capture"
    )

    async def rolls_bigger_on_two(case: Mapping[str, Any], tools: CandidateTools) -> int:
        return int(case["n"]) + await tools["roll"](sides=100 if case["n"] == 2 else 20)

    replay = await run_eval(
        CASES,
        rolls_bigger_on_two,
        [total],
        scope_id=SCOPE,
        host=host,
        k=1,
        tools={"roll": dice.roll},
        cassette_mode="replay",
        cassette_corpus_id=capture.run_id,
    )
    assert dice.calls == 3
    assert (replay.status, replay.n_scored, replay.n_candidate_failed, replay.n_excluded) == ("completed", 2, 0, 1)
    assert len(replay.errors) == 1 and "CassetteMiss" in replay.errors[0]


async def test_a_miss_the_candidate_swallows_still_excludes_the_cell() -> None:
    host, dice = callable_host([total]), Dice()
    capture = await run_eval(
        CASES, adds_a_roll, [total], scope_id=SCOPE, host=host, k=1, tools={"roll": dice.roll}, cassette_mode="capture"
    )

    async def shrugs_off_failures(case: Mapping[str, Any], tools: CandidateTools) -> int:
        try:
            return int(case["n"]) + await tools["roll"](sides=12)  # never captured
        except Exception:  # noqa: BLE001 — the point: a candidate that absorbs every tool failure
            return 0

    replay = await run_eval(
        CASES,
        shrugs_off_failures,
        [total],
        scope_id=SCOPE,
        host=host,
        k=1,
        tools={"roll": dice.roll},
        cassette_mode="replay",
        cassette_corpus_id=capture.run_id,
    )
    # Every cell missed, so the rig measured nothing: the run fails, and no cell is scored as the candidate's 0.
    assert dice.calls == 3
    assert (replay.status, replay.n_scored, replay.n_candidate_failed, replay.n_excluded) == ("failed", 0, 0, 3)


async def test_asking_more_often_than_the_capture_did_is_an_exhaustion_not_a_neighbours_answer() -> None:
    host, dice = callable_host([total]), Dice()
    capture = await run_eval(
        CASES, adds_a_roll, [total], scope_id=SCOPE, host=host, k=1, tools={"roll": dice.roll}, cassette_mode="capture"
    )

    async def rolls_twice_on_three(case: Mapping[str, Any], tools: CandidateTools) -> int:
        answer = await adds_a_roll(case, tools)
        return answer + (await tools["roll"](sides=20) if case["n"] == 3 else 0)

    replay = await run_eval(
        CASES,
        rolls_twice_on_three,
        [total],
        scope_id=SCOPE,
        host=host,
        k=1,
        tools={"roll": dice.roll},
        cassette_mode="replay",
        cassette_corpus_id=capture.run_id,
    )
    assert dice.calls == 3
    assert (replay.n_scored, replay.n_excluded) == (2, 1)
    assert "CassetteExhausted" in replay.errors[0]


async def test_a_tool_answer_no_cassette_can_record_excludes_the_cell() -> None:
    def unrecordable(case_n: int) -> object:
        return object()

    async def asks(case: Mapping[str, Any], tools: CandidateTools) -> int:
        await tools["odd"](case_n=case["n"])
        return 1

    summary = await run_eval(CASES[:1], asks, [total], scope_id=SCOPE, k=1, tools={"odd": unrecordable})
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (0, 0, 1)
    assert "which a cassette cannot record" in summary.errors[0]


async def test_a_tool_that_raises_is_the_candidates_to_handle() -> None:
    def broken(**_: Any) -> int:
        raise TimeoutError("the service timed out")

    async def asks(case: Mapping[str, Any], tools: CandidateTools) -> int:
        return int(await tools["broken"]())

    summary = await run_eval(CASES[:1], asks, [total], scope_id=SCOPE, k=1, tools={"broken": broken})
    assert (summary.n_scored, summary.n_candidate_failed, summary.n_excluded) == (0, 1, 0)
    assert "TimeoutError: the service timed out" in summary.errors[0]


@pytest.mark.parametrize("mode", ["capture", "replay"])
async def test_a_cassette_run_of_a_candidate_that_declares_no_tools_is_refused_before_anything_is_stored(
    mode: Any,
) -> None:
    async def plain(case: Mapping[str, Any]) -> int:
        return 1

    host = callable_host([total])
    corpus = "some-capture" if mode == "replay" else None
    with pytest.raises(ValueError, match="declares none"):
        await run_eval(
            CASES, plain, [total], scope_id=SCOPE, host=host, k=1, cassette_mode=mode, cassette_corpus_id=corpus
        )
    assert list_runs(host, SCOPE) == []


@pytest.mark.parametrize(
    ("tools", "said"),
    [
        ({}, "non-empty mapping"),
        ([("roll", Dice().roll)], "non-empty mapping"),
        ({"not a name": Dice().roll}, "no identifier"),
        ({"roll": 3}, "not callable"),
    ],
    ids=["empty", "not a mapping", "a name that is no identifier", "not callable"],
)
async def test_tools_no_candidate_could_be_handed_are_refused(tools: Any, said: str) -> None:
    with pytest.raises(ValueError, match=said):
        await run_eval(CASES, adds_a_roll, [total], scope_id=SCOPE, k=1, tools=tools)


async def test_a_replay_naming_no_capture_of_these_cases_is_refused_by_the_launch() -> None:
    host, dice = callable_host([total]), Dice()
    with pytest.raises(ValidationFailedError, match="names no corpus"):
        await run_eval(
            CASES,
            adds_a_roll,
            [total],
            scope_id=SCOPE,
            host=host,
            k=1,
            tools={"roll": dice.roll},
            cassette_mode="replay",
        )
    other = await run_eval(
        CASES[:2],
        adds_a_roll,
        [total],
        scope_id=SCOPE,
        host=host,
        k=1,
        tools={"roll": dice.roll},
        cassette_mode="capture",
    )
    with pytest.raises(ValidationFailedError, match="captured template"):
        await run_eval(
            CASES,
            adds_a_roll,
            [total],
            scope_id=SCOPE,
            host=host,
            k=1,
            tools={"roll": dice.roll},
            cassette_mode="replay",
            cassette_corpus_id=other.run_id,
        )
    with pytest.raises(ValidationFailedError, match="name a corpus only with cassette_mode='replay'"):
        await run_eval(
            CASES,
            adds_a_roll,
            [total],
            scope_id=SCOPE,
            host=host,
            k=1,
            tools={"roll": dice.roll},
            cassette_corpus_id=other.run_id,
        )


async def test_compare_serves_every_arm_one_capture() -> None:
    host, dice = callable_host([total]), Dice()
    capture = await run_eval(
        CASES, adds_a_roll, [total], scope_id=SCOPE, host=host, k=1, tools={"roll": dice.roll}, cassette_mode="capture"
    )
    rolled: dict[str, list[int]] = {}

    def arm(name: str, offset: int) -> Any:
        async def candidate(case: Mapping[str, Any], tools: CandidateTools) -> int:
            roll = await tools["roll"](sides=20)
            rolled.setdefault(name, []).append(roll)
            return int(case["n"]) + roll + offset

        return candidate

    comparison = await compare(
        CASES,
        {"base": arm("base", 0), "plus": arm("plus", 1)},
        [total],
        control="base",
        scope_id=SCOPE,
        host=host,
        k=1,
        tools={"roll": dice.roll},
        cassette_mode="replay",
        cassette_corpus_id=capture.run_id,
    )
    assert dice.calls == 3
    assert sorted(rolled["base"]) == sorted(rolled["plus"])
    means = {name: summary.measures[0].mean for name, summary in comparison.arms.items()}
    assert means["plus"] == pytest.approx(means["base"] + 1)
