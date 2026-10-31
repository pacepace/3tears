"""an answer is compressed on a worker thread, never on the event loop, and written as JSON a piece at a time.

A large answer takes tens of milliseconds to gzip, and every other request on the pod waits for the loop while
it does. So ``VersionedAnswers`` compresses on a worker thread, both where it stores an answer and where it
answers one uncached; ``answer_json`` writes ``json.dumps``' exact text in calls short enough that the loop runs
between them, whatever the answer's shape: a short list of long series, a mapping keyed by numbers, rows of long
text (one call of 0.5 s and one of 50 ms on such answers before, 2026-10-31).
"""

from __future__ import annotations

import asyncio
import gzip
import json
import random
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from threetears.core.collections.registry import CollectionRegistry

from threetears.core.collections.versioned_answers import AnswerOutcome, VersionedAnswers, answer_json
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient

_SCOPE = "tool_pod-answersoffloop"


def _answers() -> VersionedAnswers:
    nats = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l1_backend=None, l2_client=nats, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    return VersionedAnswers(registry, DefaultCoreConfig(), nats, table_name="offloop_answers")


@pytest.fixture
def compressing_threads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """the thread each gzip of this test ran on."""
    threads: list[int] = []
    original: Callable[..., bytes] = gzip.compress

    def recording(*args: Any, **kwargs: Any) -> bytes:
        threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(gzip, "compress", recording)
    return threads


async def _text() -> str:
    return '{"rows": [1, 2, 3]}'


class TestAnAnswerIsCompressedOffTheLoop:
    async def test_a_computed_answer_is_compressed_on_a_worker_thread(self, compressing_threads: list[int]) -> None:
        given = await _answers().answer_with_outcome("v1", "page", _text, order=1)
        assert given.outcome is AnswerOutcome.COMPUTED
        assert gzip.decompress(given.body) == b'{"rows": [1, 2, 3]}'
        assert compressing_threads, "the answer was not compressed at all"
        assert threading.get_ident() not in compressing_threads, "the answer was compressed on the event loop"

    async def test_an_answer_given_uncached_is_compressed_on_a_worker_thread(
        self, compressing_threads: list[int]
    ) -> None:
        answers = _answers()
        await answers.retire_older_than(5)
        given = await answers.answer_with_outcome("v2", "page", _text, order=2)
        assert given.outcome is AnswerOutcome.RETIRED
        assert gzip.decompress(given.body) == b'{"rows": [1, 2, 3]}'
        assert compressing_threads and threading.get_ident() not in compressing_threads


class TestAnswerJson:
    @pytest.mark.parametrize(
        "answer",
        [
            {"rows": [{"id": n, "at": datetime(2026, 10, 24, tzinfo=UTC), "v": n / 3} for n in range(2000)]},
            [[n, str(n)] for n in range(1000)],
            {"a": {"b": [1, 2, {"c": None}]}, "d": "é\n"},
            {1: "a key that is not text", "b": 2},
            list(range(257)),
            (1, "two"),
            "text",
            3.5,
            None,
        ],
    )
    def test_it_is_exactly_what_json_dumps_writes(self, answer: Any) -> None:
        assert answer_json(answer) == json.dumps(answer, default=str)

    def test_a_decimal_is_written_as_the_json_number_its_float_writes(self) -> None:
        exact = Decimal("19380.254913924450")
        answer = {"total": exact, "rows": [{"share": Decimal("0.125")}] * 300, "when": date(2026, 11, 3)}
        expected = json.dumps(
            {"total": float(exact), "rows": [{"share": 0.125}] * 300, "when": "2026-11-03"}, default=str
        )
        assert answer_json(answer) == expected

    @pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity"), Decimal("sNaN")])
    def test_a_non_finite_decimal_is_written_as_null(self, bad: Decimal) -> None:
        """``NaN`` and ``Infinity`` are not JSON; a NUMERIC column holding ``'NaN'`` must not make an answer unreadable."""
        answer = {"total": bad, "rows": [{"share": bad, "n": 1}] * 300, "flat": [bad]}
        text = answer_json(answer)
        assert json.loads(text) == {"total": None, "rows": [{"share": None, "n": 1}] * 300, "flat": [None]}
        assert "NaN" not in text and "Infinity" not in text

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_float_is_written_as_null(self, bad: float) -> None:
        answer = {"ratio": bad, "rows": [{"ratio": bad, "ok": 0.5}] * 300, "pair": (bad, 2.5), "nested": {"x": [bad]}}
        text = answer_json(answer)
        expected = {
            "ratio": None,
            "rows": [{"ratio": None, "ok": 0.5}] * 300,
            "pair": [None, 2.5],
            "nested": {"x": [None]},
        }
        assert json.loads(text) == expected
        assert "NaN" not in text and "Infinity" not in text

    def test_no_one_encoder_call_writes_a_long_list_or_a_whole_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        answer = {"rows": [{"id": n} for n in range(5000)], "meta": {"states": list(range(600))}}
        expected = json.dumps(answer, default=str)
        sizes: list[int] = []
        original = json.dumps

        def recording(value: Any, *args: Any, **kwargs: Any) -> str:
            sizes.append(len(value) if isinstance(value, list | dict) else 0)
            assert value is not answer, "the whole answer went through one encoder call"
            return original(value, *args, **kwargs)

        monkeypatch.setattr(json, "dumps", recording)
        assert answer_json(answer) == expected
        assert max(sizes) <= 256, f"an encoder call wrote {max(sizes)} items at once"


#: how much later than a thread running plain Python the event loop may run while an answer is written: one
#: encoder call's worth. Any thread running Python costs the loop a few switch intervals (5 ms each) a tick, 15-40 ms
#: on a laptop; what this rules out is one call holding the lock for a whole answer (0.9 s for the series, 2026-10-31)
LOOP_MAY_BE_LATER = 0.015


#: the most work one encoder call may do: a value, a mapping's key, or 64 characters of text each one unit
CALL_BUDGET = 8192

#: the most items of one list or mapping one encoder call may write
CALL_ITEMS = 256


def _work(value: Any) -> int:
    """the units of work writing ``value`` takes, as :data:`CALL_BUDGET` counts them."""
    if isinstance(value, str):
        return 1 + len(value) // 64
    if isinstance(value, dict):
        return 1 + len(value) + sum(_work(item) for item in value.values())
    if isinstance(value, list | tuple):
        return 1 + sum(_work(item) for item in value)
    return 1


def _series(count: int, points: int) -> dict[str, Any]:
    """a short list of long series: each item is far past one call, though the list holds few."""
    return {"series": [{"name": f"s{n}", "points": [[k, k / 3] for k in range(points)]} for n in range(count)]}


def _by_number(count: int) -> dict[str, Any]:
    """a mapping keyed by numbers, which ``json`` writes as text."""
    return {"by_fips": {n: [n, n / 2, "x"] for n in range(count)}}


def _hidden_giant() -> list[Any]:
    """small rows, and among them, where no evenly spread sample lands, one row holding a long list."""
    rows: list[Any] = [[n, f"r{n}"] for n in range(4001)]
    rows[1234] = [1234, list(range(250)), {"deep": [[k] * 200 for k in range(200)]}]
    return rows


def _random_value(rng: random.Random, depth: int, room: list[int]) -> Any:
    """a random JSON-ish value: every scalar type json writes, and lists, tuples and mappings of every key type,
    some wide enough to be written in runs, of at most ``room[0]`` values in all.
    """
    pick = rng.random()
    room[0] -= 1
    if depth <= 0 or pick < 0.45 or room[0] <= 0:
        return rng.choice(
            [
                rng.randint(-(10**12), 10**12),
                rng.random() * 1e6,
                True,
                False,
                None,
                "x" * rng.randint(0, 300),
                'é\n\u2028"',
                datetime(2026, 10, 31, tzinfo=UTC),
            ]
        )
    width = min(room[0], rng.choice([0, 1, 3, 40, 300, 700]))
    if pick < 0.7:
        items = [_random_value(rng, depth - 1, room) for _ in range(width)]
        return tuple(items) if rng.random() < 0.2 else items
    keys = [rng.choice([f"k{n}", n, n + 0.5, True, False, None]) for n in range(width)]
    return {key: _random_value(rng, depth - 1, room) for key in keys}


@pytest.fixture
def encoder_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """each encoder call's work (in ``answer_json``'s units, counted in full) and its list's or mapping's length."""
    calls: list[tuple[int, int]] = []
    original = json.dumps

    def recording(value: Any, *args: Any, **kwargs: Any) -> str:
        calls.append((_work(value), len(value) if isinstance(value, list | tuple | dict) else 0))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", recording)
    return calls


class TestAnyShapeIsWrittenInShortCalls:
    @pytest.mark.parametrize(
        "answer",
        [
            pytest.param(_series(6, 20_000), id="a short list of long series"),
            pytest.param(_by_number(20_000), id="a mapping keyed by numbers"),
            pytest.param({"rows": [{"html": "x" * 20_000, "n": n} for n in range(400)]}, id="rows of long text"),
            pytest.param({"rows": _hidden_giant()}, id="one heavy row hidden among small ones"),
            pytest.param([[list(range(200)) for _ in range(200)] for _ in range(3)], id="lists nested three deep"),
        ],
    )
    def test_no_call_passes_the_budget_and_the_text_is_what_json_dumps_writes(
        self, answer: Any, encoder_calls: list[tuple[int, int]]
    ) -> None:
        text = answer_json(answer)
        work = [each for each, _ in encoder_calls]
        lengths = [each for _, each in encoder_calls]
        assert max(work) <= CALL_BUDGET, f"an encoder call did {max(work)} units of work"
        assert max(lengths) <= CALL_ITEMS, f"an encoder call wrote {max(lengths)} items at once"
        assert text == _plain_json(answer)

    @pytest.mark.parametrize("seed", range(40))
    def test_any_answer_is_exactly_what_json_dumps_writes(self, seed: int) -> None:
        answer = _random_value(random.Random(seed), 4, [20_000])
        assert answer_json(answer) == json.dumps(answer, default=str)

    @pytest.mark.parametrize(
        "answer",
        [{}, [], (), {"a": []}, {"a": {}}, [[]] * 600, [{}] * 600, {n: {} for n in range(9000)}, [None] * 9000],
    )
    def test_empty_containers_are_written_as_json_writes_them(self, answer: Any) -> None:
        assert answer_json(answer) == json.dumps(answer, default=str)

    def test_a_non_finite_number_deep_in_a_run_is_null(self) -> None:
        answer = {"series": [{"points": [[k, float("nan") if k == 7777 else k / 2] for k in range(9000)]}] * 3}
        text = answer_json(answer)
        assert "NaN" not in text
        assert json.loads(text)["series"][2]["points"][7777] == [7777, None]

    def test_a_key_json_refuses_is_refused(self) -> None:
        with pytest.raises(TypeError):
            answer_json({("a", "tuple"): list(range(9000))})


def _plain_json(answer: Any) -> str:
    """``json.dumps(answer, default=str)``, by the encoder itself whatever a test patched."""
    return json.JSONEncoder(default=str).encode(answer)


def _busy(seconds: float) -> None:
    """plain Python for ``seconds``: what any thread running Python costs the loop."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


async def _late_while(work: Callable[..., Any], *args: Any) -> tuple[float, float, Any]:
    """how late a ticker on the loop runs at most while ``work(*args)`` runs on a worker thread, how long it ran,
    and what it gave."""
    late = 0.0
    ticking = True

    async def tick() -> None:
        nonlocal late
        while ticking:
            start = time.perf_counter()
            await asyncio.sleep(0.002)
            late = max(late, time.perf_counter() - start - 0.002)

    ticker = asyncio.create_task(tick())
    await asyncio.sleep(0.05)
    began = time.perf_counter()
    given = await asyncio.to_thread(work, *args)
    took = time.perf_counter() - began
    await asyncio.sleep(0.05)
    ticking = False
    await ticker
    return late, took, given


@pytest.mark.xdist_group("loop_timing")
class TestTheLoopRunsWhileALargeAnswerIsWritten:
    @pytest.mark.parametrize(
        "answer",
        [pytest.param(_series(10, 200_000), id="series"), pytest.param(_by_number(200_000), id="numbered")],
    )
    async def test_the_loop_runs_as_it_does_beside_any_python_thread_while_an_answer_is_written(
        self, answer: Any
    ) -> None:
        late, took, text = await _late_while(answer_json, answer)
        floor, _, _ = await _late_while(_busy, took)
        assert len(text) > 5_000_000
        assert late < floor + LOOP_MAY_BE_LATER, (
            f"the loop ran {late * 1000:.1f} ms late while the answer was written, "
            f"{floor * 1000:.1f} ms beside plain Python as long"
        )
