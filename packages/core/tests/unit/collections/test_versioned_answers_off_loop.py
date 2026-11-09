"""an answer is compressed on a worker thread, never on the event loop, and written as JSON a piece at a time.

A large answer takes tens of milliseconds to gzip, and every other request on the pod waits for the loop while
it does. So ``VersionedAnswers`` compresses on a worker thread, both where it stores an answer and where it
answers one uncached; ``answer_json`` writes ``json.dumps``' exact text in calls short enough that the loop runs
between them, whatever the answer's shape: a short list of long series, a mapping keyed by numbers, rows of long
text (one call of 0.5 s and one of 50 ms on such answers before, 2026-10-31).
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import os
import random
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import MetaData

from threetears.core.cache.sqlite import SQLiteBackend
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


#: the type ``answer_json`` returns, a ``bytearray`` whose pieces are added as they are written
_Answer = type(answer_json(None))


class TestTheAnswerIsNeverCopiedWhole:
    """a ``"".join`` of the pieces into one ``str``, and its ``encode`` for gzip, each held the interpreter lock for
    one copy of the whole answer: 32-40 ms for 51 MB on a 4-core CI runner (2026-10-31)."""

    def test_each_piece_is_added_as_it_is_written(self, monkeypatch: pytest.MonkeyPatch) -> None:
        answer = _series(6, 20_000)
        added: list[int] = []
        original = _Answer.append

        def recording(self: _Answer, piece: str) -> None:
            added.append(len(piece))
            original(self, piece)

        monkeypatch.setattr(_Answer, "append", recording)
        text = answer_json(answer)
        assert sum(added) == len(text)
        assert max(added) * 100 < len(text), f"one piece of {max(added)} characters in an answer of {len(text)}"

    def test_it_reads_as_the_text_it_holds(self) -> None:
        answer = {"rows": [{"n": n, "s": "é\n"} for n in range(3000)], "nan": float("nan")}
        text = answer_json(answer)
        plain = json.dumps({**answer, "nan": None}, default=str)
        assert isinstance(text, bytearray)
        assert text == plain and not text != plain
        assert text != plain[:-1] + " " and text != plain + " "
        assert len(text) == len(plain)
        assert '"n": 2999' in text and "é" not in text and "NaN" not in text
        assert str(text) == plain
        assert json.loads(text) == json.loads(plain)
        assert text.encode("utf-8") is text and text.encode("ascii") is text
        assert text.encode("utf-16") == plain.encode("utf-16")
        assert bytes(text) == plain.encode("ascii")

    async def test_it_is_gzipped_as_it_is_to_the_same_bytes_as_its_text(self) -> None:
        answer = _series(3, 5_000)
        expected = gzip.compress(json.dumps(answer).encode("utf-8"), mtime=0)
        assert await _gz(answer_json(answer)) == expected

    async def test_a_computed_answer_json_is_stored_and_given(self) -> None:
        answer = _series(3, 5_000)

        async def compute() -> _Answer:
            return answer_json(answer)

        answers = _answers()
        given = await answers.answer_with_outcome("v1", "series", compute, order=1)
        assert given.outcome is AnswerOutcome.COMPUTED
        assert gzip.decompress(given.body) == json.dumps(answer).encode("utf-8")
        again = await answers.answer_with_outcome("v1", "series", compute, order=1)
        assert again.outcome is AnswerOutcome.READ and again.body == given.body


async def _gz(text: _Answer) -> bytes:
    """the answer's gzip, as the answers write it."""
    answers = _answers()

    async def compute() -> _Answer:
        return text

    await answers.retire_older_than(5)
    return (await answers.answer_with_outcome("v1", "page", compute, order=1)).body


#: how much later than a thread running plain Python the event loop may run while an answer is written: one
#: encoder call's worth. Any thread running Python costs the loop a few switch intervals (5 ms each) a tick, 15-40 ms
#: on a laptop; what this rules out is one call holding the lock for a whole answer (0.9 s for the series, 2026-10-31)
LOOP_MAY_BE_LATER = 0.015

#: how many times a loop-timing bound is measured, each time the work's lateness and then the floor beside plain
#: Python for as long, back to back. A runner's other batches add to both only while they run, so a measurement taken
#: while one started beside it can miss by their load, not the code's; the bound holds when one pair meets it. A
#: call holding the lock for a whole answer misses in every pair (0.9 s against 15 ms)
TIMING_PAIRS = 3


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


@pytest.mark.loop_timing
class TestTheLoopRunsWhileALargeAnswerIsWritten:
    @pytest.mark.parametrize(
        "answer",
        [pytest.param(_series(10, 200_000), id="series"), pytest.param(_by_number(200_000), id="numbered")],
    )
    async def test_the_loop_runs_as_it_does_beside_any_python_thread_while_an_answer_is_written(
        self, answer: Any
    ) -> None:
        pairs: list[tuple[float, float]] = []
        for _ in range(TIMING_PAIRS):
            late, took, text = await _late_while(answer_json, answer)
            floor, _, _ = await _late_while(_busy, took)
            assert len(text) > 5_000_000
            pairs.append((late, floor))
            if late < floor + LOOP_MAY_BE_LATER:
                break
        assert any(late < floor + LOOP_MAY_BE_LATER for late, floor in pairs), (
            "the loop ran late while the answer was written, in every pair: "
            + "; ".join(f"{late * 1000:.1f} ms, {floor * 1000:.1f} ms beside plain Python" for late, floor in pairs)
        )


class TestARowIsStoredAndReadAsJsonWrites:
    """the row's L2 bytes are ``json.dumps``' of the row, written on a worker thread and read by their layout."""

    async def test_a_computed_row_is_stored_as_json_dumps_writes_it(self) -> None:
        answer = _series(3, 5_000)
        stored: list[bytes] = []
        answers = _answers()
        original = answers.serialize

        def recording(data: dict[str, Any]) -> bytes:
            written = original(data)
            stored.append(written)
            return written

        answers.serialize = recording  # type: ignore[method-assign]

        async def compute() -> bytearray:
            return answer_json(answer)

        given = await answers.answer_with_outcome("v1", "series", compute, order=1)
        assert given.outcome is AnswerOutcome.COMPUTED
        assert stored, "no row was stored"
        row = json.loads(stored[-1])
        assert stored[-1] == json.dumps({key: row[key] for key in ("version", "request", "body")}).encode("utf-8")
        assert gzip.decompress(base64.b64decode(row["body"])) == json.dumps(answer).encode("utf-8")
        assert given.body == base64.b64decode(row["body"])

    @pytest.mark.parametrize(
        "row",
        [
            {"version": "v1", "request": "ab" * 32, "body": base64.b64encode(os.urandom(3 << 20)).decode("ascii")},
            {"version": "v1", "request": "ab", "body": ""},
            {"version": "v1", "request": "ab", "body": 'a "quoted" \\ body'},
            {"version": "v1", "request": "ab", "body": "é"},
            {"version": "v1", "request": "ab", "body": "x", "extra": 1},
            {"request": "ab", "version": "v1", "body": "x"},
            {"version": 'a", "body": "b', "request": "ab", "body": "x"},
        ],
    )
    def test_any_row_reads_back_as_json_loads_reads_it(self, row: dict[str, Any]) -> None:
        stored = json.dumps(row).encode("utf-8")
        assert _answers().deserialize(stored) == json.loads(stored)

    def test_a_row_json_refuses_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _answers().deserialize(b'{"version": "v1", "request": , "body": "x"}')


def _large_answer_text() -> str:
    """an answer whose gzip is megabytes (random hex compresses about half), as a large cached view's is."""
    return json.dumps({"blob": random.Random(7).randbytes(8_000_000).hex()})


async def _late_during(work: Callable[[], Awaitable[Any]]) -> tuple[float, float, Any]:
    """how late a ticker on the loop runs at most while ``work()`` is awaited on the loop, how long it took, and
    what it gave."""
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
    given = await work()
    took = time.perf_counter() - began
    await asyncio.sleep(0.05)
    ticking = False
    await ticker
    return late, took, given


@pytest.mark.loop_timing
class TestTheLoopRunsWhileALargeAnswerIsStoredAndRead:
    async def test_storing_and_reading_a_large_answer_holds_the_loop_no_longer_than_any_python_thread(self) -> None:
        text = _large_answer_text()

        def replica(bus: FakeNatsClient, *, l1: bool = False) -> VersionedAnswers:
            registry = CollectionRegistry()
            backend: Any = None
            if l1:
                backend = SQLiteBackend(db_name=f"offloop-l1-{uuid.uuid4().hex}")
                backend.initialize(MetaData())
            registry.configure(l1_backend=backend, l2_client=bus, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
            return VersionedAnswers(registry, DefaultCoreConfig(), bus, table_name="offloop_large_answers")

        async def compute() -> str:
            return text

        expected = text.encode("utf-8")
        steps = ("stored", "read from L2", "read from L1")
        pairs: dict[str, list[tuple[float, float]]] = {name: [] for name in steps}
        for _ in range(TIMING_PAIRS):
            # a fresh bus and replicas each time, so each is stored, read from L2 and read from L1 again. Both keep
            # an L1, as a serving pod does: the writer keeps the answer it stored
            bus = FakeNatsClient()
            writer, reader = replica(bus, l1=True), replica(bus, l1=True)
            for name, answers, outcome in (
                (steps[0], writer, AnswerOutcome.COMPUTED),
                (steps[1], reader, AnswerOutcome.READ),
                (steps[2], reader, AnswerOutcome.READ),
            ):
                late, took, given = await _late_during(
                    lambda answers=answers: answers.answer_with_outcome("v1", "large", compute, order=1)
                )
                floor, _, _ = await _late_while(_busy, took)
                assert given.outcome is outcome
                assert len(given.body) > 4_000_000 and gzip.decompress(given.body) == expected
                pairs[name].append((late, floor))
            if all(any(late < floor + LOOP_MAY_BE_LATER for late, floor in pairs[name]) for name in steps):
                break
        for name in steps:
            assert any(late < floor + LOOP_MAY_BE_LATER for late, floor in pairs[name]), (
                f"the loop ran late while a large answer was {name}, in every pair: "
                + "; ".join(
                    f"{late * 1000:.1f} ms, {floor * 1000:.1f} ms beside plain Python" for late, floor in pairs[name]
                )
            )


class TestAWorkerLetsTheLoopRunBetweenSlices:
    """a worker encoding or decoding a large answer pauses between its slices, for longer than zero, so the event
    loop's thread takes the interpreter lock then rather than a switch interval later; a small answer never pauses,
    and either way the answer given is exactly the one stored."""

    @staticmethod
    def _replica(bus: FakeNatsClient, *, l1: bool = False) -> VersionedAnswers:
        registry = CollectionRegistry()
        backend: Any = None
        if l1:
            backend = SQLiteBackend(db_name=f"pauses-l1-{uuid.uuid4().hex}")
            backend.initialize(MetaData())
        registry.configure(l1_backend=backend, l2_client=bus, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
        return VersionedAnswers(registry, DefaultCoreConfig(), bus, table_name="pauses_answers")

    async def _pauses_storing_then_reading(self, text: str, monkeypatch: pytest.MonkeyPatch) -> tuple[list, list]:
        bus = FakeNatsClient()
        writer, reader = self._replica(bus), self._replica(bus, l1=True)

        async def compute() -> str:
            return text

        pauses: list[float] = []
        monkeypatch.setattr(time, "sleep", pauses.append)
        stored = await writer.answer_with_outcome("v1", "q", compute, order=1)
        storing, pauses[:] = list(pauses), []
        read = await reader.answer_with_outcome("v1", "q", compute, order=1)
        assert stored.outcome is AnswerOutcome.COMPUTED and read.outcome is AnswerOutcome.READ
        assert gzip.decompress(stored.body) == gzip.decompress(read.body) == text.encode("utf-8")
        return storing, list(pauses)

    async def test_a_large_answer_pauses_between_slices_when_stored_and_when_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        storing, reading = await self._pauses_storing_then_reading(_large_answer_text(), monkeypatch)
        assert storing and reading, f"stored with {len(storing)} pauses, read with {len(reading)}"
        assert all(pause > 0 for pause in storing + reading), "a zero sleep hands the lock back to the worker"

    async def test_a_small_answer_never_pauses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        storing, reading = await self._pauses_storing_then_reading(json.dumps({"small": 1}), monkeypatch)
        assert storing == [] and reading == []


def _l1_replica(bus: FakeNatsClient) -> VersionedAnswers:
    """a replica on ``bus`` keeping answers in its own SQLite L1."""
    registry = CollectionRegistry()
    backend = SQLiteBackend(db_name=f"kept-l1-{uuid.uuid4().hex}")
    backend.initialize(MetaData())
    registry.configure(l1_backend=backend, l2_client=bus, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    return VersionedAnswers(registry, DefaultCoreConfig(), bus, table_name="kept_answers")


def _l2_replica(bus: FakeNatsClient) -> VersionedAnswers:
    """a replica on ``bus`` keeping answers in L2 alone."""
    registry = CollectionRegistry()
    registry.configure(l1_backend=None, l2_client=bus, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    return VersionedAnswers(registry, DefaultCoreConfig(), bus, table_name="kept_answers")


async def _stored(text: str) -> tuple[FakeNatsClient, Callable[[], Awaitable[str]]]:
    """a bus holding ``text`` as the answer to ``"q"`` at ``v1``, stored by a peer, and the compute that gave it."""
    bus = FakeNatsClient()

    async def compute() -> str:
        return text

    stored = await _l2_replica(bus).answer_with_outcome("v1", "q", compute, order=1)
    assert stored.outcome is AnswerOutcome.COMPUTED
    return bus, compute


class TestALargeAnswerReadFromL2IsDecodedAndKeptOffTheLoop:
    """reading a large answer a peer stored decodes its row and writes it into this replica's L1 on worker threads:
    on the loop the two held it 4 ms for 12.5 MB on a laptop, and four to five times that on a CI runner (2026-11-07).
    A small answer's row is decoded and kept on the loop, as any collection's is."""

    @pytest.fixture
    def threads(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[int]]:
        """the thread each answer row was decoded on, and each answer row was written into L1 on."""
        seen: dict[str, list[int]] = {"decoded": [], "kept": []}
        decode = VersionedAnswers.deserialize
        upsert = SQLiteBackend.upsert

        def decoding(self: VersionedAnswers, data: bytes) -> dict[str, Any]:
            row = decode(self, data)
            if "body" in row:
                seen["decoded"].append(threading.get_ident())
            return row

        def keeping(self: SQLiteBackend, table: str, data: dict[str, Any], primary_key: Any = "id") -> None:
            if "body" in data:
                seen["kept"].append(threading.get_ident())
            upsert(self, table, data, primary_key)

        monkeypatch.setattr(VersionedAnswers, "deserialize", decoding)
        monkeypatch.setattr(SQLiteBackend, "upsert", keeping)
        return seen

    async def test_a_large_answer_is_decoded_and_kept_on_worker_threads(self, threads: dict[str, list[int]]) -> None:
        text = _large_answer_text()
        bus, compute = await _stored(text)
        reader = _l1_replica(bus)
        threads["decoded"].clear()
        given = await reader.answer_with_outcome("v1", "q", compute, order=1)
        assert given.outcome is AnswerOutcome.READ and gzip.decompress(given.body) == text.encode("utf-8")
        assert threads["decoded"] and threads["kept"], f"decoded {threads['decoded']}, kept {threads['kept']}"
        loop = threading.get_ident()
        assert loop not in threads["decoded"], "the large answer's row was decoded on the event loop"
        assert loop not in threads["kept"], "the large answer's row was written into L1 on the event loop"
        assert reader.get_row_sync(reader.key_of("v1", "q")) is not None, "the answer was not kept in L1"

    async def test_a_small_answer_is_decoded_and_kept_on_the_loop(self, threads: dict[str, list[int]]) -> None:
        bus, compute = await _stored(json.dumps({"small": 1}))
        reader = _l1_replica(bus)
        threads["decoded"].clear()
        given = await reader.answer_with_outcome("v1", "q", compute, order=1)
        assert given.outcome is AnswerOutcome.READ
        loop = threading.get_ident()
        assert threads["decoded"] == [loop] and threads["kept"] == [loop]


class TestAnEvictionDuringTheOffLoopWriteLeavesNoCopy:
    """a retirement here, or a peer's broadcast evicting the key, that lands while a large answer is being written
    into L1 on a worker thread leaves no copy of it there: the eviction's delete can run before the write lands, so
    the read deletes its copy again once it sees the key moved on."""

    @pytest.mark.parametrize("eviction", ["retired here", "evicted by a peer's broadcast"])
    async def test_no_copy_is_left(self, eviction: str, monkeypatch: pytest.MonkeyPatch) -> None:
        text = _large_answer_text()
        bus, compute = await _stored(text)
        reader = _l1_replica(bus)
        writing, evicted = threading.Event(), threading.Event()
        upsert = SQLiteBackend.upsert
        loop = threading.get_ident()

        def held(self: SQLiteBackend, table: str, data: dict[str, Any], primary_key: Any = "id") -> None:
            if "body" in data and threading.get_ident() != loop:
                # the eviction lands, and deletes, before this write does
                writing.set()
                assert evicted.wait(10), "the eviction never landed"
            upsert(self, table, data, primary_key)

        monkeypatch.setattr(SQLiteBackend, "upsert", held)
        reading = asyncio.create_task(reader.answer_with_outcome("v1", "q", compute, order=1))
        while not writing.is_set() and not reading.done():
            await asyncio.sleep(0.001)
        assert writing.is_set(), "the answer was not written into L1 off the event loop"
        key = reader.key_of("v1", "q")
        if eviction == "retired here":
            assert await reader.retire_older_than(2) == 1
        else:
            assert reader.evict_from_cache_sync(key)
        evicted.set()
        given = await reading
        assert given.outcome is AnswerOutcome.READ and gzip.decompress(given.body) == text.encode("utf-8")
        assert reader.get_row_sync(key) is None, f"L1 still holds the answer {eviction} while it was written"


class TestALargeAnswerStoredIsKeptOffTheLoop:
    """a replica that computes a large answer writes the row its compare-and-swap stored into its own L1 on a
    worker thread, and an eviction that lands while it does leaves no copy; a small answer is kept on the loop."""

    @pytest.fixture
    def kept(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """the thread each answer row was written into L1 on."""
        threads: list[int] = []
        upsert = SQLiteBackend.upsert

        def keeping(self: SQLiteBackend, table: str, data: dict[str, Any], primary_key: Any = "id") -> None:
            if "body" in data:
                threads.append(threading.get_ident())
            upsert(self, table, data, primary_key)

        monkeypatch.setattr(SQLiteBackend, "upsert", keeping)
        return threads

    async def test_a_large_answer_stored_is_kept_on_a_worker_thread(self, kept: list[int]) -> None:
        text = _large_answer_text()

        async def compute() -> str:
            return text

        answers = _l1_replica(FakeNatsClient())
        given = await answers.answer_with_outcome("v1", "q", compute, order=1)
        assert given.outcome is AnswerOutcome.COMPUTED and gzip.decompress(given.body) == text.encode("utf-8")
        assert kept, "the stored answer was not kept in L1"
        assert threading.get_ident() not in kept, "the stored answer was written into L1 on the event loop"
        assert answers.get_row_sync(answers.key_of("v1", "q")) is not None

    async def test_a_small_answer_stored_is_kept_on_the_loop(self, kept: list[int]) -> None:
        async def compute() -> str:
            return json.dumps({"small": 1})

        answers = _l1_replica(FakeNatsClient())
        assert (await answers.answer_with_outcome("v1", "q", compute, order=1)).outcome is AnswerOutcome.COMPUTED
        assert kept and set(kept) == {threading.get_ident()}

    @pytest.mark.parametrize("eviction", ["retired here", "evicted by a peer's broadcast"])
    async def test_an_eviction_during_the_write_leaves_no_copy(
        self, eviction: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = _large_answer_text()

        async def compute() -> str:
            return text

        answers = _l1_replica(FakeNatsClient())
        writing, evicted = threading.Event(), threading.Event()
        upsert = SQLiteBackend.upsert
        loop = threading.get_ident()

        def held(self: SQLiteBackend, table: str, data: dict[str, Any], primary_key: Any = "id") -> None:
            if "body" in data and threading.get_ident() != loop:
                # the eviction lands, and deletes, before this write does
                writing.set()
                assert evicted.wait(10), "the eviction never landed"
            upsert(self, table, data, primary_key)

        monkeypatch.setattr(SQLiteBackend, "upsert", held)
        storing = asyncio.create_task(answers.answer_with_outcome("v1", "q", compute, order=1))
        while not writing.is_set() and not storing.done():
            await asyncio.sleep(0.001)
        assert writing.is_set(), "the stored answer was not written into L1 off the event loop"
        key = answers.key_of("v1", "q")
        if eviction == "retired here":
            assert await answers.retire_older_than(2) == 1
        else:
            assert answers.evict_from_cache_sync(key)
        evicted.set()
        given = await storing
        assert gzip.decompress(given.body) == text.encode("utf-8")
        assert answers.get_row_sync(key) is None, f"L1 still holds the answer {eviction} while it was written"
