"""an answer is compressed on a worker thread, never on the event loop, and written as JSON a piece at a time.

A large answer takes tens of milliseconds to gzip, and every other request on the pod waits for the loop while
it does. So ``VersionedAnswers`` compresses on a worker thread, both where it stores an answer and where it
answers one uncached; ``answer_json`` writes ``json.dumps``' exact text in calls short enough that the loop runs
between them.
"""

from __future__ import annotations

import gzip
import json
import threading
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
