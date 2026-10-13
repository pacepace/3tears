"""``RecordedCompletions`` records what a client answered once and replays it exactly, refusing what it cannot serve.

Every test records through a scripted inner client, so no provider is called. The recording is read back as the
file it is (one JSON object per line), and every refusal is held to naming what was asked and the remedy.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import pytest
from pydantic import BaseModel, ConfigDict

from threetears.evals.kernel.host import ApparatusError
from threetears.evals.schema import CompletionResult, StopReason
from threetears.evals.testing import (
    CompletionConformanceFailure,
    ReplayedCompletion,
    RecordedCompletions,
    RecordedPrompt,
    RecordingRefused,
    check_completion_conformance,
)

SYSTEM = "You answer support tickets in one sentence."
SCHEMA = {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": {"type": "object"}}}


@dataclass(frozen=True, kw_only=True)
class _Reply:
    """A provider-shaped completion: the ten members, and one attribute of the host's own beyond them."""

    content: str
    input_tokens: int | None = 120
    output_tokens: int | None = 30
    cost_usd: float | None = 0.00042
    price_source: str | None = "openrouter:models@2026-10-13T09:00:00Z"
    model: str = "acme/large"
    served_model: str | None = "acme/large-2026-09"
    reasoning_tokens: int | None = 4
    temperature: float | None = 0.0
    stop_reason: StopReason = "end_turn"
    served_provider: str | None = "Acme Cloud"


# parity-with: threetears.evals.schema.CompletionClient
class _FakeInner:
    """A scripted client: each call returns the next reply, and every prompt and release is counted."""

    def __init__(self, *replies: _Reply) -> None:
        self.replies = list(replies)
        self.asked: list[tuple[str, str, dict[str, Any] | None]] = []
        self.aclose_calls = 0

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:
        self.asked.append((system, user, response_format))
        await asyncio.sleep(0)
        return self.replies.pop(0)

    async def aclose(self) -> None:
        self.aclose_calls += 1

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def _record(path: Path, *calls: tuple[str, _Reply], replace: bool = False) -> None:
    """Record one call per (user, reply), in order, through a fresh recording client."""
    async with RecordedCompletions.record(_FakeInner(*(reply for _, reply in calls)), path, replace=replace) as client:
        for user, _ in calls:
            await client.generate(system=SYSTEM, user=user)


async def test_recording_appends_one_line_per_call_and_hands_back_the_inner_completion(tmp_path: Path) -> None:
    path = tmp_path / "recordings" / "triage.jsonl"
    first, second, other = _Reply(content="Refund it."), _Reply(content="Escalate it."), _Reply(content="Close it.")
    inner = _FakeInner(first, second, other)
    client = RecordedCompletions.record(inner, path)
    assert client.mode == "record"

    got = [
        await client.generate(system=SYSTEM, user="My order never came.", response_format=SCHEMA),
        await client.generate(system=SYSTEM, user="My order never came.", response_format=SCHEMA),
        await client.generate(system=SYSTEM, user="How do I reset my password?"),
    ]

    assert got[0] is first and got[1] is second and got[2] is other, "the inner completion is handed back unchanged"
    assert inner.asked[0] == (SYSTEM, "My order never came.", SCHEMA)
    lines = _lines(path)
    assert [line["index"] for line in lines] == [0, 1, 0], "an identical prompt's calls are numbered under its key"
    assert lines[0]["key"] == lines[1]["key"] != lines[2]["key"]
    assert len(lines[0]["key"]) == 64
    assert lines[0]["result"] == {
        "content": "Refund it.",
        "input_tokens": 120,
        "output_tokens": 30,
        "cost_usd": 0.00042,
        "price_source": "openrouter:models@2026-10-13T09:00:00Z",
        "model": "acme/large",
        "served_model": "acme/large-2026-09",
        "reasoning_tokens": 4,
        "temperature": 0.0,
        "stop_reason": "end_turn",
    }
    assert lines[0]["extra"] == {"served_provider": "Acme Cloud"}
    assert lines[0]["user_head"] == "My order never came."
    assert all(line["recorder_version"] == RecordedCompletions.FORMAT_VERSION for line in lines)
    assert all(datetime.fromisoformat(line["recorded_at"]).tzinfo is not None for line in lines)


async def test_the_key_covers_the_model_and_temperature_the_completion_reports(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(
        path,
        ("Same prompt.", _Reply(content="a")),
        ("Same prompt.", _Reply(content="b", model="acme/small")),
        ("Same prompt.", _Reply(content="c", temperature=0.7)),
    )
    lines = _lines(path)
    assert len({line["key"] for line in lines}) == 3
    assert [line["index"] for line in lines] == [0, 0, 0]


async def test_concurrent_identical_calls_are_numbered_without_a_gap(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    client = RecordedCompletions.record(_FakeInner(*(_Reply(content=str(n)) for n in range(4))), path)
    await asyncio.gather(*(client.generate(system=SYSTEM, user="Roll a die.") for _ in range(4)))
    assert sorted(line["index"] for line in _lines(path)) == [0, 1, 2, 3]
    assert [line["index"] for line in _lines(path)] == [0, 1, 2, 3], "indexes are given in the order lines are written"


async def test_replay_serves_repeats_in_order_and_refuses_the_call_past_the_recording(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Roll a die.", _Reply(content="4")), ("Roll a die.", _Reply(content="2")))
    client = RecordedCompletions.replay(path)
    assert client.mode == "replay"

    served = [(await client.generate(system=SYSTEM, user="Roll a die.")).content for _ in range(2)]
    assert served == ["4", "2"]
    with pytest.raises(RecordingRefused) as refused:
        await client.generate(system=SYSTEM, user="Roll a die.")
    message = str(refused.value)
    assert "'Roll a die.'" in message and "recorded 2 time(s)" in message and "call 2" in message
    assert "RECORD=1" in message and str(path) in message
    assert refused.value.remedy in message


async def test_a_prompt_the_recording_does_not_hold_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("My order never came.", _Reply(content="Refund it.")))
    client = RecordedCompletions.replay(path, rerecord="RECORD=1 python llm_judge.py")

    long_user = "My parcel never came. " * 10
    with pytest.raises(RecordingRefused) as refused:
        await client.generate(system=SYSTEM, user=long_user)
    message = str(refused.value)
    assert repr(long_user[:80] + "…") in message, "the prompt is named by its first 80 characters"
    assert "call 0" in message and "`RECORD=1 python llm_judge.py`" in message
    assert isinstance(refused.value, ApparatusError), "a replay miss is the rig's fault, so a cell is excluded on it"

    for changed in ({"system": "Another system prompt."}, {"response_format": SCHEMA}):
        with pytest.raises(RecordingRefused, match="no call is recorded for the prompt 'My order never came.'"):
            await client.generate(**{"system": SYSTEM, "user": "My order never came.", **changed})


async def test_a_replayed_completion_is_the_recorded_ten_members_and_conforms(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    reply = _Reply(content="Refund it.", cost_usd=None, price_source=None, reasoning_tokens=None, temperature=None)
    await _record(path, ("My order never came.", reply))

    replayed = await RecordedCompletions.replay(path).generate(system=SYSTEM, user="My order never came.")

    check_completion_conformance(replayed)
    assert isinstance(replayed, ReplayedCompletion)
    assert replayed == ReplayedCompletion(
        content="Refund it.",
        input_tokens=120,
        output_tokens=30,
        cost_usd=None,
        price_source=None,
        model="acme/large",
        served_model="acme/large-2026-09",
        reasoning_tokens=None,
        temperature=None,
        stop_reason="end_turn",
        extra={"served_provider": "Acme Cloud"},
    )
    with pytest.raises(TypeError):
        replayed.extra["served_provider"] = "elsewhere"  # type: ignore[index]


async def test_a_price_is_replayed_as_recorded(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Price me.", _Reply(content="ok", cost_usd=0, temperature=0)))
    replayed = await RecordedCompletions.replay(path).generate(system=SYSTEM, user="Price me.")
    assert (replayed.cost_usd, replayed.price_source) == (0.0, "openrouter:models@2026-10-13T09:00:00Z")
    assert isinstance(replayed.cost_usd, float) and isinstance(replayed.temperature, float)


async def test_recording_onto_an_existing_file_appends_and_replace_starts_afresh(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Roll a die.", _Reply(content="4")))
    await _record(path, ("Roll a die.", _Reply(content="2")))
    assert [(line["index"], line["result"]["content"]) for line in _lines(path)] == [(0, "4"), (1, "2")]
    client = RecordedCompletions.replay(path)
    assert [(await client.generate(system=SYSTEM, user="Roll a die.")).content for _ in range(2)] == ["4", "2"]

    await _record(path, ("Roll a die.", _Reply(content="6")), replace=True)
    assert [(line["index"], line["result"]["content"]) for line in _lines(path)] == [(0, "6")]


async def test_two_recording_clients_on_one_file_number_one_key_together(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    first = RecordedCompletions.record(_FakeInner(_Reply(content="a"), _Reply(content="c")), path)
    second = RecordedCompletions.record(_FakeInner(_Reply(content="b")), path)
    await first.generate(system=SYSTEM, user="Same.")
    await second.generate(system=SYSTEM, user="Same.")
    await first.generate(system=SYSTEM, user="Same.")
    assert [line["index"] for line in _lines(path)] == [0, 1, 2]


async def test_aclose_closes_the_inner_client_recording_and_nothing_replaying(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    inner = _FakeInner(_Reply(content="ok"))
    async with RecordedCompletions.record(inner, path) as client:
        await client.generate(system=SYSTEM, user="Hello.")
    assert inner.aclose_calls == 1

    replaying = RecordedCompletions.replay(path)
    async with replaying:
        pass
    await replaying.aclose()
    assert (await replaying.generate(system=SYSTEM, user="Hello.")).content == "ok", "closing a replay releases nothing"


async def test_one_file_holding_two_models_answering_one_prompt_replays_each_by_name(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Same prompt.", _Reply(content="large")), ("Same prompt.", _Reply(content="small", model="m")))

    with pytest.raises(
        RecordingRefused, match=r"recorded under model 'acme/large' at temperature 0\.0, model 'm'"
    ) as e:
        await RecordedCompletions.replay(path).generate(system=SYSTEM, user="Same prompt.")
    assert "replay(path, model=..., temperature=...)" in e.value.remedy

    large = RecordedCompletions.replay(path, model="acme/large")
    small = RecordedCompletions.replay(path, model="m", temperature=0.0)
    assert (await large.generate(system=SYSTEM, user="Same prompt.")).content == "large"
    assert (await small.generate(system=SYSTEM, user="Same prompt.")).content == "small"
    with pytest.raises(RecordingRefused, match="under model 'm' at temperature 0.7"):
        await RecordedCompletions.replay(path, model="m", temperature=0.7).generate(system=SYSTEM, user="Same prompt.")


async def test_check_covers_names_every_prompt_the_recording_no_longer_covers(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Roll a die.", _Reply(content="4")), ("Pick a card.", _Reply(content="7")))

    RecordedCompletions.check_covers(
        path, [RecordedPrompt(SYSTEM, "Roll a die."), RecordedPrompt(SYSTEM, "Pick a card.")]
    )
    with pytest.raises(RecordingRefused) as refused:
        RecordedCompletions.check_covers(
            path,
            [
                RecordedPrompt(SYSTEM, "Roll a die."),
                RecordedPrompt(SYSTEM, "Roll a die."),
                RecordedPrompt(SYSTEM, "Pick a card, any card."),
            ],
            rerecord="RECORD=1 python dice.py",
        )
    message = str(refused.value)
    assert "no longer covers 2 prompt(s)" in message
    assert "'Roll a die.' is asked 2 time(s) and recorded 1" in message
    assert "'Pick a card, any card.' is asked 1 time(s) and recorded 0" in message
    assert message.endswith("re-record " + str(path) + " with `RECORD=1 python dice.py`, the provider's key set")


async def test_replay_refuses_a_missing_or_empty_recording_with_the_remedy(tmp_path: Path) -> None:
    with pytest.raises(RecordingRefused, match="there is no recording at this path; re-record .* with `RECORD=1"):
        RecordedCompletions.replay(tmp_path / "never.jsonl")
    (empty := tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RecordingRefused, match="holds no calls"):
        RecordedCompletions.replay(empty)


def _good_line(**overrides: Any) -> dict[str, Any]:
    line: dict[str, Any] = {
        "recorder_version": RecordedCompletions.FORMAT_VERSION,
        "recorded_at": "2026-10-13T09:00:00+00:00",
        "key": "0" * 64,
        "index": 0,
        "user_head": "Hello.",
        "result": {
            "content": "ok",
            "input_tokens": 1,
            "output_tokens": 1,
            "cost_usd": None,
            "price_source": None,
            "model": "acme/large",
            "served_model": None,
            "reasoning_tokens": None,
            "temperature": None,
            "stop_reason": "end_turn",
        },
        "extra": {},
    }
    return {**line, **overrides}


@pytest.mark.parametrize(
    ("third", "problem"),
    [
        ("{not json", "it is not JSON"),
        ("[1, 2]", "it holds a JSON list"),
        ("", "it is not JSON"),
        (json.dumps(_good_line(index=1)), "its call index is 1, and the lines before it hold 0 call(s)"),
        (json.dumps(_good_line(key="f" * 64, index=2)), "its call index is 2, and the lines before it hold 0"),
        (json.dumps({k: v for k, v in _good_line().items() if k != "extra"}), "it lacks extra"),
        (json.dumps(_good_line(note="hand-edited")), "it adds note"),
        (json.dumps(_good_line(recorder_version="1")), "its recorder_version is '1'"),
        (json.dumps(_good_line(result={**_good_line()["result"], "input_tokens": "12"})), "`input_tokens` holds '12'"),
        (json.dumps(_good_line(result={**_good_line()["result"], "stop_reason": "length"})), "`stop_reason` holds"),
        (json.dumps(_good_line(result={"content": "ok"})), "its result lacks cost_usd"),
        (json.dumps(_good_line(recorded_at="yesterday")), "its recorded_at 'yesterday' is not an ISO 8601 time"),
    ],
)
async def test_a_malformed_line_refuses_the_recording_by_its_number(tmp_path: Path, third: str, problem: str) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Hello.", _Reply(content="a")), ("Hello.", _Reply(content="b")))
    with path.open("a", encoding="utf-8") as recording:
        recording.write(third + "\n")

    with pytest.raises(RecordingRefused, match="line 3 is not a recorded call") as refused:
        RecordedCompletions.replay(path)
    assert problem in str(refused.value)
    with pytest.raises(RecordingRefused, match="line 3 is not a recorded call"):
        RecordedCompletions.record(_FakeInner(), path)


async def test_a_line_from_a_newer_recording_format_is_refused_with_the_remedy(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Hello.", _Reply(content="a")))
    with path.open("a", encoding="utf-8") as recording:
        recording.write(json.dumps(_good_line(recorder_version=RecordedCompletions.FORMAT_VERSION + 1)) + "\n")

    newer = f"line 2 was written in recording format {RecordedCompletions.FORMAT_VERSION + 1}, newer than this build's"
    with pytest.raises(RecordingRefused, match=newer) as refused:
        RecordedCompletions.replay(path)
    assert refused.value.remedy.startswith("upgrade 3tears-evals to the build that recorded it, or re-record")
    with pytest.raises(RecordingRefused, match=newer):
        RecordedCompletions.record(_FakeInner(), path)


async def test_a_completion_the_recording_could_not_replay_is_refused_when_recorded(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"

    @dataclass(frozen=True, kw_only=True)
    class _WithRawMessage(_Reply):
        raw: object = None

    cases: list[tuple[CompletionResult, str]] = [
        (_WithRawMessage(content="ok", raw=object()), "`raw` holds a value of type object, which JSON cannot hold"),
        (replace(_Reply(content="ok"), input_tokens="12"), "`input_tokens` holds '12' (str)"),  # type: ignore[arg-type]
        (replace(_Reply(content="ok"), stop_reason="length"), "stop_reason 'length' is not one of"),  # type: ignore[arg-type]
    ]
    for completion, problem in cases:
        inner = _FakeInner()
        inner.replies.append(completion)  # type: ignore[arg-type]
        with pytest.raises(CompletionConformanceFailure, match=re.escape(problem)):
            await RecordedCompletions.record(inner, path).generate(system=SYSTEM, user="Hello.")
    assert not path.exists(), "nothing is written for a completion the recording refused"


class _PlainReply:
    """A completion as a plain object: its instance attributes are what it declares."""

    def __init__(self, **members: Any) -> None:
        for name, value in members.items():
            setattr(self, name, value)


class _ModelReply(BaseModel):
    """A completion as a frozen Pydantic model: its fields are what it declares."""

    model_config = ConfigDict(frozen=True)

    content: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    price_source: str | None
    model: str
    served_model: str | None
    reasoning_tokens: int | None
    temperature: float | None
    stop_reason: StopReason
    served_provider: str


async def test_every_attribute_a_completion_type_declares_beyond_the_ten_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    ten = {name: value for name, value in vars(_Reply(content="ok")).items() if name != "served_provider"}
    plain = _PlainReply(**ten, served_provider="Plain Cloud", _cache="internal")
    model = _ModelReply(**ten, served_provider="Model Cloud")
    await _record(path, ("Plain.", plain), ("Model.", model))  # type: ignore[arg-type]

    assert [line["extra"] for line in _lines(path)] == [
        {"served_provider": "Plain Cloud"},
        {"served_provider": "Model Cloud"},
    ]
    replayed = await RecordedCompletions.replay(path).generate(system=SYSTEM, user="Model.")
    assert isinstance(replayed, ReplayedCompletion)
    assert replayed.extra == {"served_provider": "Model Cloud"}
