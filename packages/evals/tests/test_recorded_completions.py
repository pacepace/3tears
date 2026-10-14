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

from threetears.evals.kernel import OutOfRunBudget, PlannedCall
from threetears.evals.kernel.host import ApparatusError
from threetears.evals.schema import ClientRequestSettings, CompletionResult, OutOfRunSpend, StopReason
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

#: The settings a reasoning judge is built with, and the same judge asked for reasoning by an effort level instead.
BUDGETED = ClientRequestSettings(max_tokens=16000, reasoning_max_tokens=8192, strict_output=True)
BY_EFFORT = ClientRequestSettings(max_tokens=16000, reasoning_effort="high", strict_output=True)


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


async def _record(
    path: Path,
    *calls: tuple[str, _Reply],
    settings: ClientRequestSettings | None = BUDGETED,
    replace: bool = False,
) -> None:
    """Record one call per (user, reply), in order, through a fresh recording client built with ``settings``."""
    inner = _FakeInner(*(reply for _, reply in calls))
    async with RecordedCompletions.record(inner, path, settings=settings, replace=replace) as client:
        for user, _ in calls:
            await client.generate(system=SYSTEM, user=user)


async def test_recording_appends_one_line_per_call_and_hands_back_the_inner_completion(tmp_path: Path) -> None:
    path = tmp_path / "recordings" / "triage.jsonl"
    first, second, other = _Reply(content="Refund it."), _Reply(content="Escalate it."), _Reply(content="Close it.")
    inner = _FakeInner(first, second, other)
    client = RecordedCompletions.record(inner, path, settings=BUDGETED)
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
    assert all(line["settings"] == BUDGETED.model_dump(mode="json") for line in lines), "every line carries them"
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
    client = RecordedCompletions.record(
        _FakeInner(*(_Reply(content=str(n)) for n in range(4))), path, settings=BUDGETED
    )
    await asyncio.gather(*(client.generate(system=SYSTEM, user="Roll a die.") for _ in range(4)))
    assert sorted(line["index"] for line in _lines(path)) == [0, 1, 2, 3]
    assert [line["index"] for line in _lines(path)] == [0, 1, 2, 3], "indexes are given in the order lines are written"


async def test_replay_serves_repeats_in_order_and_refuses_the_call_past_the_recording(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Roll a die.", _Reply(content="4")), ("Roll a die.", _Reply(content="2")))
    client = RecordedCompletions.replay(path, settings=BUDGETED)
    assert client.mode == "replay"

    served = [(await client.generate(system=SYSTEM, user="Roll a die.")).content for _ in range(2)]
    assert served == ["4", "2"]
    with pytest.raises(RecordingRefused) as refused:
        await client.generate(system=SYSTEM, user="Roll a die.")
    message = str(refused.value)
    assert "'Roll a die.'" in message and "recorded 2 time(s)" in message and "call 2" in message
    assert f"re-record {path} with the client it stands in for" in message, "no command given: the generic remedy"
    assert "RECORD=1" not in message, "no convention is named that the caller did not pass"
    assert refused.value.remedy in message


async def test_a_prompt_the_recording_does_not_hold_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("My order never came.", _Reply(content="Refund it.")))
    client = RecordedCompletions.replay(path, settings=BUDGETED, rerecord="RECORD=1 python llm_judge.py")

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

    replayed = await RecordedCompletions.replay(path, settings=BUDGETED).generate(
        system=SYSTEM, user="My order never came."
    )

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
        sent_settings=None,
        extra={"served_provider": "Acme Cloud"},
    )
    with pytest.raises(TypeError):
        replayed.extra["served_provider"] = "elsewhere"  # type: ignore[index]


async def test_a_truncated_completion_round_trips_as_truncated(tmp_path: Path) -> None:
    """Output cut short for an unnamed cause is recorded and replayed as such, never as a failed call or a finish."""
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Summarise the thread.", _Reply(content='{"summary": "The cust', stop_reason="truncated")))

    assert _lines(path)[0]["result"]["stop_reason"] == "truncated"
    replayed = await RecordedCompletions.replay(path, settings=BUDGETED).generate(
        system=SYSTEM, user="Summarise the thread."
    )
    check_completion_conformance(replayed)
    assert (replayed.stop_reason, replayed.content) == ("truncated", '{"summary": "The cust')


async def test_a_price_is_replayed_as_recorded(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Price me.", _Reply(content="ok", cost_usd=0, temperature=0)))
    replayed = await RecordedCompletions.replay(path, settings=BUDGETED).generate(system=SYSTEM, user="Price me.")
    assert (replayed.cost_usd, replayed.price_source) == (0.0, "openrouter:models@2026-10-13T09:00:00Z")
    assert isinstance(replayed.cost_usd, float) and isinstance(replayed.temperature, float)


async def test_recording_onto_an_existing_file_appends_and_replace_starts_afresh(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Roll a die.", _Reply(content="4")))
    await _record(path, ("Roll a die.", _Reply(content="2")))
    assert [(line["index"], line["result"]["content"]) for line in _lines(path)] == [(0, "4"), (1, "2")]
    client = RecordedCompletions.replay(path, settings=BUDGETED)
    assert [(await client.generate(system=SYSTEM, user="Roll a die.")).content for _ in range(2)] == ["4", "2"]

    await _record(path, ("Roll a die.", _Reply(content="6")), replace=True)
    assert [(line["index"], line["result"]["content"]) for line in _lines(path)] == [(0, "6")]


async def test_two_recording_clients_on_one_file_number_one_key_together(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    first = RecordedCompletions.record(_FakeInner(_Reply(content="a"), _Reply(content="c")), path, settings=BUDGETED)
    second = RecordedCompletions.record(_FakeInner(_Reply(content="b")), path, settings=BUDGETED)
    await first.generate(system=SYSTEM, user="Same.")
    await second.generate(system=SYSTEM, user="Same.")
    await first.generate(system=SYSTEM, user="Same.")
    assert [line["index"] for line in _lines(path)] == [0, 1, 2]


async def test_aclose_closes_the_inner_client_recording_and_nothing_replaying(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    inner = _FakeInner(_Reply(content="ok"))
    async with RecordedCompletions.record(inner, path, settings=BUDGETED) as client:
        await client.generate(system=SYSTEM, user="Hello.")
    assert inner.aclose_calls == 1

    replaying = RecordedCompletions.replay(path, settings=BUDGETED)
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
        await RecordedCompletions.replay(path, settings=BUDGETED).generate(system=SYSTEM, user="Same prompt.")
    assert "replay(path, model=..., temperature=...)" in e.value.remedy

    large = RecordedCompletions.replay(path, settings=BUDGETED, model="acme/large")
    small = RecordedCompletions.replay(path, settings=BUDGETED, model="m", temperature=0.0)
    assert (await large.generate(system=SYSTEM, user="Same prompt.")).content == "large"
    assert (await small.generate(system=SYSTEM, user="Same prompt.")).content == "small"
    with pytest.raises(RecordingRefused, match="under model 'm' at temperature 0.7"):
        await RecordedCompletions.replay(path, settings=BUDGETED, model="m", temperature=0.7).generate(
            system=SYSTEM, user="Same prompt."
        )


async def test_check_covers_names_every_prompt_the_recording_no_longer_covers(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Roll a die.", _Reply(content="4")), ("Pick a card.", _Reply(content="7")))

    RecordedCompletions.check_covers(
        path, [RecordedPrompt(SYSTEM, "Roll a die."), RecordedPrompt(SYSTEM, "Pick a card.")], settings=BUDGETED
    )
    with pytest.raises(RecordingRefused) as refused:
        RecordedCompletions.check_covers(
            path,
            [
                RecordedPrompt(SYSTEM, "Roll a die."),
                RecordedPrompt(SYSTEM, "Roll a die."),
                RecordedPrompt(SYSTEM, "Pick a card, any card."),
            ],
            settings=BUDGETED,
            rerecord="RECORD=1 python dice.py",
        )
    message = str(refused.value)
    assert "no longer covers 2 prompt(s)" in message
    assert "'Roll a die.' is asked 2 time(s) and recorded 1" in message
    assert "'Pick a card, any card.' is asked 1 time(s) and recorded 0" in message
    assert message.endswith("re-record " + str(path) + " with `RECORD=1 python dice.py`, the provider's key set")


async def test_check_covers_counts_prompts_that_carry_a_response_format(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    inner = _FakeInner(_Reply(content="{}"), _Reply(content="{}"))
    async with RecordedCompletions.record(inner, path, settings=BUDGETED) as client:
        for _ in range(2):
            await client.generate(system=SYSTEM, user="Shape it.", response_format=SCHEMA)
    reordered = json.loads(json.dumps(SCHEMA, sort_keys=True))  # the same directive, its keys in another order

    RecordedCompletions.check_covers(
        path,
        [RecordedPrompt(SYSTEM, "Shape it.", SCHEMA), RecordedPrompt(SYSTEM, "Shape it.", reordered)],
        settings=BUDGETED,
    )
    with pytest.raises(RecordingRefused, match="'Shape it.' is asked 3 time\\(s\\) and recorded 2"):
        RecordedCompletions.check_covers(path, [RecordedPrompt(SYSTEM, "Shape it.", SCHEMA)] * 3, settings=BUDGETED)
    with pytest.raises(RecordingRefused, match="'Shape it.' is asked 1 time\\(s\\) and recorded 0"):
        RecordedCompletions.check_covers(path, [RecordedPrompt(SYSTEM, "Shape it.")], settings=BUDGETED)


async def test_replay_refuses_a_missing_or_empty_recording_with_the_remedy(tmp_path: Path) -> None:
    with pytest.raises(RecordingRefused, match="there is no recording at this path; re-record .* with the client it"):
        RecordedCompletions.replay(tmp_path / "never.jsonl", settings=BUDGETED)
    (empty := tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RecordingRefused, match="holds no calls"):
        RecordedCompletions.replay(empty, settings=BUDGETED)


def _good_line(**overrides: Any) -> dict[str, Any]:
    line: dict[str, Any] = {
        "recorder_version": RecordedCompletions.FORMAT_VERSION,
        "recorded_at": "2026-10-13T09:00:00+00:00",
        "key": "0" * 64,
        "index": 0,
        "user_head": "Hello.",
        "settings": None,
        "sent_settings": None,
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
        (json.dumps({k: v for k, v in _good_line().items() if k != "settings"}), "it lacks settings"),
        (json.dumps(_good_line(settings="high")), "its settings are a JSON str, not an object or null"),
        (
            json.dumps(_good_line(settings={"max_tokens": "16000"})),
            "its settings are not request settings this build reads",
        ),
        (
            json.dumps(
                _good_line(settings={"max_tokens": 16000, "reasoning_max_tokens": 8192, "reasoning_effort": "high"})
            ),
            "not both",
        ),
        (json.dumps(_good_line(settings={"max_tokens": 16000, "seed": 7})), "seed"),
        (json.dumps({k: v for k, v in _good_line().items() if k != "sent_settings"}), "it lacks sent_settings"),
        (json.dumps(_good_line(sent_settings="high")), "its sent_settings are a JSON str, not an object or null"),
        (
            json.dumps(_good_line(sent_settings={"max_tokens": 0})),
            "its sent_settings are not request settings this build reads",
        ),
    ],
)
async def test_a_malformed_line_refuses_the_recording_by_its_number(tmp_path: Path, third: str, problem: str) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Hello.", _Reply(content="a")), ("Hello.", _Reply(content="b")))
    with path.open("a", encoding="utf-8") as recording:
        recording.write(third + "\n")

    with pytest.raises(RecordingRefused, match="line 3 is not a recorded call") as refused:
        RecordedCompletions.replay(path, settings=BUDGETED)
    assert problem in str(refused.value)
    with pytest.raises(RecordingRefused, match="line 3 is not a recorded call"):
        RecordedCompletions.record(_FakeInner(), path, settings=BUDGETED)


async def test_a_line_from_a_newer_recording_format_is_refused_with_the_remedy(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Hello.", _Reply(content="a")))
    with path.open("a", encoding="utf-8") as recording:
        recording.write(json.dumps(_good_line(recorder_version=RecordedCompletions.FORMAT_VERSION + 1)) + "\n")

    newer = f"line 2 was written in recording format {RecordedCompletions.FORMAT_VERSION + 1}, newer than this build's"
    with pytest.raises(RecordingRefused, match=newer) as refused:
        RecordedCompletions.replay(path, settings=BUDGETED)
    assert refused.value.remedy.startswith("upgrade 3tears-evals to the build that recorded it, or re-record")
    with pytest.raises(RecordingRefused, match=newer):
        RecordedCompletions.record(_FakeInner(), path, settings=BUDGETED)


async def test_a_line_from_an_older_recording_format_is_refused_by_name(tmp_path: Path) -> None:
    """Format 1 kept no sent settings, so its calls cannot say what they were sent with: re-recorded, never guessed."""
    path = tmp_path / "rec.jsonl"
    older = {k: v for k, v in _good_line(recorder_version=1).items() if k != "sent_settings"}
    path.write_text(json.dumps(older) + "\n", encoding="utf-8")

    with pytest.raises(
        RecordingRefused, match="line 1 was written in recording format 1, older than this build's 2"
    ) as refused:
        RecordedCompletions.replay(path, settings=BUDGETED)
    assert "kept no sent_settings" in str(refused.value)
    assert refused.value.remedy.startswith("re-record")
    with pytest.raises(RecordingRefused, match="older than this build's"):
        RecordedCompletions.record(_FakeInner(), path, settings=BUDGETED)


@dataclass(frozen=True, kw_only=True)
class _ReportingReply(_Reply):
    """A completion whose client reports what its request carried: the optional ``sent_settings`` member."""

    sent_settings: ClientRequestSettings | None = None


async def test_what_a_call_was_sent_with_is_recorded_and_replayed_never_the_settings_asked(tmp_path: Path) -> None:
    """The client was built with a budget and sent an effort word: the line keeps both, the key only what was asked."""
    path = tmp_path / "rec.jsonl"
    reply = _ReportingReply(content="Refund it.", sent_settings=BY_EFFORT)
    await _record(path, ("My order never came.", reply))

    (line,) = _lines(path)
    assert line["settings"] == BUDGETED.model_dump(mode="json")
    assert line["sent_settings"] == BY_EFFORT.model_dump(mode="json")
    assert "sent_settings" not in line["extra"], "an optional member is first-class, never an extra"

    replayed = await RecordedCompletions.replay(path, settings=BUDGETED).generate(
        system=SYSTEM, user="My order never came."
    )
    assert isinstance(replayed, ReplayedCompletion) and replayed.sent_settings == BY_EFFORT
    check_completion_conformance(replayed, require_sent_settings=True)


async def test_a_call_whose_client_does_not_report_what_it_sent_replays_unverified(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Hello.", _Reply(content="ok")))

    assert _lines(path)[0]["sent_settings"] is None
    replayed = await RecordedCompletions.replay(path, settings=BUDGETED).generate(system=SYSTEM, user="Hello.")
    assert isinstance(replayed, ReplayedCompletion) and replayed.sent_settings is None


async def test_a_completion_whose_sent_settings_are_not_request_settings_is_refused_when_recorded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rec.jsonl"
    reply = _ReportingReply(content="ok", sent_settings={"max_tokens": 10240})  # type: ignore[arg-type]
    with pytest.raises(CompletionConformanceFailure, match="sent_settings holds a dict"):
        await _record(path, ("Hello.", reply))
    assert not path.exists() or not path.read_text(encoding="utf-8"), "nothing is written for a refused completion"


async def test_a_completion_the_recording_could_not_replay_is_refused_when_recorded(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"

    @dataclass(frozen=True, kw_only=True)
    class _WithRawMessage(_Reply):
        raw: object = None

    cases: list[tuple[CompletionResult, str]] = [
        (_WithRawMessage(content="ok", raw=object()), "`raw` holds a value of type object, which JSON cannot hold"),
        (replace(_Reply(content="ok"), input_tokens="12"), "`input_tokens` holds '12' (str)"),  # type: ignore[arg-type]
        (replace(_Reply(content="ok"), stop_reason="length"), "stop_reason 'length' is not one of"),  # type: ignore[arg-type]
        (replace(_Reply(content="ok"), cost_usd=float("nan")), "`cost_usd` holds nan, a number JSON cannot hold"),
        (replace(_Reply(content="ok"), temperature=float("inf")), "`temperature` holds inf, a number JSON cannot hold"),
    ]
    for completion, problem in cases:
        inner = _FakeInner()
        inner.replies.append(completion)  # type: ignore[arg-type]
        with pytest.raises(CompletionConformanceFailure, match=re.escape(problem)):
            await RecordedCompletions.record(inner, path, settings=BUDGETED).generate(system=SYSTEM, user="Hello.")
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
    replayed = await RecordedCompletions.replay(path, settings=BUDGETED).generate(system=SYSTEM, user="Model.")
    assert isinstance(replayed, ReplayedCompletion)
    assert replayed.extra == {"served_provider": "Model Cloud"}


async def test_a_recording_replays_under_the_settings_it_was_made_with_and_refuses_others_by_name(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Score this answer.", _Reply(content="4")), settings=BUDGETED)

    same = ClientRequestSettings(max_tokens=16000, reasoning_max_tokens=8192, strict_output=True)
    served = await RecordedCompletions.replay(path, settings=same).generate(system=SYSTEM, user="Score this answer.")
    assert served.content == "4", "settings equal to the recorded ones are the recorded ones"

    asked_other = [
        (
            BY_EFFORT,
            "recorded with reasoning_max_tokens=8192, reasoning_effort=None, "
            "asked with reasoning_max_tokens=None, reasoning_effort='high'",
        ),
        (
            BUDGETED.model_copy(update={"max_tokens": 4096}),
            "recorded with max_tokens=16000, asked with max_tokens=4096",
        ),
        (
            None,
            "recorded with max_tokens=16000, reasoning_max_tokens=8192, reasoning_effort=None, strict_output=True, "
            "asked with no request settings",
        ),
    ]
    for settings, drift in asked_other:
        client = RecordedCompletions.replay(path, settings=settings, rerecord="RECORD=1 python llm_judge.py")
        with pytest.raises(RecordingRefused) as refused:
            await client.generate(system=SYSTEM, user="Score this answer.")
        message = str(refused.value)
        assert "'Score this answer.' is recorded only under other request settings" in message
        assert f"({drift})" in message, message
        assert message.endswith(
            "re-record " + str(path) + " with `RECORD=1 python llm_judge.py`, the provider's key set"
        )

    with pytest.raises(RecordingRefused, match="no call is recorded for the prompt 'Another answer.'"):
        await RecordedCompletions.replay(path, settings=BY_EFFORT).generate(system=SYSTEM, user="Another answer.")


@pytest.mark.parametrize(
    ("first", "second"), [(None, BUDGETED), (BUDGETED, BY_EFFORT)], ids=["none-and-settings", "two-settings"]
)
async def test_one_prompt_recorded_under_two_settings_shares_a_file_and_each_client_replays_its_own(
    tmp_path: Path, first: ClientRequestSettings | None, second: ClientRequestSettings | None
) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Same prompt.", _Reply(content="first")), settings=first)
    await _record(path, ("Same prompt.", _Reply(content="second")), settings=second)

    lines = _lines(path)
    assert lines[0]["key"] != lines[1]["key"], "settings are part of the key"
    assert [line["index"] for line in lines] == [0, 0]
    assert [line["settings"] for line in lines] == [
        None if settings is None else settings.model_dump(mode="json") for settings in (first, second)
    ]
    for settings, answer in ((first, "first"), (second, "second")):
        client = RecordedCompletions.replay(path, settings=settings)
        assert (await client.generate(system=SYSTEM, user="Same prompt.")).content == answer

    neither = ClientRequestSettings(max_tokens=2048)
    with pytest.raises(RecordingRefused, match="recorded only under other request settings") as refused:
        await RecordedCompletions.replay(path, settings=neither).generate(system=SYSTEM, user="Same prompt.")
    assert str(refused.value).count("recorded with") == 2, "each settings value the prompt is held under is named"


async def test_check_covers_names_a_prompt_held_only_under_other_settings(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Roll a die.", _Reply(content="4")), settings=BUDGETED)
    await _record(path, ("Pick a card.", _Reply(content="7")), settings=BY_EFFORT)

    prompts = [RecordedPrompt(SYSTEM, "Roll a die."), RecordedPrompt(SYSTEM, "Pick a card.")]
    with pytest.raises(RecordingRefused) as refused:
        RecordedCompletions.check_covers(path, prompts, settings=BUDGETED)
    message = str(refused.value)
    assert "no longer covers 1 prompt(s)" in message
    assert (
        "'Pick a card.' is recorded only under other request settings (recorded with reasoning_max_tokens=None, "
        "reasoning_effort='high', asked with reasoning_max_tokens=8192, reasoning_effort=None)"
    ) in message


async def test_a_line_whose_settings_differ_from_its_keys_earlier_lines_refuses_the_recording(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    await _record(path, ("Hello.", _Reply(content="a")), settings=BUDGETED)
    (first,) = _lines(path)
    with path.open("a", encoding="utf-8") as recording:
        recording.write(json.dumps({**first, "index": 1, "settings": BY_EFFORT.model_dump(mode="json")}) + "\n")

    with pytest.raises(RecordingRefused, match="line 2 is not a recorded call: its settings are not the ones") as e:
        RecordedCompletions.replay(path, settings=BUDGETED)
    assert "they carry reasoning_max_tokens=8192" in str(e.value) and "the file was edited" in str(e.value)


def test_settings_are_a_required_keyword_with_no_default(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    with pytest.raises(TypeError, match="settings"):
        RecordedCompletions.record(_FakeInner(), path)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="settings"):
        RecordedCompletions.replay(path)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="settings"):
        RecordedCompletions.check_covers(path, [])  # type: ignore[call-arg]
    assert not path.exists(), "the signature refuses the call before anything is read or written"


# --- standing in for a bound client: an analysis writer, an out-of-run call ------------------------------


# parity-with: threetears.evals.schema.BoundCompletionClient
class _FakeBoundInner(_FakeInner):
    """A scripted client that names its model and prices each call, as a role factory's client does."""

    model_name = "acme/large"

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        return 0.25


# parity-with: threetears.evals.schema.out_of_run_spend.OutOfRunSpendStore
class _FakeLedger:
    """Keeps every out-of-run row the budget writes."""

    def __init__(self) -> None:
        self.rows: list[OutOfRunSpend] = []

    def save_out_of_run_spend(self, spend: OutOfRunSpend, /) -> None:
        self.rows.append(spend)


async def _analysis_call(client: Any) -> tuple[list[float | None], Any, _FakeLedger]:
    """One analysis call through the engine's out-of-run budget, as ``generate_analysis`` makes it: priced, then made."""
    ledger = _FakeLedger()
    budget = OutOfRunBudget(ledger, scope_id="s", cap_usd=1.0, blocking_executor=None)
    [admitted] = budget.admit(client, "analysis", [PlannedCall(system=SYSTEM, user="Write the analysis.")])
    made = await budget.generate(client, admitted)
    return [admitted.ceiling_usd], made, ledger


async def test_a_recorder_of_a_bound_client_stands_in_for_an_analysis_client(tmp_path: Path) -> None:
    path = tmp_path / "analysis.jsonl"
    inner = _FakeBoundInner(_Reply(content="The analysis."))
    recorder = RecordedCompletions.record(inner, path, settings=None)
    assert recorder.model_name == "acme/large", "the inner client's model, read through"
    assert recorder.price_ceiling(system=SYSTEM, user="u") == 0.25, "the inner client's ceiling: every call goes to it"
    ceilings, made, ledger = await _analysis_call(recorder)
    assert ceilings == [0.25] and made.result.content == "The analysis." and len(ledger.rows) == 1
    assert len(_lines(path)) == 1


async def test_a_replay_stands_in_for_an_analysis_client_and_costs_nothing(tmp_path: Path) -> None:
    path = tmp_path / "analysis.jsonl"
    await _record(path, ("Write the analysis.", _Reply(content="The analysis.")), settings=None)
    replay = RecordedCompletions.replay(path, settings=None)
    assert replay.model_name == "acme/large", "the one model the recording holds"
    assert replay.price_ceiling(system=SYSTEM, user="anything") == 0.0, "a replay sends nothing and costs nothing"
    ceilings, made, _ = await _analysis_call(replay)
    assert ceilings == [0.0] and made.result.content == "The analysis."


async def test_a_replay_s_model_is_the_one_it_names_and_ambiguity_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "two.jsonl"
    await _record(path, ("Hello.", _Reply(content="a")), ("Hello.", replace(_Reply(content="b"), model="acme/small")))
    assert RecordedCompletions.replay(path, settings=BUDGETED, model="acme/small").model_name == "acme/small"
    with pytest.raises(RecordingRefused, match="holds calls of 2 models .* ambiguous") as refused:
        _ = RecordedCompletions.replay(path, settings=BUDGETED).model_name
    assert "model=" in refused.value.remedy


def test_a_recorder_of_a_plain_client_names_no_model_and_prices_nothing(tmp_path: Path) -> None:
    """A plain ``CompletionClient`` has neither member, so its recorder says so by name rather than inventing one."""
    recorder = RecordedCompletions.record(_FakeInner(), tmp_path / "plain.jsonl", settings=None)
    with pytest.raises(AttributeError, match="model_name: the recorded client _FakeInner names no model"):
        _ = recorder.model_name
    with pytest.raises(AttributeError, match="price_ceiling: the recorded client _FakeInner prices no call"):
        recorder.price_ceiling(system=SYSTEM, user="u")
