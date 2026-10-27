"""Recorded tool turns: replies kept per session in order, the snapshot id a placeholder, drift refused by its turn.

Recording goes through the shipped ``Model`` over ``httpx.MockTransport``, so what is kept is what a live turn
returns; nothing reaches a live API.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from threetears.evals.providers import Catalogue, Model
from threetears.evals.schema import TurnToolResult, TurnToolSpec, TurnMessage
from threetears.evals.testing import RecordedTurns, RecordingRefused

MODEL: Final = "claude-opus-5-5"

TOOLS: Final = (TurnToolSpec("evals_analyst", "Read the snapshot.", {"type": "object", "properties": {}}),)

#: The snapshot the recording was made on, and the one a later freeze of the same fixture gives it.
RECORDED_ON: Final = "snap-7f3a91c2"
REPLAYED_ON: Final = "snap-0b44de10"


def reply(*content: Mapping[str, Any], stop: str = "tool_use", cost_tokens: int = 100) -> dict[str, Any]:
    """A Messages API reply holding ``content``."""
    return {
        "model": "claude-opus-5-5-20261001",
        "content": list(content),
        "stop_reason": stop,
        "usage": {"input_tokens": cost_tokens, "output_tokens": 10, "cache_read_input_tokens": 0},
    }


def call(snapshot: str, action: str = "campaign_overview") -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": f"toolu_{action}",
        "name": "evals_analyst",
        "input": {"snapshot": snapshot, "action": action},
    }


class Server:
    """A Messages API answering each request with the next reply; it refuses once it has none left."""

    def __init__(self, *replies: Mapping[str, Any]) -> None:
        self._replies = list(replies)
        self.sent = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.sent += 1
        if not self._replies:
            raise AssertionError("a replay sent a request")
        return httpx.Response(200, json=self._replies.pop(0))


def opus(server: Server, **options: Any) -> Model:
    rows = {"data": [{"id": "anthropic/claude-opus-5.5", "pricing": {"prompt": "0.000004", "completion": "0.00002"}}]}
    return Model(
        MODEL,
        provider="anthropic",
        api_key="test-key",
        catalogue=Catalogue(
            http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=rows)))
        ),
        http=httpx.AsyncClient(transport=httpx.MockTransport(server)),
        **options,
    )


def ask(text: str) -> list[TurnMessage]:
    return [TurnMessage("user", text)]


async def record_two_sessions(path: Path) -> Server:
    """Record an analyst session of two turns and an editor session of one onto ``path``."""
    server = Server(
        reply({"type": "text", "text": f"Reading {RECORDED_ON}."}, call(RECORDED_ON)),
        reply({"type": "text", "text": "Reviewing."}, stop="end_turn"),
        reply({"type": "text", "text": f'```json\n{{"snapshot": "{RECORDED_ON}"}}\n```'}, stop="end_turn"),
    )
    model = opus(server, turn_recording=path)
    analyst = model.turn_session("analyst", snapshot=RECORDED_ON)
    editor = model.turn_session("editor-1", snapshot=RECORDED_ON)
    await analyst.turn(system="SYSTEM-ONLY-MARK", messages=ask("REQUEST-ONLY-MARK"), tools=TOOLS, max_tokens=1000)
    await editor.turn(system="e", messages=ask("page"), tools=TOOLS, max_tokens=1000)
    await analyst.turn(
        system="s",
        messages=[
            *ask("REQUEST-ONLY-MARK"),
            TurnMessage("user", [TurnToolResult("toolu_campaign_overview", "RESULT-ONLY-MARK")]),
        ],
        tools=TOOLS,
        max_tokens=1000,
    )
    return server


async def test_a_recording_replays_each_session_in_order_with_the_snapshot_written_back(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    server = await record_two_sessions(path)
    assert server.sent == 3

    replay = RecordedTurns.replay(path)
    analyst = replay.turn_session("analyst", snapshot=REPLAYED_ON)
    editor = replay.turn_session("editor-1", snapshot=REPLAYED_ON)
    first = await analyst.turn(system="anything", messages=ask("else entirely"), tools=(), max_tokens=1)
    review = await editor.turn(system="", messages=ask("?"), tools=(), max_tokens=1)
    assert replay.unconsumed() == {"analyst": 1}
    second = await analyst.turn(system="", messages=ask("?"), tools=(), max_tokens=1)

    assert first.text == f"Reading {REPLAYED_ON}."
    assert dict(first.tool_uses[0].input) == {"snapshot": REPLAYED_ON, "action": "campaign_overview"}
    assert first.stop_reason == "tool_use"
    assert second.text == f'```json\n{{"snapshot": "{RECORDED_ON}"}}\n```'.replace(RECORDED_ON, REPLAYED_ON)
    assert review.text == "Reviewing."
    assert first.cost_usd == pytest.approx(100 * 0.000004 + 10 * 0.00002), "a replayed cost is the recorded one"
    assert replay.unconsumed() == {}


async def test_the_file_holds_replies_alone_with_the_snapshot_as_a_placeholder(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    await record_two_sessions(path)
    text = path.read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.splitlines()]

    assert RECORDED_ON not in text
    assert "{snapshot}" in text
    assert [(line["session"], line["index"]) for line in lines] == [("analyst", 0), ("editor-1", 0), ("analyst", 1)]
    assert all(line["kind"] == "reply" for line in lines), "no ceiling was asked, so none is kept"
    assert all(set(line) == {"recorder_version", "recorded_at", "session", "index", "kind", "reply"} for line in lines)
    assert "ONLY-MARK" not in text, "no request bytes are kept: no system prompt, message or tool result"


async def test_a_session_asked_past_its_recording_is_refused_naming_the_turn(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    await record_two_sessions(path)
    replay = RecordedTurns.replay(path)
    editor = replay.turn_session("editor-1", snapshot=REPLAYED_ON)
    await editor.turn(system="", messages=ask("?"), tools=(), max_tokens=1)

    with pytest.raises(RecordingRefused, match=r"session 'editor-1' holds 1 recorded turns, and its turn 2 was asked"):
        await editor.turn(system="", messages=ask("?"), tools=(), max_tokens=1)
    with pytest.raises(RecordingRefused, match=r"session 'editor-2' holds no recorded turns, and its turn 1 was asked"):
        await replay.turn_session("editor-2", snapshot=REPLAYED_ON).turn(
            system="", messages=ask("?"), tools=(), max_tokens=1
        )
    with pytest.raises(RecordingRefused, match="cut short .* delete the file and record it again; otherwise"):
        await editor.turn(system="", messages=ask("?"), tools=(), max_tokens=1)
    assert replay.unconsumed() == {"analyst": 2}, "the turns a replay left are named by session"


async def test_a_session_asked_for_again_carries_on_where_it_was(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    await record_two_sessions(path)
    replay = RecordedTurns.replay(path)

    await replay.turn_session("analyst", snapshot=REPLAYED_ON).turn(
        system="", messages=ask("?"), tools=(), max_tokens=1
    )
    again = await replay.turn_session("analyst", snapshot=REPLAYED_ON).turn(
        system="", messages=ask("?"), tools=(), max_tokens=1
    )

    assert again.text == f'```json\n{{"snapshot": "{REPLAYED_ON}"}}\n```', (
        "the analyst's second turn, not its first again"
    )


async def test_a_model_given_an_existing_turn_recording_replays_it_and_sends_nothing(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    await record_two_sessions(path)
    idle = Server()

    session = opus(idle, turn_recording=path).turn_session("analyst", snapshot=REPLAYED_ON)
    first = await session.turn(system="", messages=ask("?"), tools=(), max_tokens=1)

    assert first.text == f"Reading {REPLAYED_ON}."
    assert idle.sent == 0


async def test_an_existing_recording_is_never_recorded_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "turns.jsonl"
    await record_two_sessions(path)
    held = path.read_text(encoding="utf-8")

    with pytest.raises(RecordingRefused, match="it exists, and a turn recording is replayed, never recorded again"):
        RecordedTurns.record(opus(Server()), path)
    monkeypatch.setenv("RECORD", "1")
    idle = Server()
    session = opus(idle, turn_recording=path).turn_session("analyst", snapshot=REPLAYED_ON)
    first = await session.turn(system="", messages=ask("?"), tools=(), max_tokens=1)

    assert first.text == f"Reading {REPLAYED_ON}.", "RECORD=1 still replays a turn recording that exists"
    assert idle.sent == 0
    assert path.read_text(encoding="utf-8") == held


async def test_each_ceiling_a_session_answered_is_kept_for_its_turn_and_replayed(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    model = opus(Server(reply({"type": "text", "text": "one"}, stop="end_turn")), turn_recording=path)
    await model.refresh_prices()
    witness = opus(Server())
    await witness.refresh_prices()
    asked: dict[str, Any] = {"system": "s", "tools": TOOLS, "max_tokens": 1000}
    recording = model.turn_session("analyst", snapshot=RECORDED_ON)

    first_ceiling = recording.turn_price_ceiling(messages=ask("a"), **asked)
    await recording.turn(messages=ask("a"), **asked)
    # The budget refuses the second turn on its ceiling: no reply is ever recorded for it.
    refused_ceiling = recording.turn_price_ceiling(messages=ask("a much longer conversation " * 50), **asked)

    assert first_ceiling == witness.turn_price_ceiling(messages=ask("a"), **asked), "a recording passes it through"
    assert first_ceiling is not None and refused_ceiling is not None and refused_ceiling > first_ceiling
    replaying = RecordedTurns.replay(path)
    replay = replaying.turn_session("analyst", snapshot=REPLAYED_ON)
    assert replay.turn_price_ceiling(messages=ask("anything"), **asked) == first_ceiling
    replayed = await replay.turn(messages=ask("anything"), **asked)
    assert replaying.unconsumed() == {"analyst": 1}, "the refused turn's ceiling is a line not yet asked"
    assert replay.turn_price_ceiling(messages=ask("anything"), **asked) == refused_ceiling, (
        "the refused turn's ceiling replays, so the budget ends the replay where it ended the recording"
    )
    assert replayed.cost_usd == pytest.approx(100 * 0.000004 + 10 * 0.00002), "the cost is still the recorded one"
    assert replaying.unconsumed() == {}


async def test_a_ceiling_asked_for_a_turn_that_recorded_none_is_refused_naming_the_turn(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    await record_two_sessions(path)
    replay = RecordedTurns.replay(path).turn_session("editor-1", snapshot=REPLAYED_ON)

    with pytest.raises(RecordingRefused, match=r"session 'editor-1' recorded no price ceiling for its turn 1"):
        replay.turn_price_ceiling(system="", messages=ask("?"), tools=(), max_tokens=1)


async def test_a_model_with_a_turn_recording_sends_turns_only_through_a_session(tmp_path: Path) -> None:
    model = opus(Server(), turn_recording=tmp_path / "turns.jsonl")

    with pytest.raises(ValueError, match=r"send it through model\.turn_session\(name, snapshot=\.\.\.\)"):
        await model.turn(system="", messages=ask("?"), tools=(), max_tokens=1)
    with pytest.raises(ValueError, match="turn_session"):
        model.turn_price_ceiling(system="", messages=ask("?"), tools=(), max_tokens=1)


def test_a_turn_recording_beside_a_temperature_is_refused_when_the_model_is_built(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tool turns send no temperature"):
        opus(Server(), turn_recording=tmp_path / "turns.jsonl", temperature=0.3)
    assert not (tmp_path / "turns.jsonl").exists()


async def replaying_completions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A completion recording of one call, which a model built on it replays with no live client."""
    monkeypatch.delenv("RECORD", raising=False)
    path = tmp_path / "completions.jsonl"
    recorder = opus(Server(reply({"type": "text", "text": "hello"}, stop="end_turn")), recording=path)
    await recorder.generate(system="", user="hi")
    return path


async def test_a_turn_recording_to_record_on_a_model_replaying_its_completions_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completions = await replaying_completions(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="is recorded live, and this model replays its completions"):
        Model(MODEL, provider="anthropic", recording=completions, turn_recording=tmp_path / "turns.jsonl")


async def test_a_model_replaying_its_completions_sends_no_tool_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replaying = Model(MODEL, provider="anthropic", recording=await replaying_completions(tmp_path, monkeypatch))

    with pytest.raises(ValueError, match="replays its completions from .* and sends nothing live"):
        await replaying.turn(system="", messages=ask("?"), tools=(), max_tokens=1)
    with pytest.raises(ValueError, match="sends nothing live"):
        replaying.turn_session("analyst", snapshot=RECORDED_ON)


async def test_a_reply_already_holding_the_placeholder_is_refused_when_it_is_recorded(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    model = opus(Server(reply({"type": "text", "text": "the key is {snapshot}"}, stop="end_turn")), turn_recording=path)

    with pytest.raises(RecordingRefused, match=r"holds the literal text \{snapshot\}"):
        await model.turn_session("analyst", snapshot=RECORDED_ON).turn(
            system="", messages=ask("?"), tools=(), max_tokens=1
        )
    assert not path.exists() or path.read_text(encoding="utf-8") == "", "nothing ambiguous is written"


@pytest.mark.parametrize(
    ("lines", "problem"),
    [
        (["not json"], r"line 1 is not JSON"),
        ([{"session": "a"}], r"line 1 is not a line this recorder writes"),
        (
            [{"recorder_version": 1, "recorded_at": "t", "session": "a", "index": 0, "reply": {}}],
            r"line 1 is not a line this recorder writes",
        ),
        (
            [{"recorder_version": 9, "recorded_at": "t", "session": "a", "index": 0, "kind": "reply", "reply": {}}],
            r"line 1 is recorder format 9",
        ),
        (
            [{"recorder_version": 1, "recorded_at": "t", "session": "a", "index": 1, "kind": "reply", "reply": {}}],
            r"line 1 is for turn index 1 of session 'a', whose next is 0",
        ),
        (
            [{"recorder_version": 1, "recorded_at": "t", "session": "a", "index": 1, "kind": "ceiling", "ceiling": 1}],
            r"line 1 is for turn index 1 of session 'a', whose next is 0",
        ),
        (
            [
                {
                    "recorder_version": 1,
                    "recorded_at": "t",
                    "session": "a",
                    "index": 0,
                    "kind": "ceiling",
                    "ceiling": "1",
                }
            ],
            r"line 1's ceiling '1' is neither a finite number nor null",
        ),
        (
            [
                {
                    "recorder_version": 1,
                    "recorded_at": "t",
                    "session": "a",
                    "index": 0,
                    "kind": "reply",
                    "reply": {"content": []},
                }
            ],
            r"line 1's reply cannot be read",
        ),
    ],
)
def test_a_line_this_recorder_does_not_write_refuses_the_file_by_number(
    tmp_path: Path, lines: list[Any], problem: str
) -> None:
    path = tmp_path / "turns.jsonl"
    path.write_text("\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines), encoding="utf-8")

    with pytest.raises(RecordingRefused, match=problem):
        RecordedTurns.replay(path)


def test_a_missing_recording_and_an_empty_session_name_are_refused(tmp_path: Path) -> None:
    with pytest.raises(RecordingRefused, match="no such turn recording"):
        RecordedTurns.replay(tmp_path / "absent.jsonl")
    path = tmp_path / "turns.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="needs a name"):
        RecordedTurns.replay(path).turn_session("", snapshot=RECORDED_ON)
    with pytest.raises(ValueError, match="needs the snapshot id"):
        RecordedTurns.replay(path).turn_session("analyst", snapshot="")
