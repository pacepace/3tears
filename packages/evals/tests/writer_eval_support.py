"""What the writer eval's tests share: a scripted scoring judge, Kairo's report as a turn recording keeps it, and its
case file.

Nothing here reaches a model: the judge scores every dimension 4 and sends nothing, and the recording's replies are
scripted, its tool calls quoting the snapshot as the recording's placeholder.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.report_writer_support import call, reply, review, submitted
from packages.evals.tests.writer_case_support import SEED
from threetears.evals.harness import WriterCaseFile
from threetears.evals.schema.completion import StopReason
from threetears.evals.schema.turns import TurnReply
from threetears.evals.testing import RecordedTurns

#: The scoring judge: OpenRouter's, of the google family.
JUDGE: Final = "google/gemini-3.8-flash"

#: The score the scripted judge gives every dimension.
JUDGED_SCORE: Final = 4


@dataclass(frozen=True)
class ScoredReply:
    """One scripted judge reply, in the attribute names the engine reads."""

    content: str
    input_tokens: int | None = 200
    output_tokens: int | None = 30
    cost_usd: float | None = 0.0
    price_source: str | None = "scripted rate"
    model: str = JUDGE
    served_model: str | None = JUDGE
    reasoning_tokens: int | None = None
    stop_reason: StopReason = "end_turn"
    temperature: float | None = None


# parity-with: threetears.evals.harness.writer_host.WriterJudge
class FakeScoringJudge:
    """A scoring judge that sends nothing: it scores the dimension its system prompt names, and keeps every call."""

    model_name = JUDGE
    provider = "openrouter"
    base_url = None

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> ScoredReply:
        match = re.search(r'single key "([^"]+)"', system)
        assert match is not None, "the judge's system prompt names no dimension"
        self.calls.append(match[1])
        return ScoredReply(json.dumps({"reasoning": f"read {match[1]}", "criteria_scores": {match[1]: JUDGED_SCORE}}))


def kairo_report(analyst: str, editor: str) -> dict[str, list[TurnReply]]:
    """Kairo's report, by session: the analyst, on ``analyst``, reads the dominance and submits the document; the
    editor, on ``editor``, reads it too and clears it."""
    return {
        "analyst": [
            reply(call(1, "options_dominance", "{snapshot}"), model=analyst),
            reply(submitted(), model=analyst),
        ],
        "editor-1": [reply(call(9, "options_dominance", "{snapshot}"), model=editor), reply(review(), model=editor)],
    }


def lay_recording(path: Path, sessions: Mapping[str, Sequence[TurnReply]]) -> Path:
    """Write ``sessions`` at ``path`` as a turn recording keeps them: each turn's price ceiling, then its reply."""
    lines: list[dict[str, Any]] = []
    for session, replies in sessions.items():
        for index, said in enumerate(replies):
            stamp = {
                "recorder_version": RecordedTurns.FORMAT_VERSION,
                "recorded_at": "2026-11-03T00:00:00+00:00",
                "session": session,
                "index": index,
            }
            lines += [
                {**stamp, "kind": "ceiling", "ceiling": 0.05},
                {**stamp, "kind": "reply", "reply": said.to_json()},
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def kairo_case(directory: Path) -> WriterCaseFile:
    """Kairo's fixture snapshot written as a case file in ``directory``: a held-out case at any share above 0.1."""
    group, withhold = SEED[KAIRO]
    case = WriterCaseFile(
        name="kairo",
        group=group,
        stratum="reference",
        withhold=list(withhold),
        snapshot=fixture_snapshots("frozen-elsewhere")[KAIRO],
    )
    case.write(directory / "kairo.json.gz")
    return case
