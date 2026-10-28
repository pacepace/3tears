"""The report writer's recorded runs, replayed: each ends as recorded, with its tools run live on the fixture snapshot.

A recording (``turns.jsonl``, :class:`~threetears.evals.testing.RecordedTurns`) holds each session's model replies in
order and nothing sent, the snapshot id written as ``{snapshot}``; ``replay.json`` beside it says how the run was made
(the fixture snapshot, the model, the budget, the notes withheld) and what it came to. A replay runs ``write_report``
over a host seeded with every fixture snapshot, every session's turns served from the recording and every tool call
run live, and holds it to the record: the ending, the analyst's and the editor's round counts, whether a document was
stored, the tally's turns, tool calls, bounces and cost, and every recorded turn and price ceiling consumed. A run
that ended ``failed`` is refused as a replay, naming why: its replay would end ``failed`` on the recording running out
whatever failed live, so it would pass for the wrong cause.

**Drift is a red that names its cause, and nothing is recorded again.** A replay that asks a turn the recording does
not hold ends ``failed`` with the recording's refusal (the session and the turn) as its reason; one that leaves turns
unasked names each session. The fix is to the code, or to the recorded reply as data (plan §4.7).

The recordings replayed are each directory under ``fixtures/report_writer/`` (hand-scripted, made through the
recorder from scripted replies) and under ``fixtures/report_docs/prototype4/`` (the acceptance runs,
``analyst_cli.py write``) that holds a ``replay.json``.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans.analyst_cli import RECORDING, REPLAY, snapshot_ids
from packages.evals.tests.report_writer_support import RecordedModel, replay_cases, written
from threetears.evals.actions.report_writing import ReportNotStored, ReportWritten
from threetears.evals.testing import RecordedTurns


@dataclass(frozen=True)
class Replayed:
    """A replay: what ``replay.json`` recorded, what the replay came to, what it left unasked, and why its store
    refused, where it did."""

    expected: dict[str, Any]
    written: ReportWritten
    unconsumed: dict[str, int]
    refused: str | None = None


async def replay(case: Path) -> Replayed:
    """Replay one recorded run (its directory) as ``replay.json`` says it was made."""
    expected: dict[str, Any] = json.loads((case / REPLAY).read_text())
    recording = RecordedTurns.replay(case / RECORDING)
    refusal: str | None = None
    try:
        result = await written(
            RecordedModel(recording, expected["model"]),
            snapshot=snapshot_ids()[expected["snapshot_of"]],
            budget_usd=expected["budget_usd"],
            withhold=expected["withhold"],
        )
    except ReportNotStored as refused:
        # A store refused in the replay: the run's record rides on it, and drift reads it and says why.
        result, refusal = refused.written, str(refused)
    return Replayed(expected, result, recording.unconsumed(), refusal)


def drift(replayed: Replayed) -> list[str]:
    """How a replay differs from its record, each difference naming its cause; empty when it ran as recorded."""
    expected, result = replayed.expected, replayed.written
    if expected["ending"] == "failed":
        return [
            f"the recorded run ended failed ({expected['reason'] or 'no reason recorded'}), so it is no replay: a "
            "replay of it ends failed on the recording running out, whatever failed live, and would pass for the "
            "wrong cause; record a run that reached an ending of its own"
        ]
    outcome = result.outcome
    found: list[str] = []
    if result.ending != expected["ending"]:
        found.append(f"ended {result.ending} ({outcome.reason or 'no reason given'}), recorded {expected['ending']}")
    if (result.stored is not None) != expected["stored"]:
        why = f" (the store refused: {replayed.refused})" if replayed.refused is not None else ""
        found.append(f"stored is {result.stored is not None}, recorded {expected['stored']}{why}")
    for name, now in (("analyst_rounds", outcome.analyst_rounds), ("editor_rounds", outcome.editor_rounds)):
        if now != expected[name]:
            found.append(f"{name} is {now}, recorded {expected[name]}")
    total = result.generation["tally"]["total"]
    for name, recorded in expected["tally"].items():
        now = total[name]
        same = (
            now is None and recorded is None
            if now is None or recorded is None
            else math.isclose(now, recorded, rel_tol=1e-9, abs_tol=1e-12)
        )
        if not same:
            found.append(f"the tally's {name} is {now}, recorded {recorded}")
    for session, left in replayed.unconsumed.items():
        found.append(f"session {session!r} left {left} recorded lines unasked: the code now stops sooner than it did")
    return found


@pytest.mark.parametrize("case", [path.parent for path in replay_cases()], ids=lambda path: path.name)
async def test_a_recorded_run_replays_to_its_recorded_ending(case: Path) -> None:
    replayed = await replay(case)
    assert not drift(replayed), f"{case.name} drifted from its recording:\n" + "\n".join(drift(replayed))


def test_the_hand_scripted_recordings_are_replayed() -> None:
    """The harness reaches the recordings it is built on: one that clears in round 1, one revised once."""
    assert {"hand-pass", "hand-revise"} <= {path.parent.name for path in replay_cases()}


async def test_a_replay_stored_what_its_run_stored() -> None:
    (case,) = [path.parent for path in replay_cases() if path.parent.name == "hand-pass"]
    replayed = await replay(case)
    stored = replayed.written.stored
    assert stored is not None and stored.ending == "stored"
    assert [finding["severity"] for finding in stored.findings] == ["should_fix"]
    assert replayed.written.page is not None and replayed.written.page.made


async def test_a_refused_store_is_named_in_the_drift_line() -> None:
    """A replay whose store refused says why beside the stored line, so the red names its cause."""
    (case,) = [path.parent for path in replay_cases() if path.parent.name == "hand-pass"]
    replayed = await replay(case)
    assert replayed.refused is None and not drift(replayed), "the control: it stores as recorded"
    unstored = replace(replayed.written, stored=None, page=None)

    found = drift(Replayed(replayed.expected, unstored, replayed.unconsumed, "the store gate found 1 verdict"))

    assert found == ["stored is False, recorded True (the store refused: the store gate found 1 verdict)"], found


@pytest.fixture
def hand_revise(tmp_path: Path) -> Path:
    """A copy of the revised recording, to change as a drifted recording would be."""
    (case,) = [path.parent for path in replay_cases() if path.parent.name == "hand-revise"]
    copied = tmp_path / case.name
    shutil.copytree(case, copied)
    return copied


def _lines(case: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (case / RECORDING).read_text().splitlines() if line.strip()]


def _write_lines(case: Path, lines: list[dict[str, Any]]) -> None:
    (case / RECORDING).write_text("".join(json.dumps(line) + "\n" for line in lines))


async def test_a_replay_asking_a_turn_the_recording_lacks_is_a_red_naming_the_session_and_turn(
    hand_revise: Path,
) -> None:
    lines = _lines(hand_revise)
    last = max(index for index, line in enumerate(lines) if line["session"] == "editor-2" and line["kind"] == "reply")
    _write_lines(hand_revise, lines[:last])

    found = drift(await replay(hand_revise))

    assert found and "ended failed" in found[0], found
    assert "session 'editor-2' holds no recorded turns, and its turn 1 was asked" in found[0], found


async def test_a_replay_leaving_recorded_turns_unasked_is_a_red_naming_the_session(hand_revise: Path) -> None:
    lines = _lines(hand_revise)
    editor_lines = [line for line in lines if line["session"] == "editor-2"]
    index = 1 + max(line["index"] for line in editor_lines)
    extra = [{**line, "index": index} for line in editor_lines if line["index"] == 0]
    _write_lines(hand_revise, [*lines, *extra])

    found = drift(await replay(hand_revise))

    assert found == ["session 'editor-2' left 2 recorded lines unasked: the code now stops sooner than it did"], found


async def test_a_recording_of_a_run_that_ended_failed_is_refused_as_a_replay(hand_revise: Path) -> None:
    expected = json.loads((hand_revise / REPLAY).read_text())
    expected.update(ending="failed", reason="ProviderError: anthropic 529 overloaded", stored=False)
    (hand_revise / REPLAY).write_text(json.dumps(expected))

    found = drift(await replay(hand_revise))

    assert found == [
        "the recorded run ended failed (ProviderError: anthropic 529 overloaded), so it is no replay: a replay of it "
        "ends failed on the recording running out, whatever failed live, and would pass for the wrong cause; record a "
        "run that reached an ending of its own"
    ]
