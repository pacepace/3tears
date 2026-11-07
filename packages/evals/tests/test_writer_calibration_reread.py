"""The writer calibration's reread: the judge asked the anchor set again, set beside its own stored read.

``fixtures/writer_cases/writer_calibration.py --reread`` is a test-retest on the judge's own verdicts over the pair set
(every two rounds of one recording, in both orders). These tests hold it on scripted judges that call no model:

- a judge that flips some preferences and leans to the first slot is reported moved on all three measures, and the
  served-model change between the reads is named as the likely cause;
- the same judge again is reported held on all three, and the agreement with the Opus editor is its own line;
- each served-model sentence is reached: changed or not, moved or held, and a read that names no model;
- "held" is a set tolerance of 0.05, pinned at its edge on preference and on first-slot lean;
- the stored record is only read: the new read goes to a new ``--out`` directory, and nowhere without one;
- the cost printed before a live reread is the stored record's own cost per call, printed before any judge call.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.judge import ToyJudgeCompletion
from packages.evals.tests.fixtures.writer_cases import writer_calibration
from packages.evals.tests.fixtures.writer_cases.writer_calibration import (
    MEASURES,
    PAIR_SET,
    RECORDS,
    Page,
    Reread,
    cost_note,
    main,
    pages,
    pairs,
    reread,
    stored_record,
)
from packages.evals.tests.fixtures.writer_cases.writer_smoke import JUDGE as SMOKE_JUDGE
from packages.evals.tests.fixtures.writer_cases.writer_smoke import WRITER as SMOKE_WRITER
from threetears.evals.ops.report_writer.preference import preference_rubric

#: The scripted judge's name, as the anchor rows record the judge asked.
_JUDGE = "scripted/anchor-judge"

#: Where the built-in prompt names the one key its reply maps.
_SCORE_KEY = re.compile(r'single key "([^"]+)"')

#: The lean table row's measure, as the reread prints it.
_LEAN = "First-slot lean: 0 none, up to 0.5 for Report A at the scale's end every time, down to -0.5 for Report B"


# parity-with: threetears.evals.schema.CompletionClient
class FakeAnchorJudge:
    """A judge that answers each pair case from a script keyed by the case's two pages, names its served model (or
    none), and notes each call in ``events`` when given a list."""

    def __init__(self, scores: dict[str, int], served: str | None, events: list[str] | None = None) -> None:
        self.scores = scores
        self.served = served
        self.events = events

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, str] | None = None
    ) -> ToyJudgeCompletion:
        if self.events is not None:
            self.events.append("judge called")
        key = _SCORE_KEY.search(system)
        assert key is not None, "the built-in judge prompt names its score key"
        (score,) = [score for text, score in self.scores.items() if text in user]
        reply = {"reasoning": "scripted", "criteria_scores": {key.group(1): score}}
        return ToyJudgeCompletion(
            content=json.dumps(reply),
            input_tokens=10,
            output_tokens=10,
            cost_usd=0.01,
            model=_JUDGE,
            price_source="scripted",
            served_model=self.served,
        )

    async def aclose(self) -> None:
        return None


class _Printed:
    """Standard output as a list of events, so a print's place among the judge's calls can be read."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def write(self, text: str) -> int:
        if text.strip():
            self.events.append(text)
        return len(text)

    def flush(self) -> None:
        return None


def _clients(judge: FakeAnchorJudge) -> Any:
    return lambda _role, _model, *, temperature=None: judge


def _script(found: list[Page], *, flip: int = 0, lean: int = 0, nudge: int = 0) -> dict[str, int]:
    """Each pair case's score: the Opus editor's verdict, but for three kinds of pairs of rounds, disjoint.

    :param flip: Pairs the editor did not tie, answered the other way in both orders: preference moves, nothing else.
    :param lean: Further such pairs answered 1 (Report A, read first) in both orders: each changes side, loses its
        order consistency, and leans 0.5.
    :param nudge: Pairs the editor scored 1 or 5, their order answered 5 answered 4 instead: one step toward Report A
        in one order, so each leans 0.125 and keeps its side and its consistency.
    """
    criterion = preference_rubric()
    ordered = pairs(found)
    firsts = range(0, len(ordered), 2)
    decided = [n for n in firsts if ordered[n].expected != 3]
    flipped, leaning = set(decided[:flip]), set(decided[flip : flip + lean])
    strong = [n for n in firsts if ordered[n].expected in (1, 5) and n not in flipped | leaning]
    assert len(strong) >= nudge and len(decided) >= flip + lean, "the anchor set holds enough pairs for the script"
    nudged = set(strong[:nudge])
    script: dict[str, int] = {}
    for n, pair in enumerate(ordered):
        first, score = n - n % 2, pair.expected
        if first in flipped:
            score = 6 - score
        elif first in leaning:
            score = 1
        elif first in nudged and score == 5:
            score = 4
        script[pair.case(criterion).output] = score
    return script


def _read(stored: dict[str, Any], found: list[Page], judge: FakeAnchorJudge, cap_usd: float | None = None) -> Reread:
    return asyncio.run(reread(stored, found, judge=_JUDGE, clients=_clients(judge), k=2, cap_usd=cap_usd))


@pytest.fixture(scope="module")
def found() -> list[Page]:
    return asyncio.run(pages())[0]


@pytest.fixture(scope="module")
def stored(found: list[Page]) -> dict[str, Any]:
    """The stored record: the steady judge's first read, as ``--out`` writes it and a file gives it back."""
    first = _read({}, found, FakeAnchorJudge(_script(found), "judge-v1"))
    assert all(held is None for held in first.held().values())
    lines = list(first.lines())
    # A first anchor says so in its opening line and prints its own read alone, compared with nothing.
    assert lines[0] == (
        f"A reread asks {_JUDGE} the anchor cases again to detect a judge or version shift, never whether the judge is "
        f"right; the stored record holds no per-case verdicts from {_JUDGE}, so this read is its first anchor."
    )
    assert lines[2:7] == [
        "| Measure | First read |",
        "|---|---|",
        "| Pairs of rounds preferring the later round | 17 of 21 |",
        "| Pairs whose two orders pick the same round | 21 of 21 |",
        f"| {_LEAN} | 0.00 |",
    ]
    assert not any("Held" in line or "served model" in line for line in lines)
    record: dict[str, Any] = json.loads(json.dumps(first.record(Path("stored.json")), default=str))
    return record


def test_a_shifted_judge_moves_on_all_three_measures_and_the_served_model_change_is_named(
    found: list[Page], stored: dict[str, Any]
) -> None:
    read = _read(stored, found, FakeAnchorJudge(_script(found, flip=4, lean=4), "judge-v2"))
    assert read.held() == dict.fromkeys(MEASURES, False)
    lines = list(read.lines())
    assert lines[0] == (
        f"A reread asks {_JUDGE} the anchor cases again and compares it with its own stored read: it detects a judge "
        "or version shift, never whether the judge is right."
    )
    # The four flipped and the four leaning pairs each changed side; the leaning four lost their order consistency
    # and lean the whole way, 0.5 each over 21 pairs.
    assert read.changed() == (8, 21)
    assert lines[4] == (
        "| Pairs of rounds preferring the later round | 17 of 21 | 12 of 21 | 8 of 21 changed side, 0.381 | no |"
    )
    assert lines[5] == "| Pairs whose two orders pick the same round | 21 of 21 | 17 of 21 | -0.190 | no |"
    assert lines[6] == f"| {_LEAN} | 0.00 | 0.10 | +0.095 | no |"
    assert lines[8] == (
        "Held means a measure changed by 0.05 or less, a set tolerance rather than a measured one; for preference the "
        "change is the share of the pairs both reads scored that changed side."
    )
    assert lines[9] == "The judge moved on preference, order consistency and first-slot lean."
    assert lines[10] == "Its served model changed from judge-v1 to judge-v2, the likely cause."
    assert lines[-1].startswith("Agreement with the Opus editor's rubric totals, an agent's labels, as weighted κ")
    assert not any(word in line.lower() for line in lines for word in ("calibrat", "accura"))


def test_an_unchanged_judge_holds_on_all_three_and_agrees_with_the_opus_editor_on_its_own_line(
    found: list[Page], stored: dict[str, Any]
) -> None:
    read = _read(stored, found, FakeAnchorJudge(_script(found), "judge-v1"))
    assert read.held() == dict.fromkeys(MEASURES, True)
    assert read.changed() == (0, 21)
    lines = list(read.lines())
    assert lines[4].endswith("| 17 of 21 | 17 of 21 | 0 of 21 changed side, 0.000 | yes |")
    assert lines[5] == "| Pairs whose two orders pick the same round | 21 of 21 | 21 of 21 | +0.000 | yes |"
    assert lines[6] == f"| {_LEAN} | 0.00 | 0.00 | +0.000 | yes |"
    assert lines[9] == "The judge held on all three measures."
    assert lines[10] == "Its served model is judge-v1 in both reads: no model change."
    # The script answers the Opus editor's own verdicts, so the second line reads full agreement both times.
    assert lines[-1] == (
        "Agreement with the Opus editor's rubric totals, an agent's labels, as weighted κ on the anchor cases: 1.00 in "
        "the stored read and 1.00 in this one."
    )
    # Every trial is a row of the new anchor, as the next reread reads it.
    record = read.record(Path("stored.json"))
    assert len(record["anchor"]) == 2 * 42 and record["case_sets"] == {PAIR_SET: {"cases": 42}}
    assert {(row["judge"], row["served_model"]) for row in record["anchor"]} == {(_JUDGE, "judge-v1")}


@pytest.mark.parametrize(
    ("shift", "served", "says"),
    [
        # The drift a scheduled reread exists to report: the judge moved under the same served model.
        (
            {"flip": 4, "lean": 4},
            "judge-v1",
            "Its served model is judge-v1 in both reads, so no model change explains the shift.",
        ),
        ({}, "judge-v2", "Its served model changed from judge-v1 to judge-v2, and the measures held."),
        ({}, None, "A read names no served model, so a model change is neither named nor ruled out."),
    ],
    ids=["moved-same-model", "held-changed-model", "no-served-model"],
)
def test_each_served_model_sentence_is_the_one_its_reads_call_for(
    found: list[Page], stored: dict[str, Any], shift: dict[str, int], served: str | None, says: str
) -> None:
    read = _read(stored, found, FakeAnchorJudge(_script(found, **shift), served))
    assert read.held() == dict.fromkeys(MEASURES, not shift)
    lines = list(read.lines())
    assert lines[10] == says
    # This read's agreement is found by the judge's name when no served model is named, as the stored read's is.
    if not shift:
        assert lines[-1].endswith("1.00 in the stored read and 1.00 in this one.")


@pytest.mark.parametrize(
    ("shift", "measure", "change", "holds"),
    [
        # One pair of rounds in 21 changing side is 0.048, within 0.05; two are 0.095, past it.
        ({"flip": 1}, "preference", "1 of 21 changed side, 0.048", True),
        ({"flip": 2}, "preference", "2 of 21 changed side, 0.095", False),
        # Each nudged pair leans 0.125: eight over 21 pairs are 0.048, nine are 0.054.
        ({"nudge": 8}, "first-slot lean", "+0.048", True),
        ({"nudge": 9}, "first-slot lean", "+0.054", False),
    ],
    ids=["one-flip-holds", "two-flips-move", "eight-nudges-hold", "nine-nudges-move"],
)
def test_held_is_a_change_of_at_most_the_set_tolerance(
    found: list[Page], stored: dict[str, Any], shift: dict[str, int], measure: str, change: str, holds: bool
) -> None:
    read = _read(stored, found, FakeAnchorJudge(_script(found, **shift), "judge-v1"))
    assert read.held() == {name: holds if name == measure else True for name in MEASURES}
    row = list(read.lines())[4 if measure == "preference" else 6]
    assert row.endswith(f"| {change} | {'yes' if holds else 'no'} |"), row


def test_a_change_shown_as_exactly_the_tolerance_holds(found: list[Page], stored: dict[str, Any]) -> None:
    """20 of 20 pairs consistent to 19 of 20 is a change of 0.05, which a float difference puts just over it."""
    last = pairs(found)[-1]
    dropped = (last.a.run, *sorted((last.a.round, last.b.round)))

    def kept(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if (r["run"], *sorted((int(r["first"]), int(r["second"])))) != dropped]

    read = _read(stored, found, FakeAnchorJudge(_script(found, lean=1), "judge-v1"))
    read = dataclasses.replace(read, stored={**stored, "anchor": kept(stored["anchor"])}, rows=kept(read.rows))
    assert read.before is not None and read.before.consistent() == (20, 20)
    assert read.now is not None and read.now.consistent() == (19, 20)
    assert read.held()["order consistency"] is True
    row = list(read.lines())[5]
    assert row.endswith("| -0.050 | yes |"), row


def test_the_reread_only_reads_the_stored_record_and_writes_to_a_new_out_directory(
    found: list[Page],
    stored: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def replayed() -> tuple[list[Page], list[Any]]:
        return found, []

    monkeypatch.setattr(writer_calibration, "pages", replayed)
    records = tmp_path / "records"
    records.mkdir()
    path = records / "20261107-steady.json"
    path.write_text(json.dumps(stored), encoding="utf-8")
    before = path.read_bytes()

    # Without --out a dry reread prints and writes nothing; a directory names its newest record.
    assert main(["--reread", str(records), "--dry"]) == 0
    assert sorted(tmp_path.rglob("*")) == [records, path] and path.read_bytes() == before
    printed = capsys.readouterr().out
    assert printed.startswith("A reread asks dry/scripted-judge the anchor cases again") and "sends" not in printed

    out = tmp_path / "new"
    assert main(["--reread", str(path), "--dry", "--out", str(out)]) == 0
    assert path.read_bytes() == before
    written = json.loads((out / "reread.json").read_text(encoding="utf-8"))
    assert written["reread_of"] == str(path) and len(written["anchor"]) == 2 * 42
    # An --out already there is refused before anything is judged, so neither file is written over.
    with pytest.raises(FileExistsError):
        main(["--reread", str(path), "--dry", "--out", str(out)])
    assert path.read_bytes() == before and json.loads((out / "reread.json").read_text(encoding="utf-8")) == written


def test_a_live_reread_prints_its_cost_before_the_judge_is_called_and_runs_under_its_cap(
    found: list[Page], stored: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def replayed() -> tuple[list[Page], list[Any]]:
        return found, []

    events: list[str] = []
    judge = FakeAnchorJudge(_script(found), "judge-v1", events)

    def live_clients(*, dry: bool, judges: tuple[str, ...]) -> Any:
        assert not dry and judges == (_JUDGE,)
        return _clients(judge)

    monkeypatch.setattr(writer_calibration, "pages", replayed)
    monkeypatch.setattr(writer_calibration, "judge_clients", live_clients)
    path = tmp_path / "stored.json"
    path.write_text(json.dumps(stored), encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", _Printed(events))
    out = tmp_path / "new"
    assert main(["--reread", str(path), "--judge", _JUDGE, "--cap-usd", "0.3", "--out", str(out)]) == 0
    monkeypatch.undo()
    # The first thing out is the cost, at the stored read's own $0.01 a call; every judge call comes after it.
    assert events[0] == (
        f"This reread sends 84 calls to {_JUDGE}, each of the 42 anchor cases 2 times, about $0.84 at the $0.0100 a "
        "call the stored record spent."
    )
    assert "judge called" in events and "judge called" not in events[:1]
    # The cap reached the arm: $0.30 at $0.01 a call stops it short of the 84 calls.
    (arm,) = json.loads((out / "reread.json").read_text(encoding="utf-8"))["arms"]
    assert arm["stopped"] is not None and arm["trials"] < 84 and events.count("judge called") < 84


def test_the_cost_printed_before_a_live_reread_is_the_stored_records_own_cost_per_call(stored: dict[str, Any]) -> None:
    # The 2026-11-03 record kept cost only where flash's pair arm stopped at its cap: $1.2098 over 76 trials.
    record = json.loads(stored_record(RECORDS).read_text(encoding="utf-8"))
    assert cost_note(record, judge=SMOKE_JUDGE, k=2) == (
        f"This reread sends 84 calls to {SMOKE_JUDGE}, each of the 42 anchor cases 2 times, about $1.34 at the $0.0159 "
        "a call the stored record spent."
    )
    once = cost_note(record, judge=SMOKE_JUDGE, k=1)
    assert "42 calls" in once and "$0.67" in once
    assert cost_note(record, judge=SMOKE_WRITER, k=2).endswith("the stored record holds no cost per call for it.")
    # A record a reread wrote keeps each trial's cost.
    assert cost_note(stored, judge=_JUDGE, k=1).endswith("about $0.42 at the $0.0100 a call the stored record spent.")
