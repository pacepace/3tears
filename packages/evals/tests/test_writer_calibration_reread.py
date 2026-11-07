"""The writer calibration's reread: the judge asked the anchor set again, set beside its own stored read.

``fixtures/writer_cases/writer_calibration.py --reread`` is a test-retest on the judge's own verdicts over the pair set
(every two rounds of one recording, in both orders). These tests hold it on scripted judges that call no model:

- a judge that flips some preferences and leans to the first slot is reported moved on all three measures, and the
  served-model change between the reads is named as the likely cause;
- the same judge again is reported held on all three, and the agreement with the Opus editor is its own line;
- the stored record is only read: the new read goes to a new ``--out`` directory, and nowhere without one;
- the cost printed before a live reread is the stored record's own cost per call.
"""

from __future__ import annotations

import asyncio
import json
import re
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


# parity-with: threetears.evals.schema.CompletionClient
class FakeAnchorJudge:
    """A judge that answers each pair case from a script keyed by the case's two pages, and names its served model."""

    def __init__(self, scores: dict[str, int], served: str) -> None:
        self.scores = scores
        self.served = served

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, str] | None = None
    ) -> ToyJudgeCompletion:
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


def _clients(judge: FakeAnchorJudge) -> Any:
    return lambda _role, _model, *, temperature=None: judge


def _script(found: list[Page], *, shifted: bool) -> dict[str, int]:
    """Each pair case's score: the Opus editor's verdict, or when ``shifted`` four pairs of rounds flipped in both
    orders and four more answered 1 (Report A, the page read first) in both, each a pair the editor did not tie."""
    criterion = preference_rubric()
    ordered = pairs(found)
    decided = [n for n in range(0, len(ordered), 2) if ordered[n].expected != 3]
    flip, lean = (set(decided[:4]), set(decided[4:8])) if shifted else (set(), set())
    script: dict[str, int] = {}
    for n, pair in enumerate(ordered):
        first = n - n % 2
        score = 1 if first in lean else (6 - pair.expected if first in flip else pair.expected)
        script[pair.case(criterion).output] = score
    return script


def _read(stored: dict[str, Any], found: list[Page], judge: FakeAnchorJudge) -> Reread:
    return asyncio.run(reread(stored, found, judge=_JUDGE, clients=_clients(judge), k=2, cap_usd=None))


@pytest.fixture(scope="module")
def found() -> list[Page]:
    return asyncio.run(pages())[0]


@pytest.fixture(scope="module")
def stored(found: list[Page]) -> dict[str, Any]:
    """The stored record: the steady judge's first read, as ``--out`` writes it and a file gives it back."""
    first = _read({}, found, FakeAnchorJudge(_script(found, shifted=False), "judge-v1"))
    assert all(held is None for held in first.held().values())
    assert f"The stored record holds no per-case verdicts from {_JUDGE}, so this read is its first anchor." in list(
        first.lines()
    )
    record: dict[str, Any] = json.loads(json.dumps(first.record(Path("stored.json")), default=str))
    return record


def test_a_shifted_judge_moves_on_all_three_measures_and_the_served_model_change_is_named(
    found: list[Page], stored: dict[str, Any]
) -> None:
    read = _read(stored, found, FakeAnchorJudge(_script(found, shifted=True), "judge-v2"))
    assert read.held() == dict.fromkeys(MEASURES, False)
    lines = list(read.lines())
    assert lines[0] == (
        f"A reread asks {_JUDGE} the anchor cases again and compares it with its own stored read: it detects a judge "
        "or version shift, never whether the judge is right."
    )
    # The four flipped and the four leaning pairs each changed side; the leaning four lost their order consistency
    # and lean the whole way, 0.5 each over 21 pairs.
    assert read.changed() == (8, 21)
    assert "| Pairs of rounds preferring the later round |" in lines[4] and "| 8 changed side | no |" in lines[4]
    assert lines[5] == "| Pairs whose two orders pick the same round | 21 of 21 | 17 of 21 | -0.19 | no |"
    assert lines[6] == "| First-slot lean, 0 none and 0.5 always the page read first | 0.00 | 0.10 | +0.10 | no |"
    assert "The judge moved on preference, order consistency and first-slot lean." in lines
    assert "Its served model changed from judge-v1 to judge-v2, the likely cause." in lines
    assert lines[-1].startswith("Agreement with the Opus editor's rubric totals, an agent's labels, as weighted κ")
    assert not any(word in line.lower() for line in lines for word in ("calibrat", "accura"))


def test_an_unchanged_judge_holds_on_all_three_and_agrees_with_the_opus_editor_on_its_own_line(
    found: list[Page], stored: dict[str, Any]
) -> None:
    read = _read(stored, found, FakeAnchorJudge(_script(found, shifted=False), "judge-v1"))
    assert read.held() == dict.fromkeys(MEASURES, True)
    assert read.changed() == (0, 21)
    lines = list(read.lines())
    assert "| 0 changed side | yes |" in lines[4]
    assert lines[5] == "| Pairs whose two orders pick the same round | 21 of 21 | 21 of 21 | +0.00 | yes |"
    assert lines[6] == "| First-slot lean, 0 none and 0.5 always the page read first | 0.00 | 0.00 | +0.00 | yes |"
    assert "The judge held on all three measures." in lines
    assert "Its served model is judge-v1 in both reads: no model change." in lines
    # The script answers the Opus editor's own verdicts, so the second line reads full agreement both times.
    assert lines[-1] == (
        "Agreement with the Opus editor's rubric totals, an agent's labels, as weighted κ on the anchor cases: 1.00 in "
        "the stored read and 1.00 in this one."
    )
    # Every trial is a row of the new anchor, as the next reread reads it.
    record = read.record(Path("stored.json"))
    assert len(record["anchor"]) == 2 * 42 and record["case_sets"] == {PAIR_SET: {"cases": 42}}
    assert {(row["judge"], row["served_model"]) for row in record["anchor"]} == {(_JUDGE, "judge-v1")}


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


def test_the_cost_printed_before_a_live_reread_is_the_stored_records_own_cost_per_call(stored: dict[str, Any]) -> None:
    # The 2026-11-03 record kept cost only where flash's pair arm stopped at its cap: $1.2098 over 76 trials.
    record = json.loads(stored_record(RECORDS).read_text(encoding="utf-8"))
    assert cost_note(record, judge=SMOKE_JUDGE, k=2) == (
        f"This reread sends 84 calls to {SMOKE_JUDGE}, each of the 42 anchor cases 2 times, about $1.34 at the $0.0159 "
        "a call the stored record spent."
    )
    assert "42 calls" in cost_note(record, judge=SMOKE_JUDGE, k=1) and "$0.67" in cost_note(
        record, judge=SMOKE_JUDGE, k=1
    )
    assert cost_note(record, judge=SMOKE_WRITER, k=2).endswith("the stored record holds no cost per call for it.")
    # A record a reread wrote keeps each trial's cost.
    assert cost_note(stored, judge=_JUDGE, k=1).endswith("about $0.42 at the $0.0100 a call the stored record spent.")
