"""The file door's one protocol, and ``python -m threetears.evals.harness FILE --write-report MODEL``.

``python -m threetears.evals FILE`` runs every module-level value that satisfies ``Runnable`` (a ``name`` and an
``async run()`` whose result satisfies ``RunResult``), with nothing naming its kind: a test double here is found and
run as an ``Eval`` is. The harness door runs a file the same way, then the report writer analyses each result
(``harness.analyse``), replaying its recording with no key and writing ``analysis.html``; with no recording and no key
it is refused, naming the file and ``ANTHROPIC_API_KEY``; a replay that leaves recorded lines unasked says so.

The re-run check: rung 1 run twice makes two campaigns, and a session recorded on the first run's campaign replays on
the second's as recorded. Nothing reaches a model: every reply is scripted, and every tool call runs live on the
frozen snapshot.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pytest

from packages.evals.tests.report_writer_support import (
    MODEL,
    FakeToolTurnClient,
    RecordedModel,
    call,
    reply,
    review,
    submitted,
)
from packages.evals.tests.rung_support import rung_copy
from threetears.evals.actions.catalogue import Caller
from threetears.evals.actions.report_writing import write_report
from threetears.evals.harness.__main__ import main as harness_main
from threetears.evals.harness.analyse import TURN_CAP, analyse, analysis_page, recording_path
from threetears.evals.ops.analyst import report_snapshot_freeze
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.quick import EXIT_GATE_FAILED, EXIT_OK, EXIT_REFUSED
from threetears.evals.quick.files import load_evals, main
from threetears.evals.quick.result import CREATED_BY, Result
from threetears.evals.schema.turns import TurnReply
from threetears.evals.testing import RecordedTurns

#: Rung 1's eval, named for its cases file.
RUNG_ONE: Final = "emails"

#: A report on rung 1's one-arm campaign that passes validity and every invariant on its snapshot: it recommends no
#: arm, lists the caveat code assigns, and shows its lead claim on the lookalike examples, by query.
RUNG_ONE_DOC: Final[dict[str, Any]] = {
    "schema": "report-doc/1",
    "story": "status",
    "headline": "Triage passes most emails, and lookalikes fail most often.",
    "lede": "One arm was measured, so there is nothing to choose between.",
    "flag": None,
    "recommends": [],
    "confidence": {
        "form": "low",
        "caveats": [
            {
                "id": "nothing_declared",
                "label": None,
                "material_if": "only if an owner bar sits near the pass rate",
                "checked": None,
            }
        ],
    },
    "decision": [{"slot": "recommendation", "text": "No call: one arm was measured."}],
    "claims": [
        {
            "id": "c1",
            "text": "Lookalike emails fail more often than plain ones.",
            "weight": "lead",
            "evidence": [
                {
                    "block": "examples",
                    "query": {
                        "source": "examples",
                        "aspect": "classify",
                        "measures": [],
                        "arms": ["triage"],
                        "strata": ["lookalike"],
                        "limit": 3,
                        "transform": "none",
                        "small_multiples_by": "none",
                    },
                    "reference": None,
                    "emphasis": "",
                    "caption": "Three failed lookalike emails.",
                }
            ],
            "caveats": [],
            "durable": True,
        }
    ],
    "explore": [],
    "omitted": [],
    "next": [],
    "method": [{"term": "Arms", "text": "One triage function.", "caveats": []}],
}


def _session(snapshot: str) -> dict[str, list[TurnReply]]:
    """The scripted report, by session, its tool calls quoting ``snapshot``: the analyst reads the lookalike
    examples and submits; the editor reads them too and clears it."""
    examples = {"arm_filter": "triage", "stratum_filter": "lookalike"}
    return {
        "analyst": [reply(call(1, "examples_list", snapshot, **examples)), reply(submitted(RUNG_ONE_DOC))],
        "editor-1": [reply(call(2, "examples_list", snapshot, **examples)), reply(review())],
    }


def _hand_written(path: Path, sessions: Mapping[str, Sequence[TurnReply]]) -> Path:
    """Write ``sessions`` as a turn recording by hand, the snapshot already the placeholder: each turn's ceiling, then
    its reply."""
    lines: list[dict[str, Any]] = []
    for session, replies in sessions.items():
        for index, said in enumerate(replies):
            line = {
                "recorder_version": RecordedTurns.FORMAT_VERSION,
                "recorded_at": "2026-11-03T00:00:00+00:00",
                "session": session,
                "index": index,
            }
            lines += [{**line, "kind": "ceiling", "ceiling": 0.05}, {**line, "kind": "reply", "reply": said.to_json()}]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


@pytest.fixture
def rung_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of ``examples/1-triage/`` as the working directory, with no key."""
    return rung_copy("1-triage", tmp_path, monkeypatch)


# parity-with: threetears.evals.quick.files.RunResult
class FakeRunResult:
    """What the double's run comes to: a line, a report page, and the exit code it was built with."""

    def __init__(self, name: str, code: int) -> None:
        self.name = name
        self.code = code

    def render(self, *, full: bool = False) -> str:
        return f"{self.name}: ran through the protocol{' (whole)' if full else ''}"

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        page = Path(out_dir) / "report.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(f"<p>{self.name}</p>", encoding="utf-8")
        return {"html": page}

    def exit_code(self) -> int:
        return self.code

    def why_failed(self) -> str | None:
        return None if self.code == EXIT_OK else f"the double {self.name} failed its gate on purpose"


# parity-with: threetears.evals.quick.files.Runnable
class FakeRunnable:
    """A runnable no part of the package defines: a name, and a run."""

    def __init__(self, name: str, code: int = EXIT_OK) -> None:
        self.name = name
        self.code = code

    async def run(self) -> FakeRunResult:
        return FakeRunResult(self.name, self.code)


#: An eval file defining two doubles, the class itself, and a value with a ``name`` and a ``run`` that is no coroutine.
DOUBLES = f"""\
from types import SimpleNamespace

from {__name__} import FakeRunnable

passes = FakeRunnable("passes")
gated = FakeRunnable("gated", code={EXIT_GATE_FAILED})
the_class = FakeRunnable
not_async = SimpleNamespace(name="not async", run=lambda: None)
"""


def test_the_file_door_runs_every_value_the_protocol_reads_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("doubles.py").write_text(DOUBLES, encoding="utf-8")

    assert [runnable.name for runnable in load_evals(Path("doubles.py"))] == ["passes", "gated"]
    assert main(["doubles.py"]) == EXIT_GATE_FAILED

    printed = capsys.readouterr().out
    assert printed.count("ran through the protocol") == 2
    assert "report: .evals/reports/passes/report.html" in printed
    assert f"exit {EXIT_GATE_FAILED}: the double gated failed its gate on purpose" in printed
    assert "not async" not in printed


def test_write_report_writes_the_analysis_from_a_hand_written_recording_with_no_key(
    rung_one: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = _hand_written(rung_one / recording_path(RUNG_ONE, MODEL), _session("{snapshot}"))

    assert harness_main(["triage.py", "--write-report", MODEL]) == EXIT_OK

    printed = capsys.readouterr().out
    page = analysis_page(RUNG_ONE)
    assert f"analysis: {page}, written from" in printed and str(recording) in printed
    assert "drift:" not in printed
    html = page.read_text(encoding="utf-8")
    assert "Triage passes most emails" in html and "<html" in html
    assert (page.parent / "report.html").is_file(), "the eval's own report is written first, as the file door writes it"


def test_no_recording_and_no_key_is_refused_naming_the_file_and_the_key(
    rung_one: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert harness_main(["triage.py", "--write-report", MODEL]) == EXIT_REFUSED

    refused = capsys.readouterr().err
    assert str(rung_one / recording_path(RUNG_ONE, MODEL)) in refused and "ANTHROPIC_API_KEY" in refused
    assert not analysis_page(RUNG_ONE).exists()


def test_a_replay_that_leaves_lines_unasked_prints_a_disclosure(
    rung_one: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session = _session("{snapshot}")
    leftover = {**session, "editor-9": [reply(review())]}
    recording = _hand_written(rung_one / recording_path(RUNG_ONE, MODEL), leftover)

    assert harness_main(["triage.py", "--write-report", MODEL]) == EXIT_OK

    printed = capsys.readouterr().out
    drift = [line for line in printed.splitlines() if line.startswith("drift:")]
    assert len(drift) == 1 and "'editor-9'" in drift[0] and str(recording) in drift[0]
    assert analysis_page(RUNG_ONE).is_file(), "the recorded sessions still ran as recorded"


@pytest.mark.parametrize(
    ("args", "refusal"),
    [
        (["triage.py"], "--write-report MODEL"),
        (["triage.py", "--write-report"], "names no value"),
        (["triage.py", "--write-report", MODEL, "--budget-usd", "free"], "not 'free'"),
        (["triage.py", "--write-report", MODEL, "--budget-usd", "0"], "above 0"),
        (["report", "--write-report", MODEL], "eval file"),
    ],
)
def test_the_harness_door_refuses_what_it_cannot_run(
    args: list[str], refusal: str, rung_one: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert harness_main(args) == EXIT_REFUSED
    assert refusal in capsys.readouterr().err


async def _run_rung_one() -> Result:
    """Rung 1 loaded afresh and run, as the file door runs it: every call replayed from its recording."""
    (ev,) = load_evals(Path("triage.py"))
    result = await ev.run()
    assert isinstance(result, Result)
    return result


async def test_a_session_recorded_on_one_run_replays_on_a_re_run_of_rung_one(
    rung_one: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = await _run_rung_one()
    ops = OpsHost.reading(first.host)
    frozen = await report_snapshot_freeze(ops, first.scope_id, campaign_id=first.campaign_id, created_by=CREATED_BY)
    # Recorded as a live model would be: its replies quote the first run's snapshot, which the recording keeps as the
    # placeholder; one client answers the analyst's turns, then the editor's.
    script = _session(frozen.snapshot_id)
    live = FakeToolTurnClient(*script["analyst"], *script["editor-1"], ceiling=0.05)
    path = recording_path(RUNG_ONE, MODEL)
    recorded = await write_report(
        ops,
        Caller(scope_id=first.scope_id, identity=CREATED_BY),
        model=RecordedModel(RecordedTurns.record(live, path), MODEL),
        budget=ReportBudget(usd=3.0, turns=TURN_CAP),
        snapshot_id=frozen.snapshot_id,
    )
    assert recorded.ending == "stored" and path.is_file()

    second = await _run_rung_one()
    assert second.campaign_id != first.campaign_id

    replayed = await analyse(second, MODEL)

    assert replayed.snapshot_id != frozen.snapshot_id, "the second run's campaign freezes its own snapshot"
    assert (replayed.ending, replayed.outcome.analyst_rounds, replayed.outcome.editor_rounds) == ("stored", 1, 1)
    assert "drift:" not in capsys.readouterr().out
    assert "Triage passes most emails" in analysis_page(RUNG_ONE).read_text(encoding="utf-8")
