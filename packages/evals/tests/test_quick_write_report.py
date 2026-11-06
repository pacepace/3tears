"""The file door's one protocol, and ``harness.analyse``, and ``python -m threetears.evals.harness FILE --write-report MODEL``.

``python -m threetears.evals FILE`` runs every module-level value that satisfies ``Runnable`` (a ``name`` and an
``async run()`` whose result satisfies ``RunResult``), with nothing naming its kind: a test double here is found and
run as an ``Eval`` is. ``analyse`` has the report writer write a result's report, replaying its recording with no key,
writing ``analysis.html`` and linking the stored document as the campaign's analysis, so ``report_read`` serves the
writer's page; a replay that leaves recorded lines unasked says so. The harness door runs a file as the file door does, then analyses each result; with no recording and no key it is refused, naming the file and
``ANTHROPIC_API_KEY``. Over a result held in hand (a runnable whose run hands it back, so its snapshot is fixed and
its recording laid in advance), the door's every after-hook ending is held offline: the page written (exit 0), the
writer storing nothing (1), the store refusing the report (1), a campaign the freeze cannot find (2), and a result
that files no campaign (2).

The re-run check: rung 1 run twice makes two campaigns, two snapshots, and a recording keyed to the first's, which the
second finds none of. Nothing reaches a model: every reply is scripted, and every tool call runs live on the
frozen snapshot.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Final

import pytest

from packages.evals.tests.report_writer_support import (
    MODEL,
    FakeToolTurnClient,
    RecordedModel,
    call,
    hand_written,
    reply,
    review,
    submitted,
)
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.rung_support import rung_copy
from threetears.evals.actions.catalogue import Caller
from threetears.evals.actions.report_writing import write_report
from threetears.evals.harness.__main__ import PROG
from threetears.evals.harness.__main__ import main as harness_main
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.harness.analyse import analyse, analysis_page, recording_path
from threetears.evals.ops import campaigns_list, report_read
from threetears.evals.ops.analyst import report_snapshot_freeze
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.report_writer.loop import MOST_REVISIONS
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.kernel import EvalReportDoc, EvalReportSession
from threetears.evals.kernel.errors import StorageError
from threetears.evals.quick import EXIT_GATE_FAILED, EXIT_OK, EXIT_REFUSED, EXIT_RUN_DID_NOT_COMPLETE
from threetears.evals.quick.files import RunResult, load_evals, main
from threetears.evals.quick.result import CREATED_BY, Result
from threetears.evals.schema.turns import TurnReply, TurnText
from threetears.evals.testing import RecordedTurns, RecordingRefused

#: Rung 1's eval, named for its cases file.
RUNG_ONE: Final = "emails"

#: How many analyst replies a report that never submits asks for before the loop gives up on it.
SILENT_TURNS: Final = 4

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


@pytest.fixture(autouse=True)
def _own_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading a Python file puts its directory on ``sys.path`` for the process; the test's process gets its own back."""
    monkeypatch.setattr(sys, "path", [*sys.path])


def _session(snapshot: str) -> dict[str, list[TurnReply]]:
    """The scripted report, by session, its tool calls quoting ``snapshot``: the analyst reads the lookalike
    examples and submits; the editor reads them too and clears it."""
    examples = {"arm_filter": "triage", "stratum_filter": "lookalike"}
    return {
        "analyst": [reply(call(1, "examples_list", snapshot, **examples)), reply(submitted(RUNG_ONE_DOC))],
        "editor-1": [reply(call(2, "examples_list", snapshot, **examples)), reply(review())],
    }


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


async def _run_rung_one() -> Result:
    """Rung 1 loaded afresh and run, as the file door runs it: every call replayed from its recording."""
    (ev,) = load_evals(Path("triage.py"))
    result = await ev.run()
    assert isinstance(result, Result)
    return result


async def _snapshot_of(result: Result) -> str:
    """The snapshot ``analyse`` freezes ``result``'s campaign as: the same evidence frozen twice names one."""
    ops = OpsHost.reading(result.host)
    frozen = await report_snapshot_freeze(ops, result.scope_id, campaign_id=result.campaign_id, created_by=CREATED_BY)
    return frozen.snapshot_id


async def test_analyse_writes_the_page_from_a_hand_written_recording_with_no_key(
    rung_one: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = await _run_rung_one()
    recording = hand_written(recording_path(RUNG_ONE, MODEL, await _snapshot_of(result)), _session("{snapshot}"))
    assert recording.is_relative_to(rung_one / "recordings" / "report")

    written = await analyse(result, MODEL)

    assert (written.ending, written.outcome.analyst_rounds, written.outcome.editor_rounds) == ("stored", 1, 1)
    assert "drift:" not in capsys.readouterr().out
    html = analysis_page(RUNG_ONE).read_text(encoding="utf-8")
    assert "Triage passes most emails" in html and "<html" in html


async def test_analyse_links_its_stored_report_as_the_campaign_s_analysis(
    rung_one: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The page ``report`` and ``report_read`` serve is the writer's once ``analyse`` has stored it: the document is
    the campaign's analysis (``doc``), one attempt reads ``stored``, and the listing reads the report written."""
    result = await _run_rung_one()
    hand_written(recording_path(RUNG_ONE, MODEL, await _snapshot_of(result)), _session("{snapshot}"))
    before = off_loop(report_read, result.host, result.campaign_id, result.scope_id, format="json")
    assert (before.basis, before.analysis_id) == ("code_only", None)

    written = await analyse(result, MODEL)

    assert written.stored is not None
    served = off_loop(report_read, result.host, result.campaign_id, result.scope_id, format="json")
    assert (served.basis, served.analysis_basis, served.needs_regeneration) == ("analysis", "doc", None)
    assert served.doc is not None and served.doc["headline"] == RUNG_ONE_DOC["headline"]
    page = off_loop(report_read, result.host, result.campaign_id, result.scope_id, format="markdown")
    assert RUNG_ONE_DOC["headline"] in page.body
    (attempt,) = off_loop(result.host.storage.list_analysis_attempts_by_campaign, result.campaign_id, result.scope_id)
    assert (attempt.outcome, attempt.analysis_id, attempt.generator_model) == ("stored", served.analysis_id, MODEL)
    (line,) = off_loop(campaigns_list, result.host, result.scope_id).campaigns
    assert line.report_state == "written" and line.report is not None
    assert line.report.analysis_id == served.analysis_id


async def test_a_replay_that_leaves_lines_unasked_prints_a_disclosure(
    rung_one: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = await _run_rung_one()
    leftover = {**_session("{snapshot}"), "editor-9": [reply(review())]}
    recording = hand_written(recording_path(RUNG_ONE, MODEL, await _snapshot_of(result)), leftover)

    written = await analyse(result, MODEL)

    drift = [line for line in capsys.readouterr().out.splitlines() if line.startswith("drift:")]
    assert len(drift) == 1 and "'editor-9'" in drift[0] and str(recording) in drift[0]
    assert written.ending == "stored", "the recorded sessions still ran as recorded"


def test_write_report_with_no_recording_and_no_key_is_refused_naming_the_file_and_the_key(
    rung_one: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert harness_main(["triage.py", "--write-report", MODEL]) == EXIT_REFUSED

    captured = capsys.readouterr()
    assert "report: .evals/reports/emails/report.html" in captured.out, "the file door ran and reported first"
    refused = captured.err
    directory = (
        rung_one / "recordings" / "report" / f"shipped-{load_skill().digest[:12]}-{MODEL}-default-r{MOST_REVISIONS}"
    )
    assert f"{directory}/{RUNG_ONE}-" in refused and ".jsonl" in refused and "ANTHROPIC_API_KEY" in refused
    assert not analysis_page(RUNG_ONE).exists()


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


async def test_a_re_run_of_rung_one_is_a_new_snapshot_no_earlier_recording_replays_over(
    rung_one: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The re-run check (P1 design, section 3). A session recorded on the first run's campaign is keyed to its
    snapshot, so the second run's campaign finds no recording and, with no key, is refused: ``--write-report`` over
    a fresh run is a keyed, live action. The replies themselves do not drift: laid at the second snapshot's path,
    the same recording replays to the same ending with nothing unasked."""
    first = await _run_rung_one()
    ops = OpsHost.reading(first.host)
    snapshot = await _snapshot_of(first)
    # Recorded as a live model would be: its replies quote the first run's snapshot, which the recording keeps as the
    # placeholder; one client answers the analyst's turns, then the editor's.
    script = _session(snapshot)
    live = FakeToolTurnClient(*script["analyst"], *script["editor-1"], ceiling=0.05)
    path = recording_path(RUNG_ONE, MODEL, snapshot)
    recorded = await write_report(
        ops,
        Caller(scope_id=first.scope_id, identity=CREATED_BY),
        model=RecordedModel(RecordedTurns.record(live, path), MODEL),
        budget=ReportBudget(usd=3.0, turns=40),
        snapshot_id=snapshot,
    )
    assert recorded.ending == "stored" and path.is_file()

    second = await _run_rung_one()
    again = await _snapshot_of(second)
    assert second.campaign_id != first.campaign_id and again != snapshot

    with pytest.raises(RecordingRefused, match="ANTHROPIC_API_KEY") as refused:
        await analyse(second, MODEL)
    assert refused.value.path == recording_path(RUNG_ONE, MODEL, again)

    shutil.copyfile(path, recording_path(RUNG_ONE, MODEL, again))
    replayed = await analyse(second, MODEL)

    assert (replayed.ending, replayed.outcome.analyst_rounds, replayed.outcome.editor_rounds) == ("stored", 1, 1)
    assert "drift:" not in capsys.readouterr().out


# parity-with: threetears.evals.quick.files.Runnable
class FakeHeldRunnable:
    """A runnable whose run hands back a result already run, set on the class by the test: its campaign, and so the
    snapshot the writer reads, are fixed before the door runs it, so the writer's recording is laid in advance."""

    name = RUNG_ONE
    held: ClassVar[RunResult | None] = None

    async def run(self) -> RunResult:
        assert self.held is not None, "the test holds a result before the door runs the file"
        return self.held


#: An eval file defining one runnable over a result held in hand.
HELD = f"""\
from {__name__} import FakeHeldRunnable

held = FakeHeldRunnable()
"""


def _held(monkeypatch: pytest.MonkeyPatch, result: RunResult) -> Path:
    """An eval file in the working directory whose one runnable hands back ``result``."""
    monkeypatch.setattr(FakeHeldRunnable, "held", result)
    path = Path("held.py")
    path.write_text(HELD, encoding="utf-8")
    return path


def _ran_rung_one() -> tuple[Result, str]:
    """Rung 1 run in a loop of its own, and the snapshot its campaign freezes as, in another."""
    result = asyncio.run(_run_rung_one())
    return result, asyncio.run(_snapshot_of(result))


def test_the_harness_door_writes_the_analysis_of_a_held_result_from_its_recording(
    rung_one: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result, snapshot = _ran_rung_one()
    recording = hand_written(recording_path(RUNG_ONE, MODEL, snapshot), _session("{snapshot}"))

    assert harness_main([str(_held(monkeypatch, result)), "--write-report", MODEL]) == EXIT_OK

    printed = capsys.readouterr().out
    assert f"analysis: {analysis_page(RUNG_ONE)}, written from {recording}" in printed
    assert "drift:" not in printed
    assert "Triage passes most emails" in analysis_page(RUNG_ONE).read_text(encoding="utf-8")


def test_a_writer_that_stores_nothing_exits_1_with_no_page(
    rung_one: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result, snapshot = _ran_rung_one()
    silent = [reply(TurnText("I have no report yet.")) for _ in range(SILENT_TURNS)]
    hand_written(recording_path(RUNG_ONE, MODEL, snapshot), {"analyst": silent})

    assert harness_main([str(_held(monkeypatch, result)), "--write-report", MODEL]) == EXIT_RUN_DID_NOT_COMPLETE

    (said,) = [line for line in capsys.readouterr().out.splitlines() if line.startswith("analysis:")]
    assert said.startswith("analysis: none; the writer ended refused: ")
    assert not analysis_page(RUNG_ONE).exists()


def test_a_report_the_store_refuses_exits_1_saying_so(
    rung_one: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result, snapshot = _ran_rung_one()
    hand_written(recording_path(RUNG_ONE, MODEL, snapshot), _session("{snapshot}"))

    def refuse(report: EvalReportDoc, *, sessions: Sequence[EvalReportSession] = ()) -> None:
        raise StorageError("the document is larger than the store takes")

    monkeypatch.setattr(result.host.storage, "save_report_doc", refuse)
    path = _held(monkeypatch, result)

    assert harness_main([str(path), "--write-report", MODEL]) == EXIT_RUN_DID_NOT_COMPLETE

    refused = capsys.readouterr().err
    assert f"{PROG} {path}: {RUNG_ONE}: the writer's report was not stored: the report ended stored" in refused
    assert not analysis_page(RUNG_ONE).exists()


def test_a_campaign_the_freeze_cannot_find_is_refused_with_exit_2(
    rung_one: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result, _ = _ran_rung_one()
    path = _held(monkeypatch, replace(result, campaign_id="campaign-nowhere"))

    assert harness_main([str(path), "--write-report", MODEL]) == EXIT_REFUSED

    refused = capsys.readouterr().err
    assert f"{PROG} {path}: {RUNG_ONE}: " in refused and "campaign-nowhere" in refused
    assert not analysis_page(RUNG_ONE).exists()


def test_a_result_that_files_no_campaign_is_refused_with_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("double.py").write_text(f"from {__name__} import FakeRunnable\n\npasses = FakeRunnable('passes')\n")

    assert harness_main(["double.py", "--write-report", MODEL]) == EXIT_REFUSED

    captured = capsys.readouterr()
    assert "report: .evals/reports/passes/report.html" in captured.out, "the file door ran and reported first"
    assert f"{PROG} double.py: passes: its result files no campaign for the writer to read" in captured.err


def test_the_file_door_refusing_write_report_names_the_door_that_takes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("doubles.py").write_text(DOUBLES, encoding="utf-8")

    assert main(["doubles.py", "--write-report", MODEL]) == EXIT_REFUSED

    assert "(--write-report is python -m threetears.evals.harness's)" in capsys.readouterr().err
