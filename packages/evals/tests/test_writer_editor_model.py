"""The editor model lever: a writer arm whose editor's reviews run on a model of their own.

``Writer(model, editor_model=...)`` names the editor's model. The launcher asks the host's model factory for the
analyst's model and then for the editor's, each with every lever of the arm, and the editor's review sessions run on
the second: each session's replies name the model it ran on, and a session handed another model's client is refused.
The run records the lever on its overlays, and its candidate model, the page's author, stays the analyst's; the
editor's turns are on its usage under the editor's model. Two arms differing only in the editor are a comparison, its
design axis the lever. (An editor of the scoring judge's family is refused at launch, as a writer of it is:
``tests/test_report_writer_launch.py``.)

Nothing reaches a model: Kairo's report is replayed from hand-laid recordings, the analyst's and the editor's in two
files, and a scripted judge scores the pages; every socket connect fails.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Final

import pytest

from packages.evals.tests.offline_support import Connect, assert_no_connects, refuse_connects
from packages.evals.tests.writer_eval_support import FakeScoringJudge, kairo_case, kairo_report, lay_recording
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.harness import Writer, WriterEval, WriterResult
from threetears.evals.harness.writer_turns import writer_turns
from threetears.evals.kernel import EvalStorage
from threetears.evals.providers.wire import KEY_VARIABLES as SHIPPED_KEY_VARIABLES
from threetears.evals.schema.models import EvalRun
from threetears.evals.storage import SqliteDocumentStore

#: The writer every arm's analyst runs on.
WRITER: Final = "report-writer-candidate"

#: The model the edited arm's editor runs on: a model of its own, of the writer's provider.
EDITOR: Final = "report-editor-candidate"

SCOPE: Final = "editor-lever"


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> list[Connect]:
    """No key in the environment, and every socket connect refused and counted."""
    for variable in SHIPPED_KEY_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    return refuse_connects(monkeypatch)


def _arms(editor: str = EDITOR) -> dict[str, Writer]:
    return {"own editor": Writer(WRITER), "other editor": Writer(WRITER, editor_model=editor)}


def _laid(tmp_path: Path, arms: dict[str, Writer]) -> None:
    """Kairo's case file, and its report laid where each arm's router reads it: one file for the arm whose editor runs
    on the writer, the analyst's and the editor's apart for the arm naming one."""
    kairo = kairo_case(tmp_path / "cases")
    for arm, writer in arms.items():
        routed = writer_turns(
            tmp_path / "turns",
            arm=arm,
            skill=writer.analyst_skill,
            cases={kairo.snapshot.id: "kairo"},
            editor_model=writer.editor_model,
        )
        report = kairo_report(WRITER, writer.editor_model or WRITER)
        if writer.editor_model is None:
            lay_recording(routed.path(WRITER, None, kairo.snapshot.id), report)
            continue
        lay_recording(routed.path(WRITER, None, kairo.snapshot.id), {"analyst": report["analyst"]})
        lay_recording(routed.path(WRITER, None, kairo.snapshot.id, role="editor"), {"editor-1": report["editor-1"]})


def _eval(tmp_path: Path, arms: dict[str, Writer], judge: FakeScoringJudge) -> WriterEval:
    return WriterEval(
        tmp_path / "cases",
        arms=arms,
        judge=judge,
        recordings=tmp_path / "turns",
        budget_usd=1.0,
        max_cost_usd=4.0,
        name=SCOPE,
        store=tmp_path / "evals.sqlite",
    )


def _runs(tmp_path: Path, result: WriterResult) -> dict[str, EvalRun]:
    """Each arm's stored run, by arm."""
    storage = EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite"))
    runs = {}
    for arm, summary in result.arms.items():
        run = storage.load_eval_run(summary.run_id, SCOPE)
        assert run is not None
        runs[arm] = run
    return runs


def _usage_models(tmp_path: Path, run: EvalRun) -> set[str]:
    """The models the candidate role's usage rows name, over every result of ``run``."""
    storage = EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite"))
    results = storage.query_eval_results_by_run(run.id, SCOPE)
    return {row.model for result in results for row in result.usage if row.role == "candidate"}


async def test_the_editor_s_reviews_run_on_its_own_model_recorded_on_the_run(
    tmp_path: Path, offline: list[Connect]
) -> None:
    arms = _arms()
    _laid(tmp_path, arms)
    judge = FakeScoringJudge()

    result = await _eval(tmp_path, arms, judge).run()

    # Both pages were written, cleared and judged, and every recorded line was asked: the editor's replies, which name
    # the editor's model, were served to a session running on it.
    for summary in result.arms.values():
        assert summary.status == "completed" and summary.errors == [] and summary.n_cases == 1
    text = result.render()
    assert "unasked" not in text
    assert len(judge.calls) == 2 * len({*judge.calls})
    runs = await asyncio.to_thread(_runs, tmp_path, result)
    # The lever is on the run; the run's model, the page's author, stays the analyst's.
    assert runs["other editor"].overlays["editor_model"] == EDITOR
    assert runs["own editor"].overlays.get("editor_model") is None
    assert {run.candidate_model for run in runs.values()} == {WRITER}
    # The editor's turns are on the run's usage under the editor's model, the analyst's under the writer's.
    assert await asyncio.to_thread(_usage_models, tmp_path, runs["other editor"]) == {WRITER, EDITOR}
    assert await asyncio.to_thread(_usage_models, tmp_path, runs["own editor"]) == {WRITER}
    # Two arms differing only in the editor are a comparison on the lever.
    campaign = await asyncio.to_thread(
        EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite")).load_campaign, result.campaign_id, SCOPE
    )
    assert campaign is not None and campaign.aspects[0].design is not None
    assert [axis.axis_id for axis in campaign.aspects[0].design.axes] == ["report_writer.editor_model"]
    assert_no_connects(offline)


def test_a_writer_naming_a_blank_editor_is_refused() -> None:
    with pytest.raises(ValueError, match="names its editor's model by id, and this one is blank"):
        Writer(WRITER, editor_model=" ")
    assert Writer(WRITER, skill=load_skill()).editor_model is None
