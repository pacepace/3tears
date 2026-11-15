"""A run launched now records every coordinate of its identity: "unrecorded" is for runs stored before this build.

Stating a coordinate after the fact (``runs_attest``) is an upgrade step for old data, never something a new run
needs. So every kind of launch the package offers is held to it here: the run it stores records the definitions of
its host's measures, and each of its candidate calls names what answered it. A scripted or replayed completion
records what answered (a replay whose recorded reply named none is answered by the recording itself: completions and
writer turns alike). Where the package cannot know, because the host's own code answered and named nothing (a log
whose lines name no served model, an ``Answer`` with a model and no ``served_model``), the run says so, naming the
coordinate and the remedy: a log as it launches, an ``Answer`` once its arm ran, both on the result and as a WARNING,
never left to surface later as a campaign that will not compare. A take-up of a run stored before this build is held
in ``test_run_resume.py``: it stays unrecorded, and says so as it is taken up.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from threetears.evals import Answer, Eval, Log, Model, Result
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.metrics import holds_host_measure
from threetears.evals.kernel.scoring import definitions_unrecorded
from threetears.evals.ops import LaunchArguments, OpsHost, run_launch
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.schema import TurnMessage
from threetears.evals.testing import RecordedTurns
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import launched_runs
from packages.evals.tests.recorded_model_support import record_model

TICKETS = [
    {"id": "refund", "text": "please refund me", "label": "billing"},
    {"id": "crash", "text": "the app crashes", "label": "bug"},
]


def _unrecorded(storage: EvalStorage, run_id: str, scope_id: str) -> list[str]:
    """Each coordinate of the stored run's identity that nothing recorded, by name; empty for a run that says all."""

    def read() -> list[str]:
        run = storage.load_eval_run(run_id, scope_id)
        assert run is not None and run.status == "completed", run
        results = storage.query_eval_results_by_run(run_id, scope_id)
        assert results, "the run stored its results"
        missing: list[str] = []
        holds = any(holds_host_measure(result.host_measures) for result in results)
        if definitions_unrecorded(run, holds_host_measures=holds):
            missing.append("measure_definition")
        if any(row.role == "candidate" and not row.served_model for result in results for row in result.usage):
            missing.append("served_model")
        if run.template_digest is None:
            missing.append("template")
        if run.context_key is None:
            missing.append("rig")
        return missing

    return off_loop(read)


def _unrecorded_of(result: Result) -> list[str]:
    """:func:`_unrecorded` for a quick result's one run."""
    (run_id,) = result.run_ids.values()
    return _unrecorded(result.host.storage, run_id, result.scope_id)


async def test_a_quick_run_of_a_plain_function_records_every_coordinate() -> None:
    def keywords(case: Mapping[str, Any]) -> dict[str, str]:
        """Label the ticket by its keywords."""
        return {"label": "billing" if "refund" in str(case["text"]) else "bug"}

    result = await Eval(TICKETS, keywords, ["label"], k=2).run(store=":memory:")

    assert _unrecorded_of(result) == [], "a function that calls no model recorded that: no candidate call"


async def test_a_quick_run_replaying_a_model_recording_records_what_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RECORD", raising=False)
    path = tmp_path / "recordings" / "label.jsonl"
    await record_model(path, "scripted/labeller", [str(t["text"]) for t in TICKETS], lambda _: "billing", repeats=2)
    model = Model("scripted/labeller", recording=path)

    async def labeller(case: Mapping[str, Any]) -> dict[str, str]:
        """Label the ticket."""
        return {"label": await model.text(str(case["text"]))}

    result = await Eval(TICKETS, labeller, ["label"], k=2).run(store=":memory:")

    assert _unrecorded_of(result) == []


async def test_a_quick_run_whose_answer_names_its_served_model_records_it() -> None:
    def scripted(case: Mapping[str, Any]) -> Answer:
        """A scripted stand-in for a model, which knows what answers: itself."""
        return Answer({"label": "billing"}, model="scripted/labeller", served_model="scripted/labeller", cost_usd=0.0)

    result = await Eval(TICKETS, scripted, ["label"], k=1).run(store=":memory:")

    assert _unrecorded_of(result) == []


async def test_a_host_launch_over_its_scripted_client_records_every_coordinate() -> None:
    def store() -> EvalStorage:
        storage = EvalStorage(InMemoryDocumentStore())
        storage.save_template(toyhost_template())
        return storage

    storage = off_loop(store)
    launch, _client = toyhost_launch_host(storage=storage)
    host = OpsHost(launch=launch)
    arguments = LaunchArguments(
        template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=[RUN_MODELS[0]], k_runs=1
    )

    (run_id,) = await launched_runs(host, await run_launch(host, arguments, TOYHOST_SCOPE))

    assert _unrecorded(storage, run_id, TOYHOST_SCOPE) == []


async def test_a_log_whose_lines_name_their_served_model_records_it_and_says_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log = Log(tmp_path / "outputs.jsonl")
    for ticket in TICKETS:
        log(ticket, {"label": "billing"}, model="vendor/alias-latest", served_model="vendor/model-2026-08")

    with caplog.at_level(logging.WARNING):
        result = await Eval(TICKETS, log, ["label"], k=1).run(store=":memory:")

    assert _unrecorded_of(result) == []
    assert not [line for line in result.disclosures if "served model" in line]
    assert not [record for record in caplog.records if "served_model" in record.getMessage()]


async def test_a_log_that_names_no_served_model_says_so_as_it_launches_with_the_coordinate_and_the_remedy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The one launch that cannot know what answered: it is said at launch, on the result and in the log, and never
    left to a campaign that will not compare."""
    log = Log(tmp_path / "outputs.jsonl")
    for ticket in TICKETS:
        log(ticket, {"label": "billing"}, model="vendor/alias-latest")

    with caplog.at_level(logging.WARNING):
        result = await Eval(TICKETS, log, ["label"], k=1).run(store=":memory:")

    assert _unrecorded_of(result) == ["served_model"], "the coordinate really is unknown, and only that one"
    (said,) = [line for line in result.disclosures if "served model" in line]
    assert "identity coordinate served_model" in said and "pools with no other run" in said
    assert "log served_model= with each output" in said and "runs_attest, coordinate served_model" in said
    assert said in result.render(), "the run's own print says it"
    assert [record for record in caplog.records if record.getMessage() == said and record.levelno == logging.WARNING]


async def test_a_quick_answer_naming_a_model_and_no_served_model_says_so_once_its_arm_ran(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The host's code answered and named nothing: the package cannot know what answered, and says so on the result
    and in the log, with the coordinate and the remedy."""

    def scripted(case: Mapping[str, Any]) -> Answer:
        """A host function that called something it does not name."""
        return Answer({"label": "billing"}, model="vendor/alias-latest", cost_usd=0.0)

    with caplog.at_level(logging.WARNING):
        result = await Eval(TICKETS, scripted, ["label"], k=1).run(store=":memory:")

    assert _unrecorded_of(result) == ["served_model"], "the coordinate really is unknown, and only that one"
    (said,) = [line for line in result.disclosures if "served model" in line]
    assert f"{len(TICKETS)} of {len(TICKETS)} result(s)" in said and "identity coordinate served_model" in said
    assert "Answer(served_model=...)" in said and "runs_attest, coordinate served_model" in said
    assert [record for record in caplog.records if record.getMessage() == said and record.levelno == logging.WARNING]


async def test_a_quick_answer_naming_its_served_model_says_nothing(caplog: pytest.LogCaptureFixture) -> None:
    def scripted(case: Mapping[str, Any]) -> Answer:
        return Answer({"label": "billing"}, model="m", served_model="m-2026-08", cost_usd=0.0)

    with caplog.at_level(logging.WARNING):
        result = await Eval(TICKETS, scripted, ["label"], k=1).run(store=":memory:")

    assert not [line for line in result.disclosures if "served model" in line]
    assert not [record for record in caplog.records if "served_model" in record.getMessage()]


async def test_a_replayed_writer_turn_whose_recorded_reply_named_no_model_is_answered_by_the_recording(
    tmp_path: Path,
) -> None:
    """A writer replay knows what answers it: the recording, the same identity for every turn of one file."""
    path = tmp_path / "turns.jsonl"
    reply = {
        "content": [{"type": "text", "text": "Done."}],
        "stop_reason": "end_turn",
        "model": "m",
        "served_model": None,
        **dict.fromkeys(
            (
                "provider_stop_reason",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "cost_usd",
                "price_source",
                "pricing_failure",
                "sent_binding",
                "input_transformations",
            )
        ),
    }
    lines = [
        {
            "recorder_version": RecordedTurns.FORMAT_VERSION,
            "recorded_at": "2026-11-01T00:00:00+00:00",
            "session": "analyst",
            "index": index,
            "kind": "reply",
            "reply": reply,
        }
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    session = RecordedTurns.replay(path).turn_session("analyst", snapshot="snap-1")

    first = await session.turn(system="", messages=[TurnMessage("user", "?")], tools=(), max_tokens=1)
    second = await session.turn(system="", messages=[TurnMessage("user", "?")], tools=(), max_tokens=1)

    assert first.served_model is not None and first.served_model.startswith("recording:")
    assert first.served_model == second.served_model, "one recording, one identity"
