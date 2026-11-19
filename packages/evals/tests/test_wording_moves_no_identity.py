"""A word the package prints moves no identity: no recording key, no replayed number, no fingerprint, no digest.

Identity is over inputs only (the structure design, section 3). A recorded judge call over a report writer's page is
keyed by what the page was rendered from, never by the page's text (``testing/recorded_completions.py`` holds the
key's components as a table), so a wording change on the page replays every call it did, with the same answers.

What is held, each part by perturbing text the package renders, through the front door:

- **the judge's input key** (:meth:`JudgeService.dimension_request`): every string the evidence carries reworded
  moves no ``input_key``, while the user prompt sent does change; and each real change to what is asked (another
  document, snapshot or findings; another criterion; another intent, variation or goal outcome; the pair's other
  order) moves it. So a key that reads rendered text again turns this red;
- **a recorded judge call**: scored through the judge service onto a recording, then replayed over a reworded page,
  it is served the recorded answer and counted as served over a changed prompt; over a changed input it is refused
  naming that input;
- **rung 5 whole** (``examples/5-writer``): replayed as shipped and again with a word changed throughout every page
  and every page's computed facts, and the pair's labels reworded, it exits 0 both times and prints the same
  readings, the reworded replay adding one disclosure that its judge calls were served over pages rendered otherwise
  since they were recorded; every stored score (its dimension, score and reasoning, the replayed responses) is the same, and so is
  every stored fingerprint and digest (each snapshot's bundle fingerprint and report-input digest, the case set's
  fingerprints, the template's) but the judged output's, below.

**One identity moves, by the label rule:** a label pairs with exactly what the labeller read. A judged output's
fingerprint (the output half of a label key, :func:`~threetears.evals.schema.models.fingerprint_judged_output`) is over
the evidence's three strings, which a labeller reads, so a reworded page moves it on every page, and a label does not
carry across a re-render; the evidence's ``inputs``, which no one reads, move it not at all. Recording keys are over
what was asked, label keys over what was read.

The design's other perturbation (a host vocabulary overlay renaming every measure, lever, arm and stratum, over the
four hot-path fixtures) turns on with S7, which separates labels from ids.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest

from packages.evals.tests.ladder_support import offline_copy
from packages.evals.tests.offline_support import assert_no_connects, refuse_connects
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.ops.report_writer import writer_kind, writer_pairwise
from threetears.evals.ops.report_writer.preference import pair_evidence
from threetears.evals.quick.files import main
from threetears.evals.run.judge_service import JudgeContext, JudgeService
from threetears.evals.schema import ClientRequestSettings, StopReason
from threetears.evals.schema.models import (
    GoalStateOutcome,
    JudgedArtifact,
    JudgeEvidence,
    RubricDim,
    fingerprint_judged_output,
)
from threetears.evals.testing import RecordedCompletions, RecordingRefused

#: The criterion the unit parts judge on.
_FIRST_SCREEN = RubricDim(
    name="report_writer.first_screen",
    description="From the headline alone a reader can say what to do.",
    scale="ordinal",
)

#: What a page is rendered from, as the writer kind names it.
_PAGE = {"document": "a" * 64, "snapshot": "report-snapshot:0123", "findings": "f" * 64}

#: The judge's settings in the unit parts.
_SETTINGS = ClientRequestSettings(max_tokens=2048, reasoning_effort="low")

#: The word every reworded page changes, and the line it gains.
_WORD, _REWORDED, _MARKER = "the ", "teh ", "A line the package now prints."


def _evidence(text: str = "Lean to Brisk-S: it clears the gate.", **inputs: Any) -> JudgeEvidence:
    return JudgeEvidence(
        case_material=f"Computed facts, from the snapshot: {text}", artifact=f"# {text}", inputs={**_PAGE, **inputs}
    )


def _context(evidence: JudgeEvidence, **fields: Any) -> JudgeContext:
    return JudgeContext(
        case_id=fields.get("case_id", "tc-1"),
        intent=fields.get("intent", "Write the report (writer)."),
        variation=fields.get("variation", {}),
        goal_outcomes=fields.get("goal_outcomes", []),
        judged_artifact=JudgedArtifact.DOCUMENT,
        judge_evidence=evidence,
    )


@dataclass(frozen=True, kw_only=True)
class _Reply:
    """A judge's completion: the ten members a recording keeps."""

    content: str
    input_tokens: int | None = 900
    output_tokens: int | None = 40
    cost_usd: float | None = 0.0003
    price_source: str | None = "openrouter:reported"
    model: str = "google/gemini-3.8-flash"
    served_model: str | None = "google/gemini-3.8-flash"
    reasoning_tokens: int | None = 0
    temperature: float | None = 0.0
    stop_reason: StopReason = "end_turn"


class _Judge:
    """Answers every call with a 4 and a reason naming how many calls it has answered."""

    def __init__(self) -> None:
        self.users: list[str] = []

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        self.users.append(user)
        reasoning = f"answer {len(self.users)}"
        return _Reply(content=json.dumps({"reasoning": reasoning, "criteria_scores": {_FIRST_SCREEN.name: "4"}}))

    async def aclose(self) -> None:
        return None


def _service(client: Any) -> JudgeService:
    return JudgeService(client_factory=lambda _model, _temperature: client, failure_describer=withhold_failure_detail)


def _reworded(evidence: JudgeEvidence, *, subject: bool = True) -> JudgeEvidence:
    """Every string the evidence carries with a word changed and a line added (and, unless told not to, a subject
    given where it had none); its inputs as they were."""
    update = {
        "case_material": evidence.case_material.replace(_WORD, _REWORDED) + f"\n{_MARKER}",
        "artifact": evidence.artifact.replace(_WORD, _REWORDED) + f"\n{_MARKER}",
    }
    return evidence.model_copy(update={**update, "subject": f"{_MARKER} (subject)"} if subject else update)


# --- the judge's input key ---------------------------------------------------------------------------------------


def test_a_reworded_page_moves_no_judge_input_key_while_the_prompt_it_sends_changes() -> None:
    service = _service(_Judge())
    evidence = _evidence()
    asked = service.dimension_request(_FIRST_SCREEN, _context(evidence))
    reworded = service.dimension_request(_FIRST_SCREEN, _context(_reworded(evidence)))

    assert asked.input_key is not None
    assert reworded.input_key == asked.input_key
    assert reworded.user_prompt != asked.user_prompt and _MARKER in reworded.user_prompt, "not vacuous"
    # Nothing rendered is in the key: no value of it holds any line of the evidence's text.
    flat = json.dumps(asked.input_key)
    for text in (evidence.case_material, evidence.artifact):
        assert all(line not in flat for line in text.splitlines() if line.strip())
    # A pair of reworded pages is keyed as the pair it was; the other order is another pair.
    other = _evidence(document="b" * 64)
    pair = service.dimension_request(_FIRST_SCREEN, _context(pair_evidence(evidence, other)))
    assert (
        service.dimension_request(
            _FIRST_SCREEN, _context(pair_evidence(_reworded(evidence, subject=False), _reworded(other, subject=False)))
        ).input_key
        == pair.input_key
    )
    assert (
        service.dimension_request(_FIRST_SCREEN, _context(pair_evidence(other, evidence))).input_key != pair.input_key
    )


@pytest.mark.parametrize(
    ("change", "context"),
    [
        ("another document", _context(_evidence(document="b" * 64))),
        ("another snapshot", _context(_evidence(snapshot="report-snapshot:4567"))),
        ("other open findings", _context(_evidence(findings="e" * 64))),
        ("another intent", _context(_evidence(), intent="Review the report.")),
        ("another variation", _context(_evidence(), variation={"audience": "operator"})),
        (
            "a goal outcome shown",
            _context(_evidence(), goal_outcomes=[GoalStateOutcome(expression="doc.claims >= 1", passed=True)]),
        ),
    ],
)
def test_a_real_change_to_what_is_asked_moves_the_judge_input_key(change: str, context: JudgeContext) -> None:
    service = _service(_Judge())
    asked = service.dimension_request(_FIRST_SCREEN, _context(_evidence()))
    assert service.dimension_request(_FIRST_SCREEN, context).input_key != asked.input_key, change


def test_a_reworded_criterion_moves_the_judge_input_key() -> None:
    service = _service(_Judge())
    context = _context(_evidence())
    reworded = _FIRST_SCREEN.model_copy(update={"description": "From the headline a reader can act."})
    assert (
        service.dimension_request(reworded, context).input_key
        != service.dimension_request(_FIRST_SCREEN, context).input_key
    )


def test_evidence_that_names_no_inputs_is_keyed_by_its_prompt_as_before() -> None:
    plain = JudgeEvidence(case_material="the facts", artifact="the reply")
    assert _service(_Judge()).dimension_request(_FIRST_SCREEN, _context(plain)).input_key is None
    assert "inputs" not in plain.model_dump(mode="json"), "an evidence naming none is stored as before the field"


# --- a recorded judge call ---------------------------------------------------------------------------------------


async def test_a_recorded_judge_call_replays_over_a_reworded_page_and_is_refused_over_a_changed_input(
    tmp_path: Path,
) -> None:
    path = tmp_path / "judge.jsonl"
    evidence = _evidence()
    async with RecordedCompletions.record(_Judge(), path, settings=_SETTINGS) as recorder:
        recorded = await _service(recorder).score_dimension(_FIRST_SCREEN, _context(evidence))
    assert recorded.score is not None

    replay = RecordedCompletions.replay(path, settings=_SETTINGS)
    replayed = await _service(replay).score_dimension(_FIRST_SCREEN, _context(_reworded(evidence)))
    assert replayed.score is not None
    assert (replayed.score.score, replayed.score.reasoning) == (recorded.score.score, recorded.score.reasoning)
    assert replay.served_over_changed_prompts == 1, "served over a reworded page, and counted"

    moved = RecordedCompletions.replay(path, settings=_SETTINGS)
    with pytest.raises(RecordingRefused) as refused:
        await _service(moved).score_dimension(_FIRST_SCREEN, _context(_evidence(snapshot="report-snapshot:4567")))
    assert "the inputs differ at evidence.snapshot" in str(refused.value)
    assert 'recorded "report-snapshot:0123"' in str(refused.value)


# --- rung 5 whole ------------------------------------------------------------------------------------------------

#: Every judge call rung 5 replays: its 98 scores (7 criteria over 14 pages) and the pairwise pass's 14.
_JUDGE_CALLS: Final = 112

#: The stored fields that are identities: every one must be the same over a reworded page, but the one below.
_IDENTITY_FIELDS: Final = ("fingerprint", "digest")

#: The one stored identity a reworded page moves, on every page: the judged output's fingerprint, a label key's half.
_OVER_TEXT: Final = "output_fingerprint"


@dataclass(frozen=True)
class _Replayed:
    """What one replay of rung 5 printed and stored."""

    exit_code: int
    printed: str
    scores: list[tuple[str, str, str, int | None, str]]
    identities: dict[str, list[str]]
    artifacts: list[str]


def _replay(copy: Path, capsys: pytest.CaptureFixture[str]) -> _Replayed:
    exit_code = main(["writer.py"])
    printed = capsys.readouterr().out.replace(str(copy), "<copy>")
    with sqlite3.connect(f"file:{copy / '.evals' / 'evals.sqlite'}?mode=ro", uri=True) as db:
        documents = [(kind, json.loads(body)) for kind, body in db.execute("select doc_type, body from documents")]
    snapshot_of = {
        body["id"]: body["host_payload"]["writer_case"]["snapshot"]["id"]
        for kind, body in documents
        if kind == "eval_test_case"
    }
    scores = sorted(
        (
            body["kind_payload"]["skill"],
            snapshot_of[body["test_case_id"]],
            score["dim"],
            score["score"],
            score["reasoning"],
        )
        for kind, body in documents
        if kind == "eval_result"
        for score in body["rubric_scores"]
    )
    identities: dict[str, list[str]] = {}

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for name, held in value.items():
                at = f"{path}.{name}"
                if any(word in name for word in _IDENTITY_FIELDS) and not isinstance(held, dict | list):
                    identities.setdefault(at, []).append(str(held))
                walk(held, at)
        elif isinstance(value, list):
            for held in value:
                walk(held, f"{path}[]")

    for kind, body in documents:
        walk(body, kind)
    artifacts = sorted(
        body["judge_evidence"]["artifact"]
        for kind, body in documents
        if kind == "eval_trace" and body.get("judge_evidence")
    )
    return _Replayed(exit_code, printed, scores, {k: sorted(v) for k, v in identities.items()}, artifacts)


def test_rung_five_replays_every_judge_call_with_the_same_answers_over_reworded_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    attempts = refuse_connects(monkeypatch)
    plain = _replay(offline_copy("5-writer", tmp_path / "plain", monkeypatch), capsys)

    page, facts, paired = writer_kind.doc_page_text, writer_kind.computed_facts, writer_pairwise.pair_evidence

    def reworded_page(*args: Any, **kwargs: Any) -> str:
        return page(*args, **kwargs).replace(_WORD, _REWORDED) + f"\n{_MARKER}"

    def reworded_facts(*args: Any, **kwargs: Any) -> str:
        return facts(*args, **kwargs).replace(_WORD, _REWORDED) + f"\n{_MARKER}"

    def relabelled(a: JudgeEvidence, b: JudgeEvidence) -> JudgeEvidence:
        pair = paired(a, b)
        return pair.model_copy(update={"artifact": pair.artifact.replace("=== Report", "=== Page")})

    monkeypatch.setattr(writer_kind, "doc_page_text", reworded_page)
    monkeypatch.setattr(writer_kind, "computed_facts", reworded_facts)
    monkeypatch.setattr(writer_pairwise, "pair_evidence", relabelled)
    moved = _replay(offline_copy("5-writer", tmp_path / "reworded", monkeypatch), capsys)

    assert_no_connects(attempts)
    # Not vacuous: every page the judge read was reworded.
    assert len(moved.artifacts) == len(plain.artifacts) == 14
    assert all(
        _MARKER in artifact and _MARKER not in before for artifact, before in zip(moved.artifacts, plain.artifacts)
    )
    # Every call replayed, with the answers it was recorded with.
    assert plain.exit_code == moved.exit_code == 0, moved.printed
    assert "7 result(s): 7 scored, 0 excluded" in moved.printed and "| lead-first | every pair |" in moved.printed
    # It says so: every judge call was served over a page rendered otherwise since it was recorded, and the
    # replay as shipped served none.
    rerendered = f"judge: {_JUDGE_CALLS} call(s) replayed over a page rendered otherwise since it was recorded"
    assert "replayed over a page rendered otherwise" not in plain.printed
    assert rerendered in moved.printed, moved.printed
    assert [line for line in moved.printed.splitlines() if rerendered not in line] == plain.printed.splitlines()
    assert len(plain.scores) == 98 and moved.scores == plain.scores
    # Every stored fingerprint and digest, but the judged output's, is where it was.
    kept = {path: held for path, held in plain.identities.items() if not path.endswith(_OVER_TEXT)}
    assert {path: held for path, held in moved.identities.items() if not path.endswith(_OVER_TEXT)} == kept
    assert any("bundle_fingerprint" in path for path in kept) and any("report_input_digest" in path for path in kept)
    # The judged output's fingerprint moves on every reworded page: a label of the page as it read does not carry over.
    over_text = {path for path in plain.identities if path.endswith(_OVER_TEXT)}
    assert over_text and over_text == {path for path in moved.identities if path.endswith(_OVER_TEXT)}
    for path in over_text:
        assert len(moved.identities[path]) == len(plain.identities[path]), path
        assert not set(moved.identities[path]) & set(plain.identities[path]), path


def test_a_reworded_page_moves_the_judged_output_fingerprint_and_its_inputs_alone_do_not() -> None:
    """A label pairs with exactly what the labeller read: a re-rendered page is another judged output, while evidence
    read identically under other inputs is the same one."""
    evidence = _evidence()
    assert fingerprint_judged_output(_reworded(evidence)) != fingerprint_judged_output(evidence)
    assert fingerprint_judged_output(_reworded(evidence, subject=False)) != fingerprint_judged_output(evidence)
    other_inputs = _evidence(document="b" * 64, snapshot="report-snapshot:4567", findings="e" * 64)
    assert fingerprint_judged_output(other_inputs) == fingerprint_judged_output(evidence)
    assert fingerprint_judged_output(evidence.model_copy(update={"inputs": None})) == fingerprint_judged_output(
        evidence
    )
