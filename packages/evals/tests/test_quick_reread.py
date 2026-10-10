"""A stored comparison re-read in a new process is decided on what it was launched under, never on who reads it.

``compare(margins=, ranges=, guardrails=)`` declares how each scorer is read on the host it builds, and a later
process reads the stored campaign through a host of its own (``callable_host(..., store=)``). Before launches
recorded their declarations, that reader's host decided the verdicts: one that declared no margin read no
``equivalent``, one that declared another margin read against it, and one that declared no guardrail read a
guardrail as a capability contrast. Each run now freezes its launching host's declarations
(``EvalRun.declared_measures``), the analysis reads every measure on them, and a reader that declares otherwise is
named in the report.

Mutations that turn this file red: a launch recording no declarations; the bundle reading the reader's descriptor;
the difference going unnamed; runs launched under different declarations read on either one; runs stored before
the field read on anything but the reader's declarations.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle, get_campaign
from threetears.evals.contracts import EvalRun, MeasureDeclaration
from threetears.evals.contracts.host import EvalHost
from threetears.evals.quick import Comparison, Guardrail, callable_host, compare
from threetears.evals.run import list_runs
from threetears.evals.storage import SqliteDocumentStore

#: Enough agreeing cases for a 0.1 margin to be shown on a pass rate, and a guardrail held within it.
CASES = [{"n": index} for index in range(48)]

#: The verdicts as a reader compares them: kind, reading, arm, outcome and the margin it was read against.
Verdicts = list[tuple[str, str, str, str, float | None]]


def correct(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer is right."""
    return answer == "right"


def no_leak(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer keeps the secret."""
    return "SECRET" not in answer


async def same(case: Mapping[str, Any]) -> str:
    return "right"


#: A reader in a fresh process: its own host over the reopened file, declaring what it is handed.
READER = """
import json
from collections.abc import Mapping
from typing import Any
from threetears.evals.ops import report_read
from threetears.evals.quick import Guardrail, callable_host
from threetears.evals.storage import SqliteDocumentStore

{scorers}

host = callable_host([correct, no_leak], arms=True, store=SqliteDocumentStore({path!r}), {declares})
report = json.loads(report_read(host, {campaign!r}, "kept", format="json").body)
print(json.dumps({{
    "verdicts": [[v["kind"], v["name"], v["arm"], v["outcome"], v["margin"]] for v in report["verdicts"]],
    "disclosures": [
        block["text"] for block in report["blocks"] if block["kind"] == "disclosure"
    ],
}}))
"""


async def _compare(path: Path) -> Comparison:
    with SqliteDocumentStore(path) as store:
        return await compare(
            CASES,
            {"current": same, "cheaper": same},
            [correct, no_leak],
            control="current",
            scope_id="kept",
            k=1,
            store=store,
            margins={"correct": 0.1},
            guardrails={"no_leak": Guardrail(margin=0.1, direction="higher_is_better")},
        )


def _read(path: Path, campaign_id: str, declares: str) -> tuple[Verdicts, list[str]]:
    """The verdicts and disclosures of the campaign's report, as a new process declaring ``declares`` reads it."""
    scorers = "\n\n".join(inspect.getsource(scorer) for scorer in (correct, no_leak))
    script = textwrap.dedent(READER).format(scorers=scorers, path=str(path), campaign=campaign_id, declares=declares)
    ran = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False, timeout=300)
    assert ran.returncode == 0, ran.stderr
    read = json.loads(ran.stdout)
    return sorted((kind, name, arm, outcome, margin) for kind, name, arm, outcome, margin in read["verdicts"]), read[
        "disclosures"
    ]


def _launched(comparison: Comparison) -> Verdicts:
    return sorted((v.kind, v.name, v.arm, v.outcome, v.margin) for v in comparison.report.verdicts)


def _rewrite_runs(host: EvalHost, rewrite: Any) -> None:
    """Store each of the scope's runs, read whole, as ``rewrite`` returns it."""
    for listed in list_runs(host, "kept"):
        run = host.storage.load_eval_run(listed.id, "kept")
        assert isinstance(run, EvalRun)
        host.storage.save_eval_run(rewrite(run))


def _bundle(host: EvalHost, comparison: Comparison) -> AnalysisContextBundle:
    return assemble_context_bundle(
        get_campaign(host.storage, comparison.campaign_id, "kept"), storage=host.storage, profile=host.profile
    )


async def test_a_reader_declaring_otherwise_reads_the_launched_verdicts_and_is_named(tmp_path: Path) -> None:
    path = tmp_path / "evals.sqlite"
    comparison = await _compare(path)
    launched = _launched(comparison)
    assert ("contrast", "correct", "candidate=cheaper", "equivalent", 0.1) in launched
    assert ("guardrail", "no_leak", "candidate=cheaper", "held", 0.1) in launched

    # A reader declaring nothing, one declaring another margin, and one declaring the guardrail a quality score.
    for declares, named in (
        ("", "correct"),
        ('margins={"correct": 0.3}', "correct"),
        ('margins={"no_leak": 0.2}', "no_leak"),
    ):
        verdicts, disclosures = _read(path, comparison.campaign_id, declares)
        assert verdicts == launched, declares
        assert [text for text in disclosures if text.startswith(f"{named} is read as its runs were launched")], declares
    _, disclosures = _read(path, comparison.campaign_id, "")
    (sentence,) = [text for text in disclosures if text.startswith("correct is read as")]
    assert "margin 0.1" in sentence and "no margin" in sentence


async def test_a_reader_declaring_alike_is_named_nowhere(tmp_path: Path) -> None:
    path = tmp_path / "evals.sqlite"
    comparison = await _compare(path)
    declares = 'margins={"correct": 0.1}, guardrails={"no_leak": Guardrail(margin=0.1, direction="higher_is_better")}'
    verdicts, disclosures = _read(path, comparison.campaign_id, declares)
    assert verdicts == _launched(comparison)
    assert not [text for text in disclosures if "launched" in text]


async def test_runs_launched_under_different_declarations_read_no_margin(tmp_path: Path) -> None:
    path = tmp_path / "evals.sqlite"
    comparison = await _compare(path)
    cheaper = comparison.arms["cheaper"].run_id

    def moved(run: EvalRun) -> EvalRun:
        if run.id != cheaper:
            return run
        # The cheaper arm's run as if launched when the host declared another margin on correct.
        declared = run.declared_measures["correct"].model_copy(update={"materiality_threshold": 0.2})
        return run.model_copy(update={"declared_measures": {**run.declared_measures, "correct": declared}})

    with SqliteDocumentStore(path) as store:
        host = callable_host([correct, no_leak], arms=True, margins={"correct": 0.1}, store=store)
        _rewrite_runs(host, moved)
        bundle = _bundle(host, comparison)
    assert [text for text in bundle.launch_declarations if "different declarations of correct" in text]
    assert bundle.measure_catalog["correct"].materiality_threshold is None


async def test_runs_stored_before_declarations_were_recorded_read_on_the_reader(tmp_path: Path) -> None:
    path = tmp_path / "evals.sqlite"
    comparison = await _compare(path)

    def unrecorded(run: EvalRun) -> EvalRun:
        assert isinstance(run.declared_measures["correct"], MeasureDeclaration)
        return run.model_copy(update={"declared_measures": {}})

    with SqliteDocumentStore(path) as store:
        host = callable_host([correct, no_leak], arms=True, margins={"correct": 0.3}, store=store)
        _rewrite_runs(host, unrecorded)
        bundle = _bundle(host, comparison)
    assert bundle.launch_declarations == []
    assert bundle.measure_catalog["correct"].materiality_threshold == 0.3
