"""The reset's P2 acceptance: does the writer gate read a worse writer as writing worse reports?

Two arms of one writer over the held-out writer cases: the shipped skill (the control) and one candidate variant,
chosen by ``--variant``, each a function of the shipped skill:

- ``hollow`` (the default): ``principles.md`` and ``statistics.md`` each left as one stub line;
- ``minimal``: ``SKILL.md`` and ``editor.md`` kept, every other module, the exemplars' index and every note left as a
  stub (the index keeps one bare row per note, so the loader still finds each note and a case may still withhold one);
- ``no-editor``: the shipped skill, its arm given 0 revisions, so the editor reviews each report once and never sends
  it back (the loop has no lever below one review);
- ``one-paragraph``: ``SKILL.md`` replaced by a short instruction to write one paragraph with no tables.

The loader requires every module, so blanked files stay as stubs; ``README.md`` is in no session's text and stays as
shipped. ``editor.md`` is byte-identical in every variant, so the in-loop editor is the same instrument for both arms
and the code measures stay comparable. Then:

1. the writer eval (:class:`~threetears.evals.harness.WriterEval`) runs the arms, each launched naming its part of
   the cap, its pages scored by the fixed scoring judge; then its pairwise pass shows that judge each case's two pages
   in both orders, under the pass's part; then its promotion read reads the pass as its primary reading, which the
   criterion reads again whole, off the pass's stored job;
2. the pass criterion is decided by code: the read answers ``keep``, and its primary reading
   (``report_writer.preferred``) reads ``regressed``, separated, its interval below 0, with the order check standing
   and bounces not separated. Anything else fails; a primary that does not separate states the held-out cases that
   would settle it, and the run stops there: nobody re-rolls.

Commands::

    uv run python -m packages.evals.tests.fixtures.report_plans.writer_acceptance --dry
    uv run python -m packages.evals.tests.fixtures.report_plans.writer_acceptance --cap-usd 12 --out <dir> --cases <dir>
    uv run python -m packages.evals.tests.fixtures.report_plans.writer_acceptance --dry --variant minimal

``--dry`` runs the whole path on scripted models: each case's analyst submits the accepted prototype 5 document for
its snapshot (or nothing, for a snapshot that has none) and the editor clears it; the scoring judge scores a digest
of its prompt. Nothing is sent and the verdict carries no signal. The live run is the owner's to approve. It reads
``ANTHROPIC_API_KEY`` (the writer) and ``OPENROUTER_API_KEY`` (a ``vendor/model`` judge) from the environment, and
records every writer turn and judge call under ``--out``. It refuses, before anything is sent, a case directory with
fewer than 10 cases on the held-out side (design §4), so the default seven run dry only.

**The cap.** The writer eval's rule splits it: a fifth goes to the pairwise pass, and the arms share the rest equally,
each arm's part holding its reports and their absolute judging. At ``--cap-usd 12`` over eleven cases the pass gets
$2.40 and each arm $4.80: its reports' budgets come to 11 × $0.35 = $3.85, leaving $0.95 for 77 absolute judge calls
(about $0.77 on flash). The expected spend is about $6 in all; the cap bounds the worst case.

**The cases.** ``--cases DIR`` reads a directory of writer case files (``*.json.gz``). Without it, the cases are the
seven fixture snapshots prototype 5 wrote over (Kairo round 2 and six misfits, in four split groups), written as
case files into a scratch directory. ``--held-out-fraction`` 0.99 puts every one of their groups on the held-out
side, since the promotion read reads held-out cases only. A misfit is its reference reshaped, so cases correlate
within a group; the test reads cases, not groups.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, get_args

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.fixtures.report_plans.writer_smoke import JUDGE as SMOKE_JUDGE
from packages.evals.tests.fixtures.report_plans.writer_smoke import WRITER as SMOKE_WRITER
from packages.evals.tests.report_writer_support import reply, review, submitted
from packages.evals.tests.writer_case_support import SEED
from packages.evals.tests.writer_eval_support import lay_recording
from threetears.evals.analysis import EvalSummary
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.analysis.report.blocks import Verdict
from threetears.evals.harness import Writer, WriterCaseFile, WriterEval, WriterResult, read_writer_cases
from threetears.evals.harness.writer_host import WriterJudge
from threetears.evals.harness.writer_turns import writer_turns
from threetears.evals.ops import OpsHost, job_poll
from threetears.evals.ops.report_writer.loop import MOST_REVISIONS
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.ops.report_writer.writer_bank import writer_group_side
from threetears.evals.ops.report_writer.writer_contract import BOUNCES_MEASURE, EffortWord
from threetears.evals.ops.report_writer.writer_pairwise import WriterPairwiseReport
from threetears.evals.ops.report_writer.writer_promotion import WriterPromotion, writer_promotion
from threetears.evals.schema.turns import TurnReply, TurnText

#: The writer both arms run on: the smoke's, the owner-ruled build-and-test ceiling, on Anthropic's own API.
WRITER: Final = SMOKE_WRITER

#: The provider the writer is built on.
WRITER_PROVIDER: Final = "anthropic"

#: The fixed scoring judge: the smoke's, of another family than the writer, through OpenRouter.
JUDGE: Final = SMOKE_JUDGE

#: The writer's thinking effort, the same for both arms.
EFFORT: Final = "low"

#: Each report's dollar cap. Haiku reports cost about $0.22 in the smoke; this leaves room for extra rounds.
BUDGET_USD: Final = 0.35

#: Each report's turn cap, which bounds a cell's wall clock and so how many cases an arm's job admits.
TURNS: Final = 30

#: The ceiling for one writer request, in seconds.
REQUEST_S: Final = 90.0

#: The share of groups placed on the held-out side: every group of the default cases.
HELD_OUT_FRACTION: Final = 0.99

#: The fewest held-out cases a live run reads (reset P2 §4).
MIN_LIVE_CASES: Final = 10

#: The arms, control first.
SHIPPED: Final = "shipped"
HOLLOW: Final = "hollow"

#: The files the hollow variant leaves as one stub line each.
HOLLOWED: Final = ("principles.md", "statistics.md")

#: The stub line each hollowed file holds.
STUB: Final = "This module is intentionally empty.\n"

#: The question the acceptance asks of the hollow variant.
QUESTION: Final = (
    "Does the gate read the hollow skill (principles.md and statistics.md left blank) as writing worse reports than "
    "the shipped skill?"
)

#: The other variants' arm names.
MINIMAL: Final = "minimal"
NO_EDITOR: Final = "no-editor"
ONE_PARAGRAPH: Final = "one-paragraph"

#: The files the minimal variant keeps as shipped: the role file, the editor's text, and the README no session reads.
MINIMAL_KEPT: Final = ("SKILL.md", "editor.md", "README.md")

#: The exemplars' index, which the minimal variant leaves as a stub with one bare row per note.
INDEX: Final = "exemplars/INDEX.md"

#: The one-paragraph variant's whole ``SKILL.md``: a deliberately bad writer.
ONE_PARAGRAPH_SKILL: Final = (
    "# Report writer\n\n"
    "Write the whole report as one paragraph of plain prose. Use no tables, no charts, no headings and no lists. "
    "Submit it with the tools you are given.\n"
)

#: The eval's name, its scope's and its campaign's.
NAME: Final = "writer acceptance"

#: Where the accepted prototype 5 runs live, one directory per snapshot.
PROTOTYPE5: Final = Path(__file__).parents[1] / "report_docs" / "prototype5"

#: Each prototype 5 fixture's stratum: the reference, or a misfit reshaped from one.
_STRATA: Final = {KAIRO: "reference"}

#: How many replies a dry analyst with no document gives before the loop gives up on it.
_SILENT_TURNS: Final = 4

#: The model name of the dry judge's replies.
_DRY_SERVED: Final = "dry/scripted-judge"


def hollow_variant(shipped: AnalystSkill | None = None) -> AnalystSkill:
    """The shipped skill with :data:`HOLLOWED` each left as :data:`STUB`; every other file byte-identical.

    :param shipped: The skill to hollow; ``None`` for the one this build ships.
    :return: The variant, checked whole.
    """
    skill = load_skill() if shipped is None else shipped
    for name in HOLLOWED:
        skill = skill.with_file(name, STUB)
    return skill


def minimal_variant(shipped: AnalystSkill | None = None) -> AnalystSkill:
    """The shipped skill with every file but :data:`MINIMAL_KEPT` left as :data:`STUB`, and the exemplars' index as
    the stub and one bare row per note, since the loader refuses a note the index does not row and a case withholds
    notes by name.

    :param shipped: The skill to cut down; ``None`` for the one this build ships.
    :return: The variant, checked whole.
    """
    skill = load_skill() if shipped is None else shipped
    notes = sorted(name for name in skill.files if name.startswith("exemplars/") and name != INDEX)
    for name in sorted(skill.files):
        if name not in MINIMAL_KEPT and name != INDEX:
            skill = skill.with_file(name, STUB)
    rows = "".join(f"| `{name.removeprefix('exemplars/')}` |\n" for name in notes)
    return skill.with_file(INDEX, f"{STUB}\n| Note |\n|---|\n{rows}")


def one_paragraph_variant(shipped: AnalystSkill | None = None) -> AnalystSkill:
    """The shipped skill with ``SKILL.md`` replaced by :data:`ONE_PARAGRAPH_SKILL`; every other file byte-identical.

    :param shipped: The skill to change; ``None`` for the one this build ships.
    :return: The variant, checked whole.
    """
    skill = load_skill() if shipped is None else shipped
    return skill.with_file("SKILL.md", ONE_PARAGRAPH_SKILL)


def _shipped(shipped: AnalystSkill | None = None) -> AnalystSkill:
    """The shipped skill unchanged: the no-editor variant changes the arm's revisions, not its skill."""
    return load_skill() if shipped is None else shipped


def files_differing(a: AnalystSkill, b: AnalystSkill) -> list[str]:
    """The files whose text differs between two skills, or that one holds and the other does not, by name."""
    return sorted(name for name in a.files.keys() | b.files.keys() if a.files.get(name) != b.files.get(name))


@dataclass(frozen=True)
class Variant:
    """One candidate the acceptance runs against the shipped skill.

    :ivar name: Its arm's name, the one ``--variant`` takes.
    :ivar noun: How the verdict names it.
    :ivar question: The question the acceptance asks of it, which its campaign is filed under.
    :ivar build: The variant's skill, made from the shipped one.
    :ivar changes: The files it is allowed to change, given the shipped skill.
    :ivar revisions: The revisions its arm is given; the control's are :data:`MOST_REVISIONS`.
    """

    name: str
    noun: str
    question: str
    build: Callable[[AnalystSkill | None], AnalystSkill]
    changes: Callable[[AnalystSkill], list[str]]
    revisions: int = MOST_REVISIONS


#: Every variant, by the name ``--variant`` takes; the hollow variant first, the default.
VARIANTS: Final[Mapping[str, Variant]] = {
    variant.name: variant
    for variant in (
        Variant(HOLLOW, "the hollow skill", QUESTION, hollow_variant, lambda _skill: sorted(HOLLOWED)),
        Variant(
            MINIMAL,
            "the minimal skill",
            "Does the gate read the minimal skill (SKILL.md and editor.md alone, every other module, the index and "
            "every note left blank) as writing worse reports than the shipped skill?",
            minimal_variant,
            lambda shipped: sorted(name for name in shipped.files if name not in MINIMAL_KEPT),
        ),
        Variant(
            NO_EDITOR,
            "the shipped skill with no editor revisions",
            "Does the gate read the shipped skill with no editor revisions (one review, never sent back) as writing "
            f"worse reports than the shipped skill with {MOST_REVISIONS}?",
            _shipped,
            lambda _skill: [],
            revisions=0,
        ),
        Variant(
            ONE_PARAGRAPH,
            "the one-paragraph skill",
            "Does the gate read the one-paragraph skill (SKILL.md asking for one paragraph and no tables) as writing "
            "worse reports than the shipped skill?",
            one_paragraph_variant,
            lambda _skill: ["SKILL.md"],
        ),
    )
}


def refuse_unlike_editors(shipped: AnalystSkill, candidate: AnalystSkill, variant: Variant = VARIANTS[HOLLOW]) -> None:
    """Refuse a variant whose editor text is not the control's, or that differs anywhere but the files it changes."""
    if candidate.editor().digest != shipped.editor().digest:
        raise SystemExit(
            f"writer_acceptance.py: the {variant.name} variant's editor.md differs, so the editor is not one instrument"
        )
    if (differing := files_differing(shipped, candidate)) != (allowed := variant.changes(shipped)):
        raise SystemExit(f"writer_acceptance.py: the {variant.name} variant differs in {differing}, not {allowed}")


# --- the cases -------------------------------------------------------------------------------------------------


def _prototype5() -> dict[str, Path]:
    """Each fixture prototype 5 wrote over, by fixture name: its run's directory."""
    runs: dict[str, Path] = {}
    for made in sorted(PROTOTYPE5.glob("*/replay.json")):
        name = json.loads(made.read_text(encoding="utf-8"))["snapshot_of"]
        runs[KAIRO if name == "kairo" else name] = made.parent
    return runs


def default_cases(directory: Path) -> Path:
    """Write the default cases into ``directory``: every fixture prototype 5 wrote over, with its seed group and
    withheld note.

    :param directory: Where the case files go.
    :return: The directory.
    """
    snapshots = fixture_snapshots("writer-acceptance-cases")
    for name in _prototype5():
        group, withhold = SEED[name]
        case = WriterCaseFile(
            name=name.replace(" ", "-"),
            group=group,
            stratum=_STRATA.get(name, "misfit"),
            withhold=list(withhold),
            snapshot=snapshots[name],
        )
        case.write(directory / f"{case.name}.json.gz")
    return directory


def accepted_documents() -> dict[str, dict[str, Any]]:
    """The document prototype 5's accepted run submitted, by the id of the snapshot it was written over."""
    snapshots = fixture_snapshots("writer-acceptance-cases")
    return {
        snapshots[name].id: json.loads((run / "doc.json").read_text(encoding="utf-8"))
        for name, run in _prototype5().items()
    }


# --- the dry models ----------------------------------------------------------------------------------------------


def dry_sessions(
    document: Mapping[str, Any] | None, *, model: str, editor_model: str | None = None
) -> dict[str, list[TurnReply]]:
    """One dry report's sessions: the analyst submits ``document`` and the editor (on ``editor_model``, else
    ``model``) clears it, or, with no document, the analyst writes none and the loop gives up on it."""
    if document is None:
        return {"analyst": [reply(TurnText("I have no report yet."), model=model) for _ in range(_SILENT_TURNS)]}
    return {
        "analyst": [reply(submitted(document), model=model)],
        "editor-1": [reply(review(), model=editor_model or model)],
    }


def lay_dry_recordings(
    recordings: Path, cases: Sequence[WriterCaseFile], arms: Mapping[str, Writer], *, provider: str
) -> int:
    """Lay one scripted recording per arm and case where the writer eval's router reads it (the editor's sessions in a
    file of their own for an arm naming an editor's model), so a dry run replays them all and records nothing; how
    many of the cases have a document.

    :param recordings: The directory every arm's recordings sit under.
    :param cases: The cases.
    :param arms: The arms, by name.
    :param provider: The writer's provider.
    """
    documents = accepted_documents()
    named = {case.snapshot.id: case.name for case in cases}
    for arm, writer in arms.items():
        routed = writer_turns(
            recordings,
            arm=arm,
            skill=writer.analyst_skill,
            cases=named,
            revisions=writer.revisions,
            editor_model=writer.editor_model,
            provider=provider,
        )
        effort = next((word for word in get_args(EffortWord) if word == writer.effort), None)
        for snapshot in named:
            sessions = dry_sessions(documents.get(snapshot), model=writer.model_id, editor_model=writer.editor_model)
            if writer.editor_model is None:
                lay_recording(routed.path(writer.model_id, effort, snapshot), sessions)
                continue
            analyst = {name: replies for name, replies in sessions.items() if not name.startswith("editor")}
            editor = {name: replies for name, replies in sessions.items() if name.startswith("editor")}
            lay_recording(routed.path(writer.model_id, effort, snapshot), analyst)
            lay_recording(routed.path(writer.model_id, effort, snapshot, role="editor"), editor)
    return sum(1 for snapshot in named if snapshot in documents)


#: Where the built-in prompt names the one key its reply maps.
_KEY: Final = re.compile(r'single key "([^"]+)"')


@dataclass(frozen=True)
class _DryReply:
    """One dry judge reply, in the attribute names the engine reads."""

    content: str
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    cost_usd: float | None = 0.0
    price_source: str | None = "dry"
    model: str = _DRY_SERVED
    served_model: str | None = _DRY_SERVED
    reasoning_tokens: int | None = None
    stop_reason: str = "end_turn"
    temperature: float | None = None


# parity-with: threetears.evals.harness.writer_host.WriterJudge
class FakeDryJudge:
    """The dry scoring judge: its score is a digest of the prompt, so the path runs and measures no signal. It names
    the live judge's model and route, so the family checks run as they do live, and prices every call at nothing."""

    base_url = None

    def __init__(self, model_name: str = JUDGE, *, provider: str = "openrouter") -> None:
        self.model_name = model_name
        self.provider = provider
        self.calls: list[str] = []

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """Nothing: a dry judge sends nothing."""
        return 0.0

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _DryReply:
        match = _KEY.search(system)
        if match is None:
            raise ValueError("the built-in judge prompt names no score key")
        self.calls.append(match[1])
        score = 1 + int(hashlib.sha256(user.encode()).hexdigest(), 16) % 5
        answer = {"reasoning": "dry: a digest of the prompt, not a reading", "criteria_scores": {match[1]: score}}
        return _DryReply(json.dumps(answer), model=self.model_name, served_model=self.model_name)

    async def aclose(self) -> None:
        """Nothing to release."""


# --- the criterion ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Requirement:
    """One requirement of the pass criterion: what it asks, what the read shows, and whether it holds."""

    asks: str
    reads: str
    holds: bool


@dataclass(frozen=True)
class Criterion:
    """The pass criterion, decided by code on the promotion read."""

    requirements: tuple[Requirement, ...]
    cases_to_settle: int | None
    noun: str = VARIANTS[HOLLOW].noun

    @property
    def passed(self) -> bool:
        """Whether every requirement holds."""
        return all(requirement.holds for requirement in self.requirements)

    def verdict(self) -> str:
        """The criterion's one line: PASS, or FAIL naming what did not hold."""
        if self.passed:
            return f"PASS: the gate reads {self.noun} as writing worse reports, on the primary reading."
        failed = "; ".join(requirement.asks for requirement in self.requirements if not requirement.holds)
        settle = (
            ""
            if self.cases_to_settle is None
            else f" About {self.cases_to_settle} held-out cases like these would settle the primary; this run stops here."
        )
        return f"FAIL: not held: {failed}.{settle}"


def _interval(verdict: Verdict | None) -> str:
    if verdict is None or verdict.interval is None:
        return "none"
    low, high = verdict.interval
    return f"{low:.3f} to {high:.3f} at {verdict.interval_level}"


#: A verdict's reason, as the criterion's table reads it.
_REASON_WORDS: Final = {
    "separated": "separated",
    "no_margin": "not separated",
    "too_few_cases": "too few cases",
    "untestable": "untestable",
}


def pass_criterion(
    read: WriterPromotion, runs: Mapping[str, EvalSummary], *, variant: Variant = VARIANTS[HOLLOW]
) -> Criterion:
    """Decide the pass criterion on a promotion read (reset P2 §4).

    Both arms' runs must have completed with a result for every case; the read must answer ``keep``; its primary
    reading must read ``regressed``, separated, with its interval below 0; the judge's order check must not void it;
    and bounces must not separate, since a variant that changes bounces broke the format rather than the writing.

    :param read: The promotion read of the candidate arm, made with the pass as its ``pairwise_job``.
    :param runs: Each arm's run summary, by arm.
    :param variant: The candidate the verdict names.
    :return: The criterion, each requirement decided.
    """
    primary = read.primary
    bounces = next((verdict for verdict in read.verdicts if verdict.name == BOUNCES_MEASURE), None)
    reliability = read.reliability
    consistency = None if reliability is None else reliability.order_consistency
    consistency_reads = (
        "not measured"
        if reliability is None or consistency is None
        else f"{consistency:.2f} ({reliability.order_consistent} of {reliability.both_orders} cases)"
    )
    ran = "; ".join(f"{arm} {run.status}, {run.n_results} of {run.n_cases}" for arm, run in runs.items())
    requirements = (
        Requirement(
            "both arms ran every case",
            ran,
            all(run.status == "completed" and run.n_results == run.n_cases * run.k_runs for run in runs.values()),
        ),
        Requirement("the read answers keep", read.decision, read.decision == "keep"),
        Requirement(
            "the primary reading reads regressed",
            "not read" if primary is None else primary.outcome.replace("_", " "),
            primary is not None and primary.outcome == "regressed",
        ),
        Requirement(
            "it separated, its interval below 0",
            "not read"
            if primary is None
            else f"{_REASON_WORDS.get(primary.reason, primary.reason.replace('_', ' '))}, {_interval(primary)}",
            primary is not None
            and primary.reason == "separated"
            and primary.interval is not None
            and primary.interval[1] < 0,
        ),
        Requirement(
            "the judge's order consistency stands at 0.7 or more",
            consistency_reads,
            read.void is None and consistency is not None,
        ),
        Requirement(
            "bounces do not separate",
            "not read" if bounces is None else bounces.outcome.replace("_", " "),
            bounces is None or bounces.outcome not in ("improved", "regressed"),
        ),
    )
    return Criterion(requirements, read.cases_to_settle, variant.noun)


# --- the run -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Acceptance:
    """What one acceptance run found: the eval's result, the pass, the read and the criterion on it."""

    dry: bool
    shipped: AnalystSkill
    candidate: AnalystSkill
    result: WriterResult
    with_documents: int | None
    report: WriterPairwiseReport
    read: WriterPromotion
    criterion: Criterion
    cap_usd: float
    editor_model: str | None = None
    variant: Variant = VARIANTS[HOLLOW]

    def readout(self) -> Iterator[str]:
        """The printed readout: the question, the arms, the caps and spend, the read's tables, then the verdict."""
        result = self.result
        yield f"P2 acceptance{' (dry: scripted models, nothing sent, no signal)' if self.dry else ''}"
        yield ""
        yield self.variant.question
        yield ""
        yield "| Arm | Role | Skill | Files differing from shipped | Editor digest | Cases | Run |"
        yield "|---|---|---|---|---|---|---|"
        for arm, skill in ((SHIPPED, self.shipped), (self.variant.name, self.candidate)):
            summary = result.arms[arm]
            yield (
                f"| {arm} | {'control' if arm == result.control else 'candidate'} | {skill.digest[:12]} | "
                f"{', '.join(files_differing(self.shipped, skill)) or 'none'} | {skill.editor().digest[:12]} | "
                f"{summary.n_cases} | {summary.status} |"
            )
        yield ""
        writer = next(iter(result.arms.values())).candidate_model
        editor = "" if self.editor_model is None else f", its editor on {self.editor_model}"
        yield f"Both arms write on {writer}{editor}; the scoring judge is {result.judge}."
        if self.variant.revisions != MOST_REVISIONS:
            yield (
                f"The {self.variant.name} arm is given {self.variant.revisions} revisions and the control "
                f"{MOST_REVISIONS}, so its editor reviews each report once and never sends it back."
            )
        if self.with_documents is not None:
            cases = next(iter(result.arms.values())).n_cases
            blank = cases - self.with_documents
            yield (
                f"Dry: {self.with_documents} of {cases} cases have a scripted document"
                + (f"; the other {blank} write no page, so their pairs tie." if blank else ".")
            )
        yield ""
        yield "| Part | Cap | Spent |"
        yield "|---|---|---|"
        for arm, summary in result.arms.items():
            spent = "unpriced" if summary.total_cost_usd is None else f"${summary.total_cost_usd:.2f}"
            yield f"| {arm} | {_cap(summary.max_cost_usd)} | {spent} |"
        pass_cost = "unpriced" if self.report.cost_usd is None else f"${self.report.cost_usd:.2f}"
        yield f"| pairwise pass ({self.report.calls_made} calls) | {_cap(self.report.cap_usd)} | {pass_cost} |"
        if self.dry:
            yield ""
            yield "In a dry run the spend is the scripted replies' own prices; nothing was paid."
        yield ""
        yield self.read.readout
        yield ""
        yield self.read.why
        yield ""
        yield "| Pass criterion | Reads | Holds |"
        yield "|---|---|---|"
        for requirement in self.criterion.requirements:
            yield f"| {requirement.asks} | {requirement.reads} | {'yes' if requirement.holds else 'no'} |"
        yield ""
        yield self.criterion.verdict()
        yield from result.disclosures

    def summary(self) -> dict[str, Any]:
        """What the run found, as JSON: the read, the pass's report, the criterion and the readout."""
        return {
            "dry": self.dry,
            "variant": self.variant.name,
            "question": self.variant.question,
            "campaign": self.result.campaign_id,
            "scope": self.result.scope_id,
            "skills": {SHIPPED: self.shipped.digest, self.variant.name: self.candidate.digest},
            "revisions": {SHIPPED: MOST_REVISIONS, self.variant.name: self.variant.revisions},
            "pass_job": self.read.pairwise_job,
            "pairwise": self.report.model_dump(mode="json"),
            "promotion": self.read.model_dump(mode="json"),
            "criterion": {
                "passed": self.criterion.passed,
                "requirements": [vars(requirement) for requirement in self.criterion.requirements],
                "cases_to_settle": self.criterion.cases_to_settle,
            },
            "readout": "\n".join(self.readout()),
        }


def _cap(usd: float | None) -> str:
    """A part of the cap as the readout's table prints it; none when no cap was enforced."""
    return "none" if usd is None else f"${usd:.2f}"


#: Each provider's key, by the variable it is read from.
_KEYS: Final = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def judge_provider(model: str) -> str:
    """Where the judge is sent: OpenRouter for a ``vendor/model`` id, Anthropic's own API for a bare one."""
    return "openrouter" if "/" in model else "anthropic"


def refuse_missing_keys(judge: str) -> None:
    """Refuse a live run, by variable, when the writer's key or the judge's is not in the environment."""
    wanted = {_KEYS[judge_provider(judge)], _KEYS[WRITER_PROVIDER]}
    if missing := sorted(name for name in wanted if not os.environ.get(name)):
        raise SystemExit(f"writer_acceptance.py: a live run needs {', '.join(missing)}; --dry needs none")


def held_out_cases(cases_dir: Path, held_out_fraction: float) -> int:
    """How many of a directory's cases a fresh store puts on the held-out side, each by its split group's point."""
    return sum(
        1
        for case in read_writer_cases(cases_dir)
        if writer_group_side(case.group or case.name, held_out_fraction) == "held_out"
    )


def refuse_too_few_cases(cases_dir: Path | None, held_out_fraction: float) -> None:
    """Refuse a live run over fewer than :data:`MIN_LIVE_CASES` held-out cases, before anything is sent."""
    if cases_dir is None:
        raise SystemExit(
            f"writer_acceptance.py: a live run names --cases, at least {MIN_LIVE_CASES} held-out cases; "
            "the default seven run dry only"
        )
    if (held := held_out_cases(cases_dir, held_out_fraction)) < MIN_LIVE_CASES:
        raise SystemExit(
            f"writer_acceptance.py: {cases_dir} puts {held} case(s) on the held-out side at {held_out_fraction}, "
            f"and a live run reads at least {MIN_LIVE_CASES}"
        )


#: The live judge's output cap, the one rung 5 records with.
JUDGE_MAX_TOKENS = 2048


def _live_judge(judge: str, out: Path) -> WriterJudge:
    """The live scoring judge, its calls recorded under ``out``; refused by name when its key is missing."""
    refuse_missing_keys(judge)
    provider = judge_provider(judge)
    from threetears.evals.providers import Model

    # The same cap and effort as rung 5's judge: a score and a short reason fit in 2,048 tokens, and the pre-launch
    # bound prices every rubric call at this cap, so the engine's 8,192 default would refuse a run the cap admits.
    return Model(
        judge,
        provider=provider,
        api_key=os.environ[_KEYS[provider]],
        max_tokens=JUDGE_MAX_TOKENS,
        effort="low",
        recording=out / "judge.jsonl",
    )


async def _measured(
    scoring: WriterJudge,
    cases: Path,
    arms: Mapping[str, Writer],
    *,
    store_at: Path | None,
    recordings: Path,
    budget: ReportBudget,
    request_s: float,
    held_out_fraction: float,
    cap_usd: float,
    variant: Variant,
) -> tuple[WriterResult, WriterPairwiseReport, WriterPromotion]:
    """Run the writer eval, which runs the arms, the pairwise pass and the read; the result, the pass and the read."""
    assert budget.usd is not None and budget.turns is not None
    result = await WriterEval(
        cases,
        arms=arms,
        judge=scoring,
        recordings=recordings,
        budget_usd=budget.usd,
        max_cost_usd=cap_usd,
        held_out_fraction=held_out_fraction,
        name=NAME,
        question=variant.question,
        turns=budget.turns,
        request_s=request_s,
        writer_provider=WRITER_PROVIDER,
        store=":memory:" if store_at is None else store_at,
    ).run()
    if variant.name in result.pairwise_unread:
        raise SystemExit(f"writer_acceptance.py: the pairwise pass {result.pairwise_unread[variant.name]}")
    # The eval read the promotion on its pass; the criterion reads the whole of both, off the pass's stored job.
    job = result.pairwise[variant.name].job_id
    status = await job_poll(OpsHost.reading(result.host), job, result.scope_id)
    read = await asyncio.to_thread(
        writer_promotion,
        result.host,
        result.campaign_id,
        result.scope_id,
        candidate=variant.name,
        pairwise_job=status,
    )
    return result, WriterPairwiseReport.model_validate(status.result), read


async def acceptance(
    *,
    dry: bool,
    cases_dir: Path | None,
    cap_usd: float,
    out: Path | None,
    writer: str = WRITER,
    editor_model: str | None = None,
    judge: str = JUDGE,
    effort: str | None = EFFORT,
    budget_usd: float = BUDGET_USD,
    turns: int = TURNS,
    request_s: float = REQUEST_S,
    held_out_fraction: float = HELD_OUT_FRACTION,
    variant: str = HOLLOW,
) -> Acceptance:
    """Run both arms, the pairwise pass and the promotion read, and decide the pass criterion.

    :param dry: Scripted models, nothing sent; ``out`` may be ``None``.
    :param cases_dir: The writer case files; ``None`` for the default cases, written into a scratch directory.
    :param cap_usd: The whole run's cap, split by the writer eval's rule (the module docstring).
    :param out: Where a live run's recordings, store and result go.
    :param variant: The candidate, by its name in :data:`VARIANTS`.
    :return: What it found.
    """
    chosen = VARIANTS[variant]
    shipped = load_skill()
    candidate = chosen.build(shipped)
    refuse_unlike_editors(shipped, candidate, chosen)
    arms = {
        name: Writer(writer, skill=skill, effort=effort, revisions=revisions, editor_model=editor_model)
        for name, skill, revisions in (
            (SHIPPED, shipped, MOST_REVISIONS),
            (chosen.name, candidate, chosen.revisions),
        )
    }
    with tempfile.TemporaryDirectory(prefix="writer-acceptance-") as scratch:
        root = Path(scratch) if out is None else out
        cases = cases_dir if cases_dir is not None else default_cases(Path(scratch) / "cases")
        recordings = root / "recordings"
        with_documents = None
        if dry:
            with_documents = lay_dry_recordings(recordings, read_writer_cases(cases), arms, provider=WRITER_PROVIDER)
            scoring: WriterJudge = FakeDryJudge(judge, provider=judge_provider(judge))
        else:
            assert out is not None
            scoring = _live_judge(judge, out)
        try:
            result, report, read = await _measured(
                scoring,
                cases,
                arms,
                store_at=None if dry else root / "acceptance.sqlite",
                recordings=recordings,
                budget=ReportBudget(usd=budget_usd, turns=turns),
                request_s=request_s,
                held_out_fraction=held_out_fraction,
                cap_usd=cap_usd,
                variant=chosen,
            )
        finally:
            await scoring.aclose()
    return Acceptance(
        dry=dry,
        shipped=shipped,
        candidate=candidate,
        result=result,
        with_documents=with_documents,
        report=report,
        read=read,
        criterion=pass_criterion(read, result.arms, variant=chosen),
        cap_usd=cap_usd,
        editor_model=editor_model,
        variant=chosen,
    )


def _amount(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"an amount above 0, not {text!r}")
    return value


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="writer_acceptance.py", description=__doc__.splitlines()[0])
    parser.add_argument("--dry", action="store_true", help="scripted models, nothing sent")
    parser.add_argument(
        "--variant", choices=list(VARIANTS), default=HOLLOW, help="the candidate run against the shipped skill"
    )
    parser.add_argument("--cases", type=Path, help="a directory of writer case files; default: prototype 5's seven")
    parser.add_argument("--cap-usd", type=_amount, help="the owner's cap for the whole run; a live run names it")
    parser.add_argument("--writer", default=WRITER, help="the model both arms write on, on Anthropic's API")
    parser.add_argument("--editor-model", help="the model the editor runs on; default: the writer's")
    parser.add_argument("--judge", default=JUDGE, help="the fixed scoring judge, of another family than the writer")
    parser.add_argument("--effort", default=EFFORT, help="the writer's reasoning effort, for both arms")
    parser.add_argument("--budget-usd", type=_amount, default=BUDGET_USD, help="each report's dollar cap")
    parser.add_argument("--turns", type=int, default=TURNS, help="each report's turn cap")
    parser.add_argument("--request-s", type=_amount, default=REQUEST_S, help="the ceiling for one writer request")
    parser.add_argument("--held-out-fraction", type=float, default=HELD_OUT_FRACTION, help="groups on held-out")
    parser.add_argument("--out", type=Path, help="where a live run's recordings, store and result go")
    options = parser.parse_args(list(argv))
    if not options.dry:
        if options.out is None or options.cap_usd is None:
            parser.error("a live run names --cap-usd and --out")
        refuse_too_few_cases(options.cases, options.held_out_fraction)
        refuse_missing_keys(options.judge)
        # Made before anything is sent, so a directory already there refuses the run rather than its result.
        options.out.mkdir(parents=True, exist_ok=False)
    found = asyncio.run(
        acceptance(
            dry=options.dry,
            cases_dir=options.cases,
            cap_usd=options.cap_usd or 10.0,
            out=options.out,
            writer=options.writer,
            editor_model=options.editor_model,
            judge=options.judge,
            effort=options.effort,
            budget_usd=options.budget_usd,
            turns=options.turns,
            request_s=options.request_s,
            held_out_fraction=options.held_out_fraction,
            variant=options.variant,
        )
    )
    summary = found.summary()
    if options.out is not None:
        (options.out / "acceptance.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(summary["readout"])
    if found.result.exit_code() == 1:
        return 1  # an arm's run did not complete
    if found.dry:
        return 0  # the dry run's verdict carries no signal
    return 0 if found.criterion.passed else 4


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
