"""The plan-scoring dataset beside this file: its inputs, anchors, rubric, score records and the brief a scorer reads.

The planner eval keeps every plan the planner writes and every score of one, to calibrate the package's plan judge
later. This module is the dataset's one reader:

- **Inputs.** ``r1`` to ``r5`` each hold ``input.json``, a report input (:func:`report_input_of`). The Kairo round 2
  campaign bundle the r5 input was adapted from is committed gzipped (``kairo-round2-bundle.json.gz``,
  :func:`kairo_bundle`), so ``report_input(bundle)`` runs against the real stored campaign. It has no generator: it is
  real data. It is moved across an aspect bundle shape by a migration script that recomputes only what the shape
  changes from what the bundle holds, and refuses when any other field moves. At shape 65 that was
  ``migrate_bundle_shape_65.py kairo-round2-bundle.json.gz``, which moved eight rate contrasts' intervals (the pass
  rate, a goal check and two pass/fail judged dimensions, two arms each) onto their stored bounded intervals, every
  verdict holding, named each corrected comparison's interval basis, and moved the shape.
- **Plans.** ``rN/anchor.plan.json`` is a hand-written plan for that input; ``anchors/weak/*.plan.json`` are copies
  weakened on one plan-check rule or one rubric item (``weaken_anchors.py`` writes them). ``anchors/LABELS.json`` says
  what each anchor is expected to be and who labeled it: every anchor starts ``pending_owner`` and counts as a label
  only once the owner approves it. ``rN/runs/<run_id>.plan.json`` are the planner's plans (``plan_eval.py``), each
  beside ``<run_id>.planner.json``, what wrote it (:class:`PlannerInfo`: the model, its settings, the digests of the
  prompt and contracts it was asked under), written when it was recorded.
- **Runs are kept.** A re-record adds a run and never deletes one or its score (owner, 2026-10-25: every plan and every
  score is the calibration dataset). The **current** run of an input is the one its recording replays to, named in
  ``recordings/runs.json`` (:func:`current_run`); the planner eval's checks (the plan checks, R1, the replay) hold for
  it. Every other run is a historical record, held only to record soundness: its provenance and its scores' digests,
  rubric and quotes.
- **Rubric.** ``rubric/v<N>.md`` and ``rubric/v<N>.json``, one pair per version: immutable, pinned by
  ``rubric/DIGESTS.json``. R1 is code's (:func:`code_score`); the rest are a model judge's. The brief and a record
  use the newest version (:func:`latest_rubric`) unless told otherwise, and a record names the version it scored.
- **Scores.** ``rN/runs/<run_id>.scores.json``, one :class:`PlanScores` per run plan. A record is sound when
  :func:`score_faults` finds nothing: its plan and rubric digests match the files, its input digest resolves to an
  input the dataset keeps, and every model-scored item is scored once. Records pool only by :func:`pool_key`: the
  planner prompt digest, the rubric digest and the scorer. Nothing else may join them.
- **Inputs are kept by digest.** An input changes (a fixture rebuilt, a field filled), and a record names the input it
  was scored against by its sha256, so every input a record was scored against is archived byte for byte as
  ``inputs/<sha256>.json`` (:func:`archived_input`), written when the record is (``dataset.py record``). History stays
  verifiable without scoring again.
- **History is immutable.** A quote is text the plan rendered when it was scored; the renderer moves on. So quotes are
  checked when a record is written and for each input's current run, and a historical record (any run not current) is
  pinned instead: ``history.json`` holds the sha256 of every historical record's file (:func:`history_faults`), and
  ``dataset.py pin`` adds a run that a re-record made historical, never changing a pin already there. Nobody edits
  history to pass.

Scoring one plan is two commands from the repository root, with the scorer's reply saved to a file between them::

    uv run python packages/evals/tests/fixtures/report_plans/dataset.py brief r5/runs/<run_id>.plan.json
    uv run python packages/evals/tests/fixtures/report_plans/dataset.py record r5/runs/<run_id>.plan.json reply.json claude-opus-5-5

After a re-record makes the previous current runs historical, ``dataset.py pin`` pins their records.

The brief holds the rubric, R1 as code scored it, the plan in page order with every prose field as written and as
rendered, the input's facts and the planner's view of the input, and the reply's shape. It is blind: it names neither
the plan's file nor its planner, and asks for no time, since a blind scorer reads no clock. ``record`` adds the run,
the digests, the planner, the rubric and the time it is made to the reply, checks the record (:func:`score_faults`,
quotes included), writes ``<run_id>.scores.json`` beside the plan and archives the input it was scored against.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.errors import PlanRefusal
from threetears.evals.analysis.report.figure_keys import render
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import planner_view
from threetears.evals.analysis.report_plan_checks import validate_plan
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.kernel.report_plan import ChartSpec, ReportPlan, prose_response_format, structure_response_format
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.schema.prose import field_is_prose

__all__ = [
    "HERE",
    "KAIRO_BUNDLE",
    "HISTORY",
    "INPUTS",
    "LABELS",
    "OPUS_SCORER",
    "PAGE_ORDER",
    "SCORES_SCHEMA",
    "PlanScores",
    "PlannerInfo",
    "archive_input",
    "archived_input",
    "archived_input_path",
    "history_faults",
    "is_current",
    "pin_history",
    "pinned_records",
    "RubricRef",
    "ScoreItem",
    "ScorerReply",
    "code_score",
    "current_run",
    "current_runs",
    "kairo_bundle",
    "labels",
    "load_plan",
    "model_items",
    "plan_input",
    "planner_json",
    "planner_of",
    "planner_path",
    "pool_key",
    "pools",
    "prose_fields",
    "recorded_planner",
    "report_input_of",
    "latest_rubric",
    "rubric",
    "record",
    "rubric_digest",
    "rubric_versions",
    "run_plans",
    "score_faults",
    "score_skeleton",
    "scores_json",
    "scores_path",
    "scoring_brief",
    "sha256_file",
]

HERE = Path(__file__).resolve().parent
#: The Kairo round 2 campaign bundle, as the writer read it, gzipped.
KAIRO_BUNDLE = HERE / "kairo-round2-bundle.json.gz"
#: What each anchor is expected to be, and who labeled it.
LABELS = HERE / "anchors" / "LABELS.json"
RUBRICS = HERE / "rubric"
RECORDINGS = HERE / "recordings"
#: Each input's current run: the run its recording replays to, by input.
CURRENT_RUNS = RECORDINGS / "runs.json"
#: The schema id every score record carries.
SCORES_SCHEMA = "plan-scores/1"
#: Every report input a score record was scored against, by its sha256: ``inputs/<sha256>.json``.
INPUTS = HERE / "inputs"
#: The sha256 of every historical score record's file, by its path under this directory.
HISTORY = HERE / "history.json"
#: The schema id the history file carries.
HISTORY_SCHEMA = "plan-score-history/1"
#: The scorer the dataset's model-scored records name: the id the score records were written under, an identity in
#: recorded calibration data, never a model this code calls (records pool by it, ``pool_key``).
OPUS_SCORER = "claude-opus-5-5"
#: Who wrote an anchor's plan, in place of a planner model.
HAND_WRITTEN = "hand-written"


def sha256_file(path: Path) -> str:
    """A file's sha256, as lowercase hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kairo_bundle() -> CampaignBundle:
    """The Kairo round 2 campaign bundle, read from its gzipped fixture."""
    with gzip.open(KAIRO_BUNDLE, "rt", encoding="utf-8") as stream:
        return CampaignBundle.from_dict(json.load(stream))


def report_input_of(name: str) -> ReportInput:
    """One fixture input (``r1`` to ``r5``)."""
    return ReportInput.model_validate_json((HERE / name / "input.json").read_text(encoding="utf-8"))


def archived_input_path(digest: str) -> Path:
    """Where the input with sha256 ``digest`` is archived."""
    return INPUTS / f"{digest}.json"


def archived_input(digest: str) -> ReportInput:
    """The archived input with sha256 ``digest``, checked against its digest.

    :raises ValueError: No input with that digest is archived, or the archived file is not that input.
    """
    path = archived_input_path(digest)
    if not path.is_file():
        raise ValueError(f"no input with sha256 {digest} is archived under {INPUTS.name}/")
    if sha256_file(path) != digest:
        raise ValueError(f"{INPUTS.name}/{path.name} is not the input its name says: its bytes changed")
    return ReportInput.model_validate_json(path.read_text(encoding="utf-8"))


def archive_input(name: str) -> str:
    """Archive one fixture input's current bytes under its digest, if not archived yet.

    :return: Its sha256.
    """
    current = (HERE / name / "input.json").read_bytes()
    digest = hashlib.sha256(current).hexdigest()
    path = archived_input_path(digest)
    if not path.is_file():
        INPUTS.mkdir(exist_ok=True)
        path.write_bytes(current)
    return digest


def _input_at(name: str, digest: str) -> ReportInput | None:
    """The input with sha256 ``digest``: the archived one, else ``name``'s current input where it is that one."""
    if archived_input_path(digest).is_file():
        return archived_input(digest)
    if sha256_file(HERE / name / "input.json") == digest:
        return report_input_of(name)
    return None


def load_plan(path: Path) -> ReportPlan:
    """A stored plan."""
    return ReportPlan.model_validate_json(path.read_text(encoding="utf-8"))


def labels() -> list[dict[str, Any]]:
    """Every anchor's label entry (``anchors/LABELS.json``)."""
    loaded: dict[str, Any] = json.loads(LABELS.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = loaded["anchors"]
    return entries


def plan_input(path: Path) -> str:
    """The input a plan in the dataset was written against: its ``rN`` directory, or its anchor label's input."""
    relative = path.resolve().relative_to(HERE)
    if relative.parts[0] != "anchors":
        return relative.parts[0]
    entry = next((entry for entry in labels() if entry["plan"] == relative.as_posix()), None)
    if entry is None:
        raise ValueError(f"{relative} has no entry in {LABELS.name}")
    found: str = entry["input"]
    return found


# --- the rubric ------------------------------------------------------------------------------------------------------


def rubric_versions() -> list[str]:
    """Every rubric version, oldest first, as ``rubric/DIGESTS.json`` pins them."""
    pinned: dict[str, Any] = json.loads((RUBRICS / "DIGESTS.json").read_text(encoding="utf-8"))
    return sorted(pinned, key=lambda version: int(version.removeprefix("v")))


def latest_rubric() -> str:
    """The newest rubric version: what a brief and a record use unless told otherwise."""
    return rubric_versions()[-1]


def rubric(version: str | None = None) -> dict[str, Any]:
    """A rubric version's items, as data; the newest version by default."""
    loaded: dict[str, Any] = json.loads((RUBRICS / f"{version or latest_rubric()}.json").read_text(encoding="utf-8"))
    return loaded


def rubric_digest(version: str | None = None) -> str:
    """A rubric version's digest: the sha256 of its JSON file, which a score record names; the newest by default."""
    return sha256_file(RUBRICS / f"{version or latest_rubric()}.json")


def model_items(version: str | None = None) -> list[str]:
    """The items a model (or the owner) scores, in the rubric's order; the rest are code's. The newest by default."""
    return [item["id"] for item in rubric(version)["items"] if item["scored_by"] == "model"]


def code_score(plan: ReportPlan, report_input: ReportInput, version: str | None = None) -> tuple[str, str]:
    """R1, scored by code: the plan checks pass and the first evidence section draws the shape's spine chart.

    :return: ``(verdict, reason)``.
    """
    try:
        validate_plan(plan, report_input)
    except PlanRefusal as refused:
        return "fail", f"the plan checks refuse it: {refused}"
    spines: list[str] = rubric(version)["spine"].get(plan.shape, [])
    if not plan.sections:
        return "fail", "the plan has no evidence section, so no spine chart"
    first = plan.sections[0].chart.kind
    if first not in spines:
        return "fail", f"the first evidence section draws {first!r}, and shape {plan.shape} opens on {spines}"
    return "pass", f"the plan checks pass and the first section draws {first!r}, shape {plan.shape}'s spine"


# --- score records ---------------------------------------------------------------------------------------------------


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScoreItem(_Record):
    """One rubric item's score."""

    id: str = Field(description="The rubric item's id, R2 onward.")
    verdict: Literal["pass", "fail"]
    reason: str = Field(min_length=1, description="One sentence: why.")
    quote: str = Field(description="Text copied exactly from the plan's prose; required for a fail.")
    labeled_by: Literal["model", "owner"] = Field(description="Who gave the verdict: a model judge or the owner.")


class PlannerInfo(_Record):
    """What wrote the plan: the planner model and how it was asked, or ``hand-written`` for an anchor."""

    model: str
    provider: str
    settings: dict[str, Any]
    prompt_digest: str = Field(description="sha256 of the planner prompt the plan was written under; empty for none.")
    contract_digest: str = Field(description="sha256 of both planner contracts; empty for a hand-written plan.")


class RubricRef(_Record):
    """The rubric version scored against, and its digest."""

    version: str
    digest: str


class PlanScores(_Record):
    """One plan's scores against one rubric version, by one scorer."""

    schema_id: Literal["plan-scores/1"] = Field(alias="schema")
    run_id: str = Field(description="The plan file's name without ``.plan.json``.")
    input: str = Field(description="The input directory the plan was written against, ``r1`` to ``r5``.")
    input_digest: str = Field(
        description="sha256 of that input's ``input.json`` as scored, archived as ``inputs/<sha256>.json``."
    )
    plan_digest: str = Field(description="sha256 of the plan file scored.")
    planner: PlannerInfo
    scorer: str = Field(description="The scorer: a model id (``claude-opus-5-5``) or ``owner``.")
    rubric: RubricRef
    items: list[ScoreItem]
    scored_at: str = Field(description="When code recorded the scorer's reply, ISO-8601, by code's clock.")


def prose_fields(plan: BaseModel, path: str = "") -> Iterator[tuple[str, str]]:
    """Every prose field a plan holds (a field typed model prose), by its path, in the plan's order, and the masthead's
    lines, which code writes with figure keys in them."""
    for name, field in type(plan).model_fields.items():
        value = getattr(plan, name)
        here = f"{path}.{name}" if path else name
        if isinstance(value, str):
            if field_is_prose(field) or path == "masthead":
                yield here, value
        elif isinstance(value, BaseModel):
            yield from prose_fields(value, here)
        elif isinstance(value, Sequence):
            for index, item in enumerate(value):
                if isinstance(item, BaseModel):
                    yield from prose_fields(item, f"{here}[{index}]")


#: A thousands separator between digits, which the renderer of the day may or may not have written.
_GROUPED = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _ungrouped(text: str) -> str:
    """``text`` with no thousands separator between digits: a quote a scorer recorded is read as recorded, whether the
    renderer then grouped a count (``1190``) or now does (``1,190``)."""
    return _GROUPED.sub("", text)


def _readable(plan: ReportPlan, report_input: ReportInput) -> list[tuple[str, str, str]]:
    """Each prose field as written and as rendered."""
    return [(path, text, render(text, report_input, {})) for path, text in prose_fields(plan)]


def score_faults(scores: PlanScores, plan_path: Path, *, quotes: bool = True) -> list[str]:
    """Everything wrong with a score record of the plan at ``plan_path``; empty when it is sound.

    Sound means: it names the plan's file by its digest, and its input by a digest that resolves to an input the
    dataset keeps (archived, or the input as it stands); its rubric version exists and its digest is that version's;
    every model-scored item of the version is scored exactly once and nothing else is; and every fail quotes what
    failed. With ``quotes``, every quote is also a substring of one of the plan's prose fields, as written or as
    rendered against that input, a thousands separator between digits set aside (:func:`_ungrouped`): checked when a record is written and for a current run, while a historical record is
    pinned instead (:func:`history_faults`).
    """
    faults: list[str] = []
    name = plan_path.name.removesuffix(".plan.json")
    if scores.run_id != name:
        faults.append(f"run_id is {scores.run_id!r}, and the plan file is {name!r}")
    if scores.plan_digest != sha256_file(plan_path):
        faults.append("plan_digest is not the plan file's sha256: the plan changed after it was scored")
    expected_input = plan_input(plan_path)
    scored_against: ReportInput | None = None
    if scores.input != expected_input:
        faults.append(f"input is {scores.input!r}, and the plan was written against {expected_input!r}")
    else:
        scored_against = _input_at(expected_input, scores.input_digest)
        if scored_against is None:
            faults.append(
                f"input_digest {scores.input_digest} names no input the dataset keeps: neither archived under "
                f"{INPUTS.name}/ nor {expected_input}/input.json as it stands"
            )
    if not (RUBRICS / f"{scores.rubric.version}.json").is_file():
        return [*faults, f"rubric version {scores.rubric.version!r} does not exist"]
    if scores.rubric.digest != rubric_digest(scores.rubric.version):
        faults.append(f"rubric digest is not {scores.rubric.version}.json's sha256")
    wanted = model_items(scores.rubric.version)
    ids = [item.id for item in scores.items]
    for item_id in wanted:
        if ids.count(item_id) != 1:
            faults.append(f"item {item_id} is scored {ids.count(item_id)} times, and every item is scored once")
    faults += [f"item {item_id} is not a model-scored item of the rubric" for item_id in ids if item_id not in wanted]
    for item in scores.items:
        if item.verdict == "fail" and not item.quote.strip():
            faults.append(f"item {item.id} fails with no quote of what failed")
    if not quotes or scored_against is None:
        return faults
    readable = _readable(load_plan(plan_path), scored_against)
    for item in scores.items:
        quote = _ungrouped(item.quote)
        if item.quote and not any(
            quote in _ungrouped(text) or quote in _ungrouped(shown) for _, text, shown in readable
        ):
            faults.append(f"item {item.id} quotes {item.quote[:60]!r}, which no prose field of the plan holds")
    return faults


def is_current(plan_path: Path) -> bool:
    """Whether a run plan is its input's current run (:func:`current_run`)."""
    return any(plan_path.resolve() == current_run(name).resolve() for name in current_runs())


def pinned_records() -> dict[str, str]:
    """Every pinned historical record's sha256, by its path under this directory (``history.json``)."""
    if not HISTORY.is_file():
        return {}
    loaded: dict[str, Any] = json.loads(HISTORY.read_text(encoding="utf-8"))
    records: dict[str, str] = loaded["records"]
    return records


def _historical_records() -> list[Path]:
    return [scores_path(plan) for plan in run_plans() if not is_current(plan) and scores_path(plan).is_file()]


def history_faults() -> list[str]:
    """Everything wrong with the pinned history; empty when it holds: every pinned record is on disk with the bytes
    it was pinned with, and every historical run's record is pinned."""
    pinned = pinned_records()
    faults: list[str] = []
    for relative, digest in pinned.items():
        path = HERE / relative
        if not path.is_file():
            faults.append(f"{relative} is pinned and gone: history is kept, never deleted")
        elif sha256_file(path) != digest:
            faults.append(f"{relative} changed after it was pinned: a historical record is immutable")
    for path in _historical_records():
        relative = path.relative_to(HERE).as_posix()
        if relative not in pinned:
            faults.append(f"{relative} is a historical run's record and is not pinned: run `dataset.py pin`")
    return faults


def pin_history() -> list[str]:
    """Pin every historical run's record not pinned yet, keeping every pin already there.

    :return: The paths pinned now.
    """
    pinned = pinned_records()
    added = [
        path.relative_to(HERE).as_posix()
        for path in _historical_records()
        if path.relative_to(HERE).as_posix() not in pinned
    ]
    records = pinned | {relative: sha256_file(HERE / relative) for relative in added}
    HISTORY.write_text(
        json.dumps({"schema": HISTORY_SCHEMA, "records": dict(sorted(records.items()))}, indent=2) + "\n",
        encoding="utf-8",
    )
    return added


def pool_key(scores: PlanScores) -> tuple[str, str, str]:
    """What score records must share to pool: the planner prompt digest, the rubric digest and the scorer."""
    return scores.planner.prompt_digest, scores.rubric.digest, scores.scorer


def pools(records: Sequence[PlanScores]) -> dict[tuple[str, str, str], list[PlanScores]]:
    """Score records grouped by :func:`pool_key`, and by nothing else."""
    grouped: dict[tuple[str, str, str], list[PlanScores]] = {}
    for record in records:
        grouped.setdefault(pool_key(record), []).append(record)
    return grouped


# --- the brief -------------------------------------------------------------------------------------------------------


def _contract_digest() -> str:
    both = [structure_response_format(), prose_response_format()]
    return hashlib.sha256(json.dumps(both, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def planner_path(plan_path: Path) -> Path:
    """Where a run plan's provenance is kept: beside it, ``<run_id>.planner.json``."""
    return plan_path.with_name(plan_path.name.removesuffix(".plan.json") + ".planner.json")


def _is_run(plan_path: Path) -> bool:
    relative = plan_path.resolve().relative_to(HERE)
    return len(relative.parts) == 3 and relative.parts[1] == "runs"


def planner_of(plan_path: Path) -> PlannerInfo:
    """What wrote a plan: a run's provenance, kept beside it since it was recorded, else hand-written.

    :raises ValueError: The plan is a run with no provenance beside it.
    """
    if not _is_run(plan_path):
        return PlannerInfo(model=HAND_WRITTEN, provider="", settings={}, prompt_digest="", contract_digest="")
    kept = planner_path(plan_path)
    if not kept.is_file():
        raise ValueError(f"{plan_path.name} is a run with no {kept.name} beside it saying what wrote it")
    return PlannerInfo.model_validate_json(kept.read_text(encoding="utf-8"))


def recorded_planner(name: str) -> PlannerInfo:
    """What the current recording of one input was made with: its model and settings, read off the recording, and
    the digests of the recorded prompt and of today's contracts. True only while recording, which is when
    ``plan_eval.py`` writes it beside the new run."""
    first = json.loads((RECORDINGS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()[0])
    model = str(first["result"].get("model") or "")
    return PlannerInfo(
        model=model,
        provider="openrouter" if "/" in model else "anthropic",
        settings=first["settings"],
        prompt_digest=sha256_file(RECORDINGS / "prompt.txt"),
        contract_digest=_contract_digest(),
    )


def planner_json(info: PlannerInfo) -> str:
    """A run's provenance as the dataset keeps it."""
    return info.model_dump_json(indent=2) + "\n"


def current_runs() -> dict[str, str]:
    """Each input's current run id: the run its recording replays to."""
    loaded: dict[str, str] = json.loads(CURRENT_RUNS.read_text(encoding="utf-8"))
    return loaded


def current_run(name: str) -> Path:
    """The plan of one input's current run."""
    return HERE / name / "runs" / f"{current_runs()[name]}.plan.json"


def run_plans() -> list[Path]:
    """Every run plan the dataset keeps, current and historical, in path order."""
    return sorted(HERE.glob("r*/runs/*.plan.json"))


def score_skeleton(plan_path: Path, *, scorer: str = OPUS_SCORER, version: str | None = None) -> dict[str, Any]:
    """A score record for the plan at ``plan_path`` with every field but the verdicts set."""
    name = plan_input(plan_path)
    version = version or latest_rubric()
    return {
        "schema": SCORES_SCHEMA,
        "run_id": plan_path.name.removesuffix(".plan.json"),
        "input": name,
        "input_digest": sha256_file(HERE / name / "input.json"),
        "plan_digest": sha256_file(plan_path),
        "planner": planner_of(plan_path).model_dump(),
        "scorer": scorer,
        "rubric": {"version": version, "digest": rubric_digest(version)},
        "items": [
            {"id": item_id, "verdict": "", "reason": "", "quote": "", "labeled_by": "model"}
            for item_id in model_items(version)
        ],
        "scored_at": "",
    }


#: GENERATOR §2's page order, top to bottom, as the brief lists the plan: each part's heading and the plan fields
#: whose prose it holds. Durable claims are not on the page; they close the brief.
PAGE_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Masthead", ("masthead",)),
    ("Headline", ("headline",)),
    ("Lede", ("lede",)),
    ("Flag", ("flag",)),
    ("Decision block", ("decision", "answers")),
    ("Options table", ("options",)),
    ("Summary charts", ("summary",)),
    ("Arms key", ()),
    ("Evidence sections", ("sections",)),
    ("Explore", ("explore",)),
    ("Full results", ("full_results",)),
    ("Method", ("method",)),
    ("Durable claims (not on the page)", ("durable_claims",)),
)


def _top(path: str) -> str:
    """The plan field a prose path sits under: ``sections[1].headline`` is under ``sections``."""
    return path.split(".", 1)[0].split("[", 1)[0]


def _chart_line(where: str, chart: ChartSpec) -> str:
    query = chart.query
    return (
        f"Chart {where}: {chart.kind} of {query.source} aspect={query.aspect!r} measures={query.measures} "
        f"arms={query.arms} strata={query.strata} transform={query.transform} by={query.small_multiples_by} "
        f"emphasis={chart.emphasis!r} level={chart.level!r} unit={chart.unit!r}"
    )


def _choices(part: str, plan: ReportPlan) -> Iterator[str]:
    """The choices one part of the page makes, each in a line, after its prose."""
    if part == "Flag":
        for flag in plan.flag:
            yield f"Flag discloses {flag.discloses}; weakens {flag.weakens}; survives {flag.survives}"
    elif part == "Decision block":
        yield f"Confidence form: {plan.decision.confidence.form}"
        for slot in plan.decision.slots:
            if slot.caveats:
                yield f"Caveats landed on decision slot {slot.key}: {slot.caveats}"
    elif part == "Options table":
        yield "Options table rows: " + ", ".join(f"{row.option} ({row.role})" for row in plan.options.rows)
        yield f"Options columns: {plan.options.columns}"
    elif part == "Summary charts":
        yield from (_chart_line(f"summary[{i}]", chart) for i, chart in enumerate(plan.summary))
    elif part == "Arms key":
        yield f"Shown: {plan.show_arms_key}"
    elif part == "Evidence sections":
        for index, section in enumerate(plan.sections):
            yield _chart_line(f"sections[{index}] {section.id}", section.chart)
            if section.caveats:
                yield f"Caveats landed on sections[{index}] {section.id}: {section.caveats}"
    elif part == "Full results":
        yield f"Tables: {plan.full_results}"
    elif part == "Method":
        for index, item in enumerate(plan.method):
            if item.caveats:
                yield f"Caveats landed on method[{index}]: {item.caveats}"


def _page(plan: ReportPlan, report_input: ReportInput) -> Iterator[str]:
    """The plan in page order (GENERATOR §2, :data:`PAGE_ORDER`): each part under its heading, its prose fields as
    written and rendered, then the choices it makes, so a scorer sees what sits above Full results and what below."""
    yield f"Shape {plan.shape}, mode {plan.mode}, headline kind {plan.headline_kind}."
    readable = _readable(plan, report_input)
    for part, fields in PAGE_ORDER:
        yield f"\n### {part}"
        for path, text, shown in readable:
            if _top(path) in fields:
                yield f"\n[{path}]\n  as written: {text}\n  rendered:   {shown}"
        yield from _choices(part, plan)


class ScorerReply(_Record):
    """What a scorer returns for one plan: its items. Code adds everything else, the time included: a blind scorer
    reads no clock, so a time it wrote would be a guess."""

    items: list[ScoreItem]


def scoring_brief(plan_path: Path, *, version: str | None = None) -> str:
    """The brief a scorer reads for one plan: the task, the rubric, the plan, the input's facts and the reply's shape.

    It is blind: it names neither the plan's file nor what wrote it, since an anchor's file name says what it is
    weakened on and a run's names its planner model. :func:`record` adds both to the reply.
    """
    version = version or latest_rubric()
    name = plan_input(plan_path)
    report_input = report_input_of(name)
    plan = load_plan(plan_path)
    verdict, reason = code_score(plan, report_input, version)
    facts = plan_facts(report_input).model_dump(mode="json")
    rubric_text = (RUBRICS / f"{version}.md").read_text(encoding="utf-8")
    reply = {
        "items": [
            {"id": item_id, "verdict": "pass | fail", "reason": "", "quote": "", "labeled_by": "model"}
            for item_id in model_items(version)
        ]
    }
    parts = [
        f"# Score one report plan against rubric {version}",
        (
            "Score every model-scored item of the rubric below, each on its own, as a reviewer reads the page. Return "
            "ONLY a JSON object shaped like the reply at the end of this brief: for each item its verdict (pass or "
            "fail), a one-sentence reason, and a quote copied exactly from the plan's prose (as written or as "
            "rendered; a fail always quotes what failed)."
        ),
        f"R1 is code's and already scored: {verdict}, {reason}.",
        rubric_text,
        "## The plan, in page order",
        "\n".join(_page(plan, report_input)),
        "## What code decided about the input (facts the planner was handed)",
        json.dumps(facts, ensure_ascii=False),
        "## The report input, as the planner read it",
        planner_view(report_input),
        "## The reply",
        json.dumps(reply, indent=2),
    ]
    return "\n\n".join(parts) + "\n"


def scores_path(plan_path: Path) -> Path:
    """Where a plan's score record is kept: beside it, ``<run_id>.scores.json``."""
    return plan_path.with_name(plan_path.name.removesuffix(".plan.json") + ".scores.json")


def record(plan_path: Path, reply: str, *, scorer: str, version: str | None = None) -> PlanScores:
    """A scorer's reply for one plan made into its score record, checked; the caller writes it.

    :param plan_path: The plan scored.
    :param reply: The scorer's reply: the JSON object the brief asks for, optionally inside a code fence.
    :param scorer: Who scored it: a model id (``claude-opus-5-5``) or ``owner``.
    :param version: The rubric version scored against.
    :raises ValueError: The reply is not the object the brief asks for, or the record is unsound (:func:`score_faults`).
    """
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    answered = ScorerReply.model_validate_json(text)
    skeleton = score_skeleton(plan_path, scorer=scorer, version=version)
    skeleton["items"] = [item.model_dump() for item in answered.items]
    skeleton["scored_at"] = utc_now_iso()
    scores = PlanScores.model_validate(skeleton)
    if faults := score_faults(scores, plan_path):
        raise ValueError(f"the score record of {plan_path.name} is unsound: {'; '.join(faults)}")
    return scores


def scores_json(scores: PlanScores) -> str:
    """A score record as the dataset keeps it."""
    return scores.model_dump_json(by_alias=True, indent=2) + "\n"


_USAGE = """usage, from the repository root (paths relative to report_plans/):
  dataset.py brief <plan>                  print the plan's scoring brief
  dataset.py record <plan> <reply> <scorer> [<rubric version>]
                                           write <run_id>.scores.json beside the plan from a scorer's reply file,
                                           scored against the newest rubric unless a version is named, and archive
                                           the input it was scored against
  dataset.py pin                           pin every historical run's record not pinned yet"""


def main(argv: Sequence[str]) -> int:
    """``brief <plan>`` prints a plan's scoring brief; ``record <plan> <reply file> <scorer>`` writes its record;
    ``pin`` pins the historical records."""

    def at(given: str) -> Path:
        path = Path(given)
        return path if path.is_absolute() else HERE / path

    if len(argv) == 2 and argv[0] == "brief":
        print(scoring_brief(at(argv[1])))
        return 0
    if len(argv) in (4, 5) and argv[0] == "record":
        plan = at(argv[1])
        version = argv[4] if len(argv) == 5 else None
        scores = record(plan, Path(argv[2]).read_text(encoding="utf-8"), scorer=argv[3], version=version)
        archive_input(scores.input)
        scores_path(plan).write_text(scores_json(scores), encoding="utf-8")
        print(f"Wrote {scores_path(plan).relative_to(HERE)}.")
        return 0
    if argv == ["pin"]:
        added = pin_history()
        print(f"Pinned {len(added)} historical record(s) in {HISTORY.name}.")
        return 0
    print(_USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
