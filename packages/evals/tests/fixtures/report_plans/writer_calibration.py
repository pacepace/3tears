"""The report writer's judges against the Opus editor: the calibration set, and the judge arms that read it.

Every recorded writer run of prototypes 4 and 5 (``fixtures/report_docs/prototype{4,5}``) holds the Opus editor's
review of each round's page. This script turns those reviews into judge-kind cases (reset P2 §3):

1. **Pages.** Each recording is replayed through :class:`~threetears.evals.ops.report_writer.writer_kind.ReportWriterKind`
   once per round it recorded, at ``revisions`` 0, 1 and 2, so each replay ends on that round's page. Its
   ``CandidateOutput.judge_evidence`` is the page exactly as the gate's judge reads it. A replay that does not reach
   its round's page is left out, and named with why.
2. **Reviews.** Each page is paired with the Opus review of its round (``generation.json``'s ``reviews``), keyed by
   (recording, round). The replay's open findings must match that review's, or the page is left out.
3. **Absolute cases.** One :class:`~threetears.evals.kernel.AuthoredJudgeCase` per page and rubric item, on the
   gate's own criterion (``writer_scoring_rubric()[i]``, verbatim), its verdict the Opus score mapped 2 → 5, 1 → 3,
   0 → 1, its note Opus's reason.
4. **Pair cases.** Every two rounds of one recording, in both orders, on ``preference_rubric()``. The verdict comes
   from the Opus totals, Δ = total(B) − total(A): +2 or more → 5, +1 → 4, 0 → 3, −1 → 2, −2 or less → 1.
5. **Freeze** both into one judge template, as the case sets ``writer-opus-abs`` and ``writer-opus-pair``.
6. **Judge arms.** One judge-kind run per judge over each set, read per criterion through ``judge_kind_readings``.

The labels are an agent's (owner ruling R3). The readout calls κ "agreement with the Opus editor", and this script
never records a judge profile from it: no campaign's judge tier reads ``calibrated`` from these cases::

    uv run python -m packages.evals.tests.fixtures.report_plans.writer_calibration --dry
    uv run python -m packages.evals.tests.fixtures.report_plans.writer_calibration --cap-usd 10 --out <dir>

``--dry`` runs every step on a scripted judge that calls no model; its scores carry no signal. The live run needs
an owner's approval. It reads ``OPENROUTER_API_KEY`` for a judge named ``vendor/model`` and ``ANTHROPIC_API_KEY`` for a
bare Claude id. Each judge arm runs under an equal share of ``--cap-usd``, enforced on its landed spend.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import itertools
import json
import os
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.report_plans.analyst_cli import RECORDING, REPLAY
from packages.evals.tests.fixtures.report_plans.writer_smoke import JUDGE as SMOKE_JUDGE
from packages.evals.tests.fixtures.report_plans.writer_smoke import WRITER as SMOKE_WRITER
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT, TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.judge import ToyJudgeCompletion
from packages.evals.tests.report_writer_support import RecordedModel
from packages.evals.tests.scripted_table import FakeCellSink
from packages.evals.tests.writer_host_support import CALLER, WRITER_TEMPLATE, writer_host
from threetears.evals.actions.report_writing import report_writer_kind
from threetears.evals.analysis import judge_kind_readings
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.judge_kind_readings import JudgeKindReadings
from threetears.evals.kernel import (
    JUDGE_KIND,
    AuthoredJudgeCase,
    JudgeKindOverlays,
    VariantConfig,
    withhold_failure_detail,
)
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops.report_writer.preference import pair_evidence, preference_rubric
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.ops.report_writer.writer_contract import (
    EDITOR_ROUNDS_MEASURE,
    MUST_FIX_OPEN_MEASURE,
    SHOULD_FIX_OPEN_MEASURE,
    WRITER_KIND,
    WriterOverlays,
    writer_scoring_rubric,
)
from threetears.evals.ops.report_writer.writer_kind import WriterCase, writer_case_payload, writer_subject_components
from threetears.evals.providers import Model
from threetears.evals.run import (
    EvalRunCostCap,
    JudgeCaseFreezeReport,
    BudgetStoppedError,
    RunnerOptions,
    execute_run,
    freeze_judge_cases,
    judge_kind,
)
from threetears.evals.schema import EvalResult, EvalTemplate, EvalTestCase, JudgeEvidence, RubricDim
from threetears.evals.schema.subject import SubjectSnapshot
from threetears.evals.testing import RecordedTurns

#: Where the recorded runs live, a directory per run.
REPORT_DOCS: Final = Path(__file__).parents[1] / "report_docs"

#: The prototypes whose runs are replayed: prototype 5's three rounds each, prototype 4's final pages and drafts.
PROTOTYPES: Final = ("prototype5", "prototype4")

#: The two case sets this script freezes.
ABSOLUTE_SET: Final = "writer-opus-abs"
PAIR_SET: Final = "writer-opus-pair"

#: The editor's item score as the gate's 1-5 scale states it (``writer_scoring_rubric``'s guide).
EXPECTED: Final = {2: 5, 1: 3, 0: 1}

#: The judges a live run measures: the smoke's flash judge, and Haiku (the smoke's writer) as the cheap Claude reference.
LIVE_JUDGES: Final = (SMOKE_JUDGE, SMOKE_WRITER)

#: The dry judge's model name: a scripted reader that calls nothing.
DRY_JUDGE: Final = "dry/scripted-judge"

#: The judge template the cases are frozen into.
JUDGE_TEMPLATE: Final = EvalTemplate(
    id="writer-opus-calibration",
    scope_id=TOYHOST_SCOPE,
    name="Report writer judge, against the Opus editor",
    intent="Score a report page on the editor rubric as the Opus editor of its round scored it.",
    candidate_kind=JUDGE_KIND,
    created_at=TOYHOST_INSTANT,
    updated_at=TOYHOST_INSTANT,
)


@dataclass(frozen=True)
class Page:
    """One round's page of one recorded run, as the gate's judge reads it, beside the Opus review of that round.

    :ivar run: The recorded run, as ``<prototype>/<directory>``.
    :ivar round: The editor round whose page this is, from 1.
    :ivar evidence: The replay's judge evidence: the page and the facts code computed.
    :ivar measures: The replay's own measures, which say which round it ended on.
    :ivar review: The Opus review of this round, from ``generation.json``.
    :ivar editor: The digest of the ``editor.md`` the review was made under.
    :ivar model: The model the recorded editor ran on.
    """

    run: str
    round: int
    evidence: JudgeEvidence
    measures: dict[str, bool | float | str]
    review: dict[str, Any]
    editor: str
    model: str

    @property
    def author(self) -> str:
        """Who the page's labels are by: the Opus editor, under its own ``editor.md``."""
        return f"agent:{self.model} report editor @{self.editor}"

    @property
    def total(self) -> int:
        """The review's total over the seven items, out of 14."""
        return sum(int(item["score"]) for item in self.review["items"])


@dataclass(frozen=True)
class LeftOut:
    """A round whose page is not in the set, and why."""

    run: str
    round: int
    why: str


def recordings() -> list[Path]:
    """Every recorded run of :data:`PROTOTYPES`, in order."""
    return [path.parent for name in PROTOTYPES for path in sorted((REPORT_DOCS / name).glob(f"*/{REPLAY}"))]


def _counts(findings: Sequence[dict[str, Any]]) -> tuple[float, float]:
    return (
        float(sum(1 for f in findings if f["severity"] == "must_fix")),
        float(sum(1 for f in findings if f["severity"] == "should_fix")),
    )


class _NoSpans:
    """A cell's tracing windows when a replay is driven alone: both open nothing."""

    def identity(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    collecting = identity


async def replay_round(case_dir: Path, round_: int) -> Page | LeftOut:
    """Replay one recording through the writer kind until its ``round_``'s page, and pair it with that round's review.

    :param case_dir: The recorded run's directory.
    :param round_: The round whose page to end on, from 1: the kind is given ``round_ - 1`` revisions.
    :return: The page, or why it is left out.
    """
    run = f"{case_dir.parent.name}/{case_dir.name}"
    made: dict[str, Any] = json.loads((case_dir / REPLAY).read_text())
    generation: dict[str, Any] = json.loads((case_dir / "generation.json").read_text())
    (review,) = [review for review in generation["reviews"] if review["round"] == round_]
    recording = RecordedTurns.replay(case_dir / RECORDING)
    fixture = await asyncio.to_thread(writer_host)
    name = KAIRO if made["snapshot_of"] == "kairo" else made["snapshot_of"]
    case = EvalTestCase(
        id=f"{case_dir.name}-r{round_}",
        template_id=WRITER_TEMPLATE.id,
        scope_id=CALLER.scope_id,
        host_payload=writer_case_payload(
            WriterCase.of(fixture_snapshots(CALLER.scope_id)[name], withhold=made["withhold"])
        ),
    )
    skill = load_skill()
    kind = report_writer_kind(
        fixture.host,
        CALLER,
        overlays=WriterOverlays(revisions=round_ - 1),
        models=lambda _effort: RecordedModel(recording, made["model"]),
        budget=ReportBudget(usd=made["budget_usd"]),
    )
    prepared = await kind.prepare(
        subject_snapshot=SubjectSnapshot(
            subject_id="the report writer",
            subject_label="the report writer",
            components=writer_subject_components(skill),
            state={},
        ),
        variant_config=VariantConfig(candidate_model=made["model"]),
        world_seed=None,
        span_window=_NoSpans(),
        cassettes=None,
        world=None,
    )
    output = await kind.invoke(prepared, case, FakeCellSink())
    measures = dict(output.host_measures)
    if output.judge_evidence is None:
        # The loop's own reason names the session and turn a recording ran out at; the errors only its type.
        reason = (output.kind_payload or {}).get("reason") or "; ".join(
            [*output.candidate_errors, *output.infra_errors]
        )
        return LeftOut(run, round_, f"the replay stored no page: {reason or 'no reason recorded'}")
    if measures.get(EDITOR_ROUNDS_MEASURE) != float(round_):
        return LeftOut(run, round_, f"the replay ended on editor round {measures.get(EDITOR_ROUNDS_MEASURE)}")
    opened = (measures.get(MUST_FIX_OPEN_MEASURE), measures.get(SHOULD_FIX_OPEN_MEASURE))
    if opened != _counts(review["findings"]):
        return LeftOut(run, round_, f"the replay's open findings {opened} are not round {round_}'s review's")
    return Page(
        run, round_, output.judge_evidence, measures, review, generation["skill"]["editor"], generation["model"]
    )


async def pages() -> tuple[list[Page], list[LeftOut]]:
    """Every round's page of every recording, and every round left out with why.

    :return: The pages in recording and round order, and the rounds left out.
    """
    found: list[Page] = []
    left: list[LeftOut] = []
    for case_dir in recordings():
        generation = json.loads((case_dir / "generation.json").read_text())
        for review in generation["reviews"]:
            page = await replay_round(case_dir, int(review["round"]))
            if isinstance(page, Page):
                found.append(page)
            else:
                left.append(page)
    return found, left


def absolute_cases(found: Sequence[Page]) -> list[AuthoredJudgeCase]:
    """One case per page and rubric item: the gate's criterion, the Opus score as its verdict, Opus's reason as its note.

    :param found: The pages.
    :return: The cases, page by page, in the rubric's order.
    """
    rubric = writer_scoring_rubric()
    cases: list[AuthoredJudgeCase] = []
    for page in found:
        items = sorted(page.review["items"], key=lambda item: int(item["n"]))
        if len(items) != len(rubric):
            raise ValueError(f"{page.run} round {page.round}'s review scores {len(items)} items, not {len(rubric)}")
        for dim, item in zip(rubric, items, strict=True):
            if dim.name != f"{WRITER_KIND}.{str(item['name']).lower().replace(' ', '_')}":
                raise ValueError(
                    f"{page.run} round {page.round}'s item {item['n']} is {item['name']!r}, not {dim.name}"
                )
            cases.append(
                AuthoredJudgeCase(
                    criterion=dim,
                    output=page.evidence.artifact,
                    case_material=page.evidence.case_material,
                    intent=WRITER_TEMPLATE.intent,
                    expected=EXPECTED[int(item["score"])],
                    author=page.author,
                    note=item["reason"],
                )
            )
    return cases


def preferred(delta: int) -> int:
    """The pair verdict for Δ = total(B) − total(A): 5 when B leads by 2 or more, 1 when A does, 3 on a tie."""
    return 3 + max(-2, min(2, delta))


@dataclass(frozen=True)
class Pair:
    """Two rounds of one recording, ``a`` read first."""

    a: Page
    b: Page

    @property
    def expected(self) -> int:
        return preferred(self.b.total - self.a.total)

    def case(self, criterion: RubricDim) -> AuthoredJudgeCase:
        evidence = pair_evidence(self.a.evidence, self.b.evidence)
        return AuthoredJudgeCase(
            criterion=criterion,
            output=evidence.artifact,
            case_material=evidence.case_material,
            intent=WRITER_TEMPLATE.intent,
            expected=self.expected,
            author=self.a.author,
            note=(
                f"{self.a.run}: Report A is round {self.a.round} ({self.a.total} of 14 from the Opus editor), Report B "
                f"is round {self.b.round} ({self.b.total} of 14)"
            ),
        )


def pairs(found: Sequence[Page]) -> list[Pair]:
    """Every two rounds of one recording, in both orders: the earlier first, then swapped.

    :param found: The pages.
    :return: The pairs.
    """
    by_run: dict[str, list[Page]] = {}
    for page in found:
        by_run.setdefault(page.run, []).append(page)
    ordered: list[Pair] = []
    for run_pages in by_run.values():
        for a, b in itertools.combinations(sorted(run_pages, key=lambda page: page.round), 2):
            ordered.extend((Pair(a, b), Pair(b, a)))
    return ordered


def pair_cases(found: Sequence[Page]) -> list[AuthoredJudgeCase]:
    """The pair cases on ``preference_rubric()``: each pair of rounds in both orders, the swapped one mirrored."""
    criterion = preference_rubric()
    return [pair.case(criterion) for pair in pairs(found)]


def freeze(host: EvalHost, found: Sequence[Page]) -> dict[str, JudgeCaseFreezeReport]:
    """Freeze both case sets into :data:`JUDGE_TEMPLATE` on ``host``'s store; synchronous, a store call.

    :param host: The judge host.
    :param found: The pages.
    :return: Each case set's freeze, by name.
    """
    host.storage.save_template(JUDGE_TEMPLATE)
    return {
        name: freeze_judge_cases(
            host.storage, template=JUDGE_TEMPLATE, scope_id=TOYHOST_SCOPE, authored=cases, case_set=name
        )
        for name, cases in ((ABSOLUTE_SET, absolute_cases(found)), (PAIR_SET, pair_cases(found)))
    }


#: Where the built-in prompt names the one key its reply maps.
_KEY: Final = re.compile(r'single key "([^"]+)"')


# parity-with: threetears.evals.schema.CompletionClient
class FakeDryJudge:
    """A judge that reads nothing: its score is a digest of the prompt, so the readout runs and measures no signal."""

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, str] | None = None
    ) -> ToyJudgeCompletion:
        match = _KEY.search(system)
        if match is None:
            raise ValueError("the built-in judge prompt names no score key")
        score = 1 + int(hashlib.sha256(user.encode()).hexdigest(), 16) % 5
        reply = {"reasoning": "dry: a digest of the prompt, not a reading", "criteria_scores": {match.group(1): score}}
        return ToyJudgeCompletion(
            content=json.dumps(reply),
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            model=DRY_JUDGE,
            price_source="dry",
            served_model=DRY_JUDGE,
        )

    async def aclose(self) -> None:
        return None


#: Each provider's key, by the environment variable it is read from.
_KEYS: Final = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def _provider(model: str) -> str:
    """Where a live judge is sent: OpenRouter for a ``vendor/model`` id, Anthropic's own API for a bare one."""
    return "openrouter" if "/" in model else "anthropic"


def _live_client(keys: dict[str, str]) -> Any:
    def clients(role: str, model: str | None, *, temperature: float | None = None) -> Model:
        if role != "judge" or model is None:
            raise SystemExit(f"writer_calibration.py: a judge arm builds no {role!r} client")
        provider = _provider(model)
        return Model(model, provider=provider, api_key=keys[_KEYS[provider]])

    return clients


def _keys(judges: Sequence[str]) -> dict[str, str]:
    wanted = {_KEYS[_provider(judge)] for judge in judges}
    keys = {name: os.environ.get(name, "") for name in sorted(wanted)}
    if missing := [name for name, value in keys.items() if not value]:
        raise SystemExit(f"writer_calibration.py: a live run needs {', '.join(missing)}; --dry needs none")
    return keys


#: What one call of each set reads, against an absolute call: a pair call reads two pages.
CALL_WEIGHT: Final = {ABSOLUTE_SET: 1, PAIR_SET: 2}


@dataclass(frozen=True)
class Arm:
    """One judge's run over one case set: the cases, the trials it landed, and why it stopped early, if it did."""

    judge: str
    case_set: str
    cases: int
    trials: int
    stopped: str | None


def arm_caps(frozen: dict[str, JudgeCaseFreezeReport], judges: Sequence[str], cap_usd: float) -> dict[str, float]:
    """Each case set's share of one judge's part of ``cap_usd``, by the calls it plans weighted by :data:`CALL_WEIGHT`.

    :param frozen: The freezes, by case set.
    :param judges: The judges, each given an equal part.
    :param cap_usd: The whole run's cap.
    :return: Each case set's cap for one judge's arm over it.
    """
    weights = {name: len(report.cases) * CALL_WEIGHT[name] for name, report in frozen.items()}
    total = sum(weights.values())
    return {name: cap_usd / len(judges) * weight / total for name, weight in weights.items()}


async def judge_arm(
    host: EvalHost, report: JudgeCaseFreezeReport, *, judge: str, k: int, cap_usd: float | None
) -> tuple[list[EvalResult], Arm]:
    """One judge's run over one frozen case set, through the engine's runner; its results and what it ran.

    An arm its cap stops keeps what it landed, and says it stopped.

    :param host: The judge host, the cases frozen on its store.
    :param report: The case set's freeze.
    :param judge: The judge model.
    :param k: Trials per case.
    :param cap_usd: The arm's ceiling on landed spend; ``None`` for a dry arm, which spends nothing.
    """
    assert report.case_set is not None
    case_ids = [case.test_case_id for case in report.cases]
    run = make_eval_run(
        id=f"{report.case_set.name}-v{report.case_set.version}-{re.sub(r'[^a-z0-9]+', '-', judge.lower())}",
        scope_id=TOYHOST_SCOPE,
        template_id=JUDGE_TEMPLATE.id,
        candidate_kind=JUDGE_KIND,
        candidate_model=judge,
        k_runs=k,
        test_case_ids=case_ids,
        overlays=JudgeKindOverlays().model_dump(mode="json"),
    )
    cases = await asyncio.to_thread(host.storage.load_test_cases_by_ids, case_ids, TOYHOST_SCOPE)
    await asyncio.to_thread(host.storage.save_eval_run, run)
    kind = await asyncio.to_thread(
        judge_kind,
        host.storage,
        host.completion_clients("a judge calibration"),
        model=judge,
        scope_id=TOYHOST_SCOPE,
        failure_describer=withhold_failure_detail,
    )
    cap = None if cap_usd is None else EvalRunCostCap(run.id, cap_usd, enabled=True)
    stopped: str | None = None
    async with kind:
        try:
            await execute_run(
                host,
                run=run,
                template=JUDGE_TEMPLATE,
                test_cases=cases,
                judge_service=None,
                options=RunnerOptions(candidate_kinds={JUDGE_KIND: lambda _cell: kind}),
                budget_gate=None if cap is None else cap.check,
                on_cost=None if cap is None else cap.record,
            )
        except BudgetStoppedError as cut:  # the arm's cap: what it landed is kept and read, and the stop is reported
            stopped = str(cut)
    results = await asyncio.to_thread(host.storage.query_eval_results_by_run, run.id, TOYHOST_SCOPE)
    return results, Arm(judge, report.case_set.name, len(case_ids), len(results), stopped)


def agreement_rows(readings: JudgeKindReadings) -> list[dict[str, Any]]:
    """One row per judge and criterion: the cases, the parse rate and the agreement with the Opus editor."""
    rows: list[dict[str, Any]] = []
    for reading in readings.readings:
        agreement = reading.label_agreement
        rows.append(
            {
                "judge": reading.judge_model,
                "criterion": reading.rubric_dim,
                "cases": reading.cases,
                "parse_rate": reading.parse_validity.rate,
                "kappa": None if agreement is None else agreement.kappa,
                "weighted_kappa": None if agreement is None else agreement.weighted_kappa,
                "interval": None if agreement is None else agreement.agreement_interval,
                "self_kappa": None if reading.self_agreement is None else reading.self_agreement.kappa,
            }
        )
    return rows


def _cell(value: Any) -> str:
    if value is None:
        return "undetermined"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, tuple | list):
        return " to ".join(f"{bound:.2f}" for bound in value)
    return str(value)


def readout(rows: Sequence[dict[str, Any]], arms: Sequence[Arm], left: Sequence[LeftOut]) -> Iterator[str]:
    """The printed readout: agreement with the Opus editor per judge and criterion, each arm's trials, then every
    round left out."""
    yield "Agreement with the Opus editor (an agent's labels; no judge profile is recorded from them)."
    yield ""
    yield "| Judge | Criterion | Cases | Parse rate | κ | Weighted κ | Weighted κ bounds | Self κ |"
    yield "|---|---|---|---|---|---|---|---|"
    for row in rows:
        yield "| " + " | ".join(_cell(row[key]) for key in row) + " |"
    yield ""
    yield f"Each pair of rounds is two {PAIR_SET} cases, one per order, so its bounds count each pair twice."
    yield ""
    yield "| Judge | Case set | Cases | Trials | Stopped early |"
    yield "|---|---|---|---|---|"
    for arm in arms:
        yield f"| {arm.judge} | {arm.case_set} | {arm.cases} | {arm.trials} | {arm.stopped or 'no'} |"
    if left:
        yield ""
        yield "| Round left out | Why |"
        yield "|---|---|"
        for out in left:
            yield f"| {out.run} round {out.round} | {out.why} |"


@dataclass(frozen=True)
class Calibration:
    """What one calibration run found: the pages and the rounds left out, both freezes, the judge host and its arms.

    :ivar found: The pages, in recording and round order.
    :ivar left: The rounds left out, with why.
    :ivar host: The judge host the cases are frozen on and the arms ran on.
    :ivar frozen: Each case set's freeze, by name.
    :ivar results: Every judge arm's results.
    :ivar arms: What each judge arm ran.
    :ivar readings: The arms read per judge and criterion.
    """

    found: list[Page]
    left: list[LeftOut]
    host: EvalHost
    frozen: dict[str, JudgeCaseFreezeReport]
    results: list[EvalResult]
    arms: list[Arm]
    readings: JudgeKindReadings

    def summary(self, *, dry: bool) -> dict[str, Any]:
        """What the run found, as JSON: the pages, the rounds left out, the case sets, the arms and the agreement rows."""
        rows = agreement_rows(self.readings)
        return {
            "dry": dry,
            "pages": [{"run": p.run, "round": p.round, "total": p.total, "author": p.author} for p in self.found],
            "left_out": [{"run": out.run, "round": out.round, "why": out.why} for out in self.left],
            "case_sets": {
                name: {"version": report.case_set.version if report.case_set else None, "cases": len(report.cases)}
                for name, report in self.frozen.items()
            },
            "arms": [
                {"judge": a.judge, "case_set": a.case_set, "cases": a.cases, "trials": a.trials, "stopped": a.stopped}
                for a in self.arms
            ],
            "agreement_with_the_opus_editor": rows,
            "readout": "\n".join(readout(rows, self.arms, self.left)),
        }


async def calibration(*, dry: bool, k: int, cap_usd: float, judges: Sequence[str]) -> Calibration:
    """Build the pages, freeze both case sets, and run one judge arm per judge over each.

    :param dry: Run on :class:`FakeDryJudge` instead of ``judges``.
    :param k: Trials per case.
    :param cap_usd: The live run's cap: an equal part per judge, split over the case sets by :func:`arm_caps`.
    :param judges: The live judges.
    :return: What it found.
    """
    found, left = await pages()
    dry_judge = FakeDryJudge()
    arms = (DRY_JUDGE,) if dry else tuple(judges)
    clients = (lambda _role, _model, *, temperature=None: dry_judge) if dry else _live_client(_keys(arms))
    host = await asyncio.to_thread(lambda: toyhost_host(clients=clients))
    frozen = await asyncio.to_thread(freeze, host, found)
    caps = arm_caps(frozen, arms, cap_usd)
    results: list[EvalResult] = []
    ran: list[Arm] = []
    for judge in arms:
        for name, report in frozen.items():
            landed, arm = await judge_arm(host, report, judge=judge, k=k, cap_usd=None if dry else caps[name])
            results.extend(landed)
            ran.append(arm)
    return Calibration(found, left, host, frozen, results, ran, judge_kind_readings(results))


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="writer_calibration.py", description=__doc__.splitlines()[0])
    parser.add_argument("--dry", action="store_true", help="a scripted judge, nothing sent")
    parser.add_argument("--k", type=int, default=2, help="trials per case")
    parser.add_argument("--cap-usd", type=float, default=10.0, help="the owner's cap for every judge arm together")
    parser.add_argument("--judge", action="append", help=f"a live judge; default {', '.join(LIVE_JUDGES)}")
    parser.add_argument("--out", type=Path, help="where the live run's result goes")
    options = parser.parse_args(list(argv))
    if not options.dry and options.out is None:
        parser.error("a live run needs --out")
    if options.out is not None:
        # Made before anything is sent, so a directory already there refuses the run rather than its result.
        options.out.mkdir(parents=True, exist_ok=False)
    found = asyncio.run(
        calibration(dry=options.dry, k=options.k, cap_usd=options.cap_usd, judges=options.judge or LIVE_JUDGES)
    ).summary(dry=options.dry)
    if options.out is not None:
        (options.out / "calibration.json").write_text(json.dumps(found, indent=2, default=str), encoding="utf-8")
    print(found["readout"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
