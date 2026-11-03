"""``WriterEval``: the report writer measured as the thing under test, over portable cases, from recordings.

A writer eval is the fifth rung of the ladder: the eval under test is the report writer itself (the analyst and its
editor), and each arm is a :class:`Writer`, a model with a skill and an effort. ``await WriterEval(...).run()``:

1. **reads the case files** in ``cases_dir`` (:func:`~threetears.evals.harness.writer_cases.read_writer_cases`), and
   refuses a case that withholds a note some arm's skill does not hold (below);
2. **opens the store** (``.evals/evals.sqlite`` unless told) and builds the writer host over it
   (:func:`~threetears.evals.harness.writer_host.writer_ops_host`), every arm's model routed to its recordings
   (:func:`~threetears.evals.harness.writer_turns.writer_turns`);
3. **creates the writer template** in the eval's scope, or reuses the one there, and **restores and mints every case**
   into it with its stratum, each landing on its split group's side at ``held_out_fraction``;
4. **launches every arm on one side's cases** (``side``, the held-out side by default), each naming its equal share of
   ``max_cost_usd``. A replayed writer prices nothing, and the engine admits an unpriced arm only because its launch
   names that share, so the share is the ceiling every arm runs under;
5. **files the campaign**: the control is the first arm, one design axis per lever the arms differ on (the skill, the
   effort, the model), and the declared ``margin``;
6. **reads the promotion** of each candidate against the control
   (:func:`~threetears.evals.ops.report_writer.writer_promotion.writer_promotion`), on the held-out side only: the dev
   side is where a change is tuned, so the read refuses it by design, and a dev run says so instead of reading it;
7. returns a :class:`WriterResult`, which prints the campaign's readings as a table and writes the campaign's page.

**The job budget sets how many cases an arm may run.** A writer cell's deadline is a whole report: every turn its
budget allows at the request ceiling, and its page's judging after. The engine gives an arm's job a budget sized as
if its cells ran one after another and clamps it at its ceiling, and the host refuses an arm whose cells, two at a
time, could outlast it. Without a turn cap a report's turns are bounded only by its dollars, so its deadline runs
past any job, which is why ``turns`` defaults to 40. The most cases one arm may run:

=========  ===========  ===========
request_s  turns = 40   no turn cap
=========  ===========  ===========
60 s       16           2
120 s      8            0
180 s      4            0
=========  ===========  ===========

**A recorded arm runs each case once.** Each case's sessions replay from one file per arm
(``<recordings>/<arm>-<skill12>-<model>-<effort>-r<revisions>/<case>-<snapshot12>.jsonl``), with no key; a missing file is recorded when
``ANTHROPIC_API_KEY`` is set and refused before anything runs when it is not. After the arms have run, each session a
replay left recorded lines of is printed with the result: the code now stops sooner than it did when it was recorded.

**Withheld notes are checked against every arm's skill.** A case withholds the exemplar note reasoned on its own
campaign, and each cell takes that note out of its own arm's skill (a frozen copy of the control's included), so a case
withholding a note some arm's skill lacks could never run on that arm. It is refused here, before anything is minted.
The mint itself checks the notes against the skill this build ships, for the bank, which outlives this eval.

**A case's example results stay where it was frozen.** A snapshot frozen from a campaign names the stored result
behind each example it shows. The writer's tools read the snapshot alone, and none of them fetches a stored result,
so nothing is missing from what the writer reads; but a page that quotes such an id names a result this store does not
hold, and the result says how many cases name one.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from threetears.evals.analysis import (
    CampaignReport,
    EvalSummary,
    GateResult,
    campaign_report,
    campaign_report_markdown,
    gate_verdicts,
    summarize_run,
)
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.kernel.eval_file import beside_eval_file
from threetears.evals.kernel.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.text_block import one_line
from threetears.evals.ops import OpsHost
from threetears.evals.ops.analysis import CampaignDefinition, campaign_create
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.ops.report_writer.writer_kind import WriterSide
from threetears.evals.ops.report_writer.writer_bank import (
    MintedWriterCase,
    WriterSplit,
    WriterTemplateCreate,
    writer_template_create,
)
from threetears.evals.ops.report_writer.writer_contract import (
    SKILL_LEVER,
    WRITER_KIND,
    EffortWord,
    WriterModelFactory,
    WriterModels,
)
from threetears.evals.ops.report_writer.loop import MOST_REVISIONS, TurnModel
from threetears.evals.ops.report_writer.writer_promotion import writer_promotion
from threetears.evals.quick.contract import open_store
from threetears.evals.schema import DocumentStore
from threetears.evals.quick.result import CREATED_BY
from threetears.evals.harness.writer_cases import WriterCaseFile, read_writer_cases
from threetears.evals.harness.writer_host import WriterJudge, writer_ops_host
from threetears.evals.harness.writer_turns import KEY_VARIABLE, WriterTurns, writer_turns
from threetears.evals.run import quote_launch, start_run
from threetears.evals.schema.models import DECLARED_NAME_CHARS, CaseSetRef

#: The effort lever's id, as the writer's overlays name it.
_EFFORT_LEVER = f"{WRITER_KIND}.reasoning_effort"

#: The revisions lever's id, as the writer's overlays name it.
_REVISIONS_LEVER = f"{WRITER_KIND}.revisions"

#: The efforts a writer may name.
_EFFORTS: frozenset[str] = frozenset(get_args(EffortWord))

#: A side in words.
_SIDE_WORDS: Mapping[WriterSide, str] = {"held_out": "held-out", "dev": "dev"}

#: The writer template every writer eval's cases are minted into, in its scope.
_TEMPLATE_NAME = "report writer"


@dataclass(frozen=True)
class Writer:
    """One arm of a writer eval: the model that writes, the skill it starts from, and its reasoning effort.

    :ivar model_id: The writer's model, by the id its provider takes (``claude-sonnet-5-5``).
    :ivar skill: The analyst skill both its sessions start from: a frozen copy (``AnalystSkill.at(path)``) or a
        variant of one (``.with_file(name, text)``); ``None`` for the skill this build ships.
    :ivar effort: The reasoning effort every turn is sent at (``low``, ``medium``, ``high`` ...); ``None`` for the
        model's own.
    :ivar revisions: The revisions the analyst is given after a review that does not clear its report.
    """

    model_id: str
    skill: AnalystSkill | None = None
    effort: str | None = None
    revisions: int = MOST_REVISIONS

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("a Writer names its model by id, and this one is blank")
        if self.effort is not None and self.effort not in _EFFORTS:
            raise ValueError(f"effort {self.effort!r} is none of {', '.join(get_args(EffortWord))}")
        if not 0 <= self.revisions <= MOST_REVISIONS:
            raise ValueError(f"a Writer's revisions are 0 to {MOST_REVISIONS}, not {self.revisions}")

    @property
    def analyst_skill(self) -> AnalystSkill:
        """The skill the arm runs: its own, or the one this build ships."""
        return self.skill if self.skill is not None else load_skill()


@dataclass(frozen=True)
class WriterResult:
    """What a writer eval ran: each arm's run, the campaign holding them, its page, and the promotion read.

    ``print(result)`` is the quiet summary (:meth:`render`); :meth:`write` writes the campaign's page.

    :ivar name: The eval's name, which titles its campaign.
    :ivar scope_id: The scope the cases, runs and campaign are stored in.
    :ivar side: The side of the split the arms ran on.
    :ivar control: The arm every other arm is read against: the first.
    :ivar arms: Each arm's run summary, by arm name, in the order the arms were given.
    :ivar host: The writer host the runs are stored through, whose measure descriptors name the readings.
    :ivar campaign_id: The campaign holding every arm's run.
    :ivar page: The campaign's page, as :func:`~threetears.evals.analysis.campaign_report` reads it.
    :ivar judge: The scoring judge's model.
    :ivar strata: How many of the side's cases each stratum holds, ``none`` for a case naming none.
    :ivar decisions: Each candidate's promotion decision (``promote``, ``keep`` or ``undecided``); empty on the dev
        side, where no promotion is read.
    :ivar promotion_why: Each candidate's decision in one sentence.
    :ivar gates: Each candidate's verdicts, gated as the promotion read gates them: what a failing exit code reads.
    :ivar store: Where the runs are stored, in words.
    :ivar disclosures: What the run must say about how it was run: replays, recordings, drift, results held elsewhere.
    """

    name: str
    scope_id: str
    side: WriterSide
    control: str
    arms: dict[str, EvalSummary]
    host: EvalHost = field(repr=False)
    campaign_id: str
    page: CampaignReport = field(repr=False)
    judge: str
    strata: dict[str, int]
    decisions: dict[str, str]
    promotion_why: dict[str, str]
    gates: dict[str, GateResult] = field(repr=False)
    store: str = ""
    disclosures: tuple[str, ...] = ()

    def __str__(self) -> str:
        """The quiet summary (:meth:`render`)."""
        return self.render()

    def render(self, *, full: bool = False) -> str:
        """The result as text: the readings table, each arm's results and spend, the promotion read and disclosures.

        Each reading is a measure the writer host declares, named as its descriptor names it, then each judged
        dimension of the template's rubric. ``full`` adds every arm's whole summary and the campaign's page.

        :param full: Every arm's whole summary and the whole page, as Markdown, after the quiet summary.
        :ptype full: bool
        :return: The text.
        :rtype: str
        """
        candidates = " and ".join(arm for arm in self.arms if arm != self.control)
        cases = next(iter(self.arms.values())).n_cases
        lines = [
            f"{self.name}: {candidates} against {self.control}, {cases} {_SIDE_WORDS[self.side]} case(s) x 1, "
            f"judged by {self.judge}"
        ]
        lines += self._readings()
        lines += ["| stratum | cases |", "|---|---|", *(f"| {name} | {n} |" for name, n in self.strata.items())]
        for arm, summary in self.arms.items():
            label = f"{arm}{' (control)' if arm == self.control else ''}"
            lines.append(f"{label}: {summary.results_text}")
            lines += [f"  {one_line(text)}" for text in (summary.stopped_text, summary.total_spend_text) if text]
            lines += [f"  error: {one_line(error)}" for error in summary.errors]
        if self.decisions:
            lines += [f"promotion of {arm}: {self.promotion_why[arm]}" for arm in self.decisions]
        else:
            lines.append("promotion: not read on the dev side, where a change is tuned; launch side='held_out'")
        lines += self.disclosures
        lines.append(f"stored in {self.store}, scope {self.scope_id!r}")
        text = "\n".join(lines)
        if not full:
            return text
        return "\n\n".join(
            [text, *(summary.render() for summary in self.arms.values()), campaign_report_markdown(self.page)]
        )

    def _readings(self) -> list[str]:
        """The readings table: a row per measure some arm carries, then per judged dimension, a column per arm."""
        measures = self.host.profile.measures
        rows: dict[str, dict[str, float]] = {}
        for arm, summary in self.arms.items():
            for measure in summary.measures:
                if measure.mean is not None:
                    descriptor = measures.get(measure.name)
                    label = (
                        descriptor.reader_name if descriptor is not None and descriptor.reader_name else measure.name
                    )
                    rows.setdefault(label, {})[arm] = measure.mean
            for dimension in summary.judged:
                if dimension.mean is not None:
                    rows.setdefault(f"{dimension.name} (judged)", {})[arm] = dimension.mean
        head = ["| reading | " + " | ".join(self.arms) + " |", "|---|" + "---|" * len(self.arms)]
        body = [
            f"| {label} | " + " | ".join(f"{row[arm]:.2f}" if arm in row else "-" for arm in self.arms) + " |"
            for label, row in rows.items()
        ]
        return [*head, *body] if body else ["no reading: no arm's results carry a measure"]

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        """Write the campaign's page as one self-contained file, its Markdown, its JSON and its aspect's bundle.

        :param out_dir: The directory; created when missing.
        :ptype out_dir: Path | str
        :return: Each file written, by role (``html``, ``markdown``, ``json``, ``bundle:<aspect>``).
        :rtype: dict[str, Path]
        """
        return self.page.write(out_dir)

    def exit_code(self) -> int:
        """What a command line exits with: 0, 1 when an arm's run did not complete, 4 when a promotion gate failed."""
        if any(summary.status != "completed" for summary in self.arms.values()):
            return 1
        return 4 if any(gate.failed for gate in self.gates.values()) else 0

    def why_failed(self) -> str | None:
        """Why :meth:`exit_code` is 4, in one line and the page's words; ``None`` when no gate failed.

        :return: ``lead-first's promotion gate failed on ...``, one clause per candidate whose gate failed.
        :rtype: str | None
        """
        failed = [f"{arm}'s promotion gate {gate.why()}" for arm, gate in self.gates.items() if gate.failed]
        return "; ".join(failed) or None


class WriterEval:
    """The report writer, measured: arms of :class:`Writer` over portable writer cases, replayed from recordings.

    ``await writer_eval.run()`` returns a :class:`WriterResult`; ``python -m threetears.evals FILE`` runs every
    writer eval a file defines, as it runs an ``Eval``. What a run does, and why its defaults are what they are: the
    module docstring.
    """

    def __init__(
        self,
        cases_dir: str | os.PathLike[str],
        *,
        arms: Mapping[str, Writer],
        judge: WriterJudge,
        recordings: str | os.PathLike[str],
        budget_usd: float,
        max_cost_usd: float,
        held_out_fraction: float = 0.35,
        side: WriterSide = "held_out",
        name: str = "writer",
        turns: int = 40,
        request_s: float = 120.0,
        margin: float = 5.0,
        writer_provider: str = "anthropic",
        store: DocumentStore | str | os.PathLike[str] | None = None,
        scope: str | None = None,
    ) -> None:
        """Declare the eval; nothing is read or run until :meth:`run`.

        :param cases_dir: The directory of writer case files (``*.json.gz``); relative, it is read beside the eval file
            being loaded.
        :ptype cases_dir: str | os.PathLike[str]
        :param arms: The arms, by name; the first is the control. At least two, no two on one model, skill and effort.
        :ptype arms: Mapping[str, Writer]
        :param judge: The scoring judge, of another family than the writers, with its recording.
        :ptype judge: WriterJudge
        :param recordings: The directory every arm's turn recordings sit under, read beside the eval file.
        :ptype recordings: str | os.PathLike[str]
        :param budget_usd: Each report's dollar cap.
        :ptype budget_usd: float
        :param max_cost_usd: The most the whole eval may spend, judge included; each arm runs under an equal share.
        :ptype max_cost_usd: float
        :param held_out_fraction: The share of split groups placed on the held-out side when first minted.
        :ptype held_out_fraction: float
        :param side: The side whose cases the arms run on; a promotion is read on ``held_out`` alone.
        :ptype side: WriterSide
        :param name: The eval's name, its scope's unless ``scope`` says, and its campaign's.
        :ptype name: str
        :param turns: Each report's turn cap, which bounds how long a cell can run (the module docstring's table).
        :ptype turns: int
        :param request_s: The ceiling for one request on the writer's client, in seconds.
        :ptype request_s: float
        :param margin: How far, in points of a 0-100 rate, a candidate may fall below the control before it is shown
            worse.
        :ptype margin: float
        :param writer_provider: The provider every writer is built on, which tells a writer's family.
        :ptype writer_provider: str
        :param store: Where the cases and runs are kept: ``None`` for ``.evals/evals.sqlite``, ``":memory:"``, a file
            or a document store.
        :ptype store: DocumentStore | str | os.PathLike[str] | None
        :param scope: The scope; ``None`` is ``name``.
        :ptype scope: str | None
        :raises ValueError: Fewer than two arms, two arms on one model, skill and effort, a name too long for a campaign, or a share, budget,
            turn cap or ceiling that is not above 0.
        """
        if len(arms) < 2:
            raise ValueError("a writer eval compares arms: name a control and at least one candidate")
        conditions: dict[tuple[str, str, str | None], str] = {}
        for arm, writer in arms.items():
            key = (writer.model_id, writer.analyst_skill.digest, writer.effort)
            if (twin := conditions.get(key)) is not None:
                # The launcher hands the model factory the model, the skill and the effort, never the revisions, so
                # two arms alike in those three could not be told apart when a cell's recording is found.
                raise ValueError(
                    f"arms {twin!r} and {arm!r} run one model, skill and effort, which tells no cell's recording apart; "
                    "vary one of the three, or compare their revisions in two evals"
                )
            conditions[key] = arm
        if not 0 < held_out_fraction < 1:
            raise ValueError(f"held_out_fraction is a share strictly between 0 and 1, not {held_out_fraction}")
        if side not in _SIDE_WORDS:
            raise ValueError(f"side is 'held_out' or 'dev', not {side!r}")
        if len(name) > DECLARED_NAME_CHARS:
            raise ValueError(f"a writer eval's name titles its campaign, at most {DECLARED_NAME_CHARS} characters")
        for what, value in (
            ("budget_usd", budget_usd),
            ("max_cost_usd", max_cost_usd),
            ("turns", turns),
            ("request_s", request_s),
        ):
            if not value > 0:
                raise ValueError(f"{what} is above 0, not {value}")
        self.name = name
        self.cases_dir = beside_eval_file(cases_dir).absolute()
        self.recordings = beside_eval_file(recordings).absolute()
        self.arms = dict(arms)
        self.judge = judge
        self.budget = ReportBudget(usd=budget_usd, turns=turns)
        self.max_cost_usd = max_cost_usd
        self.split = WriterSplit(held_out_fraction=held_out_fraction)
        self.side: WriterSide = side
        self.request_s = request_s
        self.margin = margin
        self.writer_provider = writer_provider
        self.store = store
        self.scope_id = name if scope is None else scope

    async def run(self) -> WriterResult:
        """Mint the cases, launch every arm on one side's cases, file the campaign and read the promotion.

        :return: The result.
        :rtype: WriterResult
        :raises ValueError: The case files do not read; a case withholds a note an arm's skill lacks; the side holds
            no case, or its set lists a case the directory no longer holds; a recording is missing with no key to
            make it; or the reports' budgets could pass an arm's share.
        :raises ValidationFailedError: The launch refused an arm (a judge of the writer's family, an arm whose cells
            could outlast its job), or the campaign's declaration.
        """
        cases = read_writer_cases(self.cases_dir)
        _refuse_unheld_notes(cases, self.arms)
        opened, where = open_store(self.store)
        turns: dict[str, WriterTurns] = {}
        host = writer_ops_host(
            opened,
            judge=self.judge,
            models=_routed(self.arms, turns),
            budget=self.budget,
            request_s=lambda _max_tokens: self.request_s,
            max_cost_usd=self.max_cost_usd,
            skills={writer.analyst_skill.digest: writer.analyst_skill for writer in self.arms.values()},
            writer_provider=self.writer_provider,
            max_arms=len(self.arms),
        )
        try:
            return await self._run(host, cases, turns, where)
        finally:
            for routed in turns.values():
                await routed.aclose()
            await host.launch.job_manager.shutdown()

    async def _run(
        self, host: OpsHost, cases: Sequence[WriterCaseFile], turns: dict[str, WriterTurns], where: str
    ) -> WriterResult:
        eval_host = host.eval_host
        template_id = await run_blocking(eval_host.blocking_executor, self._template, host)
        minted = [
            (case, await case.mint(host, template_id, self.scope_id, created_by=CREATED_BY, split=self.split))
            for case in cases
        ]
        on_side = [(case, receipt) for case, receipt in minted if receipt.side == self.side]
        case_set = await run_blocking(eval_host.blocking_executor, self._case_set, host, on_side)
        named = {receipt.snapshot_id: case.name for case, receipt in on_side}
        for arm in self.arms:
            writer = self.arms[arm]
            turns[arm] = writer_turns(
                self.recordings, arm=arm, skill=writer.analyst_skill, cases=named, revisions=writer.revisions
            )
        held_elsewhere = await run_blocking(
            eval_host.blocking_executor, _results_held_elsewhere, host, self.scope_id, on_side
        )
        disclosures = [*self._recording_disclosures(turns, named), *held_elsewhere]
        share = self.max_cost_usd / len(self.arms)
        if self.budget.usd is not None and len(on_side) * self.budget.usd > share:
            raise ValueError(
                f"{len(on_side)} report(s) at ${self.budget.usd:.2f} each could pass an arm's ${share:.2f} share of "
                f"max_cost_usd ${self.max_cost_usd:.2f}; raise max_cost_usd or lower budget_usd"
            )
        launches = {arm: self._launch(template_id, case_set, writer, share) for arm, writer in self.arms.items()}
        for arguments in launches.values():  # every arm is quoted, and refused, before any is started
            await quote_launch(host.launch, **arguments)
        runs = {arm: (await start_run(host.launch, **arguments))[0] for arm, arguments in launches.items()}
        await host.launch.job_manager.wait_for([run.id for run in runs.values()])
        for arm, routed in turns.items():
            disclosures += [f"{arm}: {left}" for left in routed.unconsumed()]
        case_names = {receipt.test_case_id: case.name for case, receipt in on_side}
        return await run_blocking(
            eval_host.blocking_executor,
            lambda: self._declared(
                eval_host,
                {arm: run.id for arm, run in runs.items()},
                template_id,
                case_names=case_names,
                strata=Counter(case.stratum or "none" for case, _ in on_side),
                where=where,
                disclosures=tuple(disclosures),
            ),
        )

    def _template(self, host: OpsHost) -> str:
        """The writer template in the scope: the one there, or one created now. Blocking."""
        held = host.eval_host.storage.load_template_by_name(_TEMPLATE_NAME, self.scope_id)
        if held is None:
            create = WriterTemplateCreate(template_name=_TEMPLATE_NAME, intent=f"Write the report ({self.name}).")
            return writer_template_create(host, create, self.scope_id).id
        if held.candidate_kind != WRITER_KIND:
            raise ValueError(
                f"scope {self.scope_id!r} holds a template named {_TEMPLATE_NAME!r} of kind {held.candidate_kind!r}; "
                "run the writer eval in another scope (scope=)"
            )
        return held.id

    def _case_set(self, host: OpsHost, on_side: Sequence[tuple[WriterCaseFile, MintedWriterCase]]) -> CaseSetRef:
        """The side's set at its latest version, refused when it lists a case the directory no longer holds. Blocking."""
        side = _SIDE_WORDS[self.side]
        if not on_side:
            raise ValueError(
                f"none of the cases in {self.cases_dir} sits on the {side} side at held_out_fraction "
                f"{self.split.held_out_fraction}; add cases, or move the share"
            )
        versions = host.eval_host.storage.query_case_sets(self.scope_id, name=self.split.set_of(self.side))
        latest = versions[0]
        if extra := sorted(set(latest.test_case_ids) - {receipt.test_case_id for _, receipt in on_side}):
            raise ValueError(
                f"the {side} set of scope {self.scope_id!r} lists {len(extra)} case(s) no file in {self.cases_dir} "
                f"holds ({', '.join(extra)}); retire them (writer_case_archive), or run in a fresh scope (scope=)"
            )
        return latest.ref

    def _recording_disclosures(self, turns: Mapping[str, WriterTurns], named: Mapping[str, str]) -> list[str]:
        """Where each arm's writer comes from: replayed, or recorded now; refused when a file is missing and no key is."""
        said: list[str] = []
        recording = bool(os.environ.get(KEY_VARIABLE))
        for arm, routed in turns.items():
            writer = self.arms[arm]
            paths = [routed.path(writer.model_id, _effort(writer), snapshot) for snapshot in named]
            missing = [path for path in paths if not path.is_file()]
            if missing and not recording:
                raise ValueError(
                    f"arm {arm!r} has no turn recording for {len(missing)} of {len(paths)} case(s) ({missing[0]} first), "
                    f"and {KEY_VARIABLE} is not set to record them; restore the files, or set it to record them once"
                )
            folder = routed.directory(writer.model_id, _effort(writer))
            done = len(paths) - len(missing)
            said.append(
                f"{arm}: writer turns replayed from {folder}"
                if not missing
                else f"{arm}: writer turns of {len(missing)} case(s) recorded to {folder}, {done} replayed"
            )
        return said

    def _launch(
        self, template_id: str, case_set: CaseSetRef, writer: Writer, share: float
    ) -> dict[str, Any]:  # Any: the launch's keyword arguments, each of its own type
        """One arm's launch: its model, its levers, the side's set and its share of the ceiling."""
        return {
            "max_cost_usd": share,
            "template_id": template_id,
            "scope_id": self.scope_id,
            "subject_id": self.name,
            "models": [writer.model_id],
            "k_runs": 1,
            "case_set": case_set,
            "judge_model": self.judge.model_name,
            "overlays": {
                "skill": writer.analyst_skill.digest,
                "reasoning_effort": writer.effort,
                "revisions": writer.revisions,
            },
        }

    def _declared(
        self,
        host: EvalHost,
        run_ids: Mapping[str, str],
        template_id: str,
        *,
        case_names: Mapping[str, str],
        strata: Mapping[str, int],
        where: str,
        disclosures: tuple[str, ...],
    ) -> WriterResult:
        """File the campaign, read each candidate's promotion, and summarise each arm. Blocking."""
        control, *candidates = self.arms
        levels = {arm: _levels(writer) for arm, writer in self.arms.items()}
        axes = []
        for lever in (SKILL_LEVER, _EFFORT_LEVER, _REVISIONS_LEVER, CANDIDATE_MODEL_LEVER):
            values = dict.fromkeys(levels[arm][lever] for arm in self.arms)
            if len(values) > 1:
                shown = {value: next(arm for arm in self.arms if levels[arm][lever] == value) for value in values}
                axes.append({"axis_id": lever, "values": [{"content": v, "display": shown[v]} for v in values]})
        line = campaign_create(
            host,
            CampaignDefinition(
                name=self.name,
                subject_id=self.name,
                behavior="report-writing",
                question=f"Does {' or '.join(candidates)} write reports better than {control}?",
                arms=[
                    {"label": arm, "role": "control" if arm == control else "candidate", "from_run_ids": [run_id]}
                    for arm, run_id in run_ids.items()
                ],
                aspects=[
                    {
                        "template_id": template_id,
                        "run_ids": list(run_ids.values()),
                        "margin": self.margin,
                        "design": {
                            "axes": axes,
                            "held_fixed": {"stimulus": "controlled", "apparatus": "commissioned"},
                            "intended_repetitions": 1,
                        },
                    }
                ],
            ),
            self.scope_id,
            created_by=CREATED_BY,
        )
        decisions: dict[str, str] = {}
        why: dict[str, str] = {}
        gates: dict[str, GateResult] = {}
        if self.side == "held_out":
            for arm in candidates:
                read = writer_promotion(host, line.id, self.scope_id, candidate=arm)
                decisions[arm], why[arm], gates[arm] = read.decision, read.why, gate_verdicts(read.verdicts)
        return WriterResult(
            name=self.name,
            scope_id=self.scope_id,
            side=self.side,
            control=control,
            arms={
                arm: summarize_run(host, run_id, self.scope_id, case_names=case_names)
                for arm, run_id in run_ids.items()
            },
            host=host,
            campaign_id=line.id,
            page=campaign_report(host, line.id, self.scope_id),
            judge=self.judge.model_name,
            strata=dict(sorted(strata.items())),
            decisions=decisions,
            promotion_why=why,
            gates=gates,
            store=where,
            disclosures=disclosures,
        )


def _levels(writer: Writer) -> dict[str, str]:
    """A writer's level of each lever a writer eval may vary, as a design axis states it."""
    return {
        SKILL_LEVER: writer.analyst_skill.digest,
        _EFFORT_LEVER: writer.effort or "default",
        _REVISIONS_LEVER: str(writer.revisions),
        CANDIDATE_MODEL_LEVER: writer.model_id,
    }


def _effort(writer: Writer) -> EffortWord | None:
    """A writer's effort as the overlays type it, which its construction checked."""
    return next((word for word in get_args(EffortWord) if word == writer.effort), None)


def _routed(arms: Mapping[str, Writer], turns: Mapping[str, WriterTurns]) -> WriterModelFactory:
    """The host's model factory: each cell's writer, routed to its arm's recordings by model, skill and effort.

    The launcher hands the factory the candidate model and the arm's skill, and the cell's effort after; those three
    are what tell the arms apart (no two arms share all three), so they find the arm whose recordings the cell replays,
    whose own factory refuses any other skill. ``turns`` is filled once the cases are minted, before any arm launches.
    """

    def factory(model: str, skill: AnalystSkill) -> WriterModels:
        by_effort = {
            writer.effort: arm
            for arm, writer in arms.items()
            if writer.model_id == model and writer.analyst_skill.digest == skill.digest
        }
        if not by_effort:
            raise ValueError(f"no arm writes on {model!r} with skill {skill.digest[:12]}")

        def at(effort: EffortWord | None) -> TurnModel:
            arm = by_effort.get(effort)
            if arm is None:
                raise ValueError(f"no arm writes on {model!r} with skill {skill.digest[:12]} at effort {effort!r}")
            return turns[arm](model, skill)(effort)

        return at

    return factory


def _refuse_unheld_notes(cases: Sequence[WriterCaseFile], arms: Mapping[str, Writer]) -> None:
    """Refuse a case withholding a note some arm's skill lacks: no cell of that arm could run it (module docstring)."""
    faults: list[str] = []
    for case in cases:
        for arm, writer in arms.items():
            try:
                writer.analyst_skill.analyst(withhold=case.withhold)
            except ValueError as unknown:
                faults.append(f"case {case.name!r} on arm {arm!r}: {unknown}")
    if faults:
        raise ValueError(
            "a case withholds a note an arm's skill does not hold, so no cell of that arm could run it: "
            + "; ".join(faults)
        )


def _results_held_elsewhere(
    host: OpsHost, scope_id: str, on_side: Sequence[tuple[WriterCaseFile, MintedWriterCase]]
) -> list[str]:
    """The disclosure for cases naming example results the scope does not hold (the module docstring). Blocking."""
    storage = host.eval_host.storage
    naming = [
        case
        for case, _ in on_side
        if any(
            storage.load_eval_result(result_id, scope_id) is None
            for result_id in case.snapshot.example_result_ids.values()
        )
    ]
    if not naming:
        return []
    return [
        f"{len(naming)} case(s) name example results stored only where they were frozen; the writer reads each example "
        "from its snapshot, and a page that names such a result names one this store does not hold"
    ]


__all__ = ["Writer", "WriterEval", "WriterResult"]
