"""``WriterEval``: the report writer measured as the thing under test, over portable cases, from recordings.

A writer eval is the fifth rung of the ladder: the eval under test is the report writer itself (the analyst and its
editor), and each arm is a :class:`Writer`: a model with a skill, an effort, its revisions and, if it names one, its
editor's own model. ``await WriterEval(...).run()``:

1. **reads the case files** in ``cases_dir`` (:func:`~threetears.evals.harness.writer_cases.read_writer_cases`), and
   refuses a case that withholds a note some arm's skill does not hold (below);
2. **opens the store** (``.evals/evals.sqlite`` unless told) and builds the writer host over it
   (:func:`~threetears.evals.harness.writer_host.writer_ops_host`), every arm's model routed to its recordings
   (:func:`~threetears.evals.harness.writer_turns.writer_turns`);
3. **creates the writer template** in the eval's scope, or reuses the one there, and **restores and mints every case**
   into it with its stratum, each landing on its split group's side at ``held_out_fraction``, its withheld notes
   checked against the control arm's skill;
4. **launches every arm on one side's cases** (``side``, the held-out side by default), each naming its part of
   ``max_cost_usd`` (below); under a rule to stop, ``batch_size`` cases at a time (step 6). A replayed writer prices nothing, and the engine admits an unpriced arm only because its
   launch names that part, so the part is the ceiling every arm runs under;
5. **files the campaign**: the control is the first arm, one design axis per lever the arms differ on (the skill, the
   effort, the revisions, the editor's model, the model), the eval's ``question`` and its ``margin``, if it declares one,
   all dated when the arms launched (``create_campaign``'s ``declared_at``): the eval fixed them in the call, before
   any result existed, so a replay of a declared eval never reads as criteria set after its results;
6. **runs the pairwise pass** (:func:`~threetears.evals.ops.report_writer.writer_pairwise.writer_pairwise`) for each
   candidate against the control, on the eval's own host over the same store: the eval's judge reads each case's two
   pages side by side, in both orders, its calls admitted under the candidate's part of the pass's reserve;
   The pass judges under the eval's stopping rule (``stop_when``), declared as the eval runs, before any arm
   launches, over the side's cases (its ``max_cases``): ``separated`` stops the judge at the first case after which
   the preference separates, either way; ``never``, the default, launches and reads every case at once.
   Under ``separated`` the arms launch ``batch_size`` cases at a time, in the bet's order, each batch a run of every
   arm filed into the one campaign; after each batch the pass judges only the new pairs, carrying the answers it has,
   and the arms stop launching once every candidate's preference has separated, or at the last case. The gated
   readings and guardrails are fixed-n tests, read once, at the stop, on the cases launched, and the result and the
   page say how many; a candidate whose preference separated improved while a guardrail is not shown held on them
   goes on to the last case, and its pass says those readings had a second look. At the last case the side-by-side
   reading is the one a single batch of every case reads, number for number, and so are the gated readings: a run that
   stored no page sits in its arm's test setup, its unwritten reports failures in Report finished and absent from the
   judged readings, its pairs read side by side, and said once.
   Either way each look is recorded, the result says what happened in one sentence, and the launch's disclosures say
   the resolution floor: the smallest preference, held on every case, the side's cases can separate
   (:func:`~threetears.evals.ops.report_writer.writer_pairwise.resolution_line`), never a refusal, under the
   ``fewest_cases`` the author asks for or not, with the projection a
   :class:`~threetears.evals.ops.report_writer.writer_pairwise.WriterPrior` run on the same cases made;
7. **reads the promotion** of each candidate against the control
   (:func:`~threetears.evals.ops.report_writer.writer_promotion.writer_promotion`), its pass's completed job as the
   primary reading, on the held-out side only: the dev side is where a change is tuned, so the read refuses it by
   design, and a dev run says so instead of reading it (its pass is still run and printed);
8. returns a :class:`WriterResult`, each arm's summary read over its batches' runs as one
   (:func:`~threetears.evals.analysis.summarize_runs`), which prints the campaign's readings and each pass's as tables and writes the
   campaign's page, led by the answer: each candidate's promotion reading and the judge's side-by-side preference,
   with the preference drawn against even and the side-by-side tables at the head of its Scorecard.

**A report budget that cannot admit even one turn is refused before anything runs; one under two turns is warned.**
The cap holds back the next turn's ceiling against what was actually spent, so the arithmetic proves a budget too small
only when it is under a first turn's ceiling. Each arm's first turn of each role is priced before the store is opened:
the analyst's at the writer's output cap and the arm's skill text (its shortest over the cases, each without the note it
withholds), the editor's at its output cap and the editor's text, on the editor's own model, both at the rates
Anthropic publishes, which a recording run is built on. A first turn's brief and tools only add to that, so each is a
floor of the real ceiling. A ``budget_usd`` under either is refused by name, at $0 and once, with the budget that admits
the turn (:meth:`~threetears.evals.ops.report_writer.tally.ReportBudget.first_turn_refusal`). A budget under
:data:`~threetears.evals.ops.report_writer.tally.MIN_TURNS_FOR_A_REPORT` of the analyst's first-turn ceiling is launched,
with one line among the result's disclosures, on any model, since whether it fits turns on what the turns really
cost; on the model the recorded acceptance runs wrote on, the line cites what they stored at such budgets, and on any
other it says no acceptance run is recorded on it. A model those rates do not list is not checked here; each report's
first turn is then refused, at $0, when its real ceiling is above the budget. After the arms run, an arm whose reports ended over budget with no page says
how many, and what the reports of its run that stored a page needed: what they spent, plus one turn's ceiling.

**The cap is split before anything launches.** ``pairwise_share`` of ``max_cost_usd`` (a fifth unless told) is
reserved for the pairwise passes, in one equal part per candidate; each part is the out-of-run cap its pass's calls are
admitted under. The arms share the rest equally, and each arm's part holds its reports and their absolute judging. The
run is refused before any arm launches when an arm's reports, each at ``budget_usd``, could pass the arm's part, so
the reports always leave the pass its reserve. With two arms, ``max_cost_usd`` $12 and the default share, the pass may
spend $2.40 and each arm $4.80.

**The judge's calls are priced before launch, as far as they can be.** A judge call is priced at the judge's output
cap and a prompt bound read off the pages it shows, which are not yet written; but one call with no page in it, the
judge's ``price_ceiling`` over an empty prompt (its output cap, and the
:data:`~threetears.evals.kernel.capped_completion.PROMPT_FRAMING_TOKENS` every prompt is allowed), is a floor no real
call comes in under. Each arm's cases are judged on every dimension of the rubric, and each pass's pairs in both
orders, each call with :data:`~threetears.evals.run.judge.JUDGE_CALL_ATTEMPTS` attempts, so the run is refused before
any arm launches when either part cannot hold its floor:

==========  ==========================================================================================
part        refused when
==========  ==========================================================================================
an arm's    reports x ``budget_usd`` + cases x dimensions x attempts x floor is above the arm's part
a pass's    cases x 2 orders x attempts x floor is above the pass's part
==========  ==========================================================================================

A judge that cannot price a call yet (its rates unread) is not checked here; a pass its part cannot cover is then
refused with nothing spent once its pages are written, and the result says so and exits 1, as it does for a pass that
stops before its last pair.

Once a page is written its judging is priced in full, page and all, and the engine admits it against what the run's
cap has left before any call is paid for: a case whose judging does not fit is set aside unjudged, never scored low,
and stops its run there (a re-judge takes it up). So the floor decides only what a launch is refused on, and whether a
batch launches; each batch's run runs under all its arm's part has left, so a case is set aside only when the arm
cannot afford its judging, and an arm's judging stays within its part wherever the judge's price ceiling bounds what
it is billed. Its reports are never refused, so its running cases can still take it past its part by what they write.

**The job budget sets how many cases an arm may run.** A writer cell's deadline is a whole report: every turn its
budget allows at the request ceiling, and its page's judging after. The engine gives an arm's job a budget sized as
if its cells ran one after another and clamps it at its ceiling, and the host refuses an arm whose cells, two at a
time, could outlast it. Without a turn cap a report's turns are bounded only by its dollars, so its deadline runs
past any job, which is why ``turns`` defaults to 40 (``TURN_CAP``, the harness door's cap too). The most cases one
arm may run:

=========  ===========  ===========
request_s  turns = 40   no turn cap
=========  ===========  ===========
60 s       16           2
120 s      8            0
180 s      4            0
=========  ===========  ===========

**A recorded arm runs each case once** (``k`` is 1, and refused above it). Each case's sessions replay from one file
per arm (``<recordings>/<arm>-<skill12>-<model>-<effort>-r<revisions>[-e<editor>]/<case>-<snapshot12>.jsonl``, and
the editor's beside it for an arm whose editor runs on a model of its own), with no key; a second repeat of a case
would replay that same file, so it would repeat the first report rather than measure another. A missing file is
recorded when the writer provider's key (``ANTHROPIC_API_KEY`` by default) is set and refused before anything runs
when it is not. After the arms have run, each session a replay left recorded lines of is printed with the result: the
code now stops sooner than it did when it was recorded.

**Withheld notes are checked against every arm's skill.** A case withholds the exemplar note reasoned on its own
campaign, and each cell takes that note out of its own arm's skill (a frozen copy of the control's included), so a case
withholding a note some arm's skill lacks could never run on that arm. It is refused here, before anything is minted.
The mint checks the notes against the control arm's skill, so a frozen copy is checked against itself, never against
the skill this build ships, which may since have dropped or renamed a note.

**A case's example results stay where it was frozen.** A snapshot frozen from a campaign names the stored result
behind each example it shows. The writer's tools read the snapshot alone, and none of them fetches a stored result,
so nothing is missing from what the writer reads; but a page that quotes such an id names a result this store does not
hold, and the result says how many cases name one.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, get_args

from threetears.evals.analysis import (
    CampaignReport,
    EvalSummary,
    GateResult,
    add_runs_to_campaign,
    campaign_report,
    campaign_report_markdown,
    create_campaign,
    gate_verdicts,
    summarize_run,
    summarize_runs,
)
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.analysis.report.blocks import (
    ChartBlock,
    DisclosureBlock,
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
    Verdict,
    table_row,
)
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.report.figures import ContrastPlaces, MeasureRuler, report_rulers
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.number_forms.printed import PreferenceShares
from threetears.evals.analysis.report.number_forms.sets import EVEN_SHARE
from threetears.evals.analysis.report.charts import (
    ChartReference,
    DotIntervalChart,
    DotIntervalPanel,
    DotIntervalRow,
)
from threetears.evals.analysis.report.md_parts import escaped, markdown_table, table_lines
from threetears.evals.analysis.stats import PROJECTION_CASE_CAP, bet_order
from threetears.evals.analysis.style import CANNOT_TELL_APART
from threetears.evals.kernel.capped_completion import PROMPT_FRAMING_TOKENS, prompt_token_bound, refresh_prices
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.eval_file import beside_eval_file
from threetears.evals.kernel.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.kernel.metrics import reads_as_share
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.run_pages import iter_stored_results
from threetears.evals.kernel.usage_capture import maybe_billed_usd
from threetears.evals.kernel.text_block import one_line
from threetears.evals.ops import JobStatus, OpsHost, await_job
from threetears.evals.ops.analysis import CampaignDefinition
from threetears.evals.ops.report_writer.tally import MIN_TURNS_FOR_A_REPORT, ReportBudget
from threetears.evals.ops.report_writer.writer_kind import WriterSide
from threetears.evals.ops.report_writer.writer_bank import (
    MintedWriterCase,
    WriterSplit,
    WriterTemplateCreate,
    writer_template_create,
)
from threetears.evals.ops.report_writer.writer_contract import (
    ENDING_MEASURE,
    MUST_FIX_OPEN_MEASURE,
    SKILL_LEVER,
    WRITER_KIND,
    EffortWord,
    WriterArm,
    WriterModelFactory,
    WriterModels,
)
from threetears.evals.ops.report_writer.loop import MOST_REVISIONS, TurnModel, WriterLimits
from threetears.evals.ops.report_writer.writer_pairwise import (
    PairwiseStopping,
    StoppingRule,
    StopWhen,
    WriterPairwiseReport,
    WriterPrior,
    resolution_line,
    writer_pairwise,
)
from threetears.evals.ops.report_writer.writer_promotion import (
    WriterPromotion,
    held_out_batch_name,
    writer_promotion,
)
from threetears.evals.quick.contract import open_store
from threetears.evals.schema import DocumentStore
from threetears.evals.quick.result import CREATED_BY
from threetears.evals.harness.analyse import TURN_CAP
from threetears.evals.harness.writer_cases import WriterCaseFile, read_writer_cases
from threetears.evals.harness.writer_host import WriterJudge, writer_ops_host
from threetears.evals.harness.writer_turns import WriterTurns, writer_turns
from threetears.evals.run import quote_launch, start_run
from threetears.evals.run.case_sets import mint_case_set
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import DECLARED_NAME_CHARS, CaseSetRef, EvalResult, utc_now_iso
from threetears.evals.schema.turns import TurnCeiling

log = get_logger(__name__)

#: The model the recorded writer acceptance runs wrote on, whose figures :data:`_RECORDED_BUDGETS` states.
_RECORDED_MODEL: Final = "claude-sonnet-5-5"

#: What the recorded writer acceptance runs read at budgets near two first-turn ceilings, in their own words: Sonnet 5.5
#: at medium effort, the shipped skill against its hollow, no-editor and minimal variants
#: (``tests/fixtures/report_docs/writer_acceptance``: 2026-11-03 and 2026-11-04 at $1, 2026-11-06 at $0.75, and the
#: no-editor and minimal runs at $1.50, which stored 50 of 50 and 49 of 50 pages).
_RECORDED_BUDGETS: Final = (
    "in the recorded Sonnet 5.5 acceptance runs, at $0.75 16 of 50 reports stored a page, at $1.00 17 to 22 of 22, and "
    "at $1.50 99 of 100"
)

#: The effort lever's id, as the writer's overlays name it.
_EFFORT_LEVER = f"{WRITER_KIND}.reasoning_effort"

#: The revisions lever's id, as the writer's overlays name it.
_REVISIONS_LEVER = f"{WRITER_KIND}.revisions"

#: The editor's model lever's id, as the writer's overlays name it.
_EDITOR_LEVER = f"{WRITER_KIND}.editor_model"

#: What tells one arm from another: every lever the launcher hands the model factory, and the effort.
_Condition = tuple[str, str, int, str | None, str | None]

#: What a model id may not hold, since it names a recording directory.
_NOT_IN_A_SEGMENT: Final = ("/", "\\", "\0")

#: The efforts a writer may name.
_EFFORTS: frozenset[str] = frozenset(get_args(EffortWord))

#: A side in words.
_SIDE_WORDS: Mapping[WriterSide, str] = {"held_out": "held-out", "dev": "dev"}

#: The writer template every writer eval's cases are minted into, in its scope.
_TEMPLATE_NAME = "report writer"

#: The part of ``max_cost_usd`` a writer eval reserves for its pairwise passes unless told (the module docstring).
PAIRWISE_SHARE: Final = 0.2

#: The cases a writer eval launches at a time under a rule to stop (``batch_size``). A host picks it from its
#: concurrent-cell count and its admitted-runs limit: one batch is one launch of every arm, so a batch the size of the
#: cells a run holds in flight (four on the host this default is set for) is one full wave of each run, where one more
#: case would add a straggler that lengthens the batch by a whole cell for one case of signal, and arms x batch stays
#: within the runs the host admits at once (8 arms x 4 is that host's 32).
_BATCH_SIZE: Final = 4

#: The stopping rules a writer eval may declare.
_STOP_WHEN: frozenset[str] = frozenset(get_args(StopWhen))

#: The preference table's columns, by key and header: a row per candidate and the pairs it reads, every figure on the
#: share scale a reader sees (the two shares sum to 100%, 50% is even); the interval's header carries its level when a
#: pass read one.
_PREFERENCE_COLUMNS: Final = (
    ("candidate", "Candidate"),
    ("pairs", "Pairs"),
    ("reads", "Reads"),
    ("share", "Candidate's share (%)"),
    ("control_share", "Control's share (%)"),
    ("interval", "Interval (%)"),
)

#: The preference figure's axis ticks, in percent: the whole share scale, even at its middle.
_SHARE_TICKS: Final = (0.0, 25.0, 50.0, 75.0, 100.0)

#: The judge's checks table's columns, by key and header: a row per candidate.
_CHECK_COLUMNS: Final = (
    ("candidate", "Candidate"),
    ("order_consistency", "Order consistency"),
    ("first_slot_lean", "First-slot lean (pts)"),
    ("cannot_tell", "Can't tell"),
    ("to_settle", "Held-out cases to settle"),
    ("spent", "Spent"),
)

#: The two sets of pairs a preference row reads: every pair, and those whose two pages were both stored finished.
_EVERY_PAIR: Final = "every pair"
_BOTH_FINISHED: Final = "both pages finished"

#: What the side-by-side tables say once, above them: what a share, its interval, order consistency and the lean are.
_PAIRWISE_LEAD: Final = (
    "Side by side, the judge read each case's two pages in both orders and said how strongly it preferred one: a "
    "page's share is the part of its preference it gave that page, averaged over the cases, so the candidate's and the "
    "control's sum to 100% (50% is even); the interval is on the candidate's share, and the reading over every pair is "
    "the promotion's primary reading; order consistency is the part of cases whose two orders agreed, and first-slot "
    "lean how far the judge favoured the page it read first, in points (-50 to 50)."
)

#: What a promotion decision leads the page's answer with, by decision: the candidate's name and the control's fill it.
_DECISION_LEADS: Final[Mapping[str, str]] = {
    "promote": "Promote {candidate}",
    "keep": "Keep {control}",
    "undecided": "Undecided, so {control} stays",
}


@dataclass(frozen=True)
class WriterPairwise:
    """One candidate's pairwise pass against the control, as a writer eval read it: the judge's side-by-side
    preference, the same over the pairs whose pages were both finished, the judge's order checks, and what the
    promotion read made of it.

    :ivar job_id: The pass's job, whose record holds its whole report.
    :ivar read_on: The job the promotion read took as its primary reading: ``job_id`` on the held-out side, ``None``
        on the dev side, where no promotion is read.
    :ivar primary: The verdict over every pair, a missing page counted as the side it leaves: the promotion's primary
        reading. Its ``delta`` is the candidate's mean share less the control's, and its ``interval`` the interval on
        it.
    :ivar share: The candidate's mean share of a pair, 0 to 1 (0.5 is no preference); ``None`` when nothing scored.
    :ivar finished_only: The verdict over the pairs whose two pages were both stored finished, deciding nothing.
    :ivar finished_share: The candidate's mean share over those pairs.
    :ivar both_orders: The cases the judge scored in both orders.
    :ivar order_consistent: Of those, the cases whose two orders picked the same side.
    :ivar order_consistency: That share; ``None`` when no case was scored both ways.
    :ivar first_slot_bias: The judge's mean lean to the page it read first, -0.5 to 0.5.
    :ivar cannot_tell: The answers the judge gave that it could not tell.
    :ivar void: Why the promotion read voided the primary reading (order consistency under its floor); ``None`` when it
        stands or no read was made.
    :ivar cases_to_settle: The held-out cases like these at which a primary that did not separate would, as the
        promotion read projects it; ``None`` when it separated, is untested or void, or no read was made.
    :ivar stopped: Why the pass stopped before its last pair, when it did; its verdicts are then untested.
    :ivar stopping: Its stopping rule and how it ran against it: where it stopped, each look, and its one sentence;
        ``None`` for a report that records none.
    :ivar calls_made: The judge calls it made.
    :ivar cost_usd: What they cost; ``None`` when any went unpriced.
    :ivar cap_usd: The part of ``max_cost_usd`` its calls were admitted under; ``None`` when none was enforced.
    :ivar cases_read: The cases its primary reading read: every case run, or those up to its stop.
    """

    job_id: str
    read_on: str | None
    primary: Verdict
    share: float | None
    finished_only: Verdict
    finished_share: float | None
    both_orders: int
    order_consistent: int
    order_consistency: float | None
    first_slot_bias: float | None
    cannot_tell: int
    void: str | None
    cases_to_settle: int | None
    stopped: str | None
    stopping: PairwiseStopping | None
    calls_made: int
    cost_usd: float | None
    cap_usd: float | None
    cases_read: int = 0


def _read_pass(status: JobStatus, read: WriterPromotion | None) -> WriterPairwise:
    """A completed pass's report, and the promotion read made on it if one was, as the result carries them."""
    report = WriterPairwiseReport.model_validate(status.result)
    reliability = report.reliability
    return WriterPairwise(
        job_id=status.job_id,
        read_on=None if read is None else read.pairwise_job,
        primary=report.whole.verdict,
        share=report.whole.candidate_share,
        finished_only=report.finished_only.verdict,
        finished_share=report.finished_only.candidate_share,
        both_orders=reliability.both_orders,
        order_consistent=reliability.order_consistent,
        order_consistency=reliability.order_consistency,
        first_slot_bias=reliability.first_slot_bias,
        cannot_tell=reliability.cannot_tell,
        void=None if read is None else read.void,
        cases_to_settle=None if read is None else read.cases_to_settle,
        stopped=report.stopped,
        stopping=report.stopping,
        calls_made=report.calls_made,
        cost_usd=report.cost_usd,
        cap_usd=report.cap_usd,
        cases_read=report.whole.cases,
    )


@dataclass(frozen=True)
class _Pass:
    """One candidate's pairwise pass: its completed job, or why it has none."""

    status: JobStatus | None
    why_not: str | None


@dataclass(frozen=True)
class _BatchCosts:
    """What a batch of cases can cost at most: each report at the budget, each judge call at the judge's floor (``None``
    when it cannot price one yet, which adds nothing)."""

    report: float
    floor: float | None
    dims: int
    judge: str

    def arm(self, cases: int) -> float:
        """One arm's worst case over ``cases``: its reports and their judging on every dimension, every attempt."""
        return cases * (self.report + self.dims * JUDGE_CALL_ATTEMPTS * (self.floor or 0.0))

    def pairs(self, cases: int) -> float:
        """One candidate's pass over ``cases`` pairs at worst: both orders, every attempt."""
        return cases * 2 * JUDGE_CALL_ATTEMPTS * (self.floor or 0.0)

    def short(
        self,
        cases: int,
        arms: Sequence[str],
        spent: Mapping[str, float],
        spent_pass: Mapping[str, float],
        part: float,
        pass_part: float,
    ) -> tuple[str, float, float] | None:
        """The first part a batch of ``cases`` does not fit, what the batch would reserve of it and what its spend so
        far leaves of it; ``None`` when it fits every launching arm's part and every launching candidate's part of the
        pass's reserve.

        :return: ``(whose part, reservation, left)``, or ``None``.
        """
        for arm in arms:
            if spent[arm] + self.arm(cases) > part + 1e-9:
                return f"{arm}'s part", self.arm(cases), part - spent[arm]
        for arm in arms[1:]:
            if spent_pass[arm] + self.pairs(cases) > pass_part + 1e-9:
                return f"the pass's part for {arm}", self.pairs(cases), pass_part - spent_pass[arm]
        return None


@dataclass(frozen=True)
class _Held:
    """A candidate settled at its own stop: the promotion read made there (``None`` on the dev side), and the cases
    launched by then."""

    read: WriterPromotion | None
    launched: int


def _guardrails_unheld(read: WriterPromotion) -> bool:
    """Whether a promotion read finds the primary improved and the guardrails not shown held: the one stop that does
    not decide, so the run goes on to the last case."""
    primary = read.primary
    improved = (
        primary is not None
        and primary.outcome == "improved"
        and primary.reason != "too_few_cases"
        and read.void is None
    )
    return improved and read.decision == "undecided"


def _report_of(ran: _Pass | None) -> WriterPairwiseReport | None:
    """A pass's report, when it completed and stopped short of nothing: what the next batch's pass carries."""
    if ran is None or ran.status is None or ran.why_not is not None:
        return None
    return WriterPairwiseReport.model_validate(ran.status.result)


def _launched_line(
    launched: int,
    planned: int,
    batch: int,
    side: str,
    passes: Mapping[str, _Pass],
    read_on: Mapping[str, int],
    *,
    capped: str | None = None,
) -> str:
    """Where a batched run stopped launching, in one sentence: the cases the gated readings and guardrails are read on,
    and how often. ``read_on`` is the cases each candidate's were read on, at its own stop or the last batch."""
    reports = [report for ran in passes.values() if (report := _report_of(ran)) is not None]
    again = any(report.stopping is not None and report.stopping.past_the_stop for report in reports)
    ran = (
        f"The arms ran every {side} case"
        if launched >= planned
        else f"Read on {launched} cases: the arms ran {launched} of the {planned} {side} cases"
    )
    ran += f", {batch} at a time in the bet's order"
    if len(set(read_on.values())) > 1:
        each = ", ".join(f"{count} for {arm}" for arm, count in read_on.items())
        return (
            f"{ran}, and each candidate stopped launching at its own stop, so its gated readings and guardrails were "
            f"read {'' if again else 'once, '}on the cases launched by then: {each}."
        )
    if launched >= planned:
        read = "read on every case" if again else "read once, on every case"
        return f"{ran}, so the gated readings and guardrails were {read}."
    separated = len(reports) == len(passes) and all(
        report.stopping is not None and report.stopping.stopped_at is not None for report in reports
    )
    those = f"those {launched}"
    if capped is not None:
        return (
            f"{ran}, until the spend cap stopped it (the pass's line says where), so the gated readings and guardrails "
            f"were read once, on {those}."
        )
    why = "every candidate's preference had separated" if separated else "no candidate's pass could read further"
    return f"{ran}, and stopped once {why}, so the gated readings and guardrails were read once, on {those}."


@dataclass(frozen=True)
class Writer:
    """One arm of a writer eval: the model that writes, the skill it starts from, and its reasoning effort.

    :ivar model_id: The writer's model, by the id its provider takes (``claude-sonnet-5-5``).
    :ivar skill: The analyst skill both its sessions start from: a frozen copy (``AnalystSkill.at(path)``) or a
        variant of one (``.with_file(name, text)``); ``None`` for the skill this build ships.
    :ivar effort: The reasoning effort every turn is sent at (``low``, ``medium``, ``high`` ...); ``None`` for the
        model's own.
    :ivar revisions: The revisions the analyst is given after a review that does not clear its report.
    :ivar editor_model: The model the editor's reviews run on, by the id its provider takes; ``None`` runs them on
        ``model_id``, which stays the arm's model either way.
    """

    model_id: str
    skill: AnalystSkill | None = None
    effort: str | None = None
    revisions: int = MOST_REVISIONS
    editor_model: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("a Writer names its model by id, and this one is blank")
        if self.editor_model is not None and not self.editor_model.strip():
            raise ValueError("a Writer names its editor's model by id, and this one is blank")
        for what, model in (("model", self.model_id), ("editor's model", self.editor_model)):
            if model is not None and any(sep in model for sep in _NOT_IN_A_SEGMENT):
                # Refused here, before any case is minted: each id names a recording directory.
                raise ValueError(
                    f"a Writer's {what} {model!r} names its recordings' directory, so it holds no '/' or '\\'; an "
                    "aggregator's route (vendor/model) cannot be recorded by the harness: name the provider's own id "
                    "and build it on that provider (writer_provider)"
                )
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
    :ivar k: The repeats of each case each arm ran.
    :ivar pairwise: Each candidate's pairwise pass against the control, by arm, as :class:`WriterPairwise` reads it;
        a pass that did not start, or whose job ended other than completed, has none.
    :ivar pairwise_unread: Why a candidate's pass did not complete, by arm: refused before any call, ended other than
        completed, or stopped before its last pair.
    :ivar cases: The side's case files the arms ran, by name, sorted: what a later run's ``WriterPrior`` names.
    :ivar resolution: The resolution floor's line, as the disclosures carry it.
    :ivar launch_line: Where a run launched in batches stopped launching, as the disclosures carry it; empty for one
        launched at once.
    :ivar one_editor: Each candidate, by arm: whether its promotion was read as running the control's editor (the
        same ``editor.md``, editor model and effort), so its editor's own readings compare the writing, as the read
        records it; empty on the dev side, where no promotion is read.
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
    k: int = 1
    pairwise: dict[str, WriterPairwise] = field(default_factory=dict, repr=False)
    pairwise_unread: dict[str, str] = field(default_factory=dict)
    cases: tuple[str, ...] = ()
    resolution: str = ""
    launch_line: str = ""
    one_editor: dict[str, bool] = field(default_factory=dict)

    def __str__(self) -> str:
        """The quiet summary (:meth:`render`)."""
        return self.render()

    def render(self, *, full: bool = False) -> str:
        """The result as text: the readings table, the pairwise table, each arm's results and spend, the promotion read
        and disclosures.

        Each reading is a measure the writer host declares, named as its descriptor names it, then each judged
        dimension of the template's rubric; then the side-by-side tables (:meth:`pairwise_text`). ``full`` adds every
        arm's whole summary and the campaign's page.

        :param full: Every arm's whole summary and the whole page, as Markdown, after the quiet summary.
        :ptype full: bool
        :return: The text.
        :rtype: str
        """
        candidates = " and ".join(arm for arm in self.arms if arm != self.control)
        cases = next(iter(self.arms.values())).n_cases
        lines = [
            f"{self.name}: {candidates} against {self.control}, {cases} {_SIDE_WORDS[self.side]} case(s) x {self.k}, "
            f"judged by {self.judge}"
        ]
        lines += self._readings()
        lines += ["", self.pairwise_text(), ""]
        lines += markdown_table(["stratum", "cases"], [[name, str(n)] for name, n in self.strata.items()])
        for arm, summary in self.arms.items():
            label = f"{arm}{' (control)' if arm == self.control else ''}"
            lines.append(f"{label}: {summary.results_text}")
            lines += [f"  {one_line(text)}" for text in (summary.stopped_text, summary.total_spend_text) if text]
            lines += [f"  error: {one_line(error)}" for error in summary.errors]
        lines += self._stopping_lines()
        if self.decisions:
            lines += [f"promotion of {arm}: {self.promotion_why[arm]}" for arm in self.decisions]
        else:
            lines.append("promotion: read on the held-out side only")
        lines += self.disclosures
        lines.append(f"stored in {self.store}, scope {self.scope_id!r}")
        text = "\n".join(lines)
        if not full:
            return text
        return "\n\n".join(
            [text, *(summary.render() for summary in self.arms.values()), campaign_report_markdown(self.page)]
        )

    def _readings(self) -> list[str]:
        """The readings table: a row per measure some arm carries, then per judged dimension, a column per arm. A rate
        (a boolean or pass/fail measure's, a pass/fail dimension's) reads as a percent and the count it is of, every
        other reading as its mean."""
        measures = self.host.profile.measures
        rows: dict[str, dict[str, tuple[float, int]]] = {}
        # Each row's measure, by its key and kind, which the label a reader sees may rename, and whether it is a rate.
        read: dict[str, tuple[str, str, bool]] = {}
        for arm, summary in self.arms.items():
            for measure in summary.measures:
                if measure.mean is not None:
                    descriptor = measures.get(measure.name)
                    label = (
                        descriptor.reader_name if descriptor is not None and descriptor.reader_name else measure.name
                    )
                    rows.setdefault(label, {})[arm] = (measure.mean, measure.n)
                    # A share reads in percent as every page reads it (number rule 4), a pass/fail measure too.
                    rate = descriptor is not None and (reads_as_share(descriptor) or descriptor.pass_fail)
                    read.setdefault(label, (measure.name, "measure", rate))
            for dimension in summary.judged:
                if dimension.mean is not None:
                    rows.setdefault(f"{dimension.name} (judged)", {})[arm] = (dimension.mean, dimension.n)
                    passes = dimension.scale == "pass_fail"
                    read.setdefault(f"{dimension.name} (judged)", (dimension.name, "judged", passes))
        forms = PageForms.off_page()
        # A measure the campaign's aspect page states a contrast of reads as that page prints its arms' levels (one form
        # per measure, number rule 1): Must-fix findings open reads 0.43 here as on the page, never another form.
        bundles = [fold.report.bundle for fold in self.page.aspects]
        aspects = PageForms.of_aspects(bundles, self.page.readings)
        on_page: dict[str, tuple[PageForms, MeasureRuler, ContrastPlaces]] = {}
        for bundle in bundles:
            rulers = report_rulers(bundle_decision_surface(bundle), bundle.measure_catalog)
            for family in bundle.multiple_comparisons.families:
                for comparison in family.comparisons:
                    places = aspects[bundle.aspect].contrast_places(comparison.name, comparison.reading)
                    if comparison.reading == "measure" and places is not None:
                        ruler = rulers.of(comparison.name, comparison.reading)
                        on_page.setdefault(comparison.name, (aspects[bundle.aspect], ruler, places))
        body: list[list[str]] = []
        for label, row in rows.items():
            key, kind, rate = read[label]
            if rate:
                said = {arm: printed.text for arm, printed in forms.rate_readings(row, measure=key, kind=kind).items()}
            elif kind == "judged":
                # A judged score at the precision its cases support: a tenth below about 30 cases, where a hundredth is noise.
                said = {arm: printed.text for arm, printed in forms.score_readings(row, measure=key, kind=kind).items()}
            elif kind == "measure" and key in on_page:
                page, ruler, places = on_page[key]
                said = {
                    arm: page.ruled_level(ruler, mean, places, measure=key, kind=kind).text
                    for arm, (mean, _) in row.items()
                }
            else:
                said = {arm: forms.reading(mean, measure=key, kind=kind).text for arm, (mean, _) in row.items()}
            body.append([label, *(said.get(arm, "-") for arm in self.arms)])
        return (
            markdown_table(["reading", *self.arms], body) if body else ["no reading: no arm's results carry a measure"]
        )

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        """Write the campaign's page as one self-contained file, its Markdown, its JSON and its aspect's bundle.

        :param out_dir: The directory; created when missing.
        :ptype out_dir: Path | str
        :return: Each file written, by role (``html``, ``markdown``, ``json``, ``bundle:<aspect>``).
        :rtype: dict[str, Path]
        """
        written = self.page.write(out_dir)
        pairwise = Path(out_dir) / "pairwise.md"
        said = "".join(f"\n{escaped(one_line(line))}\n" for line in self._stopping_lines())
        pairwise.write_text(f"# Side by side\n\n{self.pairwise_text()}\n{said}", encoding="utf-8")
        return {**written, "pairwise": pairwise}

    def _stopping_lines(self) -> list[str]:
        """What each candidate's pass did against its stopping rule, one sentence each."""
        return [
            f"pairwise pass of {arm}: {read.stopping.line}"
            for arm, read in self.pairwise.items()
            if read.stopping is not None
        ]

    def pairwise_text(self) -> str:
        """Each candidate's pairwise pass as two tables under the one sentence defining their terms, then a line for
        each pass that did not complete: the tables the page's Scorecard prints under its preference figure
        (:func:`_side_by_side_blocks`).

        The preference table has a row per candidate and the pairs it reads (every pair, a missing page counted as the
        side it leaves, then the pairs whose two pages were both stored finished): what the preference reads (``void``
        beside it when the order checks void it), the candidate's mean share, the control's beside it and the interval
        on the candidate's, all on the share scale (50% is even). The judge's checks table has a row per candidate:
        order consistency, the lean to the page read first, the answers the judge could not tell, the held-out cases
        like these that would settle a reading that did not separate, and what the pass spent of its part.

        :return: The text, as Markdown.
        :rtype: str
        """
        lines = [_PAIRWISE_LEAD]
        for block in _side_by_side_blocks(
            PageForms.off_page(),
            self.control,
            self.pairwise,
            self.pairwise_unread,
            candidates=self._candidates,
            why=self.promotion_why,
        ):
            if isinstance(block, TableBlock):
                lines += ["", *table_lines(block)]
        if self.pairwise_unread:
            lines += [
                "",
                *(escaped(one_line(f"pairwise pass of {arm}: {why}")) for arm, why in self.pairwise_unread.items()),
            ]
        return "\n".join(lines)

    @property
    def _candidates(self) -> list[str]:
        """Every arm but the control, in the order the arms were given."""
        return [arm for arm in self.arms if arm != self.control]

    def exit_code(self) -> int:
        """What a command line exits with: 0; 1 when an arm's run or a pairwise pass did not complete; 4 when a
        promotion gate failed."""
        if any(summary.status != "completed" for summary in self.arms.values()) or self.pairwise_unread:
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
        question: str | None = None,
        k: int = 1,
        pairwise_share: float = PAIRWISE_SHARE,
        turns: int = TURN_CAP,
        request_s: float = 120.0,
        margin: float | None = None,
        writer_provider: str = "anthropic",
        store: DocumentStore | str | os.PathLike[str] | None = None,
        scope: str | None = None,
        stop_when: StopWhen = "never",
        prior: WriterPrior | None = None,
        batch_size: int = _BATCH_SIZE,
        fewest_cases: int | None = None,
    ) -> None:
        """Declare the eval; nothing is read or run until :meth:`run`.

        :param cases_dir: The directory of writer case files (``*.json.gz``); relative, it is read beside the eval file
            being loaded.
        :ptype cases_dir: str | os.PathLike[str]
        :param arms: The arms, by name; the first is the control. At least two, no two alike in every lever (model,
            skill, effort, revisions, editor's model).
        :ptype arms: Mapping[str, Writer]
        :param judge: The scoring judge, of another family than the writers, with its recording.
        :ptype judge: WriterJudge
        :param recordings: The directory every arm's turn recordings sit under, read beside the eval file.
        :ptype recordings: str | os.PathLike[str]
        :param budget_usd: Each report's dollar cap.
        :ptype budget_usd: float
        :param max_cost_usd: The most the whole eval may spend, judge and pairwise pass included, split as the module
            docstring says. The run is refused when an arm's reports, each at ``budget_usd``, could pass its part,
            and, for a judge that prices its calls, when they and the judge's calls at their floor could, or a pass's
            calls at their floor could pass its part.
        :ptype max_cost_usd: float
        :param held_out_fraction: The share of split groups placed on the held-out side when first minted.
        :ptype held_out_fraction: float
        :param side: The side whose cases the arms run on; a promotion is read on ``held_out`` alone.
        :ptype side: WriterSide
        :param name: The eval's name, its scope's unless ``scope`` says, and its campaign's.
        :ptype name: str
        :param question: The question the campaign is filed asking; ``None`` asks whether the candidates write reports
            better than the control.
        :ptype question: str | None
        :param k: Repeats of each case per arm: 1, the only count a recorded arm can run (the module docstring).
        :ptype k: int
        :param pairwise_share: The part of ``max_cost_usd`` reserved for the pairwise passes, strictly between 0 and 1.
        :ptype pairwise_share: float
        :param turns: Each report's turn cap, which bounds how long a cell can run (the module docstring's table).
        :ptype turns: int
        :param request_s: The ceiling for one request on the writer's client, in seconds.
        :ptype request_s: float
        :param margin: How far, in points of a 0-100 rate, a candidate may fall below the control before it is shown
            worse: the author's own, a non-inferiority question's. ``None`` declares none, and the question is a
            superiority one, a candidate's drop read against zero.
        :ptype margin: float | None
        :param writer_provider: The provider every writer (and editor) is built on to record, whose key a recording run
            reads, and which tells a writer's family to the judge's check.
        :ptype writer_provider: str
        :param store: Where the cases and runs are kept: ``None`` for ``.evals/evals.sqlite``, ``":memory:"``, a file
            or a document store.
        :ptype store: DocumentStore | str | os.PathLike[str] | None
        :param scope: The scope; ``None`` is ``name``.
        :ptype scope: str | None
        :param stop_when: The pairwise pass's stopping rule, declared as :meth:`run` starts, before any arm launches,
            over the side's cases: ``separated`` stops the judge at the first case after which the preference
            separates, either way; ``never`` reads every case.
        :ptype stop_when: StopWhen
        :param prior: An earlier run on these cases, whose projection the resolution line carries when it ran the same
            case files; one over other cases is said and not carried.
        :ptype prior: WriterPrior | None
        :param batch_size: Under ``separated``, the cases every arm launches at a time, in the bet's order, before the
            pass reads on (4 unless told; a host picks it from its concurrent-cell count and its admitted-runs limit, as the constant beside the class says); ``never`` launches every case at once.
        :ptype batch_size: int
        :param fewest_cases: The fewest side cases the author asks a run to read; a side with fewer runs, and its
            resolution line says it is under them: a floor is a disclosure, never a refusal. ``None`` asks none.
        :ptype fewest_cases: int | None
        :raises ValueError: Fewer than two arms, two arms alike in every lever, a name too long for a campaign, a
            blank question, a ``k`` other than 1, a stopping rule that is neither, or a share, budget, turn cap or
            ceiling out of its range.
        """
        if len(arms) < 2:
            raise ValueError("a writer eval compares arms: name a control and at least one candidate")
        conditions: dict[_Condition, str] = {}
        for arm, writer in arms.items():
            if (twin := conditions.get(key := _condition(writer))) is not None:
                raise ValueError(
                    f"arms {twin!r} and {arm!r} are alike in every lever (model, skill, effort, revisions, editor's "
                    "model), so they compare nothing; vary one"
                )
            conditions[key] = arm
        if not 0 < held_out_fraction < 1:
            raise ValueError(f"held_out_fraction is a share strictly between 0 and 1, not {held_out_fraction}")
        if side not in _SIDE_WORDS:
            raise ValueError(f"side is 'held_out' or 'dev', not {side!r}")
        if len(name) > DECLARED_NAME_CHARS:
            raise ValueError(f"a writer eval's name titles its campaign, at most {DECLARED_NAME_CHARS} characters")
        if question is not None and not question.strip():
            raise ValueError("a writer eval's question is filed on its campaign, and this one is blank")
        if k != 1:
            raise ValueError(
                f"k is 1 for a writer eval, not {k}: each arm replays one recording per case, so a second repeat would "
                "replay the first's turns and repeat its report rather than measure another; run more cases instead"
            )
        if not 0 < pairwise_share < 1:
            raise ValueError(f"pairwise_share is a share strictly between 0 and 1, not {pairwise_share}")
        if stop_when not in _STOP_WHEN:
            raise ValueError(f"stop_when is 'separated' or 'never', not {stop_when!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size is at least 1 case, not {batch_size}")
        if fewest_cases is not None and fewest_cases < 1:
            raise ValueError(f"fewest_cases is at least 1 case, not {fewest_cases}")
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
        self.question = question
        self.k = k
        self.pairwise_share = pairwise_share
        self.request_s = request_s
        self.margin = margin
        self.writer_provider = writer_provider
        self.store = store
        self.scope_id = name if scope is None else scope
        self.stop_when: StopWhen = stop_when
        self.prior = prior
        self.batch_size = batch_size
        self.fewest_cases = fewest_cases

    async def run(self) -> WriterResult:
        """Mint the cases, launch every arm on one side's cases, file the campaign and read the promotion.

        :return: The result.
        :rtype: WriterResult
        :raises ValueError: The case files do not read; a case withholds a note an arm's skill lacks; the side holds
            no case, or its set lists a case the directory no longer holds; a recording is missing with no key to
            make it; or the reports' budgets, or they and the judge's calls at their floor, could pass an arm's part,
            or a pass's calls at their floor its part.
        :raises ValidationFailedError: The launch refused an arm (a judge of the writer's family, an arm whose cells
            could outlast its job), or the campaign's declaration.
        """
        # The stopping rule is declared now, before any arm launches, so before any result it could be read against.
        declared_at = utc_now_iso()
        cases = read_writer_cases(self.cases_dir)
        _refuse_unheld_notes(cases, self.arms)
        floors, warned = self._launch_budget(cases)
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
            out_of_run_cost_usd=self._pass_part,
        )
        try:
            return await self._run(host, cases, turns, where, floors=floors, warned=warned, declared_at=declared_at)
        finally:
            for routed in turns.values():
                await routed.aclose()
            await host.launch.job_manager.shutdown()

    async def _run(
        self,
        host: OpsHost,
        cases: Sequence[WriterCaseFile],
        turns: dict[str, WriterTurns],
        where: str,
        *,
        floors: Mapping[str, float],
        warned: Sequence[str],
        declared_at: str,
    ) -> WriterResult:
        eval_host = host.eval_host
        template_id = await run_blocking(eval_host.blocking_executor, self._template, host)
        control = next(iter(self.arms.values())).analyst_skill
        minted = [
            (
                case,
                await case.mint(
                    host, template_id, self.scope_id, created_by=CREATED_BY, split=self.split, skill=control
                ),
            )
            for case in cases
        ]
        on_side = [(case, receipt) for case, receipt in minted if receipt.side == self.side]
        case_set = await run_blocking(eval_host.blocking_executor, self._case_set, host, on_side)
        named = {receipt.snapshot_id: case.name for case, receipt in on_side}
        for arm in self.arms:
            writer = self.arms[arm]
            turns[arm] = writer_turns(
                self.recordings,
                arm=arm,
                skill=writer.analyst_skill,
                cases=named,
                revisions=writer.revisions,
                editor_model=writer.editor_model,
                provider=self.writer_provider,
            )
        held_elsewhere = await run_blocking(
            eval_host.blocking_executor, _results_held_elsewhere, host, self.scope_id, on_side
        )
        planned = sorted({receipt.test_case_id for _, receipt in on_side})
        batched = self.stop_when == "separated"
        rule = StoppingRule(
            stop_when=self.stop_when,
            case_ids=planned,
            max_cases=len(planned),
            declared_at=declared_at,
            # Recorded as declared, so the batches launch, and every later read judges, in this one order.
            order=[planned[index] for index in bet_order(len(planned))] if batched else None,
            batch_size=self.batch_size if batched else None,
        )
        names = frozenset(case.name for case, _ in on_side)
        resolution = self._resolution(len(on_side), names)
        # What each arm's recordings hold before anything launches; what it recorded is said once the run is over,
        # counted over the cases it ran.
        before = self._recorded_before(turns, named)
        disclosures = [
            *warned,
            resolution,
            *held_elsewhere,
        ]
        share = self._arm_part
        # Under a rule to stop, the cases launch in the bet's own order, a batch at a time, so each look reads a
        # prefix of the one bet; otherwise the side's set launches whole, as one batch.
        order = rule.bet
        batches: list[list[str] | None] = (
            [order[start : start + self.batch_size] for start in range(0, len(order), self.batch_size)]
            if batched
            else [None]
        )
        # The cap is a budget the run runs under: each batch is launched only when its worst case fits what the
        # batches before it left, and only a cap that cannot admit the first batch is refused.
        costs = await self._batch_costs(host, template_id)
        first = len(planned) if batches[0] is None else len(batches[0])
        self._refuse_a_first_batch_past_its_part(costs, cases=first * self.k)
        # The design, the question, the margin and the control are fixed here, in the call, before any arm runs: the
        # campaign, filed once they have, is dated now, so its criteria read as set before its results.
        declared_at = utc_now_iso()
        # What each part has spent so far, as its runs and passes recorded it (a run that priced nothing counts the
        # whole cap it ran under): the spend cap bounds actual spend, so what a batch did not spend is left to the next.
        spent = {arm: 0.0 for arm in self.arms}
        spent_pass = {arm: 0.0 for arm in list(self.arms)[1:]}
        cap_stop: str | None = None
        ran_cases: dict[str, list[str]] = {arm: [] for arm in self.arms}
        run_ids: dict[str, list[str]] = {arm: [] for arm in self.arms}
        campaign_id: str | None = None
        passes: dict[str, _Pass] = {}
        going_on: set[str] = set()
        # A candidate settled at its own stop: its pass and its promotion read, held as they were read there, and the
        # cases launched by then. It launches no further batch, and nothing of it is read again.
        held: dict[str, _Held] = {}
        launched: list[str] = []
        for index, batch in enumerate(batches):
            size = (len(planned) if batch is None else len(batch)) * self.k
            arms = [arm for arm in self.arms if arm not in held]
            short = costs.short(size, arms, spent, spent_pass, share, self._pass_part)
            if index > 0 and short is not None:
                whose, reserve, left = short
                total = math.fsum([*spent.values(), *spent_pass.values()])
                forms = PageForms.off_page()
                cap_stop = (
                    f"Stopped at the spend cap after {len(launched)} of {len(planned)} cases, "
                    f"{forms.money_total(total).text} spent: the next batch of {size // self.k} would reserve "
                    f"{forms.money_total(reserve).text} against the {forms.money_total(max(left, 0.0)).text} left of "
                    f"{whose}."
                )
                log.warning(
                    "eval.writer_batch_cap_stop scope=%s batch=%d/%d launched_cases=%d planned_cases=%d spent_usd=%.4f "
                    "next_reserve_usd=%.4f left_usd=%.4f part=%s",
                    self.scope_id,
                    index + 1,
                    len(batches),
                    len(launched),
                    len(planned),
                    total,
                    reserve,
                    left,
                    whose,
                )
                break
            ref = case_set
            if batch is not None and len(batches) > 1:
                ref = await run_blocking(
                    eval_host.blocking_executor,
                    self._batch_set,
                    host,
                    template_id,
                    case_set,
                    batch,
                    index,
                    len(batches),
                )
            launched += planned if batch is None else batch
            for arm in [arm for arm in self.arms if arm not in held]:
                ran_cases[arm] += planned if batch is None else batch
            # The reservation decides whether a batch launches: its reports at the budget and its judging at the
            # judge's floor, the one worst case known before a page is written, must fit what each part has left. Once
            # launched, each arm's run runs under all its part has left, not under the reservation: a page's judging
            # is priced in full only once the page exists, and the engine admits it against the run's cap before
            # paying for any of it, so a run capped at a floor-priced reservation would set aside judging the arm
            # could afford. The arm's runs run one after another, each under what the ones before it left, and the
            # engine holds each to its cap, so together they stay within the part.
            reservation = costs.arm(size) if len(batches) > 1 else share
            caps = {arm: share - spent[arm] for arm in arms}
            log.info(
                "eval.writer_batch_launch scope=%s batch=%d/%d cases=%d arms=%s reserve_usd=%.4f caps=%s",
                self.scope_id,
                index + 1,
                len(batches),
                size // self.k,
                ",".join(arms),
                reservation,
                ",".join(f"{arm}:{PageForms.off_page().money_total(cap).text}" for arm, cap in caps.items()),
            )
            ran = await self._launch_batch(host, template_id, ref, caps, arms)
            for arm, run_id in ran.items():
                run_ids[arm].append(run_id)
                run_spent, status, stopped = await run_blocking(
                    eval_host.blocking_executor, self._spent, eval_host, run_id, caps[arm]
                )
                spent[arm] += run_spent
                log.info(
                    "eval.writer_batch_ran scope=%s batch=%d/%d arm=%s run=%s status=%s spent_usd=%.4f "
                    "cap_usd=%.4f arm_spent_usd=%.4f arm_part_usd=%.4f stopped=%s",
                    self.scope_id,
                    index + 1,
                    len(batches),
                    arm,
                    run_id,
                    status,
                    run_spent,
                    caps[arm],
                    spent[arm],
                    share,
                    stopped or "-",
                )
            # Every batch run joins the campaign, a run that stored no page among them: nothing of it was judged, so a
            # judge dimension none of its results exercised reads at its arm's level, and it sits in its arm's test
            # setup. Its unwritten reports count as failures in Report finished, the judged readings find no score of
            # it, and the pass reads a missing page as losing its pair, or tying it when both are. A case whose judging
            # the cost cap set aside did write its report: Report finished counts it as what it did, the judged readings
            # leave it out, and the pass reads its page as any other.
            if campaign_id is None:
                campaign_id = await run_blocking(
                    eval_host.blocking_executor, self._file_campaign, eval_host, run_ids, template_id, declared_at
                )
            else:
                await run_blocking(
                    eval_host.blocking_executor,
                    add_runs_to_campaign,
                    eval_host.storage,
                    campaign_id,
                    self.scope_id,
                    list(ran.values()),
                )
            for arm in arms[1:]:
                passes[arm] = await self._pairwise(
                    host,
                    campaign_id,
                    arm,
                    rule,
                    judged=_report_of(passes.get(arm)),
                    past_the_stop=arm in going_on,
                    launched=launched if batched else None,
                )
                report = _report_of(passes[arm])
                spent_pass[arm] = (
                    spent_pass[arm] + costs.pairs(size)
                    if report is None or report.cost_usd is None
                    else report.cost_usd
                )
            if index == len(batches) - 1:
                break
            await self._settle(host, campaign_id, passes, going_on, held, launched=len(launched))
            if len(held) == len(self.arms) - 1:
                break
        # The first batch always launches, and files the campaign.
        assert campaign_id is not None
        if cap_stop is not None:
            # The candidates still reading are read once more, asking nothing, so their records carry the stop.
            for arm in [arm for arm in list(self.arms)[1:] if arm not in held]:
                passes[arm] = await self._pairwise(
                    host,
                    campaign_id,
                    arm,
                    rule,
                    judged=_report_of(passes.get(arm)),
                    past_the_stop=arm in going_on,
                    launched=launched,
                    out_of_budget=cap_stop,
                )
        snapshot_of = {receipt.test_case_id: receipt.snapshot_id for _, receipt in on_side}
        wrote = {arm: {snapshot_of[case_id] for case_id in ids} for arm, ids in ran_cases.items()}
        at = len(warned) + 1
        disclosures[at:at] = self._recording_disclosures(turns, before, wrote)
        for arm, routed in turns.items():
            disclosures += [f"{arm}: {left}" for left in routed.unconsumed()]
        disclosures += await run_blocking(
            eval_host.blocking_executor,
            _over_budget_with_no_page,
            host,
            self.scope_id,
            run_ids,
            self.budget.spelled,
            floors,
        )
        if len(batches) > 1:
            pageless = await run_blocking(
                eval_host.blocking_executor, _pageless_runs, eval_host, run_ids, self.scope_id
            )
            if pageless:
                disclosures.append(
                    await run_blocking(
                        eval_host.blocking_executor, _pageless_line, eval_host, run_ids, pageless, self.scope_id
                    )
                )
        launch_line = None
        if batched:
            read_on = {arm: held[arm].launched if arm in held else len(launched) for arm in passes}
            launch_line = _launched_line(
                len(launched),
                len(planned),
                self.batch_size,
                _SIDE_WORDS[self.side],
                passes,
                read_on,
                capped=cap_stop,
            )
            disclosures.append(launch_line)
        case_names = {receipt.test_case_id: case.name for case, receipt in on_side}
        filed = campaign_id
        return await run_blocking(
            eval_host.blocking_executor,
            lambda: self._declared(
                eval_host,
                run_ids,
                filed,
                passes,
                case_names=case_names,
                strata=Counter(case.stratum or "none" for case, _ in on_side),
                where=where,
                disclosures=tuple(disclosures),
                cases=tuple(sorted(names)),
                resolution=resolution,
                launch_line=launch_line,
                held={arm: kept.read for arm, kept in held.items() if kept.read is not None},
            ),
        )

    def _spent(self, host: EvalHost, run_id: str, cap: float) -> tuple[float, str, str | None]:
        """What a run spent, as its results recorded it (the whole cap it ran under when any of it went unpriced), its
        status, and why a designed stop ended it, if one did. Blocking."""
        summary = summarize_run(host, run_id, self.scope_id)
        cost = summary.total_cost_usd
        if cost is None:
            return cap, summary.status, summary.stopped_because
        # Beside its cost, what its priced cells' dropped sends may have billed, as the run's cap counted it: the next
        # batch's cap is what the arm's part has left after both.
        dropped = math.fsum(
            maybe_billed_usd(row.usage, row.cost_roles)
            for row in iter_stored_results(host.storage, run_id, self.scope_id)
            if row.cost_usd is not None
        )
        return cost + dropped, summary.status, summary.stopped_because

    async def _launch_batch(
        self, host: OpsHost, template_id: str, case_set: CaseSetRef, parts: Mapping[str, float], arms: Sequence[str]
    ) -> dict[str, str]:
        """Launch each of ``arms`` over ``case_set``, each under its part of the ceiling, quoting and refusing every arm
        before any starts, and await them; each arm's run, by arm."""
        launches = {arm: self._launch(template_id, case_set, self.arms[arm], parts[arm]) for arm in arms}
        for arguments in launches.values():  # every arm is quoted, and refused, before any is started
            await quote_launch(host.launch, **arguments)
        runs = {arm: (await start_run(host.launch, **arguments))[0] for arm, arguments in launches.items()}
        await host.launch.job_manager.wait_for([run.id for run in runs.values()])
        return {arm: run.id for arm, run in runs.items()}

    def _batch_set(
        self, host: OpsHost, template_id: str, whole: CaseSetRef, batch: Sequence[str], index: int, batches: int
    ) -> CaseSetRef:
        """The set one batch launches: its cases, as a set made for one launch named for the side's set version it is
        cut from (:func:`~threetears.evals.ops.report_writer.writer_promotion.held_out_batch_name`). The version there
        is reused when it already lists exactly these cases, as a later run on the same set's batches finds it.
        Blocking."""
        storage = host.eval_host.storage
        name = held_out_batch_name(whole, index, batches)
        versions = storage.query_case_sets(self.scope_id, name=name)
        if versions and list(versions[0].test_case_ids) == list(batch):
            return versions[0].ref
        listed = mint_case_set(
            storage,
            scope_id=self.scope_id,
            name=name,
            template_id=template_id,
            test_case_ids=list(batch),
            tracked=False,
        )
        return listed.ref

    async def _settle(
        self,
        host: OpsHost,
        campaign_id: str,
        passes: Mapping[str, _Pass],
        going_on: set[str],
        held: dict[str, _Held],
        *,
        launched: int,
    ) -> None:
        """Hold each candidate that reads no further, as it stands at its own stop: a pass that did not complete, or one
        whose preference separated. Its promotion is read there, once (on the held-out side). A candidate whose
        preference separated improved while its guardrails are not shown held on the cases run by then is not held:
        it goes on to the last case (``going_on``), and its readings have a second look there. The writer template declares
        no guardrail, so that branch is reached only when the scope's template declares one, as a host's may."""
        for arm, ran in passes.items():
            if arm in held or arm in going_on:
                continue
            stopping = None if ran.status is None else WriterPairwiseReport.model_validate(ran.status.result).stopping
            complete = ran.status is not None and ran.why_not is None
            if complete and (stopping is None or stopping.stopped_at is None):
                continue
            read = None if self.side != "held_out" else await self._read_at_stop(host, campaign_id, arm, ran)
            if complete and read is not None and _guardrails_unheld(read):
                going_on.add(arm)
                continue
            held[arm] = _Held(read=read, launched=launched)

    async def _read_at_stop(self, host: OpsHost, campaign_id: str, candidate: str, ran: _Pass) -> WriterPromotion:
        """The promotion read on a candidate's pass at its stop."""
        return await run_blocking(
            host.eval_host.blocking_executor,
            lambda: writer_promotion(
                host.eval_host,
                campaign_id,
                self.scope_id,
                candidate=candidate,
                pairwise_job=ran.status,
                one_editor=self._one_editor(candidate),
            ),
        )

    def _one_editor(self, candidate: str) -> bool:
        """Whether a candidate runs the control's editor, so its editor's own readings compare the writing: the same
        ``editor.md``, on the same model (the candidate model, for an arm naming no editor's model of its own), at the
        same effort. Any of them differing makes the editor another instrument."""
        mine, control = self.arms[candidate], next(iter(self.arms.values()))
        return all(
            ours == theirs
            for ours, theirs in (
                (mine.analyst_skill.editor().digest, control.analyst_skill.editor().digest),
                (mine.editor_model or mine.model_id, control.editor_model or control.model_id),
                (mine.effort, control.effort),
            )
        )

    def _resolution(self, cases: int, names: frozenset[str]) -> str:
        """The resolution floor's line at ``cases``, with the prior run's projection when it ran these same case files,
        and saying so when it ran others."""
        prior, fewest = self.prior, self.fewest_cases
        if prior is None:
            return resolution_line(cases, fewest=fewest)
        if prior.case_names != names:
            return (
                f"{resolution_line(cases, fewest=fewest)} The prior run given ran other cases, so its projection is "
                "not carried."
            )
        return resolution_line(cases, projected=prior.cases_to_settle, fewest=fewest)

    @property
    def _pass_part(self) -> float:
        """Each candidate's part of the pairwise reserve: the out-of-run cap its pass is admitted under."""
        return self.max_cost_usd * self.pairwise_share / (len(self.arms) - 1)

    @property
    def _arm_part(self) -> float:
        """Each arm's part of what the pairwise reserve leaves: the ceiling its launch names."""
        return self.max_cost_usd * (1 - self.pairwise_share) / len(self.arms)

    def _launch_budget(self, cases: Sequence[WriterCaseFile]) -> tuple[dict[str, float], list[str]]:
        """Each arm's analyst first-turn ceiling, priced before any snapshot is read, and the launch's warning, if any;
        the run is refused first when an arm's analyst or editor cannot be admitted even one turn (the module
        docstring). An arm whose model the published rates do not list is left to each report's first turn."""
        floors: dict[str, float] = {}
        twice: list[tuple[float, str]] = []
        forms = PageForms.off_page()
        for arm, writer in self.arms.items():
            analyst_texts = [writer.analyst_skill.analyst(withhold=case.withhold).text for case in cases] or [
                writer.analyst_skill.analyst().text
            ]
            limits = WriterLimits()
            roles = (
                ("analyst", writer.model_id, min(analyst_texts, key=len), limits.analyst.session.max_tokens),
                (
                    "editor",
                    writer.editor_model or writer.model_id,
                    writer.analyst_skill.editor().text,
                    limits.editor.session.max_tokens,
                ),
            )
            for role, model, text, max_tokens in roles:
                ceiling = self._first_turn_floor(model, text, max_tokens)
                if ceiling is None:
                    continue
                refused = self.budget.first_turn_refusal(
                    ceiling,
                    priced=(
                        f"the {role}'s first turn of arm {arm!r} on {model}, priced at its "
                        f"{forms.count(max_tokens).text}-token output cap and its text before any snapshot is read,"
                    ),
                )
                if refused is not None:
                    raise ValueError(
                        f"the writer eval was not launched, and nothing was sent or spent: {refused}: raise budget_usd"
                    )
                if role == "analyst":
                    floors[arm] = ceiling.usd
                    twice.append((ceiling.usd, model))
        if self.budget.usd is None or not twice:
            return floors, []
        dearest, model = max(twice)
        if dearest * MIN_TURNS_FOR_A_REPORT <= self.budget.usd:
            return floors, []
        needed = forms.money_total(dearest * MIN_TURNS_FOR_A_REPORT).text
        return floors, [
            f"the {self.budget.spelled} report budget is under {MIN_TURNS_FOR_A_REPORT} of a first turn's ceiling on "
            f"{model} ({needed}), so the cap may stop a report before its "
            f"review; {_RECORDED_BUDGETS if model == _RECORDED_MODEL else f'no acceptance run is recorded on {model}'}"
        ]

    def _first_turn_floor(self, model: str, text: str, max_tokens: int) -> TurnCeiling | None:
        """A floor of a first turn's ceiling on ``model``: its system ``text`` and output cap at Anthropic's published
        rates, or ``None`` when they do not list the model on the writer's provider."""
        # The client's own table, loaded once per launch: importing the harness loads no client (test_providers_boundary).
        from threetears.evals.providers import AnthropicPublishedRates, Provider

        provider = next((known for known in get_args(Provider) if known == self.writer_provider), None)
        rated = None if provider is None else AnthropicPublishedRates().rates_for(model, provider=provider)
        if rated is None:
            return None
        dollars = rated.rates.ceiling(prompt_tokens=prompt_token_bound(text, "", None), output_tokens=max_tokens)
        return TurnCeiling(usd=float(dollars), estimate=not rated.guaranteed)

    async def _batch_costs(self, host: OpsHost, template_id: str) -> _BatchCosts:
        """What a batch can cost at most, per case: each arm's report at ``budget_usd`` and its judging, each call at
        the judge's floor, and the pass's two orders of each pair at that floor. A judge that cannot price a call yet
        (its rates unread) adds nothing here; each pass then refuses, with nothing spent, what its part cannot cover."""
        await refresh_prices(self.judge)
        floor = self.judge.price_ceiling(system="", user="", response_format=None)
        template = await run_blocking(
            host.eval_host.blocking_executor, host.eval_host.storage.load_template, template_id, self.scope_id
        )
        return _BatchCosts(
            report=self.budget.usd or 0.0,
            floor=floor,
            dims=0 if template is None else len(template.rubric),
            judge=self.judge.model_name,
        )

    def _refuse_a_first_batch_past_its_part(self, costs: _BatchCosts, *, cases: int) -> None:
        """Refuse the run before any arm launches when the first batch's worst case could pass an arm's part or the
        pass's (the module docstring), naming the smallest ``max_cost_usd`` that admits it: the one cap no run under it
        could keep."""
        arms, candidates = len(self.arms), len(self.arms) - 1
        smallest = max(
            costs.arm(cases) * arms / (1 - self.pairwise_share), costs.pairs(cases) * candidates / self.pairwise_share
        )
        forms = PageForms.off_page()
        admitted = forms.money_total(math.ceil(smallest * 100 - 1e-9) / 100)
        admits = f"a max_cost_usd of {admitted.text} or more admits it"
        which = "the first batch's" if self.stop_when == "separated" else "the"
        if self.budget.usd is not None and cases * self.budget.usd > self._arm_part:
            raise ValueError(
                f"{which} {cases} report(s) at {forms.money_total(self.budget.usd).text} each could pass an arm's "
                f"{forms.money_total(self._arm_part).text} part of max_cost_usd "
                f"{forms.money_total(self.max_cost_usd).text}, after the pairwise pass's "
                f"{forms.money_total(self.max_cost_usd * self.pairwise_share).text} (pairwise_share "
                f"{self.pairwise_share}); {admits}, or lower budget_usd"
            )
        floor = costs.floor
        if floor is None:
            return
        each = (
            f"{forms.money_ledger(floor).text} each before any page is read ({costs.judge}'s output cap and "
            f"{PROMPT_FRAMING_TOKENS} prompt tokens)"
        )
        pass_calls = cases * 2 * JUDGE_CALL_ATTEMPTS
        if pass_calls * floor > self._pass_part:
            raise ValueError(
                f"a pairwise pass's {pass_calls} judge call(s) ({cases} pair(s) in both orders, {JUDGE_CALL_ATTEMPTS} "
                f"attempts each) are priced at {each}, {forms.money_total(pass_calls * floor).text} in all, above the "
                f"pass's {forms.money_total(self._pass_part).text} part of max_cost_usd "
                f"{forms.money_total(self.max_cost_usd).text} (pairwise_share "
                f"{self.pairwise_share}); {admits}, or raise pairwise_share or lower the judge's max_tokens"
            )
        judge_calls = cases * costs.dims * JUDGE_CALL_ATTEMPTS
        if costs.arm(cases) > self._arm_part:
            raise ValueError(
                f"an arm's {cases} report(s) at {forms.money_total(self.budget.usd or 0).text} and its {judge_calls} "
                "judge call(s) "
                f"({cases} case(s) x {costs.dims} dimension(s) x {JUDGE_CALL_ATTEMPTS} attempts), priced at {each}, "
                f"come to {forms.money_total(costs.arm(cases)).text}, above the arm's "
                f"{forms.money_total(self._arm_part).text} part of max_cost_usd "
                f"{forms.money_total(self.max_cost_usd).text}; {admits}, or lower budget_usd or the judge's max_tokens"
            )

    async def _pairwise(
        self,
        host: OpsHost,
        campaign_id: str,
        candidate: str,
        rule: StoppingRule,
        *,
        judged: WriterPairwiseReport | None = None,
        past_the_stop: bool = False,
        launched: Sequence[str] | None = None,
        out_of_budget: str | None = None,
    ) -> _Pass:
        """Run one candidate's pairwise pass against the control as a job, and await it; its status, and why it did not
        complete when it did not: a refusal the pass makes before any call (its calls priced over its part, a judge of
        a writer's family, no case both arms ran), a job that ended other than completed, or a pass stopped before its
        last pair (an account refusal), whose report is still read. ``judged`` is the candidate's pass over the earlier
        batches, whose answers stand; ``past_the_stop`` reads on to the last case."""
        try:
            started = await writer_pairwise(
                host,
                campaign_id,
                self.scope_id,
                candidate=candidate,
                stopping_rule=rule,
                judged=judged,
                past_the_stop=past_the_stop,
                launched=launched,
                out_of_budget=out_of_budget,
            )
        except (ValidationFailedError, NotFoundError) as refused:
            return _Pass(status=None, why_not=f"refused before any call: {refused}")
        (status,) = await await_job(host, started, self.scope_id)
        if status.state != "completed" or status.result is None:
            return _Pass(status=None, why_not=f"ended {status.state}: {status.detail or 'no detail recorded'}")
        stopped = WriterPairwiseReport.model_validate(status.result).stopped
        # A pass stopped part-way completes its job with its verdicts untested: read, printed, and still not complete.
        return _Pass(status=status, why_not=None if stopped is None else f"stopped before its last pair: {stopped}")

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

    def _recorded_before(self, turns: Mapping[str, WriterTurns], named: Mapping[str, str]) -> dict[str, set[str]]:
        """Each arm's cases whose turn recordings are there before anything launches, by snapshot; the run is refused
        when a case's file is missing and no key is set to record it."""
        before: dict[str, set[str]] = {}
        for arm, routed in turns.items():
            writer = self.arms[arm]
            by_case = {snapshot: routed.paths(writer.model_id, _effort(writer), snapshot) for snapshot in named}
            missing = [snapshot for snapshot, paths in by_case.items() if not all(path.is_file() for path in paths)]
            if missing and not os.environ.get(routed.key_variable):
                first = next(path for path in by_case[missing[0]] if not path.is_file())
                raise ValueError(
                    f"arm {arm!r} has no turn recording for {len(missing)} of {len(by_case)} case(s) ({first} first), "
                    f"and {routed.key_variable} is not set to record them; restore the files, or set it to record "
                    "them once"
                )
            before[arm] = set(by_case) - set(missing)
        return before

    def _recording_disclosures(
        self, turns: Mapping[str, WriterTurns], before: Mapping[str, set[str]], ran: Mapping[str, set[str]]
    ) -> list[str]:
        """Where each arm's writer came from, counted over the cases it ran: replayed from a recording there before
        the run, or recorded by it."""
        said: list[str] = []
        for arm, routed in turns.items():
            writer = self.arms[arm]
            folder = routed.directory(writer.model_id, _effort(writer))
            cases = ran.get(arm, set())
            recorded = cases - before.get(arm, set())
            replayed = len(cases) - len(recorded)
            said.append(
                f"{arm}: writer turns replayed from {folder}"
                if not recorded
                else f"{arm}: writer turns of {len(recorded)} case(s) recorded to {folder}, {replayed} replayed"
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
            "k_runs": self.k,
            "case_set": case_set,
            "judge_model": self.judge.model_name,
            "overlays": {
                "skill": writer.analyst_skill.digest,
                "reasoning_effort": writer.effort,
                "revisions": writer.revisions,
                "editor_model": writer.editor_model,
            },
        }

    def _file_campaign(
        self, host: EvalHost, run_ids: Mapping[str, Sequence[str]], template_id: str, declared_at: str
    ) -> str:
        """File the campaign over every arm's runs (its first batch's, the later batches joining it as they run),
        control first, its design, question, margin and control dated ``declared_at``, when the arms launched; its id.
        Blocking."""
        control, *candidates = self.arms
        levels = {arm: _levels(writer) for arm, writer in self.arms.items()}
        axes = []
        for lever in (SKILL_LEVER, _EFFORT_LEVER, _REVISIONS_LEVER, _EDITOR_LEVER, CANDIDATE_MODEL_LEVER):
            values = dict.fromkeys(levels[arm][lever] for arm in self.arms)
            if len(values) > 1:
                shown = {value: next(arm for arm in self.arms if levels[arm][lever] == value) for value in values}
                axes.append({"axis_id": lever, "values": [{"content": v, "display": shown[v]} for v in values]})
        asked = self.question or f"Does {' or '.join(candidates)} write reports better than {control}?"
        definition = CampaignDefinition(
            name=self.name,
            subject_id=self.name,
            behavior="report-writing",
            question=asked,
            arms=[
                {"label": arm, "role": "control" if arm == control else "candidate", "from_run_ids": list(ids)}
                for arm, ids in run_ids.items()
            ],
            aspects=[
                {
                    "template_id": template_id,
                    "run_ids": [run_id for ids in run_ids.values() for run_id in ids],
                    "margin": self.margin,
                    "design": {
                        "axes": axes,
                        "held_fixed": {"stimulus": "controlled", "apparatus": "commissioned"},
                        "intended_repetitions": self.k,
                    },
                }
            ],
        )
        campaign = create_campaign(
            host.storage,
            definition.model_dump(exclude={"aspects_from", "exploratory"}),
            scope_id=self.scope_id,
            created_by=CREATED_BY,
            profile=host.profile,
            exploratory=definition.exploratory,
            aspects_from=definition.aspects_from,
            declared_at=declared_at,
        )
        return campaign.id

    def _declared(
        self,
        host: EvalHost,
        run_ids: Mapping[str, Sequence[str]],
        campaign_id: str,
        passes: Mapping[str, _Pass],
        *,
        case_names: Mapping[str, str],
        strata: Mapping[str, int],
        where: str,
        disclosures: tuple[str, ...],
        cases: tuple[str, ...],
        resolution: str,
        launch_line: str | None = None,
        held: Mapping[str, WriterPromotion] | None = None,
    ) -> WriterResult:
        """Read each candidate's promotion on its pass, and summarise each arm over its batches' runs. Blocking."""
        control = next(iter(self.arms))
        promotions: dict[str, WriterPromotion] = {}
        if self.side == "held_out":
            # A candidate held at its own stop keeps the read made there; the rest are read now, once.
            kept = held or {}
            promotions = {
                arm: kept[arm]
                if arm in kept
                else writer_promotion(
                    host,
                    campaign_id,
                    self.scope_id,
                    candidate=arm,
                    pairwise_job=ran.status,
                    one_editor=self._one_editor(arm),
                )
                for arm, ran in passes.items()
            }
        arms = {arm: summarize_runs(host, ids, self.scope_id, case_names=case_names) for arm, ids in run_ids.items()}
        # An arm launched in batches ran under its part of the cap across them, each batch under what the ones before
        # it left, so its ceiling is the part: the runs' ceilings summed would read above the cap.
        arms = {
            arm: summary if len(run_ids[arm]) == 1 else summary.model_copy(update={"max_cost_usd": self._arm_part})
            for arm, summary in arms.items()
        }
        pairwise = {
            arm: _read_pass(ran.status, promotions.get(arm)) for arm, ran in passes.items() if ran.status is not None
        }
        unread = {arm: ran.why_not for arm, ran in passes.items() if ran.why_not is not None}
        decisions: dict[str, str] = {arm: read.decision for arm, read in promotions.items()}
        why = {arm: read.why for arm, read in promotions.items()}
        # The side-by-side section and the answer state one candidate's shares alike, so they share one set of forms.
        forms = PageForms.off_page()
        side_by_side = _side_by_side_blocks(
            forms, control, pairwise, unread, candidates=list(passes), why=why, launch_line=launch_line
        )
        answer = _answer_block(
            forms,
            control,
            list(passes),
            pairwise,
            unread,
            decisions=decisions,
            why=why,
            cases=next(iter(arms.values())).n_cases,
            side=_SIDE_WORDS[self.side],
        )
        return WriterResult(
            name=self.name,
            scope_id=self.scope_id,
            side=self.side,
            control=control,
            arms=arms,
            host=host,
            campaign_id=campaign_id,
            page=_with_side_by_side(campaign_report(host, campaign_id, self.scope_id, lead=[answer]), side_by_side),
            judge=self.judge.model_name,
            strata=dict(sorted(strata.items())),
            decisions=decisions,
            promotion_why=why,
            gates={arm: gate_verdicts(read.verdicts) for arm, read in promotions.items()},
            store=where,
            disclosures=disclosures,
            k=self.k,
            pairwise=pairwise,
            pairwise_unread=unread,
            cases=cases,
            resolution=resolution,
            launch_line=launch_line or "",
            one_editor={arm: read.one_editor for arm, read in promotions.items()},
        )


def _pairwise_reads(verdict: Verdict) -> str:
    """What a pairwise verdict reads, as its cell prints it: its outcome, too few cases beside it when it was read on
    fewer cases than a rate needs; the undecided outcome in the glossary's words, ``cannot tell apart``."""
    outcome = CANNOT_TELL_APART if verdict.outcome == "not_separated" else verdict.outcome.replace("_", " ")
    if verdict.reason == "too_few_cases":
        return f"{outcome} (too few cases)"
    # An untested reading names why, in the one sentence its row has.
    return f"{outcome}: {verdict.reason_detail}" if verdict.outcome == "untested" and verdict.reason_detail else outcome


def _primary_reads(read: WriterPairwise) -> str:
    """What a candidate's primary reading reads, ``void`` beside it when the order checks void it."""
    reads = _pairwise_reads(read.primary)
    return reads if read.void is None else f"{reads} (void)"


def _level_text(forms: PageForms, pairwise: Mapping[str, WriterPairwise]) -> str | None:
    """The level the passes read their intervals at (``95%``); ``None`` when none read one."""
    level = next((read.primary.interval_level for read in pairwise.values() if read.primary.interval_level), None)
    return None if level is None else f"{forms.lone(level * 100).text}%"


def _preference_figure(
    forms: PageForms, control: str, pairwise: Mapping[str, WriterPairwise], candidates: Sequence[str]
) -> list[ReportBlock]:
    """The preference drawn at the head of the Scorecard, so a reader sees at once which page the judge preferred, by
    how much and whether it is settled: each candidate's share as a dot on the 0 to 100% scale, its interval where its
    pass read one (an open mark where it read none), against a line at even (50%), each end of the axis saying whose
    page it prefers ("← prefers shipped", "prefers lead-first →"); then one line per candidate saying
    the same in words, the control's share beside the candidate's. Nothing when no pass scored a pair. An interval is
    drawn and said only with its level, as every interval on a page is (GENERATOR §5.2)."""
    level = _level_text(forms, pairwise)
    # Each share at the precision the table prints it, its interval dropped from the figure where it has no level.
    drawn = [
        (arm, read, shares if read.primary.interval_level else replace(shares, interval=None))
        for arm in candidates
        if (read := pairwise.get(arm)) is not None
        and (shares := forms.preference(read.share, read.primary.interval)) is not None
    ]
    if not drawn:
        return []

    # Each candidate's share reads at its own precision everywhere it appears: its table row, its line, the answer and
    # its mark, whose tooltip and values table print the text its forms decided. Where the candidates' precisions
    # differ, each is a panel of its own, with its own format, so no share is padded to another's decimals
    # (``61.000%`` beside ``49.998%``).
    def row(arm: str, shares: PreferenceShares) -> DotIntervalRow:
        ends = shares.interval
        return DotIntervalRow(
            label=arm,
            value=shares.candidate,
            lo=None if ends is None else ends[0],
            hi=None if ends is None else ends[1],
            hollow=True if ends is None else None,
            value_text=shares.share_percent(shares.candidate),
            # The interval's unit is written once (number rule 3): on the share beside it in a tooltip (``70% (33 to
            # 100)``), after its high end where the values table states it alone (``33 to 100%``).
            lo_text=None if ends is None else shares.share_text(ends[0]),
            hi_text=None if ends is None else shares.share_text(ends[1]),
        )

    formats = {shares.fmt for _, _, shares in drawn}
    panels = (
        None
        if len(formats) == 1
        else [DotIntervalPanel(title=arm, rows=[row(arm, shares)], fmt=shares.fmt) for arm, _, shares in drawn]
    )
    chart = DotIntervalChart(
        rows=[row(arm, shares) for arm, _, shares in drawn] if panels is None else None,
        panels=panels,
        domain=(0.0, 100.0),
        fmt=drawn[0][2].fmt,
        level=level if any(shares.interval is not None for _, _, shares in drawn) else None,
        ticks=list(_SHARE_TICKS),
        reference=ChartReference(
            value=EVEN_SHARE, kind="no_preference", label=f"even ({forms.lone(EVEN_SHARE).text}%)"
        ),
        # Right of even the judge preferred the candidate's page, left of it the control's: each end says whose.
        better=(f"prefers {control}", f"prefers {drawn[0][0] if len(drawn) == 1 else 'the candidate'}"),
    )
    lines = []
    for arm, read, shares in drawn:
        candidate_share, control_share = shares.share_text(shares.candidate), shares.share_text(shares.control)
        line = f"{arm} {candidate_share}% · {control} {control_share}% (50% is even)"
        if shares.interval_text is not None:
            line += f", {level} interval {shares.interval_text}"
        lines.append(
            DisclosureBlock(section="scorecard", source="comparisons", text=f"{line}; {_primary_reads(read)}.")
        )
    title = f"Share of the judge's preference against {control}"
    return [ChartBlock(section="scorecard", chart="dot_interval", title=title, data=chart), *lines]


def _preference_rows(forms: PageForms, arm: str, read: WriterPairwise | None) -> list[dict[str, TableCell]]:
    """A candidate's rows of the preference table: every pair, then the pairs whose two pages were both finished; one
    row of the candidate alone when its pass was not read."""
    if read is None:
        return [table_row({key: arm if key == "candidate" else None for key, _ in _PREFERENCE_COLUMNS})]
    rows = []
    for pairs, verdict, share in (
        (_EVERY_PAIR, read.primary, read.share),
        (_BOTH_FINISHED, read.finished_only, read.finished_share),
    ):
        reads = _primary_reads(read) if pairs == _EVERY_PAIR else _pairwise_reads(verdict)
        # Both shares and the interval's ends at one precision, the fewest decimals that keep each on its side of
        # none, even and all, and the three apart.
        shares = forms.preference(share, verdict.interval)
        rows.append(
            table_row(
                {
                    "candidate": arm,
                    "pairs": pairs,
                    "reads": reads,
                    "share": None if shares is None else shares.cell(shares.candidate),
                    "control_share": None if shares is None else shares.cell(shares.control),
                    "interval": None if shares is None or shares.ends is None else " to ".join(shares.ends),
                }
            )
        )
    return rows


def _check_row(forms: PageForms, arm: str, read: WriterPairwise | None) -> dict[str, TableCell]:
    """A candidate's row of the judge's checks table; the candidate alone when its pass was not read."""
    if read is None:
        return table_row({key: arm if key == "candidate" else None for key, _ in _CHECK_COLUMNS})
    rate, settle = read.order_consistency, read.cases_to_settle
    spent, cap = forms.money_set([read.cost_usd, read.cap_usd])  # the spend and its part read in one money form
    return table_row(
        {
            "candidate": arm,
            "order_consistency": None
            if rate is None
            else f"{forms.lone(forms.check(rate)).text}% ({read.order_consistent} of {read.both_orders})",
            "first_slot_lean": forms.check_cell(read.first_slot_bias),
            "cannot_tell": forms.count_cell(read.cannot_tell),
            "to_settle": None
            if settle is None
            else f"more than {forms.count(PROJECTION_CASE_CAP).text}"
            if settle > PROJECTION_CASE_CAP
            else forms.count(settle).text,
            "spent": spent.text if read.cap_usd is None else f"{spent.text} of {cap.text}",
        }
    )


def _side_by_side_blocks(
    forms: PageForms,
    control: str,
    pairwise: Mapping[str, WriterPairwise],
    unread: Mapping[str, str],
    *,
    candidates: Sequence[str],
    why: Mapping[str, str],
    launch_line: str | None = None,
) -> list[ReportBlock]:
    """The side-by-side section the page's Scorecard opens with, whose tables the print and ``pairwise.md`` lay out:
    the preference figure and its line per candidate (:func:`_preference_figure`), the one sentence defining its
    terms, the preference table (a row per candidate and the pairs it reads), the judge's
    checks table (a row per candidate), a line for each pass that did not complete, and each candidate's promotion
    reading in its own sentence. A run launched in batches (``launch_line``) says where it stopped launching and what
    each pass did against its rule, so the page says how many cases its readings were read on."""
    level = _level_text(forms, pairwise)
    interval = "Interval (%)" if level is None else f"{level} interval (%)"
    preference = [
        TableColumn(key=key, header=interval if key == "interval" else header) for key, header in _PREFERENCE_COLUMNS
    ]
    rows = [row for arm in candidates for row in _preference_rows(forms, arm, pairwise.get(arm))]
    checks = [_check_row(forms, arm, pairwise.get(arm)) for arm in candidates]
    return [
        *_preference_figure(forms, control, pairwise, candidates),
        DisclosureBlock(section="scorecard", source="comparisons", text=_PAIRWISE_LEAD),
        TableBlock(
            section="scorecard",
            name="side_by_side",
            title=f"Side by side against {control}",
            columns=preference,
            rows=rows,
            order="the candidates in the order the arms were given, every pair before both pages finished",
            total_rows=len(rows),
        ),
        TableBlock(
            section="scorecard",
            name="judge_checks",
            title="The judge's checks",
            columns=[TableColumn(key=key, header=header) for key, header in _CHECK_COLUMNS],
            rows=checks,
            order="the candidates in the order the arms were given",
            total_rows=len(checks),
        ),
        *(
            DisclosureBlock(section="scorecard", source="comparisons", text=f"Pairwise pass of {arm}: {reason}")
            for arm, reason in unread.items()
        ),
        *(
            []
            if launch_line is None
            else [
                DisclosureBlock(section="scorecard", source="comparisons", text=launch_line),
                *(
                    DisclosureBlock(
                        section="scorecard", source="comparisons", text=f"Pairwise pass of {arm}: {read.stopping.line}"
                    )
                    for arm in candidates
                    if (read := pairwise.get(arm)) is not None and read.stopping is not None
                ),
            ]
        ),
        *(
            DisclosureBlock(section="scorecard", source="comparisons", text=f"Promotion of {arm}: {why[arm]}")
            for arm in candidates
            if arm in why
        ),
    ]


def _answer_block(
    forms: PageForms,
    control: str,
    candidates: Sequence[str],
    pairwise: Mapping[str, WriterPairwise],
    unread: Mapping[str, str],
    *,
    decisions: Mapping[str, str],
    why: Mapping[str, str],
    cases: int,
    side: str,
) -> DisclosureBlock:
    """The page's answer, its lead under the question: for each candidate, its promotion reading and the judge's
    side-by-side preference in one plain sentence (``Undecided, so shipped stays: across 7 held-out cases the judge gave
    lead-first 68% of its preference over shipped (50% is even), too few cases to tell from an even split; about 39
    cases would settle it.``), the promotion's own sentence after it where the preference alone does not explain the
    decision. ``cases`` counts the side's cases run; a pass that stopped part-way through them reads its preference on
    the cases up to its stop, and its sentence counts those."""
    sentences = [
        _answer_sentence(
            forms,
            control,
            arm,
            pairwise.get(arm),
            unread.get(arm),
            decisions.get(arm),
            why.get(arm),
            _cases(_read_on(pairwise.get(arm), cases), side),
        )
        for arm in candidates
    ]
    decided = all(decisions.get(arm) in ("promote", "keep") for arm in candidates)
    return DisclosureBlock(
        section="answer", source="verdict", text=" ".join(sentences), tone="decided" if decided else "open"
    )


def _answer_sentence(
    forms: PageForms,
    control: str,
    arm: str,
    read: WriterPairwise | None,
    unread: str | None,
    decision: str | None,
    why: str | None,
    cases: str,
) -> str:
    """One candidate's sentence of the page's answer (:func:`_answer_block`)."""
    if read is None:
        missed = f"the judge's side-by-side pass of {arm} against {control} did not complete: {unread}"
        return why if why is not None else f"{missed[0].upper()}{missed[1:]}."
    primary = read.primary
    improved = primary.outcome == "improved" and primary.reason != "too_few_cases" and read.void is None
    regressed = primary.outcome == "regressed" and primary.reason != "too_few_cases" and read.void is None
    shares = forms.preference(read.share, primary.interval)
    if shares is None:
        preference = f"across {cases} the judge scored no pair of {arm} against {control}"
    else:
        preference = (
            f"across {cases} the judge gave {arm} {shares.share_text(shares.candidate)}% of its preference over "
            f"{control} (50% is even)"
        )
    settle = read.cases_to_settle
    if read.void is not None:
        # The reason carries its own count in parentheses, so it follows a colon, never nested in another pair.
        preference += f", but its two reading orders disagreed too often for that to count: {read.void}"
    elif primary.reason == "too_few_cases":
        preference += ", too few cases to read"
    elif improved or regressed:
        preference += ", enough cases to tell from an even split"
    elif primary.outcome == "not_separated":
        preference += ", too few cases to tell from an even split"
        if settle is not None:
            # A projection that assumes the observed share is the true one: said as conditional and coarse (one
            # significant figure, as a count reads, PageForms.count), since a share on these few cases moves between two
            # recordings of near-identical pages: ``if the share stayed near 52%, roughly 3,000 cases would settle it``.
            near = shares.share_text(shares.candidate) if shares is not None else None
            about = (
                f"more than {forms.count(PROJECTION_CASE_CAP).text}"
                if settle > PROJECTION_CASE_CAP
                else f"roughly {forms.count(_coarse(settle)).text}"
            )
            condition = f"if the share stayed near {near}%, " if near is not None else ""
            preference += f"; {condition}{about} cases would settle it"
    else:
        preference += f", which reads as {primary.outcome.replace('_', ' ')}"
    if decision is None:
        return f"{preference[0].upper()}{preference[1:]}; a promotion is read on the held-out side only."
    lead = _DECISION_LEADS[decision].format(candidate=arm, control=control)
    explained = (
        (decision == "promote" and improved)
        or (decision == "keep" and regressed)
        or (decision == "undecided" and not improved)
    )
    return f"{lead}: {preference}." + ("" if explained or why is None else f" {why}")


def _read_on(read: WriterPairwise | None, cases: int) -> int:
    """The cases a candidate's preference reads: those up to its pass's stop, or every case run."""
    if read is None or read.stopping is None or read.stopping.stopped_at is None:
        return cases
    return read.cases_read


def _coarse(count: int) -> int:
    """A projected count to one significant figure (``2,969`` to ``3,000``, ``44`` to ``40``): the precision a
    projection on a handful of cases supports."""
    if count < 10:
        return count
    step: int = 10 ** (len(str(count)) - 1)
    return (count + step // 2) // step * step


def _cases(n: int, side: str) -> str:
    """The side's cases in words: ``7 held-out cases``, ``1 dev case``."""
    return f"{n} {side} case{'' if n == 1 else 's'}"


def _with_side_by_side(page: CampaignReport, side_by_side: Sequence[ReportBlock]) -> CampaignReport:
    """The campaign's page, which states the writer eval's answer as its lead, with the side-by-side section at the
    head of its Scorecard: each section lays its blocks out in their order, so it leads its section. A page a stored
    report document draws holds no blocks of its own, and is kept as it is."""
    if page.doc_page is not None:
        return page
    blocks = list(page.blocks)
    at = next((index for index, block in enumerate(blocks) if block.section == "scorecard"), len(blocks))
    return page.model_copy(update={"blocks": [*blocks[:at], *side_by_side, *blocks[at:]]})


def _levels(writer: Writer) -> dict[str, str]:
    """A writer's level of each lever a writer eval may vary, as a design axis states it."""
    return {
        SKILL_LEVER: writer.analyst_skill.digest,
        _EFFORT_LEVER: writer.effort or "default",
        _REVISIONS_LEVER: str(writer.revisions),
        _EDITOR_LEVER: writer.editor_model or "default",
        CANDIDATE_MODEL_LEVER: writer.model_id,
    }


def _effort(writer: Writer) -> EffortWord | None:
    """A writer's effort as the overlays type it, which its construction checked."""
    return next((word for word in get_args(EffortWord) if word == writer.effort), None)


def _condition(writer: Writer) -> _Condition:
    """What tells a writer's arm apart: its model, skill digest, revisions, editor's model and effort."""
    return (writer.model_id, writer.analyst_skill.digest, writer.revisions, writer.editor_model, writer.effort)


def _routed(arms: Mapping[str, Writer], turns: Mapping[str, WriterTurns]) -> WriterModelFactory:
    """The host's model factory: each cell's writer, routed to its arm's recordings by every lever and the effort.

    The launcher hands the factory every lever of the arm (its model, skill, revisions and editor's model), and the
    cell's effort after; together they tell the arms apart (no two arms share them all), so they find the arm whose
    recordings the cell replays, whose own factory refuses any other lever. ``turns`` is filled once the cases are
    minted, before any arm launches.
    """

    def factory(launched: WriterArm) -> WriterModels:
        levers = (launched.model, launched.skill.digest, launched.revisions, launched.editor_model)
        by_effort = {writer.effort: arm for arm, writer in arms.items() if _condition(writer)[:4] == levers}
        named = (
            f"{launched.model!r} with skill {launched.skill.digest[:12]}, {launched.revisions} revision(s) and editor "
            f"{launched.editor_model or 'the candidate model'}"
        )
        if not by_effort:
            raise ValueError(f"no arm writes on {named}")

        def at(effort: EffortWord | None) -> TurnModel:
            arm = by_effort.get(effort)
            if arm is None:
                raise ValueError(f"no arm writes on {named} at effort {effort!r}")
            return turns[arm](launched)(effort)

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


def _pageless_runs(host: EvalHost, run_ids: Mapping[str, Sequence[str]], scope_id: str) -> list[str]:
    """The runs that stored no page: none of their results carries a stored page's open findings, a run that stored no
    result at all among them. Blocking."""
    return [
        run_id
        for ids in run_ids.values()
        for run_id in ids
        if not any(
            MUST_FIX_OPEN_MEASURE in result.host_measures
            for result in host.storage.query_eval_results_by_run(run_id, scope_id)
        )
    ]


def _pageless_line(host: EvalHost, run_ids: Mapping[str, Sequence[str]], pageless: Sequence[str], scope_id: str) -> str:
    """Say once that batch runs stored no page, naming each arm by the cases its such runs launched, and what that does
    to the readings. The cases are read off each run's own record, never off its stored results: a run whose job ended
    before its first cell stored one stored no result at all, and its cases are no less unwritten. Blocking."""
    storage = host.storage
    left = set(pageless)
    launched: dict[str, set[str]] = {}
    none: set[str] = set()
    for run_id in (run_id for ids in run_ids.values() for run_id in ids):
        none |= {
            result.test_case_id
            for result in storage.query_eval_results_by_run(run_id, scope_id)
            if MUST_FIX_OPEN_MEASURE not in result.host_measures
        }
        if run_id in left:
            run = storage.load_eval_run(run_id, scope_id)
            if run is None:
                raise NotFoundError("writer eval run", run_id, remedy="restore the run, then read the eval again")
            launched[run_id] = set(run.test_case_ids)
            none |= launched[run_id]
    who = " and ".join(
        f"{len(cases)} of {arm}'s"
        for arm, ids in run_ids.items()
        if (cases := {case for run_id in ids if run_id in left for case in launched[run_id]})
    )
    return (
        f"{len(none)} case{'' if len(none) == 1 else 's'} stored no page on at least one arm. A batch run that stored "
        f"none, {who}, is left out of the judged readings, since no judge was asked of it; its unwritten reports count "
        "against its arm in Report finished and side by side, where a missing page loses its pair, or ties it when "
        "both are."
    )


def _over_budget_with_no_page(
    host: OpsHost, scope_id: str, run_ids: Mapping[str, Sequence[str]], budget: str, floors: Mapping[str, float]
) -> list[str]:
    """For each arm whose reports ended over budget with no page, how many did, and what the reports of its run that
    stored a page needed under the cap: what they spent, plus one turn's ceiling (the arm's analyst first-turn floor,
    where it was priced), so the next budget is sized from them. Blocking."""
    storage = host.eval_host.storage
    forms = PageForms.off_page()
    said: list[str] = []
    for arm, ids in run_ids.items():
        results = [result for run_id in ids for result in storage.query_eval_results_by_run(run_id, scope_id)]
        # A stored page carries its open findings' count; a report that stored none carries none.
        paged = [result for result in results if MUST_FIX_OPEN_MEASURE in result.host_measures]
        blank = [
            result
            for result in results
            if result.host_measures.get(ENDING_MEASURE) == "over_budget"
            and MUST_FIX_OPEN_MEASURE not in result.host_measures
        ]
        if not blank:
            continue
        spent = sorted(cost for result in paged if (cost := _writer_spend(result)) is not None)
        if spent:
            most = spent[-1]
            floor = floors.get(arm)
            needed = (
                "that plus one turn's ceiling"
                if floor is None
                else f"about {forms.money_total(most + floor).text}, that plus one turn's ceiling of at least "
                f"{forms.money_total(floor).text}"
            )
            low, high = forms.money_total(spent[0]).text, forms.money_total(most).text
            span = low if low == high else f"{low} to {high}"
            evidence = (
                f"the {len(spent)} that stored a page under it spent {span}{' each' if len(spent) > 1 else ''}, so "
                f"under the cap the costliest needed {needed}"
            )
        elif paged:
            evidence = f"the {len(paged)} that stored a page under it were not all priced"
        else:
            evidence = "none stored a page, so this run shows no budget that admits one"
        said.append(
            f"{arm}: {len(blank)} of {len(results)} report(s) ended over their {budget} budget with no page; {evidence}"
        )
    return said


def _writer_spend(result: EvalResult) -> float | None:
    """What a result's report cost its writer (the candidate's usage), or ``None`` when any of it went unpriced."""
    total = 0.0
    for usage in result.usage:
        if usage.role != "candidate":
            continue
        if usage.cost_usd is None:
            return None
        total += usage.cost_usd
    return total


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
    return [f"{len(naming)} case(s) name example results stored only where they were frozen"]


__all__ = ["Writer", "WriterEval", "WriterPairwise", "WriterResult"]
