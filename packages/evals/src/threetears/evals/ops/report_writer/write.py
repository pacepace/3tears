"""Writing a report: from a campaign, an analysis or a frozen snapshot to a stored report document.

:func:`report_write` freezes the evidence first (``report_snapshot_freeze``), unless it is handed a snapshot already
frozen, so neither tool it is handed writes: the ``evals_analyst`` tool on the analyst's cut and on the editor's, each
bound to its host and caller by the action that calls this (``actions/report_writing.py``'s ``write_report``). It
reads the snapshot's report input, loads the skill and runs the writing loop
(:func:`~threetears.evals.ops.report_writer.loop.write_rounds`), whose submissions are checked by the one checker
(:func:`~threetears.evals.analysis.report_doc_store.doc_verdicts`). It stores what the loop's ending stores through the
store gate (:func:`~threetears.evals.analysis.report_doc_store.store_report_doc`), which checks it again, and builds
the stored page with its made line (:func:`~threetears.evals.analysis.report_doc_store.made_line`).

**A paid run never loses its record.** The run's generation record in memory (the tally, the reviews, and every
session held inline) is on the result whatever the ending, and on :class:`ReportNotStored` when the store gate or the
store refuses the document. What is stored is the same record naming each session by ref: the report's id is minted
before anything is written, each session is stored as its own document under the id that report and its name give,
and the document's ``generation`` holds each session's digest, turns and ref, never a message.

**A paid run never loses its gaps either.** Every ``capability_gap`` call the write's sessions made (the write's mount
stores none itself), every figure key that did not resolve and sent a draft back, and every rubric item a review scored
below 2 is stored as an ``eval_analyst_gap`` when the write ends, on every ending
(:func:`~threetears.evals.ops.report_writer.gaps.write_gaps`): after the report, naming it, when the report is stored;
naming no report when it is not, :class:`ReportNotStored` included. Each is on the result too
(:attr:`ReportWritten.gaps`). A store that refuses a gap fails neither the report nor the result: the failure is
recorded on this module's logger at error level and on the result (:attr:`ReportWritten.gaps_not_stored`), beside the
records to store again. A write that crashes before its end loses its gaps, as it loses the write.

**Keys.** The caller passes the model (plan §4.6): a host builds ``Model(id, provider="anthropic", api_key=...)``
from the key it holds. Nothing here reads an environment variable.

**Under a dollar budget, a run that cannot be priced never starts.** A dollar budget (:class:`ReportBudget`) admits
every turn on its price ceiling, so before anything is frozen or sent the model is asked whether its turns can be
priced (:meth:`~threetears.evals.ops.report_writer.loop.TurnModel.turn_pricing_refusal`); one that cannot is refused
by name, with nothing frozen, sent or spent. The budget's own refusal of a turn it cannot bound stays as the backstop.
A token and turn budget admits turns by their tokens and the turns sent, so it asks for no price: a model nothing
prices writes its report under it, tallied unpriced.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report_doc_checks import DocVerdict
from threetears.evals.analysis.report_doc_store import doc_verdicts, generation_record, made_line, store_report_doc
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.analyst_gap import EvalAnalystGap
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.report_doc import EvalReportDoc, ReportDoc, new_report_doc_id
from threetears.evals.ops.analyst import report_snapshot_freeze, snapshot_input
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.report_writer.gaps import CheckedDrafts, write_gaps
from threetears.evals.ops.report_writer.loop import TurnModel, WriterEnding, WriterLimits, WritingOutcome, write_rounds
from threetears.evals.ops.report_writer.session import ToolMount
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.schema.logs import contain, get_logger

#: The names the report writer's entry point (``actions/report_writing.py``) calls and returns.
__all__ = ["ReportNotStored", "ReportWritten", "report_write"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReportWritten:
    """What writing a report came to.

    :ivar ending: How the writing loop ended (plan §4.4).
    :ivar snapshot_id: The snapshot it was written from.
    :ivar stored: The stored document; ``None`` for an ending that stores nothing.
    :ivar outcome: The loop's outcome: the reviews, the rounds, a refusal's faults, why it ended.
    :ivar generation: The run's generation record in memory (the tally, the reviews, every session held inline under
        its name's ``session``, with no ``ref``), kept whatever the ending. A stored document's own ``generation``
        names each session by ref instead (``stored.generation``), and its sessions are stored beside it.
    :ivar page: The stored document's page as a reader reads it, built strict with its open findings in Method and
        its made line (:func:`~threetears.evals.analysis.report_doc_store.made_line`); ``None`` when nothing is stored.
    :ivar gaps: Every gap and notice the write noted (the module docstring), whatever the ending, each naming the
        stored report or none.
    :ivar gaps_not_stored: Why the gaps were not stored, when they were not (the store refused them, or they could not
        be read from the sessions, which :attr:`generation` still holds); ``None`` when every one is stored. The records
        read are :attr:`gaps`, to store again (``EvalStorage.save_analyst_gap`` keeps one per id).
    """

    ending: WriterEnding
    snapshot_id: str
    stored: EvalReportDoc | None
    outcome: WritingOutcome
    generation: dict[str, Any]  # Any: a JSON object
    page: DocPage | None
    gaps: tuple[EvalAnalystGap, ...] = ()
    gaps_not_stored: str | None = None


class ReportNotStored(Exception):
    """A report the writing loop ended with a document to store, and the store gate or the store refused.

    The run was paid for, so its record rides on the refusal: :attr:`written` is what :func:`report_write` would have
    returned, with nothing stored. The cause is the exception this one was raised from. A store that failed part way
    may have written some of the report's sessions under :attr:`report_doc_id` before it failed; no stored document
    names them, and ``EvalStorage.delete_report_doc(report_doc_id, scope_id)`` removes them without touching anything
    else in the scope.

    :ivar written: The run's outcome and generation record, ``stored`` and ``page`` ``None``.
    :ivar report_doc_id: The id minted for the report, which any session written before the failure is stored under.
    """

    def __init__(self, message: str, *, written: ReportWritten, report_doc_id: str) -> None:
        """Say why the report was not stored, and carry the run's record and the id it would have been stored under.

        :param message: What failed.
        :ptype message: str
        :param written: The run's outcome and generation record.
        :ptype written: ReportWritten
        :param report_doc_id: The id minted for the report.
        :ptype report_doc_id: str
        """
        super().__init__(message)
        self.written = written
        self.report_doc_id = report_doc_id


def _checker(report_input: ReportInput) -> Callable[[ReportDoc], list[DocVerdict]]:
    """The analyst's checker on the snapshot: the one checker, the store gate's."""

    def check(doc: ReportDoc) -> list[DocVerdict]:
        return doc_verdicts(doc, report_input)

    return check


def _stored_as(ending: WriterEnding) -> Literal["stored", "stored_with_findings", "over_budget"] | None:
    """The ending a stored document records, for an ending that stores one; ``None`` for one that stores nothing."""
    if ending == "stored" or ending == "stored_with_findings" or ending == "over_budget":
        return ending
    return None


async def report_write(
    host: OpsHost,
    scope_id: str,
    *,
    created_by: str,
    model: TurnModel,
    budget: ReportBudget,
    analyst_mount: ToolMount,
    editor_mount: ToolMount,
    snapshot_id: str | None = None,
    campaign_id: str | None = None,
    analysis_id: str | None = None,
    withhold: Collection[str] = (),
    limits: WriterLimits | None = None,
    tally: Tally | None = None,
) -> ReportWritten:
    """Write a report over one frozen snapshot, and store what its ending stores (the module docstring).

    :param host: The host whose store holds the evidence, the snapshot and the report.
    :ptype host: OpsHost
    :param scope_id: The scope the report is written in.
    :ptype scope_id: str
    :param created_by: Who asks for the report, as the host names the caller.
    :ptype created_by: str
    :param model: The model every session runs on (``Model(id, provider="anthropic", api_key=...)``).
    :ptype model: TurnModel
    :param budget: The report's budget (dollars, or tokens and turns), which admits every turn.
    :ptype budget: ReportBudget
    :param analyst_mount: The ``evals_analyst`` tool on the analyst's cut, read-only, bound to the host and caller.
    :ptype analyst_mount: ToolMount
    :param editor_mount: The tool on the editor's cut, bound the same way.
    :ptype editor_mount: ToolMount
    :param snapshot_id: A snapshot already frozen; exactly one of it, ``campaign_id`` and ``analysis_id``.
    :ptype snapshot_id: str | None
    :param campaign_id: A campaign to freeze and write the report on.
    :ptype campaign_id: str | None
    :param analysis_id: A stored analysis to freeze and write the report on.
    :ptype analysis_id: str | None
    :param withhold: Exemplar notes, by file name, the analyst is not given (a blind run's own campaign's note).
    :ptype withhold: Collection[str]
    :param limits: What the loop is held to; plan §4.3's defaults when ``None``.
    :ptype limits: WriterLimits | None
    :param tally: The tally every turn is counted in, when the caller keeps it (a job that must say what a run it
        stopped had spent); a fresh one when ``None``.
    :ptype tally: Tally | None
    :return: How it ended, the stored document if any, and the generation record.
    :rtype: ReportWritten
    :raises ValueError: Not exactly one of the sources named; under a dollar budget, a model whose turns cannot be
        bounded, refused with the budget's lead-in and the model's own pricing refusal; or one the model refuses for
        its route or settings, raised as the model words it. Each before anything is frozen or sent.
    :raises ReportNotStored: The loop ended with a document to store, and the store gate or the store refused it;
        the run's record rides on it (:attr:`ReportNotStored.written`), raised from the cause.
    :raises EvalServiceError: The freeze or the snapshot's read refused (no such campaign, analysis or snapshot).
    """
    named = [
        name
        for name, given in (("snapshot_id", snapshot_id), ("campaign_id", campaign_id), ("analysis_id", analysis_id))
        if given is not None
    ]
    if len(named) != 1:
        raise ValueError(
            f"a report is written from one source: name a snapshot_id, a campaign_id or an analysis_id, not {named or 'none'}"
        )
    if budget.usd is not None and (unpriced := await model.turn_pricing_refusal()) is not None:
        raise ValueError(
            f"the report was not started, and nothing was frozen, sent or spent: its {budget.spelled} budget admits "
            f"each turn on its price ceiling, and {unpriced}; a model nothing prices is written under a token and turn "
            "budget (ReportBudget(tokens=, turns=))"
        )
    if snapshot_id is None:
        frozen = await report_snapshot_freeze(
            host, scope_id, campaign_id=campaign_id, analysis_id=analysis_id, created_by=created_by
        )
        snapshot_id = frozen.snapshot_id
    report_input = await snapshot_input(host, scope_id, snapshot_id)
    eval_host = host.eval_host
    snapshot = await run_blocking(
        eval_host.blocking_executor, eval_host.storage.load_report_snapshot, snapshot_id, scope_id
    )
    skill = load_skill()
    analyst_skill, editor_skill = skill.analyst(withhold=withhold), skill.editor()
    tally = tally if tally is not None else Tally()
    drafts = CheckedDrafts(_checker(report_input))
    outcome = await write_rounds(
        model=model,
        snapshot=snapshot_id,
        report_input=report_input,
        analyst_mount=analyst_mount,
        editor_mount=editor_mount,
        analyst_skill=analyst_skill,
        editor_skill=editor_skill,
        check=drafts,
        tally=tally,
        budget=budget,
        limits=limits,
    )
    written: dict[str, Any] = {  # Any: the record's parts, as JSON
        "model": model.model_name,
        "tally": tally.to_json(),
        "skill": {"analyst": analyst_skill.digest, "editor": editor_skill.digest, "package": skill.digest},
        "reviews": [review.to_json() for review in outcome.reviews],
        "sessions": outcome.sessions,
    }
    unstored = ReportWritten(
        ending=outcome.ending,
        snapshot_id=snapshot_id,
        stored=None,
        outcome=outcome,
        generation=generation_record(report_doc_id=None, **written),
        page=None,
    )

    async def noted(report_doc_id: str | None, written: ReportWritten) -> ReportWritten:
        # The write's gaps, naming the report it stored or none, stored and carried on its result. Neither reading
        # them nor storing them may cost the paid run its record, so a failure of either is recorded and returned.
        try:
            gaps = write_gaps(
                scope_id=scope_id,
                snapshot_id=written.snapshot_id,
                report_doc_id=report_doc_id,
                model=model.model_name,
                created_by=created_by,
                outcome=outcome,
                tally=written.generation["tally"],
                drafts=drafts,
                report_input=report_input,
            )
        except Exception as exc:
            contain(exc, log=log, what="the report write's gaps could not be read from its sessions", level="error")
            return replace(written, gaps_not_stored=f"the gaps could not be read ({type(exc).__name__}: {exc})")
        refused = await _store_gaps(host, gaps)
        return replace(written, gaps=tuple(gaps), gaps_not_stored=refused)

    ending = _stored_as(outcome.ending)
    if outcome.doc is None or ending is None:
        return await noted(None, unstored)
    report_doc_id = new_report_doc_id()
    try:
        stored = await run_blocking(
            eval_host.blocking_executor,
            store_report_doc,
            eval_host.storage,
            report_doc_id=report_doc_id,
            scope_id=scope_id,
            snapshot_id=snapshot_id,
            campaign_id=snapshot.campaign_id if snapshot is not None else None,
            doc=outcome.doc,
            report_input=report_input,
            ending=ending,
            findings=outcome.findings,
            generation=generation_record(report_doc_id=report_doc_id, **written),
            sessions=outcome.sessions,
            created_by=created_by,
        )
    except Exception as exc:
        # Re-raised with the run's record on it: every turn was paid for, so the tally, the reviews and the sessions
        # reach the caller beside the failure, never instead of it.
        raise ReportNotStored(
            f"the report ended {outcome.ending} and was not stored ({type(exc).__name__}: {exc}); its generation "
            "record (tally, reviews, sessions) is this error's .written, with its gaps, stored naming no report, and "
            f"any session the store wrote before it failed is stored under report {report_doc_id} (this error's "
            ".report_doc_id)",
            written=await noted(None, unstored),
            report_doc_id=report_doc_id,
        ) from exc
    page = build(outcome.doc, report_input, strict=True, findings=outcome.findings, made=made_line(stored))
    return await noted(report_doc_id, replace(unstored, stored=stored, page=page))


async def _store_gaps(host: OpsHost, gaps: Sequence[EvalAnalystGap]) -> str | None:
    """Store each gap where none is stored under its id; why the store refused, or ``None`` when it took them all.

    A refusal is recorded at error level for the operator and returned for the result, never raised: the run was paid
    for, and its record and its report outlive a gap the store would not take.
    """
    eval_host = host.eval_host
    try:
        for gap in gaps:
            await run_blocking(eval_host.blocking_executor, eval_host.storage.save_analyst_gap, gap)
    except Exception as exc:
        # Recorded, and returned on the result beside the records: the write's report and record stand without them.
        contain(exc, log=log, what="the report write's gaps were not stored; they ride on its result", level="error")
        return f"{type(exc).__name__}: {exc}"
    return None
