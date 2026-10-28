"""The report writer's entry point: from a campaign, an analysis or a frozen snapshot to a stored report document.

:func:`write_report` freezes the evidence first (``report_snapshot_freeze``), unless it is handed a snapshot already
frozen, so neither tool it mounts writes. It mounts the ``evals_analyst`` tool twice from the engine's catalogue, as
an MCP client would mount it: the analyst's cut read-only (``analyst_tools(writes=False)``) and the editor's
(``analyst_tools(role="editor")``), each bound to the host and the caller its calls run as, each described by its own
description and the catalogue's primer (:func:`~threetears.evals.actions.catalogue.catalogue_primer`), and each a
read of the frozen snapshot, so a stored session may drop a result's body and call the tool again for it. It then
runs the writing loop (:func:`~threetears.evals.ops.report_writer.loop.write_rounds`), whose submissions are checked
by the one checker (:func:`~threetears.evals.analysis.report_doc_store.doc_verdicts`), and stores what the loop's
ending stores through the store gate (:func:`~threetears.evals.analysis.report_doc_store.store_report_doc`), which
checks it again.

**Keys.** The caller passes the model (plan §4.6): a host builds ``Model(id, provider="anthropic", api_key=...)``
from the key it holds. Nothing here reads an environment variable.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from threetears.evals.actions.analyst import analyst_tools
from threetears.evals.actions.catalogue import DEFAULT_PREFIX, Caller, MountedTool, catalogue_primer
from threetears.evals.actions.engine import eval_catalogue
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.report_doc_checks import DocVerdict
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report_doc_store import doc_verdicts, generation_record, made_line, store_report_doc
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.report_doc import EvalReportDoc, ReportDoc
from threetears.evals.ops.analyst import report_snapshot_freeze, snapshot_input
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.report_writer.analyst import InvariantCheck
from threetears.evals.ops.report_writer.loop import TurnModel, WriterEnding, WriterLimits, WritingOutcome, write_rounds
from threetears.evals.ops.report_writer.session import ToolOutcome
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally

#: Empty until the report writer has a production caller outside this module: the analysis job of plan §9 (W4), which
#: a host's job factory runs. ``tests/test_report_writer_exports.py`` lists the names awaiting it.
__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _HostMount:
    """A mounted catalogue tool as a session's :class:`~threetears.evals.ops.report_writer.session.ToolMount`: bound
    to the host and the caller every call runs as, and described with the catalogue's primer."""

    tool: MountedTool
    host: OpsHost
    caller: Caller
    primer: str

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def description(self) -> str:
        return f"{self.tool.description}\n\n{self.primer}"

    @property
    def rederivable(self) -> bool:
        # Every action it carries reads the frozen snapshot: a result is a function of the call's arguments.
        return self.tool.hints.read_only

    def input_schema(self) -> dict[str, Any]:  # Any: a JSON schema
        return self.tool.input_schema()

    async def call(self, arguments: Mapping[str, Any]) -> ToolOutcome:  # Any: the model's JSON arguments
        return await self.tool.call(arguments, host=self.host, caller=self.caller)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReportWritten:
    """What writing a report came to.

    :ivar ending: How the writing loop ended (plan §4.4).
    :ivar snapshot_id: The snapshot it was written from.
    :ivar stored: The stored document; ``None`` for an ending that stores nothing.
    :ivar outcome: The loop's outcome: the reviews, the rounds, a refusal's faults, why it ended.
    :ivar generation: The generation record (the tally, the reviews, every session), kept whatever the ending.
    :ivar page: The stored document's page as a reader reads it, built strict with its open findings in Method and
        its made line (:func:`~threetears.evals.analysis.report_doc_store.made_line`); ``None`` when nothing is stored.
    """

    ending: WriterEnding
    snapshot_id: str
    stored: EvalReportDoc | None
    outcome: WritingOutcome
    generation: dict[str, Any]  # Any: a JSON object
    page: DocPage | None


class ReportNotStored(Exception):
    """A report the writing loop ended with a document to store, and the store gate or the store refused.

    The run was paid for, so its record rides on the refusal: :attr:`written` is what :func:`write_report` would have
    returned, with nothing stored. The cause is the exception this one was raised from.

    :ivar written: The run's outcome and generation record, ``stored`` and ``page`` ``None``.
    """

    def __init__(self, message: str, *, written: ReportWritten) -> None:
        """Say why the report was not stored, and carry the run's record.

        :param message: What failed.
        :ptype message: str
        :param written: The run's outcome and generation record.
        :ptype written: ReportWritten
        """
        super().__init__(message)
        self.written = written


def _mount(host: OpsHost, caller: Caller, *, prefix: str, role: Literal["analyst", "editor"]) -> _HostMount:
    """The ``evals_analyst`` tool on ``role``'s cut, read-only, bound to ``host`` and ``caller``."""
    (spec,) = analyst_tools(prefix, writes=False, role=role)
    tool = eval_catalogue().mount(spec)
    if not tool.hints.read_only:
        raise ValueError(
            f"the {role}'s tool {tool.name!r} carries an action that writes; the report writer mounts reads"
        )
    return _HostMount(tool, host, caller, catalogue_primer([tool]))


def _checker(report_input: ReportInput) -> InvariantCheck:
    """The analyst's checker on the snapshot: the one checker, the store gate's."""

    def check(doc: ReportDoc) -> list[DocVerdict]:
        return doc_verdicts(doc, report_input)

    return check


def _stored_as(ending: WriterEnding) -> Literal["stored", "stored_with_findings", "over_budget"] | None:
    """The ending a stored document records, for an ending that stores one; ``None`` for one that stores nothing."""
    if ending == "stored" or ending == "stored_with_findings" or ending == "over_budget":
        return ending
    return None


async def write_report(
    host: OpsHost,
    caller: Caller,
    *,
    model: TurnModel,
    budget: ReportBudget,
    snapshot_id: str | None = None,
    campaign_id: str | None = None,
    analysis_id: str | None = None,
    withhold: Collection[str] = (),
    limits: WriterLimits | None = None,
    prefix: str = DEFAULT_PREFIX,
) -> ReportWritten:
    """Write a report over one frozen snapshot, and store what its ending stores (the module docstring).

    :param host: The host whose store holds the evidence, the snapshot and the report.
    :ptype host: OpsHost
    :param caller: Who asks for the report, and the scope it is written in.
    :ptype caller: Caller
    :param model: The model every session runs on (``Model(id, provider="anthropic", api_key=...)``).
    :ptype model: TurnModel
    :param budget: The report's dollar budget, which admits every turn.
    :ptype budget: ReportBudget
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
    :param prefix: The host's prefix for its eval tools.
    :ptype prefix: str
    :return: How it ended, the stored document if any, and the generation record.
    :rtype: ReportWritten
    :raises ValueError: Not exactly one of the sources named.
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
    if snapshot_id is None:
        frozen = await report_snapshot_freeze(
            host, caller.scope_id, campaign_id=campaign_id, analysis_id=analysis_id, created_by=caller.identity
        )
        snapshot_id = frozen.snapshot_id
    report_input = await snapshot_input(host, caller.scope_id, snapshot_id)
    eval_host = host.eval_host
    snapshot = await run_blocking(
        eval_host.blocking_executor, eval_host.storage.load_report_snapshot, snapshot_id, caller.scope_id
    )
    skill = load_skill()
    analyst_skill, editor_skill = skill.analyst(withhold=withhold), skill.editor()
    tally = Tally()
    outcome = await write_rounds(
        model=model,
        snapshot=snapshot_id,
        report_input=report_input,
        analyst_mount=_mount(host, caller, prefix=prefix, role="analyst"),
        editor_mount=_mount(host, caller, prefix=prefix, role="editor"),
        analyst_skill=analyst_skill,
        editor_skill=editor_skill,
        check=_checker(report_input),
        tally=tally,
        budget=budget,
        limits=limits,
    )
    generation = generation_record(
        model=model.model_name,
        tally=tally.to_json(),
        skill={"analyst": analyst_skill.digest, "editor": editor_skill.digest, "package": skill.digest},
        reviews=[review.to_json() for review in outcome.reviews],
        sessions=outcome.sessions,
    )
    unstored = ReportWritten(
        ending=outcome.ending, snapshot_id=snapshot_id, stored=None, outcome=outcome, generation=generation, page=None
    )
    ending = _stored_as(outcome.ending)
    if outcome.doc is None or ending is None:
        return unstored
    try:
        stored = await run_blocking(
            eval_host.blocking_executor,
            store_report_doc,
            eval_host.storage,
            scope_id=caller.scope_id,
            snapshot_id=snapshot_id,
            campaign_id=snapshot.campaign_id if snapshot is not None else None,
            doc=outcome.doc,
            report_input=report_input,
            ending=ending,
            findings=outcome.findings,
            generation=generation,
            created_by=caller.identity,
        )
    except Exception as exc:
        # Re-raised with the run's record on it: every turn was paid for, so the tally, the reviews and the sessions
        # reach the caller beside the failure, never instead of it.
        raise ReportNotStored(
            f"the report ended {outcome.ending} and was not stored ({type(exc).__name__}: {exc}); its generation "
            "record (tally, reviews, sessions) is this error's .written",
            written=unstored,
        ) from exc
    page = build(outcome.doc, report_input, strict=True, findings=outcome.findings, made=made_line(stored))
    return replace(unstored, stored=stored, page=page)
