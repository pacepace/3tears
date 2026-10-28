"""The report writer's entry point: from a campaign, an analysis or a frozen snapshot to a stored report document.

:func:`write_report` mounts the ``evals_analyst`` tool twice from the analyst's actions, as an MCP client would mount
it: the analyst's cut read-only (``analyst_tools(writes=False)``) and the editor's (``analyst_tools(role="editor")``),
each bound to the host and the caller its calls run as, each described by its own description and the catalogue's
primer (:func:`~threetears.evals.actions.catalogue.catalogue_primer`), and each a read of the frozen snapshot, so a
stored session may drop a result's body and call the tool again for it. Both mount the gap note that stores nothing
(``analyst_actions(gaps_kept_by="write")``): the write stores each gap its sessions noted when it ends. The writing itself is the operation's
(:func:`~threetears.evals.ops.report_writer.write.report_write`): it freezes the evidence first unless handed a
snapshot already frozen, so neither tool writes, runs the writing loop, and stores what its ending stores through the
store gate. Its result and its refusal (``ReportWritten``, ``ReportNotStored``) are the operation's, named here too.

**Keys.** The caller passes the model (plan §4.6): a host builds ``Model(id, provider="anthropic", api_key=...)``
from the key it holds. Nothing here reads an environment variable.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from threetears.evals.actions.analyst import analyst_actions, analyst_tools
from threetears.evals.actions.catalogue import ActionCatalogue, DEFAULT_PREFIX, Caller, MountedTool, catalogue_primer
from threetears.evals.ops.analysis import ReportWriter, ReportWriterCall
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.report_writer.loop import TurnModel, WriterLimits
from threetears.evals.ops.report_writer.session import ToolOutcome
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.ops.report_writer.write import ReportNotStored as ReportNotStored
from threetears.evals.ops.report_writer.write import ReportWritten as ReportWritten
from threetears.evals.ops.report_writer.write import report_write

#: What ``analysis_generate``'s action hands its job (actions/engine.py). ``write_report`` stays out until a caller
#: outside this module imports it; ``tests/test_report_writer_exports.py`` lists it as awaiting one.
__all__ = ["campaign_report_writer"]


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


def _mount(host: OpsHost, caller: Caller, *, prefix: str, role: Literal["analyst", "editor"]) -> _HostMount:
    """The ``evals_analyst`` tool on ``role``'s cut, read-only, bound to ``host`` and ``caller``."""
    (spec,) = analyst_tools(prefix, writes=False, role=role)
    # The analyst's actions alone, which are every action its cut holds: the engine's catalogue adds actions no
    # analyst tool mounts, and the action that starts a generation (actions/engine.py) hands its job this module's
    # writer, so this module reads none of it. Its gap note stores nothing: the write stores every gap its sessions
    # noted when it ends, with the session each came from (report_write), so each is stored once.
    tool = ActionCatalogue(analyst_actions(gaps_kept_by="write")).mount(spec)
    if not tool.hints.read_only:
        raise ValueError(
            f"the {role}'s tool {tool.name!r} carries an action that writes; the report writer mounts reads"
        )
    return _HostMount(tool, host, caller, catalogue_primer([tool]))


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
    tally: Tally | None = None,
) -> ReportWritten:
    """Write a report over one frozen snapshot, and store what its ending stores (the module docstring).

    :param host: The host whose store holds the evidence, the snapshot and the report.
    :ptype host: OpsHost
    :param caller: Who asks for the report, and the scope it is written in.
    :ptype caller: Caller
    :param model: The model every session runs on (``Model(id, provider="anthropic", api_key=...)``).
    :ptype model: TurnModel
    :param budget: The report's budget (dollars, or tokens and turns), which admits every turn.
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
    :param tally: The tally every turn is counted in, when the caller keeps it; a fresh one when ``None``.
    :ptype tally: Tally | None
    :return: How it ended, the stored document if any, and the generation record.
    :rtype: ReportWritten
    :raises ValueError: Not exactly one of the sources named.
    :raises ReportNotStored: The loop ended with a document to store, and the store gate or the store refused it;
        the run's record rides on it (:attr:`ReportNotStored.written`), raised from the cause.
    :raises EvalServiceError: The freeze or the snapshot's read refused (no such campaign, analysis or snapshot).
    """
    return await report_write(
        host,
        caller.scope_id,
        created_by=caller.identity,
        model=model,
        budget=budget,
        analyst_mount=_mount(host, caller, prefix=prefix, role="analyst"),
        editor_mount=_mount(host, caller, prefix=prefix, role="editor"),
        snapshot_id=snapshot_id,
        campaign_id=campaign_id,
        analysis_id=analysis_id,
        withhold=withhold,
        limits=limits,
        tally=tally,
    )


def campaign_report_writer(caller: Caller, *, prefix: str = DEFAULT_PREFIX) -> ReportWriter:
    """The report writer ``analysis_generate``'s job runs, bound to the caller who started it: :func:`write_report` on
    the campaign, which it freezes first.

    :param caller: Who started the generation, and the scope it runs in: the tool calls of both sessions run as them,
        and the stored document names them.
    :ptype caller: Caller
    :param prefix: The host's prefix for its eval tools.
    :ptype prefix: str
    :return: The writer.
    :rtype: ReportWriter
    """

    async def write(
        host: OpsHost,
        *,
        model: TurnModel,
        budget: ReportBudget,
        campaign_id: str,
        limits: WriterLimits,
        tally: Tally,
    ) -> ReportWritten:
        return await write_report(
            host,
            caller,
            model=model,
            budget=budget,
            campaign_id=campaign_id,
            limits=limits,
            prefix=prefix,
            tally=tally,
        )

    # Checked against the call the job makes, then handed over opaque (ReportWriter's docstring says why).
    typed: ReportWriterCall = write
    return cast("ReportWriter", typed)
