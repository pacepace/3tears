"""``analyse``: the report writer's report over one quick-path campaign, replayed from its recording or recorded once.

:func:`analyse` freezes the campaign's evidence (``report_snapshot_freeze``, through the result's host read as
:meth:`OpsHost.reading <threetears.evals.ops.host.OpsHost.reading>`), runs
:func:`~threetears.evals.actions.report_writing.write_report` over the snapshot in the result's scope, and writes the
stored document's page to ``.evals/reports/<name>/analysis.html``. Nothing is checked about the campaign first: a
world, classifier or judged campaign is frozen and read like any other, and the freeze and the analyst's tools refuse
what they cannot read, each with its own remedy.

**The model is a recording.** Every session of the report goes to one file, routed as a writer arm's are
(:func:`~threetears.evals.harness.writer_turns.writer_turns`), under the shipped skill's digest, the model, its effort,
the revisions and the snapshot, so a recording made under one skill or model, or over other evidence, never replays:

    ``<recording>/shipped-<skill12>-<model>-default-r<revisions>/<name>-<snapshot12>.jsonl``

The snapshot's id is derived from the campaign, so a recording replays over the campaign it was made on, and a run
filed again as a new campaign is written live (the key), never from another campaign's recording.

- **A file that exists is replayed, with no key**, through
  :meth:`RecordedTurns.replay <threetears.evals.testing.RecordedTurns.replay>`, never through ``Model``.
- **A missing file is recorded when ``ANTHROPIC_API_KEY`` is set**, and refused when it is not: the refusal
  (:class:`~threetears.evals.testing.RecordingRefused`) names the file and the variable.
- **Drift is printed, never silent.** A replay that leaves recorded lines unasked prints one line per session, naming
  it and the file: the code now stops sooner than it did when it was recorded.

A turn cap (:data:`TURN_CAP`) rides on the dollar budget: dollars bound what a report spends, never how long it runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from threetears.evals.actions.catalogue import Caller
from threetears.evals.actions.report_writing import ReportWritten, write_report
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.analyst import report_snapshot_freeze
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.ops.report_writer.writer_contract import WriterArm
from threetears.evals.quick.files import report_dir
from threetears.evals.quick.result import CREATED_BY
from threetears.evals.harness.writer_turns import WriterTurns, writer_turns

#: The page the writer's report is written to, in the eval's report directory.
ANALYSIS_PAGE: Final = "analysis.html"

#: Where the recordings sit by default: beside the eval file being loaded, else under the working directory.
DEFAULT_RECORDING: Final = "recordings/report"

#: What one report may spend by default, in US dollars.
DEFAULT_BUDGET_USD: Final = 3.0

#: The most turns one report may send, beside its dollars: what bounds how long it runs.
TURN_CAP: Final = 40

#: The label the recording directory leads with: the writer runs the shipped skill.
_ARM: Final = "shipped"


@runtime_checkable
class Filed(Protocol):
    """A run filed as a campaign: what the writer reads, and where.

    :class:`~threetears.evals.quick.result.Result` is one.
    """

    @property
    def name(self) -> str:
        """What titles the campaign, and names its report's directory."""
        ...

    @property
    def host(self) -> EvalHost:
        """The host whose store holds the campaign."""
        ...

    @property
    def scope_id(self) -> str:
        """The scope the campaign is stored in."""
        ...

    @property
    def campaign_id(self) -> str:
        """The campaign."""
        ...


def recording_path(
    name: str, model: str, snapshot_id: str, *, recording: str | os.PathLike[str] = DEFAULT_RECORDING
) -> Path:
    """The file the writer's sessions on the eval named ``name`` replay from, or are recorded onto, on ``model``, over
    the snapshot ``snapshot_id`` (:attr:`ReportWritten.snapshot_id <threetears.evals.actions.report_writing.ReportWritten>`).

    :param name: The eval's name.
    :ptype name: str
    :param model: The writer's model id.
    :ptype model: str
    :param snapshot_id: The snapshot the report is written over.
    :ptype snapshot_id: str
    :param recording: The directory the recordings sit under, read as :func:`analyse` reads it.
    :ptype recording: str | os.PathLike[str]
    :return: ``<recording>/shipped-<skill12>-<model>-default-r<revisions>/<name>-<snapshot12>.jsonl``.
    :rtype: Path
    :raises ValueError: The model id or the snapshot id is no single path segment.
    """
    return _turns(name, snapshot_id, recording).path(model, None, snapshot_id)


def _turns(name: str, snapshot_id: str, recording: str | os.PathLike[str]) -> WriterTurns:
    """The writer's recorded models over one snapshot, the case named for the eval's report directory."""
    return writer_turns(recording, arm=_ARM, skill=load_skill(), cases={snapshot_id: analysis_page(name).parent.name})


def analysis_page(name: str) -> Path:
    """Where the writer's report on the eval named ``name`` is written: its report directory's :data:`ANALYSIS_PAGE`."""
    return report_dir(name) / ANALYSIS_PAGE


async def analyse(
    result: Filed,
    model: str,
    *,
    budget_usd: float = DEFAULT_BUDGET_USD,
    recording: str | os.PathLike[str] = DEFAULT_RECORDING,
) -> ReportWritten:
    """The report writer's report on ``result``'s campaign, its page written to :func:`analysis_page`.

    :param result: The run, filed as a campaign.
    :ptype result: Filed
    :param model: The writer's model id (``claude-sonnet-5-5``), which a recording is made on.
    :ptype model: str
    :param budget_usd: The most the report may spend, in US dollars; :data:`TURN_CAP` turns ride on it.
    :ptype budget_usd: float
    :param recording: The directory the report's recordings sit under (the module docstring); a relative path is read
        beside the eval file being loaded, else in the working directory.
    :ptype recording: str | os.PathLike[str]
    :return: How the writing ended, the stored document and its page. With nothing stored there is no page, and a
        page an earlier run wrote is removed, so the directory never shows a report this run did not write.
    :rtype: ReportWritten
    :raises ValueError: A budget that is no finite amount above 0, or a model id that is no single path segment.
    :raises RecordingRefused: No recording over this campaign's snapshot, and ``ANTHROPIC_API_KEY`` is not set to make
        one.
    :raises EvalServiceError: The freeze refused the campaign, with its remedy.
    :raises ReportNotStored: The writer ended with a document, and the store refused it.
    """
    budget = ReportBudget(usd=budget_usd, turns=TURN_CAP)
    ops = OpsHost.reading(result.host)
    frozen = await report_snapshot_freeze(ops, result.scope_id, campaign_id=result.campaign_id, created_by=CREATED_BY)
    turns = _turns(result.name, frozen.snapshot_id, recording)
    try:
        written = await write_report(
            ops,
            Caller(scope_id=result.scope_id, identity=CREATED_BY),
            model=turns(WriterArm(model=model, skill=load_skill()))(None),
            budget=budget,
            snapshot_id=frozen.snapshot_id,
        )
    finally:
        await turns.aclose()
    for left in turns.unconsumed():
        sys.stdout.write(f"drift: {left}\n")
    await run_blocking(result.host.blocking_executor, _write_page, analysis_page(result.name), written.page)
    return written


def _write_page(path: Path, page: DocPage | None) -> None:
    """Write the page at ``path``, or remove an earlier run's when there is none."""
    if page is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc_page_html(page), encoding="utf-8")


#: The names the harness's command line calls, the constants above read through them; and the turn cap, which the
#: writer eval's reports run under too.
__all__ = ["DEFAULT_BUDGET_USD", "TURN_CAP", "Filed", "analyse", "analysis_page", "recording_path"]
