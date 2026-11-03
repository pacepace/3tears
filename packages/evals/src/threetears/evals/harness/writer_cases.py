"""Portable writer cases: one report snapshot, frozen whole into a file a writer eval ships beside itself.

A writer case is a frozen report snapshot (:class:`~threetears.evals.kernel.report_snapshot.EvalReportSnapshot`)
and what its mint names: its split group, its stratum (the domain it comes from, which a writer campaign's readings
split by) and the exemplar notes its analyst is never given. A store's case lives under a template in one scope; a
:class:`WriterCaseFile` carries the same case anywhere, as one gzipped JSON document (``<name>.json.gz``):
``{format, name, group, stratum, withhold, snapshot}``.

**Every source is one format.** A snapshot frozen from a campaign (:func:`writer_case_freeze`, over any host whose
profile declares the campaign's measures, such as a quick eval's own) and a bare report input frozen with no campaign
behind it are both an ``EvalReportSnapshot``, so one reader reads both (:meth:`WriterCaseFile.read`,
:func:`read_writer_cases`), and nothing here asks where a snapshot came from.

**Minting restores, then mints by id.** :meth:`WriterCaseFile.mint` puts the snapshot back in the minting scope
(:func:`~threetears.evals.ops.analyst.report_snapshot_restore`: its id is derived from its evidence, so it is stored
exactly as frozen) and mints it there by its id
(:func:`~threetears.evals.ops.report_writer.writer_bank.writer_case_mint`), carrying its group, its stratum and its
withheld notes. The side it lands on is the split's, read off its group, and recorded on the stored case.

**The bytes are reproducible.** A file is the case's compact JSON, gzipped with no timestamp, so writing the same
case twice writes the same bytes, and a committed case moves in review only when its content does.
"""

from __future__ import annotations

import gzip
import zlib
from collections.abc import Collection
from pathlib import Path
from typing import Literal

from pydantic import Field

from threetears.evals.kernel.errors import NotFoundError
from threetears.evals.kernel.eval_file import beside_eval_file
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.report_snapshot import FIND_SNAPSHOT, EvalReportSnapshot
from threetears.evals.ops.analyst import report_snapshot_freeze, report_snapshot_restore
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.report_writer.writer_bank import (
    MintedWriterCase,
    WriterCaseMint,
    WriterSplit,
    writer_case_mint,
)
from threetears.evals.schema.base import EvalBaseModel

#: What a writer case file's name ends in: the directory reader reads these and nothing else.
WRITER_CASE_SUFFIX = ".json.gz"

#: The format every file written by this build declares, and the only one it reads.
WriterCaseFormat = Literal["writer-case/1"]


class WriterCaseFile(EvalBaseModel):
    """One writer case, portable: its snapshot frozen whole, and what minting it names.

    Built from any frozen snapshot, read from a file (:meth:`read`), written to one (:meth:`write`) and minted into a
    writer template in any scope (:meth:`mint`).
    """

    format: WriterCaseFormat = Field(default="writer-case/1", description="The file format, which a reader checks.")
    name: str = Field(min_length=1, description="The case's name, unique among the files a writer eval reads.")
    group: str | None = Field(
        default=None,
        min_length=1,
        description="The split group: related cases named alike sit on one side. Omitted, the case's name is its group.",
    )
    stratum: str | None = Field(
        default=None,
        min_length=1,
        description="The kind of case it is (the domain it comes from), stored as the minted case's stratum.",
    )
    withhold: list[str] = Field(
        default_factory=list, description="The exemplar notes, by file name, the case's analyst is never given."
    )
    snapshot: EvalReportSnapshot = Field(description="The report snapshot the case is written over, frozen whole.")

    def write(self, path: str | Path) -> Path:
        """Write the case to a file, its directory made when missing.

        :param path: The file, ending in ``.json.gz``.
        :ptype path: str | Path
        :return: The file written.
        :rtype: Path
        :raises ValueError: When the path does not end in ``.json.gz``, so no directory read would find it.
        """
        target = Path(path)
        if not target.name.endswith(WRITER_CASE_SUFFIX):
            raise ValueError(f"a writer case file's name ends in {WRITER_CASE_SUFFIX}, and {target.name!r} does not")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(self.to_json().encode("utf-8"), mtime=0))
        return target

    @classmethod
    def read(cls, path: str | Path) -> WriterCaseFile:
        """Read a case from its file, whatever source its snapshot was frozen from.

        :param path: The file; a relative path is read beside the eval file being loaded, else in the working
            directory.
        :ptype path: str | Path
        :return: The case, its snapshot's identity checked.
        :rtype: WriterCaseFile
        :raises ValueError: When the file is not gzip, or its content does not read as a writer case of this format.
        :raises OSError: When the file cannot be opened.
        """
        source = beside_eval_file(path)
        try:
            return cls.from_json(gzip.decompress(source.read_bytes()).decode("utf-8"))
        except (gzip.BadGzipFile, EOFError, zlib.error, ValueError) as unread:
            raise ValueError(f"{source} does not read as a writer case: {unread}") from unread

    async def mint(
        self,
        host: OpsHost,
        template_id: str,
        scope_id: str,
        *,
        created_by: str,
        split: WriterSplit | None = None,
    ) -> MintedWriterCase:
        """Restore the snapshot into a scope and mint it there as a case of a writer template.

        :param host: The host whose store receives the snapshot and the case.
        :ptype host: OpsHost
        :param template_id: The writer template.
        :ptype template_id: str
        :param scope_id: The scope the template lives in, which the snapshot is restored into.
        :ptype scope_id: str
        :param created_by: Who mints it, as the host names the caller.
        :ptype created_by: str
        :param split: The sets the case joins; ``None`` for the default names and share.
        :ptype split: WriterSplit | None
        :return: The mint's receipt: the case, its stratum, its side and the set version listing it.
        :rtype: MintedWriterCase
        :raises NotFoundError: The template is not in the scope.
        :raises ValidationFailedError: As :func:`~threetears.evals.ops.report_writer.writer_bank.writer_case_mint`
            refuses.
        """
        restored = await report_snapshot_restore(host, scope_id, self.snapshot)
        mint = WriterCaseMint(
            template_id=template_id,
            snapshot_id=restored.snapshot_id,
            withhold=self.withhold,
            group=self.group or self.name,
            stratum=self.stratum,
            split=split or WriterSplit(),
        )
        return await writer_case_mint(host, mint, scope_id, created_by=created_by)


async def writer_case_freeze(
    host: OpsHost,
    scope_id: str,
    *,
    name: str,
    created_by: str,
    campaign_id: str | None = None,
    analysis_id: str | None = None,
    group: str | None = None,
    stratum: str | None = None,
    withhold: Collection[str] = (),
) -> WriterCaseFile:
    """Freeze a campaign's evidence, or a stored analysis's, as a portable writer case.

    The freeze runs on ``host``, which must read the campaign: for a quick eval, its own host
    (``OpsHost.reading(result.host)``), whose profile declares the eval's measures.

    :param host: The host whose store holds the campaign.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param name: The case's name.
    :ptype name: str
    :param created_by: Who freezes it, as the host names the caller.
    :ptype created_by: str
    :param campaign_id: The campaign to freeze; exactly one of it and ``analysis_id``.
    :ptype campaign_id: str | None
    :param analysis_id: The stored analysis to freeze.
    :ptype analysis_id: str | None
    :param group: The split group; ``None`` for the case's name.
    :ptype group: str | None
    :param stratum: The kind of case it is.
    :ptype stratum: str | None
    :param withhold: The notes its analyst is never given.
    :ptype withhold: Collection[str]
    :return: The case, unwritten.
    :rtype: WriterCaseFile
    :raises ValidationFailedError: As :func:`~threetears.evals.ops.analyst.report_snapshot_freeze` refuses.
    :raises NotFoundError: No such campaign or analysis in the scope.
    """
    frozen = await report_snapshot_freeze(
        host, scope_id, campaign_id=campaign_id, analysis_id=analysis_id, created_by=created_by
    )
    eval_host = host.eval_host
    snapshot = await run_blocking(
        eval_host.blocking_executor, eval_host.storage.load_report_snapshot, frozen.snapshot_id, scope_id
    )
    if snapshot is None:
        raise NotFoundError("report snapshot", frozen.snapshot_id, remedy=FIND_SNAPSHOT)
    return WriterCaseFile(name=name, group=group, stratum=stratum, withhold=list(withhold), snapshot=snapshot)


def read_writer_cases(directory: str | Path) -> tuple[WriterCaseFile, ...]:
    """Every writer case file in a directory, by file name.

    :param directory: The directory; only its ``*.json.gz`` files are read, and not its subdirectories. A relative
        path is read beside the eval file being loaded, else in the working directory.
    :ptype directory: str | Path
    :return: The cases, in file-name order.
    :rtype: tuple[WriterCaseFile, ...]
    :raises ValueError: When it is not a directory, holds no case file, holds one that does not read, or holds two
        cases of one name.
    """
    root = beside_eval_file(directory)
    if not root.is_dir():
        raise ValueError(f"no writer case directory at {root}")
    paths = sorted(path for path in root.glob(f"*{WRITER_CASE_SUFFIX}") if path.is_file())
    if not paths:
        raise ValueError(f"{root} holds no writer case file (*{WRITER_CASE_SUFFIX})")
    cases = tuple(WriterCaseFile.read(path) for path in paths)
    seen: dict[str, Path] = {}
    for path, case in zip(paths, cases, strict=True):
        if case.name in seen:
            raise ValueError(f"{seen[case.name].name} and {path.name} both hold a case named {case.name!r}")
        seen[case.name] = path
    return cases


#: The writer eval reads a case directory and mints each case; the freeze and the format's names wait for a caller.
__all__ = ["WriterCaseFile", "read_writer_cases"]
