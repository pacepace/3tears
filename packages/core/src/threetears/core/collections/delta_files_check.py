"""The start check of the delta files' directory: a process that writes or copies a table that opts in to delta files
does not start without it.

**Why.** A table that opts in (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_files`) refuses
every commit whose file cannot be saved (:func:`~threetears.core.collections.caller_transaction.save_delta_files`), and
its copies apply those files off the same directory (``docs/design-pod-delta-files.md`` decisions 7, 12 and 29). A process
started without the directory serves until its first write and then fails every load, and a copy waits on files it
can never read: found in a test or on the night, not at the deploy. Checked once at the start, a process missing it
exits before it serves or contends for a write lease, with a message saying what is wrong and how to fix it.

**What is checked.** ``THREETEARS_DELTA_FILES_DIR`` is set, names an existing directory, and a file can be written
there as a delta file is (:func:`~threetears.core.collections.delta_files.write_delta_file`): written under a temporary
name, flushed, renamed, the directory flushed, then removed. Only when at least one table given opts in; a process
whose tables do not opt in is never checked.

**No marker file.** A marker the writer leaves on the volume at its first start would be left just the same on local
disk or a per-pod ``emptyDir``, so it cannot tell a shared volume from a private one; and a new, empty shared volume
has none yet. The one proof that the directory is shared is a file one replica writes being read whole by another,
which needs two running replicas: the deployment's own check after a deploy (reports: ``scripts/check-delta-volume.sh``).
This check proves what one process can on its own.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

from threetears.core.collections.delta_files import _DELTA_FILES_DIR_ENV, _fsync_directory
from threetears.observe import get_logger
from threetears.observe.metrics import counter

if TYPE_CHECKING:
    from threetears.core.collections.schema_backed import TableSchema

__all__ = ["DeltaFilesDirectoryError", "require_delta_files_root", "require_delta_files_root_off_loop"]

log = get_logger(__name__)

#: the start probe's name under the root: a dot name at the top, never in a table's directory, so no sweep or fold
#: ever reads it (delta files live one directory deep, ``<table>/<id>.arrow``)
_PROBE_PREFIX: Final = ".start-check-"

#: the start checks of the directory, by outcome: ``usable``, or what was wrong (``unset``, ``not_absolute``,
#: ``missing``, ``not_directory``, ``not_writable``); a process whose tables do not opt in is never checked, so never
#: counted. Any but ``usable`` refuses the start
_START_CHECKS = counter(
    "delta_files_start_checks_total",
    description="a process's start checks of its delta files' directory, by outcome: usable, unset, not_absolute, "
    "missing, not_directory, not_writable; any but usable refuses the start",
    label_names=("outcome",),
)

#: the rule line around the message, so it stands out in a pod's log among everything else at its start
_RULE: Final = "=" * 100


class DeltaFilesDirectoryError(RuntimeError):
    """the delta files' directory a process needs is not usable: it must not start.

    :ivar tables: the tables that opt in to delta files and need it
    :ivar found: what was found, one line (unset, missing, not a directory, not writable)
    """

    def __init__(self, message: str, *, tables: tuple[str, ...], found: str) -> None:
        super().__init__(message)
        self.tables = tables
        self.found = found


def _probe(root: Path) -> None:
    """write, flush, rename and remove a file under ``root``, as a delta file is saved, the directory flushed after
    the rename and after the removal.

    :param root: the delta files' directory
    :ptype root: Path
    :return: nothing
    :rtype: None
    :raises OSError: when any step fails
    """
    token = uuid.uuid7().hex
    partial = root / f"{_PROBE_PREFIX}{token}.partial"
    final = root / f"{_PROBE_PREFIX}{token}"
    try:
        with partial.open("wb") as sink:
            sink.write(b"threetears delta files start check\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(partial, final)
        _fsync_directory(root)
    finally:
        # a step that failed leaves neither name behind; a removal that fails is the error the caller reports
        partial.unlink(missing_ok=True)
        final.unlink(missing_ok=True)
    _fsync_directory(root)


def _found(named: str | None) -> str | None:
    """what is wrong with the directory ``named``, or None when it is usable; counted by outcome (:data:`_START_CHECKS`).

    :param named: the variable's value; None or empty when unset
    :ptype named: str | None
    :return: one line saying what was found, or None
    :rtype: str | None
    """
    problem: str | None = None
    outcome = "usable"
    if not named:
        outcome = "unset"
        problem = f"{_DELTA_FILES_DIR_ENV} is not set"
    elif not os.path.isabs(named):
        outcome = "not_absolute"
        problem = f"{_DELTA_FILES_DIR_ENV}={named} is not an absolute path"
    elif not os.path.exists(named):
        outcome = "missing"
        problem = f"{_DELTA_FILES_DIR_ENV}={named} does not exist (nothing is mounted there)"
    elif not os.path.isdir(named):
        outcome = "not_directory"
        problem = f"{_DELTA_FILES_DIR_ENV}={named} is not a directory"
    else:
        try:
            _probe(Path(named))
        except OSError as exc:
            outcome = "not_writable"
            problem = (
                f"{_DELTA_FILES_DIR_ENV}={named} is not writable by this process (uid {os.getuid()}, gid "
                f"{os.getgid()}): writing, flushing, renaming and removing a file there failed: {exc}"
            )
    _START_CHECKS.labels(outcome=outcome).inc()
    return problem


def _message(tables: tuple[str, ...], found: str, fixes: Mapping[str, str]) -> str:
    """the error's text: what is wrong, why the process cannot run, and how to fix it wherever it runs.

    :param tables: the tables that opt in
    :ptype tables: tuple[str, ...]
    :param found: what was found
    :ptype found: str
    :param fixes: where the process runs -> how to give it the directory there
    :ptype fixes: Mapping[str, str]
    :return: the message, many lines
    :rtype: str
    """
    lines = [
        "",
        _RULE,
        "THIS PROCESS CANNOT START: ITS DELTA FILES DIRECTORY IS NOT USABLE",
        _RULE,
        f"Found: {found}.",
        "",
        f"Needed by {len(tables)} table(s) that opt in to delta files: {', '.join(tables)}.",
        "",
        "Why: every write to those tables saves its delta file in this directory before it commits, and is refused",
        "without it; every copy of them applies those files from it. Without the directory this process would serve",
        "until its first load and then fail every write, and its copies would wait for files they can never read.",
        f"It needs {_DELTA_FILES_DIR_ENV} set to an existing directory this process can write: on a cluster a",
        "ReadWriteMany volume (EFS) mounted in every replica at the same path; locally any directory.",
        "",
        "HOW TO FIX, by where this process runs:",
    ]
    for place, fix in fixes.items():
        lines.append(f"  * {place}:")
        lines.extend(f"      {line}" for line in fix.splitlines())
    lines.extend(
        [
            "  * a test that starts this process or writes those tables to L3:",
            f"      set {_DELTA_FILES_DIR_ENV} to a temporary directory (pytest: monkeypatch.setenv with tmp_path).",
            _RULE,
        ]
    )
    return "\n".join(lines)


def require_delta_files_root(
    schemas: Iterable[TableSchema],
    *,
    fixes: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """refuse to start without the delta files' directory when any of ``schemas`` opts in to delta files.

    Blocking: a file is written, flushed and renamed, the directory flushed twice (each ``fsync`` waits on the volume:
    on EFS a round trip each, tens of milliseconds). Safe only where no event loop runs on this thread: a process's
    start before its loop does (``main`` before ``asyncio.run``), a script, a test. From a coroutine call
    :func:`require_delta_files_root_off_loop`, which runs this on a thread.

    :param schemas: every table the process writes or copies
    :ptype schemas: Iterable[TableSchema]
    :param fixes: where the process runs -> how to give it the directory there (the chart value and claim, the
        local runs, the check after a deploy), each printed under "HOW TO FIX"; a test's fix is always printed
    :ptype fixes: Mapping[str, str] | None
    :param environ: the environment to read ``THREETEARS_DELTA_FILES_DIR`` from; None reads the process's
    :ptype environ: Mapping[str, str] | None
    :return: the directory, checked; None when no table opts in (nothing is checked)
    :rtype: Path | None
    :raises DeltaFilesDirectoryError: when a table opts in and the directory is unset, missing, not a directory or
        not writable
    """
    tables = tuple(sorted(schema.name for schema in schemas if schema.delta_files))
    env = os.environ if environ is None else environ
    named = env.get(_DELTA_FILES_DIR_ENV)
    root: Path | None = None
    if tables:
        found = _found(named)
        if found is not None:
            raise DeltaFilesDirectoryError(_message(tables, found, fixes or {}), tables=tables, found=found)
        root = Path(named or "")
        log.info(
            "the delta files' directory is usable",
            extra={"extra_data": {"directory": str(root), "tables": len(tables)}},
        )
    return root


async def require_delta_files_root_off_loop(
    schemas: Iterable[TableSchema],
    *,
    fixes: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """:func:`require_delta_files_root` from a coroutine: the probe's write and flushes run on a thread.

    Why: the probe's ``fsync`` calls wait on the volume (on EFS a network round trip each), and a process that checks
    from its async start would hold its event loop that long, answering nothing else.

    :param schemas: every table the process writes or copies
    :ptype schemas: Iterable[TableSchema]
    :param fixes: where the process runs -> how to give it the directory there (:func:`require_delta_files_root`)
    :ptype fixes: Mapping[str, str] | None
    :param environ: the environment to read ``THREETEARS_DELTA_FILES_DIR`` from; None reads the process's
    :ptype environ: Mapping[str, str] | None
    :return: the directory, checked; None when no table opts in (nothing is checked)
    :rtype: Path | None
    :raises DeltaFilesDirectoryError: as :func:`require_delta_files_root` raises it
    """
    tables = list(schemas)
    return await asyncio.to_thread(require_delta_files_root, tables, fixes=fixes, environ=environ)
