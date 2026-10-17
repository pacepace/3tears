"""a git repository as a durable store: its working tree, its one writer, its commits and its origin.

:class:`GitRepo` owns one working tree. Every change to it is a commit made by the repository's
writer -- the process holding its NATS lease (:mod:`threetears.git_store.writer`); any other process
forwards its change set to the writer and gets the commit, or the refusal, back. Without NATS
(``nats=None``: tests, one-process development) this process is always the writer.

**How a commit happens**, on the writer, under the repository's lock:

1. HEAD is checked against the last commit this store saw. A commit made by hand since is reported
   first (its rows dropped from every cache by :func:`~threetears.git_store.bind.bind_collections`),
   so the checks below compare against the files as they are now.
2. Every change's compare-and-swap is checked against its file. One miss refuses the whole set:
   nothing is written (:class:`~threetears.git_store.errors.ChangeConflictError`).
3. A file someone edited by hand and has not committed is never written over
   (:class:`~threetears.git_store.errors.HandEditPendingError`).
4. A journal naming the paths about to change is written into the git directory; each file is
   written with ``atomic_write`` (temporary file, fsync, rename), staged, and committed with
   ``git commit --only <paths>``, so nothing else staged or edited joins the commit. A set that
   changes no bytes makes no commit. The journal is removed.
5. The commit is pushed when the policy says so (a plain push, never forced) and announced.

A crash between 4's first write and its commit leaves the journal: the next writer puts those
paths back as HEAD has them. The caller never heard "saved".

**Two ways to share a repository.** ``clone="shared"`` (the default): every process on the disk
reads the one working tree, only the writer changes it, and the writer's cache drops reach the
others over NATS. ``clone="own"``: each process has its own clone; one that is not the writer
fetches and fast-forwards when the writer announces a push, and every ``watch_interval``.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar

from pydantic import SecretStr

from threetears.core.collections import CallerTransaction
from threetears.core.utils.atomic_write import atomic_write_sync
from threetears.git_store.advanced import AdvancedListener, RepoAdvanced, publish_advanced
from threetears.git_store.change import Change, ChangeSet, CommitResult, same_version
from threetears.git_store.errors import (
    ChangeConflictError,
    ConflictedRow,
    GitStoreError,
    HandEditPendingError,
    NotTheWriterError,
)
from threetears.git_store.git import Author, CommitEntry, FileChange, Git, GitError
from threetears.git_store.layout import Layout, locate
from threetears.git_store.wire import (
    ForwardedRequest,
    decode_reply,
    decode_request,
    encode_failure,
    encode_request,
    encode_result,
)
from threetears.git_store.writer import COMMIT_HANDLER, WriterRole
from threetears.nats.errors import NatsClientError
from threetears.nats.forward import ForwardedHandlerError, ForwardHandler, NoOwnerError
from threetears.observe import PeriodicTask, get_logger, retry_bounded

if TYPE_CHECKING:
    from threetears.nats import NatsClient

__all__ = ["DirtyFile", "GitRepo", "GitRepoConfig", "RepoStatus", "RowChange", "RowsChanged"]

log = get_logger(__name__)

_T = TypeVar("_T")

#: a repository key: one lease, forward and broadcast name. Letters, digits, ``_``, ``-`` and ``.``.
_KEY: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
#: a branch this store commits to.
_BRANCH: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]*")
#: a trailer key (``Op-Seq``).
_TRAILER_KEY: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")
#: the journal of a change being written, and the last commit this store reported, in the git dir.
_JOURNAL: Final[str] = "git-store/inflight.json"
_SEEN_HEAD: Final[str] = "git-store/seen-head"
#: the most paths one dirty-file warning names.
_DIRTY_LOG_LIMIT: Final[int] = 20


@dataclass(frozen=True)
class GitRepoConfig:
    """one repository's settings.

    :ivar key: the repository's stable name: its lease, forward and broadcast key
        (``"samhla-characters"``, ``"aria"``)
    :ivar work_tree: the working tree's root
    :ivar branch: the branch committed to
    :ivar remote_url: origin; the working tree is cloned from it when absent. ``None``: no origin
        (a missing working tree is created empty)
    :ivar deploy_key: the private SSH key origin is reached with; never logged. ``None`` uses the
        ambient SSH configuration
    :ivar clone: ``"shared"`` -- the working tree is shared by every process on its disk and only
        the writer changes it; ``"own"`` -- this process's own clone, which follows origin while
        another process is the writer
    :ivar seek_writer: try to become the writer at open, and keep trying while another process is;
        ``False`` becomes the writer only when a change finds no writer to forward to (a command line)
    :ivar push: ``"after_commit"`` pushes each commit at once; ``"interval"`` every ``push_interval``;
        ``"never"`` leaves origin alone
    :ivar push_interval: how often ``push="interval"`` pushes
    :ivar safety_idle: how long a hand edit may sit uncommitted before the sweep warns about it
    :ivar sweep_interval: how often the sweep looks
    :ivar committer: who every commit names as committer, and as author when the change names none
    :ivar lease_bucket: the writer lease's bucket name (the NATS client adds its namespace)
    :ivar lease_ttl: how long the lease outlives its holder's last renewal
    :ivar lease_renew: how often the holder renews it
    :ivar watch_interval: how often the writer looks for hand commits, a clone follows origin, and a
        process seeking the lease tries for it
    :ivar forward_timeout: how long a forwarded change waits for the writer's answer (a commit and
        its push)
    """

    key: str
    work_tree: Path
    branch: str = "main"
    remote_url: str | None = None
    deploy_key: SecretStr | None = None
    clone: Literal["shared", "own"] = "shared"
    seek_writer: bool = True
    push: Literal["never", "after_commit", "interval"] = "after_commit"
    push_interval: timedelta = timedelta(minutes=5)
    safety_idle: timedelta = timedelta(hours=6)
    sweep_interval: timedelta = timedelta(hours=1)
    committer: Author = Author("git-store", "git-store@localhost")
    lease_bucket: str = "git-store-writers"
    lease_ttl: timedelta = timedelta(seconds=30)
    lease_renew: timedelta = timedelta(seconds=10)
    watch_interval: timedelta = timedelta(seconds=10)
    forward_timeout: timedelta = timedelta(seconds=30)

    def __post_init__(self) -> None:
        """refuse a key or branch that cannot name a lease.

        :return: None
        :rtype: None
        :raises ValueError: for a bad key or branch, or a lease that renews no sooner than it expires
        """
        if not _KEY.fullmatch(self.key) or ".." in self.key:
            raise ValueError(f"git-store repository key {self.key!r}: letters, digits, '_', '-' and '.' only")
        if not _BRANCH.fullmatch(self.branch) or ".." in self.branch or self.branch.endswith((".", "/")):
            raise ValueError(f"git-store branch {self.branch!r} is not a plain branch name")
        if self.lease_renew >= self.lease_ttl:
            raise ValueError("lease_renew must be shorter than lease_ttl")

    @property
    def lease_key(self) -> str:
        """the writer lease's key.

        :return: ``"{key}.{branch}"``
        :rtype: str
        """
        return f"{self.key}.{self.branch}"


@dataclass(frozen=True, slots=True)
class RowChange:
    """a row a commit added, changed or removed.

    :ivar table: the table
    :ivar pk: the row's key
    :ivar deleted: the commit removed the row's file
    :ivar local: this process made the commit (through this store)
    :ivar sha: the commit
    """

    table: str
    pk: Mapping[str, Any]
    deleted: bool
    local: bool
    sha: str


#: a callback told of the rows a commit changed.
RowsChanged = Callable[[list[RowChange]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class DirtyFile:
    """a file in the working tree that differs from HEAD and was not written by this store.

    :ivar path: the path, relative to the root
    :ivar status: two letters, as ``git status --porcelain`` gives them (``" M"``, ``"??"``)
    :ivar since: when the file was last changed on disk; ``None`` for a removed file
    """

    path: str
    status: str
    since: datetime | None


@dataclass(frozen=True, slots=True)
class RepoStatus:
    """a repository at a glance.

    :ivar key: its key
    :ivar branch: its branch
    :ivar head: the commit HEAD names, or ``None`` before the first
    :ivar is_writer: whether this process is its writer
    :ivar writer: the writer's holder id, when there is one (``None`` without NATS)
    :ivar ahead: commits origin lacks, as last fetched
    :ivar behind: commits origin has that this clone lacks, as last fetched
    :ivar dirty: hand edits not yet committed
    :ivar push_blocked: why pushing stopped (origin moved in a way this repository cannot
        fast-forward); ``None`` while pushing works. A person decides; :meth:`GitRepo.push` retries
    :ivar last_push_error: the last push failure that was not a refusal (the network), or ``None``
    :ivar last_push_at: when origin last took a push from this process
    """

    key: str
    branch: str
    head: str | None
    is_writer: bool
    writer: str | None
    ahead: int
    behind: int
    dirty: tuple[DirtyFile, ...]
    push_blocked: str | None
    last_push_error: str | None
    last_push_at: datetime | None


@dataclass
class _Plan:
    """what a change set comes to against the files as they are.

    :ivar writes: each path to write, with its new bytes, or ``None`` to remove it
    :ivar rows: each row a write touches: ``(table, pk, deleted)``
    :ivar conflicts: rows whose compare-and-swap missed
    """

    writes: dict[str, bytes | None] = field(default_factory=dict)
    rows: list[tuple[str, Mapping[str, Any], bool]] = field(default_factory=list)
    conflicts: list[ConflictedRow] = field(default_factory=list)


def _message_with_trailers(message: str, trailers: Mapping[str, str] | None) -> str:
    """a commit message with ``Key: value`` trailer lines after a blank line.

    :param message: the message
    :ptype message: str
    :param trailers: the trailers
    :ptype trailers: Mapping[str, str] | None
    :return: the whole message
    :rtype: str
    :raises ValueError: for an empty message, or a trailer key or value that would break the block
    """
    text = message.strip()
    if not text:
        raise ValueError("a git-store commit needs a message")
    if not trailers:
        return text + "\n"
    for key, value in trailers.items():
        if not _TRAILER_KEY.fullmatch(key) or "\n" in value:
            raise ValueError(f"trailer {key!r}: {value!r} cannot be written as one trailer line")
    return text + "\n\n" + "\n".join(f"{key}: {value}" for key, value in trailers.items()) + "\n"


def _checked_paths(paths: Sequence[str]) -> list[str]:
    """paths a person named, refused if any leaves the working tree or names the git directory.

    :param paths: the paths, relative to the root
    :ptype paths: Sequence[str]
    :return: the paths, normalized
    :rtype: list[str]
    :raises ValueError: for an absolute path, one that climbs out, or one inside ``.git``
    """
    checked: list[str] = []
    for path in paths:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or (pure.parts and pure.parts[0] == ".git") or not pure.parts:
            raise ValueError(f"{path!r} is not a path inside the working tree")
        checked.append(pure.as_posix())
    return checked


class GitRepo:
    """one working tree, its writer, its commits and its origin.

    :ivar config: the settings
    :ivar layout: where each table's rows live in it
    :ivar git: the ``git`` program bound to the working tree, for a product's own read-only questions
    """

    def __init__(
        self,
        config: GitRepoConfig,
        layout: Layout,
        *,
        nats: NatsClient | None,
        listener: AdvancedListener | None = None,
    ) -> None:
        """describe the repository; nothing on disk or on NATS changes until :meth:`open`.

        :param config: the settings
        :ptype config: GitRepoConfig
        :param layout: where each table's rows live
        :ptype layout: Layout
        :param nats: the connected client the writer lease, forwards and broadcasts use; ``None``
            makes this process always the writer, with no forwarding and no broadcast
        :ptype nats: NatsClient | None
        :param listener: the process's shared broadcast listener (:class:`GitRepos` passes one);
            ``None`` makes one for this repository when it needs it
        :ptype listener: AdvancedListener | None
        :return: None
        :rtype: None
        """
        self.config = config
        self.layout = layout
        self.git = Git(config.work_tree, deploy_key=config.deploy_key)
        self._nats = nats
        self._listener = listener
        self._lock = asyncio.Lock()
        self._seen_head: str | None = None
        self._listeners: list[RowsChanged] = []
        self._role: WriterRole | None = None
        self._tasks: list[PeriodicTask] = []
        self._push_blocked: str | None = None
        self._last_push_error: str | None = None
        self._last_push_at: datetime | None = None
        self._opened = False
        self._log = {"key": config.key, "branch": config.branch}

    # ------------------------------------------------------------------ lifecycle

    async def open(self) -> None:
        """make the working tree ready and take part: clone or create it, claim or follow the writer.

        Without NATS this process is the writer: an interrupted change is recovered and unpushed
        commits are pushed now.

        :return: None
        :rtype: None
        :raises GitError: when the working tree cannot be cloned or created
        """
        if self._opened:
            return
        await asyncio.to_thread(self._ensure_checkout)
        self._seen_head = await asyncio.to_thread(self._recorded_head)
        self._opened = True
        if self._nats is None:
            await self._takeover()
        else:
            self._role = WriterRole(
                self._nats,
                lease_key=self.config.lease_key,
                bucket=self.config.lease_bucket,
                ttl=self.config.lease_ttl,
                renew_every=self.config.lease_renew,
                takeover=self._takeover,
                forward_timeout=self.config.forward_timeout,
                log_extra=self._log,
            )
            await self._role.serve(COMMIT_HANDLER, self._handle_forwarded)
            if self.config.seek_writer:
                await self._role.claim()
            if self.config.clone == "own":
                if self._listener is None:
                    self._listener = AdvancedListener(self._nats)
                await self._listener.register(self.config.key, self.config.branch, self)
        self._start_tasks()

    def _start_tasks(self) -> None:
        """start the background loops: watch, sweep, and the interval push.

        :return: None
        :rtype: None
        """
        name = f"git-store:{self.config.lease_key}"
        self._tasks.append(
            PeriodicTask(self._watch, interval=self.config.watch_interval, name=f"{name}:watch", logger=log)
        )
        self._tasks.append(
            PeriodicTask(self._sweep, interval=self.config.sweep_interval, name=f"{name}:sweep", logger=log)
        )
        if self.config.push == "interval" and self.config.remote_url is not None:
            self._tasks.append(
                PeriodicTask(self._interval_push, interval=self.config.push_interval, name=f"{name}:push", logger=log)
            )
        for task in self._tasks:
            task.start()

    async def aclose(self) -> None:
        """stop the loops, stop listening, and give the writer lease up for the next writer.

        :return: None
        :rtype: None
        """
        for task in self._tasks:
            await task.stop()
        self._tasks.clear()
        if self._listener is not None and self.config.clone == "own":
            await self._listener.unregister(self.config.key, self.config.branch)
        if self._role is not None:
            await self._role.release()
        self._opened = False

    def _ensure_checkout(self) -> None:
        """clone the working tree from origin, or create it empty, when it is not there. BLOCKING.

        :return: None
        :rtype: None
        :raises GitError: when git refuses
        :raises GitStoreError: when the working tree sits inside another git repository
        """
        if self.git.exists():
            if self.config.remote_url is not None and not self.git.has_remote():
                self.git.set_remote(self.config.remote_url)
            return
        outer = self.git.enclosing()
        if outer is not None:
            # never init or clone inside another checkout, nor use it: commits would land in that repository
            raise GitStoreError(f"{self.config.work_tree} is inside the git repo {outer}, not a repo of its own")
        if self.config.remote_url is not None:
            self.git.clone(self.config.remote_url, self.config.branch)
        else:
            self.git.init(self.config.branch)
        log.info("git_store.checkout: the working tree was made", extra={"extra_data": self._log})

    def _recorded_head(self) -> str | None:
        """the last commit this repository's store reported, as recorded in the git directory. BLOCKING.

        Falls back to HEAD when nothing (valid) is recorded: a fresh store starts from what is there.

        :return: the commit, or ``None`` on an empty branch
        :rtype: str | None
        """
        path = self.git.git_path(_SEEN_HEAD)
        if path.is_file():
            recorded = path.read_text(encoding="utf-8").strip()
            if recorded and self.git.rev_parse(recorded) is not None:
                return recorded
        return self.git.head()

    def _record_head(self, head: str | None) -> None:
        """remember the last commit reported, for the next writer. BLOCKING.

        :param head: the commit
        :ptype head: str | None
        :return: None
        :rtype: None
        """
        if head is None:
            return
        path = self.git.git_path(_SEEN_HEAD)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_sync(path, head + "\n", mode=0o644)

    # ------------------------------------------------------------------ the writer

    @property
    def is_writer(self) -> bool:
        """whether this process commits to the repository itself.

        :return: ``True`` without NATS, or while this process holds the writer lease
        :rtype: bool
        """
        return self._role is None or self._role.is_writer

    def _require_writer(self) -> None:
        """refuse to change the working tree from a process that is not the writer.

        :return: None
        :rtype: None
        :raises NotTheWriterError: when this process does not hold the lease
        """
        if not self.is_writer:
            raise NotTheWriterError(f"this process is not the writer of {self.config.lease_key}; nothing was committed")

    async def serve(self, name: str, handler: ForwardHandler) -> None:
        """serve a product's own operation on this repository's writer lease.

        :param name: the operation's name (not ``"commit"``)
        :ptype name: str
        :param handler: ``bytes -> bytes``, run on the writer
        :ptype handler: ForwardHandler
        :return: None
        :rtype: None
        :raises RuntimeError: without NATS, where there is nobody to serve
        """
        if self._role is None:
            raise RuntimeError("a repository opened without NATS serves no forwarded operations")
        await self._role.serve(name, handler)

    async def forward(self, name: str, payload: bytes) -> bytes:
        """send a product's own operation to the writer's handler for it.

        :param name: the operation's name
        :ptype name: str
        :param payload: the request
        :ptype payload: bytes
        :return: the reply
        :rtype: bytes
        :raises RuntimeError: without NATS
        :raises NoOwnerError: when no writer serves it now
        """
        if self._role is None:
            raise RuntimeError("a repository opened without NATS forwards nothing")
        return await self._role.forward(name, payload)

    async def _takeover(self) -> None:
        """make this process's working tree the repository's current state, on becoming the writer.

        Recover a change that was being written when its writer stopped (its paths go back to HEAD:
        its caller never heard "saved"), catch up with origin when this branch is only behind it,
        report every commit made since the store last looked, and push what origin lacks.

        :return: None
        :rtype: None
        """
        async with self._lock:
            recovered = await asyncio.to_thread(self._recover_journal)
            if recovered:
                log.warning(
                    "git_store.recovered: an interrupted change was put back as HEAD has it",
                    extra={"extra_data": {**self._log, "paths": recovered[:_DIRTY_LOG_LIMIT]}},
                )
            if self.config.remote_url is not None:
                await self._catch_up_with_origin()
            await self._notice_head()
            await asyncio.to_thread(self._record_head, self._seen_head)
            if self.config.push != "never" and self.config.remote_url is not None:
                ahead, _ = await asyncio.to_thread(self.git.ahead_behind, self.config.branch)
                if ahead:
                    await self._push_locked(explicit=False)
        log.info("git_store.takeover: the working tree is current", extra={"extra_data": self._log})

    async def _catch_up_with_origin(self) -> None:
        """fetch origin and fast-forward to it when this branch is only behind; never merge or force.

        :return: None
        :rtype: None
        """
        branch = self.config.branch
        try:
            await asyncio.to_thread(self.git.fetch)
            remote = await asyncio.to_thread(self.git.rev_parse, f"origin/{branch}")
            head = await asyncio.to_thread(self.git.head)
            if remote is None or remote == head:
                return
            if head is None or await asyncio.to_thread(self.git.is_ancestor, head, remote):
                await asyncio.to_thread(self.git.merge_ff_only, f"origin/{branch}")
            elif not await asyncio.to_thread(self.git.is_ancestor, remote, head):
                self._push_blocked = (
                    f"{branch} and origin/{branch} have both moved on; a person must reconcile them "
                    f"(this store never merges or forces)"
                )
                log.error(
                    "git_store.diverged: the branch and origin have both moved on; pushing stopped",
                    extra={"extra_data": {**self._log, "head": head, "origin": remote}},
                )
        except GitError as exc:
            log.warning(
                "git_store.fetch.failed: carrying on with the working tree as it is",
                extra={"extra_data": {**self._log, "error": str(exc)}},
            )

    def _recover_journal(self) -> list[str]:
        """put the paths an interrupted change was writing back as HEAD has them. BLOCKING.

        :return: the paths put back
        :rtype: list[str]
        """
        journal = self.git.git_path(_JOURNAL)
        if not journal.is_file():
            return []
        paths = [str(path) for path in json.loads(journal.read_text(encoding="utf-8")).get("paths", [])]
        self.git.restore(paths)
        for path in paths:
            target = self.config.work_tree / path
            for leftover in target.parent.glob(f"{target.name}.tmp.*"):
                leftover.unlink(missing_ok=True)
        journal.unlink(missing_ok=True)
        return paths

    # ------------------------------------------------------------------ committing

    @asynccontextmanager
    async def change(
        self,
        message: str,
        *,
        author: Author | None = None,
        trailers: Mapping[str, str] | None = None,
        date: datetime | None = None,
    ) -> AsyncIterator[ChangeSet]:
        """a change set, committed as ONE commit when the block ends cleanly.

        Collection writes join it with ``conn=cs``; when it ends, committed or refused, every key
        they wrote is dropped from every cache (:class:`~threetears.core.collections.CallerTransaction`)::

            async with repo.change("aria: bank 3 keepers") as cs:
                await keepers.save_entity(k1, conn=cs)
            cs.result

        :param message: the commit message
        :ptype message: str
        :param author: who the commit names as author; ``None`` for the committer
        :ptype author: Author | None
        :param trailers: ``Key: value`` lines added to the message
        :ptype trailers: Mapping[str, str] | None
        :param date: the author date (a replayed history); ``None`` for now
        :ptype date: datetime | None
        :return: the change set, for the block
        :rtype: AsyncIterator[ChangeSet]
        :raises ChangeConflictError: when the block ends and a row changed since it was read
        """
        change_set = ChangeSet(self, message, author=author, trailers=dict(trailers or {}), date=date)
        async with CallerTransaction(change_set):
            yield change_set

    async def commit(
        self,
        changes: Sequence[Change],
        message: str,
        *,
        author: Author | None = None,
        trailers: Mapping[str, str] | None = None,
        date: datetime | None = None,
    ) -> CommitResult:
        """write ``changes`` as one commit, or refuse them whole.

        On a process that is not the writer the changes are forwarded to it; when no process is the
        writer, this one tries once to become it.

        :param changes: the writes
        :ptype changes: Sequence[Change]
        :param message: the commit message
        :ptype message: str
        :param author: who the commit names as author; ``None`` for the committer
        :ptype author: Author | None
        :param trailers: ``Key: value`` lines added to the message
        :ptype trailers: Mapping[str, str] | None
        :param date: the author date; ``None`` for now
        :ptype date: datetime | None
        :return: the commit; ``sha`` is ``None`` when nothing changed
        :rtype: CommitResult
        :raises ChangeConflictError: when a row changed since it was read; nothing was written
        :raises HandEditPendingError: when a file to write has an uncommitted hand edit
        :raises NotTheWriterError: when no writer could take the change
        """
        full = _message_with_trailers(message, trailers)
        request = ForwardedRequest(op="commit", changes=tuple(changes), message=full, author=author, date=date)
        return await self._as_writer(request, lambda: self._commit_local(list(changes), full, author, date))

    async def commit_paths(
        self, paths: Sequence[str] | None, message: str, *, author: Author | None = None
    ) -> CommitResult:
        """commit files someone edited by hand, so the change counts and every cache drops the old rows.

        :param paths: the files, relative to the root; ``None`` for every changed file
        :ptype paths: Sequence[str] | None
        :param message: the commit message
        :ptype message: str
        :param author: who the commit names as author; ``None`` for the committer
        :ptype author: Author | None
        :return: the commit; ``sha`` is ``None`` when none of the files differed from HEAD
        :rtype: CommitResult
        :raises ValueError: for a path outside the working tree
        :raises NotTheWriterError: when no writer could take it
        """
        checked = None if paths is None else tuple(_checked_paths(paths))
        full = _message_with_trailers(message, None)
        request = ForwardedRequest(op="commit_paths", paths=checked, message=full, author=author)
        return await self._as_writer(request, lambda: self._commit_paths_local(checked, full, author))

    async def push(self) -> bool:
        """push to origin now, even after a refusal stopped automatic pushes.

        :return: whether origin took it
        :rtype: bool
        """
        request = ForwardedRequest(op="push")
        result = await self._as_writer(request, self._push_now)
        return result.pushed

    async def _push_now(self) -> CommitResult:
        """push, on the writer, under the lock.

        :return: the result, ``pushed`` saying whether origin took it
        :rtype: CommitResult
        """
        async with self._lock:
            pushed = await self._push_locked(explicit=True)
            head = await asyncio.to_thread(self.git.head)
        return CommitResult(sha=head, pushed=pushed)

    async def _as_writer(self, request: ForwardedRequest, local: Callable[[], Awaitable[CommitResult]]) -> CommitResult:
        """run an operation here when this process is the writer, else on the writer.

        :param request: the operation, for forwarding
        :ptype request: ForwardedRequest
        :param local: the operation, run here
        :ptype local: Callable[[], Awaitable[CommitResult]]
        :return: the result
        :rtype: CommitResult
        :raises NotTheWriterError: when no process is the writer and this one could not become it
        """
        if self.is_writer:
            return await local()
        assert self._role is not None  # is_writer is True without a role
        reply: bytes | None = None
        try:
            reply = await self._role.forward(COMMIT_HANDLER, encode_request(request))
        except NoOwnerError:
            reply = None  # nobody serves it: become the writer, below
        except ForwardedHandlerError as exc:
            raise GitStoreError(
                f"the writer of {self.config.lease_key} failed: {exc.type_name}: {exc.message}"
            ) from exc
        except NatsClientError as exc:
            # NATS itself failed (this process lost its connection): the change may not have reached the
            # writer, and nothing was committed here
            raise NotTheWriterError(
                f"the writer of {self.config.lease_key} could not be reached ({type(exc).__name__}: {exc}); "
                f"nothing was committed by this process"
            ) from exc
        if reply is None:
            try:
                claimed = await self._role.claim()
            except NatsClientError as exc:
                raise NotTheWriterError(
                    f"no process serves {self.config.lease_key} and this one could not take the lease "
                    f"({type(exc).__name__}: {exc}); nothing was committed"
                ) from exc
            if not claimed:
                raise NotTheWriterError(
                    f"no process is the writer of {self.config.lease_key} and this one could not become it; "
                    f"nothing was committed"
                )
            return await local()
        return decode_reply(reply)

    async def _handle_forwarded(self, payload: bytes) -> bytes:
        """run another process's operation, on the writer, and answer with its result or refusal.

        :param payload: the request
        :ptype payload: bytes
        :return: the reply
        :rtype: bytes
        """
        request = decode_request(payload)
        try:
            self._require_writer()
            if request.op == "commit":
                result = await self._commit_local(list(request.changes), request.message, request.author, request.date)
            elif request.op == "commit_paths":
                result = await self._commit_paths_local(request.paths, request.message, request.author)
            else:
                result = await self._push_now()
        except (GitStoreError, ChangeConflictError) as exc:
            return encode_failure(exc)
        return encode_result(result)

    async def _retrying(self, call: Callable[[], _T]) -> _T:
        """run a blocking git call off the loop, retrying while another git process holds the index.

        :param call: the call
        :ptype call: Callable[[], _T]
        :return: its result
        :rtype: _T
        :raises GitError: once the index stays locked, or for any other failure at once
        """

        async def attempt() -> _T:
            return await asyncio.to_thread(call)

        return await retry_bounded(
            attempt,
            retry_on=lambda exc: isinstance(exc, GitError) and exc.index_locked,
            first_delay=0.1,
            max_delay=1.0,
            max_attempts=6,
        )

    async def _commit_local(
        self, changes: list[Change], message: str, author: Author | None, date: datetime | None
    ) -> CommitResult:
        """check and commit a change set, on the writer.

        :param changes: the writes
        :ptype changes: list[Change]
        :param message: the whole message
        :ptype message: str
        :param author: the author
        :ptype author: Author | None
        :param date: the author date
        :ptype date: datetime | None
        :return: the commit
        :rtype: CommitResult
        :raises ChangeConflictError: when a compare-and-swap missed
        :raises HandEditPendingError: when a hand edit is in the way
        :raises NotTheWriterError: when the lease is not (or no longer) held
        """
        async with self._lock:
            self._require_writer()
            await self._notice_head()
            plan = await asyncio.to_thread(self._plan, changes)
            if plan.conflicts:
                log.info(
                    "git_store.conflict: a change set was refused; rows changed since they were read",
                    extra={"extra_data": {**self._log, "rows": len(plan.conflicts)}},
                )
                raise ChangeConflictError(plan.conflicts)
            if not plan.writes:
                return CommitResult(sha=None)
            # once files start changing the sequence runs to its end even if the caller is
            # cancelled: a commit abandoned halfway leaves files written and nothing committed
            sha = await asyncio.shield(self._write_and_commit(plan, message, author or self.config.committer, date))
            rows = [RowChange(table, pk, deleted, True, sha) for table, pk, deleted in plan.rows]
            await self._reported(sha, rows)
            pushed = await self._push_after_commit()
            paths = tuple(sorted(plan.writes))
        log.info(
            "git_store.commit: a change set was committed",
            extra={"extra_data": {**self._log, "sha": sha, "files": len(paths), "pushed": pushed}},
        )
        return CommitResult(sha=sha, paths=paths, pushed=pushed)

    def _plan(self, changes: Sequence[Change]) -> _Plan:
        """check each change against its file and work out the bytes to write. BLOCKING.

        :param changes: the writes
        :ptype changes: Sequence[Change]
        :return: the plan
        :rtype: _Plan
        :raises ValueError: for a compare-and-swap on a table with no version column, or a row the
            layout cannot place
        """
        plan = _Plan()
        root = self.config.work_tree
        for change in changes:
            path = locate(self.layout, root, change.table, change.pk, change.row)
            if path is None:
                if change.row is None:
                    continue  # deleting a row that has no file: nothing to do
                raise ValueError(f"the layout cannot place a new {change.table} row {dict(change.pk)}")
            key = path.as_posix()
            if key in plan.writes:
                current = plan.writes[key]
            else:
                target = root / key
                current = target.read_bytes() if target.is_file() else None
            if change.cas is not None:
                column = self.layout.version_column(change.table)
                if column is None:
                    raise ValueError(
                        f"{change.table} has no version column, so a compare-and-swap write cannot be honoured; "
                        f"it is refused rather than written unfenced"
                    )
                if current is None:
                    plan.conflicts.append(ConflictedRow(change.table, change.pk, "deleted", change.cas))
                    continue
                stored = self.layout.decode(change.table, path, current)
                if not same_version(stored.get(column), change.cas):
                    plan.conflicts.append(ConflictedRow(change.table, change.pk, "changed", change.cas))
                    continue
            if change.row is None:
                if current is not None:
                    plan.writes[key] = None
                    plan.rows.append((change.table, change.pk, True))
                continue
            if current is not None and change.on_conflict == "ignore":
                continue
            if current is not None and change.on_conflict == "raise":
                plan.conflicts.append(ConflictedRow(change.table, change.pk, "exists"))
                continue
            data = self.layout.encode(change.table, change.row)
            if data != current:
                plan.writes[key] = data
                plan.rows.append((change.table, change.pk, False))
        return plan

    async def _write_and_commit(self, plan: _Plan, message: str, author: Author, date: datetime | None) -> str:
        """refuse a hand edit in the way, then journal, write, stage and commit the plan's files.

        :param plan: the plan
        :ptype plan: _Plan
        :param message: the whole message
        :ptype message: str
        :param author: the author
        :ptype author: Author
        :param date: the author date
        :ptype date: datetime | None
        :return: the new commit
        :rtype: str
        :raises HandEditPendingError: when a file to write differs from HEAD
        :raises NotTheWriterError: when the lease was lost before the commit; nothing was committed
        """
        paths = sorted(plan.writes)
        dirty = await asyncio.to_thread(self.git.dirty, paths)
        if dirty:
            raise HandEditPendingError([path for _, path in dirty])
        await asyncio.to_thread(self._write_journal, paths)
        try:
            await asyncio.to_thread(self._apply, plan)
            written = [path for path in paths if plan.writes[path] is not None]
            removed = [path for path in paths if plan.writes[path] is None]
            await self._retrying(lambda: self.git.stage(written, removed))
            self._require_writer()
            sha = await self._retrying(
                lambda: self.git.commit(paths, message, author=author, committer=self.config.committer, date=date)
            )
        except BaseException:
            await asyncio.to_thread(self._undo, paths)
            raise
        await asyncio.to_thread(self._clear_journal)
        return sha

    def _write_journal(self, paths: Sequence[str]) -> None:
        """record the paths about to change, for recovery after a crash. BLOCKING.

        :param paths: the paths
        :ptype paths: Sequence[str]
        :return: None
        :rtype: None
        """
        journal = self.git.git_path(_JOURNAL)
        journal.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_sync(journal, json.dumps({"paths": list(paths)}), mode=0o644)

    def _clear_journal(self) -> None:
        """remove the journal once its change is committed or undone. BLOCKING.

        :return: None
        :rtype: None
        """
        self.git.git_path(_JOURNAL).unlink(missing_ok=True)

    def _apply(self, plan: _Plan) -> None:
        """write and remove the plan's files. BLOCKING.

        :param plan: the plan
        :ptype plan: _Plan
        :return: None
        :rtype: None
        """
        for path, data in plan.writes.items():
            target = self.config.work_tree / path
            if data is None:
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_sync(target, data, mode=0o644)

    def _undo(self, paths: Sequence[str]) -> None:
        """put ``paths`` back as HEAD has them after a change failed to commit. BLOCKING.

        :param paths: the paths
        :ptype paths: Sequence[str]
        :return: None
        :rtype: None
        """
        try:
            self.git.restore(paths)
        except GitError as exc:
            # the journal stays, so the next takeover finishes the job
            log.error(
                "git_store.undo.failed: a failed change could not be put back; the next writer will",
                extra={"extra_data": {**self._log, "error": str(exc)}},
            )
            return
        self._clear_journal()

    async def _commit_paths_local(
        self, paths: Sequence[str] | None, message: str, author: Author | None
    ) -> CommitResult:
        """commit hand-edited files, on the writer.

        :param paths: the files; ``None`` for every changed file
        :ptype paths: Sequence[str] | None
        :param message: the whole message
        :ptype message: str
        :param author: the author
        :ptype author: Author | None
        :return: the commit
        :rtype: CommitResult
        """
        async with self._lock:
            self._require_writer()
            await self._notice_head()
            dirty = await asyncio.to_thread(self.git.dirty, list(paths) if paths is not None else None)
            if not dirty:
                return CommitResult(sha=None)
            targets = sorted({path for _, path in dirty})
            root = self.config.work_tree
            written = [path for path in targets if (root / path).exists()]
            removed = [path for path in targets if not (root / path).exists()]
            head_before = await asyncio.to_thread(self.git.head)
            committer = self.config.committer

            async def stage_and_commit() -> str:
                await self._retrying(lambda: self.git.stage(written, removed))
                self._require_writer()
                return await self._retrying(
                    lambda: self.git.commit(targets, message, author=author or committer, committer=committer)
                )

            sha = await asyncio.shield(stage_and_commit())
            changes = await asyncio.to_thread(self.git.changed, head_before, sha)
            rows = await asyncio.to_thread(self._rows_of, changes, head_before, sha, True)
            await self._reported(sha, rows)
            pushed = await self._push_after_commit()
        log.info(
            "git_store.commit_paths: hand edits were committed",
            extra={"extra_data": {**self._log, "sha": sha, "files": len(targets), "pushed": pushed}},
        )
        return CommitResult(sha=sha, paths=tuple(targets), pushed=pushed)

    # ------------------------------------------------------------------ reporting changed rows

    def on_rows_changed(self, callback: RowsChanged) -> Callable[[], None]:
        """be told of the rows every commit changes: this store's own, hand commits, and commits
        pulled from origin.

        :param callback: called with the rows of each commit, or of each batch of commits noticed at once
        :ptype callback: RowsChanged
        :return: a function that stops the calls
        :rtype: Callable[[], None]
        """
        self._listeners.append(callback)

        def remove() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return remove

    async def _reported(self, head: str, rows: list[RowChange]) -> None:
        """record ``head`` as seen and tell every listener of ``rows``.

        A listener that fails is logged; the others are still told.

        :param head: the commit now seen
        :ptype head: str
        :param rows: the rows
        :ptype rows: list[RowChange]
        :return: None
        :rtype: None
        """
        self._seen_head = head
        if self.is_writer or self.config.clone == "own":
            await asyncio.to_thread(self._record_head, head)
        if not rows:
            return
        for callback in list(self._listeners):
            try:
                await callback(rows)
            except (
                Exception
            ) as exc:  # prawduct:allow prawduct/broad-except -- one listener's failure must not stop the rest
                log.warning(
                    "git_store.listener.failed: a changed-rows listener raised",
                    extra={"extra_data": {**self._log, "error": f"{type(exc).__name__}: {exc}"}},
                    exc_info=True,
                )

    def _rows_of(self, changes: Sequence[FileChange], old: str | None, new: str, local: bool) -> list[RowChange]:
        """the rows a set of changed files hold. BLOCKING when the layout reads content.

        :param changes: the changed files
        :ptype changes: Sequence[FileChange]
        :param old: the commit before
        :ptype old: str | None
        :param new: the commit after
        :ptype new: str
        :param local: whether this process made the commit
        :ptype local: bool
        :return: the rows, files that hold none left out
        :rtype: list[RowChange]
        """
        rows: list[RowChange] = []
        for change in changes:
            data: bytes | None = None
            if self.layout.reads_content:
                rev = old if change.status == "D" else new
                data = None if rev is None else self.git.show(rev, change.path)
            found = self.layout.identify(PurePosixPath(change.path), data)
            if found is not None:
                rows.append(RowChange(found[0], found[1], change.status == "D", local, new))
        return rows

    async def _notice_head(self) -> list[RowChange]:
        """report every commit HEAD gained since this store last looked. Call under the lock.

        :return: the rows those commits changed
        :rtype: list[RowChange]
        """
        head = await asyncio.to_thread(self.git.head)
        if head is None or head == self._seen_head:
            return []
        old = self._seen_head
        try:
            changes = await asyncio.to_thread(self.git.changed, old, head)
        except GitError as exc:
            # the old commit is gone (history was rewritten and collected): nothing to diff against
            log.warning(
                "git_store.head.unreadable: HEAD moved from a commit git no longer has; no rows reported",
                extra={"extra_data": {**self._log, "old": old, "head": head, "error": str(exc)}},
            )
            changes = []
        rows = await asyncio.to_thread(self._rows_of, changes, old, head, False)
        log.info(
            "git_store.head_moved: commits made elsewhere were noticed",
            extra={"extra_data": {**self._log, "old": old, "head": head, "rows": len(rows)}},
        )
        await self._reported(head, rows)
        return rows

    async def sync(self) -> list[RowChange]:
        """notice what moved now rather than at the next watch: a hand commit, or (on a clone that is
        not the writer) origin's new commits, fetched and fast-forwarded.

        :return: the rows the newly seen commits changed
        :rtype: list[RowChange]
        """
        async with self._lock:
            if not self.is_writer and self.config.clone == "own" and self.config.remote_url is not None:
                await self._catch_up_with_origin()
            return await self._notice_head()

    async def advanced(self, message: RepoAdvanced) -> None:
        """origin moved: a clone that is not the writer follows it.

        :param message: the broadcast
        :ptype message: RepoAdvanced
        :return: None
        :rtype: None
        """
        if self.is_writer or self.config.clone != "own" or message.head == self._seen_head:
            return
        await self.sync()

    # ------------------------------------------------------------------ origin

    async def _push_after_commit(self) -> bool:
        """push a commit just made, when the policy says to. Call under the lock.

        :return: whether origin took it
        :rtype: bool
        """
        if self.config.push != "after_commit":
            return False
        return await self._push_locked(explicit=False)

    async def _interval_push(self) -> None:
        """push what origin lacks, on the writer, for ``push="interval"``.

        :return: None
        :rtype: None
        """
        if not self.is_writer:
            return
        async with self._lock:
            ahead, _ = await asyncio.to_thread(self.git.ahead_behind, self.config.branch)
            if ahead:
                await self._push_locked(explicit=False)

    async def _push_locked(self, *, explicit: bool) -> bool:
        """push the branch to origin. Call under the lock.

        A refusal (origin has commits this branch lacks) is checked once against a fresh fetch: when
        origin turns out to be behind after all, the push is tried again. Otherwise pushing stops
        until someone pushes explicitly, and commits carry on locally.

        :param explicit: a person asked for it, so a stop from an earlier refusal does not apply
        :ptype explicit: bool
        :return: whether origin took it
        :rtype: bool
        """
        if self.config.remote_url is None or (self._push_blocked is not None and not explicit):
            return False
        branch = self.config.branch
        head = await asyncio.to_thread(self.git.head)
        if head is None:
            return False
        try:
            await asyncio.to_thread(self.git.push, branch)
        except GitError as first:
            if not await self._origin_is_behind(first):
                return False
            try:
                await asyncio.to_thread(self.git.push, branch)
            except GitError as second:
                self._push_failed(second)
                return False
        self._push_blocked = None
        self._last_push_error = None
        self._last_push_at = datetime.now(UTC)
        if self._nats is not None:
            await publish_advanced(self._nats, RepoAdvanced(key=self.config.key, branch=branch, head=head))
        return True

    async def _origin_is_behind(self, error: GitError) -> bool:
        """after a failed push: whether origin is only behind, so a second push is worth it.

        :param error: the failed push
        :ptype error: GitError
        :return: ``True`` when origin is an ancestor of HEAD after a fresh fetch
        :rtype: bool
        """
        rejected = "rejected" in error.stderr or "non-fast-forward" in error.stderr or "fetch first" in error.stderr
        if not rejected:
            self._push_failed(error)
            return False
        branch = self.config.branch
        try:
            await asyncio.to_thread(self.git.fetch)
        except GitError as exc:
            self._push_failed(exc)
            return False
        remote = await asyncio.to_thread(self.git.rev_parse, f"origin/{branch}")
        head = await asyncio.to_thread(self.git.head)
        if remote is not None and head is not None and await asyncio.to_thread(self.git.is_ancestor, remote, head):
            return True
        self._push_blocked = (
            f"origin/{branch} has commits {branch} lacks; pushing stopped and commits carry on locally "
            f"until a person reconciles them (this store never merges or forces)"
        )
        log.error(
            "git_store.push.rejected: origin moved without this repository; pushing stopped",
            extra={"extra_data": {**self._log, "head": head, "origin": remote}},
        )
        return False

    def _push_failed(self, error: GitError) -> None:
        """record a push that failed for a reason other than a refusal.

        :param error: the failure
        :ptype error: GitError
        :return: None
        :rtype: None
        """
        self._last_push_error = str(error)
        log.warning(
            "git_store.push.failed: origin did not take the push; the commit is safe locally",
            extra={"extra_data": {**self._log, "error": str(error)}},
        )

    # ------------------------------------------------------------------ the loops

    async def _watch(self) -> None:
        """one watch: seek the lease, notice hand commits as the writer, follow origin as a clone.

        :return: None
        :rtype: None
        """
        if self._role is not None and self.config.seek_writer and not self._role.is_writer:
            await self._role.claim()
        if self.is_writer or self.config.clone == "own":
            await self.sync()

    async def _sweep(self) -> None:
        """warn about hand edits left uncommitted longer than ``safety_idle``, on the writer.

        :return: None
        :rtype: None
        """
        if not self.is_writer:
            return
        dirty = await asyncio.to_thread(self._dirty_files)
        cutoff = datetime.now(UTC) - self.config.safety_idle
        stale = [item.path for item in dirty if item.since is None or item.since < cutoff]
        if stale:
            log.warning(
                "git_store.dirty: hand edits have sat uncommitted past the idle window; commit them",
                extra={"extra_data": {**self._log, "files": len(stale), "paths": stale[:_DIRTY_LOG_LIMIT]}},
            )

    def _dirty_files(self) -> list[DirtyFile]:
        """every hand edit in the working tree, with when it was made. BLOCKING.

        :return: the files
        :rtype: list[DirtyFile]
        """
        found: list[DirtyFile] = []
        for status, path in self.git.dirty():
            target = self.config.work_tree / path
            since: datetime | None = None
            with suppress(FileNotFoundError):  # NOSILENT: a removed file has no time; None says so
                since = datetime.fromtimestamp(target.stat().st_mtime, UTC)
            found.append(DirtyFile(path, status, since))
        return found

    # ------------------------------------------------------------------ reading

    async def history(self, path: str, *, limit: int = 50) -> list[CommitEntry]:
        """the commits that changed ``path``, newest first.

        :param path: the path, relative to the root
        :ptype path: str
        :param limit: the most to return
        :ptype limit: int
        :return: the commits
        :rtype: list[CommitEntry]
        """
        return await asyncio.to_thread(self.git.log, path=path, limit=limit)

    async def read_at(self, rev: str, path: str) -> bytes | None:
        """a file's bytes as of a commit.

        :param rev: the commit
        :ptype rev: str
        :param path: the path, relative to the root
        :ptype path: str
        :return: the bytes, or ``None`` when it did not exist there
        :rtype: bytes | None
        """
        return await asyncio.to_thread(self.git.show, rev, path)

    async def status(self) -> RepoStatus:
        """the repository at a glance: HEAD, the writer, origin, hand edits, pushing.

        :return: the status
        :rtype: RepoStatus
        """
        head = await asyncio.to_thread(self.git.head)
        ahead, behind = (0, 0)
        if self.config.remote_url is not None:
            ahead, behind = await asyncio.to_thread(self.git.ahead_behind, self.config.branch)
        dirty = await asyncio.to_thread(self._dirty_files)
        writer = None if self._role is None else await self._role.holder()
        return RepoStatus(
            key=self.config.key,
            branch=self.config.branch,
            head=head,
            is_writer=self.is_writer,
            writer=writer,
            ahead=ahead,
            behind=behind,
            dirty=tuple(dirty),
            push_blocked=self._push_blocked,
            last_push_error=self._last_push_error,
            last_push_at=self._last_push_at,
        )
