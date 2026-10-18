"""a git repository as a durable store: its working tree, its one writer, its commits and its origin.

:class:`GitRepo` owns one working tree. Every change to it is a commit made by the repository's
writer -- the process holding its NATS lease (:mod:`threetears.core.backends.git.writer`); any other process
forwards its change set to the writer and gets the commit, or the refusal, back. Without NATS
(``nats=None``: tests, one-process development) this process is always the writer.

**How a commit happens**, on the writer, under the repository's lock:

1. HEAD is checked against the last commit this store saw. A commit made by hand since is reported
   first (its rows dropped from every cache by :func:`~threetears.core.backends.git.bind.bind_collections`),
   so the checks below compare against the files as they are now.
2. Every change's compare-and-swap is checked against its file. One miss refuses the whole set:
   nothing is written (:class:`~threetears.core.backends.git.errors.ChangeConflictError`).
3. A file someone edited by hand and has not committed is never written over
   (:class:`~threetears.core.backends.git.errors.HandEditPendingError`).
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

**Saved, not yet committed.** A folder with a :class:`~threetears.core.backends.git.tail.PendingTail`
(:meth:`GitRepo.add_tail`) holds edits saved before they are committed (scriob's op-log). Its files are
the store's own work, never a hand edit: they are rebuilt from the tail when this process becomes the
writer, joined to any commit that touches the folder (with the tail's trailers, so the newest trailer
always says how far git reaches), committed by :meth:`GitRepo.commit_intent`, and committed by the sweep
once the folder has had no commit for ``safety_idle``. Taking over, the writer first puts that folder's
changed files back as HEAD has them (hand edits there included), then replays the tail.
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
from threetears.core.backends.git.envelope import DEFAULT_ISSUER, ForwardEnvelope, GitDirReplayAnchor
from threetears.core.backends.git.advanced import AdvancedListener, RepoAdvanced, publish_advanced
from threetears.core.backends.git.change import Change, ChangeSet, CommitResult, same_version
from threetears.core.backends.git.errors import (
    FileConflictError,
    ChangeConflictError,
    ConflictedRow,
    DivergedError,
    GitStoreError,
    HandEditPendingError,
    NotTheWriterError,
)
from threetears.core.backends.git.git import Author, CommitEntry, FileChange, Git, GitError
from threetears.core.backends.git.layout import Layout, locate
from threetears.core.backends.git.tail import PendingTail
from threetears.core.backends.git.wire import (
    ForwardedRequest,
    decode_read,
    decode_reply,
    decode_request,
    decode_rows,
    encode_failure,
    encode_read,
    encode_request,
    encode_result,
    encode_rows,
)
from threetears.core.backends.git.writer import COMMIT_HANDLER, READ_HANDLER, WriterRole
from threetears.nats.errors import NatsClientError
from threetears.nats.forward import ForwardedHandlerError, ForwardHandler, NoOwnerError
from threetears.observe import PeriodicTask, get_logger, retry_bounded

if TYPE_CHECKING:
    from threetears.core.coordination import ReplayAnchor
    from threetears.nats import NatsClient

__all__ = [
    "Committed",
    "DirtyFile",
    "GitRepo",
    "GitRepoConfig",
    "PendingWrite",
    "RepoStatus",
    "RowChange",
    "RowsChanged",
]

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
#: the message of the sweep's commit of saved edits left idle.
_SAFETY_MESSAGE: Final[str] = "safety commit: saved edits idle past the window"


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
    :ivar safety_idle: how long a hand edit may sit uncommitted before the sweep warns about it, and how long a
        folder with saved edits (a pending tail) may go without a commit before the sweep commits them
    :ivar sweep_interval: how often the sweep looks
    :ivar committer: who every commit names as committer, and as author when the change names none
    :ivar lease_bucket: the writer lease's bucket name (the NATS client adds its namespace)
    :ivar lease_ttl: how long the lease outlives its holder's last renewal
    :ivar lease_renew: how often the holder renews it
    :ivar watch_interval: how often the writer looks for hand commits, a clone follows origin, and a
        process seeking the lease tries for it
    :ivar forward_signing_seed: the deployment's Ed25519 seed (32 bytes, base64) every forwarded request is sealed
        with and the writer admits by (envelope.py); required with NATS, the same in every process of a deployment,
        never logged
    :ivar forward_issuer: the sending service's name in the seal, the same in every process of a deployment
    :ivar envelope_bucket: the KV bucket the writer spends each request's single-use id in
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
    forward_signing_seed: SecretStr | None = None
    forward_issuer: str = DEFAULT_ISSUER
    envelope_bucket: str = "git-store-envelopes"

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
    :ivar local: this process made the commit (through this store), or saved the edit
    :ivar sha: the commit; for an edit saved and not yet committed (a pending tail), the HEAD it sits on
    """

    table: str
    pk: Mapping[str, Any]
    deleted: bool
    local: bool
    sha: str


#: a callback told of the rows a commit changed.
RowsChanged = Callable[[list[RowChange]], Awaitable[None]]
#: a callback told of every commit this store makes, on the writer.
Committed = Callable[[CommitResult], Awaitable[None]]


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
    :ivar dirty: hand edits not yet committed (saved edits of a pending tail are not hand edits)
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


def _scope_key(scope: str | None) -> str | None:
    """a tail's folder, normalized: ``None`` (or ``"."``) for the whole repository.

    :param scope: the folder, relative to the root
    :ptype scope: str | None
    :return: the folder, or ``None``
    :rtype: str | None
    :raises ValueError: for a folder outside the working tree
    """
    if scope is None or scope in ("", "."):
        return None
    return _checked_paths([scope])[0]


def _covers(scope: str | None, path: str) -> bool:
    """whether ``path`` is under ``scope``.

    :param scope: the folder; ``None`` for the whole repository
    :ptype scope: str | None
    :param path: the path, relative to the root
    :ptype path: str
    :return: ``True`` when it is
    :rtype: bool
    """
    return scope is None or path == scope or path.startswith(scope + "/")


def _with_trailers(message: str, trailers: Mapping[str, str]) -> str:
    """a whole commit message with more trailer lines, added to its trailer block when it has one.

    :param message: the whole message
    :ptype message: str
    :param trailers: the lines to add
    :ptype trailers: Mapping[str, str]
    :return: the message
    :rtype: str
    :raises ValueError: for a trailer that cannot be one line
    """
    if not trailers:
        return message
    for key, value in trailers.items():
        if not _TRAILER_KEY.fullmatch(key) or "\n" in value:
            raise ValueError(f"trailer {key!r}: {value!r} cannot be written as one trailer line")
    text = message.rstrip("\n")
    lines = "\n".join(f"{key}: {value}" for key, value in trailers.items())
    paragraphs = text.split("\n\n")
    last = paragraphs[-1].split("\n")
    in_block = len(paragraphs) > 1 and all(re.match(r"[A-Za-z0-9][A-Za-z0-9-]*: ", line) for line in last)
    return f"{text}\n{lines}\n" if in_block else f"{text}\n\n{lines}\n"


class PendingWrite:
    """a save into a pending tail's folder, under the repository's lock (:meth:`GitRepo.pending`)."""

    def __init__(self, scope: str | None, marked: set[str]) -> None:
        """start the save.

        :param scope: the tail's folder
        :ptype scope: str | None
        :param marked: the folder's uncommitted paths, added to as the save writes
        :ptype marked: set[str]
        :return: None
        :rtype: None
        """
        self.scope = scope
        self._marked = marked
        self.written: list[str] = []

    def wrote(self, paths: Sequence[str]) -> None:
        """say which files the save wrote: they are the store's, uncommitted, until the next commit takes them.

        :param paths: the paths, relative to the root
        :ptype paths: Sequence[str]
        :return: None
        :rtype: None
        :raises ValueError: for a path outside the tail's folder
        """
        checked = _checked_paths(list(paths))
        outside = [path for path in checked if not _covers(self.scope, path)]
        if outside:
            raise ValueError(f"a save into {self.scope!r} wrote outside it: {', '.join(outside)}")
        self._marked.update(checked)
        self.written.extend(checked)


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
        replay_anchor: ReplayAnchor | None = None,
    ) -> None:
        """describe the repository; nothing on disk or on NATS changes until :meth:`open`.

        :param config: the settings
        :ptype config: GitRepoConfig
        :param layout: where each table's rows live
        :ptype layout: Layout
        :param nats: the connected client the writer lease, forwards and broadcasts use; ``None``
            makes this process always the writer, with no forwarding and no broadcast
        :ptype nats: NatsClient | None
        :param replay_anchor: where the forward seal's replay ledger remembers its first existence; ``None`` keeps
            it in the repository's git directory (:class:`~threetears.core.backends.git.envelope.GitDirReplayAnchor`)
        :ptype replay_anchor: ReplayAnchor | None
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
        self._replay_anchor = replay_anchor
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
        self._tails: dict[str | None, PendingTail] = {}
        #: each tail's files written and not yet committed
        self._pending: dict[str | None, set[str]] = {}
        self._commit_listeners: list[Committed] = []

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
            if self.config.forward_signing_seed is None:
                # forwarding unsigned would let any connection on the forward family commit (envelope.py)
                raise GitStoreError(
                    f"git-store repository {self.config.key!r}: forward_signing_seed is required with NATS"
                )
            self._role = WriterRole(
                self._nats,
                envelope=ForwardEnvelope(
                    self._nats,
                    seed=self.config.forward_signing_seed,
                    audience=self.config.lease_key,
                    issuer=self.config.forward_issuer,
                    bucket=self.config.envelope_bucket,
                    anchor=self._replay_anchor or GitDirReplayAnchor(self.git.git_path("git-store")),
                ),
                lease_key=self.config.lease_key,
                bucket=self.config.lease_bucket,
                ttl=self.config.lease_ttl,
                renew_every=self.config.lease_renew,
                takeover=self._takeover,
                forward_timeout=self.config.forward_timeout,
                log_extra=self._log,
            )
            await self._role.serve(COMMIT_HANDLER, self._handle_forwarded)
            await self._role.serve(READ_HANDLER, self._handle_read)
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

    async def forward(
        self, name: str, payload: bytes, *, local: Callable[[bytes], Awaitable[bytes]] | None = None
    ) -> bytes:
        """run a product's own operation on the writer: here when this process is it, else sent to its handler.

        With ``local``, the rule the store's own commits follow: when this process is the writer (always, without
        NATS) ``local(payload)`` runs here; when no process is, this one tries once to become it and then runs
        ``local``. Without ``local`` the request is only ever sent.

        :param name: the operation's name
        :ptype name: str
        :param payload: the request
        :ptype payload: bytes
        :param local: the operation, run on this process when it is (or becomes) the writer
        :ptype local: Callable[[bytes], Awaitable[bytes]] | None
        :return: the reply
        :rtype: bytes
        :raises RuntimeError: without NATS and without ``local``
        :raises NoOwnerError: without ``local``, when no writer serves it now
        :raises NotTheWriterError: with ``local``, when no process is the writer and this one could not become it
        :raises ForwardedHandlerError: when the writer's handler raised
        """
        if local is not None:
            if self.is_writer:
                return await local(payload)
            assert self._role is not None  # is_writer is True without a role
            try:
                return await self._role.forward(name, payload)
            except NoOwnerError:
                await self._become_writer()
                return await local(payload)
            except ForwardedHandlerError:
                raise  # the writer ran it and its handler raised: the sender sees that error, not "unreachable"
            except NatsClientError as exc:
                raise NotTheWriterError(
                    f"the writer of {self.config.lease_key} could not be reached ({type(exc).__name__}: {exc}); "
                    f"{name} did not run here"
                ) from exc
        if self._role is None:
            raise RuntimeError("a repository opened without NATS forwards nothing")
        return await self._role.forward(name, payload)

    async def _become_writer(self) -> None:
        """take the lease when no process holds it (it runs the takeover), or say why not.

        :return: None
        :rtype: None
        :raises NotTheWriterError: when the lease could not be taken
        """
        assert self._role is not None
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

    async def _takeover(self) -> None:
        """make this process's working tree the repository's current state, on becoming the writer.

        Recover a change that was being written when its writer stopped (its paths go back to HEAD:
        its caller never heard "saved"), put each pending tail's folder back as HEAD has it, catch up
        with origin when this branch is only behind it, report every commit made since the store last
        looked, rebuild each tail's saved edits on top, and push what origin lacks.

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
            for scope in list(self._tails):
                await asyncio.to_thread(self._restore_scope, scope)
            if self.config.remote_url is not None:
                await self._catch_up_with_origin()
            await self._notice_head()
            await asyncio.to_thread(self._record_head, self._seen_head)
            for scope in list(self._tails):
                await self._replay_tail(scope)
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
        recorded = json.loads(journal.read_text(encoding="utf-8"))
        paths = [str(path) for path in recorded.get("paths", [])]
        self.git.restore(paths)
        for path in paths:
            target = self.config.work_tree / path
            for leftover in target.parent.glob(f"{target.name}.tmp.*"):
                leftover.unlink(missing_ok=True)
        self._remove_new_dirs([str(d) for d in recorded.get("dirs", [])])
        journal.unlink(missing_ok=True)
        return paths

    # ------------------------------------------------------------------ saved, not yet committed

    async def add_tail(self, tail: PendingTail) -> None:
        """make ``tail``'s folder hold saved edits not yet committed; on the writer, rebuild them now.

        Rebuilding first puts the folder's changed files back as HEAD has them (a hand edit there too).

        :param tail: the tail
        :ptype tail: PendingTail
        :return: None
        :rtype: None
        :raises ValueError: when its folder overlaps another tail's
        """
        scope = _scope_key(tail.scope)
        for other in self._tails:
            if other is None or scope is None or _covers(other, scope) or _covers(scope, other):
                raise ValueError(f"a pending tail over {scope!r} overlaps the one over {other!r}")
        self._tails[scope] = tail
        self._pending[scope] = set()
        if self._opened and self.is_writer:
            async with self._lock:
                await asyncio.to_thread(self._restore_scope, scope)
                await self._replay_tail(scope)

    def remove_tail(self, scope: str | None) -> PendingTail | None:
        """stop keeping a folder's saved edits (a story deleted or moved). Its uncommitted files stay as they are.

        :param scope: the folder
        :ptype scope: str | None
        :return: the tail, or ``None`` when the folder had none
        :rtype: PendingTail | None
        """
        key = _scope_key(scope)
        self._pending.pop(key, None)
        return self._tails.pop(key, None)

    def pending_paths(self, scope: str | None = None) -> set[str]:
        """the files saved and not yet committed.

        :param scope: one tail's folder; ``None`` for every tail's
        :ptype scope: str | None
        :return: the paths
        :rtype: set[str]
        """
        if scope is not None:
            return set(self._pending.get(_scope_key(scope), ()))
        return set().union(*self._pending.values()) if self._pending else set()

    @asynccontextmanager
    async def pending(self, scope: str | None) -> AsyncIterator[PendingWrite]:
        """save into a tail's folder without committing, on the writer, under the repository's lock.

        :class:`~threetears.core.backends.git.tail.OpLogTail` saves through it. The block writes files and says
        which with :meth:`PendingWrite.wrote`; when it ends, their rows are reported (so caches drop them)::

            async with repo.pending("stories/one") as save:
                save.wrote(["stories/one/scenes/a.md"])

        :param scope: the tail's folder
        :ptype scope: str | None
        :return: the save, for the block
        :rtype: AsyncIterator[PendingWrite]
        :raises GitStoreError: when the folder has no tail
        :raises NotTheWriterError: when this process isn't the writer
        """
        key = _scope_key(scope)
        if key not in self._tails:
            raise GitStoreError(f"{self.config.key}: no pending tail keeps {key!r}; add_tail first")
        async with self._lock:
            self._require_writer()
            await self._notice_head()
            save = PendingWrite(key, self._pending.setdefault(key, set()))
            yield save
            if save.written:
                head = await asyncio.to_thread(self.git.head)
                rows = await asyncio.to_thread(self._rows_in_tree, sorted(set(save.written)), head or "")
                await self._tell(rows)

    async def commit_intent(
        self,
        scope: str | None,
        message: str,
        *,
        author: Author | None = None,
        idle: timedelta | None = None,
    ) -> CommitResult:
        """commit a tail's saved edits now (the user means it), on the writer.

        The tail is replayed first, so the commit holds every saved edit; it carries the tail's trailers.
        Edits that net to no change make no commit.

        :param scope: the tail's folder
        :ptype scope: str | None
        :param message: the commit message
        :ptype message: str
        :param author: who the commit names as author; ``None`` for the committer
        :ptype author: Author | None
        :param idle: commit only when the folder's last commit is older than this (a safety commit);
            ``None`` commits now
        :ptype idle: timedelta | None
        :return: the commit; ``sha`` is ``None`` when nothing was committed
        :rtype: CommitResult
        :raises GitStoreError: when the folder has no tail
        :raises NotTheWriterError: when no writer could take it
        """
        key = _scope_key(scope)
        full = _message_with_trailers(message, None)
        request = ForwardedRequest(
            op="commit_intent",
            scope=key,
            message=full,
            author=author,
            idle=None if idle is None else idle.total_seconds(),
        )
        return await self._as_writer(request, lambda: self._commit_intent_local(key, full, author, idle))

    def on_committed(self, callback: Committed) -> Callable[[], None]:
        """be told of every commit this store makes (on the writer, where it is made).

        :param callback: called with each commit's result, after it is pushed
        :ptype callback: Committed
        :return: a function that stops the calls
        :rtype: Callable[[], None]
        """
        self._commit_listeners.append(callback)

        def remove() -> None:
            if callback in self._commit_listeners:
                self._commit_listeners.remove(callback)

        return remove

    async def _announce(self, result: CommitResult) -> CommitResult:
        """tell every commit listener of a commit; one that fails is logged. Call outside the lock.

        :param result: the commit
        :ptype result: CommitResult
        :return: the same result
        :rtype: CommitResult
        """
        if result.sha is None:
            return result
        for callback in list(self._commit_listeners):
            try:
                await callback(result)
            except (
                Exception
            ) as exc:  # prawduct:allow prawduct/broad-except -- one listener's failure must not stop the rest
                log.warning(
                    "git_store.commit_listener.failed: a commit listener raised",
                    extra={"extra_data": {**self._log, "error": f"{type(exc).__name__}: {exc}"}},
                    exc_info=True,
                )
        return result

    def _restore_scope(self, scope: str | None) -> None:
        """put a tail's folder back as HEAD has it: its saved files and every tracked change. BLOCKING.

        :param scope: the folder
        :ptype scope: str | None
        :return: None
        :rtype: None
        """
        changed = {path for status, path in self.git.dirty(None if scope is None else [scope]) if status != "??"}
        paths = sorted(changed | self._pending.get(scope, set()))
        if paths:
            self.git.restore(paths)
            log.info(
                "git_store.tail.reset: a tail's folder was put back as HEAD has it before its replay",
                extra={"extra_data": {**self._log, "scope": scope, "paths": paths[:_DIRTY_LOG_LIMIT]}},
            )
        self._pending[scope] = set()

    async def _replay_tail(self, scope: str | None) -> set[str]:
        """rebuild a tail's saved edits and mark its files as the store's. Call under the lock, on the writer.

        :param scope: the folder
        :ptype scope: str | None
        :return: the folder's uncommitted paths
        :rtype: set[str]
        """
        save = PendingWrite(scope, self._pending.setdefault(scope, set()))
        save.wrote(list(await self._tails[scope].replay(self)))
        if save.written:
            head = await asyncio.to_thread(self.git.head)
            await self._tell(await asyncio.to_thread(self._rows_in_tree, sorted(set(save.written)), head or ""))
        return self._pending[scope]

    async def _flush_tails(self, paths: Sequence[str]) -> tuple[list[str], dict[str, str], list[str | None]]:
        """the saved edits a commit of ``paths`` takes along: every tail whose folder it touches. Under the lock.

        :param paths: the commit's own paths
        :ptype paths: Sequence[str]
        :return: the tails' uncommitted paths, their trailers, and their folders
        :rtype: tuple[list[str], dict[str, str], list[str | None]]
        :raises GitStoreError: when two tails would give one commit different values of one trailer
        """
        extra: set[str] = set()
        trailers: dict[str, str] = {}
        scopes = [scope for scope in self._tails if any(_covers(scope, path) for path in paths)]
        for scope in scopes:
            extra |= await self._replay_tail(scope)
            for key, value in (await self._tails[scope].trailers(self)).items():
                if trailers.get(key, value) != value:
                    raise GitStoreError(
                        f"one commit would carry two {key} trailers ({trailers[key]} and {value}): it touches the "
                        f"folders of two pending tails; commit them apart"
                    )
                trailers[key] = value
        return sorted(extra - set(paths)), trailers, scopes

    async def _commit_intent_local(
        self, scope: str | None, message: str, author: Author | None, idle: timedelta | None
    ) -> CommitResult:
        """replay a tail and commit what it saved, on the writer.

        :param scope: the folder
        :ptype scope: str | None
        :param message: the whole message
        :ptype message: str
        :param author: the author
        :ptype author: Author | None
        :param idle: commit only when the folder's last commit is older than this
        :ptype idle: timedelta | None
        :return: the commit
        :rtype: CommitResult
        :raises GitStoreError: when the folder has no tail
        """
        async with self._lock:
            self._require_writer()
            await self._notice_head()
            tail = self._tails.get(scope)
            if tail is None:
                raise GitStoreError(f"{self.config.key}: no pending tail keeps {scope!r}; nothing was committed")
            if idle is not None:
                last = await asyncio.to_thread(self.git.last_commit, scope)
                if last is not None and datetime.now(UTC) - last[1] <= idle:
                    return CommitResult(sha=None)
            marked = await self._replay_tail(scope)
            dirty = (
                sorted({path for _, path in await asyncio.to_thread(self.git.dirty, sorted(marked))}) if marked else []
            )
            if not dirty:
                marked.clear()  # the saved edits net to no change: nothing to commit
                return CommitResult(sha=None)
            full = _with_trailers(message, await tail.trailers(self))
            root = self.config.work_tree
            written = [path for path in dirty if (root / path).exists()]
            removed = [path for path in dirty if not (root / path).exists()]
            head_before = await asyncio.to_thread(self.git.head)
            committer = self.config.committer

            async def stage_and_commit() -> str:
                await self._retrying(lambda: self.git.stage(written, removed))
                self._require_writer()
                return await self._retrying(
                    lambda: self.git.commit(dirty, full, author=author or committer, committer=committer)
                )

            sha = await asyncio.shield(stage_and_commit())
            marked.clear()
            changes = await asyncio.to_thread(self.git.changed, head_before, sha)
            await self._reported(sha, await asyncio.to_thread(self._rows_of, changes, head_before, sha, True))
            pushed = await self._push_after_commit()
        log.info(
            "git_store.commit_intent: saved edits were committed",
            extra={"extra_data": {**self._log, "scope": scope, "sha": sha, "files": len(dirty), "pushed": pushed}},
        )
        return await self._announce(CommitResult(sha=sha, paths=tuple(dirty), pushed=pushed))

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

    async def commit_files(
        self,
        files: Mapping[str, bytes | None],
        message: str,
        *,
        author: Author | None = None,
        base: str | None = None,
        moves: Mapping[str, str] | None = None,
    ) -> CommitResult:
        """write these files' bytes (``None`` removes one) and commit them as one commit, on the writer.

        For files that aren't rows (a log, a note) and for a hand edit made on a clone of its own: that
        clone can't commit for itself (its commit would race the writer's pushes), so it sends the bytes.
        Rows among the files are reported like any commit's, so every cache drops them.

        ``moves`` renames files in the same commit: each new path gets the old one's bytes as the writer's tree
        holds them, saved edits of a pending tail included (they ride along, with the tail's trailer), and the old
        path is removed. A rename that also edits text (scriob's) saves the edit first, then moves the file.

        :param files: each path, relative to the root, with its new bytes or ``None``
        :ptype files: Mapping[str, bytes | None]
        :param message: the commit message
        :ptype message: str
        :param author: who the commit names as author; ``None`` for the committer
        :ptype author: Author | None
        :param base: the commit the sender read these files at; the writer refuses any of them changed
            since. ``None``: no check
        :ptype base: str | None
        :param moves: old path -> new path, moved in the same commit
        :ptype moves: Mapping[str, str] | None
        :return: the commit; ``sha`` is ``None`` when every file already had those bytes
        :rtype: CommitResult
        :raises ValueError: for a path outside the working tree, or a move that also appears in ``files``
        :raises GitStoreError: for a move from a file that isn't there, or onto one that is
        :raises FileConflictError: when a file changed since ``base``; nothing was written
        :raises HandEditPendingError: when the writer's tree has an uncommitted edit of one of them
        :raises NotTheWriterError: when no writer could take it
        """
        checked = dict(zip(_checked_paths(list(files)), files.values(), strict=True))
        moved = dict(zip(_checked_paths(list(moves or {})), _checked_paths(list((moves or {}).values())), strict=True))
        touched = [*moved, *moved.values()]
        if len(set(touched)) != len(touched) or set(touched) & set(checked):
            raise ValueError("a moved path appears twice, or also among the files written")
        full = _message_with_trailers(message, None)
        request = ForwardedRequest(
            op="commit_files", files=checked, base=base, message=full, author=author, moves=moved or None
        )
        return await self._as_writer(request, lambda: self._commit_files_local(checked, full, author, base, moved))

    async def pull(self) -> CommitResult:
        """take what origin has that this branch lacks, now (someone pushed from outside), on the writer.

        Fetch; fast-forward when this branch is only behind. A pending tail's saved edits are put aside for
        the fast-forward and rebuilt on the new HEAD. The rows the new commits changed are reported, so every
        cache drops them. A branch only ahead of origin is pushed instead.

        :return: ``sha``: HEAD after it; ``paths``: the files the pulled commits changed (none when there was
            nothing to take)
        :rtype: CommitResult
        :raises GitStoreError: when the repository has no origin
        :raises DivergedError: when the branch and origin have both moved on; nothing changed
        :raises GitError: when origin could not be fetched
        :raises NotTheWriterError: when no writer could take it
        """
        return await self._as_writer(ForwardedRequest(op="pull"), self._pull_local)

    async def _pull_local(self) -> CommitResult:
        """fetch and fast-forward, on the writer, under the lock.

        :return: the result
        :rtype: CommitResult
        """
        if self.config.remote_url is None:
            raise GitStoreError(f"{self.config.key} has no origin to pull from")
        branch = self.config.branch
        async with self._lock:
            self._require_writer()
            await self._notice_head()
            await asyncio.to_thread(self.git.fetch)
            remote = await asyncio.to_thread(self.git.rev_parse, f"origin/{branch}")
            head = await asyncio.to_thread(self.git.head)
            if remote is None or remote == head:
                return CommitResult(sha=head)
            if head is not None and await asyncio.to_thread(self.git.is_ancestor, remote, head):
                pushed = await self._push_locked(explicit=False)  # only ahead: origin is what lags
                return CommitResult(sha=head, pushed=pushed)
            if head is not None and not await asyncio.to_thread(self.git.is_ancestor, head, remote):
                raise DivergedError(branch, head, remote)
            for scope in list(self._tails):
                await asyncio.to_thread(self._restore_scope, scope)
            try:
                await asyncio.to_thread(self.git.merge_ff_only, f"origin/{branch}")
            finally:
                for scope in list(self._tails):  # rebuilt on whatever HEAD is now
                    await self._replay_tail(scope)
            changes = await asyncio.to_thread(self.git.changed, head, remote)
            await self._notice_head()
            self._push_blocked = None
        log.info(
            "git_store.pull: origin's new commits were taken",
            extra={"extra_data": {**self._log, "old": head, "head": remote, "files": len(changes)}},
        )
        return CommitResult(sha=remote, paths=tuple(sorted(change.path for change in changes)))

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
            await self._become_writer()
            return await local()
        return decode_reply(reply)

    def read_local(self, table: str, pk: Mapping[str, Any]) -> dict[str, Any] | None:
        """a row as this process's working tree holds it now. BLOCKING.

        :param table: the table
        :ptype table: str
        :param pk: the key
        :ptype pk: Mapping[str, Any]
        :return: the row, or ``None``
        :rtype: dict[str, Any] | None
        """
        root = self.config.work_tree
        path = locate(self.layout, root, table, pk)
        if path is None:
            return None
        target = root / path.as_posix()
        if not target.is_file():
            return None
        return self.layout.decode(table, path, target.read_bytes())

    def scan_local(self, table: str, filters: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        """every row of ``table`` matching ``filters`` in this process's working tree. BLOCKING.

        :param table: the table
        :ptype table: str
        :param filters: equality filters
        :ptype filters: Mapping[str, Any] | None
        :return: the rows, in path order
        :rtype: list[dict[str, Any]]
        """
        layout = self.layout
        root = self.config.work_tree
        wanted = dict(filters or {})
        rows: list[dict[str, Any]] = []
        for candidate in sorted(root.glob(layout.candidates(table, wanted))):
            if not candidate.is_file():
                continue
            relative = PurePosixPath(candidate.relative_to(root).as_posix())
            data = candidate.read_bytes()
            found = layout.identify(relative, data if layout.reads_content else None)
            if found is None or found[0] != table:
                continue
            row = layout.decode(table, relative, data)
            if all(column in row and row[column] == value for column, value in wanted.items()):
                rows.append(row)
        return rows

    def _reads_from_writer(self) -> bool:
        """whether this process's reads go to the writer: a clone of its own that isn't the writer.

        Its working tree lags the writer's until it fetches, so a read there could return a row the writer has
        already changed, and a collection would cache it. With every such read answered by the writer, the git tier
        is one current copy from every process's view, as a shared database is, and 3tears' collections (evict on
        every commit, re-read on a miss) show current data on every pod (Pace, 2026-10-18: switch pods and see the
        proper data). A shared working tree is the writer's own files, so it reads them directly.

        :return: ``True`` to ask the writer
        :rtype: bool
        """
        return self.config.clone == "own" and self._role is not None and not self.is_writer

    async def _read_via_writer(
        self, kind: str, table: str, key: Mapping[str, Any] | None
    ) -> list[dict[str, Any]] | None:
        """the writer's answer to a read, or ``None`` when no writer serves (this process then reads its own tree).

        :param kind: ``"fetch"`` or ``"scan"``
        :ptype kind: str
        :param table: the table
        :ptype table: str
        :param key: the key or the filters
        :ptype key: Mapping[str, Any] | None
        :return: the rows, or ``None``
        :rtype: list[dict[str, Any]] | None
        :raises GitStoreError: when the writer failed the read, or NATS could not be reached (a read from this
            process's own clone could be stale, so it isn't offered in its place)
        """
        assert self._role is not None
        try:
            reply = await self._role.forward(READ_HANDLER, encode_read(kind, table, key))
        except NoOwnerError:
            # NOSILENT: no writer means nobody is changing the repository, so this clone is as current as any;
            # the caller reads it locally
            return None
        except ForwardedHandlerError as exc:
            raise GitStoreError(
                f"the writer of {self.config.lease_key} failed a read: {exc.type_name}: {exc.message}"
            ) from exc
        except NatsClientError as exc:
            raise GitStoreError(
                f"the writer of {self.config.lease_key} could not be reached for a read ({type(exc).__name__}: {exc})"
            ) from exc
        return decode_rows(reply)

    async def read_row(self, table: str, pk: Mapping[str, Any]) -> dict[str, Any] | None:
        """the current row with key ``pk``: this working tree's, or the writer's when this clone may lag.

        :param table: the table
        :ptype table: str
        :param pk: the key
        :ptype pk: Mapping[str, Any]
        :return: the row, or ``None``
        :rtype: dict[str, Any] | None
        """
        if self._reads_from_writer():
            rows = await self._read_via_writer("fetch", table, pk)
            if rows is not None:
                return rows[0] if rows else None
        return await asyncio.to_thread(self.read_local, table, pk)

    async def read_rows(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """the current rows of ``table`` matching ``filters``: this working tree's, or the writer's.

        :param table: the table
        :ptype table: str
        :param filters: equality filters; ``None`` for every row
        :ptype filters: Mapping[str, Any] | None
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        if self._reads_from_writer():
            rows = await self._read_via_writer("scan", table, filters)
            if rows is not None:
                return rows
        return await asyncio.to_thread(self.scan_local, table, filters)

    async def _handle_read(self, payload: bytes) -> bytes:
        """answer another process's read, on the writer, from the writer's working tree.

        :param payload: the read
        :ptype payload: bytes
        :return: the rows
        :rtype: bytes
        """
        self._require_writer()
        kind, table, key = decode_read(payload)
        if kind == "fetch":
            row = await asyncio.to_thread(self.read_local, table, key)
            return encode_rows([] if row is None else [row])
        return encode_rows(await asyncio.to_thread(self.scan_local, table, key))

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
            elif request.op == "commit_files":
                result = await self._commit_files_local(
                    dict(request.files or {}), request.message, request.author, request.base, dict(request.moves or {})
                )
            elif request.op == "pull":
                result = await self._pull_local()
            elif request.op == "commit_intent":
                idle = None if request.idle is None else timedelta(seconds=request.idle)
                result = await self._commit_intent_local(request.scope, request.message, request.author, idle)
            else:
                result = await self._push_now()
        except (GitStoreError, ChangeConflictError, GitError, ValueError) as exc:
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
            sha, taken = await asyncio.shield(
                self._write_and_commit(plan, message, author or self.config.committer, date)
            )
            rows = [RowChange(table, pk, deleted, True, sha) for table, pk, deleted in plan.rows]
            await self._reported(sha, rows)
            pushed = await self._push_after_commit()
            paths = tuple(sorted({*plan.writes, *taken}))
        log.info(
            "git_store.commit: a change set was committed",
            extra={"extra_data": {**self._log, "sha": sha, "files": len(paths), "pushed": pushed}},
        )
        return await self._announce(CommitResult(sha=sha, paths=paths, pushed=pushed))

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

    async def _write_and_commit(
        self, plan: _Plan, message: str, author: Author, date: datetime | None, *, moving: set[str] | None = None
    ) -> tuple[str, list[str]]:
        """refuse a hand edit in the way, then journal, write, stage and commit the plan's files, with the saved
        edits of every pending tail whose folder they touch (and its trailers).

        :param plan: the plan
        :ptype plan: _Plan
        :param message: the whole message
        :ptype message: str
        :param author: the author
        :ptype author: Author
        :param date: the author date
        :ptype date: datetime | None
        :param moving: files being moved away: their saved edits travel with the move, so they may be removed
        :ptype moving: set[str] | None
        :return: the new commit, and the saved files of pending tails it took along
        :rtype: tuple[str, list[str]]
        :raises HandEditPendingError: when a file to write differs from HEAD
        :raises GitStoreError: when a file to write holds saved edits not yet committed
        :raises NotTheWriterError: when the lease was lost before the commit; nothing was committed
        """
        paths = sorted(plan.writes)
        saved = self.pending_paths()
        dirty = await asyncio.to_thread(self.git.dirty, paths)
        self._refuse_dirty([(status, path) for status, path in dirty if not (path in (moving or ()) and path in saved)])
        taken, trailers, scopes = await self._flush_tails(paths)
        message = _with_trailers(message, trailers)
        await asyncio.to_thread(self._write_journal, paths)
        try:
            await asyncio.to_thread(self._apply, plan)
            root = self.config.work_tree
            written = [path for path in paths if plan.writes[path] is not None]
            removed = [path for path in paths if plan.writes[path] is None]
            written += [path for path in taken if (root / path).exists()]
            removed += [path for path in taken if not (root / path).exists()]
            everything = sorted({*paths, *taken})
            await self._retrying(lambda: self.git.stage(written, removed))
            self._require_writer()
            sha = await self._retrying(
                lambda: self.git.commit(everything, message, author=author, committer=self.config.committer, date=date)
            )
        except BaseException:
            await asyncio.to_thread(self._undo, paths)
            for scope in scopes:  # a saved file the undo put back as HEAD has it gets its saved edits again
                await self._replay_tail(scope)
            raise
        await asyncio.to_thread(self._clear_journal)
        for scope in scopes:
            self._pending[scope].clear()
        return sha, taken

    def _refuse_dirty(self, dirty: Sequence[tuple[str, str]]) -> None:
        """refuse to write over files that differ from HEAD: saved edits not yet committed, or hand edits.

        :param dirty: ``(status, path)`` of the files to write that differ from HEAD
        :ptype dirty: Sequence[tuple[str, str]]
        :return: None
        :rtype: None
        :raises GitStoreError: when one holds saved edits (commit them first: :meth:`commit_intent`)
        :raises HandEditPendingError: when one holds a hand edit
        """
        if not dirty:
            return
        saved = self.pending_paths()
        held = sorted(path for _, path in dirty if path in saved)
        if held:
            raise GitStoreError(
                f"saved edits not yet committed in the way, nothing was written: {', '.join(held)}. "
                f"Commit them first (commit_intent)"
            )
        raise HandEditPendingError([path for _, path in dirty])

    def _write_journal(self, paths: Sequence[str]) -> None:
        """record the paths about to change, for recovery after a crash. BLOCKING.

        :param paths: the paths
        :ptype paths: Sequence[str]
        :return: None
        :rtype: None
        """
        journal = self.git.git_path(_JOURNAL)
        journal.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_sync(journal, json.dumps({"paths": list(paths), "dirs": self._new_dirs(paths)}), mode=0o644)

    def _new_dirs(self, paths: Sequence[str]) -> list[str]:
        """the folders writing ``paths`` would create (none of them there now), deepest first. BLOCKING.

        :param paths: the paths about to be written
        :ptype paths: Sequence[str]
        :return: the folders, relative to the root
        :rtype: list[str]
        """
        root = self.config.work_tree
        missing: set[str] = set()
        for path in paths:
            for parent in PurePosixPath(path).parents:
                if parent == PurePosixPath(".") or (root / parent.as_posix()).exists():
                    break
                missing.add(parent.as_posix())
        return sorted(missing, key=lambda d: (-d.count("/"), d))

    def _remove_new_dirs(self, dirs: Sequence[str]) -> None:
        """remove folders an undone change created, deepest first, each only when it is empty. BLOCKING.

        :param dirs: the folders, relative to the root, deepest first
        :ptype dirs: Sequence[str]
        :return: None
        :rtype: None
        """
        for name in dirs:
            folder = self.config.work_tree / name
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()

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
        journal = self.git.git_path(_JOURNAL)
        try:
            self.git.restore(paths)
            dirs = json.loads(journal.read_text(encoding="utf-8")).get("dirs", []) if journal.is_file() else []
            self._remove_new_dirs([str(d) for d in dirs])
        except (GitError, OSError) as exc:
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
            saved = self.pending_paths()
            hand = sorted({path for _, path in dirty} - saved)
            if not hand:
                return CommitResult(sha=None)
            taken, trailers, scopes = await self._flush_tails(hand)
            full = _with_trailers(message, trailers)
            targets = sorted({*hand, *taken})
            root = self.config.work_tree
            written = [path for path in targets if (root / path).exists()]
            removed = [path for path in targets if not (root / path).exists()]
            head_before = await asyncio.to_thread(self.git.head)
            committer = self.config.committer

            async def stage_and_commit() -> str:
                await self._retrying(lambda: self.git.stage(written, removed))
                self._require_writer()
                return await self._retrying(
                    lambda: self.git.commit(targets, full, author=author or committer, committer=committer)
                )

            sha = await asyncio.shield(stage_and_commit())
            for scope in scopes:
                self._pending[scope].clear()
            changes = await asyncio.to_thread(self.git.changed, head_before, sha)
            rows = await asyncio.to_thread(self._rows_of, changes, head_before, sha, True)
            await self._reported(sha, rows)
            pushed = await self._push_after_commit()
        log.info(
            "git_store.commit_paths: hand edits were committed",
            extra={"extra_data": {**self._log, "sha": sha, "files": len(targets), "pushed": pushed}},
        )
        return await self._announce(CommitResult(sha=sha, paths=tuple(targets), pushed=pushed))

    async def _moved_bytes(self, moves: Mapping[str, str]) -> dict[str, bytes | None]:
        """the writes a set of moves comes to: each new path gets the old one's bytes as the tree holds them now
        (saved edits rebuilt first), each old path is removed. Under the lock.

        :param moves: old path -> new path
        :ptype moves: Mapping[str, str]
        :return: the writes
        :rtype: dict[str, bytes | None]
        :raises GitStoreError: for a move from a missing file, or onto an existing one
        """
        for scope in [scope for scope in self._tails if any(_covers(scope, path) for path in moves)]:
            await self._replay_tail(scope)
        root = self.config.work_tree
        writes: dict[str, bytes | None] = {}
        for old, new in moves.items():
            source, target = root / old, root / new
            if not source.is_file():
                raise GitStoreError(f"cannot move {old}: there is no such file")
            if target.exists():
                raise GitStoreError(f"cannot move {old} onto {new}: {new} exists")
            writes[new] = await asyncio.to_thread(source.read_bytes)
            writes[old] = None
        return writes

    async def _commit_files_local(
        self,
        files: dict[str, bytes | None],
        message: str,
        author: Author | None,
        base: str | None,
        moves: dict[str, str] | None = None,
    ) -> CommitResult:
        """write and commit files sent whole, on the writer.

        :param files: each path with its new bytes, or ``None`` to remove it
        :ptype files: dict[str, bytes | None]
        :param message: the whole message
        :ptype message: str
        :param author: the author
        :ptype author: Author | None
        :param base: the commit the sender read them at; ``None`` for no check
        :ptype base: str | None
        :param moves: old path -> new path
        :ptype moves: dict[str, str] | None
        :return: the commit
        :rtype: CommitResult
        :raises FileConflictError: when a file changed since ``base``
        :raises GitStoreError: when this repository doesn't have ``base``
        :raises HandEditPendingError: when one of them has an uncommitted edit here
        """
        async with self._lock:
            self._require_writer()
            await self._notice_head()
            head_before = await asyncio.to_thread(self.git.head)
            if base is not None and base != head_before:
                if await asyncio.to_thread(self.git.rev_parse, base) is None:
                    raise GitStoreError(f"the writer of {self.config.lease_key} has no commit {base}")
                moved = {change.path for change in await asyncio.to_thread(self.git.changed, base, head_before or base)}
                stale = sorted(moved & {*files, *(moves or {})})
                if stale:
                    raise FileConflictError(stale)
            if moves:
                files = {**files, **await self._moved_bytes(moves)}

            def differing() -> dict[str, bytes | None]:
                current = {path: None if head_before is None else self.git.show(head_before, path) for path in files}
                return {path: data for path, data in files.items() if current[path] != data}

            writes = await asyncio.to_thread(differing)
            if not writes:
                return CommitResult(sha=None)
            sha, taken = await asyncio.shield(
                self._write_and_commit(
                    _Plan(writes=writes), message, author or self.config.committer, None, moving=set(moves or {})
                )
            )
            changes = await asyncio.to_thread(self.git.changed, head_before, sha)
            rows = await asyncio.to_thread(self._rows_of, changes, head_before, sha, True)
            await self._reported(sha, rows)
            pushed = await self._push_after_commit()
            paths = tuple(sorted({*writes, *taken}))
        log.info(
            "git_store.commit_files: files were committed",
            extra={"extra_data": {**self._log, "sha": sha, "files": len(paths), "pushed": pushed}},
        )
        return await self._announce(CommitResult(sha=sha, paths=paths, pushed=pushed))

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
        await self._tell(rows)

    async def _tell(self, rows: list[RowChange]) -> None:
        """tell every changed-rows listener of ``rows``; one that fails is logged, the others are still told.

        :param rows: the rows
        :ptype rows: list[RowChange]
        :return: None
        :rtype: None
        """
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

    def _rows_in_tree(self, paths: Sequence[str], head: str) -> list[RowChange]:
        """the rows files in the working tree hold, as saved edits of this process. BLOCKING.

        :param paths: the files, relative to the root
        :ptype paths: Sequence[str]
        :param head: the commit they sit on
        :ptype head: str
        :return: the rows, files that hold none left out
        :rtype: list[RowChange]
        """
        rows: list[RowChange] = []
        for path in paths:
            target = self.config.work_tree / path
            exists = target.is_file()
            data = target.read_bytes() if exists and self.layout.reads_content else None
            if not exists and self.layout.reads_content:
                data = self.git.show(head, path) if head else None
            found = self.layout.identify(PurePosixPath(path), data)
            if found is not None:
                rows.append(RowChange(found[0], found[1], not exists, True, head))
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
            if not self.is_writer and self.pending_paths():
                # this process saved edits as the writer and is no longer it: they are safe in the tail and the
                # new writer rebuilds them; on a clone of its own they would block following origin
                if self.config.clone == "own":
                    await asyncio.to_thread(self.git.restore, sorted(self.pending_paths()))
                for marked in self._pending.values():
                    marked.clear()
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
        """commit saved edits whose folder had no commit for ``safety_idle``, and warn about hand edits left
        uncommitted that long, on the writer.

        :return: None
        :rtype: None
        """
        if not self.is_writer:
            return
        for scope in list(self._tails):
            try:
                await self._commit_intent_local(scope, _SAFETY_MESSAGE, None, self.config.safety_idle)
            except (GitStoreError, GitError) as exc:
                log.warning(
                    "git_store.safety_commit.failed: saved edits could not be committed; the next sweep tries again",
                    extra={"extra_data": {**self._log, "scope": scope, "error": f"{type(exc).__name__}: {exc}"}},
                )
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
        saved = self.pending_paths()
        for status, path in self.git.dirty():
            if path in saved:
                continue
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
