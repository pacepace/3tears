"""saved but not yet committed: a pending tail of edits under one folder, and the op-log that holds it.

Some products save an edit at once but commit only when the user means it (a file switch, an accept,
a checkpoint) or after a quiet stretch (scriob: an edit is saved in a NATS op-log and in the working
tree, and committed on Save or by the 6-hour safety commit). Between the save and the commit the
working tree holds changes no commit has; they are the store's own work, not a hand edit.

A :class:`PendingTail` is that work for one folder of a repository (its ``scope``; ``None`` for the
whole repository). The repository asks it to rebuild the uncommitted files (:meth:`PendingTail.replay`)
when it becomes the writer, before a commit that takes them, and when the commit intent comes; and for
the trailers that commit carries (:meth:`PendingTail.trailers`), which say how far the commits reach.

:class:`OpLogTail` is the tail kept in a :class:`threetears.nats.OpLog`, lifted from scriob's
coordinator. Each commit carries ``Op-Seq: <n>``: every op up to ``n`` is in git. The ops after it are
the tail; after a crash or a writer change they are replayed onto HEAD. The op type and how ops change
a file stay the product's (``decode``, ``apply``).

One repair is kept from scriob, where it fixed a real incident: when the op-log is younger than git
(reset, purged, a fresh stream beside a cloned history), the last trailer names a sequence the log never
reached. That trailer is stale; every op in the log is then uncommitted (committed-through 0), and it
stays so until HEAD moves, however far the log grows meanwhile. A plain ``trailer > log head``
comparison would flip once the log caught up with the stale number and drop saved edits silently.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Generic, Protocol, TypeVar, runtime_checkable

from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.core.backends.git.repo import GitRepo
    from threetears.nats.oplog import AppendResult, OpRecord

__all__ = ["OpLogLike", "OpLogTail", "PendingTail", "committed_seq_from"]

log = get_logger(__name__)

OpT = TypeVar("OpT")

#: the trailer key a commit records its committed-through sequence under.
DEFAULT_TRAILER: Final[str] = "Op-Seq"
_TRAILER_KEY: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")


@runtime_checkable
class PendingTail(Protocol):
    """saved work under one folder that no commit has yet.

    :ivar scope: the folder, relative to the root (``"stories/one"``); ``None`` for the whole repository
    """

    scope: str | None

    async def replay(self, repo: GitRepo) -> Sequence[str]:
        """write the uncommitted work into the working tree, rebuilt from HEAD. Runs on the writer, under
        the repository's lock; the same tail gives the same bytes every time.

        :param repo: the repository
        :ptype repo: GitRepo
        :return: every path the tail touches, relative to the root
        :rtype: Sequence[str]
        """
        ...

    async def trailers(self, repo: GitRepo) -> Mapping[str, str]:
        """the trailers of a commit that takes everything the last :meth:`replay` wrote.

        :param repo: the repository
        :ptype repo: GitRepo
        :return: ``Key: value`` lines (``{"Op-Seq": "41"}``)
        :rtype: Mapping[str, str]
        """
        ...


class OpLogLike(Protocol):
    """what :class:`OpLogTail` needs of an op-log (:class:`threetears.nats.OpLog` has it)."""

    async def append(self, *, payload: bytes, op_id: str, expected_last_seq: int) -> AppendResult:
        """append one op, only if the log's last sequence is still ``expected_last_seq``.

        :param payload: the op's bytes
        :ptype payload: bytes
        :param op_id: the op's id, so a retried append is not logged twice
        :ptype op_id: str
        :param expected_last_seq: the last sequence the caller read
        :ptype expected_last_seq: int
        :return: where the op landed
        :rtype: AppendResult
        """
        ...

    def replay(self, from_seq: int) -> AsyncIterator[OpRecord]:
        """the ops from ``from_seq`` on, in order.

        :param from_seq: the first sequence wanted
        :ptype from_seq: int
        :return: the ops
        :rtype: AsyncIterator[OpRecord]
        """
        ...

    async def last_seq(self) -> int:
        """the sequence of the log's newest op (0 when empty).

        :return: the sequence
        :rtype: int
        """
        ...


def committed_seq_from(message: str, trailer: str = DEFAULT_TRAILER) -> int:
    """the committed-through sequence a commit message records: its last ``trailer`` line (none: 0).

    :param message: the whole message
    :ptype message: str
    :param trailer: the trailer key
    :ptype trailer: str
    :return: the sequence
    :rtype: int
    """
    found = re.findall(rf"^{re.escape(trailer)}:\s*(\d+)\s*$", message, flags=re.MULTILINE)
    return int(found[-1]) if found else 0


class OpLogTail(Generic[OpT]):
    """the uncommitted ops after the last commit's ``Op-Seq`` trailer, as a :class:`PendingTail`.

    Save an edit with :meth:`save` (on the writer: a product forwards its own save there with
    :meth:`GitRepo.serve` / :meth:`GitRepo.forward`); commit with :meth:`GitRepo.commit_intent`.

    :ivar scope: the folder its ops change; ``None`` for the whole repository
    :ivar oplog: the log
    :ivar trailer: the trailer key
    """

    def __init__(
        self,
        oplog: OpLogLike,
        *,
        scope: str | None,
        decode: Callable[[bytes], OpT],
        apply: Callable[[GitRepo, Sequence[OpT]], Awaitable[Sequence[str]]],
        trailer: str = DEFAULT_TRAILER,
    ) -> None:
        """describe the tail.

        :param oplog: the op-log, one per scope (scriob keys it by story)
        :ptype oplog: OpLogLike
        :param scope: the folder its ops change, relative to the root; ``None`` for the whole repository
        :ptype scope: str | None
        :param decode: an op from its logged bytes
        :ptype decode: Callable[[bytes], OpT]
        :param apply: write the files ``ops`` (in sequence order) make of HEAD's files, and return every path
            written. It must rebuild from HEAD (``repo.git.show("HEAD", path)``), never edit the file in place,
            so a second replay of the same ops gives the same bytes
        :ptype apply: Callable[[GitRepo, Sequence[OpT]], Awaitable[Sequence[str]]]
        :param trailer: the trailer key commits record the committed-through sequence under
        :ptype trailer: str
        :return: None
        :rtype: None
        :raises ValueError: for a trailer key that cannot be one trailer line
        """
        if not _TRAILER_KEY.fullmatch(trailer):
            raise ValueError(f"trailer key {trailer!r}: letters, digits and '-' only")
        self.oplog = oplog
        self.scope = scope
        self.trailer = trailer
        self._decode = decode
        self._apply = apply
        #: HEAD when a stale trailer was found; committed-through stays 0 until HEAD moves from it
        self._stale_head: str | None = None
        #: the sequence the last replay reached, for the next commit's trailer
        self._through: int | None = None

    async def committed_through(self, repo: GitRepo) -> int:
        """the last sequence git holds: the newest trailer under the scope, unless it is stale.

        :param repo: the repository
        :ptype repo: GitRepo
        :return: the sequence; 0 when no commit records one, or the one found is ahead of the log
        :rtype: int
        """
        grep = rf"^{self.trailer}:[[:space:]]*[0-9]+"
        found = await asyncio.to_thread(repo.git.last_commit, self.scope, grep=grep)
        in_git = 0 if found is None else committed_seq_from(found[2], self.trailer)
        if self._stale_head is not None:
            head = await asyncio.to_thread(repo.git.head)
            if head == self._stale_head:
                return 0
            self._stale_head = None
            log.info(
                "git_store.tail.healed: a commit moved HEAD past the stale trailer",
                extra={"extra_data": {"key": repo.config.key, "scope": self.scope}},
            )
        log_head = await self.oplog.last_seq()
        if in_git > log_head:
            self._stale_head = await asyncio.to_thread(repo.git.head)
            log.warning(
                "git_store.tail.stale_trailer: git's trailer is ahead of the op-log; every logged op counts as "
                "uncommitted until the next commit",
                extra={"extra_data": {"key": repo.config.key, "scope": self.scope, "git": in_git, "log": log_head}},
            )
            return 0
        return in_git

    async def records(self, after: int) -> list[OpRecord]:
        """the logged ops after sequence ``after``.

        :param after: the sequence
        :ptype after: int
        :return: the records, in sequence order
        :rtype: list[OpRecord]
        """
        return [record async for record in self.oplog.replay(from_seq=after + 1)]

    async def pending_ops(self, repo: GitRepo) -> list[OpRecord]:
        """the uncommitted ops.

        :param repo: the repository
        :ptype repo: GitRepo
        :return: the records after the committed-through sequence
        :rtype: list[OpRecord]
        """
        return await self.records(await self.committed_through(repo))

    async def head_seq(self, repo: GitRepo) -> int:
        """the sequence the next op is appended after: the tail's last, else the committed-through one.

        :param repo: the repository
        :ptype repo: GitRepo
        :return: the sequence
        :rtype: int
        """
        committed = await self.committed_through(repo)
        tail = await self.records(committed)
        return tail[-1].seq if tail else committed

    async def _materialise(self, repo: GitRepo, committed: int, tail: Sequence[OpRecord]) -> list[str]:
        """apply ``tail`` onto HEAD and remember how far it reaches.

        :param repo: the repository
        :ptype repo: GitRepo
        :param committed: the committed-through sequence the tail follows
        :ptype committed: int
        :param tail: the records
        :ptype tail: Sequence[OpRecord]
        :return: the paths written
        :rtype: list[str]
        """
        self._through = tail[-1].seq if tail else committed
        if not tail:
            return []
        return list(await self._apply(repo, [self._decode(record.payload) for record in tail]))

    async def replay(self, repo: GitRepo) -> list[str]:
        """write the uncommitted ops into the working tree, rebuilt from HEAD.

        :param repo: the repository
        :ptype repo: GitRepo
        :return: the paths written
        :rtype: list[str]
        """
        committed = await self.committed_through(repo)
        return await self._materialise(repo, committed, await self.records(committed))

    async def trailers(self, repo: GitRepo) -> dict[str, str]:
        """``{trailer: <the sequence the last replay reached>}``, for the commit that takes it.

        Every commit under the scope carries it, an empty tail's too, so the newest trailer always says
        how far git reaches.

        :param repo: the repository
        :ptype repo: GitRepo
        :return: the trailer
        :rtype: dict[str, str]
        """
        through = self._through if self._through is not None else await self.committed_through(repo)
        return {self.trailer: str(through)}

    async def save(
        self,
        repo: GitRepo,
        *,
        payload: bytes,
        op_id: str,
        expected_last_seq: int | None = None,
        check: Callable[[Sequence[OpT], OpT], Awaitable[None] | None] | None = None,
    ) -> AppendResult:
        """save one op: append it to the log, then write the whole tail into the working tree. No commit.

        Runs on the writer, under the repository's lock. The committed-through sequence is worked out once,
        before the append, and the same value is used after it: the append moves the log's head, and working
        it out again could count the op just appended as committed and leave it out of the files.

        :param repo: the repository
        :ptype repo: GitRepo
        :param payload: the op's bytes, as the log keeps them
        :ptype payload: bytes
        :param op_id: the op's id; a retry with the same id is not appended twice
        :ptype op_id: str
        :param expected_last_seq: the sequence the sender saw last; a stale one is refused by the log.
            ``None``: append after the current head (the writer decides the order)
        :ptype expected_last_seq: int | None
        :param check: refuse an op that would not replay, before it is logged (raise): called with the
            pending ops and the new one. An op that can't replay, once logged, would fail every replay after it
        :ptype check: Callable[[Sequence[OpT], OpT], Awaitable[None] | None] | None
        :return: the log's answer; ``seq`` is the op's sequence (the original one for a retry)
        :rtype: AppendResult
        :raises NotTheWriterError: when this process isn't the writer
        """
        async with repo.pending(self.scope) as pending:
            committed = await self.committed_through(repo)
            prior = await self.records(committed)
            if check is not None:
                verdict = check([self._decode(record.payload) for record in prior], self._decode(payload))
                if inspect.isawaitable(verdict):
                    await verdict
            expected = expected_last_seq
            if expected is None:
                expected = prior[-1].seq if prior else committed
            result = await self.oplog.append(payload=payload, op_id=op_id, expected_last_seq=expected)
            pending.wrote(await self._materialise(repo, committed, await self.records(committed)))
        return result
