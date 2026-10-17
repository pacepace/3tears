"""change sets: many row writes, one commit.

A :class:`ChangeSet` is the connection a collection write joins. It is handed to
``save_entity(..., conn=cs)`` (and to the store's ``upsert`` / ``delete``) and collects each write as
a :class:`Change`; nothing touches the working tree until the outermost transaction on it ends
cleanly, when the whole set is checked and written as ONE commit, or refused whole.

It fits 3tears' :class:`~threetears.core.collections.CallerTransaction`: that opens
``cs.transaction()``, the writes inside it enroll the keys they wrote, and once the transaction has
ended -- committed or refused -- every key is evicted from L1 and L2 everywhere. ``GitRepo.change``
opens both for you::

    async with repo.change("aria: bank 3 keepers") as cs:
        await keepers.save_entity(k1, conn=cs)
        await recipes.save_entity(recipe, conn=cs)
    cs.result.sha   # the commit, or None when nothing changed
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from threetears.git_store.git import Author

if TYPE_CHECKING:
    from threetears.git_store.repo import GitRepo

__all__ = ["Change", "ChangeSet", "CommitResult", "OnConflict", "change_key", "same_version"]

#: what a write does when its row already exists: replace it, leave it, or refuse the set.
OnConflict = Literal["update", "ignore", "raise"]


def change_key(table: str, pk: Mapping[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    """the identity of one row, for collapsing several writes of it into one.

    :param table: the table
    :ptype table: str
    :param pk: the row's key
    :ptype pk: Mapping[str, Any]
    :return: a hashable key
    :rtype: tuple[str, tuple[tuple[str, str], ...]]
    """
    return table, tuple(sorted((column, str(value)) for column, value in pk.items()))


@dataclass(frozen=True, slots=True)
class Change:
    """one row write.

    :ivar table: the table
    :ivar pk: the row's key, column to value
    :ivar row: the whole row to write, or ``None`` to delete it
    :ivar cas: the version the writer read (the row's ``date_updated``); the write applies only while
        the stored row still holds it, and a row deleted since is a miss. ``None``: no check
    :ivar on_conflict: when the row already exists: ``"update"`` it, ``"ignore"`` this write, or
        ``"raise"`` (refuse the set)
    """

    table: str
    pk: Mapping[str, Any]
    row: Mapping[str, Any] | None
    cas: datetime | None = None
    on_conflict: OnConflict = "update"


@dataclass(frozen=True, slots=True)
class CommitResult:
    """what a change set or a commit came to.

    :ivar sha: the new commit, or ``None`` when nothing changed (the files already said it)
    :ivar paths: the files the commit changed
    :ivar pushed: whether origin took it before this returned
    """

    sha: str | None
    paths: tuple[str, ...] = ()
    pushed: bool = False


@dataclass
class ChangeSet:
    """the writes of one commit, gathered until the transaction on it ends.

    Writing one row twice keeps one change: the last row, with the FIRST write's compare-and-swap
    version, which is the one read from the file. A nested transaction is a savepoint: if it ends
    by raising, only the writes made inside it are dropped.

    :ivar repo: the repository the set commits to
    :ivar message: the commit message
    :ivar author: who the commit names as author; ``None`` for the repository's committer
    :ivar trailers: ``Key: value`` lines added to the message
    :ivar date: the author date; ``None`` for now
    :ivar changes: the writes so far, one per row, in first-written order
    :ivar result: the commit, once the outermost transaction ended cleanly
    """

    repo: GitRepo
    message: str
    author: Author | None = None
    trailers: Mapping[str, str] = field(default_factory=dict)
    date: datetime | None = None
    changes: list[Change] = field(default_factory=list)
    result: CommitResult | None = None
    _depth: int = field(default=0, init=False, repr=False)
    _index: dict[tuple[str, tuple[tuple[str, str], ...]], int] = field(default_factory=dict, init=False, repr=False)

    def add(self, change: Change) -> None:
        """record a write.

        :param change: the write
        :ptype change: Change
        :return: None
        :rtype: None
        :raises RuntimeError: when the set has already been committed
        """
        if self.result is not None:
            raise RuntimeError("this change set was already committed; open another")
        key = change_key(change.table, change.pk)
        at = self._index.get(key)
        if at is None:
            self._index[key] = len(self.changes)
            self.changes.append(change)
        else:
            first = self.changes[at]
            self.changes[at] = replace(change, cas=first.cas, on_conflict=first.on_conflict)

    def pending(self, table: str, pk: Mapping[str, Any]) -> Change | None:
        """this set's write of a row, so a read inside the set sees it.

        :param table: the table
        :ptype table: str
        :param pk: the row's key
        :ptype pk: Mapping[str, Any]
        :return: the write, or ``None`` when the set has not written the row
        :rtype: Change | None
        """
        at = self._index.get(change_key(table, pk))
        return None if at is None else self.changes[at]

    def _restore(self, saved: list[Change]) -> None:
        """put the writes back as they were when a savepoint opened.

        :param saved: the writes then, rows a later write replaced included
        :ptype saved: list[Change]
        :return: None
        :rtype: None
        """
        self.changes[:] = saved
        self._index = {change_key(c.table, c.pk): i for i, c in enumerate(self.changes)}

    def transaction(self, **options: Any) -> AbstractAsyncContextManager[ChangeSet]:
        """the transaction ``CallerTransaction(cs)`` opens: committed as one commit when the outermost ends.

        :param options: accepted for the connection protocol; none apply
        :ptype options: Any
        :return: an async context manager yielding this set
        :rtype: AbstractAsyncContextManager[ChangeSet]
        """
        del options
        return self._transaction()

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[ChangeSet]:
        """open one level of the set's transaction.

        :return: this set, for the body
        :rtype: AsyncIterator[ChangeSet]
        :raises ChangeConflictError: when the outermost level commits and a row changed since read
        """
        # a copy, not a length: a write inside the savepoint may replace a row written before it
        saved = list(self.changes)
        self._depth += 1
        try:
            yield self
        except BaseException:
            self._depth -= 1
            self._restore(saved if self._depth > 0 else [])
            raise
        self._depth -= 1
        if self._depth == 0:
            self.result = await self.repo.commit(
                list(self.changes), self.message, author=self.author, trailers=self.trailers, date=self.date
            )


def same_version(stored: Any, expected: datetime) -> bool:
    """whether a stored version is the one a writer read.

    :param stored: the file's version value (a datetime, or text a codec left undecoded)
    :ptype stored: Any
    :param expected: the version the writer read
    :ptype expected: datetime
    :return: ``True`` when they name the same instant
    :rtype: bool
    """
    if isinstance(stored, str):
        try:
            stored = datetime.fromisoformat(stored)
        except ValueError:
            return False
    return isinstance(stored, datetime) and stored == expected
