"""the errors a git-backed store raises, each naming what to do about it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from threetears.core.exceptions import ConcurrentModificationError

__all__ = [
    "ChangeConflictError",
    "ConflictedRow",
    "DivergedError",
    "FileConflictError",
    "GitStoreError",
    "HandEditPendingError",
    "NotTheWriterError",
]


class GitStoreError(Exception):
    """the base of every git-store refusal."""


@dataclass(frozen=True, slots=True)
class ConflictedRow:
    """one row a change set could not apply.

    :ivar table: the table
    :ivar pk: the row's key
    :ivar reason: ``"changed"`` -- its version is not the one the writer read; ``"deleted"`` -- it was
        read as existing and is gone; ``"exists"`` -- an insert-only write found it already there
    :ivar expected: the version the writer read, for ``"changed"`` and ``"deleted"``
    """

    table: str
    pk: Mapping[str, Any]
    reason: Literal["changed", "deleted", "exists"]
    expected: datetime | None = None


class ChangeConflictError(ConcurrentModificationError):
    """a change set was refused whole: rows it would write changed since they were read.

    Nothing was written and nothing committed. A :class:`ConcurrentModificationError`, so code that
    already handles a collection's optimistic-lock miss handles this; its ``table_name`` and
    ``entity_id`` are the first conflicted row's.

    :ivar rows: every conflicted row
    """

    def __init__(self, rows: Sequence[ConflictedRow]) -> None:
        """name the conflicted rows.

        :param rows: the rows, at least one
        :ptype rows: Sequence[ConflictedRow]
        :return: None
        :rtype: None
        """
        self.rows = tuple(rows)
        first = self.rows[0]
        super().__init__(first.table, dict(first.pk), first.expected)
        described = "; ".join(f"{row.table} {dict(row.pk)} {row.reason}" for row in self.rows)
        self.args = (f"change set refused, nothing was written: {described}",)

    def __str__(self) -> str:
        """the message, naming every row.

        :return: the message
        :rtype: str
        """
        return str(self.args[0])


class HandEditPendingError(GitStoreError):
    """a change would write over a file someone edited by hand and has not committed.

    Commit the edit first (``GitRepo.commit_paths``), or put the file back, then retry.

    :ivar paths: the edited files
    """

    def __init__(self, paths: Sequence[str]) -> None:
        """name the edited files.

        :param paths: the files, relative to the working tree's root
        :ptype paths: Sequence[str]
        :return: None
        :rtype: None
        """
        self.paths = tuple(paths)
        super().__init__(
            f"uncommitted hand edits in the way, nothing was written: {', '.join(self.paths)}. "
            f"Commit the edit first, or put the file back"
        )


class FileConflictError(GitStoreError):
    """files sent to be committed changed in the repository since the commit they were based on.

    Nothing was written. Read them again (pull, or fetch them from the writer) and send the change again.

    :ivar paths: the files that changed
    """

    def __init__(self, paths: Sequence[str]) -> None:
        """name the files that changed.

        :param paths: the files, relative to the working tree's root
        :ptype paths: Sequence[str]
        :return: None
        :rtype: None
        """
        self.paths = tuple(paths)
        super().__init__(
            f"files changed since the commit they were based on, nothing was written: {', '.join(self.paths)}"
        )


class DivergedError(GitStoreError):
    """the branch and origin have both moved on; this store never merges or forces, so a person reconciles them.

    Nothing changed. Raised by an explicit :meth:`~threetears.git_store.repo.GitRepo.pull`; on its own the
    writer only stops pushing (``RepoStatus.push_blocked``).

    :ivar head: this repository's commit
    :ivar origin: origin's commit
    """

    def __init__(self, branch: str, head: str, origin: str) -> None:
        """name the two commits.

        :param branch: the branch
        :ptype branch: str
        :param head: this repository's commit
        :ptype head: str
        :param origin: origin's commit
        :ptype origin: str
        :return: None
        :rtype: None
        """
        self.head = head
        self.origin = origin
        super().__init__(
            f"{branch} ({head[:12]}) and origin/{branch} ({origin[:12]}) have both moved on; nothing was pulled. "
            f"A person must reconcile them (this store never merges or forces)"
        )


class NotTheWriterError(GitStoreError):
    """this process may not commit to the repository and no writer could take the change.

    Another process holds the writer lease and did not answer, or this process lost the lease while
    committing (nothing was committed). Retry; the change was not applied.
    """
