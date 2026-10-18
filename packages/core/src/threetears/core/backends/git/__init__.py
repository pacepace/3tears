"""a git working tree as a durable store: rows as files, every change set one commit, one writer per repository.

A product brings a :class:`RowCodec` (how a row becomes bytes; :class:`JsonCodec` ships) and a
:class:`Layout` (where a row's file lives; :class:`TemplateLayout` builds one from path templates).
:class:`GitRepo` owns the working tree and its writer; :class:`GitDurableStore` is the
:class:`~threetears.core.backends.DurableStore` a
:class:`~threetears.core.collections.DurableStoreCollection` runs on; :class:`GitRepos` and
:class:`RoutedGitStore` keep many repositories at once; :func:`bind_collections` drops rows other
commits changed from every cache. A :class:`PendingTail` (:class:`OpLogTail`) keeps edits saved before they are
committed, for products that commit only when the user means it.

An L3 beside :mod:`threetears.core.backends.sql` and :mod:`threetears.core.backends.nats_proxy`, imported
by its own path (``threetears.core.backends`` does not load it). It needs core's ``nats`` extra at runtime
(``pip install 3tears[nats]``): the writer lease, forwarding and the advanced broadcast use the NATS
client. It also needs the ``git`` program on the path; no pygit2.
"""

from threetears.core.backends.git.advanced import AdvancedListener, RepoAdvanced
from threetears.core.backends.git.bind import bind_collections
from threetears.core.backends.git.change import Change, ChangeSet, CommitResult
from threetears.core.backends.git.codec import JsonCodec, RowCodec
from threetears.core.backends.git.envelope import ForwardRefusedError, new_signing_seed
from threetears.core.backends.git.errors import (
    ChangeConflictError,
    ConflictedRow,
    DivergedError,
    FileConflictError,
    GitStoreError,
    HandEditPendingError,
    NotTheWriterError,
)
from threetears.core.backends.git.git import Author, CommitEntry, GitError
from threetears.core.backends.git.layout import Layout, TableLayout, TemplateLayout
from threetears.core.backends.git.repo import (
    Committed,
    DirtyFile,
    GitRepo,
    GitRepoConfig,
    PendingWrite,
    RepoStatus,
    RowChange,
)
from threetears.core.backends.git.repos import GitRepos, RoutedGitStore
from threetears.core.backends.git.store import GitDurableStore
from threetears.core.backends.git.tail import OpLogLike, OpLogTail, PendingTail, committed_seq_from

__all__ = [
    "AdvancedListener",
    "Author",
    "Change",
    "ChangeConflictError",
    "ChangeSet",
    "CommitEntry",
    "CommitResult",
    "Committed",
    "ConflictedRow",
    "DirtyFile",
    "DivergedError",
    "FileConflictError",
    "ForwardRefusedError",
    "GitDurableStore",
    "GitError",
    "GitRepo",
    "GitRepoConfig",
    "GitRepos",
    "GitStoreError",
    "HandEditPendingError",
    "JsonCodec",
    "Layout",
    "NotTheWriterError",
    "OpLogLike",
    "OpLogTail",
    "PendingTail",
    "PendingWrite",
    "RepoAdvanced",
    "RepoStatus",
    "RoutedGitStore",
    "RowChange",
    "RowCodec",
    "TableLayout",
    "TemplateLayout",
    "bind_collections",
    "committed_seq_from",
    "new_signing_seed",
]
