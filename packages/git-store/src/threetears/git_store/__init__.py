"""a git working tree as a durable store: rows as files, every change set one commit, one writer per repository.

A product brings a :class:`RowCodec` (how a row becomes bytes; :class:`JsonCodec` ships) and a
:class:`Layout` (where a row's file lives; :class:`TemplateLayout` builds one from path templates).
:class:`GitRepo` owns the working tree and its writer; :class:`GitDurableStore` is the
:class:`~threetears.core.backends.DurableStore` a
:class:`~threetears.core.collections.DurableStoreCollection` runs on; :class:`GitRepos` and
:class:`RoutedGitStore` keep many repositories at once; :func:`bind_collections` drops rows other
commits changed from every cache.
"""

from threetears.git_store.advanced import AdvancedListener, RepoAdvanced
from threetears.git_store.bind import bind_collections
from threetears.git_store.change import Change, ChangeSet, CommitResult
from threetears.git_store.codec import JsonCodec, RowCodec
from threetears.git_store.envelope import ForwardRefusedError
from threetears.git_store.errors import (
    ChangeConflictError,
    ConflictedRow,
    GitStoreError,
    HandEditPendingError,
    NotTheWriterError,
)
from threetears.git_store.git import Author, CommitEntry, GitError
from threetears.git_store.layout import Layout, TableLayout, TemplateLayout
from threetears.git_store.repo import DirtyFile, GitRepo, GitRepoConfig, RepoStatus, RowChange
from threetears.git_store.repos import GitRepos, RoutedGitStore
from threetears.git_store.store import GitDurableStore

__all__ = [
    "AdvancedListener",
    "Author",
    "Change",
    "ChangeConflictError",
    "ChangeSet",
    "CommitEntry",
    "CommitResult",
    "ConflictedRow",
    "DirtyFile",
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
    "RepoAdvanced",
    "RepoStatus",
    "RoutedGitStore",
    "RowChange",
    "RowCodec",
    "TableLayout",
    "TemplateLayout",
    "bind_collections",
]
