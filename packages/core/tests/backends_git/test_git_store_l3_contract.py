"""the git store keeps the L3 contract every durable store keeps (``tests/l3_contract.py``, its first tier).

The same tier Postgres and SQLite run: a row written whole and read back, the conflict mode a write asks for, the
compare-and-swap on ``date_updated``, deletes, composite keys, equality scans, many rows at once, and a transaction
-- here a change set, committed as one commit (``GitDurableStore.transaction``, ``GitRepo.change``). The git store is a store of JSON files, so it
does not run the second tier (typed values from a registered schema, column-wise updates, raw SQL); that is what it
is, not a gap.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest

from packages.core.tests.backends_git.conftest import MakeRepo
from packages.core.tests.l3_contract import CollectionOverStoreContract, DurableStoreContract
from threetears.core.backends.git import (
    ChangeConflictError,
    GitDurableStore,
    JsonCodec,
    TableLayout,
    TemplateLayout,
)


def _contract_layout() -> TemplateLayout:
    """the contract's tables as files: one per row, under a folder per table."""
    codec = JsonCodec()
    return TemplateLayout(
        [
            TableLayout("l3c_docs", "docs/{id}.json", ("id",), codec),
            TableLayout("l3c_ignoring", "ignoring/{id}.json", ("id",), codec, version_column=None),
            TableLayout("l3c_raising", "raising/{id}.json", ("id",), codec, version_column=None),
            TableLayout("l3c_composite", "composite/{tenant}/{id}.json", ("tenant", "id"), codec, version_column=None),
        ]
    )


@pytest.fixture
async def store(make_repo: MakeRepo) -> AsyncIterator[GitDurableStore]:
    """a store over a fresh repository laid out with the contract's tables."""
    yield GitDurableStore(await make_repo("contract", layout=_contract_layout()))


class TestTheGitStoreKeepsTheL3Contract(DurableStoreContract):
    duplicate_error = ChangeConflictError


class TestCollectionsOverTheGitStore(CollectionOverStoreContract):
    """a collection over this store, with no L2 and with L2."""

    def caller_transaction(self, store: Any) -> contextlib.AbstractAsyncContextManager[Any]:  # type: ignore[override]
        """the git store's caller transaction is a change set, which opens its own ``CallerTransaction``."""
        change: contextlib.AbstractAsyncContextManager[Any] = store.repo.change("contract: a caller's transaction")
        return change
