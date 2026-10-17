"""many repositories behind one store, the forwarding wire, collections bound to commits, and no force push."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import NO_L2, DurableStoreCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.git_store import (
    Author,
    Change,
    ChangeConflictError,
    CommitResult,
    ConflictedRow,
    GitDurableStore,
    GitRepoConfig,
    GitRepos,
    HandEditPendingError,
    NotTheWriterError,
    RoutedGitStore,
    bind_collections,
)
from threetears.git_store.wire import (
    ForwardedRequest,
    decode_reply,
    decode_request,
    encode_failure,
    encode_request,
    encode_result,
)

T0 = datetime(2026, 10, 17, 9, 0, tzinfo=UTC)


def recipe(character: str, look: str = "x") -> dict[str, Any]:
    return {"character": character, "look": look, "date_updated": T0}


class TestManyRepositories:
    async def test_one_repository_per_character_opened_on_first_use(self, tmp_path: Path, layout: Any) -> None:
        repos = GitRepos(layout, nats=None, config_for=lambda key: GitRepoConfig(key=key, work_tree=tmp_path / key))
        store = RoutedGitStore(repos, lambda table, values: values.get("character"))
        try:
            await store.upsert("recipes", recipe("aria"))
            await store.upsert("recipes", recipe("june"))
            assert (tmp_path / "aria/aria/recipe.json").is_file()
            assert (tmp_path / "june/june/recipe.json").is_file()
            assert {repo.config.key for repo in repos} == {"aria", "june"}
            assert await store.fetch_one("recipes", {"character": "june"}) == recipe("june")
            assert len(await store.scan("recipes")) == 2  # no filter on the routing column: every repository
            assert len(await store.scan("recipes", {"character": "aria"})) == 1
            assert [entry.subject for entry in await store.history("recipes", {"character": "aria"})] == [
                "recipes aria: save"
            ]
            aria = await repos.repo("aria")
            async with aria.change("wrong repository") as change_set:
                with pytest.raises(ValueError, match="lives in repository 'june'"):
                    await store.upsert("recipes", recipe("june", "y"), conn=change_set)
        finally:
            await repos.aclose()

    async def test_one_shared_repository_for_everyone(self, tmp_path: Path, layout: Any) -> None:
        repos = GitRepos(layout, nats=None)
        await repos.add(GitRepoConfig(key="characters", work_tree=tmp_path / "shared"))
        store = RoutedGitStore(repos, lambda table, values: "characters")
        try:
            written = await store.upsert_many("recipes", [recipe("aria"), recipe("june")], max_rows=9, max_bytes=9)
            assert written == 2
            shared = repos.get("characters")
            assert shared is not None and len(shared.git.log()) == 1
            with pytest.raises(KeyError):
                await repos.repo("unknown")
        finally:
            await repos.aclose()


class TestWire:
    def test_a_change_set_crosses_with_its_types(self) -> None:
        row = {
            "when": T0,
            "day": date(2026, 10, 17),
            "id": UUID(int=7),
            "price": Decimal("2.50"),
            "nested": {"$looks_like_a_tag": 1},
            "items": [T0, {"x": UUID(int=1)}],
        }
        request = ForwardedRequest(
            op="commit",
            changes=(
                Change("keepers", {"tile": "1"}, row, cas=T0, on_conflict="raise"),
                Change("keepers", {"tile": "2"}, None),
            ),
            message="m\n",
            author=Author("Pace", "p@example.org"),
            trailers={"Op-Seq": "3"},
            date=T0,
        )
        assert decode_request(encode_request(request)) == request

    def test_results_and_refusals_come_back_as_they_were(self) -> None:
        assert decode_reply(encode_result(CommitResult("abc", ("a.json",), True))) == CommitResult(
            "abc", ("a.json",), True
        )
        conflict = ChangeConflictError([ConflictedRow("keepers", {"tile": "1"}, "changed", T0)])
        with pytest.raises(ChangeConflictError) as raised:
            decode_reply(encode_failure(conflict))
        assert raised.value.rows == conflict.rows
        with pytest.raises(HandEditPendingError) as edited:
            decode_reply(encode_failure(HandEditPendingError(["a.json"])))
        assert edited.value.paths == ("a.json",)
        with pytest.raises(NotTheWriterError):
            decode_reply(encode_failure(NotTheWriterError("gone")))

    def test_an_unknown_operation_is_refused(self) -> None:
        with pytest.raises(ValueError):
            decode_request(b'{"op": "reset_hard", "changes": []}')


class _Recipe(BaseEntity):
    primary_key_field = "character"


class _Recipes(DurableStoreCollection[_Recipe]):
    primary_key_column = "character"

    @property
    def table_name(self) -> str:
        return "recipes"

    @property
    def entity_class(self) -> type[_Recipe]:
        return _Recipe


@pytest.fixture()
def registry() -> Iterator[CollectionRegistry]:
    metadata = MetaData()
    Table(
        "recipes",
        metadata,
        Column("character", String(64), primary_key=True),
        Column("look", String(255)),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    backend = SQLiteBackend(db_name=f"git_store_bind_{uuid.uuid4().hex[:8]}")
    backend.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=backend, kv_key_scope="test-principal")
    yield registry
    from threetears.core.testing.bridge import drain_and_shutdown_bridge

    drain_and_shutdown_bridge()


async def test_a_hand_commit_drops_the_row_from_the_collections_cache(
    make_repo: Any, registry: CollectionRegistry, git_cli: Any
) -> None:
    repo = await make_repo()
    recipes = _Recipes(registry, DefaultCoreConfig(), GitDurableStore(repo), nats_client=NO_L2)
    bind_collections(repo, {"recipes": recipes})
    await recipes.save_entity(_Recipe({"character": "aria", "look": "one", "date_updated": None}, is_new=True))
    cached = await recipes.get("aria")
    assert cached is not None and cached.look == "one"
    assert recipes.get_row_sync("aria") is not None

    path = repo.config.work_tree / "aria/recipe.json"
    path.write_text(path.read_text().replace('"one"', '"two"'))
    git_cli(repo.config.work_tree, "commit", "-qam", "edited by hand")
    assert recipes.get_row_sync("aria") is not None  # stale until the store notices
    await repo.sync()
    assert recipes.get_row_sync("aria") is None
    fresh = await recipes.get("aria")
    assert fresh is not None and fresh.look == "two"


def test_nothing_in_the_package_can_force_a_push() -> None:
    """a force push rewrites origin's history; the store must only ever push fast-forwards."""
    source = Path(__file__).resolve().parents[2] / "src"
    offenders = [
        f"{path.name}:{number}"
        for path in sorted(source.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if re.search(r"""["'](--force(-with-lease)?|-f|\+refs/)""", line) or re.search(r"""["']\+[\w/]+:""", line)
    ]
    assert offenders == []
