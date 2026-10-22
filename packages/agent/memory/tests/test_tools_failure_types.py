"""A memory tool answers a storage failure with a result, and lets a bug in our own code propagate.

Each handler in :mod:`threetears.agent.memory.tools` catches the failures its try body can
meet (storage unreachable, a lost compare-and-swap, a cycle in the provenance graph, the
supersede race) and nothing else. Each site is driven twice here: once with a failure the
store can raise, which must still read as the result it always did, and once with an
``AttributeError`` or ``TypeError`` from the callee, which must reach the caller rather than
read as an outage.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from threetears.agent.memory import tools
from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import ConsolidationCycleError
from threetears.agent.memory.revisions import RETRACTED_TAG, WriteIncompleteError, Written
from threetears.core.exceptions import (
    ConcurrentModificationError,
    DataLayerUnavailableError,
    GenerationNotCommittedError,
    GenerationUnavailableError,
)
from threetears.nats.errors import KvError

_USER = uuid4()
_AGENT = uuid4()
_CUSTOMER = uuid4()
_TARGET = uuid4()

#: what a bug in our own code looks like from the handler's side
_BUGS = (AttributeError("'NoneType' object has no attribute 'fetch'"), TypeError("fetch() takes 2 arguments"))


def _bug_ids() -> list[str]:
    return [type(bug).__name__ for bug in _BUGS]


def _row(**overrides: Any) -> dict[str, Any]:
    return {
        "memory_id": _TARGET,
        "type_memory": "fact",
        "type": "fact",
        "content": "lives in Leeds",
        "date_created": None,
        "similarity": 0.99,
        **overrides,
    }


def _unique_violation(constraint: str) -> Exception:
    """the error a direct pool raises, and the NATS proxy rebuilds, for a duplicate key."""
    return asyncpg.PostgresError.new(
        {"C": "23505", "M": f'duplicate key value violates unique constraint "{constraint}"', "n": constraint}
    )


# parity-exempt: one stand-in answering the read and write calls every memory tool makes across the memories, media, chunk and edge collections; the SQL behind them is covered by the collection tests
class _FakeStore:
    """a collection whose every call answers a fixed value, or raises what ``fail`` names for it."""

    def __init__(self, fail: dict[str, BaseException] | None = None, **answers: Any) -> None:
        self.fail = fail or {}
        self.answers = answers
        self.saved: list[Any] = []

    async def _answer(self, name: str, default: Any) -> Any:
        if name in self.fail:
            raise self.fail[name]
        return self.answers.get(name, default)

    async def find_by_alias(self, **kwargs: Any) -> dict[str, Any] | None:
        _ = kwargs
        return await self._answer("find_by_alias", _row())  # type: ignore[no-any-return]

    async def search_by_ids(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        _ = args, kwargs
        return await self._answer("search_by_ids", [])  # type: ignore[no-any-return]

    async def search_by_semantic(self, **kwargs: Any) -> list[dict[str, Any]]:
        _ = kwargs
        return await self._answer("search_by_semantic", [])  # type: ignore[no-any-return]

    async def search_by_fts(self, **kwargs: Any) -> list[dict[str, Any]]:
        _ = kwargs
        return await self._answer("search_by_fts", [])  # type: ignore[no-any-return]

    async def find_for_recall(self, **kwargs: Any) -> dict[str, Any] | None:
        _ = kwargs
        return await self._answer("find_for_recall", _row())  # type: ignore[no-any-return]

    async def bump_salience(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        await self._answer("bump_salience", None)

    async def find_by_memory_id(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        _ = args, kwargs
        return await self._answer("find_by_memory_id", [])  # type: ignore[no-any-return]

    async def count_by_memory_id(self, *args: Any, **kwargs: Any) -> int:
        _ = args, kwargs
        return await self._answer("count_by_memory_id", 0)  # type: ignore[no-any-return]

    async def find_sources(self, *args: Any) -> list[UUID]:
        _ = args
        return await self._answer("find_sources", [])  # type: ignore[no-any-return]

    async def list_for_user(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        _ = kwargs
        return await self._answer("list_for_user", ([], 0))  # type: ignore[no-any-return]

    async def count_by_user(self, *args: Any, **kwargs: Any) -> bool:
        _ = args, kwargs
        return await self._answer("count_by_user", True)  # type: ignore[no-any-return]

    async def find_similar_for_dedup(self, **kwargs: Any) -> list[dict[str, Any]]:
        _ = kwargs
        return await self._answer("find_similar_for_dedup", [])  # type: ignore[no-any-return]

    async def get(self, key: Any) -> Any:
        _ = key
        return await self._answer("get", SimpleNamespace(user_id=_USER, embedding=[0.1] * 8, alias=None, salience=0.5))

    async def save_entity(self, entity: Any) -> None:
        self.saved.append(entity)
        await self._answer("save_entity", None)

    async def hybrid_search(self, **kwargs: Any) -> list[dict[str, Any]]:
        _ = kwargs
        return await self._answer("hybrid_search", [])  # type: ignore[no-any-return]


# parity-exempt: answers only get_by_owner_and_customer, the namespace read memory_add makes before it writes
class _FakeNamespaces:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def get_by_owner_and_customer(self, **kwargs: Any) -> Any:
        _ = kwargs
        raise self.error


class _Context:
    conversation_id = uuid4()
    correlation_id = uuid4()


def _raising(error: BaseException) -> Callable[..., Awaitable[Any]]:
    async def _call(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        raise error

    return _call


async def _wrote(*args: Any, **kwargs: Any) -> Written:
    _ = args, kwargs
    return Written(memory_id=uuid4())


# -- reads: a storage failure is the tool's error line ---------------------------------------


async def _search(
    authorizer: MemoryAuthorizerDependencies,
    store: _FakeStore,
    *,
    media: _FakeStore | None = None,
    chunks: _FakeStore | None = None,
    **call: Any,
) -> str:
    (tool,) = await tools.load_memory_search_tool(
        _USER,
        DeterministicFakeEmbedding(size=8),
        _AGENT,
        _CUSTOMER,
        authorizer,
        store,  # type: ignore[arg-type]
        media or store,  # type: ignore[arg-type]
        chunks or store,  # type: ignore[arg-type]
    )
    return str(await tool.ainvoke(call))


def _side_search(branch: str) -> Callable[[MemoryAuthorizerDependencies, _FakeStore], Awaitable[str]]:
    """a search whose memories are found and whose ``branch`` side search runs on ``store``."""

    async def _call(authorizer: MemoryAuthorizerDependencies, store: _FakeStore) -> str:
        found = _FakeStore(search_by_semantic=[_row()])
        sides = {"media": _FakeStore(), "chunks": _FakeStore(), branch: store}
        return await _search(authorizer, found, media=sides["media"], chunks=sides["chunks"], query="where")

    return _call


async def _recall(authorizer: MemoryAuthorizerDependencies, store: _FakeStore, **call: Any) -> str:
    (tool,) = await tools.load_memory_recall_tool(
        _USER,
        _AGENT,
        _CUSTOMER,
        authorizer,
        store,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        consolidations_collection=store,  # type: ignore[arg-type]
        salience_bump=0.1,
    )
    return str(await tool.ainvoke({"memory_id": str(_TARGET), **call}))


async def _list(authorizer: MemoryAuthorizerDependencies, store: _FakeStore) -> str:
    (tool,) = tools.load_memory_list_tool(_USER, _AGENT, _CUSTOMER, authorizer, store)  # type: ignore[arg-type]
    return str(await tool.ainvoke({}))


async def _chunk_search(authorizer: MemoryAuthorizerDependencies, store: _FakeStore) -> str:
    (tool,) = await tools.load_chunk_search_tool(
        _USER,
        _AGENT,
        _CUSTOMER,
        DeterministicFakeEmbedding(size=8),
        authorizer,
        store,  # type: ignore[arg-type]
    )
    return str(await tool.ainvoke({"query": "the lease terms"}))


_CHUNK_HIT = {"chunk_id": uuid4(), "memory_id": _TARGET, "content": "The tenant pays monthly.", "score": 0.9}

#: (name, how the tool is called, the collection call that fails, what the failure reads as)
_READ_SITES: list[tuple[str, Callable[[MemoryAuthorizerDependencies, _FakeStore], Awaitable[str]], str, str]] = [
    ("search by ids", lambda a, s: _search(a, s, ids=[str(_TARGET)]), "search_by_ids", "id_lookup failed"),
    ("search by alias", lambda a, s: _search(a, s, alias="home"), "find_by_alias", "alias failed"),
    ("search semantic", lambda a, s: _search(a, s, query="where", mode="fuzzy"), "search_by_semantic", "query failed"),
    ("recall by alias", lambda a, s: _recall(a, s, memory_id="", alias="home"), "find_by_alias", "alias failed"),
    ("recall chunks", lambda a, s: _recall(a, s), "find_by_memory_id", "fetch chunks (default) failed"),
    ("list", _list, "list_for_user", "query failed"),
    ("chunk search", _chunk_search, "hybrid_search", "search failed"),
]


class TestAReadThatCannotReachStorageSaysSo:
    @pytest.mark.parametrize(("name", "call", "method", "said"), _READ_SITES, ids=[s[0] for s in _READ_SITES])
    @pytest.mark.parametrize(
        "outage",
        [DataLayerUnavailableError("broker down"), KvError("kv down"), _unique_violation("x"), TimeoutError("slow")],
        ids=["DataLayerUnavailableError", "KvError", "PostgresError", "TimeoutError"],
    )
    async def test_an_outage_is_the_tools_error_line(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        name: str,
        call: Callable[[MemoryAuthorizerDependencies, _FakeStore], Awaitable[str]],
        method: str,
        said: str,
        outage: Exception,
    ) -> None:
        _ = name
        result = await call(permissive_memory_authorizer, _FakeStore(fail={method: outage}))
        assert result.startswith("[TOOL ERROR]") and said in result, result

    @pytest.mark.parametrize(("name", "call", "method", "said"), _READ_SITES, ids=[s[0] for s in _READ_SITES])
    @pytest.mark.parametrize("bug", _BUGS, ids=_bug_ids())
    async def test_a_bug_propagates(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        name: str,
        call: Callable[[MemoryAuthorizerDependencies, _FakeStore], Awaitable[str]],
        method: str,
        said: str,
        bug: Exception,
    ) -> None:
        _ = name, said
        with pytest.raises(type(bug)):
            await call(permissive_memory_authorizer, _FakeStore(fail={method: bug}))


#: (name, how the tool is called, the collection call that fails, what still shows)
_DEGRADED_READS: list[tuple[str, Callable[[MemoryAuthorizerDependencies, _FakeStore], Awaitable[str]], str, str]] = [
    (
        "search full-text",
        lambda a, s: _search(a, s, query="where", mode="precise"),
        "search_by_fts",
        "search incomplete",
    ),
    ("search media", _side_search("media"), "search_by_semantic", "media search failed"),
    ("search chunks", _side_search("chunks"), "search_by_semantic", "document chunk search failed"),
    ("recall salience bump", _recall, "bump_salience", "lives in Leeds"),
    ("recall sources", _recall, "find_sources", "lives in Leeds"),
    ("chunk search dates", _chunk_search, "search_by_ids", "The tenant pays monthly."),
]


class TestAReadThatLosesAnExtraStillAnswers:
    @pytest.mark.parametrize(("name", "call", "method", "shown"), _DEGRADED_READS, ids=[s[0] for s in _DEGRADED_READS])
    async def test_an_outage_leaves_the_rest_of_the_answer(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        name: str,
        call: Callable[[MemoryAuthorizerDependencies, _FakeStore], Awaitable[str]],
        method: str,
        shown: str,
    ) -> None:
        _ = name
        store = _FakeStore(fail={method: DataLayerUnavailableError("broker down")}, hybrid_search=[_CHUNK_HIT])
        assert shown in await call(permissive_memory_authorizer, store)

    @pytest.mark.parametrize(("name", "call", "method", "shown"), _DEGRADED_READS, ids=[s[0] for s in _DEGRADED_READS])
    @pytest.mark.parametrize("bug", _BUGS, ids=_bug_ids())
    async def test_a_bug_propagates(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        name: str,
        call: Callable[[MemoryAuthorizerDependencies, _FakeStore], Awaitable[str]],
        method: str,
        shown: str,
        bug: Exception,
    ) -> None:
        _ = name, shown
        with pytest.raises(type(bug)):
            await call(permissive_memory_authorizer, _FakeStore(fail={method: bug}, hybrid_search=[_CHUNK_HIT]))


class TestASideSearchWithCorruptMetadataIsDegraded:
    """a stored file or passage whose metadata is not JSON loses that branch, not the search."""

    async def test_undecodable_metadata_is_a_degraded_branch(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        corrupt = {**_row(), "content_id": uuid4(), "media_id": uuid4(), "media_category": "image"}
        corrupt |= {"metadata_json": "{not json", "chunk_id": uuid4(), "heading_context": None, "page_number": None}
        result = await _search(
            permissive_memory_authorizer,
            _FakeStore(search_by_semantic=[_row()]),
            media=_FakeStore(search_by_semantic=[corrupt]),
            chunks=_FakeStore(search_by_semantic=[corrupt]),
            query="where",
        )
        assert result.startswith("(Partial results: media search, document chunk search failed"), result
        assert "lives in Leeds" in result


# -- writes: memory_add ----------------------------------------------------------------------


async def _add(authorizer: MemoryAuthorizerDependencies, store: _FakeStore, **call: Any) -> str:
    (tool,) = await tools.load_memory_add_tool(
        _USER,
        DeterministicFakeEmbedding(size=8),
        _AGENT,
        _CUSTOMER,
        authorizer,
        store,  # type: ignore[arg-type]
        context_resolver=_Context,
        consolidations_collection=store,  # type: ignore[arg-type]
    )
    return str(await tool.ainvoke({"content": "lives in York", **call}))


class TestMemoryAddBeforeItWrites:
    async def test_a_namespace_outage_is_the_authorize_error(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        authorizer = MemoryAuthorizerDependencies(
            acl_cache=permissive_memory_authorizer.acl_cache,
            namespace_collection=_FakeNamespaces(KvError("kv down")),  # type: ignore[arg-type]
            group_collection=permissive_memory_authorizer.group_collection,
            group_member_collection=permissive_memory_authorizer.group_member_collection,
            role_collection=permissive_memory_authorizer.role_collection,
            role_assignment_collection=permissive_memory_authorizer.role_assignment_collection,
        )
        assert "authorize failed" in await _add(authorizer, _FakeStore())

    @pytest.mark.parametrize("bug", _BUGS, ids=_bug_ids())
    async def test_a_bug_in_the_namespace_read_propagates(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies, bug: Exception
    ) -> None:
        authorizer = MemoryAuthorizerDependencies(
            acl_cache=permissive_memory_authorizer.acl_cache,
            namespace_collection=_FakeNamespaces(bug),  # type: ignore[arg-type]
            group_collection=permissive_memory_authorizer.group_collection,
            group_member_collection=permissive_memory_authorizer.group_member_collection,
            role_collection=permissive_memory_authorizer.role_collection,
            role_assignment_collection=permissive_memory_authorizer.role_assignment_collection,
        )
        with pytest.raises(type(bug)):
            await _add(authorizer, _FakeStore())

    @pytest.mark.parametrize(
        "failure",
        [DataLayerUnavailableError("broker down"), ConcurrentModificationError("groups", 1, None)],
        ids=["DataLayerUnavailableError", "ConcurrentModificationError"],
    )
    async def test_a_failed_first_write_ensure_still_stores(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        failure: Exception,
    ) -> None:
        monkeypatch.setattr(tools, "write_once", _wrote)
        result = await _add(permissive_memory_authorizer, _FakeStore(fail={"count_by_user": failure}))
        assert result.startswith("Stored as [memory:"), result

    @pytest.mark.parametrize("bug", _BUGS, ids=_bug_ids())
    async def test_a_bug_in_the_first_write_ensure_propagates(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies, bug: Exception
    ) -> None:
        with pytest.raises(type(bug)):
            await _add(permissive_memory_authorizer, _FakeStore(fail={"count_by_user": bug}))


#: what a write may fail on and still be answered as not stored
_WRITE_OUTAGES = [
    DataLayerUnavailableError("broker down"),
    ConcurrentModificationError("memories", _TARGET, None),
    GenerationUnavailableError("generation store down"),
    ConsolidationCycleError("the gist is its own source"),
]
_WRITE_OUTAGE_IDS = [type(e).__name__ for e in _WRITE_OUTAGES]


class TestMemoryAddWhenTheWriteFails:
    @pytest.mark.parametrize("failure", _WRITE_OUTAGES, ids=_WRITE_OUTAGE_IDS)
    async def test_a_failed_correction_is_the_store_error(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        failure: Exception,
    ) -> None:
        monkeypatch.setattr(tools, "supersede", _raising(failure))
        result = await _add(permissive_memory_authorizer, _FakeStore(), corrects=[str(_TARGET)])
        assert result.startswith("[TOOL ERROR] memory_add: store failed"), result

    @pytest.mark.parametrize("bug", _BUGS, ids=_bug_ids())
    async def test_a_bug_in_the_correction_propagates(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        bug: Exception,
    ) -> None:
        monkeypatch.setattr(tools, "supersede", _raising(bug))
        with pytest.raises(type(bug)):
            await _add(permissive_memory_authorizer, _FakeStore(), corrects=[str(_TARGET)])

    @pytest.mark.parametrize(
        "failure",
        [*_WRITE_OUTAGES, ValueError("memory is permanent and is never superseded")],
        ids=[*_WRITE_OUTAGE_IDS, "ValueError"],
    )
    async def test_a_failed_near_duplicate_replace_is_the_store_error_and_writes_no_second_copy(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        failure: Exception,
    ) -> None:
        # the replace may have committed the new memory before it failed, so a plain
        # write after it would store the same fields twice
        plain_writes: list[object] = []

        async def _write_once(*args: Any, **kwargs: Any) -> Written:
            plain_writes.append((args, kwargs))
            return await _wrote(*args, **kwargs)

        monkeypatch.setattr(tools, "supersede", _raising(failure))
        monkeypatch.setattr(tools, "write_once", _write_once)
        store = _FakeStore(find_similar_for_dedup=[_row()])
        result = await _add(permissive_memory_authorizer, store)
        assert result.startswith("[TOOL ERROR] memory_add: store failed"), result
        assert plain_writes == []

    @pytest.mark.parametrize(
        "outage",
        [DataLayerUnavailableError("broker down"), KvError("kv down")],
        ids=["DataLayerUnavailableError", "KvError"],
    )
    @pytest.mark.parametrize("read", ["find_similar_for_dedup", "find_for_recall"])
    async def test_a_near_duplicate_search_that_cannot_reach_storage_still_stores(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        outage: Exception,
        read: str,
    ) -> None:
        monkeypatch.setattr(tools, "supersede", _raising(AssertionError("a failed search replaces nothing")))
        monkeypatch.setattr(tools, "write_once", _wrote)
        store = _FakeStore(fail={read: outage}, find_similar_for_dedup=[_row()])
        result = await _add(permissive_memory_authorizer, store)
        assert result.startswith("Stored as [memory:") and "replaces" not in result, result

    @pytest.mark.parametrize("bug", _BUGS, ids=_bug_ids())
    async def test_a_bug_in_the_near_duplicate_search_propagates(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        bug: Exception,
    ) -> None:
        with pytest.raises(type(bug)):
            await _add(permissive_memory_authorizer, _FakeStore(fail={"find_similar_for_dedup": bug}))

    @pytest.mark.parametrize("bug", _BUGS, ids=_bug_ids())
    async def test_a_bug_in_the_near_duplicate_replace_propagates(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        bug: Exception,
    ) -> None:
        monkeypatch.setattr(tools, "supersede", _raising(bug))
        with pytest.raises(type(bug)):
            await _add(permissive_memory_authorizer, _FakeStore(find_similar_for_dedup=[_row()]))

    @pytest.mark.parametrize(
        "failure",
        [*_WRITE_OUTAGES, ValueError("memory is permanent and is never superseded")],
        ids=[*_WRITE_OUTAGE_IDS, "ValueError"],
    )
    async def test_a_failed_insert_is_the_store_error(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        failure: Exception,
    ) -> None:
        monkeypatch.setattr(tools, "write_once", _raising(failure))
        result = await _add(permissive_memory_authorizer, _FakeStore())
        assert result.startswith("[TOOL ERROR] memory_add: store failed"), result

    @pytest.mark.parametrize("bug", _BUGS, ids=_bug_ids())
    async def test_a_bug_in_the_insert_propagates(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        bug: Exception,
    ) -> None:
        monkeypatch.setattr(tools, "write_once", _raising(bug))
        with pytest.raises(type(bug)):
            await _add(permissive_memory_authorizer, _FakeStore())


class TestATakenAliasIsToldFromAnyOtherRefusal:
    @pytest.mark.parametrize(
        "refusal",
        [
            _unique_violation("ix_memories_user_alias"),
            DataLayerUnavailableError(
                'L3 execute failed: QUERY_FAILED: duplicate key value violates unique constraint "ix_memories_user_alias"'
            ),
        ],
        ids=["UniqueViolationError", "a broker that forwards no fields"],
    )
    async def test_the_alias_index_refusing_is_alias_taken(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        refusal: Exception,
    ) -> None:
        monkeypatch.setattr(tools, "write_once", _raising(refusal))
        result = await _add(permissive_memory_authorizer, _FakeStore(), alias="home")
        assert result.startswith("[TOOL ERROR] memory_add: alias failed — Alias 'home' already used."), result

    async def test_a_duplicate_primary_key_is_not_alias_taken(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the old check read any 23505 as a taken alias; a duplicate id is a store failure."""
        monkeypatch.setattr(tools, "write_once", _raising(_unique_violation("memories_pkey")))
        result = await _add(permissive_memory_authorizer, _FakeStore(), alias="home")
        assert result.startswith("[TOOL ERROR] memory_add: store failed"), result
        assert "already used" not in result

    async def test_memory_update_tells_them_apart_too(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tools, "supersede", _raising(_unique_violation("ix_memories_user_alias")))
        assert "already used by another memory" in await _update(permissive_memory_authorizer, alias="home")
        monkeypatch.setattr(tools, "supersede", _raising(_unique_violation("memories_pkey")))
        result = await _update(permissive_memory_authorizer, alias="home")
        assert result.startswith("[TOOL ERROR] memory_update: store failed"), result


# -- writes: memory_update, memory_forget, memory_restore -------------------------------------


async def _update(authorizer: MemoryAuthorizerDependencies, store: _FakeStore | None = None, **call: Any) -> str:
    store = store or _FakeStore()
    (tool,) = await tools.load_memory_update_tool(
        _USER,
        DeterministicFakeEmbedding(size=8),
        _AGENT,
        _CUSTOMER,
        authorizer,
        store,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        context_resolver=_Context,
    )
    return str(await tool.ainvoke({"memory_id": str(_TARGET), **call}))


async def _forget(authorizer: MemoryAuthorizerDependencies) -> str:
    (tool,) = tools.load_memory_forget_tool(_USER, _AGENT, _CUSTOMER, authorizer, _FakeStore())  # type: ignore[arg-type]
    return str(await tool.ainvoke({"memory_id": str(_TARGET), "reason": "moved away"}))


async def _restore(authorizer: MemoryAuthorizerDependencies) -> str:
    store = _FakeStore(find_for_recall=_row(tags=[RETRACTED_TAG]))
    (tool,) = tools.load_memory_restore_tool(_USER, _AGENT, _CUSTOMER, authorizer, store)  # type: ignore[arg-type]
    return str(await tool.ainvoke({"memory_id": str(_TARGET)}))


#: (name, how the tool is called, the revision it makes, what a failed one reads as)
_REVISION_SITES: list[tuple[str, Callable[[MemoryAuthorizerDependencies], Awaitable[str]], str, str]] = [
    ("update", lambda a: _update(a, content="lives in York"), "supersede", "memory_update: store failed"),
    ("forget", _forget, "retract", "memory_forget: store failed"),
    ("restore", _restore, "restore", "memory_restore: store failed"),
]


class TestARevisionThatCannotBeWrittenChangesNothing:
    @pytest.mark.parametrize(("name", "call", "revision", "said"), _REVISION_SITES, ids=[s[0] for s in _REVISION_SITES])
    @pytest.mark.parametrize("failure", _WRITE_OUTAGES[:3], ids=_WRITE_OUTAGE_IDS[:3])
    async def test_an_outage_is_the_store_error(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        call: Callable[[MemoryAuthorizerDependencies], Awaitable[str]],
        revision: str,
        said: str,
        failure: Exception,
    ) -> None:
        _ = name
        monkeypatch.setattr(tools, revision, _raising(failure))
        result = await call(permissive_memory_authorizer)
        assert result.startswith(f"[TOOL ERROR] {said}"), result

    @pytest.mark.parametrize(
        "failure",
        [ConsolidationCycleError("the gist is its own source"), ValueError("memory is permanent")],
        ids=["ConsolidationCycleError", "ValueError"],
    )
    async def test_a_refused_replacement_is_the_update_store_error(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        failure: Exception,
    ) -> None:
        monkeypatch.setattr(tools, "supersede", _raising(failure))
        result = await _update(permissive_memory_authorizer, content="lives in York")
        assert result.startswith("[TOOL ERROR] memory_update: store failed"), result

    @pytest.mark.parametrize(("name", "call", "revision", "said"), _REVISION_SITES, ids=[s[0] for s in _REVISION_SITES])
    @pytest.mark.parametrize("bug", _BUGS, ids=_bug_ids())
    async def test_a_bug_propagates(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        call: Callable[[MemoryAuthorizerDependencies], Awaitable[str]],
        revision: str,
        said: str,
        bug: Exception,
    ) -> None:
        _ = name, said
        monkeypatch.setattr(tools, revision, _raising(bug))
        with pytest.raises(type(bug)):
            await call(permissive_memory_authorizer)


def _landed(unfinished: str = "mark", source_ids: tuple[UUID, ...] = (_TARGET,)) -> WriteIncompleteError:
    cause = DataLayerUnavailableError("broker down")
    landed = WriteIncompleteError(uuid4(), unfinished, cause, source_ids)  # type: ignore[arg-type]
    landed.__cause__ = cause
    return landed


class TestAWriteThatLandedIsSaidToHaveLanded:
    """a memory written before a later step failed is stored: saying "store failed" would have it written twice."""

    async def test_a_near_duplicate_replace_says_the_memory_is_stored_and_what_still_reads_as_current(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        landed = _landed()
        plain_writes: list[object] = []

        async def _write_once(*args: Any, **kwargs: Any) -> Written:
            plain_writes.append((args, kwargs))
            return await _wrote(*args, **kwargs)

        monkeypatch.setattr(tools, "supersede", _raising(landed))
        monkeypatch.setattr(tools, "write_once", _write_once)
        result = await _add(permissive_memory_authorizer, _FakeStore(find_similar_for_dedup=[_row()]))
        assert result.startswith(f"Stored as [memory:{landed.memory_id}]"), result
        assert f"[memory:{_TARGET}]" in result and "still read as current" in result, result
        assert "do not store it again" in result, result
        assert plain_writes == []

    async def test_a_correction_says_the_memory_is_stored(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        landed = _landed("edges")
        monkeypatch.setattr(tools, "supersede", _raising(landed))
        result = await _add(permissive_memory_authorizer, _FakeStore(), corrects=[str(_TARGET)])
        assert result.startswith(f"Stored as [memory:{landed.memory_id}]"), result
        assert "do not store it again" in result, result

    async def test_a_plain_write_whose_generation_did_not_advance_says_it_is_stored(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        landed = _landed("generation", ())
        monkeypatch.setattr(tools, "write_once", _raising(landed))
        result = await _add(permissive_memory_authorizer, _FakeStore())
        assert result.startswith(f"Stored as [memory:{landed.memory_id}]"), result
        assert "may not show in searches" in result, result

    async def test_an_update_says_the_replacement_is_written_and_keeps_its_name(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        landed = _landed()
        monkeypatch.setattr(tools, "supersede", _raising(landed))
        result = await _update(permissive_memory_authorizer, content="lives in York")
        assert result.startswith(f"Updated: [memory:{landed.memory_id}] replaces [memory:{_TARGET}]."), result
        assert "do not update it again" in result, result

    @pytest.mark.parametrize(
        ("unfinished", "said"),
        [("mark", "could not be marked as replaced by it"), ("edges", "could not be linked to")],
    )
    async def test_the_reply_names_the_step_that_was_not_done(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
        monkeypatch: pytest.MonkeyPatch,
        unfinished: str,
        said: str,
    ) -> None:
        monkeypatch.setattr(tools, "supersede", _raising(_landed(unfinished)))
        result = await _add(permissive_memory_authorizer, _FakeStore(find_similar_for_dedup=[_row()]))
        assert said in result, result

    async def test_a_correction_whose_link_beside_a_permanent_memory_failed_says_it_is_stored(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        permanent = uuid4()
        new_id = uuid4()

        # parity-exempt: _FakeStore answering a permanent memory for one id and a replaceable one for the other
        class _OnePermanent(_FakeStore):
            async def find_for_recall(self, **kwargs: Any) -> dict[str, Any] | None:
                if kwargs["memory_id"] == permanent:
                    return _row(memory_id=permanent, evergreen=True)
                return _row()

            def create(self, data: dict[str, Any]) -> SimpleNamespace:
                return SimpleNamespace(**data)

        async def _superseded(*args: Any, **kwargs: Any) -> UUID:
            _ = args, kwargs
            return new_id

        monkeypatch.setattr(tools, "supersede", _superseded)
        store = _OnePermanent(fail={"save_entity": KvError("kv down")})
        result = await _add(permissive_memory_authorizer, store, corrects=[str(_TARGET), str(permanent)])
        assert result.startswith(f"Stored as [memory:{new_id}]"), result
        assert f"[memory:{permanent}]" in result and "do not store it again" in result, result

    async def test_an_update_whose_replacement_landed_leaves_the_name_on_the_replacement(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # putting the name back on the old memory would collide with the replacement that holds it
        monkeypatch.setattr(tools, "supersede", _raising(_landed()))
        store = _FakeStore(find_for_recall=_row(alias="home"))
        result = await _update(permissive_memory_authorizer, store, content="lives in York")
        assert result.startswith("Updated: [memory:"), result
        assert [getattr(e, "alias", None) for e in store.saved] == [None], store.saved

    async def test_an_update_whose_name_came_off_under_a_stale_generation_goes_ahead(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        new_id = uuid4()

        async def _superseded(*args: Any, **kwargs: Any) -> UUID:
            _ = args, kwargs
            return new_id

        monkeypatch.setattr(tools, "supersede", _superseded)
        store = _FakeStore(
            fail={"save_entity": GenerationUnavailableError("generation store down")},
            find_for_recall=_row(alias="home"),
        )
        result = await _update(permissive_memory_authorizer, store, content="lives in York")
        assert result.startswith(f"Updated: [memory:{new_id}] replaces [memory:{_TARGET}]."), result

    async def test_an_update_whose_name_did_not_come_off_changes_nothing(
        self, permissive_memory_authorizer: MemoryAuthorizerDependencies, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tools, "supersede", _raising(AssertionError("nothing is replaced")))
        store = _FakeStore(
            fail={"save_entity": GenerationNotCommittedError("rolled back")},
            find_for_recall=_row(alias="home"),
        )
        result = await _update(permissive_memory_authorizer, store, content="lives in York")
        assert result.startswith("[TOOL ERROR] memory_update: store failed"), result
