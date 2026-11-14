"""the seeds the memory equivalence tests share: two agents, a caller and another user, the memories and files the
seeded rows hang from, a fresh schema holding the memory tables, and a collection over a connection.

Runs on PostgreSQL and YugabyteDB (the session's ``db_container``); each module that imports :func:`conn` gets a
schema of its own.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest_asyncio

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.same_results import Seeded

from .conftest import AsyncpgStore


NS = uuid.UUID("0c7f3e7a-51b2-4f43-9d0e-6a1f9b3e2d10")


AGENTS = (uuid.uuid5(NS, "agent-0"), uuid.uuid5(NS, "agent-1"))


USER = uuid.uuid5(NS, "user")


OTHER_USER = uuid.uuid5(NS, "other-user")


CUSTOMER = uuid.uuid5(NS, "customer")


CONVERSATION = uuid.uuid5(NS, "conversation")


T0 = datetime(2026, 9, 30, 12, 0)  # noqa: DTZ001 - the tables' columns are TIMESTAMP


#: a multiple of the ties' run of three, so a LIMIT never cuts a tie
PAGE = 6


def ident(kind: str, index: int) -> uuid.UUID:
    return uuid.uuid5(NS, f"{kind}-{index}")


#: memories each case's rows hang from: in each agent, the caller's memory in the conversation, another user's
#: memory in the conversation, and the caller's memory in another conversation
MEMORIES = {
    (agent, name): (ident(f"memory-{name}", index), user, conversation)
    for index, agent in enumerate(AGENTS)
    for name, user, conversation in (
        ("mine", USER, CONVERSATION),
        ("theirs", OTHER_USER, CONVERSATION),
        ("elsewhere", USER, uuid.uuid5(NS, "another conversation")),
    )
}


#: files each case's content rows hang from: in each agent, the caller's file and another user's file
FILES = {
    (agent, name): (ident(f"file-{name}", index), user, MEMORIES[(agent, "mine")][0])
    for index, agent in enumerate(AGENTS)
    for name, user in (("mine", USER), ("theirs", OTHER_USER))
}


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def conn(pg_url: str) -> AsyncIterator[asyncpg.Connection]:
    """a fresh schema holding the memory tables."""
    schema = f"mem_same_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(pg_url)
    await connection.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    await connection.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    await runner.apply_for_agent_schema(AsyncpgStore(connection))
    try:
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


@contextlib.asynccontextmanager
async def parents(connection: asyncpg.Connection) -> AsyncIterator[None]:
    """the memories and files the seeded rows hang from, for one test; gone after it."""
    await connection.executemany(
        "INSERT INTO memories (memory_id, agent_id, customer_id, user_id, conversation_id, type_memory, content,"
        " date_created, date_updated) VALUES ($1, $2, $3, $4, $5, 'fact', 'a memory', $6, $6)",
        [(memory, agent, CUSTOMER, user, conv, T0) for (agent, _), (memory, user, conv) in MEMORIES.items()],
    )
    await connection.executemany(
        "INSERT INTO media (media_id, memory_id, agent_id, customer_id, user_id, date_created, date_updated)"
        " VALUES ($1, $2, $3, $4, $5, $6, $6)",
        [(media, memory, agent, CUSTOMER, user, T0) for (agent, _), (media, user, memory) in FILES.items()],
    )
    try:
        yield
    finally:
        await connection.execute("DELETE FROM media")
        await connection.execute("DELETE FROM memories")


def collection(kind: type[Any], pool: Any) -> Any:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    if kind is MemoriesCollection:
        return kind(registry, config, authorizer=MagicMock(spec=MemoryAuthorizerDependencies))
    return kind(registry=registry, config=config)


def content_row(seed: Seeded) -> Mapping[str, Any]:
    """a content row on the caller's file of its tenant's agent; every fourth one written by another user."""
    agent = AGENTS[seed.tenant]
    return {
        "content_id": ident("content", seed.index),
        "media_id": FILES[(agent, "mine")][0],
        "agent_id": agent,
        "customer_id": CUSTOMER,
        "user_id": OTHER_USER if seed.index % 4 == 3 else USER,
        "content_type": "text",
        "content": f"content {seed.index}",
        "summary": None if seed.null else f"summary {seed.index}",
        "date_created": T0 + timedelta(minutes=seed.tie),
    }


CHUNK_MEMORIES = ("mine", "theirs", "elsewhere")


def chunk_row(seed: Seeded) -> Mapping[str, Any]:
    """a chunk of its tenant's agent, under the caller's memory in the conversation, another user's memory in it, or
    the caller's memory in another conversation in turn; every fifth one written by another user."""
    agent = AGENTS[seed.tenant]
    return {
        "chunk_id": ident("chunk", seed.index),
        "memory_id": MEMORIES[(agent, CHUNK_MEMORIES[seed.index % 3])][0],
        "agent_id": agent,
        "customer_id": CUSTOMER,
        "user_id": OTHER_USER if seed.index % 5 == 4 else USER,
        "chunk_index": seed.index,
        "content": f"chunk {seed.index}",
        "heading_context": None if seed.null else "a heading",
        "page_number": None if seed.null else seed.index,
        "date_created": T0,
        "message_id_start": None if seed.null else ident("message", seed.tie),
    }


#: the embedding width the package registers by default
DIM = 1024


OTHER_CUSTOMER = uuid.uuid5(NS, "other-customer")


def memory_row(seed: Seeded) -> Mapping[str, Any]:
    """a memory of its tenant's agent, older than the window on even rows and newer on odd ones; every fifth one of
    another user, every seventh one of another customer, its distance from the query its tie."""
    return {
        "memory_id": ident("memory", seed.index),
        "agent_id": AGENTS[seed.tenant],
        "customer_id": OTHER_CUSTOMER if seed.index % 7 == 6 else CUSTOMER,
        "user_id": OTHER_USER if seed.index % 5 == 4 else USER,
        "conversation_id": CONVERSATION,
        "type_memory": "fact",
        "content": f"memory {seed.index}",
        "date_created": T0 - timedelta(days=30, minutes=seed.index) if seed.index % 2 == 0 else datetime(2099, 1, 1),  # noqa: DTZ001
        "date_updated": T0,
        "embedding": None if seed.null else "[" + ",".join(["1"] * (DIM - 1) + [str(seed.tie)]) + "]",
    }


async def text_vectors(connection: asyncpg.Connection) -> None:
    """send and read embeddings as their text form on ``connection``: pgvector's type is an extension type on
    PostgreSQL and a built-in one on YugabyteDB (which asyncpg then refuses without a codec), so the codec is named
    where the type lives.

    :param connection: the connection the seeds are written on
    :ptype connection: asyncpg.Connection
    :return: nothing
    :rtype: None
    """
    vector_schema = await connection.fetchval(
        "SELECT n.nspname FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'vector'"
    )
    await connection.set_type_codec("vector", schema=vector_schema, encoder=str, decoder=str, format="text")
