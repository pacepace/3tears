"""integration: the history pages whose key runs the way their order does walk their index, on PostgreSQL and YugabyteDB.

A page ordered ``date_created DESC`` and completed by its key ascending (the key-completion codemod's first shape)
has no index holding that order: PostgreSQL reads the index and adds an incremental sort, YugabyteDB (no incremental
sort) reads every row the pins name and sorts them. Completed by the key the way the order runs, a page is one walk
of an index holding ``(pins, date_created DESC, key DESC)``. Each test reads the statement the collection really
sends (its collection over a recording pool) at a size where the planner would rather scan, and asks the plan for the
index, no sort, and about the page's own rows.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from threetears.agent.tools.collections import ContextItemCollection
from threetears.agent.workspace.collections import WorkspaceFileVersionCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.captured import CapturedL3
from threetears.core.testing.plans import explain_statement, plan_uses_index

pytestmark = pytest.mark.integration

#: the page each read asks
_PAGE = 10
#: the busy workspace: 3,000 journal rows over 30 paths (100 versions each), a tenth of them at one instant, beside
#: 100 other workspaces' 20 rows each
_VERSIONS = 3000
_PATHS = 30


@pytest.fixture(params=["conn", "yugabyte_conn"])
def db(request: pytest.FixtureRequest) -> Any:
    """the session's migrated schema on whichever engine it is (the other skips)."""
    return request.getfixturevalue(request.param)


def _node_types(node: Mapping[str, Any]) -> Iterator[str]:
    yield str(node.get("Node Type", ""))
    for child in node.get("Plans", ()):
        yield from _node_types(child)


async def _busy_workspace(db: Any) -> uuid.UUID:
    """seed the busy workspace and the others (committed: the session's schema is dropped at its end), then ANALYZE."""
    busy = uuid.uuid4()
    others = [uuid.uuid4() for _ in range(100)]
    await db.executemany(
        "INSERT INTO workspaces (workspace_id, agent_id, name, created_by, current_version, date_created,"
        " date_updated) VALUES ($1, gen_random_uuid(), $2, gen_random_uuid(), 0, now(), now())",
        [(workspace, f"plan-{workspace}") for workspace in [busy, *others]],
    )
    insert = (
        "INSERT INTO workspace_file_versions (version_id, workspace_id, relative_path, version, content, sha256,"
        " action, actor_id, correlation_id, date_created)"
        " SELECT gen_random_uuid(), $1, 'p' || (g % $3) || '.md', g, 'c'::bytea, repeat('0', 64), 'update',"
        " gen_random_uuid(), gen_random_uuid(),"
        " CASE WHEN g % 10 = 0 THEN timestamptz '2026-11-14' ELSE timestamptz '2026-11-01' + g * interval '1 s' END"
        " FROM generate_series(1, $2) g"
    )
    await db.execute(insert, busy, _VERSIONS, _PATHS)
    for workspace in others:
        await db.execute(insert, workspace, 20, 4)
    await db.execute("ANALYZE workspace_file_versions")
    return busy


def _versions(captured: CapturedL3) -> WorkspaceFileVersionCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=captured)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return WorkspaceFileVersionCollection(registry, config, captured)


async def _page_plan(db: Any, sent: CapturedL3) -> Any:
    (sql, args), *_ = [asked for asked in sent.asked if "workspace_file_versions" in asked[0]]
    return await explain_statement(db, sql, *args)


async def test_workspace_file_versions_history_key(db: Any) -> None:
    """Flips if a workspace's history page (newest first, completed by version_id DESC) stops reading
    idx_workspace_file_versions_history_key in its order, so it reads and sorts the workspace's journal (v006: v001's
    (workspace_id, date_created) held no key; YugabyteDB sorted the busy workspace's 3,000 rows for a page of 10)."""
    busy = await _busy_workspace(db)
    sent = CapturedL3(real=db)
    await _versions(sent).find_by_workspace(busy, _PAGE)
    plan = await _page_plan(db, sent)
    assert plan_uses_index(plan, "idx_workspace_file_versions_history_key"), plan.plan
    assert not set(_node_types(plan.plan)) & {"Sort", "Incremental Sort"}, plan.plan
    # the page with keyset_page's one row past it (whether a next page exists), from the index's first rows
    assert plan.rows_returned == _PAGE + 1, plan
    assert plan.rows_scanned <= 2 * (_PAGE + 1), plan


async def test_workspace_file_versions_path_history(db: Any) -> None:
    """Flips if one path's history page stops reading idx_workspace_file_versions_path_history in its order: reading
    the UNIQUE (workspace_id, relative_path, version) sorts every version of the path, and the workspace-wide index
    walks the whole journal filtering the path (here 30 rows read for each one kept)."""
    busy = await _busy_workspace(db)
    sent = CapturedL3(real=db)
    await _versions(sent).find_by_workspace_and_path(busy, "p7.md", _PAGE)
    plan = await _page_plan(db, sent)
    assert plan_uses_index(plan, "idx_workspace_file_versions_path_history"), plan.plan
    assert not set(_node_types(plan.plan)) & {"Sort", "Incremental Sort"}, plan.plan
    assert plan.rows_returned == _PAGE + 1, plan
    assert plan.rows_scanned <= 2 * (_PAGE + 1), plan


#: the busy conversation's context rows, beside :data:`_OTHER_CONVERSATIONS` others of :data:`_OTHER_ITEMS` each
_ITEMS = 300
_OTHER_CONVERSATIONS = 200
_OTHER_ITEMS = 20


async def _busy_conversation(db: Any) -> uuid.UUID:
    """seed the busy conversation's context rows and the others' (committed, as :func:`_busy_workspace`), then
    ANALYZE."""
    busy = uuid.uuid4()
    conversations = [busy, *(uuid.uuid4() for _ in range(_OTHER_CONVERSATIONS))]
    await db.execute(
        "INSERT INTO conversations (agent_id, conversation_id, customer_id, user_id, channel_type, status,"
        " date_created, date_updated, message_count)"
        " SELECT gen_random_uuid(), c, gen_random_uuid(), gen_random_uuid(), 'web', 'active', now(), now(), 0"
        " FROM unnest($1::uuid[]) c",
        conversations,
    )
    insert = (
        "INSERT INTO context_items (conversation_id, context_id, context_type, key, short_desc, content,"
        " date_accessed, date_created, date_updated)"
        " SELECT $1, gen_random_uuid(), 'tool_result', 'k' || g, 's', 'c', now(),"
        " timestamptz '2026-11-01' + g * interval '1 s', now() FROM generate_series(1, $2) g"
    )
    await db.execute(insert, busy, _ITEMS)
    for conversation in conversations[1:]:
        await db.execute(insert, conversation, _OTHER_ITEMS)
    await db.execute("ANALYZE context_items")
    return busy


async def test_find_by_conversation_reads_the_primary_key(db: Any) -> None:
    """Flips if a conversation's context rows stop being read by an index leading with conversation_id: the held
    contract step (tools v006) drops ix_context_items_conv, whose one column the primary key (and
    ix_context_items_type) leads with, so the read must stay bounded to the one conversation's rows on both engines
    without it."""
    busy = await _busy_conversation(db)
    sent = CapturedL3(real=db)
    registry = CollectionRegistry()
    registry.configure(l3_pool=sent)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    rows = await ContextItemCollection(registry, config).find_by_conversation(busy)
    assert len(rows) == _ITEMS
    (sql, args), *_ = [asked for asked in sent.asked if "context_items" in asked[0]]
    plan = await explain_statement(db, sql, *args)
    # the primary key or ix_context_items_type, each led by conversation_id (the planner's pick between them)
    assert plan_uses_index(plan, "context_items_pkey") or plan_uses_index(plan, "ix_context_items_type"), plan.plan
    assert "Seq Scan" not in set(_node_types(plan.plan)), plan.plan
    # the one conversation's rows, never the others' 4,000
    assert plan.rows_returned == _ITEMS, plan
    assert plan.rows_scanned <= 2 * _ITEMS, plan
