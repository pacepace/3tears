"""integration: audit envelope lands at a stub Hub-side consumer.

REALISM / SCOPING LIMIT
-----------------------

- **real :func:`audit.publish_workspace_event`** from the agent side;
  :class:`DocSetTool` invokes it on every successful write.
- **fake NATS client** with in-process subject dispatch: a subscription
  registered on ``{namespace}.audit.workspace.`` invokes the stub
  consumer handler synchronously within :meth:`publish`, so publish
  latency is zero and there is no timing slack to manage.
- **stub :class:`AuditEventCollection`**: records ``save_entity``
  calls; the real implementation in the aibots repo writes to
  ``platform_audit.audit_events``. the shard explicitly permits this
  scoping trade-off ("If WorkspaceAuditConsumer lives in aibots, [...]
  verify the fake NATS bus captured the published envelope with
  correct subject + shape"). we go one step further: wire an in-
  process stub consumer that mirrors the real consumer's persist
  contract, so the end-to-end envelope handling is validated here
  even though the real DB row insert belongs to the aibots integration
  suite.

see ``README.md`` for graduation notes.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import pytest

from threetears.agent.workspace.config import AllowConfig, WorkspaceConfig
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.tools.doc_set import DocSetTool


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class _StubAuditEventCollection:
    """minimal stand-in for the Hub-side ``AuditEventCollection``.

    records every ``save_entity`` call verbatim. the shape mirrors the
    aibots-side collection well enough for the consumer protocol test.
    """

    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []

    async def save_entity(self, entity: dict[str, Any]) -> None:
        self.saved.append(entity)


class _InProcessAuditConsumer:
    """in-process stand-in for the Hub-side ``WorkspaceAuditConsumer``.

    decodes each incoming NATS envelope and forwards it to the injected
    :class:`_StubAuditEventCollection`. no timer, retries, or DLQ here;
    the real consumer adds those.
    """

    def __init__(self, collection: _StubAuditEventCollection) -> None:
        self._collection = collection

    async def handle(self, msg: Any) -> None:
        """
        subscription handler: decode payload -> stub collection.

        :param msg: fake NATS message with ``.subject`` and ``.data``
        :ptype msg: Any
        :return: None
        :rtype: None
        """
        envelope = json.loads(msg.data.decode("utf-8"))
        await self._collection.save_entity(envelope)


async def test_doc_set_publishes_audit_envelope_to_consumer(
    workspace_with_audience_fixture: Any,
    permissive_acl_cache: Any,
) -> None:
    """doc_set fires one envelope with canonical shape onto the bus.

    the in-process stub consumer receives the envelope and the stub
    collection records exactly one ``save_entity`` with
    ``event_type=workspace.doc_set``.

    :param workspace_with_audience_fixture: pre-seeded fixture bag
    :ptype workspace_with_audience_fixture: WorkspaceFixture
    :return: None
    :rtype: None
    """
    fx = workspace_with_audience_fixture
    collection = _StubAuditEventCollection()
    consumer = _InProcessAuditConsumer(collection)
    namespace = "threetears-test"
    fx.nats.register_subscription(f"{namespace}.audit.workspace.", consumer.handle)

    sandbox = WorkspaceSandbox.from_config(
        WorkspaceConfig(
            allow=AllowConfig(read=["**/*"], write=["**/*.yaml"]),
        )
    )

    from threetears.agent.workspace.pin import set_pin

    await set_pin(
        fx.context,
        workspace_id=fx.workspace_id,
        workspace_name=fx.workspace_name,
        pinned_by_actor_id=fx.agent_id,
    )

    doc_set = DocSetTool(
        workspace_collection=fx.workspace_collection,
        workspace_file_collection=fx.file_collection,
        workspace_file_version_collection=fx.version_collection,
        sandbox=sandbox,
        context_provider=lambda: fx.context,
        agent_id=fx.agent_id,
        db_pool=fx.pool,
        nats_client=fx.nats,
        namespace=namespace,
        acl_cache=permissive_acl_cache,
    )

    result = await doc_set.execute(
        relative_path="audience_settings.yaml",
        jsonpath="$.audience_units[0].vb_candidates",
        value=42,
    )
    assert result.success is True, result.error

    # the fake NATS bus captured exactly one publish on the expected subject.
    audit_publishes = [(s, p) for s, p in fx.nats.published if s.startswith(f"{namespace}.audit.workspace.")]
    assert len(audit_publishes) == 1, audit_publishes
    subject, payload = audit_publishes[0]
    assert subject == f"{namespace}.audit.workspace.set"
    envelope = json.loads(payload.decode("utf-8"))
    assert envelope["event_type"] == "workspace.doc_set"
    assert envelope["actor_type"] == "agent"
    # WS-ACL-10: actor_user_id replaces legacy actor_id; calling +
    # owner + customer + namespace carry full identity.
    assert UUID(envelope["actor_user_id"])  # present + parseable
    assert envelope["agent_id"] == str(fx.agent_id)
    assert envelope["calling_agent_id"]
    assert envelope["owner_agent_id"] == str(fx.agent_id)
    assert envelope["customer_id"]
    assert UUID(envelope["namespace_id"])
    assert envelope["resource_type"] == "workspace_file"
    assert envelope["action"] == "set"

    # the in-process consumer forwarded the envelope into the stub
    # collection exactly once.
    assert len(collection.saved) == 1
    persisted = collection.saved[0]
    assert persisted["event_type"] == "workspace.doc_set"
    assert persisted["resource_id"].endswith("/audience_settings.yaml")
