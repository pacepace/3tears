"""a durable that already exists takes the ack wait and delivery budget its code now asks for.

nats-py creates a durable only when it is missing and binds an existing one as it stands. The aibots
hub shortened its audit durable's ack wait from 60 to 20 seconds, rebuilt, and the live durable on
the running stack still answered 60: a deployed stream never sees a changed config unless the bind
updates it. Both consumer shapes, against a real server.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from threetears.nats import NatsClient, Subjects, set_default_namespace

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("consumer_kind", ["push", "pull"])
async def test_a_rebind_updates_the_ack_wait_and_budget_of_a_durable_made_earlier(
    nats_container: str, consumer_kind: str
) -> None:
    namespace = f"follows{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    client = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="hub")
    try:
        stream = await client.ensure_jetstream_stream(
            name=f"FOLLOWS{namespace}", subjects=[Subjects.audit_wildcard().path]
        )

        async def handle(msg: Any) -> None:
            await msg.ack()

        async def bind(ack_wait_seconds: float, max_deliver: int) -> Any:
            kwargs: dict[str, Any] = {
                "subject": Subjects.audit_wildcard(),
                "durable": "follows",
                "cb": handle,
                "max_deliver": max_deliver,
                "ack_wait_seconds": ack_wait_seconds,
                "stream": stream,
            }
            if consumer_kind == "push":
                return await client.jetstream_subscribe_durable(**kwargs)
            return await client.jetstream_pull_subscribe(**kwargs)

        first = await bind(60.0, 5)
        await first.stop()
        before = await client.jetstream_context().consumer_info(stream, "follows")
        second = await bind(20.0, 4)
        after = await client.jetstream_context().consumer_info(stream, "follows")
        await second.stop()

        assert (before.config.ack_wait, before.config.max_deliver) == (60.0, 5)
        assert (after.config.ack_wait, after.config.max_deliver) == (20.0, 4)
        assert after.created == before.created, "the durable was updated in place, not recreated"
    finally:
        await client.shutdown()
