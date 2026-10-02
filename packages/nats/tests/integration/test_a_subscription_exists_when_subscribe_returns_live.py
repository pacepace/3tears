"""a subscription is live on the server by the time ``NatsClient.subscribe`` returns.

Found through the aibots hub's audit-anonymize test, intermittent under parallel load: a pod's
request answered "no responders" although the hub's responder had subscribed before the pod
asked. ``subscribe`` returned while the ``SUB`` was still in nats-py's pending buffer, so a request
sent at once from another connection reached a server that did not yet know of it -- 108 of 200
requests, measured. Every request here is sent the moment ``subscribe`` returns, from a second
connection, and every one must be answered.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import BaseModel

from threetears.nats import IncomingMessage, NatsClient, Subject

pytestmark = pytest.mark.integration

_TRIALS = 100


class _Ping(BaseModel):
    """the request and its answer."""

    text: str


async def test_a_request_sent_as_subscribe_returns_is_answered(nats_container: str) -> None:
    namespace = f"subnow{uuid.uuid4().hex[:6]}"
    responder = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="r")
    caller = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="c")
    answered = 0
    try:
        for _ in range(_TRIALS):
            subject = Subject.raw(f"{namespace}.{uuid.uuid4().hex}")

            async def answer(msg: IncomingMessage) -> None:
                assert msg.reply_subject is not None
                await responder.publish_reply(reply_subject=msg.reply_subject, message=_Ping(text="pong"))

            subscription = await responder.subscribe(subject, cb=answer)
            reply = await caller.request(subject=subject, message=_Ping(text="ping"), response_type=_Ping, timeout=2.0)
            answered += reply.text == "pong"
            await responder.unsubscribe(subscription)
    finally:
        await responder.shutdown()
        await caller.shutdown()
    assert answered == _TRIALS
