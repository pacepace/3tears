"""a read routed to a hub replica that stops without answering is answered by a sibling, within its bound.

Two broker replicas share the ``l3-broker`` queue group on a real bus. One stops after taking its
first request and never answers it, as a hub replica killed mid-rollout does; the other answers every
request. Each read the stopped replica took times out at the proxy's bound and is sent again, to the
sibling. Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import pytest
from threetears.nats import IncomingMessage, NatsClient, Subjects, set_default_namespace

from threetears.core.backends.nats_proxy import NatsProxyL3Backend

pytestmark = pytest.mark.integration

#: the proxy's statement bound; it waits this and two seconds for a reply
_TIMEOUT_MS = 300
_REPLY_BOUND_SECONDS = _TIMEOUT_MS / 1000 + 2


async def test_a_read_the_stopped_replica_took_times_out_and_is_answered_by_its_sibling(nats_container: str) -> None:
    namespace = f"retry_{uuid.uuid4().hex[:8]}"
    set_default_namespace(namespace)
    subject = Subjects.l3_query()
    taken = asyncio.Event()
    stopped_took: list[bytes] = []
    answered: list[bytes] = []

    async with (
        await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="stops"
        ) as stops,
        await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="serves"
        ) as serves,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod") as pod,
    ):

        async def never_answers(msg: IncomingMessage) -> None:
            stopped_took.append(msg.data)
            taken.set()

        async def answers(msg: IncomingMessage) -> None:
            answered.append(msg.data)
            assert msg.reply_subject is not None
            reply = json.dumps({"success": True, "rows": [{"n": 1}], "row_count": 1}).encode()
            await serves.publish_raw_reply(reply_subject=msg.reply_subject, payload=reply)

        stopping = await stops.subscribe(subject, cb=never_answers, queue="l3-broker", max_in_flight=10)
        await serves.subscribe(subject, cb=answers, queue="l3-broker", max_in_flight=10)
        proxy = NatsProxyL3Backend(
            nats_client=pod,
            namespace_prefix=namespace,
            agent_id=str(uuid.uuid4()),
            identity_token=lambda: "identity-token",
            timeout_ms=_TIMEOUT_MS,
        )

        async def timed_read() -> tuple[list[dict[str, int]], float]:
            started = time.monotonic()
            rows = await proxy.fetch("SELECT 1 AS n FROM items")
            return rows, time.monotonic() - started

        reads = [asyncio.create_task(timed_read()) for _ in range(12)]
        await asyncio.wait_for(taken.wait(), timeout=5)
        # the replica stops: NATS routes it nothing more, and what it took is never answered
        await stops.unsubscribe(stopping)
        results = await asyncio.gather(*reads)

    assert stopped_took, "some read was routed to the replica that stopped"
    assert all(rows == [{"n": 1}] for rows, _ in results), results
    retried = [elapsed for _, elapsed in results if elapsed >= _REPLY_BOUND_SECONDS]
    assert len(retried) >= 1, "a read the stopped replica took waited out its bound, then was answered"
    assert all(elapsed < 2 * _REPLY_BOUND_SECONDS + 1 for _, elapsed in results), (
        "each read the stopped replica took is answered by its next attempt, within one more bound"
    )
    assert len(answered) == len(results), "the sibling answered every read once, the retried ones included"
