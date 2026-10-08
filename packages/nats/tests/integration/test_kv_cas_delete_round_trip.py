"""Integration test: a delete fenced on a revision the key has moved past is refused, and says so.

A remover that judged a key gone at one revision must not delete what a writer put there since.
The server refuses such a delete with a wrong-last-sequence error that nats-py, unlike on put and
update, does not map to ``KeyWrongLastSequenceError``; only a real server shows which error it is.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid

import pytest

from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration


async def test_a_delete_at_a_revision_the_key_moved_past_is_refused(nats_container: str) -> None:
    namespace = f"casdel{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="casdel"
    ) as nc:
        bucket = await nc.kv_bucket(name="casdel")
        judged = await bucket.put(key="k", value=b"old")
        moved = await bucket.put(key="k", value=b"new")
        assert await bucket.delete(key="k", revision=judged) is False
        assert await bucket.get(key="k") == b"new", "a refused delete removed the key"
        assert await bucket.delete(key="k", revision=moved) is True
        assert await bucket.get(key="k") is None
