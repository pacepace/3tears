"""integration: the memory-extraction cooldown against a real JetStream broker.

What an in-memory double cannot prove, proved against the session ``nats_container``:

- the cooldown lives on the KEY: a bucket some other component created with a longer
  ``max_age`` (live dev had ``KV_metallm-locks`` at 600 s against a 300 s cooldown) no longer
  stretches it;
- the claim is atomic on the broker: of many turns racing past the read, exactly one claims;
- a bucket whose stream lacks ``allow_msg_ttl`` is reconciled in place by the extractor's
  declaring open, and when the only handle is a bind-only one that cannot reconcile, the claim
  fails loud naming the bucket instead of writing a key that would outlive its cooldown.

A checkout without docker skips cleanly through the fixture.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from nats.js.api import StorageType

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.extraction import MemoryExtractor
from threetears.agent.memory.types import MemoryConfig
from threetears.nats import NatsClient, set_default_namespace
from threetears.nats.errors import KvError
from threetears.nats.kv import build_kv_stream_config

pytestmark = pytest.mark.integration

_BUCKET = "locks"


def _extractor(nc: NatsClient, authorizer: MemoryAuthorizerDependencies, cooldown: int) -> MemoryExtractor:
    """an extractor whose rate-limit hooks run against ``nc``; nothing else is exercised.

    :param nc: connected client
    :ptype nc: NatsClient
    :param authorizer: rbac bundle, unused by the rate-limit hooks
    :ptype authorizer: MemoryAuthorizerDependencies
    :param cooldown: extraction cooldown in seconds
    :ptype cooldown: int
    :return: extractor
    :rtype: MemoryExtractor
    """
    return MemoryExtractor(
        config=MemoryConfig(extraction_rate_limit_cooldown_seconds=cooldown),
        embedding_provider=MagicMock(),
        chat_model_factory=MagicMock(),
        authorizer=authorizer,
        memories_collection=MagicMock(),
        nats_client=nc,
        rate_limit_bucket=_BUCKET,
    )


async def _connect(url: str, namespace: str) -> NatsClient:
    """connect one client in ``namespace``.

    :param url: broker url
    :ptype url: str
    :param namespace: subject namespace
    :ptype namespace: str
    :return: connected client
    :rtype: NatsClient
    """
    return await NatsClient.connect(nats_url=url, nats_subject_namespace=namespace, client_name="extractor")


async def _legacy_bucket(nc: NatsClient, namespace: str) -> None:
    """create the bucket's stream the way a pre-TTL opener did: long max_age, no allow_msg_ttl.

    :param nc: connected client
    :ptype nc: NatsClient
    :param namespace: subject namespace
    :ptype namespace: str
    :return: nothing
    :rtype: None
    """
    legacy = build_kv_stream_config(
        bucket=f"{namespace}-{_BUCKET}", ttl_seconds=600, history=1, storage_type=StorageType.MEMORY, direct=None
    )
    legacy.allow_msg_ttl = False
    await nc.jetstream_context().add_stream(legacy)


async def test_the_cooldown_lives_on_the_key_not_the_shared_bucket(
    nats_container: str,
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    namespace = f"memrl{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await _connect(nats_container, namespace) as nc:
        # another component owns the bucket and gave it a ten-minute max_age.
        await nc.kv_bucket(name=_BUCKET, ttl=timedelta(seconds=600))
        ext = _extractor(nc, permissive_memory_authorizer, cooldown=2)
        conversation_id = uuid.uuid7()
        assert await ext.claim_rate_limit(conversation_id) == (True, 0)
        assert await ext.check_rate_limit(conversation_id) == (False, 2)
        await asyncio.sleep(3.5)
        assert await ext.check_rate_limit(conversation_id) == (True, 0), (
            "the cooldown key outlived its own lifetime -- it took the bucket's max_age"
        )


async def test_exactly_one_of_many_racing_turns_claims(
    nats_container: str,
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    namespace = f"memrl{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await _connect(nats_container, namespace) as nc:
        ext = _extractor(nc, permissive_memory_authorizer, cooldown=30)
        conversation_id = uuid.uuid7()
        results = await asyncio.gather(*(ext.claim_rate_limit(conversation_id) for _ in range(8)))
        assert results.count((True, 0)) == 1, results
        assert results.count((False, 30)) == 7, results


async def test_a_legacy_bucket_is_reconciled_and_then_carries_the_key_lifetime(
    nats_container: str,
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    namespace = f"memrl{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await _connect(nats_container, namespace) as nc:
        await _legacy_bucket(nc, namespace)
        ext = _extractor(nc, permissive_memory_authorizer, cooldown=2)
        conversation_id = uuid.uuid7()
        assert await ext.claim_rate_limit(conversation_id) == (True, 0)
        info = await nc.jetstream_context().stream_info(f"KV_{namespace}-{_BUCKET}")
        assert info.config.allow_msg_ttl is True, "the extractor's open did not enable per-key lifetimes"
        await asyncio.sleep(3.5)
        assert await ext.check_rate_limit(conversation_id) == (True, 0)


async def test_a_bucket_that_cannot_carry_a_key_lifetime_fails_loud(
    nats_container: str,
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
    caplog: pytest.LogCaptureFixture,
) -> None:
    namespace = f"memrl{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await _connect(nats_container, namespace) as nc:
        await _legacy_bucket(nc, namespace)
        # a bind-only open cached first: this handle can never reconcile allow_msg_ttl.
        await nc.kv_bucket(name=_BUCKET, create_if_missing=False)
        ext = _extractor(nc, permissive_memory_authorizer, cooldown=2)
        with caplog.at_level(logging.ERROR, logger="threetears.agent.memory.extraction"), pytest.raises(KvError):
            await ext.claim_rate_limit(uuid.uuid7())
        full_name = f"{namespace}-{_BUCKET}"
        assert any(r.levelno == logging.ERROR and full_name in r.getMessage() for r in caplog.records), [
            r.getMessage() for r in caplog.records
        ]
