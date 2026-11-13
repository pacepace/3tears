"""``threetears.nats.capabilities``: the client satisfies every capability protocol, a double holding only one
capability's methods is that capability, and importing them loads no NATS client."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from threetears.nats import (
    DurableSubscriber,
    JetStreamPublisher,
    KvBucketOpener,
    KvPublisher,
    NatsClient,
    Publisher,
    PullSubscriber,
    RawReplier,
    RawRequester,
    Subscriber,
    Subscription,
    SubscriptionHandle,
    TypedSubscriber,
)
from threetears.nats.kv import KvStreamClient

_CAPABILITIES: list[type[Any]] = [
    DurableSubscriber,
    JetStreamPublisher,
    KvBucketOpener,
    KvPublisher,
    Publisher,
    PullSubscriber,
    RawReplier,
    RawRequester,
    Subscriber,
    TypedSubscriber,
]


# held by mypy (typecheck.sh checks tests): a NatsClient is each capability, signatures and all
def _requester(client: NatsClient) -> RawRequester:
    return client


def _subscriber(client: NatsClient) -> Subscriber:
    return client


def _typed_subscriber(client: NatsClient) -> TypedSubscriber:
    return client


def _publisher(client: NatsClient) -> Publisher:
    return client


def _replier(client: NatsClient) -> RawReplier:
    return client


def _jetstream_publisher(client: NatsClient) -> JetStreamPublisher:
    return client


def _durable_subscriber(client: NatsClient) -> DurableSubscriber:
    return client


def _pull_subscriber(client: NatsClient) -> PullSubscriber:
    return client


def _kv_bucket_opener(client: NatsClient) -> KvBucketOpener:
    return client


def _kv_publisher(client: NatsClient) -> KvPublisher:
    return client


# what a bucket holds its client for (NatsKvBucket.open(client=...)), and the handle a typed subscription answers
def _kv_stream_client(client: NatsClient) -> KvStreamClient:
    return client


def _subscription_handle(subscription: Subscription) -> SubscriptionHandle:
    return subscription


@pytest.mark.parametrize("capability", _CAPABILITIES, ids=lambda c: c.__name__)
def test_the_client_is_every_capability(capability: type[Any]) -> None:
    client = NatsClient.__new__(NatsClient)
    assert isinstance(client, capability)


class _OnlyRequests:
    async def request_raw(self, *, subject: Any, payload: bytes, timeout: Any = None) -> bytes:
        return payload


def test_a_double_with_one_capability_s_methods_is_that_capability_and_no_other() -> None:
    double = _OnlyRequests()
    assert isinstance(double, RawRequester)
    assert not any(isinstance(double, other) for other in _CAPABILITIES if other is not RawRequester)


def test_the_client_is_what_a_bucket_holds_and_its_subscription_is_a_handle() -> None:
    assert isinstance(NatsClient.__new__(NatsClient), KvStreamClient)
    assert isinstance(Subscription.__new__(Subscription), SubscriptionHandle)
    assert not isinstance(_OnlyRequests(), KvStreamClient)


def test_importing_the_capabilities_loads_no_nats_client() -> None:
    code = "import sys, threetears.nats.capabilities; print('nats' in sys.modules, 'threetears.nats.client' in sys.modules)"
    answer = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert answer.stdout.split() == ["False", "False"]
