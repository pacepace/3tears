"""which delivered reply subjects count as the receiver's own consumer's, in both ack-reply forms.

nats-server puts a delivery's ack reply in one of two forms: nine tokens with no JetStream domain
(``$JS.ACK.<stream>.<consumer>.…``), and eleven or more with one
(``$JS.ACK.<domain>.<account>.<stream>.<consumer>.…``). A forgery in either form naming another
stream or consumer is refused, and so is anything that is not an ack reply at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from threetears.nats.own_consumer import delivered_by_own_consumer, random_consumer_name

_STREAM = "3tears-tools-results"
_CONSUMER = "result-waiter-0123456789abcdef0123456789abcdef"
_V1_TAIL = "1.7.1.1700000000000000000.0"
_V2_TAIL = "1.7.1.1700000000000000000.0.rnd"


@dataclass
class _Delivery:
    reply: str
    subject: str = "_INBOX.x"


@pytest.mark.parametrize(
    "reply",
    [
        f"$JS.ACK.{_STREAM}.{_CONSUMER}.{_V1_TAIL}",
        f"$JS.ACK.hub.ACCOUNTHASH.{_STREAM}.{_CONSUMER}.{_V2_TAIL}",
        f"$JS.ACK._.ACCOUNTHASH.{_STREAM}.{_CONSUMER}.1.7.1.1700000000000000000.0",
    ],
    ids=["v1", "v2-domain", "v2-no-domain-eleven-tokens"],
)
def test_the_receivers_own_ack_reply_counts(reply: str) -> None:
    assert delivered_by_own_consumer(_Delivery(reply), stream=_STREAM, consumer=_CONSUMER, receiver="test")


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ("", "no_jetstream_reply"),
        ("_INBOX_tool_pod_forger.reply", "no_jetstream_reply"),
        (f"$JS.ACK.{_STREAM}.{_CONSUMER}.1.7.1.1700000000000000000", "no_jetstream_reply"),
        (f"$JS.ACK.hub.ACC.{_STREAM}.{_CONSUMER}.1.7.1.0", "no_jetstream_reply"),
        (f"$JS.ACK.forger.{_CONSUMER}.{_V1_TAIL}", "other_stream"),
        (f"$JS.ACK.{_STREAM}.forger.{_V1_TAIL}", "other_consumer"),
        (f"$JS.ACK.hub.ACCOUNTHASH.forger.{_CONSUMER}.{_V2_TAIL}", "other_stream"),
        (f"$JS.ACK.hub.ACCOUNTHASH.{_STREAM}.forger.{_V2_TAIL}", "other_consumer"),
    ],
    ids=[
        "empty",
        "not-an-ack",
        "v1-short-by-one",
        "ten-tokens",
        "v1-other-stream",
        "v1-other-consumer",
        "v2-other-stream",
        "v2-other-consumer",
    ],
)
def test_a_forged_reply_is_refused_with_its_reason(reply: str, reason: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="threetears.nats.own_consumer"):
        assert not delivered_by_own_consumer(_Delivery(reply), stream=_STREAM, consumer=_CONSUMER, receiver="test")
    (record,) = caplog.records
    assert record.getMessage() == f"test refused a delivery not made by its own consumer ({reason})"


def test_consumer_names_are_random_past_the_prefix() -> None:
    names = {random_consumer_name("p-") for _ in range(1000)}
    assert len(names) == 1000
    assert all(len(name) == len("p-") + 32 for name in names)
