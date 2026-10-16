"""a channel answer's dedup key is its turn's CSPRNG id, which a forger cannot pre-claim."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from uuid import uuid4

from threetears.channels import ChannelDeliveryMessage


def _delivery(**overrides):
    fields = {
        "correlation_id": uuid4(),
        "conversation_id": uuid4(),
        "agent_id": uuid4(),
        "channel_type": "slack",
        "content": "the answer",
    }
    fields.update(overrides)
    return ChannelDeliveryMessage(**fields)


def test_an_answer_with_a_turn_id_dedups_on_it_not_the_correlation_id() -> None:
    turn_id = "a" * 32
    delivery = _delivery(turn_id=turn_id)
    assert delivery.dedup_key == f"{delivery.conversation_id.hex}-{turn_id}"
    assert delivery.correlation_id.hex not in delivery.dedup_key


def test_a_forged_answer_on_the_correlation_id_claims_another_key() -> None:
    conversation, correlation = uuid4(), uuid4()
    forged = _delivery(conversation_id=conversation, correlation_id=correlation)
    real = _delivery(conversation_id=conversation, correlation_id=correlation, turn_id="b" * 32)
    assert forged.dedup_key != real.dedup_key


def test_an_answer_without_a_turn_id_keeps_its_old_key() -> None:
    delivery = _delivery()
    assert delivery.dedup_key == f"{delivery.conversation_id.hex}-{delivery.correlation_id.hex}"


def test_a_turn_id_that_is_not_one_hex_token_is_refused() -> None:
    with pytest.raises(ValidationError):
        _delivery(turn_id="not.a.kv-key*")
