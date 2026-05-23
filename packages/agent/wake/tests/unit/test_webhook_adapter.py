"""Unit tests for pure pieces of :mod:`threetears.agent.wake.webhook_adapter`.

The end-to-end ``webhook_receive`` flow is covered in the integration
suite (it requires a real pool + Collections). Here we exercise the
payload-decoding helper + the result envelope contract.
"""

from __future__ import annotations

import pytest

from threetears.agent.wake.webhook_adapter import (
    WebhookReceiveResult,
    _decode_payload,
)


def test_decode_payload_json_object() -> None:
    assert _decode_payload(b'{"type": "push"}') == {"type": "push"}


def test_decode_payload_json_array() -> None:
    assert _decode_payload(b"[1, 2, 3]") == [1, 2, 3]


def test_decode_payload_plain_text_falls_back_to_string() -> None:
    assert _decode_payload(b"hello world") == "hello world"


def test_decode_payload_empty_returns_empty_dict() -> None:
    assert _decode_payload(b"") == {}


def test_decode_payload_invalid_utf8_raises() -> None:
    with pytest.raises(ValueError):
        _decode_payload(b"\xff\xfe\xfd")


def test_webhook_receive_result_is_frozen() -> None:
    res = WebhookReceiveResult(status_code=202, fire_id=None, message="ok")
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        res.status_code = 500  # type: ignore[misc]
