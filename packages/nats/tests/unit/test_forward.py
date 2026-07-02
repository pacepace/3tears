"""unit tests for :mod:`threetears.nats.forward`.

cover the wire-framing round-trip (ok payload vs error frame), the
handler-exception -> error-frame mapping, subject derivation
(deterministic, subject-safe, collision-distinct), and the
empty/unknown-tag guards. the two-pod owner-routing proof against a
real broker lives in the integration suite.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from threetears.nats import (
    ForwardedHandlerError,
    ForwardError,
    Subjects,
    set_default_namespace,
)
from threetears.nats.forward import _TAG_ERR, _TAG_OK, _decode_reply, _encode_err, _encode_ok


@pytest.fixture(autouse=True)
def _reset_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """each test starts from the documented default namespace."""
    monkeypatch.delenv("THREETEARS_NATS_SUBJECT_NAMESPACE", raising=False)
    set_default_namespace("3tears")


# --------------------------------------------------------------------------
# subject derivation
# --------------------------------------------------------------------------


def test_forward_subject_is_namespaced_sha256_token() -> None:
    """the forward subject is ``{ns}.forward.{sha256hex(key)}``."""
    key = "repo-x:branch-y"
    token = hashlib.sha256(key.encode("utf-8")).hexdigest()
    subject = Subjects.forward(key)
    assert subject.path == f"{'3tears'}.forward.{token}"
    assert subject.kind == "point"


def test_forward_subject_is_deterministic() -> None:
    """the same key derives the same subject every time (cross-pod agreement)."""
    assert Subjects.forward("k").path == Subjects.forward("k").path


def test_forward_subject_is_collision_distinct() -> None:
    """distinct keys derive distinct subjects."""
    assert Subjects.forward("alpha").path != Subjects.forward("beta").path


def test_forward_subject_token_is_subject_safe_for_hostile_keys() -> None:
    """keys with NATS-illegal chars (dots, spaces, wildcards) yield a safe token."""
    subject = Subjects.forward("a.b c*>:/weird")
    token = subject.path.rsplit(".", 1)[-1]
    # sha256 hex is [0-9a-f] only -- no illegal subject characters survive.
    assert all(c in "0123456789abcdef" for c in token)


def test_forward_subject_rejects_empty_key() -> None:
    """an empty key is a programming error, not a silent empty token."""
    with pytest.raises(ValueError, match="key must be non-empty"):
        Subjects.forward("")


# --------------------------------------------------------------------------
# wire framing: ok frame round-trip
# --------------------------------------------------------------------------


def test_ok_frame_round_trip() -> None:
    """an ok frame decodes back to the exact handler bytes."""
    payload = b"\x00\x01\x02 arbitrary bytes \xff"
    frame = _encode_ok(payload)
    assert frame[0] == _TAG_OK
    assert _decode_reply(frame) == payload


def test_ok_frame_round_trip_empty_payload() -> None:
    """an empty handler reply is unambiguous from an error frame (tag byte present)."""
    frame = _encode_ok(b"")
    assert frame == bytes([_TAG_OK])
    assert _decode_reply(frame) == b""


# --------------------------------------------------------------------------
# wire framing: error frame round-trip + handler-exception mapping
# --------------------------------------------------------------------------


def test_error_frame_carries_type_name_and_message() -> None:
    """encoding an exception captures its type name + message as JSON body."""

    class OpLogSequenceConflict(Exception):
        pass

    frame = _encode_err(OpLogSequenceConflict("expected seq 7, got 9"))
    assert frame[0] == _TAG_ERR
    decoded = json.loads(frame[1:].decode("utf-8"))
    assert decoded == {"type": "OpLogSequenceConflict", "message": "expected seq 7, got 9"}


def test_error_frame_decodes_to_forwarded_handler_error() -> None:
    """decoding an error frame raises ForwardedHandlerError with type + message preserved."""
    frame = _encode_err(ValueError("bad input"))
    with pytest.raises(ForwardedHandlerError) as excinfo:
        _decode_reply(frame)
    assert excinfo.value.type_name == "ValueError"
    assert excinfo.value.message == "bad input"
    # the str carries both so a log line is self-describing.
    assert "ValueError" in str(excinfo.value)
    assert "bad input" in str(excinfo.value)


def test_error_frame_round_trip_preserves_custom_type_name() -> None:
    """a consumer can map the forwarded type name back onto its own exception."""
    frame = _encode_err(RuntimeError("cas conflict"))
    with pytest.raises(ForwardedHandlerError) as excinfo:
        _decode_reply(frame)
    assert excinfo.value.type_name == "RuntimeError"


# --------------------------------------------------------------------------
# wire framing: malformed-frame guards
# --------------------------------------------------------------------------


def test_decode_rejects_empty_frame() -> None:
    """an empty frame (no tag byte) is a protocol error, not silently treated as ok."""
    with pytest.raises(ForwardError, match="empty frame"):
        _decode_reply(b"")


def test_decode_rejects_unknown_tag() -> None:
    """an unknown tag byte raises rather than silently mis-decoding."""
    with pytest.raises(ForwardError, match="unknown frame tag"):
        _decode_reply(bytes([0x7F]) + b"body")


def test_decode_rejects_malformed_error_frame() -> None:
    """an error frame with non-JSON / missing fields raises ForwardError, not KeyError."""
    bad = bytes([_TAG_ERR]) + b"not json"
    with pytest.raises(ForwardError, match="malformed error frame"):
        _decode_reply(bad)

    missing_fields = bytes([_TAG_ERR]) + json.dumps({"type": "X"}).encode("utf-8")
    with pytest.raises(ForwardError, match="malformed error frame"):
        _decode_reply(missing_fields)
