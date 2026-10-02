"""unit tests for :mod:`threetears.nats.forward`.

cover the wire-framing round-trip (ok payload vs error frame), the
handler-exception -> error-frame mapping, subject derivation for both the
unscoped and the family-scoped shape (deterministic, subject-safe,
collision-distinct, and pinned byte-for-byte on the unscoped one every
deployed owner already subscribes), and the empty/unknown-tag guards. the
two-pod owner-routing proof against a real broker lives in the integration
suite, as does the proof that a real server's matcher enforces the
family-scoped grant.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import timedelta

import pytest

from threetears.nats import (
    ForwardedHandlerError,
    ForwardError,
    IncomingMessage,
    Subject,
    Subjects,
    set_default_namespace,
)
from threetears.nats.forward import forward, serve_owner


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


def test_forward_subject_output_is_byte_identical_to_the_shipped_form() -> None:
    """the unscoped subject is pinned to a literal, not re-derived from the implementation.

    every deployed pod already subscribes this exact string, and the
    family-scoped sibling shares the hashing helper with it. re-deriving the
    expected value with ``hashlib`` here would agree with any change to that
    helper, including one that silently moved every existing owner's subject.
    """
    assert (
        Subjects.forward("repo-x:branch-y").path
        == "3tears.forward.b4b7d226624dbb97a4f939231c6434aa33f96c0cd1bc5b7621eaeedf86975a62"
    )
    assert Subjects.forward("repo-x:branch-y").kind == "point"


# --------------------------------------------------------------------------
# subject derivation: family-scoped
# --------------------------------------------------------------------------


def test_forward_scoped_subject_hashes_family_and_key() -> None:
    """the scoped subject is ``{ns}.forward.{sha256hex(family)}.{sha256hex(key)}``."""
    subject = Subjects.forward_scoped("hitl-tools.scrape-zone_alpha.1-0-0", "session-42")
    assert subject.path == (
        "3tears.forward."
        "89238c846facc157dce6f7338116d60b771c8ff32a658c0f75fd7723368ee131."
        "92e76c732d82ec49fb40ff0bb444430c52f63577fe1a055ea119693241b2d291"
    )
    assert subject.kind == "point"


def test_forward_scoped_subject_does_not_collide_with_the_unscoped_family() -> None:
    """the scoped shape carries two segments after ``forward``, the unscoped one carries one.

    an unscoped subscriber therefore never receives a scoped message and a
    permission granted over one shape never spans the other.
    """
    scoped = Subjects.forward_scoped("hitl-tools.scrape-zone_alpha.1-0-0", "session-42")
    unscoped = Subjects.forward("session-42")
    assert len(scoped.path.split(".")) == len(unscoped.path.split(".")) + 1
    assert scoped.path != unscoped.path
    assert not scoped.path.startswith(unscoped.path)


def test_forward_scoped_subject_separates_families_for_one_key() -> None:
    """the same key under two families is two subjects -- the point of the segment."""
    key = "session-42"
    alpha = Subjects.forward_scoped("hitl-tools.scrape-zone_alpha.1-0-0", key)
    beta = Subjects.forward_scoped("hitl-tools.scrape-zone_beta.1-0-0", key)
    assert alpha.path != beta.path
    # and only the family segment moved: the key segment is shared.
    assert alpha.path.rsplit(".", 1)[-1] == beta.path.rsplit(".", 1)[-1]


def test_forward_scoped_subject_is_deterministic() -> None:
    """two processes starting from the same family + key derive the same subject."""
    a = Subjects.forward_scoped("hitl-t", "k")
    b = Subjects.forward_scoped("hitl-t", "k")
    assert a.path == b.path


def test_forward_scoped_subject_survives_a_hostile_tool_name() -> None:
    """a family carrying a space, a ``*`` and a ``>`` still yields ``[0-9a-f]`` tokens.

    not hypothetical: ``ToolManifestEntry.name`` is an unvalidated bare ``str``
    and ``RegistrationHandler._validate_manifest`` checks only that ``pod_id``
    and ``tools`` are non-empty, so a name like this reaches the builder. a
    sanitizer would not close it -- both sanitizers in the codebase replace
    dots and nothing else, so the space would produce an illegal subject and
    the ``*`` / ``>`` would inject wildcards into a GRANT.
    """
    hostile = Subjects.hitl_forward_family("tools.evil name.* > .1-0-0")
    subject = Subjects.forward_scoped(hostile, "a key with > and * in it")

    prefix, family_token, key_token = subject.path.rsplit(".", 2)
    assert prefix == "3tears.forward"
    for token in (family_token, key_token):
        assert set(token) <= set("0123456789abcdef"), token
        assert len(token) == 64
    # and nothing hostile survived anywhere in the rendered subject.
    for illegal in (" ", "*", ">"):
        assert illegal not in subject.path


@pytest.mark.parametrize(
    ("family", "key", "match"),
    [
        ("", "k", "family must be non-empty"),
        ("hitl-t", "", "key must be non-empty"),
    ],
)
def test_forward_scoped_subject_rejects_empty_segments(family: str, key: str, match: str) -> None:
    """an empty segment is a programming error, never a silent empty token."""
    with pytest.raises(ValueError, match=match):
        Subjects.forward_scoped(family, key)


def test_forward_scoped_wildcard_is_an_exact_family_with_a_wildcard_key() -> None:
    """the grant pattern pins the family literal and wildcards only the key."""
    pattern = Subjects.forward_scoped_wildcard("hitl-tools.scrape-zone_alpha.1-0-0")
    assert pattern.path == "3tears.forward.89238c846facc157dce6f7338116d60b771c8ff32a658c0f75fd7723368ee131.*"
    assert pattern.kind == "pattern"
    # it matches the concrete subject the same family builds, and no other.
    concrete = Subjects.forward_scoped("hitl-tools.scrape-zone_alpha.1-0-0", "session-42")
    assert concrete.path.rsplit(".", 1)[0] == pattern.path.rsplit(".", 1)[0]
    other = Subjects.forward_scoped("hitl-tools.scrape-zone_beta.1-0-0", "session-42")
    assert other.path.rsplit(".", 1)[0] != pattern.path.rsplit(".", 1)[0]


def test_forward_scoped_wildcard_rejects_empty_family() -> None:
    """an empty family would render a grant with an empty token in it."""
    with pytest.raises(ValueError, match="family must be non-empty"):
        Subjects.forward_scoped_wildcard("")


def test_hitl_family_is_derived_from_the_tool_namespace_name() -> None:
    """the family is a plain string derived from the registered tool namespace name."""
    assert Subjects.hitl_forward_family("tools.scrape-zone_alpha.1-0-0") == "hitl-tools.scrape-zone_alpha.1-0-0"


def test_hitl_family_separates_tools_and_is_deterministic() -> None:
    """two tools derive two families; one tool derives one, in every process."""
    alpha = Subjects.hitl_forward_family("tools.scrape-zone_alpha.1-0-0")
    beta = Subjects.hitl_forward_family("tools.scrape-zone_beta.1-0-0")
    assert alpha != beta
    assert alpha == Subjects.hitl_forward_family("tools.scrape-zone_alpha.1-0-0")


def test_hitl_family_rejects_an_empty_owned_node() -> None:
    """an empty node would collapse every provider onto one family."""
    with pytest.raises(ValueError, match="owned_node must be non-empty"):
        Subjects.hitl_forward_family("")


@pytest.mark.asyncio
async def test_owner_and_caller_meet_on_the_scoped_subject_only_when_a_family_is_named() -> None:
    """both public functions route by ``family``: unscoped without one, family-scoped with one."""

    async def _echo(payload: bytes) -> bytes:
        return payload

    bus = _LoopbackBus()
    async with serve_owner(bus, "k", _echo):  # type: ignore[arg-type]
        await forward(bus, "k", b"x")  # type: ignore[arg-type]
    assert bus.served == [Subjects.forward("k").path]
    assert bus.requested == [Subjects.forward("k").path]

    scoped = _LoopbackBus()
    async with serve_owner(scoped, "k", _echo, family="hitl-t"):  # type: ignore[arg-type]
        await forward(scoped, "k", b"x", family="hitl-t")  # type: ignore[arg-type]
    assert scoped.served == [Subjects.forward_scoped("hitl-t", "k").path]
    assert scoped.requested == [Subjects.forward_scoped("hitl-t", "k").path]


# --------------------------------------------------------------------------
# wire framing: ok frame round-trip
# --------------------------------------------------------------------------
#
# The frame is the wire contract between an owner and a caller on different pods, possibly on
# different releases, so its bytes are asserted literally: tag ``0x00`` and the handler's reply
# verbatim, or tag ``0x01`` and UTF-8 JSON ``{type, message}``.

_TAG_OK_BYTE = 0x00
_TAG_ERR_BYTE = 0x01


@pytest.mark.asyncio
async def test_ok_frame_round_trip() -> None:
    """an ok frame decodes back to the exact handler bytes."""
    payload = b"\x00\x01\x02 arbitrary bytes \xff"

    async def _handler(_request: bytes) -> bytes:
        return payload

    bus = _LoopbackBus()
    async with serve_owner(bus, "k", _handler):  # type: ignore[arg-type]
        assert await forward(bus, "k", b"req") == payload  # type: ignore[arg-type]
    assert bus.replies == [bytes([_TAG_OK_BYTE]) + payload]


@pytest.mark.asyncio
async def test_ok_frame_round_trip_empty_payload() -> None:
    """an empty handler reply is unambiguous from an error frame (tag byte present)."""

    async def _handler(_request: bytes) -> bytes:
        return b""

    bus = _LoopbackBus()
    async with serve_owner(bus, "k", _handler):  # type: ignore[arg-type]
        assert await forward(bus, "k", b"req") == b""  # type: ignore[arg-type]
    assert bus.replies == [bytes([_TAG_OK_BYTE])]


# --------------------------------------------------------------------------
# wire framing: error frame round-trip + handler-exception mapping
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_frame_carries_type_name_and_message() -> None:
    """a handler exception is framed as its type name + message, as JSON."""

    class OpLogSequenceConflict(Exception):
        pass

    async def _handler(_request: bytes) -> bytes:
        raise OpLogSequenceConflict("expected seq 7, got 9")

    bus = _LoopbackBus()
    async with serve_owner(bus, "k", _handler):  # type: ignore[arg-type]
        with pytest.raises(ForwardedHandlerError):
            await forward(bus, "k", b"req")  # type: ignore[arg-type]
    (frame,) = bus.replies
    assert frame[0] == _TAG_ERR_BYTE
    decoded = json.loads(frame[1:].decode("utf-8"))
    assert decoded == {"type": "OpLogSequenceConflict", "message": "expected seq 7, got 9"}


@pytest.mark.asyncio
async def test_error_frame_decodes_to_forwarded_handler_error() -> None:
    """decoding an error frame raises ForwardedHandlerError with type + message preserved."""

    async def _handler(_request: bytes) -> bytes:
        raise ValueError("bad input")

    bus = _LoopbackBus()
    async with serve_owner(bus, "k", _handler):  # type: ignore[arg-type]
        with pytest.raises(ForwardedHandlerError) as excinfo:
            await forward(bus, "k", b"req")  # type: ignore[arg-type]
    assert excinfo.value.type_name == "ValueError"
    assert excinfo.value.message == "bad input"
    # the str carries both so a log line is self-describing.
    assert "ValueError" in str(excinfo.value)
    assert "bad input" in str(excinfo.value)


@pytest.mark.asyncio
async def test_error_frame_round_trip_preserves_custom_type_name() -> None:
    """a consumer can map the forwarded type name back onto its own exception."""

    async def _handler(_request: bytes) -> bytes:
        raise RuntimeError("cas conflict")

    bus = _LoopbackBus()
    async with serve_owner(bus, "k", _handler):  # type: ignore[arg-type]
        with pytest.raises(ForwardedHandlerError) as excinfo:
            await forward(bus, "k", b"req")  # type: ignore[arg-type]
    assert excinfo.value.type_name == "RuntimeError"


# --------------------------------------------------------------------------
# wire framing: malformed-frame guards
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decode_rejects_empty_frame() -> None:
    """an empty frame (no tag byte) is a protocol error, not silently treated as ok."""
    with pytest.raises(ForwardError, match="empty frame"):
        await forward(_LoopbackBus(answer=b""), "k", b"req")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_decode_rejects_unknown_tag() -> None:
    """an unknown tag byte raises rather than silently mis-decoding."""
    with pytest.raises(ForwardError, match="unknown frame tag"):
        await forward(_LoopbackBus(answer=bytes([0x7F]) + b"body"), "k", b"req")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_decode_rejects_malformed_error_frame() -> None:
    """an error frame with non-JSON / missing fields raises ForwardError, not KeyError."""
    bad = bytes([_TAG_ERR_BYTE]) + b"not json"
    with pytest.raises(ForwardError, match="malformed error frame"):
        await forward(_LoopbackBus(answer=bad), "k", b"req")  # type: ignore[arg-type]

    missing_fields = bytes([_TAG_ERR_BYTE]) + json.dumps({"type": "X"}).encode("utf-8")
    with pytest.raises(ForwardError, match="malformed error frame"):
        await forward(_LoopbackBus(answer=missing_fields), "k", b"req")  # type: ignore[arg-type]


class _LoopbackBus:
    """the slice of :class:`NatsClient` the forward helpers use, wired back to itself in-process.

    ``serve_owner`` subscribes here; ``forward`` requests here, and the request is handed to that
    subscription with a reply subject whose reply is recorded and returned -- or, when ``answer``
    is given, ``forward`` receives that frame as if a misbehaving owner had sent it.

    :param answer: a raw reply frame to return to every request instead of dispatching it
    :ptype answer: bytes | None
    :ivar served: the subject of each subscription ``serve_owner`` made
    :ivar requested: the subject of each request ``forward`` made
    :ivar replies: every reply frame the owner published, in order
    """

    def __init__(self, *, answer: bytes | None = None) -> None:
        self._answer = answer
        self._callbacks: dict[str, Callable[[IncomingMessage], Awaitable[None]]] = {}
        self.served: list[str] = []
        self.requested: list[str] = []
        self.replies: list[bytes] = []

    async def subscribe(
        self, *, subject: Subject, cb: Callable[[IncomingMessage], Awaitable[None]], **_options: object
    ) -> str:
        self.served.append(subject.path)
        self._callbacks[subject.path] = cb
        return subject.path

    async def flush(self) -> None:
        return None

    async def unsubscribe(self, subscription: str) -> None:
        self._callbacks.pop(subscription, None)

    async def publish_raw_reply(self, *, reply_subject: str, payload: bytes) -> None:
        self.replies.append(payload)

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        self.requested.append(subject.path)
        if self._answer is not None:
            return self._answer
        before = len(self.replies)
        await self._callbacks[subject.path](
            IncomingMessage(data=payload, reply_subject="_INBOX.loopback", subject=subject.path)
        )
        return self.replies[before]
