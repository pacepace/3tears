"""one hub ask: sent with its token request, decoded, matched to its request by correlation id, classified.

Every pod -> hub client (object store, collection keys, geography reload, audit anonymization) asks
through :func:`threetears.nats.hub_requests.ask_hub`, so its rules are pinned once here.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import UUID, uuid7

import pytest
from pydantic import BaseModel

from threetears.nats.errors import RequestError
from threetears.nats.hub_requests import ask_hub
from threetears.nats.subjects import Subject, Subjects, set_default_namespace


class _Request(BaseModel):
    correlation_id: UUID


class _Reply(BaseModel):
    success: bool
    correlation_id: UUID | None = None
    count: int | None = None
    error_code: str | None = None
    error_message: str | None = None


class _Unavailable(Exception):
    pass


class _Refused(Exception):
    def __init__(self, reply: _Reply) -> None:
        self.reply = reply
        super().__init__(f"{reply.error_code}: {reply.error_message}")


class _Hub:
    """answers with fixed bytes, or a transport failure; records what it was sent."""

    def __init__(self, answer: bytes | Exception) -> None:
        self.answer = answer
        self.sent: list[tuple[str, bytes, timedelta]] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        self.sent.append((subject.path, payload, timeout))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


_ID = uuid7()


async def _ask(answer: bytes | Exception) -> _Reply:
    return await ask_hub(
        _Hub(answer),  # type: ignore[arg-type]
        subject=Subjects.hub_collection_keys_purge(),
        request=_Request(correlation_id=_ID),
        reply_type=_Reply,
        what="a test ask",
        timeout_seconds=2.0,
        unavailable=_Unavailable,
        refused=_Refused,
        retryable={"HUB_FAILED"},
    )


def _reply(**fields: object) -> bytes:
    return _Reply.model_validate(fields).model_dump_json().encode()


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("3tears")


class TestAskHub:
    async def test_a_success_answering_this_request_is_returned(self) -> None:
        hub = _Hub(_reply(success=True, correlation_id=_ID, count=3))
        reply = await ask_hub(
            hub,  # type: ignore[arg-type]
            subject=Subjects.hub_collection_keys_purge(),
            request=_Request(correlation_id=_ID),
            reply_type=_Reply,
            what="a test ask",
            timeout_seconds=2.0,
            unavailable=_Unavailable,
            refused=_Refused,
        )
        assert reply.count == 3
        [(subject, payload, timeout)] = hub.sent
        assert subject == Subjects.hub_collection_keys_purge().path
        assert _Request.model_validate_json(payload).correlation_id == _ID
        assert timeout == timedelta(seconds=2.0)

    @pytest.mark.parametrize(
        "answer",
        [
            pytest.param({"success": True}, id="a success with no id"),
            pytest.param({"success": True, "correlation_id": str(uuid7())}, id="a success under another id"),
            pytest.param(
                {"success": False, "correlation_id": str(uuid7()), "error_code": "IDENTITY_REFUSED"},
                id="a refusal under another id",
            ),
        ],
    )
    async def test_a_reply_that_does_not_answer_this_request_is_unavailable(self, answer: dict[str, object]) -> None:
        with pytest.raises(_Unavailable, match="correlation_id"):
            await _ask(_reply(**answer))

    async def test_a_refusal_with_no_id_is_this_requests_and_final(self) -> None:
        with pytest.raises(_Refused) as raised:
            await _ask(_reply(success=False, error_code="INVALID_REQUEST", error_message="no body"))
        assert raised.value.reply.error_code == "INVALID_REQUEST"

    async def test_a_retryable_code_is_unavailable(self) -> None:
        with pytest.raises(_Unavailable, match="HUB_FAILED"):
            await _ask(_reply(success=False, correlation_id=_ID, error_code="HUB_FAILED"))

    async def test_any_other_refusal_is_built_by_the_caller(self) -> None:
        with pytest.raises(_Refused, match="IDENTITY_REFUSED"):
            await _ask(_reply(success=False, correlation_id=_ID, error_code="IDENTITY_REFUSED", error_message="who"))

    async def test_a_reply_that_does_not_decode_is_unavailable(self) -> None:
        with pytest.raises(_Unavailable, match="did not decode"):
            await _ask(b"not json")

    async def test_a_transport_failure_is_unavailable_with_its_cause(self) -> None:
        with pytest.raises(_Unavailable) as raised:
            await _ask(RequestError("no responders"))
        assert isinstance(raised.value.__cause__, RequestError)
