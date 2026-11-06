"""a tool pod tells the hub that a CDN-cached index it serves moved, and the hub purges the customer's copy."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import UUID, uuid7

import pytest
from pydantic import SecretStr, ValidationError

from threetears.nats.cache_index_requests import (
    CACHE_INDEX_REQUEST_ERROR_CODES,
    CacheIndexMovedReply,
    CacheIndexMovedRequest,
    CacheIndexRequestRefusedError,
    CacheIndexRequestUnavailableError,
    report_cache_index_moved,
)
from threetears.nats.errors import RequestTimeoutError
from threetears.nats.subject_permissions import Principal, build_permissions
from threetears.nats.subjects import Subject, Subjects, set_default_namespace


_CUSTOMER = UUID("0198a3f2-1c4e-7b9a-8d2f-3e4a5b6c7d8e")


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("3tears")


class _Hub:
    """answers an index-moved report the way a hub does, or not at all."""

    def __init__(self, reply: CacheIndexMovedReply | None = None, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.asked: list[CacheIndexMovedRequest] = []
        self.subjects: list[str] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        del timeout
        self.subjects.append(subject.path)
        request = CacheIndexMovedRequest.model_validate_json(payload)
        self.asked.append(request)
        if self.error is not None:
            raise self.error
        assert self.reply is not None
        return self.reply.model_copy(update={"correlation_id": request.correlation_id}).model_dump_json().encode()


class TestTheSubjectAndItsGrant:
    def test_the_subject_name(self) -> None:
        assert Subjects.hub_cache_index_moved().path == "3tears.hub.cache.index.moved"

    def test_a_tool_pod_publishes_and_the_hub_answers(self) -> None:
        tool_pod = build_permissions(Principal.TOOL_POD, pod_id=str(uuid7()))
        hub = build_permissions(Principal.HUB, conn_id="hub-1")
        agent = build_permissions(Principal.AGENT_POD, agent_id=str(uuid7()), pod_id=str(uuid7()))
        subject = "3tears.hub.cache.index.moved"
        assert subject in tool_pod.publish and subject not in tool_pod.subscribe
        assert subject in hub.subscribe
        assert subject not in agent.publish


class TestTheRequest:
    def test_it_round_trips_with_the_token_on_the_wire_and_redacted_elsewhere(self) -> None:
        request = CacheIndexMovedRequest(
            identity_token=SecretStr("s3cret-value"),
            correlation_id=uuid7(),
            customer_id=_CUSTOMER,
            conversion="enr-2026",
            version="v1f3a",
            scope="VA",
        )
        wire = request.model_dump_json()
        assert '"identity_token":"s3cret-value"' in wire
        assert "s3cret-value" not in repr(request)
        back = CacheIndexMovedRequest.model_validate_json(wire)
        assert back == request

    @pytest.mark.parametrize("field", ["customer_id", "conversion", "version"])
    def test_an_empty_name_is_refused(self, field: str) -> None:
        values: dict[str, Any] = {"customer_id": _CUSTOMER, "conversion": "c", "version": "v"} | {field: ""}
        with pytest.raises(ValidationError):
            CacheIndexMovedRequest(identity_token=SecretStr("tok"), correlation_id=uuid7(), **values)

    def test_the_reply_round_trips(self) -> None:
        reply = CacheIndexMovedReply(
            success=False, correlation_id=uuid7(), error_code="PURGE_FAILED", error_message="x"
        )
        assert CacheIndexMovedReply.model_validate_json(reply.model_dump_json()) == reply


class TestThePodsReport:
    async def test_the_hub_is_told_the_customer_conversion_version_and_scope(self) -> None:
        hub = _Hub(CacheIndexMovedReply(success=True))
        reply = await report_cache_index_moved(
            hub,  # type: ignore[arg-type]
            identity_token="tok",
            customer_id=_CUSTOMER,
            conversion="enr-2026",
            version="v2",
            scope="VA",
        )
        assert reply.success
        assert hub.subjects == [Subjects.hub_cache_index_moved().path]
        asked = hub.asked[0]
        assert (asked.customer_id, asked.conversion, asked.version, asked.scope) == (_CUSTOMER, "enr-2026", "v2", "VA")

    async def test_a_hub_that_does_not_answer_is_unavailable(self) -> None:
        with pytest.raises(CacheIndexRequestUnavailableError):
            await report_cache_index_moved(
                _Hub(error=RequestTimeoutError("no answer")),  # type: ignore[arg-type]
                identity_token="tok",
                customer_id=_CUSTOMER,
                conversion="c",
                version="v",
            )

    @pytest.mark.parametrize(
        ("code", "raised"),
        [("IDENTITY_REFUSED", CacheIndexRequestRefusedError), ("PURGE_FAILED", CacheIndexRequestUnavailableError)],
    )
    async def test_a_refusal_is_refused_and_a_hub_failure_is_unavailable(
        self, code: str, raised: type[Exception]
    ) -> None:
        hub = _Hub(CacheIndexMovedReply(success=False, error_code=code, error_message="no"))
        with pytest.raises(raised):
            await report_cache_index_moved(
                hub, identity_token="tok", customer_id=_CUSTOMER, conversion="c", version="v"
            )  # type: ignore[arg-type]

    async def test_no_token_asks_nothing(self) -> None:
        hub = _Hub(CacheIndexMovedReply(success=True))
        with pytest.raises(CacheIndexRequestUnavailableError):
            await report_cache_index_moved(hub, identity_token="", customer_id=_CUSTOMER, conversion="c", version="v")  # type: ignore[arg-type]
        assert hub.asked == []


class TestTheCodes:
    def test_a_responder_answers_with_these_and_only_a_purge_failure_is_retried(self) -> None:
        assert CACHE_INDEX_REQUEST_ERROR_CODES == {
            "INVALID_REQUEST",
            "IDENTITY_REFUSED",
            "INDEX_NOT_OWNED",
            "PURGE_FAILED",
        }

    async def test_a_pod_that_does_not_own_the_index_is_refused_finally(self) -> None:
        hub = _Hub(CacheIndexMovedReply(success=False, error_code="INDEX_NOT_OWNED", error_message="not tools.enr"))
        with pytest.raises(CacheIndexRequestRefusedError) as refused:
            await report_cache_index_moved(
                hub, identity_token="tok", customer_id=_CUSTOMER, conversion="c", version="v"
            )  # type: ignore[arg-type]
        assert refused.value.error_code == "INDEX_NOT_OWNED"

    def test_a_customer_that_is_no_uuid_is_refused_before_it_is_sent(self) -> None:
        with pytest.raises(ValidationError):
            CacheIndexMovedRequest(
                identity_token=SecretStr("tok"), correlation_id=uuid7(), customer_id="14th", conversion="c", version="v"
            )
