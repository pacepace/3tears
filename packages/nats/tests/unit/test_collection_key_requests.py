"""a pod's retired collection keys are purged by the hub, under the pod's verified scope and nowhere else."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from threetears.nats.collection_key_requests import (
    CollectionKeysPurgeReply,
    CollectionKeysPurgeRequest,
    CollectionKeysRequestRefusedError,
    CollectionKeysRequestUnavailableError,
    purge_pod_collection_keys,
    purge_scoped_keys,
)
from threetears.nats.errors import RequestError, RequestTimeoutError
from threetears.nats.subjects import Subject, Subjects, set_default_namespace


class _JetStream:
    """records each filtered purge."""

    def __init__(self) -> None:
        self.purged: list[tuple[str, str]] = []

    async def purge_stream(self, name: str, *, subject: str) -> bool:
        self.purged.append((name, subject))
        return True


class _Hub:
    """answers a purge request the way a hub does, or not at all."""

    def __init__(
        self, reply: CollectionKeysPurgeReply | None = None, error: Exception | None = None, *, echo: bool = True
    ) -> None:
        self.reply = reply
        self.error = error
        self.echo = echo
        self.asked: list[CollectionKeysPurgeRequest] = []
        self.subjects: list[str] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        del timeout
        self.subjects.append(subject.path)
        request = CollectionKeysPurgeRequest.model_validate_json(payload)
        self.asked.append(request)
        if self.error is not None:
            raise self.error
        assert self.reply is not None
        reply = self.reply.model_copy(update={"correlation_id": request.correlation_id}) if self.echo else self.reply
        return reply.model_dump_json().encode()


class TestTheHubPurgesOnlyUnderTheVerifiedScope:
    async def test_each_key_is_its_one_exact_subject_below_the_scope(self) -> None:
        js = _JetStream()
        purged = await purge_scoped_keys(
            js, bucket="aibots-collections", scope="tool_pod-abc", keys=["enr_answers.v3_1f", "enr_answers_index.v3.1"]
        )
        assert purged == 2
        assert js.purged == [
            ("KV_aibots-collections", "$KV.aibots-collections.tool_pod-abc.enr_answers.v3_1f"),
            ("KV_aibots-collections", "$KV.aibots-collections.tool_pod-abc.enr_answers_index.v3.1"),
        ]

    async def test_a_key_naming_another_scope_still_lands_under_the_callers(self) -> None:
        js = _JetStream()
        await purge_scoped_keys(js, bucket="b", scope="tool_pod-mine", keys=["tool_pod-other.enr_answers.v3_1f"])
        assert js.purged == [("KV_b", "$KV.b.tool_pod-mine.tool_pod-other.enr_answers.v3_1f")]

    @pytest.mark.parametrize(
        "key",
        [">", "*", "enr_answers.>", "enr_answers.*", "..tool_pod-other", ".enr", "enr.", "enr..x", "a b", ""],
    )
    async def test_a_key_that_could_leave_the_scope_purges_nothing(self, key: str) -> None:
        js = _JetStream()
        with pytest.raises(ValidationError):
            await purge_scoped_keys(js, bucket="b", scope="tool_pod-mine", keys=[key])
        assert js.purged == []


class TestThePodsAsk:
    async def test_the_hub_purges_and_says_how_many(self) -> None:
        hub = _Hub(CollectionKeysPurgeReply(success=True, purged=2))
        purged = await purge_pod_collection_keys(hub, identity_token="tok", keys=["t.a_1", "t.a_2"])  # type: ignore[arg-type]
        assert purged == 2
        assert hub.asked[0].keys == ["t.a_1", "t.a_2"]
        assert hub.subjects == [Subjects.hub_collection_keys_purge().path]

    @pytest.mark.parametrize("error", [RequestTimeoutError("no answer"), RequestError("no responders")])
    async def test_a_hub_that_does_not_know_the_request_is_unavailable(self, error: Exception) -> None:
        with pytest.raises(CollectionKeysRequestUnavailableError):
            await purge_pod_collection_keys(_Hub(error=error), identity_token="tok", keys=["t.a_1"])  # type: ignore[arg-type]

    async def test_a_refusal_is_a_refusal(self) -> None:
        hub = _Hub(CollectionKeysPurgeReply(success=False, error_code="IDENTITY_REFUSED", error_message="who?"))
        with pytest.raises(CollectionKeysRequestRefusedError, match="IDENTITY_REFUSED"):
            await purge_pod_collection_keys(hub, identity_token="tok", keys=["t.a_1"])  # type: ignore[arg-type]

    async def test_a_success_with_no_correlation_id_is_not_taken_as_this_requests(self) -> None:
        """the one correlation rule every hub ask keeps: only a refusal may come back without the id."""
        hub = _Hub(CollectionKeysPurgeReply(success=True, purged=2), echo=False)
        with pytest.raises(CollectionKeysRequestUnavailableError, match="correlation_id=None"):
            await purge_pod_collection_keys(hub, identity_token="tok", keys=["t.a_1"])  # type: ignore[arg-type]

    async def test_a_success_under_another_requests_id_is_not_this_ones(self) -> None:
        hub = _Hub(CollectionKeysPurgeReply(success=True, purged=2, correlation_id=uuid4()), echo=False)
        with pytest.raises(CollectionKeysRequestUnavailableError, match="correlation_id"):
            await purge_pod_collection_keys(hub, identity_token="tok", keys=["t.a_1"])  # type: ignore[arg-type]

    async def test_a_refusal_with_no_correlation_id_is_this_requests(self) -> None:
        """a hub that could not decode the body had no id to echo; its refusal is final, not a stray."""
        hub = _Hub(CollectionKeysPurgeReply(success=False, error_code="INVALID_REQUEST", error_message="?"), echo=False)
        with pytest.raises(CollectionKeysRequestRefusedError, match="INVALID_REQUEST"):
            await purge_pod_collection_keys(hub, identity_token="tok", keys=["t.a_1"])  # type: ignore[arg-type]

    async def test_a_hub_side_failure_is_unavailable_not_refused(self) -> None:
        hub = _Hub(CollectionKeysPurgeReply(success=False, error_code="PURGE_FAILED", error_message="js down"))
        with pytest.raises(CollectionKeysRequestUnavailableError, match="PURGE_FAILED"):
            await purge_pod_collection_keys(hub, identity_token="tok", keys=["t.a_1"])  # type: ignore[arg-type]

    async def test_the_token_never_shows_outside_the_wire(self) -> None:
        request = CollectionKeysPurgeRequest(identity_token="secret-token", correlation_id=uuid4(), keys=["t.k"])  # type: ignore[arg-type]
        assert "secret-token" not in repr(request)
        assert "secret-token" in request.model_dump_json()


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("3tears")
