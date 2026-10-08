"""the tool pod -> hub contract for a pod's own Object Store: declare it, retire what is no longer served.

every client test drives the real decode: a reply JSON-encoded as the hub sends it, handed back from a
scripted ``request_raw``.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from uuid import uuid7

import pytest

from threetears.nats import RequestTimeoutError, Subject, set_default_namespace
from threetears.nats.object_store_requests import (
    MAX_RETIRED_OBJECTS,
    OBJECT_STORE_REQUEST_ERROR_CODES,
    DeclaredObjectStore,
    ObjectStoreDeclareRequest,
    ObjectStoreNotDeclaredError,
    ObjectStoreRequestRefusedError,
    ObjectStoreRequestUnavailableError,
    RetiredObjects,
    bind_pod_object_store,
    declare_pod_object_store,
    retire_pod_objects,
)

_TOKEN = "hub-minted-identity-token"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """bind the subject namespace a connected client would have set."""
    set_default_namespace("3tears")


class _ScriptedRequests:
    """answers ``request_raw`` with scripted replies, recording each request it was sent."""

    def __init__(self, *replies: dict[str, Any] | Exception, echo: bool = True) -> None:
        self._replies = list(replies)
        self._echo = echo
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        del timeout
        self.sent.append((subject.path, json.loads(payload)))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if self._echo:
            reply = {"correlation_id": self.sent[-1][1]["correlation_id"], **reply}
        return json.dumps(reply).encode()


_DECLARED = {
    "success": True,
    "bucket": "3tears-tool_pod-x-objects",
    "pointers_bucket": "3tears-tool_pod-x-pointers",
    "max_bytes": 1024,
}


class TestDeclare:
    async def test_the_ask_goes_to_the_hub_with_the_token_and_names_no_bucket(self) -> None:
        nc = _ScriptedRequests(_DECLARED)
        declared = await declare_pod_object_store(nc, identity_token=_TOKEN)  # type: ignore[arg-type]
        assert declared == DeclaredObjectStore(
            bucket="3tears-tool_pod-x-objects", pointers_bucket="3tears-tool_pod-x-pointers", max_bytes=1024
        )
        subject, body = nc.sent[0]
        assert subject == "3tears.hub.object_store.declare"
        assert body["identity_token"] == _TOKEN
        assert set(body) == {"identity_token", "correlation_id"}

    def test_the_token_is_redacted_off_the_wire(self) -> None:
        request = ObjectStoreDeclareRequest(identity_token=_TOKEN, correlation_id=uuid7())
        assert _TOKEN not in repr(request)
        assert _TOKEN in request.model_dump_json()

    async def test_no_token_is_unavailable_without_asking(self) -> None:
        nc = _ScriptedRequests()
        with pytest.raises(ObjectStoreRequestUnavailableError):
            await declare_pod_object_store(nc, identity_token="")  # type: ignore[arg-type]
        assert nc.sent == []

    async def test_a_refusal_carries_the_hubs_code(self) -> None:
        nc = _ScriptedRequests({"success": False, "error_code": "IDENTITY_REFUSED", "error_message": "no"})
        with pytest.raises(ObjectStoreRequestRefusedError) as raised:
            await declare_pod_object_store(nc, identity_token=_TOKEN)  # type: ignore[arg-type]
        assert raised.value.error_code == "IDENTITY_REFUSED"

    async def test_a_hub_failure_is_retryable(self) -> None:
        nc = _ScriptedRequests({"success": False, "error_code": "DECLARE_FAILED", "error_message": "nats"})
        with pytest.raises(ObjectStoreRequestUnavailableError):
            await declare_pod_object_store(nc, identity_token=_TOKEN)  # type: ignore[arg-type]

    async def test_a_transport_failure_is_retryable(self) -> None:
        nc = _ScriptedRequests(RequestTimeoutError("no answer"))
        with pytest.raises(ObjectStoreRequestUnavailableError):
            await declare_pod_object_store(nc, identity_token=_TOKEN)  # type: ignore[arg-type]

    async def test_a_reply_to_another_request_is_not_trusted(self) -> None:
        nc = _ScriptedRequests({**_DECLARED, "correlation_id": str(uuid7())}, echo=False)
        with pytest.raises(ObjectStoreRequestUnavailableError):
            await declare_pod_object_store(nc, identity_token=_TOKEN)  # type: ignore[arg-type]

    async def test_a_success_without_its_buckets_is_not_trusted(self) -> None:
        nc = _ScriptedRequests({"success": True})
        with pytest.raises(ObjectStoreRequestUnavailableError):
            await declare_pod_object_store(nc, identity_token=_TOKEN)  # type: ignore[arg-type]


class TestRetire:
    async def test_the_ask_names_the_objects(self) -> None:
        nc = _ScriptedRequests({"success": True, "retired": 2, "absent": 1, "orphan_chunks": 0})
        retired = await retire_pod_objects(  # type: ignore[arg-type]
            nc, identity_token=_TOKEN, names=["a/TX/1", "a/TX/2", "a/CA/1"]
        )
        assert retired == RetiredObjects(retired=2, absent=1, orphan_chunks=0)
        subject, body = nc.sent[0]
        assert subject == "3tears.hub.object_store.retire"
        assert body["names"] == ["a/TX/1", "a/TX/2", "a/CA/1"]

    @pytest.mark.parametrize(
        "names",
        [[], ["has space"], ["a*"], [f"n{i}" for i in range(MAX_RETIRED_OBJECTS + 1)]],
    )
    async def test_an_invalid_ask_is_refused_without_asking(self, names: list[str]) -> None:
        nc = _ScriptedRequests()
        with pytest.raises(ObjectStoreRequestRefusedError) as raised:
            await retire_pod_objects(nc, identity_token=_TOKEN, names=names)  # type: ignore[arg-type]
        assert raised.value.error_code == "INVALID_REQUEST"
        assert nc.sent == []

    async def test_a_hub_failure_is_retryable(self) -> None:
        nc = _ScriptedRequests({"success": False, "error_code": "RETIRE_FAILED", "error_message": "nats"})
        with pytest.raises(ObjectStoreRequestUnavailableError):
            await retire_pod_objects(nc, identity_token=_TOKEN, names=["a"])  # type: ignore[arg-type]


class TestRefusalsThePodActsOn:
    async def test_a_pod_not_opted_in_is_refused_for_good(self) -> None:
        nc = _ScriptedRequests({"success": False, "error_code": "OBJECT_STORE_NOT_GRANTED", "error_message": "no"})
        with pytest.raises(ObjectStoreRequestRefusedError) as raised:
            await declare_pod_object_store(nc, identity_token=_TOKEN)  # type: ignore[arg-type]
        assert raised.value.error_code == "OBJECT_STORE_NOT_GRANTED"

    async def test_an_exhausted_budget_is_refused_for_good(self) -> None:
        nc = _ScriptedRequests(
            {"success": False, "error_code": "OBJECT_STORE_BUDGET_EXHAUSTED", "error_message": "full"}
        )
        with pytest.raises(ObjectStoreRequestRefusedError) as raised:
            await declare_pod_object_store(nc, identity_token=_TOKEN)  # type: ignore[arg-type]
        assert raised.value.error_code == "OBJECT_STORE_BUDGET_EXHAUSTED"

    async def test_a_retire_against_a_bucket_not_declared_says_to_declare_again(self) -> None:
        nc = _ScriptedRequests({"success": False, "error_code": "OBJECT_STORE_NOT_DECLARED", "error_message": "gone"})
        with pytest.raises(ObjectStoreNotDeclaredError):
            await retire_pod_objects(nc, identity_token=_TOKEN, names=["a"])  # type: ignore[arg-type]
        assert issubclass(ObjectStoreNotDeclaredError, ObjectStoreRequestRefusedError)


def test_the_vocabulary_is_the_one_the_hub_answers() -> None:
    assert (
        frozenset(
            {
                "INVALID_REQUEST",
                "IDENTITY_REFUSED",
                "OBJECT_STORE_NOT_GRANTED",
                "OBJECT_STORE_BUDGET_EXHAUSTED",
                "OBJECT_STORE_NOT_DECLARED",
                "DECLARE_FAILED",
                "RETIRE_FAILED",
            }
        )
        == OBJECT_STORE_REQUEST_ERROR_CODES
    )


class _BindingClient(_ScriptedRequests):
    """a scripted client that also binds buckets, recording what it was asked to bind."""

    namespace = "3tears"

    def __init__(self, *replies: dict[str, Any] | Exception) -> None:
        super().__init__(*replies)
        self.bound: list[tuple[str, dict[str, Any]]] = []

    async def object_store(self, *, name: str, prefix_namespace: bool = True) -> str:
        self.bound.append(("object_store", {"name": name, "prefix_namespace": prefix_namespace}))
        return "store"

    async def kv_bucket(self, *, name: str, create_if_missing: bool = True) -> str:
        self.bound.append(("kv_bucket", {"name": name, "create_if_missing": create_if_missing}))
        return "pointers"


class TestBind:
    async def test_a_pod_binds_the_buckets_the_hub_declared_and_creates_none(self) -> None:
        nc = _BindingClient(_DECLARED)
        bound = await bind_pod_object_store(nc, identity_token=lambda: _TOKEN)  # type: ignore[arg-type]
        assert (bound.store, bound.pointers) == ("store", "pointers")
        assert nc.bound == [
            ("object_store", {"name": "3tears-tool_pod-x-objects", "prefix_namespace": False}),
            ("kv_bucket", {"name": "tool_pod-x-pointers", "create_if_missing": False}),
        ]

    async def test_a_pointer_bucket_outside_the_namespace_is_refused(self) -> None:
        nc = _BindingClient({**_DECLARED, "pointers_bucket": "other-tool_pod-x-pointers"})
        with pytest.raises(ObjectStoreRequestUnavailableError, match="outside"):
            await bind_pod_object_store(nc, identity_token=lambda: _TOKEN)  # type: ignore[arg-type]

    async def test_a_retire_after_nats_lost_the_bucket_declares_it_and_asks_again(self) -> None:
        nc = _BindingClient(
            _DECLARED,
            {"success": False, "error_code": "OBJECT_STORE_NOT_DECLARED", "error_message": "gone"},
            _DECLARED,
            {"success": True, "retired": 1, "absent": 0, "orphan_chunks": 0},
        )
        tokens = iter(["t1", "t2", "t3", "t4"])
        bound = await bind_pod_object_store(nc, identity_token=lambda: next(tokens))  # type: ignore[arg-type]

        retired = await bound.retire(["enr/VA/3/t.abc"])

        assert retired == RetiredObjects(retired=1, absent=0, orphan_chunks=0)
        assert [path.rsplit(".", 1)[-1] for path, _ in nc.sent] == ["declare", "retire", "declare", "retire"]
        assert [body["identity_token"] for _, body in nc.sent] == ["t1", "t2", "t3", "t4"], "a stale token was sent"
