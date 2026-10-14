"""tests for the client's ``list_own_exports``: a pod's own recorded exports, read to the last page.

The hub here is the contract in ``packages/datasources/README.md`` ("A pod's own exports"), kept in
memory: it answers for the pod its token names, pages by the core keyset, and deletes by destination.
A hub one release older is stood in for by that release's request shape, so the client's reading of
"this hub does not know the ask" is tested against what such a hub does with the real wire bytes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid7

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from threetears.core.pagination import Keyset, decode_cursor
from threetears.datasources.query_client import (
    EXPORT_LIST_UNSUPPORTED,
    DatasourceExportDeleteRequest,
    DatasourceExportListRequest,
    DatasourceExportListResult,
    DatasourceExportListUnsupportedError,
    DatasourceExportRecord,
    DatasourceExportRequest,
    DatasourceQueryClient,
    DatasourceQueryError,
    DatasourceQueryRequest,
    DatasourceQueryResponse,
    RelationFingerprintRequest,
)
from threetears.nats.errors import RequestError
from threetears.nats.subjects import Subject, Subjects, set_default_namespace

_POD = "pod-a-token"
_OTHER_POD = "pod-b-token"
_MADE = datetime(2026, 10, 13, 9, 0, tzinfo=UTC)
_BY_DESTINATION = Keyset(columns=("destination",), descending=False)


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace("3tears")


class _Release066Request(BaseModel):
    """the request as release 0.66.0 declares it: every ask but the export list, unknown fields refused."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: UUID
    identity_token: str
    user_identity_token: str | None = None
    query: str | None = None
    params: list[Any] = []
    fingerprint: dict[str, Any] | None = None
    export: dict[str, Any] | None = None
    export_delete: dict[str, Any] | None = None


def _record(destination: str) -> DatasourceExportRecord:
    return DatasourceExportRecord(
        destination=destination, object_prefix=f"exports/warehouse/{destination}/", date_created=_MADE
    )


# parity-exempt: NatsClient subset standing in for the hub's datasource responder; the client reaches NATS through the typed request form alone
class _FakeHub:
    """the hub's answer to the export asks, from records it keeps per verified pod.

    :param records: token -> the destinations recorded for the pod that token names
    :param page_size: the most exports one page carries
    :param knows_the_ask: False for a hub of release 0.66.0, which reads requests in that release's shape
    :param older_replica_from_page: the first page (0-based) answered by a replica of release 0.66.0,
        the ones before it by replicas that know the ask; for a hub part-way through a roll
    """

    def __init__(
        self,
        records: dict[str, list[str]] | None = None,
        *,
        page_size: int = 2,
        knows_the_ask: bool = True,
        older_replica_from_page: int | None = None,
    ) -> None:
        self.records = {token: sorted(destinations) for token, destinations in (records or {}).items()}
        self.page_size = page_size
        self.knows_the_ask = knows_the_ask
        self.older_replica_from_page = older_replica_from_page
        self.subjects: list[str] = []
        self.wire: list[dict[str, Any]] = []

    async def request(
        self, *, subject: Subject, message: BaseModel, response_type: type[BaseModel], timeout: timedelta
    ) -> BaseModel:
        del response_type, timeout
        payload = message.model_dump_json()
        self.subjects.append(subject.path)
        self.wire.append(json.loads(payload))
        older = not self.knows_the_ask or (
            self.older_replica_from_page is not None and len(self.wire) > self.older_replica_from_page
        )
        if older:
            try:
                _Release066Request.model_validate_json(payload)
            except ValidationError:
                # what the hub's responder answers a body its model refuses: no correlation id,
                # because it never read one
                return DatasourceQueryResponse(
                    success=False, error_code="MALFORMED_REQUEST", error_message="malformed datasource query request"
                )
        request = DatasourceQueryRequest.model_validate_json(payload)
        own = self.records.setdefault(request.identity_token.get_secret_value(), [])
        if request.export_delete is not None:
            own.remove(request.export_delete.destination)
            return DatasourceQueryResponse(success=True, correlation_id=request.correlation_id, export_deleted=1)
        assert request.export_list is not None
        after = decode_cursor(request.export_list.cursor)[0] if request.export_list.cursor is not None else ""
        rows = [destination for destination in own if destination > after][: self.page_size + 1]
        page = _BY_DESTINATION.page(rows, self.page_size, key_of=lambda destination: (destination,))
        return DatasourceQueryResponse(
            success=True,
            correlation_id=request.correlation_id,
            export_list=DatasourceExportListResult(
                exports=[_record(destination) for destination in page.items], next_cursor=page.next_cursor
            ),
        )


# parity-exempt: NatsClient subset answering each request with the next scripted reply, or raising in its place
class _FakeNats:
    def __init__(self, *replies: DatasourceQueryResponse | Exception) -> None:
        self._replies = list(replies)
        self.messages: list[BaseModel] = []

    async def request(
        self, *, subject: Subject, message: BaseModel, response_type: type[BaseModel], timeout: timedelta
    ) -> BaseModel:
        del subject, response_type, timeout
        self.messages.append(message)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _client(nats: Any, token: str | None = _POD) -> DatasourceQueryClient:
    return DatasourceQueryClient(nats, identity_token=lambda: token)


def _page(*destinations: str, next_cursor: str | None = None) -> DatasourceQueryResponse:
    return DatasourceQueryResponse(
        success=True,
        correlation_id=uuid7(),
        export_list=DatasourceExportListResult(exports=[_record(d) for d in destinations], next_cursor=next_cursor),
    )


class TestTheWire:
    def test_a_list_request_round_trips_and_names_no_pod(self) -> None:
        request = DatasourceQueryRequest(
            correlation_id=uuid7(), identity_token="t", export_list=DatasourceExportListRequest(cursor="abc")
        )
        wire = json.loads(request.model_dump_json())
        assert wire["export_list"] == {"cursor": "abc"}
        assert DatasourceQueryRequest.model_validate_json(request.model_dump_json()).export_list == request.export_list
        # the pod asked about is the verified caller: the ask has no field to name one, and refuses one
        assert set(DatasourceExportListRequest.model_fields) == {"cursor"}
        for field in ("tool_pod_id", "pod_id", "principal_id", "agent_id"):
            with pytest.raises(ValidationError):
                DatasourceExportListRequest.model_validate({field: str(uuid7())})

    def test_every_other_ask_is_still_what_an_older_hub_reads(self) -> None:
        others = [
            DatasourceQueryRequest(correlation_id=uuid7(), identity_token="t", query="SELECT 1"),
            DatasourceQueryRequest(
                correlation_id=uuid7(),
                identity_token="t",
                fingerprint=RelationFingerprintRequest(relation="s.t", key_columns=["a"]),
            ),
            DatasourceQueryRequest(
                correlation_id=uuid7(),
                identity_token="t",
                export=DatasourceExportRequest(select="SELECT 1", destination="enr/x"),
            ),
            DatasourceQueryRequest(
                correlation_id=uuid7(),
                identity_token="t",
                export_delete=DatasourceExportDeleteRequest(destination="enr/x"),
            ),
        ]
        for request in others:
            assert "export_list" not in json.loads(request.model_dump_json())
            _Release066Request.model_validate_json(request.model_dump_json())

    def test_a_list_is_one_ask_and_binds_nothing(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            DatasourceQueryRequest(
                correlation_id=uuid7(), identity_token="t", query="SELECT 1", export_list=DatasourceExportListRequest()
            )
        with pytest.raises(ValidationError, match="binds no parameters"):
            DatasourceQueryRequest(
                correlation_id=uuid7(), identity_token="t", params=[1], export_list=DatasourceExportListRequest()
            )

    def test_an_older_client_reads_a_reply_that_carries_a_page(self) -> None:
        # the response model ignores what it does not know, so a hub that answers the new ask
        # breaks no reader of the old ones
        reply = _page("enr/a").model_dump_json()
        assert DatasourceQueryResponse.model_validate_json(reply).export_list is not None
        assert "export_list" in json.loads(DatasourceQueryResponse(success=True).model_dump_json())


class TestListOwnExports:
    @pytest.mark.asyncio
    async def test_every_page_is_read_to_the_end_in_order(self) -> None:
        destinations = [f"enr/r{i}" for i in range(5)]
        hub = _FakeHub({_POD: destinations, _OTHER_POD: ["enr/theirs"]})
        trace = uuid7()
        exports = await _client(hub).list_own_exports("warehouse", correlation_id=trace)
        assert exports == [_record(destination) for destination in destinations]
        assert len(hub.wire) == 3
        assert hub.wire[0]["export_list"] == {"cursor": None}
        assert [decode_cursor(sent["export_list"]["cursor"]) for sent in hub.wire[1:]] == [["enr/r1"], ["enr/r3"]]
        # the same subject as a query, and one trace for the whole listing
        assert set(hub.subjects) == {Subjects.datasource_query("warehouse").path}
        assert {sent["correlation_id"] for sent in hub.wire} == {str(trace)}
        # nothing but the token says which pod: the request holds no other name for it
        assert set(hub.wire[0]) == {"correlation_id", "identity_token", "user_identity_token", "query", "params"} | {
            "fingerprint",
            "export_list",
        }

    @pytest.mark.asyncio
    async def test_a_pod_with_no_exports_gets_an_empty_list_in_one_ask(self) -> None:
        hub = _FakeHub({_OTHER_POD: ["enr/theirs"]})
        assert await _client(hub).list_own_exports("warehouse") == []
        assert len(hub.wire) == 1

    @pytest.mark.asyncio
    async def test_each_listed_export_is_deleted_by_its_destination_and_the_next_list_is_empty(self) -> None:
        hub = _FakeHub({_POD: ["enr/r1/VA", "enr/r1/NC", "enr/r2"], _OTHER_POD: ["enr/theirs"]})
        client = _client(hub)
        for export in await client.list_own_exports("warehouse"):
            assert await client.delete_export("warehouse", destination=export.destination) == 1
        assert await client.list_own_exports("warehouse") == []
        assert hub.records[_OTHER_POD] == ["enr/theirs"]

    @pytest.mark.asyncio
    async def test_a_hub_that_predates_the_ask_is_its_own_outcome_not_an_empty_list(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        hub = _FakeHub({_POD: ["enr/left-behind"]}, knows_the_ask=False)
        with caplog.at_level("WARNING"), pytest.raises(DatasourceExportListUnsupportedError) as raised:
            await _client(hub).list_own_exports("warehouse")
        assert isinstance(raised.value, DatasourceQueryError)
        assert raised.value.error_code == EXPORT_LIST_UNSUPPORTED
        assert raised.value.datasource_name == "warehouse"
        assert "does not know the export list ask" in caplog.text
        assert len(hub.wire) == 1

    @pytest.mark.asyncio
    async def test_an_older_hub_still_answers_the_delete_this_client_sends(self) -> None:
        hub = _FakeHub({_POD: ["enr/x"]}, knows_the_ask=False)
        assert await _client(hub).delete_export("warehouse", destination="enr/x") == 1

    @pytest.mark.asyncio
    async def test_an_older_replica_answering_a_later_page_lists_nothing(self) -> None:
        # part-way through a hub's roll: no prefix of the list is handed back as the list
        hub = _FakeHub({_POD: [f"enr/r{i}" for i in range(5)]}, older_replica_from_page=1)
        with pytest.raises(DatasourceExportListUnsupportedError):
            await _client(hub).list_own_exports("warehouse")
        assert len(hub.wire) == 2

    @pytest.mark.asyncio
    async def test_a_hub_that_read_the_ask_but_has_no_answer_for_it_is_unsupported_too(self) -> None:
        # a hub on this release's models with no handler yet answers the ask as one it cannot place
        refusal = DatasourceQueryResponse(
            success=False,
            error_code="MALFORMED_REQUEST",
            error_message="a datasource request carries exactly one of query or fingerprint",
            correlation_id=uuid7(),
        )
        with pytest.raises(DatasourceExportListUnsupportedError):
            await _client(_FakeNats(refusal)).list_own_exports("warehouse")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code", ["EXPORT_NOT_GRANTED", "IDENTITY_REFUSED", "DATASOURCE_NOT_FOUND", "ACCESS_DENIED"]
    )
    async def test_any_other_refusal_carries_the_hubs_code_and_is_not_unsupported(self, code: str) -> None:
        fake = _FakeNats(DatasourceQueryResponse(success=False, error_code=code, error_message="no"))
        with pytest.raises(DatasourceQueryError) as raised:
            await _client(fake).list_own_exports("warehouse")
        assert raised.value.error_code == code
        assert not isinstance(raised.value, DatasourceExportListUnsupportedError)

    @pytest.mark.asyncio
    async def test_a_repeated_cursor_is_refused_rather_than_followed_or_cut_short(self) -> None:
        fake = _FakeNats(
            _page("enr/a", next_cursor="one"),
            _page("enr/b", next_cursor="two"),
            _page("enr/c", next_cursor="one"),
            _page("enr/d"),
        )
        with pytest.raises(DatasourceQueryError, match="repeated a cursor") as raised:
            await _client(fake).list_own_exports("warehouse")
        assert raised.value.error_code == "MALFORMED_RESPONSE"
        assert len(fake.messages) == 3

    @pytest.mark.asyncio
    async def test_a_success_with_no_page_is_not_taken_as_no_exports(self) -> None:
        fake = _FakeNats(DatasourceQueryResponse(success=True, correlation_id=uuid7()))
        with pytest.raises(DatasourceQueryError) as raised:
            await _client(fake).list_own_exports("warehouse")
        assert raised.value.error_code == "MALFORMED_RESPONSE"

    @pytest.mark.asyncio
    async def test_a_bus_failure_on_a_later_page_fails_the_listing(self) -> None:
        fake = _FakeNats(_page("enr/a", next_cursor="one"), RequestError("bus down"))
        with pytest.raises(DatasourceQueryError) as raised:
            await _client(fake).list_own_exports("warehouse")
        assert raised.value.error_code == "REQUEST_FAILED"

    @pytest.mark.asyncio
    async def test_nothing_is_sent_without_a_datasource_or_a_token(self) -> None:
        fake = _FakeNats()
        with pytest.raises(DatasourceQueryError) as unnamed:
            await _client(fake).list_own_exports("")
        assert unnamed.value.error_code == "INVALID_DATASOURCE_NAME"
        with pytest.raises(DatasourceQueryError) as untokened:
            await _client(fake, token=None).list_own_exports("warehouse")
        assert untokened.value.error_code == "NO_IDENTITY_TOKEN"
        assert fake.messages == []
