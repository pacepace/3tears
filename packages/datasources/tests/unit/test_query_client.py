"""tests for :mod:`threetears.datasources.query_client`.

the client carries a FORWARDED TOKEN rather than a self-asserted principal: the
hub reads the caller off a signature it verifies in-process, so there is no
field on the request with which a caller could name somebody else. these tests
pin that shape on the wire, the client's refusal to send without a token, and its
translation of every reply -- rows, a refusal envelope, undecodable bytes, a
transport failure -- into either rows or one typed error carrying the hub's code.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import UUID, uuid7

import pytest
from pydantic import BaseModel, ValidationError

from threetears.datasources.query_client import (
    DEFAULT_QUERY_TIMEOUT_SECONDS,
    QUERY_STATEMENT_TIMEOUT_SECONDS,
    DatasourceQueryClient,
    DatasourceQueryError,
    DatasourceQueryRequest,
    DatasourceQueryResponse,
)
from threetears.nats.errors import RequestError, RequestTimeoutError
from threetears.nats.subjects import Subject, set_default_namespace

_NS = "3tears"
_TOKEN = "hub.minted.token"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace(_NS)


# parity-exempt: NatsClient subset for the query-client unit test; production NatsClient is the full canonical wrapper and the client reaches it through the typed request form alone
class _FakeNatsClient:
    """records the typed request the client issues and answers with a canned reply.

    :param reply: the response model to hand back, or ``None`` to raise
    :ptype reply: BaseModel | None
    :param raise_exc: the exception to raise instead of answering
    :ptype raise_exc: Exception | None
    """

    def __init__(self, reply: BaseModel | None = None, raise_exc: Exception | None = None) -> None:
        """bind the canned outcome.

        :param reply: the response model to hand back
        :ptype reply: BaseModel | None
        :param raise_exc: the exception to raise instead
        :ptype raise_exc: Exception | None
        """
        self._reply = reply
        self._raise_exc = raise_exc
        self.calls: list[dict[str, Any]] = []

    async def request(
        self,
        *,
        subject: Subject,
        message: BaseModel,
        response_type: type[BaseModel],
        timeout: timedelta,
    ) -> BaseModel:
        """record the call and answer.

        :param subject: the subject the client published to
        :ptype subject: Subject
        :param message: the typed request body
        :ptype message: BaseModel
        :param response_type: the model the client decodes into
        :ptype response_type: type[BaseModel]
        :param timeout: the per-call deadline
        :ptype timeout: timedelta
        :return: the canned reply
        :rtype: BaseModel
        :raises Exception: when the fake is configured to raise
        """
        self.calls.append(
            {"subject": subject, "message": message, "response_type": response_type, "timeout": timeout},
        )
        if self._raise_exc is not None:
            raise self._raise_exc
        assert self._reply is not None
        return self._reply


def _client(fake: _FakeNatsClient, token: str | None = _TOKEN) -> DatasourceQueryClient:
    """build a client over the fake with a constant token provider.

    :param fake: the fake NATS client
    :ptype fake: _FakeNatsClient
    :param token: what the provider returns
    :ptype token: str | None
    :return: the client under test
    :rtype: DatasourceQueryClient
    """
    return DatasourceQueryClient(fake, identity_token=lambda: token)  # type: ignore[arg-type]


def _rows(correlation_id: UUID, rows: list[dict[str, Any]]) -> DatasourceQueryResponse:
    """a success envelope carrying ``rows``.

    :param correlation_id: the id echoed back
    :ptype correlation_id: UUID
    :param rows: the rows to carry
    :ptype rows: list[dict[str, Any]]
    :return: the envelope
    :rtype: DatasourceQueryResponse
    """
    return DatasourceQueryResponse(success=True, rows=rows, row_count=len(rows), correlation_id=correlation_id)


class TestRequestShape:
    """the wire model refuses what a caller could use to name a principal."""

    def test_request_refuses_an_unknown_field(self) -> None:
        """an ``agent_id`` on the body is refused, not ignored.

        ignoring it would let a stale client send one beside a valid token, have
        it dropped, and be authorized as the token's principal with no error
        anywhere; refusing it makes the wire border say so.
        """
        with pytest.raises(ValidationError):
            DatasourceQueryRequest.model_validate(
                {
                    "correlation_id": str(uuid7()),
                    "identity_token": _TOKEN,
                    "query": "SELECT 1",
                    "agent_id": "somebody-else",
                },
            )

    def test_request_round_trips_its_fields(self) -> None:
        """every declared field survives serialization unchanged."""
        cid = uuid7()
        request = DatasourceQueryRequest(
            correlation_id=cid,
            identity_token=_TOKEN,
            user_identity_token="user.assertion",
            query="SELECT $1",
            params=[7],
        )
        parsed = DatasourceQueryRequest.model_validate_json(request.model_dump_json())
        assert parsed == request
        assert parsed.correlation_id == cid


class TestSend:
    """what the client puts on the bus."""

    @pytest.mark.asyncio
    async def test_publishes_on_the_datasource_query_subject(self) -> None:
        """the subject names the datasource through the canonical factory."""
        cid = uuid7()
        fake = _FakeNatsClient(reply=_rows(cid, []))
        await _client(fake).query("central-reporting", "SELECT 1", correlation_id=cid)

        assert len(fake.calls) == 1
        assert fake.calls[0]["subject"].path == f"{_NS}.datasource.central-reporting.query"
        assert fake.calls[0]["response_type"] is DatasourceQueryResponse

    @pytest.mark.asyncio
    async def test_forwards_the_current_token_and_the_query(self) -> None:
        """the body carries the provider's token verbatim, the sql and the params."""
        cid = uuid7()
        fake = _FakeNatsClient(reply=_rows(cid, []))
        await _client(fake).query(
            "central-reporting",
            "SELECT * FROM t WHERE id = $1",
            [42],
            user_identity_token="user.assertion",
            correlation_id=cid,
        )

        sent = fake.calls[0]["message"]
        assert isinstance(sent, DatasourceQueryRequest)
        assert sent.identity_token == _TOKEN
        assert sent.user_identity_token == "user.assertion"
        assert sent.query == "SELECT * FROM t WHERE id = $1"
        assert sent.params == [42]
        assert sent.correlation_id == cid

    @pytest.mark.asyncio
    async def test_reads_the_token_on_every_call(self) -> None:
        """the provider is consulted per call, so a re-minted token is forwarded."""
        tokens = iter(["first", "second"])
        fake = _FakeNatsClient(reply=_rows(uuid7(), []))
        client = DatasourceQueryClient(fake, identity_token=lambda: next(tokens))  # type: ignore[arg-type]

        await client.query("ds", "SELECT 1")
        await client.query("ds", "SELECT 1")

        assert [c["message"].identity_token for c in fake.calls] == ["first", "second"]

    @pytest.mark.asyncio
    async def test_mints_a_correlation_id_when_none_is_given(self) -> None:
        """a call with no correlation id still carries one the hub can echo."""
        fake = _FakeNatsClient(reply=_rows(uuid7(), []))
        await _client(fake).query("ds", "SELECT 1")
        assert isinstance(fake.calls[0]["message"].correlation_id, UUID)

    @pytest.mark.asyncio
    async def test_honours_the_configured_timeout(self) -> None:
        """the deadline handed to the bus is the one the client was built with."""
        fake = _FakeNatsClient(reply=_rows(uuid7(), []))
        client = DatasourceQueryClient(fake, identity_token=lambda: _TOKEN, timeout=12.5)  # type: ignore[arg-type]
        await client.query("ds", "SELECT 1")
        assert fake.calls[0]["timeout"] == timedelta(seconds=12.5)


class TestDeadlineOrdering:
    """the hub cancels a statement before the client stops listening."""

    def test_statement_timeout_sits_under_the_client_deadline(self) -> None:
        """a stuck warehouse must surface as the hub's typed refusal, not a transport fault.

        if the client gave up first its ``REQUEST_FAILED`` would steer a retry
        that stacks a second statement on the one still running; the margin is
        what the cancel and the refusal need to travel back.
        """
        assert QUERY_STATEMENT_TIMEOUT_SECONDS < DEFAULT_QUERY_TIMEOUT_SECONDS
        assert DEFAULT_QUERY_TIMEOUT_SECONDS - QUERY_STATEMENT_TIMEOUT_SECONDS >= 10

    @pytest.mark.asyncio
    async def test_default_deadline_is_the_documented_one(self) -> None:
        """a client built with no explicit timeout waits exactly the default."""
        fake = _FakeNatsClient(reply=_rows(uuid7(), []))
        await _client(fake).query("ds", "SELECT 1")
        assert fake.calls[0]["timeout"] == timedelta(seconds=DEFAULT_QUERY_TIMEOUT_SECONDS)


class TestRefusesToSendWithoutIdentity:
    """an unauthenticatable request is refused here, before the bus sees it."""

    @pytest.mark.asyncio
    async def test_empty_datasource_name_raises_before_any_publish(self) -> None:
        """an empty name composes no subject, and that is the client's error to name."""
        fake = _FakeNatsClient(reply=_rows(uuid7(), []))
        with pytest.raises(DatasourceQueryError) as exc_info:
            await _client(fake).query("", "SELECT 1")
        assert exc_info.value.error_code == "INVALID_DATASOURCE_NAME"
        assert fake.calls == []

    @pytest.mark.asyncio
    async def test_empty_token_raises_before_any_publish(self) -> None:
        """an empty token is refused by the hub exactly as a missing one is."""
        fake = _FakeNatsClient(reply=_rows(uuid7(), []))
        with pytest.raises(DatasourceQueryError) as exc_info:
            await _client(fake, token="").query("ds", "SELECT 1")
        assert exc_info.value.error_code == "NO_IDENTITY_TOKEN"
        assert fake.calls == []

    @pytest.mark.asyncio
    async def test_none_token_raises_before_any_publish(self) -> None:
        """a provider that has not been handed a token yet is the same refusal."""
        fake = _FakeNatsClient(reply=_rows(uuid7(), []))
        with pytest.raises(DatasourceQueryError) as exc_info:
            await _client(fake, token=None).query("ds", "SELECT 1")
        assert exc_info.value.error_code == "NO_IDENTITY_TOKEN"
        assert fake.calls == []


class TestReplies:
    """every reply becomes rows or one typed error carrying the hub's code."""

    @pytest.mark.asyncio
    async def test_success_returns_the_rows(self) -> None:
        """rows come back as plain dicts in the hub's order, and a full result says so."""
        rows = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        cid = uuid7()
        fake = _FakeNatsClient(reply=_rows(cid, rows))

        result = await _client(fake).query("ds", "SELECT id, name FROM t")

        assert result.rows == rows
        assert result.row_count == 2
        assert result.truncated is False
        assert result.correlation_id == cid

    @pytest.mark.asyncio
    async def test_a_cut_result_is_returned_flagged_never_as_the_whole(self) -> None:
        """the hub's row cap rides through: a caller deriving state from a full read
        can refuse the prefix instead of treating the missing rows as absent upstream."""
        rows = [{"id": index} for index in range(3)]
        reply = DatasourceQueryResponse(success=True, rows=rows, row_count=3, truncated=True, correlation_id=uuid7())
        fake = _FakeNatsClient(reply=reply)

        result = await _client(fake).query("ds", "SELECT id FROM t")

        assert result.rows == rows
        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_refusal_envelope_raises_with_the_code(self) -> None:
        """the hub's code rides on the exception, so a caller branches on it."""
        cid = uuid7()
        refusal = DatasourceQueryResponse(
            success=False,
            error_code="ACCESS_DENIED",
            error_message="datasource access denied: ds (datasource.read)",
            correlation_id=cid,
        )
        fake = _FakeNatsClient(reply=refusal)
        with pytest.raises(DatasourceQueryError) as exc_info:
            await _client(fake).query("ds", "SELECT 1", correlation_id=cid)
        assert exc_info.value.error_code == "ACCESS_DENIED"
        assert "datasource access denied" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_refusal_with_no_code_is_still_a_refusal(self) -> None:
        """a ``success=false`` envelope carrying no code cannot be read as rows."""
        fake = _FakeNatsClient(reply=DatasourceQueryResponse(success=False, correlation_id=uuid7()))
        with pytest.raises(DatasourceQueryError) as exc_info:
            await _client(fake).query("ds", "SELECT 1")
        assert exc_info.value.error_code == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_transport_failure_raises_with_a_transport_code(self) -> None:
        """a timeout or a bus fault is reported as such, never as a refusal."""
        fake = _FakeNatsClient(raise_exc=RequestTimeoutError("request timed out"))
        with pytest.raises(DatasourceQueryError) as exc_info:
            await _client(fake).query("ds", "SELECT 1")
        assert exc_info.value.error_code == "REQUEST_FAILED"
        assert isinstance(exc_info.value.__cause__, RequestError)

    @pytest.mark.asyncio
    async def test_undecodable_reply_raises_with_a_transport_code(self) -> None:
        """the wrapper reports a decode failure as a request error; so does the client."""
        fake = _FakeNatsClient(raise_exc=RequestError("response decode failed"))
        with pytest.raises(DatasourceQueryError) as exc_info:
            await _client(fake).query("ds", "SELECT 1")
        assert exc_info.value.error_code == "REQUEST_FAILED"
