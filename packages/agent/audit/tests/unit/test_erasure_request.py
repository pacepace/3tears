"""the agent -> hub contract for anonymizing the audit rows an agent published.

the survey engine erases a respondent; the hub holds the audit rows the survey's agent
published about them. the pod asks, the hub answers, and neither owns the contract, so
it lives here beside the envelope: the request, the reply, the subject, and the client a
pod calls. every client test drives the real decode -- a reply JSON-encoded as the hub
sends it, handed back from a scripted ``request_raw``.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid7

import pytest
from pydantic import ValidationError

from threetears.agent.audit import (
    AUDIT_ANONYMIZE_ERROR_CODES,
    MAX_ANONYMIZE_ACTORS,
    AuditAnonymization,
    AuditAnonymizeError,
    AuditAnonymizeRefusedError,
    AuditAnonymizeReply,
    AuditAnonymizeRequest,
    AuditAnonymizeUnavailableError,
    request_audit_anonymization,
)
from threetears.nats import RequestTimeoutError, Subject, Subjects, set_default_namespace

_AGENT = UUID("0190aaaa-0000-7000-8000-000000000001")
_OTHER_AGENT = UUID("0190bbbb-0000-7000-8000-000000000002")
_TOKEN = "hub-minted-identity-token"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """bind the subject namespace a connected client would have set."""
    set_default_namespace("3tears")


class _ScriptedRequests:
    """answers ``request_raw`` with scripted reply bytes, recording every request it was sent.

    the client under test needs only this one method of the NATS client; a reply can also
    be an exception, raised as the transport would raise it.
    """

    def __init__(self, *replies: dict[str, Any] | Exception) -> None:
        """
        :param replies: reply dicts (JSON-encoded on the way out) or exceptions, in order
        :ptype replies: dict[str, Any] | Exception
        """
        self._replies = list(replies)
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        """
        :param subject: request subject
        :ptype subject: Subject
        :param payload: the encoded request
        :ptype payload: bytes
        :param timeout: reply deadline
        :ptype timeout: timedelta
        :return: the next scripted reply
        :rtype: bytes
        """
        del timeout
        self.sent.append((subject.path, json.loads(payload)))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        # the hub echoes the correlation id it was sent.
        reply = {**reply, "correlation_id": self.sent[-1][1]["correlation_id"]}
        return json.dumps(reply).encode()


def _success(rows_matched: int, rows_changed: int, *, agent_id: UUID = _AGENT) -> dict[str, Any]:
    """
    :param rows_matched: rows the hub matched
    :ptype rows_matched: int
    :param rows_changed: rows whose content it changed
    :ptype rows_changed: int
    :param agent_id: the verified agent it echoes
    :ptype agent_id: UUID
    :return: a success reply
    :rtype: dict[str, Any]
    """
    return {"success": True, "agent_id": str(agent_id), "rows_matched": rows_matched, "rows_changed": rows_changed}


async def _call(nats: _ScriptedRequests, actors: list[UUID], **kwargs: Any) -> AuditAnonymization:
    """
    :param nats: the scripted transport
    :ptype nats: _ScriptedRequests
    :param actors: the actor ids to anonymize
    :ptype actors: list[UUID]
    :param kwargs: overrides for identity_token / agent_id
    :ptype kwargs: Any
    :return: the result
    :rtype: AuditAnonymization
    """
    arguments: dict[str, Any] = {"identity_token": _TOKEN, "agent_id": _AGENT, "actor_user_ids": actors}
    arguments.update(kwargs)
    return await request_audit_anonymization(nats, **arguments)  # type: ignore[arg-type]


class TestTheModels:
    """the wire models are the contract; both round-trip and both refuse what is not in it."""

    def test_the_request_round_trips(self) -> None:
        """every field survives the JSON border as its own type."""
        request = AuditAnonymizeRequest(
            identity_token=_TOKEN, correlation_id=uuid7(), agent_id=_AGENT, actor_user_ids=[uuid7(), uuid7()]
        )

        assert AuditAnonymizeRequest.model_validate_json(request.model_dump_json()) == request

    @pytest.mark.parametrize("count", [0, MAX_ANONYMIZE_ACTORS + 1])
    def test_the_actor_list_is_bounded(self, count: int) -> None:
        """an empty request asks nothing; an unbounded one is refused before it is sent.

        :param count: the refused list length
        :ptype count: int
        """
        with pytest.raises(ValidationError):
            AuditAnonymizeRequest(
                identity_token=_TOKEN,
                correlation_id=uuid7(),
                agent_id=_AGENT,
                actor_user_ids=[uuid7() for _ in range(count)],
            )

    def test_a_field_outside_the_contract_is_refused(self) -> None:
        """a caller cannot smuggle a ``customer_id`` or a second agent into the request."""
        with pytest.raises(ValidationError):
            AuditAnonymizeRequest.model_validate(
                {
                    "identity_token": _TOKEN,
                    "correlation_id": str(uuid7()),
                    "agent_id": str(_AGENT),
                    "actor_user_ids": [str(uuid7())],
                    "customer_id": str(uuid7()),
                }
            )

    def test_the_reply_round_trips_both_shapes(self) -> None:
        """a success and a refusal decode through the one reply model."""
        success = AuditAnonymizeReply(
            success=True, correlation_id=uuid7(), agent_id=_AGENT, rows_matched=3, rows_changed=2
        )
        refusal = AuditAnonymizeReply(success=False, error_code="AGENT_MISMATCH", error_message="not yours")

        assert AuditAnonymizeReply.model_validate_json(success.model_dump_json()) == success
        assert AuditAnonymizeReply.model_validate_json(refusal.model_dump_json()) == refusal

    def test_the_error_codes_are_named(self) -> None:
        """the vocabulary a responder answers with and a caller branches on."""
        assert AUDIT_ANONYMIZE_ERROR_CODES == frozenset(
            {"INVALID_REQUEST", "IDENTITY_UNVERIFIED", "AGENT_MISMATCH", "ANONYMIZE_FAILED"}
        )


class TestTheClient:
    """what a pod calls: one request per bounded batch, decoded, refusals and timeouts typed."""

    async def test_a_success_returns_the_counts(self) -> None:
        """the hub's counts come back, and the request carried the token, agent and actors."""
        actors = [uuid7(), uuid7()]
        nats = _ScriptedRequests(_success(5, 4))

        result = await _call(nats, actors)

        assert result == AuditAnonymization(rows_matched=5, rows_changed=4)
        ((subject, sent),) = nats.sent
        assert subject == Subjects.hub_audit_anonymize().path
        assert sent["identity_token"] == _TOKEN
        assert sent["agent_id"] == str(_AGENT)
        assert sent["actor_user_ids"] == [str(actor) for actor in actors]

    async def test_a_refusal_raises_the_typed_refusal_with_its_code(self) -> None:
        """a hub refusal is not retryable and says why."""
        nats = _ScriptedRequests({"success": False, "error_code": "AGENT_MISMATCH", "error_message": "not yours"})

        with pytest.raises(AuditAnonymizeRefusedError) as refused:
            await _call(nats, [uuid7()])

        assert refused.value.error_code == "AGENT_MISMATCH"
        assert isinstance(refused.value, AuditAnonymizeError)

    async def test_a_timeout_raises_unavailable(self) -> None:
        """a hub that does not answer is retryable, and distinct from a refusal."""
        nats = _ScriptedRequests(RequestTimeoutError("request timed out"))

        with pytest.raises(AuditAnonymizeUnavailableError):
            await _call(nats, [uuid7()])

    async def test_a_reply_that_does_not_decode_raises_unavailable(self) -> None:
        """a malformed reply is a broker fault, never a success."""
        nats = _ScriptedRequests({"success": True, "rows_matched": "many"})

        with pytest.raises(AuditAnonymizeUnavailableError):
            await _call(nats, [uuid7()])

    async def test_a_success_without_counts_raises_unavailable(self) -> None:
        """a success that reports nothing cannot be told apart from one that did nothing."""
        nats = _ScriptedRequests({"success": True, "agent_id": str(_AGENT)})

        with pytest.raises(AuditAnonymizeUnavailableError):
            await _call(nats, [uuid7()])

    async def test_a_success_for_a_different_agent_raises_unavailable(self) -> None:
        """the hub echoes the agent it verified; one that is not this pod's answered someone else."""
        nats = _ScriptedRequests(_success(1, 1, agent_id=_OTHER_AGENT))

        with pytest.raises(AuditAnonymizeUnavailableError):
            await _call(nats, [uuid7()])

    async def test_no_token_raises_before_anything_is_sent(self) -> None:
        """a pod with no identity cannot ask, and does not try."""
        nats = _ScriptedRequests()

        with pytest.raises(AuditAnonymizeUnavailableError):
            await _call(nats, [uuid7()], identity_token="")

        assert nats.sent == []

    async def test_no_actors_sends_nothing(self) -> None:
        """nothing to anonymize is an answer, not a request."""
        nats = _ScriptedRequests()

        assert await _call(nats, []) == AuditAnonymization(rows_matched=0, rows_changed=0)
        assert nats.sent == []

    async def test_a_long_list_is_sent_in_bounded_batches_and_summed(self) -> None:
        """the bound is the wire's, not the caller's: batches go out in order and their counts add."""
        actors = [uuid7() for _ in range(MAX_ANONYMIZE_ACTORS + 3)]
        nats = _ScriptedRequests(_success(10, 7), _success(2, 1))

        result = await _call(nats, actors)

        assert result == AuditAnonymization(rows_matched=12, rows_changed=8)
        assert [len(sent["actor_user_ids"]) for _, sent in nats.sent] == [MAX_ANONYMIZE_ACTORS, 3]
        assert [UUID(actor) for _, sent in nats.sent for actor in sent["actor_user_ids"]] == actors

    async def test_duplicate_actors_are_asked_about_once(self) -> None:
        """the same person named twice is one request entry."""
        actor = uuid7()
        nats = _ScriptedRequests(_success(1, 1))

        await _call(nats, [actor, actor])

        assert nats.sent[0][1]["actor_user_ids"] == [str(actor)]
