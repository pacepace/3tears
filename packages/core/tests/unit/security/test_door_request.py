"""a door request proves its credential over its own body, door and operation, so a lifted one is useless."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from typing import Any

import pytest
from prometheus_client import REGISTRY
from pydantic import BaseModel, SecretStr, field_serializer
from threetears.nats.errors import KvError

from threetears.core.security import (
    DOOR_PROOF_FIELD,
    DOOR_PROOF_REPLAY_TTL,
    DelegatedTokenError,
    DoorCredential,
    DoorRequestDoor,
    DoorVerdict,
    InProcessEnvelopeLedger,
    PopKey,
    ProofAbsentError,
    RegistrationKey,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    mint_delegated_token,
    sign_identity_token,
    verify_door_request,
)
from threetears.core.security.door_proof import DoorProofSwitch, DoorWindow
from threetears.core.security.identity_token import IdentityClaims

_REGISTRY, _REGISTRY_PUB = generate_signing_keypair()
_HUB, _HUB_PUB = generate_signing_keypair()
_JWKS = build_jwks({"registry-1": _REGISTRY_PUB, "hub-1": _HUB_PUB})
_POD, _POD_PUB = generate_signing_keypair()
_AGENT, _AGENT_PUB = generate_signing_keypair()
_THIEF, _ = generate_signing_keypair()
_BODY = {"namespace": "agents.a", "query": "select 1", "params": [1, "x", None]}
_TX = {"correlation_id": "c-1", "tx_id": "tx-1"}


def _delegated(*, door: str = "l3") -> str:
    now = int(time.time())
    return mint_delegated_token(
        signing_key=_REGISTRY,
        kid="registry-1",
        agent_id="agent-1",
        pod_id="pod-1",
        customer_id="cust-1",
        user_id="user-1",
        doors=(door,),
        holder_jkt=jwk_thumbprint(_POD_PUB),
        correlation_id="corr-1",
        deadline=now + 60,
        now=now,
    )


def _agent_token(*, bound: bool = True, sid: str = "sid-1") -> str:
    now = int(time.time())
    return sign_identity_token(
        IdentityClaims(
            sub="agent-1",
            customer_id="cust-1",
            sid=sid,
            pod_id="pod-a",
            iss="hub",
            iat=now,
            exp=now + 600,
            cnf=jwk_thumbprint(_AGENT_PUB) if bound else None,
        ),
        signing_key=_HUB,
        kid="hub-1",
    )


class TestTheCredentialBindsOnlyItsOwnKey:
    """a key the token does not bind makes a bearer credential, never a proof the door would refuse."""

    def test_a_key_the_token_binds_is_kept(self) -> None:
        assert DoorCredential(token=_agent_token(), holder_key=_AGENT).proves

    def test_a_token_that_binds_no_key_is_bearer(self) -> None:
        credential = DoorCredential(token=_agent_token(bound=False), holder_key=_AGENT)
        assert not credential.proves
        assert DOOR_PROOF_FIELD not in credential.present(_BODY, door="l3", op="query")

    def test_a_token_binding_another_key_is_bearer(self) -> None:
        assert not DoorCredential(token=_agent_token(), holder_key=_THIEF).proves
        assert not DoorCredential(token=_delegated(), holder_key=_AGENT).proves

    def test_a_string_that_is_not_a_token_is_bearer(self) -> None:
        assert not DoorCredential(token="not-a-token", holder_key=_AGENT).proves

    def test_a_tool_pod_proves_the_delegation_with_its_own_key(self) -> None:
        delegated = _delegated()
        credential = DoorCredential.for_tool_call(
            delegated_token=delegated, identity_token="fwd", key=RegistrationKey(_POD)
        )
        assert credential == DoorCredential(token=delegated, holder_key=_POD)
        assert credential.proves

    def test_an_in_process_server_proves_the_agents_own_token(self) -> None:
        token = _agent_token()
        credential = DoorCredential.for_tool_call(delegated_token=None, identity_token=token, key=PopKey(_AGENT))
        assert credential == DoorCredential(token=token, holder_key=_AGENT)
        assert credential.proves

    def test_neither_key_proves_the_other_principals_token(self) -> None:
        token, delegated = _agent_token(), _delegated()
        # a pop key never proves a delegation, a registration key never the agent's own token
        assert DoorCredential.for_tool_call(
            delegated_token=delegated, identity_token=token, key=PopKey(_POD)
        ) == DoorCredential(token=token)
        assert DoorCredential.for_tool_call(
            delegated_token=None, identity_token=token, key=RegistrationKey(_AGENT)
        ) == DoorCredential(token=token)

    def test_a_pod_without_the_delegations_key_forwards_the_callers_token_bearer(self) -> None:
        token = _agent_token()
        credential = DoorCredential.for_tool_call(
            delegated_token=_delegated(), identity_token=token, key=RegistrationKey(_THIEF)
        )
        assert credential == DoorCredential(token=token)
        assert (
            DoorCredential.for_tool_call(delegated_token=None, identity_token=None, key=RegistrationKey(_POD)) is None
        )


class TestTheProofBindsTheBodyTheDoorAndTheOperation:
    """a proven request verifies only as sent, where it was sent, for what it asked."""

    def test_a_pod_proves_the_calls_delegated_token(self) -> None:
        sent = DoorCredential(token=_delegated(), holder_key=_POD).present(_BODY, door="l3", op="query")
        caller = verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3", op="query")
        assert (caller.agent_id, caller.pod_id, caller.user_id) == ("agent-1", "pod-1", "user-1")

    def test_an_agent_proves_its_own_token(self) -> None:
        sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_BODY, door="l3", op="query")
        caller = verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3", op="query")
        assert (caller.agent_id, caller.pod_id) == ("agent-1", None)

    def test_a_changed_body_is_refused(self) -> None:
        sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_BODY, door="l3", op="query")
        sent["query"] = "delete from t"
        with pytest.raises(DelegatedTokenError, match="proof of possession"):
            verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3", op="query")

    @pytest.mark.parametrize(
        ("signed", "replayed"),
        [
            ("tx.rollback", "tx.commit"),
            ("tx.execute", "tx.fetch"),
            ("tx.fetch", "tx.fetchrow"),
            ("tx.fetchrow", "tx.execute"),
        ],
    )
    def test_a_proof_for_one_operation_is_refused_for_another_with_the_same_body(
        self, signed: str, replayed: str
    ) -> None:
        sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_TX, door="l3", op=signed)
        verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3", op=signed)
        with pytest.raises(DelegatedTokenError, match="proof of possession"):
            verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3", op=replayed)

    def test_a_proof_for_one_door_is_refused_at_another(self) -> None:
        sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_BODY, door="l3", op="query")
        with pytest.raises(DelegatedTokenError, match="proof of possession"):
            verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="datasource", op="query")

    def test_a_delegated_token_at_another_door_is_refused(self) -> None:
        sent = DoorCredential(token=_delegated(door="datasource"), holder_key=_POD).present(
            _BODY, door="l3", op="query"
        )
        with pytest.raises(DelegatedTokenError, match="does not verify"):
            verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3", op="query")

    def test_a_bearer_request_is_left_to_the_doors_window(self) -> None:
        sent = DoorCredential(token=_agent_token()).present(_BODY, door="l3", op="query")
        assert DOOR_PROOF_FIELD not in sent
        with pytest.raises(ProofAbsentError):
            verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3", op="query")

    def test_a_request_without_a_credential_is_refused_outright(self) -> None:
        with pytest.raises(DelegatedTokenError, match="no credential") as caught:
            verify_door_request(_BODY, jwks=_JWKS, hub_issuer="hub", door="l3", op="query")
        assert not isinstance(caught.value, ProofAbsentError)


class _Request(BaseModel):
    identity_token: SecretStr
    identity_proof: str | None = None
    object_id: str

    @field_serializer("identity_token", when_used="json")
    def _reveal(self, value: SecretStr) -> str:
        return value.get_secret_value()


class _NoProofField(BaseModel):
    identity_token: str


class TestAWireModel:
    """a model is proven over what ``model_dump_json`` puts on the wire."""

    def test_it_verifies_as_sent(self) -> None:
        token = _agent_token()
        proven = DoorCredential(token=token, holder_key=_AGENT).presented(
            _Request(identity_token=SecretStr(token), object_id="o-1"), door="hub.object.resolve", op="resolve"
        )
        caller = verify_door_request(
            json.loads(proven.model_dump_json()), jwks=_JWKS, hub_issuer="hub", door="hub.object.resolve", op="resolve"
        )
        assert caller.agent_id == "agent-1"

    def test_a_bearer_credential_sends_it_as_it_is(self) -> None:
        message = _Request(identity_token=SecretStr("t"), object_id="o-1")
        assert DoorCredential(token="t").presented(message, door="d", op="o") is message

    def test_one_carrying_another_token_is_not_proven(self) -> None:
        with pytest.raises(ValueError, match="does not carry"):
            DoorCredential(token=_agent_token(), holder_key=_AGENT).presented(
                _Request(identity_token=SecretStr("other"), object_id="o-1"), door="d", op="o"
            )

    def test_one_with_no_proof_field_is_a_wiring_error(self) -> None:
        with pytest.raises(ValueError, match="no identity_proof field"):
            DoorCredential(token="t").presented(_NoProofField(identity_token="t"), door="d", op="o")


# --- the door ---------------------------------------------------------------------------------

_WINDOW = DoorWindow(door="l3", opened=date(2026, 10, 8), required_from=date(2027, 1, 5))


class _Keys:
    """a hub key set that can start empty, rotate, and count its refreshes."""

    def __init__(self, *, first: Mapping[str, Any], after_refresh: Mapping[str, Any] | None = None) -> None:
        self._first = first
        self._after = after_refresh if after_refresh is not None else first
        self.refreshes = 0

    async def get_jwks(self, *, refresh: bool = False) -> Mapping[str, Any]:
        if refresh:
            self.refreshes += 1
        return self._after if self.refreshes else self._first


class _BrokenLedger(InProcessEnvelopeLedger):
    """a ledger that cannot be reached."""

    async def record_unique(self, nonce: str, *, issued_at: datetime) -> bool:
        raise KvError("the ledger bucket is unreachable")


def _door(
    *,
    today: date = date(2026, 10, 9),
    delegated: bool = True,
    keys: Any = None,
    ledger: Any = None,
) -> DoorRequestDoor:
    return DoorRequestDoor(
        switch=DoorProofSwitch(_WINDOW, today=lambda: today),
        jwks=keys if keys is not None else _Keys(first=_JWKS),
        hub_issuer="hub",
        replay_guard=ledger if ledger is not None else InProcessEnvelopeLedger(retention=DOOR_PROOF_REPLAY_TTL),
        accepts_delegated=delegated,
    )


def _refused(cause: str) -> float:
    value = REGISTRY.get_sample_value("door_request_refused_total", {"door": "l3", "cause": cause})
    return value or 0.0


async def _counted(cause: str, verdict_of: Any) -> DoorVerdict:
    before = _refused(cause)
    verdict: DoorVerdict = await verdict_of
    assert _refused(cause) == before + 1, cause
    return verdict


class TestTheDoor:
    """each verdict, and the refusal class it is counted under."""

    async def test_a_proven_request_is_served_once_then_counted_as_replayed(self) -> None:
        door = _door()
        sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_BODY, door="l3", op="query")
        first = await door.admit(sent, op="query")
        assert first.admitted
        assert first.caller is not None
        assert first.caller.agent_id == "agent-1"
        assert not (await _counted("replayed", door.admit(sent, op="query"))).admitted

    async def test_a_proof_for_a_rollback_is_not_admitted_as_a_commit(self) -> None:
        door = _door()
        sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_TX, door="l3", op="tx.rollback")
        assert not (await _counted("unverified", door.admit(sent, op="tx.commit"))).admitted
        assert (await door.admit(sent, op="tx.rollback")).admitted

    async def test_a_lifted_token_under_another_requests_proof_is_counted_unverified(self) -> None:
        token = _agent_token()
        sent = DoorCredential(token=token).present(_BODY, door="l3", op="query")
        other = DoorCredential(token=_agent_token(sid="sid-2"), holder_key=_AGENT).present(_BODY, door="l3", op="query")
        sent[DOOR_PROOF_FIELD] = other[DOOR_PROOF_FIELD]
        assert not (await _counted("unverified", _door().admit(sent, op="query"))).admitted

    async def test_the_door_reads_the_bytes_as_sent(self) -> None:
        body = {**_BODY, "a_field_a_later_caller_adds": "x"}
        sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(body, door="l3", op="query")
        assert (await _door().admit_message(json.dumps(sent).encode(), op="query")).admitted
        assert not (await _counted("malformed", _door().admit_message(b"[]", op="query"))).admitted

    async def test_a_bearer_request_is_served_while_the_window_is_open_and_refused_after(self) -> None:
        sent = DoorCredential(token=_agent_token()).present(_BODY, door="l3", op="query")
        assert await _door().admit(sent, op="query") == DoorVerdict(admitted=True)
        assert not (await _counted("absent", _door(today=date(2027, 1, 5)).admit(sent, op="query"))).admitted

    async def test_a_non_binding_key_is_served_as_bearer_while_the_window_is_open(self) -> None:
        credential = DoorCredential(token=_agent_token(bound=False), holder_key=_AGENT)
        sent = credential.present(_BODY, door="l3", op="query")
        assert await _door().admit(sent, op="query") == DoorVerdict(admitted=True)

    async def test_a_delegated_token_has_no_bearer_form(self) -> None:
        sent = DoorCredential(token=_delegated()).present(_BODY, door="l3", op="query")
        assert not (await _counted("absent", _door().admit(sent, op="query"))).admitted

    async def test_a_door_no_pod_reaches_refuses_a_delegated_token(self) -> None:
        sent = DoorCredential(token=_delegated(), holder_key=_POD).present(_BODY, door="l3", op="query")
        assert not (await _counted("delegated", _door(delegated=False).admit(sent, op="query"))).admitted
        assert (await _door(delegated=True).admit(sent, op="query")).admitted

    async def test_an_empty_key_set_is_counted_as_keys_unavailable(self) -> None:
        sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_BODY, door="l3", op="query")
        door = _door(keys=_Keys(first={}))
        assert not (await _counted("keys_unavailable", door.admit(sent, op="query"))).admitted

    @pytest.mark.parametrize("source", ["pinned-none", "source-fails"])
    async def test_a_key_source_that_has_no_keys_is_counted_as_keys_unavailable(self, source: str) -> None:
        """the source protocol's "no keys" is one error, so the door refuses rather than raising."""
        from threetears.core.security import PinnedEnvelopeJwks, ServiceEnvelopeError  # noqa: PLC0415

        class _Fails:
            async def get_jwks(self, *, refresh: bool = False) -> Mapping[str, Any]:
                raise ServiceEnvelopeError("the hub JWKS could not be fetched")

        keys = PinnedEnvelopeJwks(None) if source == "pinned-none" else _Fails()
        sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_BODY, door="l3", op="query")
        assert not (await _counted("keys_unavailable", _door(keys=keys).admit(sent, op="query"))).admitted

    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            ({"identity_token": "own"}, "agent-1"),
            ({"user_identity_token": "own"}, "agent-1"),
            ({}, "unclaimed"),
            ({"identity_token": "not-a-token"}, "unclaimed"),
        ],
    )
    async def test_a_request_without_a_proof_is_counted_under_the_sender_its_token_claims(
        self, fields: dict[str, str], expected: str
    ) -> None:
        """the door reads the sender off the token it already reads; no caller attributes it by hand."""
        switch = DoorProofSwitch(_WINDOW, today=lambda: date(2026, 10, 9))
        door = DoorRequestDoor(
            switch=switch,
            jwks=_Keys(first=_JWKS),
            hub_issuer="hub",
            replay_guard=InProcessEnvelopeLedger(retention=DOOR_PROOF_REPLAY_TTL),
            accepts_delegated=True,
        )
        body = {**_BODY, **{name: _agent_token() if value == "own" else value for name, value in fields.items()}}
        assert (await door.admit(body, op="query")).admitted
        assert [sender.claimed_sender for sender in switch.absent_senders()] == [expected]

    async def test_a_hub_rekey_is_found_by_one_refresh(self) -> None:
        keys = _Keys(first=build_jwks({"old-1": generate_signing_keypair()[1]}), after_refresh=_JWKS)
        sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_BODY, door="l3", op="query")
        assert (await _door(keys=keys).admit(sent, op="query")).admitted
        assert keys.refreshes == 1

    async def test_an_unreachable_ledger_refuses_and_never_raises(self) -> None:
        sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_BODY, door="l3", op="query")
        door = _door(ledger=_BrokenLedger(retention=DOOR_PROOF_REPLAY_TTL))
        assert not (await _counted("ledger_unavailable", door.admit(sent, op="query"))).admitted

    def test_a_ledger_that_forgets_an_id_before_its_proof_expires_fails_the_boot(self) -> None:
        with pytest.raises(ValueError, match="must remember an id"):
            _door(ledger=InProcessEnvelopeLedger(retention=DOOR_PROOF_REPLAY_TTL - timedelta(seconds=1)))


def test_a_datasource_operation_is_read_off_its_subject() -> None:
    """the client and the door derive it from the subject itself, the sanitised segment included."""
    from threetears.core.security import datasource_query_op
    from threetears.nats import Subjects, set_default_namespace

    set_default_namespace("test")
    assert datasource_query_op(Subjects.datasource_query("warehouse")) == "warehouse.query"
    # a dot in the name would split the subject; the builder writes it as a hyphen, and the op follows
    assert datasource_query_op(Subjects.datasource_query("central.reporting")) == "central-reporting.query"


class TestTheHashFrameIsInjective:
    """no door or operation, whatever characters it holds, hashes as another split of the same text."""

    @pytest.mark.parametrize(
        ("one", "other"),
        [(("a:b", "c"), ("a", "b:c")), (("l3", "tx.commit"), ("l3:tx", "commit")), (("a", "1:b"), ("a1", "b"))],
    )
    def test_two_splits_never_share_a_hash(self, one: tuple[str, str], other: tuple[str, str]) -> None:
        from threetears.core.security import door_request_hash

        assert door_request_hash(_TX, door=one[0], op=one[1]) != door_request_hash(_TX, door=other[0], op=other[1])


class _RecordingLedger(InProcessEnvelopeLedger):
    """keeps the ids it was asked to spend."""

    def __init__(self) -> None:
        super().__init__(retention=DOOR_PROOF_REPLAY_TTL)
        self.spent: list[str] = []

    async def record_unique(self, nonce: str, *, issued_at: datetime) -> bool:
        self.spent.append(nonce)
        return await super().record_unique(nonce, issued_at=issued_at)


async def test_a_proof_is_spent_under_its_door_proof_and_agent_namespace() -> None:
    """the ledger key cannot collide with another kind of id or another principal's proof."""
    ledger = _RecordingLedger()
    sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_BODY, door="l3", op="query")
    assert (await _door(ledger=ledger).admit(sent, op="query")).admitted
    (spent,) = ledger.spent
    assert spent.startswith("door-proof|agent-1|")
    assert spent.removeprefix("door-proof|agent-1|")


@pytest.mark.parametrize("token", ["a.!!!.c", "a." + "e30" + ".c", "a.bm90LWpzb24.c", "only.two"])
def test_a_token_whose_payload_does_not_parse_binds_no_key(token: str) -> None:
    """a malformed or claim-less token makes a bearer credential, never an exception."""
    assert not DoorCredential(token=token, holder_key=_AGENT).proves


def test_an_l3_operation_is_its_subjects_own_part() -> None:
    """the L3 client and the broker read the operation off the subject the same way."""
    from threetears.core.security import l3_request_op

    assert l3_request_op("test.l3.query") == "query"
    assert l3_request_op("test.l3.tx.commit") == "tx.commit"
    with pytest.raises(ValueError, match="not an l3"):
        l3_request_op("test.datasource.warehouse.query")


def test_a_subject_that_is_not_a_datasource_query_has_no_operation() -> None:
    """a door never signs or verifies a made-up operation such as ``None.query``."""
    from threetears.core.security import datasource_query_op
    from threetears.nats import Subject

    with pytest.raises(ValueError, match="not a datasource query"):
        datasource_query_op(Subject.raw("test.l3.query"))


def test_a_delegation_names_only_the_doors_a_pod_reaches_as_its_caller() -> None:
    """the registry's default audience is the set the hub takes a delegation at: one source."""
    from threetears.core.security import DEFAULT_DELEGATED_DOORS

    assert frozenset({"hub.object.resolve", "namespace.discover", "hub.engagement"}) == DEFAULT_DELEGATED_DOORS
