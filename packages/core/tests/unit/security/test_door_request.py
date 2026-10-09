"""a door request proves its credential over its own body, so a lifted token or request is useless."""

from __future__ import annotations

import json
import time

import pytest
from pydantic import BaseModel, SecretStr, field_serializer

from threetears.core.security import (
    DOOR_PROOF_FIELD,
    DelegatedTokenError,
    DoorCredential,
    ProofAbsentError,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    mint_delegated_token,
    sign_identity_token,
    verify_door_request,
)
from threetears.core.security.identity_token import IdentityClaims

_REGISTRY, _REGISTRY_PUB = generate_signing_keypair()
_HUB, _HUB_PUB = generate_signing_keypair()
_JWKS = build_jwks({"registry-1": _REGISTRY_PUB, "hub-1": _HUB_PUB})
_POD, _POD_PUB = generate_signing_keypair()
_AGENT, _AGENT_PUB = generate_signing_keypair()
_BODY = {"namespace": "agents.a", "query": "select 1", "params": [1, "x", None]}


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


def _agent_token() -> str:
    now = int(time.time())
    return sign_identity_token(
        IdentityClaims(
            sub="agent-1",
            customer_id="cust-1",
            sid="sid-1",
            pod_id="pod-a",
            iss="hub",
            iat=now,
            exp=now + 600,
            cnf=jwk_thumbprint(_AGENT_PUB),
        ),
        signing_key=_HUB,
        kid="hub-1",
    )


def test_a_pod_proves_the_calls_delegated_token_with_its_own_key() -> None:
    credential = DoorCredential.for_tool_call(delegated_token=_delegated(), identity_token="agent", pod_key=_POD)
    assert credential is not None
    caller = verify_door_request(credential.present(_BODY), jwks=_JWKS, hub_issuer="hub", door="l3")
    assert (caller.agent_id, caller.pod_id, caller.user_id) == ("agent-1", "pod-1", "user-1")


def test_an_agent_proves_its_own_token_with_its_holder_key() -> None:
    sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_BODY)
    caller = verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3")
    assert (caller.agent_id, caller.pod_id) == ("agent-1", None)


def test_a_changed_body_is_refused() -> None:
    sent = DoorCredential(token=_agent_token(), holder_key=_AGENT).present(_BODY)
    sent["query"] = "delete from t"
    with pytest.raises(DelegatedTokenError, match="proof of possession"):
        verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3")


def test_a_lifted_token_proven_with_another_key_is_refused() -> None:
    thief, _ = generate_signing_keypair()
    sent = DoorCredential(token=_delegated(), holder_key=thief).present(_BODY)
    with pytest.raises(DelegatedTokenError, match="proof of possession"):
        verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3")


def test_a_delegated_token_at_another_door_is_refused() -> None:
    sent = DoorCredential(token=_delegated(door="datasource"), holder_key=_POD).present(_BODY)
    with pytest.raises(DelegatedTokenError, match="does not verify"):
        verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3")


def test_a_bearer_request_is_left_to_the_doors_window() -> None:
    sent = DoorCredential(token=_agent_token()).present(_BODY)
    assert DOOR_PROOF_FIELD not in sent
    with pytest.raises(ProofAbsentError):
        verify_door_request(sent, jwks=_JWKS, hub_issuer="hub", door="l3")


def test_a_request_without_a_credential_is_refused_outright() -> None:
    with pytest.raises(DelegatedTokenError, match="no credential") as caught:
        verify_door_request(_BODY, jwks=_JWKS, hub_issuer="hub", door="l3")
    assert not isinstance(caught.value, ProofAbsentError)


def test_a_pod_without_its_key_falls_back_to_the_forwarded_token() -> None:
    credential = DoorCredential.for_tool_call(delegated_token=_delegated(), identity_token="agent-tok", pod_key=None)
    assert credential == DoorCredential(token="agent-tok")
    assert DoorCredential.for_tool_call(delegated_token=None, identity_token=None, pod_key=_POD) is None


class _Request(BaseModel):
    identity_token: SecretStr
    identity_proof: str | None = None
    object_id: str

    @field_serializer("identity_token", when_used="json")
    def _reveal(self, value: SecretStr) -> str:
        return value.get_secret_value()


class _NoProofField(BaseModel):
    identity_token: str


def test_a_wire_model_is_proven_over_what_it_puts_on_the_wire() -> None:
    token = _agent_token()
    proven = DoorCredential(token=token, holder_key=_AGENT).presented(
        _Request(identity_token=SecretStr(token), object_id="o-1")
    )
    caller = verify_door_request(json.loads(proven.model_dump_json()), jwks=_JWKS, hub_issuer="hub", door="l3")
    assert caller.agent_id == "agent-1"


def test_a_wire_model_from_a_caller_without_a_key_is_sent_as_it_is() -> None:
    message = _Request(identity_token=SecretStr("t"), object_id="o-1")
    assert DoorCredential(token="t").presented(message) is message


def test_a_wire_model_carrying_another_token_is_not_proven() -> None:
    with pytest.raises(ValueError, match="does not carry"):
        DoorCredential(token=_agent_token(), holder_key=_AGENT).presented(
            _Request(identity_token=SecretStr("other"), object_id="o-1")
        )


def test_a_wire_model_with_no_proof_field_is_a_wiring_error() -> None:
    with pytest.raises(ValueError, match="no identity_proof field"):
        DoorCredential(token="t", holder_key=_AGENT).presented(_NoProofField(identity_token="t"))
