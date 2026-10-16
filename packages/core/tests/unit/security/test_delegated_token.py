"""a delegated token reaches only its doors, only with its pod's key, only until its call ends."""

from __future__ import annotations

import time

import pytest

from threetears.core.security import (
    DelegatedTokenError,
    ProofAbsentError,
    build_jwks,
    canonical_call_hash,
    generate_signing_keypair,
    jwk_thumbprint,
    make_pop_proof,
    mint_delegated_token,
    sign_identity_token,
    verify_delegated_token,
    verify_door_caller,
)
from threetears.core.security.identity_token import IdentityClaims
from threetears.core.security.pop import access_token_hash

_REGISTRY, _REGISTRY_PUB = generate_signing_keypair()
_HUB, _HUB_PUB = generate_signing_keypair()
_JWKS = build_jwks({"registry-1": _REGISTRY_PUB, "hub-1": _HUB_PUB})
_POD, _POD_PUB = generate_signing_keypair()
_BODY = canonical_call_hash("l3.query", {"sql": "select 1"}, "corr-1")


def _delegated(*, doors=("l3.query",), holder=_POD_PUB, deadline=None, now=None) -> str:
    issued = int(time.time()) if now is None else now
    return mint_delegated_token(
        signing_key=_REGISTRY,
        kid="registry-1",
        agent_id="agent-1",
        pod_id="pod-1",
        customer_id="cust-1",
        user_id="user-1",
        doors=doors,
        holder_jkt=jwk_thumbprint(holder),
        correlation_id="corr-1",
        deadline=deadline if deadline is not None else issued + 60,
        now=issued,
    )


def _proof(token: str, *, key=_POD, body: str = _BODY, nonce: str = "n-1") -> str:
    return make_pop_proof(
        holder_key=key, access_token_hash=access_token_hash(token), body_hash=body, nonce=nonce, iat=int(time.time())
    )


def test_the_pod_acts_as_its_caller_at_a_named_door() -> None:
    token = _delegated()
    caller = verify_door_caller(token, _proof(token), jwks=_JWKS, hub_issuer="hub", door="l3.query", body_hash=_BODY)
    assert (caller.agent_id, caller.customer_id, caller.user_id, caller.pod_id) == (
        "agent-1",
        "cust-1",
        "user-1",
        "pod-1",
    )
    assert caller.proof.jti == "n-1"


def test_another_door_is_refused() -> None:
    token = _delegated()
    with pytest.raises(DelegatedTokenError, match="does not verify"):
        verify_door_caller(token, _proof(token), jwks=_JWKS, hub_issuer="hub", door="datasource.query", body_hash=_BODY)


def test_a_token_without_the_pods_key_is_useless() -> None:
    other, _ = generate_signing_keypair()
    token = _delegated()
    with pytest.raises(DelegatedTokenError, match="proof of possession"):
        verify_door_caller(
            token, _proof(token, key=other), jwks=_JWKS, hub_issuer="hub", door="l3.query", body_hash=_BODY
        )


def test_a_proof_for_another_request_is_refused() -> None:
    token = _delegated()
    with pytest.raises(DelegatedTokenError, match="proof of possession"):
        verify_door_caller(
            token, _proof(token, body="other-body"), jwks=_JWKS, hub_issuer="hub", door="l3.query", body_hash=_BODY
        )


def test_after_the_call_ends_the_token_is_refused() -> None:
    long_ago = int(time.time()) - 3600
    token = _delegated(now=long_ago, deadline=long_ago + 60)
    with pytest.raises(DelegatedTokenError, match="ExpiredSignature"):
        verify_delegated_token(token, jwks=_JWKS, door="l3.query")


def test_no_proof_leaves_it_to_the_doors_window() -> None:
    with pytest.raises(ProofAbsentError):
        verify_door_caller(_delegated(), None, jwks=_JWKS, hub_issuer="hub", door="l3.query", body_hash=_BODY)


def test_an_agent_proves_its_own_token_with_its_own_key() -> None:
    agent_key, agent_pub = generate_signing_keypair()
    now = int(time.time())
    token = sign_identity_token(
        IdentityClaims(
            sub="agent-1",
            customer_id="cust-1",
            sid="sid-1",
            pod_id="pod-a",
            iss="hub",
            iat=now,
            exp=now + 600,
            cnf=jwk_thumbprint(agent_pub),
        ),
        signing_key=_HUB,
        kid="hub-1",
    )
    caller = verify_door_caller(
        token, _proof(token, key=agent_key), jwks=_JWKS, hub_issuer="hub", door="l3.query", body_hash=_BODY
    )
    assert (caller.agent_id, caller.pod_id) == ("agent-1", None)


def test_minting_past_the_maximum_or_for_no_door_is_refused() -> None:
    now = int(time.time())
    with pytest.raises(ValueError, match="at most"):
        _delegated(now=now, deadline=now + 7200)
    with pytest.raises(ValueError, match="at least one door"):
        _delegated(doors=())


class TestThePodVerifiesItsOwnDelegation:
    """a declaring pod takes the caller from the call's delegation, for itself and this call only."""

    def test_it_names_the_caller(self) -> None:
        from threetears.core.security import verify_delegation_at_pod

        claims = verify_delegation_at_pod(_delegated(), jwks=_JWKS, pod_id="pod-1", correlation_id="corr-1")
        assert (claims.agent_id, claims.customer_id, claims.user_id) == ("agent-1", "cust-1", "user-1")

    @pytest.mark.parametrize(
        ("pod_id", "correlation_id", "match"),
        [("pod-2", "corr-1", "does not verify"), ("pod-1", "corr-2", "another call")],
    )
    def test_another_pod_or_call_is_refused(self, pod_id: str, correlation_id: str, match: str) -> None:
        from threetears.core.security import verify_delegation_at_pod

        with pytest.raises(DelegatedTokenError, match=match):
            verify_delegation_at_pod(_delegated(), jwks=_JWKS, pod_id=pod_id, correlation_id=correlation_id)

    def test_an_identity_token_is_not_a_delegation(self) -> None:
        from threetears.core.security import verify_delegation_at_pod

        now = int(time.time())
        token = sign_identity_token(
            IdentityClaims(sub="agent-1", customer_id="cust-1", sid="s", pod_id="p", iss="hub", iat=now, exp=now + 60),
            signing_key=_HUB,
            kid="hub-1",
        )
        with pytest.raises(DelegatedTokenError, match="not an EdDSA delegated token"):
            verify_delegation_at_pod(token, jwks=_JWKS, pod_id="pod-1", correlation_id="corr-1")


class TestThePodsChecksStandOnTheirOwn:
    """each check the pod makes refuses on its own, not only behind the audience."""

    @staticmethod
    def _signed(**overrides: object) -> str:
        import jwt  # noqa: PLC0415

        from threetears.core.security import DELEGATED_TOKEN_ISSUER, DELEGATED_TOKEN_TYPE  # noqa: PLC0415

        now = int(time.time())
        payload: dict[str, object] = {
            "iss": DELEGATED_TOKEN_ISSUER,
            "aud": ["l3.query", "tool-pod:pod-1"],
            "sub": "agent-1",
            "act": "pod-1",
            "customer_id": "cust-1",
            "user_id": None,
            "cnf": "jkt-1",
            "corr": "corr-1",
            "iat": now,
            "exp": now + 60,
            **overrides,
        }
        return jwt.encode(
            payload, key=_REGISTRY, algorithm="EdDSA", headers={"kid": "registry-1", "typ": DELEGATED_TOKEN_TYPE}
        )

    def test_a_delegation_addressed_to_this_pod_but_naming_another_actor_is_refused(self) -> None:
        from threetears.core.security import verify_delegation_at_pod  # noqa: PLC0415

        token = self._signed(act="pod-2")
        with pytest.raises(DelegatedTokenError, match="another pod as its actor"):
            verify_delegation_at_pod(token, jwks=_JWKS, pod_id="pod-1", correlation_id="corr-1")

    def test_the_conversation_travels_signed(self) -> None:
        from threetears.core.security import verify_delegation_at_pod  # noqa: PLC0415

        token = self._signed(conversation_id="conv-1")
        claims = verify_delegation_at_pod(token, jwks=_JWKS, pod_id="pod-1", correlation_id="corr-1")
        assert claims.conversation_id == "conv-1"
        bare = verify_delegation_at_pod(self._signed(), jwks=_JWKS, pod_id="pod-1", correlation_id="corr-1")
        assert bare.conversation_id is None

    def test_only_an_unknown_key_is_told_apart_for_a_refresh(self) -> None:
        from threetears.core.security import DelegatedKeyNotFoundError, verify_delegation_at_pod  # noqa: PLC0415

        with pytest.raises(DelegatedKeyNotFoundError):
            verify_delegation_at_pod(
                self._signed(), jwks=build_jwks({"hub-1": _HUB_PUB}), pod_id="pod-1", correlation_id="corr-1"
            )
        expired = self._signed(iat=int(time.time()) - 600, exp=int(time.time()) - 300)
        with pytest.raises(DelegatedTokenError) as caught:
            verify_delegation_at_pod(expired, jwks=_JWKS, pod_id="pod-1", correlation_id="corr-1")
        assert not isinstance(caught.value, DelegatedKeyNotFoundError)
