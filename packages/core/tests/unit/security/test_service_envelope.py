"""a service envelope verifies only from its signer, for its door and operation, over its body, now, once."""

from __future__ import annotations

import time
from datetime import timedelta

import jwt as pyjwt
import pytest

from threetears.core.security import (
    SERVICE_ENVELOPE_FIELD,
    SERVICE_ENVELOPE_LEEWAY,
    SERVICE_ENVELOPE_MAX_LIFETIME,
    SERVICE_ENVELOPE_REPLAY_TTL,
    ServiceEnvelopeError,
    body_hash,
    build_jwks,
    generate_signing_keypair,
    mint_service_envelope,
    verify_and_record,
    verify_service_envelope,
)
from threetears.core.testing.replay_guard import FakeReplayGuard

_BODY = {"principal_id": "p-1", "grant": "workspace.read", "nested": {"b": 2, "a": 1}}


def _keys(kid: str = "idc-1"):
    priv, pub = generate_signing_keypair()
    return priv, build_jwks({kid: pub})


def _mint(priv, **overrides):
    args = {
        "signing_key": priv,
        "kid": "idc-1",
        "issuer": "identity-core",
        "audience": "hub.claim_grant",
        "op": "ensure",
        "body": _BODY,
    }
    args.update(overrides)
    return mint_service_envelope(**args)


def _verify(envelope, jwks, **overrides):
    args = {"jwks": jwks, "issuer": "identity-core", "audience": "hub.claim_grant", "op": "ensure", "body": _BODY}
    args.update(overrides)
    return verify_service_envelope(envelope, **args)


class TestVerify:
    def test_round_trip_carries_the_signed_facts(self) -> None:
        priv, jwks = _keys()
        claims = _verify(_mint(priv), jwks)
        assert claims.issuer == "identity-core"
        assert claims.audience == "hub.claim_grant"
        assert claims.op == "ensure"
        assert claims.body_hash == body_hash(_BODY)
        assert claims.jti
        assert claims.exp - claims.iat == int(SERVICE_ENVELOPE_MAX_LIFETIME.total_seconds())

    def test_another_signers_key_is_refused(self) -> None:
        priv, _ = _keys()
        _, other_jwks = _keys()
        with pytest.raises(ServiceEnvelopeError, match="does not verify"):
            _verify(_mint(priv), other_jwks)

    def test_an_unknown_kid_is_refused(self) -> None:
        priv, jwks = _keys()
        with pytest.raises(ServiceEnvelopeError, match="kid"):
            _verify(_mint(priv, kid="idc-2"), jwks)

    def test_another_issuer_is_refused(self) -> None:
        priv, jwks = _keys()
        with pytest.raises(ServiceEnvelopeError, match="InvalidIssuer"):
            _verify(_mint(priv, issuer="gateway"), jwks)

    def test_an_envelope_for_another_door_is_refused(self) -> None:
        priv, jwks = _keys()
        with pytest.raises(ServiceEnvelopeError, match="InvalidAudience"):
            _verify(_mint(priv, audience="hub.audit_merge"), jwks)

    def test_an_envelope_for_another_operation_is_refused(self) -> None:
        priv, jwks = _keys()
        with pytest.raises(ServiceEnvelopeError, match="another operation"):
            _verify(_mint(priv, op="revoke"), jwks)

    def test_an_envelope_moved_onto_another_body_is_refused(self) -> None:
        priv, jwks = _keys()
        with pytest.raises(ServiceEnvelopeError, match="another body"):
            _verify(_mint(priv), jwks, body={**_BODY, "grant": "workspace.admin"})

    def test_the_envelope_field_itself_is_not_hashed(self) -> None:
        # the envelope travels inside the body it signs, so it cannot be part of what it signs
        priv, jwks = _keys()
        envelope = _mint(priv)
        _verify(envelope, jwks, body={**_BODY, SERVICE_ENVELOPE_FIELD: envelope})

    def test_key_order_does_not_change_the_hash(self) -> None:
        reordered = {"nested": {"a": 1, "b": 2}, "grant": "workspace.read", "principal_id": "p-1"}
        assert body_hash(reordered) == body_hash(_BODY)

    def test_a_non_json_value_is_refused_rather_than_stringified(self) -> None:
        with pytest.raises(TypeError):
            body_hash({"when": object()})

    def test_an_expired_envelope_is_refused(self) -> None:
        priv, jwks = _keys()
        stale = int(time.time()) - int((SERVICE_ENVELOPE_MAX_LIFETIME + SERVICE_ENVELOPE_LEEWAY).total_seconds()) - 1
        with pytest.raises(ServiceEnvelopeError, match="ExpiredSignature"):
            _verify(_mint(priv, now=stale), jwks)

    def test_an_issue_time_beyond_the_leeway_ahead_is_refused(self) -> None:
        priv, jwks = _keys()
        ahead = int(time.time()) + int(SERVICE_ENVELOPE_LEEWAY.total_seconds()) + 5
        with pytest.raises(ServiceEnvelopeError, match="ImmatureSignature"):
            _verify(_mint(priv, now=ahead), jwks)

    def test_minting_a_longer_life_than_the_maximum_is_refused(self) -> None:
        priv, _ = _keys()
        with pytest.raises(ValueError, match="at most"):
            _mint(priv, ttl=SERVICE_ENVELOPE_MAX_LIFETIME + timedelta(seconds=1))

    def test_a_signed_longer_life_than_the_maximum_is_refused(self) -> None:
        # a signer that bypasses the minter's bound is still held to it by the verifier
        priv, jwks = _keys()
        now = int(time.time())
        payload = {
            "iss": "identity-core",
            "aud": "hub.claim_grant",
            "op": "ensure",
            "bh": body_hash(_BODY),
            "jti": "j-1",
            "iat": now,
            "exp": now + 3600,
        }
        envelope = pyjwt.encode(payload, key=priv, algorithm="EdDSA", headers={"kid": "idc-1"})
        with pytest.raises(ServiceEnvelopeError, match="longer life"):
            _verify(envelope, jwks)

    def test_a_missing_claim_is_refused(self) -> None:
        priv, jwks = _keys()
        now = int(time.time())
        payload = {"iss": "identity-core", "aud": "hub.claim_grant", "op": "ensure", "bh": body_hash(_BODY)}
        payload |= {"iat": now, "exp": now + 30}
        envelope = pyjwt.encode(payload, key=priv, algorithm="EdDSA", headers={"kid": "idc-1"})
        with pytest.raises(ServiceEnvelopeError, match="MissingRequiredClaim"):
            _verify(envelope, jwks)

    def test_a_non_eddsa_envelope_is_refused(self) -> None:
        _, jwks = _keys()
        now = int(time.time())
        payload = {"iss": "identity-core", "aud": "hub.claim_grant", "op": "ensure", "bh": body_hash(_BODY)}
        payload |= {"jti": "j", "iat": now, "exp": now + 30}
        envelope = pyjwt.encode(payload, key="a-shared-secret-long-enough-for-hs256!", algorithm="HS256")
        with pytest.raises(ServiceEnvelopeError, match="not EdDSA"):
            _verify(envelope, jwks)

    def test_garbage_is_refused(self) -> None:
        _, jwks = _keys()
        with pytest.raises(ServiceEnvelopeError, match="malformed"):
            _verify("not-a-jws", jwks)


class TestVerifyAndRecord:
    async def test_the_first_use_is_accepted_and_a_replay_refused(self) -> None:
        priv, jwks = _keys()
        envelope = _mint(priv)
        guard = FakeReplayGuard()
        args = {"jwks": jwks, "issuer": "identity-core", "audience": "hub.claim_grant", "op": "ensure"}
        await verify_and_record(envelope, body=_BODY, replay_guard=guard, **args)
        with pytest.raises(ServiceEnvelopeError, match="already used"):
            await verify_and_record(envelope, body=_BODY, replay_guard=guard, **args)

    async def test_an_envelope_that_does_not_verify_spends_nothing(self) -> None:
        priv, jwks = _keys()
        envelope = _mint(priv, jti="shared-id")
        guard = FakeReplayGuard()
        args = {"jwks": jwks, "issuer": "identity-core", "audience": "hub.claim_grant", "op": "ensure"}
        with pytest.raises(ServiceEnvelopeError, match="another body"):
            await verify_and_record(envelope, body={"other": 1}, replay_guard=guard, **args)
        await verify_and_record(envelope, body=_BODY, replay_guard=guard, **args)

    async def test_ids_are_scoped_to_their_issuer(self) -> None:
        # one service choosing another's id spends only its own
        idc_priv, idc_jwks = _keys()
        gw_priv, gw_jwks = _keys("gw-1")
        guard = FakeReplayGuard()
        await verify_and_record(
            _mint(gw_priv, kid="gw-1", issuer="gateway", jti="same"),
            jwks=gw_jwks,
            issuer="gateway",
            audience="hub.claim_grant",
            op="ensure",
            body=_BODY,
            replay_guard=guard,
        )
        await verify_and_record(
            _mint(idc_priv, jti="same"),
            jwks=idc_jwks,
            issuer="identity-core",
            audience="hub.claim_grant",
            op="ensure",
            body=_BODY,
            replay_guard=guard,
        )

    async def test_a_guard_not_sized_for_the_leeway_is_refused(self) -> None:
        priv, jwks = _keys()
        guard = FakeReplayGuard(verifier_future_tolerance=timedelta(0))
        with pytest.raises(ValueError, match="sized for"):
            await verify_and_record(
                _mint(priv),
                jwks=jwks,
                issuer="identity-core",
                audience="hub.claim_grant",
                op="ensure",
                body=_BODY,
                replay_guard=guard,
            )

    def test_the_replay_memory_covers_the_whole_accept_window(self) -> None:
        assert SERVICE_ENVELOPE_REPLAY_TTL >= SERVICE_ENVELOPE_MAX_LIFETIME + SERVICE_ENVELOPE_LEEWAY


class _Jwks:
    """a sender's key set, counting fetches, rotating to ``after_refresh`` on a forced one."""

    def __init__(self, jwks, *, after_refresh=None) -> None:
        self.jwks = jwks
        self.after_refresh = after_refresh
        self.refreshes = 0

    async def get_jwks(self, *, refresh: bool = False):
        if refresh:
            self.refreshes += 1
            if self.after_refresh is not None:
                self.jwks = self.after_refresh
        return self.jwks


def _door(jwks_source, *, guard=None, today=None):
    from datetime import date

    from threetears.core.security import ServiceEnvelopeDoor
    from threetears.core.security.door_proof import DoorProofSwitch, DoorWindow

    window = DoorWindow(door="claim_grant", opened=date(2026, 10, 7), required_from=date(2027, 1, 5))
    switch = DoorProofSwitch(window, today=(lambda: today) if today else (lambda: date(2026, 10, 8)))
    return ServiceEnvelopeDoor(
        switch=switch,
        issuer="identity-core",
        audience="hub.claim_grant",
        jwks=jwks_source,
        replay_guard=guard if guard is not None else FakeReplayGuard(),
    )


def _with_envelope(envelope):
    return {**_BODY, SERVICE_ENVELOPE_FIELD: envelope}


class TestServiceEnvelopeDoor:
    async def test_a_verified_envelope_is_admitted_once(self) -> None:
        priv, jwks = _keys()
        door = _door(_Jwks(jwks))
        body = _with_envelope(_mint(priv))
        assert await door.admit(body, op="ensure") is True
        assert await door.admit(body, op="ensure") is False

    async def test_a_forged_envelope_is_refused(self) -> None:
        forger, _ = _keys()
        _, jwks = _keys()
        door = _door(_Jwks(jwks))
        assert await door.admit(_with_envelope(_mint(forger)), op="ensure") is False

    async def test_an_envelope_for_another_op_is_refused(self) -> None:
        priv, jwks = _keys()
        door = _door(_Jwks(jwks))
        assert await door.admit(_with_envelope(_mint(priv, op="revoke")), op="ensure") is False

    async def test_a_rotated_key_is_fetched_afresh_once(self) -> None:
        old_priv, old_jwks = _keys("idc-old")
        new_priv, new_jwks = _keys("idc-new")
        source = _Jwks(old_jwks, after_refresh=new_jwks)
        door = _door(source)
        assert await door.admit(_with_envelope(_mint(new_priv, kid="idc-new")), op="ensure") is True
        assert source.refreshes == 1

    async def test_a_known_kid_is_not_refetched(self) -> None:
        priv, jwks = _keys()
        source = _Jwks(jwks)
        await _door(source).admit(_with_envelope(_mint(priv)), op="ensure")
        assert source.refreshes == 0

    async def test_no_envelope_is_admitted_while_the_window_is_open(self) -> None:
        _, jwks = _keys()
        assert await _door(_Jwks(jwks)).admit(_BODY, op="ensure") is True

    async def test_no_envelope_is_refused_from_the_windows_date(self) -> None:
        from datetime import date

        _, jwks = _keys()
        door = _door(_Jwks(jwks), today=date(2027, 1, 5))
        assert await door.admit(_BODY, op="ensure") is False

    async def test_an_unreachable_ledger_refuses_rather_than_admits(self) -> None:
        from threetears.nats.errors import KvError

        priv, jwks = _keys()
        door = _door(_Jwks(jwks), guard=FakeReplayGuard(record_error=KvError("down")))
        assert await door.admit(_with_envelope(_mint(priv)), op="ensure") is False

    async def test_no_key_set_available_refuses(self) -> None:
        class _Unavailable:
            async def get_jwks(self, *, refresh: bool = False):
                raise ServiceEnvelopeError("no trusted key set")

        priv, _ = _keys()
        assert await _door(_Unavailable()).admit(_with_envelope(_mint(priv)), op="ensure") is False

    def test_a_ledger_not_sized_for_the_leeway_fails_construction(self) -> None:
        _, jwks = _keys()
        with pytest.raises(ValueError, match="sized for"):
            _door(_Jwks(jwks), guard=FakeReplayGuard(verifier_future_tolerance=timedelta(0)))


class TestServiceEnvelopeSigner:
    def test_a_sealed_body_verifies_at_its_door(self) -> None:
        from threetears.core.security import ServiceEnvelopeSigner

        priv, jwks = _keys()
        signer = ServiceEnvelopeSigner(signing_key=priv, kid="idc-1", issuer="identity-core")
        sealed = signer.seal(audience="hub.claim_grant", op="ensure", body=_BODY)
        assert set(sealed) == {*_BODY, SERVICE_ENVELOPE_FIELD}
        _verify(sealed[SERVICE_ENVELOPE_FIELD], jwks, body=sealed)


class TestAdmitMessage:
    async def test_a_field_the_receiver_does_not_model_still_counts_toward_the_hash(self) -> None:
        import json

        priv, jwks = _keys()
        door = _door(_Jwks(jwks))
        newer = {**_BODY, "added_by_a_newer_sender": True}
        signer_body = {**newer, SERVICE_ENVELOPE_FIELD: _mint(priv, body=newer)}
        assert await door.admit_message(json.dumps(signer_body).encode(), op="ensure") is True

    async def test_a_body_that_is_not_a_json_object_is_refused(self) -> None:
        _, jwks = _keys()
        door = _door(_Jwks(jwks))
        assert await door.admit_message(b"[1, 2]", op="ensure") is False
        assert await door.admit_message(b"\xff", op="ensure") is False


def _seed_and_public():
    import base64

    from cryptography.hazmat.primitives import serialization

    priv, pub = generate_signing_keypair()
    raw = priv.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    return base64.b64encode(raw).decode(), pub


class TestPinnedSender:
    """a sender with no JWKS of its own: a seed on its side, the public half pinned at the door."""

    async def test_an_envelope_from_the_seed_verifies_under_the_pinned_half(self) -> None:
        from threetears.core.security import PinnedEnvelopeJwks, ServiceEnvelopeSigner

        seed, pub = _seed_and_public()
        signer = ServiceEnvelopeSigner.from_seed_text(seed, issuer="identity-core", what="SEED")
        door = _door(PinnedEnvelopeJwks(pub))
        sealed = signer.seal(audience="hub.claim_grant", op="ensure", body=_BODY)
        assert await door.admit(sealed, op="ensure") is True

    async def test_without_a_pinned_key_no_envelope_verifies(self) -> None:
        from threetears.core.security import PinnedEnvelopeJwks, ServiceEnvelopeSigner

        seed, _ = _seed_and_public()
        signer = ServiceEnvelopeSigner.from_seed_text(seed, issuer="identity-core", what="SEED")
        sealed = signer.seal(audience="hub.claim_grant", op="ensure", body=_BODY)
        assert await _door(PinnedEnvelopeJwks(None)).admit(sealed, op="ensure") is False

    def test_a_malformed_seed_names_the_setting_and_not_the_value(self) -> None:
        from threetears.core.security import ServiceEnvelopeSigner

        with pytest.raises(ValueError, match="GATEWAY_SEED is not") as caught:
            ServiceEnvelopeSigner.from_seed_text("c2hvcnQ=", issuer="gateway", what="GATEWAY_SEED")
        assert "c2hvcnQ" not in str(caught.value)


class TestSignerFromSecretRef:
    @staticmethod
    def _window():
        from datetime import date

        from threetears.core.security.door_proof import DoorWindow

        return DoorWindow(door="hub.claim_grant", opened=date(2026, 10, 8), required_from=date(2027, 1, 5))

    def test_a_provisioned_seed_gives_a_signer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from threetears.core.security import ServiceEnvelopeSigner

        seed, pub = _seed_and_public()
        monkeypatch.setenv("TEST_ENVELOPE_SEED", seed)
        signer = ServiceEnvelopeSigner.from_secret_ref(
            "env://TEST_ENVELOPE_SEED", issuer="svc", what="TEST_ENVELOPE_SEED", window=self._window()
        )
        assert signer is not None
        assert signer.signing_key.public_key().public_bytes_raw() == pub.public_bytes_raw()

    def test_an_unprovisioned_seed_gives_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from threetears.core.security import ServiceEnvelopeSigner

        monkeypatch.delenv("TEST_ENVELOPE_SEED", raising=False)
        assert (
            ServiceEnvelopeSigner.from_secret_ref(
                "env://TEST_ENVELOPE_SEED", issuer="svc", what="TEST_ENVELOPE_SEED", window=self._window()
            )
            is None
        )


class TestInProcessLedger:
    async def test_an_id_is_spent_once(self) -> None:
        from datetime import UTC, datetime

        from threetears.core.security import InProcessEnvelopeLedger

        ledger = InProcessEnvelopeLedger()
        now = datetime.now(UTC)
        assert await ledger.record_unique("a", issued_at=now) is True
        assert await ledger.record_unique("a", issued_at=now) is False

    async def test_an_id_is_forgotten_only_once_its_envelope_can_no_longer_be_accepted(self) -> None:
        from datetime import UTC, datetime

        from threetears.core.security import InProcessEnvelopeLedger

        clock = {"now": datetime(2026, 10, 8, 12, 0, tzinfo=UTC)}
        ledger = InProcessEnvelopeLedger(clock=lambda: clock["now"])
        issued = clock["now"]
        await ledger.record_unique("a", issued_at=issued)
        clock["now"] = issued + SERVICE_ENVELOPE_REPLAY_TTL
        assert await ledger.record_unique("a", issued_at=issued) is False
        clock["now"] = issued + SERVICE_ENVELOPE_REPLAY_TTL + timedelta(seconds=1)
        assert await ledger.record_unique("b", issued_at=clock["now"]) is True
        assert await ledger.record_unique("a", issued_at=issued) is True  # its envelope expired long ago

    def test_a_verifier_wider_than_the_leeway_is_refused(self) -> None:
        from threetears.core.security import InProcessEnvelopeLedger

        with pytest.raises(ValueError, match="sized for"):
            InProcessEnvelopeLedger().require_covers(SERVICE_ENVELOPE_LEEWAY + timedelta(seconds=1))

    async def test_a_door_on_the_ledger_refuses_a_replay(self) -> None:
        from threetears.core.security import InProcessEnvelopeLedger

        priv, jwks = _keys()
        door = _door(_Jwks(jwks), guard=InProcessEnvelopeLedger())
        body = _with_envelope(_mint(priv))
        assert await door.admit(body, op="ensure") is True
        assert await door.admit(body, op="ensure") is False


class TestHubJwksSource:
    async def test_it_serves_the_providers_set_and_asks_it_to_refresh(self) -> None:
        from threetears.core.security import HubJwksEnvelopeSource

        _, jwks = _keys()

        class _Provider:
            refreshed = 0

            def __call__(self):
                return jwks

            async def refresh_now(self) -> bool:
                self.refreshed += 1
                return True

        provider = _Provider()
        source = HubJwksEnvelopeSource(provider)
        assert await source.get_jwks() == jwks
        await source.get_jwks(refresh=True)
        assert provider.refreshed == 1


class TestSealedModel:
    def test_a_sealed_model_verifies_over_what_goes_on_the_wire(self) -> None:
        import json

        from pydantic import BaseModel

        from threetears.core.security import ServiceEnvelopeSigner

        class _Wire(BaseModel):
            pod_id: str
            service_envelope: str | None = None

        priv, jwks = _keys()
        signer = ServiceEnvelopeSigner(signing_key=priv, kid="idc-1", issuer="identity-core")
        sent = json.loads(signer.sealed(_Wire(pod_id="p"), audience="hub.claim_grant", op="ensure").model_dump_json())
        _verify(sent[SERVICE_ENVELOPE_FIELD], jwks, body=sent)

    def test_a_model_without_the_field_is_refused(self) -> None:
        from pydantic import BaseModel

        from threetears.core.security import ServiceEnvelopeSigner

        class _NoField(BaseModel):
            pod_id: str

        priv, _ = _keys()
        signer = ServiceEnvelopeSigner(signing_key=priv, kid="idc-1", issuer="identity-core")
        with pytest.raises(ValueError, match="no service_envelope field"):
            signer.sealed(_NoField(pod_id="p"), audience="a", op="o")


class TestARequiredDoor:
    async def test_a_message_without_an_envelope_is_refused_and_a_signed_one_admitted(self) -> None:
        from threetears.core.security import ServiceEnvelopeDoor

        priv, jwks = _keys()
        door = ServiceEnvelopeDoor.required(
            door="hub.merge",
            issuer="identity-core",
            audience="hub.claim_grant",
            jwks=_Jwks(jwks),
            replay_guard=FakeReplayGuard(),
        )
        assert door.door == "hub.merge"
        assert await door.admit(_BODY, op="ensure") is False
        assert await door.admit(_with_envelope(_mint(priv)), op="ensure") is True

    def test_a_door_with_neither_switch_nor_name_is_refused(self) -> None:
        from threetears.core.security import ServiceEnvelopeDoor

        _, jwks = _keys()
        with pytest.raises(ValueError, match="needs a name"):
            ServiceEnvelopeDoor(switch=None, issuer="i", audience="a", jwks=_Jwks(jwks), replay_guard=FakeReplayGuard())
