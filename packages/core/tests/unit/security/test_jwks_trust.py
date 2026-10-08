"""a fetched JWKS is never trusted blind: thumbprint = kid, add-only merge, a signed statement, a gated refresh.

Any client in the account can get bytes delivered onto a verifier's JWKS reply inbox. These tests
script the forged replies such a client would send -- an attacker key beside the real ones, a key
renamed, the real key dropped, a statement signed by the wrong key -- and check each is refused at
the provider, and that the publisher's real reply is still taken.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from datetime import date, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from threetears.core.security.door_proof import DoorProofSwitch, DoorWindow
from threetears.core.security.identity_token import (
    IdentityClaims,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
    verify_identity_token,
)
from threetears.core.security.jwks_provider import HUB_JWKS_ISSUER, CachedHubJwksProvider
from threetears.core.security.jwks_trust import (
    HUB_JWKS_ROOT_PUBLIC_KEY_ENV,
    HUB_JWKS_STATEMENT_WINDOW,
    MAX_STATEMENT_LIFETIME,
    THUMBPRINT_WINDOW,
    JWKS_STATEMENT_FIELD,
    ROTATION_OVERLAP,
    JwksKeyring,
    JwksStatementError,
    RefreshGate,
    hub_jwks_root_public_key_from_env,
    root_public_key_from_text,
    sign_jwks_statement,
    thumbprint_mismatches,
    verify_jwks_statement,
)

_HUB_SIGNING, _HUB_PUBLIC = generate_signing_keypair()
_HUB_KID = jwk_thumbprint(_HUB_PUBLIC)
_ATTACKER_SIGNING, _ATTACKER_PUBLIC = generate_signing_keypair()
_ATTACKER_KID = jwk_thumbprint(_ATTACKER_PUBLIC)
_ROOT = Ed25519PrivateKey.generate()
_WRONG_ROOT = Ed25519PrivateKey.generate()


def _keys(*publics: Any) -> list[dict[str, Any]]:
    keys: list[dict[str, Any]] = build_jwks({jwk_thumbprint(p): p for p in publics})["keys"]
    return keys


def _reply(keys: list[dict[str, Any]], *, signed_by: Ed25519PrivateKey | None = None) -> bytes:
    document: dict[str, Any] = {"keys": keys}
    if signed_by is not None:
        document[JWKS_STATEMENT_FIELD] = sign_jwks_statement(
            keys, signing_key=signed_by, issuer=HUB_JWKS_ISSUER, now=int(time.time()), ttl=timedelta(minutes=5)
        )
    return json.dumps(document).encode("utf-8")


def _client(*replies: bytes) -> MagicMock:
    nc = MagicMock()
    nc.request_raw = AsyncMock(side_effect=list(replies))
    return nc


def _switch(*, required: bool, window: DoorWindow = HUB_JWKS_STATEMENT_WINDOW) -> DoorProofSwitch:
    today = window.required_from if required else window.opened
    return DoorProofSwitch(window, today=lambda: today)


def _token(signing: Ed25519PrivateKey, kid: str) -> str:
    now = int(time.time())
    claims = IdentityClaims(
        iss="aibots-hub",
        sub="019470a8-b5c3-7def-8123-0000000000aa",
        pod_id="019470a8-b5c3-7def-8123-0000000000bb",
        sid="019470a8-b5c3-7def-8123-0000000000cc",
        iat=now,
        exp=now + 300,
        customer_id="019470a8-b5c3-7def-8123-0000000000dd",
    )
    return sign_identity_token(claims, signing_key=signing, kid=kid)


def _kids(provider: CachedHubJwksProvider) -> set[str]:
    return {entry["kid"] for entry in provider()["keys"]}


class TestThumbprintIsKid:
    """an entry whose key material does not hash to its kid refuses the whole reply."""

    def test_real_keys_pass(self) -> None:
        assert thumbprint_mismatches(_keys(_HUB_PUBLIC, _ATTACKER_PUBLIC)) == []

    def test_a_renamed_key_is_a_mismatch(self) -> None:
        renamed = _keys(_ATTACKER_PUBLIC)[0] | {"kid": _HUB_KID}
        assert thumbprint_mismatches([renamed]) == ["entry 0: kid is not the key's thumbprint"]

    @pytest.mark.parametrize(
        "entry",
        [{"kty": "OKP", "crv": "Ed25519", "x": "abc"}, {"kid": "k1", "kty": "OKP", "crv": "Ed25519", "x": "abc"}, "k"],
    )
    def test_a_malformed_entry_is_a_mismatch(self, entry: Any) -> None:
        assert len(thumbprint_mismatches([entry])) == 1

    async def test_from_the_windows_date_an_unsigned_reply_with_a_renamed_key_changes_nothing(self) -> None:
        renamed = _keys(_ATTACKER_PUBLIC)[0] | {"kid": "the-hubs-next-kid"}
        provider = CachedHubJwksProvider(
            _client(_reply(_keys(_HUB_PUBLIC)), _reply([*_keys(_HUB_PUBLIC), renamed])),
            request_timeout_seconds=1.0,
            statement_switch=_switch(required=False),
            thumbprint_switch=_switch(required=True, window=THUMBPRINT_WINDOW),
        )
        await provider.refresh()
        await provider.refresh()
        assert _kids(provider) == {_HUB_KID}

    async def test_before_the_windows_date_a_misnamed_key_is_taken_and_counted(self) -> None:
        """a publisher still on a release that names a key otherwise is not refused before the date."""
        renamed = _keys(_HUB_PUBLIC)[0] | {"kid": "a-pre-thumbprint-kid"}
        thumbprint = _switch(required=False, window=THUMBPRINT_WINDOW)
        provider = CachedHubJwksProvider(
            _client(_reply([renamed])),
            request_timeout_seconds=1.0,
            statement_switch=_switch(required=False),
            thumbprint_switch=thumbprint,
        )
        await provider.refresh()
        assert _kids(provider) == {"a-pre-thumbprint-kid"}
        (absent,) = thumbprint.absent_senders()
        assert absent.door == "hub.jwks.thumbprint"

    async def test_a_signed_reply_needs_no_thumbprint_check_its_statement_covers_the_keys(self) -> None:
        renamed = _keys(_HUB_PUBLIC)[0] | {"kid": "a-pre-thumbprint-kid"}
        provider = CachedHubJwksProvider(
            _client(_reply([renamed], signed_by=_ROOT)),
            request_timeout_seconds=1.0,
            root_public_key=_ROOT.public_key(),
            statement_switch=_switch(required=True),
            thumbprint_switch=_switch(required=True, window=THUMBPRINT_WINDOW),
        )
        await provider.refresh()
        assert _kids(provider) == {"a-pre-thumbprint-kid"}


class TestAddOnlyMerge:
    """a refresh adds keys and never replaces the cache; a key is retired only after the overlap."""

    async def test_a_signed_reply_missing_a_key_that_verified_keeps_it(self) -> None:
        provider = CachedHubJwksProvider(
            _client(_reply(_keys(_HUB_PUBLIC), signed_by=_ROOT), _reply(_keys(_ATTACKER_PUBLIC), signed_by=_ROOT)),
            request_timeout_seconds=1.0,
            root_public_key=_ROOT.public_key(),
            statement_switch=_switch(required=True),
        )
        await provider.refresh()
        token = _token(_HUB_SIGNING, _HUB_KID)
        verify_identity_token(token, jwks=provider(), issuer="aibots-hub")
        await provider.refresh()  # the publisher's next set omits the key
        verify_identity_token(token, jwks=provider(), issuer="aibots-hub")
        assert _HUB_KID in _kids(provider)

    async def test_an_unsigned_forgery_inside_the_window_is_gone_at_the_next_genuine_reply(self) -> None:
        """a reply nothing authenticated replaces the cache; it is never merged add-only.

        The forged key is named by its own correct thumbprint, so the thumbprint check passes it,
        and the window still accepts an unsigned reply. Merged add-only it would outlive every
        genuine reply for the rotation overlap; replaced, it is gone at the next one.
        """
        provider = CachedHubJwksProvider(
            _client(
                _reply(_keys(_HUB_PUBLIC)),
                _reply(_keys(_HUB_PUBLIC, _ATTACKER_PUBLIC)),
                _reply(_keys(_HUB_PUBLIC)),
            ),
            request_timeout_seconds=1.0,
            statement_switch=_switch(required=False),
        )
        await provider.refresh()
        await provider.refresh()  # the forgery, accepted while the window is open
        assert _ATTACKER_KID in _kids(provider)
        await provider.refresh()  # the hub's own next reply
        assert _kids(provider) == {_HUB_KID}
        with pytest.raises(Exception, match="kid"):
            verify_identity_token(_token(_ATTACKER_SIGNING, _ATTACKER_KID), jwks=provider(), issuer="aibots-hub")

    def test_a_key_is_retired_only_when_missing_for_the_overlap(self) -> None:
        now = [0.0]
        ring = JwksKeyring(overlap=timedelta(hours=1), clock=lambda: now[0])
        ring.adopt(_keys(_HUB_PUBLIC), verified=True)
        now[0] = 10.0
        first = ring.adopt(_keys(_ATTACKER_PUBLIC), verified=True)
        assert first.held == (_HUB_KID,) and first.retired == ()
        now[0] = 10.0 + 3599.0
        assert ring.adopt(_keys(_ATTACKER_PUBLIC), verified=True).held == (_HUB_KID,)
        now[0] = 10.0 + 3600.0
        assert ring.adopt(_keys(_ATTACKER_PUBLIC), verified=True).retired == (_HUB_KID,)
        assert _HUB_KID not in ring

    def test_a_key_that_comes_back_restarts_its_overlap(self) -> None:
        now = [0.0]
        ring = JwksKeyring(overlap=timedelta(hours=1), clock=lambda: now[0])
        ring.adopt(_keys(_HUB_PUBLIC), verified=True)
        now[0] = 100.0
        ring.adopt([], verified=True)
        now[0] = 200.0
        ring.adopt(_keys(_HUB_PUBLIC), verified=True)
        now[0] = 3700.0
        assert ring.adopt([], verified=True).held == (_HUB_KID,)

    def test_the_overlap_outlives_the_longest_platform_token(self) -> None:
        assert timedelta(hours=1) <= ROTATION_OVERLAP


class TestSignedStatement:
    """the publisher signs the exact key set with a root key the verifier pins."""

    def test_a_statement_over_the_keys_verifies(self) -> None:
        keys = _keys(_HUB_PUBLIC)
        statement = sign_jwks_statement(
            keys, signing_key=_ROOT, issuer=HUB_JWKS_ISSUER, now=int(time.time()), ttl=timedelta(minutes=5)
        )
        verify_jwks_statement(statement, keys, root_public_key=_ROOT.public_key(), issuer=HUB_JWKS_ISSUER)

    def test_a_statement_over_other_keys_is_refused(self) -> None:
        statement = sign_jwks_statement(
            _keys(_HUB_PUBLIC),
            signing_key=_ROOT,
            issuer=HUB_JWKS_ISSUER,
            now=int(time.time()),
            ttl=timedelta(minutes=5),
        )
        with pytest.raises(JwksStatementError, match="different key set"):
            verify_jwks_statement(
                statement,
                _keys(_HUB_PUBLIC, _ATTACKER_PUBLIC),
                root_public_key=_ROOT.public_key(),
                issuer=HUB_JWKS_ISSUER,
            )

    def test_a_statement_signed_by_another_key_is_refused(self) -> None:
        keys = _keys(_HUB_PUBLIC, _ATTACKER_PUBLIC)
        statement = sign_jwks_statement(
            keys, signing_key=_WRONG_ROOT, issuer=HUB_JWKS_ISSUER, now=int(time.time()), ttl=timedelta(minutes=5)
        )
        with pytest.raises(JwksStatementError, match="pinned root"):
            verify_jwks_statement(statement, keys, root_public_key=_ROOT.public_key(), issuer=HUB_JWKS_ISSUER)

    def test_an_expired_statement_is_refused(self) -> None:
        keys = _keys(_HUB_PUBLIC)
        statement = sign_jwks_statement(
            keys, signing_key=_ROOT, issuer=HUB_JWKS_ISSUER, now=int(time.time()) - 3600, ttl=timedelta(minutes=5)
        )
        with pytest.raises(JwksStatementError, match="ExpiredSignature"):
            verify_jwks_statement(statement, keys, root_public_key=_ROOT.public_key(), issuer=HUB_JWKS_ISSUER)

    def test_a_token_that_is_not_a_statement_is_refused(self) -> None:
        with pytest.raises(JwksStatementError, match="not a JWKS statement"):
            verify_jwks_statement(
                _token(_ROOT, jwk_thumbprint(_ROOT.public_key())),
                _keys(_HUB_PUBLIC),
                root_public_key=_ROOT.public_key(),
                issuer=HUB_JWKS_ISSUER,
            )


class TestForgedRepliesAtTheProvider:
    """the forged JWKS a pod would deliver: the real keys plus its own."""

    async def test_with_the_door_required_an_unsigned_forgery_is_refused_and_its_token_with_it(self) -> None:
        provider = CachedHubJwksProvider(
            _client(_reply(_keys(_HUB_PUBLIC), signed_by=_ROOT), _reply(_keys(_HUB_PUBLIC, _ATTACKER_PUBLIC))),
            request_timeout_seconds=1.0,
            root_public_key=_ROOT.public_key(),
            statement_switch=_switch(required=True),
        )
        await provider.refresh()
        await provider.refresh()
        assert _kids(provider) == {_HUB_KID}
        with pytest.raises(Exception, match="kid"):
            verify_identity_token(_token(_ATTACKER_SIGNING, _ATTACKER_KID), jwks=provider(), issuer="aibots-hub")

    async def test_a_forgery_signed_by_another_root_is_refused_in_either_mode(self) -> None:
        for required in (False, True):
            provider = CachedHubJwksProvider(
                _client(_reply(_keys(_HUB_PUBLIC, _ATTACKER_PUBLIC), signed_by=_WRONG_ROOT)),
                request_timeout_seconds=1.0,
                root_public_key=_ROOT.public_key(),
                statement_switch=_switch(required=required),
            )
            await provider.refresh()
            assert provider() == {"keys": []}
            assert provider.is_warmed is False

    async def test_the_publishers_signed_reply_is_taken(self) -> None:
        provider = CachedHubJwksProvider(
            _client(_reply(_keys(_HUB_PUBLIC), signed_by=_ROOT)),
            request_timeout_seconds=1.0,
            root_public_key=_ROOT.public_key(),
            statement_switch=_switch(required=True),
        )
        await provider.refresh()
        verify_identity_token(_token(_HUB_SIGNING, _HUB_KID), jwks=provider(), issuer="aibots-hub")

    async def test_with_the_door_required_and_no_root_given_every_reply_is_refused(self) -> None:
        provider = CachedHubJwksProvider(
            _client(_reply(_keys(_HUB_PUBLIC), signed_by=_ROOT)),
            request_timeout_seconds=1.0,
            statement_switch=_switch(required=True),
        )
        await provider.refresh()
        assert provider.is_warmed is False


class TestNoDowngrade:
    """once the hub has signed a reply, an unsigned one is refused, inside the window too."""

    async def test_an_attacker_key_under_its_own_thumbprint_is_not_adopted_after_a_signed_reply(self) -> None:
        provider = CachedHubJwksProvider(
            _client(_reply(_keys(_HUB_PUBLIC), signed_by=_ROOT), _reply(_keys(_HUB_PUBLIC, _ATTACKER_PUBLIC))),
            request_timeout_seconds=1.0,
            root_public_key=_ROOT.public_key(),
            statement_switch=_switch(required=False),
        )
        await provider.refresh()
        await provider.refresh()

        assert _kids(provider) == {_HUB_KID}
        with pytest.raises(Exception, match="kid"):
            verify_identity_token(_token(_ATTACKER_SIGNING, _ATTACKER_KID), jwks=provider(), issuer="aibots-hub")


class TestTheWindow:
    """before its date an unsigned reply is accepted and counted; from it, refused; traffic moves nothing."""

    async def test_an_unsigned_reply_before_the_date_is_accepted_and_counted(self) -> None:
        switch = _switch(required=False)
        provider = CachedHubJwksProvider(
            _client(*[_reply(_keys(_HUB_PUBLIC))] * 5),
            request_timeout_seconds=1.0,
            root_public_key=_ROOT.public_key(),
            statement_switch=switch,
        )
        for _ in range(5):
            await provider.refresh()
        assert _kids(provider) == {_HUB_KID}
        (absent,) = switch.absent_senders()
        assert absent.door == "hub.jwks.statement" and absent.count == 5

    def test_the_date_not_the_count_ends_it(self) -> None:
        today = [HUB_JWKS_STATEMENT_WINDOW.opened]
        switch = DoorProofSwitch(HUB_JWKS_STATEMENT_WINDOW, today=lambda: today[0])
        for _ in range(1000):
            switch.judge(presented=False, claimed_sender="hub")
        assert switch.mode.value == "optional"
        today[0] = HUB_JWKS_STATEMENT_WINDOW.required_from
        assert switch.mode.value == "required"

    def test_the_window_is_within_the_maximum(self) -> None:
        DoorWindow(
            door=HUB_JWKS_STATEMENT_WINDOW.door,
            opened=HUB_JWKS_STATEMENT_WINDOW.opened,
            required_from=HUB_JWKS_STATEMENT_WINDOW.required_from,
        )
        assert HUB_JWKS_STATEMENT_WINDOW.opened == date(2026, 10, 7)


class TestRootKeyFromTheEnvironment:
    def test_unset_is_none(self) -> None:
        assert hub_jwks_root_public_key_from_env({}) is None

    def test_a_base64url_key_is_read(self) -> None:
        raw = _ROOT.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        encoded = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        key = hub_jwks_root_public_key_from_env({HUB_JWKS_ROOT_PUBLIC_KEY_ENV: encoded})
        assert key is not None and jwk_thumbprint(key) == jwk_thumbprint(_ROOT.public_key())

    def test_the_standard_base64_form_is_read_too(self) -> None:
        raw = _ROOT.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        key = root_public_key_from_text(base64.b64encode(raw).decode("ascii"), what="setting")
        assert jwk_thumbprint(key) == jwk_thumbprint(_ROOT.public_key())

    def test_a_malformed_key_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match=HUB_JWKS_ROOT_PUBLIC_KEY_ENV):
            hub_jwks_root_public_key_from_env({HUB_JWKS_ROOT_PUBLIC_KEY_ENV: "not-a-key"})


class TestNoAttackerTimedRefresh:
    """a burst of unknown-kid triggers costs one fetch per window."""

    async def test_a_hundred_concurrent_triggers_cause_one_fetch(self) -> None:
        nc = MagicMock()

        async def slow_fetch(**_: Any) -> bytes:
            await asyncio.sleep(0.01)
            return _reply(_keys(_HUB_PUBLIC))

        nc.request_raw = AsyncMock(side_effect=slow_fetch)
        provider = CachedHubJwksProvider(nc, request_timeout_seconds=1.0, statement_switch=_switch(required=False))

        ran = await asyncio.gather(*(provider.refresh_now() for _ in range(100)))

        assert sum(ran) == 1
        assert nc.request_raw.await_count == 1

    async def test_the_gate_runs_once_per_window(self) -> None:
        now = [0.0]
        runs = []

        async def refresh() -> None:
            runs.append(now[0])

        gate = RefreshGate(refresh, window_seconds=5.0, clock=lambda: now[0])
        assert await gate.run() is True
        now[0] = 4.9
        assert await gate.run() is False
        now[0] = 5.0
        assert await gate.run() is True
        assert runs == [0.0, 5.0]


class TestProvenance:
    """a key only an unsigned reply vouched for never outlives the next genuine reply, signed or not."""

    async def test_an_unsigned_forgery_then_a_signed_genuine_reply_leaves_no_attacker_key(self) -> None:
        """the sequence a forger wins inside the window: its key, then the hub's first signed reply.

        The signed reply holds a key it omits through the rotation overlap only when a verified
        reply brought that key in; the attacker's was brought by an unsigned forgery, so it goes.
        """
        provider = CachedHubJwksProvider(
            _client(_reply(_keys(_HUB_PUBLIC, _ATTACKER_PUBLIC)), _reply(_keys(_HUB_PUBLIC), signed_by=_ROOT)),
            request_timeout_seconds=1.0,
            root_public_key=_ROOT.public_key(),
            statement_switch=_switch(required=False),
        )
        await provider.refresh()  # the forgery, taken while the window is open
        assert _ATTACKER_KID in _kids(provider)
        await provider.refresh()  # the hub's first signed reply

        assert _kids(provider) == {_HUB_KID}
        with pytest.raises(Exception, match="kid"):
            verify_identity_token(_token(_ATTACKER_SIGNING, _ATTACKER_KID), jwks=provider(), issuer="aibots-hub")

    def test_the_keyring_holds_only_what_a_verified_reply_brought(self) -> None:
        ring = JwksKeyring(overlap=timedelta(hours=1))
        ring.adopt(_keys(_HUB_PUBLIC), verified=True)
        ring.adopt(_keys(_HUB_PUBLIC, _ATTACKER_PUBLIC), verified=False)

        merged = ring.adopt([], verified=True)

        assert merged.retired == (_ATTACKER_KID,)
        assert merged.held == (_HUB_KID,)


class TestTheReplyShape:
    @pytest.mark.parametrize("bad", ['"x"', '{"a": 1}', "7"])
    async def test_a_reply_whose_keys_is_not_a_list_changes_nothing_and_does_not_warm(self, bad: str) -> None:
        provider = CachedHubJwksProvider(
            _client(f'{{"keys": {bad}}}'.encode("utf-8")),
            request_timeout_seconds=1.0,
            statement_switch=_switch(required=False),
        )
        await provider.refresh()
        assert provider() == {"keys": []}
        assert provider.is_warmed is False


class TestNoReplay:
    """a stored genuine reply cannot be replayed to bring back a key the hub has rotated out."""

    async def test_a_statement_older_than_one_already_verified_is_refused(self) -> None:
        now = int(time.time())

        def signed(keys: list[dict[str, Any]], at: int) -> bytes:
            statement = sign_jwks_statement(
                keys, signing_key=_ROOT, issuer=HUB_JWKS_ISSUER, now=at, ttl=timedelta(minutes=5)
            )
            return json.dumps({"keys": keys, JWKS_STATEMENT_FIELD: statement}).encode("utf-8")

        provider = CachedHubJwksProvider(
            _client(signed(_keys(_HUB_PUBLIC), now), signed(_keys(_HUB_PUBLIC, _ATTACKER_PUBLIC), now - 60)),
            request_timeout_seconds=1.0,
            root_public_key=_ROOT.public_key(),
            statement_switch=_switch(required=True),
        )
        await provider.refresh()
        await provider.refresh()  # the replay, signed a minute before the reply already taken

        assert _kids(provider) == {_HUB_KID}

    def test_a_statement_claiming_a_longer_life_than_the_maximum_is_refused(self) -> None:
        keys = _keys(_HUB_PUBLIC)
        statement = sign_jwks_statement(
            keys,
            signing_key=_ROOT,
            issuer=HUB_JWKS_ISSUER,
            now=int(time.time()),
            ttl=MAX_STATEMENT_LIFETIME + timedelta(minutes=1),
        )
        with pytest.raises(JwksStatementError, match="longer life"):
            verify_jwks_statement(statement, keys, root_public_key=_ROOT.public_key(), issuer=HUB_JWKS_ISSUER)


class TestTheMissingRootIsSaid:
    def test_a_provider_built_without_a_root_key_warns_naming_the_date(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            CachedHubJwksProvider(_client(), request_timeout_seconds=1.0, statement_switch=_switch(required=False))
        assert any("no pinned root key" in r.getMessage() and "2027-01-05" in r.getMessage() for r in caplog.records)
