"""Refresh-token rotation and reuse detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.iam.rotation import (
    RefreshTokenLedger,
    ReplayGraceCache,
    ReplayGraceEntry,
    RotationError,
    SessionLifetimeCaps,
    SessionRevoker,
    rotate_refresh_token,
)
from threetears.iam.tokens import (
    HmacSigner,
    HmacVerifier,
    TokenPair,
    TokenType,
    mint_token_pair,
    new_session_id,
    verify_session_token,
)

_SECRET = "s" * 48
_ISSUER = "https://issuer.example"
_AUDIENCE = "platform:internal"

_SIGNER = HmacSigner(_SECRET)
_VERIFIER = HmacVerifier(_SECRET, issuer=_ISSUER, audience=_AUDIENCE)
_THUMBPRINT = "thumbprint-abc"


class _Ledger:
    """In-memory single-use ledger.

    # parity-with: threetears.iam.rotation.RefreshTokenLedger
    """

    def __init__(self) -> None:
        self.spent: set[str] = set()

    async def redeem(self, jti: str) -> bool:
        if jti in self.spent:
            return False
        self.spent.add(jti)
        return True


class _Revoker:
    """In-memory session revoker.

    # parity-with: threetears.iam.rotation.SessionRevoker
    """

    def __init__(self) -> None:
        self.revoked: set[str] = set()

    async def revoke_session(self, sid: str) -> None:
        self.revoked.add(sid)

    async def is_session_revoked(self, sid: str) -> bool:
        return sid in self.revoked


class _GraceCache:
    """In-memory replay-grace cache with no TTL of its own -- a test drops entries by hand.

    # parity-with: threetears.iam.rotation.ReplayGraceCache
    """

    def __init__(self) -> None:
        self.entries: dict[str, ReplayGraceEntry] = {}

    async def remember(self, jti: str, *, pair: TokenPair, holder_key: str) -> None:
        self.entries[jti] = ReplayGraceEntry(pair=pair, holder_key=holder_key)

    async def recall(self, jti: str) -> ReplayGraceEntry | None:
        return self.entries.get(jti)


def _pair(**overrides: Any) -> TokenPair:
    """Mint a starting pair, with any minting argument overridable per test.

    Holder-key bound by default: sessions bind at issuance, so an unbound pair is the
    exceptional case a test must ask for explicitly (``cnf=None``).
    """
    kwargs: dict[str, Any] = {
        "subject": "user-1",
        "session_id": new_session_id(),
        "issuer": _ISSUER,
        "audience": _AUDIENCE,
        "signer": _SIGNER,
        "step_up_window": 300,
        "cnf": _THUMBPRINT,
    }
    kwargs.update(overrides)
    return mint_token_pair(**kwargs)


def test_the_doubles_satisfy_their_protocols() -> None:
    assert isinstance(_Ledger(), RefreshTokenLedger)
    assert isinstance(_Revoker(), SessionRevoker)


async def test_rotation_issues_a_new_pair() -> None:
    original = _pair()
    rotated = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=_Ledger(),
        holder_key=_THUMBPRINT,
    )
    assert rotated.access_token != original.access_token
    assert rotated.refresh_token != original.refresh_token


async def test_the_session_survives_rotation() -> None:
    original = _pair()
    rotated = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=_Ledger(),
        holder_key=_THUMBPRINT,
    )
    # sid persists so revoking a session still revokes everything descended from it.
    assert rotated.claims.sid == original.claims.sid


async def test_auth_time_is_carried_forward_unchanged() -> None:
    # The property that keeps step-up meaningful: letting auth_time move would mean every
    # refresh silently re-satisfies it, so a stolen refresh token becomes a way to perform
    # sensitive actions without ever proving possession of a credential.
    started = int((datetime.now(UTC) - timedelta(hours=6)).timestamp())
    original = _pair(auth_time=started, session_started_at=started)
    rotated = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=_Ledger(),
        holder_key=_THUMBPRINT,
    )
    assert rotated.claims.auth_time == started
    assert rotated.claims.session_started_at == started
    assert rotated.claims.iat > started


async def test_impersonation_context_is_carried_forward() -> None:
    original = _pair(act="admin-1", act_reason="impersonation", act_restriction="view")
    rotated = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=_Ledger(),
        holder_key=_THUMBPRINT,
    )
    # A refresh never re-derives the acting identity -- it proves continuity only.
    assert rotated.claims.act == "admin-1"
    assert rotated.claims.act_restriction == "view"


async def test_reuse_is_rejected_and_revokes_the_family() -> None:
    original = _pair()
    ledger, revoker = _Ledger(), _Revoker()
    await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=ledger,
        revoker=revoker,
        holder_key=_THUMBPRINT,
    )
    with pytest.raises(RotationError, match="already been used"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
            revoker=revoker,
            holder_key=_THUMBPRINT,
        )
    # Two parties held the same token; forcing re-authentication is something the real user
    # can do and the thief cannot.
    assert original.claims.sid in revoker.revoked


async def test_a_revoked_session_cannot_refresh() -> None:
    original = _pair()
    revoker = _Revoker()
    await revoker.revoke_session(original.claims.sid)
    with pytest.raises(RotationError, match="revoked"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
            revoker=revoker,
        )


async def test_an_access_token_cannot_be_used_to_refresh() -> None:
    original = _pair()
    with pytest.raises(RotationError, match="not trustworthy"):
        await rotate_refresh_token(
            original.access_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
        )


async def test_a_forged_token_is_rejected() -> None:
    forged = mint_token_pair(
        subject="user-1",
        session_id=new_session_id(),
        issuer=_ISSUER,
        audience=_AUDIENCE,
        signer=HmacSigner("a-completely-different-secret-value-here"),
    )
    with pytest.raises(RotationError, match="not trustworthy"):
        await rotate_refresh_token(forged.refresh_token, verifier=_VERIFIER, signer=_SIGNER, ledger=_Ledger())


async def test_the_absolute_lifetime_cap_ends_a_session() -> None:
    # Without this a session refreshes forever and "deprovisioned" means nothing.
    started = int((datetime.now(UTC) - timedelta(days=40)).timestamp())
    original = _pair(auth_time=started, session_started_at=started)
    with pytest.raises(RotationError, match="absolute lifetime"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
            lifetime_caps=SessionLifetimeCaps(absolute=timedelta(days=30)),
        )


async def test_a_session_within_the_absolute_cap_still_refreshes() -> None:
    started = int((datetime.now(UTC) - timedelta(days=5)).timestamp())
    original = _pair(auth_time=started, session_started_at=started)
    await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=_Ledger(),
        lifetime_caps=SessionLifetimeCaps(absolute=timedelta(days=30)),
        holder_key=_THUMBPRINT,
    )


async def test_the_inactivity_timeout_ends_a_session() -> None:
    original = _pair(now=datetime.now(UTC) - timedelta(hours=10))
    with pytest.raises(RotationError, match="idle"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
            lifetime_caps=SessionLifetimeCaps(inactivity=timedelta(hours=1)),
        )


async def test_a_bound_session_requires_a_proof() -> None:
    original = _pair(cnf="thumbprint-abc")
    with pytest.raises(RotationError, match="no proof of possession"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
        )


async def test_a_wrong_holder_key_is_rejected() -> None:
    original = _pair(cnf="thumbprint-abc")
    with pytest.raises(RotationError, match="does not match"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
            holder_key="thumbprint-xyz",
        )


async def test_a_wrong_holder_key_does_not_burn_the_token() -> None:
    # THE subtle one. If a mismatched key consumed the jti, anyone holding a COPY of the
    # refresh token bytes could deny service to the legitimate holder at will, without ever
    # possessing the key. The mismatch must cost the attacker everything and the victim
    # nothing.
    original = _pair(cnf="thumbprint-abc")
    ledger = _Ledger()
    with pytest.raises(RotationError):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
            holder_key="thumbprint-xyz",
        )
    assert not ledger.spent
    # The real holder can still exchange it.
    rotated = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=ledger,
        holder_key="thumbprint-abc",
    )
    assert rotated.claims.cnf == "thumbprint-abc"


async def test_a_revoked_session_does_not_burn_the_token_either() -> None:
    # Same reasoning as the holder-key case: nothing above the redemption step may consume a
    # token it then refuses to exchange.
    original = _pair()
    revoker = _Revoker()
    await revoker.revoke_session(original.claims.sid)
    ledger = _Ledger()
    with pytest.raises(RotationError):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
            revoker=revoker,
        )
    assert not ledger.spent


async def test_the_rotated_token_verifies_and_is_a_refresh_token() -> None:
    original = _pair()
    rotated = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=_Ledger(),
        holder_key=_THUMBPRINT,
    )
    verify_session_token(rotated.refresh_token, verifier=_VERIFIER, expected_type=TokenType.REFRESH)
    verify_session_token(rotated.access_token, verifier=_VERIFIER, expected_type=TokenType.ACCESS)


async def test_rotation_chains() -> None:
    # Every link works because the binding carries forward: each rotated pair is minted with
    # the same cnf the redeemed token proved, so the chain never produces an unrefreshable
    # (unbound) token.
    ledger = _Ledger()
    current = _pair()
    for _ in range(5):
        current = await rotate_refresh_token(
            current.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
            holder_key=_THUMBPRINT,
        )
    assert len(ledger.spent) == 5
    assert current.claims.cnf == _THUMBPRINT


# -- the holder key is resolved lazily, at step 5 and not before ---------------------------


def _counting_holder_key(thumbprint: str, calls: list[str]) -> Any:
    """A holder-key resolver that records every time it is actually awaited."""

    async def resolve() -> str:
        calls.append(thumbprint)
        return thumbprint

    return resolve


async def test_a_holder_key_resolver_is_awaited_for_a_bound_session() -> None:
    original = _pair(cnf="thumbprint-abc")
    calls: list[str] = []
    rotated = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=_Ledger(),
        holder_key=_counting_holder_key("thumbprint-abc", calls),
    )
    assert rotated.claims.cnf == "thumbprint-abc"
    assert calls == ["thumbprint-abc"]


async def test_a_doomed_refresh_never_pays_for_proof_validation() -> None:
    # THE reason the holder key is a resolver rather than a value: proof validation is a
    # round trip, and every check that can deny this refresh has already run by step 5.
    original = _pair(cnf="thumbprint-abc")
    revoker = _Revoker()
    await revoker.revoke_session(original.claims.sid)
    calls: list[str] = []
    with pytest.raises(RotationError, match="revoked"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
            revoker=revoker,
            holder_key=_counting_holder_key("thumbprint-abc", calls),
        )
    assert calls == []


# -- sessions bind at issuance -------------------------------------------------------------


async def test_an_unbound_session_cannot_refresh() -> None:
    # Sessions bind AT ISSUANCE: the login flow collects the proof of possession and mints
    # the pair with cnf already set. A refresh token carrying no cnf therefore has no key
    # anyone could prove, and re-authentication is the only path that can bind one.
    ledger = _Ledger()
    with pytest.raises(RotationError, match="not holder-key bound"):
        await rotate_refresh_token(
            _pair(cnf=None).refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
        )
    # Same discipline as a wrong holder key: the refusal burns nothing, so it can never be
    # used to deny service or to trip reuse detection against the session's real holder.
    assert not ledger.spent


async def test_a_thief_cannot_bind_their_own_key_to_an_unbound_session() -> None:
    # The attack bind-on-first-refresh allowed: steal the refresh token before the victim's
    # first refresh, present YOUR key, own the session. Presenting a key changes nothing --
    # an unbound session is unrefreshable, whoever shows up.
    original = _pair(cnf=None)
    ledger = _Ledger()
    calls: list[str] = []
    with pytest.raises(RotationError, match="not holder-key bound"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
            holder_key=_counting_holder_key("thumbprint-attacker", calls),
        )
    assert not ledger.spent
    # And the refusal never pays for proof validation it cannot use.
    assert calls == []


async def test_the_bind_on_first_use_parameter_is_gone() -> None:
    # Deleted, not deprecated: with binding at issuance there is no first-use bind to widen,
    # and a surviving flag would silently re-open the stolen-token window it existed in.
    with pytest.raises(TypeError):
        await rotate_refresh_token(
            _pair().refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
            holder_key=_THUMBPRINT,
            bind_holder_key_on_first_use=True,  # type: ignore[call-arg]
        )


# -- caller-supplied deny rules ------------------------------------------------------------


def _gate(reason: str | None, seen: list[str]) -> Any:
    async def check(claims: Any) -> str | None:
        seen.append(claims.sub)
        return reason

    return check


async def test_a_pre_redemption_gate_denies_without_burning_the_token() -> None:
    # Same guarantee steps 2 and 3 already have: nothing above the redemption step may
    # consume a token it then refuses to exchange.
    ledger = _Ledger()
    with pytest.raises(RotationError, match="principal is blocked"):
        await rotate_refresh_token(
            _pair().refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
            pre_redemption_checks=[_gate("principal is blocked.", [])],
        )
    assert not ledger.spent


async def test_pre_redemption_gates_stop_at_the_first_refusal() -> None:
    seen: list[str] = []
    with pytest.raises(RotationError, match="first"):
        await rotate_refresh_token(
            _pair().refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
            pre_redemption_checks=[_gate("first.", seen), _gate("second.", seen)],
        )
    assert len(seen) == 1


async def test_a_post_redemption_gate_denies_a_token_that_is_already_spent() -> None:
    # The placement IS the point: a denial here must still force full re-authentication,
    # never permit a retry with the same token.
    ledger = _Ledger()
    with pytest.raises(RotationError, match="re-check failed"):
        await rotate_refresh_token(
            _pair().refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
            holder_key=_THUMBPRINT,
            post_redemption_checks=[_gate("re-check failed.", [])],
        )
    assert len(ledger.spent) == 1


async def test_a_gate_returning_none_allows_the_refresh() -> None:
    seen: list[str] = []
    await rotate_refresh_token(
        _pair().refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=_Ledger(),
        holder_key=_THUMBPRINT,
        pre_redemption_checks=[_gate(None, seen)],
        post_redemption_checks=[_gate(None, seen)],
    )
    assert seen == ["user-1", "user-1"]


# -- per-session lifetime caps -------------------------------------------------------------


async def test_lifetime_caps_can_be_resolved_per_session() -> None:
    # Caps that come from per-tenant policy cannot be known until the token is verified,
    # because the tenant is a claim on the token.
    started = int((datetime.now(UTC) - timedelta(days=40)).timestamp())
    original = _pair(auth_time=started, session_started_at=started, customer_id="tenant-strict")

    async def caps_for(claims: Any) -> SessionLifetimeCaps:
        assert claims.customer_id == "tenant-strict"
        return SessionLifetimeCaps(absolute=timedelta(days=30))

    with pytest.raises(RotationError, match="absolute lifetime"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
            lifetime_caps=caps_for,
        )


# ---------------------------------------------------------------------------
# Replay grace: a lost response is not a theft signal.
# ---------------------------------------------------------------------------


def _refresh_jti(pair: TokenPair) -> str:
    """The jti the ledger redeems -- read off the REFRESH token.

    `TokenPair.claims` is the ACCESS token's claims (see its docstring), whose jti is a
    different value entirely; keying a grace lookup on it would silently never match.
    """
    return verify_session_token(pair.refresh_token, verifier=_VERIFIER, expected_type=TokenType.REFRESH).jti


def test_the_grace_cache_double_satisfies_its_protocol() -> None:
    assert isinstance(_GraceCache(), ReplayGraceCache)


async def test_a_successful_rotation_remembers_its_pair_under_the_redeemed_jti() -> None:
    original = _pair()
    grace = _GraceCache()
    rotated = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=_Ledger(),
        holder_key=_THUMBPRINT,
        replay_grace=grace,
    )
    remembered = grace.entries[_refresh_jti(original)]
    assert remembered.pair.refresh_token == rotated.refresh_token
    assert remembered.holder_key == _THUMBPRINT


async def test_a_replay_within_grace_returns_the_same_pair_and_does_not_revoke() -> None:
    # The laptop slept (or the network dropped) between the server rotating and the client
    # storing the reply, so the client presents the same token again. It proved possession
    # of the bound key to get here, which a thief holding only the token bytes cannot do.
    original = _pair()
    ledger, revoker, grace = _Ledger(), _Revoker(), _GraceCache()
    first = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=ledger,
        revoker=revoker,
        holder_key=_THUMBPRINT,
        replay_grace=grace,
    )
    second = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=ledger,
        revoker=revoker,
        holder_key=_THUMBPRINT,
        replay_grace=grace,
    )
    assert second.refresh_token == first.refresh_token
    assert second.access_token == first.access_token
    # The session survives: this was the legitimate holder all along.
    assert original.claims.sid not in revoker.revoked


async def test_a_replay_outside_grace_still_revokes_the_family() -> None:
    # The cache is TTL-bounded; once the entry is gone the replay is indistinguishable from
    # theft again, and the original response stands unchanged.
    original = _pair()
    ledger, revoker, grace = _Ledger(), _Revoker(), _GraceCache()
    await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=ledger,
        revoker=revoker,
        holder_key=_THUMBPRINT,
        replay_grace=grace,
    )
    grace.entries.clear()  # what the store's TTL does on its own
    with pytest.raises(RotationError, match="already been used"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
            revoker=revoker,
            holder_key=_THUMBPRINT,
            replay_grace=grace,
        )
    assert original.claims.sid in revoker.revoked


async def test_a_grace_entry_recorded_under_another_key_is_not_honoured() -> None:
    # Defence in depth. Step 6 already refuses a mismatched proof before redemption, so this
    # can only arise from a cache serving an entry that was never this holder's -- which
    # must read as theft, not as grace.
    original = _pair()
    ledger, revoker, grace = _Ledger(), _Revoker(), _GraceCache()
    rotated = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=ledger,
        revoker=revoker,
        holder_key=_THUMBPRINT,
        replay_grace=grace,
    )
    grace.entries[_refresh_jti(original)] = ReplayGraceEntry(pair=rotated, holder_key="someone-elses-key")
    with pytest.raises(RotationError, match="already been used"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
            revoker=revoker,
            holder_key=_THUMBPRINT,
            replay_grace=grace,
        )
    assert original.claims.sid in revoker.revoked


async def test_without_a_grace_cache_reuse_behaves_exactly_as_before() -> None:
    # The parameter is additive: a deployment that supplies nothing keeps the original
    # revoke-the-family response, which is what every existing caller gets.
    original = _pair()
    ledger, revoker = _Ledger(), _Revoker()
    await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=ledger,
        revoker=revoker,
        holder_key=_THUMBPRINT,
    )
    with pytest.raises(RotationError, match="already been used"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
            revoker=revoker,
            holder_key=_THUMBPRINT,
        )
    assert original.claims.sid in revoker.revoked


async def test_post_redemption_checks_still_run_on_a_grace_replay() -> None:
    # A subject who lost the right to hold the session must not get it back by replaying.
    original = _pair()
    ledger, grace = _Ledger(), _GraceCache()
    await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=ledger,
        holder_key=_THUMBPRINT,
        replay_grace=grace,
    )

    async def _deny(_claims: Any) -> str:
        return "principal is blocked."

    with pytest.raises(RotationError, match="principal is blocked"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
            holder_key=_THUMBPRINT,
            replay_grace=grace,
            post_redemption_checks=(_deny,),
        )
