"""Refresh-token rotation and reuse detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.iam.rotation import (
    RefreshTokenLedger,
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


def _pair(**overrides: Any) -> TokenPair:
    """Mint a starting pair, with any minting argument overridable per test."""
    kwargs: dict[str, Any] = {
        "subject": "user-1",
        "session_id": new_session_id(),
        "issuer": _ISSUER,
        "audience": _AUDIENCE,
        "signer": _SIGNER,
        "step_up_window": 300,
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
    )
    with pytest.raises(RotationError, match="already been used"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
            revoker=revoker,
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
    )
    verify_session_token(rotated.refresh_token, verifier=_VERIFIER, expected_type=TokenType.REFRESH)
    verify_session_token(rotated.access_token, verifier=_VERIFIER, expected_type=TokenType.ACCESS)


async def test_rotation_chains() -> None:
    ledger = _Ledger()
    current = _pair()
    for _ in range(5):
        current = await rotate_refresh_token(
            current.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=ledger,
        )
    assert len(ledger.spent) == 5


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


async def test_an_unbound_session_never_resolves_a_holder_key() -> None:
    # Nothing to match against, so paying for proof validation would be pure waste.
    calls: list[str] = []
    await rotate_refresh_token(
        _pair().refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=_Ledger(),
        holder_key=_counting_holder_key("thumbprint-abc", calls),
    )
    assert calls == []


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


# -- bind on first use ---------------------------------------------------------------------


async def test_a_holder_key_binds_on_first_refresh() -> None:
    # For a login flow that collects no proof of possession: the first refresh is where the
    # session can start being bound at all.
    original = _pair()
    rotated = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=_Ledger(),
        holder_key="thumbprint-abc",
        bind_holder_key_on_first_use=True,
    )
    assert original.claims.cnf is None
    assert rotated.claims.cnf == "thumbprint-abc"


async def test_binding_on_first_use_refuses_a_refresh_that_presents_no_key() -> None:
    # Otherwise a client could simply decline to bind, indefinitely.
    with pytest.raises(RotationError, match="binds a holder key"):
        await rotate_refresh_token(
            _pair().refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
            bind_holder_key_on_first_use=True,
        )


async def test_an_already_bound_session_still_has_to_match() -> None:
    # bind-on-first-use widens the FIRST refresh only; it never relaxes a later one.
    original = _pair(cnf="thumbprint-abc")
    with pytest.raises(RotationError, match="does not match"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
            holder_key="thumbprint-xyz",
            bind_holder_key_on_first_use=True,
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
