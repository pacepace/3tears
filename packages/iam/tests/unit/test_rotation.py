"""Refresh-token rotation and reuse detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.iam.rotation import (
    RefreshTokenLedger,
    RotationError,
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
            absolute_session_lifetime=timedelta(days=30),
        )


async def test_a_session_within_the_absolute_cap_still_refreshes() -> None:
    started = int((datetime.now(UTC) - timedelta(days=5)).timestamp())
    original = _pair(auth_time=started, session_started_at=started)
    await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=_Ledger(),
        absolute_session_lifetime=timedelta(days=30),
    )


async def test_the_inactivity_timeout_ends_a_session() -> None:
    original = _pair(now=datetime.now(UTC) - timedelta(hours=10))
    with pytest.raises(RotationError, match="idle"):
        await rotate_refresh_token(
            original.refresh_token,
            verifier=_VERIFIER,
            signer=_SIGNER,
            ledger=_Ledger(),
            inactivity_timeout=timedelta(hours=1),
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
            holder_key_thumbprint="thumbprint-xyz",
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
            holder_key_thumbprint="thumbprint-xyz",
        )
    assert not ledger.spent
    # The real holder can still exchange it.
    rotated = await rotate_refresh_token(
        original.refresh_token,
        verifier=_VERIFIER,
        signer=_SIGNER,
        ledger=ledger,
        holder_key_thumbprint="thumbprint-abc",
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
