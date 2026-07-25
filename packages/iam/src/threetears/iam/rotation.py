"""Refresh-token rotation with reuse detection.

Presenting a valid refresh token issues a new pair and invalidates the one
presented. Presenting a refresh token that has ALREADY been rotated away is the
theft signal: a legitimate client never replays a token it has already
exchanged, so a second redemption means two parties hold the same token and one
of them stole it. The response is to revoke the entire session family, forcing
re-authentication -- which the attacker cannot do and the real user can.

**The check order is load-bearing.** Each step is placed where it is for a
reason, and reordering them reintroduces a specific bug:

1. Verify the token. A forged, expired, or wrong-type token is rejected before
   anything else runs or costs anything.
2. Standing revocation. Cheap and read-only, so an already-revoked session does
   not pay for a round trip it cannot recover from.
3. Session lifetime caps. Without an absolute cap a session refreshes forever,
   and "deprovisioned" means nothing.
4. Validate the proof of possession, if the session is bound.
5. Match the holder key against the token's ``cnf`` -- **without consuming the
   refresh token's ``jti``**. This is the subtle one. If a wrong-key attempt
   burned the ``jti``, anyone holding a COPY of the refresh token bytes could
   deny service to the legitimate holder at will, without ever possessing the
   key. The mismatch must cost the attacker everything and the victim nothing.
6. Redeem the ``jti``. LAST, and the only consuming step, so nothing above can
   burn a token it then refuses to exchange.

**``auth_time`` is carried forward unchanged.** A refresh proves session
continuity; it does not re-confirm a credential. Letting it move would mean
every refresh silently re-satisfies step-up, turning a stolen refresh token
into a way to perform sensitive actions without ever proving possession of a
password or a second factor.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from threetears.iam.tokens import (
    DEFAULT_ACCESS_TTL,
    DEFAULT_REFRESH_TTL,
    SessionClaims,
    TokenError,
    TokenPair,
    TokenSigner,
    TokenType,
    TokenVerifier,
    mint_token_pair,
    verify_session_token,
)
from threetears.observe import get_logger

__all__ = ["RefreshTokenLedger", "RotationError", "SessionRevoker", "rotate_refresh_token"]

log = get_logger(__name__)


class RotationError(Exception):
    """A refresh could not be honoured.

    Covers an untrustworthy token, a revoked session, an expired session lifetime, a failed
    proof of possession, and reuse of an already-rotated token. One type for all of them:
    the client's correct response to every one is to re-authenticate, and distinguishing
    them tells an attacker which of their assumptions was wrong.

    Carries only structural reasons -- never the token or key material.
    """


@runtime_checkable
class RefreshTokenLedger(Protocol):
    """Tracks which refresh tokens have been spent."""

    async def redeem(self, jti: str) -> bool:
        """Atomically mark ``jti`` spent.

        :return: ``True`` if this call spent it, ``False`` if it was ALREADY spent -- which
            is the reuse signal. Concurrent redemptions of one ``jti`` must produce exactly
            one ``True``; a check-then-act would let a stolen token and the real one both
            through, which is precisely the case this exists to catch.
        """
        ...


@runtime_checkable
class SessionRevoker(Protocol):
    """Revokes whole session families."""

    async def revoke_session(self, sid: str) -> None:
        """Revoke every token descended from session ``sid``."""
        ...

    async def is_session_revoked(self, sid: str) -> bool:
        """Whether ``sid`` has been revoked."""
        ...


async def rotate_refresh_token(
    refresh_token: str,
    *,
    verifier: TokenVerifier,
    signer: TokenSigner,
    ledger: RefreshTokenLedger,
    revoker: SessionRevoker | None = None,
    holder_key_thumbprint: str | None = None,
    absolute_session_lifetime: timedelta | None = None,
    inactivity_timeout: timedelta | None = None,
    access_ttl: timedelta = DEFAULT_ACCESS_TTL,
    refresh_ttl: timedelta = DEFAULT_REFRESH_TTL,
    now: datetime | None = None,
) -> TokenPair:
    """Exchange a refresh token for a new pair, invalidating the one presented.

    :param refresh_token: the presented token.
    :ptype refresh_token: str
    :param verifier: verifies the presented token.
    :ptype verifier: TokenVerifier
    :param signer: signs the new pair.
    :ptype signer: TokenSigner
    :param ledger: the single-use ledger. Its atomicity IS the reuse detection.
    :ptype ledger: RefreshTokenLedger
    :param revoker: consulted for standing revocation, and told to revoke the family on
        reuse. ``None`` disables both -- which means reuse is rejected but NOT escalated, so
        a thief simply retries. Supply one in any deployment that can.
    :ptype revoker: SessionRevoker | None
    :param holder_key_thumbprint: the thumbprint proven by an accompanying proof of
        possession, from :func:`~threetears.iam.dpop.validate_dpop_proof`. Required when the
        presented token carries a ``cnf`` binding.
    :ptype holder_key_thumbprint: str | None
    :param absolute_session_lifetime: cap measured from ``session_started_at``. Without one a
        session refreshes forever.
    :ptype absolute_session_lifetime: timedelta | None
    :param inactivity_timeout: cap measured from the presented token's ``iat``.
    :ptype inactivity_timeout: timedelta | None
    :param access_ttl: lifetime of the new access token.
    :ptype access_ttl: timedelta
    :param refresh_ttl: lifetime of the new refresh token.
    :ptype refresh_ttl: timedelta
    :param now: injectable clock.
    :ptype now: datetime | None
    :return: the new pair.
    :rtype: TokenPair
    :raises RotationError: on any failure. The caller must treat it as "re-authenticate".
    """
    moment = now or datetime.now(UTC)

    # 1. Verify before anything else runs.
    try:
        claims = verify_session_token(refresh_token, verifier=verifier, expected_type=TokenType.REFRESH)
    except TokenError as exc:
        raise RotationError(f"refresh token is not trustworthy ({exc}).") from None

    # 2. Standing revocation -- cheap, read-only, before any round trip.
    if revoker is not None and await revoker.is_session_revoked(claims.sid):
        raise RotationError("session has been revoked.")

    # 3. Lifetime caps. The floor that makes deprovisioning mean something.
    _enforce_lifetime_caps(
        claims,
        moment=moment,
        absolute_session_lifetime=absolute_session_lifetime,
        inactivity_timeout=inactivity_timeout,
    )

    # 4/5. Holder-key binding, checked WITHOUT consuming the jti (module docstring, step 5).
    if claims.cnf is not None:
        if holder_key_thumbprint is None:
            raise RotationError("this session is holder-key bound and no proof of possession was presented.")
        if holder_key_thumbprint != claims.cnf:
            raise RotationError("proof of possession does not match the session's bound key.")

    # 6. Redemption -- last, and the only consuming step.
    if not await ledger.redeem(claims.jti):
        # Already spent. A legitimate client never replays a token it has exchanged, so two
        # parties hold this one and the session is compromised.
        log.warning(
            "refresh token reuse detected; revoking the session family",
            extra={"extra_data": {"sid": claims.sid}},
        )
        if revoker is not None:
            await revoker.revoke_session(claims.sid)
        raise RotationError("refresh token has already been used.")

    return mint_token_pair(
        subject=claims.sub,
        session_id=claims.sid,
        issuer=claims.iss,
        audience=claims.aud,
        signer=signer,
        # Carried forward unchanged: a refresh proves continuity, not authentication.
        auth_time=claims.auth_time,
        step_up_window=claims.step_up_window,
        session_started_at=claims.session_started_at,
        customer_id=claims.customer_id,
        cnf=claims.cnf,
        act=claims.act,
        act_reason=claims.act_reason,
        act_restriction=claims.act_restriction,
        access_ttl=access_ttl,
        refresh_ttl=refresh_ttl,
        now=moment,
    )


def _enforce_lifetime_caps(
    claims: SessionClaims,
    *,
    moment: datetime,
    absolute_session_lifetime: timedelta | None,
    inactivity_timeout: timedelta | None,
) -> None:
    """Reject a session that has outlived either configured cap.

    Both are measured against the token's own claims rather than a stored record, so the
    check costs nothing and cannot be skipped by a caller that forgot to load the session.
    """
    seconds_now = int(moment.timestamp())
    if absolute_session_lifetime is not None:
        age = seconds_now - claims.session_started_at
        if age > absolute_session_lifetime.total_seconds():
            raise RotationError("session has exceeded its absolute lifetime.")
    if inactivity_timeout is not None:
        idle = seconds_now - claims.iat
        if idle > inactivity_timeout.total_seconds():
            raise RotationError("session has been idle beyond the inactivity timeout.")
