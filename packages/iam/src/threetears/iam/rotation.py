"""Refresh-token rotation with reuse detection.

Presenting a valid refresh token issues a new pair and invalidates the one
presented. Presenting a refresh token that has ALREADY been rotated away is the
theft signal: a second redemption ordinarily means two parties hold the same
token and one of them stole it. The response is to revoke the entire session
family, forcing re-authentication -- which the attacker cannot do and the real
user can.

**With one exception, and it is not a hole.** A legitimate client also replays
when it never RECEIVED the reply -- a laptop suspended mid-request, a network
changed while moving, a tab discarded. It still holds the old token because the
new one never reached it. Without :class:`ReplayGraceCache` that user is logged
out for being unlucky rather than for being attacked, and the logs read as
theft. Supplying one lets a replay be answered with the SAME pair the lost reply
carried, for a few tens of seconds, and only when the replay proves possession
of the key bound in ``cnf`` -- which step 6 has already checked and a thief
holding only the token bytes cannot do. Outside that window, or under any other
key, the family is revoked exactly as before.

**The check order is load-bearing.** Each step is placed where it is for a
reason, and reordering them reintroduces a specific bug:

1. Verify the token. A forged, expired, or wrong-type token is rejected before
   anything else runs or costs anything.
2. Standing revocation. Cheap and read-only, so an already-revoked session does
   not pay for a round trip it cannot recover from.
3. Session lifetime caps. Without an absolute cap a session refreshes forever,
   and "deprovisioned" means nothing.
4. ``pre_redemption_checks`` -- the deployment's own additional deny rules, in
   the same "cheap, read-only, cannot recover anyway" band as steps 2 and 3.
5. Refuse a session that is not holder-key bound, then validate the proof of
   possession. Sessions bind AT ISSUANCE: the login flow collects a proof and
   mints the pair with ``cnf`` already set, so a refresh token carrying none
   has no key anyone could prove, and re-authentication is the only path that
   can bind one. Binding on first refresh instead would hand the session to
   whoever presents the token first -- a thief who refreshes before the victim
   does binds THEIR key. The refusal sits here, before redemption, so it never
   burns the ``jti`` and never trips reuse detection. The holder key is
   resolved LAZILY here rather than passed in already-computed, so a refresh
   that steps 2-4 were always going to deny never pays for proof validation.
6. Match the holder key against the token's ``cnf`` -- **without consuming the
   refresh token's ``jti``**. This is the subtle one. If a wrong-key attempt
   burned the ``jti``, anyone holding a COPY of the refresh token bytes could
   deny service to the legitimate holder at will, without ever possessing the
   key. The mismatch must cost the attacker everything and the victim nothing.
7. Redeem the ``jti``. The only consuming step, so nothing above can burn a
   token it then refuses to exchange.
8. ``post_redemption_checks`` -- deny rules that must run against a token that
   is already spent either way. A re-check whose subject may have lost the
   right to hold the session belongs here rather than at step 4: denying it
   must still force full re-authentication, never permit a retry with the same
   token.

**``auth_time`` is carried forward unchanged.** A refresh proves session
continuity; it does not re-confirm a credential. Letting it move would mean
every refresh silently re-satisfies step-up, turning a stolen refresh token
into a way to perform sensitive actions without ever proving possession of a
password or a second factor.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
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
    sole_audience,
    verify_session_token,
)
from threetears.observe import get_logger

__all__ = [
    "HolderKeySource",
    "LifetimeCapsSource",
    "RefreshGate",
    "RefreshTokenLedger",
    "ReplayGraceCache",
    "ReplayGraceEntry",
    "RotationError",
    "SessionLifetimeCaps",
    "SessionRevoker",
    "rotate_refresh_token",
]

log = get_logger(__name__)


class RotationError(Exception):
    """A refresh could not be honoured.

    Covers an untrustworthy token, a revoked session, an expired session lifetime, a session
    that was never holder-key bound, a failed proof of possession, and reuse of an
    already-rotated token. One type for all of them:
    the client's correct response to every one is to re-authenticate, and distinguishing
    them tells an attacker which of their assumptions was wrong.

    Carries only structural reasons -- never the token or key material.
    """


@dataclass(frozen=True, slots=True)
class SessionLifetimeCaps:
    """How long a session may live, independently of how long its tokens live.

    :ivar absolute: cap measured from ``session_started_at`` and unmoved by rotation.
        Without one a session refreshes forever, and deprovisioning means nothing.
    :ivar inactivity: cap measured from the presented token's ``iat``.
    """

    absolute: timedelta | None = None
    inactivity: timedelta | None = None


#: A caller-supplied deny rule. Returns the structural reason to refuse the refresh, or
#: ``None`` to allow it. Raising :class:`RotationError` directly is equivalent; returning a
#: reason simply saves every gate from constructing the exception itself.
RefreshGate = Callable[[SessionClaims], Awaitable[str | None]]

#: The holder key proven by an accompanying proof of possession, either already computed or
#: as a zero-argument coroutine function that produces it. The callable form is what keeps
#: proof validation at step 5 of the check order instead of before step 1.
HolderKeySource = str | Callable[[], Awaitable[str]]

#: Fixed caps, or a resolver that reads them per session -- from tenant policy keyed on
#: ``claims.customer_id``, say, which cannot be known until the token has been verified.
LifetimeCapsSource = SessionLifetimeCaps | Callable[[SessionClaims], Awaitable[SessionLifetimeCaps]]


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


@dataclass(frozen=True, slots=True)
class ReplayGraceEntry:
    """What a rotation issued, kept briefly so its holder can ask for it twice.

    :ivar pair: the pair the original redemption minted -- returned verbatim to a replay
        inside the grace window, so the client ends up holding exactly what it would have
        held had the first reply arrived.
    :ivar holder_key: the ``cnf`` thumbprint the original redeemer proved. A replay must
        prove the SAME key; an entry recorded under any other is not this holder's and is
        treated as reuse.
    """

    pair: TokenPair
    holder_key: str


@runtime_checkable
class ReplayGraceCache(Protocol):
    """Short-lived memory of what each redemption issued, so a LOST REPLY is not read as theft.

    Reuse detection assumes a legitimate client never replays a token it has exchanged.
    That is false in one ordinary case: the server redeems the token and mints the reply,
    and the reply never arrives -- a laptop suspended mid-request, a network changed on the
    move, a tab discarded. The client still holds the old token, presents it again, and a
    detector without this cache revokes the whole family and logs the user out for being
    unlucky rather than for being attacked.

    What makes honouring the replay safe is that it happens AFTER step 6: the caller has
    already proven possession of the key bound in ``cnf``, which a thief holding only the
    token bytes cannot do. The replay is therefore the same client, and it is handed the
    same pair -- no new grant, no extended lifetime, nothing the first reply would not have
    given it.

    **Implementations MUST bound entries by a short TTL** (tens of seconds -- long enough
    to cover a lost reply, short enough that a token copied off the wire is useless by the
    time it is used). The window is the store's own, not this module's: outside it, `recall`
    simply misses and the original revoke-the-family response stands.
    """

    async def remember(self, jti: str, *, pair: TokenPair, holder_key: str) -> None:
        """Record what redeeming ``jti`` issued, under the holder key that redeemed it."""
        ...

    async def recall(self, jti: str) -> ReplayGraceEntry | None:
        """Return what ``jti`` issued, or ``None`` once the window has closed."""
        ...


async def rotate_refresh_token(
    refresh_token: str,
    *,
    verifier: TokenVerifier,
    signer: TokenSigner,
    ledger: RefreshTokenLedger,
    revoker: SessionRevoker | None = None,
    holder_key: HolderKeySource | None = None,
    lifetime_caps: LifetimeCapsSource | None = None,
    replay_grace: ReplayGraceCache | None = None,
    pre_redemption_checks: Sequence[RefreshGate] = (),
    post_redemption_checks: Sequence[RefreshGate] = (),
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
    :param holder_key: the thumbprint proven by an accompanying proof of possession, from
        :func:`~threetears.iam.dpop.validate_dpop_proof`. Required for every refresh that
        can succeed: sessions bind their holder key at issuance, so the presented token
        always carries a ``cnf`` and a refresh presenting no key is refused. Pass a
        coroutine function rather than a string wherever producing it costs anything: the
        callable is awaited at step 5 of the check order, so a refresh the earlier steps
        deny never pays for it.
    :ptype holder_key: HolderKeySource | None
    :param lifetime_caps: how long the session may live. Pass a resolver where the caps come
        from per-tenant policy, which cannot be read until ``customer_id`` is known.
    :ptype lifetime_caps: LifetimeCapsSource | None
    :param replay_grace: brief memory of what each redemption issued, so a client whose
        reply was LOST (a suspended laptop, a network changed mid-request) is handed that
        same reply again instead of having its session revoked for replaying. Safe because
        the replay has already proven possession of the bound holder key at step 6. ``None``
        keeps the original behaviour exactly: every replay is reuse.
    :ptype replay_grace: ReplayGraceCache | None
    :param pre_redemption_checks: additional deny rules, run in order at step 4 -- after the
        standing revocation and lifetime checks, before the proof of possession, and before
        anything has been consumed. This is where a deployment's own revocation shapes
        (principal, tenant) and its administrative disable check belong.
    :ptype pre_redemption_checks: Sequence[RefreshGate]
    :param post_redemption_checks: deny rules run at step 8, after the presented token has
        been consumed and before the new pair is minted.
    :ptype post_redemption_checks: Sequence[RefreshGate]
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
    await _enforce_lifetime_caps(claims, moment=moment, caps=lifetime_caps)

    # 4. The deployment's own deny rules, in the same nothing-consumed-yet band.
    await _run_gates(pre_redemption_checks, claims)

    # 5/6. Holder-key binding, checked WITHOUT consuming the jti (module docstring, step 6).
    resolved_cnf = await _resolve_holder_binding(claims, holder_key=holder_key)

    # 7. Redemption -- the only consuming step.
    if not await ledger.redeem(claims.jti):
        # Already spent. Two readings: the holder never received the reply and is asking
        # again, or two parties hold this token. `_replayed_within_grace` separates them by
        # the one thing a thief cannot forge -- the holder key proven at step 6.
        replayed = await _replayed_within_grace(claims, resolved_cnf, replay_grace)
        if replayed is None:
            log.warning(
                "refresh token reuse detected; revoking the session family",
                extra={"extra_data": {"sid": claims.sid}},
            )
            if revoker is not None:
                await revoker.revoke_session(claims.sid)
            raise RotationError("refresh token has already been used.")
        # The same client, handed the same pair the lost reply carried. Deny rules that run
        # against an already-spent token still apply: a subject who lost the right to hold
        # this session must not recover it by replaying.
        await _run_gates(post_redemption_checks, claims)
        return replayed

    # 8. Deny rules that must run against an already-spent token.
    await _run_gates(post_redemption_checks, claims)

    pair = mint_token_pair(
        subject=claims.sub,
        session_id=claims.sid,
        issuer=claims.iss,
        # Read back through the checked accessor rather than passed as the tuple it is
        # stored in: the rotated pair must carry the SAME single audience the redeemed
        # token did, and `sole_audience` is what refuses to guess if it somehow carries
        # more than one.
        audience=sole_audience(claims),
        signer=signer,
        # Carried forward unchanged: a refresh proves continuity, not authentication.
        auth_time=claims.auth_time,
        step_up_window=claims.step_up_window,
        session_started_at=claims.session_started_at,
        customer_id=claims.customer_id,
        cnf=resolved_cnf,
        act=claims.act,
        act_reason=claims.act_reason,
        act_restriction=claims.act_restriction,
        access_ttl=access_ttl,
        refresh_ttl=refresh_ttl,
        now=moment,
    )
    if replay_grace is not None:
        # Filed under the jti just REDEEMED -- the one a lost reply leaves the client still
        # holding, and therefore the one it will present again.
        await replay_grace.remember(claims.jti, pair=pair, holder_key=resolved_cnf)
    return pair


async def _replayed_within_grace(
    claims: SessionClaims,
    resolved_cnf: str,
    replay_grace: ReplayGraceCache | None,
) -> TokenPair | None:
    """The pair a lost reply carried, when this replay is provably its holder, else ``None``.

    ``None`` means "treat as reuse": no cache configured, the window has closed, or the
    entry belongs to another holder key -- which cannot arise from step 6 alone and so must
    read as theft rather than as grace.
    """
    if replay_grace is None:
        return None
    entry = await replay_grace.recall(claims.jti)
    if entry is None or entry.holder_key != resolved_cnf:
        return None
    log.info(
        "refresh token replayed inside the grace window; returning the issued pair",
        extra={"extra_data": {"sid": claims.sid}},
    )
    return entry.pair


async def _run_gates(gates: Sequence[RefreshGate], claims: SessionClaims) -> None:
    """Run caller-supplied deny rules in order, stopping at the first refusal."""
    for gate in gates:
        reason = await gate(claims)
        if reason is not None:
            raise RotationError(reason)


async def _resolve_holder_binding(
    claims: SessionClaims,
    *,
    holder_key: HolderKeySource | None,
) -> str:
    """Settle what ``cnf`` the new pair carries, without consuming anything.

    Exactly two cases: a session with no ``cnf`` was never holder-key bound and is refused
    -- binding happens at issuance, so re-authentication is the only path that can bind it
    -- and a bound session must present a key whose thumbprint matches exactly. The refusal
    of an unbound session comes BEFORE the holder key is resolved: there is nothing to match
    it against, so paying for proof validation would be pure waste.
    """
    if claims.cnf is None:
        raise RotationError("this session is not holder-key bound; re-authenticate.")
    if holder_key is None:
        raise RotationError("this session is holder-key bound and no proof of possession was presented.")
    thumbprint = holder_key if isinstance(holder_key, str) else await holder_key()
    if thumbprint != claims.cnf:
        raise RotationError("proof of possession does not match the session's bound key.")
    return claims.cnf


async def _enforce_lifetime_caps(
    claims: SessionClaims,
    *,
    moment: datetime,
    caps: LifetimeCapsSource | None,
) -> None:
    """Reject a session that has outlived either configured cap.

    Both are measured against the token's own claims rather than a stored record, so the
    check costs nothing and cannot be skipped by a caller that forgot to load the session.
    """
    if caps is None:
        return
    resolved = caps if isinstance(caps, SessionLifetimeCaps) else await caps(claims)
    seconds_now = int(moment.timestamp())
    if resolved.absolute is not None:
        age = seconds_now - claims.session_started_at
        if age > resolved.absolute.total_seconds():
            raise RotationError("session has exceeded its absolute lifetime.")
    if resolved.inactivity is not None:
        idle = seconds_now - claims.iat
        if idle > resolved.inactivity.total_seconds():
            raise RotationError("session has been idle beyond the inactivity timeout.")
