"""OAuth ``state``: signing, the two-stage algorithm pin, and the single-use nonce."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from threetears.iam.oauth_state import (
    DEFAULT_STATE_TYPE,
    OAuthStateError,
    consume_state_nonce,
    mint_oauth_state,
    record_state_nonce,
    verify_oauth_state,
)
from threetears.iam.stores.memory import MemoryStateStore

_SECRET = "test-signing-secret-not-a-real-one"
_ISSUER = "test-issuer"
_AUDIENCE = "test-issuer:oauth-state"


def _mint(**overrides: object) -> str:
    kwargs: dict[str, object] = {"secret": _SECRET, "issuer": _ISSUER, "audience": _AUDIENCE}
    kwargs.update(overrides)
    return mint_oauth_state(**kwargs)  # type: ignore[arg-type]


def _verify(state: str, **overrides: object) -> object:
    kwargs: dict[str, object] = {"secret": _SECRET, "issuer": _ISSUER, "audience": _AUDIENCE}
    kwargs.update(overrides)
    return verify_oauth_state(state, **kwargs)  # type: ignore[arg-type]


def test_a_minted_state_verifies_and_carries_a_nonce() -> None:
    verified = verify_oauth_state(_mint(), secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE)
    assert verified.nonce
    assert verified.expires_at > verified.issued_at


def test_each_mint_carries_a_distinct_nonce() -> None:
    first = verify_oauth_state(_mint(), secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE)
    second = verify_oauth_state(_mint(), secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE)
    assert first.nonce != second.nonce


def test_a_wrong_secret_is_refused() -> None:
    with pytest.raises(OAuthStateError):
        _verify(_mint(), secret="a-different-secret")


def test_a_wrong_audience_is_refused() -> None:
    with pytest.raises(OAuthStateError):
        _verify(_mint(), audience="someone-elses-audience")


def test_a_wrong_issuer_is_refused() -> None:
    with pytest.raises(OAuthStateError):
        _verify(_mint(), issuer="someone-else")


def test_a_token_of_another_type_is_refused() -> None:
    """A token minted for some other purpose under the SAME secret is not a state."""
    other = _mint(state_type="password_reset")
    with pytest.raises(OAuthStateError):
        _verify(other)


def test_an_expired_state_is_refused() -> None:
    stale = _mint(now=datetime.now(UTC) - timedelta(hours=1), ttl=timedelta(minutes=5))
    with pytest.raises(OAuthStateError):
        _verify(stale)


def test_the_alg_none_header_never_reaches_signature_verification() -> None:
    """The declared-algorithm pin: an unsigned token is refused before a key is chosen."""
    forged = jwt.encode(
        {
            "type": DEFAULT_STATE_TYPE,
            "nonce": "deadbeef",
            "iss": _ISSUER,
            "aud": _AUDIENCE,
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(minutes=5)).timestamp()),
        },
        key="",
        algorithm="none",
    )
    with pytest.raises(OAuthStateError):
        _verify(forged)


def test_garbage_is_refused_rather_than_raising_a_jwt_error() -> None:
    with pytest.raises(OAuthStateError):
        _verify("not-a-jwt-at-all")


def test_a_state_without_a_nonce_is_refused() -> None:
    """Minted by hand: a state with no nonce cannot be made single-use, so it is not usable."""
    nonceless = jwt.encode(
        {
            "type": DEFAULT_STATE_TYPE,
            "iss": _ISSUER,
            "aud": _AUDIENCE,
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(minutes=5)).timestamp()),
        },
        _SECRET,
        algorithm="HS256",
    )
    with pytest.raises(OAuthStateError):
        _verify(nonceless)


@pytest.mark.asyncio
async def test_a_recorded_nonce_consumes_exactly_once() -> None:
    store = MemoryStateStore()
    state = _mint()
    await record_state_nonce(store, state, secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE)

    await consume_state_nonce(store, state, secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE)

    # The replay: same bytes, already spent.
    with pytest.raises(OAuthStateError):
        await consume_state_nonce(store, state, secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE)


@pytest.mark.asyncio
async def test_an_unrecorded_state_is_a_replay() -> None:
    """Signature-valid but never recorded: the stateless check passes and single-use does not."""
    store = MemoryStateStore()
    state = _mint()
    verify_oauth_state(state, secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE)

    with pytest.raises(OAuthStateError):
        await consume_state_nonce(store, state, secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE)


@pytest.mark.asyncio
async def test_recording_refuses_a_state_that_does_not_verify() -> None:
    store = MemoryStateStore()
    with pytest.raises(OAuthStateError):
        await record_state_nonce(store, _mint(secret="wrong"), secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE)


def test_a_state_minted_on_a_slightly_fast_clock_still_verifies() -> None:
    """One second of NTP drift between two pods must not reject a legitimate login.

    PyJWT validates ``iat`` as not-in-the-future, so with no leeway a state minted on a pod whose
    clock is a second ahead of the pod handling the callback is refused -- and refused with the
    same message as a forged or replayed state, by this module's deliberate design, so the outage
    is indistinguishable from an attack.
    """
    ahead = mint_oauth_state(
        secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE, now=datetime.now(UTC) + timedelta(seconds=30)
    )

    verified = verify_oauth_state(ahead, secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE)

    assert verified.nonce


def test_skew_tolerance_is_bounded() -> None:
    """Leeway is tolerance, not a blank cheque -- a state from far in the future is still refused."""
    far = mint_oauth_state(
        secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE, now=datetime.now(UTC) + timedelta(hours=1)
    )
    with pytest.raises(OAuthStateError):
        _verify(far)


def test_an_empty_secret_is_refused_as_a_state_error_not_a_jwt_error() -> None:
    """``InvalidKeyError`` derives from ``PyJWTError``, NOT ``InvalidTokenError``.

    Catching only the latter let a deployment booted with an unset secret raise a raw jwt exception
    out of a function whose contract says it raises ``OAuthStateError`` -- a 500 from every
    callback, uncontainable by callers catching the documented type.
    """
    with pytest.raises(OAuthStateError):
        _verify(_mint(), secret="")


def test_an_asymmetric_key_cannot_be_used_as_the_hmac_secret_at_all() -> None:
    """The asymmetric-confusion surface does not exist here, and this pins why.

    The classic downgrade is an attacker signing HS256 with the deployment's PUBLIC key bytes as
    the HMAC secret. PyJWT refuses that at MINT time -- it detects a PEM/x509 key and will not use
    it as an HMAC secret -- so the forged token cannot be constructed against this module in the
    first place. Asserted rather than assumed: my first attempt at this test tried to build the
    attack and discovered the library had already closed it.
    """
    public_key = "-----BEGIN PUBLIC KEY-----\nnot-the-hmac-secret\n-----END PUBLIC KEY-----"

    with pytest.raises(jwt.InvalidKeyError):
        mint_oauth_state(secret=public_key, issuer=_ISSUER, audience=_AUDIENCE)


def test_a_header_declaring_another_algorithm_is_refused() -> None:
    """The pin is on the DECLARED header, so a different alg never reaches key selection."""
    forged = jwt.encode(
        {
            "type": DEFAULT_STATE_TYPE,
            "nonce": "deadbeef",
            "iss": _ISSUER,
            "aud": _AUDIENCE,
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(minutes=5)).timestamp()),
        },
        _SECRET,
        algorithm="HS512",
    )

    with pytest.raises(OAuthStateError):
        _verify(forged)


@pytest.mark.asyncio
async def test_a_recorded_binding_round_trips_to_the_consumer() -> None:
    """The browser binding: the signature proves the flow is ours, not that it is THIS browser's.

    Without a binding an attacker can start a real flow, take the validly-minted state and their
    own authorization code, and walk a victim's browser to the callback -- signing the victim into
    the attacker's account (RFC 6749 s10.12). The payload is how a caller closes that.
    """
    store = MemoryStateStore()
    state = _mint()
    await record_state_nonce(
        store, state, secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE, payload={"browser": "cookie-hash"}
    )

    bound = await consume_state_nonce(store, state, secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE)

    assert bound["browser"] == "cookie-hash", "the caller must be able to compare this to the live request"


@pytest.mark.asyncio
async def test_consuming_a_nonce_recorded_without_a_binding_returns_empty_not_none() -> None:
    """The empty payload must not read as 'not found' -- a truthiness test here rejects every
    valid callback for callers that bind nothing."""
    store = MemoryStateStore()
    state = _mint()
    await record_state_nonce(store, state, secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE)

    assert await consume_state_nonce(store, state, secret=_SECRET, issuer=_ISSUER, audience=_AUDIENCE) == {}
