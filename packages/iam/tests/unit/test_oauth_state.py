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
