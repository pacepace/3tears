"""Step-up re-authentication freshness."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from threetears.agent.acl import ImpersonationCategory
from threetears.iam.stepup import (
    StepUpRequiredError,
    SteppableClaims,
    is_step_up_satisfied,
    require_fresh_auth,
)


@dataclass(frozen=True)
class _Claims:
    """Minimal stand-in for a verified token's step-up-relevant claims.

    # parity-with: threetears.iam.stepup.SteppableClaims
    """

    auth_time: int
    step_up_window: int


def test_the_stand_in_satisfies_the_protocol() -> None:
    assert isinstance(_Claims(auth_time=0, step_up_window=0), SteppableClaims)


def test_fresh_authentication_satisfies_the_check() -> None:
    assert is_step_up_satisfied(auth_time=1_000, step_up_window=300, now=1_100)


def test_exactly_at_the_window_edge_still_satisfies() -> None:
    assert is_step_up_satisfied(auth_time=1_000, step_up_window=300, now=1_300)


def test_one_second_past_the_window_does_not() -> None:
    assert not is_step_up_satisfied(auth_time=1_000, step_up_window=300, now=1_301)


def test_a_zero_window_demands_authentication_in_the_same_second() -> None:
    assert is_step_up_satisfied(auth_time=1_000, step_up_window=0, now=1_000)
    assert not is_step_up_satisfied(auth_time=1_000, step_up_window=0, now=1_001)


def test_clock_skew_into_the_future_is_tolerated() -> None:
    # A token minted a few seconds ahead by NTP drift must not fail the check: the subject
    # has, at worst, authenticated.
    assert is_step_up_satisfied(auth_time=1_100, step_up_window=300, now=1_000)


def test_default_clock_is_used_when_now_is_omitted() -> None:
    # A window of a century is satisfied by any real clock; a window that already expired
    # in 1970 is not. Together these prove the default clock is consulted.
    assert is_step_up_satisfied(auth_time=0, step_up_window=10**10)
    assert not is_step_up_satisfied(auth_time=0, step_up_window=1)


def test_require_fresh_auth_passes_within_the_window() -> None:
    require_fresh_auth(
        _Claims(auth_time=1_000, step_up_window=300),
        category=ImpersonationCategory.ACCOUNT_DELETION,
        now=1_200,
    )


def test_require_fresh_auth_raises_a_distinguishable_error() -> None:
    # Distinct from a generic denial on purpose: the caller must be able to tell the user
    # "confirm your password", not "access denied".
    with pytest.raises(StepUpRequiredError) as excinfo:
        require_fresh_auth(
            _Claims(auth_time=1_000, step_up_window=300),
            category=ImpersonationCategory.EMAIL_CHANGE,
            now=9_999,
        )
    assert excinfo.value.category is ImpersonationCategory.EMAIL_CHANGE
    assert "email_change" in str(excinfo.value)


@pytest.mark.parametrize("category", list(ImpersonationCategory))
def test_every_category_is_enforceable(category: ImpersonationCategory) -> None:
    with pytest.raises(StepUpRequiredError):
        require_fresh_auth(_Claims(auth_time=0, step_up_window=1), category=category, now=1_000)


def test_the_taxonomy_is_the_shared_acl_one() -> None:
    # The whole reason this package exists: two repositories previously declared these six
    # strings independently because no import path connected them.
    assert {member.value for member in ImpersonationCategory} == {
        "credential_passkey_mfa_management",
        "account_deletion",
        "apikey_mint_rotate_for_target",
        "email_change",
        "rbac_grant_change",
        "audit_editing",
    }
