"""the one reading of who a verified token names.

Three doors used to read the customer claim three ways -- a boolean, an ``is
None``, a UUID parse -- and a tool pod admitted at one was refused at the next.
:func:`principal_from_claims` is the reading they share; these pin what it says
for each kind of claim and that a malformed claim raises rather than reads as a
platform principal.
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from threetears.core.security import (
    PLATFORM_CUSTOMER_SENTINEL,
    IdentityClaims,
    VerifiedPrincipal,
    principal_from_claims,
)


def _claims(*, sub: str, customer_id: str) -> IdentityClaims:
    now = int(time.time())
    return IdentityClaims(
        sub=sub, customer_id=customer_id, sid="sid-1", pod_id="pod-1", iss="hub", iat=now, exp=now + 60
    )


class TestPrincipalFromClaims:
    def test_a_customer_uuid_is_an_agent_with_that_customer(self) -> None:
        sub, customer = uuid4(), uuid4()
        principal = principal_from_claims(_claims(sub=str(sub), customer_id=str(customer)))
        assert principal == VerifiedPrincipal(
            principal_id=sub, customer_id=customer, is_tool_pod=False, customer_claim=str(customer)
        )

    def test_the_sentinel_is_a_tool_pod_with_no_customer(self) -> None:
        sub = uuid4()
        principal = principal_from_claims(_claims(sub=str(sub), customer_id=PLATFORM_CUSTOMER_SENTINEL))
        assert principal.principal_id == sub
        assert principal.customer_id is None
        assert principal.is_tool_pod is True
        # verbatim, so a downstream assertion re-mints the claim the token carried
        assert principal.customer_claim == PLATFORM_CUSTOMER_SENTINEL

    @pytest.mark.parametrize("customer_id", ["aibots-platform-but-not-quite", "", "not-a-uuid"])
    def test_any_other_non_uuid_customer_claim_raises(self, customer_id: str) -> None:
        # the sentinel is the one non-UUID value the hub mints on purpose; anything else is a
        # malformed token, and a door catching ValueError fails closed on it.
        with pytest.raises(ValueError):
            principal_from_claims(_claims(sub=str(uuid4()), customer_id=customer_id))

    def test_a_non_uuid_sub_raises_for_a_tool_pod_too(self) -> None:
        with pytest.raises(ValueError):
            principal_from_claims(_claims(sub="not-a-uuid", customer_id=PLATFORM_CUSTOMER_SENTINEL))

    def test_the_value_is_immutable(self) -> None:
        principal = principal_from_claims(_claims(sub=str(uuid4()), customer_id=PLATFORM_CUSTOMER_SENTINEL))
        with pytest.raises(AttributeError):
            principal.is_tool_pod = False  # type: ignore[misc]
