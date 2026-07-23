"""tests for the claims-aware :func:`authorize_from_claims` primitive --
build-plan.md Chunk 13 (identity-core repo), security-model.md's
Impersonation paragraph: "the existing `authorize()` signature has no path
for `act_reason` to reach it at all."

mirrors ``test_authorize.py``'s fixtures/shape exactly; the only new
surface under test is the impersonation deny-list overlay and the
external-audience refusal `authorize_from_claims` adds on top of the
existing `authorize()` primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import (
    INTERNAL_AUDIENCE,
    AccessDenied,
    AclCache,
    ClaimsForAuthorization,
    ExternalAudienceNotSupported,
    Group,
    GroupMembership,
    ImpersonationCategory,
    MemberType,
    Role,
    RoleAssignment,
    ScopeType,
    authorize_from_claims,
)

from tests.unit._fake_loaders import FakeStore


@dataclass
class _StubNamespace:
    id: UUID
    customer_id: UUID
    namespace_type: str
    owner_agent_id: UUID


class _NamespaceCollectionStub:
    def __init__(self, entries: dict[str, _StubNamespace]) -> None:
        self._entries = dict(entries)

    async def get_by_name(self, name: str) -> _StubNamespace | None:
        return self._entries.get(name)


def _grant_user_read(
    *, store: FakeStore, user_id: UUID, namespace_id: UUID, customer_id: UUID, namespace_type: str = "memory"
) -> None:
    role = Role(id=uuid4(), name="Reader", permissions={namespace_type: frozenset(["read"])}, is_built_in=True)
    group = Group(id=uuid4(), name="readers", customer_id=customer_id)
    membership = GroupMembership(
        group_id=group.id, member_type=MemberType.USER, member_id=user_id, customer_id=customer_id
    )
    assignment = RoleAssignment(
        id=uuid4(),
        role_id=role.id,
        group_id=group.id,
        scope_type=ScopeType.NAMESPACE,
        scope_namespace_id=namespace_id,
        scope_namespace_type=None,
        scope_customer_id=None,
    )
    store.add_role(role)
    store.add_group(group)
    store.add_membership(membership)
    store.add_assignment(assignment)


def _setup(*, user_id: UUID) -> tuple[_NamespaceCollectionStub, AclCache, _StubNamespace]:
    customer = uuid4()
    owner = uuid4()
    ns = _StubNamespace(id=uuid4(), customer_id=customer, namespace_type="memory", owner_agent_id=owner)
    store = FakeStore()
    _grant_user_read(store=store, user_id=user_id, namespace_id=ns.id, customer_id=customer)
    cache = AclCache(membership_loader=store, grant_loader=store)
    ns_collection = _NamespaceCollectionStub({"memories.test": ns})
    return ns_collection, cache, ns


class TestAuthorizeFromClaimsOrdinaryTraffic:
    """no `act_reason`, or `act_reason="delegation"`: behaves exactly like
    `authorize()` -- the deny-list overlay never triggers."""

    @pytest.mark.asyncio
    async def test_allows_when_no_act_reason(self) -> None:
        user = uuid4()
        ns_collection, cache, _ns = _setup(user_id=user)
        claims = ClaimsForAuthorization(sub=str(user), is_internal_audience=True, act_reason=None)

        result = await authorize_from_claims(
            namespace_collection=ns_collection,
            namespace_name="memories.test",
            action="read",
            claims=claims,
            cache=cache,
            sensitive_category=ImpersonationCategory.CREDENTIAL_PASSKEY_MFA_MANAGEMENT,
        )
        assert result.decision is True

    @pytest.mark.asyncio
    async def test_allows_delegation_even_for_sensitive_category(self) -> None:
        """`act_reason="delegation"` is NOT the impersonation deny-list --
        only the literal `"impersonation"` value triggers it."""
        user = uuid4()
        ns_collection, cache, _ns = _setup(user_id=user)
        claims = ClaimsForAuthorization(sub=str(user), is_internal_audience=True, act_reason="delegation")

        result = await authorize_from_claims(
            namespace_collection=ns_collection,
            namespace_name="memories.test",
            action="read",
            claims=claims,
            cache=cache,
            sensitive_category=ImpersonationCategory.ACCOUNT_DELETION,
        )
        assert result.decision is True

    @pytest.mark.asyncio
    async def test_impersonation_without_sensitive_category_still_evaluates_normally(self) -> None:
        """an impersonation session performing a NON-deny-listed action defers
        entirely to the ordinary evaluator -- test-specifications.md's happy
        path: "perform an action as the target" (an ordinary support action)
        must still succeed."""
        user = uuid4()
        ns_collection, cache, _ns = _setup(user_id=user)
        claims = ClaimsForAuthorization(sub=str(user), is_internal_audience=True, act_reason="impersonation")

        result = await authorize_from_claims(
            namespace_collection=ns_collection,
            namespace_name="memories.test",
            action="read",
            claims=claims,
            cache=cache,
            sensitive_category=None,
        )
        assert result.decision is True


class TestAuthorizeFromClaimsImpersonationDenyList:
    """`act_reason="impersonation"` + a sensitive category: denied
    unconditionally, even though the target's own permissions would allow
    it (the same grant `_setup` seeds for the ordinary-traffic tests above)."""

    @pytest.mark.asyncio
    async def test_denies_sensitive_category_under_impersonation(self) -> None:
        user = uuid4()
        ns_collection, cache, _ns = _setup(user_id=user)
        claims = ClaimsForAuthorization(sub=str(user), is_internal_audience=True, act_reason="impersonation")

        with pytest.raises(AccessDenied) as exc_info:
            await authorize_from_claims(
                namespace_collection=ns_collection,
                namespace_name="memories.test",
                action="read",
                claims=claims,
                cache=cache,
                sensitive_category=ImpersonationCategory.CREDENTIAL_PASSKEY_MFA_MANAGEMENT,
            )
        assert exc_info.value.reason == "impersonation_deny_list"
        assert exc_info.value.user_id == user

    @pytest.mark.parametrize("category", list(ImpersonationCategory))
    @pytest.mark.asyncio
    async def test_denies_every_category(self, category: ImpersonationCategory) -> None:
        """every member of the fixed category set is deny-listed -- not a
        tenant-configurable subset (security-model.md)."""
        user = uuid4()
        ns_collection, cache, _ns = _setup(user_id=user)
        claims = ClaimsForAuthorization(sub=str(user), is_internal_audience=True, act_reason="impersonation")

        with pytest.raises(AccessDenied) as exc_info:
            await authorize_from_claims(
                namespace_collection=ns_collection,
                namespace_name="memories.test",
                action="read",
                claims=claims,
                cache=cache,
                sensitive_category=category,
            )
        assert exc_info.value.reason == "impersonation_deny_list"


class TestAuthorizeFromClaimsExternalAudience:
    """non-internal-audience claims are refused outright -- security-model.md:
    "resolves `sub` to a real `principal_id` correctly only on internal-
    audience tokens" -- fail closed rather than silently mis-resolving."""

    @pytest.mark.asyncio
    async def test_raises_for_external_audience(self) -> None:
        user = uuid4()
        ns_collection, cache, _ns = _setup(user_id=user)
        claims = ClaimsForAuthorization(sub=str(user), is_internal_audience=False, act_reason=None)

        with pytest.raises(ExternalAudienceNotSupported) as exc_info:
            await authorize_from_claims(
                namespace_collection=ns_collection,
                namespace_name="memories.test",
                action="read",
                claims=claims,
                cache=cache,
            )
        assert exc_info.value.reason == "external_audience_not_supported"
        # ExternalAudienceNotSupported is a subclass of AccessDenied so
        # generic catchers still trip.
        assert isinstance(exc_info.value, AccessDenied)

    @pytest.mark.asyncio
    async def test_external_audience_checked_before_deny_list(self) -> None:
        """even an impersonation + sensitive-category call on an external-
        audience token raises the audience error, not the deny-list one --
        the audience check is a precondition, not a second independent path."""
        user = uuid4()
        ns_collection, cache, _ns = _setup(user_id=user)
        claims = ClaimsForAuthorization(sub=str(user), is_internal_audience=False, act_reason="impersonation")

        with pytest.raises(ExternalAudienceNotSupported):
            await authorize_from_claims(
                namespace_collection=ns_collection,
                namespace_name="memories.test",
                action="read",
                claims=claims,
                cache=cache,
                sensitive_category=ImpersonationCategory.EMAIL_CHANGE,
            )


class TestAuthorizeFromClaimsInvalidSub:
    @pytest.mark.asyncio
    async def test_raises_access_denied_for_non_uuid_sub(self) -> None:
        ns_collection, cache, _ns = _setup(user_id=uuid4())
        claims = ClaimsForAuthorization(sub="not-a-uuid", is_internal_audience=True, act_reason=None)

        with pytest.raises(AccessDenied) as exc_info:
            await authorize_from_claims(
                namespace_collection=ns_collection,
                namespace_name="memories.test",
                action="read",
                claims=claims,
                cache=cache,
            )
        assert exc_info.value.reason == "invalid_sub"


class TestClaimsFromVerifiedClaims:
    """`ClaimsForAuthorization.from_verified_claims` derives
    `is_internal_audience` from the token's own `aud` rather than trusting the
    caller to assert it.

    `is_internal_audience` decides whether `sub` may be treated as a real
    internal principal id, so the derivation failing closed is the point: every
    audience that is not exactly the internal one -- unknown, external, or
    absent -- must yield `False` and be refused downstream.
    """

    def test_internal_audience_derives_true(self) -> None:
        claims = ClaimsForAuthorization.from_verified_claims(sub=str(uuid4()), aud=INTERNAL_AUDIENCE)
        assert claims.is_internal_audience is True

    @pytest.mark.parametrize("aud", ["aibots:external", "some-other-service", ""])
    def test_non_internal_audience_derives_false(self, aud: str) -> None:
        claims = ClaimsForAuthorization.from_verified_claims(sub=str(uuid4()), aud=aud)
        assert claims.is_internal_audience is False

    def test_absent_audience_derives_false(self) -> None:
        """A token predating the `aud` claim is NOT assumed internal here. This
        is the authorization boundary, not the authentication one -- a
        token-verifying consumer may need to tolerate a missing `aud` during
        rollout, but treating an unlabelled token as internal at THIS point
        would hand an unvouched-for subject to RBAC."""
        claims = ClaimsForAuthorization.from_verified_claims(sub=str(uuid4()), aud=None)
        assert claims.is_internal_audience is False

    def test_act_reason_passes_through(self) -> None:
        claims = ClaimsForAuthorization.from_verified_claims(
            sub=str(uuid4()), aud=INTERNAL_AUDIENCE, act_reason="impersonation"
        )
        assert claims.act_reason == "impersonation"

    def test_act_reason_defaults_to_none(self) -> None:
        claims = ClaimsForAuthorization.from_verified_claims(sub=str(uuid4()), aud=INTERNAL_AUDIENCE)
        assert claims.act_reason is None

    @pytest.mark.asyncio
    async def test_derived_external_audience_is_refused_end_to_end(self) -> None:
        """The derivation and the refusal compose: an external `aud` string in,
        `ExternalAudienceNotSupported` out, with no caller-asserted boolean
        anywhere in between."""
        user = uuid4()
        ns_collection, cache, _ns = _setup(user_id=user)
        claims = ClaimsForAuthorization.from_verified_claims(sub=str(user), aud="aibots:external")

        with pytest.raises(ExternalAudienceNotSupported):
            await authorize_from_claims(
                namespace_collection=ns_collection,
                namespace_name="memories.test",
                action="read",
                claims=claims,
                cache=cache,
            )

    @pytest.mark.asyncio
    async def test_derived_internal_audience_is_allowed_end_to_end(self) -> None:
        """The mirror case, so the test above cannot pass merely because the
        derivation returns False for everything."""
        user = uuid4()
        ns_collection, cache, _ns = _setup(user_id=user)
        claims = ClaimsForAuthorization.from_verified_claims(sub=str(user), aud=INTERNAL_AUDIENCE)

        result = await authorize_from_claims(
            namespace_collection=ns_collection,
            namespace_name="memories.test",
            action="read",
            claims=claims,
            cache=cache,
        )
        assert result is not None
