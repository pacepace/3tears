"""tests for :func:`evaluate_file_access` path-glob-bearing rbac gate.

namespace-task-01 phase 7 introduces the custom action types
``read_file_matching:<glob>`` and ``write_file_matching:<glob>``.
this module exercises the helper that translates a (namespace, actor,
file path, direction) tuple into an allow/deny decision by walking
the evaluator's trail output, filtering granted action strings by
the direction-appropriate prefix, and matching the suffix glob against
the file path via :meth:`pathlib.PurePosixPath.full_match`.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from threetears.agent.acl import (
    Group,
    GroupMembership,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    ScopeType,
    evaluate_file_access,
)

from tests.unit._fake_loaders import FakeStore, make_cache


def _ns_workspace() -> Namespace:
    """build a workspace-type :class:`Namespace` with a fresh id.

    :return: workspace namespace record
    :rtype: Namespace
    """
    return Namespace(
        id=uuid4(),
        customer_id=uuid4(),
        namespace_type="workspace",
        owner_agent_id=uuid4(),
    )


def _role_with_actions(actions: list[str]) -> Role:
    """build a :class:`Role` whose ``workspace`` bucket contains ``actions``.

    :param actions: action strings to embed in the workspace bucket
    :ptype actions: list[str]
    :return: built role
    :rtype: Role
    """
    return Role(
        id=uuid4(),
        name=f"workspace-allow:{uuid4().hex}",
        permissions={"workspace": frozenset(actions)},
        is_built_in=False,
    )


def _grant(
    namespace: Namespace,
    role: Role,
    user_id,
) -> tuple[Group, GroupMembership, RoleAssignment]:
    """build a one-group one-membership one-assignment chain for ``user_id``.

    :param namespace: workspace namespace grant targets
    :ptype namespace: Namespace
    :param role: role to grant
    :ptype role: Role
    :param user_id: member user UUID
    :return: tuple of (group, membership, assignment) for fake-store insertion
    :rtype: tuple[Group, GroupMembership, RoleAssignment]
    """
    group = Group(id=uuid4(), name="grantee", customer_id=namespace.customer_id)
    membership = GroupMembership(
        group_id=group.id,
        member_type=MemberType.USER,
        member_id=user_id,
        customer_id=namespace.customer_id,
    )
    assignment = RoleAssignment(
        id=uuid4(),
        role_id=role.id,
        group_id=group.id,
        scope_type=ScopeType.NAMESPACE,
        scope_namespace_id=namespace.id,
        scope_namespace_type=None,
        scope_customer_id=None,
    )
    return group, membership, assignment


def _grant_member(
    namespace: Namespace,
    role: Role,
    member_type: MemberType,
    member_id,
) -> tuple[Group, GroupMembership, RoleAssignment]:
    """build a one-group one-membership one-assignment chain for any member.

    symmetric to :func:`_grant` but parameterized on
    :class:`MemberType` so an agent-side grant can be built for the
    user-intersect-agent path.

    :param namespace: workspace namespace grant targets
    :ptype namespace: Namespace
    :param role: role to grant
    :ptype role: Role
    :param member_type: user or agent member type
    :ptype member_type: MemberType
    :param member_id: member UUID
    :return: tuple of (group, membership, assignment) for fake-store insertion
    :rtype: tuple[Group, GroupMembership, RoleAssignment]
    """
    group = Group(id=uuid4(), name="grantee", customer_id=namespace.customer_id)
    membership = GroupMembership(
        group_id=group.id,
        member_type=member_type,
        member_id=member_id,
        customer_id=namespace.customer_id,
    )
    assignment = RoleAssignment(
        id=uuid4(),
        role_id=role.id,
        group_id=group.id,
        scope_type=ScopeType.NAMESPACE,
        scope_namespace_id=namespace.id,
        scope_namespace_type=None,
        scope_customer_id=None,
    )
    return group, membership, assignment


class TestReadPermit:
    """direction='read' permit paths."""

    @pytest.mark.asyncio
    async def test_single_glob_exact_match_permits(self) -> None:
        """one read glob matching the path exactly permits."""
        user = uuid4()
        namespace = _ns_workspace()
        role = _role_with_actions(["read_file_matching:docs/readme.md"])
        group, membership, assignment = _grant(namespace, role, user)
        store = FakeStore()
        store.add_role(role)
        store.add_group(group)
        store.add_membership(membership)
        store.add_assignment(assignment)

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=None,
            path="docs/readme.md",
            direction="read",
            cache=make_cache(store),
        )
        assert decision is True

    @pytest.mark.asyncio
    async def test_recursive_glob_matches_nested_path(self) -> None:
        """``**/*.yaml`` permits ``sub/deep/foo.yaml``."""
        user = uuid4()
        namespace = _ns_workspace()
        role = _role_with_actions(["read_file_matching:**/*.yaml"])
        group, membership, assignment = _grant(namespace, role, user)
        store = FakeStore()
        store.add_role(role)
        store.add_group(group)
        store.add_membership(membership)
        store.add_assignment(assignment)

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=None,
            path="sub/deep/foo.yaml",
            direction="read",
            cache=make_cache(store),
        )
        assert decision is True

    @pytest.mark.asyncio
    async def test_multiple_globs_any_match_permits(self) -> None:
        """any glob in the granted action set matching the path permits."""
        user = uuid4()
        namespace = _ns_workspace()
        role = _role_with_actions(
            [
                "read_file_matching:*.yaml",
                "read_file_matching:docs/**",
            ]
        )
        group, membership, assignment = _grant(namespace, role, user)
        store = FakeStore()
        store.add_role(role)
        store.add_group(group)
        store.add_membership(membership)
        store.add_assignment(assignment)

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=None,
            path="docs/guide/intro.md",
            direction="read",
            cache=make_cache(store),
        )
        assert decision is True


class TestReadDeny:
    """direction='read' deny paths."""

    @pytest.mark.asyncio
    async def test_no_matching_glob_denies(self) -> None:
        """read glob ``*.yaml`` does not permit ``foo.txt``."""
        user = uuid4()
        namespace = _ns_workspace()
        role = _role_with_actions(["read_file_matching:*.yaml"])
        group, membership, assignment = _grant(namespace, role, user)
        store = FakeStore()
        store.add_role(role)
        store.add_group(group)
        store.add_membership(membership)
        store.add_assignment(assignment)

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=None,
            path="foo.txt",
            direction="read",
            cache=make_cache(store),
        )
        assert decision is False

    @pytest.mark.asyncio
    async def test_write_glob_does_not_permit_read(self) -> None:
        """only a write glob in the granted set -> read denies."""
        user = uuid4()
        namespace = _ns_workspace()
        role = _role_with_actions(["write_file_matching:*.yaml"])
        group, membership, assignment = _grant(namespace, role, user)
        store = FakeStore()
        store.add_role(role)
        store.add_group(group)
        store.add_membership(membership)
        store.add_assignment(assignment)

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=None,
            path="foo.yaml",
            direction="read",
            cache=make_cache(store),
        )
        assert decision is False

    @pytest.mark.asyncio
    async def test_no_grants_denies(self) -> None:
        """actor with no memberships is denied."""
        user = uuid4()
        namespace = _ns_workspace()
        store = FakeStore()

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=None,
            path="anything.yaml",
            direction="read",
            cache=make_cache(store),
        )
        assert decision is False


class TestWrite:
    """direction='write' permit + deny symmetry with read."""

    @pytest.mark.asyncio
    async def test_write_glob_match_permits_write(self) -> None:
        """write glob matching path -> write permits."""
        user = uuid4()
        namespace = _ns_workspace()
        role = _role_with_actions(["write_file_matching:out/*.yaml"])
        group, membership, assignment = _grant(namespace, role, user)
        store = FakeStore()
        store.add_role(role)
        store.add_group(group)
        store.add_membership(membership)
        store.add_assignment(assignment)

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=None,
            path="out/foo.yaml",
            direction="write",
            cache=make_cache(store),
        )
        assert decision is True

    @pytest.mark.asyncio
    async def test_read_glob_does_not_permit_write(self) -> None:
        """only a read glob in the granted set -> write denies."""
        user = uuid4()
        namespace = _ns_workspace()
        role = _role_with_actions(["read_file_matching:*.yaml"])
        group, membership, assignment = _grant(namespace, role, user)
        store = FakeStore()
        store.add_role(role)
        store.add_group(group)
        store.add_membership(membership)
        store.add_assignment(assignment)

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=None,
            path="foo.yaml",
            direction="write",
            cache=make_cache(store),
        )
        assert decision is False


class TestOwnerShortCircuit:
    """agent owner short-circuit permits every file direction."""

    @pytest.mark.asyncio
    async def test_owner_agent_permits_any_read(self) -> None:
        """agent is owner of namespace -> agent-only read permits anything."""
        agent = uuid4()
        namespace = Namespace(
            id=uuid4(),
            customer_id=uuid4(),
            namespace_type="workspace",
            owner_agent_id=agent,
        )
        store = FakeStore()

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=None,
            agent_id=agent,
            path="anything/goes.txt",
            direction="read",
            cache=make_cache(store),
        )
        assert decision is True

    @pytest.mark.asyncio
    async def test_owner_agent_permits_any_write(self) -> None:
        """agent is owner -> agent-only write permits anything."""
        agent = uuid4()
        namespace = Namespace(
            id=uuid4(),
            customer_id=uuid4(),
            namespace_type="workspace",
            owner_agent_id=agent,
        )
        store = FakeStore()

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=None,
            agent_id=agent,
            path="path.md",
            direction="write",
            cache=make_cache(store),
        )
        assert decision is True


class TestUserAgentIntersection:
    """user-intersect-agent path: the agent side is a genuine constraint.

    the namespace is NOT owned by the agent under test (``_ns_workspace``
    assigns a random ``owner_agent_id``), so the owner short-circuit does
    not fire and both sides must independently grant the path.
    """

    @pytest.mark.asyncio
    async def test_broad_user_narrow_agent_denies_outside_agent_sandbox(self) -> None:
        """broad user grant must NOT widen access past the agent's sandbox."""
        user = uuid4()
        agent = uuid4()
        namespace = _ns_workspace()

        user_role = _role_with_actions(["read_file_matching:**"])
        u_group, u_membership, u_assignment = _grant_member(
            namespace, user_role, MemberType.USER, user
        )
        agent_role = _role_with_actions(["read_file_matching:sandbox/**"])
        a_group, a_membership, a_assignment = _grant_member(
            namespace, agent_role, MemberType.AGENT, agent
        )

        store = FakeStore()
        for role in (user_role, agent_role):
            store.add_role(role)
        for group in (u_group, a_group):
            store.add_group(group)
        for membership in (u_membership, a_membership):
            store.add_membership(membership)
        for assignment in (u_assignment, a_assignment):
            store.add_assignment(assignment)

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=agent,
            path="secrets/key.txt",
            direction="read",
            cache=make_cache(store),
        )
        assert decision is False

    @pytest.mark.asyncio
    async def test_intersection_permits_when_both_sides_match(self) -> None:
        """a path granted on BOTH sides permits."""
        user = uuid4()
        agent = uuid4()
        namespace = _ns_workspace()

        user_role = _role_with_actions(["read_file_matching:**"])
        u_group, u_membership, u_assignment = _grant_member(
            namespace, user_role, MemberType.USER, user
        )
        agent_role = _role_with_actions(["read_file_matching:sandbox/**"])
        a_group, a_membership, a_assignment = _grant_member(
            namespace, agent_role, MemberType.AGENT, agent
        )

        store = FakeStore()
        for role in (user_role, agent_role):
            store.add_role(role)
        for group in (u_group, a_group):
            store.add_group(group)
        for membership in (u_membership, a_membership):
            store.add_membership(membership)
        for assignment in (u_assignment, a_assignment):
            store.add_assignment(assignment)

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=agent,
            path="sandbox/notes.txt",
            direction="read",
            cache=make_cache(store),
        )
        assert decision is True

    @pytest.mark.asyncio
    async def test_user_side_still_caps_broad_agent(self) -> None:
        """a narrow user grant caps a broad agent grant (intersection is symmetric)."""
        user = uuid4()
        agent = uuid4()
        namespace = _ns_workspace()

        user_role = _role_with_actions(["read_file_matching:docs/**"])
        u_group, u_membership, u_assignment = _grant_member(
            namespace, user_role, MemberType.USER, user
        )
        agent_role = _role_with_actions(["read_file_matching:**"])
        a_group, a_membership, a_assignment = _grant_member(
            namespace, agent_role, MemberType.AGENT, agent
        )

        store = FakeStore()
        for role in (user_role, agent_role):
            store.add_role(role)
        for group in (u_group, a_group):
            store.add_group(group)
        for membership in (u_membership, a_membership):
            store.add_membership(membership)
        for assignment in (u_assignment, a_assignment):
            store.add_assignment(assignment)

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=agent,
            path="secrets/key.txt",
            direction="read",
            cache=make_cache(store),
        )
        assert decision is False

    @pytest.mark.asyncio
    async def test_agent_with_no_grants_denies_broad_user(self) -> None:
        """an agent with NO file grants collapses the intersection to deny."""
        user = uuid4()
        agent = uuid4()
        namespace = _ns_workspace()

        user_role = _role_with_actions(["read_file_matching:**"])
        u_group, u_membership, u_assignment = _grant_member(
            namespace, user_role, MemberType.USER, user
        )

        store = FakeStore()
        store.add_role(user_role)
        store.add_group(u_group)
        store.add_membership(u_membership)
        store.add_assignment(u_assignment)

        decision = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=agent,
            path="anything.txt",
            direction="read",
            cache=make_cache(store),
        )
        assert decision is False

    @pytest.mark.asyncio
    async def test_owner_agent_with_user_caps_by_user(self) -> None:
        """owner agent side matches every path; the user side still caps."""
        user = uuid4()
        agent = uuid4()
        namespace = Namespace(
            id=uuid4(),
            customer_id=uuid4(),
            namespace_type="workspace",
            owner_agent_id=agent,
        )

        user_role = _role_with_actions(["read_file_matching:docs/**"])
        u_group, u_membership, u_assignment = _grant_member(
            namespace, user_role, MemberType.USER, user
        )

        store = FakeStore()
        store.add_role(user_role)
        store.add_group(u_group)
        store.add_membership(u_membership)
        store.add_assignment(u_assignment)
        cache = make_cache(store)

        permitted = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=agent,
            path="docs/readme.md",
            direction="read",
            cache=cache,
        )
        denied = await evaluate_file_access(
            namespace=namespace,
            user_id=user,
            agent_id=agent,
            path="secrets/key.txt",
            direction="read",
            cache=cache,
        )
        assert permitted is True
        assert denied is False


class TestValidation:
    """input validation shape."""

    @pytest.mark.asyncio
    async def test_unknown_direction_raises(self) -> None:
        """non-``read``/``write`` direction raises ``ValueError``."""
        namespace = _ns_workspace()
        store = FakeStore()
        with pytest.raises(ValueError):
            await evaluate_file_access(
                namespace=namespace,
                user_id=uuid4(),
                agent_id=None,
                path="x",
                direction="delete",  # type: ignore[arg-type]
                cache=make_cache(store),
            )

    @pytest.mark.asyncio
    async def test_no_actor_raises(self) -> None:
        """both user_id and agent_id None -> ``ValueError``."""
        namespace = _ns_workspace()
        store = FakeStore()
        with pytest.raises(ValueError):
            await evaluate_file_access(
                namespace=namespace,
                user_id=None,
                agent_id=None,
                path="x",
                direction="read",
                cache=make_cache(store),
            )
