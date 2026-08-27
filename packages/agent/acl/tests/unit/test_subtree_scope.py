"""unit tests for :attr:`ScopeType.SUBTREE` and its ``covers`` branch.

a subtree assignment is the one scope shape that answers "this node and
everything under it". the three that preceded it answer one exact row,
one type inside one customer, or everything -- so a grant meant as "the
``tools.pentest`` subtree" had to be written as either a single row or
every tool namespace the customer has.

containment is delegated to :func:`threetears.core.namespaces.namespace_contains`,
the ONE implementation, so the segment-awareness proven in that module's
own tests holds here by construction rather than by a second copy of the
rule.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import (
    EvaluationContext,
    Group,
    GroupMembership,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    ScopeType,
    evaluate_decision,
)

from ._fake_loaders import FakeStore, make_cache

CUSTOMER = UUID("11111111-1111-4111-8111-111111111111")
TOOL_CALL = "tool.call"


def _tool_namespace(name: str | None, *, customer_id: UUID | None = None) -> Namespace:
    """build a tool-type namespace value carrying ``name``.

    :param name: canonical namespace name, or ``None`` for a value
        whose construction site did not supply one
    :ptype name: str | None
    :param customer_id: owning customer, or ``None`` for a
        platform-scoped tool row (which is what every tool namespace
        actually is)
    :ptype customer_id: UUID | None
    :return: namespace value
    :rtype: Namespace
    """
    return Namespace(
        id=uuid4(),
        customer_id=customer_id,
        namespace_type="tool",
        owner_agent_id=None,
        name=name,
    )


def _subtree_assignment(node: str) -> RoleAssignment:
    """build a subtree-scoped assignment rooted at ``node``.

    :param node: subtree root name
    :ptype node: str
    :return: assignment value
    :rtype: RoleAssignment
    """
    return RoleAssignment(
        id=uuid4(),
        role_id=uuid4(),
        group_id=uuid4(),
        scope_type=ScopeType.SUBTREE,
        scope_namespace_id=None,
        scope_namespace_type=None,
        scope_customer_id=None,
        scope_namespace_name=node,
    )


class TestScopeTypeMembership:
    """the enum gains a fourth member beside the three that exist."""

    def test_subtree_is_a_scope_type(self) -> None:
        assert ScopeType.SUBTREE.value == "subtree"

    def test_the_three_existing_shapes_are_untouched(self) -> None:
        assert ScopeType.NAMESPACE.value == "namespace"
        assert ScopeType.TYPE_CUSTOMER.value == "type_customer"
        assert ScopeType.ALL.value == "all"


class TestSubtreeCovers:
    """:meth:`RoleAssignment.covers` under a subtree scope."""

    def test_covers_a_descendant(self) -> None:
        assignment = _subtree_assignment("tools.dipp")
        assert assignment.covers(_tool_namespace("tools.dipp.thing")) is True

    def test_covers_the_node_itself(self) -> None:
        assignment = _subtree_assignment("tools.dipp")
        assert assignment.covers(_tool_namespace("tools.dipp")) is True

    def test_does_not_cover_a_prefix_sibling(self) -> None:
        # the acceptance criterion, at the scope altitude: no trailing
        # dot anywhere in the stored value, and the imposter is still
        # unreachable.
        assignment = _subtree_assignment("pentest")
        assert assignment.covers(_tool_namespace("pentestimposter.sqlmap")) is False
        assert assignment.covers(_tool_namespace("pentest.sqlmap")) is True

    def test_denies_when_the_namespace_carries_no_name(self) -> None:
        # a construction site that does not supply a name NARROWS a
        # subtree grant to nothing. it can never widen one.
        assignment = _subtree_assignment("tools.dipp")
        assert assignment.covers(_tool_namespace(None)) is False

    def test_denies_when_the_assignment_carries_no_node(self) -> None:
        malformed = RoleAssignment(
            id=uuid4(),
            role_id=uuid4(),
            group_id=uuid4(),
            scope_type=ScopeType.SUBTREE,
            scope_namespace_id=None,
            scope_namespace_type=None,
            scope_customer_id=None,
            scope_namespace_name=None,
        )
        assert malformed.covers(_tool_namespace("tools.dipp.thing")) is False

    def test_an_empty_node_covers_nothing(self) -> None:
        assignment = _subtree_assignment("")
        assert assignment.covers(_tool_namespace("tools.dipp")) is False
        assert assignment.covers(_tool_namespace("anything.at.all")) is False

    def test_case_and_whitespace_do_not_widen(self) -> None:
        assert _subtree_assignment("tools.DIPP").covers(_tool_namespace("tools.dipp.thing")) is False
        assert _subtree_assignment("tools.dipp ").covers(_tool_namespace("tools.dipp.thing")) is False

    def test_the_scope_namespace_id_is_ignored_under_a_subtree_scope(self) -> None:
        # a subtree root is a NAME. an id on the row would be a second,
        # contradictory statement of the same scope.
        target = _tool_namespace("tools.dipp.thing")
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=uuid4(),
            group_id=uuid4(),
            scope_type=ScopeType.SUBTREE,
            scope_namespace_id=target.id,
            scope_namespace_type=None,
            scope_customer_id=None,
            scope_namespace_name="tools.other",
        )
        assert assignment.covers(target) is False


class TestExistingScopesAreUnchanged:
    """the three shapes that existed keep answering exactly as before."""

    def test_namespace_scope_still_matches_on_id_alone(self) -> None:
        target = _tool_namespace("tools.dipp.thing")
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=uuid4(),
            group_id=uuid4(),
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=target.id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        assert assignment.covers(target) is True

    def test_namespace_scope_ignores_the_name(self) -> None:
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=uuid4(),
            group_id=uuid4(),
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=uuid4(),
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        assert assignment.covers(_tool_namespace("tools.dipp")) is False

    def test_type_customer_scope_is_unaffected_by_the_new_field(self) -> None:
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=uuid4(),
            group_id=uuid4(),
            scope_type=ScopeType.TYPE_CUSTOMER,
            scope_namespace_id=None,
            scope_namespace_type="tool",
            scope_customer_id=CUSTOMER,
        )
        assert assignment.covers(_tool_namespace("tools.dipp", customer_id=CUSTOMER)) is True

    def test_all_scope_covers_a_namespace_with_no_name(self) -> None:
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=uuid4(),
            group_id=uuid4(),
            scope_type=ScopeType.ALL,
            scope_namespace_id=None,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        assert assignment.covers(_tool_namespace(None)) is True


class TestNamespaceNameIsAdditive:
    """the new field defaults, so no construction site is forced to change."""

    def test_name_defaults_to_none(self) -> None:
        namespace = Namespace(
            id=uuid4(),
            customer_id=None,
            namespace_type="tool",
            owner_agent_id=None,
        )
        assert namespace.name is None

    def test_scope_namespace_name_defaults_to_none(self) -> None:
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=uuid4(),
            group_id=uuid4(),
            scope_type=ScopeType.ALL,
            scope_namespace_id=None,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        assert assignment.scope_namespace_name is None


class TestSubtreeGrantThroughTheEvaluator:
    """the whole path, not just ``covers``: membership -> assignment -> role."""

    @staticmethod
    def _store_granting(node: str) -> tuple[FakeStore, UUID]:
        """build a store where one platform group holds a subtree grant.

        the group is PLATFORM-scoped because a tool namespace carries
        ``customer_id IS NULL``, and the evaluator drops a
        customer-scoped group against such a row.

        :param node: subtree root the grant is written against
        :ptype node: str
        :return: the store and the user id it grants to
        :rtype: tuple[FakeStore, UUID]
        """
        store = FakeStore()
        user_id = uuid4()
        group = Group(id=uuid4(), name="tool callers", customer_id=None)
        role = Role(
            id=uuid4(),
            name="ToolCaller",
            permissions={"tool": frozenset({TOOL_CALL})},
            is_built_in=True,
        )
        store.add_group(group)
        store.add_role(role)
        store.add_membership(
            GroupMembership(
                group_id=group.id,
                member_type=MemberType.USER,
                member_id=user_id,
                customer_id=None,
            ),
        )
        store.add_assignment(
            RoleAssignment(
                id=uuid4(),
                role_id=role.id,
                group_id=group.id,
                scope_type=ScopeType.SUBTREE,
                scope_namespace_id=None,
                scope_namespace_type=None,
                scope_customer_id=None,
                scope_namespace_name=node,
            ),
        )
        return store, user_id

    @pytest.mark.asyncio
    async def test_a_subtree_grant_allows_a_descendant(self) -> None:
        store, user_id = self._store_granting("tools.pentest")
        ctx = EvaluationContext(
            namespace=_tool_namespace("tools.pentest.sqlmap"),
            action=TOOL_CALL,
            user_id=user_id,
        )
        assert await evaluate_decision(ctx, cache=make_cache(store)) is True

    @pytest.mark.asyncio
    async def test_a_subtree_grant_denies_a_prefix_sibling(self) -> None:
        store, user_id = self._store_granting("tools.pentest")
        ctx = EvaluationContext(
            namespace=_tool_namespace("tools.pentestimposter.sqlmap"),
            action=TOOL_CALL,
            user_id=user_id,
        )
        assert await evaluate_decision(ctx, cache=make_cache(store)) is False

    @pytest.mark.asyncio
    async def test_a_subtree_grant_denies_an_unrelated_subtree(self) -> None:
        store, user_id = self._store_granting("tools.pentest")
        ctx = EvaluationContext(
            namespace=_tool_namespace("tools.dipp.thing"),
            action=TOOL_CALL,
            user_id=user_id,
        )
        assert await evaluate_decision(ctx, cache=make_cache(store)) is False


class TestSubtreeRowsPartitionWithTheCustomerlessShapes:
    """``row_scope`` is derived from the scope, and a subtree names no customer."""

    def test_the_collection_derives_the_same_scope_as_the_entity(self) -> None:
        # the derivation is written twice -- once in
        # ``RoleAssignmentEntity.__init__`` and once in
        # ``RoleAssignmentCollection.create`` -- and a row written
        # through one path must land in the same partition as one
        # written through the other.
        from threetears.agent.acl.collections import RoleAssignmentCollection

        from .test_collections import _make_collection

        collection = _make_collection(RoleAssignmentCollection)
        entity = collection.create(
            {
                "assignment_id": uuid4(),
                "role_id": uuid4(),
                "group_id": uuid4(),
                "scope_type": "subtree",
                "scope_namespace_name": "tools.pentest",
                "scope_customer_id": None,
            },
        )
        assert entity.to_dict()["row_scope"] == "platform"

    def test_the_entity_derives_platform_scope(self) -> None:
        from threetears.agent.acl.entities import RoleAssignmentEntity

        entity = RoleAssignmentEntity(
            {
                "assignment_id": uuid4(),
                "role_id": uuid4(),
                "group_id": uuid4(),
                "scope_type": "subtree",
                "scope_namespace_name": "tools.pentest",
                "scope_customer_id": None,
            },
        )
        assert entity.to_dict()["row_scope"] == "platform"

    def test_a_namespace_scoped_row_still_derives_customer_scope(self) -> None:
        from threetears.agent.acl.entities import RoleAssignmentEntity

        entity = RoleAssignmentEntity(
            {
                "assignment_id": uuid4(),
                "role_id": uuid4(),
                "group_id": uuid4(),
                "scope_type": "namespace",
                "scope_namespace_id": uuid4(),
                "scope_customer_id": None,
            },
        )
        assert entity.to_dict()["row_scope"] == "customer"
