"""equal access shapes are handed back as one object; distinct ones stay distinct.

The contract this pins:

- an equal shape, however it was built, comes back as the identical object first seen;
- distinct shapes stay distinct, each its own canonical object;
- adding shapes keeps O(log shapes) chunks, so a new shape never copies every shape held;
- past ``max_shapes`` it starts afresh, and still answers an equal shape for every shape.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import AccessShape, AccessShapeInterner
from threetears.agent.acl.types import GroupMembership, MemberType


def _shape(groups: tuple[UUID, ...], customer: UUID | None) -> AccessShape:
    person = uuid4()
    return AccessShape.of(
        GroupMembership(
            group_id=UUID(bytes=group.bytes),
            member_type=MemberType.USER,
            member_id=person,
            customer_id=None if customer is None else UUID(bytes=customer.bytes),
        )
        for group in groups
    )


def test_equal_shapes_come_back_as_the_identical_object() -> None:
    interner = AccessShapeInterner()
    groups, customer = (uuid4(), uuid4()), uuid4()
    first = _shape(groups, customer)
    again = _shape(tuple(reversed(groups)), customer)
    assert again == first
    assert again is not first
    assert interner.intern(first) is first
    assert interner.intern(again) is first
    assert interner.held == 1


def test_distinct_shapes_stay_distinct() -> None:
    interner = AccessShapeInterner()
    group, customer = uuid4(), uuid4()
    one = interner.intern(_shape((group,), customer))
    other_customer = interner.intern(_shape((group,), uuid4()))
    platform = interner.intern(_shape((group,), None))
    other_group = interner.intern(_shape((uuid4(),), customer))
    assert len({id(one), id(other_customer), id(platform), id(other_group)}) == 4
    assert interner.held == 4
    assert interner.intern(_shape((group,), customer)) is one
    assert interner.intern(_shape((group,), None)) is platform


def test_many_shapes_are_held_in_few_chunks() -> None:
    interner = AccessShapeInterner()
    shapes = [interner.intern(_shape((uuid4(),), None)) for _ in range(1_000)]
    # 1,000 = 0b1111101000: one chunk per set bit, never one per shape
    assert interner.chunks == bin(1_000).count("1")
    assert all(interner.intern(AccessShape(memberships=shape.memberships)) is shape for shape in shapes)


def test_past_its_limit_it_starts_afresh() -> None:
    interner = AccessShapeInterner(max_shapes=2)
    first = interner.intern(_shape((uuid4(),), None))
    interner.intern(_shape((uuid4(),), None))
    third = interner.intern(_shape((uuid4(),), None))
    assert interner.held == 1
    assert interner.intern(AccessShape(memberships=third.memberships)) is third
    equal_to_first = AccessShape(memberships=first.memberships)
    assert interner.intern(equal_to_first) is equal_to_first


def test_it_holds_at_least_one_shape() -> None:
    with pytest.raises(ValueError, match="at least one shape"):
        AccessShapeInterner(max_shapes=0)
