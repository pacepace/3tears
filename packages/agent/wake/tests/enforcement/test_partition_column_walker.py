"""Package-side enforcement: every Collection declares ``partition_column``.

The workspace walker
(``packages/core/tests/enforcement/test_partition_column_enforcement.py``)
audits SQL string literals for the partition predicate. This
per-package walker pins the structural shape: the three Collection
classes in this package must declare
``partition_column = "conversation_id"`` as a class attribute and
``primary_key_column`` as a composite tuple including
``conversation_id``.

The class-attribute declaration is the contract that the workspace
walker's ``_PARTITIONED_TABLES`` mapping mirrors -- if a Collection
ever drops the declaration, the SQL walker would still fire but the
local invariant is gone. This guard fires at the package boundary so
the regression is caught immediately.
"""

from __future__ import annotations

import pytest

from threetears.agent.wake.collections import (
    WakeFireCollection,
    WakeScheduleCollection,
    WebhookSubscriptionCollection,
)


@pytest.mark.parametrize(
    "cls,expected_pk",
    [
        (
            WakeScheduleCollection,
            ("conversation_id", "schedule_id"),
        ),
        (
            WakeFireCollection,
            ("conversation_id", "fire_id"),
        ),
        (
            WebhookSubscriptionCollection,
            ("conversation_id", "subscription_id"),
        ),
    ],
)
def test_partition_column_is_conversation_id(
    cls: type,
    expected_pk: tuple[str, ...],
) -> None:
    """Each Collection declares ``partition_column = 'conversation_id'``."""
    del expected_pk
    assert getattr(cls, "partition_column", None) == "conversation_id"


@pytest.mark.parametrize(
    "cls,expected_pk",
    [
        (
            WakeScheduleCollection,
            ("conversation_id", "schedule_id"),
        ),
        (
            WakeFireCollection,
            ("conversation_id", "fire_id"),
        ),
        (
            WebhookSubscriptionCollection,
            ("conversation_id", "subscription_id"),
        ),
    ],
)
def test_primary_key_is_composite_including_partition(
    cls: type,
    expected_pk: tuple[str, ...],
) -> None:
    """Each Collection's composite PK starts with the partition column."""
    pk = cls.primary_key_column  # type: ignore[attr-defined]
    assert pk == expected_pk
    assert pk[0] == "conversation_id", "partition column must be the first PK column"
