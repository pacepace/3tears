"""Package-side enforcement: every Collection declares ``partition_column``.

The workspace walker
(``packages/core/tests/enforcement/test_partition_column_enforcement.py``)
audits SQL string literals for the partition predicate. This
per-package walker pins the structural shape: the two Collection
classes in this package must declare ``partition_column = "agent_id"``
as a class attribute and ``primary_key_column`` as a composite tuple
including ``agent_id``.

The class-attribute declaration is the contract that the workspace
walker's ``_PARTITIONED_TABLES`` mapping mirrors -- if a Collection
ever drops the declaration, the SQL walker would still fire but the
local invariant is gone. This guard fires at the package boundary so
the regression is caught immediately.
"""

from __future__ import annotations

import pytest

from threetears.agent.skills.collections import (
    AgentSkillCollection,
    AgentSkillInvocationCollection,
)


@pytest.mark.parametrize(
    "cls,expected_pk",
    [
        (AgentSkillCollection, ("agent_id", "skill_id")),
        (
            AgentSkillInvocationCollection,
            ("agent_id", "invocation_id"),
        ),
    ],
)
def test_partition_column_is_agent_id(
    cls: type,
    expected_pk: tuple[str, ...],
) -> None:
    """Each Collection declares ``partition_column = 'agent_id'``."""
    assert getattr(cls, "partition_column", None) == "agent_id"


@pytest.mark.parametrize(
    "cls,expected_pk",
    [
        (AgentSkillCollection, ("agent_id", "skill_id")),
        (
            AgentSkillInvocationCollection,
            ("agent_id", "invocation_id"),
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
    assert pk[0] == "agent_id", "partition column must be the first PK column"
