"""the identity namespace name spells its ids in full.

the name is never persisted -- the owner descriptor is built in-process -- but the evaluator
judges a subtree grant against it, so two agents sharing a name would share every such grant.
a uuid7 leads with its millisecond timestamp, so a name cut from its head is shared by every
agent minted in the same moment.
"""

from __future__ import annotations

from uuid import UUID

from threetears.agent.identity import identity_namespace_name


def test_shape() -> None:
    agent_id = UUID("019470a8-b5c3-7def-8123-456789abcdef")
    customer_id = UUID("11112222-3333-4444-5555-666677778888")
    assert identity_namespace_name(agent_id, customer_id) == (
        "identity.019470a8b5c37def8123456789abcdef.11112222333344445555666677778888"
    )


def test_two_agents_minted_in_one_millisecond_get_distinct_names() -> None:
    agent_a = UUID("019470a8-b5c3-7def-8123-456789abcdef")
    agent_b = UUID("019470a8-b5c3-7a01-9fed-cba987654321")
    customer_id = UUID("019470a8-b5c4-7000-8000-000000000001")
    assert agent_a.int >> 80 == agent_b.int >> 80, "the fixture must share the uuid7 timestamp"
    assert identity_namespace_name(agent_a, customer_id) != identity_namespace_name(agent_b, customer_id)
