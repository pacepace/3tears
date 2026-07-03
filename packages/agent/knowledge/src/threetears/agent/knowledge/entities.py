"""Agent-side knowledge entity proxies -- concepts + playbook entries.

The agent pod owns its OWN :class:`ConceptEntity` / :class:`PlaybookEntryEntity`
proxies over the ``platform.concepts`` / ``platform.playbook_entries`` tables it
reads through the ``system.platform.rbac`` NATS proxy pool. They are trivial
:class:`~threetears.core.entities.base.BaseEntity` subclasses (a primary-key
declaration only): the fields match the table shape one-to-one MINUS the
``embedding`` column, which the agent-side collections exclude from their
projection (the hub never writes a real embedding and the similarity ranker
fetches vectors on its own schedule via a separate bounded read).
"""

from __future__ import annotations

from threetears.core.entities.base import BaseEntity

__all__ = ["ConceptEntity", "PlaybookEntryEntity"]


class ConceptEntity(BaseEntity):
    """Entity proxy over a ``platform.concepts`` row (agent side).

    The governed term -> binding mapping (KNW-20 / D3). Fields: ``id`` /
    scope columns (``customer_id`` / ``user_id`` / ``visibility``) /
    ``origin_concept_id`` (shadow lineage) / ``name`` / ``aliases`` (JSONB) /
    ``definition`` / ``datasource_table_id`` (binding) / ``sql_fragment``
    (curated filter fragment, never executed) / ``caveats`` / ``tags`` (JSONB) /
    ``always_inject`` (the KNW-25 invariant flag) / ``date_created`` /
    ``date_updated``.
    """

    primary_key_field: str = "id"


class PlaybookEntryEntity(BaseEntity):
    """Entity proxy over a ``platform.playbook_entries`` row (agent side).

    The atomic, promotable knowledge unit (KNW-10). Fields: ``id`` /
    ``playbook_id`` / scope columns (``customer_id`` / ``user_id`` /
    ``visibility``) / ``origin_entry_id`` (shadow lineage) / ``title`` / ``body``
    / ``tags`` (JSONB) / ``datasource_id`` (REQUIRED anchor, knowledge-task-08) /
    ``always_inject`` (the KNW-17 invariant flag) / ``date_created`` /
    ``date_updated``.
    """

    primary_key_field: str = "id"
