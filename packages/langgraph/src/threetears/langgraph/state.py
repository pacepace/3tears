"""Shared LangGraph state-channel reducers for ``create_agent`` middleware.

Homes the ``metadata`` channel reducer the before-model injection middleware
(memory, schema-priming, governed-knowledge) all write through. Each injector
stashes its own keys under the single ``metadata`` state channel; without a
merge reducer LangGraph's default channel behaviour lets a later writer's dict
OVERWRITE an earlier writer's dict, so two injectors on the same turn would
clobber each other's ledgers. :func:`merge_metadata` is the shallow-merge
reducer that lets the channel compose: every injector's keys survive because the
update dict is merged over the existing dict rather than replacing it.

A shallow (top-level) merge is deliberate: each injector owns a DISJOINT set of
top-level keys (``surfaced_memory_ids`` for memory, ``documented_schema_block``
for schema, ``governed_knowledge_block`` + the shadow-disclosure ledgers for
knowledge), so a key-level last-writer-wins is exactly the intended semantics and
a deep merge would only add cost and surprise.
"""

from __future__ import annotations

from typing import Any

__all__ = ["merge_metadata"]


def merge_metadata(
    existing: dict[str, Any] | None,
    update: dict[str, Any] | None,
) -> dict[str, Any]:
    """Shallow-merge a metadata update over the existing metadata channel value.

    The reducer for the ``metadata`` state channel shared by the before-model
    injection middleware. Returns a NEW dict carrying ``existing``'s keys
    overlaid with ``update``'s keys, so concurrent injectors (each owning a
    disjoint set of top-level keys) compose rather than clobber. Either side may
    be ``None`` (the channel is unset, or a middleware returned no metadata),
    which is treated as an empty mapping.

    :param existing: the current channel value, or ``None`` when unset.
    :ptype existing: dict[str, Any] | None
    :param update: the incoming update, or ``None`` when there is nothing to add.
    :ptype update: dict[str, Any] | None
    :return: a new dict with ``update``'s keys overlaid on ``existing``'s.
    :rtype: dict[str, Any]
    """
    return {**(existing or {}), **(update or {})}
