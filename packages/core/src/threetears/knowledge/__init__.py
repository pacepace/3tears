"""shared knowledge-layer scope vocabulary + entry merge.

this submodule is the SINGLE authority for the two pure pieces that the
hub (eval fingerprint builder, persistence) and the SDK (turn-time
retrieval node) must agree on byte-for-byte (knowledge-task-04):

- :class:`Scope` + :func:`derive_scope` — the three-scope ladder (D1),
  scope derived from ``(customer_id, user_id)`` nullability.
- :func:`merge_entry_views` and its snapshot / effective / layered
  shapes — read-time whole-entry shadow resolution (D4 / KNW-11).

hub-only concerns stay in the hub on top of this contract: the
``ScopedFields`` Pydantic mixin (API-boundary convenience) and the
``scope_visibility_clause`` SQL helper (composes against hub RBAC) are
NOT exported here because they depend on hub persistence / hub RBAC.
ships inside the ``3tears`` (core) distribution; the runtime version is
:data:`threetears.core.__version__`.
"""

from __future__ import annotations

from threetears.knowledge.merge import (
    MAX_SHADOW_CHAIN_DEPTH,
    EntryEffective,
    EntryLayered,
    EntrySnapshot,
    OriginCycleError,
    assert_no_origin_cycle,
    merge_entry_views,
)
from threetears.knowledge.scope import Scope, derive_scope

__all__ = [
    "MAX_SHADOW_CHAIN_DEPTH",
    "EntryEffective",
    "EntryLayered",
    "EntrySnapshot",
    "OriginCycleError",
    "Scope",
    "assert_no_origin_cycle",
    "derive_scope",
    "merge_entry_views",
]
