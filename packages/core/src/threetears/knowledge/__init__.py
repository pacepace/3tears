"""shared knowledge-layer scope vocabulary + entry merge.

this submodule is the SINGLE authority for the two pure pieces that the
hub (eval fingerprint builder, persistence) and the SDK (turn-time
retrieval node) must agree on byte-for-byte (knowledge-task-04):

- :class:`Scope` + :func:`derive_scope` — the three-scope ladder (D1),
  scope derived from ``(customer_id, user_id)`` nullability.
- :func:`merge_entry_views` / :func:`merge_concept_views` and their
  snapshot / effective / layered shapes — read-time whole-entry shadow
  resolution (D4 / KNW-11 entries, KNW-21 concepts). both adapt onto
  the shared :func:`resolve_shadow_chains` walk so a hub fingerprint and
  a live turn agree byte-for-byte on the effective view.

hub-only concerns stay in the hub on top of this contract: the
``ScopedFields`` Pydantic mixin (API-boundary convenience) and the
``scope_visibility_clause`` SQL helper (composes against hub RBAC) are
NOT exported here because they depend on hub persistence / hub RBAC.
ships inside the ``3tears`` (core) distribution; the runtime version is
:data:`threetears.core.__version__`.
"""

from __future__ import annotations

from threetears.knowledge.chains import (
    ResolvedChain,
    resolve_shadow_chains,
)
from threetears.knowledge.concept_merge import (
    ConceptEffective,
    ConceptLayered,
    ConceptSnapshot,
    build_table_ref,
    merge_concept_views,
)
from threetears.knowledge.events import (
    DraftCommandOp,
    DraftTarget,
    KnowledgeDraftCommand,
    KnowledgeDraftEvent,
    KnowledgeDraftMessage,
)
from threetears.knowledge.merge import (
    MAX_SHADOW_CHAIN_DEPTH,
    EntryEffective,
    EntryEnforcement,
    EntryLayered,
    EntrySnapshot,
    OriginCycleError,
    assert_no_origin_cycle,
    merge_entry_views,
)
from threetears.knowledge.scope import Scope, derive_scope

__all__ = [
    "MAX_SHADOW_CHAIN_DEPTH",
    "ConceptEffective",
    "ConceptLayered",
    "ConceptSnapshot",
    "DraftCommandOp",
    "DraftTarget",
    "EntryEffective",
    "EntryEnforcement",
    "EntryLayered",
    "EntrySnapshot",
    "KnowledgeDraftCommand",
    "KnowledgeDraftEvent",
    "KnowledgeDraftMessage",
    "OriginCycleError",
    "ResolvedChain",
    "Scope",
    "assert_no_origin_cycle",
    "build_table_ref",
    "derive_scope",
    "merge_concept_views",
    "merge_entry_views",
    "resolve_shadow_chains",
]
