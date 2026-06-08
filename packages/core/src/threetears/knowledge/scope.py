"""knowledge-layer scope contract: the three-scope ladder (D1).

every knowledge-layer entity (playbook entries, concepts, eval cases)
is born onto the same scoping shape: a nullable ``customer_id``, a
nullable ``user_id``, and a ``visibility`` value. scope is DERIVED from
nullability, never stored as a separate column, so the column set
cannot drift out of agreement with the declared scope:

================ =============== ============= ===================
scope            ``customer_id`` ``user_id``   promoted to next by
================ =============== ============= ===================
``user``         set             set           ``customer.admin``
``customer``     set             NULL          ``super_admin``
``platform``     NULL            NULL          — (top)
================ =============== ============= ===================

the ``user_id``-set / ``customer_id``-NULL combination is illegal: a
user always belongs to a customer, so a user-scope row that names no
customer is a contradiction. that illegal state is made unrepresentable
wherever the scope columns are constructed — the consuming hub layers a
Pydantic model and a per-entity table CHECK constraint over this pure
derivation so the model is convenience and the constraint is the
contract; this module owns only the derivation both reuse.

this module is the SINGLE authority on scope vocabulary shared by the
hub (fingerprint builder + persistence) and the SDK (turn-time
retrieval node). the pure :class:`Scope` enum and :func:`derive_scope`
live here; hub-only concerns (the ``ScopedFields`` Pydantic mixin and
the ``scope_visibility_clause`` SQL helper that composes against hub
RBAC) stay in the hub on top of this contract.
"""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

__all__ = [
    "Scope",
    "derive_scope",
]


class Scope(StrEnum):
    """the three scopes a knowledge-layer entity can live at (D1).

    derived from ``(customer_id, user_id)`` nullability — never stored
    as a column. the string values match the ``target_scope`` literals
    on ``platform.promotion_requests`` so a promotion target reuses the
    same vocabulary.

    :cvar USER: customer_id set AND user_id set
    :cvar CUSTOMER: customer_id set, user_id NULL
    :cvar PLATFORM: both NULL
    """

    USER = "user"
    CUSTOMER = "customer"
    PLATFORM = "platform"


def derive_scope(*, customer_id: UUID | None, user_id: UUID | None) -> Scope:
    """derive the scope of a knowledge row from its scope columns.

    scope is a pure function of nullability per D1. the
    ``user_id``-set / ``customer_id``-NULL combination is rejected: a
    user always belongs to a customer.

    :param customer_id: owning customer (``None`` for platform scope)
    :ptype customer_id: UUID | None
    :param user_id: owning user (``None`` for customer / platform scope)
    :ptype user_id: UUID | None
    :return: derived :class:`Scope`
    :rtype: Scope
    :raises ValueError: when ``user_id`` is set but ``customer_id`` is
        NULL (the illegal combination)
    """
    if user_id is not None and customer_id is None:
        raise ValueError(
            "invalid knowledge scope: user_id set without customer_id "
            "(a user always belongs to a customer)",
        )
    if user_id is not None:
        result = Scope.USER
    elif customer_id is not None:
        result = Scope.CUSTOMER
    else:
        result = Scope.PLATFORM
    return result
