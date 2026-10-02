"""owner-scoped keys for coordination primitives over a bucket SHARED by many owners.

The platform's shared pod buckets (``ratelimits``, ``proxy_assertion_nonces``, ``leases``) are each
one bucket every pod of a kind binds, and each pod is granted only the keys under its own scope
(:func:`threetears.nats.subject_permissions.kv_key_scope_for`). A primitive over such a bucket leads
every key with that scope. The ONE rendering and the one validation live here, so the key a
primitive writes and the prefix the grant is narrowed to cannot drift apart per primitive.
"""

from __future__ import annotations

from threetears.nats.subject_permissions import KV_KEY_SCOPE_GRAMMAR

__all__ = ["owner_scoped_key", "validated_key_scope"]


def validated_key_scope(key_scope: str | None, *, primitive: str) -> str | None:
    """refuse an owner scope that is not one literal subject token.

    :param key_scope: the owner scope, or ``None`` for a bucket the primitive's owner has to itself
    :ptype key_scope: str | None
    :param primitive: the primitive's name, for the message
    :ptype primitive: str
    :return: ``key_scope`` unchanged
    :rtype: str | None
    :raises ValueError: when ``key_scope`` is not one literal subject token -- a dot splits it into
        two tokens and a wildcard widens it, so every key would fall outside the owner's grant
    """
    if key_scope is not None and not KV_KEY_SCOPE_GRAMMAR.match(key_scope):
        raise ValueError(
            f"{primitive} key_scope {key_scope!r} must be one literal subject token matching "
            f"{KV_KEY_SCOPE_GRAMMAR.pattern}; a dot splits it into two tokens and a wildcard widens it, "
            f"so its keys would fall outside the owner's grant"
        )
    return key_scope


def owner_scoped_key(key_scope: str | None, key: str) -> str:
    """the KV key a primitive stores ``key`` under.

    :param key_scope: the owner scope, already validated, or ``None``
    :ptype key_scope: str | None
    :param key: the primitive's own key
    :ptype key: str
    :return: ``{key_scope}.{key}``, or ``key`` when there is no scope
    :rtype: str
    """
    return key if key_scope is None else f"{key_scope}.{key}"
