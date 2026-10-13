"""the identity that keeps one tile source's caches apart from every other's.

a tile collection and its feature caches are keyed by ``(layer, version, ...)``: a layer
NAME and a generation, nothing naming whose layer it is. two sources serving a layer of the
same name -- two datasources, or a customer's datasource and a platform layer -- would
otherwise share every cache tier the collection framework keys by table: the pod-local L1
table, the NATS L2 keys, the cross-pod build lock and the registry entry. whichever built a
tile first would answer for both, across tenants.

so each source names a cache scope, and the collections' table names carry it. the table name
is interpolated into SQL (the L1 table, the feature cache's R-Tree companions) and into NATS KV
keys, so the scope is held to a grammar both accept unquoted: a lowercase letter, then
lowercase letters, digits and underscores, at most :data:`MAX_CACHE_SCOPE_LENGTH` characters.
``ds_<32 hex>`` for a datasource and ``ns_<32 hex>`` for a provider namespace both fit.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["MAX_CACHE_SCOPE_LENGTH", "check_cache_scope"]

#: long enough for a two-letter tag, an underscore and a 32-hex UUID, with room to spare
MAX_CACHE_SCOPE_LENGTH: Final = 48

_CACHE_SCOPE: Final = re.compile(rf"[a-z][a-z0-9_]{{0,{MAX_CACHE_SCOPE_LENGTH - 1}}}")


def check_cache_scope(scope: str) -> str:
    """return ``scope`` when it is a usable cache scope; refuse anything else.

    :param scope: the tile source's cache identity
    :ptype scope: str
    :return: the same scope
    :rtype: str
    :raises ValueError: when it is not a lowercase identifier of at most
        :data:`MAX_CACHE_SCOPE_LENGTH` characters
    """
    if not isinstance(scope, str) or _CACHE_SCOPE.fullmatch(scope) is None:
        raise ValueError(
            f"cache scope {scope!r} is not a lowercase identifier of at most {MAX_CACHE_SCOPE_LENGTH} "
            "characters (a letter, then letters, digits and underscores); it names SQL tables and NATS keys"
        )
    return scope
