"""how far out an HTTP response may travel, as ONE vocabulary.

this enum was born in ``threetears.datasources.geo_config`` under the name
``CacheClassConfig``, serving map tiles only. it is promoted here because a
second consumer arrived -- the inbound REST affordance a tool may declare
(:mod:`threetears.agent.tools.http_operation`) -- and neither package may
import the other. ``threetears.datasources`` re-exports the name it used to
own, so the geo declaration surface is unchanged; there is one enum object
with two names, not two vocabularies.

the rule the vocabulary exists to encode, restated:

**cacheability is derived from the resource's own classification and may be
narrowed by a declaration, never widened.** a free-standing "this is
cacheable" flag beside an already-recorded sensitivity is a second place to
say the same thing, and the two drift; widening is how a per-caller
authorized response reaches a shared edge cache it was never cleared for.
:func:`narrow_cache_class` is that rule as a function, and it is the only
sanctioned way to turn an inherited class plus a declaration into an
effective one.

what the three resolved classes mean at an HTTP shared cache, which keys on
URL plus ``Vary`` and CANNOT key on a bearer token:

- ``PUBLIC`` -- the response genuinely does not vary by caller. shared-edge
  cacheable with no check on the way in.
- ``AUTHENTICATED`` -- shared-edge cacheable, but the edge verifies a token
  before serving. the directives match ``public``; the difference is the
  check, not the headers.
- ``PRIVATE`` -- origin only. renders ``private, no-store``, and is what an
  unrecognised classification falls back to.

a resource that varies by tenant is therefore either origin-only or carries
its tenant in the path -- there is no third option, because the tenant is
not in the cache key otherwise.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import StrEnum
from typing import Final

from threetears.observe import get_logger

__all__ = [
    "SCOPE_QUALIFIER_SEPARATOR",
    "CacheClass",
    "narrow_cache_class",
    "qualified_scope",
    "scope_qualifier",
]

#: what joins a scope to its qualifier in a URL and an edge token (``va.3f9a1c2b7e04``): never the
#: ``-`` that joins a URL's scopes, and never a character a namespace segment carries
SCOPE_QUALIFIER_SEPARATOR: Final = "."

#: hex characters of the qualifier's digest: 48 bits, far past any count of distinct child sets
_QUALIFIER_HEX: Final = 12


def scope_qualifier(children: Iterable[str]) -> str:
    """the fingerprint of a reader's children beneath a REST affordance's ``scope_qualifier_node``.

    The hub, filling an edge token, and a tool, building a cache URL, each compute it from the
    children the same discovery lists, so the two name one scope. Order and repeats do not change it.

    :param children: the child segments the reader sees (report ids)
    :ptype children: Iterable[str]
    :return: twelve lower-case hex characters
    :rtype: str
    """
    joined = ",".join(sorted(set(children)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:_QUALIFIER_HEX]


def qualified_scope(scope: str, qualifier: str | None) -> str:
    """a scope as a reader with ``qualifier`` reads it: unchanged when the reader sees every child.

    :param scope: the scope ``scope_node`` names (``va``)
    :ptype scope: str
    :param qualifier: :func:`scope_qualifier` of the reader's children, or ``None`` when the reader sees
        every child of the qualifier node
    :ptype qualifier: str | None
    :return: ``va``, or ``va.3f9a1c2b7e04``
    :rtype: str
    """
    return scope if qualifier is None else f"{scope}{SCOPE_QUALIFIER_SEPARATOR}{qualifier}"


log = get_logger(__name__)


class CacheClass(StrEnum):
    """how far out a response may travel.

    a declaration may narrow what its resource's own classification implies
    and may never widen it. widening would let a declaration overrule the
    resource's classification, which is exactly how data reaches an edge
    cache it was never cleared for.
    """

    #: take the resource's own classification. the default and the
    #: overwhelmingly common case.
    INHERIT = "inherit"
    #: shared across customers. cacheable at a shared edge with no check on
    #: the way in, so the widest reach of the three.
    PUBLIC = "public"
    #: cacheable at a shared edge, but the edge verifies a token first. the
    #: cache directives match ``public``; the difference is the check, not
    #: the headers.
    AUTHENTICATED = "authenticated"
    #: never reaches a shared cache at all. the narrowest reach, and what an
    #: unrecognised resource classification falls back to.
    PRIVATE = "private"


#: increasing exposure. narrowing means moving toward index 0. ``INHERIT`` is
#: deliberately absent: it is a declaration-side sentinel, not an exposure.
_EXPOSURE_ORDER: tuple[CacheClass, ...] = (
    CacheClass.PRIVATE,
    CacheClass.AUTHENTICATED,
    CacheClass.PUBLIC,
)


def narrow_cache_class(inherited: CacheClass, declared: CacheClass) -> CacheClass:
    """apply a declaration to an inherited class, narrowing only.

    :param inherited: the resource's own resolved classification; must not
        be :attr:`CacheClass.INHERIT`, which is a declaration-side sentinel
    :ptype inherited: CacheClass
    :param declared: the declaration's own class, or
        :attr:`CacheClass.INHERIT` to take the resource's
    :ptype declared: CacheClass
    :return: effective class, never more exposed than ``inherited``
    :rtype: CacheClass
    :raises ValueError: when ``inherited`` is :attr:`CacheClass.INHERIT`
    """
    if inherited is CacheClass.INHERIT:
        msg = "inherited cache class must be a resolved class, not INHERIT"
        raise ValueError(msg)
    if declared is CacheClass.INHERIT:
        effective = inherited
    elif _EXPOSURE_ORDER.index(declared) > _EXPOSURE_ORDER.index(inherited):
        log.warning(
            "cache class declaration is wider than its resource; using the resource's",
            extra={"extra_data": {"declared": declared.value, "inherited": inherited.value}},
        )
        effective = inherited
    else:
        effective = declared
    return effective
