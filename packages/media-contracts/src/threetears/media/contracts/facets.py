"""Carrier facets -- the additive vocabulary a media result may carry.

A facet is the per-carrier detail a *carrier-neutral* result core cannot
hold: an image's pixel dimensions, the rights it is offered under, whether
its locator addresses the file itself or the page containing it. The core
stays carrier-agnostic; facets ride alongside it, keyed by name.

Two rules make the vocabulary safe to grow:

- **Additive only.** A facet is added, never repurposed or removed within a
  family major. Everything here is optional, and absence means "not known"
  rather than "not so".
- **Unrecognised facets are ignored, not rejected.** A producer running a
  later family version may carry a facet this reader has never heard of, and
  the reader keeps working -- so :meth:`MediaFacets.from_metadata` drops keys
  it does not know instead of raising. That is what lets a new carrier ship
  without a coordinated release across every consumer.

The vocabulary lives here, beside the carrier taxonomy it belongs to
(:class:`~threetears.media.contracts.protocols.MediaInfo`'s ``media_category``
and ``extraction_status``, and the
:class:`~threetears.media.contracts.protocols.ObjectHandle` that answers "how
are the bytes fetchable"), rather than in each retrieval package that produces
one. A second media vocabulary in a search or discovery leaf is this
vocabulary, drifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "LOCATOR_KIND_CONTAINING_PAGE",
    "LOCATOR_KIND_DIRECT_FILE",
    "MediaFacets",
]

#: :attr:`MediaFacets.locator_kind`: the locator addresses the media bytes
#: themselves -- fetching it yields the file.
LOCATOR_KIND_DIRECT_FILE = "direct-file"

#: :attr:`MediaFacets.locator_kind`: the locator addresses a page the media
#: appears *on* -- fetching it yields markup, and the bytes need a second step.
LOCATOR_KIND_CONTAINING_PAGE = "containing-page"


@dataclass
class MediaFacets:
    """The facets known to this vocabulary, for one media result.

    Every facet is optional and defaults to ``None`` meaning "not reported":
    a producer fills in what its upstream told it and leaves the rest alone,
    and a consumer reads only the facets it understands.

    ``rights_status`` is an **open** vocabulary, deliberately: providers
    report usage rights in labels nobody here controls (a licence identifier,
    a Creative Commons label, a provider's own bucket name), and coercing
    those into a closed set would lose the distinction that made a consumer
    ask. Carry the upstream label verbatim; ``None`` means the provider said
    nothing, which is not the same as "unrestricted".

    ``locator_kind`` is a *closed* distinction by contrast -- it is ours, not
    a provider's -- so its two values ship as constants
    (:data:`LOCATOR_KIND_DIRECT_FILE`, :data:`LOCATOR_KIND_CONTAINING_PAGE`)
    for callers to compare against rather than as loose strings.

    ``width``/``height`` are pixel dimensions, spelled the way
    :class:`~threetears.media.contracts.protocols.GeneratedImage` already
    spells them so the two can be read across without a translation step.
    """

    rights_status: str | None = None
    width: int | None = None
    height: int | None = None
    locator_kind: str | None = None  # LOCATOR_KIND_DIRECT_FILE | LOCATOR_KIND_CONTAINING_PAGE

    def to_metadata(self) -> dict[str, Any]:
        """Project the facets that say something to a JSON-safe dict.

        Unset facets are omitted rather than carried as explicit nulls, so the
        mapping is exactly the set of facets this result asserts -- a reader
        iterating its keys never has to tell a reported ``None`` apart from an
        absent one.

        :return: a JSON-safe mapping of facet name to value, holding only the
            facets that are set
        :rtype: dict[str, Any]
        """
        facets: dict[str, Any] = {}
        if self.rights_status is not None:
            facets["rights_status"] = self.rights_status
        if self.width is not None:
            facets["width"] = self.width
        if self.height is not None:
            facets["height"] = self.height
        if self.locator_kind is not None:
            facets["locator_kind"] = self.locator_kind
        return facets

    @classmethod
    def from_metadata(cls, data: dict[str, Any]) -> MediaFacets:
        """Read the facets this vocabulary knows, ignoring every other key.

        Deliberately lenient where the rest of this package is strict: an
        unknown key is a facet from a later version of the vocabulary, and
        dropping it is the contract (a consumer that does not recognise a
        facet ignores it rather than failing). The leniency stops there --
        a *known* facet carrying an unusable value still fails, because that
        is a producer bug rather than a version gap.

        :param data: a facet mapping (as produced by :meth:`to_metadata`),
            possibly carrying facets this reader does not know
        :ptype data: dict[str, Any]
        :return: the facets this vocabulary recognises; absent ones are None
        :rtype: MediaFacets
        :raises ValueError: when ``width`` or ``height`` is present but does
            not read as an integer
        :raises TypeError: when ``width`` or ``height`` is present but is of
            a type no integer can be read from
        """
        width = data.get("width")
        height = data.get("height")
        rights_status = data.get("rights_status")
        locator_kind = data.get("locator_kind")
        return cls(
            rights_status=str(rights_status) if rights_status is not None else None,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
            locator_kind=str(locator_kind) if locator_kind is not None else None,
        )
