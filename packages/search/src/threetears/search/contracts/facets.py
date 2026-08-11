"""Well-known facet keys -- found in ``media-contracts``, not invented (SR-C3, G13).

The result core stays carrier-agnostic (SR-C1); carrier data rides
``Candidate.facets``, keyed by the vocabulary the family already publishes
in ``3tears-media-contracts``. SR-C3 is two-sided: the facets are open AND
the published vocabulary is *pinned rather than redeclared* (success check
13). The keys below are exactly ``MediaInfo``'s field names, and the pin is
structural -- if the vocabulary ever drifts, importing this module fails
loudly instead of the two vocabularies diverging silently.

The genuinely new facets SR-C3 names -- rights status, pixel dimensions,
direct-file-versus-containing-page -- landed in ``media-contracts`` itself
as ``MediaFacets`` (search-spec.md §4 item 1), and their keys are pinned
here the same structural way: the key names are that dataclass's own field
names, checked at import.

Pixel dimensions are one facet spelled as two keys, ``width`` and
``height``, because that is how ``MediaFacets`` and ``GeneratedImage``
already spell them -- the two read across without a translation step, which
is worth more here than a nested value would be.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Final

from threetears.media.contracts import MediaInfo
from threetears.media.contracts.facets import MediaFacets

__all__ = [
    "FACET_EXTRACTION_STATUS",
    "FACET_HAS_DOWNLOADABLE_DATA",
    "FACET_HEIGHT",
    "FACET_LOCATOR_KIND",
    "FACET_MEDIA_CATEGORY",
    "FACET_RIGHTS_STATUS",
    "FACET_WIDTH",
]

#: carrier taxonomy facet (``"image"``, ``"audio"``, ``"video"``,
#: ``"document"``, ... -- the open taxonomy of ``MediaInfo.media_category``).
#: The ``carrier`` criterion speaks the same vocabulary.
FACET_MEDIA_CATEGORY: Final[str] = "media_category"

#: document/PDF extraction-status facet (``MediaInfo.extraction_status``).
FACET_EXTRACTION_STATUS: Final[str] = "extraction_status"

#: whether the carrier's bytes are fetchable (``MediaInfo.has_downloadable_data``).
FACET_HAS_DOWNLOADABLE_DATA: Final[str] = "has_downloadable_data"

#: usage-rights label, verbatim from the provider (``MediaFacets.rights_status``).
#: Open vocabulary: ``None`` means the provider said nothing, which is not
#: the same as unrestricted.
FACET_RIGHTS_STATUS: Final[str] = "rights_status"

#: pixel width (``MediaFacets.width``).
FACET_WIDTH: Final[str] = "width"

#: pixel height (``MediaFacets.height``).
FACET_HEIGHT: Final[str] = "height"

#: whether the candidate's primary locator addresses the file itself or the
#: page containing it (``MediaFacets.locator_kind``; values
#: ``LOCATOR_KIND_DIRECT_FILE`` / ``LOCATOR_KIND_CONTAINING_PAGE``).
FACET_LOCATOR_KIND: Final[str] = "locator_kind"

_MEDIA_INFO_FIELD_NAMES: Final[frozenset[str]] = frozenset(field.name for field in fields(MediaInfo))
_MEDIA_FACET_FIELD_NAMES: Final[frozenset[str]] = frozenset(field.name for field in fields(MediaFacets))

#: every pinned key, paired with the ``media-contracts`` type whose field
#: names it is pinned to.
_PINNED: Final[tuple[tuple[str, str, frozenset[str]], ...]] = tuple(
    (key, "MediaInfo", _MEDIA_INFO_FIELD_NAMES)
    for key in (FACET_MEDIA_CATEGORY, FACET_EXTRACTION_STATUS, FACET_HAS_DOWNLOADABLE_DATA)
) + tuple(
    (key, "MediaFacets", _MEDIA_FACET_FIELD_NAMES)
    for key in (FACET_RIGHTS_STATUS, FACET_WIDTH, FACET_HEIGHT, FACET_LOCATOR_KIND)
)

for _key, _owner, _owner_fields in _PINNED:
    if _key not in _owner_fields:
        raise ImportError(
            f"facet key {_key!r} is no longer a field of threetears.media.contracts.{_owner}; "
            f"the facet vocabulary is pinned to media-contracts (SR-C3) and must move with it"
        )
