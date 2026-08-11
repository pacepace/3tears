"""Well-known facet keys -- found in ``media-contracts``, not invented (SR-C3, G13).

The result core stays carrier-agnostic (SR-C1); carrier data rides
``Candidate.facets``, keyed by the vocabulary the family already publishes
in ``3tears-media-contracts``. SR-C3 is two-sided: the facets are open AND
the published vocabulary is *pinned rather than redeclared* (success check
13). The keys below are exactly ``MediaInfo``'s field names, and the pin is
structural -- if the vocabulary ever drifts, importing this module fails
loudly instead of the two vocabularies diverging silently.

The genuinely new facets SR-C3 names -- rights status, pixel dimensions,
direct-file-versus-containing-page -- land in ``media-contracts`` itself
(search-spec.md §4 item 1) and gain pinned keys here when they do.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Final

from threetears.media.contracts import MediaInfo

__all__ = [
    "FACET_EXTRACTION_STATUS",
    "FACET_HAS_DOWNLOADABLE_DATA",
    "FACET_MEDIA_CATEGORY",
]

#: carrier taxonomy facet (``"image"``, ``"audio"``, ``"video"``,
#: ``"document"``, ... -- the open taxonomy of ``MediaInfo.media_category``).
#: The ``carrier`` criterion speaks the same vocabulary.
FACET_MEDIA_CATEGORY: Final[str] = "media_category"

#: document/PDF extraction-status facet (``MediaInfo.extraction_status``).
FACET_EXTRACTION_STATUS: Final[str] = "extraction_status"

#: whether the carrier's bytes are fetchable (``MediaInfo.has_downloadable_data``).
FACET_HAS_DOWNLOADABLE_DATA: Final[str] = "has_downloadable_data"

_MEDIA_INFO_FIELD_NAMES: Final[frozenset[str]] = frozenset(field.name for field in fields(MediaInfo))
_PINNED: Final[tuple[str, ...]] = (FACET_MEDIA_CATEGORY, FACET_EXTRACTION_STATUS, FACET_HAS_DOWNLOADABLE_DATA)

for _key in _PINNED:
    if _key not in _MEDIA_INFO_FIELD_NAMES:
        raise ImportError(
            f"facet key {_key!r} is no longer a field of threetears.media.contracts.MediaInfo; "
            f"the facet vocabulary is pinned to media-contracts (SR-C3) and must move with it"
        )
