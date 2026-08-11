"""Tests for the carrier-facet vocabulary (additive, ignorable, wire-safe)."""

from __future__ import annotations

import json

import pytest

from threetears.media.contracts import (
    LOCATOR_KIND_CONTAINING_PAGE,
    LOCATOR_KIND_DIRECT_FILE,
    MediaFacets,
)


def test_locator_kind_constants() -> None:
    """The two locator kinds are stable -- producer + consumer compare on them."""
    assert LOCATOR_KIND_DIRECT_FILE == "direct-file"
    assert LOCATOR_KIND_CONTAINING_PAGE == "containing-page"


def test_every_facet_is_optional() -> None:
    """A result asserting no facets at all constructs with no arguments."""
    facets = MediaFacets()
    assert facets.rights_status is None
    assert facets.width is None
    assert facets.height is None
    assert facets.locator_kind is None
    assert facets.to_metadata() == {}


def test_to_metadata_is_json_safe() -> None:
    """Every projected value survives a JSON round-trip unchanged."""
    facets = MediaFacets(
        rights_status="CC-BY-4.0",
        width=1920,
        height=1080,
        locator_kind=LOCATOR_KIND_DIRECT_FILE,
    )
    projected = facets.to_metadata()
    assert projected == {
        "rights_status": "CC-BY-4.0",
        "width": 1920,
        "height": 1080,
        "locator_kind": "direct-file",
    }
    assert json.loads(json.dumps(projected)) == projected


def test_to_metadata_omits_unset_facets() -> None:
    """Unset facets are absent from the mapping, not carried as nulls."""
    facets = MediaFacets(locator_kind=LOCATOR_KIND_CONTAINING_PAGE)
    assert facets.to_metadata() == {"locator_kind": "containing-page"}


def test_round_trip_preserves_facets() -> None:
    """from_metadata(to_metadata(f)) reconstructs an equal facet set."""
    facets = MediaFacets(rights_status="rights-reserved", width=640, height=480, locator_kind="direct-file")
    assert MediaFacets.from_metadata(facets.to_metadata()) == facets


def test_unrecognised_facets_are_ignored_not_rejected() -> None:
    """SR-C2: a facet this vocabulary does not know is dropped, never raised.

    This is the check a closed vocabulary would fail: a producer on a later
    family version carries a facet nobody here has heard of, and the reader
    keeps working on the facets it does know.
    """
    restored = MediaFacets.from_metadata(
        {
            "width": 800,
            "duration_seconds": 42,
            "x-vendor:frame_rate": 24,
        }
    )
    assert restored == MediaFacets(width=800)


def test_absent_facets_read_as_none() -> None:
    """An empty mapping is a valid facet set: nothing is known, nothing fails."""
    assert MediaFacets.from_metadata({}) == MediaFacets()


def test_from_metadata_coerces_dimensions_at_the_border() -> None:
    """Dimensions arriving as JSON strings become ints, as elsewhere in this package."""
    restored = MediaFacets.from_metadata({"width": "800", "height": "600"})
    assert restored.width == 800
    assert restored.height == 600


def test_from_metadata_bad_dimension_raises() -> None:
    """A known facet carrying an unusable value fails closed -- that is a producer bug."""
    with pytest.raises(ValueError):
        MediaFacets.from_metadata({"width": "not-a-number"})


def test_rights_status_is_an_open_vocabulary() -> None:
    """An upstream rights label nobody here anticipated is carried verbatim."""
    restored = MediaFacets.from_metadata({"rights_status": "getty-editorial-use-only"})
    assert restored.rights_status == "getty-editorial-use-only"
