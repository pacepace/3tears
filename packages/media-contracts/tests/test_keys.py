"""Tests for the scope-first object-key builder."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from threetears.media.contracts.keys import SHARED_PREFIX, build_object_key, sanitize_segment

_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_OBJECT = UUID("019f1924-1a31-72d3-81b4-855415bd34ba")
_CREATED = datetime(2026, 6, 30, 14, 5, 0, tzinfo=UTC)


def test_build_object_key_scope_first_layout() -> None:
    """Key follows <customer>/<scope>/<category>/<Y/M/D>/<object>/<file>."""
    key = build_object_key(
        customer_id=_CUSTOMER,
        scope="engagement-019f17a2",
        category="reports",
        object_id=_OBJECT,
        created=_CREATED,
        filename="ACME Corp Pentest.pdf",
    )
    assert key == (
        "06a41d51-a6d5-7824-8000-29ab66754fc0/engagement-019f17a2/reports/"
        "2026/06/30/019f1924-1a31-72d3-81b4-855415bd34ba/acme-corp-pentest.pdf"
    )


def test_customer_id_is_the_leading_prefix() -> None:
    """Tenant isolation: every key starts with the verified customer id."""
    key = build_object_key(
        customer_id=_CUSTOMER,
        scope="conversation-x",
        category="evidence",
        object_id=_OBJECT,
        created=_CREATED,
        filename="dump.pcap",
    )
    assert key.startswith(f"{_CUSTOMER}/")


def test_filename_extension_preserved() -> None:
    """The original extension survives sanitization (download naming)."""
    key = build_object_key(
        customer_id=_CUSTOMER,
        scope="s",
        category="exports",
        object_id=_OBJECT,
        created=_CREATED,
        filename="Q3 Report.PDF",
    )
    assert key.endswith("/q3-report.pdf")


def test_missing_filename_falls_back_to_object() -> None:
    """No filename yields a stable ``object`` leaf."""
    key = build_object_key(
        customer_id=_CUSTOMER,
        scope="s",
        category="media",
        object_id=_OBJECT,
        created=_CREATED,
        filename=None,
    )
    assert key.endswith(f"/{_OBJECT}/object")


def test_sanitize_segment_collapses_unsafe_chars() -> None:
    """Segments lower-case and collapse to the ``[a-z0-9-]`` alphabet."""
    assert sanitize_segment("Engagement #42: ACME!") == "engagement-42-acme"
    assert sanitize_segment("///") == "object"


@pytest.mark.parametrize(
    "evil",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "evil/../../x",
        "a/b/c",
        "..",
        "...",
        "  /  ",
    ],
)
def test_scope_and_category_cannot_escape_the_tenant_prefix(evil: str) -> None:
    """No scope/category input can inject a ``/`` or escape ``<customer_id>/``.

    The tenant prefix is the isolation boundary; the only slashes in the key
    must be the structural ones the builder inserts (8 segments exactly).
    """
    key = build_object_key(
        customer_id=_CUSTOMER,
        scope=evil,
        category=evil,
        object_id=_OBJECT,
        created=_CREATED,
        filename="x.pdf",
    )
    segments = key.split("/")
    assert key.startswith(f"{_CUSTOMER}/")
    assert len(segments) == 8
    assert ".." not in segments


def test_filename_cannot_inject_path_separators() -> None:
    """A traversal-laden filename collapses to one safe leaf segment."""
    key = build_object_key(
        customer_id=_CUSTOMER,
        scope="s",
        category="c",
        object_id=_OBJECT,
        created=_CREATED,
        filename="../../../etc/passwd",
    )
    segments = key.split("/")
    leaf = segments[-1]
    assert len(segments) == 8
    assert leaf and "/" not in leaf and ".." not in leaf


class TestSharedAndDeterministicKeys:
    """the two shapes added for objects that are not tenant-owned or not opaque.

    both exist because ``platform.datasources.customer_id`` is itself
    nullable -- NULL meaning platform-shared -- so artifacts derived from
    such a row have no tenant to scope to, and because a reader like a CDN
    must be able to derive a key from a request without a lookup.
    """

    def test_shared_objects_get_the_shared_prefix_not_a_none_segment(self) -> None:
        # "None/..." would be a key that looks tenant-scoped and is not.
        key = build_object_key(customer_id=None, scope="tiles", category="ds", path="layer/v1/8/40/98.mvt")
        assert key.startswith(f"{SHARED_PREFIX}/")
        assert "None" not in key

    def test_shared_prefix_is_still_a_grantable_boundary(self) -> None:
        # the leading segment IS what bucket policy grants on, so a shared
        # object still needs exactly one, just not a customer's.
        key = build_object_key(customer_id=None, scope="tiles", category="ds", path="a/b.mvt")
        assert key.split("/")[0] == SHARED_PREFIX

    def test_deterministic_path_is_derivable_from_a_request(self) -> None:
        key = build_object_key(customer_id=None, scope="tiles", category="ds", path="census_tracts/v3/8/40/98.mvt")
        assert key == f"{SHARED_PREFIX}/tiles/ds/census-tracts/v3/8/40/98.mvt"

    def test_deterministic_path_keeps_the_final_extension(self) -> None:
        # ``98.mvt`` collapsing to ``98-mvt`` would lose the type a reader
        # and a CDN's content handling depend on.
        key = build_object_key(customer_id=None, scope="tiles", category="ds", path="a/98.mvt")
        assert key.endswith("/98.mvt")

    def test_deterministic_path_still_sanitizes_each_component(self) -> None:
        key = build_object_key(customer_id=None, scope="tiles", category="ds", path="Census Tracts/V3/8.mvt")
        assert key == f"{SHARED_PREFIX}/tiles/ds/census-tracts/v3/8.mvt"

    def test_deterministic_path_cannot_traverse_out_of_its_prefix(self) -> None:
        key = build_object_key(customer_id=None, scope="tiles", category="ds", path="../../etc/passwd")
        assert ".." not in key
        assert key.startswith(f"{SHARED_PREFIX}/tiles/ds/")

    def test_a_tenant_may_also_use_a_deterministic_path(self) -> None:
        key = build_object_key(customer_id=_CUSTOMER, scope="tiles", category="ds", path="layer/v1/0/0/0.mvt")
        assert key.startswith(f"{_CUSTOMER}/")
        assert key.endswith("/0.mvt")

    def test_neither_path_nor_object_id_is_an_error(self) -> None:
        # silently emitting a key missing its date/id segments would collide
        # every object in a category onto one address.
        with pytest.raises(ValueError, match="either path="):
            build_object_key(customer_id=None, scope="s", category="c")

    def test_empty_path_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="at least one component"):
            build_object_key(customer_id=None, scope="s", category="c", path="///")
