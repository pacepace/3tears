"""Tests for the scope-first object-key builder."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from threetears.object_store.keys import build_object_key, sanitize_segment

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
