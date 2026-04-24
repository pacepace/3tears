"""unit tests for :mod:`threetears.core.namespaces`.

covers the builder + sanitizer pair that ships the canonical
namespace-name shape under namespace-task-01 phase 9.5.
"""

from __future__ import annotations

import pytest

from threetears.core.namespaces import (
    PLURAL_PREFIX_BY_NAMESPACE_TYPE,
    build_namespace_name,
    sanitize_segment,
)


class TestSanitizeSegment:
    """tests for :func:`sanitize_segment`."""

    def test_replaces_single_dot(self) -> None:
        assert sanitize_segment("a.b") == "a-b"

    def test_replaces_multiple_dots(self) -> None:
        assert sanitize_segment("claude-sonnet-4.5") == "claude-sonnet-4-5"

    def test_passes_through_no_dot(self) -> None:
        assert sanitize_segment("anthropic") == "anthropic"

    def test_passes_through_empty_string(self) -> None:
        assert sanitize_segment("") == ""

    def test_leaves_hyphens_intact(self) -> None:
        assert sanitize_segment("foo-bar-baz") == "foo-bar-baz"

    def test_leaves_underscores_intact(self) -> None:
        assert sanitize_segment("foo_bar") == "foo_bar"


class TestBuildNamespaceName:
    """tests for :func:`build_namespace_name`."""

    def test_single_segment(self) -> None:
        assert build_namespace_name("datasources", "my_db") == "datasources.my_db"

    def test_two_segments(self) -> None:
        assert (
            build_namespace_name("memories", "aaaaaaaa", "bbbbbbbb")
            == "memories.aaaaaaaa.bbbbbbbb"
        )

    def test_three_segments(self) -> None:
        assert (
            build_namespace_name("channels", "slack", "12345678", "extra")
            == "channels.slack.12345678.extra"
        )

    def test_no_segments_yields_prefix_only(self) -> None:
        assert build_namespace_name("system") == "system"

    def test_sanitizes_segment_with_dot(self) -> None:
        assert (
            build_namespace_name("models", "anthropic", "claude-sonnet-4.5")
            == "models.anthropic.claude-sonnet-4-5"
        )

    def test_sanitizes_every_segment_independently(self) -> None:
        assert (
            build_namespace_name("models", "v1.0", "model.2.3")
            == "models.v1-0.model-2-3"
        )

    def test_prefix_with_dots_is_not_sanitized(self) -> None:
        # the prefix argument is supplied by the caller from the
        # PLURAL_PREFIX_* constants; it is not sanitized. guard the
        # contract so callers cannot accidentally introduce a dotted
        # prefix.
        assert build_namespace_name("sys.tem", "a") == "sys.tem.a"


class TestPluralPrefixMapping:
    """tests pinning the ``namespace_type`` -> plural prefix contract."""

    @pytest.mark.parametrize(
        ("namespace_type", "expected"),
        [
            ("agent", "agents"),
            ("audit", "audits"),
            ("channel", "channels"),
            ("conversation", "conversations"),
            ("datasource", "datasources"),
            ("memory", "memories"),
            ("model", "models"),
            ("shared", "shared"),
            ("shared_agent", "shared_agents"),
            ("system", "system"),
            ("tool", "tools"),
            ("workspace", "workspaces"),
        ],
    )
    def test_namespace_type_maps_to_plural_prefix(
        self, namespace_type: str, expected: str
    ) -> None:
        assert PLURAL_PREFIX_BY_NAMESPACE_TYPE[namespace_type] == expected

    def test_mapping_closed_set_size(self) -> None:
        # pins the 12-value closed set; adding a new namespace_type
        # requires updating v018's CHECK constraint + this mapping.
        # namespace-task-01 phase 11 adds ``audit``.
        assert len(PLURAL_PREFIX_BY_NAMESPACE_TYPE) == 12
