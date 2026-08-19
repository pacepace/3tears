"""unit tests for the shared HTTP cache-exposure vocabulary.

covers the promotion of ``CacheClass`` out of
``threetears.datasources.geo_config`` (where it was spelled
``CacheClassConfig`` and served geo layers only) into
``threetears.core.http_cache``, plus the narrow-only resolution rule the
geo pipeline already encodes hub-side. a second consumer -- the tool REST
affordance -- is what makes one shared vocabulary the right shape.
"""

from __future__ import annotations

import pytest

from threetears.core.http_cache import CacheClass, narrow_cache_class


class TestCacheClassVocabulary:
    """closed vocabulary, spelled once."""

    def test_members_are_the_four_documented_values(self) -> None:
        """``INHERIT`` plus the three resolved exposure classes."""
        assert {member.value for member in CacheClass} == {
            "inherit",
            "public",
            "authenticated",
            "private",
        }

    def test_is_a_str_enum_so_it_serializes_as_its_value(self) -> None:
        """a declaration survives a JSON round trip as its bare string."""
        assert CacheClass.PRIVATE == "private"
        assert str(CacheClass.AUTHENTICATED) == "authenticated"


class TestNarrowCacheClass:
    """derive-and-narrow: a declaration may narrow, never widen."""

    def test_inherit_takes_the_resolved_class(self) -> None:
        """``INHERIT`` is the default and means 'whatever the resource is'."""
        assert narrow_cache_class(CacheClass.PUBLIC, CacheClass.INHERIT) is CacheClass.PUBLIC
        assert narrow_cache_class(CacheClass.PRIVATE, CacheClass.INHERIT) is CacheClass.PRIVATE

    def test_narrowing_is_honoured(self) -> None:
        """a declaration less exposed than the resource wins."""
        assert narrow_cache_class(CacheClass.PUBLIC, CacheClass.PRIVATE) is CacheClass.PRIVATE
        assert narrow_cache_class(CacheClass.PUBLIC, CacheClass.AUTHENTICATED) is CacheClass.AUTHENTICATED
        assert narrow_cache_class(CacheClass.AUTHENTICATED, CacheClass.PRIVATE) is CacheClass.PRIVATE

    def test_widening_is_clamped_to_the_resource(self) -> None:
        """a declaration MORE exposed than the resource does not win.

        this is the whole point of the type: a class attribute saying
        "cacheable" must not be able to publish a per-caller-authorized
        response to a shared edge.
        """
        assert narrow_cache_class(CacheClass.PRIVATE, CacheClass.PUBLIC) is CacheClass.PRIVATE
        assert narrow_cache_class(CacheClass.PRIVATE, CacheClass.AUTHENTICATED) is CacheClass.PRIVATE
        assert narrow_cache_class(CacheClass.AUTHENTICATED, CacheClass.PUBLIC) is CacheClass.AUTHENTICATED

    def test_equal_classes_pass_through(self) -> None:
        """declaring exactly what the resource already is changes nothing."""
        for member in (CacheClass.PRIVATE, CacheClass.AUTHENTICATED, CacheClass.PUBLIC):
            assert narrow_cache_class(member, member) is member

    def test_unresolved_inherited_class_is_a_programming_error(self) -> None:
        """``inherited`` is a RESOLVED class; ``INHERIT`` there means nothing."""
        with pytest.raises(ValueError, match="inherited"):
            narrow_cache_class(CacheClass.INHERIT, CacheClass.PRIVATE)
