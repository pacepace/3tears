"""catalogue name comparison enforcement: a catalogue ``name`` column is compared with an array only as ``text``.

YugabyteDB 2026.1.1.1 mis-evaluates a ``name`` column compared with a ``text[]`` (``relname = ANY(ARRAY['a'])``
fails ``compressed lz4 data is corrupt``, intermittently, once the server has taken load), while
``relname::text = ANY(...)`` does not. :func:`find_catalog_name_violations` finds every such comparison
in a source tree; a repo's gate asserts it finds none.
"""

from threetears.enforcement.catalog_name_comparison.walkers import (
    find_catalog_name_violations,
    violations_in,
)

__all__ = ["find_catalog_name_violations", "violations_in"]
