"""threetears.core.utils — general-purpose helpers shared across packages.

contains stdlib-only utilities that every package can rely on without
pulling additional dependencies into core.
"""

from threetears.core.utils.atomic_write import atomic_write

__all__ = ["atomic_write"]
