"""Distributed tracing for 3tears -- re-exports from threetears.observe.tracing.

All tracing functionality lives in ``threetears.observe.tracing``.  This
module re-exports the public API so that existing ``from threetears.core.tracing
import traced`` imports continue to work without changes.
"""

from threetears.observe.tracing import traced

__all__ = ["traced"]
