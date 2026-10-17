"""Text-block enforcement domain: free text a model reads sits under its heading, every line of it.

The enforcement half of :mod:`threetears.observe.text_block`. :func:`unbroken_text` checks one
module's source; :func:`unbroken_text_in` checks every module under a set of roots.
"""

from threetears.enforcement.text_blocks.walkers import (
    FREE_TEXT,
    MARKER,
    unbroken_text,
    unbroken_text_in,
)

__all__ = ["FREE_TEXT", "MARKER", "unbroken_text", "unbroken_text_in"]
