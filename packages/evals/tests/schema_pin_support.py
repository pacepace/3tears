"""The checks the core and durable schema pins share: a minor ledger's additivity sentence, and a read that keeps values.

Both version spaces in ``schema/versioning.py`` keep a ledger of minors and promise that an older minor's document
reads back with every value it stored, so ``test_core_schema_pins.py`` and ``test_durable_schema_pins.py`` hold them
to it with the same two helpers, kept here under public names rather than imported privately from one of them.
"""

from __future__ import annotations

import re
from typing import Any

from threetears.evals.schema.versioning import MINOR_ADDITIVITY_SENTENCE

__all__ = ["keeps", "minor_entries", "minors_missing_the_sentence"]

#: A minor's ledger entry, as the comment block above a ``*_SCHEMA_MINOR`` writes one: ``#: - **v14.1** — …``,
#: through the end of its (possibly wrapped) comment lines.
_MINOR_ENTRY = re.compile(r"^#: - \*\*v(\d+)\.(\d+)\*\*(.*(?:\n#:   .*)*)", re.MULTILINE)


def minor_entries(ledger: str) -> dict[tuple[int, int], str]:
    """Each minor's ledger entry in ``ledger``, keyed by its major and minor, its wrapped lines joined."""
    return {
        (int(major), int(minor)): " ".join(line.removeprefix("#:").strip() for line in text.splitlines())
        for major, minor, text in _MINOR_ENTRY.findall(ledger)
    }


def minors_missing_the_sentence(ledger: str, major: int, newest: int) -> list[int]:
    """The minors after a major's first, up to ``newest``, whose ledger entry is absent or lacks the sentence."""
    entries = minor_entries(ledger)
    return [minor for minor in range(1, newest + 1) if MINOR_ADDITIVITY_SENTENCE not in entries.get((major, minor), "")]


def keeps(stored: Any, written: Any) -> bool:
    """Whether ``written`` holds every value ``stored`` holds, at every depth; it may hold more keys beside them."""
    if isinstance(stored, dict):
        return isinstance(written, dict) and all(
            key in written and keeps(value, written[key]) for key, value in stored.items()
        )
    if isinstance(stored, list):
        return isinstance(written, list) and len(stored) == len(written) and all(map(keeps, stored, written))
    return bool(stored == written)
