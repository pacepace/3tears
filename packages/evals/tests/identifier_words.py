"""Split an identifier into its constituent words, for whole-word name gates.

Shared because **two** name gates match on it and they must agree: the
engine's shared-contract scan (``test_no_host_names_in_shared_contract.py``,
host nouns must not reach the engine's vocabulary) and the host-vocabulary
ceiling (``test_host_vocabulary_ceiling.py``, the register of what is left).

**Whole-word, never substring**: a short noun matched as a substring hits every
longer word that happens to contain it, and a canary that cries wolf gets
switched off. ``eval`` as a substring matches ``interval``, so a constant such
as ``EXPORT_INTERVAL_MS`` would trip a gate that never meant it. A second copy
of the predicate would let the gates drift apart silently, which is why this
lives here rather than being duplicated.
"""

from __future__ import annotations

import re

#: Splits an identifier or token into its constituent words: snake_case and kebab-case
#: on the separators, camelCase and PascalCase on the case boundary, ``HTTPServer`` into
#: ``HTTP`` + ``Server``, and ``v2Model`` on the digit boundary. An acronym keeps a bare plural
#: ``s`` (``IDs``, ``URLs``): without it the acronym split as ``I`` + ``Ds``, and a plural host
#: noun written in capitals escaped every gate matching on this.
_WORD_RE = re.compile(r"[A-Z]+s?(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def identifier_words(text: str) -> list[str]:
    """Split an identifier or token into lowercase words.

    Args:
        text: An identifier, a dotted wire name, or any token-shaped string.

    Returns:
        Its constituent words, lowercased, in order.
    """
    out: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        out.extend(word.lower() for word in _WORD_RE.findall(chunk))
    return out
