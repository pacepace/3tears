"""Incident digests: the few sentences of an incident report that go into the weekly operations email.

``digest(report, max_sentences=3)`` picks the report's most telling sentences and returns them in the order the
report has them, joined by single spaces. Rules, not a model, decide what is telling:

- each cue (:data:`CUES`) a sentence contains adds 3: what broke, who it hit, why, and what fixed it;
- a sentence with a figure in it (a time, a count, a duration) adds 1;
- the report's first sentence adds 2, as it usually says what happened;
- a sentence of fewer than :data:`MIN_WORDS` words loses 4: headings and asides are rarely what a reader needs.

Ties go to the earlier sentence. A report is split into sentences at the end of each sentence (``.``, ``!`` or
``?`` followed by a space and the capital letter that starts the next one); abbreviations such as ``approx.``,
``e.g.``, ``i.e.`` and ``a.m.`` sit inside sentences and do not end them.
"""

from __future__ import annotations

import re
from pathlib import Path

CUES = (
    "customers", "users", "impact", "root cause", "caused by", "because", "fixed", "rolled back", "restored",
    "resolved", "outage", "failed", "errors", "timeouts",
)  # fmt: skip
MIN_WORDS = 6

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_FIGURE = re.compile(r"\d")


def sentences(report: str) -> list[str]:
    """The report's sentences, in order."""
    text = " ".join(report.split())
    return [part for part in _SENTENCE_END.split(text) if part]


def score(sentence: str, position: int) -> int:
    """How telling one sentence is, by the rules in the module docstring."""
    lowered = sentence.lower()
    points = 3 * sum(cue in lowered for cue in CUES)
    points += 1 if _FIGURE.search(sentence) else 0
    points += 2 if position == 0 else 0
    points -= 4 if len(sentence.split()) < MIN_WORDS else 0
    return points


def digest(report: str, max_sentences: int = 3) -> str:
    """The digest of one report."""
    parts = sentences(report)
    ranked = sorted(range(len(parts)), key=lambda i: (-score(parts[i], i), i))
    keep = sorted(ranked[:max_sentences])
    return " ".join(parts[i] for i in keep)


def digest_file(path: str | Path, max_sentences: int = 3) -> str:
    """The digest of a report kept as a text file."""
    return digest(Path(path).read_text(encoding="utf-8"), max_sentences)
