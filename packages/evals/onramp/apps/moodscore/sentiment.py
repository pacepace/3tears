"""Review sentiment for the feedback dashboard: how warm or cold a customer's review reads, from -1 to 1.

:func:`score` reads a review word by word against the lexicon in ``data/lexicon.tsv``, where each sentiment word has
a weight from -3 (``awful``) to 3 (``amazing``), and adds the weights up:

- **Negation.** A negator (``not``, ``no``, ``never``, ``nothing``, ``without``, ``cannot`` or any word ending in
  ``n't``) turns round the first sentiment word within the next three words, and shrinks it: the word counts
  :data:`NEGATED` times its weight, the other way. It turns round that one word only: ``no complaints, great food``
  and ``no complaints great food`` are both a turned-round ``complaints`` and a plain ``great``.
- **Intensifiers.** ``very``, ``really``, ``so``, ``super``, ``extremely`` and ``absolutely`` just before a sentiment
  word make it count half as much again.
- **Clause breaks.** A full stop, comma, ``!``, ``?``, ``;`` or the word ``but`` ends any negation or intensifier
  still waiting.

The sum ``x`` is then brought into -1 to 1 as ``x / sqrt(x * x + 15)``, so a review's score grows with each kind word
and never passes 1. A review with no sentiment words scores 0. :func:`mood` names the score's side: ``positive`` above
0.05, ``negative`` below -0.05, ``neutral`` between.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

LEXICON_FILE = Path(__file__).with_name("data") / "lexicon.tsv"

#: How many words after a negator its turn-round reaches.
NEGATION_WINDOW = 3
#: What a negated sentiment word counts, as a share of its weight, the other way.
NEGATED = 0.75
#: How much an intensifier multiplies the next sentiment word.
INTENSIFY = 1.5
NEGATORS = frozenset({"not", "no", "never", "nothing", "without", "cannot"})
INTENSIFIERS = frozenset({"very", "really", "so", "super", "extremely", "absolutely"})
CLAUSE_BREAKS = frozenset({".", ",", "!", "?", ";", "but"})

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;]")


def _lexicon(path: Path = LEXICON_FILE) -> dict[str, float]:
    weights = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            word, weight = line.split("\t")
            weights[word.strip()] = float(weight)
    return weights


LEXICON = _lexicon()


def tokens(text: str) -> list[str]:
    """The review's words and clause-breaking punctuation, in lower case; curly apostrophes read as straight ones."""
    return _TOKEN.findall(text.lower().replace("’", "'"))


def _negator(token: str) -> bool:
    return token in NEGATORS or token.endswith("n't")


def raw_score(text: str) -> float:
    """The sum of the review's sentiment weights, after negation and intensifiers."""
    total = 0.0
    negation = 0  # words left in which a sentiment word is turned round
    boost = 1.0
    for token in tokens(text):
        if token in CLAUSE_BREAKS:
            negation, boost = 0, 1.0
            continue
        if _negator(token):
            negation = NEGATION_WINDOW
            continue
        if token in INTENSIFIERS:
            boost = INTENSIFY
            continue
        weight = LEXICON.get(token)
        if weight is None:
            negation = max(negation - 1, 0)
            boost = 1.0
            continue
        if negation:
            weight = -NEGATED * weight
            negation -= 1
        total += weight * boost
        boost = 1.0
    return total


def score(text: str) -> float:
    """The review's sentiment, from -1 (as cold as it gets) to 1 (as warm), rounded to three places."""
    x = raw_score(text)
    return round(x / math.sqrt(x * x + 15), 3)


def mood(text: str) -> str:
    """``positive``, ``negative`` or ``neutral``, by :func:`score`."""
    value = score(text)
    return "positive" if value > 0.05 else "negative" if value < -0.05 else "neutral"
