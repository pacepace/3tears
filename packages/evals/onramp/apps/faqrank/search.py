"""Help-centre search: ranks the help articles for what a customer typed into the search box.

``search(query, k=5)`` returns up to ``k`` :class:`Hit` objects, best first. An article scores for each query term it
contains: a rarer term counts for more (inverse document frequency), a term in the title counts
:data:`TITLE_WEIGHT` times, and repeats in the body count with diminishing returns. Articles that match no term are not
returned. Terms are lower-cased words of letters and digits, with common words (:data:`STOPWORDS`) dropped and simple
plural and verb endings taken off, so ``refunds`` finds ``refund`` and ``billing`` finds ``bill``.

Articles are read from ``data/articles.jsonl``; :class:`Index` builds an index over any list of them.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

ARTICLES = Path(__file__).with_name("data") / "articles.jsonl"
TITLE_WEIGHT = 2.0
#: How fast repeats of a term in an article's body stop adding to its score.
SATURATION = 1.2

STOPWORDS = frozenset(
    "a an and are be by can do does for from get got have how i i'm if in is it its me my of on or so that the this "
    "to was what when where which who why will with you your".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """A word with a common ending taken off: ``refunds`` -> ``refund``, ``billing`` -> ``bill``."""
    for ending in ("ies", "ing", "ed", "s"):
        if word.endswith(ending) and not word.endswith("ss") and len(word) - len(ending) >= 3:
            word = word[: -len(ending)]
            return word + "y" if ending == "ies" else word
    return word


def terms(text: str) -> list[str]:
    """The search terms in a piece of text, in order."""
    return [stem(word) for word in _WORD.findall(text.lower()) if word not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    id: str
    title: str
    score: float


class Index:
    def __init__(self, articles: list[dict[str, Any]]) -> None:
        self.articles = {article["id"]: article for article in articles}
        self.title_terms = {aid: Counter(terms(a["title"])) for aid, a in self.articles.items()}
        self.body_terms = {aid: Counter(terms(a["body"])) for aid, a in self.articles.items()}
        seen: Counter[str] = Counter()
        for aid in self.articles:
            seen.update(set(self.title_terms[aid]) | set(self.body_terms[aid]))
        n = len(self.articles)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in seen.items()}

    def score(self, article_id: str, query_terms: list[str]) -> float:
        total = 0.0
        for term in set(query_terms):
            body = self.body_terms[article_id][term]
            weight = TITLE_WEIGHT * self.title_terms[article_id][term] + body / (body + SATURATION)
            total += self.idf.get(term, 0.0) * weight
        return total

    def search(self, query: str, k: int = 5) -> list[Hit]:
        query_terms = [stem(word) for word in query.lower().split() if word not in STOPWORDS]
        scored = [(self.score(aid, query_terms), aid) for aid in self.articles]
        ranked = sorted((pair for pair in scored if pair[0] > 0), key=lambda pair: (-pair[0], pair[1]))
        return [Hit(id=aid, title=self.articles[aid]["title"], score=round(s, 4)) for s, aid in ranked[:k]]


@cache
def _index() -> Index:
    lines = ARTICLES.read_text(encoding="utf-8").splitlines()
    return Index([json.loads(line) for line in lines if line.strip()])


def search(query: str, k: int = 5) -> list[Hit]:
    """The help articles for ``query``, best first."""
    return _index().search(query, k)
