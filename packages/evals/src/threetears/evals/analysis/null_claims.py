"""The ways a headline or summary says two arms are alike — the one place code reads the writer's words for meaning.

**Why it exists.** A production memo (2026-10) was headlined "Halving the token budget from 16384 to 12288 does not
change this agent's latency, cost, or delivery" over a contrast its own findings left unresolved (adjusted p 0.13 at
five cases, its first caveat saying it "has not confirmed a null so much as failed to separate two arms"). Rule 4 of
the writer's prompt forbade that in a finding; the headline is what most readers read, and nothing checked it. A null
is established only by the bundle: a comparison ``multiple_comparisons`` reads ``equivalent``, or a guardrail
``held``. A contrast that did not separate is unresolved, and "does not change" asserts a result the campaign did not
measure.

**A phrase list over a structural join.** The package's rule is that code never grades prose, because a gate over
free text never converges: every generation finds a phrasing it does not parse (``docs/measuring-soundly.md``). This
does not parse a claim. The headline and each summary bullet name the findings they rest on
(``headline_rests_on``, ``summary_rests_on``), and what the generator checks
(:func:`~threetears.evals.analysis.generator._unestablished_null_titles`) is structural: does any finding that
sentence rests on cite a reading the bundle established alike? Only the question "does this sentence say arms are
alike" is read off the words, and only by a short, curated list of the commonest null phrasings
(:data:`NULL_CLAIM_PATTERNS`). A reworded null passes it, and the reporter eval's judge measures the rest; what it
refuses is a sentence in one of those phrasings resting only on findings that established no null.

**Chosen to miss rather than to refuse a sound sentence.** A false refusal buys a paid repair, and two lose the whole
analysis; a miss is today's behaviour. So each phrase is an assertion form, and the near-misses a recommendation
uses are left out: "do not change the budget" and "keep it unchanged" advise, where "does not change" and "was
unchanged" report. The tests pin both directions for every phrase.
"""

from __future__ import annotations

import re

#: An optional adverb before the claim's verb or adjective: "measurably", "meaningfully", "really".
_ADVERB = r"(?:\w+ly\s+)?"

#: The null claims refused, each named for the refusal and the tests. Matched case-insensitively on word
#: boundaries, over text with markdown emphasis removed and curly apostrophes made straight.
#:
#: Left out on purpose: ``do``/``don't``, ``should`` and ``would`` before a verb (an imperative or a recommendation —
#: "do not change the budget yet", "I would not change it"), a bare ``unchanged`` ("leave it unchanged"), ``no
#: change`` ("make no change until..."), and ``the same`` without a qualifier ("on the same cases").
NULL_CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        (
            "a negated effect verb",
            rf"\b(?:(?:does|did|will)\s+not|doesn't|didn't|won't)\s+{_ADVERB}"
            r"(?:change|affect|alter|move|matter|hurt|harm|impact)\b",
        ),
        (
            "a negated difference",
            r"\b(?:(?:does|did|will)\s+not|doesn't|didn't|won't)\s+make\s+(?:a|any|much)\s+(?:\w+\s+)?difference\b",
        ),
        (
            "an unaffected reading",
            rf"\b(?:is|are|was|were)(?:\s+not|n't)\s+{_ADVERB}(?:affected|changed|altered|hurt|harmed|impacted)\b",
        ),
        (
            "no effect, difference or impact",
            r"\bno\s+(?:\w+\s+)?(?:effects|effect|differences|difference|impacts|impact)\b",
        ),
        ("no worse or better", r"\bno\s+(?:worse|slower|better|faster)\b"),
        (
            "a reading that held still",
            rf"\b(?:is|are|was|were|stays|stay|stayed|remains|remain|remained)\s+{_ADVERB}(?:unchanged|unaffected|unaltered)\b",
        ),
        (
            "a flat reading",
            rf"\b(?:is|are|was|were|stays|stay|stayed|remains|remain|remained|holds|hold|held)\s+{_ADVERB}(?:steady|flat)\b",
        ),
        (
            "about the same",
            r"\b(?:about|roughly|essentially|basically|virtually|nearly|practically|much|effectively)\s+the\s+same\b",
        ),
        ("just as good", r"\b(?:just|equally)\s+as\s+(?:fast|quick|good|accurate|reliable|cheap|well|often)\b"),
        ("nothing moved", r"\bnothing\s+(?:changed|changes|moved|moves)\b"),
        ("no trend", r"\bno\s+(?:\w+\s+)?trend\b"),
        ("a wash", r"\b(?:is|are|was|were)\s+a\s+wash\b"),
    )
)

#: Markdown emphasis and code marks, removed before matching so "does **not** change" reads as written.
_EMPHASIS = re.compile(r"[*_`]")


def null_claims(text: str) -> list[str]:
    """The null claims ``text`` makes, as matched, in the order they appear; empty when it makes none.

    :param text: A headline, or one summary bullet.
    :ptype text: str
    :return: Each matched phrase once, lower-cased with its spacing collapsed, as a refusal quotes it.
    :rtype: list[str]
    """
    plain = " ".join(_EMPHASIS.sub("", text.replace("’", "'")).split())
    found = sorted(
        (match.start(), " ".join(match.group(0).lower().split()))
        for _name, pattern in NULL_CLAIM_PATTERNS
        for match in pattern.finditer(plain)
    )
    claims: list[str] = []
    for _start, phrase in found:
        if phrase not in claims:
            claims.append(phrase)
    return claims


__all__ = ["NULL_CLAIM_PATTERNS", "null_claims"]
