"""How a report speaks: one place for the names, the glossary and the rules every rendered sentence is held to.

The renderer (:mod:`threetears.evals.analysis.report`) and the memo path both read this module, so a page and the
memo on it cannot drift into two styles.

- :func:`humanize` is the one seam an identifier passes through on its way to a reader: a model id, an arm, a judged
  dimension or check, a case, a provider. :data:`HUMANIZE_OVERRIDES` names exceptions by hand.
- :class:`ArmName` names an arm twice: in full on its first mention (``deepseek-v4.1-flash (production, control)``)
  and by its given label after that (``production``).
- :data:`GLOSSARY` defines every term of art a page uses, once, in one plain sentence.
- :func:`lint_prose` holds a sentence to the rules; :func:`lint_page` holds a whole ``report.md`` to them;
  :func:`fix_prose` drops what can be dropped deterministically and returns what is left to disclose.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

#: What :func:`humanize` names.
HumanizeKind = Literal["model", "arm", "dimension", "case", "provider"]

#: Names fixed by hand, by kind and identifier, ahead of every rule below.
HUMANIZE_OVERRIDES: dict[tuple[str, str], str] = {
    ("dimension", "__outcome__"): "outcome",
    ("dimension", "__transcript__"): "transcript",
}

#: The longest a case label runs before a page names the case by its number instead.
CASE_LABEL_CHARS = 40

_LABELLED = re.compile(r"^(?P<model>\S+) \((?P<label>[^()]+)\)$")
_PAIR = re.compile(r"^(?P<axis>[\w.-]+)=(?P<level>.+)$")


def humanize(kind: HumanizeKind, identifier: str) -> str:
    """An identifier as a reader is shown it after its first mention.

    Args:
        kind: What the identifier names.
        identifier: The identifier as stored.

    Returns:
        A model id without its vendor prefix; an arm by its given label (else its model id, vendor dropped), its
        ``axis=level`` pairs read as words; a dimension or check with its underscores as spaces; a case label on one
        line (empty when it is a dump, so the caller names the case by number); a provider as given.
    """
    if (kind, identifier) in HUMANIZE_OVERRIDES:
        return HUMANIZE_OVERRIDES[(kind, identifier)]
    text = " ".join(identifier.split())
    if kind == "model":
        return text.rsplit("/", 1)[-1]
    if kind == "arm":
        return arm_name(text).label
    if kind == "dimension":
        return re.sub(r"[_.]+", " ", text).strip() or text
    if kind == "case":
        return "" if is_dump(text) else text
    return text


def is_dump(label: str) -> bool:
    """Whether a case label is the start of a world dump or a payload rather than a name."""
    return label.startswith("The world") or "{" in label or len(label) > CASE_LABEL_CHARS


@dataclass(frozen=True)
class ArmName:
    """One arm, named in full once and by its label after that."""

    label: str
    model: str | None = None
    control: bool = False
    given: bool = True

    def first(self) -> str:
        """The first mention: ``deepseek-v4.1-flash (production, control)``, ``v1 (control)``, or the label."""
        marks = [self.label] if self.model is not None and self.given else []
        marks += ["control"] if self.control else []
        head = self.model if self.model is not None else self.label
        return f"{head} ({', '.join(marks)})" if marks else head


def arm_name(display: str, *, control: bool = False) -> ArmName:
    """Read an arm's display (``model=deepseek-v4.1-flash (production)``, ``candidate=v1``) as an :class:`ArmName`.

    Args:
        display: The arm's name as :func:`~threetears.evals.analysis.arms.arm_names` spells it.
        control: Whether it is the control.

    Returns:
        The arm's given label and model, never as ``key=value``.
    """
    pairs = [_PAIR.match(part.strip()) for part in display.split(", ")]
    if len(pairs) > 1 and all(pairs):
        # The candidate's kind is no name: an arm named by its kind and one other lever is named by that lever.
        named = [match for match in pairs if match and match["axis"] != "candidate_kind"]
        pairs = named if len(named) == 1 else pairs
    if len(pairs) > 1 and all(pairs):
        label = ", ".join(f"{humanize('dimension', match['axis'])} {match['level']}" for match in pairs if match)
        return ArmName(label=label, control=control)
    level = pairs[0]["level"] if len(pairs) == 1 and pairs[0] else display
    if labelled := _LABELLED.match(level):
        return ArmName(label=labelled["label"], model=labelled["model"], control=control)
    if "/" in level and " " not in level:
        return ArmName(label=humanize("model", level), model=level, control=control, given=False)
    return ArmName(label=level, control=control)


#: Every term of art a page may use, in one plain sentence each.
GLOSSARY: dict[str, str] = {
    "control": "The arm every other arm is compared against, usually what runs in production today.",
    "trial": "One case run once on one arm.",
    "case": "One input the eval feeds every arm; repeats of a case are not independent.",
    "usable": "A trial the test setup did not spoil; only usable trials count for or against an arm.",
    "lost": "A trial the test setup spoiled (a simulator, tool or judge failed), so it counts for no arm.",
    "failed by check": "A trial that broke a rule code checks, such as a required tool call.",
    "failed by judge": "A trial the judge model scored below the bar on some dimension.",
    "cost per reply": "What production would spend per reply, from the models the arm runs.",
    "case floor": "Five usable cases: below it a rate or a difference is too unstable to estimate.",
    "separated": "The difference from the control is larger than chance explains, after correcting for every test.",
    "interval": "The range the true value likely lies in; a wide one means the run measured it loosely.",
    "margin": "The smallest change worth acting on, declared before the run.",
    "guardrail": "A reading an arm must not get worse on, whatever it gains elsewhere.",
    "bar": "A threshold a reading must clear on its own, without a control.",
    "exploratory": "No question was declared before the run, so the page offers leads, not decisions.",
    "settled": "The rule's outcome cannot change, however the remaining cases turn out.",
    "judge": "The model that scores each trial's transcript against a rubric.",
    "calibrated": "A judge whose scores have been shown to agree with people's.",
    "marks": "✓ passed, ✗ failed, – lost to the test setup; one mark per repeat.",
}

#: Constructions a page never uses: meta-prose about the page itself, and hedges on a measured fact.
BANNED_PHRASES: tuple[str, ...] = (
    "a count, not a ranking",
    "numbered f1 onward",
    "the control first",
    "what each arm ran",
    "written by",
    "every number is code's",
    "every number on the page is code's",
    "rests on",
    "stands on",
    "as the author listed",
    "no measurable",
)

#: The most words a sentence may carry, and the most numerals before it should be a table.
MAX_WORDS = 28
MAX_NUMERALS = 3
MAX_HEADER_WORDS = 3

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z▸\"'(*$])|\s·\s")
_WORD = re.compile(r"(?<![\w(\"'=./-])[A-Za-z][a-z']*(?:-[a-z]+)*(?![\w(\"'=/])")
_NUMERAL = re.compile(r"(?<![\w.$/-])\$?\d[\d,]*(?:\.\d+)?%?(?![\w-])")
_KEY_VALUE = re.compile(r"\b[a-z_][a-z0-9_]*=[^\s,;)}=]+")
_FACT_LINE = re.compile(r"^[A-Z][a-z ]+: .* · [A-Z][a-z ]+: ")
_CODE_SPAN = re.compile(r"`[^`]*`")
_CASE_NUMBER = re.compile(r"\bcase \d+\b")


@dataclass(frozen=True)
class StyleViolation:
    """One rule a piece of text breaks."""

    rule: Literal["banned", "long_sentence", "numerals", "question", "header", "dash_chain", "key_value", "facts"]
    text: str
    detail: str


def sentences(text: str) -> list[str]:
    """``text`` split into sentences (and `` · `` segments), each on one line."""
    flat = " ".join(text.split())
    return [part.strip() for part in _SENTENCE_END.split(flat) if part.strip()]


def lint_prose(text: str, *, question: str | None = None, header: bool = False) -> list[StyleViolation]:
    """Every rule ``text`` breaks.

    Args:
        text: A sentence, a paragraph or a header.
        question: The declared question; a sentence of it repeated verbatim is a violation.
        header: Whether ``text`` is a header, held to :data:`MAX_HEADER_WORDS` words.

    Returns:
        The violations, in the order found; empty when the text passes.
    """
    found: list[StyleViolation] = []
    lowered = text.casefold()
    # A code span is code a reader copies, not prose: it is held to the banned phrases only.
    text = _CODE_SPAN.sub("code", text)
    found += [StyleViolation("banned", text, phrase) for phrase in BANNED_PHRASES if phrase in lowered]
    for line in text.splitlines():
        if line.count(" — ") >= 2:
            found.append(StyleViolation("dash_chain", line, f"{line.count(' — ') + 1} segments joined by dashes"))
        if _FACT_LINE.search(line):
            found.append(StyleViolation("facts", line, "a `Key: value · Key: value` line belongs in a table"))
    found += [StyleViolation("key_value", text, match) for match in _KEY_VALUE.findall(text)]
    if header and (words := len(_WORD.findall(text))) > MAX_HEADER_WORDS:
        found.append(StyleViolation("header", text, f"{words} words"))
    for sentence in sentences(text):
        if (words := len(_WORD.findall(sentence))) > MAX_WORDS:
            found.append(StyleViolation("long_sentence", sentence, f"{words} words"))
        # A case named by its number is a name, not a figure, as a trial's ``T4`` is.
        if (numerals := len(_NUMERAL.findall(_CASE_NUMBER.sub("case", sentence)))) > MAX_NUMERALS:
            found.append(StyleViolation("numerals", sentence, f"{numerals} numerals: make it a table"))
    if question:
        found += [
            StyleViolation("question", text, asked)
            for asked in sentences(question)
            if len(asked.split()) >= 8 and asked in " ".join(text.split())
        ]
    return found


def fix_prose(text: str, *, question: str | None = None) -> tuple[str, list[StyleViolation]]:
    """Drop every sentence breaking a rule that dropping fixes, and return what is left with what it still breaks.

    A sentence carrying a banned phrase or repeating the declared question is meta-prose or a hedge: it goes.

    Args:
        text: Prose a model wrote.
        question: The declared question.

    Returns:
        ``(fixed, remaining)``: the kept sentences, and the violations they still carry.
    """
    kept = [
        sentence
        for sentence in sentences(text)
        if not any(v.rule in ("banned", "question") for v in lint_prose(sentence, question=question))
    ]
    fixed = " ".join(kept)
    return fixed, lint_prose(fixed, question=question)


def lint_page(markdown: str, *, question: str | None = None, names: Collection[str] = ()) -> list[StyleViolation]:
    """Every rule a rendered ``report.md`` breaks, outside the data it quotes.

    Headings and table headers are held to the header rule; prose lines to every rule. Table body cells, fenced
    blocks, folded trial details and the methods' disclosures fold are data and quoted engine text, not prose.

    Args:
        markdown: The page.
        question: The declared question, which only the title may carry.
        names: The arms' names, which head a column as a name, not a header's words.

    Returns:
        The violations, in page order.
    """
    found: list[StyleViolation] = []
    fenced = False
    folded = 0
    header_row = False
    for number, line in enumerate(markdown.splitlines()):
        if line.startswith("```"):
            fenced = not fenced
            continue
        folded += line.count("<details") - line.count("</details>")
        if fenced or folded > 0 or line.startswith("</details>") or not line.strip():
            header_row = False
            continue
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split(" | ")]
            if not header_row and not line.startswith("|---"):
                headers = [cell for cell in cells if cell.removeprefix("▸ ") not in names]
                found += [v for cell in headers for v in lint_prose(cell, header=True) if v.rule == "header"]
            header_row = True
            continue
        header_row = False
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            found += lint_prose(heading, header=number > 0)
            continue
        found += lint_prose(line.lstrip("-> *").strip(), question=question)
    return found


__all__ = [
    "BANNED_PHRASES",
    "CASE_LABEL_CHARS",
    "GLOSSARY",
    "HUMANIZE_OVERRIDES",
    "MAX_HEADER_WORDS",
    "MAX_NUMERALS",
    "MAX_WORDS",
    "ArmName",
    "HumanizeKind",
    "StyleViolation",
    "arm_name",
    "fix_prose",
    "humanize",
    "is_dump",
    "lint_page",
    "lint_prose",
    "sentences",
]
