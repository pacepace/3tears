"""How a report speaks: one place for the names, the glossary and the rules every rendered sentence is held to.

The renderer (:mod:`threetears.evals.analysis.report`) and the memo path both read this module, so a page and the
memo on it cannot drift into two styles.

- :func:`humanize` is the one seam an identifier passes through on its way to a reader: a model id, an arm, a judged
  dimension, a case, a provider. :data:`HUMANIZE_OVERRIDES` names exceptions by hand. A check expression is code and
  never passes through it: :func:`measure_name` names a check's measure by its expression, verbatim.
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

from threetears.evals.kernel.metrics import goal_check_of

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
#: What marks a name as code (a check expression) rather than an identifier: a call, a comparison or a quote.
_EXPRESSION = re.compile(r"[()<>=\"']")
_PAIR = re.compile(r"^(?P<axis>[\w.-]+)=(?P<level>.+)$")
#: What marks a case label as no name: whitespace (a sentence, or a stratum before a label), the run of hex a
#: generated id carries, or a bare number, which a sentence would read as a count.
_NOT_A_NAME = re.compile(r"\s|[0-9a-f]{12,}|^\d+$")


def humanize(kind: HumanizeKind, identifier: str) -> str:
    """An identifier as a reader is shown it after its first mention.

    Args:
        kind: What the identifier names.
        identifier: The identifier as stored.

    Returns:
        A model id without its vendor prefix; an arm by its given label (else its model id, vendor dropped), its
        ``axis=level`` pairs read as words; a dimension with its underscores as spaces, but a check expression (a
        call, a comparison, a quote) as written, since it is code; a case label where it reads as a name someone
        gave (``double-charge``: one token, not a generated id, not a bare number — empty for a dump, a sentence or
        an id, so the caller names the case by number); a provider as given.
    """
    if (kind, identifier) in HUMANIZE_OVERRIDES:
        return HUMANIZE_OVERRIDES[(kind, identifier)]
    text = " ".join(identifier.split())
    if kind == "model":
        return text.rsplit("/", 1)[-1]
    if kind == "arm":
        return arm_name(text).label
    if kind == "dimension":
        if _EXPRESSION.search(text):
            return text
        return re.sub(r"[_.]+", " ", text).strip() or text
    if kind == "case":
        return "" if is_dump(text) or _NOT_A_NAME.search(text) else text
    return text


def code(text: str) -> str:
    """``text`` as a code span: what a reader copies, never reworded."""
    return f"`{text}`" if "`" not in text else text


def measure_name(name: str, reader_name: str | None = None) -> str:
    """A measure as a page names it.

    A goal check's measure is named by its expression, verbatim, as a code span: never humanized and never "Pass rate
    of", since a single trial's failure has no rate. Any other measure by its reader name, else its key humanized.

    Args:
        name: The measure's key.
        reader_name: Its reader-facing name, when the catalog carries one.

    Returns:
        The name.
    """
    if (check := goal_check_of(name)) is not None:
        return code(check)
    return reader_name or humanize("dimension", name)


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
        named: list[re.Match[str] | None] = [match for match in pairs if match and match["axis"] != "candidate_kind"]
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
    "control": (
        "The arm every other arm is compared against, usually what runs in production today; ▸ marks it in a table."
    ),
    "trial": "One case run once on one arm.",
    "case": "One input the eval feeds every arm; repeats of a case are not independent.",
    "usable": "A trial the test setup did not spoil; only usable trials count for or against an arm.",
    "lost": (
        "A trial that could not be scored because the test setup broke (a tool, simulator or judge), so it counts "
        "for no arm."
    ),
    "failed by check": "A trial that broke a rule code checks, such as a required tool call.",
    "failed by judge": "A trial the judge model scored below the bar on some dimension.",
    "cost per reply": "What production would spend per reply, from the models the arm runs.",
    "case floor": (
        "Five independent cases per arm: below it a gap has no interval; from there an interval says whether the "
        "gap would hold on more cases."
    ),
    "separated": "The difference from the control is larger than chance explains, after correcting for every test.",
    "interval": "The range the true value likely lies in; a wide one means the run measured it loosely.",
    "change": "The arm's value minus the control's: in points for a rate, in the reading's own unit otherwise.",
    "cases": "How many independent cases are behind a figure; paired means both arms ran the same ones.",
    "control mean": "The control's figure on the reading, over the cases the test read.",
    "arm mean": "The arm's figure on the reading, over the same cases as the control's.",
    "Hedges' g": (
        "The change in units of the spread between trials, so readings on different scales compare; 0.2 is small, "
        "0.8 large."
    ),
    "pairing r": "How much a case's result on one arm predicts its result on the other; high means pairing helped.",
    "adjusted p": (
        "The chance of a change this large with no real difference, raised to allow for every comparison tested; "
        "under 0.05 counts as separated."
    ),
    "decision": "What the guardrail or bar came to on this arm: held or breached, cleared or missed, else undecided.",
    "stratum": "A named group of cases the author set apart, read on its own beside the whole.",
    "label": "One class a classifier may predict; precision and recall are read per label.",
    "margin": "The smallest change worth acting on, declared before the run.",
    "guardrail": "A reading an arm must not get worse on, whatever it gains elsewhere.",
    "bar": "A threshold a reading must clear on its own, without a control.",
    "exploratory": "No question was declared before the run, so the page offers leads, not decisions.",
    "settled": (
        "Keep where the run shows the arm breaks the clause or cannot show it holds. Switch only where its interval "
        "shows the clause holds, on at least 5 cases per arm."
    ),
    "judge": "The model that scores each trial's transcript against a rubric.",
    "calibrated": "A judge whose scores have been shown to agree with people's.",
    "marks": "✓ passed, ✗ failed, – lost to the test setup; one mark per repeat.",
    "ran": "Every trial the arm ran, usable or not.",
    "passed": "Usable trials that met every check and cleared every judged bar.",
    "flag": (
        "Regression: a candidate passed a smaller share of its usable trials on the case than the control; a tie or "
        "a gain never flags. No usable trial: an arm has none on the case."
    ),
    "outcome": "How a trial ended: passed, failed, or lost to the test setup.",
    "why": "Which check or judged dimension failed the trial, or what the test setup broke.",
    "clause": "One condition of the decision rule, settled on its own; the rule switches only if every clause holds.",
    "reading": "One thing measured on a trial: a check, a judged dimension, a cost or a time.",
    "confidence": "How sure the analysis's writer was of the finding: low, medium, high or very high.",
    "based on": (
        "The evidence behind a finding: checks and measures, or a judge's scores and how far that judge has been "
        "checked against people."
    ),
    "lever": "A setting the campaign can change, such as the model or a prompt; a finding names the levers it bears on.",
    "leverage": "How much a next step would change what the campaign can decide, in the writer's judgement.",
    "coverage": "How well the campaign swept a lever: measured, thin (too few repeats) or unswept (one level ran).",
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

#: Comparative words an arm's label may carry, each with what it claims against the control: ``cost`` read on cost per
#: reply, ``speed`` on latency, and which way, ``lower`` (cheaper, faster) or ``higher``. Narrow on purpose: a label
#: is a name, and only a word that plainly claims a direction is read against the arm's own row.
LABEL_CLAIMS: tuple[tuple[str, Literal["cost", "speed"], Literal["lower", "higher"]], ...] = (
    ("cheaper", "cost", "lower"),
    ("cheap", "cost", "lower"),
    ("lower-cost", "cost", "lower"),
    ("costlier", "cost", "higher"),
    ("pricier", "cost", "higher"),
    ("more expensive", "cost", "higher"),
    ("faster", "speed", "lower"),
    ("quicker", "speed", "lower"),
    ("slower", "speed", "higher"),
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
#: A line quoting something whole, as the first screen quotes the declared question (``Asked before the run: "…"``):
#: the quotation is the page's data, held to no prose rule.
_QUOTED_LINE = re.compile(r'^[A-Z][a-z]+(?: [a-z]+)*: ".+"$')


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
    blocks, folded trial details, the methods' disclosures fold and a line quoting something whole (the declared
    question under the arms line) are data and quoted text, not prose.

    Args:
        markdown: The page.
        question: The declared question, which only its own quoted line may carry.
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
        if _QUOTED_LINE.match(line):
            continue
        found += lint_prose(line.lstrip("-> *").strip(), question=question)
    return found


__all__ = [
    "BANNED_PHRASES",
    "CASE_LABEL_CHARS",
    "GLOSSARY",
    "HUMANIZE_OVERRIDES",
    "LABEL_CLAIMS",
    "MAX_HEADER_WORDS",
    "MAX_NUMERALS",
    "MAX_WORDS",
    "ArmName",
    "HumanizeKind",
    "StyleViolation",
    "arm_name",
    "code",
    "fix_prose",
    "humanize",
    "is_dump",
    "lint_page",
    "lint_prose",
    "measure_name",
    "sentences",
]
