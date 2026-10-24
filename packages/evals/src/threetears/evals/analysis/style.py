"""How a report speaks: one place for the names, the glossary and the rules every rendered sentence is held to.

The renderer (:mod:`threetears.evals.analysis.report`) reads this module for every sentence code writes on a page.

- :func:`humanize` is the one seam an identifier passes through on its way to a reader: a model id, an arm, a judged
  dimension, a case, a provider. :data:`HUMANIZE_OVERRIDES` names exceptions by hand. A check expression is code and
  never passes through it: :func:`measure_name` names a check's measure by the display name its template declares,
  when the caller passes it (``declared=``, read off the bundle with :func:`declared_check_name`), else by its
  expression, verbatim.
- :class:`ArmName` names an arm twice: in full on its first mention (``deepseek-v4.1-flash (production, control)``)
  and by its given label after that (``production``).
- :data:`GLOSSARY` defines every term of art a page uses, once, in one plain sentence.
- :func:`lint_prose` holds a sentence to the rules. The package's own tests apply them to the text code writes (the
  page, whole); a host may apply them to its own. Nothing here reads, rewrites or drops what a model wrote; the
  writer's prompt states the same rules, and a host's reporter eval can measure whether a memo keeps them, on the
  dimensions its rubric declares.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
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

    :param kind: What the identifier names.
    :ptype kind: HumanizeKind
    :param identifier: The identifier as stored.
    :ptype identifier: str
    :return: A model id without its vendor prefix; an arm by its given label (else its model id, vendor dropped), its
        ``axis=level`` pairs read as words; a dimension with its underscores as spaces, but a check expression (a
        call, a comparison, a quote) as written, since it is code; a case label where it reads as a name someone
        gave (``double-charge``: one token, not a generated id, not a bare number — empty for a dump, a sentence or
        an id, so the caller names the case by number); a provider as given.
    :rtype: str
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


def measure_name(name: str, reader_name: str | None = None, *, declared: str | None = None) -> str:
    """A measure as a page names it.

    A goal check's measure is named by ``declared``, the display name its template declares
    (``EvalTemplate.goal_check_names``), which the caller reads off the aspect's bundle
    (:meth:`~threetears.evals.analysis.bundle.schema.AnalysisContextBundle.check_name`); the bundle's
    ``measure_catalog`` never carries it, since the catalog is in every fingerprint. With none, a check is named by its
    expression, verbatim and whole, as a code span: never humanized, never shortened, and never by the catalog's
    rate name ("Pass rate of …"), since a single trial's failure has no rate. Any other measure by its reader name,
    else its key humanized; ``declared`` names checks only.

    :param name: The measure's key.
    :ptype name: str
    :param reader_name: Its reader-facing name, when the catalog carries one. Never read for a goal check.
    :ptype reader_name: str | None
    :param declared: A goal check's declared display name, or None for none declared.
    :ptype declared: str | None
    :return: The name.
    :rtype: str
    """
    if (check := goal_check_of(name)) is not None:
        return declared if declared is not None else code(check)
    return reader_name or humanize("dimension", name)


def declared_check_name(names: Mapping[str, str], measure: str) -> str | None:
    """The display name ``names`` (a check's expression to its name) declares for the goal check ``measure`` reads.

    :param names: Declared display names, by check expression (``AnalysisContextBundle.goal_check_names``).
    :ptype names: Mapping[str, str]
    :param measure: Any measure key.
    :ptype measure: str
    :return: The declared name, or None when ``measure`` is no goal check's or its check is not named.
    :rtype: str | None
    """
    check = goal_check_of(measure)
    return names.get(check) if check is not None else None


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

    :param display: The arm's name as :func:`~threetears.evals.analysis.arms.arm_names` spells it.
    :ptype display: str
    :param control: Whether it is the control.
    :ptype control: bool
    :return: The arm's given label and model, never as ``key=value``.
    :rtype: ArmName
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
    "aspect": (
        "One part of what a campaign asks, read on its own cases with its own margin; every aspect is read against "
        "the control."
    ),
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
    "depth": "How many times an arm ran each case: the fewest, the median and the most.",
    "common depth": "The fewest runs any arm gave a case all arms share, where pass^k compares them over those cases.",
    "between cases": (
        "The share of a comparison's noise that comes from cases differing, which only more cases lower, never "
        "more repeats."
    ),
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
    "exploratory": "No question was declared, so the page offers leads, not decisions.",
    "floor status": (
        "Broken where an interval shows the arm past the floor, or it costs no less; held where one shows it "
        "within, from 5 cases per arm."
    ),
    "point estimate": (
        "A difference as this run measured it, before its interval says how far more cases could move it."
    ),
    "judge": "The model that scores each trial's transcript against a rubric.",
    "calibrated": "A judge whose scores have been shown to agree with people's.",
    "marks": "✓ passed, ✗ failed, – lost to the test setup; one mark per repeat.",
    "ran": "Every trial the arm ran, usable or not.",
    "passed": "Usable trials that met every check and cleared every judged bar.",
    "flag": (
        "Regression marks a case a candidate passed on a smaller share of its usable trials than the control; a tie "
        "or a gain never flags."
    ),
    "outcome": "How a trial ended: passed, failed, or lost to the test setup.",
    "why": "Which check or judged dimension failed the trial, or what the test setup broke.",
    "floor": "A line an arm is warned for crossing, such as a pass rate dropping too far; it warns and never decides.",
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

    :param text: A sentence, a paragraph or a header.
    :ptype text: str
    :param question: The declared question; a sentence of it repeated verbatim is a violation.
    :ptype question: str | None
    :param header: Whether ``text`` is a header, held to :data:`MAX_HEADER_WORDS` words.
    :ptype header: bool
    :return: The violations, in the order found; empty when the text passes.
    :rtype: list[StyleViolation]
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
    "code",
    "declared_check_name",
    "humanize",
    "is_dump",
    "lint_prose",
    "measure_name",
    "sentences",
]
