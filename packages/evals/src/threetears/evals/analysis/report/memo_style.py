"""The memo's words held to the page's style: fixed where code can fix them, listed where it cannot, never let through.

A memo is model-written prose, and the page around it is code's. :class:`MemoStyle` passes every sentence the memo
contributes through four steps, in order:

1. **Names.** Every raw model id an arm carries becomes the arm's label (``deepseek-v4.1-flash`` → ``production``):
   the arms line named each in full once (:func:`~threetears.evals.analysis.style.humanize`). A label already beside
   the id (``production deepseek-v4.1-flash``) is kept once, not doubled.
2. **Rules code can fix.** A sentence carrying a banned phrase (meta-prose, a hedge on a measured fact) or repeating
   the declared question is dropped (:func:`~threetears.evals.analysis.style.fix_prose`).
3. **Measured numbers.** A sentence restating a number the bundle measured (a count ``6 of 12``, a cost, a rate) is
   dropped from a body, the finding's table carrying it; a one-sentence title that does so is replaced by the
   table's own row in words, so a measured fact is stated by code, flatly.
4. **Decisions.** Where a declared decision rule settled, a sentence whose decision word (``keep``, ``switch``,
   ``unanswerable``, ``no difference``) disagrees with it is replaced by the verdict line's own sentence, and the
   replacement is disclosed in the methods.
5. **Settled differences.** Where a clause the rule settled reads a finding's reading, a finding's sentence saying
   the evidence cannot confirm, establish or show the difference (:data:`_HEDGES`, a narrow set) disputes the rule:
   it is replaced by code's sentence on that reading, which states the counts and, under the case floor, that only
   whether the gap would hold on more cases is not estimated. That replacement is disclosed in the methods too.

What is left breaking a rule (a long sentence, too many numbers) is kept as written and listed, sentence by sentence,
in the methods: ``style: N violations``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.report.decision import RuleVerdict
from threetears.evals.analysis.style import ArmName, StyleViolation, fix_prose, humanize, lint_prose, sentences
from threetears.evals.kernel.metrics import goal_check_of

#: A decision word a memo sentence may assert, and the rule outcomes it agrees with.
_DECISION_WORDS: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (re.compile(r"\bunanswerable\b|\bcannot (?:be )?(?:answer|decide)", re.I), frozenset({"unsettled"})),
    (re.compile(r"\bno (?:measurable |real )?difference\b|\bdid not separate\b", re.I), frozenset({"unsettled"})),
    (re.compile(r"\bswitch(?:ing)? to\b|\badopt\b|\breplace\b", re.I), frozenset({"switch"})),
    (re.compile(r"\bkeep\b|\bstay with\b", re.I), frozenset({"keep", "unsettled"})),
)

#: A sentence saying the evidence cannot confirm, establish or show a difference: what a settled clause has already
#: decided. Narrow on purpose: a verb of proof under a negation, or a statistical non-result, nothing looser.
_HEDGES = re.compile(
    r"\b(?:cannot|can ?not|can't|could not|couldn't|unable to|(?:too few|not enough|insufficient)\b[^.;:]*?\bto)"
    r"\s+(?:statistically\s+|reliably\s+|formally\s+|yet\s+)?(?:confirm|establish|show|prove|demonstrate)\b"
    r"|\bnot (?:statistically )?(?:confirmed|established|significant)\b"
    r"|\b(?:did|does|do) not (?:statistically )?separate\b",
    re.I,
)

_NUMBER = re.compile(r"(?<![\w.])\$?(\d+(?:\.\d+)?)(?![\w])")
_N_OF_M = re.compile(r"\b(\d+) (?:of|out of|/) ?(\d+)\b")


@dataclass
class MemoStyle:
    """One memo's style pass over one bundle: what it fixed, replaced and could not fix."""

    bundle: AnalysisContextBundle
    names: dict[str, ArmName]
    question: str | None = None
    rule: RuleVerdict | None = None
    rule_sentence: str | None = None
    rule_question: str | None = None
    violations: list[StyleViolation] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    hedged: list[str] = field(default_factory=list)
    dropped: int = 0

    def __post_init__(self) -> None:
        self._ids = self._model_ids()
        self._decimals, self._pairs = self._measured()

    def _model_ids(self) -> list[tuple[re.Pattern[str], str]]:
        """Each arm's raw model id, with and without its vendor, longest first, and the label it becomes."""
        found: dict[str, str] = {}
        for name in self.names.values():
            if name.model is None or name.model == name.label:
                continue
            for spelled in (name.model, humanize("model", name.model)):
                if spelled != name.label:
                    found.setdefault(spelled, name.label)
        # A role word already naming the arm ("production deepseek-v4.1-flash", "deepseek-v4.1-flash (production)")
        # collapses with the id into the label once, never "production production".
        return [
            (
                re.compile(
                    rf"(?:(?<![\w-]){re.escape(label)}\s+)?(?<![\w./-]){re.escape(spelled)}(?![\w/-]|\.\d)"
                    rf"(?:\s*\({re.escape(label)}(?:,[^()]*)?\))?"
                ),
                label,
            )
            for spelled, label in sorted(found.items(), key=lambda item: -len(item[0]))
        ]

    def _measured(self) -> tuple[list[float], set[tuple[int, int]]]:
        """Every non-integer figure the bundle measured, and every ``passed of n`` pair it counts."""
        decimals: list[float] = []
        pairs: set[tuple[int, int]] = set()
        for board in self.bundle.scoreboard:
            decimals += [cost for cost in (board.reply_cost_usd, board.test_cost_usd) if cost is not None]
            pairs |= {(board.passed, board.usable), (board.passed, board.ran), (board.usable, board.ran)}
        for cell in self.bundle.cell_measures:
            for summary in cell.measures.measures:
                decimals += [value for value in (summary.mean, summary.rate) if value is not None]
                if summary.n_true is not None:
                    pairs.add((summary.n_true, summary.n))
                elif goal_check_of(summary.name) is not None and summary.mean is not None:
                    # A goal check's held count, as its finding's table states it: its rate times its observations.
                    held = summary.mean * summary.n
                    if abs(held - round(held)) < 1e-6:
                        pairs.add((round(held), summary.n))
        return [value for value in decimals if value != int(value)], pairs

    def holds_measured(self, text: str) -> bool:
        """Whether ``text`` states a number the bundle measured."""
        if any((int(a), int(b)) in self._pairs for a, b in _N_OF_M.findall(text)):
            return True
        for spelled in _NUMBER.findall(text):
            if "." not in spelled:
                continue
            places = len(spelled.split(".")[1])
            value = float(spelled)
            if any(abs(value - measured) <= 0.5 * 10**-places for measured in self._decimals):
                return True
        return False

    def names_arms(self, text: str) -> str:
        """``text`` with every raw model id an arm carries spelled as the arm's label."""
        for pattern, label in self._ids:
            text = pattern.sub(label, text)
        return text

    def _decides_otherwise(self, text: str) -> bool:
        """Whether ``text`` asserts a decision the settled rule contradicts."""
        if self.rule is None or self.rule.outcome == "unsettled":
            return False
        return any(pattern.search(text) and self.rule.outcome not in agrees for pattern, agrees in _DECISION_WORDS)

    def contradicts(self, text: str) -> bool:
        """Whether ``text`` asserts a decision the settled rule contradicts; one that does is recorded for the
        methods, as a replaced sentence is."""
        named = self.names_arms(" ".join(text.split()))
        if self._decides_otherwise(named):
            self.replaced.append(named)
            return True
        return False

    def sentence(
        self, text: str, *, fallback: str | None = None, decides: bool = True, settled: str | None = None
    ) -> str:
        """One sentence the memo stands on (a title, an answer, a proposal), held to every rule.

        :param text: What the memo wrote.
        :ptype text: str
        :param fallback: What code says instead when the sentence restates a measured number: the finding's table row.
        :ptype fallback: str | None
        :param decides: Whether the sentence asserts a decision (an answer, a title, an adopted proposal), so a settled
            rule it contradicts replaces it; a deferred proposal or a next step asserts none.
        :ptype decides: bool
        :param settled: Code's sentence on a reading a clause the rule settled, which replaces a sentence disputing that
            difference (:data:`_HEDGES`); None where no settled clause reads it.
        :ptype settled: str | None
        :return: The sentence to print; its remaining violations are recorded.
        :rtype: str
        """
        named = self.names_arms(" ".join(text.split()))
        if decides and self._decides_otherwise(named) and self.rule_sentence is not None:
            self.replaced.append(named)
            return self.rule_sentence
        if settled is not None and _HEDGES.search(named):
            self.hedged.append(named)
            return settled
        fixed, _ = fix_prose(named, question=self.question)
        if fixed != named:
            self.dropped += 1
        if fallback is not None and (not fixed or self.holds_measured(fixed)):
            if fixed:
                self.dropped += 1
            return fallback
        if not fixed:
            return fallback or ""
        self.violations += lint_prose(fixed, question=self.question)
        return fixed

    def prose(self, text: str, *, settled: str | None = None, said: str | None = None) -> str:
        """A paragraph in a fold (a finding's body, a caveat), each sentence held to every rule.

        :param text: What the memo wrote.
        :ptype text: str
        :param settled: Code's sentence on a reading a clause the rule settled, which replaces a sentence disputing that
            difference; None where no settled clause reads it.
        :ptype settled: str | None
        :param said: What the finding already printed: a disputing sentence is dropped, not replaced, where that holds
            code's sentence already.
        :ptype said: str | None
        :return: The kept sentences; a sentence restating a measured number or breaking a fixable rule is dropped.
        :rtype: str
        """
        kept = []
        for part in sentences(self.names_arms(text)):
            if self._decides_otherwise(part) and self.rule_sentence is not None:
                self.replaced.append(part)
                continue
            if settled is not None and _HEDGES.search(part):
                self.hedged.append(part)
                if settled not in (said or "") and settled not in kept:
                    kept.append(settled)
                continue
            fixed, remaining = fix_prose(part, question=self.question)
            if not fixed or self.holds_measured(fixed):
                self.dropped += 1
                continue
            self.violations += remaining
            kept.append(fixed)
        return " ".join(kept)

    def offending(self) -> list[str]:
        """Each sentence still breaking a rule, once, with the rules it breaks."""
        by_text: dict[str, list[str]] = {}
        for violation in self.violations:
            rules = by_text.setdefault(violation.text, [])
            if violation.rule not in rules:
                rules.append(violation.rule)
        return [f"{', '.join(rules)}: {text}" for text, rules in by_text.items()]


def row_words(reading: str, values: Iterable[tuple[str, str]]) -> str:
    """A finding's table row as one sentence code writes: ``Cost per reply: production $0.0031; candidate $0.0033.``"""
    return f"{reading}: " + "; ".join(f"{arm} {value}" for arm, value in values) + "."


def decision_disclosures(style: MemoStyle) -> Sequence[str]:
    """The methods' lines on what the memo said that the page did not print as written."""
    lines = [
        f"The memo said “{said}”, which the decision rule's settled outcome contradicts; the verdict line's sentence "
        "replaced it."
        for said in style.replaced
    ]
    lines += [
        f"The memo said “{said}”, which disputes a difference a clause of the decision rule settled; code's "
        "sentence on that reading replaced it."
        for said in style.hedged
    ]
    if style.dropped:
        lines.append(
            f"{style.dropped} memo sentence{'' if style.dropped == 1 else 's'} restated a measured number, described "
            "the page or repeated the question, and the tables say it instead."
        )
    return lines


__all__ = [
    "MemoStyle",
    "decision_disclosures",
    "row_words",
]
