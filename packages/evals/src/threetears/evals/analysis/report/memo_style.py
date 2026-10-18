"""The memo's words held to the page's style: fixed where code can fix them, listed where it cannot, never let through.

A memo is model-written prose, and the page around it is code's. :class:`MemoStyle` passes every sentence the memo
contributes through these steps, in order:

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
5. **Sample sizes.** Where the memo says what to run next (a next step's title and ``why``, a question's answer, a
   deferred decision's ``revisit_when``), it states no sample size or cost of its own (``Expand the suite to 30 cases
   for $0.40``): the cases a switch needs are code's (:func:`~threetears.evals.analysis.report.decision.rule_projection`),
   and a writer's guess at them understated them by 4-15x (#846). A sentence there that proposes more cases (a count,
   ``thirty``, ``double``, ``3x``, ``a 50-case suite``, ``k to 10``, beside a word for getting more: add, run, expand,
   more, new, another, settle) at a count the projections do not state, or that names a cost they do not price, is
   replaced by code's sentence on the projected sample, once, and otherwise dropped; each is disclosed in the methods.
   A count the run itself reaches (``the 3 cases the candidate missed``) is a measured fact, not a proposal.
6. **Settled differences.** Where a clause the rule settled reads a finding's reading, a finding's sentence saying
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
from threetears.evals.analysis.report.decision import RuleVerdict, SampleProjection, projection_sentence
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

#: The things a sample is counted in, plural: what a proposal to run more is a count of. A count of trials (repeats
#: included) is held to the trials the run reached, every other to its cases.
_TRIAL_UNITS = r"(?:trials|runs|repeats|replicates|repetitions)"
_UNITS = rf"(?:cases|samples|examples|prompts|scenarios|conversations|items|inputs|queries|tasks|{_TRIAL_UNITS})"
_IN_TRIALS = re.compile(rf"\b{_TRIAL_UNITS}$", re.I)

#: Words between a count and its unit that leave it a count of the same things: ``30 more independent test cases``.
_BETWEEN = r"(?:(?:more|new|additional|extra|independent|paired|fresh|distinct|further|total|test|different|real)\s+)*"

#: Number words a writer spells a count in, with their values; a count of many (``dozens``) is -1, never a figure.
_WORD_COUNTS: dict[str, int] = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen".split()
    )
} | {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_WORD_COUNTS |= {"hundred": 100, "a hundred": 100, "a dozen": 12, "dozens": -1, "hundreds": -1, "thousands": -1}
_SPELLED = "|".join(sorted((re.escape(word) for word in _WORD_COUNTS), key=len, reverse=True))

#: A sample size in words, each group one count: a numeral or a range of two beside a unit (``30 cases``, ``30-60
#: more test cases``, ``~30+ trials``), a compound (``a 50-case suite``), a count after a sample's name (``the suite to
#: 30``, ``a sample of 60``, ``n = 30``), or one spelled (``thirty cases``, ``thirty-five``). A case named by its
#: number (``case 4``) is a name, not a size, and reads as none.
_SAMPLE_SIZE = re.compile(
    rf"(?<![\w.$-])(?:~\s?)?(\d[\d,]*)\+?(?:\s*(?:-|–|to|or)\s*(\d[\d,]*)\+?)?\s+{_BETWEEN}{_UNITS}\b"
    r"|(?<![\w.$-])(\d[\d,]*)-(?:case|sample|trial|prompt|scenario)\b"
    r"|\b(?:sample size|sample|suite|case set|case bank|n)\s*(?:to|of|=)\s*(?:about\s+|roughly\s+|around\s+|at least\s+|~\s?)?"
    r"(\d[\d,]*)(?![\w,]|\.\d)",
    re.I,
)
_SPELLED_SIZE = re.compile(rf"\b({_SPELLED})(?:-\w+)?\s+(?:of\s+)?{_BETWEEN}{_UNITS}\b", re.I)

#: A multiple of the sample, which no projection states: ``double the cases``, ``3x more trials``, ``twice as many``.
_MULTIPLE = re.compile(
    rf"\b(?:doubl\w*|tripl\w*|quadrupl\w*|twice|thrice|tenfold|\d+(?:\.\d+)?\s?[x×])(?:\s+\S+){{0,4}}?\s+(?:{_UNITS}|suite|sample)\b",
    re.I,
)

#: The repeats per case as a count to change to: ``raise k to 10``, ``k = 9``.
_REPEATS = re.compile(r"\bk\s*(?:to|=|of|at)\s*(\d+)\b")

#: A word for getting more of a sample, which turns a count beside it into a proposal.
_ACQUIRE = re.compile(
    r"\b(?:add\w*|run|runs|running|re-?run\w*|expand\w*|grow\w*|write|writing|collect\w*|gather\w*|extend\w*|"
    r"increas\w*|rais\w*|bump\w*|scal\w*|more|another|additional|new|extra|further|larger|bigger|fund\w*|launch\w*|"
    r"settl\w*|doubl\w*|tripl\w*|build\w*|draw\w*)\b",
    re.I,
)

#: An amount of money in words: ``$0.40``, ``0.20 USD``, ``2 dollars``.
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s*(?:USD|dollars?)\b", re.I)

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
    sample: SampleProjection | None = None
    samples: tuple[SampleProjection, ...] = ()
    violations: list[StyleViolation] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    hedged: list[str] = field(default_factory=list)
    proposed: list[str] = field(default_factory=list)
    dropped: int = 0

    def __post_init__(self) -> None:
        self._ids = self._model_ids()
        self._decimals, self._pairs = self._measured()
        self._counts, self._reached, self._repeats, self._costs = self._projected()

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

    def _projected(self) -> tuple[set[int], tuple[int, int], set[int], list[float]]:
        """What a sentence on what to run next may state: every count the projections state (``sample`` and
        ``samples``), the most cases or trials the run itself reached (any count up to it is the run's, not a
        proposal), its repeats per case, and every cost the projections price with what one case costs."""
        counts: set[int] = set()
        costs: list[float] = []
        for projection in (*([self.sample] if self.sample is not None else []), *self.samples):
            counts |= set(projection.counts())
            costs += [cost for cost in map(projection.cost, (projection.if_none, projection.if_observed)) if cost]
            if projection.per_case_usd:
                costs.append(projection.per_case_usd)
        reached = [count for board in self.bundle.scoreboard for count in (board.cases, board.usable_cases)]
        reached += [
            side.n_cases
            for family in self.bundle.multiple_comparisons.families
            for comparison in family.comparisons
            for side in (comparison.control, comparison.contrast)
        ]
        # What the run itself spent is measured, and a sentence may name it.
        costs += [
            cost for board in self.bundle.scoreboard for cost in (board.reply_cost_usd, board.test_cost_usd) if cost
        ]
        repeats = {summary.k_runs for summary in self.bundle.run_summaries}
        trials = max((board.ran for board in self.bundle.scoreboard), default=0)
        return counts, (max(reached, default=0), trials), repeats, costs

    def _priced(self, spelled: str) -> bool:
        """Whether a cost a sentence names is one the projections price, or one the run measured: as rendered, or
        rounded from one to the places it is written in, never zero."""
        value = float(spelled.replace(",", ""))
        places = len(spelled.split(".")[1]) if "." in spelled else 0
        if value <= 0:
            return False
        return any(abs(round(cost, places) - value) < 1e-9 and abs(value - cost) <= 0.1 * cost for cost in self._costs)

    def proposes_sample(self, text: str) -> bool:
        """Whether ``text``, a sentence on what to run next, states a sample size or a cost code did not project.

        A size is a proposal beside a word for getting more (:data:`_ACQUIRE`), at a count above what the run reached
        that no projection states; a multiple of the sample (``double the cases``) or a count of many (``dozens of
        cases``) always is, beside such a word. A cost is one where no projection prices it and the run did not
        measure it, beside a size or not.

        :param text: One sentence.
        :ptype text: str
        :return: True where the sentence proposes a sample size or a cost of its own.
        :rtype: bool
        """
        if any(not self._priced(found.group(1) or found.group(2)) for found in _MONEY.finditer(text)):
            return True
        if not _ACQUIRE.search(text):
            return False
        if _MULTIPLE.search(text):
            return True
        cases, trials = self._reached
        sizes = [
            (int(group.replace(",", "")), trials if _IN_TRIALS.search(found.group(0)) else cases)
            for found in _SAMPLE_SIZE.finditer(text)
            for group in found.groups()
            if group
        ]
        sizes += [
            (_WORD_COUNTS[found.group(1).casefold()], trials if _IN_TRIALS.search(found.group(0)) else cases)
            for found in _SPELLED_SIZE.finditer(text)
        ]
        if any(size < 0 or (size > reached and size not in self._counts) for size, reached in sizes):
            return True
        return any(int(found.group(1)) not in self._repeats for found in _REPEATS.finditer(text))

    def _planned(self, text: str) -> str:
        """``text`` with each sentence proposing a sample size of its own (:meth:`proposes_sample`) replaced by code's
        sentence on the projected sample, the first time, and dropped after it or where there is none."""
        kept: list[str] = []
        for part in sentences(text):
            if not self.proposes_sample(part):
                kept.append(part)
                continue
            self.proposed.append(part)
            if self.sample is not None and (said := projection_sentence(self.sample)) not in kept:
                kept.append(said)
        return " ".join(kept)

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
        self,
        text: str,
        *,
        fallback: str | None = None,
        decides: bool = True,
        settled: str | None = None,
        plans: bool = False,
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
        :param plans: Whether it says what to run next (a next step's title, a question's answer), so it states no
            sample size or cost of its own: a sentence that does is replaced by code's sentence on the projected sample,
            or dropped (:meth:`_planned`).
        :ptype plans: bool
        :return: The sentence to print; its remaining violations are recorded.
        :rtype: str
        """
        named = self.names_arms(" ".join(text.split()))
        if plans and not (named := self._planned(named)):
            return ""
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

    def prose(self, text: str, *, settled: str | None = None, said: str | None = None, plans: bool = False) -> str:
        """A paragraph in a fold (a finding's body, a caveat), each sentence held to every rule.

        :param text: What the memo wrote.
        :ptype text: str
        :param settled: Code's sentence on a reading a clause the rule settled, which replaces a sentence disputing that
            difference; None where no settled clause reads it.
        :ptype settled: str | None
        :param said: What the finding already printed: a disputing sentence is dropped, not replaced, where that holds
            code's sentence already.
        :ptype said: str | None
        :param plans: Whether it says what to run next (a next step's ``why``, a deferred decision's
            ``revisit_when``), whose sentences state no sample size or cost of their own: one that does is dropped.
        :ptype plans: bool
        :return: The kept sentences; a sentence restating a measured number or breaking a fixable rule is dropped.
        :rtype: str
        """
        kept = []
        for part in sentences(self.names_arms(text)):
            if plans and self.proposes_sample(part):
                self.proposed.append(part)
                continue
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
    lines += [
        f"The memo said “{said}”, a sample size or cost no code computed, so the page does not print it."
        for said in style.proposed
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
