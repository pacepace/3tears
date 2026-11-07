"""How a report speaks: one place for the names, the glossary and the rules every rendered sentence is held to.

The renderer (:mod:`threetears.evals.analysis.report`) reads this module for every sentence code writes on a page.

- :func:`humanize` is the one seam an identifier passes through on its way to a reader: a model id, an arm, a judged
  dimension, a case, a provider. :data:`HUMANIZE_OVERRIDES` names exceptions by hand. A check expression is code and
  never passes through it: :func:`measure_name` names a check's measure by the name :func:`check_names` gives it (the
  caller passes it as ``declared=``, read off the bundle), and no header or lead prints a check's expression; the
  page's glossary carries it, once, as the definition of the check's name (:func:`check_definition`).
- :class:`ArmName` names an arm twice: in full on its first mention (``deepseek-v4.1-flash (production, control)``)
  and by its given label after that (``production``).
- :func:`arm_display_name` is the one rule a name code computed from an arm's levers is printed by, on every page:
  ``Fanout 2``, never ``fanout 2`` beside a ``Fanout 2`` elsewhere.
- :data:`GLOSSARY` defines every term of art a page uses, once, in one plain sentence.
- :func:`lint_prose` holds a sentence to the rules. The package's own tests apply them to the text code writes (the
  page, whole); a host may apply them to its own. Nothing here reads, rewrites or drops what a model wrote; the
  writer's skill states the same rules, and a host's report_writer eval can measure whether a report keeps them, on
  the dimensions its rubric declares.
"""

from __future__ import annotations

import ast
import functools
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from threetears.evals.kernel.metrics import goal_check_of
from threetears.evals.schema.goal_grammar import (
    ACTION_BUILTINS,
    FIRE_PREDICATES,
    PATH_ROOTS,
    REFUSAL_BUILTINS,
    WORLD_ROOT,
)

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


def page_date(stamp: str) -> str:
    """A date or an ISO-8601 instant in the one form every page dates in (``14 Oct 2026``), or as given when it does
    not parse.

    :param stamp: The date or instant, as stored.
    :ptype stamp: str
    :return: The date as a page prints it.
    :rtype: str
    """
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        # NOSILENT: an unparseable stamp is printed as stored rather than dropped or reformatted.
        return stamp
    return f"{moment.day} {moment:%b %Y}"


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


#: What a page calls a goal check its template names nothing and whose expression names no subject a reader can read
#: (:func:`_check_subject`): numbered where an aspect reads several such (:func:`check_names`).
GOAL_CHECK = "Goal check"


def measure_name(name: str, reader_name: str | None = None, *, declared: str | None = None) -> str:
    """A measure as a page names it, in every header and every lead.

    A goal check's measure is named by ``declared``: the name its aspect's bundle gives it
    (:meth:`~threetears.evals.analysis.bundle.schema.AnalysisContextBundle.check_name`), the display name its template
    declares (``EvalTemplate.goal_check_names``), else what it tests (``Refund amount check``, :func:`check_names`).
    The bundle's ``measure_catalog`` never carries it, since the catalog is in every fingerprint. With none passed, a
    check is named by what it tests alone. Never by its expression, which a long check makes a paragraph in a header,
    and never by the catalog's rate name ("Pass rate of …"), since a single trial's failure has no rate: the expression
    is the glossary's definition of the name (:func:`check_definition`). Any other measure by its reader name, else its
    key humanized; ``declared`` names checks only.

    :param name: The measure's key.
    :ptype name: str
    :param reader_name: Its reader-facing name, when the catalog carries one. Never read for a goal check.
    :ptype reader_name: str | None
    :param declared: A goal check's name on its aspect, or None for none given.
    :ptype declared: str | None
    :return: The name.
    :rtype: str
    """
    if (check := goal_check_of(name)) is not None:
        return declared if declared is not None else check_names([check], {})[check]
    return reader_name or humanize("dimension", name)


def _action_word(builtin: str | None) -> str | None:
    """The word a call predicate's subject takes after its action (``Send message calls``), read off the goal
    grammar's own sets, so a builtin the grammar gains is named as its set's are: ``refusals`` for a builtin reading
    the calls a tool refused, ``calls`` for any other naming an action, ``firing`` for an event predicate; None for a
    builtin naming no action."""
    if builtin in REFUSAL_BUILTINS:
        return "refusals"
    if builtin in ACTION_BUILTINS:
        return "calls"
    if builtin in FIRE_PREDICATES:
        return "firing"
    return None


#: The roots a path names no subject under: a case parameter (``variation``), the expectation a check holds the end
#: state to rather than what it tests, and the element a generator binds. A path under the end state's root
#: (``state.refund.amount``) names its subject with that root dropped.
_NO_SUBJECT_ROOTS = PATH_ROOTS - {WORLD_ROOT}


def _path(node: ast.expr) -> list[str] | None:
    """A dotted path's names (``state.refund.amount`` -> ``[state, refund, amount]``), indexes skipped; None for any
    other node."""
    names: list[str] = []
    while True:
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Name):
            names.append(node.id)
            return names[::-1]
        else:
            return None


#: A literal a call predicate's action is read from: a dotted identifier (``support.send_message``). Any other text is a
#: host's words, never a name's: a name is built from identifiers only, so no literal's markup reaches a page.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

#: The most subjects a check's name joins (``Send message or reply check``); a check reading more is a goal check.
_MOST_SUBJECTS = 2


@dataclass(frozen=True)
class _Subject:
    """One thing a check tests: its words (``send message``) and, for a call predicate, the action word after them
    (``calls``)."""

    words: str
    action: str | None = None

    def text(self, *, action: bool = True) -> str:
        return f"{self.words} {self.action}" if action and self.action else self.words


def _reads(node: ast.AST) -> set[tuple[str, ...]]:
    """Every path and call predicate an expression reads, each as a tuple: what tells a guard from its consequent. A
    path under a root that names no subject (a case parameter, a generator's element) is the expectation, not what is
    tested, so a guard reading one beside the end state (``contains(state.music.queue, variation.track)``) is still a
    guard when a later operand reads that end state."""
    found: set[tuple[str, ...]] = set()
    # A predicate's own name (``contains``, ``call_count``) is the function a call makes, never a path it reads.
    called = {id(part.func) for part in ast.walk(node) if isinstance(part, ast.Call)}
    for part in ast.walk(node):
        if id(part) in called:
            continue
        if isinstance(part, ast.Call) and isinstance(part.func, ast.Name) and _action_word(part.func.id) is not None:
            first = part.args[0] if part.args else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.add((part.func.id, first.value))
        elif isinstance(part, ast.Attribute | ast.Name) and (path := _path(part)) is not None:
            if path[0] not in _NO_SUBJECT_ROOTS:
                found.add(tuple(path))
    return found


def _consequents(operands: Sequence[ast.expr]) -> list[ast.expr]:
    """An ``and``'s operands less its guards: an operand every path of which a later operand reads too
    (``state.order.total > 0`` before ``state.refund.amount <= state.order.total``) only guards what follows it."""
    reads = [_reads(operand) for operand in operands]
    return [
        operand
        for index, operand in enumerate(operands)
        if not (reads[index] and any(reads[index] <= later for later in reads[index + 1 :]))
    ]


def _subjects_of(node: ast.AST, bound: frozenset[str]) -> list[_Subject]:
    """The subjects an expression tests, in the order it is written, each once: every operand's of an ``and`` or an
    ``or`` (an ``and``'s guards passed over, :func:`_consequents`), else the first one it reads: the action a call
    predicate names, else a path's names, the world's root and a trailing ``length`` dropped; none where nothing in it
    names one."""
    if isinstance(node, ast.BoolOp):
        operands = _consequents(node.values) if isinstance(node.op, ast.And) else node.values
        found: list[_Subject] = []
        for operand in operands:
            found.extend(one for one in _subjects_of(operand, bound) if one not in found)
        return found
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else None
        first = node.args[0] if node.args else None
        if (
            (action := _action_word(name)) is not None
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
        ):
            if not _IDENTIFIER.fullmatch(first.value):
                return []
            return [_Subject(re.sub(r"_+", " ", first.value.rsplit(".", 1)[-1]).strip(), action)]
        if name == "passed" and not node.args:
            return [_Subject("deliberate pass")]
        children: list[ast.AST] = [*node.args, *(keyword.value for keyword in node.keywords)]
    elif isinstance(node, ast.GeneratorExp | ast.ListComp):
        targets = {
            target.id for gen in node.generators for target in ast.walk(gen.target) if isinstance(target, ast.Name)
        }
        bound = bound | targets
        # A generator tests each element of a collection: the collection is its subject.
        children = [*(gen.iter for gen in node.generators), node.elt]
    elif isinstance(node, ast.expr) and (path := _path(node)) is not None:
        if path[0] in bound or path[0] in _NO_SUBJECT_ROOTS:
            return []
        words = path[1:] if path[0] == WORLD_ROOT and len(path) > 1 else path
        words = words[:-1] if len(words) > 1 and words[-1] == "length" else words
        said = " ".join(re.sub(r"_+", " ", word) for word in words).strip()
        return [_Subject(said)] if said else []
    else:
        children = list(ast.iter_child_nodes(node))
    for child in children:
        if found := _subjects_of(child, bound):
            return found
    return []


def _joined(subjects: Sequence[_Subject], node: ast.AST) -> str | None:
    """A check's subjects as its name says them: one alone; two joined by the word their ``and`` or ``or`` reads, the
    action word they share said once and dropped (``Send message or reply``); None for none or more than
    :data:`_MOST_SUBJECTS`."""
    if not subjects or len(subjects) > _MOST_SUBJECTS:
        return None
    if len(subjects) == 1:
        return subjects[0].text()
    shared = len({one.action for one in subjects}) == 1
    word = "and" if isinstance(_top_bool(node), ast.And) else "or"
    return f" {word} ".join(one.text(action=not shared) for one in subjects)


def _top_bool(node: ast.AST) -> ast.boolop | None:
    """The first ``and`` or ``or`` an expression reads, outermost first."""
    for part in ast.walk(node):
        if isinstance(part, ast.BoolOp):
            return part.op
    return None


@functools.cache
def _check_subject(expression: str) -> str | None:
    """What a goal check tests, in a reader's words, read off its expression alone: ``Refund amount`` for
    ``state.refund.amount <= state.order.total`` (and for it guarded by ``state.order.total > 0 and ...``, named by
    what the guard lets through), ``Send message or reply`` for ``call_count("support.send_message") >= 1 or
    call_count("support.reply") >= 1``, ``Field accuracy`` for ``field_accuracy >= 0.92``; None where the expression
    does not parse, names no subject, or tests more than two. A call's action is read only from a literal that is a
    dotted identifier, never from other text.

    :param expression: The check, verbatim.
    :ptype expression: str
    :return: The subject, capitalised, or None.
    :rtype: str | None
    """
    for text in (expression.strip(), f"({expression.strip()})"):
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError:
            # NOSILENT: an expression this grammar cannot parse is named a goal check, never refused here.
            continue
        found = _joined(_subjects_of(tree.body, frozenset()), tree.body)
        return f"{found[:1].upper()}{found[1:]}" if found else None
    return None


def check_names(checks: Iterable[str], declared: Mapping[str, str]) -> dict[str, str]:
    """The name a page gives each of an aspect's goal checks, by expression: the display name ``declared`` gives it,
    else what it tests (:func:`_check_subject`) with ``check`` after it (``Refund amount check``), else
    :data:`GOAL_CHECK`. An expression is named alike wherever none of the ``checks`` beside it would share its name;
    where two would (or an unnamed one would take a declared one's) each is numbered, in the checks' sorted order
    (``Ticket status check 1``, ``Ticket status check 2``), so a check added beside it can move its number.

    :param checks: The checks a page names, by expression, in any order and with repeats.
    :ptype checks: Iterable[str]
    :param declared: Declared display names, by expression.
    :ptype declared: Mapping[str, str]
    :return: Each check's name, by expression, declared ones included, in sorted order.
    :rtype: dict[str, str]
    """
    every = sorted({*checks, *declared})
    taken = {name.casefold() for name in declared.values()}
    named = {check: f"{subject} check" if (subject := _check_subject(check)) else GOAL_CHECK for check in every}
    shared: dict[str, list[str]] = {}
    for check in every:
        if check not in declared:
            shared.setdefault(named[check], []).append(check)
    names = dict(declared)
    for name, held in shared.items():
        alone = len(held) == 1 and name.casefold() not in taken
        names.update({check: name if alone else f"{name} {index}" for index, check in enumerate(held, start=1)})
    return {check: names[check] for check in every}


#: The hover a link to a check's name carries: never the check's expression or its name, which are a host's text and
#: never belong in an attribute; the glossary entry the link lands on states them.
CHECK_LINK_TITLE = "Code runs this goal check on every trial; the glossary states it in full."


#: A line break in a check's expression, with the indentation around it: laid out on one line, a reader's line wraps
#: where the page does. Whitespace inside a line, a quoted string's included, stays as written.
_ONE_LINE = re.compile(r"[ \t]*\r?\n[ \t]*")


def _check_expression(expression: str) -> str:
    """A check's expression on one line, as the glossary prints it: each line break and the indentation around it a
    single space, every other character as written."""
    return _ONE_LINE.sub(" ", expression).strip()


def check_definition(
    by_aspect: Sequence[tuple[str, str | None]],
    *,
    span: Callable[[str], str] = code,
    words: Callable[[str], str] = str,
) -> str:
    """The glossary's sentence for a check's name: the expression it stands for, whole, as code, the one place a page
    prints it beside the name.

    :param by_aspect: Each expression the name stands for, with the aspect it does so on, or None where it stands for
        one expression on every aspect the page reads.
    :ptype by_aspect: Sequence[tuple[str, str | None]]
    :param span: Sets an expression as code in the page's form: a backtick span (the default, :func:`code`), a
        Markdown span no backtick inside can close, or escaped HTML.
    :ptype span: Callable[[str], str]
    :param words: Sets an aspect's name as the page's form prints text: as written, or escaped HTML.
    :ptype words: Callable[[str], str]
    :return: The sentence.
    :rtype: str
    """
    said = [span(_check_expression(check)) + (f" on {words(aspect)}" if aspect else "") for check, aspect in by_aspect]
    checks = said[0] if len(said) == 1 else f"{', '.join(said[:-1])} and {said[-1]}"
    return f"The goal check {checks}, which code runs on every trial."


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
    #: The levers whose reader names spell ``label``, where code computed it from them (``fanout 2, retry 1``); empty
    #: for a label given, or a level value read as written (``careful``, a model's name).
    levers: tuple[str, ...] = ()

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
        label = ", ".join(f"{lever_reader_name(match['axis'])} {match['level']}" for match in pairs if match)
        return ArmName(label=label, control=control, levers=tuple(match["axis"] for match in pairs if match))
    level = pairs[0]["level"] if len(pairs) == 1 and pairs[0] else display
    if labelled := _LABELLED.match(level):
        return ArmName(label=labelled["label"], model=labelled["model"], control=control)
    if "/" in level and " " not in level:
        return ArmName(label=humanize("model", level), model=level, control=control, given=False)
    return ArmName(label=level, control=control)


def lever_reader_name(lever: str, declared: Mapping[str, str] | None = None) -> str:
    """The one name a reader is shown for a lever, everywhere a page names one: an arm name computed from it, and a
    configuration's setting. The name declared for it where one is (``declared``: a report input's factor names, by
    id, ``Reasoning budget``), else the package's reader name for its id (:func:`humanize`, ``max output tokens``,
    never ``max_output_tokens``).

    :param lever: The lever, by id.
    :ptype lever: str
    :param declared: Each lever's declared name, by id; None or empty where nothing declares one.
    :ptype declared: Mapping[str, str] | None
    :return: Its reader name.
    :rtype: str
    """
    return (declared or {}).get(lever) or humanize("dimension", lever)


def arm_display_name(name: str, levers: Iterable[str], declared: Mapping[str, str] | None = None) -> str:
    """The name every surface prints for an arm whose name code computed from its levers: its first character upper
    case where it opens on one of ``levers``' reader names (``fanout 2`` to ``Fanout 2``, ``fanout 2 · retry 1`` to
    ``Fanout 2 · retry 1``), so prose, tables, legends and keys print one name and no reader takes ``Fanout 2`` and
    ``fanout 2`` for two arms (owner, 2026-10-28). A name opening on anything else, a level value (``careful``) or a
    model's name, is a value, printed as written.

    Called only where the data says code computed the name: a label a person gave, a declared short name and a
    model's name never pass through it. The name is shown capitalised and never stored so: an arm's id, its variant
    key and :func:`~threetears.evals.analysis.arms.arm_names`' ``axis=level`` names are unchanged.

    :param name: The computed name, as code spelled it.
    :ptype name: str
    :param levers: The levers it was spelled from, by id; their reader names (:func:`lever_reader_name`) are what
        may open it.
    :ptype levers: Iterable[str]
    :param declared: Each lever's declared name, by id, as :func:`lever_reader_name` reads it.
    :ptype declared: Mapping[str, str] | None
    :return: The name as printed.
    :rtype: str
    """
    for lever in levers:
        reader = lever_reader_name(lever, declared)
        if reader and (name == reader or name.startswith(f"{reader} ")):
            return name[:1].upper() + name[1:]
    return name


#: The reader's words for a contrast the evidence leaves undecided, as a table cell says them and the glossary defines
#: them once: never "not separated", which a first-time reader takes for "the two are the same". A sentence says it
#: whole, naming the arms (:func:`cannot_tell_apart`).
CANNOT_TELL_APART = "cannot tell apart"

#: The undecided reading where a sentence says it about its arms (``the evidence cannot tell v2 and v1 apart``,
#: :func:`cannot_tell_apart`) or a cell in its short form: what a page links to the glossary's entry.
CANNOT_TELL_APART_PHRASE = re.compile(r"cannot tell\b(?:[^<>]{0,160}?\s)?apart\b", re.IGNORECASE)


def cannot_tell_apart(arms: Sequence[str], control: str) -> str:
    """The undecided reading as a sentence says it, each arm by name and the control last: ``the evidence cannot tell
    v2 and v1 apart``; with several arms ``the evidence cannot tell v2 or v3 apart from v1``.

    The one sentence form of :data:`CANNOT_TELL_APART`: every page, print and exit line that says a contrast is
    undecided says it so, and never "not separated" or "nothing separated".

    :param arms: The arms read against the control, as the page names them; at least one.
    :ptype arms: Sequence[str]
    :param control: The control, as the page names it.
    :ptype control: str
    :return: The clause, with no closing stop.
    :rtype: str
    """
    if len(arms) == 1:
        return f"the evidence cannot tell {arms[0]} and {control} apart"
    named = f"{', '.join(arms[:-1])} or {arms[-1]}"
    return f"the evidence cannot tell {named} apart from {control}"


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
    "cost per reply": "What production would spend per reply on the models the arm runs, as its runs were routed.",
    "measuring spend, all roles": (
        "What measuring a trial cost: every call billed, the judge's and any simulated user's included, not what "
        "production would spend."
    ),
    "cost per 1,000 replies": (
        "What production would spend on 1,000 replies on the models the arm runs, as its runs were routed."
    ),
    "case floor": (
        "Five independent cases per arm: below it a gap has no interval; from there an interval says whether the "
        "gap would hold on more cases."
    ),
    CANNOT_TELL_APART: (
        "The difference from the control could be chance either way, so neither arm is shown better, and it never "
        "means the two are the same."
    ),
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
        "under 0.05 the evidence tells the arms apart."
    ),
    "decision": "What the guardrail or bar came to on this arm: held or breached, cleared or missed, else undecided.",
    "stratum": "A named group of cases the author set apart, read on its own beside the whole.",
    "label": "One class a classifier may predict; precision and recall are read per label.",
    "margin": "The smallest change worth acting on, declared before the run.",
    "guardrail": "A reading an arm must not get worse on, whatever it gains elsewhere.",
    "bar": "A threshold a reading must clear on its own, without a control.",
    "exploratory": "Run before any question was declared: each reading is a lead for a later run to test.",
    "floor status": (
        "Broken where an interval shows the arm past the floor, or it costs no less; held where one shows it "
        "within, from 5 cases per arm."
    ),
    "point estimate": (
        "A difference as this run measured it, before its interval says how far more cases could move it."
    ),
    "judge": "The model that scores each trial's transcript against a rubric.",
    "judged score": ("A judge's score on one dimension of a trial, from 1 to 5 where 5 is best, or pass or fail."),
    "calibrated": "A judge whose scores have been shown to agree with people's.",
    "marks": "✓ passed, ✗ failed, – lost to the test setup; one mark per repeat.",
    "ran": "Every trial the arm ran, usable or not.",
    "verdict": (
        "What a contrast came to for the arm against the control: shown better or worse, within the margin, or the "
        "evidence cannot tell them apart."
    ),
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
    "CANNOT_TELL_APART",
    "CANNOT_TELL_APART_PHRASE",
    "CASE_LABEL_CHARS",
    "CHECK_LINK_TITLE",
    "GLOSSARY",
    "GOAL_CHECK",
    "HUMANIZE_OVERRIDES",
    "MAX_HEADER_WORDS",
    "MAX_NUMERALS",
    "MAX_WORDS",
    "ArmName",
    "HumanizeKind",
    "StyleViolation",
    "arm_display_name",
    "arm_name",
    "cannot_tell_apart",
    "check_definition",
    "check_names",
    "code",
    "declared_check_name",
    "humanize",
    "is_dump",
    "lever_reader_name",
    "lint_prose",
    "measure_name",
    "page_date",
    "sentences",
]
