"""Host vocabulary in the eval engine may only fall: a per-module, per-term ceiling over every module's full source.

The rule: **a host's vocabulary MUST NOT leak into the eval engine.** The import
boundary holds while host words reach engine *prose* -- a failure disclosure telling a reader what
"the listener" got, a docstring using a host's ``music.queue`` as the engine's example, a comment
saying how the first host bounds a persona turn. ``test_no_host_names_in_shared_contract.py`` reads
only declared names and token-shaped values in the modules it declares shared contract, so it cannot
see any of that.

This gate is the other half. It counts every term in :data:`TERMS` across the **full source** of
**every** module under ``src/`` -- docstrings, comments and strings included -- and compares each
module's count of each term with its row in
:data:`~packages.evals.tests.host_vocabulary_register.CEILINGS`. It fails in four ways, each naming
the module and what to write:

- a count **above** its ceiling fails, naming each hit's term and line;
- a module **missing** from the register fails by name: the register is total, so a new module is
  classified when it lands rather than defaulting to anything;
- a ceiling **above** its count fails too, naming the value to write -- slack left by a cut is room
  for the next leak, so a ceiling not lowered after it was earned is a failure, not a convenience;
- a row for a module that no longer exists, a row naming a term that is not counted, or a row
  spelling out a zero fails, so the register has one spelling for every state.

**The terms.** The program's list -- ``discodon``, ``persona``, ``universe``, ``id_universe``, ``DD_``,
``music``, ``research`` -- joined with every noun in
:data:`~packages.evals.tests.host_vocabulary.HOST_NOUNS`, so this gate never watches less than the
shared-contract canary does. Each term is its own column, so a change that sheds one host word and
adds another moves two numbers rather than netting to none. Matching:

- ``id_universe`` matches as that one identifier, in any case, and is taken out of the line before
  the noun pass, so it is never also counted as ``universe``: the columns are disjoint and a row's
  numbers sum to the module's host words.
- ``DD_`` is the first host's environment-variable prefix, matched case-sensitively where an
  identifier starts with it, so ``ADD_`` and ``add_`` are not hits.
- Every other term is a whole word after the canary's own identifier splitting
  (:func:`~packages.evals.tests.identifier_words.identifier_words`), any case, with an optional plural
  ``s``: ``dj`` inside ``adjust`` is not a hit, and ``Listeners`` and ``DJs`` are.

**What is not counted: the package's own path.** ``threetears.evals`` is where the engine lives --
every import and cross-reference spells it -- and it names no host, so it is removed before
counting. Anything past that prefix still counts.

The register only falls. The program's later phases take every row to ``{}``; an empty allowlist
is the end state, not something this gate already asserts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from packages.evals.tests.host_vocabulary import HOST_NOUNS
from packages.evals.tests.host_vocabulary_register import CEILINGS
from packages.evals.tests.identifier_words import identifier_words

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

#: The engine package. Register keys are paths relative to it; :func:`test_every_module_under_src_is_measured`
#: holds that nothing under ``src/`` sits outside it, so keying from here loses no module.
_EVAL_ROOT = _SRC_ROOT / "threetears" / "evals"

#: The program's term list, in its order, then the shared-contract canary's nouns it does not already name.
TERMS: tuple[str, ...] = tuple(
    dict.fromkeys(("discodon", "persona", "universe", "id_universe", "DD_", "music", "research", *HOST_NOUNS))
)

#: Terms matched on the raw line rather than as a split word, each taken out before the word pass.
_RAW_TERMS: dict[str, re.Pattern[str]] = {
    "id_universe": re.compile(r"(?<![A-Za-z0-9])id_universe(?![A-Za-z0-9])", re.IGNORECASE),
    "DD_": re.compile(r"(?<![A-Za-z0-9_])DD_"),
}

#: Terms matched as a whole word after identifier splitting, mapped back to the term from the word.
_WORD_TERMS: tuple[str, ...] = tuple(term for term in TERMS if term not in _RAW_TERMS)
_WORD_RE = re.compile(rf"^({'|'.join(_WORD_TERMS)})s?$", re.IGNORECASE)

#: The package's own path, removed before counting (see the module docstring).
_PACKAGE_PATH_RE = re.compile(r"\bthreetears[./]evals\b")


class Hit(NamedTuple):
    """One host term in a module."""

    line: int
    term: str


def host_terms(source: str) -> list[Hit]:
    """Every host term in ``source``, prose included, with its line.

    Args:
        source: A module's full text.

    Returns:
        One hit per occurrence, by line; on a line, the raw-matched terms come before the words.
    """
    hits: list[Hit] = []
    for number, line in enumerate(source.splitlines(), start=1):
        text = _PACKAGE_PATH_RE.sub(" ", line)
        for term, pattern in _RAW_TERMS.items():
            hits.extend(Hit(number, term) for _ in pattern.finditer(text))
            text = pattern.sub(" ", text)
        for word in identifier_words(text):
            matched = _WORD_RE.match(word)
            if matched:
                hits.append(Hit(number, matched.group(1).lower()))
    return hits


def measure(root: Path) -> dict[str, list[Hit]]:
    """The host terms in every module under ``root``, keyed by its path relative to ``root``.

    Args:
        root: The engine package root.

    Returns:
        Every module, including those with no hits.
    """
    return {
        path.relative_to(root).as_posix(): host_terms(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("*.py"))
    }


def _counts(hits: list[Hit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit.term] = counts.get(hit.term, 0) + 1
    return {term: counts[term] for term in TERMS if term in counts}


def _row(module: str, counts: dict[str, int]) -> str:
    return f'"{module}": {{{", ".join(f'"{term}": {n}' for term, n in counts.items())}}},'


def verdicts(measured: dict[str, list[Hit]], register: dict[str, dict[str, int]]) -> list[str]:
    """Why ``measured`` breaks ``register``, one sentence per defect; empty when every count sits at its ceiling.

    Args:
        measured: :func:`measure`'s result.
        register: Host terms allowed per module and term; a term left out of a row is allowed none.

    Returns:
        The failures, in module order.
    """
    out: list[str] = []
    for module in sorted(set(measured) | set(register)):
        if module not in measured:
            out.append(f"{module} has a register row but no such module exists under src/ any more: delete its row")
            continue
        hits = measured[module]
        counts = _counts(hits)
        if module not in register:
            out.append(
                f"{module} is not in the register, and every module under src/ is. A new module should name no "
                f"host at all; register what it carries as {_row(module, counts)}"
            )
            continue
        row = register[module]
        for term in sorted(set(row) - set(TERMS)):
            out.append(f"{module}'s row names {term!r}, which is not a counted term: delete it from the row")
        for term in TERMS:
            count, ceiling = counts.get(term, 0), row.get(term)
            if ceiling == 0:
                out.append(f"{module}'s row writes {term!r}: 0, and leaving the term out says that: delete it")
            elif count > (ceiling or 0):
                lines = ", ".join(str(hit.line) for hit in hits if hit.term == term)
                out.append(
                    f"{module} carries {count} {term!r}, over its ceiling of {ceiling or 0}. Write the added ones in "
                    f"the engine's own terms, or move what only the host can say to its adapter: line(s) {lines}"
                )
            elif ceiling is not None and count < ceiling:
                target = f"remove {term!r} from its row" if not count else f"lower {term!r} from {ceiling} to {count}"
                out.append(f"{module} fell to {count} {term!r}: {target}")
    return out


def test_every_module_sits_at_its_ceilings() -> None:
    """The gate: no module gains a host term, every module is registered, and none keeps slack after losing one."""
    failures = verdicts(measure(_EVAL_ROOT), CEILINGS)

    assert not failures, "Host vocabulary in the eval package moved off its register:\n  " + "\n  ".join(failures)


def test_every_module_under_src_is_measured() -> None:
    """A measure that read a subtree, or read nothing, would make the register's totality meaningless."""
    outside = sorted(
        path.relative_to(_SRC_ROOT).as_posix() for path in _SRC_ROOT.rglob("*.py") if _EVAL_ROOT not in path.parents
    )
    measured = measure(_EVAL_ROOT)

    assert not outside, f"modules under src/ outside the engine package, which the register does not key: {outside}"
    # Whichever module carries the most host words today, the measure must have read some: a measure
    # that read nothing would satisfy a register of empty rows. (It pinned ``schema/models.py`` until
    # that module was cleaned to zero, which is the reason it no longer names one.)
    assert any(_counts(hits) for hits in measured.values()), (
        "the measure read no host term in any module, so it read nothing"
    )


def _plant(root: Path) -> Path:
    """A minimal package tree with one module that names no host. Returns that module."""
    core = root / "core.py"
    core.write_text('"""Score a subject."""\n\nfrom threetears.evals.kernel import x\n', encoding="utf-8")
    return core


def test_a_count_above_its_ceiling_fails_naming_the_term_and_line(tmp_path: Path) -> None:
    """Prose counts: a docstring gaining ``the listener`` fails, naming the module, the term and its line."""
    core = _plant(tmp_path)
    assert verdicts(measure(tmp_path), {"core.py": {}}) == [], "the package path must not count"

    core.write_text(
        core.read_text(encoding="utf-8") + '\n\ndef f():\n    """Deliver to the listener."""\n', encoding="utf-8"
    )

    (failure,) = verdicts(measure(tmp_path), {"core.py": {}})
    assert failure.startswith("core.py carries 1 'listener', over its ceiling of 0.")
    assert failure.endswith("line(s) 7")


def test_each_term_is_its_own_column(tmp_path: Path) -> None:
    """Shedding one host word while adding another does not net to zero: two columns move."""
    core = _plant(tmp_path)
    core.write_text("# one persona\n", encoding="utf-8")
    assert verdicts(measure(tmp_path), {"core.py": {"persona": 1}}) == []

    core.write_text("# one universe\n", encoding="utf-8")

    assert verdicts(measure(tmp_path), {"core.py": {"persona": 1}}) == [
        "core.py fell to 0 'persona': remove 'persona' from its row",
        "core.py carries 1 'universe', over its ceiling of 0. Write the added ones in the engine's own terms, "
        "or move what only the host can say to its adapter: line(s) 1",
    ]


def test_a_module_missing_from_the_register_fails_by_name(tmp_path: Path) -> None:
    """Totality: an unregistered module fails even when it names no host, and the message gives its row."""
    _plant(tmp_path)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "fresh.py").write_text("# Discodon wraps the persona turn.\n", encoding="utf-8")

    assert verdicts(measure(tmp_path), {"run/fresh.py": {"discodon": 1, "persona": 1}}) == [
        "core.py is not in the register, and every module under src/ is. A new module should name no host at "
        'all; register what it carries as "core.py": {},'
    ]
    (failure,) = verdicts(measure(tmp_path), {"core.py": {}})
    assert failure.startswith("run/fresh.py is not in the register")
    assert failure.endswith('"run/fresh.py": {"discodon": 1, "persona": 1},')


def test_a_ceiling_above_its_count_fails(tmp_path: Path) -> None:
    """The ratchet: a removed host word leaves slack, and the failure names the value to write."""
    core = _plant(tmp_path)
    core.write_text("# a persona, two personas\n", encoding="utf-8")
    assert verdicts(measure(tmp_path), {"core.py": {"persona": 2}}) == []

    core.write_text("# a persona\n", encoding="utf-8")
    assert verdicts(measure(tmp_path), {"core.py": {"persona": 2}}) == [
        "core.py fell to 1 'persona': lower 'persona' from 2 to 1"
    ]

    core.write_text("# a subject\n", encoding="utf-8")
    assert verdicts(measure(tmp_path), {"core.py": {"persona": 2}}) == [
        "core.py fell to 0 'persona': remove 'persona' from its row"
    ]


def test_stale_rows_unknown_terms_and_spelled_zeros_are_refused(tmp_path: Path) -> None:
    """A row for a module that is gone, a term not counted, or a written-out 0 is slack in the register itself."""
    _plant(tmp_path)

    assert verdicts(measure(tmp_path), {"gone.py": {"persona": 3}, "core.py": {"personna": 1, "music": 0}}) == [
        "core.py's row names 'personna', which is not a counted term: delete it from the row",
        "core.py's row writes 'music': 0, and leaving the term out says that: delete it",
        "gone.py has a register row but no such module exists under src/ any more: delete its row",
    ]


def test_matching_rules() -> None:
    """Whole words with plurals; ``id_universe`` never doubles as ``universe``; ``DD_`` only as a prefix, in capitals."""
    source = (
        "adjust_adjacent = 1  # Listeners and DJs\n"
        "from threetears.evals.adapters.discodon import kind\n"
        "self.id_universe = ID_UNIVERSE  # one universe\n"
        "DD_EVAL_X, ADD_ONE, dd_x, _DD_Y  # music\n"
    )

    assert [(hit.line, hit.term) for hit in host_terms(source)] == [
        (1, "listener"),
        (1, "dj"),
        (2, "discodon"),
        (3, "id_universe"),
        (3, "id_universe"),
        (3, "universe"),
        (4, "DD_"),
        (4, "music"),
    ]


def test_the_terms_cover_the_shared_canary_and_the_program_list() -> None:
    """This gate never watches less than the shared-contract canary, nor less than the program names."""
    program = {"discodon", "persona", "universe", "id_universe", "DD_", "music", "research"}

    assert program | set(HOST_NOUNS) == set(TERMS)
