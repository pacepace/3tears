"""rationale-required exemption file parser shared across every domain.

every domain's exemption file (``tests/enforcement/_<domain>_exemptions.txt``)
follows the same shape::

    # rationale: <specific reason>
    relative/path.py:KEY:symbol

``KEY`` is a line number, ``*`` for the whole file, or -- for a domain that opts
in with ``allow_scope_keys`` -- an enclosing qualname with an optional ordinal
(``Class.method#0``). The opt-in matters: for a line-keyed domain a non-numeric
key is a TYPO, and accepting it silently would make an inert exemption out of
what should be an error naming the line.

each entry MUST be preceded by a ``# rationale: ...`` line; rationales
shorter than the threshold or matching well-known blanket phrases are
rejected outright. this is a structural choice: the cost of a per-line
rationale is the only thing that keeps the exemption list from
becoming a silent test-disabler.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from threetears.enforcement.common.ast_helpers import relative_posix_path
from threetears.enforcement.common.violations import Violation

__all__ = [
    "Exemption",
    "ExemptionError",
    "apply_exemptions",
    "parse_exemptions_with_rationale",
    "rationale_defect",
]


class ExemptionError(Exception):
    """raised when an exemption file violates the rationale-required contract."""


@dataclass(frozen=True)
class Exemption:
    """one parsed exemption entry.

    The middle field of an entry is its KEY, and there are three forms. Which one
    a domain uses is a property of what its violations are, not a style choice:

    - a **line number**, for a domain whose violation is a place in a file;
    - ``*``, for a domain where the symbol is exempt anywhere in the file;
    - a **scope key** ``qualname[#occurrence]``, for a domain whose violation is
      an access with its own reason, where a line number would be a fact about
      the file's layout rather than about the thing exempted. An edit above a
      line-keyed entry silently stops it matching; a scope key survives that.

    :ivar file: forward-slash relative path as written in the
        exemption file
    :ivar line: 1-based line number in the source file; 0 means
        "any line in the file" (the ``LINE = '*'`` shorthand used by
        some domains for file-wide exemptions), and 0 is also what a
        scope-keyed entry carries, because it names no line
    :ivar symbol: symbol/table/class name as the walker reports it
    :ivar rationale: text from the ``# rationale: <reason>`` line,
        with the prefix stripped
    :ivar scope: enclosing qualname for a scope-keyed entry (``<module>``
        at module level), or ``None`` for the line and ``*`` forms
    :ivar occurrence: 0-based ordinal among same-file, same-scope,
        same-symbol violations, ordered by line. ``None`` when the entry
        carries no ``#N``, which matches ANY occurrence -- so a domain that
        never needs to tell two accesses apart writes no ordinal at all
    """

    file: str
    line: int
    symbol: str
    rationale: str
    scope: str | None = None
    occurrence: int | None = None


# The symbol may be DOTTED. A domain whose violations are only meaningful when qualified
# reports them that way -- dict-state names `Class.attr`, because the class is what the
# entry is about and a bare attribute name is ambiguous within a module. A bare
# identifier stays valid; this only widens what is accepted.
_ENTRY_RE = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>[^:]+):(?P<symbol>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*$")

_MIN_RATIONALE_LENGTH = 30

_BLANKET_RATIONALE_PHRASES: frozenset[str] = frozenset(
    {
        "internal access",
        "tests need this",
        "temporary",
        "todo",
        "fixme",
        "needed",
        "required",
        "necessary",
    }
)


def parse_exemptions_with_rationale(path: Path, *, allow_scope_keys: bool = False) -> list[Exemption]:
    """parse an exemption file, enforcing the rationale-required contract.

    every non-comment, non-blank line must be of the form
    ``relative/path.py:KEY:symbol``. ``KEY`` is a positive integer (a specific
    line), ``*`` (any line in the file), or -- only when ``allow_scope_keys``
    is set -- a scope key ``qualname[#N]``. Both non-line forms are
    represented as ``line=0`` in the resulting :class:`Exemption`; a scope key
    is told apart from ``*`` by its non-``None`` :attr:`Exemption.scope`.

    each entry MUST be preceded by exactly one ``# rationale:
    <reason>`` line; the rationale text must be at least
    :data:`_MIN_RATIONALE_LENGTH` characters and must not match any
    blanket phrase in :data:`_BLANKET_RATIONALE_PHRASES`. comment lines
    that are NOT rationales are allowed and pass through silently.

    :param path: path to the exemption file
    :ptype path: Path
    :param allow_scope_keys: accept ``qualname[#N]`` as a third key form. OFF by
        default, and that default is the point: for a line-keyed domain a middle
        field that is not a number is a TYPO, and accepting it silently would turn
        ``a.py:1O2:_x`` into an inert exemption instead of an error naming the line
    :ptype allow_scope_keys: bool
    :return: parsed entries in file order
    :rtype: list[Exemption]
    :raises ExemptionError: missing rationale, blanket rationale,
        malformed entry, or unreadable file content
    :raises FileNotFoundError: ``path`` does not exist (caller decides
        whether that is acceptable)
    """
    if not path.exists():
        raise FileNotFoundError(f"exemption file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ExemptionError(f"{path}: not valid utf-8: {exc}") from exc
    except OSError as exc:
        raise ExemptionError(f"cannot read {path}: {exc}") from exc

    entries: list[Exemption] = []
    pending: str | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            stripped = line.lstrip("#").strip()
            if stripped.lower().startswith("rationale:"):
                rationale = stripped[len("rationale:") :].strip()
                _validate_rationale(rationale, path, lineno)
                pending = rationale
            continue
        match = _ENTRY_RE.match(line)
        if match is None:
            raise ExemptionError(f"{path}:{lineno}: malformed entry; expected 'file:line:symbol' triple, got {line!r}")
        if pending is None:
            raise ExemptionError(f"{path}:{lineno}: entry has no preceding '# rationale: ...' line")
        key_field = match.group("line")
        line_int, scope, occurrence = _parse_key(key_field, path, lineno, allow_scope_keys=allow_scope_keys)
        entries.append(
            Exemption(
                file=match.group("file"),
                line=line_int,
                symbol=match.group("symbol"),
                rationale=pending,
                scope=scope,
                occurrence=occurrence,
            )
        )
        pending = None
    return entries


def rationale_defect(rationale: str) -> str | None:
    """say why ``rationale`` is not an acceptable exemption reason, or ``None``.

    The rule, separated from where it is applied, because an exemption does not
    only live in an exemption file. A domain can let a fake carry its rationale
    as a comment on the class -- which is strictly better, since the comment
    moves with the code -- and that route must clear the same bar or the
    "better" route is also the loophole.

    Returns text rather than raising: the file parser turns it into an
    :class:`ExemptionError` at parse time, and an in-source marker turns it into
    an ordinary violation reported alongside the others.

    :param rationale: the reason text, already stripped of its marker prefix
    :ptype rationale: str
    :return: what is wrong with it, or ``None`` when it is acceptable
    :rtype: str | None
    """
    if not rationale:
        return "must be followed by a non-empty reason"
    if len(rationale) < _MIN_RATIONALE_LENGTH:
        return f"must be at least {_MIN_RATIONALE_LENGTH} characters; got {len(rationale)}: {rationale!r}"
    lower = rationale.lower()
    for phrase in _BLANKET_RATIONALE_PHRASES:
        if lower == phrase or lower.startswith(phrase + " ") or lower.startswith(phrase + "."):
            return f"{rationale!r} matches blanket phrase {phrase!r}; rationales must be specific"
    return None


#: A scope key's optional occurrence suffix: ``qualname#3``.
#:
#: ``#`` cannot appear in a Python qualname, so it partitions the field
#: unambiguously and needs no escaping.
_SCOPE_OCCURRENCE = re.compile(r"^(?P<scope>[^#]+)#(?P<occurrence>\d+)$")


def _parse_key(field: str, path: Path, lineno: int, *, allow_scope_keys: bool) -> tuple[int, str | None, int | None]:
    """resolve an entry's middle field into ``(line, scope, occurrence)``.

    Three forms, distinguished without ambiguity: ``*`` is file-wide, an integer
    is a line, anything else is a scope key. A scope can never be confused for a
    line because a line is entirely digits and a qualname cannot be.

    :param field: the raw middle field
    :ptype field: str
    :param path: exemption file path, for error context
    :ptype path: Path
    :param lineno: line within the exemption file, for error context
    :ptype lineno: int
    :return: ``(line, scope, occurrence)``; ``line`` is 0 for both non-line forms
    :rtype: tuple[int, str | None, int | None]
    :raises ExemptionError: a non-positive line, or a malformed occurrence suffix
    """
    if field == "*":
        return (0, None, None)
    if field.isdigit():
        line_int = int(field)
        if line_int < 1:
            raise ExemptionError(f"{path}:{lineno}: line number must be positive, got {line_int}")
        return (line_int, None, None)
    if not allow_scope_keys:
        raise ExemptionError(
            f"{path}:{lineno}: line number must be an integer or '*', got {field!r}. "
            f"(Scope keys are accepted only for a domain that opts in with allow_scope_keys.)"
        )

    match = _SCOPE_OCCURRENCE.match(field)
    if match is not None:
        return (0, match.group("scope"), int(match.group("occurrence")))
    if "#" in field:
        raise ExemptionError(
            f"{path}:{lineno}: malformed scope key {field!r}; expected 'qualname' or 'qualname#N' "
            f"with N a non-negative integer"
        )
    return (0, field, None)


def _validate_rationale(rationale: str, path: Path, lineno: int) -> None:
    """raise :class:`ExemptionError` if ``rationale`` is empty / blanket / too short.

    :param rationale: text of the rationale (already prefix-stripped)
    :ptype rationale: str
    :param path: exemption file path (for error context)
    :ptype path: Path
    :param lineno: line number in the exemption file (for error context)
    :ptype lineno: int
    :raises ExemptionError: rationale fails any of the contract checks
    """
    defect = rationale_defect(rationale)
    if defect is not None:
        raise ExemptionError(f"{path}:{lineno}: rationale {defect}")


def apply_exemptions(
    violations: list[Violation],
    exemptions: list[Exemption],
    repo_root: Path,
    scope_of: Callable[[Path, int], str] | None = None,
) -> list[Violation]:
    """filter ``violations`` against ``exemptions``, preserving order.

    a violation is exempted when it matches an entry under that entry's own key
    form:

    - **line** -- its ``(relative path, line, symbol)`` triple matches;
    - **``*``** (``line=0``, ``scope=None``) -- the symbol matches anywhere in
      that file;
    - **scope** (``scope`` set) -- the symbol matches inside that qualname. With
      ``occurrence`` set it matches ONLY that ordinal; written without one
      (``qualname``, no ``#N``) it matches EVERY occurrence in the scope, which
      is the wider of the two and easy to reach for by accident. Requires
      ``scope_of``; without one a scope-keyed entry matches nothing, so a domain
      cannot be changed by this form without opting in.

    returns a new list — does not mutate ``violations``.

    :param violations: walker output to filter
    :ptype violations: list[Violation]
    :param exemptions: parsed exemption entries
    :ptype exemptions: list[Exemption]
    :param repo_root: repo root for relative-path comparison
    :ptype repo_root: Path
    :param scope_of: resolves ``(source path, line)`` to the enclosing qualname,
        enabling scope-keyed matching. ``None`` leaves those entries inert.

        **Occurrence ordinals are numbered over the VIOLATIONS passed here**, by
        line within each ``(file, scope, symbol)`` group. A caller whose ledger
        numbers a different population -- ``underscore_access`` writes its ledger
        from an AST scan, not from walker output -- must ensure the two agree, or
        an entry covers a different access than the one it was written for. That
        domain asserts the agreement in
        ``tests/enforcement/test_underscore_exemptions_resolve.py``.
    :ptype scope_of: Callable[[Path, int], str] | None
    :return: violations not matched by any exemption, in original order
    :rtype: list[Violation]
    """
    file_wide: dict[str, set[str]] = {}
    line_specific: set[tuple[str, int, str]] = set()
    scoped: dict[tuple[str, str, str], set[int | None]] = {}
    for entry in exemptions:
        # Order matters: a scope-keyed entry ALSO carries line 0, because it names
        # no line. Testing the line first would drop it into the file-wide bucket
        # and exempt its symbol everywhere in the file -- silently wider than what
        # was written, in a mechanism whose whole job is to be narrow.
        if entry.scope is not None:
            scoped.setdefault((entry.file, entry.scope, entry.symbol), set()).add(entry.occurrence)
        elif entry.line == 0:
            file_wide.setdefault(entry.file, set()).add(entry.symbol)
        else:
            line_specific.add((entry.file, entry.line, entry.symbol))

    occurrences = _occurrences(violations, repo_root, scope_of) if scoped and scope_of is not None else {}

    result: list[Violation] = []
    for violation in violations:
        rel = relative_posix_path(violation.file, repo_root)
        if (rel, violation.line, violation.symbol) in line_specific:
            continue
        wide_symbols = file_wide.get(rel)
        if wide_symbols is not None and violation.symbol in wide_symbols:
            continue
        resolved = occurrences.get(id(violation))
        if resolved is not None:
            scope, occurrence = resolved
            allowed = scoped.get((rel, scope, violation.symbol))
            # ``None`` in the set is an entry written without an ordinal, which
            # covers every occurrence -- a domain that never needs to tell two
            # accesses apart should not have to number them.
            if allowed is not None and (None in allowed or occurrence in allowed):
                continue
        result.append(violation)
    return result


def _occurrences(
    violations: list[Violation],
    repo_root: Path,
    scope_of: Callable[[Path, int], str],
) -> dict[int, tuple[str, int]]:
    """number each violation within its ``(file, scope, symbol)`` group.

    Ordered by line, so the ordinals match what a regeneration walking the same
    violations in the same order would record. Keyed by ``id`` rather than by the
    violation itself because :class:`Violation` is not hashable-by-value in a way
    that distinguishes two accesses on the same line.

    :param violations: the unfiltered violation list
    :ptype violations: list[Violation]
    :param repo_root: repo root for relative-path comparison
    :ptype repo_root: Path
    :param scope_of: resolves a source path and line to an enclosing qualname
    :ptype scope_of: Callable[[Path, int], str]
    :return: ``id(violation)`` -> ``(scope, occurrence)``
    :rtype: dict[int, tuple[str, int]]
    """
    resolved: dict[int, tuple[str, str]] = {}
    for violation in violations:
        rel = relative_posix_path(violation.file, repo_root)
        resolved[id(violation)] = (rel, scope_of(violation.file, violation.line))

    counters: dict[tuple[str, str, str], int] = {}
    numbered: dict[int, tuple[str, int]] = {}
    for violation in sorted(violations, key=lambda v: (str(v.file), v.line)):
        rel, scope = resolved[id(violation)]
        key = (rel, scope, violation.symbol)
        numbered[id(violation)] = (scope, counters.get(key, 0))
        counters[key] = counters.get(key, 0) + 1
    return numbered
