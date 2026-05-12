"""rationale-required exemption file parser shared across every domain.

every domain's exemption file (``tests/enforcement/_<domain>_exemptions.txt``)
follows the same shape::

    # rationale: <specific reason>
    relative/path.py:LINE:symbol

each entry MUST be preceded by a ``# rationale: ...`` line; rationales
shorter than the threshold or matching well-known blanket phrases are
rejected outright. this is a structural choice: the cost of a per-line
rationale is the only thing that keeps the exemption list from
becoming a silent test-disabler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from threetears.enforcement.common.ast_helpers import relative_posix_path
from threetears.enforcement.common.violations import Violation

__all__ = [
    "Exemption",
    "ExemptionError",
    "apply_exemptions",
    "parse_exemptions_with_rationale",
]


class ExemptionError(Exception):
    """raised when an exemption file violates the rationale-required contract."""


@dataclass(frozen=True)
class Exemption:
    """one parsed exemption entry.

    :ivar file: forward-slash relative path as written in the
        exemption file
    :ivar line: 1-based line number in the source file; 0 means
        "any line in the file" (the ``LINE = '*'`` shorthand used by
        some domains for file-wide exemptions)
    :ivar symbol: symbol/table/class name as the walker reports it
    :ivar rationale: text from the ``# rationale: <reason>`` line,
        with the prefix stripped
    """

    file: str
    line: int
    symbol: str
    rationale: str


_ENTRY_RE = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>[^:]+):(?P<symbol>[A-Za-z_][A-Za-z_0-9]*)\s*$")

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


def parse_exemptions_with_rationale(path: Path) -> list[Exemption]:
    """parse an exemption file, enforcing the rationale-required contract.

    every non-comment, non-blank line must be of the form
    ``relative/path.py:LINE:symbol``. ``LINE`` is either a positive
    integer (a specific line) or ``*`` (any line in the file —
    represented as ``line=0`` in the resulting :class:`Exemption`).

    each entry MUST be preceded by exactly one ``# rationale:
    <reason>`` line; the rationale text must be at least
    :data:`_MIN_RATIONALE_LENGTH` characters and must not match any
    blanket phrase in :data:`_BLANKET_RATIONALE_PHRASES`. comment lines
    that are NOT rationales are allowed and pass through silently.

    :param path: path to the exemption file
    :ptype path: Path
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
        line_field = match.group("line")
        if line_field == "*":
            line_int = 0
        else:
            try:
                line_int = int(line_field)
            except ValueError as exc:
                raise ExemptionError(
                    f"{path}:{lineno}: line number must be an integer or '*', got {line_field!r}"
                ) from exc
            if line_int < 1:
                raise ExemptionError(f"{path}:{lineno}: line number must be positive, got {line_int}")
        entries.append(
            Exemption(
                file=match.group("file"),
                line=line_int,
                symbol=match.group("symbol"),
                rationale=pending,
            )
        )
        pending = None
    return entries


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
    if not rationale:
        raise ExemptionError(f"{path}:{lineno}: '# rationale:' must be followed by a non-empty reason")
    if len(rationale) < _MIN_RATIONALE_LENGTH:
        raise ExemptionError(
            f"{path}:{lineno}: rationale must be at least "
            f"{_MIN_RATIONALE_LENGTH} characters; got {len(rationale)}: "
            f"{rationale!r}"
        )
    lower = rationale.lower()
    for phrase in _BLANKET_RATIONALE_PHRASES:
        if lower == phrase or lower.startswith(phrase + " ") or lower.startswith(phrase + "."):
            raise ExemptionError(
                f"{path}:{lineno}: rationale {rationale!r} matches blanket "
                f"phrase {phrase!r}; rationales must be specific"
            )


def apply_exemptions(
    violations: list[Violation],
    exemptions: list[Exemption],
    repo_root: Path,
) -> list[Violation]:
    """filter ``violations`` against ``exemptions``, preserving order.

    a violation is exempted iff its
    ``(relative_posix_path(file, repo_root), line, symbol)`` triple
    matches an exemption. an exemption with ``line=0`` (the ``LINE='*'``
    shorthand) matches any line in that file.

    returns a new list — does not mutate ``violations``.

    :param violations: walker output to filter
    :ptype violations: list[Violation]
    :param exemptions: parsed exemption entries
    :ptype exemptions: list[Exemption]
    :param repo_root: repo root for relative-path comparison
    :ptype repo_root: Path
    :return: violations not matched by any exemption, in original order
    :rtype: list[Violation]
    """
    file_wide: dict[str, set[str]] = {}
    line_specific: set[tuple[str, int, str]] = set()
    for entry in exemptions:
        if entry.line == 0:
            file_wide.setdefault(entry.file, set()).add(entry.symbol)
        else:
            line_specific.add((entry.file, entry.line, entry.symbol))

    result: list[Violation] = []
    for violation in violations:
        rel = relative_posix_path(violation.file, repo_root)
        if (rel, violation.line, violation.symbol) in line_specific:
            continue
        wide_symbols = file_wide.get(rel)
        if wide_symbols is not None and violation.symbol in wide_symbols:
            continue
        result.append(violation)
    return result
