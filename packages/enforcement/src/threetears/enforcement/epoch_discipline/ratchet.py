"""the shrinking ledgers of the epoch-discipline domain, and the findings each rule's gate reports.

Allowlist grammar, one file per rule, one line per (module, function, symbol)::

    <repo-relative path> :: <qualified function> :: <symbol> :: <count> :: <fix that removes it>

Waiver grammar, one file for every rule::

    <rule> :: <repo-relative path> :: <qualified function> :: <symbol> :: <count> :: <reason>

An allowlist line is a violation still owed a fix; a waiver is a site whose shape the detector
matches but which is not what the rule forbids (an expiry that is the data's own lifetime, a
cache of client handles rather than rows). Both pin the site's hit count exactly: a hit beyond
it is a new violation, a count above it is a stale line. Lines starting ``#`` and blank lines
are ignored.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from threetears.enforcement.epoch_discipline.config import (
    RULE_ADVICE,
    RULE_EPOCH_BYPASS,
    RULES,
    EpochDisciplineConfig,
    allowlist_path_for,
)
from threetears.enforcement.epoch_discipline.walkers import EpochHit, EpochScan, scan_epoch_discipline

__all__ = [
    "epoch_discipline_findings",
    "epoch_violation_counts",
]

_SEPARATOR: Final[str] = " :: "
_MIN_FIX: Final[int] = 20
_MIN_REASON: Final[int] = 30


@dataclass(frozen=True)
class _LedgerLine:
    """one allowlist or waiver line.

    :ivar file_line: line number in its ledger file
    :ivar rule: the rule it answers to
    :ivar key: ``(path, function, symbol)``
    :ivar count: hits it pins
    :ivar text: the fix (allowlist) or the reason (waiver)
    """

    file_line: int
    rule: str
    key: tuple[str, str, str]
    count: int
    text: str


def _parse_ledger(path: Path, *, rule: str | None) -> tuple[list[_LedgerLine], list[str]]:
    """read one ledger: an allowlist when ``rule`` is given, the waiver file when it is not.

    :param path: the ledger file; a missing file reads as empty
    :ptype path: Path
    :param rule: the allowlist's rule, or ``None`` for the waiver file
    :ptype rule: str | None
    :return: the lines, and every malformed line as a finding
    :rtype: tuple[list[_LedgerLine], list[str]]
    """
    lines: list[_LedgerLine] = []
    defects: list[str] = []
    if not path.is_file():
        return lines, defects
    fields = 5 if rule is not None else 6
    seen: set[tuple[str, tuple[str, str, str]]] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped.split(_SEPARATOR)]
        where = f"{path.name}:{number}"
        if len(parts) != fields:
            defects.append(f"{where}: expected {fields} fields separated by '{_SEPARATOR.strip()}', got {len(parts)}")
            continue
        line_rule = rule if rule is not None else parts.pop(0)
        path_part, function, symbol, count_text, text = parts
        minimum = _MIN_FIX if rule is not None else _MIN_REASON
        if line_rule not in RULES:
            defects.append(f"{where}: unknown rule {line_rule!r}; rules are {', '.join(RULES)}")
        elif not count_text.isdigit() or int(count_text) < 1:
            defects.append(f"{where}: count {count_text!r} must be a positive integer; delete a line at 0")
        elif len(text) < minimum:
            kind = "fix" if rule is not None else "reason"
            defects.append(f"{where}: the {kind} must say specifically what it is ({minimum}+ characters)")
        elif (line_rule, (path_part, function, symbol)) in seen:
            defects.append(f"{where}: {path_part} :: {function} :: {symbol} is listed twice")
        else:
            seen.add((line_rule, (path_part, function, symbol)))
            lines.append(_LedgerLine(number, line_rule, (path_part, function, symbol), int(count_text), text))
    return lines, defects


def _sanctioned(hit: EpochHit, primitives: frozenset[str]) -> bool:
    """whether a hit sits inside a function that implements the rule's operation.

    :param hit: the hit
    :ptype hit: EpochHit
    :param primitives: ``<path>::<qualified name>`` entries
    :ptype primitives: frozenset[str]
    :return: whether one covers it
    :rtype: bool
    """
    return any(
        hit.path == path and (hit.function == qual or hit.function.startswith(f"{qual}."))
        for path, _, qual in (entry.partition("::") for entry in primitives)
    )


def _unsanctioned_hits(config: EpochDisciplineConfig, rule: str) -> tuple[list[EpochHit], list[str]]:
    """the rule's hits outside its primitives, and the walk's own defects.

    :param config: the repo's config
    :ptype config: EpochDisciplineConfig
    :param rule: the rule
    :ptype rule: str
    :return: the hits, and findings about the walk and the primitive list
    :rtype: tuple[list[EpochHit], list[str]]
    """
    scan: EpochScan = scan_epoch_discipline(config.repo_root, config.src_roots, config.excluded_prefixes)
    defects: list[str] = []
    if scan.files_scanned < config.minimum_files_scanned:
        defects.append(
            f"the walk reached {scan.files_scanned} modules, under the floor of {config.minimum_files_scanned}: "
            f"check src_roots, since a walk that reads nothing reports nothing"
        )
    defects.extend(f"excluded prefix {prefix!r} matches no module; delete it" for prefix in scan.unmatched_exclusions)
    primitives = config.primitives.get(rule, frozenset())
    defects.extend(
        f"primitive {entry!r} of {rule} names no function or class the walk found; delete or correct it"
        for entry in sorted(primitives)
        if entry not in scan.definitions
    )
    hits = [
        hit
        for hit in scan.hits
        if hit.rule == rule
        and not _sanctioned(hit, primitives)
        and not (rule == RULE_EPOCH_BYPASS and hit.function.rsplit(".", 1)[-1] in config.write_path_hooks)
    ]
    return hits, defects


def epoch_discipline_findings(config: EpochDisciplineConfig, rule: str) -> list[str]:
    """every finding of one rule: new violations, stale ledger lines, and walk defects.

    :param config: the repo's config
    :ptype config: EpochDisciplineConfig
    :param rule: one of :data:`~threetears.enforcement.epoch_discipline.config.RULES`
    :ptype rule: str
    :return: findings, empty when the rule holds
    :rtype: list[str]
    """
    hits, findings = _unsanctioned_hits(config, rule)
    allowlist_path = allowlist_path_for(config.ledger_dir, rule)
    allowed, defects = _parse_ledger(allowlist_path, rule=rule)
    findings.extend(defects)
    every_waiver, waiver_defects = _parse_ledger(config.ledger_dir / config.waivers_name, rule=None)
    findings.extend(waiver_defects)
    waived = [line for line in every_waiver if line.rule == rule]
    by_key: dict[tuple[str, str, str], list[EpochHit]] = defaultdict(list)
    for hit in hits:
        by_key[hit.key].append(hit)
    pinned: Counter[tuple[str, str, str]] = Counter()
    for ledger, lines in ((allowlist_path.name, allowed), (config.waivers_name, waived)):
        for line in lines:
            if line.key in pinned:
                findings.append(f"{ledger}:{line.file_line}: {' :: '.join(line.key)} is both allowlisted and waived")
            pinned[line.key] += line.count
            have = len(by_key.get(line.key, []))
            if have < line.count:
                findings.append(
                    f"{ledger}:{line.file_line}: {' :: '.join(line.key)} pins {line.count} but the code has {have} -- "
                    + ("delete the line" if have == 0 else f"lower the count to {have}")
                    + ": the list only shrinks"
                )
    for key, sites in sorted(by_key.items()):
        if len(sites) > pinned[key]:
            for site in sites:
                findings.append(
                    f"{site.path}:{site.line} in {site.function}: {site.symbol} -- NEW {rule} violation "
                    f"({len(sites)} here, {pinned[key]} allowed): {RULE_ADVICE[rule]}"
                )
    return findings


def epoch_violation_counts(config: EpochDisciplineConfig) -> dict[str, int]:
    """how many hits each rule has outside its primitives, ledgers ignored.

    For a repo measuring itself before it adopts the gates: each hit becomes an allowlist line
    ``<path> :: <function> :: <symbol> :: <count> :: <fix>`` once it is given its fix.

    :param config: the repo's config; its ledgers are not read
    :ptype config: EpochDisciplineConfig
    :return: rule onto hit count
    :rtype: dict[str, int]
    """
    return {rule: len(_unsanctioned_hits(config, rule)[0]) for rule in RULES}
