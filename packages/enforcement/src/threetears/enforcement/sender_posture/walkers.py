"""walker and allowlist check for the sender-posture enforcement domain.

a site is a call whose final callee name is in :attr:`SenderPostureConfig.site_callees`. its
marker is a ``# sender-posture:`` comment anywhere from the contiguous comment lines directly
above the enclosing statement to the last line of the call itself. comments are read with
:mod:`tokenize`, so text inside a string literal never declares anything.

marker grammar::

    # sender-posture: <posture> [open <ROW>[,<ROW>...]] [-- <free text>]

where a row is an audit row id such as ``H5`` or ``S4``.

allowlist grammar, one line per (module, row)::

    <repo-relative path> :: <ROW> :: <sites> :: <chunk> :: <why it is still open>

``<sites>`` is how many sites in that module declare the row open. it is pinned so a NEW receiver
cannot ride an existing line: adding one changes the count, and the count is reviewed.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from threetears.enforcement.common import iter_python_files, parse_python_path, relative_posix_path
from threetears.enforcement.sender_posture.config import POSTURES, SenderPostureConfig

__all__ = [
    "SubscribeSite",
    "find_subscribe_sites",
    "sender_posture_findings",
]

#: a comment that names the gate at all; one that then fails :data:`_MARKER` is malformed.
_MARKER_WORD: Final[str] = "sender-posture"

_ROW: Final[str] = r"[A-Z]+[0-9]+"

_MARKER: Final[re.Pattern[str]] = re.compile(
    r"^#\s*sender-posture:\s*(?P<posture>[a-z][a-z-]*)"
    rf"(?:\s+open\s+(?P<rows>{_ROW}(?:\s*,\s*{_ROW})*))?"
    r"(?:\s+--\s+\S.*)?\s*$"
)

_ROW_ONLY: Final[re.Pattern[str]] = re.compile(rf"^{_ROW}$")

_ALLOWLIST_FIELDS: Final[int] = 5


@dataclass(frozen=True)
class SubscribeSite:
    """one receiving site and what its marker declares.

    :ivar path: repo-relative posix path of the module
    :ivar line: line of the call
    :ivar callee: final name of the called function or method
    :ivar function: qualified name of the enclosing function or class, or ``<module>``
    :ivar posture: declared posture, or none when undeclared or unknown
    :ivar open_rows: audit rows the marker declares still open at this site
    """

    path: str
    line: int
    callee: str
    function: str
    posture: str | None
    open_rows: tuple[str, ...]


@dataclass(frozen=True)
class _Marker:
    """one ``sender-posture`` comment, parsed.

    :ivar line: line the comment sits on
    :ivar posture: posture it names, or none when malformed
    :ivar open_rows: rows it declares open
    """

    line: int
    posture: str | None
    open_rows: tuple[str, ...]


@dataclass
class _Scan:
    """accumulated result of walking every source root.

    :ivar sites: every site found, in walk order
    :ivar findings: every finding raised by the walk itself
    """

    sites: list[SubscribeSite] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


def _comment_lines(source: str) -> dict[int, str]:
    """
    map line number onto the real comment on it, read with tokenize.

    :param source: module source
    :ptype source: str
    :return: line number onto comment text, starting at its hash
    :rtype: dict[int, str]
    """
    comments: dict[int, str] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comments[token.start[0]] = token.string
    return comments


def _parse_marker(line: int, comment: str) -> _Marker:
    """
    parse one comment naming the gate.

    :param line: line the comment sits on
    :ptype line: int
    :param comment: comment text, starting at its hash
    :ptype comment: str
    :return: parsed marker; posture none when the comment does not match the grammar
    :rtype: _Marker
    """
    match = _MARKER.match(comment.strip())
    marker = _Marker(line=line, posture=None, open_rows=())
    if match is not None:
        rows = match.group("rows")
        open_rows = tuple(row.strip() for row in rows.split(",")) if rows else ()
        marker = _Marker(line=line, posture=match.group("posture"), open_rows=open_rows)
    return marker


def _callee_name(call: ast.Call) -> str | None:
    """
    answer final name of called function or method.

    :param call: call node
    :ptype call: ast.Call
    :return: attribute or bare name, or none for any other callee shape
    :rtype: str | None
    """
    name: str | None = None
    if isinstance(call.func, ast.Attribute):
        name = call.func.attr
    elif isinstance(call.func, ast.Name):
        name = call.func.id
    return name


_SCOPES: Final[tuple[type[ast.AST], ...]] = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _calls_with_statements(tree: ast.Module) -> list[tuple[ast.Call, ast.stmt, str]]:
    """
    pair every call with the innermost statement enclosing it and its enclosing scope's name.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: call, its enclosing statement, and the dotted name of its enclosing function or class
        (``<module>`` at top level), in walk order
    :rtype: list[tuple[ast.Call, ast.stmt, str]]
    """
    pairs: list[tuple[ast.Call, ast.stmt, str]] = []
    stack: list[tuple[ast.AST, ast.stmt | None, tuple[str, ...]]] = [(tree, None, ())]
    while stack:
        node, statement, scope = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call) and statement is not None:
                pairs.append((child, statement, ".".join(scope) or "<module>"))
            child_scope = (*scope, child.name) if isinstance(child, _SCOPES) else scope  # type: ignore[attr-defined]
            stack.append((child, child if isinstance(child, ast.stmt) else statement, child_scope))
    pairs.sort(key=lambda pair: (pair[0].lineno, pair[0].col_offset))
    return pairs


def _region(statement: ast.stmt, call: ast.Call, lines: list[str]) -> range:
    """
    answer lines a site's marker may sit on.

    the region runs from the contiguous comment lines directly above the enclosing statement to
    the last line of the call. a blank line or code above ends it.

    :param statement: innermost statement enclosing the call
    :ptype statement: ast.stmt
    :param call: the site's call
    :ptype call: ast.Call
    :param lines: module source split into lines
    :ptype lines: list[str]
    :return: one-based line range, inclusive of both ends
    :rtype: range
    """
    start = statement.lineno
    while start > 1 and lines[start - 2].lstrip().startswith("#"):
        start -= 1
    end = call.end_lineno if call.end_lineno is not None else call.lineno
    return range(start, end + 1)


def _site_from_markers(
    relative: str,
    call: ast.Call,
    callee: str,
    function: str,
    markers: list[_Marker],
    config: SenderPostureConfig,
    findings: list[str],
) -> SubscribeSite:
    """
    judge one site against the markers in its region.

    :param relative: repo-relative path of module
    :ptype relative: str
    :param call: the site's call
    :ptype call: ast.Call
    :param callee: final callee name
    :ptype callee: str
    :param function: dotted name of the enclosing function or class
    :ptype function: str
    :param markers: well-formed markers inside the site's region
    :ptype markers: list[_Marker]
    :param config: gate configuration
    :ptype config: SenderPostureConfig
    :param findings: accumulator, mutated in place
    :ptype findings: list[str]
    :return: the site, with its posture when one valid marker declares it
    :rtype: SubscribeSite
    """
    where = f"{relative}:{call.lineno}"
    posture: str | None = None
    open_rows: tuple[str, ...] = ()
    if not markers:
        findings.append(
            f"{where}: {callee}(...) declares no sender posture; add '# sender-posture: <posture>' "
            f"naming one of {sorted(POSTURES)}"
        )
    elif len({(marker.posture, marker.open_rows) for marker in markers}) > 1:
        findings.append(f"{where}: {callee}(...) carries more than one sender-posture marker")
    elif markers[0].posture not in POSTURES:
        findings.append(f"{where}: unknown sender posture {markers[0].posture!r}; use one of {sorted(POSTURES)}")
    else:
        posture = markers[0].posture
        open_rows = markers[0].open_rows
        if posture == "unauthenticated" and not open_rows:
            findings.append(
                f"{where}: unauthenticated must name the open audit row it is tracked under "
                "('# sender-posture: unauthenticated open <ROW>')"
            )
        if posture == "primitive" and f"{relative}::{function}" not in config.primitive_functions:
            findings.append(
                f"{where}: primitive is allowed only in the configured primitive functions, and "
                f"{relative}::{function} is not one; declare what this receiver checks instead"
            )
    return SubscribeSite(
        path=relative, line=call.lineno, callee=callee, function=function, posture=posture, open_rows=open_rows
    )


def _scan_module(path: Path, config: SenderPostureConfig, scan: _Scan) -> None:
    """
    walk one module, appending its sites and findings.

    :param path: module to walk
    :ptype path: Path
    :param config: gate configuration
    :ptype config: SenderPostureConfig
    :param scan: accumulator, mutated in place
    :ptype scan: _Scan
    """
    relative = relative_posix_path(path, config.repo_root)
    try:
        source = path.read_text(encoding="utf-8")
        tree = parse_python_path(path)
        comments = _comment_lines(source)
    except (UnicodeDecodeError, SyntaxError, tokenize.TokenError) as exc:
        scan.findings.append(f"{relative}: could not be parsed ({exc}); the gate cannot see its sites")
        return
    lines = source.splitlines()
    markers: dict[int, _Marker] = {}
    for line, comment in comments.items():
        if _MARKER_WORD in comment:
            marker = _parse_marker(line, comment)
            if marker.posture is None:
                scan.findings.append(
                    f"{relative}:{line}: malformed sender-posture marker {comment.strip()!r}; "
                    "expected '# sender-posture: <posture> [open <ROW>[,<ROW>]] [-- <text>]'"
                )
            else:
                markers[line] = marker
    claimed: set[int] = set()
    for call, statement, function in _calls_with_statements(tree):
        callee = _callee_name(call)
        if callee is None or callee not in config.site_callees:
            continue
        region = _region(statement, call, lines)
        in_region = [markers[line] for line in region if line in markers]
        claimed.update(marker.line for marker in in_region)
        scan.sites.append(_site_from_markers(relative, call, callee, function, in_region, config, scan.findings))
    for line in sorted(set(markers) - claimed):
        scan.findings.append(f"{relative}:{line}: sender-posture marker sits on no subscribe site; remove it")


def _scan(config: SenderPostureConfig) -> _Scan:
    """
    walk every configured source root.

    :param config: gate configuration
    :ptype config: SenderPostureConfig
    :return: sites and walk findings
    :rtype: _Scan
    """
    scan = _Scan()
    for prefix in config.excluded_prefixes:
        if not prefix.endswith("/"):
            scan.findings.append(f"excluded prefix {prefix!r} must end with '/', so it names a directory")
    matched: set[str] = set()
    for root in config.src_roots:
        for path in iter_python_files(root):
            relative = relative_posix_path(path, config.repo_root)
            excluded_by = [prefix for prefix in config.excluded_prefixes if relative.startswith(prefix)]
            matched.update(excluded_by)
            if not excluded_by:
                _scan_module(path, config, scan)
    for prefix in config.excluded_prefixes:
        if prefix not in matched:
            scan.findings.append(f"excluded prefix {prefix!r} matches no module; remove it")
    return scan


def _allowlist_findings(config: SenderPostureConfig, sites: list[SubscribeSite]) -> list[str]:
    """
    reconcile the allowlist against the open rows the sites declare, in both directions.

    :param config: gate configuration
    :ptype config: SenderPostureConfig
    :param sites: every site found
    :ptype sites: list[SubscribeSite]
    :return: findings for malformed, duplicate, unlisted and stale entries
    :rtype: list[str]
    """
    findings: list[str] = []
    listed_name = relative_posix_path(config.allowlist_path, config.repo_root)
    if not config.allowlist_path.is_file():
        findings.append(f"{listed_name}: sender-posture allowlist not found")
        return findings
    listed: dict[tuple[str, str], int] = {}
    pinned: dict[tuple[str, str], int] = {}
    text = config.allowlist_path.read_text(encoding="utf-8")
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = [part.strip() for part in stripped.split("::")]
        if (
            len(fields) != _ALLOWLIST_FIELDS
            or not all(fields)
            or not _ROW_ONLY.match(fields[1])
            or not fields[2].isdigit()
            or int(fields[2]) < 1
        ):
            findings.append(
                f"{listed_name}:{number}: malformed allowlist line {stripped!r}; "
                "expected '<path> :: <ROW> :: <sites> :: <chunk> :: <why it is still open>'"
            )
            continue
        key = (fields[0], fields[1])
        if key in listed:
            findings.append(f"{listed_name}:{number}: {key[0]} :: {key[1]} is listed twice (first at {listed[key]})")
            continue
        listed[key] = number
        pinned[key] = int(fields[2])
    declared: dict[tuple[str, str], SubscribeSite] = {}
    counts: dict[tuple[str, str], int] = {}
    for site in sites:
        for row in site.open_rows:
            declared.setdefault((site.path, row), site)
            counts[(site.path, row)] = counts.get((site.path, row), 0) + 1
    for key, site in sorted(declared.items()):
        if key not in listed:
            findings.append(
                f"{site.path}:{site.line}: open row {key[1]} is not in the allowlist {listed_name}; "
                "add its line, or fix the site and drop the row"
            )
    for key, number in sorted(listed.items(), key=lambda item: item[1]):
        if key in declared and counts[key] != pinned[key]:
            findings.append(
                f"{listed_name}:{number}: {key[0]} :: {key[1]} pins {pinned[key]} site(s), but {counts[key]} "
                "declare that row open; a new receiver needs its own reviewed count, a fixed one drops it"
            )
        if key not in declared:
            findings.append(
                f"{listed_name}:{number}: {key[0]} :: {key[1]} is listed, but no site in it declares that "
                "row open; delete the line"
            )
    return findings


def find_subscribe_sites(config: SenderPostureConfig) -> list[SubscribeSite]:
    """
    answer every receiving site under the configured source roots.

    :param config: gate configuration
    :ptype config: SenderPostureConfig
    :return: sites in walk order, declared or not
    :rtype: list[SubscribeSite]
    """
    return _scan(config).sites


def sender_posture_findings(config: SenderPostureConfig) -> list[str]:
    """
    answer every way the repo breaks the sender-posture contract.

    :param config: gate configuration
    :ptype config: SenderPostureConfig
    :return: findings; empty when every site declares a valid posture and the allowlist matches
        the open rows exactly
    :rtype: list[str]
    """
    scan = _scan(config)
    return scan.findings + _allowlist_findings(config, scan.sites)
