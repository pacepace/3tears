"""the scanner: every NATS KV bucket is memory-backed.

``storage`` is an ordinary ``str`` with a ``"memory"`` default, so a literal ``"file"``
type-checks perfectly and nothing but a reader notices. And a bucket's storage is chosen at
CREATE and never reconciled, so the mistake outlives every later deploy that would have corrected
it: a wrong value is not a bug you fix by shipping a fix, it is one you fix by deleting live
state. That asymmetry is why this is a gate rather than a convention.

Only a literal is reported. A computed value (``storage=cfg.storage``) is left alone
deliberately: the walker cannot evaluate it, and guessing would either fail a legitimate caller
or report a location nobody can act on. The literal form is what every occurrence in this estate
has used, and it is the form somebody reaches for when they want durability in a hurry.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DURABLE_STORAGE",
    "KV_OPENING_CALLS",
    "STORAGE_KEYWORD",
    "FileBackedKvCall",
    "file_backed_kv_calls",
    "scan_for_file_backed_kv",
]

#: The storage value this gate refuses. ``"memory"`` is the contract and the ``kv_bucket``
#: default, so an absent keyword is always compliant.
DURABLE_STORAGE = "file"

#: The keyword whose value decides a bucket's tier for its whole lifetime.
STORAGE_KEYWORD = "storage"

#: Call names that open a KV bucket. Matched on the ATTRIBUTE or function name rather than a
#: resolved symbol: the walker sees one file at a time and cannot know what ``nc`` is bound to,
#: and every real opener in this estate spells one of these at the call site.
KV_OPENING_CALLS = frozenset({"kv_bucket", "open_kv_stream", "build_kv_stream_config"})


@dataclass(frozen=True)
class FileBackedKvCall:
    """one call asking a KV bucket to be file-backed.

    :ivar path: the module it is in
    :ivar lineno: the call's line
    :ivar callee: which opener was called
    """

    path: Path
    lineno: int
    callee: str


def _called_name(node: ast.Call) -> str | None:
    """return the callee's own name, attribute or bare.

    :param node: the call being inspected
    :ptype node: ast.Call
    :return: the trailing name, or ``None`` for a call shape with neither
    :rtype: str | None
    """
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def file_backed_kv_calls(path: Path) -> list[tuple[int, str]]:
    """every KV-opening call in one module that asks for file storage.

    :param path: the module to scan
    :ptype path: Path
    :return: ``(line number, callee name)`` for each offending call
    :rtype: list[tuple[int, str]]
    """
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except OSError, SyntaxError:
        return []

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name not in KV_OPENING_CALLS:
            continue
        for keyword in node.keywords:
            if keyword.arg != STORAGE_KEYWORD:
                continue
            value = keyword.value
            if isinstance(value, ast.Constant) and value.value == DURABLE_STORAGE:
                found.append((node.lineno, name))
    return found


def scan_for_file_backed_kv(paths: list[Path]) -> list[FileBackedKvCall]:
    """scan every given module and return each file-backed KV open.

    :param paths: the modules to scan
    :ptype paths: list[Path]
    :return: one entry per offending call, in path then line order
    :rtype: list[FileBackedKvCall]
    """
    out: list[FileBackedKvCall] = []
    for path in sorted(paths):
        out.extend(
            FileBackedKvCall(path=path, lineno=lineno, callee=callee) for lineno, callee in file_backed_kv_calls(path)
        )
    return out
