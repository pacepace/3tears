"""find every KV bucket opened with durable storage.

**NATS is L2. Durability is L3's job.** A KV bucket asking for
``storage="file"`` is the cache tier quietly taking on the source-of-truth
role -- at ``num_replicas=1``, with no backups, no migrations and no schema.
It reads as prudence and it is a promotion nobody reviewed.

The rule this walks is one line: **every KV bucket is memory-backed.**
Anything that genuinely cannot be lost belongs in a ``BaseCollection``, which
composes L1, L2 (this same NATS, memory-backed) and L3 (Yugabyte) rather than
making anyone choose between them.

There is in-house precedent for the argument. ``threetears.epoch`` refuses file
storage for its own state on the grounds that it would be a FALSE guarantee --
file-backed JetStream is durable only if the store directory survives, and the
failure it defends against wipes JetStream wholesale -- and keeps a Postgres
row instead.

**Why an AST walk rather than a type.** ``storage`` is an ordinary ``str``
parameter with a ``"memory"`` default, so a literal ``"file"`` at a call site
type-checks perfectly. Nothing but a reader notices, and by the time anyone
reads it the bucket already exists -- and a bucket's storage is chosen at CREATE
and never reconciled, so the mistake outlives every later deploy that would have
corrected it.
"""

from __future__ import annotations

import ast
from pathlib import Path

__all__ = [
    "DURABLE_STORAGE",
    "KV_OPENING_CALLS",
    "STORAGE_KEYWORD",
    "file_backed_kv_calls",
]

#: The storage value this gate refuses. ``"memory"`` is the contract and the
#: ``kv_bucket`` default, so an absent keyword is always compliant.
DURABLE_STORAGE = "file"

#: The keyword whose value decides a bucket's tier for its whole lifetime.
STORAGE_KEYWORD = "storage"

#: Call names that open a KV bucket. Matched on the ATTRIBUTE or function name
#: rather than a resolved symbol: the walker sees one file at a time and cannot
#: know what ``nc`` is bound to, and every real opener in this estate spells one
#: of these at the call site.
KV_OPENING_CALLS = frozenset({"kv_bucket", "open_kv_stream", "build_kv_stream_config"})


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
    """return every KV-opening call in one module that asks for file storage.

    Only a literal is reported. A computed value (``storage=cfg.storage``) is
    left alone deliberately: the walker cannot evaluate it, and guessing would
    either fail a legitimate caller or report a location nobody can act on. The
    literal form is what every occurrence in this estate has used, and it is
    the form somebody reaches for when they want durability in a hurry.

    :param path: the module to scan
    :ptype path: Path
    :return: ``(line number, callee name)`` for each offending call
    :rtype: list[tuple[int, str]]
    """
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except SyntaxError, OSError:
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
