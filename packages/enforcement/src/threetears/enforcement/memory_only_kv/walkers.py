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

from threetears.enforcement.common import parse_python_path

__all__ = [
    "DURABLE_STORAGE",
    "KV_OPENING_CALLS",
    "MIN_REPLICAS",
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

#: Call names that create or declare anything JetStream stores: a KV bucket, a plain stream, an
#: Object Store, or a stream config built by hand. Matched on the ATTRIBUTE or function name rather
#: than a resolved symbol: the walker sees one file at a time and cannot know what ``nc`` is bound
#: to, and every real opener in this estate spells one of these at the call site.
#:
#: Streams were outside this gate until 2026-11-15, on the reading that an answer or an audit record
#: "must survive a broker restart". Six streams and buckets were then found on file storage on cobalt
#: dev and prod -- the audit stream at 268 MB -- among them KV buckets this gate never saw, because
#: they were opened through ``ensure_kv_bucket``, ``NatsKvBucket.open`` or ``PersistedCopyBucket``.
#: A stream's durability is the same as a bucket's: three memory copies, plus L3 for what must outlive
#: all three. So every opener is here.
KV_OPENING_CALLS = frozenset(
    {
        "kv_bucket",
        "ensure_kv_bucket",
        "open_kv_stream",
        "build_kv_stream_config",
        "ensure_jetstream_stream",
        "ensure_replicated_jetstream_stream",
        "ensure_object_store",
        "PersistedCopyBucket",
        "StreamConfig",
        "_NatsStreamConfig",
        "add_stream",
    }
)

#: ``NatsKvBucket.open`` is an opener too, but ``open`` alone names every file handle in the estate:
#: it is matched only on these receivers.
_OPEN_RECEIVERS = frozenset({"NatsKvBucket"})

#: The keywords that set how many servers hold a store. A literal below :data:`MIN_REPLICAS` is a
#: store one server restart can lose (cobalt prod, 2026-11-02: an R1 leases bucket wiped a hold).
_REPLICAS_KEYWORDS = frozenset({"replicas", "num_replicas"})

#: The fewest servers a store may be declared on, as a literal. ``None`` (the 3tears default: one
#: server, for a local single-node NATS) and a computed value (``config.nats.replicas``, three in
#: every deployment's values) are left alone.
MIN_REPLICAS = 3


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


def _is_file_storage(value: ast.expr) -> bool:
    """whether a storage value is file storage: the literal ``"file"`` or ``StorageType.FILE``.

    :param value: the keyword's value
    :ptype value: ast.expr
    :return: whether it asks for file storage
    :rtype: bool
    """
    if isinstance(value, ast.Constant):
        return value.value == DURABLE_STORAGE
    return isinstance(value, ast.Attribute) and value.attr == "FILE"


def _is_opener(node: ast.Call) -> str | None:
    """the opener a call is, or ``None`` when it opens no store.

    :param node: the call
    :ptype node: ast.Call
    :return: the opener's name
    :rtype: str | None
    """
    name = _called_name(node)
    if name in KV_OPENING_CALLS:
        return name
    func = node.func
    if (
        name == "open"
        and isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in _OPEN_RECEIVERS
    ):
        return f"{func.value.id}.open"
    return None


def _file_storage_defaults(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """whether a function's ``storage`` parameter defaults to file storage.

    A default is the value every caller that says nothing gets: ``PersistedCopyBucket`` defaulted
    to ``"file"``, and two catalogs were file-backed through it without either naming a storage.

    :param node: the function
    :ptype node: ast.FunctionDef | ast.AsyncFunctionDef
    :return: whether it does
    :rtype: bool
    """
    arguments = node.args
    positional = arguments.posonlyargs + arguments.args
    pairs = list(zip(positional[len(positional) - len(arguments.defaults) :], arguments.defaults, strict=True))
    pairs += [(a, d) for a, d in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True) if d is not None]
    return any(arg.arg == STORAGE_KEYWORD and _is_file_storage(default) for arg, default in pairs)


def file_backed_kv_calls(path: Path) -> list[tuple[int, str]]:
    """every store-opening call in one module that asks for file storage or fewer than three servers.

    Reported too: a function whose ``storage`` parameter defaults to file storage.

    :param path: the module to scan
    :ptype path: Path
    :return: ``(line number, callee name)`` for each offending call
    :rtype: list[tuple[int, str]]
    """
    try:
        tree = parse_python_path(path)
    except UnicodeDecodeError:
        tree = None
    except OSError, SyntaxError:
        return []
    if tree is None:
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            return []

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _file_storage_defaults(node):
            found.append((node.lineno, f"{node.name}(storage default)"))
            continue
        if not isinstance(node, ast.Call):
            continue
        name = _is_opener(node)
        if name is None:
            continue
        for keyword in node.keywords:
            value = keyword.value
            if keyword.arg == STORAGE_KEYWORD and _is_file_storage(value):
                found.append((node.lineno, name))
            elif (
                keyword.arg in _REPLICAS_KEYWORDS
                and isinstance(value, ast.Constant)
                and isinstance(value.value, int)
                and not isinstance(value.value, bool)
                and value.value < MIN_REPLICAS
            ):
                found.append((node.lineno, f"{name}({keyword.arg}={value.value})"))
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
