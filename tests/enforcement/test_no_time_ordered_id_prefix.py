"""enforcement: a name, key or id is never cut from the HEAD of a time-ordered id.

A uuid7 leads with its 48-bit millisecond timestamp. ``uuid7().hex[:8]`` is the top 32 bits of
that timestamp, so it is the same for every id minted in the same ~65 seconds, and
``hex[:12]`` is the same for every id minted in the same millisecond. Anything derived from
it collides:

- ``platform.namespaces`` is UNIQUE on ``name`` and on ``schema_name``, and the memory and
  conversation namespace names were ``memories.<agent hex[:8]>.<customer hex[:8]>``. Two
  agents created for one customer in the same minute -- a cluster apply creating several at
  once -- asked for ONE name, and the second agent's namespace could never be written.
- a KV lease holder id of ``pod-<uuid7 hex[:12]>`` named every factory built in the same
  millisecond, and the holder id is the lease's fence.
- a backup object key with ``<uuid7 hex[:8]>`` after a to-the-second stamp let two backups in
  one millisecond name one object, the second overwriting the first.

The rule: a prefix slice of ``.hex`` (``x.hex[:n]``, ``x.hex[0:n]``) is refused in every
package's ``src/`` and ``tests/`` and in the root ``tests/``, except on an explicit
``uuid4()`` call, whose every digit is random. Entity ids here are uuid7, so a variable
holding one is presumed time-ordered. Take the whole hex, a hash of the whole id, or the
random tail (``x.hex[-n:]``) instead.

In tests the same slice is a flake rather than an outage: two test runs a moment apart share
an in-memory database name, a stream, a scratch database.

What it does not catch: a prefix taken some other way (``str(x)[:8]``, ``f"{x}"[:8]``,
``x.int >> n``). None derives a name in this workspace today; the ones that exist truncate
log text.

Static parsing only -- no imports executed -- consistent with the rest of
``tests/enforcement``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: directory names never walked: environments, caches and build output hold code this
#: workspace does not own.
_SKIPPED_PARTS: Final[frozenset[str]] = frozenset(
    {".venv", "venv", "node_modules", "__pycache__", "build", "dist", ".mypy_cache", ".ruff_cache"}
)


def _is_uuid4_call(node: ast.expr) -> bool:
    """whether ``node`` is a call to ``uuid4`` (bare or module-qualified).

    :param node: the expression a ``.hex`` is read from
    :ptype node: ast.expr
    :return: whether every digit of its hex is random
    :rtype: bool
    """
    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Name) and node.func.id == "uuid4")
        or (isinstance(node.func, ast.Attribute) and node.func.attr == "uuid4")
    )


def _starts_at_the_head(node: ast.Slice) -> bool:
    """whether a slice starts at index 0 and stops short of the end.

    :param node: the slice
    :ptype node: ast.Slice
    :return: whether it is a prefix
    :rtype: bool
    """
    starts_at_zero = node.lower is None or (isinstance(node.lower, ast.Constant) and node.lower.value == 0)
    return starts_at_zero and node.upper is not None


def prefix_slices(tree: ast.AST) -> list[int]:
    """the line of every ``x.hex[:n]`` / ``x.hex[0:n]`` in ``tree`` not taken from a ``uuid4()``.

    :param tree: a parsed module
    :ptype tree: ast.AST
    :return: the lines, in order
    :rtype: list[int]
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "hex"
            and isinstance(node.slice, ast.Slice)
            and _starts_at_the_head(node.slice)
            and not _is_uuid4_call(node.value.value)
        ):
            lines.append(node.lineno)
    return sorted(lines)


def _scanned_files() -> list[Path]:
    """every python file under a package's ``src/`` or ``tests/``, and the root ``tests/``.

    :return: the files, sorted
    :rtype: list[Path]
    """
    found: set[Path] = set()
    for path in (_REPO_ROOT / "packages").rglob("*.py"):
        parts = set(path.relative_to(_REPO_ROOT).parts)
        if parts & _SKIPPED_PARTS:
            continue
        if "src" in parts or "tests" in parts:
            found.add(path)
    for path in (_REPO_ROOT / "tests").rglob("*.py"):
        if not set(path.relative_to(_REPO_ROOT).parts) & _SKIPPED_PARTS:
            found.add(path)
    return sorted(found)


def test_no_name_is_cut_from_the_head_of_a_time_ordered_id() -> None:
    files = _scanned_files()
    src_files = [path for path in files if "src" in path.relative_to(_REPO_ROOT).parts]
    test_files = [path for path in files if "tests" in path.relative_to(_REPO_ROOT).parts]
    # both halves of the walk must find something, or its silence means nothing
    assert len(src_files) > 500, f"the walk found {len(src_files)} source files; the glob has gone wrong"
    assert len(test_files) > 500, f"the walk found {len(test_files)} test files; the glob has gone wrong"

    found = [
        f"{path.relative_to(_REPO_ROOT)}:{line}"
        for path in files
        for line in prefix_slices(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    ]

    assert found == [], (
        "a prefix of a uuid7's hex is its timestamp, shared by every id minted in the same moment. "
        "Use the whole hex, a hash of the whole id, or the random tail (x.hex[-n:]):\n  " + "\n  ".join(found)
    )


def test_the_reader_refuses_the_head_and_admits_the_rest() -> None:
    tree = ast.parse(
        "a = uuid7().hex[:8]\n"
        "b = customer_id.hex[:12]\n"
        "c = agent.hex[0:8]\n"
        "d = uuid7().hex[-8:]\n"
        "e = uuid4().hex[:8]\n"
        "f = uuid.uuid4().hex[:8]\n"
        "g = uuid7().hex\n"
        "h = agent.hex[8:]\n"
        "i = digest.hexdigest()[:16]\n"
    )

    assert prefix_slices(tree) == [1, 2, 3]
