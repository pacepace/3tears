"""Enforcement: UUIDs stay ``UUID`` objects; ``str()`` only at borders.

Discipline contract (matches the convention used across the sibling
projects, e.g. ``metallm/api/tests/enforcement/test_uuid_stringification.py``):
UUID-typed identifiers must travel through the codebase as ``UUID``
objects, not strings. Converting a UUID to ``str`` is legitimate ONLY at
a system *border* -- where the value leaves Python typing and becomes
wire/text: structured logging, f-strings, ``json.dumps``, NATS envelope
serialization, OpenTelemetry span attributes, HTTP headers, and the L1
SQLite cache (keys stored as text).

The walker flags ``str(<x>_id)`` / ``str(<x>.id)`` conversions. A
conversion is allowed when EITHER:

- it sits next to a recognized border idiom (the walker detects these
  automatically -- no annotation needed), OR
- it carries an explicit ``# convert at border: <reason>`` marker on the
  same line (for a genuine border the detector cannot recognize).

Everything else -- a UUID stringified away from any border, with no
marker -- is a violation. Those are the "willy-nilly" conversions that
erode the type contract; each must either move the conversion to a real
border or be justified with the explicit marker. The marker stays rare
and deliberate precisely because the common borders auto-pass.

Scope: walks every ``packages/*/src/`` tree across the workspace (this
includes the ``packages/agent/*/src/`` family nested under the agent
namespace dir). Migration modules, test trees, and ``__pycache__`` are
skipped.

AST-free, regex + nearby-context only; well under the 15s budget.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PACKAGES_ROOT = _REPO_ROOT / "packages"

# Stringification idioms on a UUID-shaped identifier.
_FORBIDDEN_PATTERNS = [
    re.compile(r"str\(\w*_id\)"),  # str(user_id), str(conversation_id)
    re.compile(r"str\(\w+\.id\)"),  # str(entity.id)
]

# Idioms in the ±1-line context that mark a legitimate border. When a
# stringification sits next to one of these, the conversion is crossing a
# real system boundary (wire/text/log) and needs no explicit marker.
_LEGITIMATE_BORDERS = [
    # Structured logging + f-strings
    "log.",
    "logger.",
    "_logger.",
    ".debug(",
    ".info(",
    ".warning(",
    ".error(",
    ".exception(",
    "extra_data",
    "extra=",
    'f"',
    "f'",
    # JSON / wire serialization
    "json.dumps(",
    "model_dump(",
    "model_dump_json(",
    ".serialize(",
    "envelope",
    ".publish(",
    "model_validate",
    # OpenTelemetry
    "set_attribute(",
    "attributes[",
    "span.",
    # HTTP headers / responses
    ".headers[",
    "X-",
    "correlation_id",
    # L1 SQLite cache border (keys persisted as text)
    "select_by_id(",
    "delete_by_id(",
    ".upsert(",
    "normalize_pk(",
    "get_field_sync(",
    "set_field_sync(",
    "key=",
    # External-API string-typed handles
    "thread_id",  # LangGraph checkpoint thread id is a string by contract
    # Inbound normalization: ``UUID(str(x))`` coerces TO a UUID -- the str()
    # is consumed immediately by the UUID() constructor, not leaked.
    "UUID(str(",
    # Explicit marker for a genuine border the walker cannot detect.
    "convert at border",
]


def _collect_src_files() -> list[Path]:
    """All Python source files under any ``packages/*/src/`` tree.

    Skips ``migrations/`` (migration-time synthesis), test trees, and
    ``__pycache__``.
    """
    return sorted(
        p
        for p in _PACKAGES_ROOT.rglob("src/**/*.py")
        if p.stat().st_size > 0
        and "migrations" not in p.parts
        and "tests" not in p.parts
        and "__pycache__" not in p.parts
    )


_SRC_FILES = _collect_src_files()
_SRC_IDS = [str(p.relative_to(_REPO_ROOT)) for p in _SRC_FILES]


def _violations_in(text: str) -> list[tuple[int, str]]:
    """Return ``(line_no, line)`` for off-border, unmarked UUID->str sites."""
    lines = text.splitlines()
    out: list[tuple[int, str]] = []
    for line_no, line in enumerate(lines, 1):
        if not any(p.search(line) for p in _FORBIDDEN_PATTERNS):
            continue
        # ±1-line context for border detection.
        context_parts: list[str] = []
        for offset in (-1, 0, 1):
            idx = line_no - 1 + offset
            if 0 <= idx < len(lines):
                context_parts.append(lines[idx])
        context = "\n".join(context_parts)
        if any(border in context for border in _LEGITIMATE_BORDERS):
            continue
        out.append((line_no, line.strip()))
    return out


class TestUuidStringificationBorders:
    """UUID stringification must occur only at detectable/marked borders."""

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_no_offborder_uuid_stringification(self, src_file: Path) -> None:
        """No ``str(*_id)`` away from a border without an explicit marker."""
        violations = _violations_in(src_file.read_text(encoding="utf-8"))
        if violations:
            rel = src_file.relative_to(_REPO_ROOT)
            detail = "\n".join(f"  line {ln}: {src}" for ln, src in violations)
            pytest.fail(
                f"{rel}: UUID->str conversion away from any recognized border:\n"
                f"{detail}\n"
                f"Keep the value a UUID, move the conversion to a real border, "
                f"or -- if this IS a legitimate border the walker can't detect -- "
                f"add a '# convert at border: <reason>' comment on the line."
            )


class TestWalkerHasTeeth:
    """Self-tests: the walker must catch real violations and honor borders."""

    def test_flags_offborder_unmarked(self) -> None:
        src = "def f(user_id):\n    return {'u': str(user_id)}\n"
        assert _violations_in(src), "walker missed an off-border stringification"

    def test_allows_logging_border(self) -> None:
        src = 'def f(user_id):\n    log.info("x", extra={"u": str(user_id)})\n'
        assert not _violations_in(src), "walker flagged a logging-border conversion"

    def test_allows_explicit_marker(self) -> None:
        src = "def f(user_id):\n    token = str(user_id)  # convert at border: cache key\n"
        assert not _violations_in(src), "walker ignored the explicit border marker"
