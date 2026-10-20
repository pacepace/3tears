"""the walker: every pipe into a quiet grep under pipefail.

A logical line is a physical line joined with the ones it continues onto (a trailing ``\\`` or ``|``), so a pipeline
split across lines is seen whole. Comment lines are skipped. A pipe is ``|`` or ``|&``, never ``||``. A quiet grep is
``grep``, ``egrep`` or ``fgrep`` with ``-q`` anywhere in its short flags, or ``--quiet`` / ``--silent``, before its
pattern.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from threetears.enforcement.common import Violation

from threetears.enforcement.grep_q_pipefail.config import GrepQPipefailConfig

__all__ = ["find_grep_q_pipefail_violations", "logical_lines", "quiet_grep_pipes", "scanned_files", "sets_pipefail"]

CATEGORY = "grep_q_pipefail.pipe_into_quiet_grep"

_PIPEFAIL = re.compile(r"(^|[;&|\s])set\s+(-[A-Za-z]*o\s+pipefail|-o\s+pipefail|-[A-Za-z]*\s+-o\s+pipefail)")
# a YAML block scalar (``run: |``) ends in a bar that is not a pipe
_BLOCK_SCALAR = re.compile(r":\s*\|[-+]?\d?$")
_SHEBANG_PIPEFAIL = re.compile(r"^#!.*\bpipefail\b")
# a pipe (not ||): | or |&, then optional spaces, then [e|f]grep and its options up to the first non-option word
_PIPE_GREP = re.compile(r"(?<!\|)\|&?(?!\|)\s*(?:\S*/)?(?:e|f)?grep((?:\s+(?:-[A-Za-z]+|--[a-z][a-z-]*(?:=\S+)?))*)")


def logical_lines(text: str) -> Iterator[tuple[int, str]]:
    """``(first line number, joined text)`` for each logical line, comment lines dropped.

    :param text: the file's text
    :ptype text: str
    :return: the logical lines
    :rtype: Iterator[tuple[int, str]]
    """
    start, parts = 0, list[str]()
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if not parts and line.lstrip().startswith("#"):
            continue
        if not parts:
            start = n
        if line.endswith("\\"):
            parts.append(line[:-1])
            continue
        parts.append(line)
        if line.endswith("|") and not line.endswith("||") and not _BLOCK_SCALAR.search(line):
            continue
        yield start, " ".join(parts)
        parts = []
    if parts:
        yield start, " ".join(parts)


def quiet_grep_pipes(line: str) -> bool:
    """whether ``line`` pipes into a quiet grep.

    :param line: a logical line
    :ptype line: str
    :return: ``True`` when it does
    :rtype: bool
    """
    for m in _PIPE_GREP.finditer(line):
        for opt in m.group(1).split():
            if opt in ("--quiet", "--silent") or (opt.startswith("-") and not opt.startswith("--") and "q" in opt):
                return True
    return False


def sets_pipefail(path: Path, text: str) -> bool:
    """whether the file's pipelines run under pipefail: a workflow (GitHub runs bash steps with ``-eo pipefail``), or a
    script that sets it.

    :param path: the file
    :ptype path: Path
    :param text: its text
    :ptype text: str
    :return: ``True`` when they do
    :rtype: bool
    """
    if path.suffix in (".yml", ".yaml"):
        return True
    first = text.splitlines()[0] if text else ""
    return bool(_SHEBANG_PIPEFAIL.search(first) or any(_PIPEFAIL.search(line) for line in text.splitlines()))


def scanned_files(config: GrepQPipefailConfig) -> list[Path]:
    """every file the globs name, never descending into an excluded directory (a ``**`` glob would walk all of
    ``.venv`` and ``node_modules`` first).

    :param config: the configuration
    :ptype config: GrepQPipefailConfig
    :return: the files, sorted
    :rtype: list[Path]
    """
    import fnmatch
    import os

    root = config.repo_root
    found: set[Path] = set()
    for here, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in config.excluded_dirs]
        rel_dir = Path(here).relative_to(root)
        for name in files:
            rel = (rel_dir / name).as_posix()
            if any(
                fnmatch.fnmatch(rel, g) or (g.startswith("**/") and fnmatch.fnmatch(rel, g[3:])) for g in config.globs
            ):
                found.add(Path(here) / name)
    return sorted(found)


def find_grep_q_pipefail_violations(config: GrepQPipefailConfig) -> list[Violation]:
    """every pipe into a quiet grep in a file that runs under pipefail.

    :param config: the configuration
    :ptype config: GrepQPipefailConfig
    :return: the violations
    :rtype: list[Violation]
    :raises ValueError: when the globs find no files (a scan of nothing proves nothing)
    """
    files = scanned_files(config)
    if not files:
        raise ValueError(f"the quiet-grep scan found no files under {config.repo_root} for {config.globs}")
    out: list[Violation] = []
    for path in files:
        text = path.read_text(errors="replace")
        if not sets_pipefail(path, text):
            continue
        for n, line in logical_lines(text):
            if quiet_grep_pipes(line):
                out.append(
                    Violation(
                        category=CATEGORY,
                        file=path,
                        line=n,
                        symbol=path.name,
                        reason="pipe into a quiet grep under pipefail: grep exits on its first match, the writer dies of "
                        "SIGPIPE and pipefail fails the match; read the output whole, then match it",
                    )
                )
    return out
