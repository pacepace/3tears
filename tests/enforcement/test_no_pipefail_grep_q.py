"""no `cmd | grep -q` in a shell script that sets `pipefail`.

`grep -q` exits on its first match. When the command feeding it is still writing, that command dies of SIGPIPE (exit
141) and `pipefail` makes the whole pipeline fail, so a check that matched reads as a miss. The scrape sidecar's
entrypoint refused to boot with a running window manager this way. Read the output whole, then match it.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PIPE_INTO_GREP_Q = re.compile(r"\|\s*grep\s+(-[a-zA-Z]*q[a-zA-Z]*|--quiet)\b")


def _scripts() -> list[Path]:
    skip = {".venv", "node_modules", ".git"}
    return [p for p in _REPO.rglob("*.sh") if not skip & set(p.parts)]


def offenders(text: str) -> list[int]:
    """line numbers that pipe into a quiet grep, in a script that sets pipefail."""
    if not re.search(r"set\s+-[a-z]*o\s+pipefail|set\s+-o\s+pipefail", text):
        return []
    return [
        n
        for n, line in enumerate(text.splitlines(), 1)
        if _PIPE_INTO_GREP_Q.search(line) and not line.lstrip().startswith("#")
    ]


def test_a_quiet_grep_at_the_end_of_a_pipe_is_caught() -> None:
    assert offenders("set -euo pipefail\nxprop -root X | grep -q 'window id' && echo up\n") == [2]
    assert offenders("set -euo pipefail\nfoo | grep --quiet bar\n") == [2]
    assert offenders("set -eu\nxprop | grep -q x\n") == []  # no pipefail: grep's status is the pipeline's
    assert offenders("set -euo pipefail\nout=$(xprop)\n[[ $out == *x* ]]\n") == []


def test_no_script_pipes_into_a_quiet_grep_under_pipefail() -> None:
    found = {
        str(p.relative_to(_REPO)): lines for p in _scripts() if (lines := offenders(p.read_text(errors="replace")))
    }
    assert not found, f"pipe into `grep -q` under pipefail (read the output whole, then match it): {found}"
