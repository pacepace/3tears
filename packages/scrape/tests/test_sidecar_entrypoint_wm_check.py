"""the sidecar entrypoint's window-manager check under ``pipefail``, with xprop writing past the 64 KB pipe buffer.

``xprop | grep -q`` under pipefail fails when it matched once xprop's output outgrows the pipe: grep exits on its first
match and xprop dies of SIGPIPE. The entrypoint now reads the output whole (``wm_running``); this runs that function
from the script itself against a fake xprop, and the old pipeline beside it, to show the difference is real.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "sidecar" / "entrypoint.sh"
OLD = 'xprop -root _NET_SUPPORTING_WM_CHECK 2>/dev/null | grep -q "window id"'

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def fake_xprop(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    xprop = bin_dir / "xprop"
    # the match first, then far more than a pipe holds, so a quiet grep has exited while xprop still writes
    xprop.write_text(
        "#!/usr/bin/env bash\necho '_NET_SUPPORTING_WM_CHECK(WINDOW): window id # 0x200001'\n"
        "head -c 400000 /dev/zero | tr '\\0' 'x'\necho\n"
    )
    xprop.chmod(0o755)
    return bin_dir


def run(bin_dir: Path, body: str) -> int:
    script = f"set -euo pipefail\nexport PATH={bin_dir}:$PATH\n{body}\n"
    return subprocess.run(["bash", "-c", script], check=False).returncode


def test_the_entrypoint_finds_a_running_window_manager_whatever_xprop_writes(tmp_path: Path) -> None:
    func = re.search(r"^wm_running\(\) \{.*?^\}", ENTRYPOINT.read_text(), re.S | re.M)
    assert func is not None, "entrypoint.sh has no wm_running()"
    assert OLD not in ENTRYPOINT.read_text()
    bin_dir = fake_xprop(tmp_path)
    assert run(bin_dir, f"{func.group(0)}\nwm_running") == 0


def test_the_old_pipe_fails_on_the_same_output(tmp_path: Path) -> None:
    # the control: the same fake under the old pipeline reads the running window manager as missing
    assert run(fake_xprop(tmp_path), f"if {OLD}; then exit 0; else exit 1; fi") == 1
