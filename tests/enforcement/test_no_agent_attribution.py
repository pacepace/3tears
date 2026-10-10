"""Enforcement: CLAUDE.md's "No agent attribution. Anywhere. Ever." has a gate.

``scripts/check-attribution.sh`` reads every commit a pull request adds and refuses one that
credits an agent (a ``Co-Authored-By`` naming one, an Anthropic noreply address, a "Generated
with" footer, a ``Claude-Session`` trailer). CI runs it on every pull request. These tests run the
real script over a throwaway repository, so they fail if it stops catching a trailer, starts
flagging a mention of the tool, examines history already on the base, or ignores its exemptions;
and they fail if the CI step goes away.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check-attribution.sh"
_EXEMPTIONS = _REPO_ROOT / "scripts" / "attribution-exemptions.txt"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None, reason="needs git and bash"
)

_ATTRIBUTED = [
    pytest.param("Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", id="co-authored-by-claude"),
    pytest.param("co-authored-by: Some Model <bot@anthropic.com>", id="co-authored-by-anthropic"),
    pytest.param("Reach me at noreply@anthropic.com", id="anthropic-noreply"),
    pytest.param("🤖 Generated with [Claude Code](https://claude.com/claude-code)", id="generated-with-footer"),
    pytest.param("Claude-Session: https://claude.ai/code/session_x", id="claude-session-trailer"),
]

_CLEAN = [
    pytest.param("Scoped snapshot: a publish keeps the index true", id="plain"),
    pytest.param("Use Claude Code's hooks to run the critic", id="names-the-tool"),
    pytest.param("Co-Authored-By: Jane Doe <jane@example.com>", id="human-co-author"),
]


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, a test repository
        list(args), cwd=cwd, capture_output=True, text=True, check=False
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """a repository carrying the real script, with one commit as the base."""
    (tmp_path / "scripts").mkdir()
    shutil.copy2(_SCRIPT, tmp_path / "scripts" / "check-attribution.sh")
    for command in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "dev@example.com"],
        ["git", "config", "user.name", "Dev"],
        ["git", "add", "."],
        ["git", "commit", "-q", "-m", "base"],
    ):
        assert _run(*command, cwd=tmp_path).returncode == 0
    return tmp_path


def _commit(repo: Path, message: str) -> str:
    # --no-verify: a developer's own commit-msg hook may strip the very trailers under test
    assert _run("git", "commit", "-q", "--no-verify", "--allow-empty", "-m", message, cwd=repo).returncode == 0
    return _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()


def _check(repo: Path, base: str) -> subprocess.CompletedProcess[str]:
    return _run("bash", "scripts/check-attribution.sh", f"{base}..HEAD", cwd=repo)


class TestTheScript:
    @pytest.mark.parametrize("line", _ATTRIBUTED)
    def test_a_commit_crediting_an_agent_is_refused_by_hash(self, repo: Path, line: str) -> None:
        base = _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
        _commit(repo, "a clean change")
        bad = _commit(repo, f"a change\n\n{line}\n")
        result = _check(repo, base)
        assert result.returncode == 1, result.stdout + result.stderr
        assert bad in result.stderr

    @pytest.mark.parametrize("line", _CLEAN)
    def test_a_commit_crediting_no_agent_passes(self, repo: Path, line: str) -> None:
        base = _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
        _commit(repo, f"a change\n\n{line}\n")
        result = _check(repo, base)
        assert result.returncode == 0, result.stderr

    def test_history_already_on_the_base_is_never_examined(self, repo: Path) -> None:
        _commit(repo, "landed long ago\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n")
        base = _commit(repo, "the base now")
        _commit(repo, "this PR's own change")
        assert _check(repo, base).returncode == 0

    def test_an_exempted_hash_passes_and_nothing_else_does(self, repo: Path) -> None:
        base = _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
        landed = _commit(repo, "landed\n\nClaude-Session: https://claude.ai/code/session_x\n")
        (repo / "scripts" / "attribution-exemptions.txt").write_text(f"{landed} landed before the check\n")
        assert _check(repo, base).returncode == 0
        later = _commit(repo, "later\n\nClaude-Session: https://claude.ai/code/session_y\n")
        result = _check(repo, base)
        assert result.returncode == 1
        assert later in result.stderr and landed not in result.stderr

    @pytest.mark.parametrize("line", _ATTRIBUTED)
    def test_a_message_file_crediting_an_agent_is_refused(self, tmp_path: Path, line: str) -> None:
        message = tmp_path / "COMMIT_EDITMSG"
        message.write_text(f"subject\n\n{line}\n")
        assert _run("bash", str(_SCRIPT), "--message", str(message), cwd=tmp_path).returncode == 1


class TestTheGate:
    def test_ci_runs_the_script_over_every_pull_requests_own_commits(self) -> None:
        workflow = yaml.safe_load(_WORKFLOW.read_text())
        steps = workflow["jobs"]["check"]["steps"]
        gate = [step for step in steps if "scripts/check-attribution.sh" in str(step.get("run", ""))]
        assert len(gate) == 1, "no CI step refuses agent attribution"
        (step,) = gate
        assert "github.event.pull_request.base.sha" in step["run"]
        assert "github.event.pull_request.head.sha" in step["run"]
        # not skipped with the suite for a documentation-only PR
        assert "steps.scope" not in str(step.get("if", ""))
        assert step.get("if") == "github.event_name == 'pull_request'"

    def test_every_exemption_is_a_full_hash(self) -> None:
        entries = [line for line in _EXEMPTIONS.read_text().splitlines() if line and not line.startswith("#")]
        assert entries, "the exemption ledger is empty; delete it rather than leave it"
        assert all(re.fullmatch(r"[0-9a-f]{40} .+", line) for line in entries), entries
