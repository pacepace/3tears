"""Enforcement: CLAUDE.md's "No agent attribution. Anywhere. Ever." has a gate.

``scripts/check-attribution.sh`` reads every commit a pull request adds and refuses one that
credits an agent (a ``Co-Authored-By`` naming one, an Anthropic noreply address, a "Generated
with" footer, a ``Claude-Session`` trailer). CI runs it on every pull request. These tests run the
real script over a throwaway repository, so they fail if it stops catching a trailer, starts
flagging a mention of the tool, examines history already on the base, or ignores its exemptions;
and they fail if the CI step goes away. ``scripts/install-hooks.sh`` installs the same check as a
commit-msg hook; it is run over a throwaway repository too, and a commit through the hook it wrote
is what proves it.
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
_INSTALLER = _REPO_ROOT / "scripts" / "install-hooks.sh"
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
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


@pytest.fixture
def hooked_repo(repo: Path) -> Path:
    """a throwaway repository carrying the real check script and the real installer, and no hooks.

    ``git init`` copies the developer's template hooks in (``init.templateDir``), and a template
    commit-msg hook that strips trailers would pass a message the check must refuse; each test that
    wants a hook already there puts its own.
    """
    shutil.copy2(_INSTALLER, repo / "scripts" / "install-hooks.sh")
    hooks = Path(_run("git", "rev-parse", "--path-format=absolute", "--git-path", "hooks", cwd=repo).stdout.strip())
    for name in ("commit-msg", "commit-msg.chained"):
        (hooks / name).unlink(missing_ok=True)
    return repo


def _install(repo: Path) -> subprocess.CompletedProcess[str]:
    return _run("bash", "scripts/install-hooks.sh", cwd=repo)


def _commit_through_hooks(repo: Path, message: str) -> subprocess.CompletedProcess[str]:
    return _run("git", "commit", "-q", "--allow-empty", "-m", message, cwd=repo)


def _hook(repo: Path) -> Path:
    return (
        Path(_run("git", "rev-parse", "--path-format=absolute", "--git-path", "hooks", cwd=repo).stdout.strip())
        / "commit-msg"
    )


class TestTheHookInstaller:
    @pytest.mark.parametrize("line", _ATTRIBUTED)
    def test_the_installed_hook_refuses_a_message_crediting_an_agent(self, hooked_repo: Path, line: str) -> None:
        assert _install(hooked_repo).returncode == 0
        head = _run("git", "rev-parse", "HEAD", cwd=hooked_repo).stdout.strip()
        result = _commit_through_hooks(hooked_repo, f"a change\n\n{line}\n")
        assert result.returncode != 0, "the hook let an attributed commit through"
        assert "credits an agent" in result.stderr
        assert _run("git", "rev-parse", "HEAD", cwd=hooked_repo).stdout.strip() == head

    @pytest.mark.parametrize("line", _CLEAN)
    def test_the_installed_hook_lets_a_clean_message_through(self, hooked_repo: Path, line: str) -> None:
        assert _install(hooked_repo).returncode == 0
        result = _commit_through_hooks(hooked_repo, f"a change\n\n{line}\n")
        assert result.returncode == 0, result.stderr

    def test_installing_twice_leaves_one_working_hook(self, hooked_repo: Path) -> None:
        assert _install(hooked_repo).returncode == 0
        first = _hook(hooked_repo).read_text()
        assert _install(hooked_repo).returncode == 0
        assert _hook(hooked_repo).read_text() == first
        assert (
            _commit_through_hooks(hooked_repo, "x\n\nClaude-Session: https://claude.ai/code/session_x\n").returncode
            != 0
        )

    def test_a_hook_it_did_not_write_is_kept_and_runs_first(self, hooked_repo: Path) -> None:
        """a template's or a tool's commit-msg hook (one that strips trailers, say) keeps working."""
        hook = _hook(hooked_repo)
        hook.parent.mkdir(parents=True, exist_ok=True)
        # a stand-in for a trailer-stripping hook: it drops every Claude-Session line
        foreign = '#!/bin/sh\ngrep -v \'^Claude-Session:\' "$1" > "$1.tmp"; mv "$1.tmp" "$1"\n'
        hook.write_text(foreign)
        hook.chmod(0o755)
        assert _install(hooked_repo).returncode == 0
        assert (hook.parent / "commit-msg.chained").read_text() == foreign
        # the kept hook ran first: the trailer it strips never reached the check
        stripped = _commit_through_hooks(hooked_repo, "x\n\nClaude-Session: https://claude.ai/code/session_x\n")
        assert stripped.returncode == 0, stripped.stderr
        assert "Claude-Session" not in _run("git", "log", "-1", "--format=%B", cwd=hooked_repo).stdout
        # and the check still refuses what it does not strip
        assert (
            _commit_through_hooks(hooked_repo, "x\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n").returncode != 0
        )
        # a re-run keeps the chain as it is
        assert _install(hooked_repo).returncode == 0
        assert (hook.parent / "commit-msg.chained").read_text() == foreign

    def test_a_refusal_from_the_kept_hook_stops_the_commit(self, hooked_repo: Path) -> None:
        hook = _hook(hooked_repo)
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/bin/sh\necho kept hook refused >&2\nexit 3\n")
        hook.chmod(0o755)
        assert _install(hooked_repo).returncode == 0
        refused = _commit_through_hooks(hooked_repo, "a clean change")
        assert refused.returncode != 0
        assert "kept hook refused" in refused.stderr

    def test_two_different_foreign_hooks_are_never_overwritten(self, hooked_repo: Path) -> None:
        hook = _hook(hooked_repo)
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/bin/sh\nexit 0\n")
        (hook.parent / "commit-msg.chained").write_text("#!/bin/sh\nexit 1\n")
        result = _install(hooked_repo)
        assert result.returncode != 0
        assert str(hook) in result.stderr
        assert hook.read_text() == "#!/bin/sh\nexit 0\n"
        assert (hook.parent / "commit-msg.chained").read_text() == "#!/bin/sh\nexit 1\n"

    def test_a_commit_in_another_worktree_is_checked_too(
        self, hooked_repo: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        assert _install(hooked_repo).returncode == 0
        other = tmp_path_factory.mktemp("wt") / "other"
        assert _run("git", "worktree", "add", "-q", "-b", "side", str(other), cwd=hooked_repo).returncode == 0
        refused = _commit_through_hooks(other, "x\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n")
        assert refused.returncode != 0
        assert _commit_through_hooks(other, "a clean change").returncode == 0

    def test_a_worktree_whose_branch_has_no_check_script_still_commits(
        self, hooked_repo: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """the hook is shared by every worktree of the clone, and some are on branches older than the script."""
        assert _install(hooked_repo).returncode == 0
        # a branch whose tree never had the script
        for command in (
            ["git", "checkout", "-q", "--orphan", "older"],
            ["git", "rm", "-rqf", "--cached", "."],
            ["git", "commit", "-q", "--no-verify", "--allow-empty", "-m", "before the check existed"],
            ["git", "checkout", "-qf", "main"],
        ):
            assert _run(*command, cwd=hooked_repo).returncode == 0, command
        other = tmp_path_factory.mktemp("wt") / "older"
        assert _run("git", "worktree", "add", "-q", str(other), "older", cwd=hooked_repo).returncode == 0
        assert not (other / "scripts" / "check-attribution.sh").exists()
        committed = _commit_through_hooks(other, "a change on the older branch")
        assert committed.returncode == 0, committed.stderr
        assert "no scripts/check-attribution.sh in this checkout" in committed.stderr
        # and where the script is checked out, the same hook still runs it
        assert (
            _commit_through_hooks(hooked_repo, "x\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n").returncode != 0
        )

    def test_claude_md_tells_a_contributor_to_run_it(self) -> None:
        assert "./scripts/install-hooks.sh" in _CLAUDE_MD.read_text()
