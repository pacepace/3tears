"""the release tool against scratch git repositories: every refusal, every edit, every undo.

A refusal must leave the tree AND the index byte-identical; a failure after the
first edit must restore both. Each refusal below is asserted that way, so a
refusal that half-edits is a red test, not a review finding.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from packages.enforcement.tests.release._scratch import TODAY, commit_all, git, release, snapshot
from threetears.enforcement.release import main

_ADDED_SECTION = "## [0.1.0] - unreleased\n\nThe first release.\n\n### Added\n\n- the first thing\n"


def _refused(root: Path, action: str, expected: str) -> str:
    """runs *action*, asserts it refused with *expected* in its message and changed nothing.

    :param root: scratch repository
    :ptype root: Path
    :param action: release action
    :ptype action: str
    :param expected: text the refusal must contain
    :ptype expected: str
    :return: stderr
    :rtype: str
    """
    before = snapshot(root)
    status, _, err = release(root, action)
    assert status == 1, err
    assert expected in err, err
    assert snapshot(root) == before, "a refusal changed the tree or the index"
    return err


class TestTheCommandLine:
    """help, and a wrong invocation, behave as a command line should."""

    def test_help_prints_to_stdout_and_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        """
        ``--help`` is a request, not an error.

        :param capsys: pytest's output capture
        :ptype capsys: pytest.CaptureFixture[str]
        :return: nothing
        :rtype: None
        """
        with pytest.raises(SystemExit) as exited:
            main(["--help"])
        assert exited.value.code == 0
        captured = capsys.readouterr()
        assert "patch" in captured.out and "release" in captured.out
        assert captured.err == ""

    @pytest.mark.parametrize("argv", [["bogus"], ["verify"], ["patch", "1.2.3"], ["verify", "1.2"]])
    def test_a_wrong_invocation_exits_two_on_stderr(self, argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
        """
        an unknown action, a version on the wrong action, or a bad version is a usage error.

        :param argv: the invocation
        :ptype argv: list[str]
        :param capsys: pytest's output capture
        :ptype capsys: pytest.CaptureFixture[str]
        :return: nothing
        :rtype: None
        """
        with pytest.raises(SystemExit) as exited:
            main(argv)
        assert exited.value.code == 2
        assert capsys.readouterr().err

    def test_a_directory_that_is_not_a_checkout_root_is_refused(self, make_repo: Callable[..., Path]) -> None:
        """
        the tool acts on the top of a checkout and nothing below it.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(tags=("v0.1.0",))
        nested = root / "nested"
        nested.mkdir()
        (nested / "pyproject.toml").write_text('[project]\nname = "n"\nversion = "0.1.0"\n')
        (nested / "CHANGELOG.md").write_text("## [Unreleased]\n\n- x\n")
        status, _, err = release(nested, "patch")
        assert status == 1
        assert "is not the top of a git checkout" in err


class TestRefusals:
    """each refusal fires, and leaves the tree and the index exactly as they were."""

    def test_a_hand_edited_version_line(self, make_repo: Callable[..., Path]) -> None:
        """
        the tool owns the version line.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(tags=("v0.1.0",))
        (root / "pyproject.toml").write_text('[project]\nname = "proj"\nversion = "0.1.5"\n')
        _refused(root, "patch", "the version line is dirty")

    def test_staged_release_notes(self, make_repo: Callable[..., Path]) -> None:
        """
        the bump stages its own files; it will not mix in someone else's.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(tags=("v0.1.0",))
        (root / "CHANGELOG.md").write_text((root / "CHANGELOG.md").read_text() + "\n- staged\n")
        git(root, "add", "CHANGELOG.md")
        _refused(root, "patch", "STAGED")

    def test_uncommitted_release_notes(self, make_repo: Callable[..., Path]) -> None:
        """
        the bump commit carries only the bump.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(tags=("v0.1.0",))
        (root / "CHANGELOG.md").write_text((root / "CHANGELOG.md").read_text() + "\n- uncommitted\n")
        _refused(root, "patch", "CHANGELOG.md has uncommitted edits")

    def test_a_patch_from_an_untagged_version(self, make_repo: Callable[..., Path]) -> None:
        """
        a declared-but-unreleased version is released (or tagged), never patched past.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(version="0.1.1", tags=("v0.1.0",))
        err = _refused(root, "patch", "0.1.1 has no v0.1.1 tag")
        assert "./scripts/bump-version.sh release" in err
        assert "git fetch --tags" in err

    def test_a_minor_from_an_untagged_version_whose_line_has_no_tag(self, make_repo: Callable[..., Path]) -> None:
        """
        the untagged first release cannot be skipped by a minor: nothing anchors the line.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(version="0.1.0")
        err = _refused(root, "minor", "0.1.0 has no v0.1.0 tag")
        assert "0.1.x has none" in err

    def test_a_target_that_is_already_tagged(self, make_repo: Callable[..., Path]) -> None:
        """
        versions only move forward.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(tags=("v0.1.0", "v0.1.1"))
        _refused(root, "patch", "v0.1.1 is already tagged")

    def test_a_target_whose_section_already_exists(self, make_repo: Callable[..., Path]) -> None:
        """
        a second section for one version is refused.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(tags=("v0.1.0",), sections="## [0.1.1] - 2026-01-01\n\n- old\n")
        _refused(root, "patch", "already has a section for 0.1.1")

    def test_a_bump_with_no_notes(self, make_repo: Callable[..., Path]) -> None:
        """
        a release with nothing to say is refused.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(tags=("v0.1.0",), unreleased="")
        _refused(root, "patch", "nothing under `## [Unreleased]`")

    def test_a_release_with_no_notes(self, make_repo: Callable[..., Path]) -> None:
        """
        ``release`` with no section and nothing unreleased is refused.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(unreleased="")
        _refused(root, "release", "nothing under `## [Unreleased]`")

    def test_a_release_of_a_tagged_version(self, make_repo: Callable[..., Path]) -> None:
        """
        a tagged version is released already.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(tags=("v0.1.0",))
        _refused(root, "release", "v0.1.0 is already tagged")


class TestBumps:
    """what a successful run writes."""

    def test_a_patch_moves_the_version_the_notes_and_the_lock(
        self, make_repo: Callable[..., Path], uvx_log: Callable[[], list[str]]
    ) -> None:
        """
        version line, dated section, empty unreleased heading, relock, and a pathspec commit hint.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :param uvx_log: the stub relock log
        :ptype uvx_log: Callable[[], list[str]]
        :return: nothing
        :rtype: None
        """
        root = make_repo(tags=("v0.1.0",), sections="## [0.1.0] - 2026-01-01\n\n- old\n")
        status, out, err = release(root, "patch")
        assert status == 0, err
        assert 'version = "0.1.1"' in (root / "pyproject.toml").read_text()
        changelog = (root / "CHANGELOG.md").read_text()
        assert changelog.startswith(
            f"# Changelog\n\n## [Unreleased]\n\n## [0.1.1] - {TODAY}\n\n### Added\n\n- a new thing\n"
        )
        assert "## [0.1.0] - 2026-01-01" in changelog
        assert uvx_log() == [str(root)]
        assert "-- pyproject.toml uv.lock CHANGELOG.md" in out
        assert "git tag -a v0.1.1" in out
        assert git(root, "diff", "--cached", "--name-only") == ""

    @pytest.mark.parametrize(("kind", "expected"), [("patch", "0.1.1"), ("minor", "0.2.0"), ("major", "1.0.0")])
    def test_each_kind_resolves(self, make_repo: Callable[..., Path], kind: str, expected: str) -> None:
        """
        patch, minor and major arithmetic.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :param kind: bump kind
        :ptype kind: str
        :param expected: resulting version
        :ptype expected: str
        :return: nothing
        :rtype: None
        """
        root = make_repo(tags=("v0.1.0",))
        status, out, err = release(root, kind)
        assert status == 0, err
        assert f"Done. Version: {expected}" in out

    def test_release_folds_unreleased_into_the_declared_section(self, make_repo: Callable[..., Path]) -> None:
        """
        ``### Added`` joins ``### Added``, a new subsection goes last, and the heading is dated.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(
            unreleased="### Added\n\n- a later thing\n\n### Fixed\n\n- a fix\n",
            sections=_ADDED_SECTION,
        )
        status, out, err = release(root, "release")
        assert status == 0, err
        assert (root / "CHANGELOG.md").read_text() == (
            "# Changelog\n\n## [Unreleased]\n\n"
            f"## [0.1.0] - {TODAY}\n\nThe first release.\n\n"
            "### Added\n\n- the first thing\n- a later thing\n\n### Fixed\n\n- a fix\n"
        )
        assert 'version = "0.1.0"' in (root / "pyproject.toml").read_text()
        assert "-- CHANGELOG.md" in out

    def test_a_minor_from_an_untagged_patch_takes_over_its_notes(
        self, make_repo: Callable[..., Path], uvx_log: Callable[[], list[str]]
    ) -> None:
        """
        0.1.1 declared and never tagged, the API grows: ``minor`` goes to 0.2.0 -- the gate's remedy works.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :param uvx_log: the stub relock log
        :ptype uvx_log: Callable[[], list[str]]
        :return: nothing
        :rtype: None
        """
        root = make_repo(
            version="0.1.1",
            tags=("v0.1.0",),
            unreleased="### Added\n\n- new API\n",
            sections="## [0.1.1] - 2026-09-01\n\n### Fixed\n\n- a fix\n\n## [0.1.0] - 2026-08-01\n\n- first\n",
        )
        status, out, err = release(root, "minor")
        assert status == 0, err
        assert "0.1.1 was declared but never tagged" in out
        assert 'version = "0.2.0"' in (root / "pyproject.toml").read_text()
        changelog = (root / "CHANGELOG.md").read_text()
        assert "## [0.1.1]" not in changelog
        assert f"## [0.2.0] - {TODAY}\n\n### Fixed\n\n- a fix\n\n### Added\n\n- new API\n" in changelog
        assert "## [0.1.0] - 2026-08-01" in changelog
        assert uvx_log() == [str(root)]

    def test_a_failed_relock_restores_every_file_and_stages_nothing(
        self, make_repo: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        it changes nothing unless it can change everything.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :param monkeypatch: pytest's environment patcher
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        root = make_repo(tags=("v0.1.0",))
        monkeypatch.setenv("UVX_STUB_FAIL_MATCH", root.name)
        before = snapshot(root)
        status, _, err = release(root, "patch")
        assert status == 1
        assert "uvx uv@latest lock failed" in err
        assert "every file this run touched is restored" in err
        assert snapshot(root) == before


class TestLocalMode:
    """a sources block in the working tree never reaches the commit."""

    _SOURCES = '\n[tool.uv.sources]\nsibling = { path = "../sibling", editable = true }\n'

    def _local(self, make_repo: Callable[..., Path], scratch: Path) -> Path:
        """a tagged repository with an uncommitted sources block and local relock.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :param scratch: the scratch parent
        :ptype scratch: Path
        :return: the repository
        :rtype: Path
        """
        (scratch / "sibling").mkdir()
        root = make_repo(tags=("v0.1.0",))
        (root / "pyproject.toml").write_text((root / "pyproject.toml").read_text() + self._SOURCES)
        (root / "uv.lock").write_text((root / "uv.lock").read_text() + "# local resolution\n")
        return root

    def test_the_index_holds_released_mode_and_the_tree_keeps_its_sources(
        self, make_repo: Callable[..., Path], scratch: Path, uvx_log: Callable[[], list[str]]
    ) -> None:
        """
        staged: HEAD plus the version, relocked in a scratch worktree. working tree: sources kept, relocked locally.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :param scratch: the scratch parent
        :ptype scratch: Path
        :param uvx_log: the stub relock log
        :ptype uvx_log: Callable[[], list[str]]
        :return: nothing
        :rtype: None
        """
        root = self._local(make_repo, scratch)
        status, out, err = release(root, "patch")
        assert status == 0, err
        staged_pyproject = git(root, "show", ":pyproject.toml")
        assert staged_pyproject == '[project]\nname = "proj"\nversion = "0.1.1"'
        staged_lock = git(root, "show", ":uv.lock")
        assert "# local resolution" not in staged_lock
        worktree_relock, local_relock = uvx_log()
        assert worktree_relock.endswith("/proj") and worktree_relock != str(root)
        assert local_relock == str(root)
        assert f"# relocked {worktree_relock}" in staged_lock
        assert git(root, "diff", "--cached", "--name-only").split() == ["CHANGELOG.md", "pyproject.toml", "uv.lock"]
        working = (root / "pyproject.toml").read_text()
        assert 'version = "0.1.1"' in working and "[tool.uv.sources]" in working
        assert "# local resolution" in (root / "uv.lock").read_text()
        assert "NO pathspec" in out
        assert len(git(root, "worktree", "list").splitlines()) == 1
        assert not Path(worktree_relock).exists()

    def test_the_scratch_worktree_sees_the_siblings(
        self, make_repo: Callable[..., Path], scratch: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """
        a sibling path source resolves in the scratch worktree as it does here.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :param scratch: the scratch parent
        :ptype scratch: Path
        :param monkeypatch: pytest's environment patcher
        :ptype monkeypatch: pytest.MonkeyPatch
        :param tmp_path: pytest's per-test directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        root = self._local(make_repo, scratch)
        probe = tmp_path / "probe.log"
        monkeypatch.setenv("UVX_STUB_PROBE", "sibling")
        monkeypatch.setenv("UVX_STUB_PROBE_OUT", str(probe))
        status, _, err = release(root, "patch")
        assert status == 0, err
        seen = probe.read_text().splitlines()
        assert len(seen) == 2 and seen[1] == str(root), seen
        assert "threetears-release-" in seen[0]

    def test_a_failed_worktree_relock_restores_and_unstages(
        self, make_repo: Callable[..., Path], scratch: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        a failure after staging began leaves the index as it was.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :param scratch: the scratch parent
        :ptype scratch: Path
        :param monkeypatch: pytest's environment patcher
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        root = self._local(make_repo, scratch)
        monkeypatch.setenv("UVX_STUB_FAIL_MATCH", "threetears-release-")
        before = snapshot(root)
        status, _, err = release(root, "patch")
        assert status == 1
        assert "scratch worktree" in err
        assert snapshot(root) == before
        assert len(git(root, "worktree", "list").splitlines()) == 1


class TestPrawductChangeLog:
    """a repository whose release record is a prawduct change-log."""

    _LOG = (
        "# Change Log\n\n"
        "## 2026-09-30: pending work\n\n<!-- prawduct: scope=new-thing -->\n\nbody\n\n"
        "## 2026-09-20: shipped work\n\n<!-- prawduct: scope=old-thing | release=v0.1.0 -->\n\nbody\n"
    )

    def test_only_pending_entries_are_marked(self, make_repo: Callable[..., Path]) -> None:
        """
        an entry that already carries ``release=`` keeps it.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        root = make_repo(tags=("v0.1.0",), notes={".prawduct/change-log.md": self._LOG})
        status, out, err = release(root, "patch")
        assert status == 0, err
        log = (root / ".prawduct/change-log.md").read_text()
        assert "<!-- prawduct: scope=new-thing | release=v0.1.1 -->" in log
        assert "<!-- prawduct: scope=old-thing | release=v0.1.0 -->" in log
        assert "1 change-log entries marked release=v0.1.1" in out

    def test_no_pending_entry_is_refused(self, make_repo: Callable[..., Path]) -> None:
        """
        nothing pending, nothing to release.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        log = self._LOG.replace("scope=new-thing -->", "scope=new-thing | release=v0.1.0 -->")
        root = make_repo(tags=("v0.1.0",), notes={".prawduct/change-log.md": log})
        _refused(root, "patch", "no release-pending entry")

    def test_a_minor_from_an_untagged_patch_moves_its_entries(self, make_repo: Callable[..., Path]) -> None:
        """
        entries marked for the never-tagged 0.1.1 move to 0.2.0 with the pending ones.

        :param make_repo: scratch repository factory
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        log = self._LOG.replace("release=v0.1.0", "release=v0.1.1")
        root = make_repo(version="0.1.1", tags=("v0.1.0",), notes={".prawduct/change-log.md": log})
        status, _, err = release(root, "minor")
        assert status == 0, err
        log_after = (root / ".prawduct/change-log.md").read_text()
        assert "release=v0.1.1" not in log_after
        assert log_after.count("release=v0.2.0") == 2


def test_the_restore_covers_a_commit_made_from_a_clean_tree(make_repo: Callable[..., Path]) -> None:
    """
    after a successful bump the committed result is exactly what the hint names.

    :param make_repo: scratch repository factory
    :ptype make_repo: Callable[..., Path]
    :return: nothing
    :rtype: None
    """
    root = make_repo(tags=("v0.1.0",))
    status, _, err = release(root, "patch")
    assert status == 0, err
    commit_all(root, "release: v0.1.1")
    assert git(root, "status", "--porcelain") == ""
    git(root, "tag", "-a", "v0.1.1", "-m", "v0.1.1")
    (root / "CHANGELOG.md").write_text(
        (root / "CHANGELOG.md").read_text().replace("## [Unreleased]\n", "## [Unreleased]\n\n- next\n")
    )
    commit_all(root, "notes")
    status, out, err = release(root, "patch")
    assert status == 0, err
    assert "Done. Version: 0.1.2" in out
