"""the API-growth gate against scratch repositories: each refusal fires, and each allowed state passes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from packages.enforcement.tests.release._scratch import commit_all, git, release
from threetears.enforcement.release import ApiGrowthConfig, api_growth_findings, http_routes


@pytest.fixture
def repo(scratch: Path) -> Path:
    """an empty git repository in the scratch parent.

    :param scratch: the scratch parent
    :ptype scratch: Path
    :return: the repository
    :rtype: Path
    """
    root = scratch / "gated"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    return root


def _write(root: Path, version: str, exported: list[str], route: str | None = None) -> None:
    """declares *version* and a public module exporting *exported*.

    :param root: scratch repository
    :ptype root: Path
    :param version: ``[project] version``
    :ptype version: str
    :param exported: the module's ``__all__``
    :ptype exported: list[str]
    :param route: one ``@router.get`` path to declare, if any
    :ptype route: str | None
    :return: nothing
    :rtype: None
    """
    (root / "pyproject.toml").write_text(f'[project]\nname = "gated"\nversion = "{version}"\n')
    package = root / "src" / "pkg"
    package.mkdir(parents=True, exist_ok=True)
    body = [f"__all__ = {exported!r}", ""] + [f"def {name}() -> None:\n    pass\n" for name in exported]
    if route is not None:
        body.append(f'@router.get("{route}")\nasync def handler() -> None:\n    pass\n')
    (package / "api.py").write_text("\n".join(body))


def _release(root: Path, version: str, exported: list[str]) -> None:
    """commits and tags a release.

    :param root: scratch repository
    :ptype root: Path
    :param version: ``X.Y.Z``
    :ptype version: str
    :param exported: the module's ``__all__``
    :ptype exported: list[str]
    :return: nothing
    :rtype: None
    """
    _write(root, version, exported)
    commit_all(root, f"release {version}")
    git(root, "tag", "-a", f"v{version}", "-m", version)


def _findings(root: Path, roots: tuple[str, ...] = ("src/pkg",), anchor: str = "kept") -> list[str]:
    """the gate's findings for a scratch repository.

    :param root: scratch repository
    :ptype root: Path
    :param roots: source roots
    :ptype roots: tuple[str, ...]
    :param anchor: a name ``src/pkg/api.py`` must export
    :ptype anchor: str
    :return: findings
    :rtype: list[str]
    """
    config = ApiGrowthConfig(
        repo_root=root,
        source_roots=roots,
        first_release=(0, 1, 0),
        known_exports=(("src/pkg/api.py", anchor),),
        extractors=(http_routes,),
    )
    return api_growth_findings(config)


class TestTheGateCanFail:
    """each refusal fires on a repository this test authors."""

    def test_growth_inside_a_patch(self, repo: Path) -> None:
        """
        a name added after v0.1.0 under 0.1.1 is named.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _release(repo, "0.1.0", ["kept"])
        _write(repo, "0.1.1", ["kept", "added"])
        findings = _findings(repo)
        assert len(findings) == 1
        assert "src/pkg/api.py: added" in findings[0]
        assert "./scripts/bump-version.sh minor" in findings[0]

    def test_a_new_route_inside_a_patch(self, repo: Path) -> None:
        """
        a route with no new name is still growth.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _release(repo, "0.1.0", ["kept"])
        _write(repo, "0.1.1", ["kept"], route="/things")
        findings = _findings(repo)
        assert len(findings) == 1 and "route GET /things" in findings[0]

    def test_growth_in_a_module_without_all(self, repo: Path) -> None:
        """
        a module with no ``__all__`` exports what it defines.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _write(repo, "0.1.0", ["kept"])
        (repo / "src/pkg/plain.py").write_text("import os\n\n\ndef old() -> None:\n    pass\n")
        commit_all(repo, "release")
        git(repo, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
        _write(repo, "0.1.1", ["kept"])
        (repo / "src/pkg/plain.py").write_text("import sys\n\n\ndef old() -> None: ...\ndef new() -> None: ...\n")
        findings = _findings(repo)
        assert len(findings) == 1 and "src/pkg/plain.py: new\n" in findings[0] + "\n"

    def test_growth_in_a_new_source_root(self, repo: Path) -> None:
        """
        a source root absent at the tag is wholly new, so all of it is growth.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _release(repo, "0.1.0", ["kept"])
        _write(repo, "0.1.1", ["kept"])
        (repo / "src/newpkg").mkdir()
        (repo / "src/newpkg/mod.py").write_text('__all__ = ["fresh"]\nfresh = 1\n')
        findings = _findings(repo, roots=("src/pkg", "src/newpkg"))
        assert len(findings) == 1 and "src/newpkg/mod.py: fresh" in findings[0]

    def test_no_tag_past_the_first_release(self, repo: Path) -> None:
        """
        the no-tag state is accepted at the first release only.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _write(repo, "0.1.1", ["kept"])
        commit_all(repo, "untagged")
        findings = _findings(repo)
        assert len(findings) == 1 and "no vX.Y.Z release tag is visible" in findings[0]

    def test_a_version_below_the_latest_release(self, repo: Path) -> None:
        """
        versions only move forward.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _release(repo, "0.2.0", ["kept"])
        _write(repo, "0.1.9", ["kept"])
        findings = _findings(repo)
        assert len(findings) == 1 and "BELOW the latest release v0.2.0" in findings[0]

    def test_a_tag_whose_commit_this_clone_lacks(self, repo: Path) -> None:
        """
        a shallow clone's tag is unreadable, and that is a refusal, not an empty baseline.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _write(repo, "0.1.0", ["kept"])
        commit_all(repo, "head")
        (repo / ".git/refs/tags").mkdir(parents=True, exist_ok=True)
        (repo / ".git/refs/tags/v0.1.0").write_text("0123456789abcdef0123456789abcdef01234567\n")
        findings = _findings(repo)
        assert len(findings) == 1 and "could not be read at v0.1.0" in findings[0]

    def test_a_blob_this_clone_lacks(self, repo: Path) -> None:
        """
        a ``<sha> missing`` reply is an unreadable baseline, not a crash.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _release(repo, "0.1.0", ["kept"])
        blob = git(repo, "rev-parse", "v0.1.0:src/pkg/api.py")
        (repo / ".git/objects" / blob[:2] / blob[2:]).unlink()
        _write(repo, "0.1.1", ["kept", "added"])
        findings = _findings(repo)
        assert len(findings) == 1, findings
        assert "could not be read at v0.1.0" in findings[0] and f"object {blob} missing" in findings[0]

    def test_a_non_literal_all_is_a_finding_on_both_sides(self, repo: Path) -> None:
        """
        a computed ``__all__`` names its module, at the tag and in the tree.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _write(repo, "0.1.0", ["kept"])
        (repo / "src/pkg/computed.py").write_text('from x import base\n__all__ = [*base.__all__, "y"]\n')
        commit_all(repo, "release")
        git(repo, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
        _write(repo, "0.1.1", ["kept"])
        findings = _findings(repo)
        assert [f for f in findings if "src/pkg/computed.py" in f and "at v0.1.0" not in f]
        assert [f for f in findings if "src/pkg/computed.py" in f and "at v0.1.0" in f]

    def test_a_stale_anchor(self, repo: Path) -> None:
        """
        a reader that stops seeing a known export fails, rather than comparing empty sets.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _write(repo, "0.1.0", ["kept"])
        commit_all(repo, "first")
        findings = _findings(repo, anchor="gone")
        assert len(findings) == 1 and "no longer sees 'gone'" in findings[0]


class TestWhatTheGateAllows:
    """the states that are NOT findings."""

    def test_growth_inside_a_minor(self, repo: Path) -> None:
        """
        a minor bump is what growth is for.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _release(repo, "0.1.0", ["kept"])
        _write(repo, "0.2.0", ["kept", "added"], route="/things")
        assert _findings(repo) == []

    def test_the_first_release_with_no_tag(self, repo: Path) -> None:
        """
        before anything was released there is nothing to compare against.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _write(repo, "0.1.0", ["kept"])
        commit_all(repo, "first")
        assert _findings(repo) == []

    def test_growth_in_a_private_module(self, repo: Path) -> None:
        """
        a ``_``-prefixed module is internal.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _release(repo, "0.1.0", ["kept"])
        _write(repo, "0.1.1", ["kept"])
        (repo / "src/pkg/_internal.py").write_text('__all__ = ["helper"]\n')
        assert _findings(repo) == []

    def test_byte_identical_modules_keep_their_own_baselines(self, repo: Path) -> None:
        """
        two modules with one blob at the tag both have a baseline, so neither reads as all-new.

        :param repo: scratch repository
        :ptype repo: Path
        :return: nothing
        :rtype: None
        """
        _write(repo, "0.1.0", ["kept"])
        for name in ("one", "two"):
            (repo / f"src/pkg/{name}.py").write_text('__all__ = ["same"]\nsame = 1\n')
        commit_all(repo, "release")
        git(repo, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
        _write(repo, "0.1.1", ["kept"])
        assert _findings(repo) == []

    def test_the_untagged_patch_then_growth_remedy_works(self, repo: Path, make_repo: Callable[..., Path]) -> None:
        """
        0.1.1 declared, the API grows, the gate says ``minor``, and ``minor`` clears the gate.

        :param repo: scratch repository
        :ptype repo: Path
        :param make_repo: scratch repository factory (sets up the uvx stub)
        :ptype make_repo: Callable[..., Path]
        :return: nothing
        :rtype: None
        """
        _release(repo, "0.1.0", ["kept"])
        (repo / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n\n## [0.1.1] - 2026-09-01\n\n- a fix\n")
        (repo / "uv.lock").write_text('[[package]]\nname = "gated"\nversion = "0.1.1"\nsource = { editable = "." }\n')
        _write(repo, "0.1.1", ["kept"])
        commit_all(repo, "declare 0.1.1")
        _write(repo, "0.1.1", ["kept", "added"])
        commit_all(repo, "new API")
        findings = _findings(repo)
        assert len(findings) == 1 and "./scripts/bump-version.sh minor" in findings[0]
        status, _, err = release(repo, "minor")
        assert status == 0, err
        assert _findings(repo) == []
