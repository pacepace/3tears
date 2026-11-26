"""
enforcement: a change a consumer must act on is named in the CHANGELOG's Unreleased section, or a commit says why.

``scripts/check_changelog.py`` holds it over a pull request's range in CI (Critic R-6, 2026-11-26: five merges into
the 0.95.0 line changed agent-acl's schema, a public error and the shared retry helper, and the Unreleased section
named none of it). What counts as consumer-facing is what the repo already records: a name in
``_public_api_consumers.txt`` (a caller outside this repo), read against the modules' surface as
:mod:`threetears.enforcement.release.surface` reads it (``__all__``), new or with a changed definition; and a new
``migrations/vNNN_*.py`` module.

These tests run the real script over a throwaway repository, so they fail if it stops catching an unnamed change,
starts refusing a named or excused one, reaches back past where it starts, or reads names out of the wrong place;
and they fail if the CI step goes away.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_changelog.py"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

#: the test repository's own config and nothing of the machine's
_ITS_OWN_CONFIG = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}

_API = "packages/demo/src/threetears/demo/api.py"
_LEDGER = "tests/enforcement/_public_api_consumers.txt"
_NOT_A_COMMIT = "0" * 40


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_changelog", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # a dataclass reads its module's namespace while it is built
    spec.loader.exec_module(module)
    return module


_CHECK = _load()


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(  # noqa: S603 - fixed argv, a test repository
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True, env={**os.environ, **_ITS_OWN_CONFIG}
    )
    return done.stdout.strip()


def _write(repo: Path, files: dict[str, str]) -> None:
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)


def _commit(repo: Path, files: dict[str, str], message: str = "a change") -> str:
    _write(repo, files)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-verify", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _changelog(*entries: str) -> str:
    return "# Changelog\n\n## Unreleased\n\n" + "\n\n".join(entries) + "\n\n## v0.1.0 -- 2026-01-01\n\nold_name\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """a workspace of one package exporting ``kept`` and ``changing``, which the ledger says the hub calls."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "dev@example.com")
    _git(tmp_path, "config", "user.name", "Dev")
    _commit(
        tmp_path,
        {
            "packages/demo/pyproject.toml": "[project]\nname = 'demo'\n",
            _API: '__all__ = ["changing", "kept"]\n\n\ndef kept():\n    return 1\n\n\ndef changing():\n    """v1."""\n'
            "    return 1\n",
            _LEDGER: f"{_API}:kept  # the hub calls it\n{_API}:changing  # the hub calls it\n",
            "CHANGELOG.md": _changelog("Nothing yet."),
        },
        "base",
    )
    return tmp_path


def _findings(repo: Path, base: str, checked_from: str = _NOT_A_COMMIT) -> list[str]:
    found: list[str] = _CHECK.findings(repo, base, "HEAD", checked_from)
    return found


def _add_name(name: str) -> dict[str, str]:
    """the demo module exporting *name* too, and the ledger naming its outside caller."""
    return {
        _API: f'__all__ = ["changing", "kept", "{name}"]\n\n\ndef kept():\n    return 1\n\n\ndef changing():\n'
        f'    """v1."""\n    return 1\n\n\ndef {name}():\n    return 2\n',
        _LEDGER: f"{_API}:kept  # the hub calls it\n{_API}:changing  # the hub calls it\n{_API}:{name}  # the hub\n",
    }


class TestNames:
    def test_a_new_name_a_consumer_calls_is_refused_until_the_changelog_names_it(self, repo: Path) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        _commit(repo, _add_name("brand_new"))
        found = _findings(repo, base)
        assert len(found) == 1 and f"{_API}:brand_new is new" in found[0], found
        _commit(repo, {"CHANGELOG.md": _changelog("**Added:** `brand_new`, which the hub calls at its re-pin.")})
        assert _findings(repo, base) == []

    def test_a_name_in_a_released_section_does_not_count(self, repo: Path) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        _commit(repo, _add_name("old_name"))
        assert len(_findings(repo, base)) == 1

    def test_a_word_containing_the_name_does_not_count(self, repo: Path) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        _commit(repo, {**_add_name("fresh"), "CHANGELOG.md": _changelog("`fresh_start` and `refresh` only.")})
        assert len(_findings(repo, base)) == 1
        _commit(repo, {"CHANGELOG.md": _changelog("`fresh.attr` is new.")})
        assert _findings(repo, base) == []

    def test_a_changed_definition_is_refused_and_a_reworded_docstring_is_not(self, repo: Path) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        source = (repo / _API).read_text()
        _commit(repo, {_API: source.replace('"""v1."""', '"""v1, reworded."""')})
        assert _findings(repo, base) == []
        _commit(repo, {_API: source.replace('"""v1."""\n    return 1', '"""v1."""\n    return 3')})
        found = _findings(repo, base)
        assert len(found) == 1 and f"{_API}:changing changed" in found[0], found

    def test_a_public_name_no_ledger_entry_names_is_not_held(self, repo: Path) -> None:
        """the recorded limit: a name with callers in this repo is outside the ledger, so outside the check."""
        base = _git(repo, "rev-parse", "HEAD")
        added = _add_name("in_repo_only")
        added[_LEDGER] = f"{_API}:kept  # the hub calls it\n{_API}:changing  # the hub calls it\n"
        _commit(repo, added)
        assert _findings(repo, base) == []


class TestMigrations:
    _MIGRATION = "packages/demo/src/threetears/demo/migrations/v002_add_a_column.py"

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param("demo v002 adds a column; the hub repeats it.", id="package-and-version"),
            pytest.param("The hub repeats demo's v002.", id="possessive"),
            pytest.param("The hub repeats demo v001-v003.", id="a-span"),
            pytest.param("`v002_add_a_column` adds a column.", id="module-name"),
            pytest.param("| demo | the hub's step |\n| --- | --- |\n| v002 | its v9 |", id="table-row"),
        ],
    )
    def test_a_new_migration_named_passes(self, repo: Path, entry: str) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        _commit(repo, {self._MIGRATION: "STEPS = ()\n", "CHANGELOG.md": _changelog(entry)})
        assert _findings(repo, base) == []

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param("Nothing yet.", id="absent"),
            pytest.param("Migrations: other v002, demo v001.", id="another-packages-v002"),
            pytest.param("The hub repeats demo v003-v005.", id="a-span-without-it"),
            pytest.param("| other | step |\n| --- | --- |\n| v002 | x |", id="another-packages-table"),
        ],
    )
    def test_a_new_migration_not_named_is_refused(self, repo: Path, entry: str) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        _commit(repo, {self._MIGRATION: "STEPS = ()\n", "CHANGELOG.md": _changelog(entry)})
        found = _findings(repo, base)
        assert len(found) == 1 and f"{self._MIGRATION} is a new migration of demo" in found[0], found


class TestTheRange:
    def test_a_commit_saying_why_excuses_that_change_alone(self, repo: Path) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        _commit(repo, _add_name("excused"), "add excused\n\nChangelog-None: excused: the hub only reads it in a test\n")
        assert _findings(repo, base) == []
        _commit(repo, _add_name("another"))
        assert len(_findings(repo, base)) == 1

    def test_a_reason_too_short_is_refused(self, repo: Path) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        _commit(repo, _add_name("excused"), "add excused\n\nChangelog-None: excused: fine\n")
        found = _findings(repo, base)
        assert len(found) == 1 and "says too little" in found[0], found

    def test_nothing_before_the_checked_from_commit_is_examined(self, repo: Path) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        checked_from = _commit(repo, _add_name("landed_before"))
        assert _findings(repo, base, checked_from) == []
        source = (repo / _API).read_text()
        _commit(repo, {_API: source.replace("return 2", "return 4")})
        found = _findings(repo, base, checked_from)
        assert len(found) == 1 and "landed_before changed" in found[0], found
        assert len(_findings(repo, base, _NOT_A_COMMIT)) == 1  # new since base, so held as new

    def test_a_checked_from_commit_outside_the_range_moves_nothing(self, repo: Path) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        _git(repo, "checkout", "-q", "-b", "elsewhere")
        aside = _commit(repo, {"unrelated.txt": "x\n"})
        _git(repo, "checkout", "-q", "main")
        _commit(repo, _add_name("brand_new"))
        assert len(_findings(repo, base, aside)) == 1

    def test_the_command_exits_one_naming_what_is_missing(self, repo: Path) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        _commit(repo, _add_name("brand_new"))
        done = subprocess.run(  # noqa: S603 - fixed argv, a test repository
            [sys.executable, str(_SCRIPT), f"{base}..HEAD", "--checked-from", _NOT_A_COMMIT],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 1 and "brand_new is new" in done.stderr, done.stderr
        unreadable = subprocess.run(  # noqa: S603 - fixed argv, a test repository
            [sys.executable, str(_SCRIPT), "no-such-ref..HEAD"], cwd=repo, capture_output=True, text=True, check=False
        )
        assert unreadable.returncode == 1 and "Nothing was compared" in unreadable.stderr


class TestTheGate:
    def test_ci_runs_the_check_over_every_pull_requests_range(self) -> None:
        workflow = yaml.safe_load(_WORKFLOW.read_text())
        steps = workflow["jobs"]["check"]["steps"]
        gate = [step for step in steps if "scripts/check_changelog.py" in str(step.get("run", ""))]
        assert len(gate) == 1, "no CI step refuses a consumer-facing change the CHANGELOG does not name"
        (step,) = gate
        assert "github.event.pull_request.base.sha" in step["run"]
        assert "github.event.pull_request.head.sha" in step["run"]
        assert "github.event_name == 'pull_request'" in step["if"]
        checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@"))
        assert checkout["with"]["fetch-depth"] == 0, "the range's base is not in a shallow clone"

    def test_the_checked_from_commit_is_a_full_hash(self) -> None:
        assert re.fullmatch(r"[0-9a-f]{40}", _CHECK.CHECKED_FROM), _CHECK.CHECKED_FROM

    def test_the_check_reads_the_ledger_the_caller_gate_reads(self) -> None:
        assert (_REPO_ROOT / _CHECK.LEDGER).is_file()
        assert (_REPO_ROOT / _CHECK.CHANGELOG).is_file()
        assert _CHECK.unreleased_section((_REPO_ROOT / _CHECK.CHANGELOG).read_text()).strip(), "no Unreleased section"
